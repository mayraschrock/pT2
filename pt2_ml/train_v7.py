import uproot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import os
import json
import torch.onnx
from collections import deque
import time

from nn_common import (BASE_BRANCHES, LAYER_FEATURES, TRAIN, VAL, TEST, Model, branches, event_split,
                       feature_names, find_files, make_features)
from nn_eval import DEFAULT_KEY_EFFS, evaluate as evaluate_model

# Wall-clock timer: printed per epoch, after training, and at the end
START_TIME = time.time()

def fmt_duration(seconds):
    """3725 -> '1h 02m 05s'"""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser(
    description="Train the pT2 real-vs-fake NN on pt2_training_data.root files from `pt2 process -r`")
parser.add_argument("--data",        type=str, nargs="+", default=["data_new/"],
                    help="Training files, directories of .root files, or glob patterns")
parser.add_argument("--output_dir",  type=str, default="outputs/")
parser.add_argument("--val_frac",    type=float, default=0.1, help="Fraction of events used for validation")
parser.add_argument("--test_frac",   type=float, default=0.1, help="Fraction of events used for testing")
parser.add_argument("--epochs",      type=int, default=30)
parser.add_argument("--fast_dev_run", action="store_true",
                    help="Quick check: 2 epochs, at most 20 batches per epoch")
parser.add_argument("--skip_shap",   action="store_true",
                    help="Skip SHAP (slow on large test sets; needs the shap package)")
parser.add_argument("--layers",      action="store_true",
                    help="Also use the LS layer connection (one-hot is_<layer> branches) as inputs")
parser.add_argument("--lst_vars",    action="store_true",
                    help="Also use the LST variables (dPhi, betaIn/Out, dBeta, z residuals, dAngle) as inputs")
parser.add_argument("--batch_size",  type=int,   default=655360, help="pT2s per training batch")
parser.add_argument("--lr",          type=float, default=4.5e-3, help="Starting Adam learning rate (cosine decay to 1e-5)")
parser.add_argument("--pos_weight",  type=float, default=400.0,  help="BCE weight of real pT2s (400 ~ balances the classes)")
parser.add_argument("--gap_weight",  type=float, default=0.5,
                    help="Weight of the gap penalty pushing real scores above 0.9 (0 = off)")
parser.add_argument("--select_eff",  type=float, default=None,
                    help="Keep the epoch with the lowest validation fake rate at this real efficiency (e.g. 0.74) "
                         "instead of the highest validation AUC; early stopping uses the same metric")
parser.add_argument("--shuffle_chunks", type=int, default=0,
                    help="Training split only: read the file chunks in a new random order every epoch and mix this "
                         "many chunks (~5 events each) before cutting batches, so every batch holds pT2s from many "
                         "events (0 = old behaviour: chunks in file order, shuffled only within a chunk)")
parser.add_argument("--key_effs",    type=float, nargs="+", default=DEFAULT_KEY_EFFS,
                    help="Working points drawn on the plots (all of them are in metrics.json)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
plot_dir = os.path.join(args.output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
# What eval_wp.py needs to re-make the plots of this model (same events, same inputs)
with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
    json.dump({"data": args.data, "val_frac": args.val_frac, "test_frac": args.test_frac,
               "layers": args.layers, "lst_vars": args.lst_vars, "epochs": args.epochs,
               "batch_size": args.batch_size, "lr": args.lr, "pos_weight": args.pos_weight,
               "gap_weight": args.gap_weight, "select_eff": args.select_eff,
               "shuffle_chunks": args.shuffle_chunks}, f, indent=2)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size
EPOCHS     = 2 if args.fast_dev_run else args.epochs
MAX_BATCHES = 20 if args.fast_dev_run else None   # per pass over a split

# Rows read from a file at a time (keeps memory bounded for large files)
READ_STEP = 10_000_000

# True global ratio: ~1.2B fakes / ~3M reals
POS_WEIGHT = args.pos_weight

# Score gap penalty: weight of auxiliary loss pushing real scores above this threshold
GAP_PENALTY_WEIGHT     = args.gap_weight   # relative weight vs BCE loss (default 0.5)
GAP_PENALTY_THRESHOLD  = 0.9   # penalise real tracks scoring below this

# ---------------------------
# Features (pt2_ml/nn_common.py; same order as src/pt2_scorer.cc)
# v7: added log_abs_md0_dxy (single highest-SHAP feature, log scale)
# --layers: + the one-hot LS layer connection
# ---------------------------
FEATURE_NAMES = feature_names(args.layers, args.lst_vars)
BRANCHES = branches(args.layers, args.lst_vars)

# ---------------------------
# Dataset
# Rows are split into train / val / test by event, so all pT2s of one event land in the same split
# ---------------------------
class Pt2Dataset(IterableDataset):
    """Yields (features, is_real) batches of one split, shuffled within each read chunk
    (with --shuffle_chunks on the training split: within a group of randomly chosen chunks)."""
    def __init__(self, files, split):
        self.files = files
        self.split = split

    def _rows(self, d):
        """(features, is_real) of this split's rows in one read chunk, or None if it has none."""
        keep = event_split(d["event_idx"], args.val_frac, args.test_frac) == self.split
        if not keep.any():
            return None
        return (make_features({k: v[keep] for k, v in d.items()}, args.layers, args.lst_vars),
                d["is_real"][keep].astype(np.float32))

    def _groups(self):
        """Lists of (features, is_real) chunks; each list is permuted together and cut into batches."""
        if self.split != TRAIN or args.shuffle_chunks <= 0:
            for d in uproot.iterate([f + ":tree" for f in self.files], BRANCHES,
                                    step_size=READ_STEP, library="np"):
                rows = self._rows(d)
                if rows is not None:
                    yield [rows]
            return
        # Every chunk of every file, in a new random order each epoch, mixed shuffle_chunks at a time
        chunks = [(f, start, min(start + READ_STEP, n))
                  for f in self.files
                  for n in [uproot.open(f + ":tree").num_entries]
                  for start in range(0, n, READ_STEP)]
        order = np.random.permutation(len(chunks))
        for g in range(0, len(order), args.shuffle_chunks):
            group = []
            for k in order[g:g + args.shuffle_chunks]:
                f, start, stop = chunks[k]
                d = uproot.open(f + ":tree").arrays(BRANCHES, entry_start=start, entry_stop=stop, library="np")
                rows = self._rows(d)
                if rows is not None:
                    group.append(rows)
            if group:
                yield group

    def __iter__(self):
        n_batches = 0
        for group in self._groups():
            features = np.concatenate([g[0] for g in group]) if len(group) > 1 else group[0][0]
            is_real = np.concatenate([g[1] for g in group]) if len(group) > 1 else group[0][1]
            del group

            idx = np.random.permutation(len(is_real))
            features, is_real = features[idx], is_real[idx]

            for i in range(0, len(is_real), BATCH_SIZE):
                yield (torch.from_numpy(features[i:i+BATCH_SIZE]),
                       torch.from_numpy(is_real[i:i+BATCH_SIZE]))
                n_batches += 1
                if MAX_BATCHES is not None and n_batches >= MAX_BATCHES:
                    return

# ---------------------------
# SHAP shim
# ---------------------------
class ShapShim(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(-1)


# ---------------------------
# Loss: BCE + score gap penalty
# ---------------------------
def gap_penalty(logits, y):
    """
    Auxiliary loss that penalises real tracks (y==1) whose sigmoid score
    falls below GAP_PENALTY_THRESHOLD.  This pushes all real tracks toward
    score=1, closing the gap in the score distribution that causes the
    99% threshold to fall into noisy fake territory.
    """
    scores     = torch.sigmoid(logits)
    real_mask  = y == 1
    if real_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    real_scores = scores[real_mask]
    # Hinge-style: only penalise scores below threshold
    penalty = torch.clamp(GAP_PENALTY_THRESHOLD - real_scores, min=0.0)
    return penalty.mean()


def combined_loss(criterion, logits, y):
    bce     = criterion(logits, y)
    penalty = gap_penalty(logits, y)
    return bce + GAP_PENALTY_WEIGHT * penalty


# ---------------------------
# Train / Eval
# ---------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = combined_loss(criterion, model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def evaluate(model, loader, criterion=None, max_rows=None):
    """AUC, labels and scores on `loader` (dropout off). With `criterion`, also the mean loss
    (same combined loss as training, averaged per batch the same way) as a 4th value.
    With `max_rows`, stop after about that many rows (used for the training-set check)."""
    model.eval()
    ys, preds, losses = [], [], []
    n_rows = 0
    with torch.no_grad():
        for x, y in loader:
            if max_rows is not None and n_rows >= max_rows:
                break
            n_rows += len(y)
            x = x.to(DEVICE)
            logits = model(x)
            if criterion is not None:
                losses.append(combined_loss(criterion, logits, y.to(DEVICE)).item())
            ys.append(y.cpu().numpy())
            preds.append(torch.sigmoid(logits).cpu().numpy())
    ys    = np.concatenate(ys)
    preds = np.concatenate(preds)
    auc   = roc_auc_score(ys, preds)
    if criterion is not None:
        return auc, ys, preds, float(np.mean(losses))
    return auc, ys, preds

def curve_metrics(y_true, y_pred, eff):
    """Fake rate at real efficiency `eff`, accuracy and balanced accuracy (mean of real and fake
    efficiency) at score 0.5. Plain accuracy is ~99.7% even for "everything is fake"."""
    real = y_true == 1
    pred_real = y_pred >= 0.5
    real_eff, fake_rej = pred_real[real].mean(), (~pred_real[~real]).mean()
    return {"fake_rate": float(fake_rate_at_efficiency(y_true, y_pred, eff)[0]),
            "acc": float((pred_real == real).mean()), "bal_acc": float(0.5 * (real_eff + fake_rej))}

def fake_rate_at_efficiency(y_true, y_pred, target_eff):
    real_scores = y_pred[y_true == 1]
    fake_scores = y_pred[y_true == 0]
    threshold   = np.quantile(real_scores, 1 - target_eff)
    fake_eff    = (fake_scores > threshold).mean()
    return fake_eff, threshold

# ---------------------------
# Load files
# ---------------------------
files = find_files(args.data)
print(f"{len(files)} file(s); event split train/val/test = "
      f"{1 - args.val_frac - args.test_frac:.2f}/{args.val_frac:.2f}/{args.test_frac:.2f}")
train_files = val_files = files

# Every split needs at least one event, and training needs real pT2s
split_counts = np.zeros((3, 2), dtype=np.int64)   # [split][events, reals]
for f in files:
    events = set()
    for d in uproot.iterate(f + ":tree", ["event_idx", "is_real"], step_size=READ_STEP, library="np"):
        events.update(np.unique(d["event_idx"]).tolist())
        np.add.at(split_counts[:, 1], event_split(d["event_idx"], args.val_frac, args.test_frac), d["is_real"].astype(np.int64))
    np.add.at(split_counts[:, 0], event_split(np.array(sorted(events), dtype=np.int64), args.val_frac, args.test_frac), 1)
for name, k in [("train", TRAIN), ("val", VAL), ("test", TEST)]:
    print(f"  {name:5s}: {split_counts[k, 0]} events, {split_counts[k, 1]} real pT2s")
if (split_counts[:, 0] == 0).any() or split_counts[TRAIN, 1] == 0:
    raise SystemExit("Not enough events for a train/val/test split; "
                     "produce more with `pt2 process -r -n <N>` or adjust --val_frac/--test_frac")

# ---------------------------
# Normalization — dedicated loader, not reused
# ---------------------------
print("Computing normalization...")
norm_loader = DataLoader(Pt2Dataset(train_files, TRAIN), batch_size=None)
all_feats = []
for i, (x, _) in enumerate(norm_loader):
    all_feats.append(x.numpy())
    if i > 50: break
del norm_loader

all_feats = np.concatenate(all_feats)
mean = all_feats.mean(axis=0)
std  = all_feats.std(axis=0) + 1e-6
if args.layers:   # one-hot layer flags stay 0 / 1
    n_layers = len(LAYER_FEATURES)
    mean[-n_layers:], std[-n_layers:] = 0.0, 1.0

np.save(os.path.join(args.output_dir, "mean.npy"), mean)
np.save(os.path.join(args.output_dir, "std.npy"),  std)
with open(os.path.join(args.output_dir, "features.json"), "w") as f:
    json.dump(FEATURE_NAMES, f, indent=2)

def normalize(x):
    return (x - mean) / std

class NormWrapper:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for x, y in self.loader:
            yield torch.tensor(normalize(x.numpy()), dtype=torch.float32), y

train_loader = NormWrapper(DataLoader(Pt2Dataset(train_files, TRAIN), batch_size=None))
val_loader   = NormWrapper(DataLoader(Pt2Dataset(val_files,   VAL),   batch_size=None))

# ---------------------------
# Model setup
# ---------------------------
input_dim = len(mean)
assert input_dim == len(FEATURE_NAMES), \
    f"Feature mismatch: mean has {input_dim}, FEATURE_NAMES has {len(FEATURE_NAMES)}"
print(f"Input dim: {input_dim}")

model     = Model(input_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

# ---------------------------
# Training loop
# v7: early stopping on val AUC (stable), patience=8
#     fake rate still logged for visibility
# ---------------------------
train_losses, val_losses, val_aucs, val_fake_rates, lrs = [], [], [], [], []
val_fake_rates_sel = []   # with --select_eff: validation fake rate at that efficiency
# Training vs validation, both with dropout off, on ~TRAIN_EVAL_ROWS training pT2s and all validation pT2s
TRAIN_EVAL_ROWS = 40_000_000
COMPARE_EFF = args.select_eff if args.select_eff is not None else 0.74
compare = {f"{s}_{k}": [] for s in ("train", "val") for k in ("loss", "auc", "fake_rate", "acc", "bal_acc")}
best_val_auc   = -float("inf")
best_sel_fr    = float("inf")
patience       = 8
min_delta      = 1e-5
no_improve     = 0
fake_rate_window = deque(maxlen=3)

print(f"Setup done in {fmt_duration(time.time() - START_TIME)} (file scan + normalization)", flush=True)
train_start = time.time()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    loss = train_epoch(model, train_loader, optimizer, criterion)
    auc, y_val, p_val, val_loss = evaluate(model, val_loader, criterion)
    scheduler.step()

    fake_rate, _ = fake_rate_at_efficiency(y_val, p_val, target_eff=0.99)

    tr_auc, y_tr, p_tr, tr_loss = evaluate(model, train_loader, criterion, max_rows=TRAIN_EVAL_ROWS)
    for split, (a, l, yy, pp) in (("train", (tr_auc, tr_loss, y_tr, p_tr)), ("val", (auc, val_loss, y_val, p_val))):
        cm = curve_metrics(yy, pp, COMPARE_EFF)
        compare[f"{split}_loss"].append(l); compare[f"{split}_auc"].append(float(a))
        for k in ("fake_rate", "acc", "bal_acc"):
            compare[f"{split}_{k}"].append(cm[k])
    del y_tr, p_tr
    current_lr   = scheduler.get_last_lr()[0]

    if args.select_eff is not None:
        sel_fr, _ = fake_rate_at_efficiency(y_val, p_val, target_eff=args.select_eff)
        val_fake_rates_sel.append(sel_fr)

    fake_rate_window.append(fake_rate)
    rolling_fr = float(np.mean(fake_rate_window))

    train_losses.append(loss)
    val_losses.append(val_loss)
    val_aucs.append(auc)
    val_fake_rates.append(fake_rate)
    lrs.append(current_lr)

    epoch_time = time.time() - epoch_start
    eta = (time.time() - train_start) / (epoch + 1) * (EPOCHS - epoch - 1)   # if no early stop
    print(f"Epoch {epoch:02d}: loss={loss:.4f}  val_loss={val_loss:.4f}  val_auc={auc:.6f}  "
          f"fake_rate@99%={fake_rate*100:.4f}%  "
          f"rolling_fr={rolling_fr*100:.4f}%  "
          + (f"fake_rate@{args.select_eff:.0%}={sel_fr*100:.5f}%  " if args.select_eff is not None else "")
          + f"train_auc={tr_auc:.6f}  "
          f"fake_rate@{COMPARE_EFF:.0%} train/val={compare['train_fake_rate'][-1]*100:.5f}%/{compare['val_fake_rate'][-1]*100:.5f}%  "
          + f"lr={current_lr:.2e}  "
          f"time={fmt_duration(epoch_time)}  eta<={fmt_duration(eta)}", flush=True)

    # Per-epoch history, rewritten every epoch so a killed job still leaves it behind
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses, "val_auc": val_aucs,
                   "val_fake_rate_99": val_fake_rates, "val_fake_rate_sel": val_fake_rates_sel,
                   "select_eff": args.select_eff, "lr": lrs,
                   "compare_eff": COMPARE_EFF, **compare}, f, indent=2)

    # Checkpoint on val AUC, or on the val fake rate at --select_eff
    if args.select_eff is not None:
        improved = sel_fr < best_sel_fr
    else:
        improved = auc > best_val_auc + min_delta
    if improved:
        best_val_auc = auc
        best_sel_fr  = sel_fr if args.select_eff is not None else best_sel_fr
        no_improve   = 0
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, "model_best.pt"))
        print(f"  -> new best (val_auc={auc:.6f}"
              + (f", fake_rate@{args.select_eff:.0%}={sel_fr*100:.5f}%" if args.select_eff is not None else "") + ")")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no {'fake-rate' if args.select_eff is not None else 'AUC'} improvement for {patience} epochs)")
            break

model.load_state_dict(
    torch.load(os.path.join(args.output_dir, "model_best.pt"), weights_only=True)
)
print(f"\nTraining took {fmt_duration(time.time() - train_start)}")
print(f"Loaded best model (val_auc = {best_val_auc:.6f})")

# ---------------------------
# Final evaluation: metrics.json and all the plots (pt2_ml/nn_eval.py)
# ---------------------------
history = {"train_loss": train_losses, "val_loss": val_losses, "val_auc": val_aucs,
           "val_fake_rate_99": val_fake_rates, "val_fake_rate_sel": val_fake_rates_sel,
           "select_eff": args.select_eff, "lr": lrs, "compare_eff": COMPARE_EFF, **compare}
shap_features, shap_is_real = evaluate_model(model, mean, std, files, args.val_frac, args.test_frac, args.layers,
                                             DEVICE, args.output_dir, history=history, key_effs=args.key_effs,
                                             lst=args.lst_vars)

# ---------------------------
# SHAP
# ---------------------------
if not args.skip_shap:
    import shap
    print("Computing SHAP values (use --skip_shap to bypass)...")

    def savefig(name):
        plt.savefig(os.path.join(plot_dir, name))
        plt.close()

    bg_idx  = np.random.choice(len(shap_features), size=min(1000, len(shap_features)), replace=False)
    bg_data = torch.tensor(normalize(shap_features[bg_idx]), dtype=torch.float32).to(DEVICE)

    r_idx = np.where(shap_is_real)[0]
    f_idx = np.where(~shap_is_real)[0]
    explain_idx = np.concatenate([
        np.random.choice(r_idx, min(2000, len(r_idx)), replace=False),
        np.random.choice(f_idx, min(2000, len(f_idx)), replace=False),
    ])
    explain_data   = torch.tensor(normalize(shap_features[explain_idx]), dtype=torch.float32).to(DEVICE)
    explain_labels = shap_is_real[explain_idx]
    explain_np     = explain_data.cpu().numpy()

    model.eval()
    shim = ShapShim(model).to(DEVICE)
    shim.eval()
    shap_values = shap.DeepExplainer(shim, bg_data).shap_values(explain_data)
    shap_arr = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values)
    while shap_arr.ndim > 2 and shap_arr.shape[-1] == 1:
        shap_arr = shap_arr.squeeze(-1)
    assert shap_arr.ndim == 2 and shap_arr.shape == (len(explain_idx), len(FEATURE_NAMES)), \
        f"Unexpected shap_arr shape: {shap_arr.shape}"

    # Global importance
    mean_abs = np.abs(shap_arr).mean(axis=0)
    order    = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(FEATURE_NAMES) + 1))
    ax.barh([FEATURE_NAMES[i] for i in order], mean_abs[order], color="#2a78d6")
    ax.set(title="Feature importance (mean |SHAP|)", xlabel="Mean |SHAP value|")
    ax.grid(False, axis="y")
    fig.tight_layout()
    savefig("shap_importance.png")

    # Beeswarms: all, real only, fake only
    for mask, suffix in ((np.ones(len(explain_labels), bool), ""), (explain_labels, "_real"), (~explain_labels, "_fake")):
        if mask.sum() > 10:
            plt.figure(figsize=(8, 0.3 * len(FEATURE_NAMES) + 1))
            shap.summary_plot(shap_arr[mask], explain_np[mask], feature_names=FEATURE_NAMES,
                              show=False, plot_size=None)
            plt.tight_layout()
            savefig(f"shap_summary{suffix}.png")

    np.save(os.path.join(args.output_dir, "shap_values.npy"), shap_arr)
    print("SHAP done.")

# ---------------------------
# Save models
# ---------------------------
torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
print("\nDone.  model_best.pt = best val AUC  |  model_final.pt = last epoch")

model.eval()
dummy = torch.randn(1, input_dim).to(DEVICE)
torch.onnx.export(
    model, dummy,
    os.path.join(args.output_dir, "model.onnx"),
    input_names=["features"],
    output_names=["score"],
    # Variable batch size, so src/pt2_scorer.cc can score many pT2s per call
    dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
    opset_version=13,
    do_constant_folding=True
)
print("ONNX model exported.")
print(f"Total time: {fmt_duration(time.time() - START_TIME)}")
