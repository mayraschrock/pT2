"""
Evaluation of a trained pT2 NN: working points (metrics.json) and all the plots.

Run at the end of train_v7.py, or on an already-trained model with eval_wp.py.

Working points: for each target real efficiency, the score cut is picked on the VALIDATION events
(the cut that keeps that fraction of their real pT2s) and then measured on the TEST events.
Picking and measuring on different events keeps the test numbers unbiased; use the same test events
in `pt2 mlcut -s test`. A pT2 passes if score >= cut, as in src/ml_cut.cc.

Outputs, in <output_dir>:
  metrics.json   one entry per working point: cut, test efficiency, fake rate, purity, counts
  plots/         training_curves, roc, working_point_scan, score_dist, cut_scan, pr,
                 features, layers, eff_vs_pls_eta, eff_vs_pls_pt
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import uproot

from nn_common import (BASE_BRANCHES, BASE_FEATURES, LAYER_FEATURES, LAYER_NAMES, LST_BRANCHES, TEST, VAL,
                       event_split, layer_index, make_features)

ROWS_PER_READ  = 10_000_000   # pT2 rows read from the ROOT file at a time (bounds memory)
ROWS_PER_SCORE = 1_000_000    # pT2 rows sent through the model at a time
N_FAKE_SAMPLE  = 3_000_000    # test fakes whose features are kept for the feature plots / SHAP (all reals are kept)

# Number of fakes in the full sample, used only to turn a fake rate into a rough count
TOTAL_FAKES_ESTIMATE = 1.2e9

# Real efficiencies to compute a working point for (all in metrics.json)
DEFAULT_TARGETS = [round(0.70 + 0.01 * i, 2) for i in range(30)] + [0.995, 0.999]
# Working points drawn on the plots
DEFAULT_KEY_EFFS = [0.74, 0.80, 0.90, 0.95]

# ---------------------------
# Style
# ---------------------------
REAL_COLOR  = "#2a78d6"   # blue
FAKE_COLOR  = "#e34948"   # red
VAL_COLOR   = "#86b6ef"   # light blue: the validation curve under the test one
MUTED       = "#8a8984"   # all working points, reference lines
TEXT_MUTED  = "#52514e"
WP_MARKERS  = ["o", "s", "D", "^", "v", "P"]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#6b6a66",
    "axes.grid": True, "grid.color": "#d9d8d4", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})


def wp_colors(n):
    """Ordered working points: one hue (purple), light -> dark."""
    return [plt.cm.Purples(x) for x in np.linspace(0.45, 0.95, n)]


def sigmoid32(logits):
    """Score from logit, in float32 like Pt2Scorer::run() in src/pt2_scorer.cc."""
    with np.errstate(over="ignore"):
        return (np.float32(1) / (np.float32(1) + np.exp(-logits.astype(np.float32)))).astype(np.float32)


def logit(score):
    """Logit of a score cut, finite even for a cut of exactly 0 or 1."""
    p = np.clip(np.float64(score), 1e-15, 1 - 1e-15)
    return float(np.log(p / (1 - p)))


def pct(x):
    """0.000451 -> '0.0451%'"""
    return f"{100 * x:.3g}%"


# ---------------------------
# Scoring
# ---------------------------
def score_splits(model, mean, std, files, val_frac, test_frac, layers, device, seed=0, lst=False):
    """One pass over the files: NN logit of every validation and test pT2, plus what the plots need."""
    rng = np.random.default_rng(seed)
    total_rows = sum(uproot.open(f + ":tree").num_entries for f in files)
    keep_fake_prob = min(1.0, N_FAKE_SAMPLE / max(1.0, test_frac * total_rows))

    model.eval()
    def logits_of(raw):
        x = ((raw - mean) / std).astype(np.float32)
        out = np.empty(len(x), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(x), ROWS_PER_SCORE):
                out[i:i + ROWS_PER_SCORE] = model(torch.from_numpy(x[i:i + ROWS_PER_SCORE]).to(device)).cpu().numpy()
        return out

    keys = ["logits", "is_real", "layer", "pls_pt", "pls_eta", "sample_features", "sample_is_real"]
    chunks = {VAL: {k: [] for k in keys}, TEST: {k: [] for k in keys}}
    events = {VAL: set(), TEST: set()}
    for d in uproot.iterate([f + ":tree" for f in files], BASE_BRANCHES + LAYER_FEATURES + (LST_BRANCHES if lst else []),
                            step_size=ROWS_PER_READ, library="np"):
        split_of_row = event_split(d["event_idx"], val_frac, test_frac)
        for split in (VAL, TEST):
            rows = split_of_row == split
            if not rows.any():
                continue
            part = {k: v[rows] for k, v in d.items()}
            events[split].update(np.unique(part["event_idx"]).tolist())
            raw = make_features(part, layers, lst)
            is_real = part["is_real"].astype(bool)
            c = chunks[split]
            c["logits"].append(logits_of(raw))
            c["is_real"].append(is_real)
            if split == TEST:
                c["layer"].append(layer_index(part))
                c["pls_pt"].append(part["pls_pt"].astype(np.float32))
                c["pls_eta"].append(part["pls_eta"].astype(np.float32))
                sample = is_real | (rng.random(len(is_real)) < keep_fake_prob)
                c["sample_features"].append(raw[sample])
                c["sample_is_real"].append(is_real[sample])

    out = {"n_events": {"val": len(events[VAL]), "test": len(events[TEST])}}
    for split, name in ((VAL, "val"), (TEST, "test")):
        for k, v in chunks[split].items():
            if v:
                out[f"{name}_{k}"] = np.concatenate(v)
    return out


# ---------------------------
# Working points and curves
# ---------------------------
class Curves:
    """Real efficiency and fake rate for any cut on the logit, from the sorted logits of one split."""
    def __init__(self, logits, is_real):
        self.real = np.sort(logits[is_real])
        self.fake = np.sort(logits[~is_real])

    def eff(self, cut):
        return 1 - np.searchsorted(self.real, cut, side="left") / len(self.real)

    def fake_rate(self, cut):
        return 1 - np.searchsorted(self.fake, cut, side="left") / len(self.fake)

    def cut_grid(self):
        """Cuts dense both where the real efficiency changes and where the fake rate is small."""
        cuts = np.concatenate([np.quantile(self.real, np.linspace(0, 1, 3001)),
                               np.quantile(self.fake, 1 - np.logspace(-8, 0, 3001))])
        return np.unique(cuts)

    def auc(self):
        """Exact ROC AUC: P(real logit > fake logit), ties count half."""
        below = np.searchsorted(self.fake, self.real, side="left")
        ties  = np.searchsorted(self.fake, self.real, side="right") - below
        return float((below.sum() + 0.5 * ties.sum()) / (len(self.real) * len(self.fake)))


def working_points(val_logits, val_is_real, test_logits, test_is_real, targets):
    """Score cuts from the validation reals, measured on the test events (score >= cut passes)."""
    val_real_scores = sigmoid32(val_logits[val_is_real])
    test_scores = sigmoid32(test_logits)
    real_sorted = np.sort(test_scores[test_is_real])
    fake_sorted = np.sort(test_scores[~test_is_real])
    del test_scores

    results = {}
    for t in sorted(targets):
        cut = float(np.quantile(val_real_scores, 1 - t))
        reals = int(len(real_sorted) - np.searchsorted(real_sorted, np.float32(cut), side="left"))
        fakes = int(len(fake_sorted) - np.searchsorted(fake_sorted, np.float32(cut), side="left"))
        fake_rate = fakes / len(fake_sorted)
        results[f"{t * 100:.1f}"] = {
            "real_efficiency":      t,                         # target, on the validation events
            "threshold":            cut,                       # pass if NN score >= threshold
            "threshold_from":       "val",
            "test_real_efficiency": reals / len(real_sorted),  # measured on the test events
            "fake_rate":            fake_rate,                 # measured on the test events
            "fake_rate_pct":        round(fake_rate * 100, 6),
            "purity":               reals / max(1, reals + fakes),
            "test_reals_passing":   reals,
            "test_fakes_passing":   fakes,
            "est_abs_fakes_1p2B":   round(fake_rate * TOTAL_FAKES_ESTIMATE, 0),
        }
    return results


# ---------------------------
# Plots
# ---------------------------
def save(fig, plot_dir, name):
    fig.savefig(os.path.join(plot_dir, name))
    plt.close(fig)


def wp_vlines(ax, key_wps):
    """Vertical dashed line at each key working point's logit cut (named in the legend)."""
    for (name, wp), color in zip(key_wps, wp_colors(len(key_wps))):
        ax.axvline(logit(wp["threshold"]), color=color, lw=1.3, ls="--", zorder=1,
                   label=f"{name} cut (score ≥ {wp['threshold']:.6f})")


def legend_right(ax):
    """Legend outside the axes, on the right, so it never covers the data."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)


def logit_range(logits, is_real):
    """x range of the logit plots: from the bulk of the fakes to the top of the reals
    (a few fakes reach logits of -1000 and beyond, which would squash everything else)."""
    return np.percentile(logits[~is_real], 1), np.percentile(logits[is_real], 99.99)


def plot_training(history, plot_dir):
    """Loss, val AUC, val fake rate at 99% efficiency and learning rate per epoch (history.json)."""
    epochs = np.arange(len(history["train_loss"]))
    if history.get("select_eff") is not None and history.get("val_fake_rate_sel"):   # the epoch of model_best.pt
        best = int(np.argmin(history["val_fake_rate_sel"]))
    else:
        best = int(np.argmax(history["val_auc"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    (ax_loss, ax_auc), (ax_fr, ax_lr) = axes

    ax_loss.plot(epochs, history["train_loss"], color=REAL_COLOR, label="training")
    if history.get("val_loss"):
        ax_loss.plot(epochs, history["val_loss"], color=FAKE_COLOR, label="validation")
    ax_loss.set(title="Loss (BCE + gap penalty)", xlabel="Epoch", ylabel="Loss")
    ax_loss.legend()

    ax_auc.plot(epochs, history["val_auc"], color=REAL_COLOR, marker="o", ms=3)
    ax_auc.set(title="Validation AUC", xlabel="Epoch", ylabel="AUC")
    ax_auc.ticklabel_format(axis="y", useOffset=False)

    fake_rate_99 = 100 * np.asarray(history["val_fake_rate_99"])
    ax_fr.plot(epochs, fake_rate_99, color=FAKE_COLOR, marker="o", ms=3)
    if fake_rate_99.min() > 0 and fake_rate_99.max() / fake_rate_99.min() > 10:   # log y only over a decade
        ax_fr.set_yscale("log")
    ax_fr.set(title="Validation fake rate at 99% real efficiency", xlabel="Epoch", ylabel="Fake rate (%)")

    ax_lr.plot(epochs, history["lr"], color=MUTED)
    ax_lr.set(title="Learning rate", xlabel="Epoch", ylabel="Learning rate")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for ax in axes.flat:
        ax.axvline(best, color=MUTED, lw=1, ls=":")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_auc.text(best, 0.03, f" saved model (epoch {best})", transform=ax_auc.get_xaxis_transform(),
                color=TEXT_MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, plot_dir, "training_curves.png")


def plot_train_vs_val(history, plot_dir):
    """Training vs validation per epoch, both with dropout off (history.json from train_v7.py):
    loss, AUC, fake rate at the compare efficiency, accuracy and balanced accuracy at score 0.5."""
    if not history.get("train_auc"):
        return
    epochs = np.arange(len(history["train_auc"]))
    eff = history.get("compare_eff", 0.74)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [
        (axes[0, 0], "loss", 1, "Loss (dropout off)", "Loss"),
        (axes[0, 1], "auc", 1, "ROC AUC", "AUC"),
        (axes[1, 0], "fake_rate", 100, f"Fake rate at {eff:.0%} real efficiency", "Fake rate [%]"),
    ]
    for ax, key, scale, title, ylabel in panels:
        ax.plot(epochs, scale * np.asarray(history[f"train_{key}"]), color=REAL_COLOR, marker="o", ms=3, label="training")
        ax.plot(epochs, scale * np.asarray(history[f"val_{key}"]), color=FAKE_COLOR, marker="o", ms=3, label="validation")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.legend()
    ax = axes[1, 1]
    for key, style, name in (("bal_acc", "-", "balanced accuracy"), ("acc", "--", "accuracy")):
        ax.plot(epochs, 100 * np.asarray(history[f"train_{key}"]), style, color=REAL_COLOR, marker="o", ms=3, label=f"training, {name}")
        ax.plot(epochs, 100 * np.asarray(history[f"val_{key}"]), style, color=FAKE_COLOR, marker="o", ms=3, label=f"validation, {name}")
    ax.set(title="Accuracy at score 0.5 (plain accuracy is ~99.7% for 'all fake')", xlabel="Epoch", ylabel="[%]")
    ax.legend()
    for ax in axes.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    save(fig, plot_dir, "train_vs_val.png")


def plot_roc(val_curves, test_curves, results, key_wps, plot_dir):
    """ROC of the validation and test events, log x: the two should agree."""
    fig, ax_log = plt.subplots(figsize=(8, 5))
    for name, color, curves in (("validation", VAL_COLOR, val_curves), ("test", REAL_COLOR, test_curves)):
        cuts = curves.cut_grid()
        ax_log.plot(curves.fake_rate(cuts), curves.eff(cuts), color=color, label=f"{name}: AUC = {curves.auc():.5f}")

    for (name, wp), color, marker in zip(key_wps, wp_colors(len(key_wps)), WP_MARKERS):
        ax_log.plot(wp["fake_rate"], wp["test_real_efficiency"], marker=marker, ms=8, color=color, ls="none",
                    mec="white", mew=1, zorder=4, label=f"{name} cut: fake rate {pct(wp['fake_rate'])}")
    ax_log.set_xscale("log")
    min_fr = min(r["fake_rate"] for r in results.values() if r["fake_rate"] > 0)
    ax_log.set(title="ROC: real efficiency vs fake rate", xlabel="Fake rate (log)", ylabel="Real efficiency",
               xlim=(0.3 * min_fr, 1.2), ylim=(0.5, 1.005))
    ax_log.grid(True, which="minor", alpha=0.4)
    ax_log.legend(loc="lower right")
    save(fig, plot_dir, "roc.png")


def plot_wp_scan(test_curves, results, key_wps, plot_dir):
    """Fake rate vs real efficiency on the test events: the curve, every working point and the key ones."""
    fig, ax = plt.subplots(figsize=(8, 5))
    effs = np.linspace(0.65, 0.9995, 400)
    cuts = np.quantile(test_curves.real, 1 - effs)
    ax.semilogy(100 * test_curves.eff(cuts), 100 * test_curves.fake_rate(cuts), color=REAL_COLOR,
                label="test events, any cut")
    ax.plot([100 * r["test_real_efficiency"] for r in results.values()],
            [100 * r["fake_rate"] for r in results.values()], "o", ms=4, color=MUTED, zorder=3,
            label="working points (cut from validation)")
    for (name, wp), color, marker in zip(key_wps, wp_colors(len(key_wps)), WP_MARKERS):
        ax.plot(100 * wp["test_real_efficiency"], 100 * wp["fake_rate"], marker=marker, ms=9, color=color,
                ls="none", mec="white", mew=1, zorder=4,
                label=f"{name} cut: eff {100 * wp['test_real_efficiency']:.1f}%, fake rate {pct(wp['fake_rate'])}")
    ax.set(title="Fake rate vs real efficiency (test events)", xlabel="Real efficiency (%)",
           ylabel="Fake rate (%)", xlim=(65, 100))
    ax.grid(True, which="minor", alpha=0.4)
    ax.legend(loc="upper left")
    save(fig, plot_dir, "working_point_scan.png")


def plot_score_dist(test_logits, test_is_real, key_wps, plot_dir):
    """Score (0-1) and logit distributions of real and fake test pT2s. Most scores sit within 1e-3 of
    0 or 1, so the logit (the model output before the sigmoid) is where the cuts can be seen."""
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(13, 4.5))
    scores = sigmoid32(test_logits)
    lo, hi = logit_range(test_logits, test_is_real)
    for mask, color, name in ((test_is_real, REAL_COLOR, "real"), (~test_is_real, FAKE_COLOR, "fake")):
        label = f"{name} ({mask.sum():,})"
        ax_s.hist(scores[mask], bins=np.linspace(0, 1, 101), density=True, histtype="stepfilled",
                  color=color, alpha=0.25, label=label)
        ax_s.hist(scores[mask], bins=np.linspace(0, 1, 101), density=True, histtype="step", color=color, lw=1.5)
        ax_l.hist(test_logits[mask], bins=np.linspace(lo, hi, 151), density=True, histtype="stepfilled",
                  color=color, alpha=0.25, label=label)
        ax_l.hist(test_logits[mask], bins=np.linspace(lo, hi, 151), density=True, histtype="step", color=color, lw=1.5)
    del scores
    ax_s.set(title="NN score", xlabel="Score", ylabel="Density (log)", yscale="log")
    ax_s.legend(loc="upper center")
    ax_l.set(title="NN output before the sigmoid, with the cuts", xlabel="logit(score)",
             ylabel="Density (log)", yscale="log", xlim=(lo, hi))
    wp_vlines(ax_l, key_wps)
    legend_right(ax_l)
    fig.tight_layout()
    save(fig, plot_dir, "score_dist.png")


def plot_cut_scan(test_curves, key_wps, plot_dir):
    """Real efficiency and fake rate (test events) as the cut on logit(score) moves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cuts = test_curves.cut_grid()
    lo = np.quantile(test_curves.fake, 0.01)
    cuts = cuts[(cuts >= lo) & (cuts <= test_curves.real[-1])]
    ax.semilogy(cuts, test_curves.eff(cuts), color=REAL_COLOR, label="real efficiency")
    ax.semilogy(cuts, test_curves.fake_rate(cuts), color=FAKE_COLOR, label="fake rate")
    wp_vlines(ax, key_wps)
    ax.set(title="Real efficiency and fake rate vs cut (test events)",
           xlabel="Cut on logit(score)  [score = 1 / (1 + exp(-logit))]", ylabel="Fraction passing (log)")
    ax.grid(True, which="minor", alpha=0.4)
    legend_right(ax)
    save(fig, plot_dir, "cut_scan.png")


def plot_pr(test_curves, key_wps, plot_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    n_real, n_fake = len(test_curves.real), len(test_curves.fake)
    cuts = test_curves.cut_grid()
    reals, fakes = test_curves.eff(cuts) * n_real, test_curves.fake_rate(cuts) * n_fake
    ok = reals + fakes > 0
    ax.plot(test_curves.eff(cuts)[ok], (reals / (reals + fakes))[ok], color=REAL_COLOR)
    for (name, wp), color, marker in zip(key_wps, wp_colors(len(key_wps)), WP_MARKERS):
        ax.plot(wp["test_real_efficiency"], wp["purity"], marker=marker, ms=9, color=color, ls="none",
                mec="white", mew=1, zorder=4, label=f"{name} cut: purity {100 * wp['purity']:.1f}%")
    ax.set(title=f"Purity vs real efficiency (test events, {n_real:,} real / {n_fake:,} fake)",
           xlabel="Real efficiency (recall)", ylabel="Purity (precision)", xlim=(0.5, 1.005), ylim=(0, 1.005))
    ax.legend(loc="lower left")
    save(fig, plot_dir, "pr.png")


def plot_features(features, is_real, plot_dir):
    """Input distributions of real and fake test pT2s (all reals, a random sample of the fakes).
    The one-hot layer flags of a --layers model are in layers.png instead."""
    ncols = 4
    nrows = (len(BASE_FEATURES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows))
    for i, (ax, name) in enumerate(zip(axes.flat, BASE_FEATURES)):
        vals = features[:, i]
        if name in ("pls_charge", "pls_nhit"):   # integers: one bin per value
            bins = np.arange(vals.min() - 0.5, vals.max() + 1.5)
        else:   # 0.5-99.5 percentile range of reals and fakes, so a few outliers do not squash the plot
            lo = min(np.percentile(vals[is_real], 0.5), np.percentile(vals[~is_real], 0.5))
            hi = max(np.percentile(vals[is_real], 99.5), np.percentile(vals[~is_real], 99.5))
            bins = np.linspace(lo, hi, 61) if hi > lo else np.linspace(lo - 0.5, hi + 0.5, 3)
        peak, floor = 0, np.inf
        for mask, color, label in ((is_real, REAL_COLOR, "real"), (~is_real, FAKE_COLOR, "fake")):
            h, _, _ = ax.hist(vals[mask], bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.25,
                              label=label)
            ax.hist(vals[mask], bins=bins, density=True, histtype="step", color=color, lw=1.3)
            if (h > 0).any():
                peak, floor = max(peak, h.max()), min(floor, h[h > 0].min())
        if peak / floor > 100:   # log y only when the densities span more than two decades
            ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(BASE_FEATURES):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Model inputs, test events (each curve normalized to unit area; range: 0.5–99.5 percentiles)", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, plot_dir, "features.png")


def wp_passing(test_logits, key_wps):
    """Per key working point: which test pT2s pass its cut."""
    scores = sigmoid32(test_logits)
    return [scores >= np.float32(wp["threshold"]) for _, wp in key_wps]


def plot_layers(test_layer, test_is_real, passing, key_wps, plot_dir):
    """Per LS layer connection: share of the test pT2s, real efficiency and fake rate at the key cuts."""
    n = len(LAYER_NAMES)
    x = np.arange(n)
    real_all = np.bincount(test_layer[test_is_real & (test_layer >= 0)], minlength=n)
    fake_all = np.bincount(test_layer[~test_is_real & (test_layer >= 0)], minlength=n)
    fig, (ax_n, ax_eff, ax_fr) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    w = 0.38
    ax_n.bar(x - w / 2, 100 * real_all / real_all.sum(), w, color=REAL_COLOR, label=f"real ({real_all.sum():,})")
    ax_n.bar(x + w / 2, 100 * fake_all / fake_all.sum(), w, color=FAKE_COLOR, label=f"fake ({fake_all.sum():,})")
    ax_n.set(title="Share of the test pT2s in each layer connection", ylabel="Share (%, log)", yscale="log")
    legend_right(ax_n)

    offsets = np.linspace(-0.25, 0.25, len(key_wps))
    for (name, _), color, marker, dx, ok in zip(key_wps, wp_colors(len(key_wps)), WP_MARKERS, offsets, passing):
        real_pass = np.bincount(test_layer[test_is_real & ok & (test_layer >= 0)], minlength=n)
        fake_pass = np.bincount(test_layer[~test_is_real & ok & (test_layer >= 0)], minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff, fr = real_pass / real_all, fake_pass / fake_all
        ax_eff.plot(x + dx, 100 * eff, marker=marker, ms=7, ls="none", color=color, mec="white", mew=0.8, label=f"{name} cut")
        fr_plot = np.where(fr > 0, 100 * fr, np.nan)   # a zero fake rate cannot be shown on log y
        ax_fr.plot(x + dx, fr_plot, marker=marker, ms=7, ls="none", color=color, mec="white", mew=0.8, label=f"{name} cut")
    for i in range(n - 1):   # separate the layer columns
        for ax in (ax_n, ax_eff, ax_fr):
            ax.axvline(i + 0.5, color="#ecebe8", lw=0.8, zorder=0)
    for ax in (ax_n, ax_eff, ax_fr):
        ax.grid(False, axis="x")
    ax_eff.set(title="Real efficiency per layer connection (test events)", ylabel="Real efficiency (%)", ylim=(0, 105))
    legend_right(ax_eff)
    ax_fr.set(title="Fake rate per layer connection (test events)", ylabel="Fake rate (%, log)", yscale="log")
    legend_right(ax_fr)
    ax_fr.set_xticks(x, [name.replace("_to_", "→") for name in LAYER_NAMES], rotation=35, ha="right")
    fig.tight_layout()
    save(fig, plot_dir, "layers.png")


def plot_eff_vs(var, name, label, unit, test_is_real, passing, key_wps, plot_dir):
    """Real efficiency and fake rate in bins of a pLS variable, at the key cuts."""
    lo, hi = np.percentile(var[test_is_real], [0.5, 99.5])
    edges = np.linspace(lo, hi, 31)
    centers = 0.5 * (edges[1:] + edges[:-1])
    b = np.digitize(var, edges) - 1
    inside = (b >= 0) & (b < len(centers))
    real_all = np.bincount(b[inside & test_is_real], minlength=len(centers))
    fake_all = np.bincount(b[inside & ~test_is_real], minlength=len(centers))

    fig, (ax_eff, ax_fr) = plt.subplots(1, 2, figsize=(13, 4.8))
    for (wp_name, _), color, marker, ok in zip(key_wps, wp_colors(len(key_wps)), WP_MARKERS, passing):
        real_pass = np.bincount(b[inside & test_is_real & ok], minlength=len(centers))
        fake_pass = np.bincount(b[inside & ~test_is_real & ok], minlength=len(centers))
        with np.errstate(divide="ignore", invalid="ignore"):
            eff, fr = real_pass / real_all, fake_pass / fake_all
        ax_eff.plot(centers, 100 * eff, marker=marker, ms=5, color=color, lw=1.5, label=f"{wp_name} cut")
        ax_fr.plot(centers, np.where(fr > 0, 100 * fr, np.nan), marker=marker, ms=5, color=color, lw=1.5,
                   label=f"{wp_name} cut")
    xlabel = f"{label} ({unit})" if unit else label
    ax_eff.set(title=f"Real efficiency vs {label} (test events)", xlabel=xlabel, ylabel="Real efficiency (%)",
               ylim=(0, 105))
    ax_fr.set(title=f"Fake rate vs {label} (test events)", xlabel=xlabel, ylabel="Fake rate (%, log)", yscale="log")
    handles, labels = ax_eff.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles, labels, loc="upper center", ncol=len(key_wps), bbox_to_anchor=(0.5, 1.0))
    save(fig, plot_dir, f"eff_vs_{name}.png")


# ---------------------------
# Everything
# ---------------------------
def evaluate(model, mean, std, files, val_frac, test_frac, layers, device, output_dir,
             history=None, targets=DEFAULT_TARGETS, key_effs=DEFAULT_KEY_EFFS, lst=False):
    """Score the validation and test events, write <output_dir>/metrics.json and <output_dir>/plots/.
    Returns (sample features, sample is_real) of the test events, for SHAP."""
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Scoring the validation and test events of {len(files)} file(s)...", flush=True)
    s = score_splits(model, mean, std, files, val_frac, test_frac, layers, device, lst=lst)
    val_logits, val_is_real = s["val_logits"], s["val_is_real"]
    test_logits, test_is_real = s["test_logits"], s["test_is_real"]
    print(f"  validation: {s['n_events']['val']} events, {val_is_real.sum():,} real, {(~val_is_real).sum():,} fake pT2s")
    print(f"  test:       {s['n_events']['test']} events, {test_is_real.sum():,} real, {(~test_is_real).sum():,} fake pT2s")

    results = working_points(val_logits, val_is_real, test_logits, test_is_real, targets)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Working points (cut from the validation events, measured on the test events) ===")
    print(f"{'target':>7}  {'cut (score >=)':>18}  {'test eff':>8}  {'fake rate':>10}  {'purity':>6}")
    for name, r in results.items():
        print(f"{name:>6}%  {r['threshold']:18.12f}  {100 * r['test_real_efficiency']:7.2f}%  "
              f"{100 * r['fake_rate']:9.5f}%  {100 * r['purity']:5.1f}%")
    print(f"Saved {os.path.join(output_dir, 'metrics.json')}")

    # Working points drawn on the plots: the closest computed one to each key efficiency
    key_names = sorted({min(results, key=lambda k: abs(results[k]["real_efficiency"] - e)) for e in key_effs},
                       key=float)
    key_wps = [(f"{float(k):g}%", results[k]) for k in key_names]

    print("Making plots...", flush=True)
    val_curves, test_curves = Curves(val_logits, val_is_real), Curves(test_logits, test_is_real)
    print(f"Test AUC: {test_curves.auc():.6f}   validation AUC: {val_curves.auc():.6f}")
    if history:
        plot_training(history, plot_dir)
        plot_train_vs_val(history, plot_dir)
    plot_roc(val_curves, test_curves, results, key_wps, plot_dir)
    del val_curves
    plot_wp_scan(test_curves, results, key_wps, plot_dir)
    plot_score_dist(test_logits, test_is_real, key_wps, plot_dir)
    plot_cut_scan(test_curves, key_wps, plot_dir)
    plot_pr(test_curves, key_wps, plot_dir)
    plot_features(s["test_sample_features"], s["test_sample_is_real"], plot_dir)
    passing = wp_passing(test_logits, key_wps)
    plot_layers(s["test_layer"], test_is_real, passing, key_wps, plot_dir)
    plot_eff_vs(s["test_pls_eta"], "pls_eta", "pLS η", "", test_is_real, passing, key_wps, plot_dir)
    plot_eff_vs(s["test_pls_pt"], "pls_pt", "pLS $p_T$", "GeV", test_is_real, passing, key_wps, plot_dir)
    print(f"Saved plots to {plot_dir}")
    return s["test_sample_features"], s["test_sample_is_real"]
