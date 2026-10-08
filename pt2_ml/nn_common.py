"""
Pieces shared by train_v7.py and eval_wp.py: the model inputs, the event split and the model.

They must match src/pt2_scorer.cc (inputs, in order) and src/event_split.h (split rule and seed),
or the C++ scores / `-s test` events will not match the trained model.
"""
import glob
import os

import numpy as np
import torch.nn as nn

LOG_EPS    = 1e-6
SPLIT_SEED = 12345
TRAIN, VAL, TEST = 0, 1, 2   # event_split() labels

BASE_FEATURES = [
    "ls_pt", "ls_eta", "ls_sin_phi", "ls_cos_phi",
    "pls_pt", "pls_eta", "pls_sin_phi", "pls_cos_phi",
    "pls_charge", "pls_nhit",
    "pt2_delta_pt", "pt2_delta_eta", "pt2_delta_phi", "pt2_delta_R",
    "pt2_md0_dxy", "pt2_md0_dz",
    "pt2_md1_dxy", "pt2_md1_dz",
    "pt2_md0_rz",  "pt2_md1_rz",
    "log_abs_md0_dxy",
]

# LS layer connections, in the order of Histograms::catNames in src/histograms.h.
# A --layers model has their one-hot flags (the is_<layer> branches) as extra inputs after BASE_FEATURES.
LAYER_NAMES = [
    "L1F_to_L2F", "L1F_to_L2T", "L1T_to_L2F", "L1T_to_L2T", "L1T_to_E1PS",
    "L2F_to_L3F", "L2F_to_L3T", "L2T_to_L3F", "L2T_to_L3T", "L2T_to_E1PS",
    "E1PS_to_E2PS", "E1PS_to_E22S", "E2PS_to_E3PS",
]
LAYER_FEATURES = ["is_" + name for name in LAYER_NAMES]

# LST pT3-style variables already in the training ntuple (--lst_vars). They go between BASE_FEATURES and
# the layer flags, so the flags stay last. The two z residuals are <= LST_SENTINEL when LST could not
# compute them (~5% of pT2s): they are then set to 0 and lst_zRes_valid = 0.
LST_BRANCHES = ["lst_dPhi", "betaIn", "betaOut", "dBeta", "lst_zResGeo", "lst_zResKin", "dAngle"]
LST_FEATURES = LST_BRANCHES + ["lst_zRes_valid"]
LST_SENTINEL = -99.0

BASE_BRANCHES = [
    "event_idx", "is_real",
    "ls_pt", "ls_eta", "ls_phi", "pls_pt", "pls_eta", "pls_phi", "pls_charge", "pls_nhit",
    "pt2_delta_pt", "pt2_delta_eta", "pt2_delta_phi", "pt2_delta_R",
    "pt2_md0_dxy", "pt2_md0_dz", "pt2_md1_dxy", "pt2_md1_dz", "pt2_md0_rz", "pt2_md1_rz",
]


def feature_names(layers, lst=False):
    return BASE_FEATURES + (LST_FEATURES if lst else []) + (LAYER_FEATURES if layers else [])


def branches(layers, lst=False):
    """Branches to read for these inputs."""
    return BASE_BRANCHES + (LST_BRANCHES if lst else []) + (LAYER_FEATURES if layers else [])


def inputs_from_n_features(n):
    """(layers, lst): which optional inputs a model with n inputs (len of its mean.npy) was trained with."""
    for layers in (False, True):
        for lst in (False, True):
            if n == len(feature_names(layers, lst)):
                return layers, lst
    raise SystemExit(f"model has {n} inputs, which matches no combination of --layers / --lst_vars")


def layers_from_n_features(n):
    """Whether a model with n inputs (len of its mean.npy) was trained with --layers."""
    return inputs_from_n_features(n)[0]


def event_split(event_idx, val_frac, test_frac):
    """TRAIN, VAL or TEST per row; deterministic per event index (same rule as src/event_split.h)."""
    u = (event_idx.astype(np.uint64) * np.uint64(2654435761) + np.uint64(SPLIT_SEED)) % np.uint64(1_000_003)
    u = u.astype(np.float64) / 1_000_003
    return np.where(u < test_frac, TEST, np.where(u < test_frac + val_frac, VAL, TRAIN))


def lst_columns(d):
    """The LST_FEATURES columns: z residual sentinels -> 0, plus their validity flag."""
    zgeo, zkin = d["lst_zResGeo"], d["lst_zResKin"]
    valid = (zgeo > LST_SENTINEL) & (zkin > LST_SENTINEL)
    zero = np.float32(0)
    return [d["lst_dPhi"], d["betaIn"], d["betaOut"], d["dBeta"],
            np.where(valid, zgeo, zero), np.where(valid, zkin, zero), d["dAngle"], valid.astype(np.float32)]


def make_features(d, layers, lst=False):
    """Model inputs, in the same order as Pt2Scorer::features() in src/pt2_scorer.cc
    (21, then the 8 LST_FEATURES with lst=True, then the 13 one-hot layer flags with layers=True).
    """
    ls_phi, pls_phi = d["ls_phi"], d["pls_phi"]
    md0_dxy = d["pt2_md0_dxy"]
    x = np.stack([
        d["ls_pt"],  d["ls_eta"],
        np.sin(ls_phi), np.cos(ls_phi),
        d["pls_pt"], d["pls_eta"],
        np.sin(pls_phi), np.cos(pls_phi),
        d["pls_charge"], d["pls_nhit"],
        d["pt2_delta_pt"],  d["pt2_delta_eta"],
        d["pt2_delta_phi"], d["pt2_delta_R"],
        md0_dxy,            d["pt2_md0_dz"],
        d["pt2_md1_dxy"],   d["pt2_md1_dz"],
        d["pt2_md0_rz"],    d["pt2_md1_rz"],
        np.log(np.abs(md0_dxy) + np.float32(LOG_EPS)),
    ] + (lst_columns(d) if lst else []) + ([d[name] for name in LAYER_FEATURES] if layers else []),
        axis=1).astype(np.float32)
    # Pt2Scorer replaces non-finite inputs by 0 before normalizing; do the same
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def layer_index(d):
    """Index in LAYER_NAMES of each row's layer connection (-1 if none), from the is_<layer> branches."""
    flags = np.stack([d[name] for name in LAYER_FEATURES], axis=1)
    return np.where(flags.any(axis=1), flags.argmax(axis=1), -1).astype(np.int8)


def find_files(patterns):
    """.root files from files, directories or glob patterns."""
    files = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            files += glob.glob(os.path.join(pattern, "*.root"))
        else:
            files += glob.glob(pattern)
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No .root files found in {patterns}")
    return files


class Model(nn.Module):
    """
    v7 architecture:
      - Block 1: Linear(input→256) + BN + ReLU + Dropout(0.2)
      - Block 2: Linear(256→128)   + BN + ReLU + Dropout(0.1)
      - Residual projection: Linear(input→128, no bias) added after Block 2
      - Block 3: Linear(128→64)    + ReLU
      - Head:    Linear(64→1)      (outputs the logit; score = sigmoid(logit))

    The residual projection gives gradients a direct path back to the input,
    which helps sharpen score separation at high efficiency working points.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2))
        self.block2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1))
        # Residual projection: maps raw input directly to 128-dim space
        self.residual_proj = nn.Linear(input_dim, 128, bias=False)
        self.block3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out) + self.residual_proj(x)   # residual add
        out = self.block3(out)
        return self.head(out).squeeze(-1)
