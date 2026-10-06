"""The evaluator's normalisation and a real-data stream for the local gates.

The stream is built the way the official `local_eval.build_stream` builds it: trajectories in
sorted order, windows of 20 + 20 frames at stride 20, `is_first` marking the step at which the
platform calls `reset_ttt_state`. Raw physical units; the gates normalise as the evaluator does.
"""
from __future__ import annotations

import os

import numpy as np

T_IN = T_OUT = 20
INTERVAL = 20
SUB_S = 2
HORIZON = T_IN + T_OUT

# example_data/mean_std_real.pt of the release: the input statistics (what the evaluator
# normalises the input window with) and the target statistics (what it de-normalises the
# prediction with).
MEAN_IN = np.array([0.15496085584163666, -0.0005139928543940187, 0.0], dtype=np.float64)
STD_IN = np.array([0.09680565446615219, 0.015960684046149254, 1.0], dtype=np.float64)
MEAN_TGT = np.array([0.15496256947517395, -0.0005177936982363462, 0.0], dtype=np.float64)
STD_TGT = np.array([0.09681040793657303, 0.015963643789291382, 1.0], dtype=np.float64)

# The eight held-out trajectories (unet/data.py:VAL_TRAJ): what the gates stream.
UNSEEN = ["3750_0", "6300_10", "8850_20", "11400_20",
          "13950_10", "16500_5", "21600_15", "26700_0"]


def data_dir() -> str:
    """The `train_real` directory: `RPDE_REAL` (solution.py sets it), else resolved from `RPDE`."""
    d = os.environ.get("RPDE_REAL")
    if d:
        return d
    root = os.environ.get("RPDE")
    if not root:
        raise SystemExit("set RPDE_REAL to the directory holding train_real's *.h5 files")
    for d in (os.path.join(root, "train_real"), os.path.join(root, "train_real", "train_real")):
        if os.path.exists(d):
            return d
    raise SystemExit(f"no train_real under {root}")


def build_stream(names):
    """[(name, time_id, x_raw (20,32,64,3), y_raw (20,32,64,3), is_first)] in evaluator order."""
    import h5py

    data = data_dir()
    out = []
    prev = None
    for name in sorted(names):
        with h5py.File(os.path.join(data, f"{name}.h5"), "r") as f:
            u = f["u"][:, ::SUB_S, ::SUB_S].astype(np.float32)
            v = f["v"][:, ::SUB_S, ::SUB_S].astype(np.float32)
        d = np.stack([u, v, np.zeros_like(u)], -1)
        n = len(d)
        for t in range(0, n - HORIZON + 1, INTERVAL):
            out.append((name, t, d[t:t + T_IN], d[t + T_IN:t + HORIZON], name != prev))
            prev = name
    return out


def normalize_in(x_raw: np.ndarray) -> np.ndarray:
    return ((x_raw - MEAN_IN) / STD_IN).astype(np.float32)


def denorm_pred(pred_norm: np.ndarray) -> np.ndarray:
    return (pred_norm * STD_TGT + MEAN_TGT).astype(np.float32)
