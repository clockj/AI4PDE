"""Shared Track-2 pieces: the evaluator's normalizer, and a real-data stream builder.

The kit's ``local_eval.py`` only knows how to read ``example_data``, which is SYNTHETIC --
``make_example.py`` writes two files whose means agree to seven digits, so its rel-L2 is a
plumbing check and nothing else.  Everything here reads ``train_real`` instead, so a Track-2
number is comparable to a Track-1 one on the same trajectories.

The stream is built exactly the way ``local_eval.build_stream`` builds it (sorted by
``(filename, time_id)``, stride ``INTERVAL``, tail padded by repeating the last frame), so the
window set is bit-identical to Track 1's ``make_val.py`` on the same trajectories.
"""
from __future__ import annotations

import os

import numpy as np

T_IN = T_OUT = 20
INTERVAL = 20
SUB_S = 2
HORIZON = T_IN + T_OUT
H, W, C = 32, 64, 3

DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"

# example_data/mean_std_real.pt, read 2026-08-31.  mean_in/std_in are bit-identical to
# Track_1/submit_fno/_fno.py's MEAN/STD, so the evaluator's normalized input IS the tensor our
# FNO already expects.  mean_tgt/std_tgt differ from them in the sixth decimal; the conversion
# below is exact rather than approximate because it costs one fused multiply-add.
MEAN_IN = np.array([0.15496085584163666, -0.0005139928543940187, 0.0], dtype=np.float64)
STD_IN = np.array([0.09680565446615219, 0.015960684046149254, 1.0], dtype=np.float64)
MEAN_TGT = np.array([0.15496256947517395, -0.0005177936982363462, 0.0], dtype=np.float64)
STD_TGT = np.array([0.09681040793657303, 0.015963643789291382, 1.0], dtype=np.float64)

# Track-1 validation splits, repeated here so a Track-2 run can be compared window for window.
UNSEEN = ["3750_0", "6300_10", "8850_20", "11400_20",
          "13950_10", "16500_5", "21600_15", "26700_0"]
SEEN = ["5025_0", "7575_10", "10125_20", "12675_20",
        "17775_10", "19050_5", "22875_15", "25425_0"]


def aoa_of(name: str) -> int:
    return int(name.split("_")[1])


def re_of(name: str) -> int:
    return int(name.split("_")[0])


def all_names(data: str = DATA) -> list[str]:
    return sorted(f[:-3] for f in os.listdir(data) if f.endswith(".h5"))


def build_stream(names, data: str = DATA):
    """[(name, time_id, x_raw(20,32,64,3), y_raw(20,32,64,3), is_first)] in evaluator order.

    Raw physical units.  ``is_first`` marks the step at which the platform calls
    ``reset_ttt_state``.
    """
    import h5py  # lazy: never importable from a submission, only from the harness

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
