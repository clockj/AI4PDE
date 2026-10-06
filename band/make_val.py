"""Build the two validation window sets the band fit reads, from the released real data.

    python make_val.py --data <train_real> --out <dir>      # writes val_unseen.npz, val_seen.npz

UNSEEN: the eight trajectories every twin holds out (`unet/data.py:VAL_TRAJ`).
SEEN:   eight trajectories the twins trained on, matched in Re / AoA spread.
Windows are 20 + 20 frames at stride 20, the competition's protocol; the third channel (p) is
zero-filled, as the scorer expects.
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unet"))
from data import SUB, T_IN, T_OUT, VAL_TRAJ  # noqa: E402

INTERVAL = 20
HORIZON = T_IN + T_OUT
UNSEEN = list(VAL_TRAJ)
SEEN = ["5025_0", "7575_10", "10125_20", "12675_20",
        "17775_10", "19050_5", "22875_15", "25425_0"]


def build(data: str, names: list[str], out: str) -> None:
    xs, ys, ids = [], [], []
    for name in names:
        with h5py.File(os.path.join(data, f"{name}.h5"), "r") as f:
            u = f["u"][:, ::SUB, ::SUB].astype(np.float32)
            v = f["v"][:, ::SUB, ::SUB].astype(np.float32)
        d = np.stack([u, v, np.zeros_like(u)], -1)
        for t in range(0, len(d) - HORIZON + 1, INTERVAL):
            xs.append(d[t:t + T_IN])
            ys.append(d[t + T_IN:t + HORIZON])
            ids.append(f"{name}:{t}")
    np.savez_compressed(out, x=np.asarray(xs, np.float32), y=np.asarray(ys, np.float32),
                        ids=np.array(ids))
    print(f"{out}: {len(xs)} windows from {len(names)} trajectories", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the train_real directory of the release")
    ap.add_argument("--out", required=True, help="directory for val_unseen.npz and val_seen.npz")
    a = ap.parse_args()
    assert not set(SEEN) & set(UNSEEN)
    os.makedirs(a.out, exist_ok=True)
    build(a.data, UNSEEN, os.path.join(a.out, "val_unseen.npz"))
    build(a.data, SEEN, os.path.join(a.out, "val_seen.npz"))
