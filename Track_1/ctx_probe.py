"""How much is TRAJECTORY HISTORY worth, on the project's own 336 val_unseen windows?

`dmd_probe.context_sweep` found a parameter-free per-window linear fit goes 0.1106 -> 0.0851
as its input grows from 20 frames to 180, i.e. -23%, while at the protocol's 20 frames it
LOSES to the trivial input mean. That was on an ad-hoc window set with its own difficulty, so
it could not be compared to anything on record. This repeats it on val_unseen itself, where
persistence is 0.1384, the input time-mean 0.1338 and the shipped ensemble ~0.104.

WHY THIS IS A TRACK-2 QUESTION AND NOT A TRACK-1 ONE. Track 1's `predict()` gets 20-frame
windows and nothing else; reaching outside one costs permutation invariance and leans on the
tiling. Track 2's protocol is the opposite: the official `local_eval.py` holds ONE model
object, calls `ttt_step` in strict time order, and calls `reset_ttt_state()` at trajectory
boundaries -- a mechanism that exists only because within-trajectory state is expected to
persist. Caching frames the stream has already delivered uses PAST measurements only, which is
what a streaming forecaster is. That is categorically different from the leak recorded in
CLAUDE.md, which reads window k+1 -- window k's own answer, bit for bit.

At step k of a trajectory the stream has delivered frames [0, 20k+20). This measures what a
predictor could do with them.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"


def history_for(ids, data=DATA, sub=2):
    """Raw frames of every trajectory named in `ids`, keyed by name, plus each window's t0."""
    import h5py
    names = sorted({str(i).split(":")[0] for i in ids})
    traj = {}
    for nm in names:
        with h5py.File(os.path.join(data, nm + ".h5"), "r") as f:
            traj[nm] = np.stack([f["u"][:, ::sub, ::sub], f["v"][:, ::sub, ::sub]],
                                -1).astype(np.float32)
    t0 = np.array([int(str(i).split(":")[1]) for i in ids])
    who = np.array([str(i).split(":")[0] for i in ids])
    return traj, who, t0


def rel(p, y):
    a = p[..., :2].reshape(len(p), -1).astype(np.float64)
    b = y[..., :2].reshape(len(y), -1).astype(np.float64)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val_unseen")
    ap.add_argument("--ctx", type=int, nargs="+", default=[20, 40, 60, 80, 120, 160, 200])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--aoa", type=int, nargs="+", default=None)
    args = ap.parse_args()

    from dmd_probe import dmd_forecast

    d = np.load(os.path.join(HERE, f"{args.set}.npz"))
    x, y, ids = d["x"], d["y"], d["ids"]
    if args.aoa:
        s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ids], args.aoa)
        x, y, ids = x[s], y[s], ids[s]
    traj, who, t0 = history_for(ids)
    # A window's own input must be bit-identical to the reconstructed slice, or the history
    # is not the same measurement and every row below is meaningless.
    chk = max(float(np.abs(traj[w][t:t + 20] - x[i, :, :, :, :2]).max())
              for i, (w, t) in enumerate(zip(who, t0)))
    print(f"{args.set}: {len(x)} windows over {len(traj)} trajectories; "
          f"reconstruction check {chk:.1e}\n")
    assert chk == 0.0, "reconstructed history does not match the stored window"

    print(f"  {'persistence':28s} {rel(np.repeat(x[:, -1:], 20, 1), y).mean():.4f}")
    print(f"  {'input 20-frame time-mean':28s} "
          f"{rel(np.repeat(x.mean(1, keepdims=True), 20, 1), y).mean():.4f}\n")

    # How much history the STREAM has actually delivered by this window: frames [0, t0+20).
    avail = t0 + 20
    print(f"  available history: min {avail.min()} median {int(np.median(avail))} "
          f"max {avail.max()} frames\n")
    print(f"  {'context':>10s} {'DMD':>9s} {'mean(ctx)':>10s} {'used':>6s}")
    for c in args.ctx:
        pd_, pm_ = [], []
        for i, (w, t) in enumerate(zip(who, t0)):
            lo = max(t + 20 - c, 0)                 # never reaches past the trajectory start
            h = traj[w][lo:t + 20]
            if len(h) < 4:
                h = traj[w][max(t + 20 - 20, 0):t + 20]
            pd_.append(dmd_forecast(h, 20, min(args.rank, len(h) - 2)))
            pm_.append(np.repeat(h.mean(0, keepdims=True), 20, 0))
        pd_ = np.stack(pd_); pm_ = np.stack(pm_)
        used = float(np.mean(np.minimum(avail, c)))
        print(f"  {c:>10d} {rel(pd_, y).mean():9.4f} {rel(pm_, y).mean():10.4f} {used:6.0f}")


if __name__ == "__main__":
    main()
