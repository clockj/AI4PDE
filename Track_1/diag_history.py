"""Does the trajectory's own past history beat the 20-frame input window as a predictor?

    python diag_history.py

Track 1's predict() receives every test window at once, in (filename, time_id) order. For
window k the windows 0..k-1 of the same trajectory are INPUTS, so their frames are legally
available -- no target is read. That is ordinary transductive inference, and it is a strictly
larger conditioning set than the 20 frames the model currently sees.

It should matter most for the variance field. `tke_score` is bounded near 81 because a
20-frame variance estimate of a turbulent field is intrinsically noisy; the best possible
predictor of the target's realized 20-frame variance is the TRUE variance field, and a few
hundred frames of history estimate that far better than 20 do. Same question for the mean.

Everything here is causal: window k uses frames strictly before its own input window.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_Y = [8, 10, 12, 14, 16, 18, 20, 22, 24]
PROBE_X = [13, 21, 29, 37]


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main() -> None:
    d = np.load(os.path.join(HERE, "val_unseen.npz"))
    X, Y, ids = d["x"][..., :2], d["y"][..., :2], d["ids"]
    traj = np.array([str(i).split(":")[0] for i in ids])

    # Causal history: all input frames of this trajectory strictly before window k, plus
    # window k's own input. Window 0 of a trajectory falls back to its own 20 frames.
    hist_mean = np.empty_like(X[:, 0])
    hist_std = np.empty_like(X[:, 0])
    for name in np.unique(traj):
        w = np.flatnonzero(traj == name)
        for j, k in enumerate(w):
            frames = X[w[:j + 1]].reshape(-1, *X.shape[2:])   # (20*(j+1), H, W, C)
            hist_mean[k] = frames.mean(0)
            hist_std[k] = frames.std(0)

    win_mean, win_std = X.mean(1), X.std(1)
    tgt_mean, tgt_std = Y.mean(1), Y.std(1)

    n = np.array([np.sum(traj[:k + 1] == traj[k]) * 20 for k in range(len(X))])
    print(f"{'='*78}\nhistory available per window: {n.min()} .. {n.max()} frames "
          f"(median {int(np.median(n))})\n{'='*78}")

    print(f"\npredicting the target window's per-pixel 20-frame STD  (drives tke_score)")
    print(f"  {'estimator':38s} {'rel-L2':>8s} {'corr':>7s}")
    for label, est in (("input window std (20 frames)", win_std),
                       ("trajectory history std (causal)", hist_std)):
        print(f"  {label:38s} {rel(est, tgt_std):8.3f} {corr(est, tgt_std):7.3f}")
    print(f"  {'wp2i_s5 variance head (measured)':38s} {0.358:8.3f} {0.907:7.3f}")

    print(f"\npredicting the target window's 20-frame MEAN  (drives mvpe_score)")
    print(f"  {'estimator':38s} {'rel-L2':>8s} {'MVPE':>7s}")
    for label, est in (("input window mean (20 frames)", win_mean),
                       ("trajectory history mean (causal)", hist_mean)):
        cols = []
        for px in PROBE_X:
            pp = est[:, PROBE_Y, px, :].reshape(len(est), -1)
            tt = tgt_mean[:, PROBE_Y, px, :].reshape(len(est), -1)
            cols.append(np.linalg.norm(pp - tt, axis=1) /
                        np.linalg.norm(tt, axis=1).clip(1e-8))
        print(f"  {label:38s} {rel(est, tgt_mean):8.3f} "
              f"{float(np.mean(np.stack(cols))):7.3f}")
    print(f"  {'wp2i_s5 (measured)':38s} {'':8s} {0.079:7.3f}")
    print(f"  {'perfect-true-mean floor (measured)':38s} {'':8s} {0.067:7.3f}")

    # Does history help MORE as it accumulates? If the curve is still falling at 800 frames the
    # estimate is sampling-limited; if it flattens early the flow is not stationary enough for
    # history to buy anything.
    print(f"\nstd rel-L2 by how much history the window has:")
    for lo, hi in ((20, 20), (40, 160), (180, 400), (420, 840)):
        m = (n >= lo) & (n <= hi)
        if m.sum():
            print(f"  {lo:4d}-{hi:4d} frames  (n={m.sum():3d})   window {rel(win_std[m], tgt_std[m]):.3f}"
                  f"   history {rel(hist_std[m], tgt_std[m]):.3f}")


if __name__ == "__main__":
    main()
