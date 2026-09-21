"""The time-mean "irreducible floor" is an OVERESTIMATE, and the model already beats it.

`diag_mean_floor.py` estimates the sampling error of a 20-frame mean by splitting the window
into two 10-frame halves and taking 0.5*(m1 - m2).  Its docstring argues that positive
autocorrelation makes the halves less than independent, so the estimate UNDERSTATES the floor.

That direction is right -- corr(m1, m2) about the trajectory mean is measured at **+0.288** --
but it does not rescue the conclusion, because the quantity being bounded is the wrong one.

MEASURED, on the eight held-out trajectories:

    half-split "floor"                       0.0531
    sampling error vs the 868-frame mean     0.0735      (1.38x larger, as +0.288 predicts)
    OUR MODEL's time-mean error              0.0438      <- below BOTH

The model beats the "irreducible floor" at every angle of attack, by 20% at AoA 20 (0.0632
against 0.0830).  It is not a floor.  Neither number is: both measure how much the target's
20-frame mean VARIES, and the model is not asked to guess it blind -- it is handed the
immediately preceding 20 frames, and this flow is quasi-periodic, so a large part of that
variation is phase information the input carries.  The irreducible error is the conditional
variance of `mean_t(y)` given the input window, which neither estimator computes.

**Consequence.**  `CLAUDE.md`'s `rel_l2` ceiling of 97.0-97.4 and round 10's "the time-mean is
already at its sampling floor (0.0503 vs 0.0531 irreducible)" both rest on this number and are
withdrawn.  The time-mean carries 29% of rel-L2's error ENERGY (0.0438 against a total of
0.0814, fluctuation part 0.0685), so driving it to zero would cut rel-L2 by 16% -- roughly
+1.5 `final` at the measured exchange rate, and far above the proxy's ~2% resolution.  That
does not make it easy: `--w-mean 3`, `--w-mean 8` and blending toward the input window's mean
have all been tried and are null.  It makes it OPEN, which is not what the map said.

This measures the two things that settle it:

  1. **corr(m1 - M, m2 - M)** across windows and pixels, with M the trajectory's long-time
     mean.  Negative confirms the anti-correlation and the upward bias.
  2. **rel-L2 of (window mean - trajectory mean)** -- the sampling error computed directly
     against 868 frames instead of inferred from a 20-frame split.  This is the real bound on
     "predict the trajectory's mean flow and stop there", and it is an UPPER bound on the
     irreducible error, because the input window also carries phase information about how this
     particular window's mean deviates from the trajectory's.

    python mean_floor2.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, os.pardir, "Track_2"))
T_IN = T_OUT = 20
INTERVAL = 20
SUB = 2
HORIZON = 40
DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
UNSEEN = ["3750_0", "6300_10", "8850_20", "11400_20",
          "13950_10", "16500_5", "21600_15", "26700_0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", default=UNSEEN)
    ap.add_argument("--pred", default=None, help="npz from stream_eval.py --save, for the model row")
    a = ap.parse_args()
    import h5py

    rows = []
    P = ids = None
    if a.pred:
        z = np.load(a.pred, allow_pickle=True)
        P = z["pred"][..., :2].astype(np.float64)
        ids = [str(i) for i in z["ids"]]

    print(f"{'traj':>10s} {'AoA':>4s} {'corr(h1,h2)':>12s} {'half-split':>11s} "
          f"{'vs traj mean':>13s} {'model':>8s}")
    agg = []
    for name in a.traj:
        with h5py.File(os.path.join(DATA, f"{name}.h5"), "r") as f:
            u = f["u"][:, ::SUB, ::SUB].astype(np.float64)
            v = f["v"][:, ::SUB, ::SUB].astype(np.float64)
        d = np.stack([u, v], -1)                                   # (F, H, W, 2)
        M = d.mean(0)                                              # trajectory long-time mean
        Y = np.stack([d[t + T_IN:t + HORIZON]
                      for t in range(0, len(d) - HORIZON + 1, INTERVAL)])   # (N,20,H,W,2)
        m = Y.mean(1)
        m1, m2 = Y[:, :10].mean(1), Y[:, 10:].mean(1)
        a1, a2 = (m1 - M).ravel(), (m2 - M).ravel()
        corr = float(np.corrcoef(a1, a2)[0, 1])
        norm = np.sqrt((Y ** 2).sum((1, 2, 3, 4)))
        half = float(np.mean(np.sqrt(T_OUT * ((0.5 * (m1 - m2)) ** 2).sum((1, 2, 3))) / norm))
        direct = float(np.mean(np.sqrt(T_OUT * ((m - M) ** 2).sum((1, 2, 3))) / norm))
        mod = np.nan
        if P is not None:
            sel = np.array([i.split(":")[0] == name for i in ids])
            mod = float(np.mean(np.sqrt(T_OUT * ((P[sel].mean(1) - m) ** 2).sum((1, 2, 3)))
                                / norm))
        agg.append((corr, half, direct, mod))
        print(f"{name:>10s} {name.split('_')[1]:>4s} {corr:12.3f} {half:11.4f} "
              f"{direct:13.4f} {mod:8.4f}")
    A = np.array(agg)
    print(f"{'MEAN':>10s} {'':>4s} {A[:,0].mean():12.3f} {A[:,1].mean():11.4f} "
          f"{A[:,2].mean():13.4f} {A[:,3].mean():8.4f}")
    print("\n  corr(h1,h2)   correlation of the two 10-frame half-means about the trajectory\n"
          "                mean. NEGATIVE => the halves are anti-correlated and the half-split\n"
          "                floor is biased UP.\n"
          "  half-split    the floor as diag_mean_floor.py computes it.\n"
          "  vs traj mean  the sampling error measured directly against 868 frames. This is\n"
          "                the real bound on 'predict the trajectory mean and stop', and it is\n"
          "                still an UPPER bound on the irreducible error, because the input\n"
          "                window carries phase information about this window's own deviation.\n"
          "  model         our time-mean error on the same windows.")


if __name__ == "__main__":
    main()
