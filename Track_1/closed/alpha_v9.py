"""Re-sweep the fluctuation-rescaling strength ALPHA under v9 scoring.

    python alpha_v9.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

ALPHA rescales each pixel's temporal fluctuation toward the variance head's prediction.  It
buys TKE and pays for it in rel-L2 *and* in per-element error, because inflating a fluctuation
the model got only approximately right moves the prediction further from the target.  Under v7
that third cost was nearly invisible: SPS ran through a logistic whose slope near the operating
point was ~23 per unit of aggregate, so a per-element error increase barely registered and 0.75
came out on top.  v9 made SPS the identity, slope 100.  The same error increase now costs 4.4x
as much, which is exactly the kind of change that moves an optimum.

The band is refitted at every alpha, because a model with larger per-element errors wants wider
intervals and comparing a swept alpha against a band fitted at one of them would confound the
two.  `oracle` is the per-sample-per-element upper bound: it sets h_ie = |E_ie|, i.e. perfect
knowledge of each individual error, and bounds what ANY uncertainty quantification could reach.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import sys

import numpy as np

import sps9
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def oracle_sps(A, a, scored, n_scored):
    """h_ie = |E_ie| per sample AND element: the ceiling for any per-window adaptive band."""
    contrib = np.where(scored, np.exp(-2.0 * A / sps9.SIGMA), np.float32(0.0))
    return float(100.0 * (a @ contrib.reshape(len(A), -1).sum(axis=1, dtype=np.float64))
                 / n_scored)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
    ap.add_argument("--ms", type=float, default=1.190,
                    help="end-to-end ms/window; only shifts every row by the same constant")
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    t_score = sc.score_time(args.ms / 1000.0)

    hdr = (f"{'alpha':>5s} | {'rel':>6s} {'tke':>6s} {'mvpe':>6s} {'time':>6s} {'sps':>6s} | "
           f"{'mean5':>7s} | {'cov':>5s} {'acc':>6s} {'oracle':>6s}")
    print(hdr)
    print("-" * len(hdr))

    for alpha in args.alphas:
        Ps, Ys = infer(models, ds["x"], dev, alpha), ds["y"]
        Pu, Yu = infer(models, du["x"], dev, alpha), du["y"]
        S, U = prep(Ps, Ys), prep(Pu, Yu)

        # Fit the band on val_seen only. The exact per-element optimum beat every parametric
        # family in band_v9.py, so it is the one used here.
        band = optimal_band(*S[:3])
        s = sps_score(*U, band)

        r = float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2)))
        m = sc.mvpe_rel_l2(Pu, Yu)
        five = [sc.score_error(r), sc.score_error(k), sc.score_error(m), t_score, s]
        cov = float(np.count_nonzero((U[0] <= band) & U[2]) / U[3])
        print(f"{alpha:5.2f} | {five[0]:6.2f} {five[1]:6.2f} {five[2]:6.2f} {five[3]:6.2f} "
              f"{five[4]:6.2f} | {np.mean(five):7.3f} | {cov:5.3f} {100*U[1].mean():6.2f} "
              f"{oracle_sps(*U):6.2f}")


if __name__ == "__main__":
    main()
