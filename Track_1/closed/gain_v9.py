"""WP-D: the fluctuation-rescaling gain, as a 2-parameter family instead of a 1-parameter one.

    python gain_v9.py --ckpt unet/fno_vmlp_lift.pth ...

Inference rescales the predicted fluctuation by

    g = (alpha * sigma + (1 - alpha) * s) / s          s = the raw prediction's own temporal std

and five rounds have swept `alpha`, i.e. the line a + b = 1 of the two-parameter family

    g = (a * sigma + b * s) / s .

Nothing has swept off that line, and there is no reason the optimum should sit on it: `a` is a
calibration of the variance head's SCALE (which is fitted to a log-std MSE and has no reason to
be unbiased at the level the TKE metric charges for) while `b` is how much of the model's own
under-dispersed fluctuation to keep. Both are fitted on val_seen AoA 15-20 and reported on
val_unseen AoA 15-20, and `a + b = 1` is printed alongside so the shipped point is visible.
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import final_v9  # noqa: E402
import sps9  # noqa: E402
from sps9 import optimal_band, prep, sc, sps_score  # noqa: E402

F_BOARD = 1.39


def apply_gain(P0, sigma, a, b):
    """P0 is the alpha=0 field; sigma the head's. Mirrors _fno.predict_fno's recomposition."""
    m = P0.mean(1, keepdims=True)
    s = P0.std(1, keepdims=True)
    g = (np.float32(a) * sigma + np.float32(b) * s) / (s + np.float32(1e-6))
    return (m + (P0 - m) * g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=F_BOARD)
    ap.add_argument("--time-score", type=float, default=91.1209)
    ap.add_argument("--grid-a", type=float, nargs="+",
                    default=[0.0, 0.45, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4])
    ap.add_argument("--grid-b", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.4, 0.55, 0.7])
    args = ap.parse_args()

    import torch
    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c, dev) for c in args.ckpt]

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hs = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hu = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)

    def raw(X):
        sig = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(models, X, dev, alpha=0.0, sigma_out=sig)
        P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
        return P, np.concatenate([sig, np.zeros_like(sig[..., :1])], -1).astype(np.float32)

    P0s, Sgs = raw(ds["x"][hs])
    P0u, Sgu = raw(du["x"][hu])
    Ys, Yu = ds["y"][hs], du["y"][hu]
    f = np.float32(args.f)

    def score(P, Y, band=None, fit=False):
        A, a_, sm, n = prep(P, Y)
        if fit:
            return optimal_band(f * A, a_, sm)
        dm, tk, mv = sps9.sample_metrics(P, Y)
        s = sps_score(f * A, sps9.weights_from_metrics(f * dm, f * tk, f * mv), sm, n, band)
        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        mvv = sc.mvpe_rel_l2(P, Y)
        subs = [sc.score_error(r * args.f), sc.score_error(k * args.f),
                sc.score_error(mvv * args.f), args.time_score, s]
        return r, k, mvv, s, final_v9.predict(subs)[0]

    grid_a = args.grid_a
    grid_b = args.grid_b
    print(f"fit val_seen AoA {args.aoa} ({int(hs.sum())} w), report val_unseen "
          f"({int(hu.sum())} w), inflate {args.f}x\n")
    print(f"{'a':>5s} {'b':>5s} | {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} "
          f"{'tke':>6s} {'mvpe':>6s} {'sps':>6s} | {'final':>7s}")
    rows = []
    for a in grid_a:
        for b in grid_b:
            Ps = apply_gain(P0s, Sgs, a, b)
            Pu = apply_gain(P0u, Sgu, a, b)
            band = score(Ps, Ys, fit=True)
            r, k, mv, s, fin = score(Pu, Yu, band)
            tag = "  <- shipped" if (a, b) == (0.75, 0.25) else ""
            rows.append((fin, a, b))
            print(f"{a:5.2f} {b:5.2f} | {r:7.4f} {k:7.4f} {mv:7.4f} | "
                  f"{sc.score_error(r*args.f):6.2f} {sc.score_error(k*args.f):6.2f} "
                  f"{sc.score_error(mv*args.f):6.2f} {s:6.2f} | {fin:7.3f}{tag}")
        print()
    rows.sort(reverse=True)
    print(f"best  a={rows[0][1]:.2f} b={rows[0][2]:.2f}  final {rows[0][0]:.3f}")
    ship = [x for x in rows if (x[1], x[2]) == (0.75, 0.25)]
    if ship:
        print(f"shipped a=0.75 b=0.25       final {ship[0][0]:.3f}   "
              f"delta {rows[0][0]-ship[0][0]:+.3f}")


if __name__ == "__main__":
    main()
