"""Does a fitted per-pixel fluctuation gain beat the variance head's own prediction?

    python fluc_calib.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

ALPHA already rescales each pixel's temporal fluctuation toward the variance head's per-window
prediction.  That is an adaptive correction, but it is only as good as the head, and the head is
trained on a log-std loss rather than on what TKE actually measures.  A static multiplicative
gain g(y, x, c) fitted offline on val_seen corrects any *systematic* bias left over -- places
where the head is reliably too confident or too timid -- at the cost of one array multiply that
folds into the existing GPU compose for free.

Fitted on val_seen, reported on val_unseen.  Three forms, increasing in freedom:

  scalar        one number
  per-pixel     g(y, x, c)
  per-pixel/t   g(t, y, x, c), the most a static array can express

The objective is mean5, not TKE, because inflating the fluctuation buys TKE and pays in rel-L2
and in per-element error (hence SPS); optimizing TKE alone would just rediscover ALPHA = 1.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import sps9
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def apply_gain(P, g):
    """Scale each pixel's fluctuation about its own time-mean by g, leaving the mean alone."""
    m = P.mean(axis=1, keepdims=True)
    return (m + (P - m) * g).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--ms", type=float, default=1.390)
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    Ps, Ys = infer(models, ds["x"], dev, args.alpha), ds["y"]
    Pu, Yu = infer(models, du["x"], dev, args.alpha), du["y"]
    t_score = sc.score_time(args.ms / 1000.0)

    def mean5(P, Y, band=None):
        S = prep(P, Y)
        if band is None:
            band = optimal_band(np.float32(1.5) * S[0], S[1], S[2])
        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        mv = sc.mvpe_rel_l2(P, Y)
        five = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), t_score,
                sps_score(*S, band)]
        return float(np.mean(five)), five, band

    base_s, _, _ = mean5(Ps, Ys)
    base_u, five_u, _ = mean5(Pu, Yu)
    print(f"baseline (ALPHA={args.alpha})   seen {base_s:.3f}   unseen {base_u:.3f}   "
          f"[rel {five_u[0]:.2f} tke {five_u[1]:.2f} mvpe {five_u[2]:.2f} sps {five_u[4]:.2f}]")

    # --- scalar
    best = max(((mean5(apply_gain(Ps, g), Ys)[0], g) for g in np.linspace(0.85, 1.30, 19)),
               key=lambda z: z[0])
    v_u, f_u, _ = mean5(apply_gain(Pu, best[1]), Yu)
    print(f"scalar g={best[1]:.3f}          seen {best[0]:.3f}   unseen {v_u:.3f}   "
          f"[rel {f_u[0]:.2f} tke {f_u[1]:.2f} mvpe {f_u[2]:.2f} sps {f_u[4]:.2f}]")

    # --- per-pixel and per-pixel/time, from the ratio of true to predicted fluctuation std.
    # This is the least-squares gain for the fluctuation amplitude, then damped toward 1 by
    # `w` because the scored optimum is not the TKE optimum.
    def ratio(P, Y, axes):
        fp = P - P.mean(axis=1, keepdims=True)
        fy = Y - Y.mean(axis=1, keepdims=True)
        num = np.sqrt((fy ** 2).mean(axis=axes, keepdims=True))
        den = np.sqrt((fp ** 2).mean(axis=axes, keepdims=True)) + 1e-8
        return (num / den).astype(np.float32)

    for name, axes in (("per-pixel  g(y,x,c)", (0, 1)), ("per-pixel/t g(t,y,x,c)", (0,))):
        rs = ratio(Ps, Ys, axes)
        best = max(((mean5(apply_gain(Ps, 1 + w * (rs - 1)), Ys)[0], w)
                    for w in np.linspace(0.0, 1.0, 11)), key=lambda z: z[0])
        w = best[1]
        v_u, f_u, _ = mean5(apply_gain(Pu, 1 + w * (rs - 1)), Yu)
        print(f"{name} damp {w:.1f}  seen {best[0]:.3f}   unseen {v_u:.3f}   "
              f"[rel {f_u[0]:.2f} tke {f_u[1]:.2f} mvpe {f_u[2]:.2f} sps {f_u[4]:.2f}]")


if __name__ == "__main__":
    main()
