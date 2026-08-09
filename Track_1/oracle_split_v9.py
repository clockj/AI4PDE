"""Where is the remaining score, by component of the prediction?

    python oracle_split_v9.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

Every prediction splits exactly into three pieces, orthogonal under the time inner product:

    y = mean_t(y)  +  sigma(y) * shape(y)          shape has unit temporal std per pixel

  * `mean_t` is what MVPE reads and most of what rel-L2 reads;
  * `sigma` is what TKE reads, and what compose()/ALPHA already rescale;
  * `shape` is the phase-resolved fluctuation -- the part that needs the model to know where in
    the shedding cycle each pixel will be 20 to 40 frames from now.

Handing the model the TARGET's value for one piece at a time prices each piece's whole
remaining headroom. That is an upper bound no method can beat, so a piece worth little is
closed and a piece worth a lot is where the next round goes. `tke_branch_v9.py` is the sigma
row of this table, worked out in more detail.

Same protocol as everything else: AoA 15-20 held-out proxy, errors at the board's 1.39x, band
refit per row, `final` from the round-9 weights.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from tke_branch_v9 import F_BOARD, score_rows

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def split(A):
    """(mean_t, sigma, shape) with shape carrying unit temporal std."""
    m = A.mean(1, keepdims=True)
    s = A.std(1, keepdims=True)
    return m, s, (A - m) / (s + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=F_BOARD)
    ap.add_argument("--time-score", type=float, default=90.5066)
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hi_s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hi_u = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)

    Ps = infer(models, ds["x"], dev, args.alpha)[hi_s]
    Pu = infer(models, du["x"], dev, args.alpha)[hi_u]
    Ys, Yu = ds["y"][hi_s], du["y"][hi_u]

    def combos(P, Y):
        pm, ps, psh = split(P)
        tm, ts, tsh = split(Y)
        return {
            "base":    P,
            "+mean":   (tm + ps * psh).astype(np.float32),
            "+sigma":  (pm + ts * psh).astype(np.float32),
            "+shape":  (pm + ps * tsh).astype(np.float32),
            "mean+sig": (tm + ts * psh).astype(np.float32),
            "sig+shape": (pm + ts * tsh).astype(np.float32),
            "all":     Y.astype(np.float32),
        }

    cs, cu = combos(Ps, Ys), combos(Pu, Yu)
    print(f"AoA {args.aoa}, {len(Yu)} windows, errors {args.f}x, band refit per row.")
    print("Each row replaces one piece of the prediction with the TARGET's -- an upper bound.")
    score_rows({k: (cs[k], cu[k]) for k in cs}, Ys, Yu, args.f, args.time_score, width=9)


if __name__ == "__main__":
    main()
