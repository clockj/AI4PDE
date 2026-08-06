"""Re-optimize ALPHA against the BOARD's error scale and the board's flat time_score.

    python alpha_board.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

`alpha_v9.py` swept ALPHA on local errors and kept 0.75. Two things have changed since, and
both push the same way:

1. **time_score is a constant.** Two board submissions 2.07x apart in local latency returned
   91.697 and 91.726. It does not respond to anything we do, so it drops out of the objective
   entirely rather than sitting in it as a fixed 95.9 that flatters the local mean.

2. **The board's per-element errors are effectively 1.39x ours** (calib_board.py, inverted from
   a measured board SPS rather than assumed). ALPHA buys TKE and pays in rel-L2 and in
   per-element error -- and that third cost is paid through SPS, whose value curve is not
   linear in the error scale. At local scale SPS peaks near alpha 0.45; at 1.39x the coverage
   term is already deep enough into its tail that a little more fluctuation costs less than it
   does locally, which moves the balance.

So the objective here is the mean of the four subscores that actually respond, evaluated on the
AoA 15-20 proxy. `time` is excluded; including a constant would only dilute every difference by
a fifth.

PER-METRIC CALIBRATION
----------------------
One stretch factor for everything is too crude, and the board says so. Anchoring each metric at
the shipped configuration (all-data x3, ALPHA 0.75) against what the board actually returned:

    rel-L2   local 0.1013 -> board 0.11472   1.13x
    TKE      local 0.4293 -> board 0.54687   1.27x
    MVPE     local 0.0868 -> board 0.12970   1.49x
    per-element error tail                   1.39x  (calib_board.py, inverted from board SPS)

Four different numbers. Using 1.39 for all of them, as the first version of this script did,
overstates TKE error by 0.05 and understates MVPE by the same order -- and since ALPHA trades
TKE against rel-L2 directly, getting their relative inflation wrong is exactly the way to get
the answer wrong. Each metric is therefore scaled by its own factor, held fixed across the
sweep.

Two caveats that bound how far to trust this. The anchors come from an all-data model whose
local numbers are contaminated, so the factors' absolute level is too high; what the sweep needs
is their *ratio*, which contamination distorts far less. And one board point cannot show whether
these factors are stable.
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

F_ELEM = 1.39                       # per-element error tail, inverted from the board's SPS
BOARD = dict(rel=0.11472, tke=0.54687, mvpe=0.12970)   # 2026-08-05 evening, all-data x3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.45, 0.6, 0.75, 0.9, 1.0, 1.1])
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=F_ELEM)
    ap.add_argument("--anchor", type=float, default=0.75,
                    help="the ALPHA the board numbers were measured at; the per-metric "
                         "inflation factors are defined by this row and applied to the rest")
    args = ap.parse_args()
    if args.anchor not in args.alphas:
        args.alphas = sorted(set(args.alphas) | {args.anchor})

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev) for c in args.ckpt]
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hi_s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hi_u = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    f = np.float32(args.f)

    # Pass 1: local errors at every alpha. Pass 2 needs the anchor row to set the factors.
    raw, fields = {}, {}
    for alpha in args.alphas:
        Pf, Yf = infer(m, ds["x"], dev, alpha)[hi_s], ds["y"][hi_s]
        Pu, Yu = infer(m, du["x"], dev, alpha)[hi_u], du["y"][hi_u]
        raw[alpha] = dict(rel=float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2))),
                          tke=float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2))),
                          mvpe=sc.mvpe_rel_l2(Pu, Yu))
        fields[alpha] = (prep(Pf, Yf), prep(Pu, Yu), sps9.sample_metrics(Pu, Yu))

    fac = {k: BOARD[k] / raw[args.anchor][k] for k in BOARD}
    print(f"objective: AoA {args.aoa} proxy. per-metric inflation anchored at alpha="
          f"{args.anchor}: " + "  ".join(f"{k} {v:.3f}x" for k, v in fac.items())
          + f"   element tail {args.f:.2f}x")
    print("time excluded: the board returned 91.697 and 91.726 for models 2.07x apart in "
          "local latency")

    hdr = (f"{'alpha':>5s} | {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} {'tke':>6s} "
           f"{'mvpe':>6s} {'sps':>6s} | {'mean4':>7s}")
    print(hdr)
    print("-" * len(hdr))

    for alpha in args.alphas:
        (Af, af, sf, nf), (Au, _, su, nu), (dm, tk, mvv) = fields[alpha]
        band = optimal_band(f * Af, af, sf)
        r, k, mv = (raw[alpha][x] * fac[x] for x in ("rel", "tke", "mvpe"))
        a_b = sps9.weights_from_metrics(fac["rel"] * dm, fac["tke"] * tk, fac["mvpe"] * mvv)
        s = sps_score(f * Au, a_b, su, nu, band)
        four = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), s]
        star = "  <-- anchor" if alpha == args.anchor else ""
        print(f"{alpha:5.2f} | {r:7.4f} {k:7.4f} {mv:7.4f} | {four[0]:6.2f} {four[1]:6.2f} "
              f"{four[2]:6.2f} {four[3]:6.2f} | {np.mean(four):7.3f}{star}")


if __name__ == "__main__":
    main()
