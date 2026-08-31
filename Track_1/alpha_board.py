"""Re-optimize ALPHA against the BOARD's error scale and the board's flat time_score.

    python alpha_board.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

`alpha_v9.py` swept ALPHA on local errors and kept 0.75. Two things have changed since, and
both push the same way:

1. **time_score does not respond to ALPHA.** ALPHA is one array multiply inside a kernel that
   already runs, so it drops out of the objective. (Round 9 falsified the stronger claim that
   board time is constant in general -- it tracks the number of sequential model invocations,
   which a sweep over ALPHA does not change.)

2. **The board's per-element errors are effectively 1.39x ours** (calib_board.py, inverted from
   a measured board SPS rather than assumed). ALPHA buys TKE and pays in rel-L2 and in
   per-element error -- and that third cost is paid through SPS, whose value curve is not
   linear in the error scale. At local scale SPS peaks near alpha 0.45; at 1.39x the coverage
   term is already deep enough into its tail that a little more fluctuation costs less than it
   does locally, which moves the balance.

So the objective here is over the four subscores that actually respond, evaluated on the AoA
15-20 proxy. `mean4` weights them equally, which is what rounds 6-8 optimized; `final` uses the
EIGHT-round weights (final_v9.py) and is the one to read. They disagree about ALPHA because
mean4 implicitly prices `tke` at 0.25 and `sps` at 0.25, where the eight rounds price them
0.170 and 0.204 -- and ALPHA is precisely a trade between those two.

3. **The weights themselves are uncertain, and ALPHA sits on the axis they are least sure
   about.** The admissible family's dominant spread direction is `rel_l2` against `tke`
   (`sub_value.py`), which is exactly what ALPHA moves -- so a midpoint `final` can rank two
   alphas one way and the family's other end rank them the other. The `dlo`/`dhi` columns are
   `final_v9.delta` against the anchor: **choose on `dlo`, the worst case over the family, not
   on the midpoint.** An alpha that cannot lose is worth more than one that might win.

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

import final_v9
import sps9
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

F_ELEM = 1.39                       # per-element error tail, inverted from the board's SPS
# Round 12's board errors, and the anchor moved with them: that round shipped ALPHA 0.90, so
# the per-metric inflation factors are now defined at the alpha actually in use rather than
# extrapolated from 0.75.  Pair this with the R12 HELD-OUT TWIN set (`fno_vl_s1..s4`,
# `fno_vl_m1220`, `fno_vl_w48`, `fno_vl_w64`, `unet_g1`), whose proxy errors on AoA 15-20 are
# 0.10703 / 0.45114 / 0.09132 -- so the factors this script derives come out at 1.090 / 1.167 /
# 1.543, which is what `proxy_calib.py` measures independently over three rounds.
BOARD = dict(rel=0.116626, tke=0.526278, mvpe=0.140914)   # R12, 2026-08-31


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.45, 0.6, 0.75, 0.9, 1.0, 1.1])
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=F_ELEM)
    ap.add_argument("--time-score", type=float, default=91.8036,
                    help="board time_score for the `final` column; constant down the sweep")
    ap.add_argument("--anchor", type=float, default=0.90,
                    help="the ALPHA the board numbers were measured at; the per-metric "
                         "inflation factors are defined by this row and applied to the rest")
    args = ap.parse_args()
    if args.anchor not in args.alphas:
        args.alphas = sorted(set(args.alphas) | {args.anchor})

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
         for c in args.ckpt]
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
    print(f"`final` uses the EIGHT-round weights with time_score = {args.time_score:.3f}; "
          f"ALPHA does not change the invocation count, so time is constant DOWN this column.")
    print("`dmid/dlo/dhi` are final_v9.delta against the anchor row -- pick on dlo, the worst "
          "case over\nthe admissible weight family, because ALPHA moves along the axis that "
          "family is least sure of.")

    hdr = (f"{'alpha':>5s} | {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} {'tke':>6s} "
           f"{'mvpe':>6s} {'sps':>6s} | {'mean4':>7s} {'final':>7s} | {'dmid':>6s} {'dlo':>6s} "
           f"{'dhi':>6s}")
    print(hdr)
    print("-" * len(hdr))

    subs, rows = {}, []
    for alpha in args.alphas:
        (Af, af, sf, nf), (Au, _, su, nu), (dm, tk, mvv) = fields[alpha]
        band = optimal_band(f * Af, af, sf)
        r, k, mv = (raw[alpha][x] * fac[x] for x in ("rel", "tke", "mvpe"))
        a_b = sps9.weights_from_metrics(fac["rel"] * dm, fac["tke"] * tk, fac["mvpe"] * mvv)
        s_ = sps_score(f * Au, a_b, su, nu, band)
        four = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), s_]
        subs[alpha] = four[:3] + [args.time_score, four[3]]
        rows.append((alpha, r, k, mv, four))

    # Deltas need the anchor row, so they are a SECOND pass -- computing them inline would
    # print 0.000 for every alpha below the anchor, which is exactly the region in question.
    for alpha, r, k, mv, four in rows:
        dmid, dlo, dhi = final_v9.delta(subs[args.anchor], subs[alpha])
        star = "  <-- anchor" if alpha == args.anchor else ""
        print(f"{alpha:5.2f} | {r:7.4f} {k:7.4f} {mv:7.4f} | {four[0]:6.2f} {four[1]:6.2f} "
              f"{four[2]:6.2f} {four[3]:6.2f} | {np.mean(four):7.3f} "
              f"{final_v9.predict(subs[alpha])[0]:7.3f} | {dmid:+6.3f} {dlo:+6.3f} "
              f"{dhi:+6.3f}{star}")

    best = max(subs, key=lambda a: final_v9.delta(subs[args.anchor], subs[a])[1])
    d = final_v9.delta(subs[args.anchor], subs[best])
    print(f"\nminimax over the weight family: ALPHA {best:.2f}, worst case {d[1]:+.3f} against "
          f"the anchor {args.anchor:.2f}\n(midpoint {d[0]:+.3f}, best case {d[2]:+.3f}). A "
          f"change is only worth making if dlo is positive.")

if __name__ == "__main__":
    main()
