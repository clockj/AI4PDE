"""Compare ENSEMBLES under v9 on the board-like high-AoA proxy.

    python ens_v9.py --sets "unet/fno_grad.pth,unet/fno_grad_s2.pth" \
                            "unet/fno_grad.pth,unet/fno_grad_s2.pth,unet/fno_w48.pth"

`compare_v9.py` scores checkpoints one at a time; this scores member SETS, which is what the
submission actually ships. Each set gets its own band, fitted on the AoA 15-20 subset of
val_seen at the board-calibrated 1.39x and reported on the AoA 15-20 subset of val_unseen -- an
ensemble's errors differ from its members', so a shared band would flatter whichever set it was
fitted to.

`mean4` excludes `time`, which no local number can predict. Round 9 falsified the "time is a
constant" reading of the first two board rounds: the 8-member package is FASTER locally than
the U-Net and 2.09 ms slower on the board, because board time tracks sequential model
invocations. Members cost ~0.42 board ms each. `--time-score` feeds that back in so the
`final` column is a whole-board projection -- pass the time_score you actually expect, or the
last board's if the member count is unchanged.

The motivating question is whether a member that is WORSE on its own can still help. Ensembles
average out independent error, so what matters is member diversity as much as member quality --
a width-48 FNO overfits (81.33 alone against the width-24's 81.66) but fails differently, and
that can be worth more than its individual deficit costs.
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

F_BOARD = 1.39


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True,
                    help="comma-separated checkpoint lists, one per ensemble")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=F_BOARD)
    ap.add_argument("--time-score", type=float, default=90.5066,
                    help="board time_score to assume for the `final` projection. The default "
                         "is round 8's, i.e. an 8-member package; subtract ~0.42 board ms per "
                         "member removed and re-score with sps9.sc.score_time.")
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hi_s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hi_u = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    f = np.float32(args.f)
    cache = {}

    hdr = (f"{'members':52s} {'n':>2s} | {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} {'sps':>6s} | "
           f"{'mean4':>7s} {'final':>7s}")
    print(f"AoA {args.aoa} proxy, band refit per set at {args.f}x; `final` uses the round-9 "
          f"weights with time_score = {args.time_score:.3f}")
    print(hdr)
    print("-" * len(hdr))

    for spec in args.sets:
        names = [c.strip() for c in spec.split(",") if c.strip()]
        for c in names:
            if c not in cache:
                p = c if os.path.isabs(c) else os.path.join(HERE, c)
                cache[c] = _fno.load_any(p, dev)
        m = [cache[c] for c in names]
        Pf, Yf = infer(m, ds["x"], dev, args.alpha)[hi_s], ds["y"][hi_s]
        Pu, Yu = infer(m, du["x"], dev, args.alpha)[hi_u], du["y"][hi_u]
        Af, af, sf, _ = prep(Pf, Yf)
        Au, _, su, nu = prep(Pu, Yu)
        band = optimal_band(f * Af, af, sf)
        r = float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2)))
        mv = sc.mvpe_rel_l2(Pu, Yu)
        dm, tk, mvv = sps9.sample_metrics(Pu, Yu)
        a_b = sps9.weights_from_metrics(args.f * dm, args.f * tk, args.f * mvv)
        s = sps_score(f * Au, a_b, su, nu, band)
        four = [sc.score_error(r * args.f), sc.score_error(k * args.f),
                sc.score_error(mv * args.f), s]
        label = ", ".join(os.path.basename(c).replace(".pth", "") for c in names)
        fin = final_v9.predict(four[:3] + [args.time_score, four[3]])[0]
        print(f"{label[:52]:52s} {len(names):2d} | {r:7.4f} {k:7.4f} {mv:7.4f} {s:6.2f} | "
              f"{np.mean(four):7.3f} {fin:7.3f}")


if __name__ == "__main__":
    main()
