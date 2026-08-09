"""Score checkpoints one by one under v9 and print a comparison table.

    python compare_v9.py --ckpt unet/fno_grad.pth unet/fno_ag05.pth unet/fno_cp0.5.pth

Each checkpoint gets its OWN band, fitted on val_seen with sps9.optimal_band and reported on
val_unseen -- a shared band would flatter whichever model the band was fitted to.

`mean5` is the mean of the five subscores.  v9 stopped publishing how final_score is combined
and it is no longer that mean (the 2026-08-05 board reported 79.733 where its own five
subscores average 78.999), so mean5 is a tracking proxy, not the leaderboard's number.  Every
subscore still moves final_score monotonically, so it remains a usable ranking signal.

`wfinal` applies the weighting fitted to three board rounds (final_v9.py) instead of weighting
the five equally, which is the better ranking signal of the two: it prices `time` at ~0.07
rather than 0.20 and `sps` at ~0.18.  It is computed from LOCAL errors, so it is a ranking
number, not a board prediction -- the board runs ~1.39x our per-element error.  Use ens_v9.py,
which inflates, when you want the projection itself.

Run-to-run sigma for this recipe is 0.015 in mean5, from three seeds of fno_grad.  A difference
under ~0.04 is not a result.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--inflate", type=float, default=1.5)
    ap.add_argument("--ms", type=float, default=1.32,
                   help="end-to-end ms/window for a single-member package; identical across "
                        "rows, so it shifts the table rather than reordering it")
    ap.add_argument("--aoa", type=int, nargs="+", default=None,
                    help="restrict val_unseen to these angles of attack. Error is governed by "
                         "AoA and by essentially nothing else -- 15-20 deg carries 2.4x the "
                         "rel-L2 of 5 deg -- and the board reads higher than our worst single "
                         "trajectory, so `--aoa 15 20` is the regime that most resembles what "
                         "it is actually asking. The fit split is left alone: the band still "
                         "comes from all of val_seen.")
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    t_score = sc.score_time(args.ms / 1000.0)
    f = np.float32(args.inflate)

    sel = slice(None)
    if args.aoa:
        aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]])
        sel = np.isin(aoa, args.aoa)
        print(f"val_unseen restricted to AoA {args.aoa}: {int(sel.sum())}/{len(aoa)} windows")

    hdr = (f"{'checkpoint':22s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} {'tke':>6s} "
           f"{'mvpe':>6s} {'time':>6s} {'sps':>6s} | {'mean5':>7s} {'wfinal':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for c in args.ckpt:
        path = c if os.path.isabs(c) else os.path.join(HERE, c)
        if not os.path.exists(path):
            print(f"{os.path.basename(c):22s} (missing)")
            continue
        m = [_fno.load(path, dev)]
        Ps, Ys = infer(m, ds["x"], dev, args.alpha), ds["y"]
        Pu, Yu = infer(m, du["x"], dev, args.alpha)[sel], du["y"][sel]
        S, U = prep(Ps, Ys), prep(Pu, Yu)
        band = optimal_band(f * S[0], S[1], S[2])

        r = float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2)))
        mv = sc.mvpe_rel_l2(Pu, Yu)
        five = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), t_score,
                sps_score(*U, band)]
        print(f"{os.path.basename(c):22s} {r:7.4f} {k:7.4f} {mv:7.4f} | {five[0]:6.2f} "
              f"{five[1]:6.2f} {five[2]:6.2f} {five[3]:6.2f} {five[4]:6.2f} | "
              f"{np.mean(five):7.3f} {final_v9.predict(five)[0]:7.3f}")


if __name__ == "__main__":
    main()
