"""How well does the local proxy predict the board? Measure it, instead of assuming a ratio.

`predict_board.py` maps proxy error to board error through fixed per-metric ratios anchored on
ONE round (rel x1.083, tke x1.117, mvpe x1.452).  It called round 11 to +0.006 and round 12 to
-0.191, and the whole round-12 miss was `tke`: predicted 79.869, actual 79.168.  A single
anchor cannot tell a systematic ratio from a lucky one.

This reconstructs the proxy for the held-out twin set of every recent round and puts the two
side by side, in errors rather than scores, so the question becomes answerable:

  * is the board/proxy ratio STABLE across rounds (then the ratio instrument is sound), or
  * does it wander (then only DELTAS are usable), or
  * do proxy deltas fail to transfer at all (then we need a resolution limit, not a ratio)?

The last is the outcome that matters most, because it sets the smallest local difference worth
acting on -- and every round since 2026-08-08 has been smaller than the answer.

    python proxy_calib.py                    # all rounds
    python proxy_calib.py --aoa 15 20        # the board-like subset (default)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
U = os.path.join(HERE, "unet")

# (label, alpha, held-out twin members).  The twins stand in for the shipped all-data package:
# the same recipes trained without the validation trajectories, so their errors on val_unseen
# are predictions rather than memorization.  Round 10's and 11's original twin lists named
# fno_vmlp_lift{,_s2}.pth, which the 2026-08-29 A100 port did not recreate; fno_vl_s1/s2 are
# the same recipe and stand in for them.
S = [f"{U}/fno_vl_s{i}.pth" for i in (1, 2, 3, 4)]
ROUNDS = [
    ("R10 7 memb", 0.75, S + [f"{U}/fno_vl_m1220.pth", f"{U}/fno_vl_w64.pth"]),
    ("R11 8 memb", 0.90, S + [f"{U}/fno_vl_m1220.pth", f"{U}/fno_vl_w48.pth",
                              f"{U}/fno_vl_w64.pth"]),
    ("R12 11 memb", 0.90, S + [f"{U}/fno_vl_m1220.pth", f"{U}/fno_vl_w48.pth",
                               f"{U}/fno_vl_w64.pth", f"{U}/unet_g1.pth"]),
    # R13 shipped ALPHA 0.60, not 0.90 -- the round was submitted precisely to move that
    # constant, so scoring its twins at 0.90 would compare the proxy against a board number
    # the package never produced.
    ("R13 8 memb", 0.60, S + [f"{U}/fno_vl_m1220.pth"]),
]
# Board subscores -> board ERRORS, by inverting score = 100/(1+0.5e).
BOARD = {
    "R10 7 memb":  (94.576394, 79.152828, 93.681162),
    "R11 8 memb":  (94.570878, 79.080459, 93.350128),
    "R12 11 memb": (94.490331, 79.167748, 93.418065),
    "R13 8 memb":  (94.679785, 77.983720, 93.613336),
}


def inv(score):
    return 2.0 * (100.0 / score - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--all", action="store_true", help="also report the full val_unseen split")
    a = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer
    from sps9 import sc

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]])
    subsets = [("AoA " + "/".join(map(str, a.aoa)), np.isin(aoa, a.aoa))]
    if a.all:
        subsets.append(("full", np.ones(len(aoa), bool)))

    out = {}
    for tag, sel in subsets:
        print(f"\n===== proxy = val_unseen {tag} ({int(sel.sum())} windows) =====")
        print(f"{'round':13s} {'memb':>4s} | {'proxy rel':>9s} {'proxy tke':>9s} "
              f"{'proxy mvpe':>10s} | {'board rel':>9s} {'board tke':>9s} {'board mvpe':>10s} "
              f"| {'r_rel':>6s} {'r_tke':>6s} {'r_mvpe':>6s}")
        rows = {}
        for label, alpha, ck in ROUNDS:
            miss = [c for c in ck if not os.path.exists(c)]
            if miss:
                print(f"{label:13s} missing {[os.path.basename(m) for m in miss]}")
                continue
            m = [_fno.load_any(c, dev) for c in ck]
            P, Y = infer(m, du["x"], dev, alpha)[sel], du["y"][sel]
            pr = (float(np.mean(sc.rel_l2_per_sample(P, Y, 2))),
                  float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))),
                  float(sc.mvpe_rel_l2(P, Y)))
            rows[label] = pr
            if label in BOARD:
                b = tuple(inv(v) for v in BOARD[label])
                print(f"{label:13s} {len(ck):4d} | {pr[0]:9.5f} {pr[1]:9.5f} {pr[2]:10.5f} | "
                      f"{b[0]:9.5f} {b[1]:9.5f} {b[2]:10.5f} | "
                      f"{b[0]/pr[0]:6.3f} {b[1]/pr[1]:6.3f} {b[2]/pr[2]:6.3f}")
            else:
                print(f"{label:13s} {len(ck):4d} | {pr[0]:9.5f} {pr[1]:9.5f} {pr[2]:10.5f} | "
                      f"{'-':>9s} {'-':>9s} {'-':>10s} | {'(new)':>21s}")
        out[tag] = rows

        # --- do proxy DELTAS transfer? --------------------------------------------------
        have = [l for l, _, _ in ROUNDS if l in rows and l in BOARD]
        if len(have) >= 2:
            print(f"\n  {'transition':26s} {'metric':6s} {'proxy d%':>9s} {'board d%':>9s} "
                  f"{'residual':>9s}")
            for i in range(len(have) - 1):
                A, B = have[i], have[i + 1]
                for j, mname in enumerate(("rel", "tke", "mvpe")):
                    dp = 100.0 * (rows[B][j] / rows[A][j] - 1.0)
                    bA, bB = inv(BOARD[A][j]), inv(BOARD[B][j])
                    db = 100.0 * (bB / bA - 1.0)
                    print(f"  {A + ' -> ' + B:26s} {mname:6s} {dp:+9.2f} {db:+9.2f} "
                          f"{db - dp:+9.2f}")
            print("\n  `residual` is how much of the board's move the proxy did NOT explain.\n"
                  "  Any local delta smaller than the residual spread is unmeasurable and\n"
                  "  must not be used to choose a package.")


if __name__ == "__main__":
    main()
