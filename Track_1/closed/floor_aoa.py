"""Is there anything left in Track 1's rel-L2, in the regime the BOARD actually asks about?

`diag_mean_floor.py` measured the irreducible sampling floor of the 20-frame time-mean on the
FULL val_unseen split, where our model sits at rel-L2 ~0.086.  But error is governed by angle
of attack and by almost nothing else (0.057 / 0.044 / 0.073 / 0.103 / 0.105 at AoA 0/5/10/15/20)
and the board reads 0.1166 -- above our hardest single trajectory.  A floor measured on the
easy majority does not say whether the hard regime is exhausted.

This measures the floor per AoA, and against the model's own error there, so the answer comes
out as a headroom in per-cent rather than as an abstract bound.  It costs no GPU: the floor
comes from the TARGETS alone, and the model's error comes from a saved prediction.

    python floor_aoa.py --pred <stream_eval/ens_v9 npz with pred+target+ids>
    python floor_aoa.py                       # floor only, no model needed

Method (same half-window trick as `diag_mean_floor.py` and the round-2 TKE floor): split each
20-frame target into two 10-frame halves with means m1, m2.  For a stationary series the halves
are near-independent, so the 20-frame mean's own sampling error is 0.5*(m1 - m2) in rms.  No
predictor can beat that, and because consecutive frames are positively correlated the estimate
UNDERSTATES the floor -- the truth is at least this.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 20


def rel(a, b):
    n = np.sqrt((b ** 2).sum(axis=(1, 2, 3, 4)))
    return np.sqrt(((a - b) ** 2).sum(axis=(1, 2, 3, 4))) / np.clip(n, 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None,
                    help="npz with pred/target/ids from stream_eval.py --save")
    ap.add_argument("--split", default="val_unseen.npz")
    a = ap.parse_args()

    if a.pred:
        z = np.load(a.pred, allow_pickle=True)
        Y = z["target"][..., :2].astype(np.float64)
        P = z["pred"][..., :2].astype(np.float64)
        ids = [str(i) for i in z["ids"]]
    else:
        d = np.load(os.path.join(HERE, a.split))
        Y = d["y"][..., :2].astype(np.float64)
        P = None
        ids = [str(i) for i in d["ids"]]
    aoa = np.array([int(i.split(":")[0].split("_")[1]) for i in ids])

    m1, m2 = Y[:, :10].mean(1), Y[:, 10:].mean(1)
    err = 0.5 * (m1 - m2)                       # rms sampling error of the 20-frame mean
    norm = np.sqrt((Y ** 2).sum((1, 2, 3, 4)))
    floor_mean = np.sqrt(T * (err ** 2).sum((1, 2, 3))) / norm      # per window

    print(f"{'AoA':>4s} {'n':>5s} {'mean floor':>11s} {'model rel':>10s} {'mean part':>10s} "
          f"{'headroom':>9s}")
    for A in sorted(set(aoa.tolist())) + ["ALL"]:
        sel = np.ones(len(Y), bool) if A == "ALL" else (aoa == A)
        f = float(floor_mean[sel].mean())
        if P is None:
            print(f"{str(A):>4s} {int(sel.sum()):5d} {f:11.4f}")
            continue
        r = float(rel(P[sel], Y[sel]).mean())
        # the model's own time-mean error, the part the floor bounds
        mp = float((np.sqrt(T * ((P[sel].mean(1) - Y[sel].mean(1)) ** 2).sum((1, 2, 3)))
                    / norm[sel]).mean())
        # what rel-L2 becomes if the mean part is driven to its floor and the fluctuation part
        # (orthogonal to it under the time inner product) is left exactly where it is
        fluc = np.sqrt(max(r ** 2 - mp ** 2, 0.0))
        best = float(np.hypot(f, fluc))
        print(f"{str(A):>4s} {int(sel.sum()):5d} {f:11.4f} {r:10.4f} {mp:10.4f} "
              f"{100*(1 - best / r):8.1f}%")
    if P is not None:
        print("\n`headroom` is the per-cent of rel-L2 that would disappear if the TIME-MEAN\n"
              "part were driven to its irreducible sampling floor and the fluctuation part\n"
              "were left untouched. The proxy cannot resolve a change under ~2%, so a row\n"
              "below that is closed whatever the mechanism.")


if __name__ == "__main__":
    main()
