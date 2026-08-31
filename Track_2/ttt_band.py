"""Online SPS band calibration -- the one thing Track 2 can do that Track 1 structurally cannot.

Track 1 has to GUESS how much harder the hidden test set is than the split its band was fitted
on.  The guess is a 1.39x per-element tail inflation read back out of a single board round
(`Track_1/calib_board.py`), and `sps_score` is our weakest subscore (39.6 on the board against
a 71.0 ceiling) carrying the second-largest weight (0.204).

Track 2 does not have to guess.  Each step reveals the previous window's ground truth, and the
previous window's prediction is already in hand, so this trajectory's REALIZED per-element
error distribution is observable -- causally, with no forward pass, no backward pass and no
extra invocation.  This prices that.

    python ttt_band.py --npz <stream_eval --save output>

Read the table as: `shipped` is what Track 1's band scores when the errors are stretched by
`x infl`; `best global rescale` is the hindsight ceiling of any scale-only rule; the online
rows are what a causal estimator actually recovers.  A rule is only interesting if it tracks
the ceiling at EVERY inflation -- one that only works at 1.0 has just memorized this split.

Scoring goes through `Track_1/sps9`, whose factorization makes a band evaluation one pass over
|E| with the per-sample accuracy factors computed once; the naive route re-derives rel-L2, TKE
and MVPE for every candidate and is ~100x slower.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Track_1"))
import sps9  # noqa: E402

SIGMA = sps9.SIGMA


def groups_of(ids):
    names = [str(s).split(":")[0] for s in ids]
    g, cur = [], [0]
    for i in range(1, len(names)):
        if names[i] != names[i - 1]:
            g.append(np.array(cur)); cur = []
        cur.append(i)
    g.append(np.array(cur))
    return g


def sps_perwindow(A, a, scored, n_scored, H, scales):
    """SPS with a per-WINDOW multiplicative scale on the half-widths.

    `H` is (1,T,H,W,2) or (N,...); `scales` is (N,).  One window at a time, because the
    full-size product would be a 27M-element temporary per candidate.
    """
    tot = 0.0
    Hb = H[0] if H.shape[0] == 1 else None
    for i in range(len(A)):
        h = (Hb if Hb is not None else H[i]) * np.float32(scales[i])
        c = np.where((A[i] <= h) & scored[i], np.exp(-2.0 * h / SIGMA), 0.0)
        tot += float(a[i]) * float(c.sum(dtype=np.float64))
    return 100.0 * tot / n_scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--inflate", type=float, nargs="+", default=[1.0, 1.2, 1.39, 1.7])
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    P, Y = z["pred"].astype(np.float32), z["target"].astype(np.float32)
    H0 = ((z["upper"] - z["lower"]) / 2.0).astype(np.float32)[..., :2]
    G = groups_of(z["ids"])
    print(f"{len(P)} windows / {len(G)} trajectories;  median shipped half-width "
          f"{np.median(H0):.5f}  (sigma_global {SIGMA:.5f})")

    for infl in a.inflate:
        # Stretch the ERROR and keep the prediction: exactly the mis-specification the board
        # imposes on a band fitted locally.  The zero-target mask is preserved so the scored
        # element set does not move.
        Yi = P.copy()
        Yi[..., :2] = P[..., :2] + infl * (Y[..., :2] - P[..., :2])
        Yi[..., :2] = np.where(Y[..., :2] == 0.0, 0.0, Yi[..., :2])
        A, av, scored, n = sps9.prep(P, Yi)

        one = np.ones(len(P))
        shipped = sps_perwindow(A, av, scored, n, H0, one)
        grid = np.linspace(0.4, 3.0, 27)
        vals = [sps_perwindow(A, av, scored, n, H0, one * c) for c in grid]
        j = int(np.argmax(vals))
        ceil_c, ceil_v = float(grid[j]), float(vals[j])

        # Oracle upper bound for ANY band: the exact per-element optimum (sps9.optimal_band),
        # which knows every realized error.  Reported so the scale-only ceiling can be read as
        # a fraction of what intervals can do at all.
        ob = sps9.optimal_band(A, av, scored)
        oracle = sps_perwindow(A, av, scored, n, ob[None] if ob.ndim == 4 else ob, one)

        print(f"\n  --- errors x{infl:.2f} ---   "
              f"(median |E| {np.median(A[scored]):.5f})")
        print(f"  {'shipped Track-1 band':30s} {shipped:7.3f}")
        print(f"  {'best global rescale (hindsight)':30s} {ceil_v:7.3f}  c={ceil_c:.2f}  "
              f"{ceil_v - shipped:+.3f}")
        print(f"  {'per-element oracle band':30s} {oracle:7.3f}  {oracle - shipped:+.3f}")

        # ---- causal online rules -------------------------------------------------------
        # s_k = q(|E| over this trajectory's steps so far) / q(half-widths on the same steps).
        # A pure ratio, so nothing about the fitting split has to travel with it.
        for q in (0.5, 0.75, 0.9):
            for warm in (1, 3):
                for lam in (1.0,):
                    scales = np.ones(len(P))
                    for g in G:
                        seen_e, seen_h = [], []
                        s = 1.0
                        for j2, i in enumerate(g):
                            if j2 >= warm:
                                e = np.concatenate(seen_e); h = np.concatenate(seen_h)
                                s_new = float(np.quantile(e, q) / max(np.quantile(h, q), 1e-9))
                                s = float(np.clip(lam * s_new + (1 - lam) * s, 0.4, 3.0))
                            scales[i] = s
                            m = scored[i]
                            seen_e.append(A[i][m]); seen_h.append(np.broadcast_to(
                                H0[0] if H0.shape[0] == 1 else H0[i], A[i].shape)[m])
                    v = sps_perwindow(A, av, scored, n, H0, scales)
                    print(f"  {'online q%.2f warm=%d' % (q, warm):30s} {v:7.3f}  "
                          f"{v - shipped:+.3f}   (final scale {scales[-1]:.2f})")


if __name__ == "__main__":
    main()
