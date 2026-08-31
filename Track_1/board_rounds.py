"""Every scored Codabench round, and what the five subscores imply about `final_score`.

v9 stopped publishing the combination rule.  Round 9 inferred a weight family from three
rounds; round 10 falsified the no-intercept form with a fourth.  This is the fifth, and with
five rounds the family is tight enough to be useful -- and, for the first time, the
no-intercept model is OVERDETERMINED and can be tested rather than assumed.

    python board_rounds.py
"""
from __future__ import annotations

import itertools

import numpy as np

NAMES = ["rel_l2", "tke", "mvpe", "time", "sps"]

# (label, date, submission id, rel, tke, mvpe, time, sps, final).
# Submission ids come from the Codabench history read 2026-08-31.  Two rounds -- 883908 and
# 886137 -- were submitted by a collaborator and have no package recorded in this repo; their
# subscores are unknown, so they are listed for the record but excluded from the fit.
ROUNDS = [
    ("R6  U-Net 1 inv ", "2026-08-04", 878641, 94.4342,   78.1660,   93.7922,   91.6973,   36.9374,   79.733),
    ("R7  FNO x3 3 inv", "2026-08-05", 879768, 94.5754,   78.5278,   93.9100,   91.7264,   37.4680,   79.984883),
    ("R9  FNO x8 8 inv", "2026-08-06", 881164, 94.6342,   78.8825,   93.8030,   90.5066,   38.2338,   80.115065),
    ("R10 7 memb/3 inv", "2026-08-08", 883017, 94.5764,   79.1528,   93.6812,   91.1209,   39.6263,   80.512025),
    ("X1  collaborator", "2026-08-09", 883908, 94.461888, 79.593999, 93.655155, 92.102517, 39.328139, 80.519055),
    ("R11 8 memb/3 inv", "2026-08-10", 885003, 94.570878, 79.080459, 93.350128, 92.777073, 39.545007, 80.609074),
    ("X2  collaborator", "2026-08-11", 886137, 94.461620, 78.843945, 92.926545, 93.263011, 38.878534, 80.372179),
    ("R12 11 memb/4inv", "2026-08-31", 909571, 94.490331, 79.167748, 93.418065, 91.803630, 39.637266, 80.515842),
]

UNSCORED = []   # all eight rounds now have full subscores

# ---------------------------------------------------------------------------------------
# TRACK 2.  Empty until the first LTTTA submission is scored.  Paste rows in the same shape:
#     (label, date, submission id, rel, tke, mvpe, time, sps, final)
#
# The first Track-2 round ships the SAME 8 members as Track 1's round 13, so its four error
# subscores are a direct test of whether the two tracks score the same hidden trajectories.
# Predicted, recorded 2026-08-31 before submitting: rel 94.47 / tke 79.35 / mvpe 93.41 /
# sps 39.50 (time unknown -- Track 1's board runs 3-5x local, and local is 1.817 ms/step).
#
#   * within ~0.15 on all four  -> shared test set; every Track-1 instrument transfers, and
#     Track 2 only needs its own board-TIME model fitted;
#   * rel_l2 off by more than ~0.5 -> different sets; Track 2 needs its own calibration, and
#     nothing in CLAUDE.md's error tables may be quoted for it.
#
# Track 2's `final_score` combination is separately unpublished. Do not assume it is Track 1's:
# fit it the same way, from its own rounds, once there are enough of them.
ROUNDS_T2 = []

X = np.array([r[3:8] for r in ROUNDS], float)
Y = np.array([r[8] for r in ROUNDS], float)


def fit(nonneg: bool = True):
    """Least squares for `final = w . subscores`, with sum(w) = 1 and no intercept.

    Eight rounds against five weights plus one constraint is comfortably overdetermined, so
    unlike rounds 9-10 this is a TESTABLE model rather than an exact interpolation: the
    residual is now evidence.
    """
    from scipy.optimize import nnls
    A = np.vstack([X, np.ones(5) * 100.0])       # scale the constraint row into range
    b = np.concatenate([Y, [100.0]])
    if nonneg:
        w, _ = nnls(A, b)
    else:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return w, X @ w - Y


def weight_bounds(tol_mult: float = 2.0, tol_floor: float = 0.01):
    """Range of each weight over every non-negative weighting that fits nearly as well.

    Linear program: minimise/maximise w_i subject to |X w - Y| <= tol, sum(w) = 1, w >= 0.
    `tol` is whichever is larger -- a multiple of the best achievable max-residual, or a floor
    -- so a knife-edge best fit does not collapse the reported interval to a point.
    """
    from scipy.optimize import linprog
    w_best, r_best = fit()
    tol = max(tol_mult * np.abs(r_best).max(), tol_floor)
    n = 5
    A_ub = np.vstack([X, -X])
    b_ub = np.concatenate([Y + tol, -(Y - tol)])
    A_eq, b_eq = np.ones((1, n)), np.array([1.0])
    out = {}
    for i in range(n):
        c = np.zeros(n); c[i] = 1.0
        lo = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n)
        hi = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n)
        out[NAMES[i]] = (lo.x[i] if lo.success else np.nan,
                         hi.x[i] if hi.success else np.nan)
    return w_best, np.abs(r_best).max(), tol, out


def compare(a_label, b_label):
    """Subscore-by-subscore delta between two rounds, priced with the fitted weights."""
    ia = [r[0].startswith(a_label) for r in ROUNDS].index(True)
    ib = [r[0].startswith(b_label) for r in ROUNDS].index(True)
    w, _ = fit()
    d = X[ib] - X[ia]
    print(f"\n=== {ROUNDS[ib][0].strip()} vs {ROUNDS[ia][0].strip()} ===")
    print(f"  {'subscore':<10s} {'delta':>9s} {'x weight':>9s} {'= final':>9s}")
    for n, dv, wv in zip(NAMES, d, w):
        print(f"  {n:<10s} {dv:>+9.3f} {wv:>9.3f} {dv * wv:>+9.3f}")
    print(f"  {'TOTAL':<10s} {'':>9s} {'':>9s} {float(d @ w):>+9.3f}"
          f"   (actual {Y[ib] - Y[ia]:+.3f})")


def main():
    print("=== Scored rounds ===")
    print(f"{'round':<17s} {'date':<11s} {'id':>7s}  " +
          "  ".join(f"{n:>9s}" for n in NAMES) + "      final")
    for r in ROUNDS:
        print(f"{r[0]:<17s} {r[1]:<11s} {r[2]:>7d}  " +
              "  ".join(f"{v:9.4f}" for v in r[3:8]) + f"  {r[8]:9.5f}")

    w, res, tol, bounds = weight_bounds()
    print("\n=== final_score = w . subscores,  sum(w) = 1,  w >= 0 ===")
    print(f"  eight rounds, five weights -> 3 degrees of freedom")
    print(f"  max |residual| {res:.4f}   (an exact-fit artefact would be ~0 with 0 dof)")
    print(f"  weight interval = every non-negative weighting fitting to within {tol:.4f}\n")
    print(f"  {'subscore':<10s} {'fitted':>8s}   {'admissible':>18s}   round-9 claim")
    r9 = dict(rel_l2="0.266", tke="0.244", mvpe="0.241", time="0.069", sps="0.181")
    for n, v in zip(NAMES, w):
        lo, hi = bounds[n]
        print(f"  {n:<10s} {v:>8.3f}   {lo:>8.3f} - {hi:<7.3f}   {r9[n]}")

    print("\n=== Deltas, R12 vs the two rounds that matter ===")
    compare("R10", "R12")
    compare("R11", "R12")

    print("\n=== Time-normalised final: what each round scores at a COMMON board latency ===")
    print("  Board `time_score` swings 2.8 points across the eight rounds and is only partly")
    print("  under our control, so the raw ranking conflates model quality with eval-machine")
    print("  luck. This holds time_score fixed at the median and re-prices the other four.")
    ref = float(np.median(X[:, 3]))
    lo_t, hi_t = bounds["time"]
    print(f"\n  reference time_score = {ref:.3f}\n")
    print(f"  {'round':<17s} {'final':>9s} {'time adj':>9s} {'normalised':>11s}"
          f" {'(range)':>16s}")
    rows = []
    for r, x, y in zip(ROUNDS, X, Y):
        adj = (ref - x[3]) * w[3]
        rng = sorted([(ref - x[3]) * lo_t, (ref - x[3]) * hi_t])
        rows.append((r[0], y, adj, y + adj, y + rng[0], y + rng[1]))
    for lab, y, adj, norm, a, b in rows:
        print(f"  {lab:<17s} {y:>9.3f} {adj:>+9.3f} {norm:>11.3f}"
              f" {a:>7.3f}-{b:<7.3f}")
    best = max(rows, key=lambda t: t[3])
    print(f"\n  best at constant latency: {best[0].strip()} ({best[3]:.3f})")

    print("\n=== Board latency, inverted from time_score ===")
    print(f"  {'round':<17s} {'inv':>4s} {'local ms':>9s} {'board ms':>9s} "
          f"{'model':>7s} {'miss':>7s}")
    known = {"R6": (1, 2.648), "R7": (3, 1.282), "R9": (8, 2.211),
             "R10": (3, 3.598), "R11": (3, 2.002), "R12": (4, 2.235)}
    for r in ROUNDS:
        key = r[0].split()[0]
        t = 0.72896 * (100.0 / r[6] - 1.0) ** 2 * 1000.0
        if key in known:
            inv, loc = known[key]
            m = 4.447 + 0.324 * inv + 0.431 * loc
            print(f"  {r[0]:<17s} {inv:>4d} {loc:>9.3f} {t:>9.3f} {m:>7.3f} {t - m:>+7.3f}")
        else:
            print(f"  {r[0]:<17s} {'?':>4s} {'?':>9s} {t:>9.3f}")


if __name__ == "__main__":
    main()
