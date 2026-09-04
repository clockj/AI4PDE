"""Every scored Codabench round, and what the five subscores imply about `final_score`.

v9 stopped publishing the combination rule.  Round 9 inferred a weight family from three
rounds; round 10 falsified the no-intercept form with a fourth.  With FOURTEEN scored rounds the
no-intercept model is comfortably OVERDETERMINED (9 dof) and is tested, not assumed.

**The fit POOLS BOTH TRACKS.**  Round 13 shipped the identical package to Track 1 and Track 2
on the same day and the two boards returned the same four error subscores to <=0.3 % of the
error (mvpe excepted, and even that is one window's worth), and Track 1's fitted weights
reprice Track 2's subscores to 80.650 against a published 80.653 -- so the two tracks are one
test set under one scoring rule (ROADMAP 1.16).  Pooling costs nothing and buys three rounds.
Set `POOL_TRACKS = False` to recover the Track-1-only fit.

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
    ("R13 8memb/1inv a60", "2026-08-31", 910641, 94.679785, 77.983720, 93.613336, 93.115432, 39.689258, 80.643823),
    # R14 is R13's members, R13's band and R13's ALPHA plus the GATED MEAN-BLEND (beta 0.15,
    # fired only for windows whose mean speed leaves the training range).  All four error
    # subscores came back BIT-IDENTICAL to R13's, so the gate never fired on a single hidden
    # window: the hidden set is INSIDE the training speed range and phi = 0.  The only moving
    # part is `time`, which makes the pair a direct measurement of `w_time` (see main()).
    ("R14 blend8/1inv a60", "2026-09-02", 912521, 94.679785, 77.983720, 93.613336, 92.978725, 39.689258, 80.630755),
    # R15 is T2R2's EXACT member set on the Track-1 board -- the third cross-track identity
    # check, and it passes: rel +0.15 %, tke -0.46 %, sps -0.02 % of the error against 912522,
    # with mvpe the outlier again at -4.17 % (36 scored probe points per window, the smallest
    # effective sample of the five, so the first place a one-window pool difference shows).
    ("R15 dropout8/1inv a60", "2026-09-03", 913176, 94.780094, 77.747643, 94.122329, 93.180177, 40.140948, 80.835124),
    # R16 is T2R3's EXACT member set on the Track-1 board -- the FOURTH cross-track identity
    # check, and the account's best score on either board.  In error terms against 913187:
    # rel -0.17 %, tke -1.52 %, mvpe -4.94 %, sps +0.098.  Same shape as the three earlier
    # checks (mvpe the outlier); tke's 1.5 % is the largest tke offset of the four and is
    # worth watching, not acting on.  Board latency 4.52 ms against T2R3's 4.78.
    ("R16 w96x8/1inv a60 fp16", "2026-09-04", 914506, 94.764049, 78.433653, 94.030389, 92.699846, 41.259088, 81.119780),
]

UNSCORED = []   # every round on both tracks now has full subscores

# ---------------------------------------------------------------------------------------
# TRACK 2.  Same shape:  (label, date, submission id, rel, tke, mvpe, time, sps, final)
#
# Round 1 shipped the SAME 8 members, band and ALPHA as Track 1's round 13, so its four error
# subscores are a direct test of whether the two tracks score the same hidden trajectories.
# ANSWERED 2026-08-31, and the answer is YES for three of the four:
#
#     subscore     T1 R13      T2 R1      diff     rel. diff in the ERROR
#     rel_l2      94.679785  94.694512   +0.015          -0.28 %
#     tke         77.983720  77.942224   -0.041          +0.24 %
#     mvpe        93.613336  93.399401   -0.214          +3.59 %
#     sps         39.689258  39.730331   +0.041          -0.10 %
#
# Three of four agree to <=0.3 % of the error -- far inside anything a different test set
# could produce, and far outside float32 noise being the explanation for mvpe. So the sets
# are the same or near-identical, and MVPE, which is read at 4 probe COLUMNS (36 points of
# 2048), is simply the metric with the smallest effective sample size: it is the first place
# a small difference in the scored window set shows up.  Working conclusion: **Track-1
# instruments transfer to Track 2**; treat mvpe as the one number to re-measure per track.
#
# Track 2's `final_score` combination is separately unpublished. Do not assume it is Track
# 1's: with one round it is not identified at all.  (For the record, Track 1's fitted weights
# applied to T2 R1's subscores give 80.667 against the published 80.653.)
#
# CONFIRMED 2026-09-02, and again 2026-09-03: the pooled fit's max residual is 0.038 at twelve
# rounds and 0.044 at fourteen, no worse than the nine Track-1 rounds alone (0.035) -- the
# direct evidence that pooling is legitimate, since a different combination rule on Track 2
# would have shown up as a residual.  Round 15 tightens it further: R15 shipped T2R2's EXACT
# members to the Track-1 board and three of four error subscores came back within 0.5 % of the
# error (rel +0.15 %, tke -0.46 %, sps -0.02 %; mvpe -4.17 %, the usual outlier).
ROUNDS_T2 = [
    ("T2R1 8memb/1inv a60", "2026-08-31", 910642, 94.694512, 77.942224, 93.399401, 93.287797, 39.730331, 80.652944),
    # 8 all-data DROPOUT members (4x0.05, 4x0.15), 1 invocation, ALPHA 0.60.  +0.200 over T2R1
    # and the account's best score on either board.
    ("T2R2 dropout8/1inv a60", "2026-09-02", 912522, 94.787655, 77.667926, 93.882222, 93.633350, 40.148056, 80.852826),
    # 8 all-data WIDTH-96 FNOs (4x dropout 0.30, 4x 0.45), 400 epochs, fp16-packed, ALPHA 0.60.
    # NEW BEST on either board.  The gain is `sps` (+1.013) and `tke` (-2.9 % of error) while
    # `rel_l2` got fractionally WORSE (+0.64 %): capacity x dropout buys the per-element TAIL,
    # not the rms.  It paid 1.41 board ms for it (3.370 -> 4.780).
    ("T2R3 w96x8/1inv a60 fp16", "2026-09-03", 913187, 94.755824, 78.174302, 93.739707, 92.508902, 41.160630, 81.019213),
    # 3 all-data U-FNO members (width 96, dropout 0.30, 400 ep), 3 invocations, ALPHA 0.60.
    # MISSED its prediction by -0.45 (predicted 81.33), and most of the miss is ACCURACY, not
    # latency: against the T2R3 anchor rel_l2 came back +1.90 % WORSE where the 3-vs-3 said
    # -4.4 %, a 4.0-pp instrument residual and the largest on record.  Latency was the other
    # half: 8.69 board ms against T2R3's 4.78 (the U-Net branch is conv-heavy at batch 1 and
    # `allow_tf32 = False` makes 3x3 convs ~50 % slower -- 5.457 ms/step measured, not the
    # 1.28 the staging README claimed).  At T2R3's latency it would have scored 81.086, +0.07.
    # Working hypothesis: U-FNO's local advantage is in-distribution memorization; the
    # interior Re-band k-fold, the one instrument that called T2R3's shape, was never run on
    # it.  Do NOT submit the Track-1 twin of this package.
    ("T2R4 ufno3/3inv a60 fp16", "2026-09-04", 914508, 94.661471, 78.413683, 93.727402, 90.154359, 41.600131, 80.879200),
]

# The fit pools both tracks -- see the module docstring.  ROUNDS first, then ROUNDS_T2, so the
# row order matches Track_1/final_v9.py:BOARD exactly.
POOL_TRACKS = True

FIT_ROUNDS = ROUNDS + ROUNDS_T2 if POOL_TRACKS else ROUNDS
X = np.array([r[3:8] for r in FIT_ROUNDS], float)
Y = np.array([r[8] for r in FIT_ROUNDS], float)

# `w_time`, MEASURED DIRECTLY rather than fitted.  R13 and R14 differ in `time_score` alone
# (the blend gate never fired, so the four error subscores are bit-identical):
#   d time = 92.978725 - 93.115432 = -0.136707,   d final = 80.630755 - 80.643823 = -0.013068
W_TIME_DIRECT = (80.630755 - 80.643823) / (92.978725 - 93.115432)   # = 0.0956


def fit(nonneg: bool = True):
    """Least squares for `final = w . subscores`, with sum(w) = 1 and no intercept.

    Twelve pooled rounds against five weights plus one constraint is comfortably
    overdetermined (7 dof), so unlike rounds 9-10 this is a TESTABLE model rather than an
    exact interpolation: the residual is now evidence.
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
    ia = [r[0].startswith(a_label) for r in FIT_ROUNDS].index(True)
    ib = [r[0].startswith(b_label) for r in FIT_ROUNDS].index(True)
    w, _ = fit()
    d = X[ib] - X[ia]
    print(f"\n=== {FIT_ROUNDS[ib][0].strip()} vs {FIT_ROUNDS[ia][0].strip()} ===")
    print(f"  {'subscore':<10s} {'delta':>9s} {'x weight':>9s} {'= final':>9s}")
    for n, dv, wv in zip(NAMES, d, w):
        print(f"  {n:<10s} {dv:>+9.3f} {wv:>9.3f} {dv * wv:>+9.3f}")
    print(f"  {'TOTAL':<10s} {'':>9s} {'':>9s} {float(d @ w):>+9.3f}"
          f"   (actual {Y[ib] - Y[ia]:+.3f})")


def main():
    print("=== Scored rounds ===")
    print(f"{'round':<24s} {'tr':>2s} {'date':<11s} {'id':>7s}  " +
          "  ".join(f"{n:>9s}" for n in NAMES) + "      final")
    for r in FIT_ROUNDS:
        tr = 2 if r in ROUNDS_T2 else 1
        print(f"{r[0]:<24s} {tr:>2d} {r[1]:<11s} {r[2]:>7d}  " +
              "  ".join(f"{v:9.4f}" for v in r[3:8]) + f"  {r[8]:9.5f}")

    w, res, tol, bounds = weight_bounds()
    print("\n=== final_score = w . subscores,  sum(w) = 1,  w >= 0 ===")
    print(f"  {len(FIT_ROUNDS)} rounds"
          f"{' POOLED across both tracks' if POOL_TRACKS else ' (Track 1 only)'}"
          f", five weights -> {len(FIT_ROUNDS) - 4} degrees of freedom")
    print(f"  max |residual| {res:.4f}   (an exact-fit artefact would be ~0 with 0 dof)")
    print(f"  weight interval = every non-negative weighting fitting to within {tol:.4f}\n")
    print(f"  {'subscore':<10s} {'fitted':>8s}   {'admissible':>18s}   round-9 claim")
    r9 = dict(rel_l2="0.266", tke="0.244", mvpe="0.241", time="0.069", sps="0.181")
    for n, v in zip(NAMES, w):
        lo, hi = bounds[n]
        print(f"  {n:<10s} {v:>8.3f}   {lo:>8.3f} - {hi:<7.3f}   {r9[n]}")

    print("\n  --- and one weight is MEASURED, not fitted ---")
    print("  R13 and R14 differ in `time_score` alone: the gated mean-blend never fired on a")
    print("  single hidden window, so all four error subscores came back bit-identical.")
    print(f"    d time  {92.978725 - 93.115432:+.6f}")
    print(f"    d final {80.630755 - 80.643823:+.6f}")
    print(f"    => w_time = {W_TIME_DIRECT:.4f}   (fit {w[3]:.4f}, "
          f"admissible {bounds['time'][0]:.3f}-{bounds['time'][1]:.3f})")
    print("  A single-subscore A/B is the only way to read a weight without the fit; this one")
    print("  lands at the fitted value, which is the first independent check the family has.")

    print("\n=== Deltas, R12 vs the two rounds that matter ===")
    compare("R10", "R12")
    compare("R11", "R12")

    print("\n=== Time-normalised final: what each round scores at a COMMON board latency ===")
    print("  Board `time_score` swings 2.8 points across the twelve rounds and is only partly")
    print("  under our control, so the raw ranking conflates model quality with eval-machine")
    print("  luck. This holds time_score fixed at the median and re-prices the other four.")
    ref = float(np.median(X[:, 3]))
    lo_t, hi_t = bounds["time"]
    print(f"\n  reference time_score = {ref:.3f}\n")
    print(f"  {'round':<24s} {'final':>9s} {'time adj':>9s} {'normalised':>11s}"
          f" {'(range)':>16s}")
    rows = []
    for r, x, y in zip(FIT_ROUNDS, X, Y):
        adj = (ref - x[3]) * w[3]
        rng = sorted([(ref - x[3]) * lo_t, (ref - x[3]) * hi_t])
        rows.append((r[0], y, adj, y + adj, y + rng[0], y + rng[1]))
    for lab, y, adj, norm, a, b in rows:
        print(f"  {lab:<24s} {y:>9.3f} {adj:>+9.3f} {norm:>11.3f}"
              f" {a:>7.3f}-{b:<7.3f}")
    best = max(rows, key=lambda t: t[3])
    print(f"\n  best at constant latency: {best[0].strip()} ({best[3]:.3f})")

    print("\n=== Board latency, inverted from time_score ===")
    print(f"  {'round':<24s} {'inv':>4s} {'local ms':>9s} {'board ms':>9s} "
          f"{'model':>7s} {'miss':>7s}")
    # local ms are 4060-EQUIVALENT (the A100 runs 1.850x faster; CLAUDE.md round 12).
    known = {"R6": (1, 2.648), "R7": (3, 1.282), "R9": (8, 2.211),
             "R10": (3, 3.598), "R11": (3, 2.002), "R12": (4, 2.235),
             "R13": (1, 1.900),
             # R14: R13's package + the (never-firing) blend gate. 1.033 ms/window measured on
             # the A100; x1.850 = 1.911 4060-equivalent.
             "R14": (1, 1.911),
             # R15: T2R2's members on Track 1, 1.023 ms A100 x1.850 = 1.893.
             "R15": (1, 1.893),
             # R16: T2R3's package on the Track-1 board, so the same 3.017.
             "R16": (1, 3.017),
             "T2R1": (1, 1.900), "T2R2": (1, 2.546),
             # T2R3: 8 fused width-96 members, 1.631 ms A100 x1.850 = 3.017.
             "T2R3": (1, 3.017),
             # T2R4: 3 U-FNO members, 3 invocations. 5.457 ms/step measured in the build log
             # (slurm-7144661) x1.850 = 10.10 4060-equivalent.  The README's "1.28 ms/step"
             # was wrong and is withdrawn.
             "T2R4": (3, 10.10)}
    for r in FIT_ROUNDS:
        key = r[0].split()[0]
        t = 0.72896 * (100.0 / r[6] - 1.0) ** 2 * 1000.0
        if key in known:
            inv, loc = known[key]
            m = 4.447 + 0.324 * inv + 0.431 * loc
            print(f"  {r[0]:<24s} {inv:>4d} {loc:>9.3f} {t:>9.3f} {m:>7.3f} {t - m:>+7.3f}")
        else:
            print(f"  {r[0]:<24s} {'?':>4s} {'?':>9s} {t:>9.3f}")


if __name__ == "__main__":
    main()
