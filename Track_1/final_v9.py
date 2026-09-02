"""Project five v9 subscores onto the leaderboard's `final_score`.

v9 does not publish the combination.  Rounds 9 and 10 inferred it from three and four board
rounds respectively, where a five-weight linear model fits ANY three points exactly and the
"fit" carried no evidence.  **This version uses all TWELVE scored rounds, POOLED ACROSS BOTH
TRACKS** -- ten on Track 1 (two of them a collaborator's) and two on Track 2.  Twelve
equations against five weights with sum(w) = 1 leaves 8 degrees of freedom, so the
no-intercept linear form is TESTED rather than assumed -- and it survives, at a max residual
of 0.038 against subscores quoted to six decimals.

Pooling is licensed by round 13, which shipped the IDENTICAL package to both boards on the
same day: three of the four error subscores agreed to <=0.3 % of the error, and Track 1's
weights reprice Track 2's subscores to 80.650 against a published 80.653.  One test set, one
rule (ROADMAP 1.16).  The pooled residual (0.038) is no worse than the Track-1-only fit
(0.035), which is the evidence that pooling did not paper over a second rule.

    from final_v9 import predict
    mid, lo, hi = predict([rel, tke, mvpe, time, sps])

What changed, and it matters:

    subscore   round-9 claim   eight-round   nine-round   TWELVE-round   admissible (12)
    rel_l2         0.266          0.545        0.582         0.568       0.294 - 0.679
    tke            0.244          0.170        0.097         0.085       0.010 - 0.187
    mvpe           0.241          0.000        0.011         0.029       0.000 - 0.237
    time           0.069          0.081        0.087         0.091       0.038 - 0.169
    sps            0.181          0.204        0.224         0.226       0.192 - 0.248

`rel_l2` is worth more than twice what round 9 priced it at, and `mvpe` is worth approximately
nothing.  Round 10's "stop optimizing rel-L2" followed from the old weights and is withdrawn.
Nothing moved by more than its own interval between the nine- and twelve-round fits.

Round 13 was submitted FOR this: ALPHA 0.90 -> 0.60 is a near-pure `rel_l2` <-> `tke` trade,
which is the family's dominant undetermined axis.  It moved the board +0.128 where the
eight-round weights predicted +0.019, and the only way to fit that is `tke` DOWN and `rel_l2`
UP.  **`tke` is the cheapest error subscore on the board**, below `time` and a sixth of
`rel_l2` -- so the composition (ALPHA) and the variance head, which trade exactly along this
axis, were re-decided: see `Track_1/alpha_board.py`.

**`time` is now MEASURED, not only fitted.**  R14 is R13's package plus a gate that never
fired, so the two rounds differ in `time_score` alone: d time -0.136707, d final -0.013068
=> **w_time = 0.0956**, against a fitted 0.091.  That is the first weight in this project read
off a single-subscore A/B rather than a regression, and it agrees.

**Still a fit, not a law.**  Report the range, not the midpoint alone, and work in DELTAS
between rounds wherever possible -- a delta is insensitive to everything the level fit gets
wrong in common.  `Track_1/board_rounds.py` owns the data and the fit; this module is the
thin, dependency-free interface to it that the eval scripts import.
"""
from __future__ import annotations

import numpy as np

# rel_l2, tke, mvpe, time, sps, final -- every scored round from the shared account on BOTH
# tracks, read off Codabench (2026-08-31 for R6..T2R1, 2026-09-02 for R14 and T2R2).  Kept in
# sync with Track_1/board_rounds.py: rows are ROUNDS then ROUNDS_T2, in that order.
BOARD = np.array([
    [94.4342,   78.1660,   93.7922,   91.6973,   36.9374,   79.733],      # R6   U-Net
    [94.5754,   78.5278,   93.9100,   91.7264,   37.4680,   79.984883],   # R7   FNO x3
    [94.6342,   78.8825,   93.8030,   90.5066,   38.2338,   80.115065],   # R9   FNO x8
    [94.5764,   79.1528,   93.6812,   91.1209,   39.6263,   80.512025],   # R10  7 memb
    [94.461888, 79.593999, 93.655155, 92.102517, 39.328139, 80.519055],   # X1   collaborator
    [94.570878, 79.080459, 93.350128, 92.777073, 39.545007, 80.609074],   # R11  8 memb
    [94.461620, 78.843945, 92.926545, 93.263011, 38.878534, 80.372179],   # X2   collaborator
    [94.490331, 79.167748, 93.418065, 91.803630, 39.637266, 80.515842],   # R12  11 memb
    [94.679785, 77.983720, 93.613336, 93.115432, 39.689258, 80.643823],   # R13  8 memb a=0.60
    [94.679785, 77.983720, 93.613336, 92.978725, 39.689258, 80.630755],   # R14  R13 + blend
    [94.694512, 77.942224, 93.399401, 93.287797, 39.730331, 80.652944],   # T2R1 = R13's pkg
    [94.787655, 77.667926, 93.882222, 93.633350, 40.148056, 80.852826],   # T2R2 dropout x8
])
_X, _Y = BOARD[:, :5], BOARD[:, 5]
NAMES = ["rel_l2", "tke", "mvpe", "time", "sps"]

# The subscore matrix is atrociously conditioned -- every column sits in [36, 95] and moves by
# ~1 across twelve rounds -- so this needs a real solver, not a hand-rolled projected gradient
# (which lands on max|residual| 0.072 against nnls's 0.038 and gets the weights wrong by a
# factor of two).  scipy is in the `realpde` env; it is absent only from the SUBMISSION
# container, and nothing shipped imports this module.
from scipy.optimize import linprog, nnls  # noqa: E402

_A = np.vstack([_X, np.ones(5) * 100.0])       # constraint row scaled into the subscore range
_b = np.concatenate([_Y, [100.0]])
W0, _ = nnls(_A, _b)
RESID = _X @ W0 - _Y

# Admissible family: {w >= 0, sum w = 1, |X w - Y| <= TOL}.  TOL is twice the best achievable
# max-residual (floored), so a knife-edge best fit does not collapse the interval to a point.
TOL = max(2.0 * float(np.abs(RESID).max()), 0.01)
_A_UB = np.vstack([_X, -_X])
_B_UB = np.concatenate([_Y + TOL, -(_Y - TOL)])
_A_EQ, _B_EQ = np.ones((1, 5)), np.array([1.0])
_BOUNDS = [(0.0, None)] * 5


def _extreme(v, sign):
    """min (sign=+1) or max (sign=-1) of w . v over the admissible family."""
    r = linprog(sign * np.asarray(v, float), A_ub=_A_UB, b_ub=_B_UB,
                A_eq=_A_EQ, b_eq=_B_EQ, bounds=_BOUNDS)
    return float(r.x @ np.asarray(v, float)) if r.success else float(W0 @ np.asarray(v, float))


def weight_range() -> dict:
    """Per-weight admissible interval, the table quoted in the docstring."""
    out = {}
    for i, n in enumerate(NAMES):
        e = np.zeros(5); e[i] = 1.0
        out[n] = (_extreme(e, +1), _extreme(e, -1))
    return out


def predict(subs) -> tuple[float, float, float]:
    """(midpoint, min, max) `final_score` over the admissible weight family."""
    v = np.asarray(subs, dtype=float)
    return float(W0 @ v), _extreme(v, +1), _extreme(v, -1)


def delta(subs_a, subs_b) -> tuple[float, float, float]:
    """(midpoint, min, max) of `final(b) - final(a)`.

    Prefer this to two `predict` calls whenever both packages are ours: the level fit's
    common-mode error cancels in a difference, so a delta is the number the twelve rounds
    actually constrain.  Round 12's post-mortem turns on exactly this distinction.
    """
    d = np.asarray(subs_b, float) - np.asarray(subs_a, float)
    return float(W0 @ d), _extreme(d, +1), _extreme(d, -1)


def report(subs, label: str = "") -> str:
    mid, lo, hi = predict(subs)
    return (f"{label}predicted final {mid:.3f}  (admissible {lo:.3f}-{hi:.3f}, "
            f"+-{(hi - lo) / 2:.3f})")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print(f"W0 {W0}   sum {W0.sum():.6f}")
    print(f"max |residual| over the {len(BOARD)} pooled rounds: "
          f"{np.abs(RESID).max():.4f}   TOL {TOL:.4f}\n")
    for n, (lo, hi) in weight_range().items():
        i = NAMES.index(n)
        print(f"  {n:7s} {lo:.3f} - {hi:.3f}   fit {W0[i]:.3f}")
    print("\nself-check against every board round:")
    for row in BOARD:
        mid, lo, hi = predict(row[:5])
        print(f"  actual {row[5]:.6f}  fit {mid:.6f}  err {mid - row[5]:+.4f}  "
              f"range {hi - lo:.3f}")
