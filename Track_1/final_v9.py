"""Project five v9 subscores onto the leaderboard's `final_score`.

v9 does not publish the combination.  Rounds 9 and 10 inferred it from three and four board
rounds respectively, where a five-weight linear model fits ANY three points exactly and the
"fit" carried no evidence.  **This version uses all EIGHTEEN scored rounds, POOLED ACROSS BOTH
TRACKS** -- thirteen on Track 1 (two of them a collaborator's) and five on Track 2.

**ROUND 17 DROPPED THE `sum(w) = 1` CONSTRAINT, AND THAT IS THE WHOLE STORY OF THIS FILE.**
Every fit from round 9 to round 16 imposed it.  Nothing ever measured it; a weighted mean was
the natural guess, and no one removed the row to see what happened.  Removing it:

    subscore   round-9 claim   FOURTEEN-rd   SIXTEEN-rd    EIGHTEEN-rd    LOO range
                                (sum w = 1)   (sum w = 1)    **FREE**     (17-of-18 refits)
    rel_l2         0.266          0.573         0.578        0.4754      0.4742 - 0.4771
    tke            0.244          0.075         0.068        0.1017      0.1013 - 0.1019
    mvpe           0.241          0.035         0.039        0.0936      0.0920 - 0.0944
    time           0.069          0.088         0.083        0.0967      0.0961 - 0.0971
    sps            0.181          0.229         0.231        0.2503      0.2499 - 0.2505
    sum w          1.000          1.000         1.000        1.0177
    max residual     --           0.044         0.047        **0.0023**

**Twenty times tighter, with one fewer assumption.**  Leave-one-round-out predicts a held-out
round's `final` to +-0.004 (rms 0.002), and the LOO weight spread above is a thousandth wide:
no single round carries the fit.  This is an instrument now, not a regression.

**And the one weight ever measured WITHOUT a fit picks the free form.**  R14 is R13's package
plus a gate that never fired, so the two rounds differ in `time_score` alone: d time -0.136707,
d final -0.013068 => **w_time = 0.0956**.  The free fit says **0.0967**; the constrained fit
said 0.085.  The direct A/B is 12 % from the constrained value and 1 % from the free one.

Why `sum(w)` lands at 1.018 rather than 1 is unknown -- a hidden sixth term, or a subscore
transformed before the average.  The ~0.002 residual says whatever it is, it is small, and it
cancels in DELTAS, which is how this module should be used.

WHAT THIS RE-PRICES, and these are retractions, not refinements:

* **`mvpe` is NOT worth nothing.**  0.094, not 0.035 -- 2.4x.  "Stop spending on mvpe" (round
  12 §3) is WITHDRAWN.
* **`tke` is NOT the cheapest error subscore.**  0.102, not 0.068 -- 1.5x, and above `time`.
  Round 13's "the variance branch is retired" rested on the constrained weights and is
  WITHDRAWN; the branch is cheap, not worthless.
* At R17's levels a 10 % cut in one error is worth (direct subscore only, before SPS):
  rel **+0.238** (was +0.289) / tke **+0.175** (was +0.117) / mvpe **+0.053** (was +0.022).
  `rel_l2` + `sps` still carry ~0.73 of the total, so the strategy does not invert -- but the
  three error subscores are much closer together than four rounds of planning assumed.
* **ALPHA 0.60 WAS NEVER SETTLED.**  Round 13 read its +0.128 as evidence that `tke` is cheap.
  Re-priced free, R13's ALPHA 0.90 -> 0.60 is rel +0.090, tke -0.120, mvpe +0.018, sps +0.013
  = **+0.001, i.e. nothing**; the entire +0.128 was the 4 -> 1 invocation collapse (time
  +0.127).  Two alphas tie on the board, so the optimum is somewhere between them and the
  sweep has to be redone under these weights (plan_20260905.md Q1d).

Pooling is licensed by round 13, which shipped the IDENTICAL package to both boards on the
same day, and re-confirmed four times since (R15/T2R2, R16/T2R3, and R17/T2R5): three of the
four error subscores agree to <=0.5 % of the error every time, with mvpe the standing outlier
(36 scored probe points per window is the smallest effective sample of the five).  One test
set, one rule (ROADMAP 1.16).

    from final_v9 import predict
    mid, lo, hi = predict([rel, tke, mvpe, time, sps])

**Still a fit, not a law.**  Report the range, not the midpoint alone, and work in DELTAS
between rounds wherever possible -- a delta is insensitive to everything the level fit gets
wrong in common.  `Track_1/board_rounds.py` owns the data and the fit; this module is the
thin, dependency-free interface to it that the eval scripts import.
"""
from __future__ import annotations

import numpy as np

# rel_l2, tke, mvpe, time, sps, final -- every scored round from the shared account on BOTH
# tracks, read off Codabench (2026-08-31 for R6..T2R1, 2026-09-02 for R14/T2R2, 2026-09-03 for
# R15/T2R3, 2026-09-04 for R16/T2R4/R17/T2R5).  Listed here in SUBMISSION-DATE order and in
# Track_1/board_rounds.py by track; `_check_sync` below holds the two to the same multiset.
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
    [94.780094, 77.747643, 94.122329, 93.180177, 40.140948, 80.835124],   # R15  = T2R2's members
    [94.755824, 78.174302, 93.739707, 92.508902, 41.160630, 81.019213],   # T2R3 w96 x8, fp16
    [94.764049, 78.433653, 94.030389, 92.699846, 41.259088, 81.119780],   # R16  = T2R3's members
    [94.661471, 78.413683, 93.727402, 90.154359, 41.600131, 80.879200],   # T2R4 U-FNO x3, 3 inv
    [94.745531, 78.635462, 93.974234, 92.988361, 41.324574, 81.169843],   # R17  w96 L8 do45 x4
    [94.727329, 78.357043, 93.692366, 92.427051, 41.191134, 81.019298],   # T2R5 = R17's members
    [94.745531, 78.635462, 93.974234, 93.058602, 41.324574, 81.176572],   # R18  = R17 + hostw
    [94.727329, 78.357043, 93.692366, 92.399134, 41.191137, 81.016621],   # T2R6 = T2R5 + bmm
    [94.776881, 78.532564, 94.031877, 93.030936, 41.332982, 81.186041],   # R19  simx4
    [94.762818, 78.220131, 93.843944, 92.896212, 41.210644, 81.086639],   # T2R7 = R19's members
    [94.794521, 78.518609, 94.081696, 93.270089, 41.410372, 81.239831],   # R20  cmb3sim, best
    [94.782311, 78.278425, 93.933159, 92.597588, 41.310588, 81.106430],   # T2R8 = R20's members
    [94.847008, 78.684381, 94.298686, 92.676031, 41.755710, 81.330636],   # R21  xfam5, 2 inv
    [94.841673, 78.469938, 94.157190, 91.521812, 41.649497, 81.155762],   # T2R9 = R21's members
    [94.794522, 78.518609, 94.081696, 90.932390, 41.410354, 81.015201],   # R23  = R20 UNFUSED
    [94.782311, 78.278425, 93.933159, 90.928559, 41.310591, 80.945935],   # T2R10 = T2R8 UNFUSED
    [94.861827, 78.793104, 94.347768, 91.961259, 41.866674, 81.312041],   # R24  xfam6 = R21 + ViT
    [94.845935, 78.395628, 94.066988, 90.720061, 41.626260, 81.058649],   # T2R11 x8v = T2R9 + 4 FNO
    [94.861790, 78.793380, 94.347508, 92.297891, 41.867687, 81.344644],   # R25  = R24 + fp16 autocast
    [94.861391, 78.587250, 94.224322, 91.039757, 41.760694, 81.164374],   # T2R12 = R25's archive on T2
    [94.747724, 78.667443, 94.173298, 91.528039, 41.995785, 81.218444],   # R26  = P1, Swin x4 + ViT x2
    [94.890753, 78.781089, 94.063576, 91.638537, 42.063113, 81.314785],   # T2R13 = P2-L8, F-FNO x4 + ViT x2
])
_X, _Y = BOARD[:, :5], BOARD[:, 5]


def _check_sync() -> None:
    """BOARD must hold the same ROUNDS as board_rounds.py -- as a SET, not in order.

    This table is a hand-maintained copy so the module stays dependency-free for the submission
    container, and a hand-maintained copy is exactly the thing that silently goes stale.  The
    two files list the rounds in different orders (this one by submission date, `board_rounds`
    by track) and that is harmless -- least squares does not see row order -- so the check is
    on the multiset.  Silent on ImportError by design: nothing shipped imports `board_rounds`.
    """
    try:
        from board_rounds import ROUNDS, ROUNDS_T2          # noqa: PLC0415
    except Exception:
        try:
            from Track_1.board_rounds import ROUNDS, ROUNDS_T2   # noqa: PLC0415
        except Exception:
            return
    want = np.array([r[3:9] for r in list(ROUNDS) + list(ROUNDS_T2)], float)
    key = lambda a: sorted(map(tuple, np.round(a, 6)))
    if want.shape != BOARD.shape or key(want) != key(BOARD):
        raise AssertionError(
            f"final_v9.BOARD is out of sync with board_rounds.py "
            f"({BOARD.shape[0]} rows here, {want.shape[0]} there).  Fix the copy before "
            f"pricing anything -- a stale fit is worse than no fit.")


_check_sync()
NAMES = ["rel_l2", "tke", "mvpe", "time", "sps"]

# The subscore matrix is atrociously conditioned -- every column sits in [36, 95] and moves by
# ~1 across eighteen rounds -- so this needs a real solver.  scipy is in the `realpde` env; it
# is absent only from the SUBMISSION container, and nothing shipped imports this module.
from scipy.optimize import linprog  # noqa: E402

# THE FIT: plain least squares, no constraint row and no intercept (round 17).  The
# `sum(w) = 1` row that rounds 9-16 carried was never tested and costs a factor of twenty in
# residual -- see the docstring.  Non-negativity is not imposed either; it is CHECKED below,
# and it holds, which is one more thing the constrained fit was buying on credit.
W0, *_ = np.linalg.lstsq(_X, _Y, rcond=None)
RESID = _X @ W0 - _Y
if (W0 < 0).any():                       # never fires today; a future round could change that
    raise AssertionError(f"free fit went negative: {dict(zip(NAMES, W0))}")
# THE GATE IS PRECISION-AWARE (round 25, 2026-09-14).  A row may miss by 0.0025 PLUS the rounding it
# was recorded with, pushed through the weights.  R6 was read off Codabench to 3-4 decimals (its
# `final` is 79.733, +-0.0005 on its own), and at 32 rounds it sits at 0.002505 -- 5e-6 over a flat
# 0.0025, on a row whose own rounding is a hundred times that -- while every row recorded to six
# decimals still has to clear 0.0025 (the worst, R15, is at 0.00225).  A mistyped modern round still
# fails loudly; a refit nudging an old low-precision row across the line no longer does.
def _rounding(v) -> float:
    """Half a unit in the last decimal a board value was recorded with (repr gives the typed literal)."""
    s = repr(float(v))
    return 0.5 * 10.0 ** -min(len(s.split(".")[1]) if "." in s else 0, 6)


SLACK = np.array([float(np.abs(W0) @ np.array([_rounding(v) for v in row[:5]])) + _rounding(row[5])
                  for row in BOARD])
_OVER = np.abs(RESID) - (0.0025 + SLACK)
if (_OVER > 0).any():
    _i = int(np.argmax(_OVER))
    raise AssertionError(
        f"final_v9: the free fit no longer reproduces the board to 0.0025 plus recording precision "
        f"(BOARD row {_i}: |residual| {abs(RESID[_i]):.6f}, allowed {0.0025 + SLACK[_i]:.6f}).  A round "
        f"was mistyped, or the scoring rule changed -- do not price anything until this is understood.")

# Admissible family: {w >= 0, |X w - Y| <= TOL}.  NO equality row -- imposing sum(w) = 1 makes
# this set EMPTY at any tolerance the free fit can reach, which is the cleanest statement of
# why the constraint is gone.  TOL is twice the best achievable max-residual, floored at 0.005
# so a knife-edge best fit does not collapse the interval to a point.
TOL = max(2.0 * float(np.abs(RESID).max()), 0.005)
_A_UB = np.vstack([_X, -_X])
_B_UB = np.concatenate([_Y + TOL, -(_Y - TOL)])
_BOUNDS = [(0.0, None)] * 5


def _extreme(v, sign):
    """min (sign=+1) or max (sign=-1) of w . v over the admissible family."""
    r = linprog(sign * np.asarray(v, float), A_ub=_A_UB, b_ub=_B_UB, bounds=_BOUNDS)
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
    common-mode error cancels in a difference, so a delta is the number the eighteen rounds
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
