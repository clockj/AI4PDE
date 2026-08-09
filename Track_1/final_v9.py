"""Project five v9 subscores onto the leaderboard's `final_score`.

v9 does not publish the combination. Three board rounds plus the assumption that it is a linear
weighted mean with weights summing to 1 and no intercept determine it exactly -- up to one free
direction, because three equations cannot fix five unknowns.

    from final_v9 import predict
    mid, lo, hi = predict([rel, tke, mvpe, time, sps])

Replaces the rule `final ~ mean5 + 0.73` used in rounds 6-8, which was falsified in round 9:
the three offsets were +0.728, +0.743, +0.903, and mean5 went DOWN between the last two while
`final` went UP.

**This is a fit, not a validated model.** Any five-weight linear model reproduces three points.
What makes it usable is that the interval [lo, hi] spans the whole admissible family, so a
prediction is only as strong as its width -- report the range, never the midpoint alone. The
next submission is the falsification test: predict before looking.
"""
from __future__ import annotations

import numpy as np

# rel_l2, tke, mvpe, time, sps -> final. Rounds 6 (U-Net), 7 (FNO x3), 8 (FNO x8).
BOARD = np.array([
    [94.4342, 78.1660, 93.7922, 91.6973, 36.9374, 79.733],
    [94.5754, 78.5278, 93.9100, 91.7264, 37.4680, 79.984883],
    [94.6342, 78.8825, 93.8030, 90.5066, 38.2338, 80.115065],
])

_A = np.vstack([BOARD[:, :5], np.ones(5)])          # three rounds + sum(w) = 1
_b = np.concatenate([BOARD[:, 5], [1.0]])
W0, *_ = np.linalg.lstsq(_A, _b, rcond=None)        # minimum-norm solution
NULL = np.linalg.svd(_A)[2][-1]                     # the one unconstrained direction

# Every weight must be non-negative, which bounds how far along NULL the solution can slide.
_lo = [-W0[i] / NULL[i] for i in range(5) if NULL[i] > 0]
_hi = [-W0[i] / NULL[i] for i in range(5) if NULL[i] < 0]
S_LO, S_HI = max(_lo), min(_hi)
W_LO, W_HI = W0 + S_LO * NULL, W0 + S_HI * NULL     # the two extreme admissible weightings


def predict(subs) -> tuple[float, float, float]:
    """(midpoint, min, max) `final_score` over the admissible weight family.

    `subs` is [rel_l2_score, tke_score, mvpe_score, time_score, sps_score].
    The three corner cases -- W0, W_LO, W_HI -- bracket the family because `final` is linear
    in the weights and the family is a segment.
    """
    v = np.asarray(subs, dtype=float)
    ends = [float(w @ v) for w in (W0, W_LO, W_HI)]
    return float(W0 @ v), min(ends), max(ends)


def report(subs, label: str = "") -> str:
    mid, lo, hi = predict(subs)
    return (f"{label}predicted final {mid:.3f}  (admissible {lo:.3f}-{hi:.3f}, "
            f"+-{(hi - lo) / 2:.3f})")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print(f"W0    {W0}   (min-norm)")
    print(f"null  {NULL}   s in [{S_LO:.3f}, {S_HI:.3f}]")
    print(f"W_lo  {W_LO}")
    print(f"W_hi  {W_HI}")
    print("\nper-weight admissible range:")
    for i, n in enumerate(["rel_l2", "tke", "mvpe", "time", "sps"]):
        a, b = sorted([W_LO[i], W_HI[i]])
        print(f"  {n:7s} {a:.4f} - {b:.4f}   mid {W0[i]:.4f}")
    print("\nself-check against the three board rounds:")
    for row in BOARD:
        mid, lo, hi = predict(row[:5])
        print(f"  actual {row[5]:.6f}  fit {mid:.6f}  err {mid - row[5]:+.2e}  "
              f"range width {hi - lo:.2e}")
