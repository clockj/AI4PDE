"""Which submission buys the most information?  One submission per day, ~27 slots left.

The five-weight combination behind `final_score` is unpublished.  Eight rounds plus sum(w) = 1
leave a 3-parameter admissible family (`final_v9`), and that residual freedom is not academic:
it has already silently invalidated one tuned constant (ALPHA, `plan/plan_20260831.md` 3c) and
it reversed the priority order of three rounds of work (`mvpe` from 0.241 to 0.000).

A submission does two things.  It scores, and it adds one equation.  This module prices the
SECOND, so a slot can be spent deliberately.

    python sub_value.py

The measure.  For a candidate package with predicted subscores `x`, every `w` in the admissible
family predicts a different `final = w . x`.  The width of `[min_w w.x, max_w w.x]` is exactly
how much the eight rounds FAIL to pin that package down -- so it is what the observation
resolves.  A package whose predicted range is narrow tells us nothing we did not already know,
however good its score.  `final_v9.predict` already returns that interval; here it becomes a
design criterion instead of an error bar.

That is the raw measure.  The useful one is conditional: we do not want information about the
weights for its own sake, we want to stop being unable to DECIDE things.  So `pending` lists
the decisions currently blocked by weight ambiguity -- each one a subscore DELTA whose sign
flips across the family -- and each candidate is scored by how much of that ambiguity survives
after its result lands, averaged over the outcomes the family still permits.

Caveats that bound how far to trust the ranking:

* The candidate's own `x` is a prediction.  Its `time` column is the shaky one (board timing
  swings +-1.7 points for the same package, `ROADMAP.md` 1.7), so `--jitter` re-runs the whole
  ranking with `time` perturbed and reports how much the answer moves.  A candidate whose value
  rests on the `time` column is not a real candidate.
* Informativeness is not score.  A probe worth 0.3 of ambiguity reduction is not worth taking
  if it costs 0.5 of `final` -- during the Decision phase, nothing is.  Main Development is
  where information is cheap; the table reports both columns and never adds them.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import linprog, nnls

import board_rounds

NAMES = board_rounds.NAMES
X8, Y8 = board_rounds.X, board_rounds.Y


# ------------------------------------------------------------------ the admissible family
def fit(X, Y):
    A = np.vstack([X, np.ones(5) * 100.0])
    b = np.concatenate([Y, [100.0]])
    w, _ = nnls(A, b)
    return w, X @ w - Y


def tol_for(X, Y, floor=0.01):
    _, r = fit(X, Y)
    return max(2.0 * float(np.abs(r).max()), floor)


def extreme(v, sign, X, Y, tol):
    """min (sign=+1) / max (sign=-1) of w.v over {w>=0, sum w = 1, |Xw - Y| <= tol}."""
    v = np.asarray(v, float)
    r = linprog(sign * v,
                A_ub=np.vstack([X, -X]), b_ub=np.concatenate([Y + tol, -(Y - tol)]),
                A_eq=np.ones((1, 5)), b_eq=np.array([1.0]), bounds=[(0.0, None)] * 5)
    if not r.success:                      # infeasible: the hypothetical outcome is excluded
        return np.nan
    return float(r.x @ v)


def span(v, X, Y, tol):
    lo, hi = extreme(v, +1, X, Y, tol), extreme(v, -1, X, Y, tol)
    return lo, hi, (hi - lo)


# ------------------------------------------------------------------ what we cannot decide
# Each entry is a subscore DELTA -- the measured or extrapolated move a change makes -- whose
# VALUE the eight rounds do not determine.  Sources are cited; nothing here is invented.
PENDING = [
    # plan_20260831.md 3c, from round 11's own measured alpha trade extrapolated to 0.60.
    ("ALPHA 0.90 -> 0.60", np.array([+0.28, -1.48, 0.0, 0.0, +0.60])),
    # R11 -> R12 measured board delta: 3 more members and a 4th invocation.
    ("more members (R11->R12)", X8[7] - X8[5]),
    # A pure mvpe trade: mvpe's weight is the least determined (0.000-0.160) and three rounds
    # of work were spent on it.  This is the cheapest experiment that would settle it.
    ("give up 1.0 mvpe for 0.2 tke", np.array([0.0, +0.20, -1.00, 0.0, 0.0])),
    # rel_l2 vs tke is the axis every composition knob moves along; its split is what the
    # round-9 -> eight-round refit reversed.
    ("1.0 rel_l2 for 1.0 tke", np.array([-1.00, +1.00, 0.0, 0.0, 0.0])),
]


# ------------------------------------------------------------------ candidate submissions
# Predicted subscores.  Anything not measured is marked in `note` and should be treated as
# soft; `--jitter` shows whether the ranking survives the softest column.
CANDIDATES = [
    ("R13  8 memb / 1 inv",
     np.array([94.47, 79.35, 93.41, 92.59, 39.50]),
     "staged; predicts +0.06 over R12 mostly from the 1-invocation time draw"),
    ("R13 + ALPHA 0.60",
     np.array([94.47 + 0.28, 79.35 - 1.48, 93.41, 92.59, 39.50 + 0.60]),
     "same package, one constant changed; free to build, sign of value unknown"),
    ("R13 + ALPHA 1.10",
     np.array([94.47 - 0.19, 79.35 + 0.99, 93.41, 92.59, 39.50 - 0.40]),
     "the other direction, extrapolated on the same measured trade"),
    ("R12 11 memb / 4 inv (resubmit)",
     X8[7],
     "known package; buys only a second time draw -- listed as the null candidate"),
    ("single member, 1 inv",
     np.array([94.30, 78.60, 93.20, 93.60, 38.20]),
     "SOFT: ensemble ablation, scaled from the R7->R9 member curve; large, cheap, off-axis"),
]


def ambiguity(X, Y, tol):
    out = []
    for label, d in PENDING:
        lo, hi, w = span(d, X, Y, tol)
        out.append((label, lo, hi, w, (lo < 0.0 < hi)))
    return out


def value_of(x, X, Y, tol, n_outcomes=7):
    """Expected surviving ambiguity if `x` is submitted, averaged over admissible outcomes."""
    lo, hi, width = span(x, X, Y, tol)
    if not np.isfinite(width) or width <= 0:
        return width, np.nan, np.nan
    keep = []
    for y in np.linspace(lo, hi, n_outcomes):
        X2, Y2 = np.vstack([X, x]), np.concatenate([Y, [y]])
        tot = 0.0
        for _, d in PENDING:
            _, _, w = span(d, X2, Y2, tol)
            tot += 0.0 if not np.isfinite(w) else w
        keep.append(tot)
    return width, float(np.mean(keep)), float(np.max(keep))


def run(jitter=0.0, seed=0, without=None):
    global PENDING
    if without:
        PENDING = [d for d in PENDING if without.lower() not in d[0].lower()]
        print(f"PENDING filtered: dropped anything matching {without!r}, "
              f"{len(PENDING)} decisions left\n")
    tol = tol_for(X8, Y8)
    w0, resid = fit(X8, Y8)
    print(f"eight rounds, max |residual| {np.abs(resid).max():.4f}, tol {tol:.4f}")
    print("fit  " + "  ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, w0)))

    print("\n=== decisions currently blocked by weight ambiguity ===")
    print(f"{'decision':30s} {'d final lo':>11s} {'hi':>8s} {'width':>8s}  sign?")
    base = ambiguity(X8, Y8, tol)
    base_total = 0.0
    for label, lo, hi, w, amb in base:
        base_total += w
        print(f"{label:30s} {lo:+11.3f} {hi:+8.3f} {w:8.3f}  {'AMBIGUOUS' if amb else 'settled'}")
    print(f"{'total ambiguity width':30s} {'':11s} {'':8s} {base_total:8.3f}")

    rng = np.random.default_rng(seed)
    print("\n=== candidate submissions ===")
    print(f"{'candidate':32s} {'pred final':>10s} {'range':>7s} {'resolves':>9s} "
          f"{'left':>7s} {'worst':>7s}")
    rows = []
    for label, x, note in CANDIDATES:
        xj = x.copy()
        if jitter:
            xj[3] += rng.normal(0.0, jitter)
        width, left, worst = value_of(xj, X8, Y8, tol)
        mid = float(w0 @ xj)
        rows.append((label, mid, width, base_total - left, left, worst, note))
    for label, mid, width, res, left, worst, note in sorted(rows, key=lambda r: -r[3]):
        print(f"{label:32s} {mid:10.3f} {width:7.3f} {res:9.3f} {left:7.3f} {worst:7.3f}")
    print("\n  `range`    how much the eight rounds fail to pin this package's final = raw info")
    print("  `resolves` total decision-ambiguity removed, averaged over admissible outcomes")
    print("  `worst`    ambiguity left in the worst outcome -- a probe that can fail to settle")
    print("\nnotes:")
    for label, _, note in CANDIDATES:
        print(f"  {label:32s} {note}")



# ------------------------------------------------------------ what a probe SHOULD move
def undetermined_axes(X, Y, tol, n_dirs=400, seed=0):
    """The subscore trades the eight rounds price least confidently.

    Informativeness of a candidate is the width of the admissible family's support in the
    direction of its subscore vector.  So the most informative packages to BUILD are the ones
    that move subscores along the family's own spread directions -- which are, by construction,
    trades between subscores rather than uniform improvements.  That is why a package that is
    simply better everywhere buys almost nothing: it is nearly collinear with `1`, and sum(w)
    is the one thing already fixed.
    """
    rng = np.random.default_rng(seed)
    V = []
    for _ in range(n_dirs):
        c = rng.normal(size=5)
        r = linprog(c, A_ub=np.vstack([X, -X]), b_ub=np.concatenate([Y + tol, -(Y - tol)]),
                    A_eq=np.ones((1, 5)), b_eq=np.array([1.0]), bounds=[(0.0, None)] * 5)
        if r.success:
            V.append(r.x)
    V = np.unique(np.round(np.array(V), 9), axis=0)
    C = V - V.mean(0)
    _, sv, vt = np.linalg.svd(C, full_matrices=False)
    return V, sv, vt


def axes_report():
    tol = tol_for(X8, Y8)
    V, sv, vt = undetermined_axes(X8, Y8, tol)
    print(f"\n=== the subscore trades we cannot price ({len(V)} vertices of the family) ===")
    print("A candidate's information is the family's width along its subscore vector, so these")
    print("are the directions worth building a probe to move.  Signs are arbitrary in pairs.")
    for k in range(3):
        d = vt[k] / np.abs(vt[k]).max()
        lo, hi, w = span(d, X8, Y8, tol)
        trade = "  ".join(f"{n} {v:+.2f}" for n, v in zip(NAMES, d))
        print(f"  axis {k + 1}  spread {sv[k] / sv[0]:5.2f}  width {w:6.3f}   {trade}")
    print("\n  A package that is uniformly better moves along `1`, and sum(w) = 1 is the one")
    print("  thing already known -- which is why 'a better ensemble' is worth ~0.03 and one")
    print("  changed constant that trades rel_l2 against tke is worth ~0.65.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="std dev of a perturbation applied to the predicted `time` subscore, "
                         "to test whether the ranking rests on the column we cannot predict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--without", default=None,
                    help="drop every PENDING decision whose label contains this substring, "
                         "then re-rank. The check that matters is `--without ALPHA`: a probe "
                         "that only resolves the decision it IS would be informative about "
                         "itself and nothing else. (Measured: the ALPHA 0.60 probe still "
                         "resolves 44%% of what is left after ALPHA is removed, so it is not.)")
    a = ap.parse_args()
    run(a.jitter, a.seed, a.without)
    axes_report()
