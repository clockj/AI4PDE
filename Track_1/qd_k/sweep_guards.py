"""Four asserts for band/hyper-parameter sweeps. Import them; do not re-derive them.

STATUS: live, 2026-09-23, Devi. RELEASE CONDITION: RESOLVED 2026-09-23 -- Bram ruled KEEP
BOTH. His `work/BAND_COMPARISON_TRAPS.md` and this file do different jobs, and he gave the
reason: a guard's value is not only that it can raise, it is WHAT IT FORCES THE PRODUCER TO
RECORD -- `check_not_edge` made him change his output format, not just add an assertion.
Prose cannot force anyone to change a print statement. So: read TRAPS.md for the five traps
and the reasoning; import this for the asserts. Neither supersedes the other.

Every one of these exists because it FIRED on a real reading on 2026-09-23 -- three of them on
mine. A rule in someone's notes is read by its author; an assert is read by whoever is about to
repeat the mistake. That asymmetry is the whole point of this file.

    from sweep_guards import check_grid, check_not_edge, check_band_family, effective_n

Pure stdlib + numpy. No GPU. Seconds.
"""
from __future__ import annotations
import numpy as np


def check_grid(grid, baseline, name="baseline"):
    """The sweep grid MUST contain the value you are measuring FROM, as an exact key.

    Fired 3x on 2026-09-23 (mine), after I had written the rule down twice:
      band_p grid np.arange(0, 1.61, .10) does not contain the shipped 0.45
      k grid 0.84..0.98 excludes the deployed 1.0
    Look the baseline up by EXACT KEY, never "nearest grid point": a nearest-point lookup silently
    substitutes a different comparison and returns a plausible number, where an exact key CRASHES.
    """
    g = [round(float(x), 10) for x in np.asarray(grid).ravel()]
    b = round(float(baseline), 10)
    if b not in g:
        raise AssertionError(
            f"{name}={baseline!r} is NOT in the grid (min {min(g)}, max {max(g)}, n={len(g)}). "
            f"Insert it explicitly; do not rely on a nearby point.")
    return True


def check_near_baseline(grid, baseline, finest_step, name="baseline"):
    """Containing the baseline is NOT enough: the grid needs a neighbour CLOSE to it on each side
    that exists, or "the optimum is exactly the baseline" is only "the optimum is closer to the
    baseline than my first step".

    THIS GUARD EXISTS BECAUSE OF MY OWN "DEAD ZONE" (2026-09-23). check_grid passed -- lambda=0 was
    an exact key -- but the grid was {0, 0.25, 0.5, 0.75}. I published "rho <= 0.25 => lambda* = 0,
    no small win exists". Re-run with {0, .01, .02, .05, .10, .15, .25, ...}: lambda* = 0.05 / 0.10 /
    0.15 at rho = 0.15 / 0.25 / 0.35. Never zero. Halvor found the same on four keys the same day.

    finest_step = the smallest effect you would want to be able to see, in the parameter's units.
    The nearest grid point on each available side must be within it.
    """
    g = sorted(round(float(x), 10) for x in np.asarray(grid).ravel())
    b = round(float(baseline), 10)
    check_grid(g, b, name)
    above = [x for x in g if x > b]
    below = [x for x in g if x < b]
    bad = []
    if above and above[0] - b > finest_step:
        bad.append(f"nearest point above is {above[0]} (gap {above[0]-b:g})")
    if below and b - below[-1] > finest_step:
        bad.append(f"nearest point below is {below[-1]} (gap {b-below[-1]:g})")
    if bad:
        raise AssertionError(
            f"{name}={baseline!r}: " + "; ".join(bad) + f" > finest_step {finest_step:g}. "
            f"An argmax AT the baseline cannot be distinguished from one inside that gap. "
            f"Add points between.")
    return True


def check_not_edge(per_fold_choice, grid, name="argmax"):
    """Report the chosen hyper-parameter FOR EVERY FOLD; all folds at one endpoint => edge solution.

    Fired on my k sweep: all 8 LOTO folds returned 0.84 = the grid's lower edge, which read as
    "+0.1722 final, RESOLVED". Opening the grid down to 0 moved it to 0.80 interior and the result
    fell to t=2.04, NOT resolved. Mean and t alone would have shipped a gain larger than the whole
    T1 gap.
    Guards the grid's EDGES where check_grid guards its CONTENTS.
    """
    ch = [round(float(c), 10) for c in per_fold_choice]
    lo, hi = round(float(min(grid)), 10), round(float(max(grid)), 10)
    if all(c == lo for c in ch) or all(c == hi for c in ch):
        raise AssertionError(
            f"{name}: every fold chose {ch[0]} = a grid endpoint ({lo}..{hi}) => EDGE SOLUTION. "
            f"Widen the grid past it and re-run before quoting any delta.")
    return True


def check_band_family(band_npz_path):
    """Which interval family does this band deploy? One key decides it.

    `submission.py`: CENTRE_ALPHA is read from band.npz; NO KEY => falls back to ALPHA => centre is
    None => lower/upper = pred -+ h, SYMMETRIC. With the key it clamps:
    lo=min(c-h,P), up=max(c+h,P).
    SCOPE (Lior, 2026-09-23): repo `Track_1/bands/*.npz` are INTERMEDIATES -- 1 of 16 carries the
    key -- while SHIPPED zips carry it 10 of 10. Apply this to the band.npz inside a shipped zip,
    not to the staging copy, or you will conclude the shipped family is symmetric. It is not.
    """
    z = np.load(band_npz_path, allow_pickle=True)
    clamped = "centre_alpha" in z.files
    return {"clamped": clamped,
            "centre_alpha": float(z["centre_alpha"]) if clamped else None,
            "family": "clamped (asymmetric)" if clamped else "symmetric",
            "note": "repo bands are intermediates; check the band.npz INSIDE the shipped zip"}


def effective_n(n_elements, n_windows, n_trajectories):
    """Elements are not independent units. Report all three; the smallest binds.

    `elements/param == n_windows` is an IDENTITY for a per-element band (each element learns from
    one observation per window), so that ratio can never diagnose overfitting. My fixed field had
    126 elements/param -- looked healthy -- and the held-out result FLIPPED SIGN (-0.555).
    Neighbouring pixels, frames, and windows within one trajectory are all correlated, so the
    honest denominator is trajectories.
    """
    return {"elements": int(n_elements), "windows": int(n_windows),
            "trajectories": int(n_trajectories),
            "binding": "trajectories",
            "crit_t975": {2: 4.303, 6: 2.447, 7: 2.365}.get(int(n_trajectories) - 1, "look it up"),
            "warning": "never quote elements-per-parameter as evidence against overfitting"}


def check_sigma_provenance(source_file=None, member_set=None, sigma_ref_cell=None):
    """A sigma array without its provenance is not a quantity. Declare all three or stop.

    THIS GUARD EXISTS BECAUSE OF MY OWN ERROR ON 2026-09-23. I published a **42x** clip-activity
    ratio by putting my in-range number beside Fen's extrapolation number; Bram re-measured on one
    consistent instrument and got **5.8x**. The two came from different sigma sources AND different
    sigma_ref cells -- neither of which was recorded next to either number.
    Bram's rule, which this enforces: **never store or pass a key called "sigma".** Name the key
    explicitly and record, in the same place, WHICH MODEL / MEMBER SET produced it. Changing the
    driver moves the upper-clip bite rate by ~25%, so two "sigmas" are routinely different
    quantities wearing one word.

        check_sigma_provenance(source_file="twin_export/export_unseen.npz",
                               member_set="twin 5 (p2_ffno_l8_sim_s1..s3 + 2 ViT)",
                               sigma_ref_cell="fit split, AoA<15")
    """
    missing = [n for n, v in (("source_file", source_file), ("member_set", member_set),
                              ("sigma_ref_cell", sigma_ref_cell)) if not v]
    if missing:
        raise AssertionError(
            f"sigma provenance incomplete: {missing} not declared. A sigma is only comparable to "
            f"another sigma from the SAME source, member set and sigma_ref cell. Declare all three "
            f"or do not put the two numbers side by side.")
    return {"source_file": source_file, "member_set": member_set,
            "sigma_ref_cell": sigma_ref_cell,
            "comparable_to": "only another sigma with all three fields identical"}


def check_cell_constant(name, value, from_cell=None, used_in_cell=None, current_value=None,
                        acknowledged_stale=None):
    """A constant measured on cell A is not a constant on cell B. Make it carry its cell.

    THIS GUARD EXISTS BECAUSE OF A CASE JONAS FOUND IN HIS OWN WORK ON 2026-09-23, and he said
    it was not the first: "我今天第 N 次把一个旧格的常数带进新算式". He reported the sps share of
    a precision arm as **39.4%**, computed with **C = 0.6537** taken from the r34 cell, while the
    formula was being evaluated on the BOARD cell where **C = 0.58952**. Correct value **37.0%**.
    Nothing was wrong with the algebra; the constant had silently changed cell.

    This generalises check_sigma_provenance from sigma to ANY carried constant. The point is not
    the raise -- it is that the producer must write down which cell the number came from, next to
    the number, at the moment they use it.

        check_cell_constant("C", 0.6537, from_cell="r34", used_in_cell="board T1",
                            current_value=0.58952)

    Pass acknowledged_stale="<reason>" to proceed deliberately; the reason is returned so it lands
    in the output instead of in someone's head.
    """
    if not from_cell or not used_in_cell:
        raise AssertionError(
            f"constant {name!r}={value!r} has no cell: declare from_cell= and used_in_cell=. "
            f"A constant without its cell cannot be checked by the next reader, and the failure "
            f"mode (right algebra, wrong cell) leaves no trace in the output.")
    if from_cell == used_in_cell:
        return {"name": name, "value": value, "cell": from_cell, "status": "same cell"}
    if acknowledged_stale:
        return {"name": name, "value": value, "from_cell": from_cell,
                "used_in_cell": used_in_cell, "status": "cross-cell, acknowledged",
                "reason": acknowledged_stale}
    drift = ""
    if current_value is not None:
        try:
            rel = abs(value - current_value) / abs(current_value)
            drift = (f" The value on {used_in_cell!r} is {current_value!r} "
                     f"({rel:.1%} away) -- results carrying this constant are wrong by about that.")
        except (TypeError, ZeroDivisionError):
            drift = f" The value on {used_in_cell!r} is {current_value!r}."
    raise AssertionError(
        f"constant {name!r}={value!r} was measured on {from_cell!r} but is being used on "
        f"{used_in_cell!r}.{drift} Re-measure it on {used_in_cell!r}, or pass "
        f"acknowledged_stale='<reason>' so the reason appears in the output.")


if __name__ == "__main__":
    # self-test: each guard must FIRE on the exact case that motivated it
    try:
        check_grid(np.arange(0.0, 1.61, 0.10), 0.45, "band_p shipped")
        print("FAIL: check_grid did not fire on the real 2026-09-23 case")
    except AssertionError:
        print("ok  check_grid fires on band_p grid without 0.45")
    try:
        check_not_edge([0.84] * 8, np.arange(0.84, 1.21, 0.02), "k*")
        print("FAIL: check_not_edge did not fire")
    except AssertionError:
        print("ok  check_not_edge fires when all 8 folds sit on the lower edge")
    print("ok  effective_n:", effective_n(10_321_920, 126, 3))
    try:
        check_sigma_provenance(source_file="x.npz")   # the 42x-vs-5.8x case: fields omitted
        print("FAIL: check_sigma_provenance did not fire")
    except AssertionError:
        print("ok  check_sigma_provenance fires when member_set/sigma_ref_cell are undeclared")
    try:
        check_cell_constant("C", 0.6537, from_cell="r34", used_in_cell="board T1",
                            current_value=0.58952)
        print("FAIL: check_cell_constant did not fire on the real 39.4%-vs-37.0% case")
    except AssertionError as e:
        print("ok  check_cell_constant fires when a constant crosses cells")
    try:
        check_cell_constant("C", 0.58952)
        print("FAIL: check_cell_constant did not fire on an undeclared cell")
    except AssertionError:
        print("ok  check_cell_constant fires when the cell was never written down")
    print("ok  check_cell_constant acknowledged:",
          check_cell_constant("C", 0.6537, from_cell="r34", used_in_cell="board T1",
                              acknowledged_stale="quoting the r34 row on purpose")["status"])
    try:
        check_near_baseline([0.0, 0.25, 0.5, 0.75], 0.0, 0.05, "lambda")
        print("FAIL: check_near_baseline did not fire on the dead-zone grid")
    except AssertionError:
        print("ok  check_near_baseline fires on my dead-zone lambda grid {0,.25,.5,.75}")
    print("ok  check_near_baseline passes the fine grid:",
          check_near_baseline([0,.01,.02,.05,.10,.15,.25,.5,.75], 0.0, 0.05, "lambda"))
