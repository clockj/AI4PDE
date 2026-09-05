"""Numerical diff: OFFICIAL kit-v9 scoring.py vs our 2026-08-29 reconstruction.

The reconstruction was written from CLAUDE.md's prose because Codabench was auth-walled from
the cluster; every round-12 number went through it.  The official kit arrived 2026-08-31.
This compares the two implementations OUTPUT-BY-OUTPUT on the same arrays -- a textual diff
would drown in cosmetic differences and prove nothing about the scores.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

OFFICIAL = Path("/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9/scoring.py")
RECON = Path("/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py")

# The two files accumulate in different dtypes -- the reconstruction casts to float64 inside
# kinetic_energy where the official keeps float32 -- so they differ by float roundoff, not by
# semantics.  1e-5 on a subscore is four orders below the 0.05 run-to-run noise floor and far
# below the board's 5-decimal reporting; anything above it would be a real disagreement.
TOL = 1e-5


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def report(label, a, b, tol=TOL):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    flag = "OK  " if d <= tol else "DIFF"
    scal = a.size == 1
    extra = f"   official={float(a):.10f}  recon={float(b):.10f}" if scal else ""
    print(f"  [{flag}] {label:<28s} max|diff| = {d:.3e}{extra}")
    return d <= tol


def make_case(rng, n=12, masked=True):
    """A realistic window batch: (N,20,32,64,3), p-channel zeroed, ~10% zero targets."""
    y = rng.normal(0.155, 0.10, size=(n, 20, 32, 64, 3)).astype(np.float32)
    y[..., 1] *= 0.16          # v is ~6x smaller than u, as in the real data
    y[..., 2] = 0.0            # p is zero-filled and unscored
    if masked:
        # ~9.9% of u,v elements are exactly 0 (outside the PIV FOV / inside the airfoil)
        hole = rng.random((1, 1, 32, 64, 1)) < 0.099
        y = np.where(hole, 0.0, y).astype(np.float32)
    p = (y + rng.normal(0, 0.011, size=y.shape)).astype(np.float32)
    p[..., 2] = 0.0
    return p, y


def main() -> int:
    off = load(OFFICIAL, "scoring_official")
    rec = load(RECON, "scoring_recon")
    rng = np.random.default_rng(20260831)
    ok = True

    print(f"SIGMA_GLOBAL   official={off.SIGMA_GLOBAL!r}  recon={rec.SIGMA_GLOBAL!r}")
    ok &= off.SIGMA_GLOBAL == rec.SIGMA_GLOBAL

    for tag, masked in (("masked (realistic)", True), ("dense (no zero targets)", False)):
        print(f"\n--- {tag} ---")
        P, Y = make_case(rng, masked=masked)
        c = 2

        ok &= report("rel_l2_per_sample",
                     off.rel_l2_per_sample(P, Y, c), rec.rel_l2_per_sample(P, Y, c))
        ok &= report("tke_rel_l2_per_sample",
                     off.tke_rel_l2_per_sample(P, Y, c), rec.tke_rel_l2_per_sample(P, Y, c))
        ok &= report("mvpe_rel_l2_per_sample",
                     off.mvpe_rel_l2_per_sample(P, Y), rec.mvpe_rel_l2_per_sample(P, Y))
        ok &= report("mvpe_rel_l2",
                     off.mvpe_rel_l2(P, Y), rec.mvpe_rel_l2(P, Y))

        # SPS: the scorer's DEFAULT band, then a realistic fitted-style band.
        w_o, cov_o = off.aggregate_sps(P, Y, c)
        w_r, cov_r = rec.aggregate_sps(P, Y, c)
        ok &= report("aggregate_sps w (default)", w_o, w_r)
        ok &= report("aggregate_sps coverage", cov_o, cov_r)

        half = np.full_like(P, 0.0104, dtype=np.float32)     # our fitted median half-width
        half *= (1.0 + 0.5 * rng.random(P.shape).astype(np.float32))
        lo, up = (P - half).astype(np.float32), (P + half).astype(np.float32)
        w_o, cov_o = off.aggregate_sps(P, Y, c, lower=lo, upper=up)
        w_r, cov_r = rec.aggregate_sps(P, Y, c, lower=lo, upper=up)
        ok &= report("aggregate_sps w (our band)", w_o, w_r)
        ok &= report("aggregate_sps coverage", cov_o, cov_r)

        print("  five subscores:")
        for name, o, r in (
            ("rel_l2", off.score_error(float(np.mean(off.rel_l2_per_sample(P, Y, c)))),
                       rec.score_error(float(np.mean(rec.rel_l2_per_sample(P, Y, c))))),
            ("tke",    off.score_error(float(np.mean(off.tke_rel_l2_per_sample(P, Y, c)))),
                       rec.score_error(float(np.mean(rec.tke_rel_l2_per_sample(P, Y, c))))),
            ("mvpe",   off.score_error(off.mvpe_rel_l2(P, Y)),
                       rec.score_error(rec.mvpe_rel_l2(P, Y))),
            ("sps",    off.score_sps(w_o), rec.score_sps(w_r)),
        ):
            ok &= report(f"score_{name}", o, r)

    print("\n--- score curves (scalars) ---")
    for t in (1e-3, 2.235e-3, 6.7e-3, 1e-2, 0.1, 0.72896, 0.7):
        ok &= report(f"score_time({t})", off.score_time(t), rec.score_time(t))
    for e in (0.0, 0.031, 0.0862, 0.118, 0.483, 1.0, 2.0):
        ok &= report(f"score_error({e})", off.score_error(e), rec.score_error(e))
    for w in (0.0, 0.1, 0.3696, 0.5, 1.0):
        ok &= report(f"score_sps({w})", off.score_sps(w), rec.score_sps(w))

    print("\n" + ("ALL MATCH -- the reconstruction is score-identical to the official kit."
                  if ok else "MISMATCH -- see the DIFF rows above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
