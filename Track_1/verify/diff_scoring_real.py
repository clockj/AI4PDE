"""OFFICIAL kit-v9 scoring.py vs our reconstruction, on the REAL val_unseen windows.

Synthetic arrays cannot exercise the real zero-mask geometry (the PIV field-of-view boundary
and the airfoil body), the real u/v magnitude ratio, or the real probe-column statistics --
and the MVPE bug found on 2026-08-29 lived in exactly that territory.  This scores the two
implementations on the actual data with GPU-free predictors, so it runs on a login node.

It also re-derives CLAUDE.md's documented anchor rows against the OFFICIAL scorer, which
validates the data pipeline (loading, windowing, sub-sampling) at the same time.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9/scoring.py"
RECON = "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py"

# Far below the board's 5-decimal reporting and the ~0.05 run-to-run noise floor.
TOL_SUBSCORE = 1e-4


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def five(mod, P, Y, lo=None, up=None):
    c = 2
    rel = float(np.mean(mod.rel_l2_per_sample(P, Y, c)))
    tke = float(np.mean(mod.tke_rel_l2_per_sample(P, Y, c)))
    mvpe = float(mod.mvpe_rel_l2(P, Y))
    w, cov = mod.aggregate_sps(P, Y, c, lower=lo, upper=up)
    return dict(rel_l2=rel, tke=tke, mvpe=mvpe, coverage=cov,
                s_rel=mod.score_error(rel), s_tke=mod.score_error(tke),
                s_mvpe=mod.score_error(mvpe), s_sps=mod.score_sps(w))


def main() -> int:
    off = load(OFFICIAL, "s_off")
    rec = load(RECON, "s_rec")
    d = np.load(os.path.join(HERE, os.pardir, "val_unseen.npz"))
    X, Y = d["x"].astype(np.float32), d["y"].astype(np.float32)
    print(f"val_unseen: X {X.shape}  Y {Y.shape}")
    zero_frac = float(np.mean(Y[..., :2] == 0.0))
    print(f"zero-target fraction in u,v: {zero_frac:.4f}  (CLAUDE.md documents ~0.099)\n")

    rng = np.random.default_rng(20260831)
    cases = {
        # The two documented anchor predictors -- these pin the data pipeline.
        "persistence (repeat last input frame)":
            np.repeat(X[:, -1:], Y.shape[1], axis=1),
        "input time-mean":
            np.repeat(X.mean(axis=1, keepdims=True), Y.shape[1], axis=1),
        # A model-quality predictor, so the comparison also covers the regime we ship in.
        "target + 8% noise (model-like)":
            (Y + rng.normal(0, 0.008, Y.shape)).astype(np.float32),
    }

    worst = 0.0
    for name, P in cases.items():
        P = P.astype(np.float32).copy()
        P[..., 2] = 0.0
        half = np.full(P.shape, 0.0104, np.float32)
        lo, up = (P - half).astype(np.float32), (P + half).astype(np.float32)
        o, r = five(off, P, Y, lo, up), five(rec, P, Y, lo, up)
        print(f"--- {name} ---")
        for k in o:
            dd = abs(o[k] - r[k])
            worst = max(worst, dd)
            flag = "OK  " if dd <= TOL_SUBSCORE else "DIFF"
            print(f"  [{flag}] {k:<10s} official={o[k]:>12.6f}  recon={r[k]:>12.6f}  |d|={dd:.2e}")
        print()

    print("Documented anchors (CLAUDE.md) vs the OFFICIAL scorer:")
    for name, doc in (("persistence (repeat last input frame)",
                       dict(rel_l2=0.138, tke=1.000, mvpe=0.141)),
                      ("input time-mean", dict(rel_l2=0.134, mvpe=0.124))):
        P = cases[name].astype(np.float32).copy()
        P[..., 2] = 0.0
        o = five(off, P, Y)
        row = "  ".join(f"{k} {o[k]:.4f} (doc {v:.3f})" for k, v in doc.items())
        print(f"  {name:<40s} {row}")

    print(f"\nworst |official - recon| over every quantity: {worst:.3e}"
          f"   tolerance {TOL_SUBSCORE:g}")
    print("VERDICT: " + ("score-identical -- every round-12 number stands."
                         if worst <= TOL_SUBSCORE else "MISMATCH -- round-12 numbers need a re-run."))
    return 0 if worst <= TOL_SUBSCORE else 1


if __name__ == "__main__":
    raise SystemExit(main())
