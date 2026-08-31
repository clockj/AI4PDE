"""Prove the Track-2 plumbing is the Track-1 model, exactly.

Track 2 works in the evaluator's normalized space and Track 1 in raw physical units, so the
two paths cannot be compared by inspection -- an error in the affine conversion would look
like a small accuracy loss and be blamed on the model.  This runs BOTH on the same windows and
prints the max absolute disagreement in physical units, for the prediction and for both SPS
bounds.

    python verify_space.py --submission submit_ho --traj 3750_0 16500_5
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2_common as t2  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submit_ho")
    ap.add_argument("--t1", default=str(HERE.parent / "Track_1" / "submit_fno"))
    ap.add_argument("--traj", nargs="+", default=["3750_0"])
    a = ap.parse_args()

    stream = t2.build_stream(a.traj)
    X = np.stack([s[2] for s in stream]).astype(np.float32)
    print(f"{len(X)} windows from {a.traj}")

    # --- Track 1 path: raw in, raw out ------------------------------------------------
    d1 = Path(a.t1).resolve()
    sys.path.insert(0, str(d1))
    t1 = load(d1 / "submission.py", "t1_submission")
    out1 = t1.predict(X)
    P1, L1, U1 = out1["prediction"], out1["lower"], out1["upper"]
    for k in list(sys.modules):
        if k in ("t1_submission", "_fno"):
            del sys.modules[k]
    sys.path.remove(str(d1))

    # --- Track 2 path: normalized in, normalized out, denormalized here ----------------
    d2 = Path(a.submission).resolve()
    sys.path.insert(0, str(d2))
    t2m = load(d2 / "submission.py", "t2_submission")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = t2m.get_ttt_model(str(d2), dev)
    P2, L2, U2 = [], [], []
    for name, tid, x_raw, y_raw, first in stream:
        if first:
            model.reset_ttt_state()
        pn, info = model.ttt_step(torch.from_numpy(t2.normalize_in(x_raw)[None]), None)
        P2.append(t2.denorm_pred(np.asarray(pn)[0]))
        L2.append(t2.denorm_pred(np.asarray(info["lower"])[0]))
        U2.append(t2.denorm_pred(np.asarray(info["upper"])[0]))
    P2, L2, U2 = (np.stack(z).astype(np.float32) for z in (P2, L2, U2))

    scale = float(np.abs(P1[..., :2]).max())
    for nm, a1, a2 in (("prediction", P1, P2), ("lower", L1, L2), ("upper", U1, U2)):
        d = float(np.abs(a1[..., :2] - a2[..., :2]).max())
        print(f"  {nm:11s} max|T1-T2| = {d:.3e}   ({d/scale:.2e} of field max {scale:.4f})")
    print(f"  p channel  T1 max |p| {np.abs(P1[...,2]).max():.1e}   "
          f"T2 max |p| {np.abs(P2[...,2]).max():.1e}")


if __name__ == "__main__":
    main()
