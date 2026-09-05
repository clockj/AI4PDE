"""Prove two BUILT ARCHIVES return the same arrays -- the gate a pure-latency A/B rests on.

Round 17's slot pair changes only how the shipped path computes, never what it computes: the
Track-1 arm turns on `HOST_WRITE = "torch"` (bit-identical by construction) and the Track-2 arm
turns on `SPEC_PATH = "bmm"` (~2e-06 relative, a different accumulation order). The whole
experiment is that the four ERROR subscores come back identical to R17 / T2R5 and the pair
reads `time` alone -- which is only true if the arrays really are the same. This checks that
directly, archive against archive, on the same 336 val_unseen windows.

    python verify/pkg_identity.py --track t1 --a <extracted old> --b <extracted new>
    python verify/pkg_identity.py --track t2 --a <extracted old> --b <extracted new>

Each package is imported in its own right, from its own directory, so the comparison is between
what the two ZIPS do -- not between two settings of the repo's working tree.
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
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "Track_2"))

VAL = ["11400_20", "13950_10", "16500_5", "21600_15", "26700_0", "3750_0", "6300_10", "8850_20"]


def load_pkg(d, tag):
    d = Path(d).resolve()
    for k in list(sys.modules):
        if k in (tag, "_fno", "submission"):
            del sys.modules[k]
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(tag, d / "submission.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m, d


def unload(tag, d):
    for k in list(sys.modules):
        if k in (tag, "_fno"):
            del sys.modules[k]
    if str(d) in sys.path:
        sys.path.remove(str(d))


def run_t1(d, tag, X):
    m, d = load_pkg(d, tag)
    # getattr throughout: the BASELINE archive is the one already on the board, built before
    # these constants existed, and it has neither `_TF32` nor `HOST_WRITE`/`SPEC_PATH`. An
    # older package failing to introspect is not a difference in what it computes.
    print(f"  {tag}: ALPHA={m.ALPHA} HOST_WRITE={getattr(m, 'HOST_WRITE', '<absent>')!r} "
          f"SPEC_PATH={getattr(m, 'SPEC_PATH', '<absent>')!r} "
          f"TF32={getattr(m, '_TF32', '<absent>')}", flush=True)
    o = m.predict(X)
    out = tuple(np.asarray(o[k], dtype=np.float32).copy()
                for k in ("prediction", "lower", "upper"))
    unload(tag, d)
    return out


def run_t2(d, tag, stream):
    import t2_common as t2
    m, d = load_pkg(d, tag)
    print(f"  {tag}: ALPHA={m.ALPHA} SPEC_PATH={getattr(m, 'SPEC_PATH', '<absent>')!r} "
          f"TF32={getattr(m, '_TF32', '<absent>')}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obj = m.get_ttt_model(str(d), dev)
    print(f"  {tag}: CUDA graph {'ACTIVE' if getattr(obj, 'graph', None) is not None else 'off'}",
          flush=True)
    P, L, U = [], [], []
    for name, tid, x_raw, y_raw, first in stream:
        if first:
            obj.reset_ttt_state()
        pn, info = obj.ttt_step(torch.from_numpy(t2.normalize_in(x_raw)[None]), None)
        # .copy() -- round 13 found a gate that handed back a reused staging buffer and every
        # stored prediction became the last one.
        P.append(t2.denorm_pred(np.asarray(pn)[0]).copy())
        L.append(t2.denorm_pred(np.asarray(info["lower"])[0]).copy())
        U.append(t2.denorm_pred(np.asarray(info["upper"])[0]).copy())
    out = tuple(np.stack(z).astype(np.float32) for z in (P, L, U))
    del obj
    unload(tag, d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["t1", "t2"], required=True)
    ap.add_argument("--a", required=True, help="extracted BASELINE archive (on the board)")
    ap.add_argument("--b", required=True, help="extracted CANDIDATE archive")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max|delta| relative to the field max that still counts as identical")
    a = ap.parse_args()

    import t2_common as t2
    stream = t2.build_stream(VAL)
    X = np.stack([s[2] for s in stream]).astype(np.float32)
    print(f"{len(X)} windows from {len(VAL)} trajectories, track {a.track}", flush=True)

    if a.track == "t1":
        A = run_t1(a.a, "pkgA", X)
        B = run_t1(a.b, "pkgB", X)
    else:
        A = run_t2(a.a, "pkgA", stream)
        B = run_t2(a.b, "pkgB", stream)

    scale = float(np.abs(A[0][..., :2]).max())
    print(f"\n{'array':<12} {'max|A-B|':>12} {'rel to field max':>18} {'rel-L2':>12}")
    worst = 0.0
    for nm, x1, x2 in zip(("prediction", "lower", "upper"), A, B):
        d = float(np.abs(x1 - x2).max())
        rl = float(np.linalg.norm((x1 - x2).ravel()) / max(np.linalg.norm(x1.ravel()), 1e-30))
        worst = max(worst, d / scale)
        print(f"{nm:<12} {d:>12.3e} {d/scale:>18.3e} {rl:>12.3e}")
    ok = worst <= a.tol
    print(f"\nfield max {scale:.4f}; worst relative disagreement {worst:.3e} against a "
          f"tolerance of {a.tol:.0e} -> {'IDENTICAL' if ok else '*** DIFFERS ***'}")
    print("A pure-latency A/B is only valid if this passes: the four error subscores must be "
          "the\nsame numbers the board already returned, so that the pair reads `time` alone.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
