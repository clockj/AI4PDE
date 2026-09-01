"""Prove that two packages differ ONLY by the thing you think they differ by.

    python verify/ab_identical.py --a <extracted R13 dir> --b <extracted R14 dir> \
        --zero BLEND_BETA

Now that slots are designed in pairs against a package already on the same board (ROADMAP
§1.16), every round makes a claim of the form *"the only difference is X"* -- and the board
delta is only attributable to X if that claim is exact.  It usually is not for free: the model
files can be byte-identical while `_fno.py` has moved underneath them.  Round 14's Track-1
package is a real example: nine of eleven files matched the round-13 archive byte for byte, but
`_fno.py` had gained self-norm support, oscillatory-head fusion and `--tbasis` in between.  All
three are gated on checkpoint attributes those members do not carry, so the numerics *should*
be untouched -- and `validate_strict` reproduced rel/tke/mvpe to four figures.  That is
evidence, not proof.  This is the proof: set the new feature to its off value and require the
two packages to agree BIT for BIT on real windows.

`--zero NAME` sets module attribute NAME to 0.0 in whichever package has it, which is how the
feature under test is switched off.  Repeat it for several.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag, d):
    spec = importlib.util.spec_from_file_location(tag, os.path.join(d, "submission.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    sys.path.insert(0, d)
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path.pop(0)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="extracted reference package (already scored)")
    ap.add_argument("--b", required=True, help="extracted candidate package")
    ap.add_argument("--zero", nargs="*", default=["BLEND_BETA"],
                    help="module constants to force to 0.0 before comparing")
    args = ap.parse_args()

    X = np.concatenate([np.load(os.path.join(HERE, f"val_{s}.npz"))["x"]
                        for s in ("seen", "unseen")]).astype(np.float32)
    print(f"{len(X)} windows")
    out = {}
    for tag, d in (("a", args.a), ("b", args.b)):
        m = load("_ab_" + tag, d)
        for nm in args.zero:
            if hasattr(m, nm):
                print(f"  {tag}: {nm} {getattr(m, nm)} -> 0.0 for this comparison")
                setattr(m, nm, 0.0)
        o = m.predict(X)
        out[tag] = tuple(o[k].copy() for k in ("prediction", "lower", "upper"))
        del m

    ok = True
    for i, nm in enumerate(("prediction", "lower", "upper")):
        d = float(np.abs(out["a"][i].astype(np.float64)
                         - out["b"][i].astype(np.float64)).max())
        ok &= d == 0.0
        print(f"  {nm:<11s} max |a - b| = {d:.3e}   "
              f"{'BIT-IDENTICAL' if d == 0.0 else 'DIFFERS'}")
    print("\n" + ("PASS: the board delta is attributable to the feature under test"
                  if ok else "FAIL: something else moved; the A/B is NOT clean"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
