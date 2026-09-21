"""Runtime patch: make the reconstructed v9 scorer use the REAL kit's MVPE.

`verify_scoring_20260829.py` showed the reconstruction at
`~/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py` pools all 36 wake
probes into a single relative L2.  The real kit takes a relative L2 per probe COLUMN -- each of
the four x-stations normalized by its own 9-site x 2-channel target norm -- and averages the
four.  Anchors on val_unseen (measured through the real kit):

                        anchor   pooled-36   per-column
    persistence          0.141      0.1557       0.1408
    input time-mean      0.124      0.1414       0.1240

Per-column denominators up-weight the far wake, where velocities are small; the two forms
differ by ~10-14% and the pooled one is always pessimistic.  The correction is transcribed
independently three times in this repo from when the kit was reachable:
unet/train2.py:407-420 (commented "scoring.mvpe_rel_l2_per_sample, verbatim"),
diag_mean_floor.py:36-43 and diag_history.py:71-73.

This module does NOT edit scoring.py (another agent owns code changes).  It rebinds
`mvpe_rel_l2_per_sample` on the shared `sps9.sc` module object.  That is sufficient for every
consumer, because `mvpe_rel_l2` and `aggregate_sps` both look the name up in the module
namespace at call time, and all 24 call sites in Track_1 import the same module instance via
`from sps9 import sc`.  SPS is corrected too -- its per-sample accuracy factor weights mvpe 0.2.

Usage -- wrap any existing entry point, argv passes straight through:

    python mvpe_fix_20260829.py compare_v9 --ckpt unet/fno_vl_s1.pth --aoa 15 20
    python mvpe_fix_20260829.py ens_v9 --ckpt ...

or from inside another script:

    import mvpe_fix_20260829 as mvpefix; mvpefix.apply()
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

import sps9

_ORIGINAL = None


def _mvpe_percol(P, Y):
    """Rel-L2 per probe column, averaged over the four columns.  Returns one value per sample."""
    sc = sps9.sc
    py, px = sc.PROBE_Y, sc.PROBE_X
    mp = np.asarray(P, np.float64)[..., :2].mean(axis=1)          # (N, H, W, 2)
    mt = np.asarray(Y, np.float64)[..., :2].mean(axis=1)
    # (N, 4, 9*2): column axis kept separate so each carries its own denominator.
    p = mp[:, py[:, None], px[None, :], :].transpose(0, 2, 1, 3).reshape(len(mp), len(px), -1)
    t = mt[:, py[:, None], px[None, :], :].transpose(0, 2, 1, 3).reshape(len(mt), len(px), -1)
    per_col = np.linalg.norm(p - t, axis=2) / np.linalg.norm(t, axis=2).clip(1e-8)
    return per_col.mean(axis=1)


def _targets():
    """Every live copy of the kit module.

    `sps9` and `eval_v9` EACH build their own instance with
    importlib.util.module_from_spec(...) and neither registers it in sys.modules, so there are
    two distinct module objects with two distinct namespaces.  compare_v9 scores through
    sps9.sc, but eval_v9/ens_v9 score through eval_v9.sc -- patch both or the correction
    silently applies to only some entry points.
    """
    import sys
    mods = [sps9.sc]
    ev = sys.modules.get("eval_v9")
    if ev is not None and getattr(ev, "sc", None) is not None and ev.sc not in mods:
        mods.append(ev.sc)
    return mods


def apply() -> int:
    """Install the correction on every loaded kit copy.  Idempotent; returns how many patched."""
    global _ORIGINAL
    n = 0
    for m in _targets():
        if getattr(m, "_mvpe_patched", False):
            continue
        if _ORIGINAL is None:
            _ORIGINAL = m.mvpe_rel_l2_per_sample
        m.mvpe_rel_l2_per_sample = _mvpe_percol
        m._mvpe_patched = True
        n += 1
    return n


def revert() -> None:
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    for m in _targets():
        if getattr(m, "_mvpe_patched", False):
            m.mvpe_rel_l2_per_sample = _ORIGINAL
            m._mvpe_patched = False
    _ORIGINAL = None


def _selftest() -> None:
    """The anchors, so the patch cannot silently rot."""
    import os
    d = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "val_unseen.npz"))
    X, Y = d["x"], d["y"]
    for name, P, want in (("persistence", np.repeat(X[:, -1:], 20, 1), 0.141),
                          ("time-mean", np.repeat(X.mean(1, keepdims=True), 20, 1), 0.124)):
        got = sps9.sc.mvpe_rel_l2(P, Y)
        ok = abs(got - want) <= 0.002
        print(f"  {name:12s} mvpe {got:.4f} (kit anchor {want:.3f})  {'PASS' if ok else 'FAIL'}")
        assert ok, f"{name}: {got} != {want}"


if __name__ == "__main__":
    import runpy
    import sys

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        apply()
        print(__doc__)
        raise SystemExit(0)
    if sys.argv[1] == "selftest":
        apply()
        _selftest()
        raise SystemExit(0)
    target = sys.argv[1]
    # Import eval_v9 FIRST so its private kit copy exists and gets patched too; the target
    # module's own `from eval_v9 import infer` then reuses the already-patched instance.
    try:
        import eval_v9  # noqa: F401
    except Exception as exc:                                   # pragma: no cover
        print(f"[mvpe_fix] note: eval_v9 not pre-imported ({exc})")
    n = apply()
    print(f"[mvpe_fix] scoring.mvpe_rel_l2_per_sample -> per-column form on {n} kit copies "
          "(pooled-36 reconstruction was ~10-14% pessimistic)")
    sys.argv = [target] + sys.argv[2:]
    runpy.run_module(target, run_name="__main__", alter_sys=True)
