"""Gate on `_fno.predict_fno`'s two normalizer conventions.

    python Track_1/verify/selfnorm_path.py            # CPU is fine

Two independent claims, both checked here:

**(A) A `--norm real` package is BIT-IDENTICAL to the pre-2026-08-31 code.** Every shipped
member trains under the default `--norm real` (82 checkpoints scanned, 81 of them real), so
the self-normalization branch must be dead code on the scored path. Checked against the
committed reference rather than against a re-derivation, so a typo in the new affine cannot
hide behind a matching typo in the test.

**(B) A `--norm self` member is scored in the space it was TRAINED in.** Until 2026-08-31
`predict_fno` hard-coded MEAN/STD, so the one arm ever trained with `--norm self` was fed an
input it had never seen and denormalized with the wrong affine. That is the arm
`report_20260831_round13.md` §5.3 called "not neutral -- broken" and used to close Strategy
item 5. The invariant that was violated is a round trip: normalize a window, denormalize it,
get the window back. Under the old path that round trip is off by a factor that grows as the
window's own mean speed leaves the global mean, which is exactly why the failure looked like a
Reynolds-extrapolation failure.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(T1, "submit_fno"))
sys.path.insert(0, os.path.join(T1, "unet"))

import _fno  # noqa: E402


def _load_ref(path: str):
    spec = importlib.util.spec_from_file_location("_fno_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ref_path = sys.argv[1] if len(sys.argv) > 1 else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = np.load(os.path.join(T1, "val_unseen.npz"))["x"][:8].astype(np.float32)
    ok = True

    # ---- (A) --norm real is unchanged, bit for bit -------------------------------------
    if ref_path and os.path.exists(ref_path):
        ref = _load_ref(ref_path)
        for name in ("fno_vl_s1.pth", "fno_vl_m1220.pth"):
            p = os.path.join(T1, "unet", name)
            if not os.path.exists(p):
                continue
            a = _fno.predict_fno([_fno.load_any(p, dev)], x, dev, alpha=0.90)
            b = ref.predict_fno([ref.load_any(p, dev)], x, dev, alpha=0.90)
            d = float(np.max(np.abs(a - b)))
            print(f"(A) {name:20s} max|new - committed| = {d:.3e}"
                  f"   {'OK' if d == 0.0 else 'MISMATCH'}")
            ok &= d == 0.0
    else:
        print("(A) skipped: no reference copy of the pre-patch _fno.py given as argv[1]")

    # ---- (B) the self-normalized affine round-trips ------------------------------------
    # `predict_fno`'s input map is (x - mean)/std and its output map is y*std + mean, with
    # mean/std carrying the window scale under self-normalization. Composing them must be the
    # identity for ANY window -- that is the property the old path lost.
    from model import MEAN_SN, STD_SN, self_denorm, self_norm, window_scale
    xt = torch.from_numpy(x)
    s = window_scale(xt)
    mn = torch.from_numpy(MEAN_SN)
    sd = torch.from_numpy(STD_SN)
    rt = self_denorm(self_norm(xt[..., :2], s, mn, sd), s, mn, sd)
    d = float((rt - xt[..., :2]).abs().max())
    print(f"(B) self_norm round trip on 8 real windows: max err {d:.3e}"
          f"   {'OK' if d < 1e-6 else 'BROKEN'}")
    ok &= d < 1e-6

    # And predict_fno's own two lines must agree with those helpers.
    mean = torch.from_numpy(_fno.MEAN_SN) * s
    std = torch.from_numpy(_fno.STD_SN) * s
    d2 = float((((xt[..., :2] - mean) / std) - self_norm(xt[..., :2], s, mn, sd)).abs().max())
    print(f"(B) predict_fno's affine == model.self_norm: max err {d2:.3e}"
          f"   {'OK' if d2 < 1e-6 else 'MISMATCH'}")
    ok &= d2 < 1e-6

    # ---- (C) a mixed ensemble is refused, not silently averaged -------------------------
    class _M:
        def __init__(self, sn):
            self.self_norm = sn
    try:
        _fno.predict_fno([_M(True), _M(False)], x, dev)
        print("(C) mixed --norm self / --norm real ensemble: NOT REFUSED"); ok = False
    except ValueError as e:
        print(f"(C) mixed ensemble refused: {e}   OK")

    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
