"""Gate: `_fno.fuse` on oscillatory-head members is arithmetically the sequential loop.

    python Track_1/verify/fuse_osc.py            # CPU is fine, and CPU is the strict case

WHY THIS EXISTS. Board `time_score` tracks the number of sequential model INVOCATIONS
(`ROADMAP.md` §1.5: `board ms = 4.447 + 0.324*inv + 0.431*local ms`, and round 12 paid 0.973
`time_score` for one extra group). Until this gate, `fuse()` refused to fold any member with an
`head_osc`, so an eight-member `ozdo` package would have run **eight** invocations -- about
+2.3 board ms, ~-0.11 `final`, which is more than the arm is worth. Folding them is what makes
the oscillatory head shippable at all, and the fold is only worth having if it is exact.

WHAT IS CHECKED. The fused module against the plain per-member loop `predict_fno` would run,
on real windows, for the field, the bounds-driving sigma, and the invocation count. Also that
members disagreeing on `osc_zeromean` -- a bool with no parameters, so the state_dict shape
signature cannot see it -- land in different groups instead of being averaged into one module
that can only be right for some of them.

TF32 is disabled: the spectral einsum is a matmul, and TF32 turns a 1e-8 agreement into 1e-3
without either path being wrong. Run on CPU or with TF32 off, or the gate measures the GPU.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(T1, "submit_fno"))
sys.path.insert(0, os.path.join(T1, "unet"))     # models_alt imports its siblings

import _fno  # noqa: E402


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = [f"fno_ozdo{s}.pth" for s in ("", "_b", "_c", "_d")]
    paths = [os.path.join(T1, "unet", n) for n in names]
    have = [p for p in paths if os.path.exists(p)]
    if len(have) < 2:
        print("need at least two oscillatory-head checkpoints; none found"); sys.exit(1)
    ms = [_fno.load_any(p, dev) for p in have]

    x = np.load(os.path.join(T1, "val_unseen.npz"))["x"][:16].astype(np.float32)
    ok = True

    fused = _fno.fuse(ms)
    print(f"{len(ms)} oscillatory-head members -> {len(fused)} invocation(s) "
          f"{'OK' if len(fused) == 1 else 'NOT FUSED'}")
    ok &= len(fused) == 1

    for alpha in (0.0, 0.60, 0.90):
        a = _fno.predict_fno(ms, x, dev, alpha=alpha)          # K sequential members
        b = _fno.predict_fno(fused, x, dev, alpha=alpha)       # one fused invocation
        d = float(np.max(np.abs(a - b)))
        print(f"  alpha {alpha:.2f}: max|sequential - fused| = {d:.3e}"
              f"   {'OK' if d < 2e-5 else 'MISMATCH'}")
        ok &= d < 2e-5

    sa = np.zeros((len(x), 1, _fno.H, _fno.W, _fno.C), np.float32)
    sb = np.zeros_like(sa)
    _fno.predict_fno(ms, x, dev, alpha=0.90, sigma_out=sa)
    _fno.predict_fno(fused, x, dev, alpha=0.90, sigma_out=sb)
    d = float(np.max(np.abs(sa - sb)))
    print(f"  sigma (drives the SPS band): max diff = {d:.3e}"
          f"   {'OK' if d < 2e-5 else 'MISMATCH'}")
    ok &= d < 2e-5

    # A bool with no parameters must still separate groups.
    m2 = _fno.load_any(have[1], dev)
    m2.osc_zeromean = not m2.osc_zeromean
    n = len(_fno.fuse([ms[0], m2]))
    print(f"  members disagreeing on osc_zeromean -> {n} group(s) "
          f"{'OK' if n == 2 else 'FUSED ANYWAY'}")
    ok &= n == 2

    # --- the free temporal basis, synthetically -------------------------------------------
    # `--tbasis` has no trained checkpoints yet, and a fusion bug found after a training run
    # costs the run. Build three random members of the vendored class, check that the training
    # class agrees with it bit-for-bit, and that the fold is arithmetic.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "models_alt", os.path.join(T1, "unet", "models_alt.py"))
    ma = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ma)
    except Exception as e:
        # NOT a skip. A gate that passes when a check did not run is the "silence is not
        # success" failure -- the first version of this file printed "skipped (No module named
        # 'model')" and still reported PASS, because models_alt imports its siblings and
        # unet/ was not on the path.
        print(f"  --tbasis check COULD NOT RUN: {e}"); ma = None; ok = False
    if ma is not None:
        torch.manual_seed(0)
        mem = []
        for i in range(3):
            a = ma.FNO2d(24, 8, 12, 4, True, var_mlp=True, lift_mlp=True, tbasis=3).eval()
            a.tb.data.normal_()
            a.head_tb[2].weight.data.normal_(std=0.1)
            a.head_tb[2].bias.data.normal_(std=0.1)
            b = _fno.FNO2d(24, 8, 12, 4, True, var_mlp=True, lift_mlp=True, tbasis=3).eval()
            b.load_state_dict(a.state_dict())
            mem.append(b)
            if i == 0:
                xt = torch.randn(2, _fno.CH, _fno.H, _fno.W)
                d = float((a(xt)[0] - b(xt)[0]).abs().max())
                print(f"  --tbasis train class vs vendored class: {d:.3e}"
                      f"   {'OK' if d == 0.0 else 'MISMATCH'}")
                ok &= d == 0.0
        xt = torch.randn(4, _fno.CH, _fno.H, _fno.W)
        seq = sum(m(xt)[0] for m in mem) / 3
        fz = _fno.fuse(mem)
        d = float((fz[0](xt)[0] - seq).abs().max())
        print(f"  --tbasis 3 members -> {len(fz)} invocation(s), fused vs sequential {d:.3e}"
              f"   {'OK' if len(fz) == 1 and d < 2e-5 else 'FAIL'}")
        ok &= len(fz) == 1 and d < 2e-5

    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
