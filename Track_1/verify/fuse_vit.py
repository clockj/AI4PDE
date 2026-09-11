"""Gate: `_fno.fuse` on ViT members is arithmetically the sequential loop.

    python Track_1/verify/fuse_vit.py            # CPU is fine, and CPU is the strict case

WHY THIS EXISTS.  Round 23 shipped round 20's four FNO members UNFUSED -- same checkpoints,
same fp16 packing, same band bytes -- and the board returned every error subscore identical to
six decimals with `time` alone moving: 3.795 -> 7.249 board ms on Track 1 and 4.659 -> 7.255 on
Track 2.  That is **1.151 / 0.866 ms per invocation** against a model that had fitted 0.435
since round 10, and it means the board is LAUNCH-bound rather than FLOP-bound -- four sequential
forwards do the same arithmetic as one grouped forward, so the whole +3.45 ms is dispatch.

Priced through `w_time`, one invocation is -0.085 `final` on Track 1.  A second ViT member is
worth roughly +0.15 of containment `sps` = +0.04 `final`, so UNFUSED it loses and FUSED it wins.
`StackedViT` is what makes the difference, and a fold is only worth having if it is exact.

WHAT IS CHECKED, on real windows: the field, the bounds-driving sigma, the invocation count,
and the mixed case -- FNOs and ViTs in one list must come out as exactly two groups, each
folded by its own stacker.  Synthetic members are used when fewer than two ViT checkpoints
exist, because a fusion bug found after a training run costs the run.

TF32 is disabled: attention and the stacked einsums are matmuls, and TF32 turns a 1e-7
agreement into 1e-3 without either path being wrong.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(T1, "submit_fno"))
sys.path.insert(0, os.path.join(T1, "unet"))

import _fno  # noqa: E402

TOL = 2e-5


def _report(label, d, ok_list, tol=TOL):
    good = d < tol
    print(f"  {label:<52s} {d:.3e}   {'OK' if good else 'MISMATCH'}")
    ok_list.append(good)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ok = []

    # ---- 1. synthetic members, always run ------------------------------------------------
    # Random weights are the STRICTER test: a trained ensemble's members agree with each other,
    # so an indexing bug that swapped two members would barely move the mean. Random ones do
    # not agree at all, and every per-member quantity -- pos, the LayerNorm affines, the qkv
    # stack, the grouped heads -- is then checked by the comparison rather than hidden by it.
    torch.manual_seed(0)
    syn = []
    for _ in range(3):
        m = _fno.ViT2d(dim=96, depth=3, heads=4, var_head=True).to(dev).eval()
        for p in m.parameters():
            p.data.normal_(0.0, 0.05)
        syn.append(m)
    x = torch.randn(5, _fno.CH + 0, _fno.H, _fno.W, device=dev) * 0.1

    fused = _fno.fuse(syn)
    print(f"synthetic: {len(syn)} ViT members -> {len(fused)} invocation(s) "
          f"{'OK' if len(fused) == 1 else 'NOT FUSED'}")
    ok.append(len(fused) == 1)
    with torch.no_grad():
        fs, ls = zip(*[m(x) for m in syn])
        f_seq = torch.stack(fs).mean(0)
        l_seq = torch.stack([t.clamp(-8.0, 2.0) for t in ls]).exp().mean(0).log()
        f_fus, l_fus = fused[0](x)
    _report("field   max|sequential - fused|", float((f_seq - f_fus).abs().max()), ok)
    _report("logstd  max|sequential - fused|", float((l_seq - l_fus).abs().max()), ok)

    # A single ViT must pass through untouched, not be wrapped in a one-member stack.
    one = _fno.fuse([syn[0]])
    print(f"  one ViT alone stays a ViT2d"
          f"{'':>27s}{'OK' if isinstance(one[0], _fno.ViT2d) else 'WRAPPED'}")
    ok.append(isinstance(one[0], _fno.ViT2d))

    # Members of DIFFERENT shape must land in different groups rather than assert inside
    # StackedViT -- fuse() keys on the state_dict shape signature, and this checks it covers
    # depth and width for ViTs the way it covers width for FNOs.
    odd = _fno.ViT2d(dim=96, depth=2, heads=4, var_head=True).to(dev).eval()
    n = len(_fno.fuse([syn[0], odd]))
    print(f"  ViTs of different depth -> {n} group(s)"
          f"{'':>21s}{'OK' if n == 2 else 'FUSED ANYWAY'}")
    ok.append(n == 2)

    # ---- 2. real checkpoints, if they exist ----------------------------------------------
    names = ["arch_vit_all_cmb3sim_s1.pth", "arch_vit_all_cmb3sim_s3.pth",
             "arch_vit_all_cmb3sim_s4.pth", "arch_vit_all_cmb3sim_d2.pth",
             "arch_vit_cmb3sim.pth", "arch_vit_cmb3sim_s2.pth", "arch_vit_cmb3sim_s3.pth"]
    have = [os.path.join(T1, "unet", n) for n in names]
    have = [p for p in have if os.path.exists(p)][:3]
    if len(have) < 2:
        print("\nreal checkpoints: fewer than two ViTs on disk -- synthetic check only")
    else:
        print(f"\nreal checkpoints ({len(have)}): "
              + ", ".join(os.path.basename(p) for p in have))
        ms = [_fno.load_any(p, dev) for p in have]
        xr = np.load(os.path.join(T1, "val_unseen.npz"))["x"][:16].astype(np.float32)
        fu = _fno.fuse(ms)
        print(f"  {len(ms)} members -> {len(fu)} invocation(s) "
              f"{'OK' if len(fu) == 1 else 'NOT FUSED'}")
        ok.append(len(fu) == 1)
        for alpha in (0.0, 0.60, 0.90):
            a = _fno.predict_fno(ms, xr, dev, alpha=alpha)
            b = _fno.predict_fno(fu, xr, dev, alpha=alpha)
            _report(f"alpha {alpha:.2f}  max|sequential - fused|",
                    float(np.max(np.abs(a - b))), ok)
        sa = np.zeros((len(xr), 1, _fno.H, _fno.W, _fno.C), np.float32)
        sb = np.zeros_like(sa)
        _fno.predict_fno(ms, xr, dev, alpha=0.90, sigma_out=sa)
        _fno.predict_fno(fu, xr, dev, alpha=0.90, sigma_out=sb)
        _report("sigma (drives the SPS band)  max diff", float(np.max(np.abs(sa - sb))), ok)

        # ---- 3. the MIXED package, which is what actually ships ---------------------------
        fnos = [os.path.join(T1, "unet", f"fno_all_w96L8cmb3sim_s{i}.pth") for i in (1, 2)]
        fnos = [p for p in fnos if os.path.exists(p)]
        if len(fnos) >= 2:
            mix = [_fno.load_any(p, dev) for p in fnos] + ms
            fm = _fno.fuse(mix)
            kinds = sorted(type(o).__name__ for o in fm)
            print(f"  mixed {len(fnos)} FNO + {len(ms)} ViT -> {len(fm)} invocation(s) "
                  f"{kinds} {'OK' if len(fm) == 2 else 'WRONG'}")
            ok.append(len(fm) == 2 and kinds == ["StackedFNO", "StackedViT"])
            a = _fno.predict_fno(mix, xr, dev, alpha=0.60)
            b = _fno.predict_fno(fm, xr, dev, alpha=0.60)
            _report("mixed package  max|sequential - fused|",
                    float(np.max(np.abs(a - b))), ok)
        else:
            print("  mixed check skipped: need two cmb3sim FNO members on disk")
            ok.append(False)     # silence is not success

    print("\nPASS" if all(ok) else "\nFAIL")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
