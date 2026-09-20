"""How big is an FNO package, why, and what shrinks it -- measured, not argued.

    python compress_spectral.py --ckpt unet/fno_all_w96do30_s1.pth ... --alpha 0.60

THE FACT THAT FRAMES EVERYTHING.  An FNO's parameters are its spectral weights and essentially
nothing else: at width 96, modes (8,12), 4 layers that is 4 * width^2 * my * mx * n_layers =
14.156M of 14.246M, **99.4%**.  (The 4 is two blocks -- positive and negative ky -- times the
real and imaginary parts.)  So the archive size is one formula, the package is 8 x 28.5 MB in
fp16 = 228.6 MB against a 256 MB cap, and every "can we ship a deeper/wider model" question is
answered by that formula before any GPU is booked:

    w24  L4  1.8 MB/member     w96 L4  28.5     w96 L8  56.8     w128 L4  50.5     U-FNO 31.1

THREE WAYS TO SHRINK IT, in increasing order of interest:

1. **Fewer members.**  Free, and the accuracy cost is the ensemble curve, which is measured.
2. **int8 on the spectral weights**, one scale per output channel per layer.  A pure storage
   change: the tensors are de-quantized to fp32 at load, so inference is untouched.
3. **Low-rank factorization of the spectral tensor.**  Reshape one layer's weight from
   (width_in, width_out, ky, kx) to (width^2, n_modes) and truncate its SVD to rank r.  Storage
   goes from width^2 * n_modes to r * (width^2 + n_modes) complex -- at width 96 and 96 modes,
   rank 48 halves it and rank 24 quarters it.  This is not an arbitrary compression: it says the
   operator acts through r shared channel-mixing matrices modulated across wavenumber, which is
   exactly AFNO's structural claim -- and round 15 measured an AFNO with 0.24M parameters
   beating the 14.2M FNO, so the rank is expected to be low.

Everything here is POST-HOC on a trained checkpoint.  Nothing is retrained, so what is measured
is the accuracy the compression costs, not what a compressed model could learn.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def spectral_params(width, my, mx, n_layers):
    return 4 * width * width * my * mx * n_layers


def quant_int8(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """w is (i, o, y, x, 2). One scale per OUTPUT channel, shared by real and imag.

    Returns the de-quantized tensor and its bytes-per-element. Output-channel scaling is the
    right granularity here because the einsum contracts over `i`: an error in one output
    channel's scale is a gain error on that channel alone, not a mixing error.
    """
    s = w.abs().amax(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.round(w / s).clamp(-127, 127)
    return (q * s), 1.0 + 4.0 / (w.numel() / w.shape[1])


def low_rank(w: torch.Tensor, r: int) -> tuple[torch.Tensor, float]:
    """w is (i, o, y, x, 2) -> complex (i*o, y*x); truncate the SVD to rank r."""
    i, o, y, x, _ = w.shape
    c = torch.view_as_complex(w.contiguous()).reshape(i * o, y * x)
    u, s, vh = torch.linalg.svd(c, full_matrices=False)
    r = min(r, len(s))
    approx = (u[:, :r] * s[:r]) @ vh[:r]
    kept = float((s[:r] ** 2).sum() / (s ** 2).sum())
    bytes_per = 4.0 * r * (i * o + y * x) / (i * o * y * x)     # fp16 complex = 4 bytes
    return torch.view_as_real(approx.reshape(i, o, y, x)).to(w.dtype), bytes_per, kept


def transform(blob, mode, rank):
    sd = {k: v.clone() for k, v in blob["model_state_dict"].items()}
    spec_n = sum(v.numel() for k, v in sd.items() if k.endswith((".w1", ".w2")))
    other_n = sum(v.numel() for k, v in sd.items() if not k.endswith((".w1", ".w2")))
    keeps = []
    if mode == "fp16":
        spec_bytes = 2.0 * spec_n
    elif mode == "int8":
        spec_bytes = 0.0
        for k, v in sd.items():
            if k.endswith((".w1", ".w2")):
                sd[k], bpe = quant_int8(v)
                spec_bytes += bpe * v.numel()
    else:                                                     # low-rank
        spec_bytes = 0.0
        for k, v in sd.items():
            if k.endswith((".w1", ".w2")):
                sd[k], bpe, kept = low_rank(v, rank)
                spec_bytes += bpe * v.numel()
                keeps.append(kept)
    return sd, (spec_bytes + 2.0 * other_n) / 1e6, (float(np.mean(keeps)) if keeps else 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="the SHIPPED all-data members. Sizes only -- an accuracy read off "
                         "these is contaminated (rule 1), so prefer --err-ckpt.")
    ap.add_argument("--err-ckpt", nargs="+", default=None,
                    help="held-out twins; scored instead of --ckpt when given, which is the "
                         "honest way to read an accuracy cost (rule 1). Either flag alone is "
                         "enough.")
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--ranks", type=int, nargs="+", default=[96, 48, 24, 12])
    ap.add_argument("--aoa", type=int, nargs="+", default=None)
    args = ap.parse_args()
    if not (args.ckpt or args.err_ckpt):
        raise SystemExit("give --err-ckpt (preferred) or --ckpt")

    import _fno
    import sps9
    sc = sps9.sc
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    X, Y = du["x"], du["y"]
    if args.aoa:
        m = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
        X, Y = X[m], Y[m]
    paths = args.err_ckpt or args.ckpt
    blobs = [torch.load(p, map_location="cpu") for p in paths]
    print(f"{len(paths)} members, {len(X)} windows"
          f"{' (held-out twins)' if args.err_ckpt else ' (ALL-DATA -- contaminated, sizes only)'}")
    b0 = blobs[0]
    n = spectral_params(b0["width"], b0["modes_y"], b0["modes_x"], b0["n_layers"])
    print(f"spectral params 4*{b0['width']}^2*{b0['modes_y']}*{b0['modes_x']}*"
          f"{b0['n_layers']} = {n/1e6:.3f}M\n")
    print(f"{'variant':<14} {'MB/member':>10} {'8 memb':>8} {'energy':>7} "
          f"{'rel-L2':>8} {'TKE':>8} {'MVPE':>8}")
    base = None
    for mode, rank in [("fp16", 0), ("int8", 0)] + [("lowrank", r) for r in args.ranks]:
        mods, mb = [], 0.0
        for p, blob in zip(paths, blobs):
            sd, mb, kept = transform(blob, mode, rank)  # mb/kept identical across members
            nb = dict(blob); nb["model_state_dict"] = sd
            t = os.path.join("/tmp", f"_cs_{os.getpid()}.pth")
            torch.save(nb, t)
            mods.append(_fno.load_any(t, dev))
            os.remove(t)
        P = _fno.predict_fno(_fno.fuse(mods), X, dev, alpha=args.alpha)
        P = np.concatenate([P, np.zeros_like(P[..., :1])], -1).astype(np.float32)
        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        v = float(sc.mvpe_rel_l2(P, Y))
        if base is None:
            base = (r, k, v)
        tag = mode if rank == 0 else f"{mode} r={rank}"
        print(f"{tag:<14} {mb:10.2f} {8*mb:8.1f} {kept:7.4f} "
              f"{r:8.5f} {k:8.5f} {v:8.5f}   "
              f"({100*(r/base[0]-1):+5.2f}% {100*(k/base[1]-1):+5.2f}% "
              f"{100*(v/base[2]-1):+5.2f}% vs fp16)")
        del mods
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
