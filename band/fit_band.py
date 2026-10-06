"""Fit the static per-element half-width band of the interval from the twins' errors.

    python fit_band.py --members <6 .pth> --twins <5 .pth> --val <dir with val_seen.npz, val_unseen.npz> --out band.npz

The interval the package emits is centred on y0, the mean field of the members before the
fluctuation adjustment, with half-width

    half(window, element) = base(element) * clip(sigma / sigma_ref, 0.3, 3)^p,

sigma the members' mean predicted temporal std for that window and element. This script solves
`base` exactly (band/sps.py) on the twins' errors |y0 - y| over the AoA 15 - 20 trajectories of
the SEEN split, inflated by 1.39 (the ratio between the errors the board showed and the local
ones), deflated by the sigma factor at p = 0.45 so that the two parts compose, box-smoothed
(3 x 3 x 3), and widened where fewer than 8 fitting windows scored an element (so that an
element never scored in these recordings is not left with a near-zero width). `sigma_ref` is
the mean sigma of the MEMBERS, since it normalises the sigma they emit at test time.

The npz carries `base` (1, 20, 32, 64, 3), `sigma_ref` (1, 1, 32, 64, 3), `p`, `clip_lo`,
`clip_hi` and `centre_alpha` = 0, the keys `submission.py` reads.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
sys.path.insert(0, HERE)
import _fno  # noqa: E402
from sps import optimal_band, prep, sps_score  # noqa: E402

ALPHA = 0.80          # the package's fluctuation adjustment; a_i is taken on that prediction
CENTRE_ALPHA = 0.0    # the interval is centred on the alpha = 0 field
BAND_P = 0.45
INFLATE = 1.39
FIT_AOA = (15, 20)
SMOOTH = 3
FILL_NOK, FILL_SMOOTH = 8, 5
CLIP_LO, CLIP_HI = 0.3, 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="the six all-data checkpoints")
    ap.add_argument("--twins", nargs="+", required=True, help="the five held-out twins")
    ap.add_argument("--val", required=True, help="directory holding val_seen.npz and val_unseen.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load_any(c, dev) for c in args.members]
    m_err = [_fno.load_any(c, dev) for c in args.twins]
    ds = np.load(os.path.join(args.val, "val_seen.npz"))
    du = np.load(os.path.join(args.val, "val_unseen.npz"))

    def infer(models, X, alpha=ALPHA):
        uv = _fno.predict_fno(models, X, dev, alpha=alpha)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    def sigma_of(models, X):
        s = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        _fno.predict_fno(models, X, dev, alpha=ALPHA, sigma_out=s)
        return np.concatenate([s, np.zeros_like(s[..., :1])], -1).astype(np.float32)

    keep = np.isin(np.array([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]]), FIT_AOA)
    Pf, Yf = infer(m_err, ds["x"])[keep], ds["y"][keep]
    sig_fit = sigma_of(m, ds["x"])[keep]
    sigma_ref = np.maximum(sig_fit.mean(0, keepdims=True), 1e-8).astype(np.float32)
    ratio_f = np.clip(sig_fit / sigma_ref, CLIP_LO, CLIP_HI)

    # a_i and the scored mask belong to the prediction; |E| is taken at the interval's centre.
    As, as_, scored_s, ns = prep(Pf, Yf)
    Cf = infer(m_err, ds["x"], alpha=CENTRE_ALPHA)[keep]
    As = np.abs(Cf[..., :2] - Yf[..., :2]).astype(np.float32)
    Adef = np.float32(INFLATE) * As
    Adef = (Adef / ratio_f[..., :2] ** BAND_P).astype(np.float32)
    uv_band = optimal_band(Adef, as_, scored_s)

    k = SMOOTH
    o = uv_band.astype(np.float64)
    for ax in (1, 2, 3):                      # T, H, W; channels stay independent
        pad = [(0, 0)] * 5
        pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + uv_band.shape[ax]), axis=ax)
                     for i in range(k)], axis=0)
    uv_band = o.astype(np.float32)

    # Elements the fitting windows rarely scored are blended toward the box mean of their
    # well-sampled neighbours (never narrower). Only HOW OFTEN an element scored is read.
    n_ok = scored_s.sum(0).astype(np.float64)                     # (T, H, W, 2)
    b = uv_band[0].astype(np.float64)
    ok = (n_ok >= FILL_NOK).astype(np.float64)
    kk = FILL_SMOOTH

    def box_sum(o):
        for ax in (0, 1, 2):
            pad = [(0, 0)] * 4
            pad[ax] = (kk // 2, kk // 2)
            q = np.pad(o, pad, mode="edge")
            o = np.sum([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(kk)],
                       axis=0)
        return o

    num, den = box_sum(b * ok), box_sum(ok)
    med = np.zeros_like(b)
    for t in range(b.shape[0]):
        for c in range(b.shape[-1]):
            v = b[t, :, :, c][ok[t, :, :, c] > 0]
            med[t, :, :, c] = np.median(v) if v.size else np.median(b[t, :, :, c])
    fill = np.where(den > 0, num / np.maximum(den, 1e-12), med)
    w = np.clip(n_ok / float(FILL_NOK), 0.0, 1.0)
    newb = w * b + (1.0 - w) * np.maximum(b, fill)
    touched = int((newb != b).sum())
    print(f"fill: n_ok < {FILL_NOK} at {int((n_ok < FILL_NOK).sum())} of {n_ok.size} elements; "
          f"never scored at {int((n_ok == 0).sum())}; changed {touched}")
    uv_band = newb.astype(np.float32)[None]

    band = np.concatenate([uv_band, np.zeros_like(uv_band[..., :1])], -1).astype(np.float32)
    np.savez(args.out, base=band, sigma_ref=sigma_ref, p=np.float32(BAND_P),
             clip_lo=np.float32(CLIP_LO), clip_hi=np.float32(CLIP_HI),
             centre_alpha=np.float32(CENTRE_ALPHA))

    h_fit = (uv_band * ratio_f[..., :2] ** BAND_P).astype(np.float32)
    print(f"band  half-width min {uv_band.min():.5f} median {np.median(uv_band):.5f} "
          f"max {uv_band.max():.5f}   p {BAND_P}   fit windows {int(keep.sum())}")
    print(f"sps on the fit split (in sample, centred on the prediction) "
          f"{sps_score(As, as_, scored_s, ns, h_fit):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
