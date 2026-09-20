"""Post-hoc transforms of the SHIPPED prediction array: the levers that need no retraining.

    python posthoc_v9.py --ckpt unet/fno_r16_w96_do45_L8{,_s2,_s3}.pth

SPS carries 0.2502 of the free 20-round weight and scores 41.3 against 100*a = 70.6, and round
9's decomposition says half the return on any accuracy improvement arrives through it.  SPS is
a per-element CONTAINMENT score, so what it rewards is a shorter error TAIL, not a smaller rms
-- which is exactly what round 15's board result demonstrated (rel_l2 0.64 % WORSE, sps +1.013,
net +0.166).  Everything in this file is a cheap tail-killer applied to `pred` after the model
has run, evaluated exactly, with every fitted parameter chosen on val_seen and reported on
val_unseen.

  zero      Paint 0 where the INPUT window is identically 0 through all 20 frames.  9.95 % of
            target u,v elements are exactly 0 (outside the PIV field of view, or inside the
            airfoil) and the mask is recoverable from window k's OWN input at 98.9 % agreement,
            so this uses no forbidden information.  Those elements leave SPS entirely but they
            are in the rel_l2 and mvpe norms, where whatever the model puts there is pure error.
  zthr      The same, thresholded on max_t |input| rather than on an exact zero, because PIV
            dropout is per-frame and an exactly-zero test is 94 % precise.
  clamp     Clamp the prediction into the range the pixel actually occupied over the input
            window, widened by k times that range.  A velocity cannot leave its own recent
            envelope by much in 20 frames; this deletes outliers and touches nothing else.
  median    Combine members by the per-element MEDIAN instead of the mean.  The mean is optimal
            for Gaussian member disagreement and the median for heavy-tailed disagreement, and
            heavy-tailed is precisely what SPS charges for.
  alpha_t   A per-lead-time ALPHA ramp.  `compose` blends the raw field's own std toward the
            variance head's sigma with ONE scalar for the whole window, but the fluctuation is
            far more predictable at t = 1 than at t = 20, so the rel_l2-optimal shrinkage is
            not constant in t.  ALPHA has been swept as a scalar five times; the ramp is a
            different object.
  persist   Blend the early output frames toward the last INPUT frame, weight decaying in t.

Every transform is scored with the band REFITTED on it (fit_band's own recipe: val_seen, AoA
15-20, inflate 1.39, band-p 0.45, smooth 3), because a transform that changes the error field
and keeps the old band is being measured with the wrong instrument.
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import _fno       # noqa: E402
import final_v9   # noqa: E402
import sps9       # noqa: E402
from sps9 import optimal_band, prep, sc, sps_score  # noqa: E402


def box_blur(a, k):
    if k <= 1:
        return a
    o = a.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * o.ndim
        pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + a.shape[ax]), axis=ax) for i in range(k)], axis=0)
    return o.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--inflate", type=float, default=1.39)
    ap.add_argument("--fit-aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--band-p", type=float, default=0.45)
    ap.add_argument("--band-smooth", type=int, default=3)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load_any(c, dev) for c in args.ckpt]

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))

    def raw_and_sigma(X):
        """alpha = 0 leaves the field uncomposed, so the composition can be redone per-t here."""
        s = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        per = [_fno.predict_fno([m], X, dev, alpha=0.0) for m in models]
        _fno.predict_fno(models, X, dev, alpha=0.0, sigma_out=s)
        return np.stack(per, 0).astype(np.float32), s          # (K,N,20,32,64,2), (N,1,32,64,2)

    def compose_np(field, sigma, alpha):
        """model.compose in numpy; alpha may be a scalar or shaped (1, T, 1, 1, 1)."""
        m = field.mean(1, keepdims=True)
        f = field - m
        s = f.std(1, keepdims=True)                            # population std, as std_t does
        tgt = alpha * sigma + (1.0 - alpha) * s
        return (m + f * (tgt / (s + 1e-6))).astype(np.float32)

    def pad3(uv):
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    print("running the members ...", flush=True)
    Fs, Sg_s = raw_and_sigma(ds["x"])
    Fu, Sg_u = raw_and_sigma(du["x"])
    aoa_s = np.array([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]])
    aoa_u = np.array([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]])
    kf, ku = np.isin(aoa_s, args.fit_aoa), np.isin(aoa_u, args.fit_aoa)
    Ys, Yu = ds["y"], du["y"]
    Xs, Xu = ds["x"][..., :2], du["x"][..., :2]

    def score(Ps, Pu, tag):
        """Refit the band on val_seen AoA 15-20 and report both views of val_unseen."""
        sref = np.maximum(Sg_s[kf].mean(0, keepdims=True), 1e-8)
        rf = np.clip(Sg_s[kf] / sref, 0.3, 3.0)
        ru = np.clip(Sg_u / sref, 0.3, 3.0)
        Af, af, sf, nf = prep(Ps[kf], Ys[kf])
        dmf, tkf, mvf = sps9.sample_metrics(Ps[kf], Ys[kf])
        f = np.float32(args.inflate)
        afb = sps9.weights_from_metrics(f * dmf, f * tkf, f * mvf)
        base = box_blur(optimal_band(((f * Af) / rf ** args.band_p).astype(np.float32), afb, sf),
                        args.band_smooth)
        out = []
        for name, sel in (("AoA15-20", ku), ("full", np.ones(len(Pu), bool))):
            P, Y = Pu[sel], Yu[sel]
            rel = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
            tke = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
            mvp = float(np.mean(sc.mvpe_rel_l2_per_sample(P, Y)))
            A, a, sm, n = prep(P, Y)
            dm, tk, mv = sps9.sample_metrics(P, Y)
            ab = sps9.weights_from_metrics(f * dm, f * tk, f * mv)
            h = (base * ru[sel] ** args.band_p).astype(np.float32)
            sps = sps_score(f * A, ab, sm, n, h)
            wf = final_v9.predict([100 / (1 + 0.5 * f * rel), 100 / (1 + 0.5 * f * tke),
                                   100 / (1 + 0.5 * f * mvp), 93.0, sps])[0]
            out.append((rel, tke, mvp, sps, wf))
        print(f"{tag:<16s} " + "  ".join(
            f"[{n}] rel {o[0]:.5f} tke {o[1]:.5f} mvpe {o[2]:.5f} sps {o[3]:7.3f} wf {o[4]:.3f}"
            for n, o in zip(("AoA", " full"), out)), flush=True)
        return out

    A0 = np.float32(args.alpha)
    Ps0 = pad3(compose_np(Fs.mean(0), Sg_s, A0))
    Pu0 = pad3(compose_np(Fu.mean(0), Sg_u, A0))
    print("\n=== baseline (mean over members, scalar ALPHA) ===")
    base_scores = score(Ps0, Pu0, "baseline")

    print("\n=== zero-painting: the input window's own all-zero mask ===")
    for thr in (0.0, 1e-6, 1e-4, 1e-3):
        ms = (np.abs(Xs).max(1) <= thr)[:, None]               # (N,1,32,64,2)
        mu = (np.abs(Xu).max(1) <= thr)[:, None]
        Ps = Ps0.copy(); Pu = Pu0.copy()
        Ps[..., :2] = np.where(ms, 0.0, Ps[..., :2])
        Pu[..., :2] = np.where(mu, 0.0, Pu[..., :2])
        score(Ps, Pu, f"zero thr={thr:g}")

    print("\n=== clamp to the input window's own envelope, widened by k ===")
    for k in (0.0, 0.25, 0.5, 1.0, 2.0):
        def cl(X, P0, k=k):
            lo = X.min(1, keepdims=True); hi = X.max(1, keepdims=True); r = hi - lo
            P = P0.copy()
            P[..., :2] = np.clip(P0[..., :2], lo - k * r, hi + k * r)
            return P
        score(cl(Xs, Ps0), cl(Xu, Pu0), f"clamp k={k:g}")

    if len(models) >= 3:
        print("\n=== member combination: per-element MEDIAN instead of MEAN ===")
        score(pad3(compose_np(np.median(Fs, 0), Sg_s, A0)),
              pad3(compose_np(np.median(Fu, 0), Sg_u, A0)), "median")

    print("\n=== per-lead-time ALPHA ramp a(t) = a0 + (a1-a0)*t/19 ===")
    tt = (np.arange(20, dtype=np.float32) / 19.0).reshape(1, 20, 1, 1, 1)
    for a0, a1 in ((0.60, 0.60), (0.40, 0.80), (0.30, 0.90), (0.80, 0.40), (0.50, 0.70),
                   (0.70, 0.50)):
        al = (a0 + (a1 - a0) * tt).astype(np.float32)
        score(pad3(compose_np(Fs.mean(0), Sg_s, al)),
              pad3(compose_np(Fu.mean(0), Sg_u, al)), f"alpha {a0:.2f}->{a1:.2f}")

    print("\n=== blend the early frames toward the LAST INPUT frame, b(t) = b0*exp(-t/tau) ===")
    for b0, tau in ((0.0, 1.0), (0.1, 2.0), (0.2, 2.0), (0.2, 5.0), (0.4, 2.0)):
        w = (b0 * np.exp(-np.arange(20, dtype=np.float32) / tau)).reshape(1, 20, 1, 1, 1)
        def bl(X, P0):
            P = P0.copy()
            P[..., :2] = (1.0 - w) * P0[..., :2] + w * X[:, -1:][..., :2]
            return P
        score(bl(Xs, Ps0), bl(Xu, Pu0), f"persist {b0:.2f}/{tau:g}")


if __name__ == "__main__":
    main()
