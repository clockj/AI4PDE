"""WP-C: choose the band's FITTING SET and its (p, smooth) by honest leave-one-trajectory-out CV.

    python band_cv_v9.py --ckpt unet/fno_vmlp_lift.pth ... [--rebuild]

`fit_band.py` fits `base` on val_seen AoA 15-20 -- 126 windows for 81,920 free parameters --
and val_seen is IN-SAMPLE for the held-out models that supply the error field. Two questions
this file answers, and neither can be answered by fit_band.py's own report line, because the
band is the in-sample optimum by construction:

  1. Should the error field come from val_seen (in-sample for these models, 3 trajectories),
     val_unseen (out-of-sample, 3 trajectories), or both pooled (6)?
  2. What are the best (band-p, band-smooth) once (1) is settled? Round 10 tuned them one at a
     time on a coarse grid, and refitted `base` for each p -- which is the part that matters
     and is preserved here.

CV protocol: hold out one val_unseen AoA 15-20 trajectory, fit `base` on the chosen training
set with that trajectory removed, score SPS on it. Errors on BOTH sides are multiplied by
--inflate, so the fit sees the same error scale it is scored against -- the board's 1.39x tail
inflation is a level shift the CV cannot observe and must not be allowed to fake.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import sps9  # noqa: E402
from sps9 import optimal_band, prep, sps_score  # noqa: E402

CACHE = os.path.join(HERE, "band_cv_cache.npz")
AOA_HI = (15, 20)


def traj_of(ids):
    return np.array([str(i).split(":")[0] for i in ids])


def aoa_of(ids):
    return np.array([int(str(i).split(":")[0].split("_")[1]) for i in ids])


def build_cache(ckpts, alpha):
    import torch
    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load(c, dev) for c in ckpts]
    out = {}
    for split in ("val_seen", "val_unseen"):
        d = np.load(os.path.join(HERE, f"{split}.npz"))
        keep = np.isin(aoa_of(d["ids"]), AOA_HI)
        X, Y, ids = d["x"][keep], d["y"][keep], d["ids"][keep]
        s = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(m, X, dev, alpha=alpha, sigma_out=s)
        P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
        out[f"{split}_P"], out[f"{split}_Y"] = P, Y
        out[f"{split}_sig"], out[f"{split}_ids"] = s, ids
        print(f"{split}: {len(X)} windows, trajs {sorted(set(traj_of(ids)))}")
    np.savez(CACHE, **out)
    return out


def smooth(b, k):
    if k <= 1:
        return b
    o = b.astype(np.float64)
    for ax in (1, 2, 3):                       # T, H, W; channels stay independent
        pad = [(0, 0)] * 5
        pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(k)], axis=0)
    return o.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=False, default=None)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--inflate", type=float, default=1.39)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(CACHE):
        assert args.ckpt, "--ckpt is required to build the cache"
        z = build_cache(args.ckpt, args.alpha)
    else:
        z = dict(np.load(CACHE, allow_pickle=True))
        print(f"cache {CACHE}")

    # Per-split arrays, all restricted to AoA 15-20 already.
    S, U = {}, {}
    for tag, d in (("seen", S), ("unseen", U)):
        P, Y, sig = z[f"val_{tag}_P"], z[f"val_{tag}_Y"], z[f"val_{tag}_sig"]
        A, a, sc_, n = prep(P, Y)
        d["A"] = (np.float32(args.inflate) * A).astype(np.float32)
        d["a"], d["scored"] = a, sc_
        # sigma_ref is a per-element constant of the SHIPPED models; here every configuration
        # shares one reference so the comparison is only about the fitting set and (p, k).
        d["sig"] = sig
        d["traj"] = traj_of(z[f"val_{tag}_ids"])
    ref = np.maximum(np.concatenate([S["sig"], U["sig"]]).mean(0, keepdims=True), 1e-8)
    for d in (S, U):
        d["r"] = np.clip(d["sig"] / ref, 0.3, 3.0).astype(np.float32)

    def fit(idx_parts, p, k):
        """idx_parts: list of (dict, index array). Returns the smoothed base."""
        A = np.concatenate([d["A"][i] for d, i in idx_parts])
        a = np.concatenate([d["a"][i] for d, i in idx_parts])
        s = np.concatenate([d["scored"][i] for d, i in idx_parts])
        r = np.concatenate([d["r"][i] for d, i in idx_parts])
        if p:
            A = (A / r ** np.float32(p)).astype(np.float32)
        return smooth(optimal_band(A, a, s), k)

    def score(d, i, base, p):
        half = (base * d["r"][i] ** np.float32(p)).astype(np.float32)
        n = int(np.count_nonzero(d["scored"][i]))
        return sps_score(d["A"][i], d["a"][i], d["scored"][i], n, half)

    folds = sorted(set(U["traj"]))
    all_s = np.arange(len(S["A"]))
    print(f"\ninflate {args.inflate:.2f}x applied to both sides.  "
          f"{len(folds)} folds: {folds}\n")

    sets = {
        "seen            ": lambda f: [(S, all_s)],
        "unseen-minus-f  ": lambda f: [(U, np.where(U["traj"] != f)[0])],
        "pooled-minus-f  ": lambda f: [(S, all_s), (U, np.where(U["traj"] != f)[0])],
    }
    grid_p = (0.0, 0.3, 0.45, 0.5, 0.6, 0.75, 0.9)
    grid_k = (1, 3, 5, 7)

    print(f"{'fit set':18s} {'p':>5s} {'k':>3s}   " +
          "  ".join(f"{f:>7s}" for f in folds) + "     CV mean")
    best = []
    for name, mk in sets.items():
        for p in grid_p:
            for k in grid_k:
                per = []
                for f in folds:
                    base = fit(mk(f), p, k)
                    per.append(score(U, np.where(U["traj"] == f)[0], base, p))
                mu = float(np.mean(per))
                best.append((mu, name, p, k))
                print(f"{name:18s} {p:5.2f} {k:3d}   " +
                      "  ".join(f"{v:7.3f}" for v in per) + f"   {mu:8.3f}")
        print()
    best.sort(reverse=True)
    print("top 8 by CV mean:")
    for mu, name, p, k in best[:8]:
        print(f"   {mu:8.3f}   {name.strip():16s} p={p:.2f} k={k}")
    shipped = [b for b in best if b[1].strip() == "seen" and b[2] == 0.5 and b[3] == 3]
    if shipped:
        print(f"\nshipped (seen, p=0.50, k=3): {shipped[0][0]:.3f}   "
              f"best is {best[0][0] - shipped[0][0]:+.3f} sps")


if __name__ == "__main__":
    main()
