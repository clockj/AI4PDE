"""Does a SECOND per-window scaling axis add anything to the SPS band?

    python band2_v9.py --rebuild --ckpt <held-out members>

The shipped band is  half(i,e) = base(e) * clip(sigma_i(e)/sigma_ref(e), 0.3, 3)**p.  Round 6
tried ensemble SPREAD alone and closed it (correlation with per-element |error| 0.347 against
sigma's 0.551); round 10 adopted sigma. Nobody has tried BOTH -- and they are different
quantities: sigma is the model's belief about the flow's own fluctuation amplitude, spread is
its disagreement with itself. A feature can be weaker alone and still add.

    half(i,e) = base(e) * r_sigma**p * r_spread**q

`base` is refit for every (p, q) on errors deflated by the same factor, which is the method
rule that made round 6's conclusion wrong the first time. Scored by leave-one-trajectory-out CV
on val_unseen AoA 15-20, with the 1.39x board inflation applied to both sides.
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

from sps9 import optimal_band, prep, sps_score  # noqa: E402

CACHE = os.path.join(HERE, "band2_cache.npz")
AOA_HI = (15, 20)


def smooth(b, k):
    if k <= 1:
        return b
    o = b.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * 5
        pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(k)], 0)
    return o.astype(np.float32)


def build(ckpts, alpha):
    import torch
    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ms = [_fno.load(c, dev) for c in ckpts]
    out = {}
    for split in ("val_seen", "val_unseen"):
        d = np.load(os.path.join(HERE, f"{split}.npz"))
        keep = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in d["ids"]], AOA_HI)
        X, Y, ids = d["x"][keep], d["y"][keep], d["ids"][keep]
        sig = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(ms, X, dev, alpha=alpha, sigma_out=sig)
        P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
        # Per-member RAW fields, to get the per-element disagreement. alpha=0 so the
        # fluctuation rescaling (which uses the ensemble sigma) does not contaminate it.
        per = np.stack([_fno.predict_fno([m], X, dev, alpha=0.0) for m in ms])   # (M,N,T,H,W,2)
        spread = per.std(0).astype(np.float32)                                  # (N,T,H,W,2)
        out[f"{split}_P"], out[f"{split}_Y"] = P, Y
        out[f"{split}_sig"], out[f"{split}_spread"] = sig, spread
        out[f"{split}_traj"] = np.array([str(i).split(":")[0] for i in ids])
        print(f"{split}: {len(X)} windows, {len(set(out[f'{split}_traj']))} trajectories")
    np.savez(CACHE, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--inflate", type=float, default=1.39)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    z = (build(args.ckpt, args.alpha) if args.rebuild or not os.path.exists(CACHE)
         else dict(np.load(CACHE, allow_pickle=True)))

    D = {}
    for tag in ("seen", "unseen"):
        P, Y = z[f"val_{tag}_P"], z[f"val_{tag}_Y"]
        A, a, sc_, _ = prep(P, Y)
        sg, sp = z[f"val_{tag}_sig"], z[f"val_{tag}_spread"]
        D[tag] = dict(A=(np.float32(args.inflate) * A).astype(np.float32), a=a, scored=sc_,
                      sig=sg, spread=sp, traj=z[f"val_{tag}_traj"])
    ref_s = np.maximum(D["seen"]["sig"].mean(0, keepdims=True), 1e-8)
    ref_p = np.maximum(D["seen"]["spread"].mean(0, keepdims=True), 1e-8)
    for d in D.values():
        d["rs"] = np.clip(d["sig"] / ref_s, 0.3, 3.0).astype(np.float32)
        d["rp"] = np.clip(d["spread"] / ref_p, 0.3, 3.0).astype(np.float32)

    S, U = D["seen"], D["unseen"]
    folds = sorted(set(U["traj"]))
    print(f"\ninflate {args.inflate}x both sides.  folds {folds}")
    print(f"{'p(sigma)':>9s} {'q(spread)':>10s} | " + "  ".join(f"{f:>7s}" for f in folds)
          + "     CV mean")
    rows = []
    for p in (0.0, 0.3, 0.5, 0.7):
        for q in (0.0, 0.15, 0.3, 0.5):
            f_s = (S["rs"] ** np.float32(p) * S["rp"] ** np.float32(q)).astype(np.float32)
            base = smooth(optimal_band((S["A"] / f_s).astype(np.float32), S["a"],
                                       S["scored"]), 3)
            per = []
            for f in folds:
                i = np.where(U["traj"] == f)[0]
                half = (base * (U["rs"][i] ** np.float32(p)
                                * U["rp"][i] ** np.float32(q))).astype(np.float32)
                per.append(sps_score(U["A"][i], U["a"][i], U["scored"][i],
                                     int(np.count_nonzero(U["scored"][i])), half))
            mu = float(np.mean(per))
            rows.append((mu, p, q))
            tag = "  <- shipped" if (p, q) == (0.5, 0.0) else ""
            print(f"{p:9.2f} {q:10.2f} | " + "  ".join(f"{v:7.3f}" for v in per)
                  + f"   {mu:8.3f}{tag}")
    rows.sort(reverse=True)
    ship = [r for r in rows if (r[1], r[2]) == (0.5, 0.0)][0]
    print(f"\nbest p={rows[0][1]:.2f} q={rows[0][2]:.2f}  CV {rows[0][0]:.3f}")
    print(f"shipped p=0.50 q=0.00      CV {ship[0]:.3f}   delta {rows[0][0]-ship[0]:+.3f} sps"
          f"  ({(rows[0][0]-ship[0])*0.23:+.3f} final)")


if __name__ == "__main__":
    main()
