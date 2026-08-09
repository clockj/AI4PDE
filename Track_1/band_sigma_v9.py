"""A per-SAMPLE SPS band scaled by the model's own predicted sigma.

    python band_sigma_v9.py --ckpt unet/fno_vmlp_lift.pth unet/fno_vmlp_lift_s2.pth ...

The shipped band is per-ELEMENT and identical for every window: `optimal_band` solves
exp(-2h/sigma_g) * sum_i a_i*1[|E_ie| <= h] exactly, but only for a half-width that all samples
share. Round 6 tried making it per-sample using ENSEMBLE SPREAD and closed the idea: spread
correlated with |error| at only 0.347 once the per-element mean was removed, and the best
variant was worse in-sample while better out-of-sample, i.e. noise.

The variance head's predicted sigma is a different and better proxy. Measured on the round-10
4-member held-out ensemble, correlation with per-element |error| after removing the per-element
mean:

    predicted sigma  0.551
    ensemble spread  0.482   (round 6 measured 0.347 on 3 old-recipe members)

so the idea deserves one more attempt with the better signal. SPS carries 45% of the
accuracy->score exchange rate, so a point of SPS is worth ~0.18 `final`.

    half(i, e) = c * base(e) * (sigma(i, e) / mean_i sigma(i, e)) ** p

c and p are fitted on val_seen and reported on val_unseen. p = 0 recovers the shipped band
exactly, so the fit cannot lose in-sample; the whole question is whether it transfers.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import sps9
from sps9 import optimal_band, prep, sps_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

F_BOARD = 1.39


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=F_BOARD)
    ap.add_argument("--pgrid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--cgrid", type=float, nargs="+", default=[0.85, 1.0, 1.15, 1.3])
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer
    from sigma_blend_v9 import head_sigma

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hs = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hu = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    f = np.float32(args.f)

    def pack(d, sel):
        P = infer(models, d["x"], dev, args.alpha)[sel]
        Y = d["y"][sel]
        A, a, sc_, n = prep(P, Y)
        sig = head_sigma(models, d["x"], dev)[sel]           # (N, 1, H, W, 2)
        # Ratio to the per-element mean, so p = 0 is exactly the shipped band and p != 0 only
        # redistributes width between windows without changing the average width.
        r = sig / np.maximum(sig.mean(0, keepdims=True), 1e-8)
        return f * A, a, sc_, n, r.astype(np.float32)

    As, a_s, sc_s, n_s, r_s = pack(ds, hs)
    Au, a_u, sc_u, n_u, r_u = pack(du, hu)

    def base_for(p):
        """Per-element base refit FOR THIS p, so the comparison across p is fair.

        With half = base * r**p, the containment test |E| <= base*r**p is exactly
        |E| / r**p <= base, so solving the usual per-element problem on the deflated errors
        gives the right base. (The exp(-2h/sigma_g) term then varies per sample, which the
        exact solver does not model -- hence this is a very good approximation, not an exact
        optimum. Fitting `base` only at p = 0, as the first version did, made every p != 0
        worse in-sample BY CONSTRUCTION.)
        """
        if p == 0.0:
            return optimal_band(As, a_s, sc_s)
        return optimal_band((As / np.clip(r_s, 0.3, 3.0) ** p).astype(np.float32), a_s, sc_s)

    print(f"AoA {args.aoa}: fit on {len(As)} val_seen windows, report on {len(Au)} val_unseen; "
          f"errors {args.f}x")
    print(f"{'c':>5s} {'p':>5s} | {'sps seen':>9s} {'sps unseen':>11s}")
    print("-" * 35)

    best = None
    for p in args.pgrid:
        base = base_for(p)
        for c in args.cgrid:
            hs_ = (c * base * np.clip(r_s, 0.3, 3.0) ** p).astype(np.float32)
            hu_ = (c * base * np.clip(r_u, 0.3, 3.0) ** p).astype(np.float32)
            v_s = sps_score(As, a_s, sc_s, n_s, hs_)
            v_u = sps_score(Au, a_u, sc_u, n_u, hu_)
            flag = ""
            if best is None or v_s > best[0]:
                best, flag = (v_s, v_u, c, p), "  <- best in-sample"
            print(f"{c:5.2f} {p:5.2f} | {v_s:9.3f} {v_u:11.3f}{flag}")

    b0 = base_for(0.0)
    ref_s = sps_score(As, a_s, sc_s, n_s, b0)
    ref_u = sps_score(Au, a_u, sc_u, n_u, b0)
    print(f"\nshipped band (c=1, p=0): seen {ref_s:.3f}  unseen {ref_u:.3f}")
    print(f"best in-sample c={best[2]} p={best[3]}: seen {best[0]:.3f}  unseen {best[1]:.3f}")
    print(f"OUT-OF-SAMPLE DELTA {best[1] - ref_u:+.3f} sps  "
          f"= {0.1809 * (best[1] - ref_u):+.4f} final")


if __name__ == "__main__":
    main()
