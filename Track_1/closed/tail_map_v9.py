"""WHERE the SPS misses are, HOW FAR outside the band, and what accuracy would flip them.

    python tail_map_v9.py --ckpt <shipped members> --err-ckpt <held-out twins> \
        --band ../sub_candidates/band_r17_simx4.npz --alpha 0.60

R19 measured the thing this exists to explain. Sim cut board rel_l2 by 0.63 % and board `sps`
moved +0.008; decomposing the per-sample factor a_i accounts for ALL of that, so the
per-ELEMENT containment term contributed nothing. Round 15's capacity x dropout did the
opposite -- rel_l2 0.64 % WORSE, sps +1.013 better. `sps` is 0.250 of the board weight sitting
at 41.33 against a ceiling of 100*a = 70.61, i.e. 7.3 of the ~13 points left, and it is paid
per element for `|E| <= h`, not for rms.

Nothing in this project has ever looked at the misses themselves. Three questions, and each
answer forecloses or opens a different kind of work:

  1. HOW FAR OUTSIDE? If most misses sit just past their half-width, a small uniform accuracy
     gain flips a lot of them and the rms lever is alive after all. If they sit at 3-10x h,
     only a change that removes outliers can pay, and every rms-shaped arm is dead for `sps`.
     This is measured as the distribution of |E|/h over missed elements.
  2. WHERE? Containment resolved by lead time, by x column, by y row and by channel. A deficit
     concentrated in the wake shear layer or in the last few lead times is targetable; one
     spread uniformly is not.
  3. WHICH WINDOWS? If the deficit is carried by a minority of windows, the per-window sigma
     scaling (`half = base * (sigma/sigma_ref)^p`) is mis-sizing them and that IS fixable,
     because `p` was fitted for average behaviour. Reported as containment against window
     rel-L2 decile.

Plus the number that prices all of it: containment and `sps` under a UNIFORM shrink of every
error by 1/2/5/10/20 %, with the band held FIXED. Round 9's exchange rate refit the band at
each point, which answers a different question (what a better model plus a refitted band is
worth); holding the band fixed isolates how steep containment is right here.
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

import sps9
from sps9 import prep, sc, sps_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--err-ckpt", nargs="+", default=None,
                    help="held-out twins; the misses are measured on THESE, since the shipped "
                         "members memorized val_unseen and their error field is not honest")
    ap.add_argument("--band", required=True)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=1.39)
    args = ap.parse_args()

    import torch
    import _fno

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = np.float32(args.f)

    def load(cs):
        return [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
                for c in cs]

    m = load(args.ckpt)
    m_err = m if args.err_ckpt is None else load(args.err_ckpt)
    b = np.load(args.band if os.path.isabs(args.band) else os.path.join(HERE, args.band))
    base, sigma_ref, p = b["base"], b["sigma_ref"], float(b["p"])

    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    keep = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    X, Y = du["x"][keep], du["y"][keep]

    def infer(models, want_sigma=False):
        s = np.empty((len(X), 1, 32, 64, 2), np.float32) if want_sigma else None
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=s)
        P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
        return (P, np.concatenate([s, np.zeros_like(s[..., :1])], -1).astype(np.float32)
                ) if want_sigma else P

    # The half-width the package would deploy uses the SHIPPED members' sigma; the error field
    # it is judged against comes from the twins. Mixing them is the point -- that is what the
    # submission does.
    _, sig = infer(m, want_sigma=True)
    P = infer(m_err)
    half = (base[..., :2] * np.clip(sig[..., :2] / sigma_ref[..., :2], 0.3, 3.0) ** p
            ).astype(np.float32)

    A, a_w, scored, n = prep(P, Y)          # |E| per element, a_i per sample, scored mask
    dm, tk, mv = sps9.sample_metrics(P, Y)
    a_b = sps9.weights_from_metrics(f * dm, f * tk, f * mv)
    E = f * A                                # errors stretched to board magnitude
    print(f"AoA {args.aoa}: {len(X)} windows, {int(scored.sum())} scored elements, "
          f"a = {a_b.mean():.4f}")
    print(f"sps as deployed: {sps_score(E, a_b, scored, n, half):.3f}   "
          f"(ceiling 100*a = {100 * a_b.mean():.2f})")

    inb = (E <= half) & scored
    w = np.broadcast_to(a_b.reshape(-1, 1, 1, 1, 1), scored.shape)
    cw = float((w * inb).sum() / (w * scored).sum())
    print(f"weighted containment: {cw:.4f}   -> the missing "
          f"{100 * (1 - cw):.1f} % is what `sps` is short\n")

    # ---- 1. HOW FAR OUTSIDE ----------------------------------------------------------------
    miss = scored & ~inb
    H = np.broadcast_to(half, E.shape)
    r = E[miss] / np.maximum(H[miss], 1e-12)
    qs = [50, 75, 90, 95, 99]
    print("1. HOW FAR OUTSIDE  |E|/h over missed elements")
    print("   quantiles " + "  ".join(f"p{q} {np.percentile(r, q):.2f}" for q in qs))
    for thr in (1.1, 1.25, 1.5, 2.0, 3.0, 5.0):
        print(f"   within {thr:4.2f}x h: {100 * float((r <= thr).mean()):5.1f} % of misses"
              f"   (= {100 * float((r <= thr).mean()) * (1 - cw):4.1f} % of all scored)")
    # THE ROUND-20 SCREEN, in one line so arms can be compared without reading the table.
    # R15 -- the only change that ever moved board `sps` -- made rel_l2 WORSE while moving
    # `sps` +1.013, so what to look for is the FAR TAIL falling, not the rms. These are
    # a_i-weighted fractions of all scored elements, the same weighting `sps` itself uses.
    R = np.zeros_like(E)
    np.divide(E, np.maximum(H, 1e-12), out=R)
    den = float((w * scored).sum())
    t2 = float((w * (scored & (R > 2.0))).sum()) / den
    t3 = float((w * (scored & (R > 3.0))).sum()) / den
    print(f"   TAIL SUMMARY (the round-20 screen)   miss {100 * (1 - cw):5.2f} %"
          f"   |E|>2h {100 * t2:5.2f} %   |E|>3h {100 * t3:5.2f} %")

    # ---- 2. WHERE --------------------------------------------------------------------------
    def by(axes, labels, name):
        num = (w * inb).sum(axis=axes)
        den = (w * scored).sum(axis=axes)
        c = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        c = c.ravel()
        lo, hi = int(np.nanargmin(c)), int(np.nanargmax(c))
        print(f"   {name:11s} min {c[lo]:.3f} @{labels}{lo:<3d}  max {c[hi]:.3f} "
              f"@{labels}{hi:<3d}  spread {np.nanmax(c) - np.nanmin(c):.3f}")
        return c

    print("\n2. WHERE  weighted containment resolved along each axis")
    ct = by((0, 2, 3, 4), "t", "lead time")
    by((0, 1, 3, 4), "y", "row y")
    by((0, 1, 2, 4), "x", "col x")
    cc = by((0, 1, 2, 3), "c", "channel")
    print(f"   lead time  t=0 {ct[0]:.3f} -> t=19 {ct[19]:.3f}   "
          f"(the forecast horizon's own cost)")
    print(f"   channel    u {cc[0]:.3f}   v {cc[1]:.3f}")

    # ---- 3. WHICH WINDOWS ------------------------------------------------------------------
    print("\n3. WHICH WINDOWS  containment by window rel-L2 decile")
    rl = np.asarray(sc.rel_l2_per_sample(P, Y, 2), float)
    o = np.argsort(rl)
    per = ((scored & inb).sum(axis=(1, 2, 3, 4)) /
           np.maximum(scored.sum(axis=(1, 2, 3, 4)), 1))
    dec = np.array_split(o, 10)
    print("   decile  " + " ".join(f"{i:5d}" for i in range(10)))
    print("   relL2   " + " ".join(f"{rl[d].mean():5.3f}" for d in dec))
    print("   contain " + " ".join(f"{per[d].mean():5.3f}" for d in dec))

    # ---- the price of accuracy at a FIXED band ---------------------------------------------
    print("\n4. UNIFORM ERROR SHRINK, BAND HELD FIXED  (round 9's rate refits the band; this "
          "does not)")
    print(f"   {'cut':>5s} {'containment':>12s} {'sps':>8s} {'d sps':>8s} {'d final':>9s}")
    s0 = sps_score(E, a_b, scored, n, half)
    for cut in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
        Ec = E * np.float32(1.0 - cut)
        inb_c = (Ec <= half) & scored
        cwc = float((w * inb_c).sum() / (w * scored).sum())
        sc_ = sps_score(Ec, a_b, scored, n, half)
        print(f"   {100 * cut:4.0f}% {cwc:12.4f} {sc_:8.3f} {sc_ - s0:+8.3f} "
              f"{0.2502 * (sc_ - s0):+9.3f}")
    print("   `d final` prices the sps row alone at the free 22-round weight 0.250; the same "
          "cut\n   also moves rel_l2, tke and mvpe, which is another ~0.6 x cut/10 %.")


if __name__ == "__main__":
    main()
