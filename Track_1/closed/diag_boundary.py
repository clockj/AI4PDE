"""Where does U-FNO's local advantage live?  Error decomposed by distance to the zero set.

Round 16 asked one question the board forced on us: U-FNO beats the FNO on EVERY local
instrument -- the AoA 15-20 proxy (rel-L2 -3.0%), the full held-out split (-4.4% at 3-vs-3)
and the interior Re-band k-fold (-2.14%, z -6.9, 2152 windows) -- and came back +1.9% WORSE
on the hidden test set (T2R4 vs T2R3).  Every mundane explanation has been eliminated: the
all-data members trained normally, fp16 packing perturbs both families identically, and the
two inference paths agree in-sample.  So the hidden set must differ from every local set in
some property a LOCAL conv branch is sensitive to and a GLOBAL spectral operator is not.

Round 15 proposed exactly that mechanism when it proposed U-FNO in the first place: "mode
truncation cannot represent the sharp zero set of the PIV field of view and the airfoil body".
If that is where U-FNO's gain lives, then a hidden set whose zero set has a different geometry
would erase it -- which is what the board reported.  This script tests the antecedent.

Method.  Take the 336 val_unseen windows, three 3-member ensembles of HELD-OUT twins, and the
ALPHA=0.60 composition compare_v9.py scores (`eval_v9.infer`, raw physical units).  Define the
per-window zero set Z as the pixels whose TARGET is exactly 0 in u AND v for all 20 frames --
the PIV field of view boundary plus the airfoil body -- and give every pixel its chessboard
(8-connected BFS) distance d to Z.  Then split the squared-error field by d and ask what
fraction of the error REDUCTION each band carries against the fraction of elements it holds.

Reading rule, pre-registered: the advantage is boundary-localized if >= 40% of the reduction
falls within d <= 3 while those bands hold much less than 40% of the elements; otherwise it is
spread over the wake and the zero-set hypothesis does not explain the board.

    python diag_boundary.py                 # all three pairs, all tables
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

# The three member sets, all HELD-OUT twins of the shipped all-data members, all width 96.
SETS = {
    "ufno": ["unet/arch_ufno_w96.pth", "unet/arch_ufno_w96_s2.pth", "unet/arch_ufno_w96_s3.pth"],
    "fno_do30": ["unet/fno_cap_w96_do30_e400.pth", "unet/fno_cap_w96_do30_s2.pth",
                 "unet/fno_cap_w96_do30_s3.pth"],
    "fno_L8_do45": ["unet/fno_r16_w96_do45_L8.pth", "unet/fno_r16_w96_do45_L8_s2.pth",
                    "unet/fno_r16_w96_do45_L8_s3.pth"],
}

# d = 0 is Z itself.  The bands widen geometrically because the interesting structure is at
# the boundary: a 3-pixel skin on a 32x64 grid is already a large share of the domain.
BANDS = [("d=0 (Z)", 0, 0), ("d=1", 1, 1), ("d=2", 2, 2), ("d=3", 3, 3),
         ("d=4-6", 4, 6), ("d=7-12", 7, 12), ("d>12", 13, 10 ** 6)]


def chessboard_distance(Z: np.ndarray) -> np.ndarray:
    """Per-window chessboard (8-connected BFS) distance to the nearest True in Z.

    Z: (N, H, W) bool.  Returns int16 (N, H, W); 0 inside Z.  A window with an empty Z gets a
    saturating distance so it lands in the outermost band rather than raising -- no real window
    is like that (the PIV field of view boundary is always present), but a silent inf would
    poison every mean downstream.

    Implemented as a plain two-pass chamfer rather than scipy.ndimage so the script has no
    dependency the rest of the analysis tree does not already carry, and so the metric is
    exactly the one the docstring claims.
    """
    n, h, w = Z.shape
    BIG = np.int32(10 ** 6)
    d = np.where(Z, 0, BIG).astype(np.int32)
    # Forward pass: rows top->bottom, cols left->right, using the 4 already-final neighbours.
    for i in range(h):
        for j in range(w):
            best = d[:, i, j]
            for di, dj in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < h and 0 <= jj < w:
                    best = np.minimum(best, d[:, ii, jj] + 1)
            d[:, i, j] = best
    # Backward pass: the mirror image, which completes the chamfer.
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            best = d[:, i, j]
            for di, dj in ((1, 1), (1, 0), (1, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < h and 0 <= jj < w:
                    best = np.minimum(best, d[:, ii, jj] + 1)
            d[:, i, j] = best
    return np.minimum(d, 30000).astype(np.int16)


def band_index(d: np.ndarray) -> np.ndarray:
    """(N,H,W) distance -> (N,H,W) index into BANDS."""
    b = np.full(d.shape, len(BANDS) - 1, np.int8)
    for k, (_, lo, hi) in enumerate(BANDS):
        b[(d >= lo) & (d <= hi)] = k
    return b


def _tbl(rows, hdr):
    w = [max(len(str(hdr[i])), max((len(str(r[i])) for r in rows), default=0))
         for i in range(len(hdr))]
    line = "  ".join(str(hdr[i]).rjust(w[i]) for i in range(len(hdr)))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[i]).rjust(w[i]) for i in range(len(hdr))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.60,
                    help="composition alpha; 0.60 is what compare_v9/the package use")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N windows")
    args = ap.parse_args()

    import torch
    from eval_v9 import infer
    import _fno

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    X, Y, ids = du["x"], du["y"], du["ids"]
    if args.limit:
        X, Y, ids = X[:args.limit], Y[:args.limit], ids[:args.limit]
    N, T, H, W, _ = Y.shape
    aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in ids])
    print(f"val_unseen: {N} windows, {T}x{H}x{W}, alpha {args.alpha}, device {dev}")

    Yuv = Y[..., :2].astype(np.float64)

    # ---------------------------------------------------------------- the zero set
    # "target exactly 0 in BOTH u and v across the WHOLE window" -- the static geometry
    # (field of view + airfoil body), not the occasional element that happens to vanish.
    Z = ((Y[..., 0] == 0) & (Y[..., 1] == 0)).all(1)                    # (N,H,W)
    elem_zero = (Yuv == 0)                                              # per ELEMENT
    print(f"zero set Z: {Z.mean():6.2%} of pixels          "
          f"(per-element target==0, u&v: {((Yuv[..., 0] == 0) & (Yuv[..., 1] == 0)).mean():.2%};"
          f" any channel: {elem_zero.mean():.2%})")
    print(f"  Z pixel count per window: min {Z.sum((1, 2)).min()}  median "
          f"{int(np.median(Z.sum((1, 2))))}  max {Z.sum((1, 2)).max()}  of {H * W}")
    print("  NOTE CLAUDE.md round 14 records 3.3%; that figure is a different (per-element, "
          "pooled-over-train_real) count. The number above is what val_unseen actually has.")

    d = chessboard_distance(Z)
    bidx = band_index(d)                                                # (N,H,W)
    nb = len(BANDS)

    # Per-pixel target energy and element counts, shared by every model.
    Y2 = (Yuv ** 2).sum((1, 4))                                         # (N,H,W)
    n_elem = np.full((N, H, W), T * 2, np.int64)                        # all u,v elements
    n_scored = (~elem_zero).sum((1, 4))                                 # SPS-scored elements

    def by_band(a):
        """(N,H,W) -> (nb,) sum per distance band, pooled over windows."""
        return np.array([a[bidx == k].sum() for k in range(nb)], np.float64)

    # ---------------------------------------------------------------- predictions
    S, absZ, rel_w = {}, {}, {}
    for name, ck in SETS.items():
        paths = [c if os.path.isabs(c) else os.path.join(HERE, c) for c in ck]
        miss = [p for p in paths if not os.path.exists(p)]
        if miss:
            raise SystemExit(f"{name}: missing {miss}")
        models = [_fno.load(p, dev) for p in paths]
        P = infer(models, X, dev, args.alpha)[..., :2].astype(np.float64)
        S[name] = ((P - Yuv) ** 2).sum((1, 4))                          # (N,H,W) sum over T,C
        rel_w[name] = np.sqrt(S[name].sum((1, 2)) / Y2.sum((1, 2)))     # per-window rel-L2
        # Does the model emit anything inside the zero set?  rel_l2 keeps those elements even
        # though SPS drops them, so a non-zero emission there is a real, scored error.
        m = np.broadcast_to(Z[:, None, :, :, None], P.shape)
        absZ[name] = (float(np.abs(P[..., 0])[m[..., 0]].mean()),
                      float(np.abs(P[..., 1])[m[..., 1]].mean()),
                      float(np.abs(Yuv[..., 0])[m[..., 0]].mean()))
        del models, P
        print(f"loaded {name}: pooled rel-L2 {np.sqrt(S[name].sum() / Y2.sum()):.5f}  "
              f"mean per-window {rel_w[name].mean():.5f}")

    # ---------------------------------------------------------------- (0) zero-set emission
    print("\n=== zero-set emission: mean |prediction| inside Z (physical m/s) ===")
    _tbl([[k, f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.1e}",
           f"{100 * S[k][Z].sum() / S[k].sum():5.2f}%"] for k, v in absZ.items()],
         ["model", "|u| on Z", "|v| on Z", "|u_true| on Z", "share of tot sq err"])

    # ---------------------------------------------------------------- (a)+(b) distance bands
    cnt_e, cnt_s, y2b = by_band(n_elem), by_band(n_scored), by_band(Y2)
    for a, b in (("ufno", "fno_do30"), ("fno_L8_do45", "fno_do30")):
        sa, sb = by_band(S[a]), by_band(S[b])
        red, tot = sb - sa, (sb - sa).sum()
        print(f"\n=== distance bands: {a} vs {b}   "
              f"(total sq-err reduction {tot:.5g}, {100 * tot / S[b].sum():+.2f}% of {b}'s "
              f"total squared error) ===")
        rows = []
        for k, (lab, _, _) in enumerate(BANDS):
            # relL2 is undefined inside Z (the target is identically 0 there), so the rms
            # error carries the comparison in every band and relL2 is shown where it exists.
            ra = np.sqrt(sa[k] / cnt_e[k]) if cnt_e[k] else np.nan
            rb = np.sqrt(sb[k] / cnt_e[k]) if cnt_e[k] else np.nan
            rows.append([lab,
                         f"{100 * cnt_e[k] / cnt_e.sum():6.2f}%",
                         f"{100 * cnt_s[k] / cnt_s.sum():6.2f}%",
                         f"{100 * sb[k] / sb.sum():6.2f}%",
                         f"{100 * red[k] / tot:+7.2f}%" if tot else "     n/a",
                         f"{100 * red[k] / S[b].sum():+6.3f}%",
                         f"{rb:.5f}", f"{ra:.5f}", f"{100 * (ra / rb - 1):+6.2f}%",
                         (f"{np.sqrt(sb[k] / y2b[k]):.5f}" if y2b[k] > 0 else "   --"),
                         (f"{np.sqrt(sa[k] / y2b[k]):.5f}" if y2b[k] > 0 else "   --")])
        _tbl(rows, ["band", "%elem", "%scored", f"%errB", "%red", "red/errB",
                    f"rms {b}", f"rms {a}", "d%", f"relL2 {b}", f"relL2 {a}"])
        near = slice(0, 4)
        print(f"  d<=3: {100 * cnt_e[near].sum() / cnt_e.sum():.2f}% of elements, "
              f"{100 * sb[near].sum() / sb.sum():.2f}% of {b}'s error, "
              + (f"{100 * red[near].sum() / tot:+.2f}% of the reduction" if tot else "n/a"))
        print(f"  d>=1 only (drop Z): d<=3 holds "
              f"{100 * cnt_e[1:4].sum() / cnt_e[1:].sum():.2f}% of the non-Z elements and "
              + (f"{100 * red[1:4].sum() / red[1:].sum():+.2f}% of the non-Z reduction"
                 if red[1:].sum() else "n/a"))

    # ---------------------------------------------------------------- (c) by AoA
    for a, b in (("ufno", "fno_do30"), ("fno_L8_do45", "fno_do30")):
        print(f"\n=== by AoA: {a} vs {b} ===")
        rows = []
        tot = (S[b] - S[a]).sum()
        for ang in sorted(set(aoa.tolist())):
            m = aoa == ang
            rb = np.sqrt(S[b][m].sum() / Y2[m].sum())
            ra = np.sqrt(S[a][m].sum() / Y2[m].sum())
            redm = (S[b][m] - S[a][m]).sum()
            zsh = Z[m].mean()
            # boundary share of THIS angle's reduction
            bb = bidx[m]
            near = (S[b][m] - S[a][m])[bb <= 3].sum()
            rows.append([ang, int(m.sum()), f"{zsh:6.2%}", f"{rb:.5f}", f"{ra:.5f}",
                         f"{100 * (ra / rb - 1):+6.2f}%", f"{100 * redm / tot:+7.2f}%",
                         f"{100 * near / redm if redm != 0 else np.nan:+7.2f}%"])
        _tbl(rows, ["AoA", "n", "Z frac", f"relL2 {b}", f"relL2 {a}", "d%",
                    "%of tot red", "%red at d<=3"])

    # ---------------------------------------------------------------- (c) by domain half
    for a, b in (("ufno", "fno_do30"), ("fno_L8_do45", "fno_do30")):
        print(f"\n=== by domain half (streamwise x = W index): {a} vs {b} ===")
        rows = []
        tot = (S[b] - S[a]).sum()
        for lab, sl in (("x<32 (fore)", slice(0, 32)), ("x>=32 (aft)", slice(32, 64))):
            sb, sa, yy = S[b][:, :, sl], S[a][:, :, sl], Y2[:, :, sl]
            rows.append([lab, f"{Z[:, :, sl].mean():6.2%}",
                         f"{100 * n_elem[:, :, sl].sum() / n_elem.sum():5.1f}%",
                         f"{np.sqrt(sb.sum() / yy.sum()):.5f}",
                         f"{np.sqrt(sa.sum() / yy.sum()):.5f}",
                         f"{100 * (np.sqrt(sa.sum() / yy.sum()) / np.sqrt(sb.sum() / yy.sum()) - 1):+6.2f}%",
                         f"{100 * (sb - sa).sum() / tot:+7.2f}%"])
        _tbl(rows, ["half", "Z frac", "%elem", f"relL2 {b}", f"relL2 {a}", "d%", "%of tot red"])

    # ---------------------------------------------------------------- (d) correlation
    print("\n=== per-window rel-L2: is the gain where the control is already bad? ===")
    rows = []
    for a, b in (("ufno", "fno_do30"), ("fno_L8_do45", "fno_do30")):
        g = rel_w[b] - rel_w[a]
        gp = g / rel_w[b]
        rows.append([f"{a} vs {b}",
                     f"{np.corrcoef(g, rel_w[b])[0, 1]:+.3f}",
                     f"{np.corrcoef(gp, rel_w[b])[0, 1]:+.3f}",
                     f"{np.corrcoef(g, Z.mean((1, 2)))[0, 1]:+.3f}",
                     f"{(g > 0).mean():5.1%}",
                     f"{100 * (rel_w[a].mean() / rel_w[b].mean() - 1):+6.2f}%"])
    _tbl(rows, ["pair", "corr(gain, err_A)", "corr(gain%, err_A)", "corr(gain, Z frac)",
                "win rate", "mean rel-L2 d%"])


if __name__ == "__main__":
    main()
