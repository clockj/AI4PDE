"""Re-invert the SPS tail inflation against the ROUND-10 board point.

    python calib_board11.py --ckpt <shipped all-data members> --err-ckpt <held-out counterparts>
                            --band submit_fno/band_r10.npz

`calib_board.py` did this for round 9 and got f = 1.39, which every band since has been built
for. Round 10 shipped a different band -- per-window, sigma-scaled, smoothed -- and returned a
different SPS, so the inversion can be repeated on an independent point. Two outcomes matter:

  * f_hat lands near 1.39      -> the tail model is stable across submissions and the band is
                                  sized correctly; stop touching --inflate.
  * f_hat lands well above     -> the board's per-element errors are further out in the tail
                                  than we build for, and the band should be widened.

Unlike calib_board.py this reproduces the band the round-10 package ACTUALLY used, including
the per-window sigma scaling: half(i,e) = base(e) * clip(sigma_i(e)/sigma_ref(e), 0.3, 3)**p,
with sigma from the SHIPPED models, which is what submission.py computes.
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

import sps9  # noqa: E402
from sps9 import optimal_band, prep, sps_score  # noqa: E402

# Round 10, 2026-08-08. Errors inverted from the subscores via score_error(e) = 100/(1+0.5e).
BOARD = dict(rel_l2=94.576394, tke=79.152828, mvpe=93.681162, sps=39.626333)


def err_of(score):
    return (100.0 / score - 1.0) / 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="the SHIPPED members (sigma)")
    ap.add_argument("--err-ckpt", nargs="+", required=True, help="held-out (error field)")
    ap.add_argument("--band", required=True, help="the band.npz that produced BOARD['sps']")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    args = ap.parse_args()

    import torch
    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load_any(c, dev) for c in args.ckpt]
    m_err = [_fno.load_any(c, dev) for c in args.err_ckpt]

    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    hu = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    hs = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)

    def infer(models, X, want_sigma=False):
        sig = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=sig)
        P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
        return (P, sig) if want_sigma else P

    Pp, Yp = infer(m_err, du["x"][hu]), du["y"][hu]
    _, sig_p = infer(m, du["x"][hu], want_sigma=True)
    Ap, _, scored_p, np_ = prep(Pp, Yp)

    be = {k: err_of(BOARD[k]) for k in ("rel_l2", "tke", "mvpe")}
    a0 = float(sps9.weights_from_metrics(np.float64(be["rel_l2"]), np.float64(be["tke"]),
                                         np.float64(be["mvpe"])))
    a_board = np.full(len(Ap), a0)
    print(f"board errors  rel {be['rel_l2']:.5f}  tke {be['tke']:.5f}  mvpe {be['mvpe']:.5f}")
    print(f"board accuracy factor a = {a0:.4f}  -> SPS ceiling {100*a0:.2f}, scored "
          f"{BOARD['sps']:.3f} ({BOARD['sps']/(100*a0):.1%} of it)")
    print(f"proxy rel-L2 (AoA {args.aoa}, held-out) "
          f"{float(np.mean(sps9.sc.rel_l2_per_sample(Pp, Yp, 2))):.5f}")

    z = np.load(args.band)
    base, ref, p = z["base"][..., :2], z["sigma_ref"][..., :2], float(z["p"])
    shipped = (base * np.clip(sig_p / ref, 0.3, 3.0) ** np.float32(p)).astype(np.float32)
    print(f"band: p={p}, median deployed half-width {np.median(shipped):.5f}")

    def at(f, half):
        return sps_score(np.float32(f) * Ap, a_board, scored_p, np_, half)

    grid = np.linspace(1.0, 2.4, 141)
    vals = np.array([at(f, shipped) for f in grid])
    f_hat = float(grid[np.argmin(np.abs(vals - BOARD["sps"]))])
    print("\nshipped band vs the proxy stretched by f:")
    for f in (1.0, 1.2, 1.39, f_hat, 1.8):
        tag = "   <-- reproduces the board" if abs(f - f_hat) < 1e-9 else ""
        print(f"   f={f:.3f}  sps {at(f, shipped):6.2f}{tag}")
    print(f"\ncalibrated stretch f_hat = {f_hat:.3f}   (round 9's inversion gave 1.39)")

    # Re-solve the band at a range of build scales and score each at the calibrated stretch.
    Pf, Yf = infer(m_err, ds["x"][hs]), ds["y"][hs]
    _, sig_f = infer(m, ds["x"][hs], want_sigma=True)
    Af, af, scored_f, _ = prep(Pf, Yf)
    rf = np.clip(sig_f / ref, 0.3, 3.0)

    def smooth(b, k=3):
        o = b.astype(np.float64)
        for ax in (1, 2, 3):
            pad = [(0, 0)] * 5
            pad[ax] = (k // 2, k // 2)
            q = np.pad(o, pad, mode="edge")
            o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(k)], 0)
        return o.astype(np.float32)

    print(f"\n{'build for':>10s} {'median h':>9s} | {'sps at f_hat':>13s}")
    print("-" * 38)
    best = (None, -1.0)
    for fb in (1.10, 1.25, 1.39, 1.55, 1.75, 2.00, 2.30):
        b = smooth(optimal_band((np.float32(fb) * Af / rf ** np.float32(p)).astype(np.float32),
                                af, scored_f))
        half = (b * np.clip(sig_p / ref, 0.3, 3.0) ** np.float32(p)).astype(np.float32)
        v = at(f_hat, half)
        tag = "  <-- shipped" if abs(fb - 1.39) < 1e-9 else ""
        print(f"{fb:10.2f} {np.median(b):9.5f} | {v:13.2f}{tag}")
        if v > best[1]:
            best = (fb, v)
    ship = at(f_hat, shipped)
    print(f"\nbest build scale {best[0]:.2f} -> sps {best[1]:.2f} vs {ship:.2f} shipped "
          f"({best[1]-ship:+.2f} sps, {(best[1]-ship)*0.23:+.3f} final at the mid sps weight)")


if __name__ == "__main__":
    main()
