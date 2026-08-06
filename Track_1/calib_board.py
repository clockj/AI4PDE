"""Calibrate the SPS band against a MEASURED board score instead of a guessed inflation.

    python calib_board.py --ckpt unet/fno_all_s1.pth unet/fno_all_s2.pth unet/fno_all_s3.pth \
                          --err-ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

Until now the band was sized by guessing how much larger board errors are than local ones. The
2026-08-05 evening submission makes that unnecessary: we now know exactly what the shipped band
scored on the hidden set, so the shift can be *inverted* rather than assumed.

The projection that failed
--------------------------
Predicted `sps ~42.6`, board returned **37.468**. The miss is not explained by aggregate
accuracy -- board rel-L2 came in at 0.11472 against the AoA 15-20 proxy's 0.1082, only 1.06x,
and MVPE and TKE landed close to projection too. Yet SPS behaved like a 1.37x stretch.

SPS is the only subscore that depends on the TAIL of the per-element error rather than its rms.
rel-L2 is an rms; an element that lands just outside the interval costs SPS its entire
contribution while barely moving rel-L2. So a board whose errors are more heavy-tailed than
ours -- same rms, worse coverage -- reproduces exactly this pattern. That is the hypothesis this
script encodes and then corrects for.

Method
------
1. Hold the accuracy factors a_i at the board's ACTUAL measured metrics, since those are known
   exactly and are not in question. Only the per-element error field is stretched.
2. Find the stretch f on the AoA 15-20 proxy that reproduces the board's observed SPS with the
   band that was actually shipped. That single number absorbs the tail mismatch, whatever
   causes it.
3. Re-solve the exact band at that f. Report what it would have scored on the same board.

Step 2 is the calibration; steps 1 and 3 are bookkeeping. The result is only as good as the
assumption that the tail mismatch is stable between submissions, which one board point cannot
verify -- so the output includes the sensitivity of the answer to f.
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

# 2026-08-05 evening board, the 3-seed all-data FNO with the AoA-fitted band.
BOARD = dict(rel=94.575361, tke=78.527795, mvpe=93.909952, time=91.726397, sps=37.467958)


def board_errs():
    return {k: (100.0 / v - 1.0) / 0.5 for k, v in BOARD.items() if k in ("rel", "tke", "mvpe")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--err-ckpt", nargs="+", default=None)
    ap.add_argument("--band", default=os.path.join(HERE, "submit_fno", "band.npy"),
                    help="the band that produced BOARD['sps']")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev) for c in args.ckpt]
    m_err = m if args.err_ckpt is None else [
        _fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev) for c in args.err_ckpt]

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hi_s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hi_u = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)

    # The proxy: the AoA-restricted val_unseen error field of the ERROR model (held-out, so its
    # errors are predictions rather than memorization).
    Pp, Yp = infer(m_err, du["x"], dev, args.alpha)[hi_u], du["y"][hi_u]
    Ap, _, scored_p, np_ = prep(Pp, Yp)

    # a_i from the BOARD's own metrics, applied uniformly. The board reports set means, not
    # per-window values, so every window gets the same factor -- which loses the spread but not
    # the level, and the level is what sets the scale of the answer.
    be = board_errs()
    a_board = np.full(len(Ap), sps9.weights_from_metrics(
        np.float64(be["rel"]), np.float64(be["tke"]), np.float64(be["mvpe"])))
    print(f"board errors  rel {be['rel']:.5f}  tke {be['tke']:.5f}  mvpe {be['mvpe']:.5f}")
    print(f"board accuracy factor a = {a_board[0]:.4f}  -> SPS ceiling {100*a_board[0]:.2f}")
    print(f"proxy rel-L2 (AoA {args.aoa}, held-out models) "
          f"{float(np.mean(sps9.sc.rel_l2_per_sample(Pp, Yp, 2))):.5f}")

    shipped = np.load(args.band)[..., :2]

    def at(f, band):
        return sps_score(np.float32(f) * Ap, a_board, scored_p, np_, band)

    # --- step 2: invert
    grid = np.linspace(1.0, 2.2, 121)
    vals = np.array([at(f, shipped) for f in grid])
    f_hat = float(grid[np.argmin(np.abs(vals - BOARD["sps"]))])
    print(f"\nshipped band vs the proxy stretched by f:")
    for f in (1.0, 1.1, 1.2, f_hat, 1.6):
        print(f"   f={f:.3f}  sps {at(f, shipped):6.2f}"
              + ("   <-- reproduces the board's 37.468" if abs(f - f_hat) < 1e-9 else ""))
    print(f"\ncalibrated stretch f_hat = {f_hat:.3f}   (rel-L2 only implies ~1.06x, so the "
          f"extra\n  {f_hat/1.06:.2f}x is tail, not level: the board's per-element errors miss "
          f"the interval\n  more often than an rms-matched stretch of ours would)")

    # --- step 3: re-solve at the calibrated scale, fitting on val_seen as always
    Pf, Yf = infer(m_err, ds["x"], dev, args.alpha)[hi_s], ds["y"][hi_s]
    Af, af, scored_f, nf = prep(Pf, Yf)
    print(f"\n{'band built for':22s} {'median h':>9s} | " +
          " ".join(f"{'f='+format(f,'.2f'):>9s}" for f in (1.0, f_hat, f_hat * 1.15)))
    print("-" * 60)
    best = (None, -1.0)
    for fb in (1.0, 1.1, 1.3, f_hat, f_hat * 1.15, f_hat * 1.3, f_hat * 1.5):
        b = optimal_band(np.float32(fb) * Af, af, scored_f)
        row = [at(f, b) for f in (1.0, f_hat, f_hat * 1.15)]
        tag = "  <-- shipped" if abs(fb - 1.1) < 1e-9 else ""
        print(f"{fb:22.3f} {np.median(b):9.5f} | " + " ".join(f"{v:9.2f}" for v in row) + tag)
        if row[1] > best[1]:
            best = (fb, row[1])
    print(f"\nbest at the calibrated scale: build for {best[0]:.3f} -> sps {best[1]:.2f} "
          f"on the board, vs {at(f_hat, shipped):.2f} shipped  ({best[1]-at(f_hat, shipped):+.2f})")


if __name__ == "__main__":
    main()
