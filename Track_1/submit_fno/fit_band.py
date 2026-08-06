"""Fit the per-element SPS half-width map that ships with the FNO submission. v9 scoring.

    python submit_fno/fit_band.py --ckpt ../unet/fno_grad.pth

Fits on `val_seen.npz`, reports on `val_unseen.npz`, writes `band.npy` -- one
(1, 20, 32, 64, 3) array of half-widths that `submission.py` broadcasts over the batch.

WHAT V9 CHANGED
---------------
`score_sps` was `100 / (1 + exp(-w))` and is now `100 * w`.  The aggregate the model produces
barely moved across the rule change (0.376 -> 0.369), so the board's drop from 59.30 to 36.94
is the rescaling, not a regression.  Two consequences for this file:

  * SPS is now ~4.4x more responsive to the band (slope 100 instead of ~23), so it is worth
    solving properly rather than approximately.
  * Elements whose target is exactly 0 -- outside the PIV field of view, or inside the airfoil
    body, ~9.9% of u,v -- are now dropped from both the numerator and the denominator.  They
    used to be free points, since the prediction there is ~0 and any band around it contains
    the target.  Losing a subset that scored at the maximum is most of the rest of the drop.

HOW THE BAND IS CHOSEN
----------------------
The v9 aggregate factorizes into a per-SAMPLE accuracy factor times a per-ELEMENT interval
term (see ../sps9.py), so the band problem separates over elements and each element's optimum
sits exactly at one of its observed |error| values.  `sps9.optimal_band` solves that exactly by
sorting -- no Gaussian assumption and no grid.  Measured against the parametric families it
replaces (val_seen -> val_unseen): const 48.86 -> 47.73, gauss-on-RMS 50.22 -> 48.95,
error-quantile 48.87 -> 47.52, exact 51.54 -> 49.54.

--INFLATE exists because the band is the only fitted-against-a-distribution part of the
submission, and the board's distribution is not the local one:

    board rel-L2 err 0.1179  vs  local val_unseen 0.0807   ->  1.46x

A band sized for 0.0807 is too narrow at 0.1179, and an element outside the interval scores
zero however good the prediction was.  Inflation re-solves the band against |error| scaled by
that factor.  The gain is real but bounded: the built-for-1.0 band scores 41.32 when met with
1.4x errors and the built-for-1.4 band scores 41.97, so mismatch costs ~0.7 SPS while the error
scale itself costs 8.2.  The default is 1.5 -- near the measured 1.46, and the row is flat
enough (1.4 and 1.6 differ by 0.02 in mean over the plausible range) that the exact choice
does not matter.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
sys.path.insert(0, TRACK1)
sys.path.insert(0, HERE)

import _fno  # noqa: E402
import sps9  # noqa: E402
from sps9 import optimal_band, prep, sc, sps_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=[os.path.join(TRACK1, "unet", "fno_grad.pth")],
                    help="one checkpoint, or several to fit the band for that ensemble. "
                         "The band is derived from the model's OWN error field, so it "
                         "must be refitted whenever the member set changes -- an "
                         "ensemble's errors are smaller and its half-widths narrower.")
    ap.add_argument("--err-ckpt", nargs="+", default=None,
                    help="derive the ERROR FIELD from these checkpoints instead of --ckpt. "
                         "Needed for an --all-data package: those models trained on val_seen "
                         "AND val_unseen, so their local errors are memorization, not "
                         "prediction, and a band fitted to them is far too narrow. That is "
                         "exactly how the U-Net submission ended at sps 36.94 -- local rel-L2 "
                         "0.0721 contaminated, band median 0.0046, board errors 0.1179. Point "
                         "this at the held-out models of the SAME recipe: their error field is "
                         "genuinely out-of-sample and their error STRUCTURE is what the "
                         "all-data model will have too, only slightly larger. Erring wide is "
                         "the cheap direction (see --inflate).")
    ap.add_argument("--fit-on", choices=["seen", "unseen"], default="seen",
                    help="which split the error field comes from. 'seen' is in-sample for the "
                         "held-out models and the honest choice when --ckpt is being reported "
                         "on val_unseen; 'unseen' is out-of-sample and the better one when "
                         "--err-ckpt is set, since nothing is being reported anyway.")
    ap.add_argument("--fit-aoa", type=int, nargs="+", default=[15, 20],
                    help="restrict the fitting split to these angles of attack. Error is "
                         "governed by AoA and by almost nothing else -- AoA 15-20 carries 2.4x "
                         "the rel-L2 of AoA 5 -- and the AoA 15-20 subset of val_unseen reads "
                         "0.1082 against the board's 0.1179, where the full split reads 0.0807. "
                         "The board is therefore modelled far better as 'the high-incidence "
                         "regime, slightly harder' than as 'everything, 1.46x harder', and the "
                         "extrapolation shrinks from 1.46x to ~1.1x. Measured on the AoA 15-20 "
                         "view of val_unseen: fitting there scores 43.78 against 43.24 for the "
                         "whole split at inflate 1.5. Pass 0 5 10 15 20 to fit on everything.")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--inflate", type=float, default=1.1,
                    help="scale the local error field by this before solving, to match the "
                         "error magnitude the model will actually FACE on the board (see the "
                         "module docstring; measured 1.46x)")
    ap.add_argument("--out", default=os.path.join(HERE, "band.npy"))
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load(c, dev) for c in args.ckpt]
    m_err = m if args.err_ckpt is None else [_fno.load(c, dev) for c in args.err_ckpt]

    ds = np.load(os.path.join(TRACK1, "val_seen.npz"))
    du = np.load(os.path.join(TRACK1, "val_unseen.npz"))

    def infer(models, X):
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    fit_d = ds if args.fit_on == "seen" else du
    keep = np.isin(np.array([int(str(i).split(":")[0].split("_")[1]) for i in fit_d["ids"]]),
                   args.fit_aoa)
    Pf, Yf = infer(m_err, fit_d["x"])[keep], fit_d["y"][keep]
    Pu, Yu = infer(m, du["x"]), du["y"]

    # Solve on the fitting split's error field -- an offline quantity that uses no test target
    # and no per-window information -- with |error| scaled to board magnitude.
    As, as_, scored_s, ns = prep(Pf, Yf)
    uv_band = optimal_band(np.float32(args.inflate) * As, as_, scored_s)

    # The scorer only reads the measured channels, but the ingestion program requires bounds
    # the same shape as the prediction, so the unscored pressure channel gets a zero column.
    band = np.concatenate([uv_band, np.zeros_like(uv_band[..., :1])], -1).astype(np.float32)
    np.save(args.out, band)

    U = prep(Pu, Yu)
    hi = np.isin(np.array([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]]),
                 args.fit_aoa)
    Uh = prep(Pu[hi], Yu[hi])
    dmh, tkh, mvh = sps9.sample_metrics(Pu[hi], Yu[hi])

    def at_board(f):
        """The AoA-restricted view of val_unseen, stretched by f -- the closest proxy we have
        for the hidden set (its rel-L2 is 0.1082 against the board's 0.1179)."""
        return (np.float32(f) * Uh[0],
                sps9.weights_from_metrics(f * dmh, f * tkh, f * mvh), Uh[2], Uh[3])

    print(f"ckpt        {', '.join(os.path.basename(c) for c in args.ckpt)}")
    if args.err_ckpt is not None:
        print(f"err-ckpt    {', '.join(os.path.basename(c) for c in args.err_ckpt)}")
    print(f"fit-on      val_{args.fit_on}  AoA {args.fit_aoa}  "
          f"({int(keep.sum())}/{len(keep)} windows)   inflate {args.inflate:.2f}x")
    print(f"band        {band.shape}  half-width min {uv_band.min():.5f} "
          f"median {np.median(uv_band):.5f} max {uv_band.max():.5f}")
    print(f"sps  val_{args.fit_on} (fitted)   {sps_score(As, as_, scored_s, ns, uv_band):.2f}")
    contaminated = args.err_ckpt is not None or args.fit_on == "unseen"
    print(f"sps  val_unseen ({'NOT honest' if contaminated else '  reported'}) "
          f"{sps_score(*U, uv_band):.2f}   vs "
          f"{sc.score_sps(sc.aggregate_sps(Pu, Yu, 2)[0]):.2f} for the scorer's default band")
    print("sps  board proxy (val_unseen AoA "
          + f"{args.fit_aoa}) stretched: "
          + "  ".join(f"{f:.2f}x {sps_score(*at_board(f), uv_band):5.2f}"
                      for f in (1.0, 1.05, 1.15, 1.3)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
