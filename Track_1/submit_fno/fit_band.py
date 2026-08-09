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
submission, and the board's distribution is not the local one.  Its default is no longer a
guess: the 2026-08-05 evening submission gives a MEASURED board SPS for a known band, so the
shift can be inverted rather than assumed (`../calib_board.py`).

    board rel-L2 0.11472  vs the AoA 15-20 proxy's 0.10818   ->  1.06x on the LEVEL
    board SPS 37.468 with the shipped band                   ->  1.39x on the per-element TAIL

Those two disagree on purpose, and the disagreement is the finding.  rel-L2 is an rms; SPS
depends on whether each element lands inside its interval.  An element just outside forfeits
its entire contribution while barely moving rel-L2, so a board whose errors are more
heavy-tailed than ours -- same level, worse coverage -- produces exactly this pattern.  1.39
sizes the band against the tail, which is what SPS actually charges for.

The gain is real and small.  At the calibrated scale, building for 1.39 scores 37.93 against
37.45 for the previously shipped 1.1, and the row is nearly flat either side (1.30 -> 37.87,
1.60 -> 37.87).  ~0.5 SPS is all that remains in the band; the error scale itself is worth ~5.
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
    ap.add_argument("--inflate", type=float, default=1.39,
                    help="scale the local error field by this before solving, to match the "
                         "error magnitude the model will actually FACE on the board (see the "
                         "module docstring; measured 1.46x)")
    ap.add_argument("--band-p", type=float, default=0.5,
                    help="exponent on the per-window sigma ratio: half(i,e) = base(e) * "
                         "(sigma(i,e)/sigma_ref(e))**p. 0 reproduces the shipped per-element "
                         "band exactly. Round 6 closed this idea using ENSEMBLE SPREAD, which "
                         "correlates with per-element |error| at only 0.347 once the "
                         "per-element mean is removed; the variance head's own sigma "
                         "correlates 0.551. Refitting `base` for each p (deflating the errors "
                         "by r**p before solving, so no p is handicapped) makes p = 0.5 the "
                         "best value BOTH in-sample and out-of-sample -- +0.51 sps on the AoA "
                         "15-20 proxy and +1.64 on the full split. See ../band_sigma_v9.py.")
    ap.add_argument("--band-smooth", type=int, default=3,
                    help="box-blur the per-element base over (T, H, W) with this window. The "
                         "base has 20*32*64*2 = 81,920 free parameters fitted from ~126 "
                         "windows, so `optimal_band`'s exact in-sample optimum is badly "
                         "overfitted; smoothing is pure variance reduction. It cannot be "
                         "chosen in-sample -- the unsmoothed band is the in-sample optimum by "
                         "construction -- so 3 comes from leave-one-TRAJECTORY-out CV inside "
                         "the fitting split (CV sps 42.81 / 43.07 / 43.34 / 43.28 / 43.18 for "
                         "k = 1/2/3/5/7), and the same k is independently best on val_unseen "
                         "(40.615 against 40.414). 1 disables.")
    ap.add_argument("--out", default=os.path.join(HERE, "band.npz"))
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load(c, dev) for c in args.ckpt]
    m_err = m if args.err_ckpt is None else [_fno.load(c, dev) for c in args.err_ckpt]

    ds = np.load(os.path.join(TRACK1, "val_seen.npz"))
    du = np.load(os.path.join(TRACK1, "val_unseen.npz"))

    def infer(models, X, sigma_out=None):
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=sigma_out)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    def sigma_of(models, X):
        s = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=s)
        return np.concatenate([s, np.zeros_like(s[..., :1])], -1).astype(np.float32)

    fit_d = ds if args.fit_on == "seen" else du
    keep = np.isin(np.array([int(str(i).split(":")[0].split("_")[1]) for i in fit_d["ids"]]),
                   args.fit_aoa)
    Pf, Yf = infer(m_err, fit_d["x"])[keep], fit_d["y"][keep]
    Pu, Yu = infer(m, du["x"]), du["y"]

    # sigma_ref comes from --ckpt, NOT --err-ckpt: it normalizes the sigma the SHIPPED models
    # will emit at test time, so it has to be their scale. The error field below still comes
    # from --err-ckpt, because that is about how big the errors honestly are.
    sig_fit = sigma_of(m, fit_d["x"])[keep]
    sigma_ref = np.maximum(sig_fit.mean(0, keepdims=True), 1e-8).astype(np.float32)
    ratio_f = np.clip(sig_fit / sigma_ref, 0.3, 3.0)

    # Solve on the fitting split's error field -- an offline quantity that uses no test target
    # and no per-window information -- with |error| scaled to board magnitude. Deflating by
    # ratio**p first is what makes `base` the right partner for a per-window band: the
    # containment test |E| <= base * r**p is exactly |E| / r**p <= base.
    As, as_, scored_s, ns = prep(Pf, Yf)
    Adef = np.float32(args.inflate) * As
    if args.band_p:
        Adef = (Adef / ratio_f[..., :2] ** args.band_p).astype(np.float32)
    uv_band = optimal_band(Adef, as_, scored_s)
    if args.band_smooth > 1:
        k = args.band_smooth
        o = uv_band.astype(np.float64)
        for ax in (1, 2, 3):                      # T, H, W; channels stay independent
            pad = [(0, 0)] * 5
            pad[ax] = (k // 2, k // 2)
            q = np.pad(o, pad, mode="edge")
            o = np.mean([np.take(q, range(i, i + uv_band.shape[ax]), axis=ax)
                         for i in range(k)], axis=0)
        uv_band = o.astype(np.float32)

    # The scorer only reads the measured channels, but the ingestion program requires bounds
    # the same shape as the prediction, so the unscored pressure channel gets a zero column.
    band = np.concatenate([uv_band, np.zeros_like(uv_band[..., :1])], -1).astype(np.float32)
    np.savez(args.out, base=band, sigma_ref=sigma_ref, p=np.float32(args.band_p))

    U = prep(Pu, Yu)
    hi = np.isin(np.array([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]]),
                 args.fit_aoa)
    Uh = prep(Pu[hi], Yu[hi])
    dmh, tkh, mvh = sps9.sample_metrics(Pu[hi], Yu[hi])

    # The per-window half-widths the submission will actually use, on each reporting split.
    ratio_u = np.clip(sigma_of(m, du["x"]) / sigma_ref, 0.3, 3.0)
    h_fit = (uv_band * ratio_f[..., :2] ** args.band_p).astype(np.float32)
    h_all = (uv_band * ratio_u[..., :2] ** args.band_p).astype(np.float32)
    h_hi = h_all[hi]

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
    print(f"band        base {band.shape}  half-width min {uv_band.min():.5f} "
          f"median {np.median(uv_band):.5f} max {uv_band.max():.5f}   sigma-scaling p="
          f"{args.band_p}")
    print(f"sps  val_{args.fit_on} (fitted)   {sps_score(As, as_, scored_s, ns, h_fit):.2f}"
          f"   [p=0 band: {sps_score(As, as_, scored_s, ns, uv_band):.2f}]")
    contaminated = args.err_ckpt is not None or args.fit_on == "unseen"
    print(f"sps  val_unseen ({'NOT honest' if contaminated else '  reported'}) "
          f"{sps_score(*U[:3], U[3], h_all):.2f}   vs "
          f"{sc.score_sps(sc.aggregate_sps(Pu, Yu, 2)[0]):.2f} for the scorer's default band")
    print("sps  board proxy (val_unseen AoA "
          + f"{args.fit_aoa}) stretched: "
          + "  ".join(f"{f:.2f}x {sps_score(*at_board(f), h_hi):5.2f}"
                      for f in (1.0, 1.05, 1.15, 1.3)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
