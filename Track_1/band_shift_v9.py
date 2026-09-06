"""Is the SPS interval allowed to be OFF-CENTRE, and is it worth anything?

    python band_shift_v9.py --ckpt unet/fno_r16_w96_do45_L8{,_s2,_s3}.pth

THE OBSERVATION.  Every band this project has ever shipped is `pred +/- h`.  Nothing in the
scoring program requires that.  `aggregate_sps` reads `lower`/`upper` and asks only that they
be finite, the same shape as the prediction, and `lower <= upper`; the prediction itself never
has to lie inside.  The per-sample accuracy factor `a_i` is computed from the PREDICTION array,
which we do not touch -- so an off-centre band moves `sps_score` and leaves `rel_l2`, `tke` and
`mvpe` bit-identical.  That makes it the only remaining free lever on the largest gap in the
scorecard (sps 41.3 against a ceiling of 100*a = 70.6 at board accuracy) and the cheapest
experiment available: no retraining, no latency, and a board reading that cannot be confounded.

THE ALGEBRA.  With E = pred - target and an interval [pred - h_lo, pred + h_hi],

    target in [lower, upper]   <=>   -h_hi <= E <= h_lo,     width = h_lo + h_hi

so the per-element objective is exp(-(u - l)/sigma) * sum_i a_i * 1[l <= E_ie <= u] over the
observed signed errors, with l = -h_hi and u = h_lo.  The symmetric band is the special case
u = -l, i.e. `sps9.optimal_band` on |E|.  Two questions follow, and this file answers both:

  * THE CEILING.  What does the EXACT asymmetric optimum score in-sample?  Both endpoints sit
    at observed errors, so sorting plus one pass over pairs solves it exactly -- the same
    argument `sps9.optimal_band` makes in one dimension, in two.  If that ceiling is close to
    the symmetric optimum the direction is dead in one reading.

  * THE DEPLOYABLE VERSION.  An exact per-element two-sided optimum has 2 x 81,920 free
    parameters fitted from ~126 windows and would overfit exactly as round 10b's unsmoothed
    base did.  The shrunk version is a per-element CENTRE SHIFT mu_e: ship
    `(pred - mu_e) +/- h`, which is the existing machinery run on |E - mu| instead of |E|.
    mu is the model's systematic per-element bias, it is smoothed and shrunk by the same
    box-blur-and-scale discipline the base gets, and -- the part that decides it -- it is
    chosen by leave-one-TRAJECTORY-out CV inside the fitting split and then reported on a
    split the fit never saw.  Round 10b earned that rule: the unsmoothed band IS the in-sample
    optimum by construction, so nothing in this family can be selected in-sample.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import _fno      # noqa: E402
import sps9      # noqa: E402
from sps9 import SIGMA, optimal_band, prep, sps_score  # noqa: E402


def box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """The same (T, H, W) box blur fit_band.py applies to `base`; channels stay independent."""
    if k <= 1:
        return a
    o = a.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * o.ndim
        pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + a.shape[ax]), axis=ax) for i in range(k)], axis=0)
    return o.astype(np.float32)


def asym_ceiling(E: np.ndarray, a: np.ndarray, scored: np.ndarray, n_scored: int,
                 chunk: int = 2048, cap: float = 0.08) -> float:
    """EXACT in-sample optimum over asymmetric intervals, per element.

    Sorting the signed errors of one element makes every candidate interval a contiguous run,
    so the objective is exp(-(e_j - e_i)/sigma) * (S_j - S_{i-1}) over i <= j with S the
    cumulative a-weight.  Both endpoints are observed errors, exactly as in the symmetric case.
    Evaluated over all pairs in chunks of elements; this is a CEILING, not a shippable band.
    """
    n = len(E)
    flat = E.reshape(n, -1).astype(np.float32)
    wts = (a[:, None] * scored.reshape(n, -1)).astype(np.float32)
    tot = 0.0
    for s in range(0, flat.shape[1], chunk):
        f, w = flat[:, s:s + chunk], wts[:, s:s + chunk]
        order = np.argsort(f, axis=0, kind="stable")
        srt = np.take_along_axis(f, order, axis=0)
        cw = np.cumsum(np.take_along_axis(w, order, axis=0), axis=0)
        # S_j - S_{i-1} for every i <= j, and the width penalty for that pair.
        excl = np.concatenate([np.zeros((1, f.shape[1]), np.float32), cw[:-1]], 0)
        mass = cw[None, :, :] - excl[:, None, :]                   # (i, j, m)
        width = srt[None, :, :] - srt[:, None, :]                  # (i, j, m)
        bad = (width < 0) | (width > 2.0 * cap)
        g = np.where(bad, np.float32(0.0), mass * np.exp(-width / np.float32(SIGMA)))
        tot += float(g.reshape(-1, f.shape[1]).max(axis=0).sum(dtype=np.float64))
    return float(100.0 * tot / n_scored)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="held-out models. Both the prediction and the error field come from "
                         "these; nothing here is meant to ship, only to price the direction.")
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--inflate", type=float, default=1.39,
                    help="board/proxy per-element TAIL stretch (calib_band13.py: 1.360 T1, "
                         "1.350 T2). Applied to the SIGNED errors, so it stretches the bias too.")
    ap.add_argument("--fit-aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--band-p", type=float, default=0.45)
    ap.add_argument("--band-smooth", type=int, default=3)
    ap.add_argument("--shift-smooth", type=int, nargs="+", default=[1, 3, 5, 9])
    ap.add_argument("--shift-scale", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--ceiling", action="store_true",
                    help="also compute the exact asymmetric in-sample ceiling (a few minutes)")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load_any(c, dev) for c in args.ckpt]

    def infer(X, sigma_out=None):
        uv = _fno.predict_fno(m, X, dev, alpha=args.alpha, sigma_out=sigma_out)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    def sigma_of(X):
        s = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        _fno.predict_fno(m, X, dev, alpha=args.alpha, sigma_out=s)
        return s.astype(np.float32)

    def aoa_of(ds):
        return np.array([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]])

    def traj_of(ds):
        return np.array([str(i).split(":")[0] for i in ds["ids"]])

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    kf = np.isin(aoa_of(ds), args.fit_aoa)
    ku = np.isin(aoa_of(du), args.fit_aoa)
    print(f"fit  val_seen   AoA {args.fit_aoa}: {int(kf.sum())}/{len(kf)} windows")
    print(f"rep  val_unseen AoA {args.fit_aoa}: {int(ku.sum())}/{len(ku)} windows")

    Pf, Yf = infer(ds["x"])[kf], ds["y"][kf]
    Pu, Yu = infer(du["x"])[ku], du["y"][ku]
    trf = traj_of(ds)[kf]

    # Signed errors, board-stretched. `prep` gives |E|, a_i, the scored mask and n_scored; the
    # sign is what this file is about, so E is rebuilt and the rest reused verbatim.
    Af, af, sf, nf = prep(Pf, Yf)
    Au, au, su, nu = prep(Pu, Yu)
    f = np.float32(args.inflate)
    Ef = f * (Pf[..., :2] - Yf[..., :2]).astype(np.float32)
    Eu = f * (Pu[..., :2] - Yu[..., :2]).astype(np.float32)
    # The reporting split's `a_i` must be stretched the same way the errors are, exactly as
    # fit_band.at_board does -- a_i is a function of the three per-sample metrics.
    dmu, tku, mvu = sps9.sample_metrics(Pu, Yu)
    au_b = sps9.weights_from_metrics(f * dmu, f * tku, f * mvu)
    dmf, tkf, mvf = sps9.sample_metrics(Pf, Yf)
    af_b = sps9.weights_from_metrics(f * dmf, f * tkf, f * mvf)

    rf = np.clip(sigma_of(ds["x"])[kf] / np.maximum(sigma_of(ds["x"])[kf].mean(0, keepdims=True),
                                                    1e-8), 0.3, 3.0)
    sref = np.maximum(sigma_of(ds["x"])[kf].mean(0, keepdims=True), 1e-8)
    ru = np.clip(sigma_of(du["x"])[ku] / sref, 0.3, 3.0)

    print("\n=== how much per-element BIAS is there at all? ===")
    mu_raw = Ef.mean(0, keepdims=True)
    sd_raw = Ef.std(0, keepdims=True)
    r = np.abs(mu_raw) / np.maximum(sd_raw, 1e-9)
    print(f"  |mean(E)| / std(E) per element: median {np.median(r):.3f}  "
          f"p90 {np.percentile(r, 90):.3f}  fraction > 0.5: {float((r > 0.5).mean()):.3f}")
    print(f"  a zero-bias model would sit at ~1/sqrt(n) = {1/np.sqrt(len(Ef)):.3f}")

    def fit_and_score(shift_f, shift_u):
        """base solved on |E - mu| (deflated by ratio**p), then scored on both splits."""
        Ad = np.abs(Ef - shift_f).astype(np.float32)
        if args.band_p:
            Ad = (Ad / rf ** args.band_p).astype(np.float32)
        base = optimal_band(Ad, af_b, sf)
        base = box_blur(base, args.band_smooth)
        hf = (base * rf ** args.band_p).astype(np.float32)
        hu = (base * ru ** args.band_p).astype(np.float32)
        ins = sps_score(np.abs(Ef - shift_f), af_b, sf, nf, hf)
        out = sps_score(np.abs(Eu - shift_u), au_b, su, nu, hu)
        return ins, out

    print("\n=== centre shift: in-sample and out-of-sample ===")
    print(f"{'k':>3} {'scale':>6} {'in-sample':>10} {'val_unseen':>11} {'delta':>7}")
    base_in, base_out = fit_and_score(np.float32(0.0), np.float32(0.0))
    print(f"{'-':>3} {0.0:6.2f} {base_in:10.3f} {base_out:11.3f} {0.0:+7.3f}   (symmetric)")
    best = (base_out, 0, 0.0)
    for k in args.shift_smooth:
        mu_s = box_blur(mu_raw, k)
        for sc_ in args.shift_scale:
            if sc_ == 0.0:
                continue
            mu = (sc_ * mu_s).astype(np.float32)
            ins, out = fit_and_score(mu, mu)
            print(f"{k:3d} {sc_:6.2f} {ins:10.3f} {out:11.3f} {out - base_out:+7.3f}")
            if out > best[0]:
                best = (out, k, sc_)
    print(f"  best out-of-sample: k={best[1]} scale={best[2]} -> {best[0]:.3f} "
          f"({best[0] - base_out:+.3f} sps)")

    print("\n=== leave-one-TRAJECTORY-out CV inside the fitting split (the honest selector) ===")
    trs = sorted(set(trf.tolist()))
    print(f"  {len(trs)} trajectories in the fitting split")
    print(f"{'k':>3} {'scale':>6} {'CV sps':>9}")
    cv_rows = []
    for k in args.shift_smooth:
        for sc_ in args.shift_scale:
            tot, ntot = 0.0, 0
            for t in trs:
                ho = trf == t
                if ho.all() or not ho.any():
                    continue
                mu = (sc_ * box_blur(Ef[~ho].mean(0, keepdims=True), k)).astype(np.float32)
                Ad = np.abs(Ef[~ho] - mu).astype(np.float32)
                if args.band_p:
                    Ad = (Ad / rf[~ho] ** args.band_p).astype(np.float32)
                base = box_blur(optimal_band(Ad, af_b[~ho], sf[~ho]), args.band_smooth)
                h = (base * rf[ho] ** args.band_p).astype(np.float32)
                nh = int(np.count_nonzero(sf[ho]))
                tot += sps_score(np.abs(Ef[ho] - mu), af_b[ho], sf[ho], nh, h) * nh
                ntot += nh
            cv_rows.append((k, sc_, tot / max(ntot, 1)))
            print(f"{k:3d} {sc_:6.2f} {cv_rows[-1][2]:9.3f}")
    kb, sb, vb = max(cv_rows, key=lambda r: r[2])
    print(f"  CV picks k={kb} scale={sb} (CV sps {vb:.3f})")
    ins, out = fit_and_score((sb * box_blur(mu_raw, kb)).astype(np.float32),
                             (sb * box_blur(mu_raw, kb)).astype(np.float32))
    print(f"  CV choice on val_unseen: {out:.3f} vs symmetric {base_out:.3f} "
          f"({out - base_out:+.3f} sps)")

    if args.ceiling:
        print("\n=== EXACT asymmetric in-sample ceiling (per element, both endpoints free) ===")
        sym_in = sps_score(Af * f, af_b, sf, nf,
                           box_blur(optimal_band((Af * f).astype(np.float32), af_b, sf), 1))
        asym_in = asym_ceiling(Ef, af_b, sf, nf)
        print(f"  exact symmetric  (in-sample, unsmoothed): {sym_in:.3f}")
        print(f"  exact ASYMMETRIC (in-sample, unsmoothed): {asym_in:.3f}  "
              f"({asym_in - sym_in:+.3f} sps of ceiling)")


if __name__ == "__main__":
    main()
