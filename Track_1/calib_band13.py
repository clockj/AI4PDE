"""Measure the board's per-element error TAIL from a round whose band and SPS are both known.

    python calib_band13.py --band /path/to/the/shipped/band.npz --alpha 0.60 \
        --ckpt unet/fno_all_vl_s1.pth ... --err-ckpt unet/fno_vl_s1.pth ...

`calib_board.py` did this once, in round 7, for a band that was a plain `.npy` array of
half-widths at ALPHA 0.75.  Everything in that sentence has since changed: the band is an
`.npz` whose width is `base * (sigma/sigma_ref)**p`, the package is 8 fused members, and ALPHA
is 0.60.  The number it produced -- **f = 1.39**, the factor by which the board's per-element
errors miss their interval more often than the AoA 15-20 proxy's do -- has been used to size
every band since, and has never been re-measured.

Why it is worth re-measuring now.  SPS is 0.224 of `final` (final_v9.py, nine rounds) and 52%
of the return on any accuracy improvement, and it is the ONLY subscore that depends on the tail
of the per-element error rather than its rms.  If f has drifted, the shipped band is the wrong
width and the fix is free -- no retraining, no extra latency, one refit.

Method, unchanged from `calib_board.py` and correct for the same reason: hold the per-sample
accuracy factors `a_i` at the board's OWN measured rel/tke/mvpe (those are known exactly and
are not what is in question), stretch only the per-element error field, and find the stretch
that reproduces the board's observed SPS with the band that actually shipped.  Then re-solve
the exact band at that stretch and report what it would have scored.

TWO independent reads.  Round 13 shipped the identical package to both tracks, so there are two
board SPS values for one band -- Track 1's 39.689 and Track 2's 39.730.  They should imply the
same f; that they do is the check that this number is a property of the scoring, not of a
single draw.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import band_io
import opoint
import sps9
from sps9 import optimal_band, prep

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

# Round 13, 2026-08-31, submissions 910641 (T1) and 910642 (T2): the SAME 8 members, the same
# band.npz and the same ALPHA 0.60 on both boards.
# 2026-09-28: R37 added -- the Track-1 package of record (ALPHA 0.80, JOINT_GAIN, the SIGBLEND
# band centred on y0; board_rounds.py). Its Track-2 twin T2R24 streams an adaptive band no local
# instrument replays, so it is not a second read here. Pick the rows with --rounds; each must be
# run with ITS OWN band, members and alpha (R13: bands/r13_8memb.npz, --alpha 0.60).
BOARDS = {
    "T1 R13": dict(rel=94.679785, tke=77.983720, mvpe=93.613336, sps=39.689258),
    "T2 R1 ": dict(rel=94.694512, tke=77.942224, mvpe=93.399401, sps=39.730331),
    "T1 R37": dict(rel=94.776302, tke=79.589142, mvpe=94.261277, sps=43.155698),
}


def board_errs(b):
    return {k: (100.0 / b[k] - 1.0) / 0.5 for k in ("rel", "tke", "mvpe")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="the SHIPPED members -- they supply the sigma the band ratio uses")
    ap.add_argument("--err-ckpt", nargs="+", required=True,
                    help="held-out twins -- they supply an honest error field")
    ap.add_argument("--band", required=True, help="the band.npz that produced BOARDS[*]['sps'], read "
                    "through band_io.load_band (a SIGBLEND npz needs its sidecar json)")
    ap.add_argument("--sidecar", default=None, help="SIGBLEND sidecar json of --band (default: found)")
    ap.add_argument("--rounds", nargs="+", default=["T1 R37"], choices=sorted(BOARDS),
                    help="board rounds whose sps the band produced (default 'T1 R37', the package of "
                         "record; 'T1 R13' 'T2 R1 ' for the round-13 read)")
    ap.add_argument("--alpha", type=float, default=opoint.ALPHA,
                    help=f"compose alpha of that package (default {opoint.ALPHA}, opoint.ALPHA; R13: 0.60)")
    ap.add_argument("--joint-gain", action=argparse.BooleanOptionalAction, default=None,
                    help="compose with _fno.JOINT_GAIN. Neither flag (default) = opoint.JOINT_GAIN "
                         f"({opoint.JOINT_GAIN}) for a band centred on y0, else off (R13's band)")
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--band-smooth", type=int, default=3,
                    help="box blur over (T,H,W), as fit_band.py applies it. MUST match what "
                         "the shipped band was built with, or the refit rows below are being "
                         "compared against a smoothed band and will look worse than they are.")
    args = ap.parse_args()

    import torch
    import _fno

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
         for c in args.ckpt]
    m_err = [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
             for c in args.err_ckpt]

    band = band_io.load_band(args.band, args.sidecar)
    jg = band_io.joint_gain_for(band, args.joint_gain)
    _fno.JOINT_GAIN = jg            # module state read by predict_fno's compose (2026-09-28)
    sigma_ref, p = band.sigma_ref, band.p
    # The refit rows below re-solve a STATIC band (fit_band.py's construction); a SIGBLEND band's
    # static parent is fitted at opoint.BAND_P, not at its own p' (2026-09-28).
    p_refit = p if band.kind == "static" else opoint.BAND_P
    boards = {k: BOARDS[k] for k in args.rounds}

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hi_s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hi_u = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)

    def infer(models, X, sigma_out=None, centre_out=None):
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=sigma_out,
                              centre_out=centre_out)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    def sigma_of(models, X):
        s = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=s)
        return np.concatenate([s, np.zeros_like(s[..., :1])], -1).astype(np.float32)

    # Proxy error field: held-out twins on the AoA 15-20 slice of val_unseen; their alpha-0 mean
    # is the interval centre when the band is centred on y0 (2026-09-28).
    c0u = np.zeros(du["x"].shape[:1] + (20, 32, 64, 3), np.float32)
    Pu, Yu = infer(m_err, du["x"], centre_out=c0u)[hi_u], du["y"][hi_u]
    y0u = c0u[hi_u][..., :2]
    scored_p = Yu[..., :2] != 0.0
    # The interval the SUBMISSION would build on these windows: the shipped models' sigma against
    # the band's stored sigma_ref, its centre and (SIGBLEND) its sigma-power factor -- band_io.
    sig_u = sigma_of(m, du["x"])[hi_u]
    ratio = np.clip(sig_u / sigma_ref, 0.3, 3.0)
    shipped = band_io.half_width(band, sig_u)

    def sps_at(f, a, half):
        """sps of `half` at stretch f (target moved about the band's centre), accuracy factors a."""
        lo_, up_ = band_io.interval(band, y0u, Pu, sig_u, alpha=args.alpha, joint_gain=jg, half=half)
        yf = band_io.stretch_target(band, Yu, y0u, Pu, f, alpha=args.alpha, joint_gain=jg)
        return sps9.sps_interval(a, yf, lo_, up_, scored_p)[0]

    print(f"{band.describe()}  |  alpha {args.alpha}  joint_gain {jg}  proxy = val_unseen AoA "
          f"{args.aoa} ({int(hi_u.sum())} windows)")
    print(f"proxy rel-L2 (held-out twins) "
          f"{float(np.mean(sps9.sc.rel_l2_per_sample(Pu, Yu, 2))):.5f}")
    print(f"shipped half-width  median {float(np.median(shipped)):.5f}  "
          f"max {float(shipped.max()):.5f}")

    grid = np.linspace(1.00, 2.40, 281)
    fits = {}
    for name, b in boards.items():
        be = board_errs(b)
        a = np.full(len(Pu), sps9.weights_from_metrics(
            np.float64(be["rel"]), np.float64(be["tke"]), np.float64(be["mvpe"])))
        vals = np.array([sps_at(f, a, shipped) for f in grid])
        f_hat = float(grid[np.argmin(np.abs(vals - b["sps"]))])
        fits[name] = (f_hat, a, be)
        print(f"\n{name}: board rel {be['rel']:.5f} tke {be['tke']:.5f} mvpe {be['mvpe']:.5f}"
              f"  -> a = {a[0]:.4f} (SPS ceiling {100*a[0]:.2f})")
        print(f"{name}: board sps {b['sps']:.3f}   reproduced at "
              f"**f = {f_hat:.3f}**   (round 7 measured 1.39)")
        for f in (1.20, 1.39, f_hat, 1.60):
            print(f"        f={f:.3f}  proxy sps {sps_at(f, a, shipped):6.3f}"
                  + ("   <-- matches the board" if abs(f - f_hat) < 1e-9 else ""))

    f_use = float(np.mean([v[0] for v in fits.values()]))
    fs_ = [v[0] for v in fits.values()]
    print(f"\n=== {len(fits)} board read(s) agree on f to {max(fs_) - min(fs_):.3f}; "
          f"using their mean f = {f_use:.3f} ===")

    # Re-solve the band at the calibrated stretch, fitting on val_seen as the shipped one does:
    # the error about the band's own centre (y0 for a centre-0 band, as fit_band --centre-alpha 0).
    c0s = np.zeros(ds["x"].shape[:1] + (20, 32, 64, 3), np.float32)
    Pf, Yf = infer(m_err, ds["x"], centre_out=c0s)[hi_s], ds["y"][hi_s]
    Cf = band_io.centre(band, c0s[hi_s][..., :2], Pf[..., :2], args.alpha, jg)
    Af, af, scored_f, nf = prep(np.concatenate([Cf, Pf[..., 2:]], -1), Yf)
    af = sps9.sample_weights(Pf, Yf)       # accuracy factors of the SCORED prediction, not the centre
    ratio_f = np.clip(sigma_of(m, ds["x"])[hi_s] / sigma_ref, 0.3, 3.0)
    a_use = next(iter(fits.values()))[1]

    print(f"\n{'band fitted at':>16s} {'median h':>9s} | " +
          " ".join(f"{'sps@f='+format(f,'.2f'):>12s}" for f in (1.39, f_use, f_use * 1.10)))
    print("-" * 62)
    rows = []
    for fb in (1.20, 1.39, 1.55, f_use, f_use * 1.15, f_use * 1.35, f_use * 1.6):
        Adef = np.float32(fb) * Af
        if p_refit:
            Adef = (Adef / ratio_f[..., :2] ** p_refit).astype(np.float32)
        b_ = optimal_band(Adef, af, scored_f)
        if args.band_smooth > 1:                      # identical to fit_band.py
            k, o = args.band_smooth, b_.astype(np.float64)
            for ax in (1, 2, 3):
                pad = [(0, 0)] * 5
                pad[ax] = (k // 2, k // 2)
                q = np.pad(o, pad, mode="edge")
                o = np.mean([np.take(q, range(i, i + b_.shape[ax]), axis=ax)
                             for i in range(k)], axis=0)
            b_ = o.astype(np.float32)
        h_ = (b_ * ratio[..., :2] ** p_refit).astype(np.float32)
        r = [sps_at(f, a_use, h_) for f in (1.39, f_use, f_use * 1.10)]
        rows.append((fb, r[1]))
        tag = ("  <-- what shipped" if abs(fb - 1.39) < 1e-9 and band.kind == "static" else "")
        print(f"{fb:16.3f} {float(np.median(b_)):9.5f} | "
              + " ".join(f"{v:12.3f}" for v in r) + tag)
    base_sps = sps_at(f_use, a_use, shipped)
    best = max(rows, key=lambda t: t[1])
    print(f"\nshipped band at the calibrated stretch: {base_sps:.3f}")
    print(f"best refit: fit at {best[0]:.3f} -> {best[1]:.3f}   "
          f"({best[1]-base_sps:+.3f} sps = {(best[1]-base_sps)*0.2236:+.3f} final)")


if __name__ == "__main__":
    main()
