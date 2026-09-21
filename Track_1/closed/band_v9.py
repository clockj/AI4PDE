"""Explore the SPS half-width band under v9 scoring.

    python band_v9.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

Everything is fitted on val_seen and reported on val_unseen.  v9 turned `score_sps` from a
logistic into an identity, so a gain in the aggregate is worth 4.4x what it used to be and the
band is the cheapest place in the pipeline to find one.

The candidates:

  const        one half-width everywhere -- the null model
  gauss        optimal_half(c * rms_error), the incumbent; assumes the per-element error is
               Gaussian with std equal to its RMS over windows
  quantile     h = c * per-element absolute-error quantile: distribution-free, one global knob
  exact        sps9.optimal_band -- the per-element in-sample maximum, solved exactly

`exact` is both a candidate band and a diagnostic.  Its val_seen number is the ceiling on
what ANY band-fitting procedure can reach on that data, so the gap between it and the incumbent
says how much SPS is still available from intervals alone; its val_unseen number says how much
of that survives the transfer.  A large seen/unseen gap there means the exact solution is
fitting the 336-window empirical CDF, not the error distribution.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os

import numpy as np

import sps9
from sps9 import optimal_band, prep, sps_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def optimal_half_gauss(s):
    """Per-element half-width maximizing exp(-2h/sigma) * erf(h / (s*sqrt2)) -- the incumbent.

    Unchanged by v9: the per-element trade between interval width and the chance of containing
    the target is what it always was.  v9 changed which elements enter the average and how the
    average maps to a score, neither of which touches this optimum.
    """
    import torch
    hs = torch.linspace(1e-4, 0.08, 400).view(-1, 1)
    knots = torch.logspace(-5, 0, 2000).view(1, -1)
    g = torch.exp(-2.0 * hs / sps9.SIGMA) * torch.erf(hs / (knots * 1.4142135))
    table = hs.view(-1)[g.argmax(0)].numpy()
    return np.interp(np.clip(s, 1e-5, 1.0), knots.view(-1).numpy(), table).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    args = ap.parse_args()

    import torch
    import sys
    sys.path.insert(0, os.path.join(HERE, "submit_fno"))
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    Ps, Ys = infer(models, ds["x"], dev, args.alpha), ds["y"]
    Pu, Yu = infer(models, du["x"], dev, args.alpha), du["y"]

    S, U = prep(Ps, Ys), prep(Pu, Yu)
    rms = np.sqrt(np.mean(S[0] ** 2, axis=0, keepdims=True)).astype(np.float32)

    fast, ref = sps9.self_check(Pu, Yu, np.float32(0.0065))
    print(f"self-check vs kit aggregate_sps: {fast:.6f} vs {ref:.6f}  "
          f"(delta {abs(fast-ref):.2e})")
    print(f"sigma {sps9.SIGMA:.6f}   sigma/2 {sps9.SIGMA/2:.6f}")
    print(f"accuracy ceiling (all inside, zero width): seen {100*S[1].mean()*1:.2f} -> "
          f"unseen {100*U[1].mean():.2f}")

    def fit(name, w, grid):
        best = max(((sps_score(*S, c * w), c) for c in grid), key=lambda z: z[0])
        band = (best[1] * w).astype(np.float32) if np.ndim(w) else np.float32(best[1] * w)
        med = float(np.median(band))
        print(f"  {name:24s} c {best[1]:6.3f}  median h {med:.5f}   "
              f"seen {best[0]:6.2f}   unseen {sps_score(*U, band):6.2f}")
        return band

    dflt = sps9.sc.score_sps(sps9.sc.aggregate_sps(Pu, Yu, 2)[0])
    print(f"\nscorer's own default +-5%|p| band, val_unseen: {dflt:.2f}")
    print("fitted on val_seen, reported on val_unseen:")

    fit("const", np.float32(1.0), np.linspace(0.0015, 0.030, 60))
    fit("gauss(rms)", optimal_half_gauss(rms), np.linspace(0.2, 2.5, 47))
    for q in (0.4, 0.5, 0.6, 0.7, 0.8):
        qh = np.quantile(S[0], q, axis=0, keepdims=True).astype(np.float32)
        fit(f"quantile {q:.1f}", qh, np.linspace(0.2, 2.5, 47))

    eb = optimal_band(*S[:3])
    print(f"  {'exact (in-sample opt)':24s} c  1.000  median h {np.median(eb):.5f}   "
          f"seen {sps_score(*S, eb):6.2f}   unseen {sps_score(*U, eb):6.2f}")
    fit("exact, rescaled", eb, np.linspace(0.5, 2.0, 31))


if __name__ == "__main__":
    main()
