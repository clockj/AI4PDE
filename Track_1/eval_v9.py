"""Score a checkpoint (or ensemble) under the v9 scoring rules.

    python eval_v9.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

v9 changed two things that matter, and both of them are about SPS:

1. `score_sps` was `100 / (1 + exp(-w))` and is now `100 * w`.  The old form had a floor of
   50 at w=0 and a ceiling of 73.11 at w=1; the new one runs the full 0..100.  The aggregate
   `w` the model produces barely moved between the two boards (0.376 -> 0.369), so the drop
   from 59.30 to 36.94 is almost entirely this rescaling.  It also makes SPS roughly 4.4x
   more responsive: d(score)/dw was ~23 near w=0.6 and is now 100.

2. Elements whose target is exactly 0 -- outside the PIV field of view, or inside the airfoil
   body -- are dropped from BOTH the numerator and the denominator.  They are ~9.9% of u,v
   and they used to be free points: the prediction there is ~0, any band around it contains
   the target, so each contributed the maximum (1-pm)*exp(-nil).  Removing a subset that
   scored at the maximum lowers the mean.

`final_score` is no longer published and is no longer the plain mean of the five (the mean of
the 2026-08-05 board subscores is 78.999 against a reported 79.733), so `final_mean` below is
a reference point for tracking progress, not the leaderboard's number.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KIT9 = "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9"

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

_spec = importlib.util.spec_from_file_location("sc9", os.path.join(KIT9, "scoring.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

import _fno  # noqa: E402


# --------------------------------------------------------------------------- scoring


def sps_of(P, Y, half):
    """half: None for the scorer's default +-5% band, else a broadcastable half-width."""
    if half is None:
        return sc.score_sps(sc.aggregate_sps(P, Y, 2)[0])
    lo, hi = P - half, P + half
    return sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=lo, upper=hi)[0])


def sps_parts(P, Y, half):
    """(sps score, coverage, weighted mean of (1-pm)) -- the two factors of the aggregate."""
    lo, hi = (P - half, P + half) if half is not None else (None, None)
    w, cov = sc.aggregate_sps(P, Y, 2, lower=lo, upper=hi)
    dm = sc.rel_l2_per_sample(P, Y, 2)
    tk = sc.tke_rel_l2_per_sample(P, Y, 2)
    mv = sc.mvpe_rel_l2_per_sample(P, Y)
    acc = (0.5 * np.mean(1 - dm / (0.5 + dm)) + 0.3 * np.mean(1 - tk / (0.5 + tk))
           + 0.2 * np.mean(1 - mv / (0.5 + mv)))
    return sc.score_sps(w), cov, float(acc)


def subscores(P, Y, tstep, half=None):
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
    mv = sc.mvpe_rel_l2(P, Y)
    s = [sc.score_error(r), sc.score_error(k), sc.score_error(mv),
         sc.score_time(tstep), sps_of(P, Y, half)]
    return dict(rel_err=r, tke_err=k, mvpe_err=mv, rel=s[0], tke=s[1], mvpe=s[2],
                time=s[3], sps=s[4], final_mean=float(np.mean(s)), ms=tstep * 1000)


# --------------------------------------------------------------------------- band


def err_map(P, Y):
    """Per-element RMS error over the window axis: (1, T, H, W, 3)."""
    return np.sqrt(np.mean((P - Y) ** 2, axis=0, keepdims=True)).astype(np.float32)


def optimal_half(s):
    """Per-element half-width maximizing exp(-2h/sigma) * erf(h / (s*sqrt2)).

    Unchanged by v9: the per-element trade-off between interval width and the chance of
    containing the target is exactly what it was.  What v9 changed is which elements enter the
    average and how the average maps to a score, neither of which touches this optimum.
    """
    hs = torch.linspace(1e-4, 0.08, 400).view(-1, 1)
    knots = torch.logspace(-5, 0, 2000).view(1, -1)
    g = torch.exp(-2.0 * hs / sc.SIGMA_GLOBAL) * torch.erf(hs / (knots * 1.4142135))
    table = hs.view(-1)[g.argmax(0)].numpy()
    return np.interp(np.clip(s, 1e-5, 1.0), knots.view(-1).numpy(), table).astype(np.float32)


def fit_scale(P, Y, w, grid):
    best = (-1.0, None)
    for c in grid:
        v = sps_of(P, Y, c * w if w is not None else c)
        if v > best[0]:
            best = (v, c)
    return best[1], best[0]


# --------------------------------------------------------------------------- inference


def infer(models, X, dev, alpha=0.75):
    uv = _fno.predict_fno(models, X, dev, alpha=alpha)
    return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.75])
    ap.add_argument("--ms", type=float, default=1.190,
                    help="end-to-end ms/window measured by validate_strict.py; eval_wp-style "
                         "forward-only timing understates time_score by ~3 points")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    t = args.ms / 1000.0

    for alpha in args.alphas:
        Ps, Ys = infer(models, ds["x"], dev, alpha), ds["y"]
        Pu, Yu = infer(models, du["x"], dev, alpha), du["y"]

        emap = err_map(Ps, Ys)
        c, sps_seen = fit_scale(Ps, Ys, optimal_half(emap), np.linspace(0.1, 3.0, 30))
        band = (c * optimal_half(emap)).astype(np.float32)

        print(f"\nalpha {alpha}  |  band scale c={c:.3f}  median half {np.median(band):.5f}")
        for tag, (P, Y) in (("seen", (Ps, Ys)), ("unseen", (Pu, Yu))):
            d = subscores(P, Y, t, band)
            sps, cov, acc = sps_parts(P, Y, band)
            dflt = sps_of(P, Y, None)
            print(f"  {tag:7s} rel {d['rel']:.2f}  tke {d['tke']:.2f}  mvpe {d['mvpe']:.2f}  "
                  f"time {d['time']:.2f}  sps {d['sps']:.2f} (default band {dflt:.2f})  "
                  f"| mean {d['final_mean']:.3f}")
            print(f"          errs  rel {d['rel_err']:.4f}  tke {d['tke_err']:.4f}  "
                  f"mvpe {d['mvpe_err']:.4f}  | coverage {cov:.3f}  acc-factor {acc:.4f}  "
                  f"ceiling-if-all-inside {100*acc:.2f}")


if __name__ == "__main__":
    main()
