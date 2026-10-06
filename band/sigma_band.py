"""Blend the static band with a power law in sigma, fitted on the twins' held-out errors.

    python sigma_band.py --dump dump_unseen.npz --band band_static.npz --out <dir>

The package evaluates

    half = base' * clip(sigma / sigma_ref, 0.3, 3)^p' * exp(q_c * lam * log(sigma + 1e-8))

with base' = base^(1 - lam) * kappa_{t,c}^lam, p' = (1 - lam) * p and lam = 0.5. kappa (one
value per output frame and channel) and q (one slope per channel) come from the exact per-bin
optimum of the SPS interval term on the UNSEEN windows: the twins' errors |y0 - y| (y0 the mean
of their fields), inflated by 1.29, binned into 16 quantile bins of the twins' own log sigma,
one optimal half-width per bin, then a pooled slope with per-frame intercepts.

Writes `band_sigblend_lam0.5.npz` (the static npz with base and p replaced) and
`band_sigblend_lam0.5.json`, whose `T2_sigpow` = (lam q_u, lam q_v) is baked into
submission.py as `SIGPOW` by build/build.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sps  # noqa: E402
from sps import SIGMA  # noqa: E402

ALPHA, INFL, NB, LAM = 0.80, 1.29, 16, 0.5


def compose(y0, sig, alpha=ALPHA):
    """The package's fluctuation adjustment, in numpy: the field whose per-pixel temporal std
    is moved toward sigma."""
    m = y0.mean(1, keepdims=True); s = np.maximum(y0.std(1, keepdims=True), 1e-8)
    return (m + (y0 - m) * (alpha * sig + (1 - alpha) * s) / s).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="dump_twins.py's npz")
    ap.add_argument("--band", required=True, help="fit_band.py's static band")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.dump)
    F, S, y = d["F"].astype(np.float32), d["S"].astype(np.float32), d["y"]
    traj = np.array([str(i).split(":")[0] for i in d["ids"]])
    y0 = F.mean(0); sig = S.mean(0)
    P = compose(y0, sig)
    P3 = np.concatenate([P, np.zeros_like(P[..., :1])], -1)
    Y3 = np.concatenate([y, np.zeros_like(y[..., :1])], -1)
    met = sps.sample_metrics(P3, Y3)
    scored = y != 0.0
    a1 = sps.weights_from_metrics(*(1.0 * m for m in met))
    E = np.abs(y0 - y).astype(np.float32)

    z = np.load(a.band)
    base3 = z["base"].astype(np.float32); sref3 = z["sigma_ref"].astype(np.float32); p_s = float(z["p"])
    base_s, sref = base3[..., :2], sref3[..., :2]
    logsig = np.log(sig.astype(np.float32) + np.float32(1e-8))           # (N, 1, 32, 64, 2)

    # The fit, on all eight trajectories: q (2,), log kappa (20, 2).
    fit = np.ones(len(traj), bool)
    W = (a1[fit][:, None, None, None, None] * scored[fit]).astype(np.float64)
    Ef, Ls = E[fit], np.broadcast_to(logsig, y0.shape)[fit]
    q = np.zeros(2); logk = np.zeros((20, 2))
    for c in range(2):
        X, Y, T = [], [], []
        for t in range(20):
            A = (Ef[:, t, :, :, c] * np.float32(INFL)).ravel(); s = Ls[:, t, :, :, c].ravel(); w = W[:, t, :, :, c].ravel()
            ok = w > 0; A, s, w = A[ok], s[ok], w[ok]
            edges = np.quantile(s, np.linspace(0, 1, NB + 1)); edges[0], edges[-1] = -np.inf, np.inf
            bi = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, NB - 1)
            for j in range(NB):
                m = bi == j
                Aj, wj = A[m], w[m]
                o = np.argsort(Aj, kind="stable"); Aj, wj = Aj[o], wj[o]
                gain = np.cumsum(wj) * np.exp(-2.0 * Aj.astype(np.float64) / SIGMA)
                X.append(np.median(s[m])); Y.append(np.log(max(float(Aj[gain.argmax()]), 1e-6))); T.append(t)
        X, Y, T = np.array(X), np.array(Y), np.array(T)
        xm = np.array([X[T == t].mean() for t in range(20)]); ym = np.array([Y[T == t].mean() for t in range(20)])
        q[c] = float(((X - xm[T]) * (Y - ym[T])).sum() / ((X - xm[T]) ** 2).sum())      # pooled slope
        logk[:, c] = ym - q[c] * xm                                                     # per-frame intercepts
    print("fit on all 8 trajectories:  q (u, v) =", np.round(q, 4).tolist(), flush=True)

    lam = LAM
    bp = np.zeros_like(base3)
    bp[..., :2] = (base_s ** np.float32(1 - lam)) * np.exp(lam * logk).astype(np.float32)[None, :, None, None, :]
    keys = {k: z[k] for k in z.files}
    keys.update(base=bp.astype(np.float32), p=np.float32((1 - lam) * p_s))
    out = os.path.join(a.out, f"band_sigblend_lam{lam:.1f}.npz")
    np.savez(out, **keys)
    json.dump({"lam": lam, "q": q.tolist(), "logkappa": logk.tolist(), "infl": INFL,
               "p_static": p_s, "p_new": (1 - lam) * p_s,
               "T2_sigpow": [lam * q[0], lam * q[1]], "static_band": a.band, "npz": out},
              open(os.path.join(a.out, f"band_sigblend_lam{lam:.1f}.json"), "w"), indent=1)
    print(f"wrote {out}  p' {(1 - lam) * p_s:.4f}  SIGPOW ({lam * q[0]!r}, {lam * q[1]!r})", flush=True)


if __name__ == "__main__":
    main()
