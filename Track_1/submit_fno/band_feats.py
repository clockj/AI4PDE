"""Per-element log|e| band features (09-27 T1 arm, #Leaders task #1 / #2).  Emre, 2026-09-26.

The five window-local features frozen in #Leaders:c5f77dfd (Corin 5b156c79, before any result), for the analysis
(numpy arrays from the twin export) and for the package (torch tensors, per chunk, on the GPU inside predict_fno's
GPU-band path), so the predictor is evaluated on the same numbers it was fitted on.
Two implementations of the same definitions: `features_np` (numpy, = Corin's eband_loto.py feats() operation for operation;
the analysis side, which has no torch on the VM) and `_fno.eband_features` (torch, the package); the build gate checks they agree
on the twin export (max rel <= 1e-5 on yhat). Definitions:

  logsig   log(sigma + 1e-8)                        sigma: the sigma head, member mean, physical, (B,1,H,W,2)
  loggrad  log(sqrt(gy^2 + gx^2) + 1e-6)            g = np.gradient(C0, axis=(2, 3)): unit spacing, central differences
                                                    inside, one-sided first differences at the two edges
  t        frame index / 19
  loginstd log(std_t(x[..., :2]) + 1e-6)            population std over the 20 input frames, per pixel
  logfluc  log(|C0 - mean_t C0| + 1e-6)

eband_smoke (09-26, run 20260926-091622): EBAND off == the 09-26 D package bit for bit; package x and sigma match the twin export
(x identical, sigma ratio 0.9997). torch vs numpy agree to <= 1e-6 on four features but only 3.8e-3 on logfluc: |C0 - mean_t C0| is a
difference of nearly equal numbers, so the two libraries' different reduction orders for mean_t show up after the log near the 1e-6
floor, and the band then matches the numpy reference to ~1e-4 rel instead of 1e-5. If this arm is reopened: compute mean_t in float64
on both sides (or gate at 1e-3) before trusting a 1e-5 gate.

C0 is the alpha-0 field (the member mean before compose, physical), i.e. the band centre. All inputs float32.
yhat = F @ beta[ch][:5] + beta[ch][5] per channel (u, v); half = k * exp(yhat); the interval is then
[min(C0 - half, P), max(C0 + half, P)] exactly as the centred static band.
"""
from __future__ import annotations

import numpy as np

import sys

NAMES = ("logsig", "loggrad", "t", "loginstd", "logfluc")
NF = len(NAMES)


# The torch implementation the package runs lives in _fno.py (eband_features / eband_yhat) because only submission.py, _fno.py,
# the weights and band.npz go into an archive; the build gate checks it against features_np / yhat_np below on the twin export.


def features_np(c0: np.ndarray, sig: np.ndarray, x: np.ndarray) -> np.ndarray:
    """numpy reference (the analysis side): c0 (n,20,H,W,2), sig (n,1,H,W,2), x (n,20,H,W,>=2) -> (n,20,H,W,2,5) float32."""
    c = np.asarray(c0, np.float32)
    g_y, g_x = np.gradient(c, axis=(2, 3))
    fl = [np.broadcast_to(np.log(np.asarray(sig, np.float32) + 1e-8), c.shape),
          np.log(np.sqrt(g_y ** 2 + g_x ** 2) + 1e-6),
          np.broadcast_to((np.arange(c.shape[1], dtype=np.float32) / 19.0)[None, :, None, None, None], c.shape),
          np.broadcast_to(np.log(np.asarray(x)[..., :2].astype(np.float32).std(1, keepdims=True) + 1e-6), c.shape),
          np.log(np.abs(c - c.mean(1, keepdims=True)) + 1e-6)]
    return np.stack([np.asarray(f, np.float32) for f in fl], -1)


def yhat_np(c0, sig, x, beta) -> np.ndarray:
    F = features_np(c0, sig, x)
    b = np.asarray(beta, np.float32)
    out = np.empty(F.shape[:-1], np.float32)
    for ch in range(2):
        out[..., ch] = F[..., ch, :] @ b[ch][:NF] + b[ch][NF]
    return out


def fold_variant_b(beta, beta_sigma, gamma) -> np.ndarray:
    """Variant B (Corin a8f3bb98): half = half_static * exp(gamma * (yhat - yhat_sigma)), yhat over the five features,
    yhat_sigma = beta_sigma[ch] . (logsig, 1). Linear, so it folds into ONE (2, 6) beta for mode "mult"."""
    b = np.asarray(beta, np.float64).reshape(2, NF + 1).copy()
    bs = np.asarray(beta_sigma, np.float64).reshape(2, 2)
    b[:, 0] -= bs[:, 0]
    b[:, NF] -= bs[:, 1]
    return float(gamma) * b


def const_literal(npz_path: str) -> str:
    """The EBAND literal for build_round13.sh's RPDE_SHIP_EBAND from a predictor npz: keys beta (2,6), beta_sigma (2,2), gamma
    -> mode "mult" (variant B); or beta (2,6) alone -> mode "replace". Optional key clip (2,). Floats printed with repr (exact)."""
    z = np.load(npz_path)
    if "beta_sigma" in z.files:
        b, mode = fold_variant_b(z["beta"], z["beta_sigma"], float(z["gamma"])), "mult"
    else:
        b, mode = np.asarray(z["beta"], np.float64).reshape(2, NF + 1), "replace"
    clip = tuple(float(v) for v in z["clip"]) if "clip" in z.files else None
    rows = ", ".join("(" + ", ".join(repr(float(v)) for v in r) + ")" for r in b)
    return f"(({rows}), {mode!r}, {clip!r})"


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--const":
        print(const_literal(sys.argv[2]))
    else:
        raise SystemExit("usage: band_feats.py --const <predictor.npz>")
