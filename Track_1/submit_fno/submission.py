"""RealPDE Track 1 submission: 0.90M FNO trained with a velocity-gradient matching loss.

This is a plain forecaster. It reads only its own input window, uses no cross-window
information, and would score the same if the evaluator called it one window at a time.

Zip layout (all at the archive ROOT):

    submission.py      <- this file
    _fno.py            <- self-contained FNO2d, mirrors Track_1/unet/models_alt.py
    fno.pth            <- the checkpoint (see README.md for which run)
    band.npy           <- per-element SPS half-widths from fit_band.py

Needs only torch + numpy, both present in the evaluation image
(pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime). It imports no scipy / h5py / matplotlib /
pandas / einops, none of which exist there.

WHAT THE MODEL IS
-----------------
A 4-layer Fourier neural operator, width 24, modes (8, 12) -- 0.90M parameters, 3.6 MB,
0.109 ms/window. It carries a per-pixel log-std head alongside the field head; at inference the
predicted fluctuation is rescaled toward that std (ALPHA below), which trades rel-L2 for TKE at
a rate the scorer likes.

It was trained with an extra loss term matching the TARGET's velocity-gradient tensor. Measured
on 8 held-out trajectories (336 windows), that term is worth +0.74 final over the same network
without it, and it is the only class of change in five rounds that improves rel-L2, TKE and MVPE
at the same time. Ablation: constraining one invariant of grad(u) alone (divergence OR
vorticity) gives 86.53-86.57; constraining both gives 86.67-86.68; adding the strain rates on
top changes nothing.

Local score on those held-out trajectories: 86.68. The hidden test set has historically run
~1.7 points below local, so expect ~85.

WHAT IT IS NOT
--------------
Track 1 tiles its windows at a stride equal to the output horizon and passes them all to
predict() at once, so window k's target is window k+1's input and a submission can read the
answers out of its own argument. This file deliberately does not. See ../submit_trick/ for the
measurement of that leak and the disclosure sent to the organizers.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Put our own directory on sys.path BEFORE importing the sibling module. The platform may load
# this file with importlib.util.spec_from_file_location, which does NOT add the containing
# directory to sys.path, and a bare `import _fno` would then fail at import time.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

import _fno  # noqa: E402  (must follow the sys.path insertion above)

_BAND = os.path.join(_HERE, "band.npz")
_BAND_LEGACY = os.path.join(_HERE, "band.npy")

# fno.pth, then fno_2.pth, fno_3.pth ... as many as were packaged. Members are independent
# seeds of the same recipe, averaged in raw field space inside _fno.predict_fno.
#
# Ensembling looked NEGATIVE for four rounds (-0.07 / -0.10) because latency was accounted as
# forward-pass-only, where a 2nd member nearly doubles the cost. End to end, the forward pass is
# 0.109 ms of 1.383 and the other ~1.27 ms is fixed API overhead that a 2nd member does not
# duplicate -- so a member costs ~+0.11 ms, not a doubling. Measured on val_unseen: 2 members
# +0.10 final, 3 members +0.12.
_CKPTS = [p for p in [os.path.join(_HERE, "fno.pth")]
          + [os.path.join(_HERE, f"fno_{i}.pth") for i in range(2, 17)] if os.path.exists(p)]

# Fluctuation rescaling strength. 0 returns the raw field; 1 forces the output's per-pixel
# temporal std to equal the variance head's prediction. 0.75 is the measured optimum of that
# trade and has been stable across every architecture and round in this project.
ALPHA = 0.75


def _load():
    """Load once at import, so the evaluator's timing of predict() excludes it."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # fuse() collapses same-architecture members into one grouped invocation (5 width-24 seeds
    # + w48 + m1220 + w64 goes from 8 invocations to 4). It is score-identical -- 78.4925
    # against 78.4927 on the AoA 15-20 proxy, and exact to 1.5e-8 with TF32 disabled -- so the
    # only thing it changes is latency.
    #
    # Which way it changes latency depends on a model of board time that three data points do
    # not settle. Regressing the three v9 rounds (U-Net x1 2.648 local / 5.976 board; FNO x3
    # 1.282 / 5.931; FNO x8 2.211 / 8.020):
    #     board ~ invocations : slope +0.316 ms, residuals +-0.4 ms
    #     board ~ local ms    : slope +0.380,    residuals +-1.3 ms
    # The invocation model fits three times better. Under it, 8 -> 4 invocations saves 1.26
    # board ms and is worth about +0.05 final. Under the local-time model, fusing costs
    # +0.475 ms/window locally and is worth about -0.007. Favourable but not free, and the
    # real reason to take it is that it makes ADDITIONAL same-architecture members cost zero
    # invocations.
    models = _fno.fuse([_fno.load(p, dev) for p in _CKPTS])
    # band.npz carries `base` (per-element half-widths), `sigma_ref` (the per-element mean of
    # the variance head's sigma over the fitting split) and `p`. The half-width actually used
    # is base * (sigma_i / sigma_ref)**p, i.e. a per-WINDOW interval: windows the model itself
    # expects to be turbulent get a wider band. Falls back to the legacy band.npy, where p = 0
    # reproduces the old fixed band exactly.
    band = ref = None
    p = 0.0
    if os.path.exists(_BAND):
        z = np.load(_BAND)
        band, ref, p = (z["base"].astype(np.float32), z["sigma_ref"].astype(np.float32),
                        float(z["p"]))
    elif os.path.exists(_BAND_LEGACY):
        band = np.load(_BAND_LEGACY).astype(np.float32)
    # The ingestion program rejects a non-finite or reversed interval by zeroing EVERY
    # subscore, so the band is checked -- but here, at import, where it is 0.5 MB and free,
    # not inside the timed call where the same guarantee would cost a scan of two 165 MB
    # arrays. Nothing downstream can break it: subtracting a non-negative half-width from a
    # finite prediction cannot reorder the bounds, and the sigma scaling is clipped positive.
    if band is not None and not (np.all(np.isfinite(band)) and np.all(band >= 0.0)):
        band = None
    if band is not None and p and not (ref is not None and np.all(ref > 0.0)):
        p = 0.0
    return models, dev, band, ref, p


try:
    _state = _load()
except Exception:                 # fall back to a lazy load inside predict()
    _state = None


def predict(input_array, metadata=None):
    """(N, 20, 32, 64, 3) raw physical -> dict with prediction, lower, upper."""
    global _state
    if _state is None:
        _state = _load()
    models, dev, band, ref, p = _state

    x = np.ascontiguousarray(input_array, dtype=np.float32)
    # Allocate the output once, zero-filled, and let the model write u,v into it. The pressure
    # channel is unmeasured and unscored, so zeros are the answer and never need touching
    # again. Concatenating it on afterwards cost a full-size copy inside a timed call.
    pred = np.zeros((len(x), 20, 32, 64, 3), dtype=np.float32)
    # (N, 1, 32, 64, 2), 5.5 MB at N = 336 against the field's 165 MB -- cheap enough to ask
    # for inside a timed call, and only requested when the band actually uses it.
    sig = np.empty((len(x), 1, 32, 64, 2), dtype=np.float32) if p else None
    _fno.predict_fno(models, x, dev, alpha=ALPHA, out=pred, sigma_out=sig)

    # `half` is (1, 20, 32, 64, 3) and broadcasts against pred natively, so it is NEVER
    # materialized to full size -- pred - half allocates the result and nothing else. Doing
    # this the obvious way (broadcast_to(...).copy(), then ascontiguousarray on each bound)
    # allocated three extra full-size arrays, ~500 MB at N=336, and predict() is TIMED.
    # A test horizon we did not anticipate falls back to the fitted constant rather than
    # crashing on the broadcast.
    half = band if (band is not None and band.shape[1:] == pred.shape[1:]) else np.float32(0.0065)
    if p and sig is not None and band is not None and half is band:
        # Per-window widths: base * (sigma / sigma_ref)**p, clipped to the range the fit was
        # solved over. This DOES materialize `half` at full size, which the fixed band avoided
        # -- one extra 165 MB array at N = 336. It is worth it: the sigma scaling is +0.51 sps
        # on the board-like proxy and +1.64 on the full split, both in AND out of sample,
        # where the round-6 attempt using ensemble spread was noise.
        # Order-independent: sigma_ref is a stored constant, so a window's band depends only
        # on that window.
        r = np.clip(sig / ref[..., :2], 0.3, 3.0) ** np.float32(p)
        half = band[..., :2] * r                       # (N, 20, 32, 64, 2) by broadcasting
        half = np.concatenate([half, np.zeros_like(half[..., :1])], -1)

    # No finiteness scan here. It cost 0.066 ms/window -- 5% of the whole timed call -- to
    # duplicate a check the scorer already performs on `prediction` before scoring anything,
    # and ours was strictly worse: the scorer degrades a non-finite prediction to zero scores,
    # while an assert would raise out of predict() and fail the run outright. The band's
    # finiteness and sign are established once at import instead.
    lower = (pred - half).astype(np.float32, copy=False)
    upper = (pred + half).astype(np.float32, copy=False)
    return {"prediction": pred, "lower": lower, "upper": upper}
