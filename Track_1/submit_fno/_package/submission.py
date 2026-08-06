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

_BAND = os.path.join(_HERE, "band.npy")

# fno.pth, then fno_2.pth, fno_3.pth ... as many as were packaged. Members are independent
# seeds of the same recipe, averaged in raw field space inside _fno.predict_fno.
#
# Ensembling looked NEGATIVE for four rounds (-0.07 / -0.10) because latency was accounted as
# forward-pass-only, where a 2nd member nearly doubles the cost. End to end, the forward pass is
# 0.109 ms of 1.383 and the other ~1.27 ms is fixed API overhead that a 2nd member does not
# duplicate -- so a member costs ~+0.11 ms, not a doubling. Measured on val_unseen: 2 members
# +0.10 final, 3 members +0.12.
_CKPTS = [p for p in [os.path.join(_HERE, "fno.pth")]
          + [os.path.join(_HERE, f"fno_{i}.pth") for i in range(2, 9)] if os.path.exists(p)]

# Fluctuation rescaling strength. 0 returns the raw field; 1 forces the output's per-pixel
# temporal std to equal the variance head's prediction. 0.75 is the measured optimum of that
# trade and has been stable across every architecture and round in this project.
ALPHA = 0.75


def _load():
    """Load once at import, so the evaluator's timing of predict() excludes it."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(p, dev) for p in _CKPTS]
    band = np.load(_BAND).astype(np.float32) if os.path.exists(_BAND) else None
    # The ingestion program rejects a non-finite or reversed interval by zeroing EVERY
    # subscore, so the band is checked -- but here, at import, where it is 0.5 MB and free,
    # not inside the timed call where the same guarantee would cost a scan of two 165 MB
    # arrays. Nothing downstream can break it: subtracting a non-negative half-width from a
    # finite prediction cannot reorder the bounds.
    if band is not None and not (np.all(np.isfinite(band)) and np.all(band >= 0.0)):
        band = None
    return models, dev, band


try:
    _state = _load()
except Exception:                 # fall back to a lazy load inside predict()
    _state = None


def predict(input_array, metadata=None):
    """(N, 20, 32, 64, 3) raw physical -> dict with prediction, lower, upper."""
    global _state
    if _state is None:
        _state = _load()
    models, dev, band = _state

    x = np.ascontiguousarray(input_array, dtype=np.float32)
    # Allocate the output once, zero-filled, and let the model write u,v into it. The pressure
    # channel is unmeasured and unscored, so zeros are the answer and never need touching
    # again. Concatenating it on afterwards cost a full-size copy inside a timed call.
    pred = np.zeros((len(x), 20, 32, 64, 3), dtype=np.float32)
    _fno.predict_fno(models, x, dev, alpha=ALPHA, out=pred)

    # `half` is (1, 20, 32, 64, 3) and broadcasts against pred natively, so it is NEVER
    # materialized to full size -- pred - half allocates the result and nothing else. Doing
    # this the obvious way (broadcast_to(...).copy(), then ascontiguousarray on each bound)
    # allocated three extra full-size arrays, ~500 MB at N=336, and predict() is TIMED.
    # A test horizon we did not anticipate falls back to the fitted constant rather than
    # crashing on the broadcast.
    half = band if (band is not None and band.shape[1:] == pred.shape[1:]) else np.float32(0.0065)

    # No finiteness scan here. It cost 0.066 ms/window -- 5% of the whole timed call -- to
    # duplicate a check the scorer already performs on `prediction` before scoring anything,
    # and ours was strictly worse: the scorer degrades a non-finite prediction to zero scores,
    # while an assert would raise out of predict() and fail the run outright. The band's
    # finiteness and sign are established once at import instead.
    lower = (pred - half).astype(np.float32, copy=False)
    upper = (pred + half).astype(np.float32, copy=False)
    return {"prediction": pred, "lower": lower, "upper": upper}
