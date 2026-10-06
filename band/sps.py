"""The SPS interval score, computed fast and exactly from the official `scoring.py`, and the
exact per-element optimum of the half-width band.

The official aggregate factorises: with a_i the per-SAMPLE accuracy factor
(0.5 / (0.5 + rel) weighted 0.5 / 0.3 / 0.2 over the three error metrics) and the interval term
per ELEMENT,

    w = (1 / n_scored) * sum over scored (i, e) of  a_i * exp(-2 h_e / sigma) * 1[|E_ie| <= h_e].

So scoring a band is one pass over |E|, and the band problem separates over elements: for a
fixed element the objective is a step function of h times a decreasing exponential, so its
maximum sits at one of the observed |E_ie| and sorting gives the exact in-sample optimum.

`scoring.py` next to this file is the organizers' own, from the Track 1 starting kit v9,
unchanged (sha256 a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "official_scoring", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

SIGMA = sc.SIGMA_GLOBAL


def sample_metrics(P, Y):
    """The three per-sample error metrics the accuracy factor is built from."""
    return (sc.rel_l2_per_sample(P, Y, 2), sc.tke_rel_l2_per_sample(P, Y, 2),
            sc.mvpe_rel_l2_per_sample(P, Y))


def weights_from_metrics(dm, tk, mv):
    """a_i, the per-sample accuracy factor: 0.5 g(rel) + 0.3 g(tke) + 0.2 g(mvpe), g(m) = 0.5/(0.5+m)."""
    g = lambda m: 0.5 / (0.5 + m)  # noqa: E731
    a = 0.5 * g(dm) + 0.3 * g(tk) + 0.2 * g(mv)
    return np.where(np.isfinite(a), a, 0.0).astype(np.float64)


def sample_weights(P, Y):
    return weights_from_metrics(*sample_metrics(P, Y))


def prep(P, Y):
    """(|error|, a_i, scored mask, n_scored) on the measured channels [u, v]."""
    p, t = P[..., :2], Y[..., :2]
    return (np.abs(p - t).astype(np.float32), sample_weights(P, Y),
            t != 0.0, int(np.count_nonzero(t != 0.0)))


def sps_score(A, a, scored, n_scored, half):
    """SPS in 0..100 for a half-width that broadcasts over the per-sample axes (1, T, H, W, 2)."""
    e = np.exp(-2.0 * np.asarray(half, dtype=np.float32) / SIGMA)
    contrib = np.where((A <= half) & scored, e, np.float32(0.0))
    per_sample = contrib.reshape(len(A), -1).sum(axis=1, dtype=np.float64)
    return float(100.0 * (a @ per_sample) / n_scored)


def optimal_band(A, a, scored, cap=0.08):
    """Exact per-element in-sample optimum of exp(-2h/sigma) * sum_i a_i 1[|E_ie| <= h]:
    (1, T, H, W, 2) float32. Elements whose target is zero in a window do not count for it."""
    n = len(A)
    flat = A.reshape(n, -1)
    msk = scored.reshape(n, -1)
    order = np.argsort(flat, axis=0, kind="stable")
    srt = np.take_along_axis(flat, order, axis=0)
    wts = np.take_along_axis(a[:, None] * msk, order, axis=0)
    gain = np.cumsum(wts, axis=0) * np.exp(-2.0 * srt.astype(np.float64) / SIGMA)
    best = gain.argmax(axis=0)
    h = np.take_along_axis(srt, best[None], axis=0)[0]
    return np.minimum(h, cap).reshape((1,) + A.shape[1:]).astype(np.float32)
