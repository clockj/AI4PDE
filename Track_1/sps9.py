"""Exact, fast SPS under v9 scoring, plus the exact optimal half-width band.

v9's aggregate factorizes.  Writing pm for a sample's normalized error metric,

    w = (1/n_scored) * sum over scored (i, e) of  a_i * exp(-2*h_e/sigma) * 1[|E_ie| <= h_e]
    a_i = 0.5*(1 - pm_dm_i) + 0.3*(1 - pm_tke_i) + 0.2*(1 - pm_mvpe_i)

because the three branches differ only in a per-SAMPLE factor while the interval term is
per-ELEMENT.  Two things follow:

1.  Scoring a candidate band is one pass over |E| instead of three, and `a` is computed once.
    That is what makes a real search over bands affordable.

2.  The band problem separates over elements.  For a fixed element e, the objective
    exp(-2h/sigma) * sum_i a_i*1[|E_ie| <= h] is a step function times a decreasing
    exponential, so its maximum sits exactly at one of the observed |E_ie|.  Sorting gives the
    exact in-sample optimum -- no Gaussian assumption, no grid.  That optimum is an upper bound
    on what any band-fitting procedure can do, so it says how much of the remaining SPS gap is
    reachable by better intervals and how much needs a better model.

`w` is verified against the kit's own aggregate_sps in self_check().
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

KIT9 = os.environ.get("RPDE_KIT", "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9")   # dir holding the v9 kit's scoring.py; default = the Pitzer mount (REPRODUCE §9.1)
_spec = importlib.util.spec_from_file_location("sc9", os.path.join(KIT9, "scoring.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

SIGMA = sc.SIGMA_GLOBAL


def sample_metrics(P, Y):
    """The three per-sample error metrics SPS's accuracy factor is built from."""
    return (sc.rel_l2_per_sample(P, Y, 2), sc.tke_rel_l2_per_sample(P, Y, 2),
            sc.mvpe_rel_l2_per_sample(P, Y))


def weights_from_metrics(dm, tk, mv):
    """a_i: the per-sample accuracy factor the three SPS branches share, weighted 0.5/0.3/0.2.

    1 - m/(0.5+m) is 0.5/(0.5+m), which is where the metrics enter; it is strongly concave, so
    a metric that doubles does NOT halve its term and the factor must be recomputed from the
    metric rather than rescaled.
    """
    g = lambda m: 0.5 / (0.5 + m)  # noqa: E731
    a = 0.5 * g(dm) + 0.3 * g(tk) + 0.2 * g(mv)
    return np.where(np.isfinite(a), a, 0.0).astype(np.float64)


def sample_weights(P, Y):
    return weights_from_metrics(*sample_metrics(P, Y))


def prep(P, Y):
    """(|error|, a_i, scored mask, n_scored) on the measured channels only."""
    p, t = P[..., :2], Y[..., :2]
    return (np.abs(p - t).astype(np.float32), sample_weights(P, Y),
            t != 0.0, int(np.count_nonzero(t != 0.0)))


def sps_score(A, a, scored, n_scored, half):
    """SPS in 0..100. `half` is a scalar or broadcasts over the per-sample axes (1, T, H, W, 2)."""
    e = np.exp(-2.0 * np.asarray(half, dtype=np.float32) / SIGMA)
    contrib = np.where((A <= half) & scored, e, np.float32(0.0))
    per_sample = contrib.reshape(len(A), -1).sum(axis=1, dtype=np.float64)
    return float(100.0 * (a @ per_sample) / n_scored)


def optimal_band(A, a, scored, cap=0.08):
    """Exact per-element in-sample optimum of exp(-2h/sigma) * sum_i a_i*1[|E_ie| <= h].

    Sorting |E| along the sample axis turns the inner sum into a cumulative sum of the
    corresponding a_i, so the whole field is solved in one vectorized pass.  Elements whose
    target is 0 for a given window are excluded from that window's contribution, which is why
    `a` is broadcast through `scored` rather than used directly.
    """
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


def self_check(P, Y, half):
    """The fast path must agree with the kit's own aggregate_sps."""
    A, a, scored, n = prep(P, Y)
    fast = sps_score(A, a, scored, n, half)
    lo, hi = P - half, P + half
    ref = sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=lo, upper=hi)[0])
    return fast, ref
