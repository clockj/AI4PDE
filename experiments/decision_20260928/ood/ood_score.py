"""Per-window decomposition of the kit's scores, so any subset of windows (a trajectory, a regime, the first k windows of
each stream) is scored exactly as the kit would score that subset alone.

Every kit metric is a mean of per-window values (rel, tke, mvpe), and the SPS aggregate is
    sum_i a_i * con_i / sum_i ns_i,      con_i = sum over scored elements of 1[inside] * exp(-(upper - lower) / SIGMA)
with a_i = 0.5 (1 - dm'_i) + 0.3 (1 - tke'_i) + 0.2 (1 - mvpe'_i), m' = m / (0.5 + m). Containment C = 100 sum con / sum ns.
`window_records` computes (rel_i, tke_i, mvpe_i, a_i, con_i, ns_i, ins_i) with the kit's own functions and dtypes;
`selfcheck` compares the pooled result with the kit's `aggregate_sps` on the same arrays.

Composite = final_v9.W0 . (rel, tke, mvpe, sps) -- `final` without its time term (PLAN.md section 1).
"""
from __future__ import annotations

import os
import sys

import numpy as np

WT = os.environ.get("OOD_WT", "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/wt_ood")
sys.path.insert(0, os.path.join(WT, "Track_1"))
os.environ.setdefault("RPDE_KIT", "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9")
import sps9  # noqa: E402  (loads the kit's scoring.py from RPDE_KIT)
import final_v9  # noqa: E402

sc = sps9.sc
W0 = np.asarray(final_v9.W0, dtype=np.float64)          # rel_l2, tke, mvpe, time, sps
assert list(final_v9.NAMES) == ["rel_l2", "tke", "mvpe", "time", "sps"], final_v9.NAMES


def window_records(P, L, U, Y):
    """P, L, U, Y: (N, 20, 32, 64, 3) float32 physical. -> dict of (N,) arrays."""
    c = 2
    rel = sc.rel_l2_per_sample(P, Y, c)
    tke = sc.tke_rel_l2_per_sample(P, Y, c)
    mv = sc.mvpe_rel_l2_per_sample(P, Y)

    def one_minus(m):
        g = 1.0 - m / (0.5 + m)
        return np.where(np.isfinite(g), g, 0.0)
    a = 0.5 * one_minus(rel) + 0.3 * one_minus(tke) + 0.2 * one_minus(mv)
    t = Y[..., :c]
    lo = np.asarray(L, dtype=np.float32)[..., :c]
    up = np.asarray(U, dtype=np.float32)[..., :c]
    inside = (t >= lo) & (t <= up)
    nil = (up - lo) / sc.SIGMA_GLOBAL
    scored = t != 0.0
    hit = inside & scored
    e = np.where(hit, np.exp(-nil), np.float32(0.0))
    n = len(P)
    return dict(rel=rel.astype(np.float64), tke=tke.astype(np.float64), mvpe=mv.astype(np.float64),
                a=a.astype(np.float64),
                con=e.reshape(n, -1).sum(1, dtype=np.float64),
                ns=scored.reshape(n, -1).sum(1).astype(np.int64),
                ins=hit.reshape(n, -1).sum(1).astype(np.int64))


def aggregate(rec, sel=None):
    """Kit scores of the subset `sel` (bool or index array) of windows."""
    g = (lambda k: np.asarray(rec[k])) if sel is None else (lambda k: np.asarray(rec[k])[sel])
    rel_e, tke_e, mv_e = float(np.mean(g("rel"))), float(np.mean(g("tke"))), float(np.mean(g("mvpe")))
    ns = float(np.sum(g("ns")))
    spsv = float(np.sum(g("a") * g("con")) / ns)
    out = dict(n=int(len(g("rel"))), rel_err=rel_e, tke_err=tke_e, mvpe_err=mv_e,
               rel=sc.score_error(rel_e), tke=sc.score_error(tke_e), mvpe=sc.score_error(mv_e),
               sps=sc.score_sps(spsv), C=100.0 * float(np.sum(g("con"))) / ns,
               cov=100.0 * float(np.sum(g("ins"))) / ns, abar=float(np.mean(g("a"))))
    out["composite"] = float(W0[0] * out["rel"] + W0[1] * out["tke"] + W0[2] * out["mvpe"] + W0[4] * out["sps"])
    return out


def selfcheck(P, L, U, Y, rec):
    """|ours - kit| on the pooled set, for sps (0..1 aggregate), coverage and the three errors."""
    w, cov = sc.aggregate_sps(P, Y, 2, lower=L, upper=U)
    ours = aggregate(rec)
    return dict(sps_aggregate=abs(float(np.sum(rec["a"] * rec["con"]) / np.sum(rec["ns"])) - w),
                cov=abs(ours["cov"] / 100.0 - cov),
                rel=abs(ours["rel_err"] - float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))),
                mvpe=abs(ours["mvpe_err"] - sc.mvpe_rel_l2(P, Y)))
