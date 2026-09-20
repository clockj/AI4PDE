"""Price every BACKPROP-FREE Track-2 adaptation rule, offline, from a saved stream.

Track 2 reveals window k-1's ground truth at step k.  Because the windows tile at stride 20,
that revealed target IS the current input, so it says nothing new about the current window --
what it gives you is the previous step's REALIZED ERROR, computed against a prediction you
already made and can cache.  Every rule here uses only that, so it costs no forward pass, no
backward pass and no extra invocation: the arithmetic is a handful of elementwise ops on a
(20,32,64,2) array.

That makes this the right thing to measure FIRST.  A gradient-based adapter has to beat these
by enough to pay for its own latency; these have to beat nothing.

    python ttt_offline.py --npz <stream_eval --save output>

The stream must be saved in evaluator order with `ids` of the form ``name:time_id`` so
trajectory boundaries (where a real run calls ``reset_ttt_state``) can be reconstructed.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_2/.  Putting the PARENT of closed/ on the
# path keeps `import t2_common / sps9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_2/, so every submit/ and ../Track_1/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Track_1"))
KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)
import final_v9  # noqa: E402


def subs(P, Y, L, U, t_ms):
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
    m = sc.mvpe_rel_l2(P, Y)
    s = sc.aggregate_sps(P, Y, 2, lower=L, upper=U)[0] if L is not None else \
        sc.aggregate_sps(P, Y, 2)[0]
    five = [sc.score_error(r), sc.score_error(k), sc.score_error(m),
            sc.score_time(t_ms / 1e3), sc.score_sps(s)]
    return dict(rel_err=r, tke_err=k, mvpe_err=m, five=five,
                wfinal=final_v9.predict(five)[0])


def apply_rule(P, Y, groups, mode, eta, clip=None):
    """Return a corrected copy of P.  `groups` lists index arrays, one per trajectory.

    Causal by construction: window k is corrected only from residuals of windows < k.
    """
    Q = P.copy()
    for g in groups:
        acc = None
        for j, i in enumerate(g):
            if acc is not None:
                if mode == "full":
                    Q[i, ..., :2] = P[i, ..., :2] + acc
                elif mode == "mean":
                    Q[i, ..., :2] = P[i, ..., :2] + acc.mean(0, keepdims=True)
                elif mode == "scalar":
                    Q[i, ..., :2] = P[i, ..., :2] + acc.mean(axis=(0, 1, 2), keepdims=True)
            r = Y[i, ..., :2] - P[i, ..., :2]          # residual of the step just scored
            if clip is not None:
                s = np.abs(r).mean()
                r = np.clip(r, -clip * s, clip * s)
            acc = r if acc is None else (1 - eta) * acc + eta * r
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ms", type=float, default=2.0, help="assumed board-side ms/step; the "
                    "rules here add no measurable time, so it only shifts the table")
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    P, Y = z["pred"].astype(np.float32), z["target"].astype(np.float32)
    L = z["lower"] if z["lower"].size else None
    U = z["upper"] if z["upper"].size else None
    ids = [str(s) for s in z["ids"]]
    names = [s.split(":")[0] for s in ids]
    groups, cur = [], []
    for i, n in enumerate(names):
        if cur and n != names[cur[0]]:
            groups.append(np.array(cur)); cur = []
        cur.append(i)
    groups.append(np.array(cur))
    print(f"{len(P)} windows, {len(groups)} trajectories")

    base = subs(P, Y, L, U, a.ms)
    print(f"\n{'rule':28s} {'rel':>7s} {'tke':>7s} {'mvpe':>7s} | {'wfinal':>7s} {'d':>7s}")
    print(f"{'frozen (baseline)':28s} {base['rel_err']:7.4f} {base['tke_err']:7.4f} "
          f"{base['mvpe_err']:7.4f} | {base['wfinal']:7.3f} {0.0:7.3f}")

    # ---- 1. how much of the error is even reusable? --------------------------------
    # Correlation between consecutive windows' residual fields, per trajectory.
    cs, cs_mean = [], []
    for g in groups:
        R = (Y[g][..., :2] - P[g][..., :2]).reshape(len(g), -1)
        M = (Y[g][..., :2] - P[g][..., :2]).mean(1).reshape(len(g), -1)
        for A, out in ((R, cs), (M, cs_mean)):
            A = A - A.mean(1, keepdims=True)
            n = np.linalg.norm(A, axis=1) + 1e-12
            out.append(float(np.mean(np.sum(A[:-1] * A[1:], 1) / (n[:-1] * n[1:]))))
    print(f"\nlag-1 residual correlation:  full field {np.mean(cs):+.3f}   "
          f"time-mean field {np.mean(cs_mean):+.3f}")

    # ---- 2. the rules ---------------------------------------------------------------
    for mode in ("full", "mean", "scalar"):
        for eta in (1.0, 0.5, 0.3, 0.15):
            Q = apply_rule(P, Y, groups, mode, eta)
            d = subs(Q, Y, None if L is None else L + (Q - P), None if U is None else U + (Q - P),
                     a.ms)
            print(f"{'resid-EMA ' + mode + f' eta={eta}':28s} {d['rel_err']:7.4f} "
                  f"{d['tke_err']:7.4f} {d['mvpe_err']:7.4f} | {d['wfinal']:7.3f} "
                  f"{d['wfinal'] - base['wfinal']:+7.3f}")

    # ---- 3. an ORACLE ceiling: subtract the trajectory's own mean residual ----------
    Q = P.copy()
    for g in groups:
        Q[g][..., :2] = P[g][..., :2] + (Y[g][..., :2] - P[g][..., :2]).mean((0, 1), keepdims=True)
    for g in groups:
        Q[g, ..., :2] = P[g, ..., :2] + (Y[g, ..., :2] - P[g, ..., :2]).mean((0, 1), keepdims=True)
    d = subs(Q, Y, None, None, a.ms)
    print(f"\n{'ORACLE traj-mean residual':28s} {d['rel_err']:7.4f} {d['tke_err']:7.4f} "
          f"{d['mvpe_err']:7.4f} | (sps on default band, not comparable)")


if __name__ == "__main__":
    main()
