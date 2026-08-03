"""Analog forecasting: is the wake periodic enough to predict by matching its own past?

    python diag_analog.py --suffix _sim

The TKE floor measured in report_20260731.md bounds predictors of the *true variance field*.
A predictor that gets the actual realization right scores TKE -> 0, so the floor is not a bound
on 92 -- phase prediction is. This tests the oldest method for that (Lorenz 1969, "Atmospheric
predictability as revealed by naturally occurring analogues"): find the moment in the system's
own past that most resembles now, and copy what happened next.

The setting is unusually favourable. The flow is quasi-periodic vortex shedding; a sim
trajectory is 1000 noise-free frames, i.e. tens of shedding cycles; and Track 1's predict()
receives every window of a trajectory at once, so for window k the frames of windows 0..k are
INPUTS. Searching them reads no target and is ordinary transductive inference.

Matching is done on POD coefficients rather than raw fields: the wake is low-rank, so a handful
of modes carry the phase, and the search drops from O(frames * 4096) to O(frames * r).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T_IN = T_OUT = 20


def trajectory_frames(X: np.ndarray, ids: np.ndarray) -> dict[str, np.ndarray]:
    """Windows tile each trajectory at stride 20, so their inputs concatenate to the frames."""
    traj = np.array([str(i).split(":")[0] for i in ids])
    out = {}
    for name in np.unique(traj):
        w = np.flatnonzero(traj == name)
        order = np.argsort([int(str(ids[k]).split(":")[1]) for k in w])
        out[name] = np.concatenate([X[w[o]][..., :2] for o in order], axis=0)
    return out


def pod_coeffs(frames: np.ndarray, r: int) -> np.ndarray:
    """(T,H,W,2) -> (T,r) coefficients of the temporal-mean-removed field."""
    F = frames.reshape(len(frames), -1).astype(np.float64)
    F = F - F.mean(0, keepdims=True)
    # Economy SVD on the (T x T) Gram matrix: T ~ 1000 << 4096 columns.
    G = F @ F.T
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:r]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0))[None, :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_sim")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--blend", type=int, default=4, help="average the k best analogues")
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, f"val_unseen{args.suffix}.npz"))
    X, Y, ids = d["x"], d["y"], d["ids"]
    traj = np.array([str(i).split(":")[0] for i in ids])
    frames = trajectory_frames(X, ids)

    # --- periodicity: autocorrelation of the leading POD coefficient
    print(f"{'='*78}\nquasi-periodicity of the leading POD mode\n{'='*78}")
    print(f"  {'trajectory':14s} {'shedding period':>16s} {'autocorr at that lag':>22s} "
          f"{'autocorr at lag 20':>20s}")
    periods = {}
    for name, F in frames.items():
        a = pod_coeffs(F, 2)[:, 0]
        a = (a - a.mean()) / a.std()
        ac = np.array([np.mean(a[:len(a) - L] * a[L:]) for L in range(1, 121)])
        # first local maximum after the initial decay = shedding period
        pk = next((L for L in range(2, 119) if ac[L] > ac[L - 1] and ac[L] >= ac[L + 1]
                   and ac[L] > 0.2), None)
        periods[name] = pk
        p_txt = f"{pk+1} frames" if pk else "none found"
        print(f"  {name:14s} {p_txt:>16s} {ac[pk] if pk else float('nan'):22.3f} "
              f"{ac[19]:20.3f}")

    # --- analog forecast, strictly causal
    print(f"\n{'='*78}\nanalog forecast (rank {args.rank}, blending {args.blend} best)\n{'='*78}")
    P = np.zeros_like(Y)
    used = np.zeros(len(X), dtype=int)
    for name in frames:
        F = frames[name]
        C = pod_coeffs(F, args.rank)
        w = np.flatnonzero(traj == name)
        starts = np.array([int(str(ids[k]).split(":")[1]) for k in w])
        for k, t in zip(w, starts):
            # candidate analogue starts s: the whole segment AND its continuation must lie in
            # frames strictly before this window's target, i.e. s + 40 <= t + 20.
            hi = t - T_OUT
            if hi < 1:
                P[k] = np.concatenate([X[k], np.zeros_like(X[k][..., :1])], -1)[..., :3]
                continue
            cand = np.arange(0, hi)
            q = C[t:t + T_IN]                                   # (20, r)
            seg = np.stack([C[s:s + T_IN] for s in cand])       # (nc, 20, r)
            dist = ((seg - q[None]) ** 2).sum((1, 2))
            best = cand[np.argsort(dist)[:args.blend]]
            pred = np.mean([F[s + T_IN:s + T_IN + T_OUT] for s in best], axis=0)
            P[k] = np.concatenate([pred, np.zeros_like(pred[..., :1])], -1)
            used[k] = len(cand)

    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "sc", "/home/clockj/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7/"
              "scoring.py")
    sc = importlib.util.module_from_spec(spec)
    sys.modules["sc"] = sc
    spec.loader.exec_module(sc)

    ok = used > 0
    for label, m in (("all windows", np.ones(len(X), bool)), ("windows with history", ok)):
        r = float(np.mean(sc.rel_l2_per_sample(P[m], Y[m], 2)))
        k_ = float(np.mean(sc.tke_rel_l2_per_sample(P[m], Y[m], 2)))
        mv = sc.mvpe_rel_l2(P[m], Y[m])
        print(f"  {label:22s} (n={m.sum():3d})  rel-L2 {r:.4f}  TKE {k_:.4f}  MVPE {mv:.4f}"
              f"   | {sc.score_error(r):5.1f} {sc.score_error(k_):5.1f} {sc.score_error(mv):5.1f}")
    print(f"  candidate analogues available: {used[ok].min()} .. {used.max()}")


if __name__ == "__main__":
    main()
