"""Can a per-window linear model extrapolate the wake, with no training data at all?

Every predictor in this project learns the map from 74 trajectories, and CLAUDE.md's standing
diagnosis is that those 74 are the binding constraint (seen rel-L2 0.049 against unseen 0.084).
A per-window fit sidesteps that entirely: dynamic mode decomposition reads the 20 input frames,
identifies the oscillation frequencies and growth rates actually present in THIS window, and
extrapolates them 20 frames forward. It has no parameters shared across conditions, so it
cannot fail to generalize to an unseen Reynolds number -- which is the exact failure mode the
whole robustness line is chasing.

Round 4 closed "analog forecasting" (Lorenz's method of analogues, rel-L2 0.259). That is a
different animal: analogues are nonparametric and need a densely sampled attractor, and 1000
frames is far too few. DMD fits 19 snapshot pairs of THIS window and needs no library at all.

Reference points on the same 336 windows, from CLAUDE.md:
    persistence (repeat the last frame)   0.138
    input time-mean                       0.134
    our 8-member ensemble                 ~0.104 (AoA 15-20 proxy 0.112 board)

Two questions, and the second is the one that matters:
    1. how good is DMD alone?
    2. is its error DECORRELATED from the network's?  A feature is worth adding to the input
       only if it carries something the network does not already have.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def dmd_forecast(win: np.ndarray, horizon: int = 20, rank: int = 12,
                 clip: float = 1.0) -> np.ndarray:
    """Exact DMD on one (T, H, W, C) window -> (horizon, H, W, C) continuation.

    `clip` bounds |lambda|: an eigenvalue above 1 grows like lambda^20 over the horizon and a
    19-snapshot fit produces those routinely from noise. Clipping to the unit disc keeps the
    oscillation and drops the divergence, which is the standard fix for short-record DMD.
    """
    t, h, w, c = win.shape
    m = win.reshape(t, -1).T.astype(np.float64)          # (n, T)
    mu = m.mean(1, keepdims=True)
    m = m - mu                                            # DMD on the fluctuation
    X, Y = m[:, :-1], m[:, 1:]

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, int((S > S[0] * 1e-10).sum()))
    U, S, Vt = U[:, :r], S[:r], Vt[:r]
    A = U.T @ Y @ Vt.T @ np.diag(1.0 / S)                 # (r, r) projected operator
    lam, W = np.linalg.eig(A)
    a = np.abs(lam)
    lam = np.where(a > clip, lam / np.maximum(a, 1e-12) * clip, lam)
    Phi = Y @ Vt.T @ np.diag(1.0 / S) @ W                 # exact DMD modes
    b = np.linalg.lstsq(Phi, m[:, -1], rcond=None)[0]     # amplitudes at the LAST frame

    k = np.arange(1, horizon + 1)[None, :]
    out = np.real(Phi @ (b[:, None] * lam[:, None] ** k)) + mu
    return out.T.reshape(horizon, h, w, c).astype(np.float32)


def rel_per_win(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = p[..., :2].reshape(len(p), -1).astype(np.float64)
    b = y[..., :2].reshape(len(y), -1).astype(np.float64)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val_unseen")
    ap.add_argument("--rank", type=int, nargs="+", default=[4, 6, 8, 12, 19])
    ap.add_argument("--clip", type=float, nargs="+", default=[1.0])
    ap.add_argument("--aoa", type=int, nargs="+", default=None)
    ap.add_argument("--npz", default=None, help="a saved (N,20,32,64,3) model prediction, to "
                                                "measure whether DMD's error is decorrelated")
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, f"{args.set}.npz"))
    x, y, ids = d["x"], d["y"], d["ids"]
    if args.aoa:
        s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ids], args.aoa)
        x, y, ids = x[s], y[s], ids[s]
    print(f"{args.set}: {len(x)} windows"
          + (f", AoA {args.aoa}" if args.aoa else "") + "\n")

    ref = {"persistence": np.repeat(x[:, -1:], 20, 1),
           "input time-mean": np.repeat(x.mean(1, keepdims=True), 20, 1)}
    best, best_e = None, 1e9
    for name, p in ref.items():
        e = rel_per_win(p, y)
        print(f"  {name:24s} rel-L2 {e.mean():.4f}   (p90 {np.percentile(e,90):.4f})")
    print()
    for cl in args.clip:
        for r in args.rank:
            p = np.stack([dmd_forecast(w, 20, r, cl) for w in x])
            e = rel_per_win(p, y)
            tag = f"DMD rank {r}, clip {cl}"
            print(f"  {tag:24s} rel-L2 {e.mean():.4f}   (p90 {np.percentile(e,90):.4f})")
            if e.mean() < best_e:
                best, best_e = p, e.mean()

    if args.npz and os.path.exists(args.npz):
        P = np.load(args.npz)["p"]
        if args.aoa:
            P = P[s]
        em, ed = rel_per_win(P, y), rel_per_win(best, y)
        print(f"\n  model {em.mean():.4f}   best DMD {ed.mean():.4f}")
        print(f"  per-window error correlation  {np.corrcoef(em, ed)[0,1]:+.3f}")
        # the question that decides whether DMD is worth feeding in: does an ORACLE convex
        # blend of the two beat the model? If the errors were identical it could not.
        for a in (0.1, 0.2, 0.3, 0.5):
            print(f"  blend {1-a:.1f}*model + {a:.1f}*DMD   "
                  f"rel-L2 {rel_per_win((1-a)*P + a*best, y).mean():.4f}")
        # and the per-element view SPS actually scores
        dm = np.abs(P[..., :2] - y[..., :2]); dd = np.abs(best[..., :2] - y[..., :2])
        print(f"  per-ELEMENT |error| correlation {np.corrcoef(dm.ravel(), dd.ravel())[0,1]:+.3f}")


if __name__ == "__main__":
    main()


def context_sweep(data="/users/PAS1187/qwu/RealPDE/train_real/train_real",
                  names=None, ctxs=(20, 40, 60, 80, 120, 180), rank=12, n_win=8):
    """Is 20 frames the binding constraint, or is DMD simply the wrong tool?

    The protocol window is 20 frames and the shedding period is 60-75, so a per-window fit
    sees less than a third of a cycle -- too little to identify a frequency at all. Feeding
    DMD MORE history from the raw trajectories separates the two explanations: if the error
    falls sharply with context, the 20-frame window is an information limit that no model can
    beat; if it stays flat, DMD is just wrong for this flow.

    The target is always the SAME 20 frames, so the rows are directly comparable.
    """
    import glob
    import h5py
    if names is None:
        names = ["3750_0", "6300_10", "13950_10", "16500_5", "21600_15", "26700_0"]
    print(f"\ncontext sweep: {len(names)} trajectories x {n_win} windows, DMD rank {rank}")
    print(f"{'context':>8s}" + "".join(f"{c:>10d}" for c in ctxs))
    per, ref_p, ref_m = {c: [] for c in ctxs}, [], []
    for nm in names:
        p = os.path.join(data, nm + ".h5")
        with h5py.File(p, "r") as f:
            d = np.stack([f["u"][:, ::2, ::2], f["v"][:, ::2, ::2]], -1).astype(np.float32)
        starts = np.linspace(max(ctxs), len(d) - 20 - 1, n_win).astype(int)
        for t0 in starts:
            y = d[t0:t0 + 20]
            ref_p.append(rel_per_win(np.repeat(d[t0 - 1:t0], 20, 0)[None], y[None])[0])
            ref_m.append(rel_per_win(np.repeat(d[t0 - 20:t0].mean(0, keepdims=True), 20, 0)[None],
                                     y[None])[0])
            for c in ctxs:
                f_ = dmd_forecast(d[t0 - c:t0], 20, rank)
                per[c].append(rel_per_win(f_[None], y[None])[0])
    print(f"{'DMD':>8s}" + "".join(f"{np.mean(per[c]):10.4f}" for c in ctxs))
    print(f"  persistence {np.mean(ref_p):.4f}   input 20-frame mean {np.mean(ref_m):.4f}")
    return per
