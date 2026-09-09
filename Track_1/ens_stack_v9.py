"""Two post-hoc levers on an ALREADY-TRAINED heterogeneous ensemble. No training, no latency.

Both exist because the cross-family probe (r20d/r21b) changed the ensemble from "K seeds of one
recipe" to "members of genuinely different accuracy", and two things that were correct for the
first are not obviously correct for the second.

P1  MEMBER WEIGHTS.  `_fno.predict_fno` averages members with weight `n_members` -- i.e.
    uniformly, up to fusion bookkeeping. For exchangeable seeds uniform IS optimal and there was
    never anything to fit. A ViT member and three FNO seeds are not exchangeable: they differ in
    accuracy AND in error correlation, and the optimal weight of a member that is individually
    worse but less correlated is not 1/K. The weight enters exactly where `nm` does, so anything
    this finds ships for free.

P2  THE BAND FROM CROSS-FAMILY SPREAD.  The half-width is `base * (sigma_head/sigma_ref)^p`.
    Round 6 tested ensemble spread as a band signal and rejected it: spread correlated 0.347
    with per-element |error| after the per-element mean was removed, against sigma_head's 0.551.
    That was measured on THREE WIDTH-24 SEEDS OF ONE RECIPE, where spread is seed noise. Across
    families spread measures disagreement between different inductive biases, which is a
    different quantity. Tested as a SECOND exponent, `base * r_sigma^p * r_spread^q`, so q = 0
    recovers the incumbent exactly and the question is whether q != 0 survives grouped CV.

Method note that round 10b paid for: when p or q changes, `base` MUST be refitted. `optimal_band`
on the DEFLATED errors |E| / (r_sigma^p r_spread^q) is exactly that refit, because
|E| <= base * r^k is |E|/r^k <= base. Fitting `base` once at q = 0 makes every q != 0 lose by
construction -- which is how the band was wrongly declared closed in round 6.

    python ens_stack_v9.py --ckpt <held-out twins...> --alpha 0.60 --aoa 0 5 10 15 20
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

import final_v9
import sps9
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
F_BOARD = 1.39


def compose(y: np.ndarray, sig: np.ndarray, alpha: float) -> np.ndarray:
    """`predict_fno`'s composition, in physical units.

    Equivalent to composing in normalized space and denormalizing: the ratio is invariant under
    a shared affine, so only the 1e-6 guard differs (by a factor `std` ~ 0.097, i.e. it moves
    the guard from 1e-6 to 1e-7 of the same quantity -- far below any std that reaches it).
    """
    m = y.mean(1, keepdims=True)
    s = y.std(1, keepdims=True)
    return m + (y - m) * ((alpha * sig + (1.0 - alpha) * s) / (s + 1e-6))


def simplex_lsq(A: np.ndarray, b: np.ndarray, iters: int = 400) -> np.ndarray:
    """min ||A w - b||^2 over the simplex, by projected gradient. A is (n, K), tall.

    The simplex constraint is not decoration. The composition divides by the ensemble's own
    temporal std, so a set of weights that does not sum to 1 rescales the fluctuation before
    `alpha` ever sees it, and the comparison against uniform stops being a comparison of
    ensembles. Non-negativity keeps a member from being used as a subtractive corrector fitted
    to this particular split.
    """
    G = A.T @ A
    c = A.T @ b
    K = A.shape[1]
    w = np.full(K, 1.0 / K)
    L = float(np.linalg.eigvalsh(G).max()) + 1e-12
    for _ in range(iters):
        w = w - (G @ w - c) / L
        # Euclidean projection onto the probability simplex (Duchi et al.).
        u = np.sort(w)[::-1]
        css = np.cumsum(u) - 1.0
        rho = np.nonzero(u - css / np.arange(1, K + 1) > 0)[0][-1]
        w = np.maximum(w - css[rho] / (rho + 1.0), 0.0)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="HELD-OUT twins. Shipped all-data members memorized val_unseen and "
                         "their error field is not honest, so weights fitted on them would be "
                         "fitted to memorization.")
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--aoa", type=int, nargs="+", default=[0, 5, 10, 15, 20])
    ap.add_argument("--f", type=float, default=F_BOARD)
    ap.add_argument("--time-score", type=float, default=93.2701)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    import torch
    import _fno

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = np.float32(args.f)
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    ids = [str(i) for i in du["ids"]]
    keep = np.isin([int(i.split(":")[0].split("_")[1]) for i in ids], args.aoa)
    X, Y = du["x"][keep], du["y"][keep]
    traj = np.array([i.split(":")[0] for i in ids])[keep]
    ut = sorted(set(traj.tolist()))
    print(f"AoA {args.aoa}: {len(X)} windows over {len(ut)} trajectories, "
          f"errors stretched {args.f}x to board magnitude")

    # ---- per-member RAW field and sigma, both in physical units -----------------------------
    # alpha = 0 skips the composition inside predict_fno, so `F` is the member's own field and
    # `S` its own sigma; the ensemble is then formed here, exactly where `nm` forms it there.
    names, F, S = [], [], []
    for c in args.ckpt:
        p = c if os.path.isabs(c) else os.path.join(HERE, c)
        m = _fno.load_any(p, dev)
        s = np.empty((len(X), 1, 32, 64, 2), np.float32)
        F.append(_fno.predict_fno([m], X, dev, alpha=0.0, sigma_out=s).astype(np.float32))
        S.append(s)
        names.append(os.path.basename(c).replace(".pth", ""))
        del m
        torch.cuda.empty_cache()
    F = np.stack(F)                                     # (K, N, 20, 32, 64, 2)
    S = np.stack(S)                                     # (K, N,  1, 32, 64, 2)
    K = len(F)
    Yuv = Y[..., :2].astype(np.float32)
    Ystd = Yuv.std(1, keepdims=True)

    def score(P2: np.ndarray, band=None, extra=None):
        """Four subscores + `final` for a composed (N,20,32,64,2) physical prediction."""
        P = np.concatenate([P2, np.zeros_like(P2[..., :1])], -1).astype(np.float32)
        A, a_i, scored, n = prep(P, Y)
        dm, tk, mv = sps9.sample_metrics(P, Y)
        a_b = sps9.weights_from_metrics(f * dm, f * tk, f * mv)
        b = optimal_band(f * A, a_b, scored) if band is None else band
        s = sps_score(f * A, a_b, scored, n, b)
        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        mvv = sc.mvpe_rel_l2(P, Y)
        four = [sc.score_error(r * args.f), sc.score_error(k * args.f),
                sc.score_error(mvv * args.f), s]
        fin = final_v9.predict(four[:3] + [args.time_score, four[3]])[0]
        return r, k, mvv, s, fin

    # =========================================================================================
    print("\n################ P1  MEMBER WEIGHTS ################")
    print(f"{'member':34s} {'relL2':>7s} {'w_field':>8s} {'w_sigma':>8s}")
    per = [float(np.mean(sc.rel_l2_per_sample(
        np.concatenate([compose(F[j], S[j], args.alpha),
                        np.zeros_like(F[j][..., :1])], -1), Y, 2))) for j in range(K)]

    fold_of = {t: i % args.folds for i, t in enumerate(ut)}
    fid = np.array([fold_of[t] for t in traj])

    # Fitted on ALL windows, only to display the weights; every number that decides anything
    # comes from the grouped CV below.
    Af = F.reshape(K, -1).T
    Wf = simplex_lsq(Af, Yuv.reshape(-1))
    As = S.reshape(K, -1).T
    Ws = simplex_lsq(As, Ystd.reshape(-1))
    for j in range(K):
        print(f"{names[j][:34]:34s} {per[j]:7.4f} {Wf[j]:8.3f} {Ws[j]:8.3f}")

    # Grouped CV: weights fitted on the other folds only, applied to this fold, then pooled.
    Pu = np.empty_like(F[0])
    Pw = np.empty_like(F[0])
    for k_ in range(args.folds):
        tr, te = fid != k_, fid == k_
        wf = simplex_lsq(F[:, tr].reshape(K, -1).T, Yuv[tr].reshape(-1))
        ws = simplex_lsq(S[:, tr].reshape(K, -1).T, Ystd[tr].reshape(-1))
        Pu[te] = compose(F[:, te].mean(0), S[:, te].mean(0), args.alpha)
        Pw[te] = compose(np.tensordot(wf, F[:, te], (0, 0)),
                         np.tensordot(ws, S[:, te], (0, 0)), args.alpha)
    hdr = f"{'ensemble':34s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} {'sps':>6s} {'final':>7s}"
    print("\n  out-of-fold, band refit per row:")
    print("  " + hdr)
    for lab, P in (("uniform (shipped)", Pu), ("fitted weights", Pw)):
        r, k, mv, s, fin = score(P)
        print(f"  {lab:34s} {r:7.4f} {k:7.4f} {mv:7.4f} {s:6.2f} {fin:7.3f}")

    # =========================================================================================
    print("\n################ P2  BAND FROM CROSS-FAMILY SPREAD ################")
    sig = S.mean(0)                                     # what the package deploys
    spread = F.std(0).mean(1, keepdims=True)            # per-element, time-averaged
    P = np.concatenate([compose(F.mean(0), sig, args.alpha),
                        np.zeros_like(F[0][..., :1])], -1).astype(np.float32)
    A, _, scored, n = prep(P, Y)
    dm, tk, mv = sps9.sample_metrics(P, Y)
    a_b = sps9.weights_from_metrics(f * dm, f * tk, f * mv)
    E = f * A

    # Correlation with |error| after the per-element mean is removed -- round 6's own statistic,
    # recomputed on a heterogeneous set. This is the number that decided the question then.
    Em = E[..., :2] - E[..., :2].mean(0, keepdims=True)
    for lab, q in (("sigma_head", sig), ("cross-family spread", spread)):
        qq = np.broadcast_to(q, E[..., :2].shape)
        qm = qq - qq.mean(0, keepdims=True)
        c = float((Em * qm).sum() / np.sqrt((Em ** 2).sum() * (qm ** 2).sum()))
        print(f"  corr(|E|, {lab:22s}) after per-element mean removed: {c:.3f}")

    def ratios(p, q):
        r = np.ones_like(E[..., :2])
        if p:
            r = r * np.clip(sig[..., :2] / np.median(sig[..., :2]), 0.3, 3.0) ** p
        if q:
            r = r * np.clip(spread[..., :2] / np.median(spread[..., :2]), 0.3, 3.0) ** q
        return np.concatenate([r, np.ones_like(r[..., :1])], -1).astype(np.float32)

    grid = [(p, q) for p in (0.45,) for q in (0.0, 0.15, 0.3, 0.45)] + [(0.3, 0.3), (0.0, 0.45)]
    print(f"\n  grouped-trajectory CV, {args.folds} folds "
          f"(`base` refit inside every fold at every (p,q))")
    print(f"  {'p':>5s} {'q':>5s} {'CV sps':>8s} {'in-sample':>10s}")
    best, rows = None, []
    for p, q in grid:
        R = ratios(p, q)
        Ed = (E / R).astype(np.float32)
        # sps is 100 * (weighted hits) / n_scored, so folds pool by their scored counts, not
        # by a plain mean over folds -- the folds differ in size by up to 2x.
        num = den = 0.0
        for k_ in range(args.folds):
            tr, te = fid != k_, fid == k_
            b = optimal_band(Ed[tr], a_b[tr], scored[tr])
            nk = int(scored[te].sum())
            num += sps_score(E[te], a_b[te], scored[te], nk,
                             (b * R[te]).astype(np.float32)) * nk
            den += nk
        ins = sps_score(E, a_b, scored, n,
                        (optimal_band(Ed, a_b, scored) * R).astype(np.float32))
        w = num / den
        print(f"  {p:5.2f} {q:5.2f} {w:8.3f} {ins:10.3f}")
        rows.append((w, p, q))
        if best is None or w > best[0]:
            best = (w, p, q)
    inc = [r for r in rows if r[1:] == (0.45, 0.0)][0][0]
    print(f"  -> CV picks p = {best[1]}, q = {best[2]}   CV sps {best[0]:.3f} "
          f"against the incumbent's {inc:.3f}  ({best[0] - inc:+.3f})")
    print("  q = 0 is the incumbent. A q != 0 that wins here is worth "
          f"{0.2502:.3f} `final` per sps point.")


if __name__ == "__main__":
    main()
