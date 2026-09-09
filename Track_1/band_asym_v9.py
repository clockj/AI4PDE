"""Does the SPS interval have to be CENTRED on the prediction?  Exact answer, and a CV of it.

    python band_asym_v9.py --ckpt <shipped> --err-ckpt <held-out twins> \
        --band ../sub_candidates/band_r17_simx4.npz --alpha 0.60

Five band closures (round 6, 10b, 11, 13, and round 19's fitting-sample study) all asked the
same question in different clothes: how WIDE should the interval be?  `sps9.optimal_band` solves
that exactly -- for each element it sorts the observed |E| and takes the best order statistic.
Every one of those closures inherited its one unexamined assumption: the interval is
`pred +/- h`, symmetric about the prediction.

Nothing in the scoring rule requires that.  SPS pays `(1-pm)*exp(-(upper-lower)/sigma)` when the
target lands inside, so the price is the interval's WIDTH and the placement is free.  If an
element's error distribution is skewed -- and a wake behind an airfoil has no reason to be
symmetric -- then sliding the interval buys containment for nothing.

The general problem is still exactly solvable.  Writing E = pred - target (signed), the target
lands inside [pred-hi, pred-lo] exactly when lo <= E <= hi, so the objective is

    exp(-(hi-lo)/sigma) * sum_i w_i * 1[lo <= E_i <= hi]

and its optimum is attained with both endpoints at observed E values.  Sorting E and sweeping
the count k gives every candidate in O(n^2) with O(n) memory, and the symmetric solution is the
constrained case lo = -hi, so the asymmetric optimum is >= it IN SAMPLE by construction.  The
only question is whether the offset GENERALIZES, which is what the grouped-trajectory CV here
measures -- the same protocol round 10b used to pick the band's blur.

The sigma scaling deploys unchanged: `|E| <= base * r**p` is `|E| / r**p <= base`, and that
deflation is signed-safe, so the fit runs on E / r**p and the deployed bounds are
`pred - hi*r**p`, `pred - lo*r**p`.

This is a ONE-SIDED bet.  Bounds feed no other subscore, so a null costs nothing and a win is
pure `sps` -- the 7.3 points that round 19 showed are unreachable through rms.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import sps9
from sps9 import SIGMA, prep

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

BIG = np.float32(1e9)


def solve_sym(D, W, cap):
    """Exact symmetric optimum on deflated signed errors: returns (lo, hi) with lo = -hi."""
    A = np.abs(D)
    A = np.where(W > 0, A, BIG)
    order = np.argsort(A, axis=0, kind="stable")
    srt = np.take_along_axis(A, order, axis=0)
    wts = np.take_along_axis(W, order, axis=0)
    gain = np.cumsum(wts, axis=0, dtype=np.float64) * np.exp(-2.0 * srt.astype(np.float64) / SIGMA)
    best = gain.argmax(axis=0)
    h = np.take_along_axis(srt, best[None], axis=0)[0]
    h = np.minimum(h, cap).astype(np.float32)
    return -h, h


def solve_asym(D, W, cap):
    """Exact optimum of exp(-(hi-lo)/sigma) * sum_i w_i*1[lo <= D_i <= hi], per element.

    Samples with zero weight (target == 0 for that window) are pushed to +inf so they can never
    bound an interval; an interval that reached them would pay width for no gain, so the
    optimum never selects one and the solve stays exact.
    """
    n = D.shape[0]
    S = np.where(W > 0, D, BIG)
    order = np.argsort(S, axis=0, kind="stable")
    srt = np.take_along_axis(S, order, axis=0).astype(np.float32)
    wts = np.take_along_axis(W, order, axis=0).astype(np.float64)
    cw = np.concatenate([np.zeros((1, D.shape[1])), np.cumsum(wts, axis=0)], axis=0)

    best = np.zeros(D.shape[1])
    lo = np.zeros(D.shape[1], np.float32)
    hi = np.zeros(D.shape[1], np.float32)
    for k in range(1, n + 1):
        top, bot = srt[k - 1:], srt[:n - k + 1]
        width = top - bot
        val = (cw[k:] - cw[:n - k + 1]) * np.exp(-width.astype(np.float64) / SIGMA)
        val[width > cap] = 0.0
        i = val.argmax(axis=0)
        v = np.take_along_axis(val, i[None], axis=0)[0]
        upd = v > best
        if upd.any():
            c = np.take_along_axis(bot, i[None], axis=0)[0]
            d = np.take_along_axis(top, i[None], axis=0)[0]
            best = np.where(upd, v, best)
            lo = np.where(upd, c, lo)
            hi = np.where(upd, d, hi)
    return lo, hi


def score(D, W, nsc, lo, hi):
    inb = (D >= lo) & (D <= hi)
    e = np.exp(-(hi - lo).astype(np.float64) / SIGMA)
    return float(100.0 * ((W * inb) * e).sum() / nsc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--err-ckpt", nargs="+", default=None)
    ap.add_argument("--band", required=True)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=1.39)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cap", type=float, default=0.16)
    args = ap.parse_args()

    import torch
    import _fno

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load(cs):
        return [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev) for c in cs]

    m = load(args.ckpt)
    m_err = m if args.err_ckpt is None else load(args.err_ckpt)
    b = np.load(args.band if os.path.isabs(args.band) else os.path.join(HERE, args.band))
    sigma_ref, p = b["sigma_ref"], float(b["p"])

    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    ids = [str(i) for i in du["ids"]]
    aoa = np.array([int(i.split(":")[0].split("_")[1]) for i in ids])
    keep = np.isin(aoa, args.aoa)
    X, Y = du["x"][keep], du["y"][keep]
    traj = np.array([i.split(":")[0] for i in ids])[keep]

    def infer(models, want_sigma=False):
        # predict_fno returns the MEASURED channels only, (N,20,32,64,2); the scorer wants the
        # zero-filled p channel back. tail_map_v9 does the same concatenate -- copying the call
        # without it is what crashed r20g.
        s = np.empty((len(X), 1, 32, 64, 2), np.float32) if want_sigma else None
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=s)
        P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
        if not want_sigma:
            return P
        return P, np.concatenate([s, np.zeros_like(s[..., :1])], -1).astype(np.float32)

    with torch.no_grad():
        _, sig = infer(m, want_sigma=True)
        P = infer(m_err)

    f = np.float32(args.f)
    E = f * (P[..., :2] - Y[..., :2]).astype(np.float32)          # signed, board-inflated
    A, a_w, scored, nsc_all = prep(P, Y)
    r = np.clip(sig[..., :2] / sigma_ref[..., :2], 0.3, 3.0) ** p  # (N,1,32,64,2)
    D = (E / r).reshape(len(E), -1)                                # deflated signed error
    W = (a_w[:, None] * scored[..., :2].reshape(len(E), -1)).astype(np.float64)

    print(f"windows {len(E)}  elements {D.shape[1]}  scored {int((W > 0).sum())}")

    # ---- 1. is there any skew at all? -------------------------------------------------
    ms = np.where(W > 0, D, np.nan)
    med = np.nanmedian(ms, axis=0)
    _, h0 = solve_sym(D, W, np.float32(args.cap / 2))
    ok = np.isfinite(med) & (h0 > 0)
    ratio = np.abs(med[ok]) / h0[ok]
    print("\n1. SKEW  |median signed error| / symmetric half-width, per element")
    print("   quartiles {:.3f} / {:.3f} / {:.3f}   mean {:.3f}   frac > 0.5: {:.3f}".format(
        *np.percentile(ratio, [25, 50, 75]), ratio.mean(), float((ratio > 0.5).mean())))

    # ---- 2. exact in-sample ceiling ---------------------------------------------------
    nsc = float(W.sum())
    lo_s, hi_s = solve_sym(D, W, np.float32(args.cap / 2))
    lo_a, hi_a = solve_asym(D, W, np.float32(args.cap))
    s_sym, s_asym = score(D, W, nsc, lo_s, hi_s), score(D, W, nsc, lo_a, hi_a)
    off = 0.5 * (lo_a + hi_a)
    print("\n2. IN SAMPLE (the ceiling -- asymmetric is >= symmetric by construction)")
    print(f"   symmetric  {s_sym:7.3f}      asymmetric {s_asym:7.3f}   (+{s_asym - s_sym:.3f})")
    print("   median |offset| / half-width {:.3f};  widths: sym {:.5f} asym {:.5f}".format(
        float(np.median(np.abs(off[ok]) / h0[ok])),
        float(np.median(2 * h0)), float(np.median((hi_a - lo_a)))))

    # ---- 3. does the offset generalize? -----------------------------------------------
    ut = np.unique(traj)
    g = {t: i % args.folds for i, t in enumerate(ut)}
    fold = np.array([g[t] for t in traj])
    tot = {"sym": 0.0, "asym": 0.0, "n": 0.0}
    print(f"\n3. GROUPED-TRAJECTORY CV ({len(ut)} trajectories, {args.folds} folds)")
    for k in range(args.folds):
        tr, te = fold != k, fold == k
        ls, hs = solve_sym(D[tr], W[tr], np.float32(args.cap / 2))
        la, ha = solve_asym(D[tr], W[tr], np.float32(args.cap))
        nte = float(W[te].sum())
        vs, va = score(D[te], W[te], nte, ls, hs), score(D[te], W[te], nte, la, ha)
        print(f"   fold {k}: {int(te.sum()):3d} windows   sym {vs:7.3f}   asym {va:7.3f}"
              f"   {va - vs:+.3f}")
        tot["sym"] += vs * nte
        tot["asym"] += va * nte
        tot["n"] += nte
    ds = tot["asym"] / tot["n"] - tot["sym"] / tot["n"]
    print(f"   POOLED  sym {tot['sym'] / tot['n']:7.3f}   asym {tot['asym'] / tot['n']:7.3f}"
          f"   -> {ds:+.3f} sps out of sample  ({0.2503 * ds:+.4f} final)")
    print("VERDICT: " + ("asymmetric intervals generalize -- wire into fit_band"
                         if ds > 0.05 else "null; the interval stays centred on the prediction"))


if __name__ == "__main__":
    main()
