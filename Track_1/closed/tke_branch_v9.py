"""WP6: how much `final_score` is left in the TKE branch, at most?

    python tke_branch_v9.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

`tke_ceiling.py` bounded the variance-FIELD prediction problem under v7. This asks the question
the round-9 weights make answerable: hand the model an ORACLE per-pixel temporal std and score
the result end to end. That is the ceiling on everything the TKE branch could ever pay --
adversarial training, a generative refiner, a better variance head, all of it -- because no
method can beat knowing the answer.

The composition is the one `_fno.predict_fno` applies, redone here in physical units so the
sigma can be swapped:

    y = mean_t(y) + (y - mean_t(y)) * (alpha*sigma + (1-alpha)*std_t(y)) / std_t(y)

Rows, weakest to strongest:
  head      the shipped configuration, alpha = 0.75 on the variance head's own sigma
  input     sigma from the INPUT window's own temporal std -- legal, no head needed
  traj-in   per-trajectory mean of the input-window std fields. Legal on the same argument
            --hist uses: predict() receives every window of the test set at once, so other
            windows' INPUTS are available. Tests whether the variance field is a property of
            the flow condition rather than of the window.
  traj      per-trajectory mean of the TARGET std fields -- illegal, but SMOOTH: it carries no
            per-window information at all, so the gap between it and `oracle` is exactly the
            part of the TKE error that is 20-frame sampling noise and reachable by nobody.
  oracle    sigma from the TARGET's own temporal std -- illegal, and the point of the exercise

Scored on the AoA 15-20 subset of val_unseen at the board's 1.39x per-element inflation, with
the band refit per row on val_seen. `final` uses the round-9 weights (final_v9.py).
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import sys

import numpy as np

import final_v9
import sps9
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

F_BOARD = 1.39


def recompose(P0: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Rescale P0's temporal fluctuation to carry `sigma`, in physical units."""
    m = P0.mean(1, keepdims=True)
    s = P0.std(1, keepdims=True)
    g = (alpha * sigma + (1.0 - alpha) * s) / (s + 1e-6)
    return (m + (P0 - m) * g).astype(np.float32)


def score_rows(rows, Ys, Yu, f=F_BOARD, time_score=90.5066, width=8):
    """Score {label: (pred_on_seen, pred_on_unseen)} and print one line each.

    The band is refit per row on the seen split -- rows differ in their error distribution, so
    a shared band would flatter whichever row it came from. Errors are inflated to board scale
    before scoring; `final` is the round-9 weighting.
    """
    fj = np.float32(f)
    hdr = (f"{'row':{width}s} | {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} "
           f"{'tke':>6s} {'mvpe':>6s} {'sps':>6s} | {'final':>7s} {'d(final)':>9s}")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for tag, (Ps, Pu) in rows.items():
        Af, af, sf, _ = prep(Ps, Ys)
        Au, _, su, nu = prep(Pu, Yu)
        band = optimal_band(fj * Af, af, sf)
        r = float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2)))
        mv = sc.mvpe_rel_l2(Pu, Yu)
        dm, tk, mvv = sps9.sample_metrics(Pu, Yu)
        s = sps_score(fj * Au, sps9.weights_from_metrics(f * dm, f * tk, f * mvv), su, nu, band)
        four = [sc.score_error(r * f), sc.score_error(k * f), sc.score_error(mv * f), s]
        fin = final_v9.predict(four[:3] + [time_score, four[3]])[0]
        base = fin if base is None else base
        print(f"{tag:{width}s} | {r:7.4f} {k:7.4f} {mv:7.4f} | {four[0]:6.2f} {four[1]:6.2f} "
              f"{four[2]:6.2f} {s:6.2f} | {fin:7.3f} {fin - base:+9.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--f", type=float, default=F_BOARD)
    ap.add_argument("--time-score", type=float, default=90.5066)
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hi_s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hi_u = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    f = np.float32(args.f)

    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    Ys, Yu = ds["y"][hi_s], du["y"][hi_u]
    Xs, Xu = ds["x"][hi_s], du["x"][hi_u]

    # alpha = 0 is the raw field, before any fluctuation rescaling -- the common starting
    # point every row below re-composes from.
    R0s = infer(models, ds["x"], dev, 0.0)[hi_s]
    R0u = infer(models, du["x"], dev, 0.0)[hi_u]

    def by_traj(sig: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Replace each window's sigma field with the mean over its whole trajectory."""
        name = np.array([str(i).split(":")[0] for i in ids])
        out = np.empty_like(sig)
        for t in np.unique(name):
            out[name == t] = sig[name == t].mean(0, keepdims=True)
        return out

    ids_s, ids_u = ds["ids"][hi_s], du["ids"][hi_u]
    Ss, Su = Xs.std(1, keepdims=True), Xu.std(1, keepdims=True)     # input-window std
    Ts, Tu = Ys.std(1, keepdims=True), Yu.std(1, keepdims=True)     # target std (oracle)

    Hs = infer(models, ds["x"], dev, args.alpha)[hi_s]
    Hu = infer(models, du["x"], dev, args.alpha)[hi_u]

    rows = {
        "head":    (Hs, Hu),
        "input":   (recompose(R0s, Ss, 1.0), recompose(R0u, Su, 1.0)),
        "traj-in": (recompose(R0s, by_traj(Ss, ids_s), 1.0),
                    recompose(R0u, by_traj(Su, ids_u), 1.0)),
        "traj":    (recompose(R0s, by_traj(Ts, ids_s), 1.0),
                    recompose(R0u, by_traj(Tu, ids_u), 1.0)),
    }

    # Walk sigma from where the head effectively left it to the oracle. The head's effective
    # sigma is the composed field's own std, so k = 0 reproduces the `head` row and k = 1 the
    # `oracle` row -- what the intermediate points give is the MARGINAL value of a realistic
    # partial improvement, which the endpoints alone do not.
    Es, Eu = Hs.std(1, keepdims=True), Hu.std(1, keepdims=True)
    for k in (0.25, 0.5, 0.75):
        rows[f"k={k:.2f}"] = (recompose(R0s, Es + k * (Ts - Es), 1.0),
                              recompose(R0u, Eu + k * (Tu - Eu), 1.0))
    rows["oracle"] = (recompose(R0s, Ts, 1.0), recompose(R0u, Tu, 1.0))

    print(f"AoA {args.aoa}, {len(Yu)} windows, errors inflated {args.f}x, band refit per row, "
          f"time_score {args.time_score:.3f}")
    score_rows(rows, Ys, Yu, args.f, args.time_score)


if __name__ == "__main__":
    main()
