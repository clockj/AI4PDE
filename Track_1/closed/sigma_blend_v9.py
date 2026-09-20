"""Can a blend of sigma estimators beat the variance head alone?

    python sigma_blend_v9.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

tke_branch_v9.py prices the head-to-oracle sigma gap at +7.7 `final`, +1.2 for the first 25%.
This asks whether any of that is available with no retraining at all, by combining estimators
the model already has:

    head    the variance head's sigma
    field   std_t of the predicted field itself -- what compose() falls back to at alpha = 0
    in      std_t of the INPUT window
    trajin  per-trajectory mean of `in`, legal transductively (predict() gets every window)

`head` is the best of the four alone (TKE 0.483 against 0.561 and 0.783), but they fail
differently, which is the same argument that made architecture diversity pay in round 8.

This is NOT the fitted fluctuation gain that was already falsified (CLAUDE.md: scalar,
per-pixel and per-pixel/time gains on the head's own sigma all gained in-sample and two of
three lost out-of-sample). A gain rescales one estimator; this adds independent information.

Fitted in log space on the WHOLE of val_seen, reported on the AoA 15-20 subset of val_unseen,
which is the split convention the rest of the tooling uses.
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
from tke_branch_v9 import recompose

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

F_BOARD = 1.39
EPS = 1e-6


def head_sigma(models, X, dev, batch=64):
    """Ensemble-mean sigma from the variance heads, in PHYSICAL units: (N, 1, H, W, 2).

    Mirrors the averaging `_fno.predict_fno` does -- mean of exp(logstd) across members, then
    multiplied by STD to leave normalized units.
    """
    import torch
    import _fno
    out = np.empty((len(X), 1, _fno.H, _fno.W, _fno.C), dtype=np.float32)
    std = torch.from_numpy(_fno.STD).to(dev)
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(dev)[..., :2].float()
            prev = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), _fno.CH,
                                                                     _fno.H, _fno.W)
            acc = None
            for m in models:
                _, logstd = m(prev)
                s = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)
                acc = s if acc is None else acc + s
            out[i:i + batch] = ((acc / len(models)) * std).cpu().numpy()
    return out


def by_traj(sig, ids):
    name = np.array([str(i).split(":")[0] for i in ids])
    out = np.empty_like(sig)
    for t in np.unique(name):
        out[name == t] = sig[name == t].mean(0, keepdims=True)
    return out


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
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    f = np.float32(args.f)

    def estimators(d):
        X, ids = d["x"], d["ids"]
        R0 = infer(models, X, dev, 0.0)                      # raw field, no rescaling
        sin = X[..., :2].std(1, keepdims=True)
        return R0, dict(head=head_sigma(models, X, dev),
                        field=R0[..., :2].std(1, keepdims=True),
                        into=sin, trajin=by_traj(sin, ids))

    R0s, Es = estimators(ds)
    R0u, Eu = estimators(du)
    keys = ["head", "field", "into", "trajin"]
    Ts = ds["y"][..., :2].std(1, keepdims=True)
    Tu = du["y"][..., :2].std(1, keepdims=True)

    # Least squares in log space, per channel. Elements where the target std is 0 sit outside
    # the PIV field of view or inside the airfoil; v9 drops them from scoring, and log(0) is
    # undefined, so they are dropped from the fit too.
    coef = {}
    for c in range(2):
        t = Ts[..., c].ravel()
        cols = [np.log(Es[k][..., c].ravel() + EPS) for k in keys]
        A = np.stack(cols + [np.ones_like(t)], 1)
        ok = t > 0
        coef[c] = np.linalg.lstsq(A[ok], np.log(t[ok]), rcond=None)[0]
        print(f"  channel {c}: " + "  ".join(f"{k} {coef[c][j]:+.3f}"
                                             for j, k in enumerate(keys))
              + f"  bias {coef[c][-1]:+.3f}")

    def blended(E):
        out = np.empty_like(E["head"])
        for c in range(2):
            lg = sum(coef[c][j] * np.log(E[k][..., c] + EPS) for j, k in enumerate(keys))
            out[..., c] = np.exp(lg + coef[c][-1])
        return out

    hi_s = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hi_u = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    Ys, Yu = ds["y"][hi_s], du["y"][hi_u]

    def pad(s):
        """(N,1,H,W,2) -> (N,1,H,W,3); the pressure channel is zero-filled and unscored."""
        return np.concatenate([s, np.zeros_like(s[..., :1])], -1)

    rows = {"head (alpha 0.75)": (infer(models, ds["x"], dev, args.alpha)[hi_s],
                                  infer(models, du["x"], dev, args.alpha)[hi_u])}
    for tag, Ss, Su in (("head (alpha 1)", Es["head"], Eu["head"]),
                        ("blend", blended(Es), blended(Eu))):
        rows[tag] = (recompose(R0s[hi_s], pad(Ss)[hi_s], 1.0),
                     recompose(R0u[hi_u], pad(Su)[hi_u], 1.0))
    rows["oracle"] = (recompose(R0s[hi_s], pad(Ts)[hi_s], 1.0),
                      recompose(R0u[hi_u], pad(Tu)[hi_u], 1.0))

    hdr = (f"{'sigma':18s} | {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} {'tke':>6s} "
           f"{'mvpe':>6s} {'sps':>6s} | {'final':>7s} {'d':>7s}")
    print(f"\nfit on all {len(ds['x'])} val_seen windows, reported on {int(hi_u.sum())} "
          f"AoA {args.aoa} val_unseen windows, errors {args.f}x, band refit per row")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for tag, (Ps, Pu) in rows.items():
        Af, af, sf, _ = prep(Ps, Ys)
        Au, _, su, nu = prep(Pu, Yu)
        band = optimal_band(f * Af, af, sf)
        r = float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2)))
        mv = sc.mvpe_rel_l2(Pu, Yu)
        dm, tk, mvv = sps9.sample_metrics(Pu, Yu)
        s = sps_score(f * Au, sps9.weights_from_metrics(args.f * dm, args.f * tk, args.f * mvv),
                      su, nu, band)
        four = [sc.score_error(r * args.f), sc.score_error(k * args.f),
                sc.score_error(mv * args.f), s]
        fin = final_v9.predict(four[:3] + [args.time_score, four[3]])[0]
        base = fin if base is None else base
        print(f"{tag:18s} | {r:7.4f} {k:7.4f} {mv:7.4f} | {four[0]:6.2f} {four[1]:6.2f} "
              f"{four[2]:6.2f} {s:6.2f} | {fin:7.3f} {fin - base:+7.3f}")


if __name__ == "__main__":
    main()
