"""Score a decoupled pair: field from one checkpoint, sigma from another.

    python eval_pair.py --field unet/wpA0_mse_nohead.pth --sigma unet/wpF_varonly.pth

This is the fully decoupled option flagged as untested in CLAUDE.md. With the field model
frozen and separate, its MVPE and rel-L2 are bit-identical to running it alone, so the variance
objective cannot cost the field head anything -- the failure mode that `report_20260730.md`
section 9 attributed the hidden set's -1.28 mvpe to. The price is a second forward pass, which
is charged to `time_score` and measured here rather than assumed.

`--calibrate` additionally folds in the window and history std (report_20260731.md section 2.2),
cross-fitted across trajectories so the coefficients never see the windows they score.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from calib_sigma import compose_np, features
from eval_wp import (HERE, amp, err_map, fit_scale, optimal_half, run, sc, subscores,
                     val_history)


def cross_fit(X, S, hist, Y, traj):
    """Per-channel least squares on [head, window, history, 1], fitted out-of-fold."""
    F = features(X, S, hist)
    true = Y[..., :2].std(1, keepdims=True)
    names = np.unique(traj)
    out = np.empty_like(S)
    for fold in range(2):
        te = np.isin(traj, names[fold::2])
        tr = ~te
        for ch in range(2):
            A = np.stack([f[tr][..., ch].ravel() for f in F]
                         + [np.ones(F[0][tr][..., ch].size)], 1)
            c = np.linalg.lstsq(A, true[tr][..., ch].ravel(), rcond=None)[0]
            B = np.stack([f[te][..., ch].ravel() for f in F]
                         + [np.ones(F[0][te][..., ch].size)], 1)
            out[te, ..., ch] = (B @ c).reshape(out[te, ..., ch].shape)
    return np.maximum(out, 1e-6).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--sigma", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 0.75, 1.0])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--suffix", default="", help="'_sim' for the simulation validation sets")
    args = ap.parse_args()

    data, H, traj = {}, {}, {}
    for tag in ("unseen", "seen"):
        d = np.load(os.path.join(HERE, f"val_{tag}{args.suffix}.npz"))
        data[tag] = (d["x"], d["y"])
        H[tag] = val_history(d["x"], d["ids"])
        traj[tag] = np.array([str(i).split(":")[0] for i in d["ids"]])

    # --field and --sigma may name the same checkpoint; then it is one network, not two, and
    # must be run and charged once. This is how a single two-head model is scored with the
    # calibration applied.
    same = os.path.abspath(args.field) == os.path.abspath(args.sigma)
    raw, sig, tf, ts = {}, {}, {}, {}
    for tag, (X, Y) in data.items():
        raw[tag], s_own, tf[tag] = run(os.path.join(HERE, args.field), X, 0.0, hist=H[tag])
        if same:
            sig[tag], ts[tag] = s_own, 0.0
        else:
            _, sig[tag], ts[tag] = run(os.path.join(HERE, args.sigma), X, 0.0, hist=H[tag])
        if sig[tag] is None:
            raise SystemExit(f"{args.sigma} has no variance head")
        if args.calibrate:
            sig[tag] = cross_fit(X, sig[tag], H[tag], Y, traj[tag])

    # Two separate networks both run per window, so the pair costs the sum of the passes.
    tstep = {t: tf[t] + ts[t] for t in data}
    print(f"field {os.path.basename(args.field)}  +  sigma "
          f"{os.path.basename(args.sigma)}"
          f"{'  [calibrated]' if args.calibrate else ''}")
    print(f"latency: field {tf['unseen']*1e3:.3f} ms + sigma {ts['unseen']*1e3:.3f} ms "
          f"= {tstep['unseen']*1e3:.3f} ms/window   time_score "
          f"{sc.score_time(tstep['unseen']):.2f}")

    true = data["unseen"][1][..., :2].std(1, keepdims=True)
    print(f"sigma rel-L2 on unseen: "
          f"{np.linalg.norm(sig['unseen']-true)/np.linalg.norm(true):.4f}")

    hdr = (f"{'alpha':>5s} {'set':7s} {'relL2':>6s} {'TKE':>6s} {'MVPE':>6s} | {'rel':>5s} "
           f"{'tke':>5s} {'mvpe':>5s} {'time':>5s} {'sps':>5s} | {'FINAL':>6s}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    best = (-1.0, None, None)
    for alpha in args.alphas:
        row = {}
        for tag in ("unseen", "seen"):
            P = compose_np(raw[tag], sig[tag], alpha)
            v = subscores(P, data[tag][1], tstep[tag])
            row[tag] = (P, v)
            print(f"{alpha:5.2f} {tag:7s} {v['rel_err']:6.3f} {v['tke_err']:6.3f} "
                  f"{v['mvpe_err']:6.3f} | {v['rel']:5.1f} {v['tke']:5.1f} {v['mvpe']:5.1f} "
                  f"{v['time']:5.1f} {v['sps']:5.1f} | {v['final']:6.2f}")
        if row["unseen"][1]["final"] > best[0]:
            best = (row["unseen"][1]["final"], alpha, row)

    _, alpha, row = best
    Pu, Ps = row["unseen"][0], row["seen"][0]
    Yu, Ys = data["unseen"][1], data["seen"][1]
    print(f"\nSPS bands (alpha = {alpha}, fitted on seen, reported on unseen)")
    emap = err_map(Ps, Ys)
    grid = np.linspace(0.1, 3.0, 30)
    const = np.concatenate([np.linspace(0.001, 0.02, 39), np.linspace(0.022, 0.06, 20)])
    for label, w_s, g, w_u in (
            ("constant half-width", None, const, None),
            ("c * optimal h(RMSE)", optimal_half(emap), grid, None),
            ("c * optimal h(RMSE*|w|)", optimal_half(emap * amp(Ps)), grid,
             optimal_half(emap * amp(Pu)))):
        c, _ = fit_scale(Ps, Ys, w_s, g)
        w = w_u if w_u is not None else w_s
        v = subscores(Pu, Yu, tstep["unseen"], half=c * w if w is not None else c)
        print(f"  {label:24s} {c:9.4f}  sps {v['sps']:6.2f}  FINAL {v['final']:7.2f}")


if __name__ == "__main__":
    main()
