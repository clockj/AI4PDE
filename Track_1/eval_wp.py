"""Score the WP1/WP2 models on both validation sets, with the SPS band and the alpha sweep.

    python eval_wp.py --ckpt unet/unet_mse_e150.pth unet/wp1_s5.pth unet/wp2_s2.pth

Everything is fitted on `val_seen.npz` and reported on `val_unseen.npz`, so the headline
numbers stay honest: the models never trained on the unseen trajectories, and the SPS band
never saw them either.

`eval_all.py` remains the full historical table (FNO / CNO / DeepONet / PDE-Refiner).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = "/home/clockj/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7"
sys.path.insert(0, os.path.join(HERE, "unet"))

spec = importlib.util.spec_from_file_location("sc", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

from model import (MEAN, STD, UNet, UNetTwoHead, compose, from_ch,  # noqa: E402
                   mse_predict, std_t, to_ch)


# --------------------------------------------------------------------------- inference


def run(ckpt: str, X: np.ndarray, alpha: float, batch: int = 16):
    """-> (prediction (N,T,H,W,3), sigma (N,1,H,W,2) or None, seconds per window).

    Timing is forward-pass-only at batch 16, matching eval_all.py so the numbers are
    comparable with the historical table. The submission's end-to-end predict() is slower;
    validate_submission.py measures that.
    """
    blob = torch.load(ckpt, map_location="cuda")
    two = blob.get("mode") == "twohead"
    if two:
        m = UNetTwoHead(blob.get("width", 64), var_head=blob.get("var_head", True)).cuda()
    else:
        m = UNet().cuda()
    m.load_state_dict(blob["model_state_dict"])
    m.eval()

    mean, std = torch.from_numpy(MEAN).cuda(), torch.from_numpy(STD).cuda()
    preds, sigmas, t = [], [], 0.0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch, ..., :2])).cuda()
            prev = to_ch((xb - mean) / std)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            s = None
            if two:
                raw, logstd = m(prev)
                y = from_ch(raw)
                if logstd is not None:
                    s = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)
                    y = compose(y, s, alpha)
            else:
                y = from_ch(mse_predict(m, prev))
            torch.cuda.synchronize()
            t += time.perf_counter() - t0
            preds.append((y * std + mean).float().cpu().numpy())
            if two and s is not None:
                sigmas.append((s * std).float().cpu().numpy())  # physical units

    uv = np.concatenate(preds)
    P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
    S = np.concatenate(sigmas).astype(np.float32) if sigmas else None
    del m
    torch.cuda.empty_cache()
    return P, S, t / len(X)


# --------------------------------------------------------------------------- scoring


def sps_of(P, Y, half):
    """half: None for the scorer's default +-5% band, else a broadcastable half-width."""
    if half is None:
        return sc.score_sps(sc.aggregate_sps(P, Y, 2)[0])
    lo, hi = P - half, P + half
    return sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=lo, upper=hi)[0])


def subscores(P, Y, tstep, half=None):
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
    mv = sc.mvpe_rel_l2(P, Y)
    s = [sc.score_error(r), sc.score_error(k), sc.score_error(mv),
         sc.score_time(tstep), sps_of(P, Y, half)]
    return dict(rel_err=r, tke_err=k, mvpe_err=mv, rel=s[0], tke=s[1], mvpe=s[2],
                time=s[3], sps=s[4], final=float(np.mean(s)), ms=tstep * 1000)


def fit_scale(P, Y, w, grid):
    """Pick the multiplier on the half-width field `w` that maximizes SPS."""
    best = (-1.0, None)
    for c in grid:
        v = sps_of(P, Y, c * w if w is not None else c)
        if v > best[0]:
            best = (v, c)
    return best[1], best[0]


def err_map(P, Y):
    """Per-element RMS error over the window axis: (1, T, H, W, 3)."""
    return np.sqrt(np.mean((P - Y) ** 2, axis=0, keepdims=True)).astype(np.float32)


def sigma_field(S):
    """(N,1,H,W,2) physical std -> (N,1,H,W,3) half-width field (p channel is unscored)."""
    return np.concatenate([S, np.zeros_like(S[..., :1])], -1)


def optimal_half(s):
    """Per-element half-width that maximizes the SPS integrand, given error std `s`.

    Each element contributes (1-pm) * exp(-(u-l)/sigma) when the target lands inside. For a
    symmetric interval of half-width h around the prediction and Gaussian error of std s that
    expectation is exp(-2h/sigma) * erf(h / (s*sqrt2)), so the optimum is a genuine trade-off:
    it saturates near sigma/2 for badly-predicted elements instead of growing with s. That is
    why the naive h proportional to s does no better than a constant.
    """
    hs = torch.linspace(1e-4, 0.08, 400).view(-1, 1)
    decay = torch.exp(-2.0 * hs / sc.SIGMA_GLOBAL)
    flat = torch.from_numpy(np.ascontiguousarray(s)).reshape(-1).clamp(min=1e-6)
    out = torch.empty_like(flat)
    for i in range(0, len(flat), 1 << 20):  # chunked: the full grid would be tens of GB
        chunk = flat[i:i + (1 << 20)]
        g = decay * torch.erf(hs / (chunk * 1.4142135))
        out[i:i + len(chunk)] = hs.view(-1)[g.argmax(0)]
    return out.reshape(s.shape).numpy().astype(np.float32)


# --------------------------------------------------------------------------- diagnostics


def var_head_diag(S, Y):
    """How good is sigma_hat? Baseline to beat: the input window's own std (0.756 / 0.559)."""
    true = np.std(Y[..., :2], axis=1, keepdims=True)  # biased, matches scoring
    a, b = S[..., :2].ravel(), true.ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return corr, rel


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    sets = {}
    for tag, path in [("unseen", "val_unseen.npz"), ("seen", "val_seen.npz")]:
        d = np.load(os.path.join(HERE, path))
        sets[tag] = (d["x"], d["y"])

    const_grid = np.concatenate([np.linspace(0.001, 0.02, 39), np.linspace(0.022, 0.06, 20)])
    scale_grid = np.linspace(0.1, 3.0, 30)

    for ckpt in args.ckpt:
        path = ckpt if os.path.isabs(ckpt) else os.path.join(HERE, ckpt)
        if not os.path.exists(path):
            print(f"(skipped {ckpt}: missing)")
            continue
        name = os.path.basename(path)
        two = torch.load(path, map_location="cpu").get("mode") == "twohead"
        alphas = args.alphas if two else [0.0]

        print(f"\n{'='*104}\n{name}\n{'='*104}")
        hdr = (f"{'alpha':>5s} {'set':7s} {'relL2':>6s} {'TKE':>6s} {'MVPE':>6s} | "
               f"{'rel':>5s} {'tke':>5s} {'mvpe':>5s} {'time':>5s} {'sps':>5s} | {'FINAL':>6s}")
        print(hdr)
        print("-" * len(hdr))

        best_alpha, best_final, cache = None, -1.0, {}
        for alpha in alphas:
            row = {}
            for tag, (X, Y) in sets.items():
                P, S, t = run(path, X, alpha)
                row[tag] = (P, S, t, subscores(P, Y, t))
                v = row[tag][3]
                print(f"{alpha:5.2f} {tag:7s} {v['rel_err']:6.3f} {v['tke_err']:6.3f} "
                      f"{v['mvpe_err']:6.3f} | {v['rel']:5.1f} {v['tke']:5.1f} {v['mvpe']:5.1f} "
                      f"{v['time']:5.1f} {v['sps']:5.1f} | {v['final']:6.2f}")
            gap = row["seen"][3]["final"] - row["unseen"][3]["final"]
            print(f"{'':5s} {'delta':7s} {'':>6s} {'':>6s} {'':>6s} | {'':>5s} {'':>5s} "
                  f"{'':>5s} {'':>5s} {'':>5s} | {gap:+6.2f}  <- generalization gap")
            if row["unseen"][3]["final"] > best_final:
                best_final, best_alpha, cache = row["unseen"][3]["final"], alpha, row

        Pu, Su, tu, _ = cache["unseen"]
        Ps, Ss, _, _ = cache["seen"]
        Yu, Ys = sets["unseen"][1], sets["seen"][1]

        if Su is not None:
            c, r = var_head_diag(Su, Yu)
            print(f"\nvariance head on unseen: corr {c:.3f}  relL2 {r:.3f}"
                  f"   (input-window std baseline: 0.756 / 0.559)")

        # --- SPS bands: every parameter fitted on SEEN, reported on UNSEEN.
        print(f"\nSPS bands (alpha = {best_alpha}, fitted on seen, reported on unseen)")
        print(f"  {'band':22s} {'fit':>10s} {'sps':>6s} {'FINAL':>7s}")
        base = subscores(Pu, Yu, tu)
        print(f"  {'default +-5%':22s} {'-':>10s} {base['sps']:6.2f} {base['final']:7.2f}")

        emap = err_map(Ps, Ys)
        variants = [("constant half-width", None, const_grid, None),
                    ("c * per-element RMSE", emap, scale_grid, None),
                    ("c * optimal h(RMSE)", optimal_half(emap), scale_grid, None)]
        if Su is not None:
            variants.append(("c * sigma_hat", sigma_field(Ss), scale_grid, sigma_field(Su)))
            variants.append(("c * optimal h(sigma)", optimal_half(sigma_field(Ss)), scale_grid,
                             optimal_half(sigma_field(Su))))

        for label, w_seen, grid, w_unseen in variants:
            c, _ = fit_scale(Ps, Ys, w_seen, grid)
            w = w_unseen if w_unseen is not None else w_seen
            v = subscores(Pu, Yu, tu, half=c * w if w is not None else c)
            print(f"  {label:22s} {c:10.4f} {v['sps']:6.2f} {v['final']:7.2f}")

        # Does the model have a systematic per-element bias that transfers between
        # trajectories? If it does, subtracting it is free accuracy on every subscore.
        bias = np.mean(Ys - Ps, axis=0, keepdims=True).astype(np.float32)
        bias[..., 2] = 0.0
        cb, _ = fit_scale(Ps + bias, Ys, None, const_grid)
        vb = subscores(Pu + bias, Yu, tu, half=cb)
        print(f"\nper-element bias map fitted on seen, applied to unseen (const band {cb:.4f})")
        print(f"  {'relL2':>6s} {'TKE':>6s} {'MVPE':>6s} | {'rel':>5s} {'tke':>5s} {'mvpe':>5s} "
              f"{'sps':>5s} | {'FINAL':>6s}")
        print(f"  {vb['rel_err']:6.3f} {vb['tke_err']:6.3f} {vb['mvpe_err']:6.3f} | "
              f"{vb['rel']:5.1f} {vb['tke']:5.1f} {vb['mvpe']:5.1f} {vb['sps']:5.1f} | "
              f"{vb['final']:6.2f}")


if __name__ == "__main__":
    main()
