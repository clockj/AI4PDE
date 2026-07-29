"""Score the U-Net (mse control) and PDE-Refiner on the same 336-window validation set.

    python eval_refiner.py <val.npz> unet_mse.pth unet_refiner.pth
"""
from __future__ import annotations

import importlib.util
import sys
import time

import numpy as np
import torch

from pde_refiner import MEAN, STD, UNet, from_ch, refiner_predict, to_ch

SCORING = "/home/clockj/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7/scoring.py"
spec = importlib.util.spec_from_file_location("sc", SCORING)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def predict(model, mode, X, dev="cuda", B=16):
    mean = torch.from_numpy(MEAN).to(dev)
    std = torch.from_numpy(STD).to(dev)
    outs, t = [], 0.0
    with torch.no_grad():
        for i in range(0, len(X), B):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + B, ..., :2])).to(dev)
            prev = to_ch((xb - mean) / std)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if mode == "refiner":
                y = refiner_predict(model, prev)
            else:
                k0 = torch.zeros(len(prev), dtype=torch.long, device=dev)
                y = model(prev, torch.zeros_like(prev), k0)
            torch.cuda.synchronize()
            t += time.perf_counter() - t0
            outs.append((from_ch(y) * std + mean).float().cpu().numpy())
    uv = np.concatenate(outs)
    return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32), t / len(X)


def score(P, Y, tstep, tag):
    c = 2
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, c)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, c)))
    m = sc.mvpe_rel_l2(P, Y)
    sps = sc.score_sps(sc.aggregate_sps(P, Y, c)[0])
    sps_t = max(sc.score_sps(sc.aggregate_sps(P, Y, c, lower=P - h, upper=P + h)[0])
                for h in [0.003, 0.005, 0.0075, 0.01, 0.015])
    s = [sc.score_error(r), sc.score_error(k), sc.score_error(m), sc.score_time(tstep), sps]
    print(f"{tag:22s} relL2={r:.3f} TKE={k:.3f} MVPE={m:.3f} | rel={s[0]:.1f} tke={s[1]:.1f} "
          f"mvpe={s[2]:.1f} time={s[3]:.1f} sps={sps:.1f} | FINAL={np.mean(s):.2f} "
          f"+band={np.mean(s[:4] + [sps_t]):.2f} ({tstep*1000:.2f} ms/win)")


def main() -> None:
    val, *ckpts = sys.argv[1:]
    d = np.load(val)
    X, Y = d["x"], d["y"]
    for ck in ckpts:
        blob = torch.load(ck, map_location="cuda")
        mode = blob.get("mode", "mse")
        model = UNet().cuda()
        model.load_state_dict(blob["model_state_dict"])
        model.eval()
        P, t = predict(model, mode, X)
        score(P, Y, t, f"U-Net ({mode})")


if __name__ == "__main__":
    main()
