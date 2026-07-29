"""Fit the SPS interval half-width on val_seen, report its effect on val_unseen.

Fitting and scoring on different sets keeps the number honest -- the `+band` figures in
report_20260729.md were fitted and scored on the same set and are mildly optimistic.

    python fit_band.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
KIT = "/home/clockj/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7"
sys.path.insert(0, os.path.join(TRACK1, "unet"))

from model import MEAN, STD, UNet, from_ch, mse_predict, to_ch  # noqa: E402

spec = importlib.util.spec_from_file_location("sc", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

GRID = [0.002, 0.003, 0.004, 0.005, 0.006, 0.0075, 0.01, 0.0125, 0.015, 0.02]


def predict_uv(model, X, dev="cuda", B=32):
    mean, std = torch.from_numpy(MEAN).to(dev), torch.from_numpy(STD).to(dev)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), B):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + B, ..., :2])).to(dev)
            y = mse_predict(model, to_ch((xb - mean) / std))
            outs.append((from_ch(y) * std + mean).float().cpu().numpy())
    uv = np.concatenate(outs)
    return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)


def main() -> None:
    model = UNet().cuda()
    model.load_state_dict(torch.load(os.path.join(TRACK1, "unet/unet_mse.pth"),
                                     map_location="cuda")["model_state_dict"])
    model.eval()

    sets = {}
    for tag in ("seen", "unseen"):
        d = np.load(os.path.join(TRACK1, f"val_{tag}.npz"))
        sets[tag] = (predict_uv(model, d["x"]), d["y"])

    print(f"{'h':>8s} | {'sps(seen)':>10s} {'cov':>5s} | {'sps(unseen)':>12s} {'cov':>5s}")
    best_h, best_seen = None, -1.0
    for h in GRID:
        row = {}
        for tag, (P, Y) in sets.items():
            v, cov = sc.aggregate_sps(P, Y, 2, lower=P - h, upper=P + h)
            row[tag] = (sc.score_sps(v), cov)
        if row["seen"][0] > best_seen:
            best_seen, best_h = row["seen"][0], h
        print(f"{h:8.4f} | {row['seen'][0]:10.2f} {row['seen'][1]:5.2f} | "
              f"{row['unseen'][0]:12.2f} {row['unseen'][1]:5.2f}")

    P, Y = sets["unseen"]
    default = sc.score_sps(sc.aggregate_sps(P, Y, 2)[0])
    fitted = sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=P - best_h, upper=P + best_h)[0])
    print(f"\nbest h (fitted on SEEN)      = {best_h}")
    print(f"sps on UNSEEN, default band  = {default:.2f}")
    print(f"sps on UNSEEN, fitted band   = {fitted:.2f}   (delta {fitted-default:+.2f} "
          f"=> {(fitted-default)/5:+.2f} final)")


if __name__ == "__main__":
    main()
