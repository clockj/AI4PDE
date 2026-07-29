"""Shared U-Net backbone and real-data pipeline for Track 1.

This is the core both Track 1 experiments build on:
  * `Track_1/unet/train.py`            -- plain MSE training (our best model)
  * `Track_1/diffusion/pde_refiner.py` -- same backbone, PDE-Refiner objective

Geometry follows the competition protocol exactly: 32x64 grid (native 64x128 sub-sampled by 2),
T_in = T_out = 20, channels [u, v] (pressure is unmeasured on real data and zero-filled at
submission time).

The model maps a whole 20-frame input window to a whole 20-frame output window in ONE forward
pass. Time is folded into the channel axis (CH = T_in * C = 40), so this is a 2D image-to-image
network rather than a video model -- possible only because the grid and horizon are both fixed.
"""
from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

T_IN = T_OUT = 20
H, W, C = 32, 64, 2
SUB = 2
CH = T_IN * C  # 40 channels once time is folded in

# GaussianNormalizer statistics fitted on train_real 32x64, channels [u, v].
MEAN = np.array([0.154960856, -0.000513992854], dtype=np.float32)
STD = np.array([0.0968056545, 0.015960684], dtype=np.float32)

# Trajectories held out of training; see Track_1/make_val.py (val_unseen.npz).
VAL_TRAJ = ["3750_0", "6300_10", "8850_20", "11400_20",
            "13950_10", "16500_5", "21600_15", "26700_0"]

# Sizes the step-embedding table. The MSE model only ever uses index 0, but the table stays
# K_REFINE+1 wide so one architecture serves both objectives and both checkpoints load.
K_REFINE = 3


def load_trajectories(data_dir: str, exclude: list[str]) -> list[np.ndarray]:
    """Return one (T, H, W, 2) normalized float32 array per trajectory. Real data only."""
    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        if os.path.basename(path)[:-3] in exclude:
            continue
        with h5py.File(path, "r") as f:
            u = f["u"][:, ::SUB, ::SUB].astype(np.float32)
            v = f["v"][:, ::SUB, ::SUB].astype(np.float32)
        out.append((np.stack([u, v], -1) - MEAN) / STD)
    return out


def make_windows(trajs: list[np.ndarray], stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for d in trajs:
        for t in range(0, len(d) - T_IN - T_OUT + 1, stride):
            xs.append(d[t:t + T_IN])
            ys.append(d[t + T_IN:t + T_IN + T_OUT])
    return np.asarray(xs), np.asarray(ys)


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNet(nn.Module):
    """2D encoder-decoder over (prev_window, z), 1.93M parameters.

    64ch @ 32x64 -> 128ch @ 16x32 -> 256ch @ 8x16 -> back up with concatenated skips.
    The 8x16 bottleneck spans essentially the whole domain, which is what captures the
    large-scale wake; the skips carry fine detail straight to the output.

    `z` is the refinement slot: zeros for plain MSE prediction, a noised target for
    PDE-Refiner. `k` indexes the refinement step (always 0 for MSE).
    """

    def __init__(self, width: int = 64):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 4
        self.inc = block(2 * CH, w1)
        self.down1 = block(w1, w2)
        self.down2 = block(w2, w3)
        self.up1 = block(w3 + w2, w2)
        self.up2 = block(w2 + w1, w1)
        self.outc = nn.Conv2d(w1, CH, 1)
        self.kemb = nn.Embedding(K_REFINE + 1, w1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, prev: torch.Tensor, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        h1 = self.inc(torch.cat([prev, z], 1)) + self.kemb(k)[:, :, None, None]
        h2 = self.down1(self.pool(h1))
        h3 = self.down2(self.pool(h2))
        u2 = self.up1(torch.cat([nn.functional.interpolate(h3, scale_factor=2), h2], 1))
        u1 = self.up2(torch.cat([nn.functional.interpolate(u2, scale_factor=2), h1], 1))
        return self.outc(u1)


def to_ch(x: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, C) -> (B, T*C, H, W)."""
    return x.permute(0, 1, 4, 2, 3).reshape(x.shape[0], CH, H, W)


def from_ch(x: torch.Tensor) -> torch.Tensor:
    """(B, T*C, H, W) -> (B, T, H, W, C)."""
    return x.reshape(x.shape[0], T_OUT, C, H, W).permute(0, 1, 3, 4, 2)


@torch.no_grad()
def mse_predict(model: UNet, prev: torch.Tensor) -> torch.Tensor:
    """Single forward pass: z = zeros, k = 0."""
    k0 = torch.zeros(len(prev), dtype=torch.long, device=prev.device)
    return model(prev, torch.zeros_like(prev), k0)
