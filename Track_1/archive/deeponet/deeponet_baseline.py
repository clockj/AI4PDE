"""DeepONet baseline for Track 1, for comparison against the official CNO/FNO/Transolver.

The competition ships no DeepONet checkpoint, so this trains one from scratch on train_real
(excluding the validation trajectories) and scores it on the same 336-window validation set.

Branch net: conv encoder over the 20-frame input window -> per-channel coefficients.
Trunk net : MLP over normalized (x, y, t) -> shared basis functions.
Output    : <branch, trunk> contraction, i.e. a rank-p expansion of the output field.

    python deeponet_baseline.py --epochs 40
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

T_IN = T_OUT = 20
H, W, C = 32, 64, 2  # u, v (p is zero-filled and unscored)
SUB = 2

# GaussianNormalizer statistics fitted on train_real 32x64 (channels u, v).
MEAN = np.array([0.154960856, -0.000513992854], dtype=np.float32)
STD = np.array([0.0968056545, 0.015960684], dtype=np.float32)

VAL_TRAJ = ["3750_0", "6300_10", "11400_20", "16500_5",
            "21600_15", "26700_0", "13950_10", "8850_20"]


def load_trajectories(data_dir: str, exclude: list[str]) -> list[np.ndarray]:
    """Return one (T, H, W, 2) float32 array per trajectory, normalized."""
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


class DeepONet(nn.Module):
    def __init__(self, p: int = 256, width: int = 128):
        super().__init__()
        # Branch: encode the input window (time folded into channels).
        self.branch = nn.Sequential(
            nn.Conv2d(T_IN * C, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, C * p),
        )
        # Trunk: shared basis over the fixed output grid.
        self.trunk = nn.Sequential(
            nn.Linear(3, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, p),
        )
        self.bias = nn.Parameter(torch.zeros(C))
        self.p = p
        t, y, x = torch.meshgrid(
            torch.linspace(-1, 1, T_OUT), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
            indexing="ij")
        self.register_buffer("coords", torch.stack([x, y, t], -1).reshape(-1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        z = x.permute(0, 1, 4, 2, 3).reshape(b, T_IN * C, H, W)
        coeff = self.branch(z).reshape(b, C, self.p)
        basis = self.trunk(self.coords)                       # (T*H*W, p)
        out = torch.einsum("bcp,np->bnc", coeff, basis) / self.p**0.5
        return out.reshape(b, T_OUT, H, W, C) + self.bias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/users/PAS1187/qwu/RealPDE/train_real/train_real")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="deeponet.pth")
    args = ap.parse_args()

    dev = "cuda"
    trajs = load_trajectories(args.data, exclude=VAL_TRAJ)
    X, Y = make_windows(trajs, args.stride)
    print(f"train: {len(trajs)} trajectories -> {len(X)} windows (stride {args.stride})")
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    model = DeepONet().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(X) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = n = 0.0
        for i in range(0, len(X), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = X[idx].to(dev, non_blocking=True), Y[idx].to(dev, non_blocking=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  mse {tot/n:.5f}  ({time.time()-t0:.0f}s)")

    torch.save({"model_state_dict": model.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
