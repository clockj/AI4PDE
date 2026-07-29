"""PDE-Refiner (Lippe et al., NeurIPS 2023) for Track 1, with a matched deterministic control.

Motivation: our measurements show every MSE-trained model collapses on the TKE subscore
(FNO 0.569, DeepONet 0.933, persistence 1.000) because L2 training regresses toward the
conditional mean and attenuates fluctuation amplitude. PDE-Refiner is the cheapest published
fix for exactly that failure mode -- a diffusion-style refinement that forces the network to
model all frequency components at every amplitude level -- at K+1 forward passes instead of the
20-100 a full conditional diffusion rollout (ACDM) would need.

Both modes share one U-Net so the comparison isolates the training objective:
    --mode mse       standard L2 regression, 1 forward pass at inference
    --mode refiner   PDE-Refiner objective,  K+1 forward passes at inference

    python pde_refiner.py --mode refiner --epochs 40
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
H, W, C = 32, 64, 2
SUB = 2
CH = T_IN * C  # 40 channels once time is folded in

MEAN = np.array([0.154960856, -0.000513992854], dtype=np.float32)
STD = np.array([0.0968056545, 0.015960684], dtype=np.float32)

VAL_TRAJ = ["3750_0", "6300_10", "11400_20", "16500_5",
            "21600_15", "26700_0", "13950_10", "8850_20"]

K_REFINE = 3
SIGMA_MIN = 1e-3


def load_trajectories(data_dir: str, exclude: list[str]) -> list[np.ndarray]:
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
    """Small 2D U-Net over (u_prev, z) with a refinement-step embedding."""

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


def sigma(k: int) -> float:
    return SIGMA_MIN ** (k / K_REFINE)


def refiner_loss(model: UNet, prev: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """k=0 predicts the solution; k>0 predicts the noise added to the target."""
    b = prev.shape[0]
    k = torch.randint(0, K_REFINE + 1, (b,), device=prev.device)
    is0 = k == 0
    z = torch.zeros_like(tgt)
    target = tgt.clone()
    if (~is0).any():
        eps = torch.randn_like(tgt)
        s = torch.tensor([sigma(int(i)) for i in k], device=prev.device).view(-1, 1, 1, 1)
        z = torch.where(is0.view(-1, 1, 1, 1), z, tgt + s * eps)
        target = torch.where(is0.view(-1, 1, 1, 1), tgt, eps)
    return torch.mean((model(prev, z, k) - target) ** 2)


@torch.no_grad()
def refiner_predict(model: UNet, prev: torch.Tensor) -> torch.Tensor:
    b = prev.shape[0]
    zeros = torch.zeros(b, CH, H, W, device=prev.device)
    k0 = torch.zeros(b, dtype=torch.long, device=prev.device)
    u = model(prev, zeros, k0)
    for k in range(1, K_REFINE + 1):
        s = sigma(k)
        z = u + s * torch.randn_like(u)
        kk = torch.full((b,), k, dtype=torch.long, device=prev.device)
        u = z - s * model(prev, z, kk)
    return u


def to_ch(x: torch.Tensor) -> torch.Tensor:
    """(B,T,H,W,C) -> (B,T*C,H,W)."""
    b = x.shape[0]
    return x.permute(0, 1, 4, 2, 3).reshape(b, CH, H, W)


def from_ch(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    return x.reshape(b, T_OUT, C, H, W).permute(0, 1, 3, 4, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mse", "refiner"], default="refiner")
    ap.add_argument("--data", default="/home/clockj/RealPDE/train_real/train_real")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"unet_{args.mode}.pth"

    dev = "cuda"
    trajs = load_trajectories(args.data, exclude=VAL_TRAJ)
    X, Y = make_windows(trajs, args.stride)
    print(f"train: {len(trajs)} trajectories -> {len(X)} windows, mode={args.mode}")
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)

    model = UNet().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    steps = args.epochs * ((len(X) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = n = 0.0
        for i in range(0, len(X), args.batch):
            idx = perm[i:i + args.batch]
            prev = to_ch(X[idx].to(dev))
            tgt = to_ch(Y[idx].to(dev))
            if args.mode == "refiner":
                loss = refiner_loss(model, prev, tgt)
            else:
                k0 = torch.zeros(len(idx), dtype=torch.long, device=dev)
                loss = torch.mean((model(prev, torch.zeros_like(tgt), k0) - tgt) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot/n:.5f}  ({time.time()-t0:.0f}s)")

    torch.save({"model_state_dict": model.state_dict(), "mode": args.mode}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
