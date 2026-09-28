"""PDE-Refiner (Lippe et al., NeurIPS 2023) on the shared Track 1 U-Net.

Result: this did NOT work here. 82.53 on held-out trajectories against the identical backbone
trained with plain MSE at 83.83 -- worse on rel-L2 and worse on TKE, the metric it was chosen
to fix, at 4x the inference cost. See Track_1/report/report_20260729.md for the ablation and the
two candidate explanations. Kept for reproducibility; not a line of work to continue.

The backbone, data pipeline and normalization live in Track_1/unet/model.py. This file adds only
the refinement objective and its sampler.

    python pde_refiner.py --epochs 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "unet"))

from model import (  # noqa: E402
    CH, H, K_REFINE, UNet, VAL_TRAJ, load_trajectories, make_windows, to_ch, W,
)

SIGMA_MIN = 1e-3


def sigma(k: int) -> float:
    """Noise schedule sigma_k = sigma_min^(k/K): coarse first, then progressively finer."""
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
    """K+1 forward passes: one prediction, then K denoising refinements."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/users/PAS1187/qwu/RealPDE/train_real/train_real")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="unet_refiner.pth")
    args = ap.parse_args()

    dev = "cuda"
    trajs = load_trajectories(args.data, exclude=VAL_TRAJ)
    X, Y = make_windows(trajs, args.stride)
    print(f"train: {len(trajs)} trajectories -> {len(X)} windows (refiner)")
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
            loss = refiner_loss(model, to_ch(X[idx].to(dev)), to_ch(Y[idx].to(dev)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot/n:.5f}  ({time.time()-t0:.0f}s)")

    torch.save({"model_state_dict": model.state_dict(), "mode": "refiner"}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
