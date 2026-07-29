"""Train the plain-MSE U-Net -- currently the best Track 1 model (83.83 on held-out).

Deterministic regression: one forward pass, no noise, no sampling, no refinement.
Real data only (train_real minus the held-out trajectories); no simulation pretraining.

    python train.py --epochs 40
"""
from __future__ import annotations

import argparse
import time

import torch

from model import UNet, VAL_TRAJ, load_trajectories, make_windows, to_ch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/clockj/RealPDE/train_real/train_real")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="unet_mse.pth")
    ap.add_argument("--all-data", action="store_true",
                    help="train on ALL trajectories including the held-out ones. Use for the "
                         "submission checkpoint only -- it makes local evaluation meaningless, "
                         "so keep the held-out model for every reported comparison.")
    args = ap.parse_args()

    dev = "cuda"
    trajs = load_trajectories(args.data, exclude=[] if args.all_data else VAL_TRAJ)
    X, Y = make_windows(trajs, args.stride)
    held = "NONE (all-data submission model)" if args.all_data else f"{len(VAL_TRAJ)} held out"
    print(f"train: {len(trajs)} trajectories -> {len(X)} windows (real data only), {held}")
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
            k0 = torch.zeros(len(idx), dtype=torch.long, device=dev)
            loss = torch.mean((model(prev, torch.zeros_like(tgt), k0) - tgt) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  mse {tot/n:.5f}  ({time.time()-t0:.0f}s)")

    torch.save({"model_state_dict": model.state_dict(), "mode": "mse"}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
