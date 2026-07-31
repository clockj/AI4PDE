"""WP1 + WP2 training: relative-L2 objective, no z slot, optional variance head.

Implements Track_1/plan/plan_20260729.md WP1 (loss/training alignment) and WP2 (two-head
model). One script covers the whole ablation ladder; flags select the rung:

    python train2.py --out wp1_s5.pth                          # WP1 only
    python train2.py --out wp1_s2.pth --stride 2               # WP1 + denser windows
    python train2.py --out wp2.pth --stride 2 --var-head       # WP1 + WP2

Trained on real data minus VAL_TRAJ, so every number stays honestly held out. Add
--all-data only when building a submission checkpoint.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from model import (NORM_STATS, T_IN, T_OUT, UNetTwoHead, VAL_TRAJ, compose, from_ch,
                   load_trajectories, rel_l2, self_norm, std_t, tke, to_ch, window_index,
                   window_scale)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/clockj/RealPDE/train_real/train_real")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--var-head", action="store_true", help="WP2: add the log-std head")
    ap.add_argument("--loss", choices=["rel", "mse"], default="rel",
                    help="field loss. 'rel' = per-sample relative L2 in physical units (WP1.1); "
                         "'mse' = plain MSE in normalized units, as the original train.py. The "
                         "leaderboard says 'rel' costs 1.28 mvpe_score, so 'mse' is the default "
                         "choice for submissions -- see plan_20260730.md WP-A.")
    ap.add_argument("--alpha-train", type=float, default=1.0,
                    help="fluctuation rescaling used INSIDE the field loss (var-head only)")
    ap.add_argument("--lam-v", type=float, default=1.0, help="weight on the log-std loss")
    ap.add_argument("--lam-t", type=float, default=0.5, help="weight on the TKE loss")
    ap.add_argument("--detach-var", action="store_true",
                    help="stop the variance losses from back-propagating into the shared "
                         "trunk. The head still READS the mean-flow latent, so the physical "
                         "conditioning is kept; only the reshaping of the trunk is blocked.")
    ap.add_argument("--aug-scale", type=float, default=0.0,
                    help="random freestream amplitude scaling, log-uniform in "
                         "[1/(1+s), 1+s]. 0 disables.")
    ap.add_argument("--aug-noise", type=float, default=0.0,
                    help="Gaussian noise added to the INPUT window, in units of the target's "
                         "own per-window fluctuation std. 0 disables.")
    ap.add_argument("--norm", choices=["real", "sim", "self"], default="real",
                    help="'real'/'sim' = fixed global GaussianNormalizer statistics for that "
                         "dataset. 'self' = WP-C, divide each window by its own mean streamwise "
                         "speed and work non-dimensionally, so an unseen Reynolds number is not "
                         "an unseen input. Use --norm sim for train_sim: its scale is ~6x real's "
                         "and the real constants would put the input at 7.3 sigma.")
    ap.add_argument("--all-data", action="store_true",
                    help="train on ALL trajectories including the held-out ones. Submission "
                         "checkpoints only -- it makes local evaluation meaningless.")
    args = ap.parse_args()

    dev = "cuda"
    # Self-normalization needs raw physical trajectories: the per-window scale is only
    # recoverable before the global normalizer folds every condition onto one distribution.
    self_norm_mode = args.norm == "self"
    trajs = load_trajectories(args.data, exclude=[] if args.all_data else VAL_TRAJ,
                              normalize=False)
    idx = window_index(trajs, args.stride)
    held = "NONE (all-data)" if args.all_data else f"{len(VAL_TRAJ)} held out"
    print(f"train: {len(trajs)} trajectories -> {len(idx)} windows, {held}", flush=True)
    print(f"config: {vars(args)}", flush=True)

    # Trajectories stay on the CPU (~1 GB); windows are sliced per batch. Materializing them
    # would need ~21 GB at stride 2.
    trajs_t = [torch.from_numpy(d) for d in trajs]
    idx_t = torch.from_numpy(idx)

    stats = NORM_STATS[args.norm]
    mean_v = torch.from_numpy(stats[0]).to(dev)
    std_v = torch.from_numpy(stats[1]).to(dev)

    model = UNetTwoHead(args.width, var_head=args.var_head).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * ((len(idx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    def fetch(sel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([trajs_t[i][t:t + T_IN] for i, t in sel.tolist()])
        ys = torch.stack([trajs_t[i][t + T_IN:t + T_IN + T_OUT] for i, t in sel.tolist()])
        return xs.to(dev, non_blocking=True), ys.to(dev, non_blocking=True)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(idx_t))
        acc = np.zeros(4)
        n = 0
        for i in range(0, len(idx_t), args.batch):
            sel = idx_t[perm[i:i + args.batch]]
            x, y = fetch(sel)
            b = len(sel)

            # `mul` converts a normalized difference or std to physical units; `off` completes
            # the affine map for absolute fields. Under self-normalization both carry the
            # per-window scale, so every physical-unit loss below works unchanged.
            s = window_scale(x) if self_norm_mode else torch.ones(1, device=dev)
            x, y = self_norm(x, s, mean_v, std_v), self_norm(y, s, mean_v, std_v)
            mul, off = std_v * s, mean_v * s

            if args.aug_scale > 0.0:
                # Rescale the freestream. In physical units this is u -> a*u, which is a
                # different Reynolds number to first order -- exactly the shift the hidden
                # test set makes (it contains Re 2968, below the whole training range).
                # Data is stored normalized, so a*(d*STD + MEAN) - MEAN, all over STD.
                lo = np.log(1.0 / (1.0 + args.aug_scale))
                a = torch.exp(torch.empty(b, 1, 1, 1, 1, device=dev).uniform_(lo, -lo))
                shift = (a - 1.0) * (mean_v / std_v)
                x = a * x + shift
                y = a * y + shift
            if args.aug_noise > 0.0:
                scale = args.aug_noise * std_t(y).mean((2, 3), keepdim=True)
                x = x + scale * torch.randn_like(x)

            raw, logstd = model(to_ch(x), detach_var=args.detach_var)
            raw = from_ch(raw)

            if args.var_head:
                # (B, C, H, W) -> (B, 1, H, W, C); clamp keeps exp() finite early in training.
                sigma = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)
                pred = compose(raw, sigma.detach(), args.alpha_train)
            else:
                sigma, pred = None, raw

            if args.loss == "rel":
                # Per-sample relative L2 in PHYSICAL units -- exactly the scored quantity.
                # Doing this in normalized units would be a different objective: the normalized
                # denominator is dominated by the constant MEAN/STD offset, which erases the
                # low-Re up-weighting that is the whole point of WP1.1.
                num = ((pred - y) * mul).reshape(b, -1).norm(dim=1)
                den = (y * mul + off).reshape(b, -1).norm(dim=1).clamp(min=1e-8)
                l_field = (num / den).mean()
            else:
                l_field = torch.mean((pred - y) ** 2)

            loss = l_field
            l_var = l_tke = torch.zeros((), device=dev)
            if args.var_head:
                s_true = std_t(y)
                l_var = torch.mean((sigma.log() - (s_true + 1e-4).log()) ** 2)
                # With alpha = 1 the composed field has std exactly `sigma`, so its TKE is
                # 0.5*sum_c (sigma_c * STD_c)^2 identically. Writing it that way instead of
                # calling tke(compose(...)) keeps the gradient on the variance head alone --
                # routing it through the field head would only push the raw fluctuation's
                # scale, which the rescaling then discards.
                ke_pred = 0.5 * ((sigma * mul) ** 2).squeeze(1).sum(-1)
                l_tke = rel_l2(ke_pred, tke(y, mul)).mean()
                loss = loss + args.lam_v * l_var + args.lam_t * l_tke

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc += np.array([loss.item(), l_field.item(), l_var.item(), l_tke.item()]) * b
            n += b

        if ep % 5 == 0 or ep == args.epochs - 1:
            a = acc / n
            print(f"  epoch {ep:3d}  loss {a[0]:.5f}  field {a[1]:.5f}  "
                  f"var {a[2]:.5f}  tke {a[3]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model_state_dict": model.state_dict(), "mode": "twohead",
                "width": args.width, "var_head": args.var_head,
                "self_norm": self_norm_mode, "norm": args.norm,
                "mean": stats[0], "std": stats[1], "config": vars(args)}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
