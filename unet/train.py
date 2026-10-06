"""Train one network of the submission: an F-FNO or a ViT with a field head and a per-pixel
log-standard-deviation head, on the real PIV windows and (optionally) on the mapped CFD windows.

    python train.py --data <train_real> --arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 \
        --dropout 0.3 --sim <train_sim> --sim-exclude-held --seed 1 --out twin_s1.pth     # a twin
    ... the same with --all-data and without --sim-exclude-held                            # a member

A MEMBER trains on every real trajectory; a TWIN holds out the eight trajectories of
`data.VAL_TRAJ` (the band is fitted on the twins' errors there). The eleven exact command lines
of the submission are in `solution.py` (`RECIPES`).

The loss, per step, on a real batch of windows (normalised by `data.MEAN` / `data.STD`):

    lam_f    * MSE of the field (time-mean and fluctuation parts weighted equally)
  + lam_cov  * Welsch (redescending) loss of the same residual, knee at cov_c * sigma_SPS
  + lam_grad * relative L2 of the velocity-gradient tensor,  physical units
  + lam_tgrad* relative L2 of the frame-to-frame time derivative, physical units
  + lam_v    * MSE of log(sigma) against log(std_t(y)), the target window's temporal std
  + lam_t    * relative L2 of the turbulent kinetic energy implied by sigma
  + lam_sim  * MSE of the field on one batch of mapped CFD windows (with --sim)

AdamW with a one-cycle schedule peaking at --lr. `--seed` fixes the initialisation and the
batch order; on a given GPU and software stack the same seed gives the same weights for many
epochs, across hardware the weights differ at the half-percent level (F-FNO) and the ensemble's
scores at the seed level. `--ckpt-every N --resume` makes a run restartable at epoch boundaries
(model, optimiser, schedule and RNG states), for pre-emptible queues.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

from data import (C, CH, H, MEAN, STD, T_IN, T_OUT, VAL_TRAJ, W, from_ch, load_trajectories,
                  rel_l2, std_t, tke, to_ch, window_index)

# The SPS scorer's global half-width scale (scoring.SIGMA_GLOBAL), physical units: --cov-c is
# quoted in units of it.
SPS_SIGMA = 0.0563870259


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the train_real directory of the release")
    ap.add_argument("--out", required=True, help="checkpoint to write")
    ap.add_argument("--arch", choices=["ffno", "vit"], required=True)
    ap.add_argument("--width", type=int, required=True, help="F-FNO channels, or the ViT token dimension")
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--modes-y", type=int, default=16, help="F-FNO: retained ky modes")
    ap.add_argument("--modes-x", type=int, default=32, help="F-FNO: retained kx modes")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5, help="window start stride over each trajectory")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=-1, help="-1 = unseeded")
    ap.add_argument("--all-data", action="store_true",
                    help="train on every trajectory (a member); otherwise data.VAL_TRAJ is held out (a twin)")
    ap.add_argument("--lam-f", type=float, default=0.0, help="weight of the MSE field loss")
    ap.add_argument("--lam-cov", type=float, default=2.0, help="weight of the Welsch field loss")
    ap.add_argument("--cov-c", type=float, default=0.25, help="its knee, in units of the SPS sigma")
    ap.add_argument("--lam-grad", type=float, default=1.0)
    ap.add_argument("--lam-tgrad", type=float, default=0.3)
    ap.add_argument("--lam-v", type=float, default=0.15)
    ap.add_argument("--lam-t", type=float, default=0.15)
    ap.add_argument("--sim", default="", help="the train_sim directory of the release: adds the CFD term")
    ap.add_argument("--lam-sim", type=float, default=0.3)
    ap.add_argument("--sim-stride", type=int, default=5)
    ap.add_argument("--sim-exclude-held", action="store_true",
                    help="a twin: also drop the held-out conditions from the CFD pool")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="write <out>.resume.pt every N epochs (0 = never)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.resume.pt if it exists (a no-op when it does not)")
    args = ap.parse_args()

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    dev = "cuda"
    excl = [] if args.all_data else VAL_TRAJ
    trajs = load_trajectories(args.data, exclude=excl)
    idx = window_index(trajs, args.stride)
    held = "NONE (all-data)" if args.all_data else f"{len(excl)} held out"
    print(f"train: {len(trajs)} trajectories -> {len(idx)} windows, {held}", flush=True)
    print(f"config: {vars(args)}", flush=True)

    # The trajectories stay on the CPU (~1 GB); windows are sliced per batch.
    trajs_t = [torch.from_numpy(d) for d in trajs]
    idx_t = torch.from_numpy(idx)

    sim_trajs_t = sim_idx_t = None
    if args.sim:
        from simreal import load_sim_mapped
        sim_trajs = load_sim_mapped(args.sim, args.data,
                                    exclude=excl if args.sim_exclude_held else None,
                                    real_exclude=excl)
        sim_idx = window_index(sim_trajs, args.sim_stride)
        sim_trajs_t = [torch.from_numpy(d) for d in sim_trajs]
        sim_idx_t = torch.from_numpy(sim_idx)
        print(f"sim: {len(sim_idx)} windows at stride {args.sim_stride}, lam_sim {args.lam_sim}",
              flush=True)

    mean_v = torch.from_numpy(MEAN).to(dev)
    std_v = torch.from_numpy(STD).to(dev)

    from models_alt import build, time_derivative, velocity_gradient
    model = build(args.arch, args.width, True, 0, args.modes_y, args.modes_x, args.n_layers,
                  dropout=args.dropout).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(idx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    def fetch(sel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([trajs_t[i][t:t + T_IN] for i, t in sel.tolist()])
        ys = torch.stack([trajs_t[i][t + T_IN:t + T_IN + T_OUT] for i, t in sel.tolist()])
        return xs.to(dev, non_blocking=True), ys.to(dev, non_blocking=True)

    # ---- resume: the weights, the optimiser, the schedule and the RNG states at an epoch boundary
    RESUME = (args.out + ".resume.pt") if args.ckpt_every or args.resume else None
    start_ep, elapsed0 = 0, 0.0
    if args.resume and RESUME and os.path.exists(RESUME):
        ck = torch.load(RESUME, map_location="cpu")
        if int(ck["epochs"]) != int(args.epochs):
            raise SystemExit(f"--resume: {RESUME} was written by a run with --epochs {ck['epochs']}, "
                             f"this one asks for {args.epochs}; the one-cycle schedule depends on it")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng_cpu"].cpu().to(torch.uint8))
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng_cuda"]])
        np.random.set_state(ck["rng_np"])
        start_ep, elapsed0 = int(ck["epoch"]), float(ck.get("elapsed", 0.0))
        print(f"resume: {RESUME} -> epoch {start_ep}/{args.epochs}, {elapsed0:.0f}s already spent",
              flush=True)
    elif args.resume:
        print(f"resume: no checkpoint at {RESUME}; starting from scratch", flush=True)

    def save_resume(ep_done):
        tmp = RESUME + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep_done, "epochs": args.epochs,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "rng_np": np.random.get_state(),
                    "elapsed": elapsed0 + time.time() - t0}, tmp)
        os.replace(tmp, RESUME)

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        perm = torch.randperm(len(idx_t))
        acc = np.zeros(7)
        n = 0
        for i in range(0, len(idx_t), args.batch):
            sel = idx_t[perm[i:i + args.batch]]
            x, y = fetch(sel)
            b = len(sel)
            x, y = (x - mean_v) / std_v, (y - mean_v) / std_v
            mul = std_v                           # normalised -> physical, for the physical-unit terms

            raw, logstd = model(to_ch(x), detach_var=False)
            pred = from_ch(raw)
            # (B, C, H, W) -> (B, 1, H, W, C); the clamp keeps exp() finite early in training.
            sigma = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)

            # The residual split into its time-mean and fluctuation parts (orthogonal in time).
            d = pred - y
            dm = d.mean(1, keepdim=True)
            df = d - dm
            l_field = torch.mean(dm ** 2) + torch.mean(df ** 2)
            loss = args.lam_f * l_field

            l_cov = torch.zeros((), device=dev)
            if args.lam_cov > 0.0:
                # Welsch: cn^2 * (1 - exp(-0.5 (d/cn)^2)), cn the physical knee in normalised units;
                # equal to 0.5 * MSE for small residuals, flat for large ones.
                cn = (args.cov_c * SPS_SIGMA) / mul
                l_cov = torch.mean(cn ** 2 * (1.0 - torch.exp(-0.5 * (d / cn) ** 2)))
                loss = loss + args.lam_cov * l_cov
            l_grad = torch.zeros((), device=dev)
            if args.lam_grad > 0.0:
                l_grad = rel_l2(velocity_gradient(pred * mul), velocity_gradient(y * mul)).mean()
                loss = loss + args.lam_grad * l_grad
            l_tg = torch.zeros((), device=dev)
            if args.lam_tgrad > 0.0:
                l_tg = rel_l2(time_derivative(pred * mul), time_derivative(y * mul)).mean()
                loss = loss + args.lam_tgrad * l_tg
            # The sigma head: log-MSE against the target window's temporal std, and the relative
            # L2 of the kinetic energy it implies (0.5 * sum_c (sigma_c * STD_c)^2) against the
            # target's, in physical units.
            l_var = torch.mean((sigma.log() - (std_t(y) + 1e-4).log()) ** 2)
            ke_pred = 0.5 * ((sigma * mul) ** 2).squeeze(1).sum(-1)
            l_tke = rel_l2(ke_pred, tke(y, mul)).mean()
            loss = loss + args.lam_v * l_var + args.lam_t * l_tke

            l_sim = torch.zeros((), device=dev)
            if sim_idx_t is not None and args.lam_sim > 0.0:
                # One batch of mapped CFD windows per real step, drawn with replacement.
                sj = torch.randint(len(sim_idx_t), (args.batch,))
                ss = sim_idx_t[sj]
                xs_ = torch.stack([sim_trajs_t[i][t:t + T_IN] for i, t in ss.tolist()])
                ys_ = torch.stack([sim_trajs_t[i][t + T_IN:t + T_IN + T_OUT]
                                   for i, t in ss.tolist()])
                xs_ = ((xs_.to(dev, non_blocking=True) - mean_v) / std_v)
                ys_ = ((ys_.to(dev, non_blocking=True) - mean_v) / std_v)
                ps_ = from_ch(model(to_ch(xs_), detach_var=False)[0])
                dd = ps_ - ys_
                ddm = dd.mean(1, keepdim=True)
                l_sim = torch.mean(ddm ** 2) + torch.mean((dd - ddm) ** 2)
                loss = loss + args.lam_sim * l_sim

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc += np.array([t.item() for t in (loss, l_field, l_cov, l_grad + l_tg, l_var, l_tke, l_sim)]) * b
            n += b

        if ep % 5 == 0 or ep == args.epochs - 1:
            a = acc / n
            print(f"  epoch {ep:3d}  loss {a[0]:.5f}  field {a[1]:.5f}  cov {a[2]:.5f}  "
                  f"grad {a[3]:.5f}  var {a[4]:.5f}  tke {a[5]:.5f}  sim {a[6]:.5f}  "
                  f"({elapsed0 + time.time() - t0:.0f}s)", flush=True)
        if RESUME and args.ckpt_every and (ep + 1) % args.ckpt_every == 0:
            save_resume(ep + 1)

    # The checkpoint. The keys next to the weights are the ones submission/_fno.py's `load_any`
    # reads to rebuild the module; `config` records the command line.
    torch.save({"model_state_dict": model.state_dict(), "mode": "twohead",
                "width": args.width, "var_head": True, "var_mlp": False, "lift_mlp": False,
                "point_mlp": False, "err_head": False, "var_hidden": 128, "var_deep": False,
                "var_in_std": False, "var_pred_std": False, "fgn_dim": 0, "hist": False,
                "mean_head": False, "osc_modes": 0, "osc_global": False, "osc_zeromean": False,
                "tbasis": 0, "ctx": T_IN, "arch": args.arch, "modes_y": args.modes_y,
                "modes_x": args.modes_x, "n_layers": args.n_layers, "p": 256,
                "self_norm": False, "norm": "real", "mean": MEAN, "std": STD,
                "config": {**vars(args), "argv": " ".join(sys.argv)}},
               args.out)
    print(f"saved {args.out}  ({elapsed0 + time.time() - t0:.0f}s, peak GPU memory "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
