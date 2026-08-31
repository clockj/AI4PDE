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

from model import (C, H, HIST_CH, NORM_STATS, PHASES_ALL, T_IN, T_OUT, UNetTwoHead, VAL_TRAJ, W,
                   compose, from_ch, history_stats, load_trajectories, norm_history, rel_l2,
                   self_norm, std_t, tke, to_ch, trajectory_names, window_index, window_scale)

# The scorer's wake probes on the 32x64 grid, derived in scoring.mvpe_rel_l2_per_sample from
# d=16, center=(10, 32), n_probe=9, sub_s=2. Hard-coded because the geometry is fixed for the
# season and re-deriving it per batch would only invite a mismatch with the scorer.
PROBE_Y = [8, 10, 12, 14, 16, 18, 20, 22, 24]
PROBE_X = [13, 21, 29, 37]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/users/PAS1187/qwu/RealPDE/train_real/train_real")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--var-head", action="store_true", help="WP2: add the log-std head")
    ap.add_argument("--var-mlp", action="store_true",
                    help="give the log-std head the same 128-wide MLP the field head has. "
                         "Without it the variance readout is a single 1x1 conv -- a linear "
                         "probe on a 24-channel trunk -- while the field head gets two layers. "
                         "tke_branch_v9.py prices closing 25%% of the sigma-to-oracle gap at "
                         "+1.2 final, so the variance head is worth spending parameters on.")
    ap.add_argument("--lift-mlp", action="store_true",
                    help="give the input lift a hidden layer. By default it is a single 1x1 "
                         "conv, i.e. a LINEAR map from 42 channels (20 frames x u,v, plus the "
                         "coordinate grid) down to width=24 -- the whole input window is "
                         "compressed linearly before the first spectral block sees it. Same "
                         "shape of fix as --var-mlp, at the other end of the network.")
    ap.add_argument("--point-mlp", action="store_true",
                    help="give the pointwise half of each FNO block a hidden layer. `point` is "
                         "nn.Conv2d(width, width, 1) -- linear -- and there are n_layers of "
                         "them, so it is the same 1x1 bottleneck as --lift-mlp and --var-mlp, "
                         "repeated. Those two are worth +0.086 and +0.076 and are additive.")
    ap.add_argument("--var-hidden", type=int, default=128,
                    help="hidden width of the --var-mlp log-std head. Boundary capacity, which "
                         "is the side of the boundary/trunk rule that has paid every time: "
                         "lift +0.076 and head_logstd +0.086, against --point-mlp -0.379 and "
                         "every width increase inside the trunk.")
    ap.add_argument("--var-deep", action="store_true",
                    help="feed the log-std head concat(h1..hL) instead of the last block's "
                         "output alone. Still a readout -- no trunk capacity is added -- but "
                         "the amplitude envelope is multi-scale and the final trunk state has "
                         "had four blocks to specialize toward the field head.")
    ap.add_argument("--var-in-std", action="store_true",
                    help="feed the log-std head the INPUT window's own per-pixel temporal std "
                         "(2 channels) alongside the trunk state. Unlike --var-hidden and "
                         "--var-deep this adds INFORMATION, not capacity: the trunk would have "
                         "to form a quadratic of 20 frames to recover it, and the variance "
                         "loss is a log-MSE against exactly that statistic of the TARGET "
                         "window. Uses only the window predict() is handed -- no extra frames "
                         "and no cross-window read, so it stays deployable.")
    ap.add_argument("--var-pred-std", action="store_true",
                    help="feed the log-std head the temporal std of the FIELD HEAD's own "
                         "predicted window (detached). compose() forms "
                         "(alpha*sigma + (1-alpha)*s)/s with exactly that s, so without this "
                         "the head predicts sigma in absolute terms and the answer is divided "
                         "by a quantity it never saw. --var-hidden 256 showed the head is not "
                         "capacity-limited (training `var` identical to baseline), so the "
                         "remaining lever is information, not width.")
    ap.add_argument("--fgn", type=int, default=0,
                    help="WP-1 (FGN, arXiv 2506.10772): number of NOISE SAMPLES drawn per step. "
                         "0 is off and leaves training byte-for-byte as it was. Each sample is "
                         "one extra forward pass through the SAME weights with a different z "
                         "driving the per-block FiLM, i.e. one draw of a perturbed network -- "
                         "noise in PARAMETER space, not output space, which is what makes the "
                         "samples spatially coherent when the loss only sees marginals. The "
                         "z = 0 pass is bit-identical to the plain FNO (film is zero-init and "
                         "bias-free), so every existing loss is computed on it unchanged and "
                         "this only ADDS the CRPS term. 4 is the screen value: the fair CRPS "
                         "estimator needs K >= 2 and its variance falls slowly past 4.")
    ap.add_argument("--fgn-dim", type=int, default=32,
                    help="width of the FGN noise vector. 32 is the paper's. The FiLM maps are "
                         "Linear(fgn_dim, 2*width, bias=False) -- 1 536 parameters per block at "
                         "width 24 -- which is BOUNDARY capacity, the side of the "
                         "boundary/trunk rule that has paid every time (lift +0.076, "
                         "head_logstd +0.086, against --point-mlp -0.379).")
    ap.add_argument("--lam-crps", type=float, default=0.3,
                    help="weight on the fair (unbiased) CRPS of the K sample fields against the "
                         "target, per element, in NORMALIZED units. CRPS is a proper scoring "
                         "rule for exactly what sps measures -- whether an interval contains "
                         "the truth -- and for the spread tke needs; log-MSE sigma is not. "
                         "Round 11 showed lam_v REGULARIZES the TKE objective rather than "
                         "competing with it (lam_v 0.05 / 0.3 / 0.6 -> held-out TKE 0.506 / "
                         "0.482 / 0.487, a U-shape with the shipped value at the bottom), so "
                         "the remaining lever is the objective's SHAPE, not its weight.")
    ap.add_argument("--loss", choices=["rel", "mse"], default="rel",
                    help="field loss. 'rel' = per-sample relative L2 in physical units (WP1.1); "
                         "'mse' = plain MSE in normalized units, as the original train.py. The "
                         "leaderboard says 'rel' costs 1.28 mvpe_score, so 'mse' is the default "
                         "choice for submissions -- see plan_20260730.md WP-A.")
    ap.add_argument("--alpha-train", type=float, default=1.0,
                    help="fluctuation rescaling used INSIDE the field loss (var-head only)")
    ap.add_argument("--lam-f", type=float, default=1.0,
                    help="weight on the FIELD loss. 0 trains a variance-only network, to be "
                         "paired at inference with a separately trained field model. That is "
                         "the fully decoupled option: the field head is then bit-identical to "
                         "the no-head control by construction, so the shared trunk cannot cost "
                         "MVPE (report_20260730.md section 9 measured that cost at 0.004 "
                         "locally and ~1.3 on the hidden set).")
    ap.add_argument("--lam-v", type=float, default=1.0, help="weight on the log-std loss")
    ap.add_argument("--lam-t", type=float, default=0.5, help="weight on the TKE loss")
    ap.add_argument("--w-mean", type=float, default=1.0,
                    help="weight on the TIME-MEAN half of the field loss. Time-mean and "
                         "fluctuation are orthogonal under the time inner product, so the field "
                         "loss splits exactly into the two and 1.0 reproduces it unchanged. "
                         "diag_spectral.py measures 68%% of the squared error sitting in the "
                         "fluctuation. The 'only ~31%% of it is coherently predictable' that "
                         "used to be quoted here is a CATEGORY ERROR (round 13): that Gamma^2 "
                         "is the coherence of OUR prediction with the target, i.e. what a "
                         "round-3 model achieved, not what is achievable. >1 reallocates "
                         "gradient to the time-mean (tested: 3 and 8, both null); <1 "
                         "reallocates it to the fluctuation, which the eight-round re-pricing "
                         "of oracle_split_v9 says is the SECOND largest lever (+2.96 against "
                         "the time-mean's +1.61) and which has never been tried.")
    ap.add_argument("--lam-p", type=float, default=0.0,
                    help="weight on an explicit MVPE term: the scorer's own wake-probe "
                         "quantity (4 columns x 9 rows, time-averaged u,v, per-column relative "
                         "L2). 36 points carry a fifth of the final score.")
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
    ap.add_argument("--ctx", type=int, default=T_IN,
                    help="number of input frames. The default 20 is the protocol's window, but "
                         "diag_analog.py measures the shedding period at 60-75 frames, so 20 "
                         "frames is under a third of a cycle -- too little to fix the phase or "
                         "the frequency. Larger values prepend earlier frames of the same "
                         "trajectory as extra channels; those frames are INPUTS of previous "
                         "windows, so this reads no target.")
    ap.add_argument("--hist", action="store_true",
                    help="condition on the causal trajectory history: 4 extra input channels "
                         "carrying the per-pixel mean and std of every frame up to and "
                         "including this window. Legal transductive inference -- predict() "
                         "receives all test windows and they tile each trajectory, so those "
                         "frames are inputs. See model.history_stats.")
    ap.add_argument("--swa", type=int, default=0,
                    help="average the weights of the last N epochs (Izmailov et al. 2018) and "
                         "save that as the checkpoint. A 3-seed prediction ensemble is worth "
                         "+2.0 subscore on real but costs 3x latency; weight averaging chases "
                         "the same variance reduction at 1x. Safe here because the network uses "
                         "GroupNorm, which carries no running statistics to recompute.")
    ap.add_argument("--arch", choices=["unet", "fno", "deeponet"], default="unet")
    ap.add_argument("--modes-y", type=int, default=12, help="FNO: retained ky modes")
    ap.add_argument("--modes-x", type=int, default=16, help="FNO: retained kx modes")
    ap.add_argument("--n-layers", type=int, default=4, help="FNO: spectral blocks")
    ap.add_argument("--p", type=int, default=256, help="DeepONet: basis rank")
    ap.add_argument("--lam-div", type=float, default=0.0,
                    help="physics: match the TARGET's divergence field, not zero. divergence.py "
                         "measures rms(div u)/rms(|grad u|) at 0.42-0.49 on real PIV and "
                         "0.28-0.34 on the CFD, so a div->0 penalty would fight the data; the "
                         "only defensible form is matching what the target actually has.")
    ap.add_argument("--lam-vort", type=float, default=0.0,
                    help="physics: match the target's VORTICITY field. For a vortex-shedding "
                         "wake this, not the divergence, is the coherent structure -- the whole "
                         "signal is a train of alternating vortices. Divergence matching is "
                         "worth +0.58 on FNO; this asks whether the rotational half of the "
                         "velocity gradient is worth at least as much.")
    ap.add_argument("--lam-grad", type=float, default=0.0,
                    help="physics: match all four components of grad(u). Divergence and "
                         "vorticity are two invariant combinations of that tensor, so this "
                         "subsumes both --lam-div and --lam-vort and adds the strain rates.")
    ap.add_argument("--lam-tgrad", type=float, default=0.0,
                    help="match the TARGET's frame-to-frame time derivative, in physical "
                         "units -- the temporal analogue of --lam-grad, which is the only "
                         "loss term ever to clear the proxy's ~2%% resolution. Nothing else "
                         "in the objective couples adjacent output frames.")
    ap.add_argument("--lam-spec", type=float, default=0.0,
                    help="physics: match the radially-averaged energy spectrum of the "
                         "fluctuation. Targets the measured amplitude deficit at high k "
                         "(gain 0.54 at k=32, report_20260731.md section 1.1) without "
                         "requiring phase agreement, which is unreachable there.")
    ap.add_argument("--init", default="",
                    help="warm-start the weights from this checkpoint instead of from random. "
                         "The architecture flags must match it. This is what makes the two-stage "
                         "sim -> real route possible: stage 1 trains on train_sim with --norm "
                         "sim, stage 2 fine-tunes here on train_real with --norm real and a "
                         "smaller --lr. Sim-ONLY training scores 69.0, below free persistence, "
                         "but that measures zero-shot transfer, not pretraining -- the official "
                         "sim_real_ft checkpoints get all their real skill from stage 2. Only "
                         "the shared tensors are copied, so a normalizer or head shape change "
                         "does not block the warm start.")
    ap.add_argument("--chan-pow", type=float, default=0.0,
                    help="re-weight the MSE field loss by (std_c/max std)^chan_pow. 0 is the "
                         "old behaviour, equal weight per channel in NORMALIZED units, which "
                         "weights v ~37x more than u per unit of physical error. 1 puts the "
                         "loss in physical units, where u carries 86%% of the scored error "
                         "energy. No effect on --loss rel, which is already physical.")
    ap.add_argument("--phases", type=int, default=1, choices=[1, 4],
                    help="number of ::2 decimation phases to train on. 1 is the scored grid "
                         "alone, which is what every checkpoint before round 10 used and what "
                         "the raw 64x128 frames were being thrown away for. 4 adds the three "
                         "half-cell-shifted views: real samples of the same PIV field, so this "
                         "is not the input-noise augmentation that was already falsified. "
                         "Windows and therefore gradient steps per epoch go up 4x.")
    ap.add_argument("--aoa-boost", type=float, default=1.0,
                    help="WP3: sampling weight for windows at --aoa-hi angles of attack, "
                         "relative to 1.0 for the rest. Error is governed by AoA (0 deg 0.057, "
                         "5 deg 0.044, 10 deg 0.073, 15 deg 0.103, 20 deg 0.105) and the board "
                         "reads 0.113, at or above our hardest single trajectory -- so equal "
                         "weighting spends most of the gradient on regimes the board barely "
                         "tests. Sampling becomes with-replacement; steps per epoch are "
                         "unchanged. 1.0 keeps the plain shuffle, bit for bit.")
    ap.add_argument("--aoa-hi", type=int, nargs="+", default=[15, 20],
                    help="the angles --aoa-boost and --aoa-only refer to")
    ap.add_argument("--aoa-only", action="store_true",
                    help="WP3(c): drop every window outside --aoa-hi, i.e. train a high-AoA "
                         "specialist to be blended with the generalist rather than reweighting "
                         "one model. Held-out trajectories are excluded first, as always.")
    ap.add_argument("--mean-head", action="store_true",
                    help="a decoupled time-mean readout: head_field emits the (zero-mean) "
                         "fluctuation and a separate 128-wide MLP emits mean_t(y). The "
                         "time-mean carries 29%% of rel-L2's error energy and the bound that "
                         "closed it in round 10 does not bound it -- see floor_aoa.py.")
    ap.add_argument("--osc-modes", type=int, default=0,
                    help="J oscillatory output modes, added to head_field. The shedding "
                         "period is 60-75 frames against T_IN = T_OUT = 20, so the model sees "
                         "a THIRD of a cycle and must extrapolate the next third -- and the "
                         "measured error is exactly that: amplitude is right (gain 0.95-1.0) "
                         "while coherence at k = 2-6, 57%% of the fluctuation energy, is only "
                         "0.55-0.79. This head emits (p, q, omega) per pixel per channel per "
                         "mode and synthesizes sum p cos(wt) - q sin(wt), building in the "
                         "phase advance that is awkward for a conv net. Zero-initialized, so "
                         "training starts at the plain FNO; 4.7k parameters at J = 2.")
    ap.add_argument("--osc-global", action="store_true",
                    help="one omega for the whole field per mode (pooled from the trunk) "
                         "instead of one per pixel. A vortex street has ONE shedding "
                         "frequency per trajectory, so this is the constraint that carries "
                         "the physics; a per-pixel omega lets the head fit each pixel's "
                         "20-frame trace independently, which is the memorization failure "
                         "mode the arm exists to avoid. (p, q) stay per pixel either way.")
    ap.add_argument("--osc-zeromean", action="store_true",
                    help="subtract the oscillatory term's own time-mean before adding it. "
                         "The learned period is 64.5 frames against a 20-frame window, so the "
                         "term spans 0.31 of a cycle and carries a large non-zero time-mean by "
                         "construction -- and the measured gain was ENTIRELY in the time-mean, "
                         "with coherence unmoved. This isolates the oscillatory content: if "
                         "the gain survives the phase claim is real, if it vanishes the head "
                         "was a mean-head in disguise. Note --mean-head already zero-means "
                         "everything head_field emits, so it implies this.")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout2d probability between spectral blocks. Adds no parameters, "
                         "so a checkpoint trained with it still loads into the submission's "
                         "own FNO2d, which has none and runs in eval() regardless.")
    ap.add_argument("--wd", type=float, default=1e-5,
                    help="AdamW weight decay. 1e-5 -- effectively none -- was hardcoded for "
                         "twelve rounds and never swept, while every capacity experiment in "
                         "the project said the model MEMORIZES 74 trajectories. Explicit "
                         "regularization is the textbook response to that regime and is the "
                         "one family it was never tried on.")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="held-out trajectory stems, overriding model.VAL_TRAJ. The default "
                         "split holds out 8 trajectories drawn from the INTERIOR of the Re "
                         "range, so it cannot see Re-extrapolation behaviour -- and the hidden "
                         "test set is at unseen Re. Pass a Re-extreme split to measure that.")
    ap.add_argument("--all-data", action="store_true",
                    help="train on ALL trajectories including the held-out ones. Submission "
                         "checkpoints only -- it makes local evaluation meaningless.")
    args = ap.parse_args()

    # Store the AS-TRAINED noise width: 0 when no samples are drawn, so a checkpoint's own blob
    # says whether it carries `film` weights and eval never has to infer it from --fgn.
    if args.fgn <= 0:
        args.fgn_dim = 0
    elif args.fgn_dim <= 0:
        raise SystemExit("--fgn > 0 needs --fgn-dim > 0; z would otherwise have no width")
    elif args.arch != "fno":
        # Only FNO2d takes a `z`; UNetTwoHead and DeepONetCh would raise deep inside the loop.
        raise SystemExit(f"--fgn is implemented for --arch fno only (got {args.arch})")

    dev = "cuda"
    # Self-normalization needs raw physical trajectories: the per-window scale is only
    # recoverable before the global normalizer folds every condition onto one distribution.
    self_norm_mode = args.norm == "self"
    excl = [] if args.all_data else (args.exclude if args.exclude is not None else VAL_TRAJ)
    trajs = load_trajectories(args.data, exclude=excl,
                              normalize=False, phases=PHASES_ALL[:args.phases])
    idx = window_index(trajs, args.stride)
    held = "NONE (all-data)" if args.all_data else f"{len(excl)} held out"
    print(f"train: {len(trajs)} trajectory-phases ({args.phases} phase(s)) -> "
          f"{len(idx)} windows, {held}", flush=True)

    # WP3: which windows sit at a high angle of attack. `<Re>_<AoA>` filenames, aligned with
    # `trajs` by construction.
    names = trajectory_names(args.data, exclude=excl, phases=PHASES_ALL[:args.phases])
    is_hi = np.isin([int(n.split("_")[1]) for n in names], args.aoa_hi)[idx[:, 0]]
    if args.aoa_only:
        idx = idx[is_hi]
        is_hi = is_hi[is_hi]
        print(f"--aoa-only {args.aoa_hi}: {len(idx)} windows kept", flush=True)
    w_samp = None
    if args.aoa_boost != 1.0:
        w_samp = torch.from_numpy(np.where(is_hi, args.aoa_boost, 1.0).astype(np.float64))
        print(f"--aoa-boost {args.aoa_boost} on AoA {args.aoa_hi}: "
              f"{int(is_hi.sum())}/{len(idx)} windows -> "
              f"{float(w_samp[is_hi].sum() / w_samp.sum()):.1%} of draws", flush=True)
    print(f"config: {vars(args)}", flush=True)

    # Trajectories stay on the CPU (~1 GB); windows are sliced per batch. Materializing them
    # would need ~21 GB at stride 2.
    trajs_t = [torch.from_numpy(d) for d in trajs]
    idx_t = torch.from_numpy(idx)

    stats = NORM_STATS[args.norm]
    mean_v = torch.from_numpy(stats[0]).to(dev)
    std_v = torch.from_numpy(stats[1]).to(dev)

    # Channel weights for the MSE field loss, normalized so the largest is 1 and chan_pow = 0
    # reproduces the previous loss exactly. Broadcasts over (B, T, H, W, C).
    cw = (std_v / std_v.max()) ** args.chan_pow

    hist_t = None
    if args.hist:
        hist_t = torch.from_numpy(history_stats(trajs, idx))
        print(f"history channels: {tuple(hist_t.shape)} "
              f"({hist_t.numel()*4/1e6:.0f} MB)", flush=True)

    n_ctx = max(args.ctx - T_IN, 0)  # extra frames prepended before the protocol window
    from models_alt import (build, divergence, radial_index, radial_spectrum,
                            time_derivative, velocity_gradient, vorticity)
    rad_idx, rad_nb = radial_index(H, W, dev)
    model = build(args.arch, args.width, args.var_head,
                  (HIST_CH if args.hist else 0) + n_ctx * C,
                  args.modes_y, args.modes_x, args.n_layers, args.p,
                  var_mlp=args.var_mlp, lift_mlp=args.lift_mlp,
                  point_mlp=args.point_mlp, var_hidden=args.var_hidden,
                  var_deep=args.var_deep,
                  var_in_std=args.var_in_std,
                  var_pred_std=args.var_pred_std, dropout=args.dropout,
                  mean_head=args.mean_head, osc_modes=args.osc_modes,
                  osc_global=args.osc_global, osc_zeromean=args.osc_zeromean,
                  fgn_dim=args.fgn_dim).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    if args.init:
        # Copy only the tensors that exist in both with the same shape, and say how many were
        # skipped. A silent partial load is the failure mode that would make a fine-tuning run
        # look like a from-scratch run.
        src = torch.load(args.init, map_location="cpu")["model_state_dict"]
        dst = model.state_dict()
        ok = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
        model.load_state_dict({**dst, **{k: v.to(dev) for k, v in ok.items()}})
        print(f"init: {args.init} -> {len(ok)}/{len(dst)} tensors copied "
              f"({len(dst) - len(ok)} left at their random init)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(idx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    def fetch(sel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xs = torch.stack([trajs_t[i][t:t + T_IN] for i, t in sel.tolist()])
        ys = torch.stack([trajs_t[i][t + T_IN:t + T_IN + T_OUT] for i, t in sel.tolist()])
        cs = None
        if n_ctx:
            # Frames immediately before the window. Near the start of a trajectory there are
            # fewer than n_ctx of them, so pad at the FRONT by repeating the earliest frame --
            # the same convention the official windowing uses for short tails.
            out = []
            for i, t in sel.tolist():
                seg = trajs_t[i][max(t - n_ctx, 0):t]
                if len(seg) < n_ctx:
                    # At t = 0 there is no prior frame at all; fall back to the window's own
                    # first frame, which is what eval_wp.val_context does.
                    first = seg[:1] if len(seg) else trajs_t[i][t:t + 1]
                    seg = torch.cat([first.expand(n_ctx - len(seg), -1, -1, -1), seg])
                out.append(seg)
            cs = torch.stack(out).to(dev, non_blocking=True)
        return xs.to(dev, non_blocking=True), ys.to(dev, non_blocking=True), cs

    swa_state, swa_n = None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        # With --aoa-boost this draws with replacement, so an epoch is the same number of
        # steps over a differently-weighted pool rather than one pass over every window.
        perm = (torch.randperm(len(idx_t)) if w_samp is None else
                torch.multinomial(w_samp, len(idx_t), replacement=True))
        acc = np.zeros(9)
        n = 0
        for i in range(0, len(idx_t), args.batch):
            pos = perm[i:i + args.batch]
            sel = idx_t[pos]
            x, y, ctx = fetch(sel)
            b = len(sel)
            hist = hist_t[pos].to(dev, non_blocking=True) if args.hist else None

            # `mul` converts a normalized difference or std to physical units; `off` completes
            # the affine map for absolute fields. Under self-normalization both carry the
            # per-window scale, so every physical-unit loss below works unchanged.
            s = window_scale(x) if self_norm_mode else torch.ones(1, device=dev)
            x, y = self_norm(x, s, mean_v, std_v), self_norm(y, s, mean_v, std_v)
            mul, off = std_v * s, mean_v * s
            if hist is not None:
                hist = norm_history(hist, s, mean_v, std_v)
            if ctx is not None:
                ctx = self_norm(ctx, s, mean_v, std_v)

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
                if ctx is not None:
                    ctx = a * ctx + shift
                if hist is not None:
                    # mean channels are absolute fields and take the full affine map; std
                    # channels are a scale, so they only pick up the factor.
                    a4 = a.reshape(b, 1, 1, 1)
                    sh4 = shift.reshape(b, -1).view(b, -1, 1, 1)
                    hist = torch.cat([a4 * hist[:, :2] + sh4, a4 * hist[:, 2:]], dim=1)
            if args.aug_noise > 0.0:
                scale = args.aug_noise * std_t(y).mean((2, 3), keepdim=True)
                x = x + scale * torch.randn_like(x)

            parts = [to_ch(x)]
            if ctx is not None:
                parts.append(ctx.permute(0, 1, 4, 2, 3).reshape(b, n_ctx * C, H, W))
            if hist is not None:
                parts.append(hist)
            # Hoisted out of the model call so the FGN sample passes below feed the network the
            # identical tensor rather than rebuilding it K times.
            inp = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
            raw, logstd = model(inp, detach_var=args.detach_var)
            raw = from_ch(raw)

            if args.var_head:
                # (B, C, H, W) -> (B, 1, H, W, C); clamp keeps exp() finite early in training.
                sigma = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)
                pred = compose(raw, sigma.detach(), args.alpha_train)
            else:
                sigma, pred = None, raw

            # Split the residual into its orthogonal time-mean and fluctuation parts so the
            # two can be weighted separately. d = dm + df with dm constant in time, hence
            # sum_t d^2 = T*dm^2 + sum_t df^2 exactly, and w_mean = 1 is the original loss.
            d = pred - y
            dm = d.mean(1, keepdim=True)
            df = d - dm

            if args.loss == "rel":
                # Per-sample relative L2 in PHYSICAL units -- exactly the scored quantity.
                # Doing this in normalized units would be a different objective: the normalized
                # denominator is dominated by the constant MEAN/STD offset, which erases the
                # low-Re up-weighting that is the whole point of WP1.1.
                sq_m = ((dm * mul) ** 2).reshape(b, -1).sum(1) * T_OUT
                sq_f = ((df * mul) ** 2).reshape(b, -1).sum(1)
                num = (args.w_mean * sq_m + sq_f).clamp(min=0).sqrt()
                den = (y * mul + off).reshape(b, -1).norm(dim=1).clamp(min=1e-8)
                l_field = (num / den).mean()
            else:
                # `cw` re-weights the channels. The MSE branch lives in NORMALIZED units, so
                # by default a 1-sigma error in v counts the same as a 1-sigma error in u --
                # which, since sigma_u/sigma_v = 6.1, weights v about 37x more per unit of
                # PHYSICAL error. Every scored metric disagrees: rel-L2 and MVPE are physical
                # norms in which u carries 86% of the error energy, and SPS measures every
                # interval against one global sigma in physical units. chan_pow = 1 puts the
                # loss in physical units (the `rel` branch already is); 0 leaves it as it was.
                l_field = args.w_mean * torch.mean((dm * cw) ** 2) + torch.mean((df * cw) ** 2)

            loss = args.lam_f * l_field
            l_probe = torch.zeros((), device=dev)
            if args.lam_p > 0.0:
                # scoring.mvpe_rel_l2_per_sample, verbatim: probe_y = 8..24 step 2,
                # probe_x = 13/21/29/37, time-averaged u,v, relative L2 per COLUMN (each
                # column carries its own denominator, which up-weights the far wake where
                # velocities are small), then averaged over the four columns.
                mp = (pred.mean(1) * mul.squeeze(1) + off.squeeze(1)) if self_norm_mode else \
                     (pred.mean(1) * mul + off)
                mt = (y.mean(1) * mul.squeeze(1) + off.squeeze(1)) if self_norm_mode else \
                     (y.mean(1) * mul + off)
                cols = []
                for px in PROBE_X:
                    pp = mp[:, PROBE_Y, px, :2].reshape(b, -1)
                    tt = mt[:, PROBE_Y, px, :2].reshape(b, -1)
                    cols.append((pp - tt).norm(dim=1) / tt.norm(dim=1).clamp(min=1e-8))
                l_probe = torch.stack(cols).mean()
                loss = loss + args.lam_p * l_probe

            l_div = l_spec = torch.zeros((), device=dev)
            if args.lam_div > 0.0:
                # Match the target's OWN divergence. Driving div -> 0 would fight the data:
                # divergence.py measures rms(div u)/rms(|grad u|) at 0.42-0.49 on real PIV,
                # dominated by unobservable out-of-plane dw/dz.
                l_div = rel_l2(divergence(pred * mul), divergence(y * mul)).mean()
                loss = loss + args.lam_div * l_div
            if args.lam_vort > 0.0:
                # The rotational half of the velocity gradient. For a shedding wake this is
                # the coherent structure; --lam-div constrains only the dilatational half.
                l_vort = rel_l2(vorticity(pred * mul), vorticity(y * mul)).mean()
                l_div = l_div + l_vort          # reported together in the `div` column
                loss = loss + args.lam_vort * l_vort
            if args.lam_grad > 0.0:
                # All four components. Divergence and vorticity are two invariants of this
                # tensor, so this subsumes both and adds the strain rates.
                l_grad = rel_l2(velocity_gradient(pred * mul),
                                velocity_gradient(y * mul)).mean()
                l_div = l_div + l_grad
                loss = loss + args.lam_grad * l_grad
            if args.lam_tgrad > 0.0:
                # The one derivative direction velocity_gradient does not constrain. A
                # per-element field loss is indifferent to how the predicted sequence evolves
                # from frame to frame, which is a strange silence in a forecaster's objective.
                l_tg = rel_l2(time_derivative(pred * mul), time_derivative(y * mul)).mean()
                l_div = l_div + l_tg            # reported in the `div` column with its siblings
                loss = loss + args.lam_tgrad * l_tg
            if args.lam_spec > 0.0:
                sp_p = radial_spectrum(pred * mul, rad_idx, rad_nb)
                sp_y = radial_spectrum(y * mul, rad_idx, rad_nb)
                # log-space: the spectrum spans orders of magnitude and the deficit to be
                # fixed is at high k, where the linear-scale energy is negligible.
                l_spec = torch.mean((torch.log(sp_p + 1e-12) - torch.log(sp_y + 1e-12)) ** 2)
                loss = loss + args.lam_spec * l_spec
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

            l_crps = l_svar = torch.zeros((), device=dev)
            if args.fgn > 0:
                # K extra forwards, weights shared, z ~ N(0, I). Everything above used the
                # z = 0 pass, which IS the plain FNO, so nothing here changes what the
                # deterministic model optimizes -- CRPS is strictly added on top.
                k_s = args.fgn
                zs = torch.randn(k_s, b, args.fgn_dim, device=dev)
                xs = torch.stack([from_ch(model(inp, detach_var=args.detach_var, z=zs[j])[0])
                                  for j in range(k_s)])
                # Fair (unbiased) CRPS per element:
                #   mean_k |x_k - y| - 1/(2*K*(K-1)) * sum_{k != l} |x_k - x_l|
                # The second sum runs over ORDERED pairs, so summing UNORDERED ones and
                # dividing by K*(K-1) is the same number from half the tensors. Accumulating
                # pair by pair also avoids the (K,K,B,T,H,W,C) broadcast, which is 4x the
                # memory at K = 4 for identical arithmetic. Unbiased matters: the biased
                # estimator rewards under-dispersion at small K, which is the exact failure
                # mode this whole work package exists to fix.
                spread = sum((xs[j] - xs[l]).abs().mean()
                             for j in range(k_s) for l in range(j + 1, k_s))
                l_crps = (xs - y).abs().mean() - spread / (k_s * (k_s - 1))
                loss = loss + args.lam_crps * l_crps
                with torch.no_grad():
                    # The screen from plan_20260810: does the SAMPLE SPREAD track std_t(y)
                    # better than the log-MSE head's sigma (rel-L2 0.358 on val_unseen)? This
                    # is its training-curve form, so a failing arm is visible by epoch 50
                    # instead of after a full eval -- the same cheap screen `var` provides for
                    # the sigma head.
                    sk = std_t(xs.detach().reshape(k_s * b, T_OUT, H, W, C))
                    l_svar = rel_l2(sk.reshape(k_s, b, 1, H, W, C).mean(0), std_t(y)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc += np.array([loss.item(), l_field.item(), l_var.item(), l_tke.item(),
                             l_probe.item(), l_div.item(), l_spec.item(),
                             l_crps.item(), l_svar.item()]) * b
            n += b

        if args.swa and ep >= args.epochs - args.swa:
            # Running mean of the weights over the tail of the schedule. OneCycleLR has
            # annealed the LR to near zero by then, so these iterates sit in one basin and
            # averaging them is the cheap half of an ensemble.
            sd = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().clone().float() for k, v in sd.items()}
                swa_n = 1
            else:
                swa_n += 1
                for k, v in sd.items():
                    swa_state[k] += (v.detach().float() - swa_state[k]) / swa_n

        if ep % 5 == 0 or ep == args.epochs - 1:
            a = acc / n
            # Existing field names and order are unchanged -- log scrapers depend on them; the
            # two FGN columns are appended and stay 0.00000 when --fgn is off.
            print(f"  epoch {ep:3d}  loss {a[0]:.5f}  field {a[1]:.5f}  "
                  f"var {a[2]:.5f}  tke {a[3]:.5f}  probe {a[4]:.5f}  "
                  f"div {a[5]:.5f}  spec {a[6]:.5f}  crps {a[7]:.5f}  "
                  f"svar {a[8]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    if swa_state is not None:
        print(f"SWA: averaged the last {swa_n} epochs", flush=True)
        model.load_state_dict({k: v.to(dev) for k, v in swa_state.items()})

    torch.save({"model_state_dict": model.state_dict(), "mode": "twohead",
                "width": args.width, "var_head": args.var_head, "var_mlp": args.var_mlp, "lift_mlp": args.lift_mlp, "point_mlp": args.point_mlp,
                "var_hidden": args.var_hidden, "var_deep": args.var_deep,
                "var_in_std": args.var_in_std, "var_pred_std": args.var_pred_std,
                "fgn_dim": args.fgn_dim, "hist": args.hist,
                # Both of these are read from the TOP level by _fno._fno_from_blob. mean_head
                # was missing here from the day it was added, so every --mean-head checkpoint
                # would have failed to load at deployment (loudly, on unexpected state_dict
                # keys -- but only at scoring time, after the training had been paid for).
                "mean_head": args.mean_head, "osc_modes": args.osc_modes,
                "osc_global": args.osc_global, "osc_zeromean": args.osc_zeromean,
                "ctx": args.ctx, "arch": args.arch, "modes_y": args.modes_y,
                "modes_x": args.modes_x, "n_layers": args.n_layers, "p": args.p,
                "self_norm": self_norm_mode, "norm": args.norm,
                "mean": stats[0], "std": stats[1], "config": vars(args)}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
