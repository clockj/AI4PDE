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
import json
import os
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


def aug_mask(x: torch.Tensor, y: torch.Tensor, p: float,
             zconst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Blank a random region of BOTH the input window and the target, in PHYSICAL units.

    x, y   : (B, T, H, W, C) NORMALIZED, i.e. already through self_norm / the GaussianNormalizer
    p      : per-sample probability of being masked at all
    zconst : the normalized value a PHYSICAL zero carries, (C,) -- `-mean/std`. The real data's
             own zero set (PIV field of view, airfoil body) sits at exactly this value, so a
             synthetic mask painted with it is indistinguishable, element for element, from a
             piece of geometry the sensor did not see.

    Why. Round 16's board result says U-FNO wins every local instrument and loses on the hidden
    set, and the mechanism round 15 proposed for U-FNO's gain in the first place was the sharp
    zero set: "mode truncation cannot represent the sharp zero set of the PIV field of view and
    the airfoil body". Every local split shares train_real's ONE field-of-view geometry -- the
    mask is essentially a constant of the dataset -- so nothing we own tests a model against a
    different one. This makes the geometry a nuisance variable instead of a constant.

    Two region families, drawn 50/50:
      * a border band of width 1-6 on one of the four sides -- a field of view cropped or
        extended relative to ours;
      * a rectangle 4-12 high by 8-24 wide, anywhere -- a body, a shadow, a dropout patch.
    One region per sample, held fixed across all T frames and both channels of both the input
    and the target, because the thing being modelled is static geometry, not per-frame noise.
    Masking the target as well as the input is the point: an unmasked target would ask the
    model to HALLUCINATE the field behind an occlusion, which is not the task -- the scorer
    drops zero-target elements from SPS and keeps them in rel_l2 at a prediction of ~0.

    Adds no parameters and is training-only, so a checkpoint trained with it loads into the
    submission's own FNO2d unchanged.
    """
    if p <= 0.0:
        # Belt and braces: the call site is already guarded by `if args.aug_mask > 0.0`, but
        # returning before the first draw makes the FUNCTION inert at 0 too, so the RNG stream
        # -- and therefore every subsequent batch order and dropout mask -- is untouched
        # whichever way it is reached. verify/aug_mask_check.py asserts exactly that.
        return x, y
    b, _, h, w, _ = x.shape
    hit = torch.rand(b) < p
    if not bool(hit.any()):
        return x, y
    # Drawn on the CPU generator and moved once: a per-sample torch.randint on the device would
    # cost a synchronization per draw, and the mask is 64 KB per batch.
    m = torch.zeros(b, h, w, dtype=torch.bool)
    for i in torch.nonzero(hit).flatten().tolist():
        if bool(torch.rand(()) < 0.5):
            t = int(torch.randint(1, 7, ()))            # band width 1-6
            side = int(torch.randint(0, 4, ()))
            if side == 0:
                m[i, :t, :] = True
            elif side == 1:
                m[i, h - t:, :] = True
            elif side == 2:
                m[i, :, :t] = True
            else:
                m[i, :, w - t:] = True
        else:
            rh = int(torch.randint(4, 13, ()))          # rectangle 4-12 x 8-24
            rw = int(torch.randint(8, 25, ()))
            y0 = int(torch.randint(0, h - rh + 1, ()))
            x0 = int(torch.randint(0, w - rw + 1, ()))
            m[i, y0:y0 + rh, x0:x0 + rw] = True
    m = m.to(x.device).reshape(b, 1, h, w, 1)
    return torch.where(m, zconst, x), torch.where(m, zconst, y)


def _sam_ascent(model: torch.nn.Module, rho: float) -> list:
    """Move every parameter to `w + rho * g / ||g||` in place; return `(param, w)` pairs.

    The norm is GLOBAL over all parameters that carry a gradient, which is what makes the
    perturbation an ascent step of length exactly `rho` in the full parameter space -- a
    per-tensor normalization would be a different, and much larger, neighbourhood.

    The saved `w` is a CLONE of the weights before the step, not the offset that was added.
    Restoring by subtracting the offset would be exact only up to the rounding of the add
    (`(w + e) - e != w` in fp32 whenever |e| << |w|, which is the whole regime here), and a
    weight that does not return exactly to where it started makes the SAM path silently
    non-reproducible. One clone of the parameters per step is ~114 MB at width 96 depth 8
    and is freed as soon as `_sam_restore` returns.
    """
    with torch.no_grad():
        gs = [p for p in model.parameters() if p.grad is not None]
        if not gs:
            return []
        gn = torch.norm(torch.stack([q.grad.detach().norm(2) for q in gs]), 2)
        scale = rho / (gn + 1e-12)
        saved = []
        for p in gs:
            saved.append((p, p.detach().clone()))
            p.add_(p.grad.detach() * scale)
    return saved


def _sam_restore(saved: list) -> None:
    """Undo `_sam_ascent` exactly: copy the saved weights back."""
    with torch.no_grad():
        for p, w in saved:
            p.copy_(w)


def aux_index(trajs: list[np.ndarray], k: int, stride: int) -> np.ndarray:
    """(traj, start) pairs for FRAME-STRIDE-k windows over the REAL trajectories.

    A window is `d[start : start + (T_IN+T_OUT)*k : k]` -- the same flow, sampled every k-th
    frame, so its input is 20 frames spanning 20*k real frames and its target the next 20 at
    the same spacing. At k > 1 that is the real wake at k times its own shedding frequency:
    physically plausible dynamics (every frame is a real PIV measurement, in the right order,
    of the right flow) with statistics no real condition has (no Strouhal number in the data
    matches). It is built from the SAME `trajs` list `window_index` uses, so `--exclude` is
    respected by construction -- a held-out trajectory is not in the list and cannot enter.

    k = 1 is the degenerate case and is a legitimate CONTROL rather than a no-op: identical
    distribution to the real term, one extra batch per step. The perturbation arms are k >= 2.
    """
    span = (T_IN + T_OUT) * k
    return np.asarray([(i, t) for i, d in enumerate(trajs)
                       for t in range(0, len(d) - span + 1, stride)], dtype=np.int64)


# SPS scores every element against ONE global half-width scale, frozen for the season
# (scoring.py: sigma = 0.0563870259, physical units). --cov-c is quoted in units of it so
# the saturation point of the coverage loss is stated in the currency the scorer uses.
SPS_SIGMA = 0.0563870259


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/users/PAS1187/qwu/RealPDE/train_real/train_real")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=-1,
                    help="seed torch/numpy so a control and its arm share initialization and "
                         "batch order. Seed sigma on the AoA 15-20 proxy is 0.017 wfinal "
                         "(round 11), which is the size of several shipped effects -- pairing "
                         "removes it from the comparison instead of averaging it down.")
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
    ap.add_argument("--aug-mask", type=float, default=0.0,
                    help="per-sample probability of blanking a random region of BOTH the input "
                         "window and the target with the PHYSICAL zero the real field-of-view "
                         "/ airfoil zero set already carries (a border band 1-6 px on a random "
                         "side, or a 4-12 x 8-24 rectangle; one region per sample, the same "
                         "region for all 20 frames and both channels). Every local split we "
                         "own shares train_real's single field-of-view geometry, so the zero "
                         "set is a CONSTANT of every instrument -- and round 15's argument for "
                         "U-FNO was that a conv branch represents that sharp geometry where "
                         "mode truncation cannot. If the hidden set's geometry differs, a "
                         "model tuned to ours is tuned to a nuisance variable. This makes it "
                         "a nuisance variable in training too. Applies to the REAL batch only "
                         "(the --sim term keeps its own mapped mask), adds no parameters, and "
                         "is inert at 0 -- no random number is drawn, so a p=0 run is "
                         "byte-identical to one without the flag.")
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
    ap.add_argument("--arch",
                    choices=["unet", "fno", "deeponet", "afno", "dilres", "ufno", "vit",
                             "cnext_unet", "unet_mod", "ffno", "ffno_nff", "swin_unet",
                             "simvp", "stvit", "swin_unet_b6", "swin_unet_win16",
                             "swin_unet_p1"],
                    default="unet",
                    help="round 15 adds afno / dilres / ufno / vit, all with FNO2d's "
                         "interface. --width is the trunk width (the token dim for "
                         "vit) and --n-layers the depth for all of them. The 2026-09-14 "
                         "wave 1 adds cnext_unet / unet_mod / ffno (ffno_nff: its per-layer "
                         "FF off) / swin_unet / simvp / stvit (models_alt.WAVE1_ARCHS), sized "
                         "the same way; --modes-y/-x matter only for ffno, and --var-mlp / "
                         "--lift-mlp are ignored (every wave-1 head is already an MLP).")
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
    ap.add_argument("--tbasis", type=int, default=0,
                    help="J free zero-mean temporal profiles, SHARED across windows, with "
                         "per-pixel coefficients. The discriminating control for --osc-modes: "
                         "same rank and same zero-mean constraint, but the basis does not "
                         "adapt to the window. report_20260831_round13.md §23 measured that "
                         "the osc head's omega is an affine function of the window's own mean "
                         "speed at R^2 0.99, and §25 that the branch is load-bearing -- this "
                         "asks whether that adaptivity is the mechanism or decoration.")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout2d probability between spectral blocks. Adds no parameters, "
                         "so a checkpoint trained with it still loads into the submission's "
                         "own FNO2d, which has none and runs in eval() regardless.")
    ap.add_argument("--mode-drop", type=float, default=0.0,
                    help="DropConnect over the RETAINED Fourier modes (--arch fno only). In "
                         "training, SpectralConv2d zeros a per-sample Bernoulli(p) subset of "
                         "the (ky, kx) coefficients before the spectral matmul and rescales "
                         "the survivors by 1/(1-p); the same mask is used for the +ky and "
                         "-ky blocks, which are conjugate views of one physical wavenumber. "
                         "--dropout regularizes the CHANNEL axis between blocks; the spectral "
                         "weights are 99.4%% of the parameters and live on the MODE axis, "
                         "which nothing has ever regularized. Adds no parameters and is "
                         "inert in eval(), so the checkpoint loads into _fno.py unchanged.")
    ap.add_argument("--lam-tail", type=float, default=0.0,
                    help="weight on the CVaR of the field error: per sample, the mean of the "
                         "largest 10%% of per-element squared errors, in the same cw-weighted "
                         "normalized units as the --loss mse field term. The board pays for "
                         "per-element CONTAINMENT -- round 15's w96 x dropout package made "
                         "rel_l2 slightly WORSE and sps +1.013, and sps carries 0.229 of the "
                         "weight against rel_l2's 0.573 while being far steeper -- so the "
                         "quantity to minimize is the error TAIL, not its rms. This is that "
                         "tail, written directly into the objective. --loss rel only averages "
                         "a per-sample norm and has no per-element term to take a tail of.")
    ap.add_argument("--subpix", type=float, default=0.0,
                    help="weight on a TRAINING-ONLY SUB-PIXEL target: predict the output "
                         "window at the THREE decimation phases the scored grid discards, "
                         "from the phase-(0,0) input alone. model.load_trajectories reads the "
                         "raw 64x128 frames and slices [py::2, px::2]; (0,0) is what is scored "
                         "and the other three are the SAME PIV measurement sampled half a "
                         "coarse cell over -- real flow at 4x the sample count, not synthetic "
                         "perturbation. THIS IS NOT --phases 4, which is null twice: that "
                         "makes four INDEPENDENT tasks (predict phase p from phase p) while "
                         "this makes ONE task with a 4x denser target, i.e. a "
                         "super-resolution constraint that forces the trunk to represent the "
                         "field BETWEEN the observed grid points. The head is a 1x1 conv on "
                         "the same features head_field reads and is STRIPPED at save, so the "
                         "shipped blob is a plain FNO2d and inference is unchanged. Costs 4x "
                         "the real-data RAM, exactly as --phases 4 does.")
    ap.add_argument("--lam-cov", type=float, default=0.0,
                    help="weight on a COVERAGE (redescending Welsch) field loss. SPS scores "
                         "per-element containment, exp(-2h/sigma)*1[|E|<=h], so an element "
                         "already far outside its interval is worth nothing and chasing it "
                         "costs the elements that are nearly inside. The term is "
                         "cn^2*(1-exp(-0.5*(E/cn)^2)) on the SAME cw-weighted normalized "
                         "residual --loss mse averages, with cn the physical scale "
                         "--cov-c*SPS_SIGMA expressed in those units. The cn^2 prefactor "
                         "makes the small-error limit exactly 0.5*mean(en^2), so --lam-cov 2 "
                         "is a CURVATURE-MATCHED swap for the MSE term and the only thing "
                         "that changes is the tail. Round 16's --lam-tail (CVaR of the worst "
                         "tenth) is the opposite weighting, and it lost.")
    ap.add_argument("--cov-c", type=float, default=0.25,
                    help="saturation scale of --lam-cov, in units of the scorer's frozen "
                         "sigma = 0.0563870259. The model's typical physical |E| is ~0.2 "
                         "sigma, so 0.25 puts the knee near the median error; larger is "
                         "closer to plain MSE (cov-c -> inf IS 0.5*MSE, exactly).")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="Beta(a,a) convex combinations of two windows in the batch (Zhang "
                         "et al., ICLR 2018). Non-physical by construction -- a mixture of "
                         "two solutions is not a solution -- but SPS is a per-element "
                         "CONTAINMENT score, i.e. a calibration metric, and mixup is the "
                         "standard fix for over-confident interpolation between conditions. "
                         "0 draws no random numbers, so the default RNG stream is untouched.")
    ap.add_argument("--sam-rho", type=float, default=0.0,
                    help="Sharpness-Aware Minimization (Foret et al., ICLR 2021). 0 disables "
                         "it and the training loop is byte-identical to before. Above 0 each "
                         "step takes the gradient at w + rho*g/||g|| -- the worst point of a "
                         "rho-ball around the current weights -- and applies it at w, so the "
                         "objective becomes max_{||e||<=rho} L(w+e) and the optimizer prefers "
                         "FLAT minima. The reason to want that here is round 15's board "
                         "result: capacity x dropout made rel_l2 slightly WORSE and sps "
                         "+1.013, i.e. the board pays for the per-element error TAIL, and "
                         "flatness is a tail property of the loss surface rather than a "
                         "reweighting of the training error (which --lam-tail tried, and "
                         "which memorized). Costs ~2x wall clock: two forwards, two "
                         "backwards, one optimizer step.")
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
    ap.add_argument("--teacher", default="",
                    help="DISTILLATION. A directory written by distill_labels.py: the member-"
                         "averaged field and sigma of a TEACHER ENSEMBLE, in normalized space, "
                         "one row per window in window_index order. The student is then trained "
                         "against a target blended with the teacher's, and its sigma head against "
                         "the teacher's sigma, which is what a single model has to reproduce if "
                         "it is to replace a six-member package at one invocation. The labels' "
                         "index is asserted equal to this run's, so a stride or split mismatch "
                         "fails loudly instead of training on shuffled labels.")
    ap.add_argument("--teacher-sim", action="store_true",
                    help="Also blend the SIM branch's target with the teacher's answer on the "
                         "mapped sim windows (sim_*.npy in the same directory). The teacher "
                         "labels any input, so sim stops being a wrong-but-related distribution "
                         "and becomes more of the teacher's own function.")
    ap.add_argument("--teacher-mix", type=float, default=0.5,
                    help="Weight on the TRUE target; 1 - w goes to the teacher's field. The "
                         "residual-based field terms (--loss, --lam-cov, --lam-tail, the "
                         "gradient losses, --lam-p) all see the ONE blended target, which for "
                         "the MSE branch has the same gradient as blending the two losses and "
                         "for the redescending coverage loss is the better-posed of the two. "
                         "The SIGMA target is the teacher's alone whenever --teacher is set.")
    ap.add_argument("--sim", default="",
                    help="train_sim directory. Adds a SECOND loss term on sim windows mapped "
                         "into real's measurement space (simreal.load_sim_mapped). Sim is a "
                         "strict superset of real's condition grid, so it covers Re 15225 and "
                         "27975 and the holes real leaves -- the Reynolds extrapolation the "
                         "hidden test set is suspected to need.")
    ap.add_argument("--lam-sim", type=float, default=0.3,
                    help="weight on the sim term. Real still defines the epoch: one sim batch "
                         "is drawn per real step, so this is a per-step gradient ratio.")
    ap.add_argument("--sim-loss", choices=["shape", "full"], default="shape",
                    help="'full' is the naive arm (the same field loss as real). 'shape' is "
                         "the bet: normalize BOTH prediction and target by their own time-mean "
                         "and per-pixel temporal std before comparing, which makes the term "
                         "blind to exactly the two quantities sim_map.py measures sim to be "
                         "wrong about -- the time-mean field (rel-L2 0.400 against real, where "
                         "this model is at 0.11) and the fluctuation amplitude (2.9-5.6x too "
                         "energetic). What survives is WHERE the structure goes, which no "
                         "statistic in sim_map.py rules out and which the round-10 oracle "
                         "prices as the largest reachable component under the nine-round "
                         "weights.")
    ap.add_argument("--sim-map", choices=["scalar", "std", "meanstd"], default="scalar",
                    help="how much of the real twin's STATIC statistics the domain map copies "
                         "onto sim (simreal.load_sim_mapped). 'scalar' is the shipped map: sim "
                         "keeps its own time-mean field and its own per-pixel std shape, and "
                         "only their magnitudes are matched. 'std' replaces the fluctuation "
                         "amplitude per PIXEL, 'meanstd' also replaces the time-mean field -- "
                         "the bet being that the more of sim's static statistics come from "
                         "real, the more the sim term teaches only the dynamics, which is the "
                         "one thing no statistic in sim_map.py rules out. The per-pixel gain "
                         "is clamped to a factor of 3 either side of the scalar one, because "
                         "an unclamped sigma_twin/sigma_sim is the round-14 --sim-loss shape "
                         "failure. Conditions with no usable twin keep the scalar map.")
    ap.add_argument("--sim-stride", type=int, default=5)
    ap.add_argument("--sim-batch", type=int, default=0, help="0 = same as --batch")
    ap.add_argument("--sim-frames", type=int, default=0, help="0 = all 1000")
    ap.add_argument("--sim-exclude-held", action="store_true",
                    help="also drop the held-out conditions from the SIM pool. Without it the "
                         "model sees CFD at exactly the Reynolds numbers the real data is "
                         "missing, which is the upper bound on sim's value rather than the "
                         "deployment situation -- the Track-2 example trajectory is Re 2968, "
                         "below sim's own range. Separates 'sim teaches this condition' from "
                         "'sim teaches the regime'.")
    ap.add_argument("--sim-p-head", action="store_true",
                    help="TRAINING-ONLY auxiliary PRESSURE target on sim windows. Sim has `p` "
                         "and real does not -- real's p is zero-filled and never scored -- so "
                         "it is the one piece of sim content nothing in this project has used "
                         "(CLAUDE.md Strategy item 6 proposed it in round 1 and it was never "
                         "built). Q3i measured sim's value to be its CONTENT rather than the "
                         "perturbation it delivers, and pressure is the field that couples u "
                         "and v dynamically -- the variable a 2D velocity-only forecaster has "
                         "to infer implicitly. Adds `head_p`, a 1x1 conv on the same trunk "
                         "features head_field reads, and the same field-loss form as the sim "
                         "term at --lam-p-sim, on sim windows only. The head is STRIPPED from "
                         "the saved checkpoint, so the shipped _fno.py loads the blob "
                         "unchanged. Requires --sim and --arch fno.")
    ap.add_argument("--sim-p-norm", choices=["pooled", "traj"], default="pooled",
                    help="how the auxiliary pressure TARGET is standardized. 'pooled' (the "
                         "default, and what Q3j ran) uses one mean/std over every mapped sim "
                         "trajectory -- but the speed match enters p as sc^2 ~ Re^2, so the "
                         "mapped amplitude spans ~55x across the Reynolds range and the pooled "
                         "variance is almost entirely BETWEEN conditions: Q3j's head reached "
                         "psim 0.0087, i.e. it explained 99 %% of a quantity that is mostly "
                         "'how fast is this condition', while the mid- and low-Re conditions "
                         "contributed 1/50 - 1/2500 of the loss. 'traj' standardizes each "
                         "trajectory by its OWN mean and std, so every condition contributes "
                         "equally and what is left to learn is the pressure FIELD rather than "
                         "its amplitude. Computed in memory from the cached array -- the 2.46 "
                         "GB `_p` map is not rebuilt.")
    ap.add_argument("--lam-p-sim", type=float, default=0.3,
                    help="weight on the auxiliary pressure loss, on the same scale as "
                         "--lam-sim (it is added to the TOTAL loss, not nested inside the sim "
                         "term, so the `sim` training column stays comparable with every "
                         "earlier sim arm and `psim` is reported on its own).")
    ap.add_argument("--aux-fstride", type=int, default=0,
                    help="0 = OFF (nothing is built, nothing is drawn, the RNG is untouched). "
                         "k >= 1 adds a SECOND loss term on real windows taken every k-th "
                         "frame -- the same flow at k times its shedding frequency. Q3d/f/g "
                         "measured sim's value to be flat in weight (10x), flat in window "
                         "diversity (2.5x) and to DECREASE monotonically as the domain map "
                         "makes sim more real-like, which reads as a structured PERTURBATION: "
                         "extra supervised pairs with plausible dynamics and wrong statistics. "
                         "This builds one from real data alone, so it separates that mechanism "
                         "from sim's physics content (Re/AoA coverage, CFD dynamics). k = 1 is "
                         "the control: same distribution, one extra batch per step.")
    ap.add_argument("--lam-aux", type=float, default=0.3,
                    help="weight on the aux term, on the scale of --lam-sim. Real still "
                         "defines the epoch: one aux batch is drawn per real step.")
    ap.add_argument("--aux-stride", type=int, default=0,
                    help="window START stride for the aux index. 0 = same as --stride.")
    ap.add_argument("--all-data", action="store_true",
                    help="train on ALL trajectories including the held-out ones. Submission "
                         "checkpoints only -- it makes local evaluation meaningless.")
    ap.add_argument("--drop-path", type=float, default=None,
                    help="stochastic-depth rate for the families that have it (swin_unet*, simvp, "
                         "stvit), linearly ramped over the blocks. Unset keeps each family's own "
                         "default (swin_unet 0.1). A training-time rate: nothing in the "
                         "state_dict changes, so every loader rebuilds the checkpoint as before.")
    ap.add_argument("--oracle-cond", action="store_true",
                    help="DIAGNOSTIC ONLY, never shipped (2026-09-14 wave 2). FiLM-condition "
                         "every FNO block on the TRUE (Re, AoA) of the window, read from its "
                         "trajectory's `<Re>_<AoA>` stem (models_alt.oracle_cond_features). "
                         "predict() is not given the condition, so the checkpoint cannot ship; "
                         "its value is how much held-out error knowing the condition removes "
                         "(Track_1/screen_oracle.py). Under ~2 %% and the conditioning idea "
                         "closes. Off, nothing is built and the RNG stream is untouched.")
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

    # Refuse rather than silently record an inert flag in the blob's config: build() only
    # threads mode_drop into FNO2d, so any other --arch would train the unregularized model
    # while its checkpoint claimed otherwise. That is the same failure `--fgn` guards against.
    if args.mode_drop > 0.0 and args.arch != "fno":
        raise SystemExit(f"--mode-drop is implemented for --arch fno only (got {args.arch})")
    # Same rule for the auxiliary pressure head, and checked HERE rather than beside the sim
    # loader: `load_trajectories` runs first and reads ~7 GB of HDF5, so a guard placed after it
    # makes a typo cost three minutes instead of one second.
    if args.sim_p_head:
        if not args.sim:
            raise SystemExit("--sim-p-head needs --sim: the pressure target exists in sim only")
        if args.arch != "fno":
            raise SystemExit(f"--sim-p-head is implemented on --arch fno (got {args.arch})")
    # --ctx > 20 and --hist feed the trunk tensors built OUTSIDE the 20-frame window; the mask
    # would have to be painted on those too, and neither is deployable in predict() anyway
    # (_fno.load_any refuses them). Refuse rather than mask an inconsistent input.
    if args.aug_mask > 0.0 and (args.hist or args.ctx != T_IN):
        raise SystemExit("--aug-mask is not implemented for --hist / --ctx != 20")
    if args.subpix > 0.0:
        if args.arch != "fno":
            raise SystemExit(f"--subpix is implemented on --arch fno (got {args.arch})")
        if args.loss != "mse":
            raise NotImplementedError("--subpix uses the per-element form --loss mse builds")
        if args.phases != 1:
            raise SystemExit("--subpix consumes all four phases itself; leave --phases 1")
        if args.hist or args.ctx != T_IN:
            raise SystemExit("--subpix is not implemented for --hist / --ctx != 20")
        if args.norm != "real":
            # self_norm's per-window scale is (B,1,1,1,1) and the auxiliary target is
            # (B,3,T,H,W,C); right-aligned broadcasting would silently line B up with the
            # phase axis. --norm self is rejected anyway (round 13 section 26).
            raise SystemExit("--subpix requires --norm real")
    if args.lam_cov > 0.0 and args.loss != "mse":
        raise NotImplementedError(
            "--lam-cov needs the per-element field error the --loss mse branch forms; "
            "--loss rel only produces a per-sample norm")
    # Mixup rewrites x and y together. --hist and --ctx carry frames from OUTSIDE the window
    # and would have to be mixed with the same lambda; neither is deployable anyway.
    if args.mixup > 0.0 and (args.hist or args.ctx != T_IN):
        raise SystemExit("--mixup is not implemented for --hist / --ctx != 20")
    if args.lam_tail > 0.0 and args.loss != "mse":
        raise NotImplementedError(
            "--lam-tail needs the per-element field error the --loss mse branch forms; "
            "--loss rel only produces a per-sample norm")
    if args.drop_path is not None and not (args.arch.startswith("swin_unet")
                                           or args.arch in ("simvp", "stvit")):
        raise SystemExit(f"--drop-path is not a knob of --arch {args.arch}")
    if args.oracle_cond:
        # Every forward outside the real batch (sim, aux, FGN samples) and every batch-mixing or
        # out-of-window input would see the model without its condition: refuse rather than
        # half-condition a diagnostic.
        if args.arch != "fno":
            raise SystemExit(f"--oracle-cond is implemented on --arch fno only (got {args.arch})")
        bad = [f for f, on in (("--sim", bool(args.sim)), ("--aux-fstride", args.aux_fstride > 0),
                               ("--fgn", args.fgn > 0), ("--mixup", args.mixup > 0.0),
                               ("--subpix", args.subpix > 0.0), ("--hist", args.hist),
                               ("--ctx", args.ctx != T_IN)) if on]
        if bad:
            raise SystemExit(f"--oracle-cond refuses {bad}")

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    dev = "cuda"
    # Self-normalization needs raw physical trajectories: the per-window scale is only
    # recoverable before the global normalizer folds every condition onto one distribution.
    self_norm_mode = args.norm == "self"
    excl = [] if args.all_data else (args.exclude if args.exclude is not None else VAL_TRAJ)
    if args.subpix > 0.0:
        # All four phases, then phase (0, 0) alone as `trajs` -- round 9 verified that slice is
        # bit-identical to the old `[:, ::2, ::2]`, so the window index, the AoA table and every
        # loss below are exactly what they would be at --phases 1. The other three become the
        # auxiliary target, aligned by construction: load_trajectories appends the four phases
        # of each file consecutively, so trajs_all[4*j + p] is trajectory j at phase p.
        trajs_all = load_trajectories(args.data, exclude=excl, normalize=False,
                                      phases=PHASES_ALL)
        trajs, sub_np = trajs_all[0::4], [trajs_all[k::4] for k in (1, 2, 3)]
        print(f"--subpix {args.subpix}: {len(trajs)} phase-(0,0) trajectories + "
              f"{sum(len(g) for g in sub_np)} auxiliary phase targets", flush=True)
        del trajs_all
    else:
        sub_np = None
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
    cond_tab = None
    if args.oracle_cond:
        # One row per trajectory-phase, aligned with `trajs` (same glob, same sort, same phase
        # loop), so a batch's conditions are a gather on column 0 of its (traj, start) pairs.
        from models_alt import oracle_cond_features
        cond_tab = torch.from_numpy(oracle_cond_features(
            [float(n.split("_")[0]) for n in names],
            [float(n.split("_")[1]) for n in names])).to(dev)
        print(f"--oracle-cond: FiLM on the TRUE (Re, AoA) of {len(names)} trajectory-phases "
              f"(DIAGNOSTIC ONLY; the checkpoint cannot ship)", flush=True)
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
    sub_t = ([[torch.from_numpy(d) for d in g] for g in sub_np]
             if sub_np is not None else None)
    idx_t = torch.from_numpy(idx)

    sim_trajs_t = sim_idx_t = None
    p_mean = p_std = 0.0
    p_mu_t = p_sd_t = None
    if args.sim:
        if self_norm_mode:
            raise SystemExit("--sim assumes the global real normalizer; --norm self would "
                             "rescale every window and undo the speed match")
        from simreal import load_sim_mapped
        sim_trajs = load_sim_mapped(args.sim, args.data, phases=PHASES_ALL[:args.phases],
                                    frames=args.sim_frames,
                                    exclude=excl if args.sim_exclude_held else None,
                                    real_exclude=excl, mode=args.sim_map,
                                    with_p=args.sim_p_head)
        sim_idx = window_index(sim_trajs, args.sim_stride)
        sim_trajs_t = [torch.from_numpy(d) for d in sim_trajs]
        sim_idx_t = torch.from_numpy(sim_idx)
        print(f"sim: {len(sim_idx)} windows at stride {args.sim_stride}, "
              f"lam_sim {args.lam_sim} loss '{args.sim_loss}' map '{args.sim_map}'", flush=True)
        if args.sim_p_head:
            # Pooled over the trajectories the loss will actually see -- i.e. AFTER `exclude`,
            # so the held-out semantics are the map's, unchanged. Two passes in float64 rather
            # than one concatenation: the pool is ~190M elements and materializing it would
            # cost 1.5 GB for two scalars. Zeros inside the FOV mask are included, because the
            # loss averages over them too.
            n_p = sum(d[..., 2].size for d in sim_trajs)
            s1 = sum(float(d[..., 2].sum(dtype=np.float64)) for d in sim_trajs)
            s2 = sum(float((d[..., 2].astype(np.float64) ** 2).sum()) for d in sim_trajs)
            p_mean = s1 / n_p
            p_std = max((s2 / n_p - p_mean ** 2) ** 0.5, 1e-8)
            print(f"sim p: pooled mean {p_mean:.9g} std {p_std:.9g} over {n_p} elements "
                  f"(mapped physical units, lam_p_sim {args.lam_p_sim}, "
                  f"norm '{args.sim_p_norm}')", flush=True)
            if args.sim_p_norm == "traj":
                # Per-trajectory standardization, computed from the SAME cached array -- the
                # 2.46 GB `_p` map is read, not rebuilt. Indexed by the sim trajectory id that
                # `window_index` already stores in column 0 of sim_idx, so a batch looks its
                # constants up rather than carrying them.
                mus = np.array([d[..., 2].mean(dtype=np.float64) for d in sim_trajs])
                sds = np.array([max(float(d[..., 2].std(dtype=np.float64)), 1e-8)
                                for d in sim_trajs])
                p_mu_t = torch.from_numpy(mus.astype(np.float32)).to(dev)
                p_sd_t = torch.from_numpy(sds.astype(np.float32)).to(dev)
                print(f"sim p: per-trajectory norm over {len(mus)} trajectories -- "
                      f"mean of means {mus.mean():.6g} (min {mus.min():.6g}, "
                      f"max {mus.max():.6g}), mean of stds {sds.mean():.6g} "
                      f"(min {sds.min():.6g}, max {sds.max():.6g}, "
                      f"max/min {sds.max()/sds.min():.1f}x)", flush=True)

    aux_idx_t = None
    if args.aux_fstride >= 1:
        # `trajs` is the real training list AFTER --exclude, so the aux pool inherits the split.
        a_stride = args.aux_stride or args.stride
        aux_idx = aux_index(trajs, args.aux_fstride, a_stride)
        aux_idx_t = torch.from_numpy(aux_idx)
        print(f"aux: {len(aux_idx)} windows at frame stride {args.aux_fstride}, "
              f"start stride {a_stride}, lam_aux {args.lam_aux}", flush=True)

    stats = NORM_STATS[args.norm]
    mean_v = torch.from_numpy(stats[0]).to(dev)
    std_v = torch.from_numpy(stats[1]).to(dev)

    # Channel weights for the MSE field loss, normalized so the largest is 1 and chan_pow = 0
    # reproduces the previous loss exactly. Broadcasts over (B, T, H, W, C).
    cw = (std_v / std_v.max()) ** args.chan_pow
    # The value a PHYSICAL zero takes after normalization, per channel. self_norm divides by the
    # window scale BEFORE subtracting the mean, so 0 -> (0/s - mean)/std = -mean/std whatever s
    # is, and --aug-scale maps 0 -> 0; this constant is therefore exact under every option.
    # verify/aug_mask_check.py asserts it against the real zero set's own normalized value.
    zconst = (-mean_v / std_v).to(dev)

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
                  tbasis=args.tbasis, mode_drop=args.mode_drop,
                  fgn_dim=args.fgn_dim, p_head=args.sim_p_head,
                  sub_head=args.subpix > 0.0,
                  cond_dim=(2 if args.oracle_cond else 0),
                  drop_path=args.drop_path).to(dev)
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

    # ---- DISTILLATION LABELS (forge 09-17) ------------------------------------------------
    # Rows are in window_index(trajs, stride) order, which is the order `idx` was just built
    # in, so the gather below is `pos` itself. The index is compared element by element rather
    # than trusted: a silent misalignment would train the student on another window's answer.
    teach_mean = teach_sig = sim_teach_mean = None
    if args.teacher:
        bad = [f for f, on in (("--norm self", self_norm_mode), ("--subpix", args.subpix > 0.0),
                               ("--mixup", args.mixup > 0.0), ("--aug-scale", args.aug_scale > 0.0),
                               ("--aug-noise", args.aug_noise > 0.0),
                               ("--aug-mask", args.aug_mask > 0.0),
                               ("--aoa-boost", w_samp is not None)) if on]
        if bad:
            # The teacher labelled the window as it is stored. Anything that changes x or y
            # after the gather would leave the label describing a different input.
            raise SystemExit(f"--teacher refuses {bad}: the labels describe the unaugmented "
                             f"window, and these change it")
        tman = json.load(open(os.path.join(args.teacher, "manifest.json")))
        if (tman["stride"], tman["phases"], bool(tman["all_data"]), sorted(tman["exclude"])) != \
           (args.stride, args.phases, bool(args.all_data), sorted(excl)):
            raise SystemExit(f"--teacher labels were built for stride {tman['stride']} phases "
                             f"{tman['phases']} all_data {tman['all_data']} exclude "
                             f"{tman['exclude']}, this run is stride {args.stride} phases "
                             f"{args.phases} all_data {args.all_data} exclude {sorted(excl)}")
        tidx = np.load(os.path.join(args.teacher, "real_idx.npy"))
        if tidx.shape != idx.shape or not np.array_equal(tidx, idx):
            raise SystemExit("--teacher labels are not aligned with this run's window index")
        teach_mean = torch.from_numpy(np.load(os.path.join(args.teacher, "real_mean.npy")))
        teach_sig = torch.from_numpy(np.load(os.path.join(args.teacher, "real_sig.npy")))
        print(f"teacher: {args.teacher}  {len(tidx)} labelled windows, mix {args.teacher_mix} "
              f"true / {1 - args.teacher_mix:.2f} teacher, sigma from the teacher; members "
              f"{[m['path'].split('/')[-1] for m in tman['members']]}", flush=True)
        if args.teacher_sim:
            if sim_idx_t is None:
                raise SystemExit("--teacher-sim needs --sim")
            sman = tman.get("sim")
            if sman is None:
                raise SystemExit(f"--teacher-sim: {args.teacher} has no sim labels")
            if (sman["stride"], sman["sim_map"], bool(sman["sim_exclude_held"]),
                    sman["frames"]) != (args.sim_stride, args.sim_map,
                                        bool(args.sim_exclude_held), args.sim_frames):
                raise SystemExit("--teacher-sim: the sim labels were built with different "
                                 f"sim settings ({sman}) than this run")
            sidx = np.load(os.path.join(args.teacher, "sim_idx.npy"))
            if not np.array_equal(sidx, sim_idx_t.numpy()):
                raise SystemExit("--teacher-sim labels are not aligned with this run's sim index")
            sim_teach_mean = torch.from_numpy(np.load(os.path.join(args.teacher, "sim_mean.npy")))
            print(f"teacher-sim: {len(sidx)} labelled sim windows", flush=True)
    elif args.teacher_sim:
        raise SystemExit("--teacher-sim without --teacher")

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
        acc = np.zeros(16)
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
            y_sub = None
            if sub_t is not None:
                # (B, 3, T_OUT, H, W, C), the same (trajectory, t) windows as `y`, normalized
                # by the same affine -- so it has to sit AFTER `s` is formed, not beside fetch.
                y_sub = torch.stack([torch.stack(
                    [g[i][t + T_IN:t + T_IN + T_OUT] for i, t in sel.tolist()])
                    for g in sub_t], 1).to(dev, non_blocking=True)
                y_sub = self_norm(y_sub, s, mean_v, std_v)
            if hist is not None:
                hist = norm_history(hist, s, mean_v, std_v)
            if ctx is not None:
                ctx = self_norm(ctx, s, mean_v, std_v)

            if args.mixup > 0.0:
                # FIRST, so --aug-scale / --aug-noise / --aug-mask act on the mixed window
                # exactly as they would on a real one. lam is folded to [0.5, 1] so every
                # sample keeps the identity of its first parent, which is what makes the
                # mixed target the right regression target for the mixed input.
                lam_mx = torch.distributions.Beta(args.mixup, args.mixup).sample((b,))
                lam_mx = torch.maximum(lam_mx, 1.0 - lam_mx).to(dev).view(b, 1, 1, 1, 1)
                jmx = torch.randperm(b, device=dev)
                x = lam_mx * x + (1.0 - lam_mx) * x[jmx]
                y = lam_mx * y + (1.0 - lam_mx) * y[jmx]
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
            if args.aug_mask > 0.0:
                # LAST, so the masked elements carry the physical zero exactly rather than the
                # zero plus whatever --aug-scale / --aug-noise had just put there.
                x, y = aug_mask(x, y, args.aug_mask, zconst)

            # The teacher's field replaces part of the truth, and its sigma replaces all of
            # it. `kd` reports how far the two targets are apart in the loss's own units --
            # it is a diagnostic, never added to the loss.
            t_sig = None
            kd_gap = torch.zeros((), device=dev)
            if teach_mean is not None:
                tm = teach_mean[pos].to(dev, non_blocking=True).float()
                t_sig = teach_sig[pos].to(dev, non_blocking=True).float()
                kd_gap = torch.mean(((tm - y) * cw) ** 2).detach()
                y = args.teacher_mix * y + (1.0 - args.teacher_mix) * tm

            def compute_loss():
                # Everything from the model input to the total loss, factored out so SAM
                # can evaluate the SAME expression twice: once at the weights, once at the
                # perturbed weights. The augmentations above stay OUTSIDE, so both passes
                # see the identical batch; what is redrawn inside is what the model itself
                # samples (dropout / --mode-drop masks, the FGN z, the sim minibatch),
                # which is standard for SAM and is why the RNG stream is untouched when
                # --sam-rho is 0: the closure is then called exactly once per step, in the
                # same order as the inline block it replaces.
                parts = [to_ch(x)]
                if ctx is not None:
                    parts.append(ctx.permute(0, 1, 4, 2, 3).reshape(b, n_ctx * C, H, W))
                if hist is not None:
                    parts.append(hist)
                # Hoisted out of the model call so the FGN sample passes below feed the network the
                # identical tensor rather than rebuilding it K times.
                inp = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
                if args.subpix > 0.0:
                    # One forward, not two: the auxiliary head reads the same trunk
                    # features head_field does, so asking for it here costs a 1x1 conv
                    # rather than a second pass through eight spectral blocks.
                    raw, logstd, sub = model(inp, detach_var=args.detach_var,
                                             return_sub=True)
                elif cond_tab is not None:
                    # --oracle-cond: the batch's TRUE (Re, AoA), gathered by trajectory index.
                    raw, logstd = model(inp, detach_var=args.detach_var,
                                        cond=cond_tab[sel[:, 0].to(dev)])
                    sub = None
                else:
                    raw, logstd = model(inp, detach_var=args.detach_var)
                    sub = None
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
                l_tail = torch.zeros((), device=dev)
                if args.lam_tail > 0.0:
                    # CVaR at 10%: the mean of the largest tenth of the per-element squared
                    # errors, per sample, in the SAME cw-weighted normalized units l_field
                    # averages. `l_field` splits d into its time-mean and fluctuation parts so
                    # they can be weighted separately; the tail deliberately does not -- an
                    # element that lands outside its SPS interval costs the same whichever part
                    # of the residual put it there, so the quantity to bound is the plain
                    # per-element error over the whole field.
                    e2 = ((d * cw) ** 2).reshape(b, -1)
                    kk = max(1, int(0.10 * e2.shape[1]))
                    l_tail = e2.topk(kk, dim=1).values.mean()
                    loss = loss + args.lam_tail * l_tail
                l_cov = torch.zeros((), device=dev)
                if args.lam_cov > 0.0:
                    # Redescending (Welsch) M-estimator on the same residual l_field averages.
                    # cn is --cov-c*SPS_SIGMA (PHYSICAL) mapped into cw-weighted normalized
                    # units by cw/mul; the cn^2 prefactor makes lim_{cn->inf} = 0.5*mean(en^2),
                    # so --lam-cov 2 leaves the small-error curvature of --loss mse unchanged
                    # and only bends the tail. Unlike --lam-tail this needs no per-sample sort.
                    en = d * cw
                    cn = (args.cov_c * SPS_SIGMA) * cw / mul
                    l_cov = torch.mean(cn ** 2 * (1.0 - torch.exp(-0.5 * (en / cn) ** 2)))
                    loss = loss + args.lam_cov * l_cov
                l_sub = torch.zeros((), device=dev)
                if args.subpix > 0.0:
                    # head_sub emits (B, 3*T_OUT*C, H, W) in the same time-major layout
                    # to_ch/from_ch use, so the reshape is the from_ch one with a leading
                    # phase axis.
                    sp = sub.reshape(b, 3, T_OUT, C, H, W).permute(0, 1, 2, 4, 5, 3)
                    l_sub = torch.mean(((sp - y_sub) * cw) ** 2)
                    loss = loss + args.subpix * l_sub
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
                    # With a teacher, the sigma head is distilled outright: the ensemble's
                    # sigma is a far better calibrated target than this one window's std_t(y),
                    # and the band -- hence sps containment -- is built on it.
                    s_true = t_sig if t_sig is not None else std_t(y)
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

                l_sim = l_psim = torch.zeros((), device=dev)
                if sim_idx_t is not None and args.lam_sim > 0.0:
                    # One sim batch per real step. Real defines the epoch: sim is a regularizer
                    # on a wrong-but-related distribution, not a second dataset to fit.
                    nb = args.sim_batch or args.batch
                    sj = torch.randint(len(sim_idx_t), (nb,))
                    ss = sim_idx_t[sj]
                    xs_ = torch.stack([sim_trajs_t[i][t:t + T_IN] for i, t in ss.tolist()])
                    ys_ = torch.stack([sim_trajs_t[i][t + T_IN:t + T_IN + T_OUT]
                                       for i, t in ss.tolist()])
                    yp_ = None
                    if args.sim_p_head:
                        # With --sim-p-head the mapped arrays carry [u, v, p]; split p off
                        # before the velocity normalizer, whose mean/std are length C = 2.
                        yp_ = ys_[..., 2].to(dev, non_blocking=True)   # (B, T_OUT, H, W)
                        xs_, ys_ = xs_[..., :C], ys_[..., :C]
                    xs_ = ((xs_.to(dev, non_blocking=True) - mean_v) / std_v)
                    ys_ = ((ys_.to(dev, non_blocking=True) - mean_v) / std_v)
                    if sim_teach_mean is not None:
                        ys_ = (args.teacher_mix * ys_ + (1.0 - args.teacher_mix)
                               * sim_teach_mean[sj].to(dev, non_blocking=True).float())
                    if yp_ is None:
                        ps_ = from_ch(model(to_ch(xs_), detach_var=args.detach_var)[0])
                    else:
                        # ONE forward, three outputs: the pressure head reads the same trunk
                        # features head_field does, so it costs a 1x1 conv, not a pass.
                        raw_, _, pp_ = model(to_ch(xs_), detach_var=args.detach_var,
                                             return_p=True)
                        ps_ = from_ch(raw_)
                    if args.sim_loss == "full":
                        dd = ps_ - ys_
                        ddm = dd.mean(1, keepdim=True)
                        l_sim = (args.w_mean * torch.mean((ddm * cw) ** 2)
                                 + torch.mean(((dd - ddm) * cw) ** 2))
                    else:
                        # Blind to the time-mean AND to the amplitude: divide each field by its
                        # own per-pixel temporal std. The gradient w.r.t. a global rescaling of
                        # the prediction is then exactly zero, so sim cannot teach its 2.9x
                        # fluctuation energy or its 0.40 rel-L2 mean field -- only structure.
                        def _shape(z):
                            f = z - z.mean(1, keepdim=True)
                            return f / f.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-3)
                        l_sim = torch.mean((_shape(ps_) - _shape(ys_)) ** 2)
                    loss = loss + args.lam_sim * l_sim
                    if yp_ is not None:
                        # The sim term's own field loss, on p: the residual split into its
                        # time-mean and fluctuation parts at --w-mean. No `cw` -- that
                        # re-weights u against v and p is one channel. The target is
                        # normalized by the POOLED sim p statistics printed at startup, which
                        # is the only normalizer available: real has no pressure to fit one to.
                        if p_mu_t is None:
                            yn_ = (yp_ - p_mean) / p_std
                        else:
                            # Column 0 of sim_idx is the trajectory id, so the batch's
                            # constants are a gather. Every condition then contributes the
                            # same variance to the loss, whatever its Reynolds number.
                            ii = ss[:, 0].to(dev, non_blocking=True)
                            yn_ = ((yp_ - p_mu_t[ii].view(-1, 1, 1, 1))
                                   / p_sd_t[ii].view(-1, 1, 1, 1))
                        dp = pp_ - yn_
                        dpm = dp.mean(1, keepdim=True)
                        l_psim = (args.w_mean * torch.mean(dpm ** 2)
                                  + torch.mean((dp - dpm) ** 2))
                        # Added to the TOTAL loss with its own weight rather than folded into
                        # l_sim, so `lam_p_sim` is on the same scale as `lam_sim` and the `sim`
                        # column keeps meaning what it meant in Q3d / Q3f / Q3g.
                        loss = loss + args.lam_p_sim * l_psim

                l_aux = torch.zeros((), device=dev)
                if aux_idx_t is not None and args.lam_aux > 0.0:
                    # THE PERTURBATION TEST (round 17 Q3i). Same shape as the sim term above --
                    # one batch per real step, drawn with replacement, the same `--sim-loss
                    # full` field loss (mean/fluctuation split at --w-mean, channel weights cw)
                    # -- but the windows are REAL frames at stride k. It deliberately does NOT
                    # follow --sim-loss: the `shape` variant is a closed negative (round 14).
                    #
                    # RNG ORDER. This draw is placed AFTER the sim draw on purpose, so within a
                    # step the sim minibatch sees the generator where --aux-fstride 0 would have
                    # left it, and the FIRST step's sim batch is unchanged. It does NOT hold
                    # beyond that: both draws come from the default CPU generator and this one
                    # consumes from it, so the next sim draw, the next epoch's torch.randperm
                    # and (through the extra forward's dropout masks) the CUDA stream all
                    # diverge. An aux run is a different run from its control, not a superset of
                    # it -- only --seed's shared initialization is bit-matched, exactly as for
                    # --sim-stride or --fgn. With --aux-fstride 0 none of this executes.
                    k_a = args.aux_fstride
                    aa = aux_idx_t[torch.randint(len(aux_idx_t), (args.batch,))]
                    xa = torch.stack([trajs_t[i][t:t + T_IN * k_a:k_a] for i, t in aa.tolist()])
                    ya = torch.stack([trajs_t[i][t + T_IN * k_a:t + (T_IN + T_OUT) * k_a:k_a]
                                      for i, t in aa.tolist()])
                    xa = xa.to(dev, non_blocking=True)
                    ya = ya.to(dev, non_blocking=True)
                    # Same normalizer as the real term: `trajs_t` is raw physical, and with
                    # --norm real this is exactly (x - MEAN)/STD, the sim path's expression.
                    s_a = window_scale(xa) if self_norm_mode else torch.ones(1, device=dev)
                    xa = self_norm(xa, s_a, mean_v, std_v)
                    ya = self_norm(ya, s_a, mean_v, std_v)
                    pa = from_ch(model(to_ch(xa), detach_var=args.detach_var)[0])
                    da = pa - ya
                    dam = da.mean(1, keepdim=True)
                    l_aux = (args.w_mean * torch.mean((dam * cw) ** 2)
                             + torch.mean(((da - dam) * cw) ** 2))
                    loss = loss + args.lam_aux * l_aux
                # TENSORS, not floats: `.item()` is a device sync, and the inline block it
                # replaces did all eleven of them AFTER opt.step(), where the CPU is already
                # waiting. Calling them here instead would stall the pipeline before the
                # backward is even launched.
                return loss, [loss, l_field, l_var, l_tke, l_probe, l_div,
                              l_spec, l_crps, l_svar, l_sim, l_tail, l_aux, l_psim,
                              l_cov, l_sub, kd_gap]

            loss, terms = compute_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.sam_rho > 0.0:
                # Sharpness-Aware Minimization (Foret et al., ICLR 2021): step to the
                # weights that MAXIMIZE the loss inside a rho-ball, take the gradient
                # there, and apply it at the original weights. The result is a descent
                # direction for max_{||e||<=rho} L(w + e) rather than for L(w), i.e. it
                # prefers flat minima. Two forwards and two backwards, so ~2x per step.
                saved = _sam_ascent(model, args.sam_rho)
                loss2, _ = compute_loss()
                opt.zero_grad(set_to_none=True)
                loss2.backward()
                _sam_restore(saved)
            opt.step()
            sched.step()
            acc += np.array([t.item() for t in terms]) * b
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
            # two FGN columns are appended and stay 0.00000 when --fgn is off. `tail` and `aux`
            # are appended for the same reason and stay 0.00000 when --lam-tail / --aux-fstride
            # are off, so the existing columns keep both their names and their positions;
            # `psim` is the same story for --sim-p-head.
            print(f"  epoch {ep:3d}  loss {a[0]:.5f}  field {a[1]:.5f}  "
                  f"var {a[2]:.5f}  tke {a[3]:.5f}  probe {a[4]:.5f}  "
                  f"div {a[5]:.5f}  spec {a[6]:.5f}  crps {a[7]:.5f}  "
                  f"svar {a[8]:.5f}  sim {a[9]:.5f}  tail {a[10]:.5f}  "
                  f"aux {a[11]:.5f}  psim {a[12]:.5f}  cov {a[13]:.5f}  "
                  f"sub {a[14]:.5f}  kd {a[15]:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if swa_state is not None:
        print(f"SWA: averaged the last {swa_n} epochs", flush=True)
        model.load_state_dict({k: v.to(dev) for k, v in swa_state.items()})

    sd_out = model.state_dict()
    if args.subpix > 0.0:
        # STRIP THE SUB-PIXEL HEAD, for the reason spelled out under --sim-p-head below: the
        # shipped `_fno.py` builds a plain FNO2d from the blob's flags and loads it strict, so
        # one unexpected `head_sub.*` key would fail every loader in the project, at scoring
        # time, after the training was paid for.
        n0 = len(sd_out)
        sd_out = {k: v for k, v in sd_out.items() if not k.startswith("head_sub.")}
        print(f"save: stripped {n0 - len(sd_out)} head_sub.* tensor(s)", flush=True)
    if args.sim_p_head:
        # STRIP THE AUXILIARY HEAD. `_fno._fno_from_blob` builds its vendored FNO2d from the
        # blob's flags and calls load_state_dict with strict=True, so one unexpected `head_p.*`
        # key would fail EVERY loader in the project -- and only at scoring time, after the
        # training was paid for (the failure mode `mean_head` already caused once). Stripping
        # here keeps the shipped `_fno.py` untouched, and losing the head costs nothing: it is
        # supervised on sim, its output is never scored, and no deployment path reads it.
        n_before = len(sd_out)
        sd_out = {k: v for k, v in sd_out.items() if not k.startswith("head_p.")}
        print(f"save: stripped {n_before - len(sd_out)} head_p.* tensor(s) -- the blob is a "
              f"plain FNO2d for _fno.load_any", flush=True)
    torch.save({"model_state_dict": sd_out, "mode": "twohead",
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
                "tbasis": args.tbasis,
                "ctx": args.ctx, "arch": args.arch, "modes_y": args.modes_y,
                "modes_x": args.modes_x, "n_layers": args.n_layers, "p": args.p,
                "self_norm": self_norm_mode, "norm": args.norm,
                "mean": stats[0], "std": stats[1], "config": vars(args)}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
