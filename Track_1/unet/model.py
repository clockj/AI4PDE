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

import numpy as np
import torch
import torch.nn as nn

# NOTE: h5py is imported lazily inside load_trajectories(). It is NOT present in the Track 1
# evaluation container (see smoke_test_kit.py MISSING_IN_EVAL_IMAGE), and this module is
# vendored into the submission zip, so a module-level import would crash the submission.

T_IN = T_OUT = 20
H, W, C = 32, 64, 2
SUB = 2
CH = T_IN * C  # 40 channels once time is folded in

# GaussianNormalizer statistics fitted on train_real 32x64, channels [u, v].
MEAN = np.array([0.154960856, -0.000513992854], dtype=np.float32)
STD = np.array([0.0968056545, 0.015960684], dtype=np.float32)

# WP-C statistics: the same quantities after each window is divided by its own mean streamwise
# speed. Measured over 30 train_real trajectories. Used instead of MEAN/STD when self-normalizing.
MEAN_SN = np.array([1.0, 0.0], dtype=np.float32)
STD_SN = np.array([0.526024, 0.102182], dtype=np.float32)

# The same statistics fitted on train_sim (all 100 trajectories, 2.05e8 samples, [u, v] only).
# Sim is non-dimensional so its scale is ~6x real's; normalizing sim with the real constants
# would put the input at u ~ 7.3 sigma and make the WP-D comparison meaningless.
MEAN_SIM = np.array([0.858663955, -0.004962022], dtype=np.float32)
STD_SIM = np.array([0.443825609, 0.214743194], dtype=np.float32)

NORM_STATS = {"real": (MEAN, STD), "sim": (MEAN_SIM, STD_SIM), "self": (MEAN_SN, STD_SN)}

# Trajectories held out of training; see Track_1/make_val.py (val_unseen.npz).
VAL_TRAJ = ["3750_0", "6300_10", "8850_20", "11400_20",
            "13950_10", "16500_5", "21600_15", "26700_0"]

# Sizes the step-embedding table. The MSE model only ever uses index 0, but the table stays
# K_REFINE+1 wide so one architecture serves both objectives and both checkpoints load.
K_REFINE = 3


PHASES_ALL = ((0, 0), (0, 1), (1, 0), (1, 1))


def load_trajectories(data_dir: str, exclude: list[str], normalize: bool = True,
                      phases: tuple[tuple[int, int], ...] = ((0, 0),)) -> list[np.ndarray]:
    """Return one (T, H, W, 2) float32 array per (trajectory, phase).

    `normalize=False` returns raw physical units, which self-normalization needs: the
    per-window scale is only recoverable before the global normalizer folds every flow
    condition onto one distribution.

    `phases` picks the (y, x) offsets of the ::SUB decimation. The raw frames are 64x128 and
    the scored grid is phase (0, 0), so the other three are the SAME PIV measurement sampled
    half a coarse cell over -- real flow at 4x the sample count, not synthetic perturbation.
    The airfoil sits half a cell away in the shifted phases, which is the risk: it is a
    sub-pixel translation of the geometry, and the network has to absorb it.
    """
    import h5py  # local: absent from the evaluation container, training-only dependency

    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        if os.path.basename(path)[:-3] in exclude:
            continue
        with h5py.File(path, "r") as f:
            # Read once at full resolution and slice per phase; re-reading the file four
            # times costs more than the 40 MB the raw pair holds transiently.
            u_raw = f["u"][:].astype(np.float32)
            v_raw = f["v"][:].astype(np.float32)
        for py, px in phases:
            d = np.stack([u_raw[:, py::SUB, px::SUB], v_raw[:, py::SUB, px::SUB]], -1)
            out.append((d - MEAN) / STD if normalize else d)
    return out


def trajectory_names(data_dir: str, exclude: list[str],
                     phases: tuple[tuple[int, int], ...] = ((0, 0),)) -> list[str]:
    """The `<Re>_<AoA>` stem of each array `load_trajectories` returns, in the same order.

    Same glob, same sort, same phase loop -- so index i lines up with `trajs[i]`. Kept as a
    separate function rather than a second return value so no existing caller changes.
    """
    return [os.path.basename(p)[:-3]
            for p in sorted(glob.glob(os.path.join(data_dir, "*.h5")))
            if os.path.basename(p)[:-3] not in exclude
            for _ in phases]


def make_windows(trajs: list[np.ndarray], stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for d in trajs:
        for t in range(0, len(d) - T_IN - T_OUT + 1, stride):
            xs.append(d[t:t + T_IN])
            ys.append(d[t + T_IN:t + T_IN + T_OUT])
    return np.asarray(xs), np.asarray(ys)


def window_index(trajs: list[np.ndarray], stride: int) -> np.ndarray:
    """(traj, start) pairs instead of materialized windows.

    make_windows copies every window, which costs ~9 GB at stride 5 and does not fit at
    stride 2. Slicing on the fly from the trajectories (~1 GB total) is what makes the
    denser strides of WP1 possible at all.
    """
    return np.asarray([(i, t) for i, d in enumerate(trajs)
                       for t in range(0, len(d) - T_IN - T_OUT + 1, stride)], dtype=np.int64)


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


# ---------------------------------------------------------------------------
# WP1 + WP2: same trunk, no refinement slot, optional variance head.
# See Track_1/plan/plan_20260729.md.
# ---------------------------------------------------------------------------


class UNetTwoHead(nn.Module):
    """Shared trunk, two 1x1-conv readouts.

    Differences from `UNet`:
      * no `z` slot and no step embedding -- the refinement machinery was dead weight for
        MSE-style training and doubled the cost of the first conv (WP1.3);
      * an optional per-pixel log-std head branching off the SAME final decoder feature map
        as the field head (WP2).

    The heads share everything up to `u1` on purpose: turbulent fluctuation intensity is set
    by the large-scale flow, so the variance readout must be conditioned on the same latent
    that encodes the mean field. The split costs ~130 parameters.
    """

    def __init__(self, width: int = 64, var_head: bool = True, extra_ch: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 4
        self.inc = block(CH + extra_ch, w1)
        self.down1 = block(w1, w2)
        self.down2 = block(w2, w3)
        self.up1 = block(w3 + w2, w2)
        self.up2 = block(w2 + w1, w1)
        self.head_field = nn.Conv2d(w1, CH, 1)
        self.head_logstd = nn.Conv2d(w1, C, 1) if var_head else None
        self.pool = nn.AvgPool2d(2)
        # Round 15. Dropout2d on the two ENCODER feature maps that feed the skip connections,
        # i.e. between stages and not inside a `block`. It adds NO parameters and is inert in
        # eval(), so a --dropout checkpoint is byte-compatible with the submission's vendored
        # copy of this class (`Track_1/submit_fno/_fno.py`), which does not know the flag.
        #
        # Dropout2d rather than Dropout for the same reason as FNO2d's: these are spatial
        # feature maps, and zeroing single pixels of one is nearly a no-op once the next stage
        # convolves over them. The decoder is left alone -- dropping a channel of `u1` would
        # blank it in the field head and the logstd head simultaneously, which couples the two
        # readouts through the regularizer rather than through the trunk.
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if self.head_logstd is not None:
            # Start the head predicting a uniform, roughly correct std (targets sit near 0.3 in
            # normalized units) instead of whatever a default init gives. Without this the TKE
            # loss starts ~1e2 and can drive logstd into the clamp, where the gradient is
            # exactly zero and the head never recovers -- which is what happened the first time
            # the variance losses were kept out of the trunk.
            nn.init.zeros_(self.head_logstd.weight)
            nn.init.constant_(self.head_logstd.bias, float(np.log(0.3)))

    def forward(self, prev: torch.Tensor,
                detach_var: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        h1 = self.drop(self.inc(prev))
        h2 = self.drop(self.down1(self.pool(h1)))
        h3 = self.down2(self.pool(h2))
        u2 = self.up1(torch.cat([nn.functional.interpolate(h3, scale_factor=2), h2], 1))
        u1 = self.up2(torch.cat([nn.functional.interpolate(u2, scale_factor=2), h1], 1))
        if self.head_logstd is None:
            logstd = None
        else:
            # detach_var still CONDITIONS the variance on the mean-flow latent -- it only stops
            # the variance losses from reshaping that latent, which measurably costs the field
            # head (run F: raw TKE 0.670 vs WP1's 0.599).
            logstd = self.head_logstd(u1.detach() if detach_var else u1)
        return self.head_field(u1), logstd


HIST_CH = 4  # history mean (u, v) + history std (u, v)


def history_stats(trajs: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Causal per-pixel mean and std of every frame up to and including each window.

    trajs : raw (T, H, W, 2) arrays;  idx : the (traj, start) pairs from window_index
    -> (N, HIST_CH, H, W) float32 in the same physical units as `trajs`

    Track 1's predict() is handed every window of the test set at once in (filename, time_id)
    order, and windows tile each trajectory contiguously, so for window k the frames of windows
    0..k-1 are INPUTS that are legally available -- this reads no target. It buys a far longer
    baseline for the flow's second-order statistics than the 20 frames the window itself
    carries: measured on val_unseen, combining it with the variance head's own estimate takes
    the predicted-std rel-L2 from 0.358 to 0.338 (diag_history.py).

    Computed through per-trajectory prefix sums so the cost is O(T) per trajectory rather than
    O(windows * T); the prefix arrays are freed before moving to the next trajectory.
    """
    out = np.empty((len(idx), HIST_CH, H, W), dtype=np.float32)
    for i, d in enumerate(trajs):
        rows = np.flatnonzero(idx[:, 0] == i)
        if not len(rows):
            continue
        d64 = d.astype(np.float64)
        cs1 = np.concatenate([np.zeros((1,) + d.shape[1:]), d64.cumsum(0)])
        cs2 = np.concatenate([np.zeros((1,) + d.shape[1:]), (d64 ** 2).cumsum(0)])
        for r in rows:
            n = int(idx[r, 1]) + T_IN            # frames [0, start + T_IN)
            m = cs1[n] / n
            sd = np.sqrt(np.maximum(cs2[n] / n - m ** 2, 0.0))
            out[r] = np.concatenate([m, sd], -1).transpose(2, 0, 1)
        del cs1, cs2
    return out


def norm_history(hist: torch.Tensor, s: torch.Tensor, mean: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
    """Put raw history stats on the same scale as the normalized input window.

    The mean channels are absolute fields and take the full affine map; the std channels are
    a scale and take only the division, since a standard deviation has no offset.
    """
    mv, sv = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    s4 = s.reshape(-1, 1, 1, 1) if s.numel() > 1 else s.view(1, 1, 1, 1)
    return torch.cat([(hist[:, :2] / s4 - mv) / sv, (hist[:, 2:] / s4) / sv], dim=1)


def std_t(y: torch.Tensor) -> torch.Tensor:
    """Per-pixel temporal std of (B, T, H, W, C), keepdim on T.

    Biased (1/T) to match `scoring.kinetic_energy`, which uses np.mean of the squared
    deviation. Using the unbiased estimator here would bias TKE high by T/(T-1) = 1.053.
    """
    return y.std(1, keepdim=True, unbiased=False)


def compose(y: torch.Tensor, sigma: torch.Tensor, alpha: float) -> torch.Tensor:
    """Rescale the predicted fluctuation to carry the predicted std.

    y     : (B, T, H, W, C) predicted field, normalized units
    sigma : (B, 1, H, W, C) predicted per-pixel temporal std, same units
    alpha : 0 returns `y` unchanged; 1 gives the field exactly std `sigma`.

    TKE only sees per-pixel temporal variance, so with alpha = 1 the TKE of the output is
    0.5*(sigma_u^2 + sigma_v^2) by construction -- correct even if the fluctuation is blurry
    and phase-wrong. alpha is the safety valve: at 0 this is the identity, so the composition
    cannot regress the plain-field baseline.
    """
    if alpha == 0.0:
        return y
    m = y.mean(1, keepdim=True)
    f = y - m
    s = std_t(y)
    target = alpha * sigma + (1.0 - alpha) * s
    # detach the denominator: the 1/std gradient path is unstable near-quiescent pixels and
    # carries no useful signal, since the scale is set by `target`.
    return m + f * (target / (s.detach() + 1e-6))


def tke(y: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """(B, T, H, W, C) -> (B, H, W). Mirrors scoring.kinetic_energy.

    `scale` is the per-channel std used to normalize `y`; pass it to get TKE in PHYSICAL
    units, which is what is scored. This is not cosmetic: STD_u / STD_v = 6.1, so normalized
    TKE over-weights the v fluctuation by 37x relative to the metric.
    """
    s = std_t(y)
    if scale is not None:
        s = s * scale
    return 0.5 * (s ** 2).squeeze(1).sum(-1)


def window_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-window freestream scale from raw PHYSICAL input: (B, 1, 1, 1, 1).

    The mean streamwise velocity over the whole input window approximates U_inf, which for this
    airfoil is Re*nu/c -- so dividing by it removes the Reynolds number from the input almost
    exactly. Clamped because a window whose mean u is ~0 would otherwise blow up; no real window
    is anywhere near that (measured mean_u spans 0.04-0.30 m/s).
    """
    return x[..., 0].mean(dim=(1, 2, 3)).clamp(min=1e-3).reshape(-1, 1, 1, 1, 1)


def self_norm(x: torch.Tensor, s: torch.Tensor, mean: torch.Tensor,
              std: torch.Tensor) -> torch.Tensor:
    """Raw physical (B,T,H,W,C) -> non-dimensional, standardized. See plan_20260730.md WP-C.

    Re 2968 in the hidden test set is below the whole training range, so under the global
    GaussianNormalizer its velocities land where the network has seen nothing. Dividing by the
    window's own scale first removes that axis: only the non-dimensional wake shape has to
    generalize, and that varies far less with Re than the absolute velocity does.
    """
    return (x / s - mean) / std


def self_denorm(y: torch.Tensor, s: torch.Tensor, mean: torch.Tensor,
                std: torch.Tensor) -> torch.Tensor:
    """Inverse of `self_norm`."""
    return (y * std + mean) * s


def rel_l2(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2, flattening everything but the batch axis."""
    p = p.reshape(p.shape[0], -1)
    t = t.reshape(t.shape[0], -1)
    return (p - t).norm(dim=1) / t.norm(dim=1).clamp(min=1e-8)
