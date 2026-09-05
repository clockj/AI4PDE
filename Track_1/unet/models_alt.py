"""FNO and DeepONet, parameterized for a size sweep, drop-in for `UNetTwoHead`.

Both expose the same interface as `model.UNetTwoHead`:

    forward(prev: (B, CH_in, H, W), detach_var: bool) -> (field (B, CH, H, W), logstd or None)

so `train2.py`, `eval_wp.run` and `eval_pair.py` work against them unchanged and every
comparison against the U-Net is on identical footing -- same normalization, same windowing,
same variance composition, same SPS band, same timing method.

Time is folded into the channel axis exactly as in `model.py`: a 20-frame window of 2 measured
channels is 40 channels of a 32x64 image. That makes the FNO a 2-D operator over the spatial
domain with the horizon carried in channels, which is what the official baseline does too.
"""
from __future__ import annotations

import numpy as np
import math

import torch
import torch.nn as nn

from model import C, CH, H, T_OUT, W


class SpectralConv2d(nn.Module):
    """Truncated-mode spectral convolution. Modes are (ky, kx) on the rfft2 grid.

    Parameter count is 2 * width^2 * modes_y * modes_x, which is what dominates an FNO and
    what the size sweep varies. Truncation is a low-pass: modes above the cut are passed
    through the pointwise branch only.
    """

    def __init__(self, width: int, modes_y: int, modes_x: int, mode_drop: float = 0.0):
        super().__init__()
        self.my, self.mx = modes_y, modes_x
        # DropConnect over the RETAINED Fourier modes: during training, zero a per-sample
        # random subset of the (ky, kx) coefficients before the spectral matmul and rescale
        # by 1/(1-p). The spectral weights are 99.4% of an FNO's parameters and round 15
        # measured the operator to be mode-specific, so this is dropout applied to the object
        # that IS the model -- the analogue of Dropout2d on the channel axis, on the axis the
        # parameters actually live on. Adds NO parameters and is inert in eval(), so a
        # checkpoint trained with it loads unchanged into `submit_fno/_fno.py`'s own FNO2d.
        self.mode_drop = float(mode_drop)
        scale = 1.0 / (width * width)
        # Two blocks: positive and negative ky (rfft2 keeps all ky, half of kx).
        self.w1 = nn.Parameter(scale * torch.randn(width, width, modes_y, modes_x, 2))
        self.w2 = nn.Parameter(scale * torch.randn(width, width, modes_y, modes_x, 2))

    @staticmethod
    def _mul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", a, torch.view_as_complex(w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ft = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat,
                          device=x.device)
        my, mx = min(self.my, h // 2), min(self.mx, w // 2 + 1)
        a1, a2 = ft[:, :, :my, :mx], ft[:, :, -my:, :mx]
        if self.training and self.mode_drop > 0.0:
            # One (B, 1, ky, kx) mask, shared by the +ky and -ky blocks: those two halves are
            # conjugate views of the SAME physical wavenumber, so dropping one and keeping
            # the other would ask the operator to reconstruct a real field from half a
            # Hermitian pair. Broadcast over the channel axis for the same reason Dropout2d
            # drops whole maps: a mode is a spatial pattern, not a per-channel scalar.
            keep = 1.0 - self.mode_drop
            m = (torch.rand(len(x), 1, my, mx, device=x.device) < keep).to(x.dtype) / keep
            a1, a2 = a1 * m, a2 * m
        out[:, :, :my, :mx] = self._mul(a1, self.w1[:, :, :my, :mx])
        out[:, :, -my:, :mx] = self._mul(a2, self.w2[:, :, :my, :mx])
        return torch.fft.irfft2(out, s=(h, w))


# Angular frequency band for the oscillatory head: periods of 200 down to 10 frames. The
# shedding period is 60-75 frames (omega ~ 0.084-0.105), so the band brackets it with roughly a
# factor of seven of room on each side.
OSC_W_LO, OSC_W_HI = 2.0 * math.pi / 200.0, 2.0 * math.pi / 10.0


class FNO2d(nn.Module):
    """Fourier Neural Operator over the 32x64 grid, time folded into channels."""

    def __init__(self, width: int = 32, modes_y: int = 12, modes_x: int = 16,
                 n_layers: int = 4, var_head: bool = True, extra_ch: int = 0,
                 var_mlp: bool = False, lift_mlp: bool = False, point_mlp: bool = False,
                 var_hidden: int = 128, var_deep: bool = False, var_in_std: bool = False,
                 var_pred_std: bool = False, fgn_dim: int = 0, dropout: float = 0.0,
                 mean_head: bool = False, osc_modes: int = 0,
                 osc_global: bool = False, osc_zeromean: bool = False,
                 tbasis: int = 0, mode_drop: float = 0.0):
        super().__init__()
        # Channel dropout between spectral blocks -- the one classical regularizer the project
        # never tried, against a diagnosis that is the project's own: with 74 trajectories the
        # model MEMORIZES, capacity is monotonically harmful (w16 < w24 > w48 > w64), and the
        # training loss falls while the held-out score rises. Dropout2d rather than Dropout,
        # because these are spatial feature maps and dropping single pixels is nearly a no-op
        # when the next spectral block mixes the whole grid anyway.
        #
        # It adds no parameters, so a checkpoint trained with it loads unchanged into the
        # submission's own FNO2d (which has no dropout, and runs in eval() regardless) and
        # `_fno._fno_from_blob` ignores the extra config key.
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        # The default lift is a single 1x1 conv: a LINEAR map from 42 channels (20 frames x
        # u,v, plus the coordinate grid) down to width=24. The whole input window is
        # compressed, linearly, before the first spectral block sees it. lift_mlp gives it a
        # hidden layer -- the same fix that worked on head_logstd, at the other end.
        cin = CH + extra_ch + 2                              # +2 for the coordinate grid
        self.lift = (nn.Sequential(nn.Conv2d(cin, 128, 1), nn.GELU(), nn.Conv2d(128, width, 1))
                     if lift_mlp else nn.Conv2d(cin, width, 1))
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, modes_y, modes_x, mode_drop) for _ in range(n_layers)])
        # The pointwise ("local") half of each FNO block, complementing the spectral half. It
        # is linear by default -- the third 1x1 bottleneck of the same shape as `lift` and
        # `head_logstd`, and the only one that repeats per layer.
        self.point = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, width, 1))
            if point_mlp else nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, CH, 1))
        # A DECOUPLED TIME-MEAN readout, and the reason for it is a corrected measurement.
        # The `rel_l2` ceiling this project quoted since round 3 rested on a "time-mean
        # sampling floor" of 0.0531 that the model's own mean error (0.0438) is already below
        # at every angle of attack -- so the component round 10 closed is open, and it carries
        # 29% of rel-L2's error ENERGY. See floor_aoa.py / mean_floor2.py.
        #
        # Structurally this is the same move that made --var-mlp pay: give a distinct physical
        # quantity its own MLP readout on the shared trunk instead of asking one 40-channel
        # head to encode a mean field and a 20-frame fluctuation at once. `head_field` then
        # emits only the fluctuation, which is forced zero-mean over time so the two pieces
        # cannot trade against each other. Boundary capacity, which is the side of the
        # boundary/trunk rule that has paid every time it was tested.
        self.head_mean = (nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, C, 1)) if mean_head else None)
        # An OSCILLATORY output head, and the reason is a shape argument the project has
        # never acted on. The shedding period is 60-75 frames and T_IN = T_OUT = 20, so the
        # model sees about a THIRD of a cycle and must extrapolate the next third. The
        # measured error is exactly what that predicts: `gain` is 0.95-1.0 across the
        # energy-carrying wavenumbers (amplitude is right) while coherence at k = 2-6, which
        # carry 57% of the fluctuation energy, is only 0.55-0.79 (phase is not).
        #
        # A linear `lift` can already form any temporal DFT of the input window, so the
        # spectral CONTENT is available. What is awkward for a conv net is the nonlinear step
        # after it -- estimate omega, then advance the phase by omega*t. This head builds that
        # in: per pixel and channel it emits J triples (p, q, w), and synthesizes
        # sum_j p_j cos(omega_j t) - q_j sin(omega_j t) with omega_j squashed into a band
        # spanning periods of 10-200 frames. (p, q) rather than (amplitude, phase) so there is
        # no wrap-around discontinuity to learn around.
        #
        # It is ADDITIVE on head_field, which already has full freedom to emit any 20 frames,
        # so this adds no capability -- only an inductive bias. That is the point: the
        # diagnosis is memorization of 74 trajectories, capacity is monotonically harmful, and
        # a coherent oscillation costs this path three numbers where head_field spends twenty.
        # The last conv is zero-initialized, so training starts exactly at the plain FNO.
        self.osc_modes = osc_modes
        self.osc_global = osc_global
        self.osc_zeromean = osc_zeromean
        self.head_osc = None
        if osc_modes:
            # With a per-pixel omega the head can fit each pixel's 20-frame trace on its own,
            # which is the memorization failure mode this whole line of work is aimed at. A
            # vortex street has ONE shedding frequency per trajectory (plus harmonics), so
            # `osc_global` pools the trunk spatially and emits J frequencies for the whole
            # field -- the constraint that actually carries the physics. (p, q) stay per pixel
            # either way: amplitude and phase DO vary across the wake.
            nout = (2 if osc_global else 3) * osc_modes * C
            self.head_osc = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                          nn.Conv2d(128, nout, 1))
            nn.init.zeros_(self.head_osc[2].weight); nn.init.zeros_(self.head_osc[2].bias)
            self.osc_w = (nn.Sequential(nn.Linear(width, 64), nn.GELU(),
                                        nn.Linear(64, osc_modes)) if osc_global else None)
            self.register_buffer("osc_t", torch.arange(T_OUT, dtype=torch.float32))

        # A FREE zero-mean temporal basis, the discriminating control for the osc head. The
        # osc term is rank-2J in time on a basis the window CHOOSES (omega is an affine
        # function of the window's own mean speed, R^2 0.99 -- report §23). This is the same
        # rank and the same zero-mean constraint on a basis SHARED by every window. If it
        # matches, the head's value is "low-rank, zero-mean, smooth in time" and the frequency
        # adaptivity is decoration; if it loses, the adaptivity is the mechanism. Either answer
        # decides where the fluctuation work goes next, and report §25's ablation says the
        # branch is load-bearing rather than a regularizer, so the question is worth asking.
        self.tbasis = tbasis
        self.head_tb = None
        if tbasis:
            self.head_tb = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                         nn.Conv2d(128, tbasis * C, 1))
            # Zero-init the READOUT, exactly as head_osc does, so the branch starts inert and
            # the model is the baseline at step 0. The PROFILES must NOT be zero-init'd or
            # both factors of the product have zero gradient and the branch is dead.
            nn.init.zeros_(self.head_tb[2].weight); nn.init.zeros_(self.head_tb[2].bias)
            self.tb = nn.Parameter(torch.randn(tbasis, T_OUT) * 0.5)

        # The default logstd readout is a single 1x1 conv -- a LINEAR probe on a 24-channel
        # trunk, against the field head's 128-wide MLP. var_mlp gives the two heads the same
        # shape. Worth trying because tke_branch_v9.py prices a 25% closing of the sigma gap
        # at +1.2 final, which is the largest single lever measured.
        #
        # Two further knobs, both pure BOUNDARY capacity (the trunk is untouched, which is the
        # side of the boundary/trunk rule that has paid every time it was tested):
        #   var_hidden  widens that MLP's hidden layer past 128;
        #   var_deep    feeds it concat(h1..hL) instead of the last block's output only. The
        #               20-frame amplitude envelope is a multi-scale quantity and the final
        #               trunk state has had four blocks to specialize toward the field head.
        #   var_in_std  hands the head the INPUT window's own per-pixel temporal std, the two
        #               channels the trunk would have to form a quadratic of 20 frames to
        #               recover. It is the only one of the three that adds information rather
        #               than capacity, and it is computed from the window predict() is already
        #               given -- no extra frames, no cross-window read.
        #   var_pred_std hands it the std of the FIELD HEAD's own output window. compose()
        #               forms g = (alpha*sigma + (1-alpha)*s)/s with exactly that s, so without
        #               it the head is asked for sigma in absolute terms and the answer is then
        #               divided by a quantity it never saw; with it, it can learn the ratio.
        #               Detached, so the variance loss still cannot reshape the field trunk.
        self.var_deep, self.var_in_std, self.var_pred_std = var_deep, var_in_std, var_pred_std
        var_cin = ((width * n_layers if var_deep else width)
                   + (C if var_in_std else 0) + (C if var_pred_std else 0))
        if not var_head:
            self.head_logstd = None
        elif var_mlp:
            self.head_logstd = nn.Sequential(nn.Conv2d(var_cin, var_hidden, 1), nn.GELU(),
                                             nn.Conv2d(var_hidden, C, 1))
        else:
            self.head_logstd = nn.Conv2d(var_cin, C, 1)
        if self.head_logstd is not None:
            last = self.head_logstd[-1] if var_mlp else self.head_logstd
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(np.log(0.3)))
        # FGN (arXiv 2506.10772, the engine behind WeatherNext 2): sample the WEIGHTS, not the
        # output. One draw of z in R^fgn_dim drives a per-block FiLM on the normalized
        # activation, i.e. one draw of a conditionally-normalized network, and a sample costs
        # exactly ONE forward pass -- the only reason a generative head is admissible in a
        # latency-scored task (PDE-Refiner's denoising chain scored 82.53 against its own MSE
        # control's 83.83 at 4x compute).
        #
        # bias=False and the zero init are both load-bearing, not cosmetic: film(0) is then
        # identically 0 for the whole of training, so the z = 0 (and z = None) forward is
        # BIT-IDENTICAL to the plain FNO at every point in training. The centre of the sample
        # distribution IS the baseline model, the existing losses are computed on it unchanged,
        # and a CRPS term can only add spread on top. It also means an fgn checkpoint scored
        # with z = None reproduces its own deterministic control exactly.
        self.fgn_dim = fgn_dim
        self.film = (nn.ModuleList([nn.Linear(fgn_dim, 2 * width, bias=False)
                                    for _ in range(n_layers)]) if fgn_dim > 0 else None)
        if self.film is not None:
            for lin in self.film:
                nn.init.zeros_(lin.weight)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def forward(self, prev: torch.Tensor, detach_var: bool = False,
                z: torch.Tensor | None = None):
        """`z` is the FGN noise vector, (B, fgn_dim). None (or fgn_dim = 0) is the plain FNO."""
        h = self.lift(torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1))
        # One (scale, shift) pair per block, computed once outside the loop: it is a
        # (B, fgn_dim) x (fgn_dim, 2*width) matmul against a loop body that is spectral.
        films = ([f(z).chunk(2, -1) for f in self.film]
                 if z is not None and self.film is not None else None)
        hs = []
        for i, (s, p, n) in enumerate(zip(self.spec, self.point, self.norm)):
            a = n(s(h) + p(h))
            if films is not None:
                # Conditional normalization: the noise enters where the activation statistics
                # are set, so a draw of z perturbs the whole spatial field coherently rather
                # than adding independent per-pixel noise. That coherence is what lets
                # marginal-only training (CRPS per element) produce usable joint samples.
                dg, db = films[i]
                a = (1.0 + dg[:, :, None, None]) * a + db[:, :, None, None]
            h = h + self.drop(torch.nn.functional.gelu(a))
            hs.append(h)
        field = self.head_field(h)
        if self.head_osc is not None:
            # omega is squashed into [2pi/200, 2pi/10] rad/frame, a band that brackets the
            # 60-75 frame shedding period with room on both sides. Added BEFORE the mean head
            # so head_mean still owns the time-mean and only the oscillation's fluctuating
            # part survives the zero-meaning below.
            b, Hh, Ww = len(prev), field.shape[-2], field.shape[-1]
            k = 2 if self.osc_global else 3
            o = self.head_osc(h).reshape(b, self.osc_modes, k, C, Hh, Ww)
            if self.osc_global:
                w = self.osc_w(h.mean((-2, -1)))                       # (B, J)
                om = (OSC_W_LO + (OSC_W_HI - OSC_W_LO)
                      * torch.sigmoid(w))[:, :, None, None, None]      # broadcast over C,H,W
            else:
                om = OSC_W_LO + (OSC_W_HI - OSC_W_LO) * torch.sigmoid(o[:, :, 2])
            ph = om.unsqueeze(2) * self.osc_t.view(1, 1, T_OUT, 1, 1, 1)
            osc = (o[:, :, 0].unsqueeze(2) * torch.cos(ph)
                   - o[:, :, 1].unsqueeze(2) * torch.sin(ph)).sum(1)
            if self.osc_zeromean:
                # The decisive test of what this head is actually doing. The learned period is
                # 64.5 frames against a 20-frame output window, so the oscillator spans 0.31 of
                # a cycle and mean_t cos(omega t) = +0.511 -- it carries a large non-zero
                # time-mean BY CONSTRUCTION, and the measured gain turned out to be entirely
                # in the time-mean (-2.6%) with coherence and the fluctuation error unmoved.
                # Subtracting the term's own time-mean leaves only its oscillatory content.
                # If the gain survives, the phase claim is real; if it vanishes, this head was
                # a mean-head in disguise and the phase idea is UNTESTED.
                osc = osc - osc.mean(1, keepdim=True)
            field = field + osc.reshape(b, CH, Hh, Ww)
        if self.head_tb is not None:
            b, Hh, Ww = len(prev), field.shape[-2], field.shape[-1]
            a = self.head_tb(h).reshape(b, self.tbasis, C, Hh, Ww)
            # Zero-mean the profiles rather than the product: it is the same thing (the
            # coefficients are constant in t) and it keeps the constraint on 20 numbers per
            # mode instead of on every pixel, so nothing can drift into the time-mean.
            prof = self.tb - self.tb.mean(1, keepdim=True)
            tb = torch.einsum("bjchw,jt->btchw", a, prof)
            field = field + tb.reshape(b, CH, Hh, Ww)
        if self.head_mean is not None:
            b = len(prev)
            fl = field.reshape(b, T_OUT, C, H, W)
            fl = fl - fl.mean(1, keepdim=True)              # fluctuation only, by construction
            field = (self.head_mean(h).unsqueeze(1) + fl).reshape(b, CH, H, W)
        logstd = None
        if self.head_logstd is not None:
            zv = torch.cat(hs, 1) if self.var_deep else h
            if detach_var:
                zv = zv.detach()
            if self.var_in_std:
                # prev's first CH channels are the 20 input frames x (u, v), time-major.
                s_in = prev[:, :CH].reshape(len(prev), CH // C, C, *prev.shape[-2:]).std(1)
                zv = torch.cat([zv, s_in], 1)
            if self.var_pred_std:
                # head_field is (B, T_OUT*C, H, W) time-major, same layout as to_ch/from_ch.
                sp = field.detach().reshape(len(prev), T_OUT, C, *field.shape[-2:]).std(1)
                zv = torch.cat([zv, sp], 1)
            logstd = self.head_logstd(zv)
        return field, logstd


class DeepONetCh(nn.Module):
    """DeepONet with a conv branch and an MLP trunk over the fixed (x, y, t) output grid.

    On a fixed grid this is a rank-p linear decoder over a condition-independent basis, which
    is exactly why report_20260729.md found it good at the low-rank mean field and poor at
    per-condition small-scale structure. `p` and the trunk width are what the sweep varies.
    """

    def __init__(self, p: int = 256, trunk_width: int = 128, branch_width: int = 64,
                 var_head: bool = True, extra_ch: int = 0):
        super().__init__()
        bw = branch_width
        self.branch = nn.Sequential(
            nn.Conv2d(CH + extra_ch, bw, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(bw, bw * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(bw * 2, bw * 4, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(bw * 4, bw * 8), nn.GELU(), nn.Linear(bw * 8, C * p))
        self.trunk = nn.Sequential(
            nn.Linear(3, trunk_width), nn.GELU(),
            nn.Linear(trunk_width, trunk_width), nn.GELU(),
            nn.Linear(trunk_width, trunk_width), nn.GELU(),
            nn.Linear(trunk_width, p))
        self.bias = nn.Parameter(torch.zeros(C))
        self.p = p
        # The variance head reads the branch code: a per-window global descriptor is the only
        # spatial information a DeepONet branch carries.
        self.head_logstd = nn.Sequential(nn.Linear(C * p, 256), nn.GELU(),
                                         nn.Linear(256, C * H * W)) if var_head else None
        if self.head_logstd is not None:
            nn.init.zeros_(self.head_logstd[-1].weight)
            nn.init.constant_(self.head_logstd[-1].bias, float(np.log(0.3)))
        t, y, x = torch.meshgrid(torch.linspace(-1, 1, T_OUT), torch.linspace(-1, 1, H),
                                 torch.linspace(-1, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([x, y, t], -1).reshape(-1, 3))

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        b = len(prev)
        code = self.branch(prev)                                     # (B, C*p)
        coeff = code.reshape(b, C, self.p)
        basis = self.trunk(self.coords)                              # (T*H*W, p)
        out = torch.einsum("bcp,np->bnc", coeff, basis) / self.p ** 0.5
        field = (out.reshape(b, T_OUT, H, W, C) + self.bias)
        field = field.permute(0, 1, 4, 2, 3).reshape(b, CH, H, W)
        logstd = None
        if self.head_logstd is not None:
            z = code.detach() if detach_var else code
            logstd = self.head_logstd(z).reshape(b, C, H, W)
        return field, logstd


def divergence(f: torch.Tensor) -> torch.Tensor:
    """du/dx + dv/dy on the 32x64 grid, central differences. f is (B, T, H, W, 2).

    dx == dy on this grid (0.00342 m at sub_s=2), so the spacing cancels in any ratio and is
    left out; only relative divergence is ever compared.
    """
    u, v = f[..., 0], f[..., 1]
    du = torch.zeros_like(u)
    dv = torch.zeros_like(v)
    du[..., 1:-1] = 0.5 * (u[..., 2:] - u[..., :-2])
    dv[:, :, 1:-1, :] = 0.5 * (v[:, :, 2:, :] - v[:, :, :-2, :])
    return du + dv


def vorticity(f: torch.Tensor) -> torch.Tensor:
    """dv/dx - du/dy, central differences. f is (B, T, H, W, 2).

    For a vortex-shedding wake this, not the divergence, is the coherent structure: the whole
    signal is a train of alternating vortices. Divergence matching buys +0.58 on FNO
    (report_20260802.md section 3); this asks whether the rotational half of the velocity
    gradient is worth at least as much.
    """
    u, v = f[..., 0], f[..., 1]
    dvdx = torch.zeros_like(v)
    dudy = torch.zeros_like(u)
    dvdx[..., 1:-1] = 0.5 * (v[..., 2:] - v[..., :-2])
    dudy[:, :, 1:-1, :] = 0.5 * (u[:, :, 2:, :] - u[:, :, :-2, :])
    return dvdx - dudy


def velocity_gradient(f: torch.Tensor) -> torch.Tensor:
    """All four components of grad(u) stacked: (B, T, H, W, 4).

    Divergence and vorticity are two invariant combinations of this tensor; the remaining
    degrees of freedom are the strain rates. Matching the whole tensor subsumes both of the
    individual terms and is the natural generalization to test against them.
    """
    u, v = f[..., 0], f[..., 1]
    out = []
    for a in (u, v):
        dx = torch.zeros_like(a)
        dy = torch.zeros_like(a)
        dx[..., 1:-1] = 0.5 * (a[..., 2:] - a[..., :-2])
        dy[:, :, 1:-1, :] = 0.5 * (a[:, :, 2:, :] - a[:, :, :-2, :])
        out += [dx, dy]
    return torch.stack(out, -1)


def time_derivative(f: torch.Tensor) -> torch.Tensor:
    """du/dt and dv/dt over the 20 output frames, central differences: (B, T-2, H, W, 2).

    The exact temporal analogue of `velocity_gradient`, and it exists for the same reason that
    one worked.  `--lam-grad` is the only loss term in the project's history whose effect
    (+0.74 `final`, ~8 % of the error) ever cleared the local proxy's measured resolution of
    ~2 %; everything since has been below it.  Its mechanism was not capacity and not data --
    it added a *gradient-domain* constraint, which is what pins down what a mode-truncated
    spectral operator does between the coefficients it keeps.

    Nothing in the loss currently couples ADJACENT OUTPUT FRAMES.  The network emits all 20 at
    once as 40 channels, and a per-element L2 is indifferent to whether the sequence evolves
    like a flow or flickers.  That is a strange thing for a forecaster's objective to be silent
    about, and it is the one direction of the derivative that `velocity_gradient` does not
    touch.

    dt is constant across the dataset, so it cancels in the relative L2 this is used in and is
    left out, exactly as dx is in the spatial operators.  Interior points only: a one-sided
    difference at the ends would weight the two boundary frames differently from the rest.
    """
    return 0.5 * (f[:, 2:] - f[:, :-2])


def radial_spectrum(f: torch.Tensor, idx: torch.Tensor, nb: int) -> torch.Tensor:
    """Radially binned energy of the FLUCTUATION field: (B, nb).

    Phase-free by construction, so it can ask for the missing high-k *amplitude* without
    demanding phase agreement the model cannot deliver above k ~ 15.
    """
    fl = f - f.mean(1, keepdim=True)
    e = (torch.fft.rfft2(fl, dim=(2, 3)).abs() ** 2).mean((1, 4))    # (B, H, Wr)
    b = len(e)
    out = torch.zeros(b, nb, device=f.device, dtype=e.dtype)
    return out.index_add_(1, idx.reshape(-1), e.reshape(b, -1))


def radial_index(h: int, w: int, device) -> tuple[torch.Tensor, int]:
    """Isotropic radial bin per (ky, kx); matches diag_spectral.radial_index."""
    ky = torch.fft.fftfreq(h, device=device) * h
    kx = torch.fft.rfftfreq(w, device=device) * w
    r = torch.sqrt(ky[:, None] ** 2 * (w / h) ** 2 + kx[None, :] ** 2)
    idx = r.round().long().clamp(max=w // 2)
    return idx, int(idx.max().item()) + 1


def build(arch: str, width: int, var_head: bool, extra_ch: int, modes_y: int = 12,
          modes_x: int = 16, n_layers: int = 4, p: int = 256, var_mlp: bool = False,
          lift_mlp: bool = False, point_mlp: bool = False,
          var_hidden: int = 128, var_deep: bool = False, var_in_std: bool = False,
          var_pred_std: bool = False, fgn_dim: int = 0, dropout: float = 0.0,
          mean_head: bool = False, osc_modes: int = 0,
          osc_global: bool = False,
          osc_zeromean: bool = False, tbasis: int = 0, mode_drop: float = 0.0):
    """Factory shared by train2.py and eval_wp.run so both build identical objects.

    `fgn_dim` defaults to 0, which builds no `film` module at all -- so a state_dict saved
    before FGN existed still loads with strict=True and every pre-round-12 checkpoint is
    unaffected.
    """
    from model import UNetTwoHead
    if arch == "unet":
        # `dropout` is threaded through so the U-Net can be run at round 15's operating point.
        # Dropout2d adds no parameters and is inert in eval(), so a --dropout U-Net checkpoint
        # loads unchanged into the submission's vendored UNetTwoHead.
        return UNetTwoHead(width, var_head=var_head, extra_ch=extra_ch, dropout=dropout)
    if arch == "afno":
        return AFNO2d(width, modes_y, modes_x, n_layers, var_head, extra_ch, dropout=dropout)
    if arch == "dilres":
        return DilResNet(width, n_layers, var_head, extra_ch, dropout=dropout)
    if arch == "ufno":
        return UFNO2d(width, modes_y, modes_x, n_layers, var_head, extra_ch, dropout=dropout)
    if arch == "vit":
        # --width is the token dimension and --n-layers the depth, so the sweep flags keep
        # meaning "how big" and "how deep" across every family. 64 channels per head.
        return ViT2d(width, n_layers, var_head, extra_ch, heads=max(width // 64, 1),
                     dropout=dropout)
    if arch == "fno":
        # --mode-drop is FNO-only on purpose: it is DropConnect over the retained spectral
        # coefficients, and the families above either have no spectral block (unet, dilres,
        # vit) or reach their modes through a different parameterization (afno's block MLP,
        # ufno's parallel conv branch, which would keep an undropped path open anyway).
        return FNO2d(width, modes_y, modes_x, n_layers, var_head, extra_ch, var_mlp, lift_mlp,
                     point_mlp, var_hidden, var_deep, var_in_std,
                     var_pred_std, fgn_dim, dropout, mean_head, osc_modes, osc_global,
                     osc_zeromean, tbasis, mode_drop)
    if arch == "deeponet":
        return DeepONetCh(p, trunk_width=width * 2, branch_width=width,
                          var_head=var_head, extra_ch=extra_ch)
    raise ValueError(arch)


# ===========================================================================
# Round 15: four more families at the capacity x regularization operating point.
#
# WHY THESE FOUR.  Round 14 measured that the shipped width-24 FNO UNDERFITS -- trained on
# eight trajectories and scored IN SAMPLE it reaches rho 0.588, barely above its own held-out
# 0.609, while a w96/400-epoch run reaches 0.945. Every architecture verdict this project
# recorded (rounds 8-11: "capacity is monotonically harmful", "the architecture search is
# finished") was measured on a model that could not fit the data it already had, with the
# regularizer held fixed. So the family question is reopened, but only at the new operating
# point: width 96, dropout 0.30, 400 epochs.
#
# Each one is a different answer to "what mixes information across the 32x64 grid":
#   afno   -- FNO's truncated LINEAR spectral map replaced by a nonlinear block-diagonal MLP
#             on the modes, with soft-thresholding. Keeps the global receptive field and the
#             spectral bias that made FNO win here; adds the nonlinearity FNO lacks in k-space.
#   dilres -- no spectral machinery at all: stacked dilated convolutions reach the whole grid
#             in one block. The control for "is the Fourier parameterization doing the work,
#             or just the receptive field?"
#   ufno   -- FNO block PLUS a small U-Net branch in parallel. The wake is a localized
#             structure on a uniform background; a mode-truncated operator is a low-pass and
#             cannot represent the airfoil's sharp zero set, which a conv branch can.
#   vit    -- global attention over 4x4 patches. The one family whose mixing is DATA-dependent
#             rather than fixed, and the one the project has never run (Transolver was
#             disqualified on latency at 12.5M params and 109 ms; at 128 tokens this is not
#             that model).
#
# All four expose FNO2d's exact interface -- forward(prev, detach_var) -> (field, logstd) on
# (B, CH+extra+2 -> CH, H, W) -- so train2.py, compare_v9, gap and ens_v9 drive them unchanged
# and every comparison is on identical footing. All four take `dropout` because that is the
# axis the round is testing, and none of them takes var_mlp/lift_mlp: those flags exist to give
# FNO2d's LINEAR 1x1 boundaries a hidden layer, and every model below already has MLP heads.
# ===========================================================================


def _heads(width: int, var_head: bool):
    """The two readouts FNO2d uses, so head capacity is not a confound between families.

    A 128-wide MLP on each. `--var-mlp` and `--lift-mlp` were each worth +0.08 on FNO2d
    precisely because its defaults were linear 1x1 probes; starting the new families at the
    linear default would re-run that experiment inside every one of them.
    """
    head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, CH, 1))
    head_logstd = None
    if var_head:
        head_logstd = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, C, 1))
        nn.init.zeros_(head_logstd[-1].weight)
        nn.init.constant_(head_logstd[-1].bias, float(np.log(0.3)))
    return head_field, head_logstd


class _GridIn(nn.Module):
    """Shared input stage: concat the (x, y) coordinate grid, exactly as FNO2d does."""

    def __init__(self):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def cat(self, prev: torch.Tensor) -> torch.Tensor:
        return torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1)


class AFNO2d(_GridIn):
    """Adaptive FNO (Guibas et al., ICLR 2022), at this project's operating point.

    FNO's spectral block is a LINEAR map per retained mode: `out[k] = W[k] @ in[k]`, with one
    complex matrix per (ky, kx). That is where its parameters live (2*width^2*my*mx) and it is
    exactly the term round 8 measured as monotonically harmful once width grew. AFNO replaces
    it with a two-layer MLP whose weights are SHARED across modes and block-diagonal across
    channels, so parameters go as 2*(width/blocks)^2*blocks = 2*width^2/blocks, independent of
    how many modes are kept -- 8x fewer here at blocks=8 -- and the map becomes nonlinear in
    k-space, which FNO's never is.

    The soft-threshold after the MLP is AFNO's own regularizer: it zeroes modes whose response
    is below `lam`, so the operator's spectral SUPPORT is learned per input rather than fixed
    by a truncation constant. That is the property this round wants -- sparsity chosen by the
    data instead of capacity chosen by hand.

    Modes are still truncated at (modes_y, modes_x) so the comparison against FNO2d holds the
    receptive field in k-space fixed and varies only what happens to the modes that are kept.

    CAVEAT, and it is the whole risk of this arm: because the spectral weights no longer scale
    with the mode count, an AFNO at width 96 has ~0.3 M parameters against the w96 FNO's ~14 M.
    Round 14's diagnosis is that this model class UNDERFITS, so a family that is 40x smaller is
    being asked to win from the wrong side of that. `blocks` and `mlp_ratio` are the two knobs
    that buy it back; the honest screen is rule 4 -- read the training `field` loss against the
    w96 FNO's 0.0391 before scoring anything, and drop the arm if it cannot fit.
    """

    def __init__(self, width: int = 96, modes_y: int = 8, modes_x: int = 12,
                 n_layers: int = 4, var_head: bool = True, extra_ch: int = 0,
                 blocks: int = 4, mlp_ratio: float = 4.0, lam: float = 0.01,
                 dropout: float = 0.0):
        super().__init__()
        assert width % blocks == 0, f"width {width} must divide into {blocks} blocks"
        self.my, self.mx, self.blocks, self.lam = modes_y, modes_x, blocks, lam
        d = width // blocks
        hid = max(1, int(d * mlp_ratio))
        self.lift = nn.Sequential(nn.Conv2d(CH + extra_ch + 2, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, width, 1))
        s = 1.0 / d
        # (L, blocks, d, hid, 2) -- the trailing 2 is (real, imag), viewed as complex at use.
        self.w1 = nn.Parameter(s * torch.randn(n_layers, blocks, d, hid, 2))
        self.b1 = nn.Parameter(torch.zeros(n_layers, blocks, hid, 2))
        self.w2 = nn.Parameter(s * torch.randn(n_layers, blocks, hid, d, 2))
        self.b2 = nn.Parameter(torch.zeros(n_layers, blocks, d, 2))
        self.point = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head_field, self.head_logstd = _heads(width, var_head)

    def _spectral(self, x: torch.Tensor, i: int) -> torch.Tensor:
        b, cw, h, w = x.shape
        ft = torch.fft.rfft2(x)
        out = torch.zeros_like(ft)
        my, mx = min(self.my, h // 2), min(self.mx, w // 2 + 1)
        w1, b1 = torch.view_as_complex(self.w1[i]), torch.view_as_complex(self.b1[i])
        w2, b2 = torch.view_as_complex(self.w2[i]), torch.view_as_complex(self.b2[i])
        for sl in (slice(0, my), slice(h - my, h)):
            # (B, blocks, d, ky, kx) -> block-diagonal complex MLP, shared over (ky, kx).
            z = ft[:, :, sl, :mx].reshape(b, self.blocks, -1, my, mx)
            z = torch.einsum("bkdyx,kdh->bkhyx", z, w1) + b1[None, :, :, None, None]
            # Complex GELU applied to the two parts separately: the standard AFNO nonlinearity,
            # and the only one that keeps the map equivariant to the real-input symmetry.
            z = torch.complex(torch.nn.functional.gelu(z.real),
                              torch.nn.functional.gelu(z.imag))
            z = torch.einsum("bkhyx,khd->bkdyx", z, w2) + b2[None, :, :, None, None]
            # Soft-threshold on the magnitude: shrink toward zero, keep the phase.
            mag = z.abs()
            z = z * (torch.clamp(mag - self.lam, min=0.0) / (mag + 1e-9))
            out[:, :, sl, :mx] = z.reshape(b, cw, my, mx)
        return torch.fft.irfft2(out, s=(h, w))

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = self.lift(self.cat(prev))
        for i, (p, n) in enumerate(zip(self.point, self.norm)):
            h = h + self.drop(torch.nn.functional.gelu(n(self._spectral(h, i) + p(h))))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


class DilResNet(_GridIn):
    """Dilated ResNet (Stachenfeld et al., ICLR 2022) -- the no-spectral-machinery control.

    Four residual blocks, each a stack of 3x3 convolutions at dilations 1, 2, 4, 8, 4, 2, 1.
    One block's receptive field is 1 + 2*(1+2+4+8+4+2+1) = 45 cells, larger than the 32x64
    grid's short axis, so like an FNO block it mixes the whole field in one step -- but with a
    LOCAL, translation-equivariant parameterization and no mode truncation.

    That makes it the discriminating control for the project's oldest architectural claim.
    CLAUDE.md attributes FNO's win to "a global spectral convolution with few modes" capturing
    quasi-periodic shedding directly. If the mechanism is really the global receptive field
    rather than the Fourier basis, this matches it; if the Fourier basis is doing the work,
    this loses, and the truncation is confirmed as the useful prior rather than an accident.
    It is also the cheapest family here per parameter, which matters because latency is scored.
    """

    DIL = (1, 2, 4, 8, 4, 2, 1)

    def __init__(self, width: int = 96, n_layers: int = 4, var_head: bool = True,
                 extra_ch: int = 0, dropout: float = 0.0):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv2d(CH + extra_ch + 2, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, width, 1))
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            seq = []
            for d in self.DIL:
                seq += [nn.Conv2d(width, width, 3, padding=d, dilation=d),
                        nn.GroupNorm(8, width), nn.GELU()]
            self.blocks.append(nn.Sequential(*seq[:-1]))     # no GELU before the residual add
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head_field, self.head_logstd = _heads(width, var_head)

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = self.lift(self.cat(prev))
        for blk in self.blocks:
            h = h + self.drop(torch.nn.functional.gelu(blk(h)))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


class UFNO2d(_GridIn):
    """U-FNO (Wen et al., Adv. Water Resour. 2022): an FNO block with a U-Net branch beside it.

    Each block computes spectral(h) + point(h) + unet(h) instead of spectral(h) + point(h).
    The motivation is specific to this data and measurable in it: mode truncation is a
    LOW-PASS, and 3.3 % of the scored elements are exactly zero (outside the PIV field of view
    and inside the airfoil body), i.e. a sharp indicator function that no band-limited operator
    can represent. The pointwise 1x1 branch cannot fix that either -- it has no spatial extent.
    A two-level U-Net branch does, at a fraction of the spectral block's parameters.

    The U-Net branch is deliberately SMALL (width -> width, one down/up level, 3x3 convs). This
    is meant to be an FNO with a local correction, not a U-Net with a spectral correction; the
    plain U-Net is already in the family list as its own arm.
    """

    def __init__(self, width: int = 96, modes_y: int = 8, modes_x: int = 12,
                 n_layers: int = 4, var_head: bool = True, extra_ch: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv2d(CH + extra_ch + 2, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, width, 1))
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, modes_y, modes_x) for _ in range(n_layers)])
        self.point = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for _ in range(n_layers):
            self.down.append(nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1),
                                           nn.GroupNorm(8, width), nn.GELU(),
                                           nn.Conv2d(width, width, 3, padding=1),
                                           nn.GroupNorm(8, width), nn.GELU()))
            self.up.append(nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1),
                                         nn.GroupNorm(8, width), nn.GELU()))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head_field, self.head_logstd = _heads(width, var_head)

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = self.lift(self.cat(prev))
        for s, p, n, dn, up in zip(self.spec, self.point, self.norm, self.down, self.up):
            d = nn.functional.interpolate(dn(h), size=h.shape[-2:], mode="nearest")
            h = h + self.drop(torch.nn.functional.gelu(n(s(h) + p(h) + up(torch.cat([d, h], 1)))))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


class ViT2d(_GridIn):
    """A plain space-time ViT over 4x4 patches: 8 x 16 = 128 tokens of the 32x64 grid.

    Time is already folded into channels, so one token is a 4x4x20-frame space-time block and
    self-attention mixes every such block with every other in ONE layer -- a global,
    input-DEPENDENT mixing, which is the property no other family here has (FNO's spectral
    map, the dilated stack and the U-Net branch are all fixed linear operators over space).

    Transolver's disqualification does not carry over. It scored 74.82 at 109 ms/window, but
    that is 12.5M parameters doing slice-attention over 2048 points; this is 128 tokens, where
    the attention matrix is 128x128 and the cost is dominated by the MLPs, not the attention.
    Latency is measured before anything is shipped.

    Regularization is the round's axis, so both of the transformer-specific regularizers are
    on: `dropout` inside attention and the MLPs, and stochastic depth (drop-path) at a fixed
    0.1 linearly scaled over depth, which is what makes deep ViTs trainable on small data.
    Unpatchify returns a (B, width, H, W) map so the SAME two conv heads as every other family
    read it out -- the comparison is of trunks, not of readouts.
    """

    def __init__(self, dim: int = 384, depth: int = 8, var_head: bool = True,
                 extra_ch: int = 0, patch: int = 4, heads: int = 6, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path: float = 0.1):
        super().__init__()
        self.patch = patch
        gh, gw = H // patch, W // patch
        self.gh, self.gw = gh, gw
        self.embed = nn.Conv2d(CH + extra_ch + 2, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, gh * gw, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                "n2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout)),
            }))
        # Linearly increasing drop-path rate with depth, the standard schedule.
        self.dp = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.norm_out = nn.LayerNorm(dim)
        # Unpatchify to a per-pixel feature map of `width = dim // patch**2` channels, then the
        # shared conv heads. Going straight from a token to 4x4x40 outputs would put the whole
        # readout in one linear layer and make the family incomparable to the others.
        self.wid = max(dim // (patch * patch), 8)
        self.proj = nn.Linear(dim, self.wid * patch * patch)
        self.head_field, self.head_logstd = _heads(self.wid, var_head)

    def _drop_path(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0.0 or not self.training:
            return x
        keep = 1.0 - p
        mask = x.new_empty(len(x), 1, 1).bernoulli_(keep)
        return x * mask / keep

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        b = len(prev)
        t = self.embed(self.cat(prev)).flatten(2).transpose(1, 2) + self.pos   # (B, N, dim)
        for blk, p in zip(self.blocks, self.dp):
            q = blk["n1"](t)
            t = t + self._drop_path(blk["attn"](q, q, q, need_weights=False)[0], p)
            t = t + self._drop_path(blk["mlp"](blk["n2"](t)), p)
        t = self.proj(self.norm_out(t))                        # (B, N, wid*patch*patch)
        pp = self.patch
        h = (t.reshape(b, self.gh, self.gw, self.wid, pp, pp)
              .permute(0, 3, 1, 4, 2, 5).reshape(b, self.wid, H, W))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd
