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

    def __init__(self, width: int, modes_y: int, modes_x: int):
        super().__init__()
        self.my, self.mx = modes_y, modes_x
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
        out[:, :, :my, :mx] = self._mul(ft[:, :, :my, :mx], self.w1[:, :, :my, :mx])
        out[:, :, -my:, :mx] = self._mul(ft[:, :, -my:, :mx], self.w2[:, :, :my, :mx])
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
                 osc_global: bool = False, osc_zeromean: bool = False):
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
            [SpectralConv2d(width, modes_y, modes_x) for _ in range(n_layers)])
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
          osc_zeromean: bool = False):
    """Factory shared by train2.py and eval_wp.run so both build identical objects.

    `fgn_dim` defaults to 0, which builds no `film` module at all -- so a state_dict saved
    before FGN existed still loads with strict=True and every pre-round-12 checkpoint is
    unaffected.
    """
    from model import UNetTwoHead
    if arch == "unet":
        return UNetTwoHead(width, var_head=var_head, extra_ch=extra_ch)
    if arch == "fno":
        return FNO2d(width, modes_y, modes_x, n_layers, var_head, extra_ch, var_mlp, lift_mlp,
                     point_mlp, var_hidden, var_deep, var_in_std,
                     var_pred_std, fgn_dim, dropout, mean_head, osc_modes, osc_global, osc_zeromean)
    if arch == "deeponet":
        return DeepONetCh(p, trunk_width=width * 2, branch_width=width,
                          var_head=var_head, extra_ch=extra_ch)
    raise ValueError(arch)
