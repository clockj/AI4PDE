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


class FNO2d(nn.Module):
    """Fourier Neural Operator over the 32x64 grid, time folded into channels."""

    def __init__(self, width: int = 32, modes_y: int = 12, modes_x: int = 16,
                 n_layers: int = 4, var_head: bool = True, extra_ch: int = 0,
                 var_mlp: bool = False, lift_mlp: bool = False, point_mlp: bool = False,
                 var_hidden: int = 128, var_deep: bool = False, var_in_std: bool = False,
                 var_pred_std: bool = False, fgn_dim: int = 0):
        super().__init__()
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
            h = h + torch.nn.functional.gelu(a)
            hs.append(h)
        field = self.head_field(h)
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
          var_pred_std: bool = False, fgn_dim: int = 0):
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
                     var_pred_std, fgn_dim)
    if arch == "deeponet":
        return DeepONetCh(p, trunk_width=width * 2, branch_width=width,
                          var_head=var_head, extra_ch=extra_ch)
    raise ValueError(arch)
