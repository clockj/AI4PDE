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
                 n_layers: int = 4, var_head: bool = True, extra_ch: int = 0):
        super().__init__()
        self.lift = nn.Conv2d(CH + extra_ch + 2, width, 1)   # +2 for the coordinate grid
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, modes_y, modes_x) for _ in range(n_layers)])
        self.point = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, CH, 1))
        self.head_logstd = nn.Conv2d(width, C, 1) if var_head else None
        if self.head_logstd is not None:
            nn.init.zeros_(self.head_logstd.weight)
            nn.init.constant_(self.head_logstd.bias, float(np.log(0.3)))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = self.lift(torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1))
        for s, p, n in zip(self.spec, self.point, self.norm):
            h = h + torch.nn.functional.gelu(n(s(h) + p(h)))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


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
          modes_x: int = 16, n_layers: int = 4, p: int = 256):
    """Factory shared by train2.py and eval_wp.run so both build identical objects."""
    from model import UNetTwoHead
    if arch == "unet":
        return UNetTwoHead(width, var_head=var_head, extra_ch=extra_ch)
    if arch == "fno":
        return FNO2d(width, modes_y, modes_x, n_layers, var_head, extra_ch)
    if arch == "deeponet":
        return DeepONetCh(p, trunk_width=width * 2, branch_width=width,
                          var_head=var_head, extra_ch=extra_ch)
    raise ValueError(arch)
