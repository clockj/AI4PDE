"""The CLOSED model families, moved verbatim out of `models_alt.py` on 2026-09-28.

Every class here is a family this project screened and closed (ROADMAP.md Part 2): DeepONet
(early rounds), the round-15 families AFNO, DilResNet and U-FNO, and the 2026-09-14 wave-1
backbones -- ConvNeXt U-Net, PDE-Arena's U-Net, the SwinV2 U-Net, SimVP v2 and the divided
space-time ViT -- plus the `--oracle-cond` diagnostic's `oracle_cond_features`. No final
artefact needs them; they are kept so the checkpoints on disk (arch_afno_*, arch_ufno_*,
arch_dil*, ufno_*, swin_all_*) still rebuild and the verifiers that compare them against the
vendored `_fno.py` copies still run.

Do not import this module for a model: go through `models_alt`, which re-exports every name
below (`models_alt._CLOSED_NAMES`) and whose `build` / `build_wave1` construct these classes
exactly as before. The shared stages (`_GridIn`, `_heads`, `_LayerNorm2d`, `SpectralConv2d`)
stay in `models_alt` and are imported from there; the last line of this file hands the classes
back to `models_alt` (see the note at the bottom of that file for the import order). Bodies,
defaults and parameter creation order are unchanged, so a seeded build draws the same numbers.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models_alt
from model import C, CH, H, T_OUT, W
from models_alt import SpectralConv2d, _GridIn, _heads, _LayerNorm2d


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


def _drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Per-SAMPLE stochastic depth (Huang et al. 2016) for a tensor of any rank, batch first."""
    if p <= 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
    return x * mask / keep


def _trunc_init(m: nn.Module) -> None:
    """ConvNeXt / Swin / TimeSformer initialisation: trunc-normal(0.02) weights, zero biases."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _var_mlp(cin: int, var_head: bool):
    """The sigma readout every wave-1 family uses: 128-wide MLP, last layer at log(0.3)."""
    if not var_head:
        return None
    head = nn.Sequential(nn.Conv2d(cin, 128, 1), nn.GELU(), nn.Conv2d(128, C, 1))
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, float(np.log(0.3)))
    return head


# ---------------------------------------------------------------------------------------------
# 1. cnext_unet -- The Well's CNextU-Net (Ohana et al., NeurIPS 2024 D&B)
# ---------------------------------------------------------------------------------------------
class _ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al., CVPR 2022): depthwise 7x7 -> channels-last LayerNorm ->
    Linear 4x -> GELU -> Linear -> layer scale -> Dropout2d -> residual."""

    def __init__(self, dim: int, dropout: float = 0.0, ls_init: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(ls_init)))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x).permute(0, 2, 3, 1)))))
        h = h * self.gamma.to(h.dtype)
        return x + self.drop(h.permute(0, 3, 1, 2))


class CNeXtUNet(_GridIn):
    """ConvNeXt U-Net in the form of The Well's `UNetConvNext`, sized for the 32x64 grid.

    Levels 32x64 -> 16x32 -> 8x16 -> 4x8 (STAGES = 3 stride-2 downsamplings; a fourth would leave
    a 2x4 map that a 7x7 depthwise kernel sees mostly as padding) at widths w, 2w, 4w, 8w.
    Encoder: `n_layers` ConvNeXt blocks per level, then LayerNorm + 2x2 stride-2 conv. Neck:
    one block at 8w. Decoder: LayerNorm + 2x2 stride-2 transposed conv, concat the skip,
    1x1 projection, `n_layers` blocks.

    Two deliberate departures from The Well, both because the readout here is POINTWISE (the
    shared 128-wide MLP heads) where The Well ends in a 3x3 `out_proj`: every skip is used,
    including the full-resolution one The Well's decoder never reads, and the decoder keeps blocks
    at full resolution. The neck is ONE block, as The Well's benchmark config has it (scout's copy
    notes: init dim 42, 2 blocks per stage, 1 bottleneck block).
    """

    STAGES = 3
    NECK_BLOCKS = 1

    def __init__(self, width: int = 48, n_layers: int = 2, var_head: bool = True,
                 extra_ch: int = 0, dropout: float = 0.0):
        super().__init__()
        dims = [width * 2 ** i for i in range(self.STAGES + 1)]
        rng = range(self.STAGES)

        def blocks(d, n=n_layers):
            return nn.Sequential(*[_ConvNeXtBlock(d, dropout) for _ in range(n)])

        self.in_proj = nn.Conv2d(CH + extra_ch + 2, width, 3, padding=1)
        self.enc = nn.ModuleList([blocks(dims[i]) for i in rng])
        self.down = nn.ModuleList([nn.Sequential(_LayerNorm2d(dims[i], eps=1e-6),
                                                 nn.Conv2d(dims[i], dims[i + 1], 2, stride=2))
                                   for i in rng])
        self.neck = blocks(dims[-1], self.NECK_BLOCKS)
        self.up = nn.ModuleList([nn.Sequential(_LayerNorm2d(dims[i + 1], eps=1e-6),
                                               nn.ConvTranspose2d(dims[i + 1], dims[i], 2,
                                                                  stride=2))
                                 for i in reversed(rng)])
        self.merge = nn.ModuleList([nn.Conv2d(2 * dims[i], dims[i], 1) for i in reversed(rng)])
        self.dec = nn.ModuleList([blocks(dims[i]) for i in reversed(rng)])
        self.norm_out = _LayerNorm2d(width, eps=1e-6)
        self.apply(_trunc_init)
        self.head_field, self.head_logstd = _heads(width, var_head)

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = self.in_proj(self.cat(prev))
        skips = []
        for blk, down in zip(self.enc, self.down):
            h = blk(h)
            skips.append(h)
            h = down(h)
        h = self.neck(h)
        for up, merge, blk, s in zip(self.up, self.merge, self.dec, reversed(skips)):
            h = blk(merge(torch.cat([up(h), s], 1)))
        h = self.norm_out(h)
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


# ---------------------------------------------------------------------------------------------
# 2. unet_mod -- PDE-Arena's modern U-Net "Unetmod" (Gupta & Brandstetter, 2022)
# ---------------------------------------------------------------------------------------------
class _PreActResBlock(nn.Module):
    """PDE-Arena's ResidualBlock: GroupNorm -> GELU -> 3x3 conv -> GroupNorm -> GELU -> 3x3 conv,
    plus a 1x1 shortcut when the width changes. Dropout2d on the branch before the add."""

    def __init__(self, cin: int, cout: int, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return self.skip(x) + self.drop(h)


class UNetMod(_GridIn):
    """PDE-Arena `Unet` with ch_mults (1, 2, 2, 4), `n_layers` blocks per level, no attention.

    The channel bookkeeping is PDE-Arena's exactly, including its CUMULATIVE multipliers (each
    level multiplies the previous level's width, so hidden w gives w, 2w, 4w, 16w at 32x64, 16x32,
    8x16, 4x8), the skip after EVERY down module (blocks and downsamplers), the n_layers + 1
    up-blocks per level of which the last changes width, and the 4x4 stride-2 transposed upsampler.
    Normalisation is GroupNorm(8) where PDE-Arena defaults to one group, matching FNO2d's GN(8).
    The 3x3 zero-initialised output conv is replaced by the shared MLP heads, read after the final
    GroupNorm + GELU.
    """

    CH_MULTS = (1, 2, 2, 4)

    def __init__(self, width: int = 32, n_layers: int = 2, var_head: bool = True,
                 extra_ch: int = 0, dropout: float = 0.0):
        super().__init__()
        assert width % 8 == 0, f"unet_mod width {width} must be a multiple of 8 (GroupNorm(8))"
        nres = len(self.CH_MULTS)
        self.image_proj = nn.Conv2d(CH + extra_ch + 2, width, 3, padding=1)
        down, skip_ch = [], [width]
        cin = cout = width
        for i, mult in enumerate(self.CH_MULTS):
            cout = cin * mult
            for _ in range(n_layers):
                down.append(_PreActResBlock(cin, cout, dropout))
                cin = cout
                skip_ch.append(cin)
            if i < nres - 1:
                down.append(nn.Conv2d(cin, cin, 3, stride=2, padding=1))
                skip_ch.append(cin)
        self.down = nn.ModuleList(down)
        self.mid = nn.Sequential(_PreActResBlock(cout, cout, dropout),
                                 _PreActResBlock(cout, cout, dropout))
        up = []
        cin = cout
        for i in reversed(range(nres)):
            cout = cin
            for _ in range(n_layers):
                up.append(_PreActResBlock(cin + skip_ch.pop(), cout, dropout))
            cout = cin // self.CH_MULTS[i]
            up.append(_PreActResBlock(cin + skip_ch.pop(), cout, dropout))
            cin = cout
            if i > 0:
                up.append(nn.ConvTranspose2d(cin, cin, 4, stride=2, padding=1))
        assert not skip_ch
        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, width)
        self.head_field, self.head_logstd = _heads(width, var_head)

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        x = self.image_proj(self.cat(prev))
        hs = [x]
        for m in self.down:
            x = m(x)
            hs.append(x)
        x = self.mid(x)
        for m in self.up:
            x = m(x) if isinstance(m, nn.ConvTranspose2d) else m(torch.cat([x, hs.pop()], 1))
        h = F.gelu(self.norm(x))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


# ---------------------------------------------------------------------------------------------
# 4. swin_unet -- SwinV2 blocks (Liu et al., CVPR 2022) in a Swin-Unet / scOT-style U-Net
# ---------------------------------------------------------------------------------------------
def _win_part(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.reshape(b, h // wh, wh, w // ww, ww, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, wh * ww, c)


def _win_merge(win: torch.Tensor, wh: int, ww: int, h: int, w: int) -> torch.Tensor:
    c = win.shape[-1]
    b = win.shape[0] // ((h // wh) * (w // ww))
    x = win.reshape(b, h // wh, w // ww, wh, ww, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, c)


class _WindowAttnV2(nn.Module):
    """SwinV2 window attention: SCALED COSINE attention with a learnable per-head temperature
    (clamped at 100) and a CONTINUOUS relative position bias -- a 2 -> 512 -> heads MLP over
    log-spaced relative coordinates. Runs through F.scaled_dot_product_attention with the bias
    (plus the shifted-window mask) as an additive float mask and scale 1."""

    def __init__(self, dim: int, window: tuple[int, int], heads: int, dropout: float = 0.0):
        super().__init__()
        wh, ww = window
        assert dim % heads == 0
        self.heads, self.attn_p = heads, float(dropout)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.log(10.0 * torch.ones(heads, 1, 1)))
        self.cpb = nn.Sequential(nn.Linear(2, 512), nn.ReLU(), nn.Linear(512, heads))
        dy = torch.arange(-(wh - 1), wh, dtype=torch.float32) / max(wh - 1, 1)
        dx = torch.arange(-(ww - 1), ww, dtype=torch.float32) / max(ww - 1, 1)
        tab = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), -1) * 8.0
        tab = torch.sign(tab) * torch.log2(tab.abs() + 1.0) / math.log2(8.0)
        self.register_buffer("coords", tab.reshape(-1, 2), persistent=False)
        cy, cx = torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij")
        c = torch.stack([cy.flatten(), cx.flatten()])                        # (2, N)
        rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0)              # (N, N, 2)
        idx = (rel[..., 0] + wh - 1) * (2 * ww - 1) + (rel[..., 1] + ww - 1)
        self.register_buffer("rel_index", idx.reshape(-1), persistent=False)
        # Inference cache (SwinUNet.prepare_inference): None = the as-trained path, recomputed per call.
        self._frozen = None

    @torch.no_grad()
    def freeze_cache(self) -> None:
        """Compute the logit scale and the continuous position bias ONCE, in fp32, by the very ops
        forward() runs, and keep them -- plus, lazily per (batch, dtype), the attention mask forward()
        would build -- so the eval path skips the CPB MLP, the clamp/exp and the mask arithmetic on
        every call. Same tensors, same strides: exact against the uncached forward. .train() drops it."""
        n = int(round(self.rel_index.numel() ** 0.5))
        with torch.autocast(device_type=self.coords.device.type, enabled=False):
            scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
            bias = (self.cpb(self.coords)[self.rel_index]
                    .reshape(n, n, self.heads).permute(2, 0, 1))
        self._frozen = {"scale": scale, "bias": bias, "am": {}}

    def train(self, mode: bool = True):
        if mode:
            self._frozen = None
        return super().train(mode)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bw, n, c = x.shape
        q, k, v = (self.qkv(x).reshape(bw, n, 3, self.heads, c // self.heads)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        fz = None if self.training else self._frozen
        scale = (torch.clamp(self.logit_scale, max=math.log(100.0)).exp() if fz is None
                 else fz["scale"])
        q = F.normalize(q, dim=-1) * scale
        k = F.normalize(k, dim=-1)
        dt = v.dtype
        key = (bw, dt, mask is not None)
        if fz is not None and key in fz["am"]:
            am = fz["am"][key]
        else:
            bias = ((self.cpb(self.coords)[self.rel_index]
                     .reshape(n, n, self.heads).permute(2, 0, 1)) if fz is None
                    else fz["bias"])                                         # (heads, N, N)
            if mask is None:
                am = bias.unsqueeze(0)
            else:                                                            # mask (nW, 1, N, N)
                nw = mask.shape[0]
                am = ((bias[None, None] + mask[None]).expand(bw // nw, nw, self.heads, n, n)
                      .reshape(bw, self.heads, n, n))
            am = am.to(dt)
            if fz is not None:
                fz["am"][key] = am
        o = F.scaled_dot_product_attention(q.to(dt), k.to(dt), v, attn_mask=am,
                                           dropout_p=self.attn_p if self.training else 0.0,
                                           scale=1.0)
        return self.proj_drop(self.proj(o.transpose(1, 2).reshape(bw, n, c)))


class _SwinV2Block(nn.Module):
    """RES-POST-NORM SwinV2 block on (B, H, W, C): x + DropPath(LN(attn(x))), then the same for
    the MLP. `shift` > 0 cyclically shifts by half a window along every axis the window does not
    already cover, with Swin's -100 mask so no window mixes tokens across the wrap-around."""

    def __init__(self, dim: int, res: tuple[int, int], heads: int, window: int, shift: bool,
                 dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        h, w = res
        wh, ww = min(window, h), min(window, w)
        sy = wh // 2 if (shift and wh < h) else 0
        sx = ww // 2 if (shift and ww < w) else 0
        self.win, self.shift, self.dp = (wh, ww), (sy, sx), float(drop_path)
        self.attn = _WindowAttnV2(dim, (wh, ww), heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)
        mask = None
        if sy or sx:
            img = torch.zeros(1, h, w, 1)
            ys = (slice(0, -wh), slice(-wh, -sy), slice(-sy, None)) if sy else (slice(None),)
            xs = (slice(0, -ww), slice(-ww, -sx), slice(-sx, None)) if sx else (slice(None),)
            cnt = 0
            for a in ys:
                for b_ in xs:
                    img[:, a, b_, :] = cnt
                    cnt += 1
            mw = _win_part(img, wh, ww).squeeze(-1)                          # (nW, N)
            d = mw[:, None, :] - mw[:, :, None]
            mask = torch.zeros_like(d).masked_fill(d != 0, -100.0).unsqueeze(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = x.shape
        sy, sx = self.shift
        y = torch.roll(x, (-sy, -sx), (1, 2)) if (sy or sx) else x
        y = _win_merge(self.attn(_win_part(y, *self.win), self.mask), *self.win, h, w)
        if sy or sx:
            y = torch.roll(y, (sy, sx), (1, 2))
        x = x + _drop_path(self.norm1(y), self.dp, self.training)
        return x + _drop_path(self.norm2(self.mlp(x)), self.dp, self.training)


class _PatchMergeV2(nn.Module):
    """2x2 token merge (B, H, W, C) -> (B, H/2, W/2, 2C): concat, Linear 4C -> 2C, then LN (V2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]],
                      -1)
        return self.norm(self.reduction(x))


class _PatchExpand(nn.Module):
    """Swin-Unet's patch expanding (B, H, W, C) -> (B, 2H, 2W, out): Linear C -> 4*out, depth-to-
    space, LayerNorm."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 4 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        x = (self.expand(x).reshape(b, h, w, 2, 2, -1).permute(0, 1, 3, 2, 4, 5)
             .reshape(b, 2 * h, 2 * w, -1))
        return self.norm(x)


class _TokenProj(nn.Module):
    """(B, H, W, C) -> (B, H, W, out): Linear + LayerNorm, the patch-1 stand-in for _PatchExpand."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class SwinUNet(_GridIn):
    """Hierarchical SwinV2 U-Net on patch-2 tokens (the scOT / Swin-Unet layout).

    2x2 patch embedding (42 -> w) + LN gives a 16x32 token grid. Encoder levels 16x32 (w) and 8x16
    (2w), each `n_layers` blocks alternating regular and shifted windows of 8 then a V2 patch
    merge; bottleneck 4x8 (4w, one global window); every skip passes through SKIP_BLOCKS = 2
    ConvNeXt blocks (scOT's ConvNeXt skip connections, Poseidon config [2,2,2,0]); decoder: patch
    expand, concat the skip, Linear 2d -> d, `n_layers` blocks. A final patch expand to 32x64 (max(w/2, 32) channels) and
    one residual 3x3 conv (the patch-recovery mixing scOT also adds) feed the shared MLP heads.
    Heads = d/32 per level. Absolute position enters through the coordinate grid in the embedding;
    attention itself is relative (continuous position bias).
    """

    WINDOW = 8
    STAGES = 3
    SKIP_BLOCKS = 2

    def __init__(self, width: int = 96, n_layers: int = 2, var_head: bool = True,
                 extra_ch: int = 0, dropout: float = 0.0, drop_path: float = 0.1,
                 window: int | None = None, bottleneck_depth: int | None = None,
                 patch: int = 2):
        super().__init__()
        # Phase-2 ladder knobs (2026-09-14; defaults reproduce wave 1 exactly): `window` for the
        # encoder / decoder stages (the bottleneck is always ONE global window), `bottleneck_depth`
        # blocks at the lowest resolution (default n_layers), `patch` 1 or 2 (token grid 32x64 or
        # 16x32). Arch names swin_unet_win16 / swin_unet_b6 / swin_unet_p1 carry them, because
        # _fno.load_any rebuilds from (arch, width, n_layers) alone.
        assert patch in (1, 2), f"patch must be 1 or 2, not {patch}"
        win = self.WINDOW if window is None else int(window)
        bd = n_layers if bottleneck_depth is None else int(bottleneck_depth)
        res = [((H // patch) >> i, (W // patch) >> i) for i in range(self.STAGES)]
        dims = [width * 2 ** i for i in range(self.STAGES)]
        heads = [max(d // 32, 1) for d in dims]
        n_blocks = n_layers * 2 * (self.STAGES - 1) + bd
        dpr = iter([drop_path * i / max(n_blocks - 1, 1) for i in range(n_blocks)])

        def stage(i, depth=n_layers, window=win):
            return nn.ModuleList([_SwinV2Block(dims[i], res[i], heads[i], window,
                                               bool(j % 2), dropout, next(dpr))
                                  for j in range(depth)])

        self.embed = nn.Conv2d(CH + extra_ch + 2, width, patch, stride=patch)
        self.embed_norm = nn.LayerNorm(width)
        self.enc = nn.ModuleList([stage(i) for i in range(self.STAGES - 1)])
        self.merge = nn.ModuleList([_PatchMergeV2(dims[i]) for i in range(self.STAGES - 1)])
        self.skip_blocks = nn.ModuleList([
            nn.Sequential(*[_ConvNeXtBlock(dims[i], dropout) for _ in range(self.SKIP_BLOCKS)])
            for i in range(self.STAGES - 1)])
        self.bottleneck = stage(self.STAGES - 1, bd, max(res[-1]))
        self.expand, self.skip, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(self.STAGES - 1)):
            self.expand.append(_PatchExpand(dims[i + 1], dims[i]))
            self.skip.append(nn.Linear(2 * dims[i], dims[i]))
            self.dec.append(stage(i))
        self.out_ch = max(width // 2, 32)
        self.final = (_PatchExpand(width, self.out_ch) if patch == 2
                      else _TokenProj(width, self.out_ch))
        self.refine = nn.Sequential(nn.GELU(), nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1))
        self.apply(_trunc_init)
        self.head_field, self.head_logstd = _heads(self.out_ch, var_head)
        # prepare_inference(fp32_max_batch=k): batches of <= k windows run fp32 under autocast.
        self.fp32_max_batch = 0

    @torch.no_grad()
    def prepare_inference(self, fp32_max_batch: int = 0):
        """Track-2 speedups (2026-09-14), both EXACT against the as-trained forward:
        (1) every window attention freezes its logit scale, position bias and per-(batch, dtype)
            mask (`_WindowAttnV2.freeze_cache`) -- the CPB MLP and mask arithmetic leave the step;
        (2) batches of at most `fp32_max_batch` windows run in fp32 even under autocast: at batch 1
            autocast's fp16 <-> fp32 casts around LayerNorm and normalize cost more than fp16 saves.
        Call after loading the weights; .train() drops the frozen tensors."""
        self.eval()
        for m in self.modules():
            if isinstance(m, _WindowAttnV2):
                m.freeze_cache()
        self.fp32_max_batch = int(fp32_max_batch)
        return self

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        if (self.fp32_max_batch and len(prev) <= self.fp32_max_batch and prev.is_cuda
                and torch.is_autocast_enabled()):
            with torch.autocast(device_type="cuda", enabled=False):
                return self._forward(prev.float(), detach_var)
        return self._forward(prev, detach_var)

    def _forward(self, prev: torch.Tensor, detach_var: bool = False):
        x = self.embed_norm(self.embed(self.cat(prev)).permute(0, 2, 3, 1))
        skips = []
        for blocks, merge, sblk in zip(self.enc, self.merge, self.skip_blocks):
            for blk in blocks:
                x = blk(x)
            skips.append(sblk(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
            x = merge(x)
        for blk in self.bottleneck:
            x = blk(x)
        for expand, skip, blocks, s in zip(self.expand, self.skip, self.dec, reversed(skips)):
            x = skip(torch.cat([expand(x), s], -1))
            for blk in blocks:
                x = blk(x)
        h = self.final(x).permute(0, 3, 1, 2)
        h = h + self.refine(h)
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


# ---------------------------------------------------------------------------------------------
# 5. simvp -- SimVP v2 with the gSTA translator (Tan et al., 2022; OpenSTL)
# ---------------------------------------------------------------------------------------------
class _ConvSC(nn.Module):
    """SimVP's ConvSC: 3x3 conv (stride 2 to down-sample, or conv to 4C + PixelShuffle(2) to
    up-sample) -> GroupNorm(2) -> SiLU."""

    def __init__(self, cin: int, cout: int, down: bool = False, up: bool = False):
        super().__init__()
        if up:
            self.conv = nn.Sequential(nn.Conv2d(cin, 4 * cout, 3, padding=1), nn.PixelShuffle(2))
        else:
            self.conv = nn.Conv2d(cin, cout, 3, stride=2 if down else 1, padding=1)
        self.norm = nn.GroupNorm(2, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class _GASubBlock(nn.Module):
    """gSTA sub-block, as OpenSTL's GASubBlock with GroupNorm(8) in place of its BatchNorm (scout:
    BN is fragile under fp16 and batch 32, and a batch-1 Track-2 step): GN -> [1x1 -> GELU -> gated large-kernel attention
    (dw 5x5 -> dw 7x7 dilation 3 -> 1x1 to 2C -> sigmoid(g) * f) -> 1x1] + shortcut, layer scale,
    drop-path; then GN -> MixMlp (1x1 -> dw 3x3 -> GELU -> dropout -> 1x1 -> dropout), layer scale,
    drop-path."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float = 0.0,
                 drop_path: float = 0.0, ls_init: float = 1e-2):
        super().__init__()
        hid = int(dim * mlp_ratio)
        assert dim % 8 == 0, f"gSTA width {dim} must be a multiple of 8 (GroupNorm(8))"
        self.norm1 = nn.GroupNorm(8, dim)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.norm2 = nn.GroupNorm(8, dim)
        self.fc1 = nn.Conv2d(dim, hid, 1)
        self.dw = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
        self.fc2 = nn.Conv2d(hid, dim, 1)
        self.drop = nn.Dropout(dropout)
        self.ls1 = nn.Parameter(torch.full((dim,), float(ls_init)))
        self.ls2 = nn.Parameter(torch.full((dim,), float(ls_init)))
        self.dp = float(drop_path)
        for m in self.modules():                       # OpenSTL GASubBlock._init_weights
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.norm1(x)
        f, g = self.conv1(self.conv_spatial(self.conv0(F.gelu(self.proj1(a))))).chunk(2, 1)
        a = self.proj2(torch.sigmoid(g) * f) + a
        x = x + _drop_path(self.ls1.to(a.dtype)[:, None, None] * a, self.dp, self.training)
        m = self.drop(self.fc2(self.drop(F.gelu(self.dw(self.fc1(self.norm2(x)))))))
        return x + _drop_path(self.ls2.to(m.dtype)[:, None, None] * m, self.dp, self.training)


class _MetaBlock(nn.Module):
    """OpenSTL MetaBlock: a gSTA sub-block at the INPUT width, then a 1x1 width change if needed."""

    def __init__(self, cin: int, cout: int, mlp_ratio: float, dropout: float, drop_path: float):
        super().__init__()
        self.block = _GASubBlock(cin, mlp_ratio, dropout, drop_path)
        self.reduction = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(self.block(x))


class SimVP2(_GridIn):
    """SimVP v2: shared per-frame conv encoder -> gSTA translator over T*C channels -> symmetric
    per-frame decoder with SimVP's full-resolution skip.

    width = hid_T (translator width), n_layers = N_T (translator blocks, >= 2), hid_S = width // 8
    (SimVP's 64/512 ratio). N_S = 4 ConvSC layers, down-sampling at the 2nd and 4th, so the
    translator sees 20 * hid_S channels at 8x16. Frames carry (u, v, x, y). The readout is SimVP's
    1x1 conv per frame. Sigma: concat(mean_t, std_t) of the decoder's last per-frame feature map
    -> 128-wide MLP, so it shares the whole trunk. drop-path linspace(1e-2, 0.1) as OpenSTL;
    `dropout` is the MixMlp dropout. SLIM, per scout's FLOPs count: the stock OpenSTL config is
    16.8x FNO at 20 frames (the full-resolution per-frame encoder and decoder dominate), so
    MLP_RATIO is 2 (OpenSTL: 8) and the width is chosen from the latency table (~1.7x FNO).
    """

    N_S = 4
    MLP_RATIO = 2.0

    def __init__(self, width: int = 256, n_layers: int = 8, var_head: bool = True,
                 extra_ch: int = 0, dropout: float = 0.0, drop_path: float = 0.1):
        super().__init__()
        assert extra_ch == 0, "simvp encodes frames one at a time; --hist / --ctx have no frame"
        assert n_layers >= 2, "the gSTA translator needs an input and an output block"
        hs = max(width // 8, 8)
        assert hs % 2 == 0, "GroupNorm(2) needs an even hid_S"
        self.hid_s, self.t = hs, CH // C
        samp = [False, True] * (self.N_S // 2)
        self.enc = nn.ModuleList([_ConvSC(C + 2, hs, down=samp[0])]
                                 + [_ConvSC(hs, hs, down=s) for s in samp[1:]])
        self.enc.apply(_trunc_init)
        cin = self.t * hs
        dpr = torch.linspace(1e-2, drop_path, n_layers).tolist()
        mid = [_MetaBlock(cin, width, self.MLP_RATIO, dropout, dpr[0])]
        mid += [_MetaBlock(width, width, self.MLP_RATIO, dropout, dpr[i])
                for i in range(1, n_layers - 1)]
        mid += [_MetaBlock(width, cin, self.MLP_RATIO, dropout, drop_path)]
        self.translator = nn.Sequential(*mid)
        self.dec = nn.ModuleList([_ConvSC(hs, hs, up=s) for s in reversed(samp)])
        self.dec.apply(_trunc_init)
        self.readout = nn.Conv2d(hs, C, 1)
        self.head_logstd = _var_mlp(2 * hs, var_head)

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        b, t = len(prev), self.t
        fr = prev[:, :CH].reshape(b, t, C, H, W)
        g = self.grid.unsqueeze(1).expand(b, t, -1, -1, -1).to(fr.dtype)
        z = torch.cat([fr, g], 2).reshape(b * t, C + 2, H, W)
        skip = z = self.enc[0](z)
        for layer in self.enc[1:]:
            z = layer(z)
        _, c_, h_, w_ = z.shape
        z = self.translator(z.reshape(b, t * c_, h_, w_)).reshape(b * t, c_, h_, w_)
        for layer in self.dec[:-1]:
            z = layer(z)
        feat = self.dec[-1](z + skip)
        field = self.readout(feat).reshape(b, t * C, H, W)
        logstd = None
        if self.head_logstd is not None:
            f = feat.reshape(b, t, self.hid_s, H, W)
            zf = torch.cat([f.mean(1), f.std(1)], 1)
            logstd = self.head_logstd(zf.detach() if detach_var else zf)
        return field, logstd


# ---------------------------------------------------------------------------------------------
# 6. stvit -- divided space-time transformer (TimeSformer, Bertasius et al., ICML 2021)
# ---------------------------------------------------------------------------------------------
class _MHSA(nn.Module):
    """Multi-head self-attention through F.scaled_dot_product_attention (flash / mem-efficient)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.p = heads, float(dropout)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = (self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        return self.proj_drop(self.proj(o.transpose(1, 2).reshape(b, n, c)))


class _DividedSTBlock(nn.Module):
    """TimeSformer divided attention on (B, T', S, D), pre-norm: temporal attention (each patch
    location attends across the T' time tokens), spatial attention (each time token attends across
    the S patches), MLP. Drop-path is per SAMPLE on every branch."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm_t, self.norm_s, self.norm_m = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn_t = _MHSA(dim, heads, dropout)
        self.attn_s = _MHSA(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.dp = float(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, d = x.shape
        y = self.attn_t(self.norm_t(x.transpose(1, 2).reshape(b * s, t, d)))
        x = x + _drop_path(y.reshape(b, s, t, d).transpose(1, 2), self.dp, self.training)
        y = self.attn_s(self.norm_s(x.reshape(b * t, s, d)))
        x = x + _drop_path(y.reshape(b, t, s, d), self.dp, self.training)
        return x + _drop_path(self.mlp(self.norm_m(x)), self.dp, self.training)


class STViT(_GridIn):
    """Divided space-time ViT that keeps the frame axis the channel-folded families collapse.

    DROPPED FROM WAVE 1 (plan 09-14 revision): 17-49x FNO FLOPs at 10-20 x 128 tokens, and the ViT
    family saturates at two members. Kept loadable; not trained.

    Input frames (u, v, x, y) are grouped into TUBELET = 2 frames and embedded with a 4x4 stride-4
    patch conv: 10 time tokens x 8x16 = 128 patch tokens, plus learned spatial and temporal position
    embeddings. `n_layers` divided space-time blocks at dim `width` (width/32 heads, dropout in
    attention and MLP, drop-path 0.1 linear). Output: time token j emits output frames 2j and 2j+1
    -- LN, Linear(width -> 2*4*4*16), depth-to-space to 16 channels per output frame, GELU, 1x1 conv
    to (u, v); temporal attention lets every slot read the whole input window. Sigma: the
    time-averaged tokens -> Linear(width -> 4*4*16) -> depth-to-space -> 128-wide MLP.
    """

    PATCH = 4
    TUBELET = 2
    FEAT = 16

    def __init__(self, width: int = 192, n_layers: int = 4, var_head: bool = True,
                 extra_ch: int = 0, dropout: float = 0.0, drop_path: float = 0.1):
        super().__init__()
        assert extra_ch == 0, "stvit tokenizes frames; --hist / --ctx have no frame"
        p, tu = self.PATCH, self.TUBELET
        self.t_in = CH // C
        assert self.t_in % tu == 0 and T_OUT % tu == 0 and self.t_in == T_OUT
        self.nt, self.gh, self.gw = self.t_in // tu, H // p, W // p
        heads = max(width // 32, 1)
        self.embed = nn.Conv2d(tu * (C + 2), width, p, stride=p)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, self.gh * self.gw, width))
        self.pos_t = nn.Parameter(torch.zeros(1, self.nt, 1, width))
        self.blocks = nn.ModuleList([_DividedSTBlock(width, heads, dropout,
                                                     drop_path * i / max(n_layers - 1, 1))
                                     for i in range(n_layers)])
        self.norm_out = nn.LayerNorm(width)
        self.to_frames = nn.Linear(width, tu * p * p * self.FEAT)
        self.to_sigma = nn.Linear(width, p * p * self.FEAT) if var_head else None
        self.apply(_trunc_init)
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        self.head_field = nn.Conv2d(self.FEAT, C, 1)
        self.head_logstd = _var_mlp(self.FEAT, var_head)

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        b, p, tu, nt = len(prev), self.PATCH, self.TUBELET, self.nt
        fr = prev[:, :CH].reshape(b, self.t_in, C, H, W)
        g = self.grid.unsqueeze(1).expand(b, self.t_in, -1, -1, -1).to(fr.dtype)
        z = torch.cat([fr, g], 2).reshape(b * nt, tu * (C + 2), H, W)
        x = self.embed(z).flatten(2).transpose(1, 2).reshape(b, nt, self.gh * self.gw, -1)
        x = x + (self.pos_s + self.pos_t).to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        f = self.to_frames(x).reshape(b, nt, self.gh, self.gw, tu, p, p, self.FEAT)
        f = f.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(b * T_OUT, self.FEAT, H, W)
        field = self.head_field(F.gelu(f)).reshape(b, T_OUT * C, H, W)
        logstd = None
        if self.head_logstd is not None:
            zs = x.mean(1)
            zs = zs.detach() if detach_var else zs
            s = (self.to_sigma(zs).reshape(b, self.gh, self.gw, p, p, self.FEAT)
                 .permute(0, 5, 1, 3, 2, 4).reshape(b, self.FEAT, H, W))
            logstd = self.head_logstd(s)
        return field, logstd


# --oracle-cond condition features (2026-09-14 wave 2 diagnostic). FIXED affine maps of log Re and
# AoA -- no statistic of any split enters -- shared by train2.py and Track_1/screen_oracle.py.
ORACLE_LOGRE_MU, ORACLE_LOGRE_SD = 9.3, 0.6          # log Re spans 8.23 (3750) to 10.19 (26700)
ORACLE_AOA_MU, ORACLE_AOA_SD = 10.0, 7.0710678       # AoA 0, 5, 10, 15, 20


def oracle_cond_features(re, aoa) -> np.ndarray:
    """(N,) Reynolds numbers and (N,) angles of attack in degrees -> (N, 2) float32 features."""
    re = np.asarray(re, dtype=np.float64)
    aoa = np.asarray(aoa, dtype=np.float64)
    return np.stack([(np.log(re) - ORACLE_LOGRE_MU) / ORACLE_LOGRE_SD,
                     (aoa - ORACLE_AOA_MU) / ORACLE_AOA_SD], -1).astype(np.float32)


_LOADED = True
models_alt._reexport_closed(sys.modules[__name__])
