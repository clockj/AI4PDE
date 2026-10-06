"""The two network families of the submission: the factorised Fourier neural operator (F-FNO)
and the space-time vision transformer (ViT), with the shared heads and `build()`.

The module keeps this file name and this directory (`unet/`) because the shipped `_fno.py`
(submission/_fno.py) rebuilds an F-FNO checkpoint through `unet/models_alt.py:build` whenever
that file exists next to it; the band fit therefore runs the training definition, and the
evaluation container (where the directory is absent) runs the vendored copy inside `_fno.py`.

Both families share one contract: `forward(x, detach_var)` on a (B, 42, 32, 64) input (20 frames
of [u, v] plus the coordinate grid) returns the field `(B, 40, 32, 64)` and the per-pixel log
standard deviation `(B, 2, 32, 64)` of the target window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import C, CH, H, T_OUT, W


def velocity_gradient(f: torch.Tensor) -> torch.Tensor:
    """All four components of grad(u) by central differences: (B, T, H, W, 2) -> (B, T, H, W, 4).
    The grid spacing is left out; the tensor is only ever compared in a relative L2."""
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
    """Frame-to-frame time derivative by central differences, interior frames only:
    (B, T, H, W, 2) -> (B, T-2, H, W, 2)."""
    return 0.5 * (f[:, 2:] - f[:, :-2])


def _heads(width: int, var_head: bool):
    """The two readouts: a 128-wide MLP for the field and one for the log standard deviation.
    The latter starts as the constant log(0.3) (zero weights), so the variance loss cannot
    perturb the field head's first steps."""
    head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, CH, 1))
    head_logstd = None
    if var_head:
        head_logstd = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, C, 1))
        nn.init.zeros_(head_logstd[-1].weight)
        nn.init.constant_(head_logstd[-1].bias, float(np.log(0.3)))
    return head_field, head_logstd


class _GridIn(nn.Module):
    """Shared input stage: the (x, y) coordinate grid concatenated to the input channels."""

    def __init__(self):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def cat(self, prev: torch.Tensor) -> torch.Tensor:
        return torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1)


class ViT2d(_GridIn):
    """A space-time ViT over 4 x 4 patches: 8 x 16 = 128 tokens, each a 4 x 4 x 20-frame block,
    mixed globally by self-attention. Dropout inside attention and the MLPs; stochastic depth
    at a fixed 0.1, linearly scaled over the blocks. The tokens are un-patched to a per-pixel
    feature map of `dim / 16` channels, which the same two conv heads as the F-FNO read."""

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
        self.dp = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.norm_out = nn.LayerNorm(dim)
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


class _LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel axis of an NCHW map."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _FFNOLayer(nn.Module):
    """One F-FNO layer (Tran, Mathews, Xie & Ong, ICLR 2023). The Fourier operator is separable:
    an rfft along y, one complex matrix per retained ky, irfft; the same along x; the two summed.
    Then x + Dropout2d(LayerNorm(FF(K(x)))) with FF = Linear -> GELU -> Linear.

    `dense` (set by `FFNO2d.set_spectral_path`) replaces each 1-D FFT by a matrix product on a
    precomputed cos/sin basis: the same operator up to float rounding, and what the shipped
    package runs under half precision. Training uses the FFTs, pinned to fp32."""

    def __init__(self, width: int, modes_y: int, modes_x: int, ff: bool = True,
                 dropout: float = 0.0, factor: int = 2, grid: tuple[int, int] = (H, W)):
        super().__init__()
        self.my, self.mx = min(modes_y, grid[0] // 2 + 1), min(modes_x, grid[1] // 2 + 1)
        self.wy = nn.Parameter(torch.empty(width, width, self.my, 2))
        self.wx = nn.Parameter(torch.empty(width, width, self.mx, 2))
        nn.init.xavier_normal_(self.wy)
        nn.init.xavier_normal_(self.wx)
        self.use_ff = bool(ff)
        if self.use_ff:
            self.ff = nn.Sequential(nn.Linear(width, factor * width), nn.GELU(),
                                    nn.Linear(factor * width, width))
            self.norm = nn.LayerNorm(width)
        else:
            self.point = nn.Conv2d(width, width, 1)
            self.norm = nn.GroupNorm(8, width)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.dense = False
        self._bases: dict = {}

    @staticmethod
    def _dense_basis(n: int, m: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Real (n, 2m) analysis and (2m, n) synthesis matrices equal to rfft(norm="ortho")[:m]
        and irfft(n=n, norm="ortho") of a length-m half spectrum, extracted in float64 by
        applying those very operators to unit vectors."""
        a = torch.fft.rfft(torch.eye(n, dtype=torch.float64), dim=-1, norm="ortho")[:, :m]
        e = torch.eye(m, dtype=torch.complex128)
        s_re = torch.fft.irfft(e, n=n, dim=-1, norm="ortho")
        s_im = torch.fft.irfft(1j * e, n=n, dim=-1, norm="ortho")
        return (torch.cat([a.real, a.imag], 1).to(device=device, dtype=torch.float32),
                torch.cat([s_re, s_im], 0).to(device=device, dtype=torch.float32))

    @staticmethod
    def _w2(wt: torch.Tensor) -> torch.Tensor:
        """(in, out, m, 2) complex weights -> (m, 2*out, 2*in) real 2x2-block matrices
        [[Re W, -Im W], [Im W, Re W]]."""
        wr = wt[..., 0].permute(2, 1, 0)
        wi = wt[..., 1].permute(2, 1, 0)
        return torch.cat([torch.cat([wr, -wi], 2), torch.cat([wi, wr], 2)], 1)

    def _spectral_dense(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        key = (x.device, h, w)
        if key not in self._bases:
            self._bases[key] = (self._dense_basis(h, self.my, x.device),
                                self._dense_basis(w, self.mx, x.device))
        (py, sy), (px, sx) = self._bases[key]
        my, mx = self.my, self.mx
        cy = torch.matmul(x.transpose(-1, -2), py)                            # (B, C, W, 2my)
        cy = cy.reshape(b, c, w, 2, my).permute(4, 0, 2, 3, 1).reshape(my, b * w, 2 * c)
        dy = torch.bmm(cy, self._w2(self.wy).transpose(1, 2))                  # (my, B*W, 2C')
        co = dy.shape[-1] // 2
        dy = dy.reshape(my, b, w, 2, co).permute(1, 4, 2, 3, 0).reshape(b, co, w, 2 * my)
        yy = torch.matmul(dy, sy).transpose(-1, -2)                            # (B, C', H, W)
        cx = torch.matmul(x, px)                                               # (B, C, H, 2mx)
        cx = cx.reshape(b, c, h, 2, mx).permute(4, 0, 2, 3, 1).reshape(mx, b * h, 2 * c)
        dx = torch.bmm(cx, self._w2(self.wx).transpose(1, 2))                  # (mx, B*H, 2C')
        dx = dx.reshape(mx, b, h, 2, co).permute(1, 4, 2, 3, 0).reshape(b, co, h, 2 * mx)
        xx = torch.matmul(dx, sx)                                              # (B, C', H, W)
        return yy + xx

    def _spectral(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        fy = torch.fft.rfft(x, dim=-2, norm="ortho")[:, :, :self.my]
        yy = torch.fft.irfft(torch.einsum("biyx,ioy->boyx", fy, torch.view_as_complex(self.wy)),
                             n=h, dim=-2, norm="ortho")
        fx = torch.fft.rfft(x, dim=-1, norm="ortho")[..., :self.mx]
        xx = torch.fft.irfft(torch.einsum("biyx,iox->boyx", fx, torch.view_as_complex(self.wx)),
                             n=w, dim=-1, norm="ortho")
        return yy + xx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dense:
            s = self._spectral_dense(x)
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                s = self._spectral(x.float())
        if self.use_ff:
            b = self.norm(self.ff(s.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        else:
            b = F.gelu(self.norm(s + self.point(x)))
        return x + self.drop(b)


class FFNO2d(_GridIn):
    """Factorised FNO: a lift MLP (42 -> 128 -> width), the far side of both axes zero-padded by
    PAD cells (the domain is not periodic), `n_layers` F-FNO layers, crop, LayerNorm, the two
    conv heads."""

    PAD = (4, 8)
    FF_FACTOR = 2

    def __init__(self, width: int = 64, modes_y: int = 16, modes_x: int = 32,
                 n_layers: int = 12, var_head: bool = True, extra_ch: int = 0,
                 dropout: float = 0.0, ff: bool = True):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv2d(CH + extra_ch + 2, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, width, 1))
        grid = (H + self.PAD[0], W + self.PAD[1])
        self.layers = nn.ModuleList([_FFNOLayer(width, modes_y, modes_x, ff, dropout,
                                                self.FF_FACTOR, grid)
                                     for _ in range(n_layers)])
        self.norm_out = _LayerNorm2d(width)
        self.head_field, self.head_logstd = _heads(width, var_head)

    def set_spectral_path(self, path: str = "fft"):
        """'fft' (as trained) or 'dense' (`_FFNOLayer._spectral_dense`). No parameter changes."""
        if path not in ("fft", "dense"):
            raise ValueError(f"spectral path must be 'fft' or 'dense', not {path!r}")
        for layer in self.layers:
            layer.dense = path == "dense"
        return self

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = F.pad(self.lift(self.cat(prev)), (0, self.PAD[1], 0, self.PAD[0]))
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h[..., :H, :W])
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


def build(arch: str, width: int, var_head: bool, extra_ch: int = 0, modes_y: int = 12,
          modes_x: int = 16, n_layers: int = 4, dropout: float = 0.0, **unused):
    """The one factory `train.py` and the shipped `_fno.load_any` share, so both build the same
    module from the same (arch, width, modes, depth) and a checkpoint loads strictly.

    `arch` is "ffno" or "vit". `width` is the F-FNO's channel width or the ViT's token
    dimension, `n_layers` the depth of either; `modes_y` / `modes_x` act on the F-FNO only.
    Every network of the submission has `var_head=True`."""
    if arch == "vit":
        return ViT2d(width, n_layers, var_head, extra_ch, heads=max(width // 64, 1),
                     dropout=dropout)
    if arch == "ffno":
        return FFNO2d(width, modes_y, modes_x, n_layers, var_head, extra_ch, dropout, ff=True)
    raise ValueError(f"no such family in this package: {arch!r}")
