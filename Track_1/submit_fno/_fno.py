"""Self-contained FNO for the fallback path. No dependency on the training tree.

Mirrors `Track_1/unet/models_alt.py:FNO2d` exactly -- same layer order, same parameter names --
so `fno_div_hi.pth` loads into it unchanged. Copied rather than imported so the submission zip
carries no path assumptions and no accidental h5py/scipy imports (neither is present in the
Track 1 evaluation container).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

T_IN = T_OUT = 20
H, W, C = 32, 64, 2
CH = T_IN * C

MEAN = np.array([0.154960856, -0.000513992854], dtype=np.float32)
STD = np.array([0.0968056545, 0.015960684], dtype=np.float32)


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int):
        super().__init__()
        self.my, self.mx = modes_y, modes_x
        scale = 1.0 / (width * width)
        self.w1 = nn.Parameter(scale * torch.randn(width, width, modes_y, modes_x, 2))
        self.w2 = nn.Parameter(scale * torch.randn(width, width, modes_y, modes_x, 2))

    @staticmethod
    def _mul(a, w):
        return torch.einsum("bixy,ioxy->boxy", a, torch.view_as_complex(w))

    def forward(self, x):
        b, _, h, w = x.shape
        ft = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat,
                          device=x.device)
        my, mx = min(self.my, h // 2), min(self.mx, w // 2 + 1)
        out[:, :, :my, :mx] = self._mul(ft[:, :, :my, :mx], self.w1[:, :, :my, :mx])
        out[:, :, -my:, :mx] = self._mul(ft[:, :, -my:, :mx], self.w2[:, :, :my, :mx])
        return torch.fft.irfft2(out, s=(h, w))


class FNO2d(nn.Module):
    def __init__(self, width=24, modes_y=8, modes_x=12, n_layers=4, var_head=True,
                 extra_ch=0):
        super().__init__()
        self.lift = nn.Conv2d(CH + extra_ch + 2, width, 1)
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, modes_y, modes_x) for _ in range(n_layers)])
        self.point = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, CH, 1))
        self.head_logstd = nn.Conv2d(width, C, 1) if var_head else None
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def forward(self, prev, detach_var: bool = False):
        h = self.lift(torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1))
        for s, p, n in zip(self.spec, self.point, self.norm):
            h = h + torch.nn.functional.gelu(n(s(h) + p(h)))
        logstd = None if self.head_logstd is None else self.head_logstd(h)
        return self.head_field(h), logstd


def load(path: str, device: str):
    blob = torch.load(path, map_location=device)
    m = FNO2d(blob.get("width", 24), blob.get("modes_y", 8), blob.get("modes_x", 12),
              blob.get("n_layers", 4), blob.get("var_head", True)).to(device)
    m.load_state_dict(blob["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def predict_fno(model, x_raw: np.ndarray, device: str, alpha: float = 0.75,
                batch: int = 64, out: np.ndarray | None = None) -> np.ndarray:
    """(N,20,32,64,>=2) raw physical -> (N,20,32,64,2) raw physical.

    predict() is TIMED, and at ~0.1 ms/window of actual compute this function is dominated by
    memory traffic and kernel-launch overhead, not FLOPs. Hence:

      * batch 64, not the training-time 16. Track 1 hands us every window at once.
      * the caller may pass `out`, an (N,20,32,64,>=2) array to write into, which removes a
        full-size np.concatenate of the results and lets the caller pre-zero the unscored
        pressure channel once instead of concatenating it on afterwards.
      * the whole contiguous window is uploaded and sliced on the GPU. Slicing [..., :2] on the
        host first makes a non-contiguous view that ascontiguousarray must copy.
    """
    mean = torch.from_numpy(MEAN).to(device)
    std = torch.from_numpy(STD).to(device)
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    n = len(x_raw)
    dst = np.empty((n, T_OUT, H, W, C), dtype=np.float32) if out is None else out
    for i in range(0, n, batch):
        xb = torch.from_numpy(x_raw[i:i + batch]).to(device, non_blocking=True)[..., :2].float()
        prev = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), CH, H, W)

        # Members are averaged HERE, on the GPU, in raw field and sigma space -- before the
        # fluctuation rescaling and before the transfer to host. Two consequences, both
        # deliberate: compose() runs once on a single coherent field rather than being averaged
        # across differently-rescaled ones, and K members cost K forward passes but still only
        # ONE host transfer and one output write. At 0.109 ms of compute against ~1.27 ms of
        # fixed API overhead, that is what makes ensembling affordable at all.
        y = sig = None
        for m in models:
            raw, logstd = m(prev)
            yi = raw.reshape(len(xb), T_OUT, C, H, W).permute(0, 1, 3, 4, 2)
            y = yi if y is None else y + yi
            if logstd is not None:
                si = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1)
                sig = si if sig is None else sig + si
        if len(models) > 1:
            y = y / len(models)
            if sig is not None:
                sig = sig / len(models)

        if sig is not None and alpha > 0:
            m_ = y.mean(1, keepdim=True)
            f_ = y - m_
            s_ = y.std(1, keepdim=True, unbiased=False)
            y = m_ + f_ * ((alpha * sig + (1 - alpha) * s_) / (s_ + 1e-6))
        dst[i:i + batch, ..., :2] = (y * std + mean).float().cpu().numpy()
    return dst
