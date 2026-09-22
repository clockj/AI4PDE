"""Self-contained models for the submission path. No dependency on the training tree.

`FNO2d` mirrors `Track_1/unet/models_alt.py:FNO2d` and `UNetTwoHead` mirrors
`Track_1/unet/model.py:UNetTwoHead` -- same layer order, same parameter names -- so a training
checkpoint of either loads here with `strict=True`. Copied rather than imported so the
submission zip carries no path assumptions and no accidental h5py/scipy imports (neither is
present in the Track 1 evaluation container; `unet/model.py` imports h5py, lazily, and is
therefore not importable material for a zip).

Round 12 ships a MIXED ensemble: 8 FNO members plus one U-Net. Both classes expose the same
`forward(prev, detach_var=False) -> (field, logstd)` contract, so `predict_fno` averages them
without a special case and `fuse` gives the U-Net its own group (its state_dict keys differ
from every FNO's, so it can never be folded into a StackedFNO). Use `load_any`, not `load`,
anywhere a member list may contain one.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn

# --- OPT-IN FAST PATHS (round 17 Q1) ------------------------------------------------------
# Both default to OFF and both are read ONCE, here, so a package that does not set the
# environment variable runs exactly the code it ran before these existed.  They are also
# exposed as keyword arguments on `predict_fno`, which is what the measurement drivers use so
# that one process can time several variants.
#
#   RPDE_PAD_BATCH=1     pad the final, partial batch up to `batch` rows with copies of its
#                        last window and throw the padding away.  Round 15b measured TF32
#                        making a window's output depend on its POSITION in the batch; the
#                        hypothesis this tests is that the dependence is really a dependence on
#                        the batch SHAPE -- only the last, smaller batch picks a different
#                        kernel -- in which case a fixed shape removes it.
#   RPDE_AMP=fp16|bf16   run the network under autocast.  The 1x1 convolutions are ~70 % of the
#                        FLOPs at w96 L8 and are exactly what tensor cores accelerate; the FFT
#                        and the complex einsum stay fp32 (see `_fft_in`).
_PAD_BATCH = os.environ.get("RPDE_PAD_BATCH", "") not in ("", "0")
_AMP = {"fp16": torch.float16, "half": torch.float16,
        "bf16": torch.bfloat16}.get(os.environ.get("RPDE_AMP", "").lower())

# RPDE_SPEC=bmm    run the spectral product as ONE batched REAL matmul per layer instead of
#                  two complex einsums over strided slices (see StackedFNO._spec_bmm).
# RPDE_TAIL=jit    fuse the residual add and the GELU into one scripted kernel.
# Both are OFF by default and both are checked against the shipped path by
# `Track_1/prof_q1g.py --verify`.
_SPEC = os.environ.get("RPDE_SPEC", "").lower()
_TAIL = os.environ.get("RPDE_TAIL", "").lower()
# RPDE_HOSTW=torch  do the strided output write with torch's `copy_` instead of numpy's
#                   assignment. Same bytes to the same place, but numpy's strided assignment is
#                   SINGLE-THREADED and this is bandwidth-bound -- exactly the finding round 11
#                   made about the band stage, applied to the write it did not touch.
_HOSTW = os.environ.get("RPDE_HOSTW", "").lower()


def _res_gelu_eager(h, z):
    """h + gelu(z), as one fused kernel. Two full passes over a (B, K*width, H, W) tensor
    become one; at width 96 x 4 members x batch 112 that tensor is 352 MB, so the pass that
    disappears is ~350 MB of traffic per layer.

    Scripted at import, inside a try: this file has to import in the evaluation container
    under any circumstances, and an opt-in that is OFF by default must never be the reason a
    submission fails to load. `torch.compile` is not an option there at all -- the runtime
    image carries no compiler -- but `torch.jit.script` needs none.
    """
    return h + torch.nn.functional.gelu(z)


try:
    _res_gelu = torch.jit.script(_res_gelu_eager)
except Exception:                                    # pragma: no cover
    _res_gelu = _res_gelu_eager


def _fft_in(x):
    """The spectral path always runs in fp32.

    `w1`/`w2` are real fp32 parameters viewed as complex64, and there is no complex-half
    einsum to match them with, so a half-precision `h` arriving from an autocast 1x1 conv has
    to be promoted before the FFT.  `Tensor.float()` RETURNS SELF when the tensor is already
    float32, so with autocast off this call is the identity and the shipped path is unchanged
    -- `verify/fast_paths.py` checks that bit-exactly rather than trusting the docstring.
    The bmm fast path pins its real spectral matmul to fp32 as well (`_spec_bmm`); before
    round 25 autocast ran that bmm in fp16 and built a ComplexHalf tensor from its output.
    """
    return x.float()


T_IN = T_OUT = 20
# Oscillatory-head frequency band, mirrored from unet/models_alt.py: periods of
# 200 down to 10 frames, bracketing the 60-75 frame shedding period.
OSC_W_LO, OSC_W_HI = 2.0 * math.pi / 200.0, 2.0 * math.pi / 10.0
H, W, C = 32, 64, 2
CH = T_IN * C

MEAN = np.array([0.154960856, -0.000513992854], dtype=np.float32)
STD = np.array([0.0968056545, 0.015960684], dtype=np.float32)

# The same two statistics after each window is divided by its own mean streamwise speed
# (`unet/model.py:MEAN_SN/STD_SN`). A `--norm self` checkpoint was trained against THESE, and
# feeding it MEAN/STD instead is not a small error: see `_self_norm_of` below.
MEAN_SN = np.array([1.0, 0.0], dtype=np.float32)
STD_SN = np.array([0.526024, 0.102182], dtype=np.float32)


def _self_norm_of(blob: dict) -> bool:
    """Did this checkpoint train under `--norm self`?

    Until 2026-08-31 nothing downstream of training asked. `predict_fno` hard-coded MEAN/STD,
    so a self-normalized checkpoint was fed an input it had never seen and its output was
    denormalized with the wrong affine -- and the only arm ever trained that way was scored
    through exactly that path. The damage is proportional to how far the window's own mean
    speed sits from the global mean, which is why it read as a Reynolds-extrapolation failure:
    a model whose self-normalized output is EXACTLY RIGHT scores rel-L2 0.198 / 2.224 / 0.349
    on val_rein / val_relo / val_rehi through the old path, against the arm's measured
    0.147 / 1.144 / 0.299. The arm beat a perfect model on all three sets, which is only
    possible if the instrument, not the model, was broken.

    `train2.py` writes `self_norm` at the top level and `norm` under `config`; older blobs have
    neither and are `--norm real` by construction.
    """
    if "self_norm" in blob:
        return bool(blob["self_norm"])
    return blob.get("norm", blob.get("config", {}).get("norm", "real")) == "self"


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
        ft = torch.fft.rfft2(_fft_in(x))
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat,
                          device=x.device)
        my, mx = min(self.my, h // 2), min(self.mx, w // 2 + 1)
        out[:, :, :my, :mx] = self._mul(ft[:, :, :my, :mx], self.w1[:, :, :my, :mx])
        out[:, :, -my:, :mx] = self._mul(ft[:, :, -my:, :mx], self.w2[:, :, :my, :mx])
        return torch.fft.irfft2(out, s=(h, w))


class FNO2d(nn.Module):
    def __init__(self, width=24, modes_y=8, modes_x=12, n_layers=4, var_head=True,
                 extra_ch=0, var_mlp=False, lift_mlp=False, point_mlp=False,
                 var_hidden=128, var_deep=False, var_in_std=False,
                 var_pred_std=False, fgn_dim=0, mean_head=False, osc_modes=0,
                 osc_global=False, osc_zeromean=False, tbasis=0):
        super().__init__()
        cin = CH + extra_ch + 2
        self.lift = (nn.Sequential(nn.Conv2d(cin, 128, 1), nn.GELU(), nn.Conv2d(128, width, 1))
                     if lift_mlp else nn.Conv2d(cin, width, 1))
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, modes_y, modes_x) for _ in range(n_layers)])
        self.point = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, width, 1))
            if point_mlp else nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, CH, 1))
        # Decoupled time-mean readout (--mean-head): head_field emits only the zero-mean
        # fluctuation and this emits mean_t(y). Mirrors unet/models_alt.py exactly.
        self.head_mean = (nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, C, 1)) if mean_head else None)
        # Oscillatory output head, mirrored from unet/models_alt.py. See the long comment
        # there; the band brackets the 60-75 frame shedding period.
        self.osc_modes = osc_modes
        self.osc_global = osc_global
        self.osc_zeromean = osc_zeromean
        self.head_osc = None
        if osc_modes:
            # Mirrored from unet/models_alt.py.
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
        # J free zero-mean temporal profiles SHARED across windows, with per-pixel
        # coefficients -- the discriminating control for the osc head. See
        # unet/models_alt.py for the argument; here it only has to load and run.
        self.tbasis = tbasis
        self.head_tb = None
        if tbasis:
            self.head_tb = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                         nn.Conv2d(128, tbasis * C, 1))
            self.tb = nn.Parameter(torch.zeros(tbasis, T_OUT))
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
        # FGN (round 12): checkpoints trained with --fgn carry per-block FiLM weights
        # (film.i.weight, Linear(fgn_dim, 2*width, bias=False)). The deterministic predict
        # path below never applies them -- it IS the z = 0 model, and film is zero-init and
        # bias-free in training, so z = 0 and "film unused" are the same forward bit for bit.
        # They are constructed here only so load_state_dict accepts the checkpoint; a sampling
        # deployment (K draws per window) is a separate, verdict-gated change.
        self.fgn_dim = fgn_dim
        self.film = (nn.ModuleList([nn.Linear(fgn_dim, 2 * width, bias=False)
                                    for _ in range(n_layers)]) if fgn_dim > 0 else None)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def forward(self, prev, detach_var: bool = False):
        h = self.lift(torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1))
        hs = []
        for s, p, n in zip(self.spec, self.point, self.norm):
            h = h + torch.nn.functional.gelu(n(s(h) + p(h)))
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
            prof = self.tb - self.tb.mean(1, keepdim=True)
            field = field + torch.einsum("bjchw,jt->btchw", a, prof).reshape(b, CH, Hh, Ww)
        if self.head_mean is not None:
            b = len(prev)
            fl = field.reshape(b, T_OUT, C, H, W)
            fl = fl - fl.mean(1, keepdim=True)
            field = (self.head_mean(h).unsqueeze(1) + fl).reshape(b, CH, H, W)
        logstd = None
        if self.head_logstd is not None:
            z = torch.cat(hs, 1) if self.var_deep else h
            if self.var_in_std:
                z = torch.cat([z, prev[:, :CH].reshape(len(prev), T_IN, C, H, W).std(1)], 1)
            if self.var_pred_std:
                z = torch.cat([z, field.reshape(len(prev), T_OUT, C, H, W).std(1)], 1)
            logstd = self.head_logstd(z)
        return field, logstd


class StackedFNO(nn.Module):
    """K identically-shaped FNO2d members evaluated in ONE invocation.

    Board `time_score` tracks the number of sequential model invocations, not local wall time:
    round 9's 8-member package was FASTER locally than the single U-Net (2.211 ms against
    2.648) and 2.09 ms slower on the board. Members are ~0.42 board ms each, and this removes
    all but one per architecture group.

    Every op the members use is either pointwise in the channel axis or a per-member matmul,
    so a member axis folds into `groups`:

      * 1x1 convs   -> one grouped conv, weights concatenated along the output axis;
      * GroupNorm(8, width) per member -> GroupNorm(8*K, K*width). The groups of a GroupNorm
        are contiguous channel blocks, so 8K groups over K*width channels partitions exactly
        as K separate 8-group norms do;
      * the spectral matmul  "bixy,ioxy->boxy"  ->  "bkixy,kioxy->bkoxy".

    `forward` returns the member MEAN of the field and, in the second slot,
    log(mean(exp(logstd))) -- so the sigma the caller recovers is the mean of the members'
    sigmas, exactly as the sequential loop computed it, and `predict_fno` needs no special
    case. `n_members` lets `predict_fno` weight a group by its size.
    """

    def __init__(self, members):
        super().__init__()
        first = members[0]
        self.n_members = k = len(members)
        # Retained modes are the ONE thing members may disagree on: a spectral weight is
        # zero-padded out to the group maximum below, and a zero weight on an extra mode
        # contributes nothing to the einsum, so the padded member computes exactly what it
        # computed alone. That lets modes (8,12) and (12,20) share an invocation.
        for m in members[1:]:
            assert m.state_dict().keys() == first.state_dict().keys(), "shape mismatch"
            for key, v in m.state_dict().items():
                if key.endswith(".w1") or key.endswith(".w2"):
                    continue
                assert v.shape == first.state_dict()[key].shape, f"{key} shape mismatch"

        self.n_layers = len(first.spec)
        self.my = max(s.my for m in members for s in m.spec)
        self.mx = max(s.mx for m in members for s in m.spec)
        lift_last = first.lift[-1] if isinstance(first.lift, nn.Sequential) else first.lift
        self.width = lift_last.out_channels
        self.register_buffer("grid", first.grid.clone())

        def grouped(convs):
            """K same-shape 1x1 convs -> one grouped 1x1 conv."""
            cin, cout = convs[0].in_channels, convs[0].out_channels
            g = nn.Conv2d(k * cin, k * cout, 1, groups=k)
            g.weight.data.copy_(torch.cat([c.weight.data for c in convs], 0))
            g.bias.data.copy_(torch.cat([c.bias.data for c in convs], 0))
            return g

        def shared(convs):
            """K same-shape 1x1 convs fed the SAME input -> one plain 1x1 conv.

            Only the first layer of `lift` qualifies: every member sees the identical
            (window, grid) tensor there. Conv2d(cin, K*cout, 1) with the weights stacked is
            arithmetically the same as the grouped conv over a K-fold repeat of x -- same
            FLOPs -- but it never materialises the K-times-wider input.
            """
            cin, cout = convs[0].in_channels, convs[0].out_channels
            g = nn.Conv2d(cin, k * cout, 1)
            g.weight.data.copy_(torch.cat([c.weight.data for c in convs], 0))
            g.bias.data.copy_(torch.cat([c.bias.data for c in convs], 0))
            return g

        def gnorm(norms):
            g = nn.GroupNorm(8 * k, k * self.width, eps=norms[0].eps)
            g.weight.data.copy_(torch.cat([n.weight.data for n in norms], 0))
            g.bias.data.copy_(torch.cat([n.bias.data for n in norms], 0))
            return g

        if isinstance(first.lift, nn.Sequential):        # --lift-mlp
            self.lift = nn.Sequential(shared([m.lift[0] for m in members]), nn.GELU(),
                                      grouped([m.lift[2] for m in members]))
        else:
            self.lift = shared([m.lift for m in members])
        self.point = nn.ModuleList([
            nn.Sequential(grouped([m.point[i][0] for m in members]), nn.GELU(),
                          grouped([m.point[i][2] for m in members]))
            if isinstance(first.point[i], nn.Sequential) else
            grouped([m.point[i] for m in members]) for i in range(self.n_layers)])
        self.norm = nn.ModuleList([gnorm([m.norm[i] for m in members])
                                   for i in range(self.n_layers)])
        def padded(t, j):
            """(wid, wid, my_m, mx_m, 2) -> (wid, wid, self.my, self.mx, 2), zero-filled.

            `w1` multiplies ft[:my] and `w2` multiplies ft[-my:], so their row axes run in
            opposite directions in frequency: raising `my` extends w1 toward higher positive
            wavenumbers (pad at the END) and w2 toward more negative ones (pad at the FRONT).
            Padding both at the end silently rotates the negative block and changes the field
            by O(1). Columns are `rfft` bins 0..mx-1 for both, so they always pad at the end.
            """
            out = t.new_zeros(t.shape[0], t.shape[1], self.my, self.mx, 2)
            rows = slice(None, t.shape[2]) if j == 1 else slice(self.my - t.shape[2], None)
            out[:, :, rows, :t.shape[3]] = t
            return out

        for j in (1, 2):
            w = torch.stack([torch.stack([padded(getattr(m.spec[i], f"w{j}").data, j)
                                          for m in members]) for i in range(self.n_layers)])
            self.register_parameter(f"w{j}", nn.Parameter(w))   # (L, K, wid, wid, my, mx, 2)

        self.head_field = nn.Sequential(grouped([m.head_field[0] for m in members]), nn.GELU(),
                                        grouped([m.head_field[2] for m in members]))
        self.var_deep = getattr(first, "var_deep", False)
        self.var_in_std = getattr(first, "var_in_std", False)
        self.var_pred_std = getattr(first, "var_pred_std", False)
        # The oscillatory head, fused. `head_osc` is two 1x1 convs, so it groups exactly like
        # `head_field`. `osc_w` is an MLP on the SPATIALLY POOLED trunk, i.e. a per-member
        # (width -> 64 -> J) matmul; stacking its weights along a member axis and using einsum
        # is the same block-diagonal map a grouped conv would give, without reshaping to 1x1
        # spatial. Everything the head does downstream (cos/sin synthesis, the zero-meaning,
        # the sum over modes) is LINEAR in the member axis, so adding per-member `osc` to
        # `f_all` before the mean is identical to the sequential loop's add-then-average.
        self.osc_modes = getattr(first, "osc_modes", 0)
        self.osc_global = getattr(first, "osc_global", False)
        self.osc_zeromean = getattr(first, "osc_zeromean", False)
        self.head_osc = None
        if getattr(first, "head_osc", None) is not None:
            self.head_osc = nn.Sequential(grouped([m.head_osc[0] for m in members]), nn.GELU(),
                                          grouped([m.head_osc[2] for m in members]))
            if self.osc_global:
                for li, name in ((0, "ow0"), (2, "ow1")):
                    self.register_parameter(name + "_w", nn.Parameter(torch.stack(
                        [m.osc_w[li].weight.data for m in members])))
                    self.register_parameter(name + "_b", nn.Parameter(torch.stack(
                        [m.osc_w[li].bias.data for m in members])))
            self.register_buffer("osc_t", first.osc_t.clone())

        # The free temporal basis fuses the same way: `head_tb` is two 1x1 convs, and the
        # profiles are a per-member (J, T_OUT) matrix, so an einsum over a member axis is the
        # block-diagonal map. The zero-meaning and the contraction are both linear.
        self.tbasis = getattr(first, "tbasis", 0)
        self.head_tb = None
        if getattr(first, "head_tb", None) is not None:
            self.head_tb = nn.Sequential(grouped([m.head_tb[0] for m in members]), nn.GELU(),
                                         grouped([m.head_tb[2] for m in members]))
            self.register_parameter("tb", nn.Parameter(
                torch.stack([m.tb.data for m in members])))       # (K, J, T_OUT)

        h0 = first.head_logstd
        if h0 is None:
            self.head_logstd = None
        elif isinstance(h0, nn.Sequential):
            self.head_logstd = nn.Sequential(grouped([m.head_logstd[0] for m in members]),
                                             nn.GELU(),
                                             grouped([m.head_logstd[2] for m in members]))
        else:
            self.head_logstd = grouped([m.head_logstd for m in members])

    def _packed(self, i, my, mx):
        """Layer `i`'s spectral weights as ONE real (K*M, 2*width, 2*width) block, M = 2*my*mx.

        A complex product (Ar + i.Ai)(Br + i.Bi) is a REAL matmul against the 2x2 block

            [[ Br,  Bi],
             [-Bi,  Br]]

        fed [Ar | Ai], because that returns [Ar.Br - Ai.Bi | Ar.Bi + Ai.Br] -- exactly
        [Re, Im]. Same arithmetic (a complex multiply IS four real multiplies), but it becomes
        a single contiguous real `bmm` that cuBLAS can batch, instead of a complex einsum whose
        operands are strided slices it has to copy first.

        The +ky and -ky blocks are concatenated along the MODE axis, so the two einsums of the
        shipped path become one call. Built once, on first use, and cached as a non-persistent
        buffer -- it is a repacking of `w1`/`w2`, so it must never enter the state_dict.
        """
        key = f"_pk{i}"
        p = getattr(self, key, None)
        if p is not None:
            return p
        w1 = torch.view_as_complex(self.w1[i])[:, :, :, :my, :mx]      # (K, in, out, my, mx)
        w2 = torch.view_as_complex(self.w2[i])[:, :, :, :my, :mx]
        wc = torch.cat([w1, w2], dim=3)                               # (K, in, out, 2my, mx)
        wc = wc.permute(0, 3, 4, 1, 2).reshape(-1, wc.shape[1], wc.shape[2])   # (K*M, in, out)
        br, bi = wc.real.contiguous(), wc.imag.contiguous()
        p = torch.cat([torch.cat([br, bi], 2), torch.cat([-bi, br], 2)], 1).contiguous()
        self.register_buffer(key, p, persistent=False)
        return p

    def _spec_bmm(self, i, x, my, mx):
        b, _, h, w = x.shape
        k, wid = self.n_members, self.width
        wf = w // 2 + 1
        ft = torch.fft.rfft2(_fft_in(x)).reshape(b, k, wid, h, wf)
        # (b, k, wid, 2my, mx) -> (k, 2my, mx, b, wid) -> (K*M, b, wid); the mode axis is
        # ordered [+ky block, -ky block] to match `_packed`.
        a = torch.cat([ft[:, :, :, :my, :mx], ft[:, :, :, -my:, :mx]], 3)
        a = a.permute(1, 3, 4, 0, 2).reshape(-1, b, wid)
        # The real matmul is pinned to fp32 even under autocast. Autocast puts `bmm` on its fp16
        # list, and a half `cr` makes `torch.complex` build a ComplexHalf tensor -- "experimental"
        # in torch 2.2 (r25a raised the warning here), and an unknown board GPU is exactly where
        # an experimental kernel fails. With autocast off this is the identity: `.float()`
        # returns self for fp32, and a disabled autocast context changes nothing.
        with torch.autocast(x.device.type, enabled=False):
            cr = torch.bmm(torch.cat([a.real, a.imag], 2).float().contiguous(),
                           self._packed(i, my, mx).float())
        oc = torch.complex(cr[..., :wid], cr[..., wid:])
        oc = oc.reshape(k, 2 * my, mx, b, wid).permute(3, 0, 4, 1, 2)
        out = torch.zeros(b, k, wid, h, wf, dtype=torch.cfloat, device=x.device)
        out[:, :, :, :my, :mx] = oc[..., :my, :]
        out[:, :, :, -my:, :mx] = oc[..., my:, :]
        return torch.fft.irfft2(out.reshape(b, k * wid, h, wf), s=(h, w))

    def _spec(self, i, x):
        b, _, h, w = x.shape
        k, wid = self.n_members, self.width
        my, mx = min(self.my, h // 2), min(self.mx, w // 2 + 1)
        # The bmm path indexes `w1`/`w2` with [:my, :mx] up front, so it is only equivalent
        # when the clamp is a no-op in the row axis -- `w2`'s rows run backwards in frequency
        # and a clamped `my` would have to be taken from the OTHER end. At 32x64 with
        # modes (8, 12) nothing clamps; anything else falls back rather than guess.
        if _SPEC == "bmm" and my == self.my and mx == self.mx:
            return self._spec_bmm(i, x, my, mx)
        ft = torch.fft.rfft2(_fft_in(x)).reshape(b, k, wid, h, w // 2 + 1)
        out = torch.zeros(b, k, wid, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        for sl, wp in ((slice(None, my), self.w1), (slice(-my, None), self.w2)):
            wc = torch.view_as_complex(wp[i])[:, :, :, :my, :mx]
            out[:, :, :, sl, :mx] = torch.einsum("bkixy,kioxy->bkoxy",
                                                 ft[:, :, :, sl, :mx], wc)
        return torch.fft.irfft2(out.reshape(b, k * wid, h, w // 2 + 1), s=(h, w))

    def _osc(self, h, b, k, hw):
        """Per-member oscillatory term, (b, k, CH, H, W). Mirrors FNO2d.forward exactly."""
        j = 2 if self.osc_global else 3
        o = self.head_osc(h).reshape(b, k, self.osc_modes, j, C, *hw)
        if self.osc_global:
            hm = h.mean((-2, -1)).reshape(b, k, self.width)
            a = torch.nn.functional.gelu(
                torch.einsum("bki,koi->bko", hm, self.ow0_w) + self.ow0_b)
            w = torch.einsum("bki,koi->bko", a, self.ow1_w) + self.ow1_b       # (b, k, J)
            om = (OSC_W_LO + (OSC_W_HI - OSC_W_LO)
                  * torch.sigmoid(w))[:, :, :, None, None, None]
        else:
            om = OSC_W_LO + (OSC_W_HI - OSC_W_LO) * torch.sigmoid(o[:, :, :, 2])
        ph = om.unsqueeze(3) * self.osc_t.view(1, 1, 1, T_OUT, 1, 1, 1)
        osc = (o[:, :, :, 0].unsqueeze(3) * torch.cos(ph)
               - o[:, :, :, 1].unsqueeze(3) * torch.sin(ph)).sum(2)
        if self.osc_zeromean:
            # osc is (b, k, T_OUT, C, H, W) after the sum over modes, so the TIME axis is 2 --
            # one further right than FNO2d's, which has no member axis and zero-means on 1.
            osc = osc - osc.mean(2, keepdim=True)
        return osc.reshape(b, k, CH, *hw)

    def forward(self, prev, detach_var: bool = False):
        b, k = len(prev), self.n_members
        x = torch.cat([prev, self.grid.expand(b, -1, -1, -1)], 1)
        h = self.lift(x)                    # first lift conv is shared, see `shared()`
        hs = []
        for i in range(self.n_layers):
            z = self.norm[i](self._spec(i, h) + self.point[i](h))
            h = _res_gelu(h, z) if _TAIL == "jit" else h + torch.nn.functional.gelu(z)
            hs.append(h)
        hw = h.shape[-2:]
        f_all = self.head_field(h).reshape(b, k, CH, *hw)      # per member, before averaging
        if self.head_osc is not None:
            f_all = f_all + self._osc(h, b, k, hw)
        if self.head_tb is not None:
            a = self.head_tb(h).reshape(b, k, self.tbasis, C, *hw)
            prof = self.tb - self.tb.mean(2, keepdim=True)         # (K, J, T_OUT)
            f_all = f_all + torch.einsum("bkjchw,kjt->bktchw",
                                         a, prof).reshape(b, k, CH, *hw)
        field = f_all.mean(1)
        logstd = None
        if self.head_logstd is not None:
            # The grouped head expects MEMBER-major channels, so anything appended to the
            # readout has to sit inside each member's own block, not after all of them.
            if self.var_deep or self.var_in_std or self.var_pred_std:
                parts = ([t.reshape(b, k, self.width, *hw) for t in hs] if self.var_deep
                         else [h.reshape(b, k, self.width, *hw)])
                if self.var_in_std:
                    s_in = prev[:, :CH].reshape(b, T_IN, C, *hw).std(1)
                    parts.append(s_in.unsqueeze(1).expand(b, k, C, *hw))
                if self.var_pred_std:
                    parts.append(f_all.reshape(b, k, T_OUT, C, *hw).std(2))
                z = torch.cat(parts, 2).reshape(b, -1, *hw)
            else:
                z = h
            ls = self.head_logstd(z).reshape(b, k, C, *hw)
            # Mean of the members' SIGMAS, returned as a log so the caller's exp recovers it.
            # The clamp matches predict_fno's, whose second application is then a no-op.
            logstd = ls.clamp(-8.0, 2.0).exp().mean(1).log()
        return field, logstd


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNetTwoHead(nn.Module):
    """Copy of `Track_1/unet/model.py:UNetTwoHead` -- 1.91M parameters at width 64.

    Kept byte-compatible with the training class (same submodule names inc/down1/down2/up1/up2/
    head_field/head_logstd, same `block` composition) so `unet_all_g1.pth`'s state_dict loads
    with strict=True and a silent architecture drift cannot masquerade as a working member. The
    only thing left out is the training-time constant init of `head_logstd`, which exists to
    keep the TKE loss off the clamp at step 0 and is overwritten by load_state_dict here.

    Why a second family is in the package at all: on the gate-M battery, adding this one member
    to the 7-FNO ensemble was worth +0.091 proxy / +0.123 full wfinal, against +0.019 for an
    extra FNO seed -- round 8's "architecture beats seeds" replicating across families rather
    than within one. It costs one invocation, since nothing here shares a shape with an FNO.
    """

    def __init__(self, width: int = 64, var_head: bool = True, extra_ch: int = 0):
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

    def forward(self, prev, detach_var: bool = False):
        h1 = self.inc(prev)
        h2 = self.down1(self.pool(h1))
        h3 = self.down2(self.pool(h2))
        u2 = self.up1(torch.cat([nn.functional.interpolate(h3, scale_factor=2), h2], 1))
        u1 = self.up2(torch.cat([nn.functional.interpolate(u2, scale_factor=2), h1], 1))
        logstd = None if self.head_logstd is None else \
            self.head_logstd(u1.detach() if detach_var else u1)
        return self.head_field(u1), logstd


class UFNO2d(nn.Module):
    """Copy of `Track_1/unet/models_alt.py:UFNO2d` -- U-FNO (Wen et al., 2022), 15.6M at w96.

    Vendored because `models_alt.py` is not in the archive and `load_any`'s repo-import branch
    raises inside the evaluation container by design. Keep the two definitions in step: the
    parameter NAMES and shapes are the contract, and `verify/diff_ufno.py` checks a checkpoint
    loads into both and produces the same field to float32 noise.

    Each block is spectral(h) + point(h) + unet(h) instead of the FNO's spectral + point. Mode
    truncation is a low-pass and cannot represent the sharp zero set of the PIV field of view
    and the airfoil body (3.3 % of elements, exactly zero); the 1x1 pointwise branch has no
    spatial extent and cannot either. A two-level conv branch can, at a fraction of the
    spectral block's parameters. Round 15 measured it as the largest single-model gain of the
    project, with both of its controls negative -- a dilated ResNet with the same global
    receptive field and no spectral machinery LOSES, and a plain U-Net at the same width loses
    harder, so this is the FNO plus a local branch and not a conv net in disguise.

    `dropout` is accepted and ignored: it is a training-time regularizer, the module is only
    ever run in eval() here, and Dropout2d holds no parameters, so a --dropout checkpoint loads
    into this class unchanged.
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
        self.drop = nn.Identity()
        self.head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, CH, 1))
        self.head_logstd = None
        if var_head:
            self.head_logstd = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                             nn.Conv2d(128, C, 1))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = self.lift(torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1))
        for s, p, n, dn, up in zip(self.spec, self.point, self.norm, self.down, self.up):
            d = nn.functional.interpolate(dn(h), size=h.shape[-2:], mode="nearest")
            h = h + self.drop(torch.nn.functional.gelu(n(s(h) + p(h) + up(torch.cat([d, h], 1)))))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


class AFNO2d(nn.Module):
    """Copy of `Track_1/unet/models_alt.py:AFNO2d` -- Adaptive FNO (Guibas et al., ICLR 2022).

    Vendored for the same reason as UFNO2d: `models_alt.py` is not in the archive and
    `load_any`'s repo-import branch raises inside the evaluation container by design. The
    parameter NAMES and shapes are the contract; `verify/diff_arch.py` loads one checkpoint
    into both definitions and requires the fields to agree to float32 noise.

    FNO's spectral block is a LINEAR map per retained mode. AFNO replaces it with a two-layer
    complex MLP whose weights are SHARED across modes and block-diagonal across channels, then
    soft-thresholds the magnitude, so the operator's spectral support is learned per input.
    At w96/blocks 4 that is ~0.3M parameters against the w96 FNO's ~14M.

    `dropout` is accepted and ignored -- eval() only, and Dropout2d holds no parameters, so a
    --dropout checkpoint loads unchanged.
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
        self.w1 = nn.Parameter(s * torch.randn(n_layers, blocks, d, hid, 2))
        self.b1 = nn.Parameter(torch.zeros(n_layers, blocks, hid, 2))
        self.w2 = nn.Parameter(s * torch.randn(n_layers, blocks, hid, d, 2))
        self.b2 = nn.Parameter(torch.zeros(n_layers, blocks, d, 2))
        self.point = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.drop = nn.Identity()
        self.head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, CH, 1))
        self.head_logstd = None
        if var_head:
            self.head_logstd = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                             nn.Conv2d(128, C, 1))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def _spectral(self, x: torch.Tensor, i: int) -> torch.Tensor:
        b, cw, h, w = x.shape
        ft = torch.fft.rfft2(x)
        out = torch.zeros_like(ft)
        my, mx = min(self.my, h // 2), min(self.mx, w // 2 + 1)
        w1, b1 = torch.view_as_complex(self.w1[i]), torch.view_as_complex(self.b1[i])
        w2, b2 = torch.view_as_complex(self.w2[i]), torch.view_as_complex(self.b2[i])
        for sl in (slice(0, my), slice(h - my, h)):
            z = ft[:, :, sl, :mx].reshape(b, self.blocks, -1, my, mx)
            z = torch.einsum("bkdyx,kdh->bkhyx", z, w1) + b1[None, :, :, None, None]
            z = torch.complex(torch.nn.functional.gelu(z.real),
                              torch.nn.functional.gelu(z.imag))
            z = torch.einsum("bkhyx,khd->bkdyx", z, w2) + b2[None, :, :, None, None]
            mag = z.abs()
            z = z * (torch.clamp(mag - self.lam, min=0.0) / (mag + 1e-9))
            out[:, :, sl, :mx] = z.reshape(b, cw, my, mx)
        return torch.fft.irfft2(out, s=(h, w))

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = self.lift(torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1))
        for i, (p, n) in enumerate(zip(self.point, self.norm)):
            h = h + self.drop(torch.nn.functional.gelu(n(self._spectral(h, i) + p(h))))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


class ViT2d(nn.Module):
    """Copy of `Track_1/unet/models_alt.py:ViT2d` -- a space-time ViT over 4x4 patches.

    128 tokens of the 32x64 grid; time is already folded into channels, so one token is a
    4x4x20-frame block and attention mixes every block with every other in ONE layer. It is the
    only family here whose spatial mixing is INPUT-DEPENDENT, which is why it is in the probe:
    U-FNO's board miss is hypothesized to come from its local translation-equivariant conv
    branch, and this family has no such structure to be wrong in the same way.

    Drop-path and dropout are training-time only and both are identity in eval(), so they are
    omitted here rather than reproduced; nothing they touch holds a parameter. The module names
    (`embed`, `pos`, `blocks.<i>.{n1,attn,n2,mlp}`, `norm_out`, `proj`, the two heads) are the
    contract with the training definition and `verify/diff_arch.py` enforces it.
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
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True),
                "n2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                     nn.Dropout(0.0),
                                     nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(0.0)),
            }))
        self.norm_out = nn.LayerNorm(dim)
        self.wid = max(dim // (patch * patch), 8)
        self.proj = nn.Linear(dim, self.wid * patch * patch)
        self.head_field = nn.Sequential(nn.Conv2d(self.wid, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, CH, 1))
        self.head_logstd = None
        if var_head:
            self.head_logstd = nn.Sequential(nn.Conv2d(self.wid, 128, 1), nn.GELU(),
                                             nn.Conv2d(128, C, 1))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        b = len(prev)
        x = torch.cat([prev, self.grid.expand(b, -1, -1, -1)], 1)
        t = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            q = blk["n1"](t)
            t = t + blk["attn"](q, q, q, need_weights=False)[0]
            t = t + blk["mlp"](blk["n2"](t))
        t = self.proj(self.norm_out(t))
        pp = self.patch
        h = (t.reshape(b, self.gh, self.gw, self.wid, pp, pp)
              .permute(0, 3, 1, 4, 2, 5).reshape(b, self.wid, H, W))
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


class StackedViT(nn.Module):
    """K identically-shaped ViT2d members evaluated in ONE invocation.

    Round 23 measured what this is worth.  Shipping round 20's own four FNO members UNFUSED --
    same checkpoints, same packing, same band bytes -- moved board latency 3.795 -> 7.249 ms on
    Track 1 and 4.659 -> 7.255 on Track 2 while every error subscore came back identical to six
    decimals.  That is **1.151 / 0.866 ms per invocation**, against a fitted 0.435: four
    sequential forwards do the same arithmetic as one grouped forward, so the launches alone cost
    +3.45 ms, and a member that needs its own invocation costs -0.085 `final` before it has earned
    anything.  (Round 23 also read this as "launch-bound, not FLOP-bound".  ROUND 24 FALSIFIED
    THAT HALF with this very class: one more ViT in the group cost +1.02 board ms on Track 1 for
    +0.316 A100 ms/window of local compute.  Fusion removes the launch tax, not the member's
    FLOPs.)

    The ViT is the member that makes this worth building.  On `r21b`'s marginal table over
    cmb3sim x3 it is the best cross-family addition by a wide margin (+0.39 full-split sps
    against DilRes +0.30, U-FNO +0.30, AFNO +0.10), the board paid the first one +0.148 (T1) /
    +0.152 (T2) of accuracy, and at 14.7M parameters it is 17 MB int8 against a width-96 L8
    FNO's 27.7 -- so it is simultaneously the most valuable and the smallest member available.
    Without fusion a second one is a third invocation and loses; with it, it is free.

    HOW THE MEMBER AXIS FOLDS.  Nothing here is an approximation; each op is the same
    arithmetic with a leading `k`:

      * `embed` is a patch conv on the SAME input for every member, so K of them become ONE
        Conv2d(cin, K*dim, patch, stride=patch) -- the `shared()` trick from StackedFNO, not a
        grouped conv over a repeated input;
      * LayerNorm has no cross-token mixing, so it is `F.layer_norm(t, (dim,))` once, followed
        by a per-member affine of shape (K,1,1,dim);
      * every Linear becomes `einsum("kbnd,kod->kbno")` against a stacked (K, out, in) weight;
      * attention folds K into the batch axis of `scaled_dot_product_attention`, which is
        exactly what `nn.MultiheadAttention` does internally for its own batch;
      * the two readout convs become grouped 1x1 convs over MEMBER-MAJOR channels.

    `forward` returns the member MEAN of the field and log(mean(exp(logstd))) in the second
    slot, the same contract `StackedFNO` obeys, so `predict_fno` needs no special case.

    Dropout and drop-path are training-time only and are identity in `eval()`, which is why
    `ViT2d` above does not reproduce them and neither does this.
    """

    def __init__(self, members):
        super().__init__()
        first = members[0]
        self.n_members = k = len(members)
        keys = first.state_dict().keys()
        for m in members[1:]:
            assert m.state_dict().keys() == keys, "shape mismatch"
            for key, v in m.state_dict().items():
                assert v.shape == first.state_dict()[key].shape, f"{key} shape mismatch"

        self.patch, self.gh, self.gw = first.patch, first.gh, first.gw
        self.wid = first.wid
        self.dim = dim = first.pos.shape[-1]
        self.heads = first.blocks[0]["attn"].num_heads
        self.depth = d = len(first.blocks)
        self.register_buffer("grid", first.grid.clone())

        # `embed` is a patch conv on the SAME input for every member, so K of them collapse to
        # ONE conv with K*dim output channels -- not a grouped conv over a repeated input,
        # which is the mistake round 11 fixed in StackedFNO's lift.
        e0 = first.embed
        emb = nn.Conv2d(e0.in_channels, k * dim, e0.kernel_size, stride=e0.stride)
        emb.weight.data.copy_(torch.cat([m.embed.weight.data for m in members], 0))
        emb.bias.data.copy_(torch.cat([m.embed.bias.data for m in members], 0))
        self.embed = emb
        self.register_buffer("pos", torch.stack([m.pos.data[0] for m in members])[:, None])

        # Everything per-block is stacked over BOTH depth and member, giving (L, K, ...) --
        # the same shape StackedFNO gives its spectral weights, so the forward loop indexes
        # one tensor per quantity instead of walking a ParameterList. These are constants at
        # inference; buffers, not parameters, and non-persistent because a fused module is
        # never saved (`load_any` always rebuilds it from the member checkpoints).
        def bt(name, per_block, extra_dims: int = 0):
            t = torch.stack([torch.stack([per_block(m.blocks[i]) for m in members])
                             for i in range(d)])
            for _ in range(extra_dims):
                t = t.unsqueeze(2)
            self.register_buffer(name, t.contiguous(), persistent=False)

        # LayerNorm and bias terms broadcast against (k, b, N, ...), so they carry two dummy
        # axes; weight matrices are contracted by einsum and carry none.
        bt("n1_w", lambda blk: blk["n1"].weight.data, 2)
        bt("n1_b", lambda blk: blk["n1"].bias.data, 2)
        bt("n2_w", lambda blk: blk["n2"].weight.data, 2)
        bt("n2_b", lambda blk: blk["n2"].bias.data, 2)
        bt("qkv_w", lambda blk: blk["attn"].in_proj_weight.data)
        bt("qkv_b", lambda blk: blk["attn"].in_proj_bias.data, 2)
        bt("op_w", lambda blk: blk["attn"].out_proj.weight.data)
        bt("op_b", lambda blk: blk["attn"].out_proj.bias.data, 2)
        bt("f1_w", lambda blk: blk["mlp"][0].weight.data)
        bt("f1_b", lambda blk: blk["mlp"][0].bias.data, 2)
        bt("f2_w", lambda blk: blk["mlp"][3].weight.data)
        bt("f2_b", lambda blk: blk["mlp"][3].bias.data, 2)

        def st(name, get, extra_dims: int = 0):
            t = torch.stack([get(m) for m in members])
            for _ in range(extra_dims):
                t = t.unsqueeze(1)
            self.register_buffer(name, t.contiguous(), persistent=False)

        st("no_w", lambda m: m.norm_out.weight.data, 2)
        st("no_b", lambda m: m.norm_out.bias.data, 2)
        st("pj_w", lambda m: m.proj.weight.data)
        st("pj_b", lambda m: m.proj.bias.data, 2)

        def grouped(convs):
            """K same-shape 1x1 convs -> one grouped 1x1 conv over member-major channels."""
            cin, cout = convs[0].in_channels, convs[0].out_channels
            g = nn.Conv2d(k * cin, k * cout, 1, groups=k)
            g.weight.data.copy_(torch.cat([c.weight.data for c in convs], 0))
            g.bias.data.copy_(torch.cat([c.bias.data for c in convs], 0))
            return g

        self.head_field = nn.Sequential(grouped([m.head_field[0] for m in members]), nn.GELU(),
                                        grouped([m.head_field[2] for m in members]))
        self.head_logstd = None
        if first.head_logstd is not None:
            self.head_logstd = nn.Sequential(
                grouped([m.head_logstd[0] for m in members]), nn.GELU(),
                grouped([m.head_logstd[2] for m in members]))

    @staticmethod
    def _lin(t, w, b):
        """(K,B,N,in) against a stacked (K,out,in) weight, plus a (K,1,1,out) bias."""
        return torch.einsum("kbnd,kod->kbno", t, w) + b

    def forward(self, prev, detach_var: bool = False):
        b, k, dim = len(prev), self.n_members, self.dim
        x = torch.cat([prev, self.grid.expand(b, -1, -1, -1)], 1)
        t = self.embed(x).reshape(b, k, dim, self.gh * self.gw).permute(1, 0, 3, 2)
        t = (t + self.pos).contiguous()                             # (k, b, N, dim)
        n, hd = t.shape[2], dim // self.heads
        lnorm = torch.nn.functional.layer_norm
        for i in range(self.depth):
            q = lnorm(t, (dim,)) * self.n1_w[i] + self.n1_b[i]
            qkv = self._lin(q, self.qkv_w[i], self.qkv_b[i])        # (k, b, N, 3*dim)
            # k and b are both pure batch axes to attention -- nothing mixes across either --
            # so folding them into one is exact, and it is the layout SDPA wants.
            qkv = qkv.reshape(k * b, n, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
            a = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            a = a.transpose(1, 2).reshape(k, b, n, dim)
            t = t + self._lin(a, self.op_w[i], self.op_b[i])
            h = lnorm(t, (dim,)) * self.n2_w[i] + self.n2_b[i]
            h = torch.nn.functional.gelu(self._lin(h, self.f1_w[i], self.f1_b[i]))
            t = t + self._lin(h, self.f2_w[i], self.f2_b[i])
        t = lnorm(t, (dim,)) * self.no_w + self.no_b
        t = self._lin(t, self.pj_w, self.pj_b)                      # (k, b, N, wid*p*p)
        pp, wid = self.patch, self.wid
        # (k,b,gh,gw,wid,p,p) -> (b, k, wid, gh, p, gw, p). Member-major channels, which is
        # what the grouped readout convs expect; ViT2d's own permute is this one without k.
        h = (t.reshape(k, b, self.gh, self.gw, wid, pp, pp)
              .permute(1, 0, 4, 2, 5, 3, 6).reshape(b, k * wid, H, W))
        field = self.head_field(h).reshape(b, k, CH, H, W).mean(1)
        logstd = None
        if self.head_logstd is not None:
            ls = self.head_logstd(h.detach() if detach_var else h).reshape(b, k, C, H, W)
            # Mean of the members' SIGMAS returned as a log, matching StackedFNO exactly, so
            # predict_fno needs no special case and the clamp it applies is a no-op.
            logstd = ls.clamp(-8.0, 2.0).exp().mean(1).log()
        return field, logstd


# ---------------------------------------------------------------------------------------------
# Swin U-Net members (backbone campaign, 2026-09-14): a vendored SwinUNet and its fused stack
# ---------------------------------------------------------------------------------------------
_F = torch.nn.functional

# `arch` -> SwinUNet keyword overrides, mirroring `models_alt.build_wave1`. The Phase-2 ladder
# names each structural variant in the checkpoint's own `arch` field, because `load_any`
# rebuilds a member from (arch, width, n_layers, var_head) alone.
SWIN_VARIANTS = {"swin_unet": {}, "swin_unet_b6": {"bottleneck_depth": 6},
                 "swin_unet_win16": {"window": 16}, "swin_unet_p1": {"patch": 1}}

# Which SwinUNet class `load_any` builds for a `swin_unet*` checkpoint.
#   "auto"        the training definition -- exactly the `models_alt.build` call `load_any` made
#                 before this existed -- when this file sits in the repo next to
#                 `../unet/models_alt.py` (every local instrument, so screens are untouched),
#                 and the vendored copy below everywhere else, i.e. inside every archive.
#                 Decided by the FILE's location, not by what is importable: a process that has
#                 already imported models_alt still gets the vendored class from a package dir.
#   "vendored"    the copy below, always.
#   "models_alt"  the training definition, always (raises outside the repo).
# `verify/fuse_swin.py` checks the two classes bitwise against each other on real checkpoints.
SWIN_IMPL = "auto"
_UNET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unet")


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
    """Copy of `models_alt._WindowAttnV2`: SwinV2 scaled cosine attention (temperature clamped at
    100), a continuous position bias (2 -> 512 -> heads MLP over log-spaced offsets), and the
    shifted-window mask, through SDPA with scale 1. Dropout is training-only and omitted."""

    def __init__(self, dim: int, window: tuple[int, int], heads: int):
        super().__init__()
        wh, ww = window
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.log(10.0 * torch.ones(heads, 1, 1)))
        self.cpb = nn.Sequential(nn.Linear(2, 512), nn.ReLU(), nn.Linear(512, heads))
        dy = torch.arange(-(wh - 1), wh, dtype=torch.float32) / max(wh - 1, 1)
        dx = torch.arange(-(ww - 1), ww, dtype=torch.float32) / max(ww - 1, 1)
        tab = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), -1) * 8.0
        tab = torch.sign(tab) * torch.log2(tab.abs() + 1.0) / math.log2(8.0)
        self.register_buffer("coords", tab.reshape(-1, 2), persistent=False)
        cy, cx = torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij")
        c = torch.stack([cy.flatten(), cx.flatten()])
        rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0)
        idx = (rel[..., 0] + wh - 1) * (2 * ww - 1) + (rel[..., 1] + ww - 1)
        self.register_buffer("rel_index", idx.reshape(-1), persistent=False)
        self._frozen = None

    @torch.no_grad()
    def freeze_cache(self) -> None:
        """`models_alt`'s inference cache: logit scale and position bias computed once, in fp32."""
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
        q = _F.normalize(q, dim=-1) * scale
        k = _F.normalize(k, dim=-1)
        dt = v.dtype
        key = (bw, dt, mask is not None)
        if fz is not None and key in fz["am"]:
            am = fz["am"][key]
        else:
            bias = ((self.cpb(self.coords)[self.rel_index]
                     .reshape(n, n, self.heads).permute(2, 0, 1)) if fz is None
                    else fz["bias"])
            if mask is None:
                am = bias.unsqueeze(0)
            else:
                nw = mask.shape[0]
                am = ((bias[None, None] + mask[None]).expand(bw // nw, nw, self.heads, n, n)
                      .reshape(bw, self.heads, n, n))
            am = am.to(dt)
            if fz is not None:
                fz["am"][key] = am
        o = _F.scaled_dot_product_attention(q.to(dt), k.to(dt), v, attn_mask=am, dropout_p=0.0,
                                            scale=1.0)
        return self.proj(o.transpose(1, 2).reshape(bw, n, c))


class _SwinV2Block(nn.Module):
    """Copy of `models_alt._SwinV2Block` (res-post-norm; shifted windows with Swin's -100 mask).
    Drop-path and dropout are identity in eval and omitted; the MLP keeps its indices."""

    def __init__(self, dim: int, res: tuple[int, int], heads: int, window: int, shift: bool):
        super().__init__()
        h, w = res
        wh, ww = min(window, h), min(window, w)
        sy = wh // 2 if (shift and wh < h) else 0
        sx = ww // 2 if (shift and ww < w) else 0
        self.win, self.shift = (wh, ww), (sy, sx)
        self.attn = _WindowAttnV2(dim, (wh, ww), heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Identity(),
                                 nn.Linear(4 * dim, dim), nn.Identity())
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
            mw = _win_part(img, wh, ww).squeeze(-1)
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
        x = x + self.norm1(y)
        return x + self.norm2(self.mlp(x))


class _PatchMergeV2(nn.Module):
    """2x2 token merge (B, H, W, C) -> (B, H/2, W/2, 2C): concat, Linear 4C -> 2C, LN."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]],
                      -1)
        return self.norm(self.reduction(x))


class _PatchExpand(nn.Module):
    """(B, H, W, C) -> (B, 2H, 2W, out): Linear C -> 4*out, depth-to-space, LN."""

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
    """(B, H, W, C) -> (B, H, W, out): Linear + LN, the patch-1 stand-in for _PatchExpand."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class _ConvNeXtBlock(nn.Module):
    """Copy of `models_alt._ConvNeXtBlock` (dw 7x7 -> LN -> 4x MLP -> layer scale -> residual)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(_F.gelu(self.pw1(self.norm(self.dw(x).permute(0, 2, 3, 1)))))
        h = h * self.gamma.to(h.dtype)
        return x + h.permute(0, 3, 1, 2)


class SwinUNet(nn.Module):
    """Copy of `Track_1/unet/models_alt.py:SwinUNet` -- the SwinV2 U-Net of the backbone campaign.

    2x2 (or 1x1) patch embedding + LN; encoder levels of `n_layers` blocks alternating regular
    and shifted windows, each followed by a V2 patch merge; a bottleneck of one global window;
    every skip through two ConvNeXt blocks; decoder patch expand + concat + Linear + blocks; a
    final expand to 32x64, one residual 3x3 conv, the shared MLP heads. Module names and
    registration order are the training definition's, so its checkpoints load with strict=True.
    Only the inference path is reproduced (no dropout, no drop-path, no initialization).
    """

    WINDOW = 8
    STAGES = 3
    SKIP_BLOCKS = 2

    def __init__(self, width: int = 96, n_layers: int = 2, var_head: bool = True,
                 window: int | None = None, bottleneck_depth: int | None = None,
                 patch: int = 2):
        super().__init__()
        assert patch in (1, 2), f"patch must be 1 or 2, not {patch}"
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])
        win = self.WINDOW if window is None else int(window)
        bd = n_layers if bottleneck_depth is None else int(bottleneck_depth)
        res = [((H // patch) >> i, (W // patch) >> i) for i in range(self.STAGES)]
        dims = [width * 2 ** i for i in range(self.STAGES)]
        heads = [max(d // 32, 1) for d in dims]

        def stage(i, depth=n_layers, window=win):
            return nn.ModuleList([_SwinV2Block(dims[i], res[i], heads[i], window, bool(j % 2))
                                  for j in range(depth)])

        self.embed = nn.Conv2d(CH + 2, width, patch, stride=patch)
        self.embed_norm = nn.LayerNorm(width)
        self.enc = nn.ModuleList([stage(i) for i in range(self.STAGES - 1)])
        self.merge = nn.ModuleList([_PatchMergeV2(dims[i]) for i in range(self.STAGES - 1)])
        self.skip_blocks = nn.ModuleList([
            nn.Sequential(*[_ConvNeXtBlock(dims[i]) for _ in range(self.SKIP_BLOCKS)])
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
        self.head_field = nn.Sequential(nn.Conv2d(self.out_ch, 128, 1), nn.GELU(),
                                        nn.Conv2d(128, CH, 1))
        self.head_logstd = None
        if var_head:
            self.head_logstd = nn.Sequential(nn.Conv2d(self.out_ch, 128, 1), nn.GELU(),
                                             nn.Conv2d(128, C, 1))
        self.fp32_max_batch = 0

    @torch.no_grad()
    def prepare_inference(self, fp32_max_batch: int = 0):
        """`models_alt.SwinUNet.prepare_inference`: freeze every attention's scale, bias and
        masks; batches of at most `fp32_max_batch` windows run fp32 even under autocast."""
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
        x = torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1)
        x = self.embed_norm(self.embed(x).permute(0, 2, 3, 1))
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


def _is_swin(m) -> bool:
    """A SwinUNet member of either definition (vendored above, or models_alt's)."""
    return type(m).__name__ == "SwinUNet" and all(
        hasattr(m, a) for a in ("enc", "merge", "skip_blocks", "bottleneck", "expand", "skip",
                                "dec", "final", "refine", "embed", "grid"))


def _swin_signature(m):
    """Structure a state_dict does not show, or None for every other family.

    `swin_unet_win16` has swin_unet's exact parameter shapes (window size lives only in the
    non-persistent position tables), so shapes alone would fold the two into one stack."""
    if not _is_swin(m):
        return None
    blocks = [blk for stage in list(m.enc) + [m.bottleneck] + list(m.dec) for blk in stage]
    return (tuple(m.embed.stride), "expand" if hasattr(m.final, "expand") else "proj",
            tuple((tuple(b.win), tuple(b.shift), int(b.attn.heads)) for b in blocks),
            tuple(len(s) for s in m.skip_blocks))


def _swin_member(blob: dict, device: str):
    """The vendored SwinUNet for a `swin_unet*` blob, or None when `SWIN_IMPL` resolves to the
    training definition -- `load_any` then runs its models_alt branch exactly as before."""
    impl = SWIN_IMPL
    if impl not in ("auto", "vendored", "models_alt"):
        raise ValueError(f"SWIN_IMPL must be 'auto', 'vendored' or 'models_alt', not {impl!r}")
    if impl == "models_alt" or (
            impl == "auto" and os.path.exists(os.path.join(_UNET_DIR, "models_alt.py"))):
        return None
    arch = blob.get("arch")
    if arch not in SWIN_VARIANTS:
        raise ValueError(f"{arch}: no vendored Swin variant (have {', '.join(SWIN_VARIANTS)}); "
                         "add it to SWIN_VARIANTS once verify/fuse_swin.py passes on it")
    cfg = blob.get("config", {})
    return SwinUNet(blob.get("width", cfg.get("width", 96)), blob.get("n_layers", 4),
                    blob.get("var_head", True), **SWIN_VARIANTS[arch]).to(device)


class _StkLN(nn.Module):
    """K LayerNorms over the last axis of a member-major (K, B, h, w, c) tensor.

    LayerNorm mixes nothing across tokens or members, so the normalization runs once and only
    the affine is per member -- the kernel's own arithmetic, ((x - mu) * rsqrt) * w + b, in two
    calls instead of one."""

    def __init__(self, norms):
        super().__init__()
        self.eps = float(norms[0].eps)
        assert all(float(n.eps) == self.eps for n in norms), "LayerNorm eps differs"
        k, c = len(norms), norms[0].weight.shape[0]
        self.register_buffer("w", torch.stack([n.weight.detach() for n in norms])
                             .reshape(k, 1, 1, 1, c).clone(), persistent=False)
        self.register_buffer("b", torch.stack([n.bias.detach() for n in norms])
                             .reshape(k, 1, 1, 1, c).clone(), persistent=False)

    def forward(self, x):
        return torch.addcmul(self.b, _F.layer_norm(x, x.shape[-1:], eps=self.eps), self.w)


class _StkLinear(nn.Module):
    """K Linears as ONE batched matmul over a member-major (K, ..., in) tensor."""

    def __init__(self, lins):
        super().__init__()
        self.register_buffer("w", torch.stack([lin.weight.detach() for lin in lins]).clone(),
                             persistent=False)
        has_b = [lin.bias is not None for lin in lins]
        assert all(has_b) or not any(has_b), "bias present on some members only"
        self.register_buffer("b", torch.stack([lin.bias.detach() for lin in lins])[:, None, :]
                             .clone() if has_b[0] else None, persistent=False)

    def forward(self, x):
        k, o, i = self.w.shape
        t = x.reshape(k, -1, i)
        wt = self.w.transpose(1, 2)
        y = torch.bmm(t, wt) if self.b is None else torch.baddbmm(self.b, t, wt)
        return y.view(tuple(x.shape[:-1]) + (o,))


class _StkPoint(_StkLinear):
    """K same-shape 1x1 Conv2d (bias, groups 1) as ONE batched matmul on member-major tokens
    (K, B, H, W, in) -- a Linear in all but name. Job 7287329: as grouped 1x1 convs the Swin heads
    fell onto a slow cuDNN kernel under fp16 (34.7 of the stack's 55.7 device ms at batch 64)."""

    def __init__(self, convs):
        nn.Module.__init__(self)
        assert all(tuple(c.kernel_size) == (1, 1) and tuple(c.stride) == (1, 1) and c.groups == 1
                   and tuple(c.padding) == (0, 0) and c.bias is not None for c in convs), \
            "not a plain 1x1 conv"
        self.register_buffer("w", torch.stack([c.weight.detach()[:, :, 0, 0] for c in convs])
                             .contiguous(), persistent=False)
        self.register_buffer("b", torch.stack([c.bias.detach() for c in convs])[:, None, :].clone(),
                             persistent=False)


class _StkConv(nn.Module):
    """K same-shape Conv2d as one: over a SHARED input (the embedding: K*out channels, the
    member's own groups) or over member-major channels (groups multiplied by K)."""

    def __init__(self, convs, shared_input: bool = False):
        super().__init__()
        c0 = convs[0]
        assert isinstance(c0.padding, tuple), "string padding is not supported"
        self.stride, self.padding, self.dilation = c0.stride, c0.padding, c0.dilation
        self.groups = c0.groups if shared_input else len(convs) * c0.groups
        self.register_buffer("w", torch.cat([c.weight.detach() for c in convs], 0).contiguous(),
                             persistent=False)
        self.register_buffer("b", None if c0.bias is None else
                             torch.cat([c.bias.detach() for c in convs], 0).contiguous(),
                             persistent=False)

    def forward(self, x):
        return _F.conv2d(x, self.w, self.b, self.stride, self.padding, self.dilation, self.groups)


class _StkSwinBlock(nn.Module):
    """K `_SwinV2Block`s on member-major tokens (K, B, h, w, c).

    ATTENTION LAYOUT. Members and windows are pure batch axes to attention, so they fold into
    SDPA's batch: q, k, v are (K*B*nW, heads, N, d) -- the member's own layout with a leading K --
    and the additive mask (member k's position bias plus window w's shift mask) is expanded to
    that batch once per (batch size, dtype) and cached. Job 7287329 measured this 5-8 % faster
    than folding members and windows into the head axis with a batch-free mask; the price is
    memory, ~58 MB of fp16 mask per stage-1 block at K = 4 and batch 112. The logit scale and
    position bias are computed ONCE here, in fp32, by the exact ops `_WindowAttnV2.freeze_cache`
    runs, so the stack matches a `prepare_inference()` member (bitwise equal to an as-trained one
    in fp32; under fp16 autocast the as-trained member recomputes its bias in fp16).
    """

    def __init__(self, blocks, res):
        super().__init__()
        b0, k = blocks[0], len(blocks)
        self.win, self.shift, self.heads = tuple(b0.win), tuple(b0.shift), int(b0.attn.heads)
        hh, wd = res
        wh, ww = self.win
        assert hh % wh == 0 and wd % ww == 0, f"window {self.win} does not tile {res}"
        nw, n, h = (hh // wh) * (wd // ww), wh * ww, self.heads
        self.qkv = _StkLinear([b.attn.qkv for b in blocks])
        self.proj = _StkLinear([b.attn.proj for b in blocks])
        self.norm1 = _StkLN([b.norm1 for b in blocks])
        self.fc1 = _StkLinear([b.mlp[0] for b in blocks])
        self.fc2 = _StkLinear([b.mlp[3] for b in blocks])
        self.norm2 = _StkLN([b.norm2 for b in blocks])
        dev = b0.attn.logit_scale.device
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
            scale = torch.stack([torch.clamp(b.attn.logit_scale, max=math.log(100.0)).exp()
                                 for b in blocks])                               # (K, h, 1, 1)
            bias = torch.stack([b.attn.cpb(b.attn.coords)[b.attn.rel_index]
                                .reshape(n, n, h).permute(2, 0, 1) for b in blocks])  # (K,h,N,N)
            if b0.mask is None:
                am = bias[:, None].expand(k, nw, h, n, n)
            else:
                assert b0.mask.shape[0] == nw, f"mask has {b0.mask.shape[0]} windows, not {nw}"
                assert all(torch.equal(b.mask, b0.mask) for b in blocks), "shift masks differ"
                am = bias[:, None] + b0.mask[None]                                # (K,nW,h,N,N)
        self.register_buffer("scale", scale.reshape(k, 1, h, 1, 1).contiguous(), persistent=False)
        self.register_buffer("am", am.contiguous(), persistent=False)
        self._am = {}

    def _mask(self, b, dt):
        key = (b, dt, self.am.device)
        if key not in self._am:
            k, nw, h, n, _ = self.am.shape
            self._am[key] = (self.am.to(dt)[:, None].expand(k, b, nw, h, n, n)
                             .reshape(k * b * nw, h, n, n))
        return self._am[key]

    def forward(self, x):
        k, b, hh, wd, c = x.shape
        (wh, ww), (sy, sx), h = self.win, self.shift, self.heads
        ny, nx = hh // wh, wd // ww
        nw, n, d = ny * nx, wh * ww, c // h
        y = torch.roll(x, (-sy, -sx), (2, 3)) if (sy or sx) else x
        t = y.reshape(k, b, ny, wh, nx, ww, c).permute(0, 1, 2, 4, 3, 5, 6).reshape(k, -1, c)
        q, kk, v = self.qkv(t).view(k * b * nw, n, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
        q = (_F.normalize(q, dim=-1).reshape(k, b * nw, h, n, d) * self.scale).reshape(k * b * nw, h, n, d)
        kk = _F.normalize(kk, dim=-1)
        dt = v.dtype
        o = _F.scaled_dot_product_attention(q.to(dt), kk.to(dt), v, attn_mask=self._mask(b, dt),
                                            scale=1.0)
        o = o.transpose(1, 2).reshape(k, -1, c)
        y = (self.proj(o).view(k, b, ny, nx, wh, ww, c).permute(0, 1, 2, 4, 3, 5, 6)
             .reshape(k, b, hh, wd, c))
        if sy or sx:
            y = torch.roll(y, (sy, sx), (2, 3))
        x = x + self.norm1(y)
        return x + self.norm2(self.fc2(_F.gelu(self.fc1(x))))


def _tok_to_cl(t):
    """Member-major tokens (K, B, h, w, c) -> member-major channels (B, K*c, h, w), CHANNELS-LAST.

    Every conv of a SwinUNet member sees a permuted token tensor, i.e. channels-last memory, and
    the kernel a backend picks depends on the memory format: laid out contiguous NCHW, the grouped
    versions of the skip and refine convs departed from the members by up to 7e-6 on CPU;
    laid out channels-last they reproduce them bit for bit (group count does not matter)."""
    k, b, hh, wd, c = t.shape
    return t.permute(1, 2, 3, 0, 4).reshape(b, hh, wd, k * c).permute(0, 3, 1, 2)


def _cl_to_tok(z, k: int):
    """The inverse of `_tok_to_cl`: channels-last (B, K*c, h, w) -> (K, B, h, w, c)."""
    b, kc, hh, wd = z.shape
    return z.permute(0, 2, 3, 1).reshape(b, hh, wd, k, kc // k).permute(3, 0, 1, 2, 4)


class _StkConvNeXt(nn.Module):
    """K `_ConvNeXtBlock`s on member-major channels laid out channels-last (`_tok_to_cl`): the
    depthwise conv's groups already separate every channel, so K members are one conv with K*c
    groups."""

    def __init__(self, blocks):
        super().__init__()
        self.k, self.c = len(blocks), blocks[0].dw.in_channels
        self.dw = _StkConv([b.dw for b in blocks])
        self.norm = _StkLN([b.norm for b in blocks])
        self.pw1 = _StkLinear([b.pw1 for b in blocks])
        self.pw2 = _StkLinear([b.pw2 for b in blocks])
        self.register_buffer("gamma", torch.stack([b.gamma.detach() for b in blocks])
                             .reshape(self.k, 1, 1, 1, self.c).clone(), persistent=False)

    def forward(self, z):
        t = self.norm(_cl_to_tok(self.dw(z), self.k))                         # (K, B, h, w, c)
        t = self.pw2(_F.gelu(self.pw1(t)))
        t = t * self.gamma.to(t.dtype)
        return z + _tok_to_cl(t)


class _StkPatch(nn.Module):
    """K `_PatchMergeV2` ("merge"), `_PatchExpand` ("expand") or `_TokenProj` ("proj")."""

    def __init__(self, mods, kind: str):
        super().__init__()
        self.kind = kind
        self.lin = _StkLinear([getattr(m, {"merge": "reduction"}.get(kind, kind)) for m in mods])
        self.norm = _StkLN([m.norm for m in mods])

    def forward(self, x):
        k, b, hh, wd, _ = x.shape
        if self.kind == "merge":
            x = torch.cat([x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2], x[:, :, 0::2, 1::2],
                           x[:, :, 1::2, 1::2]], -1)
            return self.norm(self.lin(x))
        if self.kind == "expand":
            x = (self.lin(x).reshape(k, b, hh, wd, 2, 2, -1).permute(0, 1, 2, 4, 3, 5, 6)
                 .reshape(k, b, 2 * hh, 2 * wd, -1))
            return self.norm(x)
        return self.norm(self.lin(x))


class StackedSwin(nn.Module):
    """K identically-structured SwinUNet members evaluated in ONE invocation.

    The board charges an invocation (+1.151 board ms on Track 1, +0.866 on Track 2), so a Swin
    package of four members is only competitive fused, as StackedFNO and StackedViT made FNO and
    ViT members. Nothing is assumed about the structure: depths, windows, shifts, heads, patch
    size and every LayerNorm eps are read off the member modules, which `load_any` built from
    each checkpoint's own `arch` / `width` / `n_layers`, so the Phase-2 ladder (`_b6`, `_win16`,
    `_p1`) folds without code changes and members that differ in any of them never share a stack
    (`fuse` keys on `_swin_signature`).

    HOW THE MEMBER AXIS FOLDS. Tokens travel member-major, (K, B, h, w, c):
      * the patch embedding reads the SAME input for every member: one conv, K*width outputs;
      * every Linear is one `baddbmm` against a stacked (K, out, in) weight;
      * every LayerNorm normalizes once and applies a per-member affine (`_StkLN`);
      * window attention folds members and windows into SDPA's batch axis (`_StkSwinBlock`);
      * the ConvNeXt skips and the refine conv run on member-major channels as grouped convs laid
        out channels-last, the members' own memory format (`_tok_to_cl`); the 1x1 heads are
        batched matmuls on tokens (`_StkPoint`): as grouped convs cuDNN ran them 20x slower.
    `forward` returns the member MEAN of the field and log(mean(exp(clamped logstd))), the
    contract StackedFNO and StackedViT obey, so `predict_fno` weights it by `n_members` with no
    special case. Head outputs are averaged in fp32 even under autocast.
    """

    def __init__(self, members):
        super().__init__()
        first = members[0]
        self.n_members = k = len(members)
        sd0, sig0 = first.state_dict(), _swin_signature(first)
        for m in members[1:]:
            sd = m.state_dict()
            assert sd.keys() == sd0.keys(), "state_dict keys differ"
            for key, v in sd.items():
                assert v.shape == sd0[key].shape, f"{key} shape mismatch"
            assert _swin_signature(m) == sig0, "window / shift / patch structure differs"
            assert torch.equal(m.grid, first.grid), "coordinate grids differ"
        self.register_buffer("grid", first.grid.detach().clone(), persistent=False)
        with torch.no_grad():
            self.embed = _StkConv([m.embed for m in members], shared_input=True)
            self.embed_norm = _StkLN([m.embed_norm for m in members])
            py, px = first.embed.stride
            n_down = len(first.enc)
            res = [((H // py) >> i, (W // px) >> i) for i in range(n_down + 1)]
            self.enc = nn.ModuleList(
                nn.ModuleList(_StkSwinBlock([m.enc[i][j] for m in members], res[i])
                              for j in range(len(first.enc[i]))) for i in range(n_down))
            self.skip_blocks = nn.ModuleList(
                nn.ModuleList(_StkConvNeXt([m.skip_blocks[i][j] for m in members])
                              for j in range(len(first.skip_blocks[i]))) for i in range(n_down))
            self.merge = nn.ModuleList(_StkPatch([m.merge[i] for m in members], "merge")
                                       for i in range(n_down))
            self.bottleneck = nn.ModuleList(
                _StkSwinBlock([m.bottleneck[j] for m in members], res[n_down])
                for j in range(len(first.bottleneck)))
            self.expand = nn.ModuleList(_StkPatch([m.expand[j] for m in members], "expand")
                                        for j in range(n_down))
            self.skip = nn.ModuleList(_StkLinear([m.skip[j] for m in members])
                                      for j in range(n_down))
            self.dec = nn.ModuleList(
                nn.ModuleList(_StkSwinBlock([m.dec[j][t] for m in members], res[n_down - 1 - j])
                              for t in range(len(first.dec[j]))) for j in range(n_down))
            self.final = _StkPatch([m.final for m in members],
                                   "expand" if hasattr(first.final, "expand") else "proj")
            assert isinstance(first.refine[0], nn.GELU) and len(first.refine) == 2
            self.refine = _StkConv([m.refine[1] for m in members])
            self.hf = nn.ModuleList([_StkPoint([m.head_field[0] for m in members]),
                                     _StkPoint([m.head_field[2] for m in members])])
            self.hl = None
            if first.head_logstd is not None:
                self.hl = nn.ModuleList([_StkPoint([m.head_logstd[0] for m in members]),
                                         _StkPoint([m.head_logstd[2] for m in members])])
        self.fp32_max_batch = int(getattr(first, "fp32_max_batch", 0) or 0)
        self.self_norm = bool(getattr(first, "self_norm", False))

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        if (self.fp32_max_batch and len(prev) <= self.fp32_max_batch and prev.is_cuda
                and torch.is_autocast_enabled()):
            with torch.autocast(device_type="cuda", enabled=False):
                return self._forward(prev.float())
        return self._forward(prev)

    def _forward(self, prev: torch.Tensor):
        b, k = len(prev), self.n_members
        x = self.embed(torch.cat([prev, self.grid.expand(b, -1, -1, -1)], 1))
        _, kc, hh, wd = x.shape
        x = self.embed_norm(x.view(b, k, kc // k, hh, wd).permute(1, 0, 3, 4, 2))
        skips = []
        for blocks, sblk, merge in zip(self.enc, self.skip_blocks, self.merge):
            for blk in blocks:
                x = blk(x)
            z = _tok_to_cl(x)
            for cb in sblk:
                z = cb(z)
            skips.append(_cl_to_tok(z, k))
            x = merge(x)
        for blk in self.bottleneck:
            x = blk(x)
        for expand, skip, blocks, s in zip(self.expand, self.skip, self.dec, reversed(skips)):
            x = skip(torch.cat([expand(x), s], -1))
            for blk in blocks:
                x = blk(x)
        z = _tok_to_cl(self.final(x))
        t = _cl_to_tok(z + self.refine(_F.gelu(z)), k)                         # (K, B, H, W, out)
        # the 1x1 heads as batched matmuls on tokens; member mean in fp32 -> (B, out, H, W)
        field = self.hf[1](_F.gelu(self.hf[0](t))).float().mean(0).permute(0, 3, 1, 2)
        logstd = None
        if self.hl is not None:
            ls = self.hl[1](_F.gelu(self.hl[0](t))).float()
            logstd = ls.clamp(-8.0, 2.0).exp().mean(0).log().permute(0, 3, 1, 2)
        return field, logstd


# ---------------------------------------------------------------------------------------------
# F-FNO members (backbone campaign, 2026-09-14): a vendored FFNO2d and its fused stack
# ---------------------------------------------------------------------------------------------
# `arch` -> FFNO2d keyword overrides, mirroring `models_alt.build_wave1`.
FFNO_VARIANTS = {"ffno": {"ff": True}, "ffno_nff": {"ff": False}}
# Which FFNO2d class `load_any` builds -- the contract of SWIN_IMPL: "auto" = the training definition
# inside the repo (every local instrument unchanged) and the vendored copy below inside an archive.
FFNO_IMPL = "auto"
# The spectral path `load_any` gives every F-FNO member: None keeps the as-built one (the as-trained
# FFTs); "dense" (1-D FFTs as matmuls on a precomputed real basis, fp16-native) or "fft". A package
# bakes it per track, as SPEC_PATH is baked: dense for Track 1's batch, fft for Track 2's batch 1.
FFNO_PATH = None
# FFNO_TILE: rows of the batch the FUSED F-FNO stack's dense spectral stage processes at a time.
# Measured 09-18 (job 7370333, profiler + interleaved bench on the L8 twins at b64 fp16): the whole
# of StackedFFNO's fixed ~30 % penalty against its own member loop lives in `_spectral_dense` --
# 4.454 ms fused against 2.313 for four member layers, 1.93x, while the lift is 1.014x. The cause is
# size, not arithmetic: the two permute-reshapes per axis materialise (b, k, ci, w, 2, my) as
# (my*k, b*w, 2ci), which is ~12 MB at K=1 and fits the A100's 40 MB L2, and ~47 MB at K=4, which
# does not. Tiling the stage over the batch restores the member loop's cache behaviour while
# keeping ONE invocation -- and one invocation, not one kernel, is what the board charges its
# 1.151 ms for. Measured: chunk 32 0.541x, 16 0.534x, 8 0.608x of the untiled stage, all three
# BITWISE identical to it (max |diff| 0.00e+00); chunk 4 regresses to 1.176x and stops being
# bitwise, so the window is real and 16 sits in the middle of it. 0 disables.
# It is a TARGET, not an exact chunk: the batch is split into ceil(b / FFNO_TILE) EQUAL tiles, and
# if it does not divide evenly the stage runs untiled. Equal tiles matter for correctness, not
# tidiness -- a ragged tail gets its own cuBLAS kernel and makes a window's fp16 output depend on
# its position in the batch, which broke validate_strict's permutation check (0.0 -> 1.4e-4).
# Batches at or below the tile take the untiled path unchanged, so Track 2's batch-1 graph is
# bit-identical to what it was.
FFNO_TILE = int(os.environ.get("RPDE_FFNO_TILE", "16"))


class _FFNOLayer(nn.Module):
    """Copy of `models_alt._FFNOLayer`: the separable Fourier operator (rfft along y, one complex
    matrix per retained ky, irfft; the same along x; summed), then x + LN(FF(K x)) (`ffno`) or
    x + GELU(GN(K x + W x)) (`ffno_nff`). Dropout is training-only and omitted."""

    def __init__(self, width: int, modes_y: int, modes_x: int, ff: bool = True, factor: int = 2,
                 grid: tuple[int, int] = (H, W)):
        super().__init__()
        self.my, self.mx = min(modes_y, grid[0] // 2 + 1), min(modes_x, grid[1] // 2 + 1)
        self.wy = nn.Parameter(torch.zeros(width, width, self.my, 2))
        self.wx = nn.Parameter(torch.zeros(width, width, self.mx, 2))
        self.use_ff = bool(ff)
        if self.use_ff:
            self.ff = nn.Sequential(nn.Linear(width, factor * width), nn.GELU(),
                                    nn.Linear(factor * width, width))
            self.norm = nn.LayerNorm(width)
        else:
            self.point = nn.Conv2d(width, width, 1)
            self.norm = nn.GroupNorm(8, width)
        self.dense = False
        self._bases: dict = {}

    @staticmethod
    def _dense_basis(n: int, m: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Real (n, 2m) analysis and (2m, n) synthesis matrices reproducing rfft(norm="ortho")[:m]
        and irfft(n=n, norm="ortho"), extracted in float64 by applying those very ops to unit
        vectors (models_alt's derivation, verbatim)."""
        a = torch.fft.rfft(torch.eye(n, dtype=torch.float64), dim=-1, norm="ortho")[:, :m]
        e = torch.eye(m, dtype=torch.complex128)
        s_re = torch.fft.irfft(e, n=n, dim=-1, norm="ortho")
        s_im = torch.fft.irfft(1j * e, n=n, dim=-1, norm="ortho")
        return (torch.cat([a.real, a.imag], 1).to(device=device, dtype=torch.float32),
                torch.cat([s_re, s_im], 0).to(device=device, dtype=torch.float32))

    @staticmethod
    def _w2(wt: torch.Tensor) -> torch.Tensor:
        """(in, out, m, 2) complex weights -> (m, 2*out, 2*in) real 2x2-block matrices."""
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
        cy = torch.matmul(x.transpose(-1, -2), py)
        cy = cy.reshape(b, c, w, 2, my).permute(4, 0, 2, 3, 1).reshape(my, b * w, 2 * c)
        dy = torch.bmm(cy, self._w2(self.wy).transpose(1, 2))
        co = dy.shape[-1] // 2
        dy = dy.reshape(my, b, w, 2, co).permute(1, 4, 2, 3, 0).reshape(b, co, w, 2 * my)
        yy = torch.matmul(dy, sy).transpose(-1, -2)
        cx = torch.matmul(x, px)
        cx = cx.reshape(b, c, h, 2, mx).permute(4, 0, 2, 3, 1).reshape(mx, b * h, 2 * c)
        dx = torch.bmm(cx, self._w2(self.wx).transpose(1, 2))
        dx = dx.reshape(mx, b, h, 2, co).permute(1, 4, 2, 3, 0).reshape(b, co, h, 2 * mx)
        xx = torch.matmul(dx, sx)
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
            b = _F.gelu(self.norm(s + self.point(x)))
        return x + b


class _LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the CHANNEL axis of an NCHW map (models_alt's)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FFNO2d(nn.Module):
    """Copy of `Track_1/unet/models_alt.py:FFNO2d` (Tran et al., ICLR 2023): lift MLP (42 -> 128
    -> w), zero-pad the far side of both axes by PAD, `n_layers` F-FNO layers, crop, LayerNorm, the
    shared MLP heads. Module names and registration order are the training definition's."""

    PAD = (4, 8)
    FF_FACTOR = 2

    def __init__(self, width: int = 64, modes_y: int = 16, modes_x: int = 32, n_layers: int = 12,
                 var_head: bool = True, ff: bool = True):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])
        self.lift = nn.Sequential(nn.Conv2d(CH + 2, 128, 1), nn.GELU(), nn.Conv2d(128, width, 1))
        grid = (H + self.PAD[0], W + self.PAD[1])
        self.layers = nn.ModuleList([_FFNOLayer(width, modes_y, modes_x, ff, self.FF_FACTOR, grid)
                                     for _ in range(n_layers)])
        self.norm_out = _LayerNorm2d(width)
        self.head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, CH, 1))
        self.head_logstd = None
        if var_head:
            self.head_logstd = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                             nn.Conv2d(128, C, 1))

    def set_spectral_path(self, path: str = "fft"):
        if path not in ("fft", "dense"):
            raise ValueError(f"spectral path must be 'fft' or 'dense', not {path!r}")
        for layer in self.layers:
            layer.dense = path == "dense"
        return self

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        x = torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1)
        h = _F.pad(self.lift(x), (0, self.PAD[1], 0, self.PAD[0]))
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h[..., :H, :W])
        logstd = None
        if self.head_logstd is not None:
            logstd = self.head_logstd(h.detach() if detach_var else h)
        return self.head_field(h), logstd


def _is_ffno(m) -> bool:
    """An FFNO2d member of either definition (vendored above, or models_alt's)."""
    return type(m).__name__ == "FFNO2d" and all(
        hasattr(m, a) for a in ("lift", "layers", "norm_out", "PAD", "grid", "head_field"))


def _ffno_signature(m):
    """Structure a state_dict does not show (padding, spectral path), or None for other families."""
    if not _is_ffno(m):
        return None
    return (tuple(m.PAD), tuple((int(l.my), int(l.mx), bool(l.use_ff), bool(l.dense))
                                for l in m.layers))


def _ffno_member(blob: dict, device: str):
    """The vendored FFNO2d for an `ffno*` blob, or None when FFNO_IMPL resolves to the training
    definition -- `load_any` then runs its models_alt branch exactly as before."""
    impl = FFNO_IMPL
    if impl not in ("auto", "vendored", "models_alt"):
        raise ValueError(f"FFNO_IMPL must be 'auto', 'vendored' or 'models_alt', not {impl!r}")
    if impl == "models_alt" or (
            impl == "auto" and os.path.exists(os.path.join(_UNET_DIR, "models_alt.py"))):
        return None
    arch = blob.get("arch")
    cfg = blob.get("config", {})
    return FFNO2d(blob.get("width", cfg.get("width", 96)), blob.get("modes_y", 8),
                  blob.get("modes_x", 12), blob.get("n_layers", 4), blob.get("var_head", True),
                  **FFNO_VARIANTS[arch]).to(device)


class _StkFFNOLayer(nn.Module):
    """K `_FFNOLayer`s on member-major channels (B, K*w, h, w).

    FFT path: ONE rfft / irfft per axis over every member's channels (the transforms act on the
    spatial axes only), the complex mixing as one einsum against stacked (K, in, out, m) weights,
    pinned to fp32 as the member pins it. Dense path: the basis projections are shared by all
    members, and the per-mode real 2x2-block mixing is one bmm over (m*K) matrices -- each member's
    `_w2` precomputed once. The feed-forward runs on member-major tokens (_StkLinear, _StkLN); the
    `ffno_nff` point conv and GroupNorm fold the member axis the way StackedFNO folds them."""

    def __init__(self, layers):
        super().__init__()
        l0, k = layers[0], len(layers)
        self.k, self.my, self.mx, self.use_ff = k, int(l0.my), int(l0.mx), bool(l0.use_ff)
        self.ci, self.co = int(l0.wy.shape[0]), int(l0.wy.shape[1])
        with torch.no_grad():
            self.register_buffer("wy", torch.stack([l.wy.detach() for l in layers]).contiguous(),
                                 persistent=False)
            self.register_buffer("wx", torch.stack([l.wx.detach() for l in layers]).contiguous(),
                                 persistent=False)
            self.register_buffer("w2y", torch.stack([_FFNOLayer._w2(l.wy.detach()) for l in layers], 1)
                                 .reshape(self.my * k, 2 * self.co, 2 * self.ci).contiguous(),
                                 persistent=False)
            self.register_buffer("w2x", torch.stack([_FFNOLayer._w2(l.wx.detach()) for l in layers], 1)
                                 .reshape(self.mx * k, 2 * self.co, 2 * self.ci).contiguous(),
                                 persistent=False)
        if self.use_ff:
            self.ff0 = _StkLinear([l.ff[0] for l in layers])
            self.ff2 = _StkLinear([l.ff[2] for l in layers])
            self.norm = _StkLN([l.norm for l in layers])
        else:
            n0 = l0.norm
            assert all(l.norm.num_groups == n0.num_groups and l.norm.eps == n0.eps for l in layers)
            self.point = _StkConv([l.point for l in layers])
            self.gn_groups, self.gn_eps = n0.num_groups * k, float(n0.eps)
            self.register_buffer("gn_w", torch.cat([l.norm.weight.detach() for l in layers]).clone(),
                                 persistent=False)
            self.register_buffer("gn_b", torch.cat([l.norm.bias.detach() for l in layers]).clone(),
                                 persistent=False)
        self.dense = bool(l0.dense)
        self._bases: dict = {}

    def _spectral_fft(self, x):
        b, _, h, w = x.shape
        k, my, mx, ci, co = self.k, self.my, self.mx, self.ci, self.co
        fy = torch.fft.rfft(x, dim=-2, norm="ortho")[:, :, :my].reshape(b, k, ci, my, w)
        yy = torch.fft.irfft(torch.einsum("bkiyx,kioy->bkoyx", fy, torch.view_as_complex(self.wy)),
                             n=h, dim=-2, norm="ortho")
        fx = torch.fft.rfft(x, dim=-1, norm="ortho")[..., :mx].reshape(b, k, ci, h, mx)
        xx = torch.fft.irfft(torch.einsum("bkiyx,kiox->bkoyx", fx, torch.view_as_complex(self.wx)),
                             n=w, dim=-1, norm="ortho")
        return (yy + xx).reshape(b, k * co, h, w)

    def _spectral_dense(self, x):
        """The dense spectral stage, in batch tiles (the note on `FFNO_TILE`).

        Same arithmetic, same order, same kernels -- only fewer rows per launch, so the
        intermediates stay in L2. Verified bitwise against the untiled path at tiles 8, 16 and 32.
        """
        b, t = x.shape[0], FFNO_TILE
        if t and b > t:
            # EVERY TILE THE SAME SIZE, or no tiling at all. A tail tile of a different length
            # gets a different cuBLAS kernel, and then a window's fp16 result depends on WHERE IN
            # THE BATCH it sat: the first cut of this (tiles of exactly FFNO_TILE, ragged tail)
            # took validate_strict's permutation drift from 0.0 to 7.6e-5 fp16 and 1.4e-4 through
            # the CUDA graph, past the 1e-4 limit, on a package that must be permutation-invariant
            # by contract. Splitting into `n` EQUAL tiles keeps one kernel shape for the whole
            # batch, so the result is position-independent again, and it wastes nothing -- unlike
            # padding the tail, which CLAUDE.md rules out for exactly the kernel-selection reason
            # that causes the drift. When b does not divide evenly the stage runs untiled, which
            # is the path that was always exact.
            n = -(-b // t)
            if b % n == 0:
                q = b // n
                return torch.cat([self._spectral_dense_rows(x[i:i + q])
                                  for i in range(0, b, q)], 0)
        return self._spectral_dense_rows(x)

    def _spectral_dense_rows(self, x):
        b, _, h, w = x.shape
        k, my, mx, ci, co = self.k, self.my, self.mx, self.ci, self.co
        key = (x.device, h, w)
        if key not in self._bases:
            self._bases[key] = (_FFNOLayer._dense_basis(h, my, x.device),
                                _FFNOLayer._dense_basis(w, mx, x.device))
        (py, sy), (px, sx) = self._bases[key]
        cy = torch.matmul(x.transpose(-1, -2), py)                            # (B, K*in, W, 2my)
        cy = cy.reshape(b, k, ci, w, 2, my).permute(5, 1, 0, 3, 4, 2).reshape(my * k, b * w, 2 * ci)
        dy = torch.bmm(cy, self.w2y.transpose(1, 2))                           # (my*K, B*W, 2out)
        dy = dy.reshape(my, k, b, w, 2, co).permute(2, 1, 5, 3, 4, 0).reshape(b, k * co, w, 2 * my)
        yy = torch.matmul(dy, sy).transpose(-1, -2)                            # (B, K*out, H, W)
        cx = torch.matmul(x, px)                                               # (B, K*in, H, 2mx)
        cx = cx.reshape(b, k, ci, h, 2, mx).permute(5, 1, 0, 3, 4, 2).reshape(mx * k, b * h, 2 * ci)
        dx = torch.bmm(cx, self.w2x.transpose(1, 2))
        dx = dx.reshape(mx, k, b, h, 2, co).permute(2, 1, 5, 3, 4, 0).reshape(b, k * co, h, 2 * mx)
        xx = torch.matmul(dx, sx)
        return yy + xx

    def forward(self, x):
        if self.dense:
            # One cast for both analysis matmuls: under autocast each would cast x (an activation, which
            # autocast does not cache) to fp16 again -- a full-size copy per layer (layout A/B, 09-14).
            xs = x.to(torch.get_autocast_gpu_dtype()) if (x.is_cuda and torch.is_autocast_enabled()) else x
            s = self._spectral_dense(xs)
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                s = self._spectral_fft(x.float())
        if self.use_ff:
            b, kc, h, w = s.shape
            t = s.reshape(b, self.k, kc // self.k, h, w).permute(1, 0, 3, 4, 2)   # (K, B, h, w, c)
            t = self.norm(self.ff2(_F.gelu(self.ff0(t))))
            # The residual is written straight into x's own layout, (B, K, c, h, w), so the next layer reads it
            # as (B, K*c, h, w) without a copy. `x + t.permute(...).reshape(...)` made a full-size fp32 copy
            # per layer -- the one copy the member loop never makes, and the whole of the stack's b64 deficit
            # (copy_ 40.3 against the loop's 16.2 device ms; layout A/B, scribe 09-14). Same sums, same order.
            out = torch.empty((b, self.k, kc // self.k, h, w), device=x.device,
                              dtype=torch.promote_types(x.dtype, t.dtype))
            torch.add(x.reshape(b, self.k, kc // self.k, h, w), t.permute(1, 0, 4, 2, 3), out=out)
            return out.view(b, kc, h, w)
        z = _F.group_norm(s + self.point(x), self.gn_groups, self.gn_w, self.gn_b, self.gn_eps)
        return x + _F.gelu(z)


class StackedFFNO(nn.Module):
    """K identically-structured FFNO2d members evaluated in ONE invocation.

    The P2-L8 package line (F-FNO L8 x4 + ViT x2) projects best on Track 1 only if the four F-FNOs
    cost one invocation (+1.151 board ms each otherwise). Structure is read off the members (modes,
    padding, feed-forward or point/GroupNorm, depth) and the spectral path is the members' own
    (`set_spectral_path` switches the whole stack): dense for Track 1's batch, fft for Track 2's
    batch 1. Channels are member-major throughout: the lift's first conv reads the shared input
    once (K*128 outputs), every later conv is grouped by K, the LayerNorms normalize once with a
    per-member affine, and the 1x1 heads are batched matmuls on tokens (`_StkPoint`).
    `forward` returns the member mean of the field and log(mean(exp(clamped logstd))), the
    StackedFNO contract, so `predict_fno` weights it by `n_members` with no special case."""

    def __init__(self, members):
        super().__init__()
        first = members[0]
        self.n_members = k = len(members)
        sd0, sig0 = first.state_dict(), _ffno_signature(first)
        for m in members[1:]:
            sd = m.state_dict()
            assert sd.keys() == sd0.keys(), "state_dict keys differ"
            for key, v in sd.items():
                assert v.shape == sd0[key].shape, f"{key} shape mismatch"
            assert _ffno_signature(m) == sig0, "padding / modes / spectral path differ"
            assert torch.equal(m.grid, first.grid), "coordinate grids differ"
        self.pad = tuple(first.PAD)
        self.register_buffer("grid", first.grid.detach().clone(), persistent=False)
        assert isinstance(first.lift[1], nn.GELU) and len(first.lift) == 3
        with torch.no_grad():
            self.lift0 = _StkConv([m.lift[0] for m in members], shared_input=True)
            self.lift2 = _StkConv([m.lift[2] for m in members])
            self.layers = nn.ModuleList(_StkFFNOLayer([m.layers[i] for m in members])
                                        for i in range(len(first.layers)))
            self.norm_out = _StkLN([m.norm_out for m in members])
            self.hf = nn.ModuleList([_StkPoint([m.head_field[0] for m in members]),
                                     _StkPoint([m.head_field[2] for m in members])])
            self.hl = None
            if first.head_logstd is not None:
                self.hl = nn.ModuleList([_StkPoint([m.head_logstd[0] for m in members]),
                                         _StkPoint([m.head_logstd[2] for m in members])])
        self.self_norm = bool(getattr(first, "self_norm", False))

    def set_spectral_path(self, path: str = "fft"):
        if path not in ("fft", "dense"):
            raise ValueError(f"spectral path must be 'fft' or 'dense', not {path!r}")
        for layer in self.layers:
            layer.dense = path == "dense"
        return self

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        b, k = len(prev), self.n_members
        h = self.lift2(_F.gelu(self.lift0(torch.cat([prev, self.grid.expand(b, -1, -1, -1)], 1))))
        h = _F.pad(h, (0, self.pad[1], 0, self.pad[0]))
        for layer in self.layers:
            h = layer(h)
        h = h[..., :H, :W]
        t = self.norm_out(h.reshape(b, k, h.shape[1] // k, H, W).permute(1, 0, 3, 4, 2))
        field = self.hf[1](_F.gelu(self.hf[0](t))).float().mean(0).permute(0, 3, 1, 2)
        logstd = None
        if self.hl is not None:
            ls = self.hl[1](_F.gelu(self.hl[0](t))).float()
            logstd = ls.clamp(-8.0, 2.0).exp().mean(0).log().permute(0, 3, 1, 2)
        return field, logstd


def fuse(models):
    """Group `models` by architecture and fuse each FNO group. Order-preserving by first sight.

    A package of 5x width-24 + w48 + m1220 + w64 goes from 8 invocations to 4, and the round-12
    mixed package (8 FNO + 1 U-Net) from 9 to 4. FNO2d members fold through StackedFNO,
    ViT2d through StackedViT, SwinUNet through StackedSwin and FFNO2d through StackedFFNO; every other family lands in its own group and is handed
    back untouched -- `predict_fno` then calls each as one more (field, logstd) module. A
    U-Net's state_dict keys differ from every FNO's, which is what puts it in its own group.

    Round 23 priced what this buys: shipping round 20's four members UNFUSED cost 1.151 board
    ms per invocation on Track 1 and 0.866 on Track 2, i.e. **-0.085 / -0.058 `final` for the
    first extra one**, with every error subscore bit-identical. Fusion is not a micro-
    optimization; it is the difference between a member being free and a member being an
    accuracy arm's worth of latency.
    """
    groups = {}
    for m in models:
        # Keyed on the full parameter signature rather than a few attributes, so any
        # architecture flag (var_mlp, lift_mlp, ...) separates groups automatically. The
        # spectral weights are excluded: StackedFNO zero-pads differing mode counts, which is
        # exact, so a modes-(12,20) member joins its width-matched (8,12) siblings instead of
        # buying its own invocation. Width and every other shape still separate, because they
        # appear in lift / point / norm / the heads.
        # `self_norm` is an ATTRIBUTE, not a parameter, so it does not appear in the shape
        # signature -- two members trained under different normalizers would otherwise fuse
        # into one StackedFNO whose input affine can only be right for one of them, and the
        # fused module would carry no flag at all. Keying on it makes that impossible.
        # `self_norm` and `osc_zeromean` are ATTRIBUTES, not parameters, so they do not appear
        # in the shape signature -- members that disagree on either would otherwise fuse into
        # one module that can only be right for some of them. `osc_modes` and `osc_global` DO
        # show up (they set head_osc's out_channels and osc_w's out_features), but they are
        # keyed explicitly rather than left to that coincidence.
        key = (bool(getattr(m, "self_norm", False)),
               getattr(m, "osc_modes", 0), bool(getattr(m, "osc_global", False)),
               bool(getattr(m, "osc_zeromean", False)), getattr(m, "tbasis", 0),
               tuple((k, tuple(v.shape)) for k, v in m.state_dict().items()
                     if not (k.endswith(".w1") or k.endswith(".w2"))),
               # Window, shift and patch live outside the state_dict (swin_unet_win16 has
               # swin_unet's exact shapes), so Swin members also key on their structure.
               # None for every other family, which leaves their grouping unchanged.
               _swin_signature(m), _ffno_signature(m))
        groups.setdefault(key, []).append(m)
    out = []
    for g in groups.values():
        # A member with a decoupled time-mean head is passed through UNFUSED. StackedFNO
        # enumerates lift / point / norm / head_field / head_logstd explicitly, so a
        # `head_mean` would be silently DROPPED and the member would emit a zero-mean field.
        # Losing an invocation costs latency; losing a head is a wrong answer, and it would
        # not show up as a crash.
        # `head_mean` is still passed through UNFUSED: StackedFNO enumerates lift / point /
        # norm / head_field / head_logstd / head_osc explicitly, so a `head_mean` would be
        # silently DROPPED and the member would emit a zero-mean field. Losing an invocation
        # costs latency; losing a head is a wrong answer that does not show up as a crash.
        # `head_osc` used to be in this list for the same reason and no longer is -- it is
        # implemented above, and `verify/fuse_osc.py` checks the fused path against the
        # sequential one.
        if len(g) == 1:
            out.extend(g)     # `extend`, not `append`: two U-Nets are two invocations, not one
        elif isinstance(g[0], ViT2d):
            # Round 23: the board is launch-bound, so a second ViT is free fused and -0.085
            # `final` unfused. StackedViT folds the member axis the same way StackedFNO does.
            out.append(StackedViT(g).to(g[0].grid.device).eval())
        elif _is_swin(g[0]):
            # 2026-09-14: K Swin U-Nets in one invocation. Built on the members' own
            # device from their own modules, so every structural choice is read, not assumed.
            out.append(StackedSwin(g).eval())
        elif _is_ffno(g[0]):
            # 2026-09-14: K F-FNOs in one invocation (StackedFFNO), on the members' own spectral
            # path; padding, modes and path are keyed, so only identical structures share a stack.
            out.append(StackedFFNO(g).eval())
        elif not isinstance(g[0], FNO2d) or g[0].head_mean is not None:
            out.extend(g)
        else:
            st = StackedFNO(g).to(g[0].grid.device).eval()
            st.self_norm = bool(getattr(g[0], "self_norm", False))
            out.append(st)
    return out


def _fno_from_blob(blob: dict, device: str):
    def cfg(key, default):
        """Read an architecture flag from the blob's top level, falling back to `config`.

        train2.py writes each flag explicitly at the top level AND dumps all of `vars(args)`
        under "config". The two lists drifted: `mean_head` was added to the model and to the
        argparse but never to the top-level list, so a --mean-head checkpoint built a
        head-less FNO2d here and died on unexpected state_dict keys -- at SCORING time, after
        the training had been paid for. Falling back to `config` makes the top-level list an
        optimization rather than a second source of truth, and fixes checkpoints already on
        disk without re-training them.
        """
        if key in blob:
            return blob[key]
        return blob.get("config", {}).get(key, default)

    m = FNO2d(cfg("width", 24), cfg("modes_y", 8), cfg("modes_x", 12),
              cfg("n_layers", 4), cfg("var_head", True),
              var_mlp=cfg("var_mlp", False),
              lift_mlp=cfg("lift_mlp", False),
              point_mlp=cfg("point_mlp", False),
              var_hidden=cfg("var_hidden", 128),
              var_deep=cfg("var_deep", False),
              var_in_std=cfg("var_in_std", False),
              var_pred_std=cfg("var_pred_std", False),
              fgn_dim=cfg("fgn_dim", 0),
              mean_head=cfg("mean_head", False),
              osc_modes=cfg("osc_modes", 0),
              osc_global=cfg("osc_global", False),
              osc_zeromean=cfg("osc_zeromean", False),
              tbasis=cfg("tbasis", 0)).to(device)
    m.load_state_dict(blob["model_state_dict"])
    m.eval()
    m.self_norm = _self_norm_of(blob)
    return m


def load(path: str, device: str):
    """Backwards-compatible alias for `load_any`.

    This used to be an FNO-ONLY loader, on the assumption that the analysis scripts never see
    anything else. Round 12 put a U-Net in the member set and falsified that: twenty scripts
    call `load`, and the first member list containing `unet_g1.pth` crashed one of them
    (`alpha_board.py`) with a wall of missing state_dict keys. Dispatching is
    behaviour-identical for FNO blobs, so the fix belongs here rather than in twenty call
    sites.
    """
    return load_any(path, device)


_Q8_SUFFIX = "._q8scale"


def dequant8(sd: dict) -> dict:
    """Undo build_zip.py --int8 in place of the caller's state_dict, if it was applied.

    THE PROBLEM IT SOLVES.  A width-96 8-layer member is 108.5 MB fp32 and 54.3 MB fp16, so
    four fill 217 MB of the 256 MB extracted cap and eight need 434.  R18 / T2R6 measured the
    board-latency slope directly (0.25 / -0.04 board ms per local ms, against the round-13
    model's 0.803; the eleven one-invocation rounds refit to `3.274 + 0.336*local`), which cuts
    the price of doubling the members from 0.09-0.10 `final` to ~0.037 on Track 1 and ~0 on
    Track 2 -- so 8 members is worth about +0.05 / +0.09 and SIZE is the only thing in the way.

    THE FORMAT.  Every quantized tensor `k` is stored as int8 with a companion fp16 tensor
    `k + "._q8scale"` that broadcasts over it; this restores `k` exactly as the packer
    dequantizes it and deletes the companion, so `load_state_dict(strict=True)` downstream sees
    precisely the original key set and every branch of `load_any` is untouched.  A blob with no
    companion keys is returned unchanged, which is what makes this safe to run unconditionally.

    THE COST, measured on two all-data w96 L8 members over 96 val_unseen windows (`quant8.py`):
    scales per (out, in) channel pair give rel-L2 8.3e-4 of the field at 29.0 MB/member against
    fp16's 2.7e-4 at 56.9 MB.  Coarser granularities are 20-40x worse (per tensor 3.0e-2, per
    out-channel 1.7e-2, per Fourier mode 3.0e-2) and none of them is shippable.  8.3e-4 against
    a model rel-L2 of 0.077 adds in quadrature to +6e-5 of the error -- the same argument that
    made fp16 free in round 15.
    """
    scales = [k for k in sd if k.endswith(_Q8_SUFFIX)]
    if not scales:
        return sd
    for sk in scales:
        k = sk[: -len(_Q8_SUFFIX)]
        sd[k] = sd[k].float() * sd[sk].float()
        del sd[sk]
    return sd


def pack_state_dict(sd: dict, fp16: bool = False, int8: bool = False,
                    min_numel: int = 4096, linear: str = "tensor") -> dict:
    """The INVERSE of dequant8, and the only place either packing is implemented.

    build_zip.py (Track 1) and Track_2/build_t2.py both write the archive's checkpoints, and
    before this existed each had its own copy of the fp16 cast.  A second copy of the int8
    format -- which carries a companion key per tensor and a scale granularity that decides
    whether the package is shippable -- is exactly the kind of duplication that drifts, so both
    callers come here and the suffix contract lives in one file with its reader.

    int8 quantizes every floating tensor of at least `min_numel` elements and at least 2 dims,
    symmetric, with an fp16 scale per (out, in) channel pair; everything else falls back to
    fp16.  Measured over 96 val_unseen windows (quant8.py) at 29.0 MB/member against fp16's
    56.9: rel-L2 8.3e-4 of the field for a single member, 3.9e-4 for an eight-member ensemble
    (the quantization noise is independent across members and averages down), against a model
    rel-L2 of 0.077.

    `linear` -- WHAT HAPPENS TO A 2-D WEIGHT, and why it is a parameter (round 24).  The scale
    reduces over `range(2, dim)`, which is right for convs and spectral weights: (out, in) keeps
    a scale per channel pair and shares it over the kernel or the retained modes.  On a 2-D
    Linear weight that range is EMPTY, and `torch.amax(x, dim=())` reduces over EVERY dimension
    (checked on torch 2.2.2), so each Linear got ONE scale for the whole matrix.  No FNO member
    has a 2-D tensor, so every FNO package was unaffected -- but a ViT is 97.5 % Linear, and
    R21 / T2R9 shipped their ViT that way: median weight error 1.33e-2 per Linear, 2.4e-2 at
    the worst (transformer rows carry max/rms up to 10.7).
      "tensor"  the historical behaviour, kept so the archives actually uploaded rebuild
                byte for byte
      "row"     one scale per output row, 7.4e-3 median weight error
      "fp16"    leave 2-D weights in fp16, exactly as free as fp16 is everywhere else

    SCORED, ALL THREE ARE FREE (r24e -- the real pack -> load_any -> dequant8 path, each
    package's own band held fixed): on R21's own twin set d sps -0.001 / -0.005 / -0.004 for
    tensor / row / fp16 with every error metric identical to four decimals, and on the ViT ALONE
    -0.004 / -0.009 / +0.002.  A 1.3e-2 weight error in a Linear does not reach the field.  So
    "tensor" stays the default -- it is the smallest (14.76 MB per ViT against fp16's 29.06) and
    it is what is already on the board -- and the per-tensor scale is a code defect with no
    score cost, not a regression.
    """
    if linear not in ("tensor", "row", "fp16"):
        raise ValueError(f"linear must be 'tensor', 'row' or 'fp16', not {linear!r}")
    if not (fp16 or int8):
        return sd
    out = {}
    for k, v in sd.items():
        if not v.is_floating_point():
            out[k] = v
        elif (int8 and v.numel() >= min_numel and v.dim() >= 2
              and not (v.dim() == 2 and linear == "fp16")):
            # () on a 2-D tensor reduces over ALL dims -- the per-tensor scale "tensor" keeps.
            red = (1,) if (v.dim() == 2 and linear == "row") else tuple(range(2, v.dim()))
            sc = v.abs().amax(dim=red, keepdim=True).clamp(min=1e-12).div(127.0).half()
            out[k] = torch.clamp(torch.round(v / sc.float()), -127, 127).to(torch.int8)
            out[k + _Q8_SUFFIX] = sc
        else:
            out[k] = v.half()
    return out


def load_any(path: str, device: str):
    """Load an FNO **or** a U-Net member, dispatching on the checkpoint's own `arch` field.

    train2.py stores `arch` in every blob it writes, and pre-round-12 checkpoints predate the
    U-Net branch, so a missing key means FNO. This is the loader every member list goes
    through -- the package, fit_band, and the endgame scripts -- because any of them may now be
    handed the mixed list.
    """
    blob = torch.load(path, map_location=device)
    # Unconditional and a no-op on every checkpoint written before --int8 existed.
    blob["model_state_dict"] = dequant8(blob["model_state_dict"])
    arch = blob.get("arch", "fno")
    cfg0 = blob.get("config", {})
    if arch in ("ufno", "afno", "vit"):
        # The three VENDORED non-FNO families. Each is a byte-compatible copy of its
        # models_alt.py definition, so a member of any of them ships without the repo.
        w = blob.get("width", cfg0.get("width", 96))
        nl = blob.get("n_layers", 4)
        vh = blob.get("var_head", True)
        if arch == "vit":
            # build() passes `width` as the ViT's `dim` and `n_layers` as its depth, and sets
            # heads = max(width // 64, 1). Reproduce that mapping exactly.
            m = ViT2d(w, nl, vh, 0, heads=max(w // 64, 1)).to(device)
        else:
            cls = UFNO2d if arch == "ufno" else AFNO2d
            m = cls(w, blob.get("modes_y", 8), blob.get("modes_x", 12), nl, vh).to(device)
        m.load_state_dict(blob["model_state_dict"])
        m.eval()
        m.self_norm = _self_norm_of(blob)
        return m
    if arch.startswith("swin_unet"):
        # Vendored 2026-09-14 so a Swin member ships without the repo. `_swin_member` returns
        # None whenever SWIN_IMPL resolves to the training definition ("auto" inside the
        # repo), and the models_alt branch below then runs exactly as it always has.
        m = _swin_member(blob, device)
        if m is not None:
            m.load_state_dict(blob["model_state_dict"])
            m.eval()
            m.self_norm = _self_norm_of(blob)
            return m
    if arch in FFNO_VARIANTS:
        # Vendored 2026-09-14 under the Swin contract: `_ffno_member` returns None whenever FFNO_IMPL
        # resolves to the training definition, and the models_alt branch below runs as before.
        m = _ffno_member(blob, device)
        if m is not None:
            m.load_state_dict(blob["model_state_dict"])
            m.eval()
            m.self_norm = _self_norm_of(blob)
            if FFNO_PATH is not None:
                m.set_spectral_path(FFNO_PATH)
            return m
    if arch not in ("fno", "unet"):
        # ROUND-15 EXPERIMENTAL FAMILIES (afno / dilres / ufno / vit). They live in
        # `Track_1/unet/models_alt.py`, which is NOT vendored into the submission zip -- so
        # this branch works for the local instruments (compare_v9, gap, ens_v9, kfold) and
        # raises, loudly, inside the evaluation container. That is deliberate: a family only
        # gets vendored once it has won its ensemble test, and until then a package that
        # silently contained one would be a submission that fails at scoring time.
        import importlib, os as _os, sys as _sys
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          "..", "unet"))
        cfg = blob.get("config", {})
        m = importlib.import_module("models_alt").build(
            arch, blob.get("width", cfg.get("width", 96)), blob.get("var_head", True), 0,
            blob.get("modes_y", 8), blob.get("modes_x", 12), blob.get("n_layers", 4),
            dropout=cfg.get("dropout", 0.0)).to(device)
        m.load_state_dict(blob["model_state_dict"])
        m.eval()
        m.self_norm = _self_norm_of(blob)
        if FFNO_PATH is not None and arch in FFNO_VARIANTS:
            m.set_spectral_path(FFNO_PATH)
        return m
    if arch != "unet":
        return _fno_from_blob(blob, device)
    # extra_ch is 0 by construction: `--hist` and `--ctx > 20` both feed the trunk information
    # from OUTSIDE the 20-frame window, which predict() never has and which would make the
    # submission order-dependent. Refuse rather than build a model nothing can drive -- the
    # strict load below would fail anyway, but on an unreadable shape error.
    assert not blob.get("hist", False) and blob.get("ctx", T_IN) == T_IN, \
        f"{path}: --hist/--ctx checkpoints are not deployable in predict()"
    m = UNetTwoHead(blob.get("width", 64), blob.get("var_head", True)).to(device)
    m.load_state_dict(blob["model_state_dict"])
    m.eval()
    m.self_norm = _self_norm_of(blob)
    return m


@torch.no_grad()
def predict_fno(model, x_raw: np.ndarray, device: str, alpha: float = 0.75,
                batch: int = 64, out: np.ndarray | None = None,
                sigma_out: np.ndarray | None = None,
                pad_batch: bool = _PAD_BATCH, amp=_AMP,
                centre_out: np.ndarray | None = None) -> np.ndarray:
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
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    # A `--norm self` member divides each window by its own mean streamwise speed before the
    # standardization, so its affine is per-WINDOW. Mixing the two conventions in one ensemble
    # would average fields that were never in the same space; refuse rather than average.
    sn = [bool(getattr(m, "self_norm", False)) for m in models]
    if any(sn) and not all(sn):
        raise ValueError("predict_fno: cannot mix --norm self and --norm real members")
    sn = sn[0] if sn else False
    mean = mean_c = torch.from_numpy(MEAN_SN if sn else MEAN).to(device)
    std = std_c = torch.from_numpy(STD_SN if sn else STD).to(device)
    n = len(x_raw)
    dst = np.empty((n, T_OUT, H, W, C), dtype=np.float32) if out is None else out
    cuda = str(device).startswith("cuda")
    for i in range(0, n, batch):
        xb = torch.from_numpy(x_raw[i:i + batch]).to(device, non_blocking=True)[..., :2].float()
        # Fixed batch SHAPE (opt-in).  Every batch but the last is already `batch` rows; the
        # tail is padded with copies of its own last window so cuDNN/cuBLAS cannot select a
        # different kernel for it.  The copies are discarded after the members are averaged,
        # below, so nothing they touch reaches the output.
        nb = len(xb)
        if pad_batch and nb < batch:
            xb = torch.cat([xb, xb[-1:].expand(batch - nb, *xb.shape[1:])], 0)
        # `mean`/`std` are (2,) and broadcast over the channel axis. Under self-normalization
        # they additionally carry the window's own scale, which makes them (B,1,1,1,2) -- the
        # SAME two lines then serve both conventions, in and out, including sigma. `s` is only
        # computed when it is needed, so the scored path is untouched for a `--norm real`
        # package (verified bit-identical).
        if sn:
            s = xb[..., 0].mean(dim=(1, 2, 3)).clamp(min=1e-3).reshape(-1, 1, 1, 1, 1)
            mean, std = mean_c * s, std_c * s
        prev = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), CH, H, W)

        # Members are averaged HERE, on the GPU, in raw field and sigma space -- before the
        # fluctuation rescaling and before the transfer to host. Two consequences, both
        # deliberate: compose() runs once on a single coherent field rather than being averaged
        # across differently-rescaled ones, and K members cost K forward passes but still only
        # ONE host transfer and one output write. At 0.109 ms of compute against ~1.27 ms of
        # fixed API overhead, that is what makes ensembling affordable at all.
        y = sig = None
        total = 0
        for m in models:
            # A StackedFNO already averaged over the members it fused, so it stands in for
            # `n_members` of them and must be weighted accordingly -- otherwise fusing would
            # silently change the ensemble's mixture.
            nm = getattr(m, "n_members", 1)
            total += nm
            if amp is None or not cuda:
                raw, logstd = m(prev)
            else:
                # Only the network runs reduced-precision; everything after this -- the member
                # average, the fluctuation rescaling, the denormalization -- stays fp32, so the
                # variant differs from the reference in exactly one place.
                with torch.autocast("cuda", dtype=amp):
                    raw, logstd = m(prev)
                raw = raw.float()
                logstd = None if logstd is None else logstd.float()
            yi = raw.reshape(len(xb), T_OUT, C, H, W).permute(0, 1, 3, 4, 2) * nm
            y = yi if y is None else y + yi
            if logstd is not None:
                si = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
                sig = si if sig is None else sig + si
        if total > 1:
            y = y / total
            if sig is not None:
                sig = sig / total
        if pad_batch and len(y) != nb:
            # Drop the padding here, before compose and before the write, so the tail costs
            # only the forward pass it was padded for.  Under `--norm self` the affine is
            # per-window and has to be sliced with it.
            y = y[:nb]
            sig = None if sig is None else sig[:nb]
            if sn:
                mean, std = mean[:nb], std[:nb]

        if centre_out is not None:
            # The member average BEFORE compose, physical units: the SPS interval centre when
            # it is decoupled from the point prediction (Bram, 2026-09-21: centring the band on
            # the un-rescaled mean is +0.61 sps held out, 8/8 trajectories). compose is linear
            # in alpha, so any other centre is y0 + (a_c/alpha)*(pred - y0) on the host.
            c_ = (y * std + mean).float().cpu().numpy()
            if _HOSTW == "torch":
                torch.from_numpy(centre_out)[i:i + batch, ..., :2].copy_(torch.from_numpy(c_))
            else:
                centre_out[i:i + batch, ..., :2] = c_
        # `alpha != 0`, not `> 0`: a NEGATIVE alpha is a legitimate (extra-shrunk) interval centre
        # that fit_band.py --centre-alpha asks for; alpha == 0 stays the untouched member mean.
        if sig is not None and alpha != 0:
            m_ = y.mean(1, keepdim=True)
            f_ = y - m_
            s_ = y.std(1, keepdim=True, unbiased=False)
            y = m_ + f_ * ((alpha * sig + (1 - alpha) * s_) / (s_ + 1e-6))
        if sigma_out is not None and sig is not None:
            # Physical units, (B, 1, H, W, 2). The SPS band is scaled by this: predicted sigma
            # correlates 0.551 with per-element |error| after the per-element mean is removed,
            # against 0.482 for ensemble spread, so it says which WINDOWS need a wider
            # interval. Costs one small copy -- 5.5 MB at N=336 against the field's 165 MB.
            sigma_out[i:i + batch] = (sig * std).float().cpu().numpy()
        hb = (y * std + mean).float().cpu().numpy()
        if _HOSTW == "torch":
            # `torch.from_numpy` is a view, not a copy, and copy_ between two float32 tensors
            # of the same shape is exact -- so this is bit-identical, only threaded.
            torch.from_numpy(dst)[i:i + batch, ..., :2].copy_(torch.from_numpy(hb))
        else:
            dst[i:i + batch, ..., :2] = hb
    return dst
