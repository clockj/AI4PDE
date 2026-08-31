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

import numpy as np
import torch
import torch.nn as nn

T_IN = T_OUT = 20
# Oscillatory-head frequency band, mirrored from unet/models_alt.py: periods of
# 200 down to 10 frames, bracketing the 60-75 frame shedding period.
OSC_W_LO, OSC_W_HI = 2.0 * math.pi / 200.0, 2.0 * math.pi / 10.0
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
                 extra_ch=0, var_mlp=False, lift_mlp=False, point_mlp=False,
                 var_hidden=128, var_deep=False, var_in_std=False,
                 var_pred_std=False, fgn_dim=0, mean_head=False, osc_modes=0,
                 osc_global=False, osc_zeromean=False):
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
        h0 = first.head_logstd
        if h0 is None:
            self.head_logstd = None
        elif isinstance(h0, nn.Sequential):
            self.head_logstd = nn.Sequential(grouped([m.head_logstd[0] for m in members]),
                                             nn.GELU(),
                                             grouped([m.head_logstd[2] for m in members]))
        else:
            self.head_logstd = grouped([m.head_logstd for m in members])

    def _spec(self, i, x):
        b, _, h, w = x.shape
        k, wid = self.n_members, self.width
        ft = torch.fft.rfft2(x).reshape(b, k, wid, h, w // 2 + 1)
        out = torch.zeros(b, k, wid, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        my, mx = min(self.my, h // 2), min(self.mx, w // 2 + 1)
        for sl, wp in ((slice(None, my), self.w1), (slice(-my, None), self.w2)):
            wc = torch.view_as_complex(wp[i])[:, :, :, :my, :mx]
            out[:, :, :, sl, :mx] = torch.einsum("bkixy,kioxy->bkoxy",
                                                 ft[:, :, :, sl, :mx], wc)
        return torch.fft.irfft2(out.reshape(b, k * wid, h, w // 2 + 1), s=(h, w))

    def forward(self, prev, detach_var: bool = False):
        b, k = len(prev), self.n_members
        x = torch.cat([prev, self.grid.expand(b, -1, -1, -1)], 1)
        h = self.lift(x)                    # first lift conv is shared, see `shared()`
        hs = []
        for i in range(self.n_layers):
            h = h + torch.nn.functional.gelu(self.norm[i](self._spec(i, h) + self.point[i](h)))
            hs.append(h)
        hw = h.shape[-2:]
        f_all = self.head_field(h).reshape(b, k, CH, *hw)      # per member, before averaging
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


def fuse(models):
    """Group `models` by architecture and fuse each FNO group. Order-preserving by first sight.

    A package of 5x width-24 + w48 + m1220 + w64 goes from 8 invocations to 4, and the round-12
    mixed package (8 FNO + 1 U-Net) from 9 to 4. Only FNO2d members are ever folded: StackedFNO
    knows how to turn a member axis into `groups` for spectral blocks and 1x1 convs, and nothing
    else. A U-Net's state_dict keys differ from every FNO's, so it lands in its own group and is
    handed back untouched -- `predict_fno` then calls it as one more (field, logstd) module.
    """
    groups = {}
    for m in models:
        # Keyed on the full parameter signature rather than a few attributes, so any
        # architecture flag (var_mlp, lift_mlp, ...) separates groups automatically. The
        # spectral weights are excluded: StackedFNO zero-pads differing mode counts, which is
        # exact, so a modes-(12,20) member joins its width-matched (8,12) siblings instead of
        # buying its own invocation. Width and every other shape still separate, because they
        # appear in lift / point / norm / the heads.
        key = tuple((k, tuple(v.shape)) for k, v in m.state_dict().items()
                    if not (k.endswith(".w1") or k.endswith(".w2")))
        groups.setdefault(key, []).append(m)
    out = []
    for g in groups.values():
        # A member with a decoupled time-mean head is passed through UNFUSED. StackedFNO
        # enumerates lift / point / norm / head_field / head_logstd explicitly, so a
        # `head_mean` would be silently DROPPED and the member would emit a zero-mean field.
        # Losing an invocation costs latency; losing a head is a wrong answer, and it would
        # not show up as a crash.
        if (len(g) == 1 or not isinstance(g[0], FNO2d)
                or g[0].head_mean is not None or g[0].head_osc is not None):
            out.extend(g)     # `extend`, not `append`: two U-Nets are two invocations, not one
        else:
            out.append(StackedFNO(g).to(g[0].grid.device).eval())
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
              osc_zeromean=cfg("osc_zeromean", False)).to(device)
    m.load_state_dict(blob["model_state_dict"])
    m.eval()
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


def load_any(path: str, device: str):
    """Load an FNO **or** a U-Net member, dispatching on the checkpoint's own `arch` field.

    train2.py stores `arch` in every blob it writes, and pre-round-12 checkpoints predate the
    U-Net branch, so a missing key means FNO. This is the loader every member list goes
    through -- the package, fit_band, and the endgame scripts -- because any of them may now be
    handed the mixed list.
    """
    blob = torch.load(path, map_location=device)
    if blob.get("arch", "fno") != "unet":
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
    return m


@torch.no_grad()
def predict_fno(model, x_raw: np.ndarray, device: str, alpha: float = 0.75,
                batch: int = 64, out: np.ndarray | None = None,
                sigma_out: np.ndarray | None = None) -> np.ndarray:
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
        total = 0
        for m in models:
            # A StackedFNO already averaged over the members it fused, so it stands in for
            # `n_members` of them and must be weighted accordingly -- otherwise fusing would
            # silently change the ensemble's mixture.
            nm = getattr(m, "n_members", 1)
            total += nm
            raw, logstd = m(prev)
            yi = raw.reshape(len(xb), T_OUT, C, H, W).permute(0, 1, 3, 4, 2) * nm
            y = yi if y is None else y + yi
            if logstd is not None:
                si = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
                sig = si if sig is None else sig + si
        if total > 1:
            y = y / total
            if sig is not None:
                sig = sig / total

        if sig is not None and alpha > 0:
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
        dst[i:i + batch, ..., :2] = (y * std + mean).float().cpu().numpy()
    return dst
