"""FNO and DeepONet, parameterized for a size sweep, drop-in for `UNetTwoHead`.

Both expose the same interface as `model.UNetTwoHead`:

    forward(prev: (B, CH_in, H, W), detach_var: bool) -> (field (B, CH, H, W), logstd or None)

so `train2.py`, `eval_wp.run` and `eval_pair.py` work against them unchanged and every
comparison against the U-Net is on identical footing -- same normalization, same windowing,
same variance composition, same SPS band, same timing method.

Time is folded into the channel axis exactly as in `model.py`: a 20-frame window of 2 measured
channels is 40 channels of a 32x64 image. That makes the FNO a 2-D operator over the spatial
domain with the horizon carried in channels, which is what the official baseline does too.

2026-09-28: the families that ship or are still re-trained live HERE -- `FFNO2d` (+ `_FFNOLayer`)
and `ViT2d`, the two member families of the final packages, and `FNO2d` (+ `SpectralConv2d`),
which `reproduce.sh r25` trains -- with the shared stages (`_GridIn`, `_heads`, `_LayerNorm2d`),
the physics helpers train2.py imports, and the two factories `build` / `build_wave1`. The CLOSED
families (DeepONet, AFNO, DilResNet, U-FNO, ConvNeXt U-Net, PDE-Arena U-Net, the Swin U-Net,
SimVP, the space-time ViT) and `oracle_cond_features` moved, verbatim, to `models_closed.py`.
This module stays the single import surface: the bottom of the file re-exports every moved name
(`_CLOSED_NAMES`), so `models_alt.SwinUNet`, `from models_alt import _win_part` and
`build("afno")` behave exactly as before. Class bodies, defaults and parameter creation order
are unchanged in both files, so a seeded build draws the same numbers.
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
                 tbasis: int = 0, mode_drop: float = 0.0, p_head: bool = False,
                 sub_head: bool = False, cond_dim: int = 0):
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

        # A TRAINING-ONLY auxiliary PRESSURE readout (round 17 Q3j). Sim has `p` and real does
        # not -- it is zero-filled in every real file and never scored -- so this head can only
        # ever be supervised on sim windows, and it is stripped from the checkpoint at save
        # time (train2.py) so the shipped `_fno.py` loads the blob unchanged with strict=True.
        #
        # A single 1x1 conv, NOT head_field's MLP, and the cheapness is not the only reason.
        # The point of an auxiliary task is to shape the TRUNK; a linear readout can only
        # succeed if the trunk itself carries p linearly, so it pushes the work where it is
        # wanted. (--var-mlp's lesson -- that a linear probe underperforms an MLP -- was about
        # a head whose OUTPUT is deployed. This one's output is thrown away.) 1,940 parameters
        # at width 96 against 14,848 for the MLP shape.
        #
        # CONSTRUCTED LAST, after every other submodule, so that at a fixed --seed a
        # p_head=True model draws the SAME initialization for the trunk and both deployed
        # heads as its p_head=False control: head_p only appends to the RNG stream instead
        # of shifting it. Without this the variance head's init differs between an arm and
        # its control, which is the one thing --seed exists to prevent.
        self.head_p = nn.Conv2d(width, T_OUT, 1) if p_head else None

        # --subpix: a TRAINING-ONLY head that predicts the target at the THREE decimation
        # phases the scored grid throws away. model.load_trajectories reads the raw 64x128
        # frames and slices [py::2, px::2]; phase (0, 0) is what is scored and (0,1)/(1,0)/(1,1)
        # are the SAME PIV measurement sampled half a coarse cell over -- real flow, not
        # synthetic perturbation.
        #
        # WHY THIS IS NOT --phases 4, which is null twice (round 17 Q2b at 400 epochs and at
        # matched optimizer steps). --phases 4 makes four INDEPENDENT tasks: predict phase p
        # from phase p. This makes ONE task with a 4x denser target: predict all four phases
        # from phase 0. That is a super-resolution constraint -- the trunk has to represent the
        # field BETWEEN the observed grid points -- and no data-side arm has ever imposed it.
        #
        # Stripped from the state_dict at save (train2.py), like head_p, so the shipped blob
        # is a plain FNO2d and inference is bit-identical to a --subpix 0 member. CONSTRUCTED
        # LAST for the same reason head_p is: at a fixed --seed an arm and its control must
        # draw the same initialization for the trunk and the two deployed heads.
        self.head_sub = nn.Conv2d(width, 3 * T_OUT * C, 1) if sub_head else None

        # --oracle-cond (2026-09-14 wave 2): a DIAGNOSTIC ONLY, never shipped -- predict() is not
        # given the condition. Per block, a FiLM on the normalized activation driven by the TRUE
        # (Re, AoA) of the window (`oracle_cond_features`), so the arm measures how much error
        # knowing the condition removes. The last Linear is zero-initialised (weight AND bias), so
        # training starts at the plain FNO. CONSTRUCTED LAST, like head_p / head_sub, so at a
        # fixed --seed the trunk and both heads draw the same initialisation as the cond_dim = 0
        # control. cond_dim = 0 builds nothing and the forward below is unchanged.
        self.cond_dim = cond_dim
        self.cond_film = None
        if cond_dim > 0:
            self.cond_film = nn.ModuleList([
                nn.Sequential(nn.Linear(cond_dim, 64), nn.GELU(), nn.Linear(64, 2 * width))
                for _ in range(n_layers)])
            for f in self.cond_film:
                nn.init.zeros_(f[-1].weight)
                nn.init.zeros_(f[-1].bias)

    def forward(self, prev: torch.Tensor, detach_var: bool = False,
                z: torch.Tensor | None = None, return_p: bool = False,
                return_sub: bool = False, cond: torch.Tensor | None = None):
        """`z` is the FGN noise vector, (B, fgn_dim). None (or fgn_dim = 0) is the plain FNO.

        `return_p` appends the auxiliary pressure prediction, (B, T_OUT, H, W), read off the
        SAME trunk features `head_field` reads. Default False, so every existing caller gets
        the same 2-tuple it always did and no other file needs to change.
        """
        h = self.lift(torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1))
        # One (scale, shift) pair per block, computed once outside the loop: it is a
        # (B, fgn_dim) x (fgn_dim, 2*width) matmul against a loop body that is spectral.
        films = ([f(z).chunk(2, -1) for f in self.film]
                 if z is not None and self.film is not None else None)
        # --oracle-cond: (B, cond_dim) true-condition features -> one (scale, shift) per block.
        cfilms = ([f(cond).chunk(2, -1) for f in self.cond_film]
                  if cond is not None and self.cond_film is not None else None)
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
            if cfilms is not None:
                cg, cb = cfilms[i]
                a = (1.0 + cg[:, :, None, None]) * a + cb[:, :, None, None]
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
        if return_sub:
            return field, logstd, (self.head_sub(h) if self.head_sub is not None else None)
        if return_p:
            return field, logstd, (self.head_p(h) if self.head_p is not None else None)
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
          osc_zeromean: bool = False, tbasis: int = 0, mode_drop: float = 0.0,
          p_head: bool = False, sub_head: bool = False, cond_dim: int = 0,
          drop_path: float | None = None, err_head: bool = False,
          err_detach: bool = False):
    """Factory shared by train2.py and eval_wp.run so both build identical objects.

    `fgn_dim` defaults to 0, which builds no `film` module at all -- so a state_dict saved
    before FGN existed still loads with strict=True and every pre-round-12 checkpoint is
    unaffected.

    `err_head` defaults to False and adds no module, so the same sentence holds for it: a
    checkpoint written without --err-head is byte-identical in structure to what this factory
    built yesterday. It is F-FNO-only and refuses loudly anywhere else, because the head is
    built by `_heads` and only FFNO2d's forward knows how to return it.
    """
    from model import UNetTwoHead
    if cond_dim and arch != "fno":
        raise ValueError(f"cond_dim (the --oracle-cond diagnostic) is FNO-only, not {arch}")
    if err_head and arch not in ("ffno", "ffno_nff"):
        raise ValueError(f"err_head (--err-head) is F-FNO-only, not {arch}")
    if drop_path is not None and not (arch.startswith("swin_unet") or arch in ("simvp", "stvit")):
        raise ValueError(f"drop_path is not a knob of {arch}")
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
                     osc_zeromean, tbasis, mode_drop, p_head=p_head,
                     sub_head=sub_head, cond_dim=cond_dim)
    if arch == "deeponet":
        return DeepONetCh(p, trunk_width=width * 2, branch_width=width,
                          var_head=var_head, extra_ch=extra_ch)
    if arch in WAVE1_ARCHS or arch in PHASE2_ARCHS:
        # 2026-09-14 wave 1 (the block at the end of this file). Each family is fully specified
        # by (width, n_layers, modes_y, modes_x) plus class constants, because _fno.load_any's
        # experimental branch rebuilds a checkpoint from exactly those blob fields.
        return build_wave1(arch, width, var_head, extra_ch, modes_y, modes_x, n_layers,
                           dropout=dropout, drop_path=drop_path, err_head=err_head,
                           err_detach=err_detach)
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


def _heads(width: int, var_head: bool, err_head: bool = False):
    """The two readouts FNO2d uses, so head capacity is not a confound between families.

    A 128-wide MLP on each. `--var-mlp` and `--lift-mlp` were each worth +0.08 on FNO2d
    precisely because its defaults were linear 1x1 probes; starting the new families at the
    linear default would re-run that experiment inside every one of them.

    `err_head` (train2 --err-head, F-FNO only for now) adds a THIRD readout, `head_logerr`,
    and is the only thing that changes the return arity: with it off this returns the same
    two-tuple every family has always unpacked, so no other call site moves. The error head
    emits CH channels, not C -- the same shape as the field output, so it carries the FRAME
    axis. That is the whole point of it: `head_logstd` regresses onto std_t(y), the target
    window's per-pixel TEMPORAL std, which is one envelope shared by all 20 forecast frames
    and is never a function of what the model got wrong. `head_logerr` is trained on the
    model's OWN |pred - y| per element per frame (train2.py, next to the l_var block).
    Init mirrors head_logstd exactly -- zero weights, bias log(0.3) -- so the head starts as
    a constant and its gradient cannot perturb the field head's first steps. It is built LAST
    so the draws it consumes come after every other parameter's, leaving the trunk's and the
    field head's initialization bit-identical to a run without it. FFNO2d.forward then feeds
    it DETACHED trunk features unconditionally; see the note there.
    """
    head_field = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, CH, 1))
    head_logstd = None
    if var_head:
        head_logstd = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, C, 1))
        nn.init.zeros_(head_logstd[-1].weight)
        nn.init.constant_(head_logstd[-1].bias, float(np.log(0.3)))
    if not err_head:
        return head_field, head_logstd
    head_logerr = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, CH, 1))
    nn.init.zeros_(head_logerr[-1].weight)
    nn.init.constant_(head_logerr[-1].bias, float(np.log(0.3)))
    return head_field, head_logstd, head_logerr


class _GridIn(nn.Module):
    """Shared input stage: concat the (x, y) coordinate grid, exactly as FNO2d does."""

    def __init__(self):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                                indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None])

    def cat(self, prev: torch.Tensor) -> torch.Tensor:
        return torch.cat([prev, self.grid.expand(len(prev), -1, -1, -1)], 1)


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


# ===========================================================================
# 2026-09-14 campaign, WAVE 1 (Track_1/plan/plan_20260914.md section 4): six backbone families,
# each screened against a fresh FNO control under ONE recipe (cmb3sim minus sim, held out).
#
# CONTRACT.  Every class keeps FNO2d's interface --
#     forward(prev (B, 40 + extra, 32, 64) normalized time-major, detach_var)
#         -> (field (B, 40, 32, 64), logstd (B, 2, 32, 64) or None)
# -- so train2.py, _fno.load_any's experimental branch, _fno.predict_fno and the v9 instruments
# (compare_v9 / ens_v9 / fit_band / predict_board) drive them unchanged. `_fno.load_any` rebuilds a
# checkpoint from (arch, width, var_head, modes_y, modes_x, n_layers) ALONE, so each family is
# fully specified by those flags plus the class constants below; a variant that needs another knob
# gets its own arch name (e.g. `ffno_nff`), never an extra flag.
#
# SHARED RULES.
#   * The (x, y) coordinate grid is concatenated to the input exactly as FNO2d does. No domain
#     padding (FNO2d adds none): convolutions zero-pad and shifted attention windows mask -- except
#     `ffno`, whose separable FFTs get F-FNO's far-side zero padding (FFNO2d.PAD).
#   * The sigma head SHARES the trunk and reads it through a 128-wide MLP whose last layer starts
#     at log(0.3), as `_heads` does for every earlier family.
#   * `dropout` is Dropout2d on each residual branch for the conv and spectral families (FNO2d's
#     placement), and attention + MLP dropout with a 0.1 linear drop-path for the transformers
#     (ViT2d's placement).
#   * Spectral ops run in fp32 under autocast (`_fno._fft_in`'s rule); everything else may run
#     fp16. Layer-scale / position tensors are cast to the activation dtype so fp16 stays fp16.
# Sizes, the latency table and the screen protocol live in Track_1/plan/screen_20260914.md.
# ===========================================================================
import torch.nn.functional as F  # noqa: E402

WAVE1_ARCHS = ("cnext_unet", "unet_mod", "ffno", "ffno_nff", "swin_unet", "simvp", "stvit")
# Phase-2 Swin ladder (2026-09-14): structural variants, one knob each (see SwinUNet.__init__).
PHASE2_ARCHS = ("swin_unet_b6", "swin_unet_win16", "swin_unet_p1")


class _LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the CHANNEL axis of an NCHW map (ConvNeXt's data_format='channels_first')."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------------------------
# 3. ffno / ffno_nff -- Factorized FNO (Tran, Mathews, Xie & Ong, ICLR 2023)
# ---------------------------------------------------------------------------------------------
class _FFNOLayer(nn.Module):
    """One F-FNO layer. The Fourier operator is SEPARABLE: rfft along y, one complex matrix per
    retained ky, irfft; the same along x; the two are summed. Parameters go as
    2 * w^2 * (modes_y + modes_x) instead of FNO's 2 * w^2 * modes_y * modes_x, which is what buys
    the deeper stack and the (near-)full mode count.

    ff=True  (`ffno`)     x + Dropout2d(LayerNorm(FF(K(x)))), FF = Linear k*w -> GELU -> Linear:
                          F-FNO's layer with its feed-forward after the operator and the residual
                          around the whole thing (its `layer_norm=True` variant, no weight norm).
    ff=False (`ffno_nff`) x + Dropout2d(GELU(GroupNorm(K(x) + W x))): FNO2d's block with only the
                          spectral operator factorised -- the per-layer FF switched off.
    """

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
        # FFNO2d.set_spectral_path("dense"): the FFT-free spectral path (off = the as-trained FFTs).
        self.dense = False
        self._bases: dict = {}

    @staticmethod
    def _dense_basis(n: int, m: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Real (n, 2m) analysis and (2m, n) synthesis matrices that reproduce rfft(norm="ortho")[:m]
        and irfft(n=n, norm="ortho") of a length-m half spectrum. They are EXTRACTED, in float64,
        by applying those very ops to unit vectors -- rows e_j through rfft give the analysis columns;
        e_k and i*e_k through irfft give the synthesis rows of the real and imaginary parts -- so no
        FFT sign or normalisation convention is re-derived by hand."""
        a = torch.fft.rfft(torch.eye(n, dtype=torch.float64), dim=-1, norm="ortho")[:, :m]
        e = torch.eye(m, dtype=torch.complex128)
        s_re = torch.fft.irfft(e, n=n, dim=-1, norm="ortho")
        s_im = torch.fft.irfft(1j * e, n=n, dim=-1, norm="ortho")
        return (torch.cat([a.real, a.imag], 1).to(device=device, dtype=torch.float32),
                torch.cat([s_re, s_im], 0).to(device=device, dtype=torch.float32))

    @staticmethod
    def _w2(wt: torch.Tensor) -> torch.Tensor:
        """(in, out, m, 2) weights viewed as complex -> (m, 2*out, 2*in) real 2x2-block matrices
        [[Re W, -Im W], [Im W, Re W]]: complex mixing written as a real matmul."""
        wr = wt[..., 0].permute(2, 1, 0)
        wi = wt[..., 1].permute(2, 1, 0)
        return torch.cat([torch.cat([wr, -wi], 2), torch.cat([wi, wr], 2)], 1)

    def _spectral_dense(self, x: torch.Tensor) -> torch.Tensor:
        """The separable operator with every 1-D FFT replaced by a dense real projection onto the
        precomputed cos/sin basis: analysis, 2x2-block mixing and synthesis are three matmuls per
        axis, which autocast runs in fp16 (the FFT path is pinned to fp32, and its casts are copies).
        Equal to `_spectral` up to float rounding (checked <= 1e-6 in fp32)."""
        b, c, h, w = x.shape
        key = (x.device, h, w)
        if key not in self._bases:
            self._bases[key] = (self._dense_basis(h, self.my, x.device),
                                self._dense_basis(w, self.mx, x.device))
        (py, sy), (px, sx) = self._bases[key]
        my, mx = self.my, self.mx
        # y axis: analysis over H, then per retained mode k one 2C -> 2C' real matmul applied to
        # every (b, w) position at once: (my, B*W, 2C) bmm (my, 2C, 2C') -- [Re; Im] channel blocks.
        cy = torch.matmul(x.transpose(-1, -2), py)                            # (B, C, W, 2my)
        cy = cy.reshape(b, c, w, 2, my).permute(4, 0, 2, 3, 1).reshape(my, b * w, 2 * c)
        dy = torch.bmm(cy, self._w2(self.wy).transpose(1, 2))                  # (my, B*W, 2C')
        co = dy.shape[-1] // 2
        dy = dy.reshape(my, b, w, 2, co).permute(1, 4, 2, 3, 0).reshape(b, co, w, 2 * my)
        yy = torch.matmul(dy, sy).transpose(-1, -2)                            # (B, C', H, W)
        # x axis: the same over W, with (b, h) positions batched per mode.
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
    """Factorized FNO: lift MLP (42 -> 128 -> w), zero-pad the far side of both axes by PAD cells
    (F-FNO's treatment of a NON-periodic domain, which its separable FFTs would otherwise wrap;
    scout's copy notes), `n_layers` F-FNO layers with FF expansion FF_FACTOR, crop, LayerNorm,
    MLP heads. Scored only as an FNO-MEMBER replacement (same family, closed capacity axes)."""

    PAD = (4, 8)
    FF_FACTOR = 2

    def __init__(self, width: int = 64, modes_y: int = 16, modes_x: int = 32,
                 n_layers: int = 12, var_head: bool = True, extra_ch: int = 0,
                 dropout: float = 0.0, ff: bool = True, err_head: bool = False,
                 err_detach: bool = False):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv2d(CH + extra_ch + 2, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, width, 1))
        grid = (H + self.PAD[0], W + self.PAD[1])
        self.layers = nn.ModuleList([_FFNOLayer(width, modes_y, modes_x, ff, dropout,
                                                self.FF_FACTOR, grid)
                                     for _ in range(n_layers)])
        self.norm_out = _LayerNorm2d(width)
        # `err_head` off -> the two-tuple every other family builds, and forward() keeps its
        # historical (field, logstd) contract, so nothing downstream sees this flag exist.
        self.head_logerr = None
        self.err_detach = bool(err_detach)
        if err_head:
            self.head_field, self.head_logstd, self.head_logerr = _heads(width, var_head, True)
        else:
            self.head_field, self.head_logstd = _heads(width, var_head)

    def set_spectral_path(self, path: str = "fft"):
        """'fft' (the default and the as-trained path) or 'dense' (`_FFNOLayer._spectral_dense`:
        the 1-D FFTs as dense real matmuls on a precomputed basis, fp16-native). Changes no
        parameter, so a checkpoint loads the same either way."""
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
        if self.head_logerr is None:
            return self.head_field(h), logstd
        # --err-head only: (field, logstd, logerr), logerr (B, CH, H, W) -- the field's own
        # shape, frame axis included. The arity changes ONLY when the head exists, so every
        # `raw, logstd = model(...)` site in the project keeps working on every checkpoint
        # trained before today.
        #
        # ALWAYS `.detach()`, not `if detach_var` -- the one place this head deliberately does
        # NOT mirror head_logstd. It is the mechanism head_logstd's `detach_var` provides,
        # applied unconditionally, and it is what makes the arm READABLE: with the trunk cut,
        # this head adds parameters and a loss term but cannot change one gradient the field
        # head receives, so an --err-head run and its control at the same --seed train the
        # identical trunk and emit the identical field. Anything that then moves in the
        # confidence band is the head, not an accuracy change. Routing it through detach_var
        # instead would have priced the two together: enabling that flag also cuts the sigma
        # head loose, which is a different recipe from the shipped twin (REPRODUCE.md 1.5) and
        # would move the field on its own.
        # CHANGED 2026-09-23 after grepping ROUNDS.md for `detach`, which I should have done
        # before writing any of this. ROUNDS.md:377-380, measured on the closest analogue:
        #   "The variance head must be allowed to reshape the shared trunk. Detached it reads
        #    the mean-flow latent and reaches corr 0.679 -- worse than copying the input
        #    window's std (0.756). Shared it reaches corr 0.907. Sharing the representation is
        #    necessary; SHARING THE GRADIENT is what makes it work."
        # An error head has it worse still: detached, it must predict where THIS MODEL is wrong
        # using only features the field task had a reason to build, and the field task has no
        # reason to encode its own failures. So the default is now the SHARED gradient, and the
        # bitwise ARM/CONTROL fingerprint is given up deliberately -- it was bought with the one
        # switch the repo had already measured as breaking the head.
        logerr = self.head_logerr(h.detach() if self.err_detach else h)
        return self.head_field(h), logstd, logerr


def build_wave1(arch: str, width: int, var_head: bool, extra_ch: int, modes_y: int,
                modes_x: int, n_layers: int, dropout: float = 0.0,
                drop_path: float | None = None, err_head: bool = False,
          err_detach: bool = False):
    """Wave-1 / Phase-2 factory, reached from `build`. Only (width, n_layers, modes_y, modes_x)
    vary per checkpoint; `drop_path` (train2 --drop-path) is a training-time rate that leaves no
    trace in the state_dict, so a loader passing None rebuilds every checkpoint unchanged.

    `err_head` is NOT of that kind: it adds `head_logerr.*` tensors, so a loader that wants an
    --err-head checkpoint must pass it (the blob records it under the top-level "err_head" key
    train2.py writes next to "var_head")."""
    kw = {} if drop_path is None else {"drop_path": float(drop_path)}
    if kw and not (arch.startswith("swin_unet") or arch in ("simvp", "stvit")):
        raise ValueError(f"drop_path is not a knob of {arch}")
    if err_head and arch not in ("ffno", "ffno_nff"):
        raise ValueError(f"err_head is not a knob of {arch}")
    if arch == "cnext_unet":
        return CNeXtUNet(width, n_layers, var_head, extra_ch, dropout)
    if arch == "unet_mod":
        return UNetMod(width, n_layers, var_head, extra_ch, dropout)
    if arch in ("ffno", "ffno_nff"):
        return FFNO2d(width, modes_y, modes_x, n_layers, var_head, extra_ch, dropout,
                      ff=(arch == "ffno"), err_head=err_head, err_detach=err_detach)
    if arch == "swin_unet":
        return SwinUNet(width, n_layers, var_head, extra_ch, dropout, **kw)
    if arch == "swin_unet_b6":
        return SwinUNet(width, n_layers, var_head, extra_ch, dropout, bottleneck_depth=6, **kw)
    if arch == "swin_unet_win16":
        return SwinUNet(width, n_layers, var_head, extra_ch, dropout, window=16, **kw)
    if arch == "swin_unet_p1":
        return SwinUNet(width, n_layers, var_head, extra_ch, dropout, patch=1, **kw)
    if arch == "simvp":
        return SimVP2(width, n_layers, var_head, extra_ch, dropout, **kw)
    if arch == "stvit":
        return STViT(width, n_layers, var_head, extra_ch, dropout, **kw)
    raise ValueError(arch)


# ===========================================================================
# 2026-09-28: re-export of the closed families (models_closed.py), so that every
# `models_alt.<name>` and `from models_alt import <name>` in the repo keeps resolving, and so
# that `build` / `build_wave1` above find the moved classes in THIS module's globals exactly
# as before. The two modules import each other, so the copy is done by a function both sides
# can call:
#   * `import models_alt` first: the import below runs models_closed to completion, and
#     its last line calls `_reexport_closed` on this (partially initialised) module;
#   * `import models_closed` first: its `from models_alt import ...` runs this file, the
#     import below then returns the half-built models_closed (no `_LOADED` yet), and
#     models_closed fills this module in on its own last line;
#   * this file exec'd under a spec that is not in sys.modules (verify/fuse_osc.py):
#     models_closed fills the registered copy, and the guarded import below fills this one.
# ===========================================================================
_CLOSED_NAMES = (
    "DeepONetCh", "AFNO2d", "DilResNet", "UFNO2d", "_drop_path", "_trunc_init", "_var_mlp",
    "_ConvNeXtBlock", "CNeXtUNet", "_PreActResBlock", "UNetMod", "_win_part", "_win_merge",
    "_WindowAttnV2", "_SwinV2Block", "_PatchMergeV2", "_PatchExpand", "_TokenProj", "SwinUNet",
    "_ConvSC", "_GASubBlock", "_MetaBlock", "SimVP2", "_MHSA", "_DividedSTBlock", "STViT",
    "ORACLE_LOGRE_MU", "ORACLE_LOGRE_SD", "ORACLE_AOA_MU", "ORACLE_AOA_SD", "oracle_cond_features",
)


def _reexport_closed(mod) -> None:
    """Bind every name in `_CLOSED_NAMES` from a fully loaded models_closed into this module."""
    g = globals()
    for name in _CLOSED_NAMES:
        g[name] = getattr(mod, name)


import models_closed  # noqa: E402

if getattr(models_closed, "_LOADED", False):
    # The explicit list is for readers and linters (pyflakes, "go to definition" from `build`);
    # it must name exactly `_CLOSED_NAMES`, which the other import order uses.
    from models_closed import (  # noqa: E402,F401
        DeepONetCh, AFNO2d, DilResNet, UFNO2d, _drop_path, _trunc_init, _var_mlp,
        _ConvNeXtBlock, CNeXtUNet, _PreActResBlock, UNetMod, _win_part, _win_merge,
        _WindowAttnV2, _SwinV2Block, _PatchMergeV2, _PatchExpand, _TokenProj, SwinUNet,
        _ConvSC, _GASubBlock, _MetaBlock, SimVP2, _MHSA, _DividedSTBlock, STViT,
        ORACLE_LOGRE_MU, ORACLE_LOGRE_SD, ORACLE_AOA_MU, ORACLE_AOA_SD, oracle_cond_features,
    )
    _reexport_closed(models_closed)
