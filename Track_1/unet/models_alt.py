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
                 tbasis: int = 0, mode_drop: float = 0.0, p_head: bool = False,
                 sub_head: bool = False):
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

    def forward(self, prev: torch.Tensor, detach_var: bool = False,
                z: torch.Tensor | None = None, return_p: bool = False,
                return_sub: bool = False):
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
        if return_sub:
            return field, logstd, (self.head_sub(h) if self.head_sub is not None else None)
        if return_p:
            return field, logstd, (self.head_p(h) if self.head_p is not None else None)
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
          osc_zeromean: bool = False, tbasis: int = 0, mode_drop: float = 0.0,
          p_head: bool = False, sub_head: bool = False):
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
                     osc_zeromean, tbasis, mode_drop, p_head=p_head,
                     sub_head=sub_head)
    if arch == "deeponet":
        return DeepONetCh(p, trunk_width=width * 2, branch_width=width,
                          var_head=var_head, extra_ch=extra_ch)
    if arch in WAVE1_ARCHS:
        # 2026-09-14 wave 1 (the block at the end of this file). Each family is fully specified
        # by (width, n_layers, modes_y, modes_x) plus class constants, because _fno.load_any's
        # experimental branch rebuilds a checkpoint from exactly those blob fields.
        return build_wave1(arch, width, var_head, extra_ch, modes_y, modes_x, n_layers,
                           dropout=dropout)
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


class _LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the CHANNEL axis of an NCHW map (ConvNeXt's data_format='channels_first')."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


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

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
        h = F.pad(self.lift(self.cat(prev)), (0, self.PAD[1], 0, self.PAD[0]))
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h[..., :H, :W])
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bw, n, c = x.shape
        q, k, v = (self.qkv(x).reshape(bw, n, 3, self.heads, c // self.heads)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        q = F.normalize(q, dim=-1) * scale
        k = F.normalize(k, dim=-1)
        bias = (self.cpb(self.coords)[self.rel_index]
                .reshape(n, n, self.heads).permute(2, 0, 1))                 # (heads, N, N)
        if mask is None:
            am = bias.unsqueeze(0)
        else:                                                                # mask (nW, 1, N, N)
            nw = mask.shape[0]
            am = ((bias[None, None] + mask[None]).expand(bw // nw, nw, self.heads, n, n)
                  .reshape(bw, self.heads, n, n))
        dt = v.dtype
        o = F.scaled_dot_product_attention(q.to(dt), k.to(dt), v, attn_mask=am.to(dt),
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
                 extra_ch: int = 0, dropout: float = 0.0, drop_path: float = 0.1):
        super().__init__()
        res = [((H // 2) >> i, (W // 2) >> i) for i in range(self.STAGES)]
        dims = [width * 2 ** i for i in range(self.STAGES)]
        heads = [max(d // 32, 1) for d in dims]
        n_blocks = n_layers * (2 * self.STAGES - 1)
        dpr = iter([drop_path * i / max(n_blocks - 1, 1) for i in range(n_blocks)])

        def stage(i):
            return nn.ModuleList([_SwinV2Block(dims[i], res[i], heads[i], self.WINDOW,
                                               bool(j % 2), dropout, next(dpr))
                                  for j in range(n_layers)])

        self.embed = nn.Conv2d(CH + extra_ch + 2, width, 2, stride=2)
        self.embed_norm = nn.LayerNorm(width)
        self.enc = nn.ModuleList([stage(i) for i in range(self.STAGES - 1)])
        self.merge = nn.ModuleList([_PatchMergeV2(dims[i]) for i in range(self.STAGES - 1)])
        self.skip_blocks = nn.ModuleList([
            nn.Sequential(*[_ConvNeXtBlock(dims[i], dropout) for _ in range(self.SKIP_BLOCKS)])
            for i in range(self.STAGES - 1)])
        self.bottleneck = stage(self.STAGES - 1)
        self.expand, self.skip, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(self.STAGES - 1)):
            self.expand.append(_PatchExpand(dims[i + 1], dims[i]))
            self.skip.append(nn.Linear(2 * dims[i], dims[i]))
            self.dec.append(stage(i))
        self.out_ch = max(width // 2, 32)
        self.final = _PatchExpand(width, self.out_ch)
        self.refine = nn.Sequential(nn.GELU(), nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1))
        self.apply(_trunc_init)
        self.head_field, self.head_logstd = _heads(self.out_ch, var_head)

    def forward(self, prev: torch.Tensor, detach_var: bool = False):
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


def build_wave1(arch: str, width: int, var_head: bool, extra_ch: int, modes_y: int,
                modes_x: int, n_layers: int, dropout: float = 0.0):
    """Wave-1 factory, reached from `build`. Only (width, n_layers, modes_y, modes_x) vary."""
    if arch == "cnext_unet":
        return CNeXtUNet(width, n_layers, var_head, extra_ch, dropout)
    if arch == "unet_mod":
        return UNetMod(width, n_layers, var_head, extra_ch, dropout)
    if arch in ("ffno", "ffno_nff"):
        return FFNO2d(width, modes_y, modes_x, n_layers, var_head, extra_ch, dropout,
                      ff=(arch == "ffno"))
    if arch == "swin_unet":
        return SwinUNet(width, n_layers, var_head, extra_ch, dropout)
    if arch == "simvp":
        return SimVP2(width, n_layers, var_head, extra_ch, dropout)
    if arch == "stvit":
        return STViT(width, n_layers, var_head, extra_ch, dropout)
    raise ValueError(arch)
