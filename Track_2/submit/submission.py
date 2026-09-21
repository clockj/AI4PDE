"""RealPDE Track 2 (LTTTA) submission: the Track-1 ensemble, streamed, frozen, CUDA-graphed.

Round T2-1.  **No test-time adaptation at all.**  ``ttt_step`` runs the same forecaster Track 1
ships and returns the same SPS band.  That is deliberate: it is the floor of this track, it
puts a number on a board where we have none, and every adaptation arm afterwards is measured
against it instead of against a guess.  The two backprop-free adaptation families that looked
obvious were both measured first and both are dead -- see ``../ttt_offline.py`` (lag-1
residual correlation is +0.05, so an EMA bias correction has nothing to correct).

WHAT IS DIFFERENT FROM TRACK 1
------------------------------
* **Space.**  Track 1 hands ``predict()`` raw physical fields; Track 2 hands ``ttt_step``
  tensors the evaluator has already normalized, and denormalizes the answer itself.  The
  evaluator's ``mean_in``/``std_in`` are bit-identical to ``_fno.MEAN``/``_fno.STD`` (checked
  against ``example_data/mean_std_real.pt``), so the normalized input IS what the trunk wants
  -- no round trip.  The TARGET stats differ in the sixth decimal, so the output carries one
  exact affine correction (``_A``, ``_B``).  ``../verify_space.py`` proves the whole path
  reproduces Track 1's ``predict()`` to 4e-6 in physical units.
* **Batch 1, and the step is what is timed.**  Track 1 amortizes one batched call over every
  window; here each step pays the full launch cost.  Measured on an A100-PCIe, eagerly:
  a single member costs **3.18 ms** of which the arithmetic is a small fraction, and each
  extra INVOCATION costs **+2.4 ms** while an extra member inside a fused group costs 0.09.
* **Hence the CUDA graph.**  The whole device-side step -- forward, ensemble average,
  fluctuation rescaling, band -- is captured once at load time (which is not timed) and
  replayed per step.  Measured bit-exact against the eager path (max abs difference 0.0) and
  3.70 ms -> 1.04 ms for the fused 8-member group.  At the fitted `time` weight of 0.081 that
  is worth about +0.25 `final`, for no change in any other subscore.  If capture fails for any
  reason the eager path runs instead, so the graph is an optimization and never a dependency.
* **Bounds go in ``info``**, in normalized space: ``half_norm = half_raw / std_tgt``.

WHAT IS THE SAME
----------------
The members, ``ALPHA`` and ``band.npz`` come from the Track-1 package.  Both tracks score the
same windows with the same five subscores, so a band fitted to this ensemble's error field on
Track 1 is the band for it here.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

import _fno  # noqa: E402

# TF32 OFF, on both the matmul and the cuDNN paths. This is a CORRECTNESS fix, not a tuning
# knob, and `verify/cpu_smoke_submission.py` is what found it.
#
# TF32 tiles a GEMM/convolution across the BATCH axis, so a window's position in the batch
# decides which accumulation order it lands in, and a 10-bit mantissa makes that visible.
# Measured on 8 val_unseen windows, same input permuted (job 7144657):
#
#                            repeat        permute (TF32 on)   permute (TF32 off)
#     w96 FNO x3 (fused)     0.00e+00           9.10e-06            5.07e-07
#     U-FNO x3               0.00e+00           1.80e-04            8.64e-07
#     U-FNO x1               0.00e+00           2.93e-04            7.15e-07
#
# The model is bit-exact on a repeat either way, so this is not nondeterminism -- it is batch
# position dependence, and the U-FNO's 3x3 convolutions carry ~20x more of it than the FNO's
# 1x1 + FFT path. 1.8e-4 fails the submission gate's 1e-4 permutation bound outright, and even
# where it passes it means the evaluator's batching order perturbs our score.
#
# It is free: round 11 measured TF32 on the spectral matmul at +0.05 ms and no subscore change
# past 1e-5, so there is nothing to trade against.
#
# OPT-IN (round 17 Q1): RPDE_TF32=1 turns it back on, for the measurement drivers only. The
# default is unchanged -- OFF -- and stays that way until a batch-position test says the
# dependence above is really a dependence on the batch SHAPE (i.e. that it disappears once
# every batch is padded to the same size), measured on the members a package actually ships.
_TF32 = os.environ.get("RPDE_TF32", "") not in ("", "0")
torch.backends.cuda.matmul.allow_tf32 = _TF32
torch.backends.cudnn.allow_tf32 = _TF32

# ROUND 17 Q1g: the spectral fast path, baked in as a constant.  The evaluation container sets
# no environment variables, so `_fno`'s env default cannot reach a submission -- this assignment
# is what ships.  "bmm" runs the spectral product as one batched real 2x2-block matmul; at
# BATCH 1 that deletes the complex einsum's per-call strided weight copy, which the profiler
# puts at 27.6 % of a step (`aten::copy_` on [4, 8, 12, 96, 1, 96], 16 calls), and is worth
# -16.6 % on the graph step.  It is ~2e-06 relative rather than bit-identical.
#
# There is NO host-write constant here and there should not be: Track 2 never does a numpy
# strided write.  `ttt_step` copies the graph's output buffers into PINNED torch tensors, so
# Track 1's `HOST_WRITE` has no counterpart on this path.
SPEC_PATH = "bmm"
# FUSE = False  keeps every member as its own module instead of collapsing same-architecture
#   members into one grouped invocation. It is NOT a speed switch and must never be flipped for
#   score: fuse() is score-identical (78.4925 against 78.4927 on the AoA 15-20 proxy, exact to
#   1.5e-8 with TF32 off), so the ONLY thing it changes is how many times the board sees a model
#   called. That is exactly what makes it an instrument. `board ms = a + 0.435*invocations +
#   0.336*local` has never had its invocation term measured -- it is fitted, on rounds whose
#   residuals are ordered by date, and R21 came in 1.2x over it on Track 1 and 2.4x on Track 2
#   for a change that added compute AND an invocation at the same time. Shipping the same
#   members unfused holds the compute BIT-IDENTICAL and moves only the invocation count, which
#   is the R13/R14 design that measured w_time directly.
FUSE = True
_fno._SPEC = SPEC_PATH

# `_body` below runs the members itself rather than going through `_fno.predict_fno`, so the
# RPDE_AMP switch has to be honoured here too. Default None = the shipped fp32 path.
#
# AMP is BAKED like FUSE (round 25) and overrides RPDE_AMP when not None. A non-None value is
# VERIFIED against fp32 in get_ttt_model (untimed) and dropped on any exception, non-finite output
# or disagreement above 2e-2 on a probe; `AMP_ACTIVE` records what actually runs. At batch 1
# autocast's casts are not free: r25a read fp16 23 % SLOWER on R21's package and 32 % faster on
# R24's, in separate processes -- which is why it is a measured, per-package choice.
AMP = None
_AMP = ({"fp16": torch.float16}.get(AMP) if AMP is not None else
        {"fp16": torch.float16, "half": torch.float16,
         "bf16": torch.bfloat16}.get(os.environ.get("RPDE_AMP", "").lower()))
AMP_ACTIVE = None
AMP_PROBE_DIFF = None

T_IN = T_OUT = 20
H, W = 32, 64
CH = T_IN * 2

# Evaluator target-space stats (example_data/mean_std_real.pt).  Our net emits fields
# normalized by _fno.MEAN / _fno.STD, which ARE the evaluator's input stats, so
#     pred_norm = (y_ours * STD + MEAN - MEAN_TGT) / STD_TGT = y_ours * _A + _B
_MEAN_TGT = np.array([0.15496256947517395, -0.0005177936982363462], dtype=np.float64)
_STD_TGT = np.array([0.09681040793657303, 0.015963643789291382], dtype=np.float64)
_A = (_fno.STD.astype(np.float64) / _STD_TGT).astype(np.float32)
_B = ((_fno.MEAN.astype(np.float64) - _MEAN_TGT) / _STD_TGT).astype(np.float32)

ALPHA = 0.60

# ---------------------------------------------------------------------------------------
# THE RUNNING-HISTORY MEAN. Off by default (HIST_BETA = 0), so the step is bit-identical to
# the round-13 package until it is switched on.
#
# Track 1's gated mean-blend replaces part of the predicted time-mean with the INPUT WINDOW's
# own measured mean, because a measurement cannot be wrong about a Reynolds number it has
# never seen. Here a strictly better measurement is available. The official `local_eval.py`
# holds ONE model object, calls `ttt_step` in strict time order, and calls
# `reset_ttt_state()` at trajectory boundaries -- a hook that exists only because
# within-trajectory state is expected to persist. So at step k every frame the stream has
# already delivered is ours to keep, and those are frames from BEFORE the forecast time,
# which is the ordinary situation of any deployed forecaster. It is the opposite of the leak
# CLAUDE.md records, which reads window k+1 -- window k's own answer, bit for bit.
#
# Measured on val_unseen's own 336 windows (`Track_1/ctx_probe.py`), estimating the target
# window's time-mean:
#     the input window's own 20 frames   0.1022     <- all Track 1 can ever use
#     every frame delivered so far       0.0783     -23%
#     the whole trajectory, incl. future 0.0739     (the offline ceiling)
# 20 frames is under a third of a 60-75 frame shedding period, so a window mean is
# contaminated by the phase it happens to sit at; a long average is not.
#
# GATED, for the same reason the Track-1 blend is: in DISTRIBUTION the model's own predicted
# mean (0.0438) is far better than any history estimator, so blending would make it worse.
# The gate fires only where the model has no right to be trusted -- windows whose mean
# streamwise speed leaves the training range.
#
# The accumulator is updated in `ttt_step`, NOT in `_body`: `_body` runs inside a captured
# CUDA graph and is also replayed five times during warm-up and twice more by the capture-time
# verification, any of which would corrupt a running sum. `_body` only READS the static
# buffers, which the graph can do because their storage never moves.
HIST_BETA = 0.0
BLEND_REF = 0.02
TRAIN_SPEED = (0.037523, 0.296857)   # per-WINDOW mean streamwise speed, all 82 real traj


class FrozenEnsemble:
    """The ``reset_ttt_state`` / ``ttt_step`` contract over a fused, frozen ensemble."""

    def __init__(self, models, device, band, sigma_ref, band_p, use_graph=True):
        self.models = models
        self.device = str(device)
        self.band_p = float(band_p or 0.0)
        cuda = self.device.startswith("cuda")

        # band / sigma_ref arrive as physical half-widths; dividing by std_tgt once, here,
        # moves them into the space `info` has to report and keeps the timed path free of it.
        self.half0 = self.sref = None
        if band is not None:
            b = np.ascontiguousarray(band[..., :2].astype(np.float32)) / _STD_TGT.astype(np.float32)
            self.half0 = torch.from_numpy(b.astype(np.float32)).to(device)
        if sigma_ref is not None:
            self.sref = torch.from_numpy(
                np.ascontiguousarray(sigma_ref[..., :2].astype(np.float32))).to(device)
        self.std = torch.from_numpy(_fno.STD).to(device)
        self.Ad = torch.from_numpy(_A).to(device)
        self.Bd = torch.from_numpy(_B).to(device)

        # Static I/O buffers.  They are the graph's inputs and outputs, so they must be the
        # same storage on every step -- which is also why they are allocated once here.
        self.gx = torch.zeros((1, T_IN, H, W, 3), dtype=torch.float32, device=device)
        self.go = torch.zeros((1, T_OUT, H, W, 3), dtype=torch.float32, device=device)
        self.gl = torch.zeros_like(self.go)
        self.gh = torch.zeros_like(self.go)
        self.host = [torch.empty((1, T_OUT, H, W, 3), dtype=torch.float32,
                                 pin_memory=cuda) for _ in range(3)]
        # Running-history accumulator. `hsum` is the sum of every delivered frame's u,v, `hn`
        # the frame count; `ghm` holds the resulting mean and `ggate` the blend strength, both
        # written by ttt_step and read by _body.
        self.hsum = torch.zeros((1, 1, H, W, 2), dtype=torch.float32, device=device)
        self.hn = 0.0
        self.ghm = torch.zeros_like(self.hsum)
        self.ggate = torch.zeros((), dtype=torch.float32, device=device)
        self.graph = None
        if cuda and use_graph:
            self._capture()

    # -- the device-side step, written once and used by both paths ----------------------
    def _body(self):
        pv = self.gx[..., :2].permute(0, 1, 4, 2, 3).reshape(1, CH, H, W).contiguous()
        y = sig = None
        total = 0
        for m in self.models:
            nm = getattr(m, "n_members", 1)      # a fused group stands in for nm members
            total += nm
            if _AMP is None:
                raw, logstd = m(pv)
            else:
                with torch.autocast("cuda", dtype=_AMP):
                    raw, logstd = m(pv)
                raw = raw.float()
                logstd = None if logstd is None else logstd.float()
            yi = raw.reshape(1, T_OUT, 2, H, W).permute(0, 1, 3, 4, 2) * nm
            y = yi if y is None else y + yi
            if logstd is not None:
                si = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
                sig = si if sig is None else sig + si
        if total > 1:
            y = y / total
            sig = None if sig is None else sig / total
        if sig is not None and ALPHA > 0:
            m_ = y.mean(1, keepdim=True)
            s_ = y.std(1, keepdim=True, unbiased=False)
            y = m_ + (y - m_) * ((ALPHA * sig + (1 - ALPHA) * s_) / (s_ + 1e-6))
        if HIST_BETA:
            # Unconditional tensor arithmetic: ggate is 0 for an in-range window, so this is
            # an identity there and no branch is needed inside the graph.
            y = y + (HIST_BETA * self.ggate) * (self.ghm - y.mean(1, keepdim=True))
        self.go[..., :2] = y * self.Ad + self.Bd
        if self.half0 is None:
            return
        half = self.half0
        if self.band_p and self.sref is not None and sig is not None:
            half = half * torch.clamp((sig * self.std) / self.sref, 0.3, 3.0) ** self.band_p
        self.gl[..., :2] = self.go[..., :2] - half
        self.gh[..., :2] = self.go[..., :2] + half

    @torch.no_grad()
    def _capture(self):
        """Capture ``_body`` once, at load time, and VERIFY it before trusting it.

        Capture is worth 3.61 ms -> 0.989 ms/step on the fused 8-member group, which is about
        +0.25 `final` at the fitted `time` weight.  It is also the only part of this file whose
        behaviour depends on a GPU we have never run on, so it is checked rather than assumed:
        capture, then replay on random input and compare against the eager path.  Anything that
        raises, or any disagreement at all, drops back to eager.  Never raises.
        """
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    self._body()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._body()

            probe = torch.randn_like(self.gx)
            self.gx.copy_(probe)
            self._body()                                   # eager, into the same buffers
            ref = [t.clone() for t in (self.go, self.gl, self.gh)]
            self.gx.copy_(probe)
            g.replay()
            torch.cuda.synchronize()
            worst = max(float((a - b).abs().max())
                        for a, b in zip((self.go, self.gl, self.gh), ref))
            self.graph = g if worst == 0.0 else None
        except Exception:
            self.graph = None

    # -- contract ----------------------------------------------------------------------
    def reset_ttt_state(self) -> None:
        """The weights never move; the running history does, and it is per TRAJECTORY.

        Failing to clear it here would carry one trajectory's mean field into the next, which
        is both wrong and undetectable from a single-trajectory test -- so `validate_t2.py`
        streams several.
        """
        self.hsum.zero_()
        self.hn = 0.0

    @torch.no_grad()
    def ttt_step(self, input_norm, prev_target_norm=None) -> Tuple[Any, Dict[str, Any]]:
        x = torch.as_tensor(input_norm)
        if x.dtype != torch.float32:
            x = x.float()
        self.gx.copy_(x, non_blocking=True)
        if HIST_BETA:
            self.hsum += self.gx[..., :2].sum(1, keepdim=True)
            self.hn += T_IN
            torch.div(self.hsum, self.hn, out=self.ghm)
            # The gate reads PHYSICAL speed, and this track's tensors are normalized.
            spd = self.gx[..., 0].mean() * float(_fno.STD[0]) + float(_fno.MEAN[0])
            lo, hi = TRAIN_SPEED
            d = torch.clamp(torch.maximum(lo - spd, spd - hi), min=0.0) / max(
                (hi - lo) * BLEND_REF, 1e-9)
            torch.clamp(d, 0.0, 1.0, out=self.ggate)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._body()
        srcs = (self.go,) if self.half0 is None else (self.go, self.gl, self.gh)
        out = self._fetch(srcs, x.is_cuda)
        if self.half0 is None:
            return out[0], {"adapt_loss": None}
        return out[0], {"adapt_loss": None, "lower": out[1], "upper": out[2]}

    def _fetch(self, srcs, want_cuda: bool):
        """Return tensors the caller OWNS, on the device it handed us the input on.

        Two things this gets right that the obvious version does not.

        **The clone is not optional.**  The evaluator keeps every prediction until the run
        ends, and these buffers are overwritten on the next step, so handing them back makes
        all N predictions equal to the last one -- rel-L2 0.67 instead of 0.074.  Every local
        harness hides it, because the kit's `local_eval` and our `stream_eval` both copy out of
        the returned tensor on the same line.  `validate_t2.py` asserts it, and caught exactly
        this on its first GPU run.

        **The device is matched to the input's**, not fixed.  The kit's `local_eval` holds its
        normalizer statistics on the CPU and would raise on a CUDA return; the kit's own
        reference `ReferenceTTTModel` returns whatever `device` it was constructed with.  The
        two cannot both be right about the platform, so the safe policy is to mirror the
        evaluator: hand back what it handed in.  On CUDA that also skips three device-to-host
        copies.
        """
        if want_cuda and self.device.startswith("cuda"):
            return [s.clone() for s in srcs]
        if not self.device.startswith("cuda"):
            return [s.clone() for s in srcs]
        for i, s in enumerate(srcs):
            self.host[i].copy_(s, non_blocking=True)   # pinned: 0.031 ms vs 0.068 pageable
        torch.cuda.synchronize()
        return [self.host[i].clone() for i in range(len(srcs))]

def _load_band(submission_dir: str):
    p_npz = os.path.join(submission_dir, "band.npz")
    if not os.path.exists(p_npz):
        return None, None, 0.0
    z = np.load(p_npz)
    band = z["base"].astype(np.float32)
    ref = z["sigma_ref"].astype(np.float32)
    p = float(z["p"])
    if not (np.all(np.isfinite(band)) and np.all(band >= 0.0)):
        return None, None, 0.0
    if band.shape[1:4] != (T_OUT, H, W) or band.shape[-1] < 2:
        return None, None, 0.0
    if p and not np.all(ref > 0.0):
        p = 0.0
    return band, ref, p


def _amp_verified(models, dev):
    """`_AMP` if every member group reproduces its fp32 output under autocast on a probe, else None."""
    global AMP_PROBE_DIFF
    if _AMP is None or not str(dev).startswith("cuda"):
        return None
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        pv = torch.randn((1, CH, H, W), generator=g).to(dev)      # already-normalized space
        worst = 0.0
        with torch.no_grad():
            for mdl in models:
                r0, l0 = mdl(pv)
                with torch.autocast("cuda", dtype=_AMP):
                    r1, l1 = mdl(pv)
                r1 = r1.float()
                l1 = None if l1 is None else l1.float()
                if not bool(torch.isfinite(r1).all()) or (l1 is not None and not bool(torch.isfinite(l1).all())):
                    return None
                worst = max(worst, float((r1 - r0).norm() / r0.norm().clamp(min=1e-12)))
                if l1 is not None:
                    worst = max(worst, float((l1 - l0).norm() / l0.norm().clamp(min=1e-12)))
        AMP_PROBE_DIFF = worst
        return _AMP if worst < 2e-2 else None
    except Exception:
        return None


def get_ttt_model(submission_dir: str, device: str):
    """Called once, before the stream, and NOT timed -- so warm-up and capture belong here.

    On CUDA the first forward of a shape pays cuFFT/cuDNN plan selection; at batch 1 that is
    tens of milliseconds and would otherwise be charged to step 0 of the first trajectory.
    """
    dev = str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    ckpts = [p for p in [os.path.join(submission_dir, "fno.pth")]
             + [os.path.join(submission_dir, f"fno_{i}.pth") for i in range(2, 17)]
             if os.path.exists(p)]
    models = [_fno.load_any(p, dev) for p in ckpts]
    if FUSE:
        models = _fno.fuse(models)
    band, ref, p = _load_band(submission_dir)
    # Decided before the graph is captured, so the graph records the mode that will run.
    global _AMP, AMP_ACTIVE
    _AMP = _amp_verified(models, dev)
    AMP_ACTIVE = None if _AMP is None else str(_AMP).replace("torch.", "")
    m = FrozenEnsemble(models, dev, band, ref, p)
    try:
        probe = torch.zeros((1, T_IN, H, W, 3), dtype=torch.float32)
        for _ in range(3):
            m.ttt_step(probe, None)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception:
        pass
    return m
