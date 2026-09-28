"""RealPDE Track 2 (LTTTA) submission: the Track-1 ensemble, streamed, with causal per-trajectory calibration.

(Docstring refreshed 2026-09-28 to describe the final package of record, 2026-09-27. Until 09-25 this track shipped the
forecaster FROZEN -- "Round T2-1, no test-time adaptation" -- and several comments below still date from then.)

THE MODEL
---------
The same six members as Track 1: four F-FNO (w40 L8, sim co-trained) and two ViT (d384 L8), fused into a
StackedFFNO(4) and a StackedViT(2), averaged uniformly; fp16 autocast (verified against fp32 at load); ALPHA 0.80 with
one compose gain per pixel for u and v (JOINT_GAIN); the SIGBLEND band (band.npz + SIGPOW). The weights never move.

WHAT ttt_step() DOES (batch 1; everything device-side is ONE CUDA graph, captured and verified at load)
----------------------
Inputs: this window's normalized input, and the previous window's target (revealed by the platform at this step).
  1. The revealed target is sanitized (non-finite -> 0, and that step updates no statistic; nor does the step after a
     non-finite forward; M3, 2026-09-28).
  2. R1 (causal band calibration): the PREVIOUS window's interval is scored against its revealed target, element by
     element, for a 9 x 9 grid of (lower, upper) multipliers; the running tables (per pixel, and per 4-frame lead block)
     give this window's multipliers k_lo, k_hi, shrunk toward 1 while few windows have been seen (R1_ASYM/TSM/BLK).
  3. D1c (causal bias removal): the running mean of (previous pre-compose mean - revealed target) over past windows,
     shrunk by n/(n+30) or by an empirical-Bayes factor, with a lead-time profile -> this window's shift -b.
  4. VARCAL: running revealed-target variance against the composed variance predicted for the same windows -> a
     per-pixel gain on the composed fluctuation amplitude, shrunk toward 1.
  5. Forward of both groups on side streams, member mean y0 and sigma; compose (ALPHA, JOINT_GAIN, VARCAL gain) -> y;
     prediction P = y + shift (target-normalized space, exact affine from the model's input space).
  6. Band: half = base * clip(sigma/sigma_ref, .3, 3)^p * (sigma + 1e-8)^q, times k_lo / k_hi; centred on y0 + shift
     (CENTRE_ALPHA 0) and widened to contain P; non-finite elements sanitized (M3).
  7. For a CPU input (the kit's runner), outputs are copied into their own slot of a pinned arena (caller-owned, no
     clone; the first ARENA_STEPS steps, then pinned staging + clone) and returned as numpy.
Nothing reads a window the platform has not shown: `stream_eval.py --causal-check` asserts it on every build.
`reset_ttt_state()` clears every running statistic at a trajectory boundary.

Space: the platform hands `ttt_step` INPUT-normalized tensors, which are exactly what the trunk wants
(`_fno.MEAN/STD` are bit-identical to the evaluator's input stats); the answer and the bounds go back in TARGET
normalization via one exact affine (`_A`, `_B`). The revealed target is normalized by the official runner with the
TARGET statistics; PREV_SPACE below says which space this file assumes (audit M1, 2026-09-28).
"""
from __future__ import annotations

import contextlib
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

# ROUND 17 Q1g: the spectral fast path, baked in as a constant. (2026-09-28: INERT in the final package -- it steers only
# FNO2d stacks and the final member set has none; the numbers below are the T2R5-era FNO package's.)  The evaluation container sets
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
_ERR = None       # err-head band settings from band.npz; None = no head term (see _load_band)

# PAR_STREAMS (09-23): run the fused groups on separate CUDA streams inside the step, so a
# batch-1 step that leaves the GPU partly idle can overlap them. Arithmetic is unchanged
# (verified package-against-package with verify/pkg_identity.py, --tol 0, run 20260923-170904:
# T2 -1.03 ms/step on a V100); False restores the serial loop.
PAR_STREAMS = True
# GRAPH_TOL (N6, Jonas 09-25): the load-time capture check compares graph replay with the eager path.
# 0.0 (the historical rule) demands bit-identity, which a different GPU's cuFFT/cuBLAS choice can break
# and then the step silently falls back to eager (~3x slower at batch 1). A positive value accepts a
# replay whose max|replay - eager| / max|eager| is at most GRAPH_TOL. GRAPH_WORST records what was
# measured at load (None = capture raised), for the load-time report. Eager path unchanged.
GRAPH_TOL = 0.0
GRAPH_WORST = None
GRAPH_ERR = None      # the exception text if capture raised (load-time report; None = no exception)
# DIAG (N1, audit 2, 09-25): one line of load-time facts on stdout from get_ttt_model (NOT timed) -- GPU, CPUs, torch/CUDA,
# AMP, graph captured or not, GRAPH_WORST / GRAPH_ERR -- plus one line on the first ttt_step (shape, dtype). Nobody had
# ever read the Codabench logs; this makes them worth reading. Printing changes no tensor.
DIAG = True
# PRUNED 2026-09-28 (never shipped in a package of record, set by no reproduce.sh recipe; the last tree that has them is
# commit fdac00b, and build_round13.sh now REFUSES their RPDE_SHIP_* variables at a non-default value): FALLBACK_FP (the
# audit-2 graph-fallback fingerprint), STACKED_OUT (one stacked return storage, Corin 09-26), R1_SNAP (legacy snapping of
# the R1 multiplier to its grid) and SIGEMA / SIGEMA_BETA / SIGEMA_N0 / SIGEMA_DDOF (ALPHA's sigma from an EMA of revealed
# target variances, 09-27 arm).
# ARENA_STEPS (K1, 09-27): > 0 = on the CPU return path every step's pred / lower / upper are written straight into
# their OWN slot of a page-locked arena allocated at load (untimed), instead of pinned staging + three fresh clones per
# step (the evaluator keeps every prediction, so each clone is a fresh mmap + first-touch faults). Same bits (md5 over 336
# steps); A100 / 2.2.2 median 3.05 -> 2.82 ms/step. Falls back to the clone path when the arena is used up or could not
# be allocated. 1024 steps = 1.5 GB page-locked. Build knob RPDE_SHIP_ARENA_STEPS.
ARENA_STEPS = 0
# NUMPY_OUT (09-26, t2_ipc3): on the CPU return path hand back numpy arrays (zero-copy views of the fresh clones) instead of torch
# tensors. The kit's local_eval takes `torch.as_tensor(...)` of what we return, so values/shape/dtype are unchanged; the process
# boundary is cheaper for numpy under every transport emulated (+0.25..0.29 ms/step torch queue, +0.8..1.06 plain pickle). False = torch.
NUMPY_OUT = False
# SPS interval centre, decoupled from the prediction -- see Track_1 submission.py CENTRE_ALPHA.
# Read from band.npz (`centre_alpha`) in _load_band; no key => ALPHA, the old bounds exactly.
CENTRE_ALPHA = ALPHA
# JOINT_GAIN (K2 09-27): one compose gain per pixel for both channels, weighted in physical units (std**2); same TKE map and
# time mean, prediction closer to y0. Requires CENTRE_ALPHA == 0 (asserted in _load_band). False = bit-identical.
JOINT_GAIN = False

# ---------------------------------------------------------------------------------------
# R1 (2026-09-25, Ramesh; approved for TRACK 2 ONLY by Jonas): causal per-pixel online calibration of the SPS
# half-width from this trajectory's own REVEALED errors. The Codabench T2 page names "calibration" as a legal
# adaptation inside ttt_step; at step t the platform hands us the target of step t-1 and nothing later.
# For every (y, x, c) keep, per candidate multiplier k in R1_KS, the running sum over PAST windows of the scorer's
# element term 1[lo<=y<=hi]*exp(-(hi-lo)/SIGMA_G) on scored elements (unweighted). Current window: per-pixel argmax,
# 3x3 replicate-padded box mean, shrink toward 1 by n/(n+R1_N0), applied continuously (not snapped to the grid);
# first window of a trajectory k=1.
# base, sigma ratio, centre and the min/max clamp are untouched: only half -> k*half. Off (False) = bit-identical.
# Reference: Ramesh's causal_band.py (numpy) / r1_ref/causal_band_t2.py (torch) cover the SYMMETRIC estimator (update /
# kfield) only; the shipped ASYM / TSM / BLK variants (update_gated_asym / kfield_asym) have no separate reference (2026-09-28).
# NEVER port to Track 1: its batch holds future windows, so the same statistic would read their answers.
R1 = False
R1_KS = tuple(0.5 * 2 ** (j / 4) for j in range(9))        # K=9 log grid 0.5..2.0, contains 1.0 exactly (Jonas 01:41Z 09-25)
R1_N0 = 2.0
R1_ASYM = False            # True (Jonas 09-26, r1_asym.py): separate causal k_lo / k_hi per pixel from a K x K table of past windows
                           # (lower bound only matters when the target is below the centre, upper when above; the width term splits
                           # into two factors), argmax over the K*K cells, each field box-smoothed and shrunk like R1. False = R1 as shipped.
D1C_N0 = 0.0              # D1c "PC" (Jeff d1c_extrap 09-27; Jonas build_d1c.py; locked by Ilse 9d32d19d): > 0 = on with this N0.
                           # Per trajectory, causal, per pixel and channel: b = sum over PAST revealed windows of mean_t(C - y) / (n + N0),
                           # n = revealed windows, C = the pre-compose (alpha-0) ensemble mean of that window BEFORE any correction; this
                           # window's P and interval centre both move by -b (R1 then scores the moved fields). Needs R1. 0.0 = off, bit-identical.
SIGPOW = None              # SIGBLEND band (K-sprint design_0, 09-27): None = off, bit-identical. A tuple (q_u, q_v): the static band is
                           # multiplied by (sigma_phys + 1e-8) ** q_c per channel (sigma_phys = sig * std, the heads' PHYSICAL sigma, float32),
                           # AFTER base * clip(sigma/sigma_ref)^p and BEFORE the R1 gains; the band npz carries base' and p'. Static/drv band
                           # only: refuses with the err-head band.
VC_N0 = 0.0                # VARCAL (design C, 09-27): > 0 = on with this N0. Causal per-pixel/channel calibration of the composed fluctuation
                           # AMPLITUDE from this trajectory's revealed windows: g = box3(sum var_t(y)) / box3(sum vpred), both sums over PAST windows
                           # (model space; vpred = that window's uncalibrated composed variance: (ALPHA*sigma_head + (1-ALPHA)*s_)^2 per channel, or
                           # (s_ * g_joint)^2 under JOINT_GAIN, the shipped case), clipped to VC_CLIP, shrunk to 1 by n/(n+VC_N0); the composed
                           # fluctuation is multiplied by sqrt(g) PER CHANNEL, so once g moves, JOINT_GAIN's one-gain-per-pixel property no longer
                           # holds exactly (comment corrected 2026-09-28). Band, centre and sigma_head untouched. Needs R1. 0.0 = off.
VC_CLIP = (0.25, 4.0)
FW_N0 = 0.0                # FAMW (design C, 09-27; the team's "family weights", per-pixel online mixing): > 0 = on with this N0. Needs exactly two
                           # fused groups A (first) and B (second). w = weight of B; w0 = n_B / n_members. Per pixel/channel, over PAST windows:
                           # w_ols = box3(sum_t d*(y - yA)) / box3(sum_t d*d), d = yB - yA (model space), clipped to [0, 1], shrunk to w0 by n/(n+FW_N0).
                           # The ensemble mean becomes yA + w*(yB - yA) BEFORE compose, so D1c accumulates the residual AFTER it. Needs R1. 0.0 = off.
R1_TSM = 1                 # TSM (design B2, 09-27): any value > 1 (3 by convention) = the running K x K table is box-smoothed (3x3 over H, W, replicate-padded) BEFORE the
                           # per-pixel argmax, i.e. the sufficient statistic is pooled over the 9 neighbours instead of only the argmax's output
                           # (which is still box-smoothed and shrunk as before). 1 = off, bit-identical. Needs R1_ASYM.
R1_BLK_N0 = 0.0            # BLK (design B2, 09-27): > 0 = on with this N0. Needs R1_ASYM. A second causal table at LEAD-TIME-BLOCK resolution
                           # (K x K x G x H x W x C, G = T_OUT / R1_BLK blocks of R1_BLK consecutive output frames; same statistic, same revealed
                           # windows of THIS trajectory; the per-pixel table is its sum over the blocks). Per block and pixel: 3x3 box of the table,
                           # argmax, 3x3 box of the multiplier -> k_blk; this window's multiplier = k_pix + (k_blk - k_pix) * n / (n + R1_BLK_N0),
                           # clamped to [min(R1_KS), max(R1_KS)], constant within a block. n = 0 -> exactly 1.0. 0.0 = off, bit-identical.
R1_BLK = 4                 # BLK: output frames per lead-time block (must divide T_OUT = 20)
R1_UNSCORED_TOL = 1e-6                                      # |yn - z_c| below this = an unscored (physical 0) element
# PREV_SPACE (audit M1, 2026-09-28): the normalization of the REVEALED target (`prev_target_norm`). "input" = this file assumes
# the evaluator normalized it with the INPUT statistics (_fno.MEAN/STD), which is what every local harness did until 09-28
# (stream_eval / validate_t2 / pkg_identity called normalize_in) and what every uploaded package assumes: R1 and D1c map it
# to target space with the affine (_A, _B), and an element is "unscored" iff it equals -MEAN/STD. "target" = it arrives
# TARGET-normalized, which is what the official runner does (kit t2_v6 local_eval.py: Normalizer.preprocess(x, y) with
# mean_tgt/std_tgt, then prev_target = that tgt_norm): R1 and D1c read it as it is, the unscored test is against
# -MEAN_TGT/STD_TGT, and VARCAL / FAMW (model space) map it back with the inverse affine. Under "input" the board's
# physical-0 elements miss the unscored test by 6e-5 / 2.3e-4 (60-230x R1_UNSCORED_TOL) and are counted as scored.
# "input" = every package of record, bit for bit. Build knob RPDE_SHIP_PREV_SPACE.
PREV_SPACE = "input"
# STATS_MASK (audit M2, 2026-09-28, a measured arm): False = every package of record, bit for bit. True = the D1c error field
# and VARCAL's sums skip UNSCORED (blanked, physical-0) target elements, the same test R1 uses (R1_UNSCORED_TOL against
# r1_z, so it is only right on the board together with PREV_SPACE "target"): D1c's per-frame error is 0 at an unscored
# element (so an always-blanked pixel gets no shift and adds 0 to both empirical-Bayes pools -- the EB factor is then the
# one of the scored pixels, since a common dilution cancels in tau2 / (tau2 + se2)), and VARCAL accumulates a pixel's
# variance pair only from windows where all 20 of its frames are scored; a pixel with no such window nearby keeps g = 1.
# Without it, a blanked pixel's error is ~|C| (~1.6 normalized for u), which dominates D1c-EB's pooled bias, and its
# target variance 0 clips VARCAL's gain to 0.25. Build knob RPDE_SHIP_STATS_MASK.
STATS_MASK = False
R1_FP32 = True             # True: width term + running sum in float32 (Ramesh 09-25: gains identical to 3 decimals, ~half the
                           # memory traffic and V100 fp32 rate); False: float64, bit-matching his numpy reference
_SIGMA_G = 0.0563870259
D1C_EB = False             # D1c-EB (design C phase 2, 09-27): True = the D1c shrink becomes max(n/(n+D1C_N0), tau2/(tau2+se2)) once n >= 2, an
                           # empirical-Bayes factor PER CHANNEL pooled over all pixels: raw = dsum/n (per pixel), b2 = mean_px(raw^2),
                           # s2 = sum_k(e_k - raw)^2 / (n - 1) (e_k = window k's time-mean error, per pixel), se2 = mean_px(s2) / n, tau2 = max(b2 - se2, 0).
                           # A trajectory whose persistent bias stands out of its own window-to-window scatter is corrected at nearly full strength
                           # within a few windows; one whose bias does not keeps D1c's n/(n+D1C_N0). No constant. Needs D1C_N0 > 0. False = D1c as before.
D1C_EB_N0 = 0.0            # damping of the empirical-Bayes factor in its first windows: factor *= n/(n+D1C_EB_N0). 0.0 = none.
D1C_LEAD_N0 = 0.0          # D1c lead-time profile (design C phase 2): > 0 = on. The shift becomes a(t) * b(pixel): a(t) = <bt, raw>_px / <raw, raw>_px per
                           # channel (bt = per-frame mean error over past windows), shrunk to 1 by n/(n+D1C_LEAD_N0), clipped to [0, 3]. 0.0 = off.


def _r1_box3(a):                             # (H,W,C) float64, replicate-padded 3x3 mean, reference op order
    o = a
    for ax in (0, 1):
        q = o.movedim(ax, 0); L = q.shape[0]
        q = torch.cat([q[:1], q, q[-1:]], 0)
        o = (((q[0:L] + q[1:L + 1]) + q[2:L + 2]) / 3.0).movedim(0, ax)
    return o


def _r1_boxn(a, d0, d1):                     # _r1_box3 over two arbitrary dims (d0, d1) of any tensor, same op order
    o = a
    for ax in (d0, d1):
        q = o.movedim(ax, 0); L = q.shape[0]
        q = torch.cat([q[:1], q, q[-1:]], 0)
        o = (((q[0:L] + q[1:L + 1]) + q[2:L + 2]) / 3.0).movedim(0, ax)
    return o


class _CausalBandScalerT:
    def __init__(self, ks, n0, std_tgt, sigma, device, shape=(32, 64, 2)):
        self.ks = torch.tensor(list(ks), dtype=torch.float64, device=device)
        self.ks32 = self.ks.to(torch.float32)
        self.n0 = torch.tensor(float(n0), dtype=torch.float64, device=device)
        self.adt = torch.float32 if R1_FP32 else torch.float64          # accumulation / width-term dtype
        self.std_tgt = torch.tensor(list(std_tgt), dtype=self.adt, device=device)
        self.sigma = float(sigma)
        self.cum = torch.zeros((len(ks),) + tuple(shape), dtype=self.adt, device=device)
        self.n = torch.zeros((), dtype=torch.float64, device=device)
        self.one = torch.ones((), dtype=torch.float64, device=device)
        self.shape = tuple(shape)
        self.npix = int(np.prod(shape))
        self.cum2 = (torch.zeros((len(ks), len(ks)) + tuple(shape), dtype=self.adt, device=device) if R1_ASYM else None)
        self.blk = int(R1_BLK) if (R1_ASYM and R1_BLK_N0 > 0) else 0          # BLK: frames per lead-time block (0 = off)
        self.cumG = (torch.zeros((len(ks), len(ks), T_OUT // self.blk) + tuple(shape), dtype=self.adt, device=device) if self.blk else None)
        self.kmin, self.kmax = float(min(ks)), float(max(ks))

    @torch.no_grad()
    def reset(self):
        self.cum.zero_(); self.n.zero_()
        if self.cum2 is not None:
            self.cum2.zero_()
        if self.cumG is not None:
            self.cumG.zero_()

    @torch.no_grad()
    def kfield_asym(self):                   # -> (k_lo, k_hi), each (1,1,H,W,C) float32, from PAST windows only
        # Jonas' spec (work/r1_asym/causal_band_asym_t.py): first max in row-major (lo, hi) order over the K*K cells.
        K = len(self.ks)
        flat = (_r1_boxn(self.cum2, 2, 3) if R1_TSM > 1 else self.cum2).flatten(0, 1).argmax(0)
        al, ah = torch.div(flat, K, rounding_mode="floor"), flat % K
        sh = self.n / (self.n + self.n0)
        klo = 1.0 + (_r1_box3(self.ks[al]) - 1.0) * sh
        khi = 1.0 + (_r1_box3(self.ks[ah]) - 1.0) * sh
        if self.blk:                         # BLK: -> (1,T,H,W,C), constant within a lead-time block
            fe = _r1_boxn(self.cumG, 3, 4).flatten(0, 1).argmax(0)                                   # (G,H,W,C)
            el = _r1_boxn(self.ks[torch.div(fe, K, rounding_mode="floor")], 1, 2)
            eh = _r1_boxn(self.ks[fe % K], 1, 2)
            s = self.n / (self.n + R1_BLK_N0)                                                        # n = 0 -> 0: k stays exactly 1.0
            G = el.shape[0]
            klo = (klo[None] + (el - klo[None]) * s).clamp(self.kmin, self.kmax)
            khi = (khi[None] + (eh - khi[None]) * s).clamp(self.kmin, self.kmax)
            klo = klo.to(torch.float32).unsqueeze(1).expand((G, self.blk) + self.shape).reshape((G * self.blk,) + self.shape)
            khi = khi.to(torch.float32).unsqueeze(1).expand((G, self.blk) + self.shape).reshape((G * self.blk,) + self.shape)
            return klo[None], khi[None]
        return klo.to(torch.float32)[None, None], khi.to(torch.float32)[None, None]

    @torch.no_grad()
    def update_gated_asym(self, P, C, h, y_tgt, scored, flag):
        """K x K table, Jonas' spec verbatim. lo <= C <= hi always, so inside = [y<C ? y>=lo_a : 1] * [y>=C ? y<=hi_b : 1] and
        exp(-(hi-lo)/s) = exp(-(C-lo)/s) * exp(-(hi-C)/s): cum[a,b] += sum_t A[t,a] B[t,b] (an einsum over t, not 81 evaluations)."""
        hk = h * self.ks32[:, None, None, None, None]                       # (K,T,H,W,C)
        lo = torch.minimum(C - hk, P); hi = torch.maximum(C + hk, P)
        st = self.std_tgt / self.sigma
        below = y_tgt < C
        A = torch.exp(-((C - lo).to(self.adt) * st)) * torch.where(below, y_tgt >= lo, True) * scored
        B = torch.exp(-((hi - C).to(self.adt) * st)) * torch.where(below, True, y_tgt <= hi)
        if self.blk:                         # one einsum at block resolution; the per-pixel table is its sum over the blocks
            K, G = len(self.ks), self.cumG.shape[2]
            inc = torch.einsum("agthwc,bgthwc->abghwc", A.reshape((K, G, self.blk) + self.shape),
                               B.reshape((K, G, self.blk) + self.shape)) * flag.to(self.adt)
            self.cumG += inc
            self.cum2 += inc.sum(2)
        else:
            self.cum2 += torch.einsum("athwc,bthwc->abhwc", A, B) * flag.to(self.adt)
        self.n += flag

    @torch.no_grad()
    def kfield(self):                        # -> (1,1,H,W,C) float32, from PAST windows only
        kh = _r1_box3(self.ks[self.cum.argmax(0)])
        k = 1.0 + (kh - 1.0) * self.n / (self.n + self.n0)          # n = 0 -> exactly 1.0
        return k.to(torch.float32)[None, None]

    @torch.no_grad()
    def update(self, P, C, h, y_tgt, scored):
        """(1,T,H,W,C) target-normalised; call AFTER the window's target is revealed."""
        hk = h * self.ks32[:, None, None, None, None]
        lo = torch.minimum(C - hk, P); hi = torch.maximum(C + hk, P)
        w = torch.exp(-((hi - lo).to(self.adt) * self.std_tgt) / self.sigma)
        cov = (y_tgt >= lo) & (y_tgt <= hi) & scored
        self.cum += torch.where(cov, w, torch.zeros_like(w)).sum(1)
        self.n += self.one

    @torch.no_grad()
    def update_gated(self, P, C, h, y_tgt, scored, flag):
        """update() times a device flag (float64 0/1): flag 0 adds exact zeros, so the state is unchanged. Lets the
        whole R1 step live inside the CUDA graph with no host-side branch (flag set per step before the replay)."""
        hk = h * self.ks32[:, None, None, None, None]
        lo = torch.minimum(C - hk, P); hi = torch.maximum(C + hk, P)
        w = torch.exp(-((hi - lo).to(self.adt) * self.std_tgt) / self.sigma)
        cov = (y_tgt >= lo) & (y_tgt <= hi) & scored
        self.cum += torch.where(cov, w, torch.zeros_like(w)).sum(1) * flag.to(self.adt)
        self.n += flag

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
        self.err = _ERR
        self.eref = None
        if _ERR is not None and _ERR["ref"] is not None:
            self.eref = torch.from_numpy(np.ascontiguousarray(_ERR["ref"][..., :2])).to(device)
        self.std_tgt = torch.from_numpy(_STD_TGT.astype(np.float32)).to(device)
        # err_floor (absent => None => the 09-24 direct band exactly): half = max(k*H, err_floor*base).
        self.floor = (self.half0 * float(_ERR["floor"])
                      if _ERR is not None and _ERR.get("floor") and self.half0 is not None else None)
        self.sigpow = None
        if SIGPOW is not None:
            if _ERR is not None:
                raise RuntimeError("SIGPOW is not supported together with the err-head band (_ERR); refusing")
            if self.half0 is None:
                raise RuntimeError("SIGPOW needs a band (band.npz); refusing")
            self.sigpow = torch.tensor([float(SIGPOW[0]), float(SIGPOW[1])], dtype=torch.float32, device=device)
        self.Ad = torch.from_numpy(_A).to(device)
        self.Bd = torch.from_numpy(_B).to(device)
        if PREV_SPACE not in ("input", "target"):
            raise RuntimeError(f"PREV_SPACE must be 'input' or 'target', not {PREV_SPACE!r}")
        self.prev_in = PREV_SPACE == "input"

        # Static I/O buffers.  They are the graph's inputs and outputs, so they must be the
        # same storage on every step -- which is also why they are allocated once here.
        self.gx = torch.zeros((1, T_IN, H, W, 3), dtype=torch.float32, device=device)
        self.go = torch.zeros((1, T_OUT, H, W, 3), dtype=torch.float32, device=device)
        self.gl = torch.zeros_like(self.go)
        self.gh = torch.zeros_like(self.go)
        # m2 (2026-09-28): a runtime that refuses page-locked memory must not make get_ttt_model raise; pageable staging is
        # slower (0.068 vs 0.031 ms per copy, the figure noted in _fetch) but correct. The arena below already had this fallback.
        try:
            self.host = [torch.empty((1, T_OUT, H, W, 3), dtype=torch.float32,
                                     pin_memory=cuda) for _ in range(3)]
        except Exception:
            self.host = [torch.empty((1, T_OUT, H, W, 3), dtype=torch.float32) for _ in range(3)]
        self.arena, self.arena_i = None, 0
        if ARENA_STEPS and cuda:
            try:
                self.arena = [torch.zeros((int(ARENA_STEPS), 1, T_OUT, H, W, 3), dtype=torch.float32).pin_memory()
                              for _ in range(3)]
            except Exception:
                self.arena = None
        # Running-history accumulator. `hsum` is the sum of every delivered frame's u,v, `hn`
        # the frame count; `ghm` holds the resulting mean and `ggate` the blend strength, both
        # written by ttt_step and read by _body.
        self.hsum = torch.zeros((1, 1, H, W, 2), dtype=torch.float32, device=device)
        self.hn = 0.0
        self.ghm = torch.zeros_like(self.hsum)
        self.ggate = torch.zeros((), dtype=torch.float32, device=device)
        # R1 state: gk = this window's multiplier field (read inside the graph), gH / gC = this window's pre-multiplier
        # half-width and centre (written inside the graph); at the NEXT step, before the replay overwrites them, they
        # and go are exactly the previous window's P / C / h that the revealed target is scored against.
        self.r1 = None
        if R1 and self.half0 is not None:
            self.r1 = _CausalBandScalerT(R1_KS, R1_N0, _STD_TGT.tolist(), _SIGMA_G, device, (H, W, 2))
            kT = T_OUT if (R1_ASYM and R1_BLK_N0 > 0) else 1                  # BLK: the multiplier varies with the lead-time block
            self.gk = torch.ones((1, kT, H, W, 2), dtype=torch.float32, device=device)
            self.gk2 = torch.ones((1, kT, H, W, 2), dtype=torch.float32, device=device) if R1_ASYM else None   # k_hi (R1_ASYM)
            self.gH = torch.zeros((1, T_OUT, H, W, 2), dtype=torch.float32, device=device)
            self.gC = torch.zeros((1, T_OUT, H, W, 2), dtype=torch.float32, device=device)
            # A physical-0 (blanked, unscored) target element, in the space the revealed target arrives in (PREV_SPACE).
            if self.prev_in:
                self.r1_z = torch.from_numpy((-_fno.MEAN.astype(np.float64).reshape(-1)[:2]
                                              / _fno.STD.astype(np.float64).reshape(-1)[:2]).astype(np.float32)).to(device)
            else:
                self.r1_z = torch.from_numpy((-_MEAN_TGT / _STD_TGT).astype(np.float32)).to(device)
            self.r1_prev = False
            self.gyp = torch.zeros((1, T_OUT, H, W, 3), dtype=torch.float32, device=device)   # revealed target (PREV_SPACE norm)
            self.gflag = torch.zeros((), dtype=torch.float64, device=device)                   # 1 = gyp is valid this step
            # R1 runs on its own CUDA stream, overlapping the member forwards (it only reads the PREVIOUS window's
            # go / gC / gH and the revealed target); joined back before go / gC / gH / gk are used or overwritten.
            self.r1_stream = torch.cuda.Stream() if cuda else None
        # D1c state: gC0 = the last window's uncorrected C (output space, written inside the graph); dsum / dn = the running
        # sum of mean_t(C - y) over revealed windows and their count; gD = this window's shift, from past windows only.
        self.d1c = bool(D1C_N0) and self.r1 is not None
        if self.d1c:
            self.gC0 = torch.zeros((1, T_OUT, H, W, 2), dtype=torch.float32, device=device)
            self.dsum = torch.zeros((1, 1, H, W, 2), dtype=torch.float64, device=device)
            self.dn = torch.zeros((), dtype=torch.float64, device=device)
            self.gD = torch.zeros((1, T_OUT if D1C_LEAD_N0 else 1, H, W, 2), dtype=torch.float32, device=device)
        self.d1x = self.d1c and (bool(D1C_EB) or bool(D1C_LEAD_N0))
        if self.d1x:
            self.dsq = torch.zeros((1, 1, H, W, 2), dtype=torch.float64, device=device)       # sum of squared time-mean errors
            self.dsumf = torch.zeros((1, T_OUT, H, W, 2), dtype=torch.float64, device=device)  # per-frame error sum (lead profile)
        # VARCAL state: cT / cP = running sums of revealed target variance and of the matching windows' uncalibrated composed variance,
        # cn = revealed windows, gVp = the last window's uncalibrated composed variance (written in the graph), gG = this window's std gain.
        self.vc = bool(VC_N0) and self.r1 is not None and ALPHA > 0
        if self.vc:
            self.cT = torch.zeros((H, W, 2), dtype=torch.float64, device=device)
            self.cP = torch.zeros((H, W, 2), dtype=torch.float64, device=device)
            self.cn = torch.zeros((), dtype=torch.float64, device=device)
            self.gVp = torch.zeros((1, 1, H, W, 2), dtype=torch.float32, device=device)
            self.gG = torch.ones((1, 1, H, W, 2), dtype=torch.float32, device=device)
        # FAMW state: fA / fB = running sums of d*d and d*(y - yA); fn = revealed windows; gYA / gYB = the last window's family means
        # (model space, written in the graph); gW = this window's weight of family B.
        self.fw = bool(FW_N0) and self.r1 is not None and len(models) == 2
        if self.fw:
            nA, nB = (getattr(m, "n_members", 1) for m in models)
            self.fw0 = float(nB) / float(nA + nB)
            self.fA = torch.zeros((H, W, 2), dtype=torch.float64, device=device)
            self.fB = torch.zeros((H, W, 2), dtype=torch.float64, device=device)
            self.fn = torch.zeros((), dtype=torch.float64, device=device)
            self.gYA = torch.zeros((1, T_OUT, H, W, 2), dtype=torch.float32, device=device)
            self.gYB = torch.zeros((1, T_OUT, H, W, 2), dtype=torch.float32, device=device)
            self.gW = torch.full((1, 1, H, W, 2), self.fw0, dtype=torch.float32, device=device)
        # M3 (2026-09-28): gok = this window's forward was finite (read by the next step's flag; see _body).
        if self.r1 is not None:
            self.gok = torch.ones((), dtype=torch.bool, device=device)
        self._streams =([torch.cuda.Stream() for _ in models]
                         if cuda and PAR_STREAMS and len(models) > 1 else None)
        self.graph = None
        if cuda and use_graph:
            self._capture()

    # -- one fused group's forward, member-weighted: the serial loop's per-group ops ---------
    @staticmethod
    def _group(m, pv, err):
        nm = getattr(m, "n_members", 1)
        if _AMP is None:
            o_ = m(pv)
        else:
            with torch.autocast("cuda", dtype=_AMP):
                o_ = m(pv)
        raw, logstd = o_[0].float(), (None if o_[1] is None else o_[1].float())
        ei = None
        if err is not None and len(o_) > 2 and o_[2] is not None:
            ei = o_[2].float().clamp(-8.0, 2.0).exp().reshape(1, T_OUT, 2, H, W) \
                .permute(0, 1, 3, 4, 2) * nm
        yi = raw.reshape(1, T_OUT, 2, H, W).permute(0, 1, 3, 4, 2) * nm
        si = None
        if logstd is not None:
            si = logstd.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
        return yi, si, ei

    # -- the device-side step, written once and used by both paths ----------------------
    def _yt(self, yp):
        """The revealed target in TARGET-normalized space (R1, D1c). PREV_SPACE "input": the exact affine, the op every
        package of record runs; "target": it already is."""
        return yp * self.Ad + self.Bd if self.prev_in else yp

    def _ym(self, yp):
        """The revealed target in MODEL (input-normalized) space, float64 (FAMW). PREV_SPACE "target": the inverse affine."""
        return yp.double() if self.prev_in else (yp.double() - self.Bd.double()) / self.Ad.double()

    def _body(self):
        """The whole device-side step (captured as one CUDA graph): M3 guards around `_step`.

        M3 (2026-09-28, shipping audit): the ingestion program zeroes every subscore on one non-finite element, and the
        running statistics accumulate the revealed target with a 0/1 flag, so a single NaN in one revealed window would
        poison the rest of the trajectory (NaN * 0 is NaN). Head: a revealed target that is not entirely finite updates
        nothing this step (flag -> 0) and is replaced by zeros, so the flag-gated products stay finite. It sits here, at
        the head of the captured step, rather than as eager ops in ttt_step: same effect, no extra host-side launches.
        Tail: a non-finite prediction element becomes 0 and a non-finite bound becomes the prediction, then
        lower <= P <= upper is re-imposed. A non-finite FORWARD (a NaN input window, an fp16 overflow) makes P non-finite,
        so the tail also records that (gok), and `_step` writes the fields the NEXT step scores this window by (gC, gH,
        gC0, gVp) through nan_to_num instead of copy_: that next step then updates nothing, exactly as for a bad target
        (measured 2026-09-28: without this, one NaN input window zeroed every later prediction of its trajectory through
        D1c). On finite values every op is the identity (flag * 1.0, gok True, nan_to_num / where pass them through,
        min/max return the bound because lower <= P <= upper already holds), so the packages of record are unchanged bit
        for bit. Written for few kernels -- in place, full tensors (channel p is 0 in go / gl / gh) -- because at batch 1
        each one is a graph node on the critical path: a first version with ~22 extra kernels cost +0.10 ms/step (A100).
        """
        if self.r1 is not None:
            self.gflag.mul_(torch.isfinite(self.gyp).all() & self.gok)
            self.gyp.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        self._step()
        if self.r1 is not None:
            torch.all(torch.isfinite(self.go), out=self.gok)
        self.go.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if self.half0 is not None:
            torch.where(torch.isfinite(self.gl), torch.minimum(self.gl, self.go), self.go, out=self.gl)
            torch.where(torch.isfinite(self.gh), torch.maximum(self.gh, self.go), self.go, out=self.gh)

    def _step(self):
        r1s = None
        if self.r1 is not None:
            # R1, inside the graph: score the PREVIOUS window (go / gC / gH are not yet overwritten) against the target
            # revealed this step (gyp, valid iff gflag == 1), then set this window's multiplier from past windows only.
            if self.r1_stream is not None:
                r1s = self.r1_stream
                r1s.wait_stream(torch.cuda.current_stream())
            with (torch.cuda.stream(r1s) if r1s is not None else contextlib.nullcontext()):
                yp = self.gyp[..., :2]
                if R1_ASYM:
                    self.r1.update_gated_asym(self.go[..., :2], self.gC, self.gH, self._yt(yp),
                                              (yp - self.r1_z).abs() >= R1_UNSCORED_TOL, self.gflag)
                    k_lo, k_hi = self.r1.kfield_asym()
                    self.gk.copy_(k_lo); self.gk2.copy_(k_hi)
                else:
                    self.r1.update_gated(self.go[..., :2], self.gC, self.gH, self._yt(yp),
                                         (yp - self.r1_z).abs() >= R1_UNSCORED_TOL, self.gflag)
                    self.gk.copy_(self.r1.kfield())
        sm = None
        if STATS_MASK and self.r1 is not None and (self.d1c or self.vc):
            sm = (self.gyp[..., :2] - self.r1_z).abs() >= R1_UNSCORED_TOL          # scored elements of the revealed target
        if self.d1c:
            # Score the PREVIOUS window's uncorrected C against the target revealed this step (gated by gflag, no host branch).
            ef = self.gC0 - self._yt(self.gyp[..., :2])
            if sm is not None:
                ef = ef * sm                          # STATS_MASK: an unscored element carries no error
            e = ef.mean(1, keepdim=True)
            self.dsum.add_(e.double() * self.gflag)
            self.dn.add_(self.gflag)
            if self.d1x:
                n_ = self.dn; nn = torch.clamp(n_, min=1.0)
                raw = self.dsum / nn                                  # n = 0: exactly 0, so every product below is 0
                shr = n_ / (n_ + D1C_N0)
                if D1C_EB:
                    self.dsq.add_(e.double() * e.double() * self.gflag)
                    ss = torch.clamp(self.dsq - nn * raw * raw, min=0.0)
                    se2 = (ss / torch.clamp(nn - 1.0, min=1.0) / nn).mean((2, 3), keepdim=True)
                    tau2 = torch.clamp((raw * raw).mean((2, 3), keepdim=True) - se2, min=0.0)
                    eb = torch.where(n_ > 1.5, tau2 / (tau2 + se2 + 1e-30), torch.zeros_like(tau2))      # n < 2: no variance estimate, EB off
                    if D1C_EB_N0:
                        eb = eb * (n_ / (n_ + D1C_EB_N0))
                    shr = torch.maximum(eb, shr)
                b_ = raw * shr
                if D1C_LEAD_N0:
                    self.dsumf.add_(ef.double() * self.gflag)
                    a_ = ((self.dsumf / nn) * raw).sum((2, 3), keepdim=True) / torch.clamp((raw * raw).sum((2, 3), keepdim=True), min=1e-30)
                    b_ = b_ * torch.clamp(1.0 + (a_ - 1.0) * (n_ / (n_ + D1C_LEAD_N0)), 0.0, 3.0)
                self.gD.copy_((-b_).to(torch.float32))
            else:
                self.gD.copy_(-(self.dsum / (self.dn + D1C_N0)))
        if self.vc:
            # PREVIOUS window: revealed target variance against that window's uncalibrated composed variance (gVp not yet overwritten).
            vt = self.gyp[..., :2].var(1, unbiased=False)[0].double()
            if not self.prev_in:
                vt = vt / (self.Ad.double() * self.Ad.double())          # target space -> model space (PREV_SPACE "target")
            vp = self.gVp[0, 0].double()
            if sm is not None:
                full = sm.all(1)[0]                   # STATS_MASK: (H,W,2) pixel scored in all 20 frames of that window
                vt = vt * full; vp = vp * full
            self.cT.add_(vt * self.gflag)
            self.cP.add_(vp * self.gflag)
            self.cn.add_(self.gflag)
            bP = _r1_box3(self.cP)
            g = torch.clamp(_r1_box3(self.cT) / torch.clamp(bP, min=1e-12), VC_CLIP[0], VC_CLIP[1])
            if sm is not None:
                g = torch.where(bP > 0, g, torch.ones_like(g))                   # no scored window near this pixel: g = 1
            g = 1.0 + (g - 1.0) * (self.cn / (self.cn + VC_N0))          # n = 0 -> exactly 1.0
            self.gG.copy_(torch.sqrt(g).to(torch.float32)[None, None])
        if self.fw:
            # PREVIOUS window's family means (gYA / gYB not yet overwritten) against the target revealed this step.
            d_ = (self.gYB - self.gYA).double()
            self.fA.add_((d_ * d_).sum(1)[0] * self.gflag)
            self.fB.add_((d_ * (self._ym(self.gyp[..., :2]) - self.gYA.double())).sum(1)[0] * self.gflag)
            self.fn.add_(self.gflag)
            w_ = torch.clamp(_r1_box3(self.fB) / torch.clamp(_r1_box3(self.fA), min=1e-12), 0.0, 1.0)
            w_ = self.fw0 + (w_ - self.fw0) * (self.fn / (self.fn + FW_N0))   # n = 0 -> exactly w0
            self.gW.copy_(torch.clamp(w_, 0.0, 1.0).to(torch.float32)[None, None])
        pv =self.gx[..., :2].permute(0, 1, 4, 2, 3).reshape(1, CH, H, W).contiguous()
        y = sig = er = None
        total = etot = 0
        # PAR_STREAMS: each fused group runs on its own side stream (forked from, and joined
        # back into, the current stream), so the groups' kernels can overlap. Every group still
        # executes the same kernels on the same inputs, and the member sums below are done on the
        # current stream in the original order, so the arrays are the serial path's bit for bit.
        outs = []
        if self._streams is not None:
            cur = torch.cuda.current_stream()
            for m, st in zip(self.models, self._streams):
                st.wait_stream(cur)
                with torch.cuda.stream(st):
                    outs.append(self._group(m, pv, self.err))
            for st in self._streams:
                cur.wait_stream(st)
            if not torch.cuda.is_current_stream_capturing():
                for o in outs:                   # eager: side streams' blocks are read on `cur`
                    for t in o:
                        if t is not None:
                            t.record_stream(cur)
        else:
            outs = [self._group(m, pv, self.err) for m in self.models]
        for m, (yi, si, ei) in zip(self.models, outs):
            nm = getattr(m, "n_members", 1)      # a fused group stands in for nm members
            total += nm
            if ei is not None:
                # --err-head: per-frame |error| of the err-bearing members, averaged over them only.
                er = ei if er is None else er + ei
                etot += nm
            y = yi if y is None else y + yi
            if si is not None:
                sig = si if sig is None else sig + si
        if total > 1:
            y = y / total
            sig = None if sig is None else sig / total
        if er is not None and etot > 1:
            er = er / etot
        if self.fw:
            # FAMW: per-pixel causal family mix replaces the uniform member mean (sigma stays the uniform mean of the heads).
            yA = outs[0][0] / getattr(self.models[0], "n_members", 1)
            yB = outs[1][0] / getattr(self.models[1], "n_members", 1)
            self.gYA.copy_(yA); self.gYB.copy_(yB)
            y = yA + self.gW * (yB - yA)
        y0 = y                            # pre-compose mean: the interval centre's anchor
        if sig is not None and ALPHA > 0:
            m_ = y.mean(1, keepdim=True)
            s_ = y.std(1, keepdim=True, unbiased=False)
            sa = sig
            if JOINT_GAIN:
                w_ = (self.std * self.std).float()
                t_ = ALPHA * sa + (1 - ALPHA) * s_
                gj_ = (torch.sqrt((t_ * t_ * w_).sum(-1, keepdim=True))
                       / (torch.sqrt((s_ * s_ * w_).sum(-1, keepdim=True)) + 1e-7))
                if self.vc:
                    # E1 merge (09-27): VARCAL must see the variance of the field JOINT_GAIN actually composes: per channel it is
                    # (s_ * gain)^2, the gain from sigma_head (uncalibrated).
                    torch.nan_to_num((s_ * gj_) ** 2, nan=0.0, posinf=0.0, neginf=0.0, out=self.gVp)          # M3: stored for the next step
                    y = m_ + (y - m_) * (gj_ * self.gG)
                else:
                    y = m_ + (y - m_) * gj_
            elif self.vc:
                torch.nan_to_num((ALPHA * sig + (1 - ALPHA) * s_) ** 2, nan=0.0, posinf=0.0, neginf=0.0, out=self.gVp)   # uncalibrated composed variance of THIS window (read next step; M3)
                y = m_ + (y - m_) * (((ALPHA * sa + (1 - ALPHA) * s_) * self.gG) / (s_ + 1e-6))
            else:
                y = m_ + (y - m_) * ((ALPHA * sa + (1 - ALPHA) * s_) / (s_ + 1e-6))
        if HIST_BETA:
            # Unconditional tensor arithmetic: ggate is 0 for an in-range window, so this is
            # an identity there and no branch is needed inside the graph.
            y = y + (HIST_BETA * self.ggate) * (self.ghm - y.mean(1, keepdim=True))
        if r1s is not None:
            torch.cuda.current_stream().wait_stream(r1s)     # R1 has finished reading go / gC / gH and writing gk
        if self.d1c:
            torch.nan_to_num(y0 * self.Ad + self.Bd, nan=0.0, posinf=0.0, neginf=0.0, out=self.gC0)           # M3: stored for the next step
            self.go[..., :2] = y * self.Ad + self.Bd + self.gD
        else:
            self.go[..., :2] = y * self.Ad + self.Bd
        if self.half0 is None:
            return
        half = self.half0
        if self.err is not None and er is not None and self.err["mode"] == "direct":
            # Head as the band itself: physical k*H, moved into the reported (target-normalized) space.
            half = (self.err["k"] * (er * self.std)) / self.std_tgt
            if self.floor is not None:
                half = torch.maximum(half, self.floor)
        else:
            if self.band_p and self.sref is not None and sig is not None:
                half = half * torch.clamp((sig * self.std) / self.sref, 0.3, 3.0) ** self.band_p
            if self.sigpow is not None:
                if sig is None:
                    raise RuntimeError("SIGPOW needs the sigma head (sig is None); refusing")
                half = half * ((sig * self.std).float() + 1e-8) ** self.sigpow
            if self.err is not None and er is not None and self.eref is not None and self.err["q"]:
                half = half * torch.clamp((er * self.std) / self.eref, 0.3, 3.0) ** self.err["q"]
        half_hi = None
        if self.r1 is not None:
            torch.nan_to_num(half.expand_as(self.gH), nan=0.0, posinf=0.0, neginf=0.0, out=self.gH)           # M3: stored for the next step
            if self.gk2 is not None:
                half_hi = half * self.gk2
            half = half * self.gk
        if half_hi is None:
            half_hi = half                   # symmetric (R1 / no R1): the same tensor, so the ops below are unchanged
        if CENTRE_ALPHA == ALPHA or sig is None or HIST_BETA:
            if self.r1 is not None:
                torch.nan_to_num(self.go[..., :2], nan=0.0, posinf=0.0, neginf=0.0, out=self.gC)
            self.gl[..., :2] = self.go[..., :2] - half
            self.gh[..., :2] = self.go[..., :2] + half_hi
            return
        # Decoupled centre, in the same normalized output space as go: c = y0 + k*(y - y0).
        # Inside the graph this is 3-4 tiny batch-1 kernels. min/max keep go inside [gl, gh].
        c = y0 * self.Ad + self.Bd
        if self.d1c:
            c = c + self.gD                  # centre moves with P ("PC"): c = (C + D) + k * ((P + D) - (C + D))
        if CENTRE_ALPHA != 0.0:
            c = c + (CENTRE_ALPHA / ALPHA) * (self.go[..., :2] - c)
        if self.r1 is not None:
            torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0, out=self.gC)                                 # M3: stored for the next step
        self.gl[..., :2] = torch.minimum(c - half, self.go[..., :2])
        self.gh[..., :2] = torch.maximum(c + half_hi, self.go[..., :2])

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
            worst = max(float((a - b).abs().max()) / max(float(b.abs().max()), 1e-12)
                        for a, b in zip((self.go, self.gl, self.gh), ref))
            global GRAPH_WORST
            GRAPH_WORST = worst
            self.graph = g if worst <= GRAPH_TOL else None
            if self.r1 is not None:                  # warm-up / probe ran with gflag 0 (no update), but be exact
                self.r1.reset(); self.gk.fill_(1.0); self.gflag.fill_(0.0); self.gok.fill_(True)
                if self.gk2 is not None:
                    self.gk2.fill_(1.0)
            if self.d1c:
                self.dsum.zero_(); self.dn.zero_(); self.gD.zero_()
            self._reset_c()
        except Exception as e:
            global GRAPH_ERR
            GRAPH_ERR = repr(e)[:300]
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
        if self.r1 is not None:
            self.r1.reset()
            self.gk.fill_(1.0)
            if self.gk2 is not None:
                self.gk2.fill_(1.0)
            self.gflag.fill_(0.0)
            self.gok.fill_(True)
            self.r1_prev = False
        if self.d1c:
            self.dsum.zero_(); self.dn.zero_(); self.gD.zero_()
        self._reset_c()

    def _reset_c(self) -> None:          # VARCAL / FAMW / D1c-EB per-trajectory state (design C); no-op when all are off
        if self.d1x:
            self.dsq.zero_(); self.dsumf.zero_()
        if self.vc:
            self.cT.zero_(); self.cP.zero_(); self.cn.zero_(); self.gG.fill_(1.0)
        if self.fw:
            self.fA.zero_(); self.fB.zero_(); self.fn.zero_(); self.gW.fill_(self.fw0)

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
        if self.r1 is not None:
            # Hand the revealed target to the graph; _body scores the previous window with it (flag 1) or skips (flag 0).
            if prev_target_norm is not None and self.r1_prev:
                self.gyp.copy_(torch.as_tensor(prev_target_norm), non_blocking=True)
                self.gflag.fill_(1.0)
            else:
                self.gflag.fill_(0.0)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._body()
        if self.r1 is not None:
            self.r1_prev = True
        srcs = (self.go,) if self.half0 is None else (self.go, self.gl, self.gh)
        out = self._fetch(srcs, x.is_cuda)
        if NUMPY_OUT and not x.is_cuda:
            out = [o.numpy() for o in out]
        if DIAG and not getattr(self, "_diag_step", False):
            self._diag_step = True
            print(f"[rpde-t2] first ttt_step: input {tuple(x.shape)} {x.dtype} on {x.device}; "
                  f"graph {'replay' if self.graph is not None else 'EAGER'}", flush=True)
        if self.half0 is None:
            return out[0], {"adapt_loss": None}
        return out[0], {"adapt_loss": None, "lower": out[1], "upper": out[2]}

    def _fetch(self, srcs, want_cuda: bool):
        """Return tensors the caller OWNS, on the device it handed us the input on.

        Two things this gets right that the obvious version does not.

        **Never hand back a buffer that the next step overwrites.**  The evaluator keeps every
        prediction until the run ends, so returning go/gl/gh (or one reused staging buffer)
        makes all N predictions equal to the last one -- rel-L2 0.67 instead of 0.074.  Every
        local harness hides it, because the kit's `local_eval` and our `stream_eval` both copy
        out of the returned tensor on the same line.  `validate_t2.py` asserts it, and caught
        exactly this on its first GPU run.  Each step therefore gets storage of its own: for a
        CPU input, its own slot of the pinned arena (ARENA_STEPS, allocated at load -- the default
        path of the final package, no clone), and once the arena is used up (or could not be
        allocated) a clone of the pinned staging buffer; for a CUDA input, a device clone.
        (Docstring corrected 2026-09-28: it used to say the clone is not optional.)

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
        if self.arena is not None and self.arena_i < int(ARENA_STEPS):
            j = self.arena_i
            self.arena_i = j + 1
            outs = [self.arena[i][j] for i in range(len(srcs))]
            for o, s in zip(outs, srcs):
                o.copy_(s, non_blocking=True)
            torch.cuda.synchronize()
            return outs
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
    global CENTRE_ALPHA, _ERR
    CENTRE_ALPHA = float(z["centre_alpha"]) if "centre_alpha" in z.files else ALPHA
    if JOINT_GAIN:
        assert CENTRE_ALPHA == 0.0, "JOINT_GAIN needs CENTRE_ALPHA == 0 (compose is no longer linear in alpha)"
    # --err-head band (2026-09-23). Absent keys => None => the band is exactly the old one.
    #   err_mode "drv":    half = base * clip(sigma/sigma_ref)^p * clip(H/err_ref, .3, 3)^err_q
    #   err_mode "direct": half = err_k * H          (H = heads' predicted |error|, physical)
    _ERR = None
    if "err_mode" in z.files:
        mode = str(z["err_mode"])
        er = z["err_ref"].astype(np.float32) if "err_ref" in z.files else None
        ok = mode in ("drv", "direct") and (mode == "direct" or (er is not None and np.all(er > 0)))
        if ok:
            _ERR = {"mode": mode, "q": float(z["err_q"]) if "err_q" in z.files else 0.0,
                    "k": float(z["err_k"]) if "err_k" in z.files else 1.0, "ref": er,
                    "floor": float(z["err_floor"]) if "err_floor" in z.files else 0.0}
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
                r0, l0 = mdl(pv)[:2]
                with torch.autocast("cuda", dtype=_AMP):
                    r1, l1 = mdl(pv)[:2]
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
    if dev.startswith("cuda") and torch.device(dev).index is not None:
        # m1 (2026-09-28): the side streams, the graph capture and every torch.cuda.synchronize() in this file act on the
        # CURRENT device. With device "cuda:N" != current they would target the wrong GPU, and _fetch's synchronize would not
        # wait for the arena copy before the host reads it. Make the requested device current once, here (untimed).
        torch.cuda.set_device(torch.device(dev))
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
        m.arena_i = 0
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception:
        pass
    if DIAG:
        try:
            aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            gpu = torch.cuda.get_device_name(torch.device(dev)) if dev.startswith("cuda") else "cpu"
            print(f"[rpde-t2] load: gpu {gpu!r} cpus {os.cpu_count()} affinity {aff} torch {torch.__version__} "
                  f"cuda {torch.version.cuda} amp {AMP_ACTIVE} par_streams {PAR_STREAMS} r1 {R1} r1_asym {R1_ASYM} r1_tsm {R1_TSM} r1_blk_n0 {R1_BLK_N0} r1_blk {R1_BLK} d1c_n0 {D1C_N0} d1c_eb {D1C_EB} d1c_eb_n0 {D1C_EB_N0} d1c_lead_n0 {D1C_LEAD_N0} vc_n0 {VC_N0} fw_n0 {FW_N0} sigpow {SIGPOW} prev_space {PREV_SPACE} stats_mask {STATS_MASK} alpha {ALPHA} joint_gain {JOINT_GAIN} centre_alpha {CENTRE_ALPHA} "
                  f"graph {'CAPTURED' if m.graph is not None else 'NONE'} graph_worst {GRAPH_WORST} graph_tol {GRAPH_TOL} "
                  f"graph_err {GRAPH_ERR} numpy_out {NUMPY_OUT} members {len(ckpts)}", flush=True)
        except Exception as e:
            print(f"[rpde-t2] load report failed: {e!r}", flush=True)
    return m
