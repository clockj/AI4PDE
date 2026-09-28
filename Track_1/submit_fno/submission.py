"""RealPDE Track 1 submission: a 6-member ensemble (4 F-FNO + 2 ViT) with a sigma-scaled SPS band.

This is a plain forecaster. It reads only its own input window, uses no cross-window
information, and would score the same if the evaluator called it one window at a time.

Zip layout (all at the archive ROOT):

    submission.py      <- this file (constants baked by build_round13.sh / build_final.sh)
    _fno.py            <- self-contained model code (FFNO2d, ViT2d, their fused stacks, predict_fno)
    fno.pth, fno_2..6.pth  <- the members, int8-packed; each blob names its own architecture
    band.npz           <- the SPS band: per-element `base`, `sigma_ref`, `p`, `centre_alpha`

Needs only torch + numpy, both present in the evaluation image
(pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime). It imports no scipy / h5py / matplotlib /
pandas / einops, none of which exist there.

THE MODEL (final package of record, 2026-09-27; this docstring refreshed 2026-09-28)
---------
fno.pth .. fno_4.pth are four F-FNO members (w40, 8 layers, modes 16 x 32, trained with sim
co-training) and fno_5 / fno_6 two ViTs (d384, 8 layers, 6 heads). `_fno.fuse` folds them into
two invocations, a StackedFFNO(4) and a StackedViT(2). Each member has a field head and a
per-pixel log-std (sigma) head.

WHAT predict() DOES (GPU, the shipped path -- `_fno.predict_fno` with `band_gpu`)
------------------
Per chunk of at most BATCH_CAP = 56 windows:
  1. upload, normalize with _fno.MEAN / STD, run both groups under fp16 autocast (AMP, verified
     against fp32 at import), and average the six members uniformly -> y0 (field) and sigma;
  2. compose: rescale each pixel's temporal fluctuation toward ALPHA*sigma + (1-ALPHA)*std_t(y0),
     ONE gain per pixel for u and v (JOINT_GAIN), ALPHA = 0.80 -> the prediction P;
  3. band: half = base * clip(sigma/sigma_ref, 0.3, 3)^p * (sigma + 1e-8)^q_c (EBAND, the
     SIGBLEND factor), centred on y0 (CENTRE_ALPHA 0) and widened to contain P:
     lower = min(y0 - half, P), upper = max(y0 + half, P); pressure bounds are 0;
  4. a non-finite element is sanitized (P -> 0, bound -> P) -- the identity on finite values;
  5. P and the bounds are written straight into caller-owned (pinned, PIN_OUT) host buffers.
Everything load-time (members, fuse, the AMP probe, the batch pick, the pinned pool, warm-up of
every tail shape) runs at import, which the evaluator does not time. Without CUDA the same model
runs in fp32 on the CPU and the band is finished on the host (`_predict`'s host path).

WHAT IT IS NOT
--------------
Track 1 tiles its windows at a stride equal to the output horizon and passes them all to
predict() at once, so window k's target is window k+1's input and a submission can read the
answers out of its own argument. This file deliberately does not. See Track_1/archive/submit_trick/
for the measurement of that leak and the disclosure sent to the organizers.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Put our own directory on sys.path BEFORE importing the sibling module. The platform may load
# this file with importlib.util.spec_from_file_location, which does NOT add the containing
# directory to sys.path, and a bare `import _fno` would then fail at import time.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

import _fno  # noqa: E402  (must follow the sys.path insertion above)

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

# ---------------------------------------------------------------------------------------
# ROUND 17 Q1g FAST PATHS, BAKED IN AS CONSTANTS.
#
# `_fno` reads these from the environment as a convenience for the measurement drivers, but the
# EVALUATION CONTAINER SETS NO ENVIRONMENT VARIABLES, so a package must state them here. These
# two assignments are what actually ship; `build_round13.sh` rewrites the constants with `sed`
# exactly as it rewrites ALPHA, and `verify/pkg_identity.py` proves the built archive returns
# the same arrays as the one already on the board.
#
# HOST_WRITE = "torch"  writes the strided output through torch's `copy_` instead of numpy's
#   assignment. BIT-IDENTICAL -- same bytes to the same place -- but numpy's strided assignment
#   is SINGLE-THREADED (1.4 GB/s measured) where torch's is not (29.0 GB/s), and this write is
#   22.8 % of the shipped 4-member path. Worth -16.8 % ms/window at 4 members, -9.9 % at 8
#   (job 7161342, contended). This is round 11's band-stage finding applied to the write three
#   lines above it, which it never touched.
#
# SPEC_PATH = "bmm"  runs the spectral product as one batched REAL 2x2-block matmul instead of
#   two complex einsums over strided slices. NOT bit-identical (~2e-06 relative: a complex
#   einsum and a real block matmul accumulate in different orders). It is a TRACK-2 change:
#   at batch 1 it deletes einsum's per-call strided WEIGHT copy, which the profiler puts at
#   27.6 % of a Track-2 step, and is worth -16.6 % there -- while on Track 1's batched path it
#   is +1.6 % SLOWER, because the copy it removes is O(1) in batch and the `cat` it adds is
#   O(batch). One switch, opposite signs on the two tracks; set it per track, never globally.
# (2026-09-28) Both are INERT in the final package: the GPU-band path writes the prediction with one direct copy
# (D2_DIRECT_PRED) and never reaches the HOST_WRITE branch, and SPEC_PATH steers only FNO2d stacks, of which the final
# member set has none. Kept because the recipes before it (reproduce.sh r25) do use them.
HOST_WRITE = "torch"
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
_fno._HOSTW = HOST_WRITE
_fno._SPEC = SPEC_PATH

# AMP is BAKED like FUSE (round 25). Round 24 measured the board charging GPU compute at ~1.75
# board ms per 4060-equivalent local ms on Track 1, so a precision that cuts compute without
# moving the error subscores is worth ~+0.07 `final` per board ms. r25a: fp16 autocast moved the
# prediction 6e-4 relative, left every error metric identical to 4-5 decimals and kept
# permutation invariance EXACT; bf16 crashes (torch.complex has no BFloat16). None is the fp32
# path every earlier round shipped. A non-None value is VERIFIED against fp32 at import, which
# is untimed, and dropped on any exception, non-finite output or disagreement above 2e-2 on a
# probe -- a kernel that is broken on the board GPU, not a 1e-3 numerics difference, which is
# priced offline on real data. `AMP_ACTIVE` records what actually runs. RPDE_AMP still steers
# the measurement drivers when AMP is None.
AMP = None
_AMP_DT = {"fp16": torch.float16}.get(AMP) if AMP is not None else _fno._AMP
AMP_ACTIVE = None
AMP_PROBE_DIFF = None

# PRUNED 2026-09-28 (none ever shipped in a package of record, and no reproduce.sh recipe sets them; the last tree that has
# them is commit fdac00b, and build_round13.sh now REFUSES their RPDE_SHIP_* variables at a non-default value):
#   GRAPH / GRAPH_FP / _Graphed / _capture_graph  -- R26/R27's CUDA graph of predict()'s chunk (+0.565 board ms, closed);
#   SMALL_GRAPH / SMALL_N / SMALL_GRAPH_LAZY       -- the L3 batch-1 graph for tiny calls (09-26 arms);
#   BATCH_FP / GPU_FP                              -- the audit-2 batch-size fingerprints on the time mean;
#   LN_MANUAL                                      -- K1's hand-written LayerNorm (not bit-identical; _fno.LN_MANUAL stays
#                                                     False and its code is left in _fno.py, unreachable from here);
#   the legacy band.npy fallback                   -- July packages; every package since carries band.npz.
#
# GPU_BAND (D2, 2026-09-25): finish the SPS interval on the GPU inside predict_fno's chunk loop (see _fno.predict_fno
# band_gpu) and hand a float64 input to the GPU unconverted. Measured on one CPU core (d2_profile run 20260925-043642):
# the host-side band cost 0.59 ms/window and a float64->float32 host conversion 0.17. Used whenever CUDA is available and
# the band is a sigma-scaled static band (no err head, no blend, centre = pred or CENTRE_ALPHA 0). Otherwise the host path
# in _predict runs; with EBAND set it applies the same factor there when EBAND is logsig-only (M4, 2026-09-28: the shipped
# case, e.g. a container without CUDA) and raises for any other EBAND, which only the GPU path implements.
GPU_BAND = True
_BAND_DEV = None
# DIAG (N1, audit 2, 09-25): one line of load-time facts on stdout at the end of _load() (NOT timed) -- GPU, CPUs, torch/
# CUDA, batch, AMP, band path -- and one line on the first predict() call (N, dtype). Changes no array.
DIAG = True
_DIAG_CALLS = 0
# EBAND (09-27 T1 arm, #Leaders task #2): None = off. Else (beta, mode, clip): beta = ((6 floats for u), (6 for v)) over the
# five band_feats.py features + intercept, mode "mult" (half = static half * exp(yhat); variant B) or "replace"
# (half = exp(yhat)), clip = (lo, hi) on yhat or None. Centred on the alpha-0 field; the prediction is untouched. The
# final package bakes the SIGBLEND literal (features 1..4 zero: half *= (sigma + 1e-8)^beta_c0). Baked by
# build_round13.sh from RPDE_SHIP_EBAND.
EBAND = None
_EBAND_DEV = None
# WARM_TAILS (Corin / Ilse 09-26): cold-start fix. predict() sees one real input, whose last chunk has N mod batch rows -- a shape
# nothing at import has run (cuFFT plans, lazy CUDA module loads, cuDNN/AMP kernel choice), so its first-use cost lands INSIDE the
# timed call. True: at import, run predict() once for every row count 1..batch, in float32 and float64 (not timed). Outputs of the
# real call are unchanged bit for bit (same kernels, same inputs); only untimed load time grows.
WARM_TAILS = False
# BATCH_CAP (K1, 09-27): > 0 caps the chunk _pick_batch chose. On the A100 the probe picks 112 (one chunk for every
# N <= 99) but the shipped path at N 99 runs 1.59 ms/window in one chunk against 1.37 in chunks of 56; N <= 56 is unchanged.
# Same bits (md5 at N 99). Build knob RPDE_SHIP_BATCH_CAP.
BATCH_CAP = 0
_WARMING = False

_BAND = os.path.join(_HERE, "band.npz")
_ERR = None       # err-head band settings read from band.npz by _load(); None = no head term

# fno.pth, then fno_2.pth, fno_3.pth ... as many as were packaged, averaged in raw field space
# inside _fno.predict_fno. The names are positional only -- build_zip.py numbers members in the
# order it was given them and `_fno.load_any` reads each one's real architecture out of the
# blob, so the list may be a MIXED ensemble. The final package (09-27) is fno.pth..fno_4.pth = four F-FNO w40 L8 and
# fno_5 / fno_6 = two ViT d384 L8 (comment refreshed 2026-09-28).
#
# Ensembling looked NEGATIVE for four rounds (-0.07 / -0.10) because latency was accounted as
# forward-pass-only, where a 2nd member nearly doubles the cost. End to end, the forward pass is
# 0.109 ms of 1.383 and the other ~1.27 ms is fixed API overhead that a 2nd member does not
# duplicate -- so a member costs ~+0.11 ms, not a doubling. Measured on val_unseen: 2 members
# +0.10 final, 3 members +0.12.
_CKPTS = [p for p in [os.path.join(_HERE, "fno.pth")]
          + [os.path.join(_HERE, f"fno_{i}.pth") for i in range(2, 17)] if os.path.exists(p)]

# Fluctuation rescaling strength. 0 returns the raw field; 1 forces the output's per-pixel
# temporal std to equal the variance head's prediction.
#
# 0.75 was carried for five rounds. It is not the optimum. `../gain_v9.py` sweeps the full
# two-parameter family g = (a*sigma + b*s)/s -- of which alpha is the line a + b = 1 -- on the
# 7-member held-out ensemble, fit on val_seen AoA 15-20 and reported on val_unseen AoA 15-20 at
# the board's 1.39x. The ridge does lie on that line, so five rounds of sweeping one number lost
# nothing, but the optimum on it is a broad plateau over alpha 0.85-0.95, worth +0.098 wfinal
# against 0.75. It trades rel-L2 (93.16 -> 93.02) and sps (36.76 -> 36.46) for TKE
# (75.31 -> 76.05), and stays positive across the whole admissible range of the sps weight.
# Round 10 saw the same thing at +0.041..0.070 and called it noise; three readings later it
# is not noise.
#
# If this changes, fit_band.py --alpha MUST change with it: both sigma_ref and the error field
# the band is solved against have to be measured at the alpha the package actually deploys.
ALPHA = 0.60

# SPS interval CENTRE, decoupled from the point prediction (2026-09-21, Bram's pre-registered
# arm: centre at the un-rescaled member mean, +0.61 sps held out, 8/8 trajectories, z 9.1).
# The scorer only reads lower <= target <= upper and lower <= upper, so the interval may sit
# anywhere; it is still widened to contain `prediction` (kit smoke test, our validators).
# compose is linear in alpha, so the centre at CENTRE_ALPHA is y0 + (CENTRE_ALPHA/ALPHA)*(pred-y0)
# with y0 the pre-compose mean. CENTRE_ALPHA == ALPHA reproduces the old band bit for bit.
# The value is READ FROM band.npz (`centre_alpha`, written by fit_band.py --centre-alpha), so the
# centre can never disagree with the centre the band was fitted at; no key => ALPHA (legacy).
CENTRE_ALPHA = ALPHA
# JOINT_GAIN (K2 09-27): one compose gain per pixel for both channels (see _fno.JOINT_GAIN, which _load() sets to this value).
# Requires CENTRE_ALPHA == 0 (asserted at load). False = the per-channel compose, bit for bit.
JOINT_GAIN = False



# ---------------------------------------------------------------------------------------
# GATED MEAN-BLEND.  Replace BLEND_BETA of the predicted time-mean with the INPUT window's own
# measured time-mean -- but ONLY for windows whose mean streamwise speed falls outside the
# training range, ramping in over BLEND_REF of that range.  BLEND_BETA = 0 disables it and the
# code below costs one mean() over the input.
#
# Round 6 closed this idea: blending made MVPE monotonically worse (0.0698 -> 0.1240), because
# in distribution the model's learned mean beats the input window's by a lot.  Round 13 found
# why that is the wrong test.  Decomposing two IDENTICAL-recipe controls on the Re-extrapolation
# split, the mean part of the error differs by 47 % between seeds while the fluctuation part
# differs by 4 %: **out of distribution the fluctuation transfers and the mean field does not.**
# And the input window's time-mean is a MEASUREMENT, so it cannot be wrong about a Reynolds
# number it has never seen in the way a learned mean can.
#
# Measured at 4 members (mean_blend_re.py --ens --gate, ALPHA 0.60), which is the gate that
# killed the oscillatory head -- a single-model gain that an ensemble already averages away is
# not a gain:
#
#     set     beta_eff   rel-L2            mean part          MVPE
#     rein    any        0.0698 (unchanged, BIT-IDENTICAL at every beta and every ref)
#     relo    ~0.15      0.1080 -> 0.1035  0.0785 -> 0.0727   0.1135 -> 0.1072
#     rehi    ~0.15      0.1137 -> 0.1120  0.0638 -> 0.0611   0.1040 -> 0.1025
#
# The optimum is in the PRODUCT beta x gate-strength -- flat over beta_eff 0.14-0.20 on both
# extrapolation sets -- which is why BLEND_REF is small.  An all-data model's speed range is
# wide, and the lowest Re the organizers have shown us (2968, the Track-2 example file) sits
# only 2.09 % outside it, so a wide ramp would deliver a fraction of the measured effect.  At
# REF = 0.02 that window gets gate 1.0, i.e. beta_eff = BLEND_BETA.
#
# TRAIN_SPEED IS MEASURED OVER WINDOWS, NOT TRAJECTORIES, and that distinction is not cosmetic.
# The first version used the per-TRAJECTORY mean range (0.03942-0.29207), which is what the
# offline experiment's `train_speed_range` computes -- but the gate is evaluated per WINDOW,
# and windows fluctuate around their trajectory's mean, so the fastest trajectory has windows
# above its own mean.  The gate then fired at 0.56 of full strength on real validation windows
# and the safety claim below was FALSE.  `verify/blend_path.py` caught it; nothing else could
# have, because `validate_strict.py` scores the package without ever leaving the training
# range.  Over the 3547 training windows the range is 0.03804-0.29634 m/s, and both local
# validation sets sit strictly inside it (max gate 0.0000).
#
# Safe by construction, and that is the point: a window inside the training range gets exactly
# the unblended prediction, bit for bit.  If the hidden set is entirely in range this submission
# returns the previous one's four error subscores unchanged -- which is itself the measurement.
# DEFAULT OFF, deliberately.
# SUPERSEDED (2026-09-23): the sentence below said "Round 14 ships 0.15 via `RPDE_BETA=0.15`".
# That was true of round 14 and is NOT true of the package of record, which ships BLEND_BETA 0 --
# see CLAUDE.md:59, CLAUDE.md:294 and bands/MANIFEST.md:114, all three of which say 0. The stale
# line sat directly above the constant, so a reader met it before the value; kept here rather than
# deleted so the history stays legible, but it is dead. (Found by @IcanFly.)
# STILL LIVE, and the reason the default stays off: the board has not ruled on this arm. Every
# number supporting it is local, and its whole case rests on the hidden set actually containing
# windows outside the training speed range -- which R14's gate found it does NOT (it fired on zero
# hidden windows; CLAUDE.md:144-145). Flip this default only after a board round says so.
BLEND_BETA = 0.0
BLEND_REF = 0.02
TRAIN_SPEED = (0.037523, 0.296857)    # per-WINDOW mean streamwise speed, all 82 traj
# ... padded by 0.2 % of the span on each side. The measured range is 0.03804-0.29634, and
# the fastest validation window IS the fastest training window, so its float32 mean lands a
# few ULP above a float64 bound and the gate fires at 0.000248 -- harmless in magnitude but
# it breaks the bit-identity property the whole design rests on. Pad, do not threshold.


def _blend_gate(x):
    """Per-window blend weight in [0, 1]; 0 inside the training speed range."""
    s = x[..., 0].mean(axis=(1, 2, 3))
    lo, hi = TRAIN_SPEED
    d = np.maximum(np.maximum(lo - s, s - hi), 0.0) / max((hi - lo) * BLEND_REF, 1e-9)
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def _apply_blend(pred, x):
    """In-place gated mean-blend on the measured channels. Per-window, so order-independent."""
    if not BLEND_BETA:
        return
    g = _blend_gate(x)
    if not g.any():                   # in distribution this is the whole cost: one mean()
        return
    b = (np.float32(BLEND_BETA) * g).reshape(-1, 1, 1, 1, 1)
    pred[..., :2] += b * (x[..., :2].mean(1, keepdims=True)
                          - pred[..., :2].mean(1, keepdims=True))

def _amp_verified(models, dev):
    """The autocast dtype predict() may use: `_AMP_DT` if it reproduces fp32 on a probe, else None."""
    global AMP_PROBE_DIFF
    if _AMP_DT is None or dev != "cuda":
        return None
    try:
        g = np.random.default_rng(0)
        probe = np.zeros((8, 20, 32, 64, 3), dtype=np.float32)
        probe[..., :2] = (_fno.MEAN + _fno.STD * g.standard_normal((8, 20, 32, 64, 2))).astype(np.float32)
        s0 = np.empty((8, 1, 32, 64, 2), dtype=np.float32)
        s1 = np.empty_like(s0)
        p0 = _fno.predict_fno(models, probe, dev, alpha=ALPHA, batch=8, sigma_out=s0, amp=None)
        p1 = _fno.predict_fno(models, probe, dev, alpha=ALPHA, batch=8, sigma_out=s1, amp=_AMP_DT)
        if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(s1))):
            return None
        d = max(float(np.linalg.norm(p1 - p0) / max(float(np.linalg.norm(p0)), 1e-12)),
                float(np.linalg.norm(s1 - s0) / max(float(np.linalg.norm(s0)), 1e-12)))
        AMP_PROBE_DIFF = d
        return _AMP_DT if d < 2e-2 else None
    except Exception:
        return None


def _pick_batch(models, dev):
    """Time a few batch sizes on THIS machine and keep the fastest.

    Called from _load(), i.e. at import, which the evaluator does not time -- so this is free.
    It is measured rather than hardcoded because the optimum is not portable: on the 8 GB card
    this was developed on, batch 12 runs at 1.657 ms/window against batch 64's 2.135, because
    the fused width-24 group's intermediates stop fitting in cache. That ordering is a property
    of one card's memory hierarchy. On a larger GPU the fixed per-launch cost (which falls as
    1/batch) should dominate instead and push the optimum back up, so guessing either way risks
    giving back more than the ~0.5 ms it is worth. Any failure falls back to 64, the value four
    rounds of board results were measured with. _load() then clips the result to BATCH_CAP (56 in
    the final package, 2026-09-27), so on the A100 every chunk of the shipped path is <= 56 rows.
    """
    if dev != "cuda":
        return 64
    try:
        import time as _time
        probe = np.zeros((112, 20, 32, 64, 3), dtype=np.float32)
        best_t, best_b = None, 64
        for b in (14, 28, 56, 112):
            _fno.predict_fno(models, probe[:b], dev, alpha=ALPHA, batch=b, amp=_AMP_DT)  # warm up
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            for _ in range(2):
                _fno.predict_fno(models, probe, dev, alpha=ALPHA, batch=b, amp=_AMP_DT)
            torch.cuda.synchronize()
            t = _time.perf_counter() - t0
            if best_t is None or t < best_t:
                best_t, best_b = t, b
        return best_b
    except Exception:
        return 64


def _load():
    """Load once at import, so the evaluator's timing of predict() excludes it."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if JOINT_GAIN:
        _fno.JOINT_GAIN = True          # predict_fno's compose (the GPU-band / direct-pred path) follows this file's constant
    # fuse() collapses same-architecture members into one grouped invocation (5 width-24 seeds
    # + w48 + m1220 + w64 goes from 8 invocations to 4). It is score-identical -- 78.4925
    # against 78.4927 on the AoA 15-20 proxy, and exact to 1.5e-8 with TF32 disabled -- so the
    # only thing it changes is latency.
    #
    # Which way it changes latency depends on a model of board time that three data points do
    # not settle. Regressing the three v9 rounds (U-Net x1 2.648 local / 5.976 board; FNO x3
    # 1.282 / 5.931; FNO x8 2.211 / 8.020):
    #     board ~ invocations : slope +0.316 ms, residuals +-0.4 ms
    #     board ~ local ms    : slope +0.380,    residuals +-1.3 ms
    # The invocation model fits three times better. Under it, 8 -> 4 invocations saves 1.26
    # board ms and is worth about +0.05 final. Under the local-time model, fusing costs
    # +0.475 ms/window locally and is worth about -0.007. Favourable but not free, and the
    # real reason to take it is that it makes ADDITIONAL same-architecture members cost zero
    # invocations.
    #
    # load_any, not load: round 12's package is a MIXED ensemble (8 FNO + 1 U-Net) and the
    # architecture is read off each checkpoint's own `arch` field, not off its packaged
    # filename -- build_zip.py names every member fno*.pth regardless of family.
    models = [_fno.load_any(p, dev) for p in _CKPTS]
    if FUSE:
        models = _fno.fuse(models)
    # band.npz carries `base` (per-element half-widths), `sigma_ref` (the per-element mean of
    # the variance head's sigma over the fitting split) and `p`. The half-width actually used
    # is base * (sigma_i / sigma_ref)**p, i.e. a per-WINDOW interval: windows the model itself
    # expects to be turbulent get a wider band. (The July band.npy fallback was removed 2026-09-28.)
    band = ref = None
    p = 0.0
    if os.path.exists(_BAND):
        z = np.load(_BAND)
        band, ref, p = (z["base"].astype(np.float32), z["sigma_ref"].astype(np.float32),
                        float(z["p"]))
        global CENTRE_ALPHA, _ERR
        CENTRE_ALPHA = float(z["centre_alpha"]) if "centre_alpha" in z.files else ALPHA
        if JOINT_GAIN:
            assert CENTRE_ALPHA == 0.0, "JOINT_GAIN needs CENTRE_ALPHA == 0 (compose is no longer linear in alpha)"
        # --err-head band (2026-09-23), same keys and rules as Track_2/submit/submission.py.
        #   err_mode "drv":    half = base * clip(sigma/sigma_ref)^p * clip(H/err_ref, .3, 3)^err_q
        #   err_mode "direct": half = err_k * H          (H = heads' predicted |error|, physical)
        # Absent keys => _ERR None => the band is bit-identical to before.
        _ERR = None
        if "err_mode" in z.files:
            mode = str(z["err_mode"])
            er = z["err_ref"].astype(np.float32) if "err_ref" in z.files else None
            if mode in ("drv", "direct") and (mode == "direct" or (er is not None and np.all(er > 0))):
                _ERR = {"mode": mode, "q": float(z["err_q"]) if "err_q" in z.files else 0.0,
                        "k": float(z["err_k"]) if "err_k" in z.files else 1.0,
                        "floor": float(z["err_floor"]) if "err_floor" in z.files else 0.0,
                        "ref": None if er is None else er[..., :2]}
    # The ingestion program rejects a non-finite or reversed interval by zeroing EVERY
    # subscore, so the band is checked -- but here, at import, where it is 0.5 MB and free,
    # not inside the timed call where the same guarantee would cost a scan of two 165 MB
    # arrays. With a finite band, subtracting a non-negative half-width cannot reorder the bounds
    # and the sigma scaling is clipped positive -- but that holds only for a FINITE prediction and
    # sigma. The GPU path (the shipped one) sanitizes non-finite elements per chunk (M3, 2026-09-28,
    # _fno.predict_fno); the host path below does not.
    if band is not None and not (np.all(np.isfinite(band)) and np.all(band >= 0.0)):
        band = None
    if band is not None and p and not (ref is not None and np.all(ref > 0.0)):
        p = 0.0
    # The err-head term is live only if (a) some member actually carries a head -- otherwise
    # predict_fno would leave the np.empty buffer unwritten -- and (b) it changes the band.
    # (_ERR is already declared global above, before its first use in this function.)
    if _ERR is not None:
        has_head = any(getattr(m, "head_logerr", None) is not None or getattr(m, "he", None) is not None
                       for m in models)
        if band is None or not has_head or (_ERR["mode"] == "drv" and not _ERR["q"]):
            _ERR = None
        else:
            ref_dev = (None if _ERR["ref"] is None
                       else torch.from_numpy(np.ascontiguousarray(_ERR["ref"])).to(dev))
            _ERR["xform"] = (_ERR["mode"], float(_ERR["q"]), float(_ERR["k"]), ref_dev)
    # Decided BEFORE the batch is timed, so the batch that wins is the one the mode will run.
    global _AMP_DT, AMP_ACTIVE
    _AMP_DT = _amp_verified(models, dev)
    AMP_ACTIVE = None if _AMP_DT is None else str(_AMP_DT).replace("torch.", "")
    batch = _pick_batch(models, dev)
    if BATCH_CAP:
        batch = min(int(batch), int(BATCH_CAP))
    if DIAG:
        try:
            aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            gpu = torch.cuda.get_device_name(0) if str(dev).startswith("cuda") else "cpu"
            print(f"[rpde-t1] load: gpu {gpu!r} cpus {os.cpu_count()} affinity {aff} torch {torch.__version__} "
                  f"cuda {torch.version.cuda} batch {batch} amp {AMP_ACTIVE} gpu_band {GPU_BAND} eband {EBAND is not None} "
                  f"direct_pred {getattr(_fno, 'D2_DIRECT_PRED', None)} alpha {ALPHA} centre_alpha {CENTRE_ALPHA} "
                  f"joint_gain {JOINT_GAIN}/{getattr(_fno, 'JOINT_GAIN', None)} members {len(_CKPTS)}", flush=True)
        except Exception as e:
            print(f"[rpde-t1] load report failed: {e!r}", flush=True)
    return models, dev, band, ref, p, batch


try:
    _state = _load()
except Exception:                 # fall back to a lazy load inside predict()
    _state = None


def _warm_tails():
    global _DIAG_CALLS, _WARMING
    if not (WARM_TAILS and _state is not None and str(_state[1]).startswith("cuda")):
        return
    try:
        _WARMING = True
        b = int(_state[5])
        for dt in (np.float32, np.float64):
            for r in range(1, b + 1):
                predict(np.zeros((r, 20, 32, 64, 3), dtype=dt))
        torch.cuda.synchronize()
        if DIAG:
            print(f"[rpde-t1] warm_tails: ran row counts 1..{b} x (f32, f64) at import", flush=True)
    except Exception as e:
        print(f"[rpde-t1] warm_tails failed (harmless): {e!r}", flush=True)
    finally:
        _WARMING = False
        _DIAG_CALLS = 0


# PIN_OUT (K1, 09-27): > 0 = the three full-size outputs of predict() are consecutive slices of a page-locked arena of
# this many WINDOWS, allocated (hence faulted in) at import, which the evaluator does not time. It removes the fresh
# allocation + first-touch page faults + pageable staging of 1.47 MB per window from the timed call. Every call gets its
# OWN slice, so nothing already returned is overwritten; when the arena is used up (or could not be allocated) the call
# allocates as before. Same bits (md5-identical at N 14/28/56/99, f32 and f64 input; k1_t1lat* 09-27).
# Measured A100 / torch 2.2.2: warm 1.51 -> 1.32 ms/window (N 56); H100 1.06 -> 0.82.  Build knob RPDE_SHIP_PIN_OUT.
PIN_OUT = 0
_POOL = None
_POOL_I = 0


def _pool_init():
    global _POOL
    if not (PIN_OUT and torch.cuda.is_available()):
        return
    try:
        _POOL = [torch.zeros((int(PIN_OUT), 20, 32, 64, 3), dtype=torch.float32).pin_memory() for _ in range(3)]
    except Exception:
        _POOL = None


def _pool_take(n):
    global _POOL_I
    if _POOL is None or _WARMING or _POOL_I + n > int(PIN_OUT):
        return None
    i = _POOL_I
    _POOL_I = i + n
    return tuple(t.numpy()[i:i + n] for t in _POOL)


_pool_init()


def predict(input_array, metadata=None):
    """(N, 20, 32, 64, 3) raw physical -> dict with prediction, lower, upper."""
    out = _predict(input_array, metadata)
    if DIAG:
        global _DIAG_CALLS
        _DIAG_CALLS += 1
        if _DIAG_CALLS == 1:
            print(f"[rpde-t1] first predict: N {len(out['prediction'])} dtype {getattr(input_array, 'dtype', type(input_array))}",
                  flush=True)
    return out


def _eband_host_factor(sig):
    """M4 (2026-09-28): the EBAND factor for the HOST band path, or raise.

    Only a logsig-only EBAND (features 1..4 exactly zero, mode "mult", no clip -- the SIGBLEND literal the final package
    bakes) has a host twin: half_uv *= exp(log(sigma + 1e-8) * beta_c0 + beta_c5), the expression of _fno.predict_fno's
    fast path, evaluated in float32 torch on the CPU. It agrees with the GPU path to float rounding, not bitwise (CPU and
    CUDA exp/log differ in the last ulp). Any other EBAND needs the feature stack that only the GPU path builds, so it
    still raises -- a silently different band would be worse than a loud failure.
    """
    beta = np.asarray(EBAND[0], dtype=np.float32)
    if not (EBAND[1] == "mult" and EBAND[2] is None and beta.shape == (2, 6) and np.all(beta[:, 1:5] == 0.0)):
        raise RuntimeError("EBAND needs the GPU-band path (cuda, static sigma band, no err head / blend) unless it is "
                           "logsig-only ('mult', no clip)")
    b = torch.from_numpy(beta)
    return torch.exp(torch.log(torch.from_numpy(sig) + 1e-8) * b[:, 0] + b[:, 5])


def _predict(input_array, metadata=None):
    global _state
    if _state is None:
        _state = _load()
    models, dev, band, ref, p, batch = _state[:6]

    use_gpu_band = (GPU_BAND and band is not None and _ERR is None and not BLEND_BETA
                    and p and str(dev).startswith("cuda") and band.shape[1:] == (20, 32, 64, 3)
                    and (CENTRE_ALPHA == ALPHA or CENTRE_ALPHA == 0.0))
    if EBAND is not None and not use_gpu_band:
        # M4 (2026-09-28): the host path below applies a logsig-only EBAND itself (a container without CUDA used to raise
        # here on every call). It needs the per-window sigma band, so everything else the GPU path requires but the device
        # must still hold; any other case raises exactly as before.
        if not (band is not None and _ERR is None and not BLEND_BETA and p and band.shape[1:] == (20, 32, 64, 3)
                and (CENTRE_ALPHA == ALPHA or CENTRE_ALPHA == 0.0)):
            raise RuntimeError("EBAND needs the GPU-band path (cuda, static sigma band, no err head / blend)")
        _eband_host_factor(np.ones((1, 1, 1, 1, 2), dtype=np.float32))      # raises now for a non-logsig EBAND
    if use_gpu_band:
        global _BAND_DEV
        if _BAND_DEV is None:
            _BAND_DEV = torch.from_numpy(np.ascontiguousarray(band)).to(dev)
        xin = np.asarray(input_array)
        # float32 or float64 go up as they are (predict_fno casts on the GPU; f64->f32 rounding is identical there)
        xin = np.ascontiguousarray(xin) if xin.dtype in (np.float32, np.float64) else np.ascontiguousarray(xin, dtype=np.float32)
        _pb = _pool_take(len(xin))
        if _pb is not None:
            pred, lower, upper = _pb
        else:
            pred = np.zeros((len(xin), 20, 32, 64, 3), dtype=np.float32)
            lower = np.empty_like(pred)
            upper = np.empty_like(pred)
        sig = np.empty((len(xin), 1, 32, 64, 2), dtype=np.float32)
        eb_ = None
        if EBAND is not None:
            global _EBAND_DEV
            if _EBAND_DEV is None:
                _EBAND_DEV = torch.tensor(EBAND[0], dtype=torch.float32, device=dev)
            eb_ = (_EBAND_DEV, EBAND[1], EBAND[2])
        _fno.predict_fno(models, xin, dev, alpha=ALPHA, batch=batch, out=pred, sigma_out=sig, amp=_AMP_DT,
                         band_gpu=(_BAND_DEV, ref, p, lower, upper, CENTRE_ALPHA != ALPHA, eb_))
        return {"prediction": pred, "lower": lower, "upper": upper}

    x = np.ascontiguousarray(input_array, dtype=np.float32)
    # Allocate the output once, zero-filled, and let the model write u,v into it. The pressure
    # channel is unmeasured and unscored, so zeros are the answer and never need touching
    # again. Concatenating it on afterwards cost a full-size copy inside a timed call.
    pred = np.zeros((len(x), 20, 32, 64, 3), dtype=np.float32)
    # (N, 1, 32, 64, 2), 5.5 MB at N = 336 against the field's 165 MB -- cheap enough to ask
    # for inside a timed call, and only requested when the band actually uses it.
    sig = np.empty((len(x), 1, 32, 64, 2), dtype=np.float32) if p else None
    # Heads' per-frame |error|, physical, only when band.npz asks for it. zeros, not empty: a
    # package whose members carry no head leaves it untouched, and zeros fail loudly below.
    # np.empty: _load() guarantees at least one err-bearing member when _ERR is set, so every row
    # is written by predict_fno (as the finished band FACTOR, computed on the GPU via err_xform).
    err = np.empty((len(x), 20, 32, 64, 2), dtype=np.float32) if _ERR is not None else None
    # Pre-compose mean, only when the interval is centred away from the prediction.
    centre = np.zeros_like(pred) if (CENTRE_ALPHA != ALPHA and band is not None) else None
    _fno.predict_fno(models, x, dev, alpha=ALPHA, batch=batch, out=pred, sigma_out=sig,
                     amp=_AMP_DT, centre_out=centre, err_out=err,
                     err_xform=None if _ERR is None else _ERR["xform"])
    # Before the band: the interval has to bracket what is actually returned.
    _apply_blend(pred, x)

    # The FALLBACK band -- used when band.npz is missing, has no sigma scaling, or the test
    # horizon is not the 20 frames it was fitted for. It is (1, 20, 32, 64, 3) or a scalar and
    # broadcasts against pred natively, so it is never materialized to full size. The live
    # path is the sigma-scaled branch below, which does materialize one full-size array.
    half = band if (band is not None and band.shape[1:] == pred.shape[1:]) else np.float32(0.0065)
    if ((p and sig is not None) or err is not None) and band is not None and half is band:
        # Per-window widths: base * (sigma / sigma_ref)**p, clipped to the range the fit was
        # solved over. The sigma scaling is +0.51 sps on the board-like proxy and +1.64 on the
        # full split, both in AND out of sample, where the round-6 attempt using ensemble
        # spread was noise.
        # Order-independent: sigma_ref is a stored constant, so a window's band depends only
        # on that window.
        #
        # This section is 1.45 ms/window of pure host memory traffic done the obvious way,
        # against 3.9 for the whole forward pass, and board time charges 0.431 ms per local
        # ms. Two changes take it to 0.13:
        #   * the scale factor carries a THIRD, zero channel, so `half` is one allocation and
        #     the (N,20,32,64,3) np.concatenate that appended the zero pressure band is gone;
        #   * the three full-size ops go through torch, which threads them. numpy's ufuncs are
        #     single-threaded and this stage is bandwidth-bound, not FLOP-bound.
        # torch.from_numpy wraps without copying, so these are views of the same buffers.
        # `upper` doubles as the scratch for `half` -- it is overwritten in place by the last
        # op, so no fourth full-size array is ever live.
        r3 = np.zeros((len(x), 1, 32, 64, 3), dtype=np.float32)
        if p and sig is not None:
            np.power(np.clip(sig / ref[..., :2], 0.3, 3.0), np.float32(p), out=r3[..., :2])
        else:
            r3[..., :2] = 1.0
        lower = np.empty_like(pred)
        upper = np.empty_like(pred)
        t_pred, t_lo, t_hi = (torch.from_numpy(a) for a in (pred, lower, upper))
        torch.mul(torch.from_numpy(band), torch.from_numpy(r3), out=t_hi)   # half -> upper
        if err is not None:
            t_e = torch.from_numpy(err)          # already the finished factor (see err_xform)
            if _ERR["mode"] == "direct":
                if _ERR.get("floor"):   # half = max(k*H, err_floor*base): the static band's floor (absent key => no floor)
                    t_e = torch.maximum(t_e, torch.from_numpy(band[..., :2]) * _ERR["floor"])
                t_hi[..., :2].copy_(t_e)                                        # half = k * H
            else:
                t_hi[..., :2].mul_(t_e)                          # half *= clip(H/ref)^q
        if EBAND is not None:
            # M4 (2026-09-28): the SIGBLEND factor, in the GPU path's order ((base * r3) * fac); checked in _predict above.
            t_hi[..., :2].mul_(_eband_host_factor(sig))
        if centre is None:
            torch.sub(t_pred, t_hi, out=t_lo)
            torch.add(t_pred, t_hi, out=t_hi)
            return {"prediction": pred, "lower": lower, "upper": upper}
        # Decoupled centre. `centre` becomes c in place; the pressure channel is 0 in both
        # pred and centre, so its bounds stay exactly 0. min/max keep pred inside the interval.
        t_c = torch.from_numpy(centre)
        if CENTRE_ALPHA != 0.0:
            t_c.add_(t_pred - t_c, alpha=CENTRE_ALPHA / ALPHA)
        torch.sub(t_c, t_hi, out=t_lo)
        torch.minimum(t_lo, t_pred, out=t_lo)
        torch.add(t_c, t_hi, out=t_hi)
        torch.maximum(t_hi, t_pred, out=t_hi)
        return {"prediction": pred, "lower": lower, "upper": upper}

    # No finiteness scan here. It cost 0.066 ms/window -- 5% of the whole timed call -- to
    # duplicate a check the scorer already performs on `prediction` before scoring anything,
    # and ours was strictly worse: the scorer degrades a non-finite prediction to zero scores,
    # while an assert would raise out of predict() and fail the run outright. The band's
    # finiteness and sign are established once at import instead.
    lower = (pred - half).astype(np.float32, copy=False)
    upper = (pred + half).astype(np.float32, copy=False)
    return {"prediction": pred, "lower": lower, "upper": upper}


_warm_tails()
