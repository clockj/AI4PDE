# xtgg233: Track 2 method

## 1. Summary
A frozen ensemble of six small networks — four factorised Fourier neural operators and two vision transformers —
maps the 20 input frames to the 20 output frames in one pass and predicts, per pixel, the temporal standard
deviation of the target window; the interval is a per-element half-width solved exactly for the SPS score on the
errors of five "twin" networks trained with eight trajectories held out, scaled by a power of the predicted standard
deviation. On Track 2 three strictly causal corrections learn from the targets the stream reveals, and train nothing:
the interval's lower and upper width per pixel, the time-mean bias of the prediction, and the amplitude of its
fluctuation. The parts that matter most for the score are the fused ensemble under a bounded (Welsch) field loss
with CFD co-training (accuracy), the exact per-element band with its online recalibration (sps), and one CUDA graph
for the whole step (time).

## 2. Model
- Architecture and number of parameters: six members.
  - Four **F-FNO** (`unet/models_alt.py: FFNO2d`): width 40, 8 factorised Fourier layers — the Fourier operator is
    separable (an rfft along y with one complex matrix per retained ky, the same along x, 16 and 32 modes, the two
    summed), followed by `x + Dropout2d(LayerNorm(Linear-GELU-Linear(K(x))))`; **1.31 M parameters** each.
  - Two **ViT** (`ViT2d`): 4 × 4 patches of the 32 × 64 grid → 128 tokens, dimension 384, 8 pre-norm blocks of
    6-head attention + MLP (ratio 4), dropout 0.3 inside attention and the MLPs, stochastic depth 0.1, tokens
    un-patched to a per-pixel feature map of 24 channels; **14.66 M parameters** each.
  - Both end in the same two convolutional heads: the field (40 channels = 20 frames × [u, v]) and a per-pixel
    log-standard-deviation for u and v. Every member takes the 40 input channels plus two coordinate channels.
- Inputs and outputs, including preprocessing and normalization: input `(1, 20, 32, 64, 3)` as the platform hands it
  (normalised with the official input statistics; the pressure channel is dropped), the 20 frames folded into the
  channel axis. Output: the 20-frame field in the same normalised space, with the platform's target statistics
  applied as one exact affine correction (they differ from the input statistics in the sixth decimal), the pressure
  channel exactly zero; `info["lower"]` and `info["upper"]` in the same space.
- Starting point (random initialization, or which official baseline checkpoint): random initialisation
  (PyTorch defaults; the ViT's position embedding truncated-normal at 0.02), seeded.

## 3. Data
- Training data used (`train_sim`, `train_real`) and how the two are combined: every real trajectory (`train_real`,
  u and v, sub-sampled by 2 to 32 × 64) as 40-frame windows at stride 5 (about 13 400 windows); every CFD trajectory
  (`train_sim`, 100) **mapped into the measurement space of the real data** by `unet/simreal.py` and used as a
  second, auxiliary stream: per step one batch of real windows and one batch of mapped CFD windows at stride 5. The
  map, per CFD condition `<Re>_<AoA>`, from the statistics of the real recording of the same condition where one
  exists: (1) the real recording's zero set (field of view, airfoil) is painted onto the CFD field; (2) the CFD
  trajectory is scaled so that its mean u equals the real mean u; (3) its fluctuation about the time mean is rescaled
  so that the norm of the per-pixel std field equals the real one. A condition without a real recording takes its
  mean speed from a freestream law `mean_u = C(AoA) · Re` fitted on the real recordings, its fluctuation from the
  pooled turbulence-intensity ratio, and its zero set from the nearest real recording in Re at the same angle. Five of
  the six members and four of the five twins use the CFD term; the plain-loss ViT does not.
- Validation split carved out of the training data, if any: eight real trajectories (`unet/data.py: VAL_TRAJ`:
  3750_0, 6300_10, 8850_20, 11400_20, 13950_10, 16500_5, 21600_15, 26700_0). The six **members** train on every
  trajectory. The five **twins** are the same recipes trained without those eight (and, for the CFD term, with
  those conditions removed from the CFD pool and their statistics withheld from the map); their errors on the eight
  are what the interval band is fitted to (sections 5), and nothing else is selected on them.
- Augmentation: none.
- Data left out, and why: `train_real/7575_0.h5`, a copy of `6300_0.h5` (the data page asks for it to be left out;
  the organizers' mount does not hold it). The submitted networks had been trained with it present (see README).
  No window is dropped otherwise; the ~10 % of target elements that are exactly zero are trained on as zeros, as the
  scorer counts them.

## 4. Training
One row per stage, in the order `train()` runs them. The eleven trainings are independent; `train()` starts them
longest first, several at a time on the one GPU, and the band fit follows them. Every network: AdamW, weight decay
1e-5, one-cycle learning-rate schedule peaking at 2e-4 over the whole run (PyTorch `OneCycleLR` defaults: 30 %
warm-up, cosine), dropout 0.3, 400 epochs, batch 32, windows at stride 5, input and target normalised by the
official input statistics. Loss terms (`unet/train.py`): `field` = MSE of the residual split into its time-mean
and fluctuation parts; `Welsch` = redescending loss `c²(1 − exp(−½ (d/c)²))` of the same residual with the knee
c = 0.25 × σ_SPS (0.0564 m/s, the scorer's interval scale) — a bounded field loss that stops the large errors of the
blanked and ill-conditioned pixels from dominating; `grad` / `tgrad` = relative L2 of the velocity-gradient tensor
and of the frame-to-frame time derivative, in physical units; `logσ` = MSE of the predicted log-standard-deviation
against the log of the target window's temporal std; `TKE` = relative L2 of the turbulent kinetic energy implied by
the predicted σ against the target's; `CFD` = the field MSE on one batch of mapped CFD windows per step.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| members F-FNO, seeds 1 2 3 4 (`member_ffno_s*`) | all real windows + mapped CFD | 2.0 Welsch + 1.0 grad + 0.3 tgrad + 0.15 logσ + 0.15 TKE + 0.3 CFD | AdamW 1e-5, one-cycle to 2e-4 | 400 epochs ≈ 167 000 steps | 32 (+ 32 CFD) |
| member ViT with the CFD term, seed 3 (`member_vit_c`) | all real windows + mapped CFD | the same | the same | 400 epochs | 32 (+ 32 CFD) |
| member ViT, plain loss, seed 1 (`member_vit_p`) | all real windows | 1.0 field + 1.0 grad + 0.3 logσ + 0.15 TKE | the same | 400 epochs | 32 |
| twins F-FNO, seeds 1 2 3 (`twin_ffno_s*`) | real windows without `VAL_TRAJ` + mapped CFD without those conditions | as the F-FNO members | the same | 400 epochs ≈ 151 000 steps | 32 (+ 32 CFD) |
| twin ViT with the CFD term, seed 1 (`twin_vit_c`) | as the F-FNO twins | as `member_vit_c` | the same | 400 epochs | 32 (+ 32 CFD) |
| twin ViT, plain loss, seed 1 (`twin_vit_p`) | real windows without `VAL_TRAJ` | as `member_vit_p` | the same | 400 epochs | 32 |
| band fit (`band/`) | the twins' predictions on the 8 held-out trajectories and on 8 trained-on ones | exact SPS optimum per element, then a 2-parameter power law | closed form / grid | — | — |
| build (`build/`) | the six members, the band | — | — | — | — |

- How runs or checkpoints are selected, if at all: not at all. One run per recipe, the last epoch is the checkpoint,
  all six members ship, all five twins fit the band. No early stopping, no best-epoch choice, no model averaging.
- Seeds, and the run-to-run spread you have seen: seeds as in the table (`solution.py: RECIPES`), set for torch,
  CUDA and numpy; cuDNN in its default mode. Re-training a member with a different seed moves our local estimate of
  `final_score` by about 0.017 for one member and about 0.01 for the ensemble; re-training all eleven networks from
  scratch on other cards (2026-09-25) moved it by +0.0006 on the held-out trajectories at angles of attack 15 – 20
  and by +0.0040 on all eight. Across GPU models the weights themselves differ (F-FNO about 0.5 %, ViT about 10 %
  after 400 epochs) because the kernels are not deterministic; on one card and software stack a seed is reproducible.

## 5. Uncertainty bounds
How `lower` and `upper` are computed, or "not returned": returned on every step, as `info["lower"]` /
`info["upper"]`, shaped like the prediction, in normalised space.

- **Centre.** The plain ensemble mean field y₀ (the mean of the six members' fields, *before* the fluctuation
  adjustment of section 6). `CENTRE_ALPHA = 0`.
- **Half-width, static part** (`band/fit_band.py`, written into `band.npz`). For each element (frame t, pixel,
  channel) the SPS score factorises into an accuracy factor per window times, per element, `exp(−2h/σ_SPS)` × 1[the
  error is inside h]; for a fixed element the in-sample optimum of h is one of the observed absolute errors, found
  exactly by sorting. The errors are those of the twins' ensemble mean field y₀ (`|y₀ − y|`; the per-window
  accuracy factor is taken on the ALPHA 0.80 prediction of section 6) on the trained-on validation windows at angles
  of attack 15 – 20 (where the field is hardest), inflated by 1.39 (the ratio between the errors the board showed
  and the local ones), deflated by the σ factor below so that the two compose, box-smoothed 3 × 3 × 3, and widened
  where fewer than 8 fitting windows scored an element; this is `base`. It is scaled per window by
  `clip(σ / σ_ref, 0.3, 3.0)^p` with `p = 0.45`, σ the ensemble's predicted std of that window and σ_ref the
  members' mean σ per element.
- **Half-width, σ power law** (`band/sigma_band.py`, the "SIGBLEND" band). On the twins' *held-out* errors the width
  is further multiplied by `(σ_phys + 1e−8)^q_c` per channel, the two exponents `q = (q_u, q_v)` fitted on 16
  σ-bins with a λ = 0.5 blend toward the static band and a 1.29 inflation; `base` and `p` are re-expressed so the
  archive carries one table. Fitted values in the submission: `q = (0.2423, 0.2454)`; the organizers' retraining
  refits them.
- **Containment.** `lower = centre − half`, `upper = centre + half`, then both widened just enough to contain the
  prediction P of section 6, so the interval never excludes the point forecast.
- **Online, Track 2 only (R1).** From the revealed targets of the current trajectory, per pixel and channel, the lower
  and the upper half-width are each multiplied by a gain: for every past window the width term and the hit indicator
  are accumulated on a 9 × 9 grid of candidate gains (0.5 … 2.0, log-spaced, containing 1), separately for the lower
  and the upper side, at the resolution of 4-frame lead-time blocks as well as per window; the argmax over the grid
  is box-smoothed 3 × 3 and shrunk toward 1 by `n / (n + N0)` (N0 = 2 per window, 20 per lead-time block). Only
  windows already revealed enter; `reset_ttt_state` zeroes the tables.

## 6. Inference
- Track 1: what `predict` does per call, numeric precision, any compilation or caching: (not this track; the same
  archive family ran on Track 1 without the online corrections.)
- Track 2: the adaptation rule in `ttt_step`: which parameters are updated, learning rate, steps per window, and what
  `reset_ttt_state` restores:
  - **No parameter is updated, there is no learning rate and no gradient step.** The networks are frozen; what
    adapts are three sets of running statistics of the current trajectory, each causal (window k's output depends on
    the targets of windows < k only, checked by `build/causal_check.py`: later windows replaced by noise must leave
    every earlier output bit-identical).
  - Per step: the six members run in two fused invocations (the four F-FNOs stacked, the two ViTs stacked); the mean
    field y₀ and the mean σ head are composed into the prediction P by moving the amplitude of y₀'s fluctuation
    about its time mean toward σ with weight ALPHA = 0.80, one gain per window shared by u and v (`JOINT_GAIN`).
    Then, from the revealed windows:
    (1) **D1c** — the time-mean bias per pixel and channel, `b = Σ_past mean_t(C − y) / (n + 30)` with C the
    uncorrected mean field of each past window, is subtracted; its shrink is replaced by an empirical-Bayes factor
    `τ² / (τ² + se²)` (pooled over pixels per channel, damped by `n / (n + 1)`) once two windows are revealed, and
    the shift is given a lead-time profile `a(t)` fitted on the past windows' per-frame mean error (shrink
    `n / (n + 5)`, clipped to [0, 3]);
    (2) **VARCAL** — the fluctuation amplitude is multiplied by `sqrt(g)`, g the ratio of the revealed to the
    predicted temporal variance summed over past windows, box-smoothed 3 × 3, clipped to [0.25, 4], shrunk to 1 by
    `n / (n + 30)`;
    (3) **R1** — the interval gains of section 5.
    The interval is centred on the corrected y₀ and widened to contain the corrected P.
  - `reset_ttt_state` sets every accumulator to zero (n = 0): the first window of every trajectory is the frozen
    ensemble's own output, exactly. `info["adapt_loss"]` is `None` (nothing is optimised).
  - Precision: fp16 autocast for the networks (verified at load against fp32 on a probe; falls back to fp32 if the
    probe disagrees), weights stored int8 per tensor and dequantised at load, statistics and corrections in fp32 and
    fp64 where the reference implementation used it. The whole device-side step — both invocations, the ensemble
    mean, the composition, the three corrections, the band — is captured once in `get_ttt_model` (untimed) as one
    CUDA graph and replayed per step; if capture fails the eager path runs (bit-identical).

## 7. Speed
What you did for the Time score: one invocation per family instead of one per member (+2.4 ms per extra
invocation measured, 0.09 ms per extra member inside a fused one); the F-FNO's spectral multiplication as a dense
batched matmul (`FFNO_PATH = "dense"`, `SPEC_PATH = "bmm"`); fp16; int8 weights (smaller archive and load); one CUDA
graph for the whole step including the online corrections (3.7 ms → 1.0 ms for the fused group when it was
introduced); a pre-allocated output arena of 1024 steps and NumPy outputs (no per-step allocation or device sync
beyond the copy). On the development board this archive family drew 5.0 – 6.7 ms per step.

## 8. What mattered
The changes that moved your development score most, with rough effect sizes (Track 2 development board, deltas in
`final_score` between consecutive uploads of the same package otherwise):
- The causal stream stack — the σ power-law band with the joint gain, the bias correction D1c with its
  empirical-Bayes shrink and lead-time profile, and VARCAL — over the same ensemble with the online band gains only:
  **+0.32** (81.740 → 82.056, the last upload).
- Centring the interval on the plain mean field instead of the amplitude-adjusted prediction: **+0.17**.
- The online interval gains (R1) over the static band: **+0.09**; ALPHA 0.80 with the exact per-element band: ≈ +0.1.
- The F-FNO (width 40, 8 layers) × 4 + ViT × 2 ensemble over a width-96 FNO × 8 ensemble: **+0.15**; dropping the
  ViT pair: −0.05; F-FNO at 4 layers instead of 8: containment −0.3.
- Dropout 0.3 in the members: **+0.20**. CFD co-training: +0.07.
- Fusing the members into one invocation per family: **+0.21** (measured by un-fusing: −0.21); the CUDA graph:
  ≈ +0.25; fp16: ≈ +0.03.

## 9. Compute
Hardware, training time, and peak GPU memory of the run behind your submission: eleven single-GPU jobs — the seven
F-FNOs on NVIDIA V100S-PCIE-32GB (4.85 h per member, 4.37 h per twin), the four ViTs on NVIDIA A100-PCIE-40GB
(3.81 h and 2.00 h for the members, 3.43 h and 1.82 h for the twins); **43.6 GPU-hours** in all, plus about 10
minutes for the band fit, the build and the gates. Peak GPU memory was not recorded by those runs; this package
measured on an A100: 2.5 GiB allocated per F-FNO training, 2.6 GiB per ViT training. Details and the logs:
README.md, `logs/`.

## 10. References
Papers, third-party code and libraries the method builds on:
- A. Tran, A. Mathews, L. Xie, C. S. Ong. *Factorized Fourier Neural Operators.* ICLR 2023 (the F-FNO layer; our own
  implementation).
- Z. Li et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021.
- A. Dosovitskiy et al. *An Image is Worth 16x16 Words.* ICLR 2021 (the ViT; our own implementation on
  `torch.nn.MultiheadAttention`); G. Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016.
- I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019 (AdamW); L. N. Smith, N. Topin,
  *Super-Convergence*, 2018 (the one-cycle schedule, PyTorch's `OneCycleLR`).
- J. E. Dennis, R. E. Welsch. *Techniques for nonlinear least squares and robust regression.* 1978 (the Welsch loss).
- PyTorch 2.2.2, numpy 1.26.4, h5py 3.16.0; the organizers' `scoring.py` (Track 1 starting kit v9) for the SPS score
  the band is fitted to. No other third-party code.
