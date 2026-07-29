# AI4PDE — NeurIPS 2026 RealPDE Competition

Solution code for the **RealPDE** competition (real-world PDE / sim-to-real fluid dynamics).
Site: https://realpdecompetition.github.io/ · Data: https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench
Track 1 (Sim2Real): https://www.codabench.org/competitions/17363/ · Track 2 (LTTTA): https://www.codabench.org/competitions/17385/

This repo (`~/RealPDE/AI4PDE`, remote `github.com/clockj/AI4PDE`) holds **our** code.
Datasets, starting kits and official checkpoints live one level up in `~/RealPDE/` and are **not**
committed (16 GB).

## The two tracks

Both tracks pose the **same underlying forecasting problem** — 20 input frames of real PIV
velocity at 32×64 → the next 20 frames, channels `[u, v, p]` with `p` zero-filled — and are
judged by the **same five subscores against the same real-world targets**. Simulation is a
training-side asset in both; it is never a target and never available at test time.

They differ entirely in *how the model is exercised*:

| | **Track 1 — Sim2Real** | **Track 2 — LTTTA** |
|---|---|---|
| Goal | transfer knowledge from clean CFD to noisy, partially observed PIV | keep a deployed model accurate as it streams, by adapting online |
| Entry point | `predict(input_array, metadata)` | `get_ttt_model(submission_dir, device)` → `ttt_step()` |
| Call pattern | once, batched over all `N` test windows | once per window, **batch 1**, strict time order |
| Space | **raw physical units** — you normalize and denormalize | **already normalized** — evaluator handles both ends |
| Feedback | none, ever | previous window's ground truth revealed each step |
| Weights | frozen | may be updated mid-stream; reset per trajectory |
| What is timed | inference only | **everything** in `ttt_step`: adaptation, controller, LLM, forward |
| Extra limit | 256 MB archive | 256 MB archive + **10 min wall clock** for the whole run |

Windows tile each trajectory with stride 20, so window *k*'s target is window *k+1*'s input.
Track 2 exploits exactly that: at step *k+1* you are handed the true answer to the question
asked at step *k*.

Note the competition page calls Track 2 "N-step autoregressive", but each step's input is the
**real measured frames**, not your own previous prediction. It is teacher-forced streaming, so
errors do not compound through a rollout — the "long-term" risk is that your *adapted weights*
drift over the ~42 windows in a trajectory.

Design consequences: Track 1 amortizes one batched forward pass over all windows, so latency is
cheap there; Track 2 pays per step at batch 1 and is charged for every gradient step it takes.
Hence a shared backbone, but adaptation logic that is cheap by construction. Per-track working
notes live in `Track_1/README.md` and `Track_2/README.md`.

## Repo layout

```
AI4PDE/
├── CLAUDE.md      # this file: contracts, metrics, measured baselines, strategy
├── Track_1/       # Sim2Real: one-shot predict(), models + training + packaging
└── Track_2/       # LTTTA: streaming ttt_step(), base model + adaptation + policy
```

Data pipeline, self-normalization and the local scorer harness are needed by both tracks —
factor them into a shared module (`common/`) once the second track needs them, rather than
duplicating.

## Layout (outside this repo)

```
~/RealPDE/
├── AI4PDE/                                  # this repo
├── train_sim/train_sim/*.h5                 # 100 CFD trajectories, 9.2 GB
├── train_real/train_real/*.h5               #  82 PIV trajectories, 7.0 GB
├── example_data/3750_0.h5                   # single example trajectory
├── baseline_checkpoints/{sim_pretrain,sim_real_ft}/*.pth
└── realpde_t{1,2}_starting_kit_v*/          # note: doubly nested, kit root is the inner dir
```

Windows download artifacts (`*:Zone.Identifier`) sit next to every downloaded file — always
filter globs with `.endswith(".h5")` / `not endswith("Identifier")`.

## Data

NACA4418 airfoil, paired real PIV + matched CFD. Filenames are `{Re}_{AoA}.h5`.

| | trajectories | frames | datasets in the h5 | dtype |
|---|---|---|---|---|
| `train_sim` | 100 (Re 3750…27975 × AoA 0,5,10,15,20) | 1000 | `u,v,p (T,64,128)`, `x,y,t`, scalars `re,aoa` | float32 |
| `train_real` | 82 (18 Re × 5 AoA, incomplete grid) | 868 | `u,v (T,64,128)`, `x,y,t`, scalars `re,aoa` | float64 |

* Real data has **no pressure** — `p` is zero-filled and **not scored** (`measured_channels()`
  returns 2). Channel order is always `[u, v, p]`.
* The T1 README claims train tarballs nest fields under `measured_data/u` — **they do not**;
  `u`/`v` are top level in what we downloaded. Handle both.
* Sim covers Re 15225 and 27975 which real does not; the T2 example test file is `2968_0.h5`
  (Re below the whole training range). **The hidden test set contains unseen Reynolds numbers**,
  so Re/AoA extrapolation matters. `predict()` gets no Re/AoA — only `metadata["sample_id"]` —
  so any conditioning must be inferred from the input window itself.
* Sim is **not** a volume multiplier: at stride 20 it is 100×49 = 4900 windows vs real's
  82×42 = 3444, only 1.4×. Its value is Re/AoA coverage, noise-free targets and `p`.

### Units: sim and real are on different scales (measured)

Grids (`x`, `y`) and `dt` are identical between sim and real. Velocities are not:

* **Sim is non-dimensional** — mean `u` ≈ 0.86–0.91 for *every* condition.
* **Real is dimensional (m/s)** — `U∞ = Re·ν/c`, measured `mean_u/Re = 1.02…1.23e-5`
  (constant to ±10% over Re 3750…26700, so ν/c ≈ 1.1e-5, chord ≈ 9 cm in water).
* Raw sim/real ratio therefore runs 3× (Re 26700) to 19× (Re 3750). Joint training on raw
  arrays is meaningless.

Dividing each trajectory by its own mean speed collapses them: rel-L2 between sim and real
mean fields drops from 2.1–17.7 (raw) to a **condition-independent 0.38–0.47**, and turbulence
intensity matches within ±40%. In self-normalized units sim and real are the same problem.

**Consequence for the normalizer.** The official GaussianNormalizer z-scores with a global
mean 0.155 / std 0.0968 from train_real. At the test Re of 2968 real mean `u` ≈ 0.033, i.e.
−1.26σ — far off-distribution. **Prefer per-window self-normalization by the input's own mean
speed, predict in non-dimensional space, rescale on output.** This fixes Re extrapolation and
puts sim and real in one space for pretraining.

### Official windowing (identical for both tracks)

`sub_s = 2` (64×128 → **32×64**), `T_in = T_out = 20`, stride 20, windows in `(filename, time_id)`
order, tail padded by repeating the last frame. Reference implementation:
`realpde_t2_starting_kit_v4/.../local_eval.py:build_stream`. Every tensor is `(N, 20, 32, 64, 3)`.

### Normalization (GaussianNormalizer, fitted on train_real 32×64)

```
mean_in  = [0.154960856, -0.000513992854, 0.0]   std_in  = [0.0968056545, 0.015960684, 1.0]
mean_tgt = [0.154962569, -0.000517793698, 0.0]   std_tgt = [0.0968104079, 0.0159636438, 1.0]
```
`std == 0` → 1 (the `p` channel). **Track 1 `predict()` receives raw physical fields and must
normalize/denormalize itself** (skipping this is the #1 cause of a correctly-loaded baseline
scoring badly). **Track 2 `ttt_step()` receives already-normalized tensors** — work entirely in
normalized space; the evaluator denormalizes, bounds included.

## Scoring (both tracks, identical formulas)

`final_score = mean(rel_l2_score, tke_score, mvpe_score, time_score, sps_score)`, each 0–100.
`realpde_t1_starting_kit_v7/.../scoring.py` is the exact official program — run it locally.

* `score = 100 / (1 + 0.5 * error)` for Rel-L2, TKE, MVPE (measured channels `u,v` only).
* **Rel-L2** — per-window `‖p−t‖/‖t‖` over the whole `u,v` field.
* **TKE** — rel-L2 of `0.5*(var_t(u) + var_t(v))`, variance over the 20 output frames.
  **Phase-free**: only the per-pixel temporal variance field is compared.
* **MVPE** — rel-L2 of time-averaged `u,v` at fixed wake probes: `x ∈ {13,21,29,37}`,
  `y ∈ {8,10,…,24}` on the 32×64 grid.
* **Time** — `100/(1+sqrt(t_neural/0.72896))`, `t_neural` = mean per-step wall time.
  Zero/missing/non-finite ⇒ 0.
* **SPS** — per element `(1−pm)·exp(−(upper−lower)/σ)` counted only where the target lies inside
  the interval; `pm = e/(0.5+e)` per branch, weights DM 0.5 / TKE 0.3 / MVPE 0.2, then
  `100/(1+exp(−w))`. **σ = 0.0563870259, frozen for the season.** Range is `[50, ~73]`.
  Bounds optional; default band is `±5%·|pred|`. Bounds must be per-element, same shape, finite,
  `lower ≤ upper` — otherwise **every** subscore is zeroed, not just SPS.

### Measured local numbers (336 real windows, 8 held-out trajectories, `t = 10 ms`)

| predictor | rel-L2 | TKE | MVPE | sps | **final** |
|---|---|---|---|---|---|
| persistence (repeat last frame) | .138 → 93.5 | 1.000 → 66.7 | .141 → 93.4 | 55.4 | **79.70** |
| input time-mean | .134 → 93.7 | 1.000 → 66.7 | .124 → 94.2 | 55.3 | 79.87 |
| mean + 0.75·echoed input fluctuation, tuned band | .153 → 92.9 | .732 → 73.2 | .124 → 94.2 | 59.3 | **81.81** |
| mean + fluctuation rescaled to **oracle** per-pixel variance | .166 → 92.3 | .007 → 99.6 | .124 → 94.2 | 61.3 | **87.39** |
| perfect prediction (oracle) | 0 → 100 | 0 → 100 | 0 → 100 | 70.8 | **92.07** |

Reproduce with the scripts in the session scratchpad (`mkval.py`, `probe.py`, `alpha.py`,
`tke_head.py`) — port them into `eval/` when building the repo.

**Read the table before designing anything.** Rel-L2 and MVPE are ~93/94 for free and cap at
100: together worth ≤ 2.7 final points. TKE (66.7) and SPS (55) hold ~4.7 of the ~12.4 points of
total headroom, and time is a *liability*, not a freebie.

## Strategy

1. **Predict the variance field, not just the mean field.** TKE is phase-free, so matching the
   per-pixel temporal variance of the 20 output frames is sufficient — no need to get eddies in
   the right place. A perfect variance head on top of a *trivial* mean predictor reaches 87.4 of
   the 92.1 oracle. Concretely: decouple into a mean-field/low-frequency prediction (drives
   Rel-L2 + MVPE) and a per-pixel fluctuation-amplitude head (drives TKE), then rescale the
   predicted fluctuation to the predicted std. Input-window std predicts target std with
   corr 0.845 and median ratio 0.617 — a learned head should do much better than that scalar.
2. **Fight spectral bias.** Any L2-trained neural operator regresses toward the conditional mean,
   which gives TKE ≈ 1.0 — i.e. the same 66.7 as persistence, after all the training cost.
   Add an explicit variance/TKE-matching term and a probe-point (MVPE) term to the loss so the
   objective matches the leaderboard.
3. **Budget against the time curve from the start.** 1 ms/step = 96.4, 10 ms = 89.5,
   100 ms = 73.0, 700 ms = 50.5. Parameter count is a poor proxy for latency here — see the
   measured table under *Architecture*: the 50M-param FNO is 4× faster than the 8M-param CNO.
   Disk size is not the binding constraint either, since fp16-packed FNO clears the 256 MB cap.
4. **Tune the SPS band offline.** The default `±5%·|pred|` band scores 55.4. A uniform half-width
   of ~0.005–0.01 physical units scores 59.2 (coverage 0.67–0.80); wider is strictly worse because
   `exp(−width/σ)` with σ = 0.056 punishes it. Fit the band on validation, then move to
   per-element widths from a predicted-error head. Cheap, model-independent, ~+1 final point.
5. **Self-normalize before anything else.** Divide each input window by its own mean speed,
   work non-dimensionally, rescale on output. This is what makes Re extrapolation work (test
   Re 2968 is 1.26σ outside the global normalizer) *and* what makes sim pretraining transfer.
   See the units section above.
6. **Sim2real.** Pretrain on `train_sim` in self-normalized space (has `p`, 1000 frames, denser
   Re grid including the two Re missing from real), fine-tune on `train_real`. Use `p` as an
   auxiliary pretraining target and drop the head. Augment sim with PIV-like noise/dropout so
   the encoder is not surprised by real measurement artifacts. Validate on **held-out (Re, AoA)
   trajectories**, never on held-out windows of trajectories the model has seen.

   **Sim does not solve TKE.** Measured on 13950_10, predicting the real target's per-pixel std
   field: from the previous *real* window's std → rel-L2 0.559 / corr 0.756; from the sim
   trajectory's std, globally calibrated → rel-L2 0.628 / corr 0.653. Sim fluctuation statistics
   are *worse* than simply reusing the real input window's variance, because real PIV variance
   carries measurement noise and different turbulence character. Sim buys the mean field and the
   structure; the variance calibration must be learned on real.
7. **Always beat the 81.81 heuristic.** It is zero-parameter and instant. A trained model that
   scores below it is not worth submitting.

### Architecture

Measured on the local RTX 4060, fp32, `(B,20,32,64,3)`, no channels_last / bf16 / CUDA-graph
tuning. The eval GPU is unknown and probably faster — treat this as a ranking, not absolutes.

| model | params | size | B=1 | B=8 | B=32 | time_score B=1 → B=32 |
|---|---|---|---|---|---|---|
| FNO | 50.4M | 403 MB / 201 MB fp16 | 15.5 ms | 11.4 ms | 10.8 ms | **87.3 → 89.2** |
| CNO | 8.0M | 32 MB | 57.5 ms | 57.5 ms | 59.8 ms | 78.1 → 77.7 |
| Transolver | 12.5M | 50 MB | 109.1 ms | 106.3 ms | 678 ms (8 GB thrash) | 72.1 → 50.9 |

* **Use fp16-packed FNO as the base for both tracks.** CNO's 32 MB looks attractive but size is
  not the binding constraint (FNO fp16 fits the 256 MB cap), and CNO is 9 `time_score` points
  behind at batch 1 while amortizing *nothing* with batch — its custom StyleGAN-derived ops are
  per-sample. Transolver is disqualified on time alone.
* **The backbone is secondary; the output parameterization is the design.** Three heads —
  (a) mean field over the horizon (Rel-L2 + MVPE), (b) per-pixel log-std field (TKE),
  (c) fluctuation — then rescale (c)'s per-pixel temporal std to match (b). A small 2D
  U-Net/ConvNeXt over time-as-channels (60→60) at 5–15M params runs sub-ms batched and is worth
  ~+2 final points over batched FNO in T1, ~+9 subscore points in T2 — real, but second-order
  next to the ~7 points in TKE. Get the variance head working first, optimize latency after.
* **T2 is stuck at batch 1** by protocol, so read the B=1 column only, and remember adaptation
  cost is charged on top. Freeze the base and adapt only a low-capacity correction (output
  affine/FiLM, low-rank adapter on the last block, or a backprop-free EMA/RLS bias correction on
  the prediction-vs-revealed-target residual). Self-normalization already removes the Re shift,
  so what is left for TTT is trajectory-specific bias.
* **Strongest T2-specific play: adapt the variance head.** Each step reveals the previous
  window's ground truth, hence its true per-pixel std — measured above as the best available
  predictor of the next window's std (rel-L2 0.56, corr 0.76). An EMA calibration of head (b)
  costs microseconds and attacks the largest subscore gap directly.

### Track 2 specifics

* Streaming, batch 1, in normalized space. `ttt_step(input_norm, prev_target_norm)` — the
  previous window's ground truth is revealed. Adapt on the **cached previous input × revealed
  previous target**; never pair the previous target with the current input.
* **Everything inside `ttt_step` is timed**, including adaptation, controller logic and any LLM
  round-trip. `reset_ttt_state()` (per trajectory) and construction are **not** timed. A 1 s LLM
  call per step alone drops `time_score` to ~46 — use the gateway offline for policy design,
  and either never or very rarely online.
* Prefer adaptation that is cheap by construction: update a small adapter / last layer / output
  affine correction, or a closed-form ridge update, rather than full-model SGD. Skip the update
  when the previous-step error is already low (this is what `agentic_demo/policy.yaml` encodes).
* Guard against drift over ~42 windows per trajectory: small lr, gradient clip, anchor to the
  checkpoint weights, reset on loss spikes.
* Hard limit: **10 min wall clock** for the entire run (loading included) or the submission fails.
* Bounds are returned via `info["lower"]/["upper"]` (both or neither, and consistently for the
  **whole run**); the evaluator otherwise reads only `info["adapt_loss"]`.

## Submission contracts

**Track 1** — `submission.py` at the zip root:
```python
def predict(input_array, metadata=None):   # (N,20,32,64,3) raw physical -> (N,20,32,64,3)
    ...                                    # or {"prediction":…, "lower":…, "upper":…}
```
Zip root also holds the checkpoint, `load_baseline.py`, `rpde_baselines/`, `_vendor/`.
Worked reference: `realpde_t1_starting_kit_v7/.../submission_example_fno.py`.

**Track 2** — `submission.py` at the zip root:
```python
def get_ttt_model(submission_dir, device):  # -> obj with reset_ttt_state() and ttt_step()
```

Both: extracted archive **< 256 MB**; `pack_ckpt_fp16.py` halves a checkpoint (complex-safe for
FNO spectral weights; load with `load_baseline.unpack_fp16`, not `Model.load_checkpoint`).

### Evaluation containers — nothing is installed at eval time

* T1: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`. **No `scipy`, `h5py`, `einops`,
  `matplotlib`, `pandas`.** `einops` ships in the kit's `_vendor/`. Never `import h5py` from
  `submission.py`.
* T2: `w3nhao/realpde-track2:v1` (PyTorch 2.2.2, CUDA 12.1, Python 3.10); `einops` **is** present.
  Model runs in an isolated subprocess; only the hidden ground truth is unreadable.
* Vendor any other pure-Python dependency inside the zip. `smoke_test_kit.py` (T1) verifies
  imports against the eval package set — run it before spending a submission.

## Baseline checkpoints

`sim_pretrain/` (sim only) and `sim_real_ft/` (fine-tuned on real — the better starting point),
each with `cno` (32 MB), `transolver` (50 MB), `fno` (403 MB / 201 MB fp16).
Hyper-parameters were read off the `state_dict`, **not** the repo yamls (which are stale —
`trainsolver.yaml` says `n_layers: 1`, the checkpoint has 3):

| model | hyper-parameters |
|---|---|
| FNO | modes (4,12,16), width 64, n_layers 4, C_in=C_out=3 |
| CNO | N_layers 3, channel_multiplier 32 |
| Transolver | n_layers 3, n_hidden 256, n_head 8, slice_num 16, space_dim 3, fun_dim 0, out_dim 3 |

```python
from load_baseline import load_baseline, make_example_input
model, meta = load_baseline("sim_real_fno_fp16.pth")   # type auto-detected from the filename
x = make_example_input("3750_0.h5", sub_s=2)           # (1,20,32,64,3)
```

## Local environment

**Use the `realpde` conda env** — it is pinned to the evaluators' stack, so what runs locally
runs on Codabench. Recreate with `conda env create -f environment.yml`.

```
python 3.10.20   torch 2.2.2+cu121   cuda 12.1   cudnn 8.9.2   numpy 1.26.4
scipy 1.13.1  h5py 3.11.0  matplotlib 3.9.2  pandas 2.2.3  einops 0.8.0
tqdm 4.67.1  pyyaml 6.0.2  requests 2.32.3  setuptools 69.5.1
```

Verified: CUDA available on sm_89, bf16 matmul ok, and **all three baselines load and run**
(FNO 16.2 ms, CNO 58.6 ms, Transolver 109.0 ms at B=1 — reproducing the timings in
*Architecture* to within noise).

Two pins that are not optional:

* **`numpy==1.26.4`** — torch 2.2.2 is not numpy-2 compatible.
* **`setuptools==69.5.1`** — CNO's `CNO_libs/custom_ops.py` imports `pkg_resources`, which was
  removed in setuptools 81; and `dnnlib/util.py` imports `requests`. Without both, CNO fails to
  import. Neither is present in the T1 eval container either, so **CNO cannot be submitted to
  Track 1 without vendoring them** — another reason to prefer FNO.

Other envs on this machine (`bubble_tracking` torch 2.7.1+cu118, `stemseg` 2.4.1+cu118) are
unrelated to this project; do not use them here.

Hardware: **RTX 4060, 8 GB**, driver 591.86 (WSL2), 16 cores, 15 GB RAM. FNO3d width 64 training
will be tight on 8 GB — use AMP / gradient checkpointing / a smaller width, or train the cheap
models first. Transolver already thrashes VRAM at B=32 in inference.

Full dataset is 16 GB — preprocess to a cached 32×64 float32 array (100×1000×32×64×3 ≈ 2.5 GB
sim, 82×868×32×64×2 ≈ 1.2 GB real) instead of re-reading h5 every epoch.

## Rules and dates

Registration (separate Google Form per track) closes **2026-08-20**; joining on Codabench is not
enough. Max 3 members, one shared account. **1 submission/day**, 100 per phase — local scoring
with `scoring.py` is mandatory before every submission. Phases: Warm-up ended 2026-07-19; **Main
Development until 2026-09-27**; Decision until 2026-10-25; results 2026-11-10; workshop 2026-12-06.
Winners open-source two weeks before the presentation. Track 2 agent steps must be explicitly
bounded; direct network access / own LLM API keys are prohibited (organizer gateway only).
Prizes per track: $6k / $3k / $1.5k; both tracks can be entered.
