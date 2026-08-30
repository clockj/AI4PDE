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
| `train_real` | 82 (18 Re × 5 AoA, incomplete grid) | 607 | `u,v (T,64,128)`, `x,y,t`, scalars `re,aoa` | float64 |

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

> **Kit v9 (2026-08-04) replaced the scoring program and the test set.** Use
> `realpde_t1_starting_kit_v9/.../scoring.py`. The SPS bullet below is rewritten for it; the
> other four subscores are unchanged. Two v9 changes, both in SPS, and both large:
>
> * `score_sps` was `100/(1+exp(-w))` and is now `100*w`. The old form was pinned in
>   `[50, 73.11]`; the new one is a full 0–100 subscore. The aggregate `w` barely moved across
>   the change (0.376 → 0.369), so the board's drop from `sps 59.30` to `36.94` is the
>   rescaling, **not** a regression. The slope went from ~23 to 100, so anything that cuts
>   per-element error is now worth **4.4× more** through SPS.
> * Elements whose target is exactly 0 — outside the PIV field of view, inside the airfoil
>   body, ~9.9% of `u,v` — now leave **both** the numerator and the denominator. They used to
>   be free points (prediction ≈ 0, any band around it contains the target), so losing them
>   costs real score.
>
> **`final_score` is no longer published and is no longer the plain mean**: the 2026-08-05
> board reported 79.733 where its own five subscores average 78.999. One equation, five
> unknowns — not solvable. Track `mean5` (the plain mean) as a proxy and say so; all five
> subscores still move `final_score` monotonically.

`realpde_t1_starting_kit_v9/.../scoring.py` is the exact official subscore program — run it
locally before every submission.

* `score = 100 / (1 + 0.5 * error)` for Rel-L2, TKE, MVPE (measured channels `u,v` only).
* **Rel-L2** — per-window `‖p−t‖/‖t‖` over the whole `u,v` field.
* **TKE** — rel-L2 of `0.5*(var_t(u) + var_t(v))`, variance over the 20 output frames.
  **Phase-free**: only the per-pixel temporal variance field is compared.
* **MVPE** — rel-L2 of time-averaged `u,v` at fixed wake probes: `x ∈ {13,21,29,37}`,
  `y ∈ {8,10,…,24}` on the 32×64 grid.
* **Time** — `100/(1+sqrt(t_neural/0.72896))`, `t_neural` = mean per-step wall time.
  Zero/missing/non-finite ⇒ 0.
* **SPS (v9)** — per element `(1−pm)·exp(−(upper−lower)/σ)` counted only where the target lies
  inside the interval **and is non-zero**; `pm = e/(0.5+e)` per branch, weights DM 0.5 / TKE 0.3
  / MVPE 0.2, then **`100·w` (was `100/(1+exp(−w))`)**. **σ = 0.0563870259, frozen for the
  season.** Range is now the full `[0, 100]`. Bounds optional; default band is `±5%·|pred|` and
  scores only 22.8 — always supply your own. Bounds must be per-element, same shape, finite,
  `lower ≤ upper` — otherwise **every** subscore is zeroed, not just SPS.

  **The aggregate factorizes**, and `Track_1/sps9.py` exploits it:
  `w = (1/n_scored)·Σ a_i·exp(−2h_e/σ)·1[|E_ie| ≤ h_e]` with
  `a_i = 0.5(1−pm_dm) + 0.3(1−pm_tke) + 0.2(1−pm_mvpe)` — the three branches differ only in a
  per-**sample** factor while the interval term is per-**element**. So scoring a band is one
  pass instead of three, and the band problem separates over elements: each element's optimum
  sits exactly at one of its observed `|E_ie|`, so sorting gives the **exact** optimum with no
  Gaussian assumption (`sps9.optimal_band`). That is also the ceiling on band fitting, and it
  is only +0.6 over the old `optimal_half(c·RMS)` — **band work is finished**, see
  `Track_1/report/report_20260805_v9.md` §4.

### Measured local numbers (336 real windows, 8 held-out trajectories, `t = 10 ms`)

| predictor | rel-L2 | TKE | MVPE | sps | **final** |
|---|---|---|---|---|---|
| persistence (repeat last frame) | .138 → 93.5 | 1.000 → 66.7 | .141 → 93.4 | 55.4 | **79.70** |
| input time-mean | .134 → 93.7 | 1.000 → 66.7 | .124 → 94.2 | 55.3 | 79.87 |
| mean + 0.75·echoed input fluctuation, tuned band | .153 → 92.9 | .732 → 73.2 | .124 → 94.2 | 59.3 | **81.81** |
| mean + fluctuation rescaled to **oracle** per-pixel variance | .166 → 92.3 | .007 → 99.6 | .124 → 94.2 | 61.3 | **87.39** |
| perfect prediction (oracle) | 0 → 100 | 0 → 100 | 0 → 100 | 70.8 | **92.07** |

### Official baselines on the same 336 windows (measured, `realpde` env, RTX 4060)

| checkpoint | rel-L2 | TKE | MVPE | rel | tke | mvpe | time | sps | **final** | +tuned band |
|---|---|---|---|---|---|---|---|---|---|---|
| **FNO** sim_real_ft | .071 | .569 | .062 | 96.6 | 77.9 | 97.0 | 88.8 | 55.1 | **83.06** | **84.46** |
| CNO sim_real_ft | .088 | .672 | .081 | 95.8 | 74.8 | 96.1 | 77.9 | 54.6 | 79.83 | 81.16 |
| Transolver sim_real_ft | .135 | .812 | .126 | 93.7 | 71.1 | 94.1 | 62.2 | 53.0 | 74.82 | 76.05 |
| FNO sim_pretrain | .760 | .985 | 1.009 | 72.5 | 67.0 | 66.5 | 88.9 | 50.1 | 69.00 | 69.50 |
| CNO sim_pretrain | .706 | 1.628 | .871 | 73.9 | 55.1 | 69.7 | 77.9 | 50.2 | 65.35 | 65.84 |
| Transolver sim_pretrain | .606 | .946 | .795 | 76.8 | 67.9 | 71.6 | 62.4 | 50.6 | 65.85 | 66.39 |
| **DeepONet** (ours, real-only, 0.85M) | .132 | .933 | .167 | 93.8 | 68.2 | 92.3 | **97.0** | 53.4 | 80.94 | 82.02 |

**FNO dominates on both axes** — best accuracy *and* 5×/23× faster than CNO/Transolver.
Likely why: the flow is quasi-periodic vortex shedding, which a global spectral convolution with
few modes captures directly, and there are only ~3.4k real training windows, so a strong
inductive bias beats data-hungry attention. CNO and Transolver are strictly dominated; do not
spend time on them.

**Sim-only pretraining scores 69.0 — far *below* trivial persistence (79.7).** All of the
baselines' real-world skill comes from the real fine-tuning stage, not from sim. Treat sim as a
weak prior, not a substitute for real data. Hypothesis for why zero-shot transfer is so bad
(untested): sim has an informative `p` channel while real is `p = 0`, so a sim-pretrained model
leans on an input that vanishes at test time — another reason to use `p` as an auxiliary
*output* only, and zero the `p` input during sim pretraining.

### Our own models (held-out trajectories — see `Track_1/report/report_20260729.md`)

> Superseded by the WP1/WP2 table further down and `Track_1/report/report_20260730.md`. The
> `time_score` column here was measured inside the prediction loop and is ~1.5 points low.

| model | rel-L2 | TKE | MVPE | rel | tke | mvpe | time | sps | **final** | +band | ms/win |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **U-Net (mse), 1.93M** | .082 | .665 | .076 | 96.1 | 75.1 | 96.4 | 96.7 | 55.0 | **83.83** | **85.15** | 0.87 |
| PDE-Refiner, 1.93M | .099 | .771 | .080 | 95.3 | 72.2 | 96.1 | 95.2 | 53.8 | 82.53 | 83.81 | 1.84 |
| DeepONet, 0.85M | .132 | .933 | .167 | 93.8 | 68.2 | 92.3 | 97.8 | 53.4 | 81.09 | 82.02 | 0.68 |

**A 1.93M MSE U-Net trained for 12 minutes on real data alone beats the 50M FNO** — 83.83 on
trajectories it never saw vs FNO's 83.06 on trajectories it trained on, at 13× the speed. Build
on this, not on FNO. Code: `Track_1/unet/` (`model.py` = shared backbone + data pipeline,
`train.py` = MSE training). It is a plain deterministic regressor — no diffusion, no sampling.
Generalization gaps at 40 epochs are tiny (U-Net +0.57, DeepONet +0.10) against an FNO difficulty
control of +0.03. But training to convergence (150 epochs, train MSE 0.0417 → 0.0242) gains only
**+0.24 on unseen** while gaining +0.86 on seen — the gap **doubles to +1.26**. Longer training is
mostly memorization; never extend the schedule or capacity without a matching held-out check.
Run-to-run noise is ~±0.1 final (error metrics are exact; `time_score` jitters).

**Diffusion is a settled negative result here.** PDE-Refiner (Lippe et al., NeurIPS 2023) on the
identical U-Net and budget scores 82.53 vs its MSE control's 83.83 — worse on rel-L2 *and* on
TKE, the metric it was chosen to fix, at 4× the compute. Likely cause: real PIV high-frequency
content is largely measurement noise, so refining toward it fits sensor noise. Full ACDM-style
autoregressive diffusion is worse still on arithmetic alone (tens of steps × 20 frames ⇒
`time_score` ≈ 26). Do not revisit without a new reason.

**DeepONet reaches 81.09 almost purely on speed** (0.68 ms/win → `time_score` 97.0). Its field
accuracy is barely above free persistence and its MVPE is worse. On a fixed output grid a
DeepONet is a rank-*p* linear decoder over a condition-independent basis: fine for the low-rank
mean field, unable to place per-condition small-scale structure, hence TKE 0.933. The lesson is
about the metric, not the architecture — **a model barely better than repeat-the-last-frame sits
2 points off the best baseline by being fast.** FNO at that latency would score ≈ 84.7, more
than the whole rel-L2 + MVPE headroom combined. Full write-up: `Track_1/report/report_20260729.md`.

> **The fine-tuned numbers are optimistic; the DeepONet one is not.** The validation windows come
> from `train_real`, which the `sim_real_ft` checkpoints were fine-tuned on, so 83.06 is closer
> to training performance than to leaderboard performance. DeepONet is the only model here
> evaluated on trajectories it never saw, and the persistence/heuristic floors are untrained. So
> FNO's +2.1 over DeepONet is an **upper bound** on its real advantage. Build a genuinely
> held-out (Re, AoA) split before trusting any model comparison.

Reproduce with the scripts in the session scratchpad (`mkval.py`, `probe.py`, `alpha.py`,
`tke_head.py`, `eval_baselines.py`) — port them into `eval/` when building the repo.

**Read the table before designing anything.** Rel-L2 and MVPE are ~93/94 for free and cap at
100: together worth ≤ 2.7 final points. TKE (66.7) and SPS (55) hold ~4.7 of the ~12.4 points of
total headroom, and time is a *liability*, not a freebie.

### Leaderboard calibration (first submission, Codabench)

`unet_mse_e150` scored **83.108** against **84.79** locally. **The hidden test set is ~1.7 points
harder than `val_unseen`** — subtract that from every local number before comparing to the board.

### The metric's ceilings are measured, not estimated — see `Track_1/report/report_20260730.md`

Read this before proposing any Track 1 idea; most of the obvious ones are already ruled out.

| subscore | best measured | bound | how the bound was measured |
|---|---|---|---|
| `tke_score` | 78.6 | **~81** | split each 20-frame target into two 10-frame halves: two estimates of the *same* variance field disagree by rel-L2 0.937 ⇒ ≈0.47 irreducible at T=20 |
| `sps_score` | 61.9 | **64.9** | oracle band with widths chosen knowing the actual error |
| `rel_l2_score` | 95.7 | ≤98.5 | PIV white-noise norm ⇒ rel-L2 floor 0.031 (loose bound) |

`sps_score = 100/(1+exp(-v))` with `v` a weighted mean of terms each ≤ 1, so it **can never
exceed 73.13**. A *perfect* predictor therefore scores (100+100+100+99+73.1)/5 = **94.4**.
Realistic local ceiling for this model class is **~87.5**. Do not set targets above that without
first explaining which bound above is wrong.

PIV white noise is only **16.0% (u) / 10.9% (v)** of the scored variance — below the 30% threshold
for explicit noise modelling. The TKE floor is *sampling* variability of a turbulent variance over
20 frames, which is baked into the target and cannot be removed.

### WP1 + WP2 results (2026-07-30) — best is 86.12, +0.40

Best model: `Track_1/unet/wp2i_s5.pth`, applied with `compose(..., alpha=0.75)` and the fitted
`c × optimal h(RMSE × |window|)` SPS band. **Not yet packaged into `Track_1/submit/`.**

| change | final | delta |
|---|---|---|
| shipped `unet_mse_e150` | 85.72 | — |
| + per-sample relative-L2 loss in **physical** units, drop the 40 dead `z` channels | 85.96 | **+0.24** |
| + variance head, decoupled, `alpha` 0.75 | 86.12 | **+0.16** |
| stride 5 → 2 · augmentation · width 96 · batch 16 | 85.86–86.14 | flat |
| WP2 trained *through* the composition | 85.12 | −0.84 |
| variance head detached from the trunk | 85.94 | −0.18 |

Traps found, all of which cost real points:

- **Never train through `compose`.** With `alpha=1` the raw fluctuation's magnitude divides back
  out and is unconstrained; the trunk drifts to raw TKE error 1.533, worse than predicting no
  fluctuation. `alpha=0` then recovers **79.20**, not the baseline — the plan's "safety valve"
  argument is wrong. Use `--alpha-train 0` and compose only at inference.
- **The variance head must be allowed to reshape the shared trunk.** Detached it reads the
  mean-flow latent and reaches corr 0.679 — *worse than copying the input window's std* (0.756).
  Shared it reaches **corr 0.907 / rel-L2 0.358**. Sharing the representation is necessary;
  sharing the gradient is what makes it work.
- **`sigma_hat` is the wrong quantity for the SPS band** (it predicts flow fluctuation, not model
  error). A per-element half-width proportional to error is also wrong: the integrand
  `exp(-2h/sigma)·erf(h/(s√2))` **saturates near `sigma/2`**, so use `optimal_half()`.
- **Time must be measured outside the prediction loop**, after warm-up, as a median of repeats.
  Inline timing swings `final` by ±0.3 — larger than most effects being measured.
- **Width 96 is worse than width 64** (86.01 vs 86.14) and slower. It also thrashes at batch 32 in
  8 GB: 183 s/epoch vs 23 s/epoch at batch 16.

**The binding constraint is 74 trajectories, not the model.** Train field loss 0.066 ≈ seen
rel-L2 0.049 against unseen 0.084; six recipes all left unseen rel-L2 pinned at 0.084–0.089.

### Round 2 (2026-07-30 evening) — read this before proposing anything

**Local validation does not predict leaderboard ranking.** Two submissions:

| | local unseen | leaderboard | `mvpe` local | `mvpe` leaderboard |
|---|---|---|---|---|
| `unet_mse_e150` | 85.73 | **83.108** | 0.077 | 91.784 |
| `wp2i_alldata` | 86.09 | **83.019** | 0.079 | 90.499 |

Local says +0.36, the board says −0.09. A **0.002** local MVPE difference became **1.28** on the
board. Both validation sets sit inside the training Re range; the hidden set does not. Every
local delta measured so far is ≤0.4, i.e. below the transfer noise. **Use local numbers as a
sanity check, not a ranking signal.** Prefer mechanism over decimals.

Leaderboard decomposition of those two: `tke +1.30`, `mvpe −1.28`, everything else ≈0. The
variance head transfers; something costs MVPE.

**Attribution (all `val_unseen`, band `c × optimal h(RMSE)`):**

| model | loss | var head | λv/λt | rel-L2 | TKE | MVPE | final |
|---|---|---|---|---|---|---|---|
| `wpA0_mse_nohead` — control | MSE | — | — | 0.083 | 0.624 | **0.077** | 85.78 |
| `wpA_mse_var` | MSE | yes | 1.0/0.5 | 0.092 | 0.539 | 0.088 | 85.73 |
| **`wpA2_mse_var_lo`** | MSE | yes | **0.3/0.15** | 0.091 | 0.547 | 0.081 | **86.07** |
| `wpC_selfnorm` | MSE | yes | 0.3/0.15 | 0.091 | 0.553 | 0.082 | 86.00 |
| `wp2i_s5` | rel-L2 | yes | 1.0/0.5 | 0.089 | 0.546 | 0.079 | 86.09 |

- **The relative-L2 loss was NOT the cause of the MVPE loss** — under MSE it got *worse* (0.088).
  `wpA0` (MSE, no head, same `UNetTwoHead` trunk) reproduces 0.077 exactly, so dropping the `z`
  channels is MVPE-neutral. **The cost comes from the variance head sharing the trunk**, under
  either loss.
- **`lam_v`/`lam_t` must be rescaled with the loss.** Under MSE the field term converges to
  ~0.042 while the variance terms give `1.0×0.267 + 0.5×0.196 = 0.365` — the trunk is driven
  ~10:1 by the variance objective. 0.3/0.15 fixes it (MVPE 0.088 → 0.081, final 85.73 → 86.07).
- **Self-normalization (`--norm self`) is locally neutral (86.00 vs 86.07).** That is the pass
  condition, not a failure: both val sets are inside the training Re range, so the thing it fixes
  is invisible locally. It can only be tested on the board.

### Round 3 (2026-07-31) — 86.39, and the forecasting ceiling is now measured

Full detail: `Track_1/report/report_20260731.md`. **Best local: 86.39** =
`wpA0_mse_nohead` (field) + `wpF1_varonly` (sigma) + cross-fit sigma calibration, `alpha=0.75`,
band `c × optimal h(RMSE)`. Not yet packaged.

**Run-to-run noise is σ = 0.040 final** (three identical-recipe runs, error metrics only; the
old ±0.1 bundled in `time_score` jitter). rel-L2 σ 0.0002, TKE σ 0.0060, MVPE σ 0.0016.

**The variance head must NOT share the trunk — this reverses run H.** Run H's detached head had
130 parameters and failed. A *width-32 variance-only network* (`--lam-f 0`) matches the shared
head's sigma to three decimals (0.3607 vs 0.3608) while leaving MVPE at the no-head control's
0.0773 instead of 0.0817, and costs only 0.087 ms. Decoupling is worth +0.15 final; the shared
trunk was pure downside and is the likely cause of the hidden set's −1.28 mvpe.

**Post-hoc sigma calibration is worth +0.16 final, free.** Least squares on
`[head_sigma, window_std, history_std, 1]` per channel takes sigma rel-L2 0.3593 → 0.3378.
**Fit it only on trajectories the model never trained on** — fitting on `val_seen` gives the
head all the weight (0.982) and buys +0.008, because the head is in-sample there.

**Every subscore is now floor-limited. Remaining forecasting envelope is ~+1.1, to ~87.5.**

| subscore | now | measured bound | how bounded |
|---|---|---|---|
| `rel_l2` | 95.8 | 97.0–97.4 | mean sampling floor 0.0531 ⊕ PIV noise 0.007–0.031 |
| `tke` | 79.8 | ~81 | 10-frame half-split ⇒ ≈0.47 irreducible |
| `mvpe` | 96.3 | **96.75** | perfect-true-mean predictor at the probes = 0.0670 |
| `time` | 98.0 | ~99 | 0.317 ms/window |
| `sps` | 62.1 | 64.9 | oracle band, hindsight widths |

Corrections to earlier rounds: **`mvpe_score`'s ceiling is 96.75, not 100** (headroom +0.11, not
+0.76), and **PIV noise is 0.6–13% of fluctuation energy, not 16%/10.9%** — the `rel_l2` floor is
dominated by the *time-mean sampling term*, not measurement noise.

**Closed with measurements — do not revisit:**
- *Residual / high-frequency reconstruction, incl. a second FNO.* Prediction–target spectral
  coherence is 0.79 at the shedding scale but <0.2 above k≈15; energy-weighted Γ² = 0.308 caps
  any per-wavenumber filter. Best variance-preserving per-k gain: **+0.013**. FNO is also the
  wrong tool — mode truncation is a low-pass, and 15.5 ms costs −2.2 final. `diag_spectral.py`.
- *The time-mean as a separate lever.* Already at its sampling floor: 0.0503 measured vs 0.0531
  irreducible. `diag_mean_floor.py`.
- *Loss reweighting.* `--w-mean 3` (+1.2σ), `--w-mean 8`, `--lam-p 0.3` (−0.8σ). Adding the
  scorer's own MVPE functional to the loss does not improve MVPE.
- *More capacity, anywhere.* Width 96 lost for the field net; width 64 loses for the variance
  net (sigma 0.3460 vs width 32's 0.3378, 2.7× latency).
- *In-training history channels* (`--hist` on the field trunk): marginal sigma gain, costs MVPE.
  Feed history to the *separate* variance net instead, or use the post-hoc calibration.

### Track 1 `predict()` receives every test window at once — and the windows tile

`IN_STEP = OUT_STEP = INTERVAL = 20` (`local_eval.py:48-51`), so **window k's target IS window
k+1's input**, verified bit-exact (`maxabsdiff 0.0`) on `3750_0.h5`. 328/336 windows on
`val_unseen`. Returning `input_array[k+1]` scores **93.55 local** (rel-L2 0.0023, TKE 0.0141,
MVPE 0.0021) — which reconciles the leaderboard's otherwise impossible 93.3, since PIV noise
caps a physically perfect forecaster near 92 and `sps_score` cannot exceed 73.11.
**No action taken. Ask the organizers before treating this as available.**

Legitimately, the same batching allows **transductive conditioning**: per-trajectory statistics
from all ~42 windows, reading no targets. Trajectory boundaries are detectable from the
per-window mean-speed jump (within ≤0.046, across ≥0.108, 0/335 errors at threshold 0.077) when
`metadata["sample_id"]` is unavailable.

### Round 4 (2026-08-03) — 86.53, and two earlier conclusions were wrong

Full detail: `Track_1/fno_deeponet/report_20260802.md`. **Best local: 86.53** =
`fno_div_hi` — a **0.90M-parameter FNO** (width 24, modes (8,12), 4 layers) with a divergence
loss at λ 1.0, its own variance head, `alpha=0.75`, band `c × optimal h(RMSE·|w|)`.
0.106 ms/window, 3.6 MB. Not yet packaged.

**1. The "FNO is dominated" verdict was a size artifact — reverse it.**

| | params | ms/win | final |
|---|---|---|---|
| official `sim_real_ft` FNO | 50.4M | 15.5 | 83.06 |
| `fno_s` (ours) | 0.90M | 0.11 | 85.95 |
| **`fno_div_hi`** | **0.90M** | **0.11** | **86.53** |

The 50M checkpoint was slower *and* worse. FNO is flat across 37× params (85.87–85.94), so use
the smallest. CNO and Transolver remain disqualified; FNO does not.

**2. Physics losses were rejected too broadly — the divergence term works.**
`plan_20260730.md` ruled them out because `rms(∇·u)/rms(|∇u|)` is 0.42–0.49 on real PIV. That
correctly rules out `∇·u → 0` and *only* that. **Matching the divergence the target actually
has** (`--lam-div`, `rel_l2(div(pred), div(target))` in physical units) is worth:

| architecture | Δ final |
|---|---|
| FNO 0.90M | **+0.58** (14σ) |
| U-Net 1.91M | +0.11 (2.8σ) |
| DeepONet | +0.08 |

It improves **rel-L2, TKE and MVPE simultaneously** — the only change in four rounds to do so
(0.0986→0.0862, 0.5360→0.5082, 0.0916→0.0757). λ_div optimum is shallow: 0.3–3.0 all within
0.05, best at 1.0 for FNO and 0.3 for the U-Net. Likely mechanism: divergence is a
gradient-domain quantity, so it constrains what a mode-truncated spectral operator does between
its retained modes — hence the gain ordering FNO ≫ U-Net > DeepONet.

**Closed this round, measured:**
- *Spectrum matching* (`--lam-spec`, log radial energy): −0.01 / −0.10 / −0.14 on FNO / U-Net /
  DeepONet. Above k≈15 coherence is <0.2, so added amplitude lands in the wrong phase.
- *DeepONet*: 81.09 → 84.30 with modern training, then plateaus. MVPE 0.1345 vs FNO's 0.0768;
  tripling p from 128→512 barely moves it. Rank-*p* fixed-basis limit, as predicted.
- *Ensembling at this operating point*: ×2 and ×3 give the best field quality in the project
  (rel-L2 0.0818, MVPE 0.0723) but **score lower** — at e≈0.08 the `100/(1+0.5e)` curve is so
  flat that accuracy is nearly free to the score while latency is not.
  **REVERSED in round 5**: that used forward-only latency. End-to-end, ensembling is +0.20 —
  see Round 5 item 3.
- *Decoupled pair on FNO*: neutral (−0.07). FNO's own variance head does not damage its field
  head the way the U-Net's did.
- *Long context* (`--ctx 80/140`, sim): monotonically worse despite a 60–75 frame shedding
  period. *SWA*: null (+0.05) — OneCycleLR anneals to ~0 so the averaged iterates are identical.
- *Analog forecasting* (Lorenz method of analogues, sim): rel-L2 0.259 vs the U-Net's 0.120.
  1000 frames is far too few for analogues in a system this dimensional.

New code: `unet/models_alt.py` (`FNO2d`, `DeepONetCh`, `divergence`, `radial_spectrum`,
`build`), `train2.py --arch/--lam-div/--lam-spec/--ctx/--swa/--lam-f`,
`fno_deeponet/eval_sweep.py`, `fno_deeponet/eval_phase3.py`.

### Round 5 (2026-08-03) — 86.30 PACKAGED, and the local time_score was wrong all along

Full detail: `Track_1/fno_deeponet/report_20260802.md` §7–12. **Shipped:
`Track_1/submit_fno/submission.zip`** — a 3-seed ensemble of 0.90M FNOs with velocity-gradient
matching. **This is the first packaged model in five rounds**; every previous round's best is
recorded as "not yet packaged" and therefore never scored.

**1. `eval_wp.py` times the FORWARD PASS ONLY. Every `time_score` in every report is inflated.**
The evaluator times all of `predict()`. Measured on the same 336 windows: 0.109 ms forward-only
(`time_score` 98.79) against **1.19 ms end-to-end (`time_score` 96.12)** — **−0.59 on `final`**,
applied uniformly to every historical number here. Rankings are unaffected (one architecture),
absolute numbers are not. The API forces three `(N,20,32,64,3)` float32 arrays out, ~0.5 GB of
writes at N=336, which is a floor no model change touches. Use
`submit_fno/validate_strict.py` for any number quoted against the leaderboard.

**2. It is not divergence — it is BOTH INVARIANTS of ∇u.** `--lam-grad` (match the target's full
velocity-gradient tensor) scores **86.68** vs `--lam-div`'s 86.53, replicated over three runs.
Decisive ablation: `--lam-div 1.0 --lam-vort 1.0` *without* strain rates reaches 86.67, i.e.
identical to the full tensor, while either invariant **alone** gives 86.53–86.57. So constraining
the trace and the antisymmetric part is what matters; the traceless symmetric part is already
fine. λ plateau 1–3 (0.3 → 86.57, 10 → 86.44). U-Net reproduces the ordering (+0.13).

**3. Ensembling was rejected on the wrong cost model — it is worth +0.20.** The old −0.07/−0.10
used forward-only latency, where a 2nd member nearly doubles cost. End-to-end the forward pass is
0.109 of 1.19 ms and the rest is fixed API overhead a 2nd member does **not** duplicate — if
members are averaged **on the GPU before the host transfer** (`_fno.predict_fno` does this:
average raw field and sigma, then `compose()` once). 3 seeds = 86.30; 7 members has the best
rel-L2/TKE/MVPE in the project but ties at 86.22, accuracy gain exactly cancelled by latency.
**Use 3.**

**4. σ for this recipe is 0.015, not 0.040.** Three seeds: 86.67 / 86.69 / 86.66. The 0.040 in
round 3 was a *different* recipe; do not reuse a borrowed σ. At 0.015, deltas of 0.05 are real.

**Closed this round, measured:**
- *Sim → real pretraining.* **Null, and not for normalizer reasons.** Self-norm both stages:
  +0.08 vs its own control, but `--norm self` itself costs −0.20, so the bundle loses. Sim-norm →
  real-norm (stage 2 in the winning config): **+0.00** (86.68 vs a from-scratch mean of 86.673).
  Sim helps a handicapped model and adds nothing to a healthy one. Strategy item 6 is now
  measured, not untested. New flag `train2.py --init` warm-starts and prints the copied-tensor
  count so a partial load cannot masquerade as from-scratch.
- *Post-hoc sigma calibration on the FNO's own head*: TKE 0.5003 → 0.5047, everything else flat.
  It helps only a **separate** sigma net (round 3's +0.16, Phase 3's +0.06); a well-conditioned
  shared head just gets noise, which `compose()` puts straight onto TKE.
- *Laplacian matching*: **not built.** λ_grad 1.0 and 3.0 tie exactly, so "weight high
  wavenumbers harder" is falsified — the λ plateau is the evidence.

**Engineering, worth +0.26 on its own** (2.425 → 1.19 ms, error metrics bit-identical): never
materialize the half-width to full size (`band` is `(1,20,…)` and broadcasts), write `u,v` into a
pre-zeroed output instead of `np.concatenate`, upload whole windows and slice on GPU, batch 64.

### Clean simulation data is HARDER to predict than noisy PIV

`wpD_sim` — same recipe, `u`/`v` only, sim's own normalizer, matched window count (11,132 vs
12,040), evaluated on held-out **sim** trajectories:

| | real | sim |
|---|---|---|
| rel-L2 | 0.091 | **0.120** (+32%) |
| MVPE | 0.081 | **0.102** (+26%) |
| TKE | 0.547 | 0.486 |

Mechanism: relative transverse fluctuation `STD_v/MEAN_u` is **0.25 for sim vs 0.103 for real**.
PIV's finite interrogation-window resolution low-pass filters the measurement, and the 32×64
sub-sampling filters it further — **the scored data is a smoothed view of the flow, and smooth
fields are easier to predict.** The CFD resolves eddies the PIV cannot see.

Consequences: **the denoise → predict → renoise route is closed** (noise is only 6% of the error
energy; renoising doubles that term since the target's realization is unknowable; and the
noise-free target is not the easier problem the idea assumed). **The TKE floor is not a PIV
artifact** — sim TKE 0.486 vs real 0.547, same regime.

Sim normalizer constants (fitted on all 100 trajectories, `[u, v]` only, 32×64):
`MEAN_SIM = [0.858663955, -0.004962022]`, `STD_SIM = [0.443825609, 0.214743194]`. Sim's scale is
~6× real's — normalizing sim with the real constants puts the input at 7.3σ.

### Round 6 (2026-08-05) — the rules changed; the band is now solved, and it is finished

Full write-up: `Track_1/report/report_20260805_v9.md`. Kit **v9** replaced the scoring program
and the test set (see the box under *Scoring*). Board **79.733** for the July 30 U-Net package
(`Track_1/submit/`), which is what was submitted — not the FNO.

Facts worth not re-deriving:

* ~~**`time_score` IS A CONSTANT ~91.7.**~~ **FALSIFIED in round 9** — see that section. Two v9
  board rounds (U-Net 5.976 ms board / 2.648 local; FNO ×3 5.931 / 1.282) happened to land
  0.045 ms apart and were over-read as "harness overhead". Round 9's 8-member package is
  **faster locally** (2.211 ms) and **2.09 ms slower on the board** (8.020 ms). Board time
  tracks the number of **sequential model invocations**, not local wall time. Members cost
  ~0.42 board ms each ≈ **−0.015 `final`** each — cheap, not free.
* **`predict()` is 1.390 ms/window locally, and the "0.109 ms forward pass" quoted all through
  rounds 1–5 is 8% of it.** Profile: H2D 0.047, three forwards + rescale 0.695, `.cpu().numpy()`
  + strided host write 0.251, `pred ± band` 0.390, `isfinite` 0.066. Only of local interest now
  — see the constant-time bullet above.
* **Two very different models land within 0.4% of each other on board rel-L2** (U-Net 0.11787,
  FNO 0.11753) across two different test sets, while differing by 0.008 locally. Local accuracy
  deltas have stopped transferring; treat a local gain under ~0.4 as unproven.
* **The loss weights the wrong channel.** `--loss mse` is in normalized units, so v gets ~37×
  u's weight per unit *physical* error, while u carries 85.9% of the rel-L2 error energy and
  every scored metric is physical. `--chan-pow` now interpolates (0 = old, 1 = physical).

* **Error is governed by angle of attack and by almost nothing else.** Per-trajectory rel-L2:
  AoA 0° ≈ 0.057, 5° ≈ 0.044, 10° ≈ 0.073, 15° ≈ 0.103, 20° ≈ 0.105, monotone from 5° up, while
  Re spans 7× across the same table and explains nothing. **The board's 0.1179 is above our
  hardest single trajectory (0.1143)** — no mixture of what we have reproduces it. The AoA 15–20
  subset (0.1082) is the only local proxy in the right neighbourhood, and it is 5% from the
  board where the full split is 46% away. Evaluate there: `compare_v9.py --aoa 15 20`.

Nulls, measured, do not retry: **ALPHA is still 0.75** under v9 (re-swept with the band refitted
at every point; SPS peaks at α=0.45 but not enough to outrun TKE). **Blending the predicted
time-mean toward the input window's mean** makes MVPE monotonically worse, 0.0698 → 0.1240 —
the model's mean beats persistence by a lot. **Ensemble spread as an adaptive band**: correlates
with |error| at only 0.347 once the per-element mean is removed, and its best variant is *worse*
in-sample while better out-of-sample, i.e. noise. **A fitted fluctuation gain on top of ALPHA**
(scalar, per-pixel, per-pixel/time): every form gains in-sample, two of three lose out-of-sample
— the variance head has no systematic residual left. **Input-noise augmentation**: 0.05 → 83.163,
0.15 → 83.181, against a seed spread of 83.177–83.274. **`--chan-pow`** (§7 of the report — the
MSE loss weights v ~37× per unit physical error while u carries 86% of the scored error energy):
falsified monotonically, p = 0.25 / 0.5 / 1.0 → 83.254 / 83.116 / 83.010. The weighting is right
as it is. **Model capacity**: `fno_s`/`m`/`l`/`xl` span only 0.31 mean5 across 37× parameters
(82.02 / 81.86 / 82.17 / 82.32) while the 0.90M model *with* the gradient loss scores 83.20.
**A 4th ensemble member**: 1→2 is +0.97 SPS, 2→3 is +0.23, a 4th under +0.1 for 0.11 ms.
Three stays.

Run-to-run σ is **~0.05 in v9 mean5**, not the 0.015 quoted in rounds 3–5 — that figure was the σ
of v7's `final`, whose logistic SPS compressed exactly the variation v9 exposes.

### Round 7 (2026-08-05 evening) — 79.985, and two false positives caught

Board: rel 94.5754 / tke 78.5278 / mvpe 93.9100 / time 91.7264 / sps 37.4680 → **79.985**, up
from 79.733. Every subscore improved. Full write-up: `Track_1/report/report_20260805_v9.md` §11.

* **The board's errors are heavy-TAILED, not large.** Board rel-L2 is only **1.06×** the AoA
  15–20 proxy, but SPS behaved like a **1.39×** stretch (`Track_1/calib_board.py`, inverted
  from the measured board SPS for a known band). rel-L2 is an rms; SPS asks whether each
  element landed *inside* its interval, and an element just outside forfeits everything while
  barely moving rel-L2. Per-metric board inflation: rel **1.13×**, tke **1.27×**, mvpe
  **1.49×**, per-element tail **1.39×**. Size the band against the tail — it is the default now.
* ~~**`final − mean5` = +0.727 and +0.743.** Working rule `final ≈ mean5 + 0.73`.~~
  **FALSIFIED in round 9**: the third offset was **+0.903**, and mean5 went *down* while `final`
  went *up*. It was two coincidences, not a rule. Use the round-9 weights instead.
* **ALPHA 0.90 was a contamination artifact.** Re-swept at board error scale it looked worth
  +0.103 mean4 over 0.75 on the *all-data* models; the held-out control gave **+0.029**. The
  all-data models trained on `val_unseen`, so their TKE there is partly memorized and unusually
  cheap for α to buy. **ALPHA stays 0.75.** Second false positive in two rounds from tuning on
  all-data checkpoints — always run a held-out control before shipping anything fitted to them.
* **The band is closed.** Three independent attacks — exact solver, AoA fit split,
  board-calibrated inflation — each returned ~+0.5 SPS and the response curve is flat either
  side of every optimum. Stop.

### Round 8 (2026-08-06) — capacity is bad, diversity is good, and both only because time is free

Report §12. Eleven runs, all converged (last-30/prev-30 loss ratio 0.25–0.32).

* **Model capacity is monotonically HARMFUL with the gradient loss.** AoA 15–20 proxy: width 24
  **81.67**, modes (12,20) 81.54, width 48 81.33, width 64 81.17 — while *training* loss goes
  the other way (0.069 → 0.038). With 74 trajectories and a dense per-pixel gradient signal,
  capacity buys memorization. Note this **inverts** the pre-gradient-loss ladder (+0.30 from
  `fno_s` to `fno_xl`); do not cite that ladder as evidence for scaling any more.
* **Ensemble diversity is monotonically GOOD, and architecture beats seeds.** Held-out sets:
  3×w24 74.418 → +m1220 74.490 → +w48 74.563 → +w64 74.614 → +cp0.25 74.645 → 9 members 74.657.
  Each new *architecture* is worth +0.05…0.07; each extra *same-width seed* only +0.01…0.03.
  Adding w48 (individually −0.34) beat adding a fourth w24 seed. This only became affordable
  once time_score turned out to be constant.
* **Board time slope, measured from two points: 0.033 ms board per ms local.** So ensemble size
  is unconstrained — even a 3 ms/window local package predicts `time_score` 91.7.
* **THIRD contamination artifact.** The all-data ensemble sweep makes `fno_all_w48` look worth
  **+0.745** where the held-out experiment measures **+0.10** for the same change: 3.6M
  parameters trained on `val_unseen` recall the proxy that scores them. Standing rule now:
  **nothing gets decided on a number computed from an all-data checkpoint** — they supply
  weights, never evidence.

### Round 9 (2026-08-07) — 80.115, and the scoring function is finally readable

Board: rel 94.6342 / tke 78.8825 / mvpe 93.8030 / time 90.5066 / sps 38.2338 → **80.115065**.
Analysis and plan: `Track_1/report/report_20260807_round9.md`. Nothing new was trained or
submitted this round; the work was reading three board rounds and the literature.

* **`final_score` is a weighted mean of the five subscores, and the weights are now bounded.**
  Three rounds + Σw = 1 leave a one-parameter family, but most weights barely move along it:
  `time` **0.034–0.099**, `sps` **0.171–0.189**, `tke` **0.219–0.273**, `rel + mvpe`
  **0.493–0.522** (their split is *not* identified — they moved together in all three rounds).
  Midpoint: `[0.266, 0.244, 0.241, 0.069, 0.181]`. Geometric and harmonic means miss by 5
  points, so the linear family is the right one. **Not yet validated** — three points, five
  unknowns, exact fit. Predict the next board `final` *before* submitting.
* **Latency is ~3× cheaper than every earlier plan assumed.** 20 ms board costs 0.33 `final`,
  50 ms costs 0.78, 100 ms costs 1.21. A 25× inference-cost increase costs ~1.2 points, so
  test-time adaptation, multi-pass inference and deep ensembles are all inside budget now.
* **The accuracy→score exchange rate, measured** (`Track_1/exchange_v9.py`, held-out 6-member
  ensemble on the AoA 15–20 proxy, errors scaled synthetically, band refit at each point):
  **+0.089 `final` per 1 % of error removed**, linear to −20 %. Decomposition of a 10 % cut:
  **sps +2.22 (45 % of the gain)**, tke +0.75 (21 %), rel +0.64 (19 %), mvpe +0.57 (15 %).
  → **SPS pays 2.3× more per unit of error than any other channel** despite its 0.181 weight,
  because interval containment is far steeper than `100/(1+0.5e)`. The band is closed; reach SPS
  through accuracy. → ~~**TKE is a trap**: it moves only 0.75 per 10 % and sits near its 0.47
  PIV-sampling floor — ≤ +0.6 `final` remains there for anyone.~~ **FALSIFIED in round 10** —
  the arithmetic was right and the conclusion was not. 10 % of a 0.483 error is a small step
  down a long curve: an oracle per-pixel variance field is worth **+7.70 `final`** and the
  first quarter of that gap **+1.21** (`Track_1/tke_branch_v9.py`). The 0.47 "floor" never was
  one — the head is at 0.483, *below* the per-trajectory-mean estimator (0.560) and below the
  disagreement between two independent 20-frame realizations (0.783), which is impossible for
  a quantity limited by sampling noise. TKE is the **largest measured lever**, not a trap.
  → Because SPS scores per-element containment, **tail reduction beats mean
  reduction**: prefer robustness and regularization over capacity and average-case accuracy.
* **Scale**: 80.1 → 81.0 needs ~10 % error reduction; → 82.0 needs ~21 %. Going from 3 to 8
  ensemble members bought ~1.6 %.
* **DPOT is not the best model on the airfoil**, contrary to a common reading of RealPDEBench.
  On the Foil dataset U-Net wins every reported metric (rel-L2 0.0145 vs DPOT-L 0.0161, DPOT-S
  0.0174; same ordering at 1/2/3 autoregressive rounds). DPOT-L is best only *averaged over the
  five datasets*. Their Foil numbers are also not comparable to ours — 99 traj × 3990 frames ×
  128×256 against our 82 × 607 × 64×128 — and DPOT's next-frame autoregressive shape is wrong
  for a one-shot 20 → 20 task. Full argument in §5 of the round-9 report.
* **Untapped data, free**: `model.py:load_trajectories` does `[:, ::2, ::2]` on **64×128** raw
  frames. Phase (0,0) is the scored grid; phases (0,1), (1,0), (1,1) are three more *real* 32×64
  views we discard — a 4× augmentation in the regime where capacity is already harmful.

### Round 10 (2026-08-08) — the variance branch, opened and largely closed

Report: `Track_1/report/report_20260808_round10.md`. Nothing submitted; the zip is still the
Aug 6 build. Baseline for every number here: **wfinal 81.895 ± 0.055** over 4 held-out seeds on
the AoA 15–20 proxy (`compare_v9.py --aoa 15 20`; `wfinal` = round-9 weights on local errors).

* **The three pieces of a prediction, priced** (`Track_1/oracle_split_v9.py`). Splitting
  `y = mean_t(y) + σ(y)·shape(y)` and handing the model the target's value for one piece at a
  time: **σ +7.71**, time-mean +2.72, shape (phase-resolved fluctuation) **+1.95**, everything
  +21.04. → **The hardest part of the problem is worth the least.** Replacing `shape` cuts
  rel-L2 by 34% and buys +1.95; replacing σ leaves rel-L2 *unchanged* and buys +7.71. Near
  e ≈ 0.1 the `100/(1+0.5e)` curve is flat — **stop optimizing rel-L2**, TKE and SPS carry the
  response.
* **Oracles are upper bounds, not forecasts.** The +7.71 σ headroom yielded **+0.08** in
  practice across five arms. Quote the oracle to rank directions, never to predict a gain.
* **The variance-trunk relationship is not monotone.** Fixed field, sigma swapped, TKE:
  `--detach-var` 0.5625 (0% of the trunk shaped by variance) / shipped shared trunk 0.4851 /
  **`--var-mlp` 0.4660** / `--lam-f 0` 0.4892 (100%). **The field task is a useful auxiliary for
  the variance task**; a trunk trained on variance alone loses structure the head needs. The
  shipped arrangement is already optimal on that axis.
* **ADOPTED — `--var-mlp`**: give `head_logstd` the field head's 128-wide MLP instead of a
  single 1×1 conv (a *linear* probe on a 24-channel trunk). 3 408 parameters, no extra
  invocation. 3 seeds **81.998 ± 0.023 vs 81.895 ± 0.055, +0.103 at 3.4σ**; 3-vs-3 ensemble
  **78.387 vs 78.305, +0.082**. rel-L2, TKE and SPS all improve — no hidden trade.
* **Closed, measured**: **`--lam-t`** — tripling the TKE loss weight moves TKE by 0.0001 and
  costs rel-L2 (the head is not under-trained). **`--detach-var`** — best rel-L2 of any arm,
  worst TKE (0.5625, no better than a trajectory-constant predictor); reproduces round 4's
  Run H on the U-Net. **`--lam-f 0`** (separate variance network) — worse than the shared
  trunk, so the extra inference invocation buys nothing. **Post-hoc sigma blending**
  (`sigma_blend_v9.py`, head + field std + input std + per-trajectory mean, log-space LSQ
  fitted on val_seen) — **−0.311** out-of-sample; third null for post-hoc sigma correction.
* **ALPHA re-swept under the round-9 weights**, held-out so it is a clean control: 0.90 beats
  0.75 by **+0.041** on the baseline and **+0.070** on the `--var-mlp` pair, both inside the
  0.05 noise floor. **ALPHA stays 0.75.**
* **`--var-mlp`'s gain is diluted twice, structurally.** ALPHA 0.75 discards a quarter of any
  σ_head improvement by construction (`0.75·σ_head + 0.25·std_t(field)`), and ensembling
  averages sigma across members before per-member head quality matters. That is why a 3.9%
  fixed-field TKE cut becomes +0.08 at the ensemble.
* **`train2.py`'s argparse defaults are NOT the shipped recipe.** Every shipped member uses
  `--lam-v 0.3 --lam-t 0.15 --alpha-train 0.0`; the defaults are 1.0 / 0.5 / **1.0**.
  `--alpha-train 1.0` computes the FIELD loss on the sigma-rescaled field, so the field head
  chases a target an untrained variance head is distorting — it collapses the model to rel-L2
  0.19 against 0.112. Three arms were lost to this. **Read the recipe off `fno_grad.pth`'s
  stored `config`, and keep a no-flags guard arm as the first line of any queue.**
* **WP1 built, verified, not shipped.** `_fno.py:StackedFNO` + `fuse()` collapse same-shape
  members into one invocation (7 members → 4), exact to **1.5e-8** with TF32 off. But it is
  0.35 ms/window *slower* locally, and its +0.07 rests on the round-9 inference that board time
  counts sequential invocations — one data point. Revisit when a second board round settles it.
* New tooling: **`final_v9.py`** (round-9 weighting as code, returns midpoint + admissible
  range, self-checks to 1e-13; wired into `compare_v9`/`ens_v9`/`alpha_board`/`validate_strict`
  — the falsified `mean5 + 0.73` is gone), `tke_branch_v9.py`, `oracle_split_v9.py`,
  `sigma_blend_v9.py`, `unet/runq.sh` + `queue.txt` (editable-while-running training queue).
  `train2.py` gains `--phases`, `--aoa-boost`, `--aoa-hi`, `--aoa-only`, `--var-mlp`;
  `model.py` gains `phases=` on `load_trajectories` (phase (0,0) verified bit-identical to the
  old `[:, ::2, ::2]`) and `trajectory_names`.

### Round 10b (2026-08-08 afternoon) — the band reopened, and the boundary/trunk rule

* **ADOPTED — `--lift-mlp`**, and the rule behind it. `lift` was `Conv2d(42, 24, 1)`: a LINEAR
  compression of the whole 20-frame input before the first spectral block. Giving it a hidden
  layer is worth **+0.076**, and it is **additive with `--var-mlp`**: together **+0.164** at
  5.9σ, **+0.146** at the 3-vs-3 ensemble, for 10 976 parameters.
  → **`--point-mlp` FAILS (−0.379).** Same linear 1×1 shape, four of them, but `point` is
  INSIDE the trunk and its training loss drops much further (field 0.052 vs 0.067). So the rule
  is **capacity at the BOUNDARIES helps, capacity in the TRUNK hurts** — which is round 8's
  capacity result seen from the other side, and the two now agree exactly.
* **Width is an interior optimum**, tested downward for the first time: w16 81.865 < **w24
  82.059** > w48 81.451. (w20 is untestable — `GroupNorm(8, 20)` needs divisibility.) The
  round-10 recipe does not rescue w48 (81.451 vs the old recipe's 81.536) and does nothing for
  w64 (81.359 vs 81.361). **The architecture search is finished.**
* **THE SPS BAND WAS NOT CLOSED — round 6 closed it against the wrong proxy.** Ensemble spread
  correlates with per-element |error| at only **0.347** after the per-element mean is removed;
  the **variance head's own sigma correlates 0.551**. `half = base·(σ/σ_ref)^p`:
  - **The obvious experiment reproduces round 6's negative and is WRONG.** Fitting `base` at
    p = 0 makes every p ≠ 0 worse in-sample *by construction*. Refit `base` per p — deflate the
    errors by `r**p`, since `|E| <= base·r**p` is exactly `|E|/r**p <= base` — and **p ≈ 0.45
    wins in BOTH directions** (in-sample 44.538 → 44.828, out 39.886 → 40.414).
  - **Smoothing the base is a second, separate gain.** It has 81 920 free parameters fitted
    from ~126 windows. A box blur over (T,H,W) cannot be selected in-sample (the unsmoothed
    band IS the in-sample optimum), so use leave-one-**TRAJECTORY**-out CV inside the fitting
    split: it picks k = 3, which is independently best on val_unseen.
  - **Total +0.729 sps = +0.132 `final`, no retraining.** `band.npy` → `band.npz`
    (`base`, `sigma_ref`, `p`); `predict_fno(..., sigma_out=)` exposes sigma.
  - Watch out: `fit_band.py`'s "board proxy 1.00x" column scores against **uninflated** errors
    and made an all-AoA fit look worth +6.9 sps. At a matched 1.39× the AoA 15-20 fit still
    wins. **`--fit-aoa 15 20` stands.**
* **WP1 fusion SHIPPED, and the earlier decision to shelve it was wrong.** "Certain local cost
  vs uncertain board gain" compares a local number to a board number. Regress the three v9
  rounds instead: board ms ~ **invocations** fits with residuals **±0.4 ms** (slope +0.316);
  board ms ~ local ms fits at ±1.3 ms. Under the better model 8 → 4 invocations is **+0.05**;
  under the worse one fusing costs −0.007. `_fno.fuse()` is score-identical (78.4925 vs
  78.4927) and exact to 1.5e-8 with TF32 off. **Extra same-architecture members are now free in
  invocation count.**
* **`w48` is out of the package.** Under fusion it earns back its own invocation group and no
  more (78.667 with, 78.668 without). Final: **5×w24 + m1220 + w64, 3 invocations.**
* **All-data training does NOT deflate the variance head**: mean sigma 0.00891 (all-data) vs
  0.00890 (held-out) on val_seen. A stored `sigma_ref` is therefore safe.
* Round-10 package projection, like-for-like against the 80.115 board: recipe+composition
  **+0.112**, fusion **+0.067**, band **+0.132** → **≈80.43**. The fusion term is the fragile
  one; if board time really tracks local time it becomes −0.007 and the package lands ≈80.36.

### Round 10 BOARD RESULT (2026-08-08) — 80.512025, +0.397

rel 94.5764 / tke 79.1528 / mvpe 93.6812 / time 91.1209 / sps 39.6263. Predicted 80.422 before
submitting (family range 80.373–80.465); actual landed 0.047 *above* the range. Plan for the
next round: `Track_1/plan/plan_20260809.md`. Best other team: **80.79**.

* **The band was the round**: sps **+1.393**, against a +0.95 forecast — roughly two-thirds of
  the gain. TKE came in near-exact (predicted 79.144, actual 79.153).
* **`rel_l2` −0.058 and `mvpe` −0.122 REGRESSED** where the held-out proxy predicted small
  gains. The package went 8 → 7 members and dropped w48 on a proxy margin of **0.001**. That
  margin was noise. **The AoA 15-20 proxy is not the authority on member count — the board is.**
  Restore w48 and add w24 seeds (free in invocations under fusion).
* ~~**`final_score` is a weighted mean with Σw = 1 and no intercept.**~~ **FALSIFIED in round
  10.** Fitting all four rounds that way needs `mvpe` weight **−0.256**. With an intercept
  allowed, non-negative fits survive and give **`sps` 0.139–0.321** (not 0.181), `time`
  0.043–0.120, and only loose bounds on rel/tke/mvpe. **SPS is worth up to 1.8× the round-9
  price.** Work in DELTAS between rounds (the intercept cancels), not the level fit.
* **BOARD TIME IS SOLVED**, four rounds to 0.064 ms:
  **`board ms = 4.447 + 0.324·invocations + 0.431·local ms`**, and **1 board ms ≈ 0.05 final**.
  R7 and R10 both ran 3 invocations and differ by 1.0 ms, so the invocation-only model of round
  10 was incomplete. **Local compute is NOT free** — round 10's fusion saved 1.62 ms of
  invocations and handed 0.60 ms back in local time (2.211 → 3.598 ms).
* Method rule earned twice now: **when testing a scaling or regularization parameter, refit
  everything downstream of it before comparing.** Fitting the SPS band's `base` once at p = 0
  is what made round 6 declare the band closed, and it nearly repeated in round 10.

### Round 11 (2026-08-09), in progress — `Track_1/report/report_20260809_round11.md`

* **Local time is 3.487 ms/window, not 1.383.** Inverting the round-10 `time_score` through the
  board-time model gives it. The per-window band added ~1.4 ms and was never re-profiled.
* **The band stage was 1.45 ms/window of avoidable memory traffic → 0.13.** Two changes: give
  the sigma scale factor a third, zero channel so `half` is one allocation and the full-size
  `np.concatenate` disappears; and run the three full-size ops through **torch**, which threads
  them (numpy's ufuncs are single-threaded and this is bandwidth-bound). SPS identical to 8e-8.
  **+0.028 final.**
* **`StackedFNO`'s lift no longer repeats a shared input.** Every member sees the same tensor at
  the lift, so `Conv2d(cin, K*width, 1)` replaces `grouped(x.repeat(1,K,1,1))`. Bit-exact.
* **Mode-padded fusion**: members may differ in retained modes; the spectral weights are
  zero-padded to the group max. m1220 now joins the width-24 group. **+0.012 final** (saves
  0.324 board ms of invocation, costs 0.199 local ms = 0.086 board ms of extra spectral work).
  **`w2` pads at the FRONT** — it multiplies `ft[-my:]`, so its row axis runs the opposite way
  in frequency from `w1`'s. Padding both at the end moves the field by 1.26.
* **An extra fused member is NOT free**: ~0.27 local ms ≈ **0.006 final** each. Round 8's curve
  is ~+0.012/member at 8→9, so past ~8 members the ensemble is roughly break-even. Above K = 10
  in one group the 8 GB card thrashes (K = 12: +1.26 ms/member).
* **The SPS band is closed** (`Track_1/band_cv_v9.py`, leave-one-trajectory-out CV, 84 configs).
  Pooling the fitting split buys **0.000** sps; the val_seen contamination costs nothing (the
  honest out-of-sample alternative has 2 trajectories and is 0.37 sps *worse*); p = 0.45 over
  0.50 is 0.026, inside the fold spread. k = 3 confirmed at every p.
* **The recomposition gain is genuinely one-dimensional** (`Track_1/gain_v9.py`): the full
  family `g = (a·σ + b·s)/s` has its ridge on `a + b = 1`, so five rounds of sweeping `alpha`
  alone lost nothing. But the optimum on that line is **alpha ≈ 0.90, not 0.75** — +0.094 here,
  +0.041…0.070 in round 10, three readings all positive. Re-measure on the final member set;
  if it ships, **`fit_band.py --alpha` must change with it.**
* **TF32 on the spectral matmul: null.** No speedup (+0.05 ms), no subscore change past 1e-5.
* **Longer context for sigma is undeployable.** `predict()` gets 20-frame windows; earlier
  frames exist only in sibling windows, which makes the submission order-dependent
  (`validate_strict.py` asserts permutation invariance to 1e-4) and leans on the same tiling
  assumption as the disclosed leak.
* **Seed sigma, measured on 4 baseline seeds:** wfinal σ = **0.017** on the AoA 15-20 proxy and
  0.030 on the full split; TKE σ = 0.0028. The often-quoted "0.055 noise floor" is pessimistic.
* **TKE is paid twice.** SPS's per-sample factor is `a_i = 0.5g(rel)+0.3g(tke)+0.2g(mvpe)`,
  `g(m)=0.5/(0.5+m)`. At board metrics a = 0.710, so SPS's ceiling is 71.0 against 39.6 scored.
  Board TKE 0.527 → 0.45 is `tke_score` +2.47 **and** +0.65 sps ≈ **+0.74 final**.
* **THE VARIANCE HEAD IS AT AN INFORMATION FLOOR, not a capacity floor.** Three arms, all null
  on the AoA proxy (82.038 / 82.047 / 82.049 against baselines 82.046 ± 0.017) **and all with a
  training `var` inside the baseline seed spread** (0.36562 / 0.36801 / 0.36796 against
  0.36708 / 0.36572 / 0.36698):
  `--var-hidden 256` (double the head's hidden width), `--var-deep` (head reads
  `concat(h1..h4)`), `--var-in-std` (head also reads the INPUT window's per-pixel temporal std).
  `var` is a mean squared log error against `std_t(y)` — the very quantity sigma is judged on —
  so **an arm that does not move `var` cannot move TKE**, and that is now the cheap screen: read
  the training curve before scoring anything. `--var-pred-std` (feed the head the field head's
  own predicted std) was cancelled on this evidence; the code is in the tree and bit-exact
  through `fuse`. What is NOT closed by this argument is sigma's **objective**: `l_var` is a
  log-MSE, uniform over pixels, while the scored `tke_rel_l2` is energy-weighted and only cares
  about the wake. `--lam-v 0.05 --lam-t 1.0` is the arm that tests it.
* **`l_var` IS NOT COMPETING WITH THE TKE OBJECTIVE — IT IS REGULARIZING IT.** `--lam-v 0.05
  --lam-t 1.0` drove the TRAINING tke loss from ~0.414 to 0.3667 (−11%) and drove held-out TKE
  the other way, 0.4800 → 0.5059; wfinal 81.463 against 82.046, a 34σ loss, with rel-L2
  0.1111 → 0.1165 and sps 42.34 → 40.73. Strip the uniform log-MSE and the head fits the
  training TKE field by memorizing it. **This explains round 10's "tripling `--lam-t` moves TKE
  by 0.0001"**: the objective was never the problem, and the term that regularizes was
  correctly resisting the term that overfits. Do not revisit `lam_t` upward, and do not try
  `--detach-var` with it — detaching protects the *trunk*, and the failure is in the *head*.
  The open question it leaves is the opposite one: `lam_v` has only ever been moved DOWN, or
  moved together with `lam_t` (1.0/0.5 in round 2, which lost 0.34 on MVPE because the trunk
  went 10:1 to the variance objective).
* **`--lam-v 0.6` closes the last direction, and completes the picture.** Training `var`
  improves MONOTONICALLY with `lam_v` (0.4941 / 0.367 / 0.36103 at 0.05 / 0.3 / 0.6) while
  held-out TKE is U-SHAPED with its minimum at the shipped 0.3 (0.5059 / 0.4823 / 0.4868).
  **Fitting log-sigma better does not predict the amplitude envelope better.** Sigma is at a
  GENERALIZATION ceiling: the readout cannot move `var` whatever its shape or inputs, the loss
  weights can move `var` but not held-out TKE, and 0.3 is already the optimum of the one family
  that moves it. Do not reopen sigma without a structurally new source of information.
* **The batch size is now measured at import, not hardcoded.** `submission._pick_batch` times
  a few sizes on whatever GPU it lands on (import is not timed). On this 8 GB card batch 12
  runs at 1.657 ms/window against batch 64's 2.135 — the fused width-24 group's intermediates
  stop fitting in cache — but that ordering is a property of one memory hierarchy and should
  invert on a larger GPU where the 1/batch launch cost dominates. Falls back to 64 on any error.
* **Round-11 package: 8 members, 3 invocations, ALPHA 0.90, band p 0.45/k 3, local 2.002
  ms/window** (round 10: 3.487). Predicted board **80.603** (+0.091); prediction recorded in
  `Track_1/plan/plan_20260809.md` before submitting.

### Architecture directions assessed 2026-08-09 — `Track_1/report/directions_20260809_architecture.md`

**The diagnosis after round 11.** Capacity, data, ensembling, the band and latency are all
closed with measurements. What is left: **we predict a conditional mean and bolt a hand-tuned
variance onto it, but ~38% of the score is distributional.** `tke` (a second moment) and `sps`
(an interval-coverage score) carry 0.47 of the 1.23 per-point weight and are our two weakest
subscores (79.15; 39.6 against a ceiling of 71.03). `rel_l2` and `mvpe` are point scores.
Round 11's `lam_v` U-shape says log-MSE is a **misaligned surrogate**, not a hard ceiling.

* **THE BET: an FGN-style probabilistic head.** arXiv 2506.10772, "Skillful joint probabilistic
  weather forecasting from marginals" — the engine behind WeatherNext 2. Sample a 32-dim
  Gaussian vector, feed it through parameter-shared **conditional-normalization** layers to
  sample new *weights* (perturbation in PARAMETER space, not output space); train on **CRPS** of
  per-location marginals; **one network call per sample**, 8x faster than GenCast's 40-step
  diffusion. Fits because CRPS is a proper scoring rule for what SPS measures, parameter-space
  noise gives spatially coherent samples from marginal-only training, and one forward pass is
  the only reason a generative model is admissible in a latency-scored task. `StackedFNO`
  already fuses weight-sharing samples into one invocation. It would replace ALPHA, the sigma
  head and `fit_band.py` with one learned object. **Tension to resolve, not ignore:** the scorer
  computes `tke` on `prediction`, so the submitted array must be close in L2 AND correctly
  dispersed; ALPHA is our current point on that trade and a generative model does not dissolve it.
* **MAYBE: phase as a co-trained second modality.** WN-C (`Papers/Track_1/s41586-026-10953-2`)
  deliberately does NOT run a tracker in the loop — it "griddifies" a sparse, physically
  meaningful modality and co-trains it with the dense field; its Extended Data Fig. 4 shows a
  two-stage "predict then track" pipeline loses the transfer benefit. Our analogue is not
  vortices but **shedding phase / Strouhal / convection speed**, derivable from the target by
  signal processing, no extra data. **Free diagnostic first:** split the per-window error into a
  phase/advection part and an amplitude part (find the error-minimizing spatial shift). Not a
  Q-criterion loss — every auxiliary physics loss except `--lam-grad` has been null.
* **REJECTED: graph neural networks.** Wrong inductive bias for a dense 32x64 uniform grid with
  fixed geometry. GraphCast/WN-C use an icosahedral mesh because the Earth is a sphere and
  lat-lon degenerates at the poles; we have no such pathology. The flow-reconstruction GNN in
  the reading list (arXiv 2411.08764) is for *sparse data in time-varying geometries*. Plus:
  capacity ladder points down, and a graph transformer costs scored latency.
* **REJECTED: KANO** (ICLR 2026, `Papers/Track_1/2509.16825v6.pdf`). Dual-domain spectral+spatial
  operator with KAN edges, for position-dependent PDEs. (1) Sells **parameter efficiency** (152
  params against FNO's 566k) — we are over-parameterized, not expressivity-limited. (2) Its core
  move, empowering the spatial basis, is what `--point-mlp` did: **−0.379**. (3) Authors concede
  "orders heavier FLOPs" and FLOPs are scored. (4) Its benchmarks are noise-free 1-D symbolic
  recovery and disjoint-family extrapolation; we interpolate in-distribution with an 8%
  board-vs-proxy gap. Its *premise* is fair (airfoil, FOV edge, 9.9% zero-target elements) but
  those elements are dropped from SPS entirely, and the lift already sees coordinates through a
  128-wide MLP. Cheap test if anyone insists: region-wise error maps, then a per-location FiLM.

**The pattern:** GNN and KANO are both proposals about **operator expressivity**, and rounds
8–11 say expressivity stopped being the binding constraint. What remains is that the fluctuation
is partly unpredictable (`diag_spectral.py`: only ~31% of fluctuation energy is coherently
predictable) and we model it with a log-MSE sigma head instead of a proper scoring rule.

### Round 12 (2026-08-29) — ported to an OSC A100, FGN and phase closed, U-Net joins the package

Full detail: `Track_1/report/report_20260829_round12.md`; raw rows in
`Track_1/plan/campaign_20260829.md`. One ~20 h autonomous session on a Slurm A100-40G (the
4060 box is gone); every binary artifact rebuilt from scratch. Git branch: `qwu/dev`.

* **Environment**: data at `/users/PAS1187/qwu/RealPDE/`, env `realpde` recreated (torch
  2.2.2+cu121). The kit `scoring.py` on this machine is a verified RECONSTRUCTION (Codabench
  is auth-walled from the cluster; agrees with `sps9.py` to 6.5e-8; MVPE is rel-L2 per probe
  COLUMN, mean over the 4 x-stations — a pooled-36-probe variant read −0.097 wfinal until the
  persistence/time-mean anchors caught it). Fetch the official kit v9 with a Codabench login
  to retire the last risk; TKE ddof=0 is the one remaining assumption. Recipe reproduces at
  +1σ (82.063 vs 82.046±0.017); 18.4 min/run; **the 12-core cpuset, not the GPU, binds** —
  never time anything while training runs, and don't judge liveness by nvidia-smi (reads 0%).
* **WP-1 FGN/CRPS probabilistic head: FAILED its pre-registered screen, closed.**
  FiLM-from-noise sample spread predicts the flow's σ at rel-L2 0.4507/0.4691 against the
  log-MSE head's 0.3293; every spread-σ composition loses ~0.7 wfinal. The CRPS term is
  harmless (its z=0 model ties the baseline seeds) but adds nothing. Parameter-space noise
  cannot manufacture amplitude information the trunk does not carry — round 11's information
  floor, measured a third way. Do not revisit generative σ without new information sources.
* **WP-2 phase/advection: DEAD.** Fluctuation-only re-registration (whose synthetic control
  recovers a planted (1,3) roll 100.0%) recovers 0.2% on the real model; zero shift preferred
  in 92% of windows. The fluctuation error is decorrelation/amplitude, not misregistration —
  no phase co-training, no phase-aligned frames, ever.
* **WP-3 mixed family: the U-Net member WINS — +0.091 proxy / +0.123 full wfinal on the 7-FNO
  ensemble, 4.5× the value of an extra FNO seed.** Round 8's "architecture beats seeds"
  replicates across families. Mode-diversity arms (m0608/m1016) are null once the set spans
  m1220+w48+w64+U-Net; composition closed. α=0.90 re-confirmed on the final member set.
* **Package (round 12): 11 members / 4 invocations** — 7×w24 all-data seeds + m1220 (one
  mode-padded StackedFNO group) + w48 + w64 + `unet_all_g1`, loading through `_fno.load_any`
  (dispatches on the blob's `arch`; `UNetTwoHead` vendored into `_fno.py`). Two latent
  `fuse()` bugs found and fixed: crash on a 2nd non-FNO member, and silent member DROPPING
  (`append` for `extend`). 82.6 MB, SUBMITTABLE, 1.208 ms/win on the A100 = 2.235 ms
  4060-equivalent (conversion r = 1.850, measured by re-timing the round-11 shape: 1.082 A100
  vs 2.002 recorded 4060).
* **Predicted board, recorded pre-submission**: rel 94.521 / tke 79.869 / mvpe 93.782 / time
  91.248 / sps 39.628 → **final 80.707 (+0.195)**, delta-based off round 10's confirmed
  80.512. TKE carries the round. **Not yet submitted** — submission is the user's call.

### Physics losses: rejected, measured

`rms(∇·u)/rms(|∇u|)` is **0.42–0.49 on real PIV and 0.28–0.34 even on the CFD**, and unchanged
when the resolution doubles (so it is not discretization error). A `∇·u → 0` penalty would fight
the target. Matching `div(target)` instead is also wrong: the scorer never looks at divergence,
and its correlated part (76%; only 24% is white noise) is dominated by out-of-plane `∂w/∂z`,
which is **not observable** from the 2D field the network sees. Momentum is worse — it needs the
unmeasured pressure. Measurements live in `Track_1/divergence.py`.

**Still untested:** a *detached variance head with capacity* (several convs on `u1.detach()`),
which would make the field head bit-identical to `wpA0` while letting the head work from a frozen
latent. Run H showed a detached 1×1 head is too weak (corr 0.679, below the trivial input-std
estimator at 0.756). Also untested: sim **pretraining of the trunk** (distinct from sim-only
training, which scores 69.0 — below free persistence).

## Strategy

> Items 1, 2 and 4 were written before the ceilings were measured and **overstate the available
> gains by roughly 10×**. The variance head was built and it works (corr 0.907); it is worth
> **+0.16 final**, not the +4 implied here, because the target's own TKE field carries ≈0.47
> irreducible sampling noise. Item 4's "per-element widths from a predicted-error head" was built
> and is worth +0.1 against an oracle ceiling of +0.6. Read
> `Track_1/report/report_20260730.md` before acting on any of them.

1. **Predict the variance field, not just the mean field.** ~~TKE is phase-free, so matching the
   per-pixel temporal variance of the 20 output frames is sufficient — no need to get eddies in
   the right place. A perfect variance head on top of a *trivial* mean predictor reaches 87.4 of
   the 92.1 oracle.~~ **Done, and the 87.4 figure was an artifact**: that oracle used the target's
   own per-pixel std, so it measured the value of *knowing* the answer, not of predicting it.
   The real structure of the fix still holds — decouple a mean-field prediction (drives rel-L2 +
   MVPE) from a per-pixel fluctuation-amplitude head (drives TKE) and rescale the predicted
   fluctuation to the predicted std — and it is implemented in `UNetTwoHead` / `compose`. It buys
   TKE 0.633 → 0.546 and **+0.16 final**. There is ~+0.5 left in TKE, total, for anyone.
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
   `exp(−width/σ)` with σ = 0.056 punishes it. **Done: 61.9 with `optimal_half()` on a per-element
   RMSE map scaled by per-window magnitude, +1.3 final over the default.** This item is now closed
   — an oracle that picks every width knowing the actual error scores only 64.9, so ≤+0.6 remains
   and no legal rule captures it. Do not spend more time on band forms.
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
  `matplotlib`, `pandas`** (`smoke_test_kit.py:63`). It **does** ship `requests`,
  `setuptools`/`pkg_resources` and `torch.utils.cpp_extension` (`smoke_test_kit.py:62`).
  `einops` ships in the kit's `_vendor/`. Never `import h5py` from `submission.py` — this
  is why `Track_1/unet/model.py` imports h5py lazily inside `load_trajectories()`.
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
  import *locally*. (Both **are** present in the T1 eval container, so this is a local-env pin
  only — an earlier note here claiming CNO could not be submitted was wrong.)

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
