# AI4PDE — NeurIPS 2026 RealPDE Competition

Our solution code for RealPDE: forecasting real PIV flow past a NACA4418 airfoil, with CFD as a
training-side asset. [Site](https://realpdecompetition.github.io/) ·
[Track 1 Sim2Real](https://www.codabench.org/competitions/17363/) ·
[Track 2 LTTTA](https://www.codabench.org/competitions/17385/) · remote `github.com/clockj/AI4PDE`,
working branch `qwu/dev`.

> **This file is loaded into every session. Keep it under ~30 k characters (Claude Code warns at
> 40 k) and keep it to what is true now.** After a board result: update *Current state*, edit in
> place any fact the result changed, append the round's write-up to [`ROUNDS.md`](ROUNDS.md), and
> put new rules in `ROADMAP.md` Part 1. `ROUNDS.md` opens with this file verbatim as of round 25
> (163 k characters: every round's detail, superseded tables, struck-through claims), so older
> pointers such as "CLAUDE.md round 14" or "Strategy item 6" resolve there.

## Read first

1. The newest `../handoffs/HANDOFF_YYYYMMDD.md`: live state, open items, running jobs. Where it
   disagrees with the repo docs, it wins.
2. `ROADMAP.md`: what the board says (Part 1), what is closed (Part 2), the plan (Part 3), the
   file map and method rules (Part 4).
3. `../sub_candidates/README.md`: staged and uploaded packages with their recorded predictions.
4. The live record, preferred over any table in prose: `Track_1/board_rounds.py` (every scored
   round), `Track_1/final_v9.py` (the weight fit), `python leaderboard/lb.py --html`.

## Current state — 2026-09-14

| | `final` | package |
|---|---|---|
| Track 1 best | **81.344644** | R25 xfam6fp16 (09-13) |
| Track 2 best | **81.164374** | T2R12 xfam6fp16 (09-13) |
| leader, read 09-14 | T1 81.85874 · T2 81.99648 | gaps 0.514 · 0.832 |

* **R25 / T2R12 (read 09-14): fp16 ran on the board**; the error subscores moved by ≤ 3e-4
  against R24, the fingerprint. Track 1 kept **37 %** of the local −21 % (5.570 → 5.076 board
  ms, +0.033). Track 2's archive also added the second ViT: its containment (+0.041) landed, but
  `time` 91.04 < 91.52, so fp16 did not pay for the ViT's FLOPs. Both estimates (81.356 /
  81.223) missed only in `time`.
* **Package of record, xfam6fp16:** 4 all-data `cmb3sim` FNOs (one fused invocation) + 2 ViTs (a
  second) · int8 `--linear tensor` · band `Track_1/bands/r24_xfam6.npz` · `ALPHA = 0.60` ·
  `AMP = "fp16"`, `SPEC_PATH = "bmm"`, `HOST_WRITE = "torch"` · 146 MB extracted.
* **Next slots (planned, not built):** Track 2 xfam5fp16 (R21's archive + fp16) against T2R12;
  Track 1 a baked TF32 constant or a CUDA graph on `predict` against R25.
* **Where the remaining score is:** the four containment levers (band, capacity, member count,
  family) are closed and precision is read. Left: TF32 (safe, not baked), a CUDA graph on the
  Track-1 `predict` path, and anything structurally new — the **backbone campaign**,
  `Track_1/plan/plan_20260914.md` (ledger `leaderboard/screen.jsonl`).

## Standing rules

* **The user uploads; agents never do.** One submission per track per day.
* **Predict before upload.** All five predicted subscores go into `../sub_candidates/README.md`
  (an estimate of record plus a bracket) before the user uploads; a prediction written afterwards
  is worthless. The `leaderboard/board.jsonl` record carries that `pred`, and is appended by the
  time the result is recorded.
* **All-data checkpoints supply weights, never evidence.** They train on every validation
  trajectory, and rounds 7, 8 and 10 were each misled by a number computed from one. Local
  evidence comes from the held-out twins.
* **Compare packages by board-subscore deltas, never by `final`**, and never resubmit for a
  better timing draw.
* **One controlled lever per slot.** A slot built so that only one subscore can move (R14's gate,
  R23's unfused package) buys a number no local work can. Main Development slots are for
  learning; the Decision phase selects the package.
* **Only baked constants ship.** `submission.py` constants override the `RPDE_*` environment
  switches, which steer measurement drivers only. Anything decided at load time (`AMP`) must
  leave a fingerprint in the board result.
* **Local timing:** never while any `rp_*` job is alive; only arms interleaved in one process
  count as evidence.
* **No generated trees, caches or bulk data under `$HOME`** (1 M-inode quota). Job scripts, logs
  and scratch go to `/fs/scratch/PAS1187/qwu/tmp/realpde/` or `/tmp`. Checkpoints in
  `Track_1/unet/` and archives in `../sub_candidates/` are the existing exceptions: a few large
  files each.
* **Wrap-up** = *Current state* here plus any changed fact, `ROUNDS.md`, `ROADMAP.md`, touched
  READMEs · board rows into `Track_1/board_rounds.py` and `final_v9.py:BOARD` (it self-checks the
  two lists and its residuals at import), `predict_board.py` anchors, `leaderboard/board.jsonl`,
  `lb.py --html` and a republish of the calibration log · uploads marked in `../sub_candidates/`
  (archive to `submitted/`, the README's §NOW section into its `HISTORY.md`, band promoted to
  `Track_1/bands/`) · a dated handoff · a commit (≤ 7 per day on `qwu/dev`).

## The problem

Both tracks: 20 frames of real PIV velocity on a 32×64 grid → the next 20 frames, channels
`[u, v, p]` with `p` zero-filled and unscored. Same hidden targets and same five subscores on both
boards. Simulation is training-side only.

| | Track 1 — Sim2Real | Track 2 — LTTTA |
|---|---|---|
| entry | `predict(input_array, metadata=None)` | `get_ttt_model(submission_dir, device)` → `reset_ttt_state()`, `ttt_step(input_norm, prev_target_norm)` |
| calls | once, all N windows batched | once per window, batch 1, time order; the previous window's target is revealed |
| units | raw physical: normalize and denormalize yourself | already normalized; the evaluator handles both ends |
| timed | all of `predict()` | all of `ttt_step`, plus `reset_ttt_state` charged to the next step; construction and loading are untimed |
| limits | extracted archive < 256 MB | < 256 MB, and 10 min wall clock for the whole run |

**The windows tile.** Stride 20 equals the horizon, so window *k*'s target is window *k+1*'s input,
bit for bit.
* Track 1 could copy answers out of its own argument (93.500 on the live board). We disclosed it
  and the organizers confirmed such scores are invalid (`Track_1/submit_trick/DISCLOSURE.md`).
  Never ship it.
* **The only clean input for window *k* is window *k*'s own input.** A statistic pooled across
  windows (a trajectory mean) contains targets and still passes a permutation-invariance check.
  Every package must be permutation-invariant (`validate_strict.py` asserts it). `predict()` gets
  no Re or AoA, only `metadata["sample_id"]`.
* **Track 2 is Track 1's model, streamed.** Anything computable from `prev_target_norm` is already
  in `input_norm`. Only the supervised pair and the realized error are new, and every adaptation
  built on them is measured and closed (`Track_2/README.md`). Ship frozen. The site's
  "N-step autoregressive" is teacher-forced, so nothing compounds.

## Data

Under `/users/PAS1187/qwu/RealPDE/` (not committed). Hugging Face mirror:
`AI4Science-WestlakeU/RealPDE-Competition-Data`. `RealPDEBench` is the separate full-resolution
benchmark, not the competition data.

| | trajectories | frames | datasets | dtype |
|---|---|---|---|---|
| `train_sim/train_sim/{Re}_{AoA}.h5` | 100: Re 3750…27975 × AoA 0/5/10/15/20 | 1000 | `u,v,p` (T,64,128), `x,y,t`, `re,aoa` | float32 |
| `train_real/train_real/{Re}_{AoA}.h5` | 82: 18 Re × 5 AoA, incomplete grid | 607 | `u,v` (T,64,128), `x,y,t`, `re,aoa` | float64 |

* **Windowing** (both tracks): subsample `[::2, ::2]` to 32×64; T_in = T_out = 20, stride 20,
  `(filename, time_id)` order, tail padded by repeating the last frame; tensors are
  `(N, 20, 32, 64, 3)`. Reference: the kit's `local_eval.py:build_stream`. `u` and `v` are
  top-level datasets (the kit README's `measured_data/u` nesting is wrong).
* **Normalizer** (GaussianNormalizer on train_real at 32×64; std 0 → 1):
  `mean_in = [0.154960856, -0.000513992854, 0.0]`, `std_in = [0.0968056545, 0.015960684, 1.0]`,
  `mean_tgt = [0.154962569, -0.000517793698, 0.0]`, `std_tgt = [0.0968104079, 0.0159636438, 1.0]`.
  Track 1 must apply it itself; skipping it is the classic way a correct model scores badly.
  `*_in` equals `_fno.MEAN/STD`. `*_tgt` differs in the sixth decimal, so Track 2's output needs
  the target affine to be exact (`Track_2/verify_space.py`).
* **Units.** Sim is non-dimensional (mean u ≈ 0.86–0.91). Real is in m/s with mean u ≈ c(AoA)·Re,
  c = [1.138, 1.149, 1.098, 1.002, 0.991]e-5 at AoA 0…20, so raw sim/real ratios run 3×–19×; never
  mix raw arrays. Sim normalizer (u, v at 32×64): `MEAN_SIM = [0.858663955, -0.004962022]`,
  `STD_SIM = [0.443825609, 0.214743194]`. `Track_1/unet/simreal.py` maps sim into real's
  measurement space (freestream-law speed, twin amplitude, twin FOV mask).
* **Sim** covers every real (Re, AoA) plus Re 15225, 27975 and real's holes. It is not a volume
  multiplier (4900 against 3444 windows), and clean CFD is *harder* to predict than smoothed PIV.
  It pays only as a small co-training regularizer.
* **The hidden set lies inside the training speed range.** R14 shipped a gate on out-of-range
  windows and it fired on none, so the hidden set's unseen Reynolds numbers are interior.
  `val_relo`, `val_rehi` and k-fold Re band 0 are retired as decision instruments.
* **Error follows AoA for `rel_l2`** (out-of-fold 0.052 / 0.053 / 0.086 / 0.106 / 0.121 at AoA
  0…20), **not for TKE** (a 1.10× span, non-monotone). Elements whose target is exactly 0
  (outside the FOV, inside the airfoil) are dropped from SPS.

## Scoring

The official program is `/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9/scoring.py`
(Track 2: `official/t2_v6/`). Our reconstruction,
`/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py`
(what `validate_strict.py` runs), agrees with it to 2.4e-6
(`Track_1/verify/diff_scoring_{official,real}.py`). Score locally before every submission.

* `rel_l2`, `tke`, `mvpe` → `100/(1 + 0.5·e)`, on `u, v` only. rel-L2 is per window. TKE is the
  rel-L2 of `0.5·(var_t u + var_t v)` over the 20 output frames, so it is phase-free. MVPE is the
  rel-L2 of the time-mean at probes x ∈ {13,21,29,37}, y ∈ {8,10,…,24}, per x-station, averaged.
* `time` → `100/(1 + sqrt(t/0.72896))`, where t is the mean board **seconds** per step
  (5.570 ms → 91.96); zero or non-finite ⇒ 0.
* `sps` (kit v9) → `100·w`. Per scored element (target inside `[lower, upper]` and non-zero):
  `(1−pm)·exp(−(upper−lower)/σ)`, `pm = e/(0.5+e)` per branch, branch weights DM 0.5 / TKE 0.3 /
  MVPE 0.2, σ = 0.0563870259. Bounds are optional but the default ±5 %·|pred| is poor, so always
  ship a band. **Malformed bounds (shape, non-finite, `lower > upper`) zero every subscore.**
* **SPS factorizes:** `sps = 100·mean_i(a_i)·containment`, `a_i = 0.5g(rel) + 0.3g(tke) +
  0.2g(mvpe)`, `g(m) = 0.5/(0.5+m)`. The exact optimal band is therefore a sort (`Track_1/sps9.py`),
  and a board round's `sps` follows from its other three subscores, `sps(B) ≈ sps(A)·a(B)/a(A)`,
  with containment as the residual. Board steps that moved containment: band +1.41 · capacity
  w24→w96 +0.99 / +0.96 · members 3→8 +0.67 · U-FNO in/out +0.46 / −0.44 · first ViT +0.18 / +0.16 ·
  second ViT +0.05 / +0.04. Everything else (sim, depth at equal FLOPs, auxiliary losses, latency
  rebuilds, 4→8 members with a ViT present) landed within ±0.05.
* **The rule combining the subscores into `final_score` is unpublished. It is a free linear fit**
  (no Σw = 1) over 32 rounds pooled across both tracks: **rel_l2 0.4775 · tke 0.1013 · mvpe
  0.0920 · time 0.0965 · sps 0.2501** (Σw 1.018, max residual 0.0025). Five single-subscore A/Bs
  read `w_time` directly at 0.0956–0.0962. `Track_1/final_v9.py` is the authority; work in deltas.
  Every weight table in `ROUNDS.md` before round 17 is the superseded constrained fit.
* **Exchange rate** with containment held fixed, per 10 % of error removed: rel_l2 **+0.346**, tke
  **+0.290**, mvpe **+0.100**, all three **+0.736** `final`.
* **Noise:** the four error subscores are deterministic for a given archive. Only `time` varies
  (same-shape drift 0.03–0.30 ms on Track 1, 0.4–0.7 ms on Track 2).
* **Both boards are one test set and one rule.** The same package lands within ~0.4 % on
  `rel_l2` and `sps` across tracks. Track 1's `mvpe` error is lower in every pair (by ~3–5 %),
  and since width 96 its `tke` error by ~1.5 %. A day's two slots form one controlled A/B, and a
  package that scored on one board is worth submitting on the other.

### Board latency — measured; move in deltas, never predict a level

* **Invocation:** +1.151 board ms on Track 1, +0.866 on Track 2 (R23 shipped R20's package
  unfused).
* **FLOPs:** ≈ 1.75 board ms per 4060-equivalent local ms on Track 1 (R16→R17 1.76, R21→R24
  1.74), 0.60 on Track 2 (one reading). A100 local ms × 1.850 = 4060-equivalent.
* **Precision** is charged less than FLOPs: fp16 kept 37 % of its local cut on Track 1, ≈ 0.65
  board ms per 4060-eq ms (R24→R25, `PREC_SLOPE_T1`); on Track 2 ~20 % (T2R12, confounded).
* At ~4.5 board ms on Track 1, 1 board ms ≈ −0.072 `final`, and +0.1 A100 ms/window ≈ −0.023.
  Track 2 carries a fixed ~+0.86 ms per-step handicap at one invocation.
* The eval machine drifts by date, so anchor on a recent round of the same shape. Code:
  `board_rounds.py` (`INV_MS_T1/T2`, `LOC_SLOPE_T1/T2`, `board_ms()`, `time_score()`,
  `invocation_cost()`) and `predict_board.py` (`LAT_INV_*`, `LAT_LOC_*`).

## What ships, and why

**Members.** Checkpoints live in `Track_1/unet/` (untracked). Read any recipe from
`torch.load(ckpt)["config"]`.
* **FNO `cmb3sim` ×4**, `fno_all_w96L8cmb3sim_s1..s4.pth` (jobs `r19e`/`r19f`): `--arch fno
  --width 96 --n-layers 8 --modes-y 8 --modes-x 12 --stride 5 --epochs 400 --batch 32 --lr 2e-4
  --dropout 0.45 --var-head --var-mlp --lift-mlp --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25
  --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 --sim <train_sim>
  --lam-sim 0.3 --sim-loss full --all-data`. `--lam-f 0.0 --lam-cov 2.0` replaces the MSE field
  term with a curvature-matched Welsch "coverage" loss that bends only the tail. Held-out twins
  drop `--all-data` and add `--sim-exclude-held`.
* **ViT ×2** (`ViT2d`, dim 384, 8 layers, 6 heads, dropout 0.3, no `--var-mlp`/`--lift-mlp`):
  `arch_vit_all_s1.pth` (plain MSE field loss, `--lam-grad 1.0 --lam-v 0.3`, no sim) and
  `arch_vit_all_cmb3sim_s3.pth` (the cmb3sim losses and sim).
* **Held-out twins**, which supply the band's error field: `fno_r18_cmb3sim{,_s2,_s3}.pth`,
  `arch_vit_d384.pth`, `arch_vit_cmb3sim.pth`.
* **`train2.py`'s argparse defaults are NOT the recipe:** `--alpha-train` 1.0 (trains through the
  composition and collapses the model), `--sim-loss shape` (the rejected arm), `--lam-f` 1.0,
  `--lam-v/--lam-t` 1.0/0.5, `--loss rel`, `--width` 64, `--n-layers` 4, modes (12,16),
  `--epochs` 150.

**Inference** (`Track_1/submit_fno/submission.py` + `_fno.py`; `Track_2/submit/_fno.py` is a
symlink to the same file):
1. Normalize → average the members' raw field and σ on the GPU → `compose`: rescale the
   fluctuation's per-pixel temporal std toward the σ head at **ALPHA 0.60** (0.55–0.70 is a flat
   plateau) → denormalize.
2. Band: `half = base · clip(σ/σ_ref, 0.3, 3.0)^p` with p 0.45. `base` is solved per element,
   box-smoothed (k 3) and fitted by `fit_band.py` on the AoA 15–20 subset at a 1.39× error stretch
   (the board's measured element tail is 1.36; refitting there buys 0.000).
3. `_fno.fuse()` folds same-architecture members into one invocation: `StackedFNO` (modes
   zero-padded; `w2` pads at the FRONT) and `StackedViT`. Exact to ~1e-8 with TF32 off.
4. Latency settings, each verified accuracy-neutral: `HOST_WRITE = "torch"` (threaded output
   write, bit-identical, −17 % on Track 1); `SPEC_PATH = "bmm"` (real 2×2-block spectral matmul,
   2e-6, −17 % of the Track-2 step); int8 packing (free for FNO and ViT); `AMP` (autocast checked
   against fp32 at import, falling back on an exception, non-finite output or a diff above 2e-2;
   `AMP_ACTIVE` records the outcome; fp16 cut compute −21 % / −46 % on R24's archive, and Track 1's board kept 37 % of it; bf16
   crashes in `torch.complex`). TF32 is off (`RPDE_TF32` opts in for drivers only). `_pick_batch`
   times batch sizes at import. Track 2 captures the whole device-side step in a CUDA graph at
   load, verified bit-exact against eager.

**Why this shape** (board evidence; detail in `ROADMAP.md` Part 1):
* Regularized capacity pays through the error tail, not the rms: dropout +0.200, then w24 → w96
  +0.166 while `rel_l2` got worse. **Attack the tail.** `rel_l2` alone would have called that a
  regression.
* Depth at equal FLOPs is a wash (4×L8 ≈ 8×L4), and the capacity ladder ends at w96 L8.
* A second family *inside* an FNO package pays through containment (first ViT +0.15 of accuracy,
  second +0.05), where U-FNO as a whole package missed by −0.45. A third ViT or AFNO does not pay
  for its FLOPs. Member count is closed (4 → 8 FNOs with a ViT present: −0.020). Uniform member
  averaging is optimal.
* Sim co-training is real but small (≈ +0.01–0.02 of accuracy); the `cmb3sim` losses on top of it
  added +0.031 / +0.049. Neither moved containment.
* Velocity-gradient matching (`--lam-grad`, both invariants of ∇u) is the only physics loss that
  works. In the FNO, the σ head must share the trunk and have an MLP readout (`--var-mlp`);
  capacity at the boundaries (`--lift-mlp`) helps, and capacity in the trunk (`--point-mlp`) hurts.

## Instruments — a sign, not a size

* **Every local number is a sign.** The proxy over-reads error cuts about 2× (R24: rel −0.75 %
  predicted, −0.30 % delivered). Board/proxy level ratios are ≈ rel 1.07, tke 1.14, mvpe 1.45, and
  the delta residual is ~4 % of the error rms. Across model families the proxy has been wrong in
  size and, for U-FNO, in sign.
* **Instrument defaults are not the operating point.** `--alpha` defaults to 0.75 in
  `compare_v9.py`, `ens_v9.py`, `fit_band.py` and `predict_board.py`, and to 0.90 in `kfold.py`;
  `fit_band.py --band-p` defaults to 0.5. Pass `--alpha 0.60` and `--band-p 0.45`.
* **Screen at the ensemble size that ships.** `Track_1/compare_v9.py` scores single checkpoints,
  each with its own band; `Track_1/ens_v9.py --aoa 15 20` scores member sets on the board-like
  proxy; add the full split. Error that an arm removes independently per member vanishes in the
  ensemble average (the oscillatory head: +3.84σ per model, −0.012 ensembled).
* **Confirm** with the interior Re-band k-fold: `Track_1/kfold.py --split re`, bands 1–3 pooled
  (~2150 windows; `--paired` bootstraps trajectories). It overstates `rel_l2` by ~1 % and gets the
  signs of tke and sps right.
* **Predict** with `Track_1/predict_board.py` (per-metric ratios, the 1.36 SPS tail, measured
  latency constants; anchors named `t2*` select Track 2's), anchored on a recent round of the same
  shape. Bracket `time` with the measured slopes, never a cross-day refit (both round-24 estimates
  fell below their floors that way). For local `sps`, transfer through the per-sample factor is
  the estimate and 1:1 transfer the optimistic end.
* **Control and replicate.** `train2.py --seed` bit-matches a control to its arm. Replicate in the
  first job; single runs at 2–3σ have regressed repeatedly. Read the training curve first: an arm
  that does not move its own training loss (e.g. `var`) cannot move the held-out metric.
* **Refit everything downstream** (band, ALPHA) before comparing a changed member set or scale
  parameter. **Derive held-out lists** (`re_split.excluded()`, `kfold.py --folds`): the grid is
  incomplete, so a typed `--exclude` leaks.
* **If an arm scores worse than the theoretical worst case for its own hypothesis, suspect the
  harness.** A representation change (normalizer, target space) is measured only once the scoring
  path implements it.

## Closed — do not re-propose without new evidence

Evidence and files: `ROADMAP.md` Part 2, the latest handoff, and `ROUNDS.md`.
* **Band**, eight times (exact per-element solver, trajectory CV, spread-adaptive, asymmetric
  −0.29, online calibration). Member weights (uniform is optimal). ALPHA (a plateau).
* **Architecture:** capacity past w96 L8 (L12, w128, modes (12,20)) and below it; ViT L12; member
  count; a third family (AFNO) or a third ViT; U-FNO as a package; CNO and Transolver (latency);
  DeepONet (rank-p decoder); GNNs; KANO; DINOv2 initialization (fits better, generalizes the same).
* **σ head:** readout width, depth and inputs; loss weights (`lam_v` is U-shaped at the recipe);
  a detached head; a separate σ network; post-hoc σ calibration (three nulls); FGN/CRPS
  generative σ.
* **Losses and regularizers:** the scorer's own functionals in the loss (`--lam-p`, `--lam-t` up,
  `--lam-tail`, `--w-mean`, spectrum, Laplacian); ∇·u → 0 and momentum; `--wd`; SAM;
  `--aug-noise`; `--aug-scale`; `--phases 4`; 800 epochs; `--chan-pow`; SWA; `--point-mlp`.
  `--mode-drop` and `--aug-mask` sit at the seed floor.
* **Representation:** `--norm self` (fixes the low-Re field, costs +21 % TKE, −0.50 ensembled);
  phase/advection re-registration; oscillatory and temporal-basis heads; diffusion
  (PDE-Refiner); analog forecasting; long context; denoise → predict → renoise.
* **Sim:** sim pretraining (+0.00); `lam_sim` dose; sim stride; per-pixel maps; `--sim-loss
  shape`; a real-data structured perturbation (`--aux-fstride`); a sim pressure head.
* **Track 2 adaptation:** residual EMA; online band calibration; gradient TTT (+0.02 of accuracy
  for −0.5 of latency); closed-form readout refit (≤ 1 %, 09-14); trajectory history (legitimate,
  worth nothing). Transductive pooled statistics are a leak.
* **Test-set range:** the gated mean blend (the hidden set is in range; `BLEND_BETA = 0.0`).

## Build and gate a package

```bash
# From the repo root. Builds and gates BOTH tracks from one member set; stages into ../sub_candidates/.
RPDE_A="<all-data members>" RPDE_M="<held-out twins>" RPDE_BAND=Track_1/bands/<band>.npz \
RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch \
RPDE_TAG=r26_<tag>_a060 bash Track_1/submit_fno/build_round13.sh 0.60
```

* Without `RPDE_BAND` the script refits the band (`fit_band.py --ckpt $A --err-ckpt $M --alpha
  ALPHA --band-p 0.45`). **Refit whenever the member set or ALPHA changes, and always take the
  error field from held-out twins.** A band fitted on memorized errors scored sps 36.94 (R6).
* **Pass every knob.** Unset ones fall back to older behaviour: `RPDE_A`/`RPDE_M` to the round-13
  width-24 members, `ALPHA` 0.90 with no argument, empty `SPEC_PATH`/`HOST_WRITE`, fp32 packing,
  `AMP = None`. The script `sed`s `ALPHA`, `FUSE`, `AMP` and `SPEC_PATH` into both tracks'
  `submission.py`, and `HOST_WRITE` and `BLEND_BETA` into Track 1's, and leaves them there; a job
  that sets `RPDE_SHIP_AMP` restores `AMP = None` itself. Never edit `_fno.py`, `submission.py` or
  the build scripts while a build runs.
* It refuses while other `rp_*` jobs run, because `validate_strict.py` times the package;
  `RPDE_FORCE=1` is for builds whose timing is not recorded.
* **Gates:** `Track_1/submit_fno/validate_strict.py` (runs inside the build: clean extract,
  extract dir off `sys.path`, eval-image packages blocked, contract, permutation invariance, the
  kit scorer, end-to-end timing; GPU) · `Track_2/validate_t2.py` (contract, isolation, no
  aliasing, statelessness, whole-run bound consistency) · `Track_1/verify/cpu_smoke_submission.py`
  (login node; `RPDE_ZIP=`) · `verify/pkg_identity.py` (whether two archives return the same
  arrays; give it the archive actually uploaded) · `verify/amp_path.py` (AMP runs, scores like
  fp32, interleaved timing) · `verify/fuse_vit.py`.
* **Provenance:** scored bands live in `Track_1/bands/` (with `MANIFEST.md`), unscored ones in
  `../sub_candidates/band_*.npz`, and every uploaded archive byte for byte in
  `../sub_candidates/submitted/` (winners open-source).

## Submission contracts

**Track 1:** `submission.py` at the zip root. `predict(input_array, metadata=None)` takes
`(N,20,32,64,3)` raw physical and returns a float array of the same shape (float32 or float64), or
`{"prediction", "lower", "upper"}`. Finite everywhere, `p` zero, bounds per element with the same
shape and `lower ≤ upper`.

**Track 2:** `get_ttt_model(submission_dir, device)` returns an object with `reset_ttt_state()` and
`ttt_step(input_norm, prev_target_norm) → (prediction, info)`. `info["adapt_loss"]` is required;
bounds go in `info["lower"]` and `info["upper"]`, both or neither, consistently for the whole run.
Adapt only on the cached previous input with the revealed previous target. **Return tensors the
caller can keep:** a reused staging buffer turns every stored prediction into the last one
(rel-L2 0.67) and passes any harness that copies out on the same line. Agent/LLM steps must be
bounded and use the organizer gateway only, and they are timed.

### Evaluation containers — nothing is installed at eval time

* Track 1, `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`: no `scipy`, `h5py`, `matplotlib` or
  `pandas` (kit `smoke_test_kit.py`); `einops` only from the kit's `_vendor/`; `requests` and
  `setuptools`/`pkg_resources` are present. Never import h5py from anything in the zip
  (`unet/model.py` imports it lazily).
* Track 2, `w3nhao/realpde-track2:v1` (PyTorch 2.2.2, CUDA 12.1, Python 3.10): `einops` present;
  runs in an isolated subprocess.
* The platform may load `submission.py` via `spec_from_file_location` with the extract directory
  off `sys.path`. Vendor any other pure-Python dependency.

## Engineering traps

* Never train through `compose` (`--alpha-train 0.0`); compose at inference only.
* Time `predict()` end to end, after warm-up, outside the loop. The forward pass is a small
  fraction of it.
* `torch.amax(x, dim=())` reduces over all dims (it gave a whole Linear one int8 scale). Score the
  real code path, never a re-implementation of it.
* Preprocessing that reads real-data statistics (domain maps, masks, per-condition normalizers)
  must be told the split. `--exclude` only filters the loss; round 14's sim map leaked 26
  held-out trajectories.
* A feature no local data exercises needs its own verifier (`verify/blend_path.py`): the strict
  gate passes such a package as if the feature were absent.
* Running jobs re-import the working tree per arm. Edit shared modules in a copy, install with an
  atomic `mv` after `ast.parse`, and keep the default path bit-identical. Build jobs that bake
  constants restore the tree themselves.
* Co-scheduled h5 readers need `HDF5_USE_FILE_LOCKING=FALSE` (set in `jobs/common.sh`). Slurm
  snapshots a batch script at submit time.
* A background waiter's sentinel must be the job script's last line, and the waiter must also exit
  when the job leaves the queue.

## Environment

* **OSC Ascend.** A100-PCIe-40G nodes via Slurm: `-A PAS1186 -M ascend -p nextgen
  --gpus-per-node=1`, `--chdir` the repo, `--output` on scratch, `--exclusive` only when timing
  (budget and concurrency limits are in the handoff). Login and OnDemand nodes have no GPU. The
  12-core cpuset binds before the GPU does: pin training and evaluation to disjoint cores, and do
  not judge liveness by `nvidia-smi`, which reads ~0 % during healthy training.
* Jobs of other projects share the allocation (`hd_*`, `p5_*`, `p6_*`, `tt26-*`), and
  `ondemand/*` is the user's desktop. Never touch them. Ours are `rp_*`.
* Repo `/users/PAS1187/qwu/workspace/realpde/main` · data and official kits
  `/users/PAS1187/qwu/RealPDE/` (`official/{t1_v9,t2_v6}/`, `baseline_checkpoints/`) · job scripts
  and logs `/fs/scratch/PAS1187/qwu/tmp/realpde/{jobs,logs}/` (`jobs/common.sh`).
* Python: `source .pyenv.sh`, or call `/users/PAS1187/qwu/.conda/envs/realpde/bin/python`.
  `environment.yml` pins the eval stack: Python 3.10, torch 2.2.2+cu121, **numpy 1.26.4** (torch
  2.2.2 is not numpy-2 compatible), **setuptools 69.5.1** (the CNO baseline imports
  `pkg_resources`). h5py, scipy, matplotlib and pandas are for local work only.

## Rules and dates

One shared Codabench account (at most 3 members); 1 submission per day per track, 100 per phase.
**Main Development ends 2026-09-27. The Decision phase (to 2026-10-25) selects the package.**
Results 2026-11-10; workshop 2026-12-06. Winners open-source two weeks before presenting. Prizes
per track: $6k / $3k / $1.5k.

## Repo map

```
CLAUDE.md  README.md  ROADMAP.md  ROUNDS.md  environment.yml  .pyenv.sh
Track_1/        instruments at the top level: board_rounds, final_v9, predict_board, compare_v9, ens_v9, kfold, sps9, ...
  unet/         model.py (data), models_alt.py (FNO2d, ViT2d, ...), train2.py, simreal.py, *.pth (untracked)
  submit_fno/   the package: submission.py, _fno.py, fit_band.py, build_zip.py, validate_strict.py, build_round13.sh
  verify/  bands/  report/  plan/          gates · scored bands · write-ups · plans and campaign ledgers
  submit/  submit_trick/  fno_deeponet/  deeponet/  diffusion/       historical
Track_2/        submit/, build_t2.py, validate_t2.py, stream_eval.py, verify_space.py, ttt_*.py (closed studies)
leaderboard/    board.jsonl, runs.jsonl, lb.py
outside the repo: ../sub_candidates/ (staging, predictions, submitted/) · ../handoffs/ · ../leaderboard.html
```
