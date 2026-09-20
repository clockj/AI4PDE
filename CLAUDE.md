# AI4PDE — NeurIPS 2026 RealPDE Competition

Our solution code for RealPDE: forecasting real PIV flow past a NACA4418 airfoil, with CFD as a
training-side asset. [Site](https://realpdecompetition.github.io/) ·
[Track 1 Sim2Real](https://www.codabench.org/competitions/17363/) ·
[Track 2 LTTTA](https://www.codabench.org/competitions/17385/) · remote `github.com/clockj/AI4PDE`,
working branch `qwu/dev`.

> **Loaded into every session. Keep it under ~30 k characters and to what is true now.** After a board
> result: update *Current state*, edit in place any fact it changed, append the write-up to
> [`ROUNDS.md`](ROUNDS.md), put new rules in `ROADMAP.md` Part 1. Cut detail goes to `ROUNDS.md` (old
> pointers such as "CLAUDE.md round 14" resolve there) or `ROADMAP.md`.

## Read first

1. The newest `../handoffs/HANDOFF_YYYYMMDD.md`: live state, open items, running jobs. Where it
   disagrees with the repo docs, it wins.
2. `ROADMAP.md`: what the board says (Part 1), what is closed (Part 2), the plan (Part 3), the file
   map and method rules (Part 4).
3. `EVALUATION.md`: how local evaluation works and how the board corrects it — scorer, splits,
   instrument catalogue, every calibration constant with its rounds, and the two checklists.
4. `../sub_candidates/README.md`: staged and uploaded packages with their predictions.
5. The live record, above any table in prose: `Track_1/board_rounds.py` (every scored round),
   `final_v9.py` (the weight fit), `python leaderboard/lb.py --html`.

## Current state — 2026-09-19

| | `final` | package |
|---|---|---|
| Track 1 best | **81.401196** | **R28** F-FNO L8 ×4 + ViT ×2 (09-18) |
| Track 2 best **by score** | **81.323990** | T2R16 (09-19) — **a time draw; the method of record is T2R13's** (81.314785) |
| leader, read 09-17 | T1 81.9613 · T2 82.05275 | |
| **top-10 cut, read 09-18** | **T1 81.624 (we are #21, +0.223) · T2 81.511 (#14, +0.197)** | the shortlist is the objective |

* **09-19, two null levers, both fingerprints exact** (detail: `ROUNDS.md`, `ROADMAP.md` Part 2).
  **R29** (R28's package on the batch-tiled `_fno.py`): all four error subscores **bit-identical to
  R28**, `time` +0.016 board ms — inside drift. **The tiled path is proven exact on the eval GPU and
  is the base of every later package; the tiling is closed as a lever and kept as code.** **T2R16**
  (T2R13's package with the `--fill-unscored` band, prediction array bit-identical): **`sps` −0.0066,
  so the fill bought nothing here** — the hidden set's blanking matches the fit split's three
  recordings. Its +0.009 net is a **−0.177 ms time draw**, not a method improvement. Dev-board cost
  **−0.0016 `final`**; memo §4.9's −0.62 exposure on a *differently* blanked private recording is
  untouched, so **the fill stays in the Decision-phase method as insurance with a measured price**.
* **NO LEVER ≥ +0.05 REMAINS ON THE DEV BOARD.** 09-18 closed the last big one: distillation fails at
  every configuration (the teacher's mean transfers, its **variance does not**, −0.5 at K1 against the
  82.3198 ceiling), ALPHA is closed, a third family is +0.001 for −0.21, and the ViT pair beats three
  FNO members. **The shortlist needs +0.22 on Track 1 and +0.20 on Track 2, the cut is drifting up,
  and there is no identified source.** Slots are not spent on timing draws.
* **09-20: HOLD on both tracks** (`../sub_candidates/README.md` §NOW). `track1_r33_p2ffnoL8_fill.zip`
  is staged but **is not a candidate** — T2R16 says the fill reads ≈ −0.002 here; it is retained for
  the Decision-phase method.
* **Two calibration facts from 09-18.** (a) **A BOARD read of a family transfers across tracks at
  ≈ 0.85×; a PROXY read does not** (R28, from the T2R13−T2R12 pair). (b) **"Halve the local deltas"
  is for arms that REMOVE error** — a member REMOVAL is charged in full (T2R15).
* **THE RULES PIVOT (09-17):** slots are **learning** slots aimed at the **top-10 cut**; every shipped
  checkpoint must be **re-creatable from the release** (`REPRODUCE.md`, `reproduce.sh`); the private
  set is OOD in AoA and Re, so an **AoA-holdout screen** replaces R14's "hidden set in range" there
  (`plan/lit_20260917.md` §2, `plan_20260917.md` §6).
* **Packages of record — the same members on both boards.** 4 all-data F-FNO w40 L8 fused **dense** +
  2 ViTs · int8 `tensor` · band `t2r13_p2ffnoL8x4.npz` (T2's Decision-phase method takes
  `t2r16_p2ffnoL8_fill.npz`) · ALPHA 0.60 · fp16, bmm, tiled `_fno.py` (T1 adds HOST_WRITE torch).
* **Where the remaining score is.** Everything package-level is closed (`ROADMAP.md` Part 2); what is
  left is **generalization, not score** — the AoA-holdout screen for the private set, and
  `reproduce.sh` verified end to end. See §Open questions in the newest handoff.

## Standing rules

* **The user uploads; agents never do.** One submission per track per day.
* **The development board is a top-10 QUALIFIER, not the prize metric.** Organizers re-train each
  shortlisted method **from scratch** on their cluster (8× A800) and score a **private test set at
  novel AoA / Re**; those scores decide the top 3, an **irreproducible method is disqualified**, and
  external data, new simulations and outside pretrained weights are **prohibited**.
* **Predict before upload.** All five predicted subscores go into `../sub_candidates/README.md` (an
  estimate plus a bracket) **before** upload; one written afterwards is worthless. `board.jsonl`
  carries that `pred` in a PENDING row from upload day.
* **All-data checkpoints supply weights, never evidence** (rounds 7, 8 and 10 were each misled by
  one). Evidence comes from the held-out twins.
* **Compare packages by board-subscore deltas, never by `final`**; never resubmit for a timing draw.
* **One controlled lever per slot.** A slot where only one subscore can move (R14's gate, R23's
  unfused package) buys a number no local work can. **Never upload a package you expect to score
  worse just to fill a slot** — HOLD is a valid decision.
* **Only baked constants ship.** `submission.py` constants override the `RPDE_*` switches, which steer
  measurement drivers only. Anything decided at load time (`AMP`, `GRAPH`) needs a board fingerprint —
  and a fingerprint is a MAGNITUDE test, not a direction test (R27).
* **Local timing:** never while an `rp_*` job is alive; only interleaved arms count, and quote both
  the kernel and the package read as a bracket.
* **No generated trees, caches or bulk data under `$HOME`** (1 M-inode quota). Jobs, logs and scratch
  go to `/fs/scratch/PAS1187/qwu/tmp/realpde/` or `/tmp`; `Track_1/unet/` and `../sub_candidates/` are
  the exceptions.
* **Wrap-up** — the numbered checklist is `EVALUATION.md` §6: *Current state* here plus any changed
  fact, `ROUNDS.md`, `ROADMAP.md`, touched READMEs · board rows into `board_rounds.py` and
  `final_v9.py:BOARD`, `predict_board.py` anchors, `board.jsonl`, `lb.py --html` · uploads marked in
  `../sub_candidates/` · a dated handoff · a commit (≤ 7/day).

## The problem

Both tracks: 20 frames of real PIV velocity on a 32×64 grid → the next 20, channels `[u, v, p]`
with `p` zero-filled and unscored. Same hidden targets and five subscores on both boards.
Simulation is training-side only.

**Track 1 (Sim2Real)** calls `predict()` once with all N windows batched, in **raw physical units** —
normalize and denormalize yourself — and times all of it. **Track 2 (LTTTA)** calls `ttt_step` once per
window at batch 1 in time order, **already normalized**, revealing the previous window's target; it
times every step plus `reset_ttt_state` (loading is untimed) and caps the run at 10 min. Both cap the
archive at 256 MB. Entry points: *Submission contracts*.

**The windows tile:** stride 20 equals the horizon, so window *k*'s target is window *k+1*'s input.
* Track 1 could copy answers out of its own argument (93.500 live). Disclosed; the organizers
  confirmed such scores are invalid (`Track_1/submit_trick/DISCLOSURE.md`). Never ship it.
* **The only clean input for window *k* is window *k*'s own input.** A statistic pooled across
  windows (a trajectory mean) contains targets and still passes a permutation-invariance check,
  which `validate_strict.py` asserts. `predict()` gets no Re or AoA, only `metadata["sample_id"]`.
* **Track 2 is Track 1's model, streamed.** Anything computable from `prev_target_norm` is already in
  `input_norm`; only the supervised pair and the realized error are new, and every adaptation built on
  them is measured and closed (`Track_2/README.md`). **Ship frozen.** The site's "N-step
  autoregressive" is teacher-forced, so nothing compounds.

## Data

Under `/users/PAS1187/qwu/RealPDE/` (not committed). HF mirror
`AI4Science-WestlakeU/RealPDE-Competition-Data`; `RealPDEBench` is the separate full-resolution
benchmark, not the competition data.

`train_sim/train_sim/{Re}_{AoA}.h5`: 100 trajectories (Re 3750…27975 × AoA 0/5/10/15/20), 1000 frames,
`u,v,p` (T,64,128) + `x,y,t,re,aoa`, float32. `train_real/train_real/{Re}_{AoA}.h5`: 82 (18 Re × 5 AoA,
incomplete grid), 607 frames, `u,v` only, float64.

* **Windowing** (both tracks): subsample `[::2, ::2]` to 32×64; T_in = T_out = 20, stride 20,
  `(filename, time_id)` order, tail padded by repeating the last frame; tensors `(N, 20, 32, 64, 3)`.
  Reference: the kit's `local_eval.py:build_stream`. `u`/`v` are top-level datasets (the kit
  README's `measured_data/u` nesting is wrong).
* **Normalizer** (GaussianNormalizer on train_real at 32×64; std 0 → 1):
  `mean_in = [0.154960856, -0.000513992854, 0.0]`, `std_in = [0.0968056545, 0.015960684, 1.0]`,
  `mean_tgt = [0.154962569, -0.000517793698, 0.0]`, `std_tgt = [0.0968104079, 0.0159636438, 1.0]`.
  Track 1 applies it itself (skipping it is the classic way a correct model scores badly); `*_in` is
  `_fno.MEAN/STD`, and `*_tgt` differs in the sixth decimal, so Track 2's output affine must be exact
  (`Track_2/verify_space.py`).
* **Units.** Sim is non-dimensional (mean u ≈ 0.86–0.91); real is m/s with mean u ≈ c(AoA)·Re,
  c = [1.138, 1.149, 1.098, 1.002, 0.991]e-5 at AoA 0…20, so raw ratios run 3×–19× — never mix raw
  arrays. Sim normalizer (u, v at 32×64): `MEAN_SIM = [0.858663955, -0.004962022]`, `STD_SIM =
  [0.443825609, 0.214743194]`. `unet/simreal.py` maps sim into real's measurement space.
* **Sim** covers every real (Re, AoA) plus Re 15225, 27975 and real's holes; not a volume multiplier
  (4900 vs 3444 windows), clean CFD is *harder* than smoothed PIV, and it pays only as a small
  co-training regularizer.
* **The PUBLIC hidden set lies inside the training speed range** (R14's gate fired on no window), so
  `val_relo`, `val_rehi` and k-fold Re band 0 are retired. **The private set is not** — see the pivot.
* **Error follows AoA for `rel_l2`** (out-of-fold 0.052 / 0.053 / 0.086 / 0.106 / 0.121 at AoA 0…20),
  **not for TKE** (1.10× span, non-monotone). Elements whose target is exactly 0 are dropped from SPS.

## Scoring

The official program is `RealPDE/official/t1_v9/.../scoring.py` (Track 2: `official/t2_v6/`). Our
reconstruction, `RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py` (what
`validate_strict.py` and `sps9.py` run), agrees with it to 2.4e-6
(`Track_1/verify/diff_scoring_{official,real}.py`). Score locally before every submission.

* `rel_l2`, `tke`, `mvpe` → `100/(1 + 0.5·e)`, on `u, v` only. rel-L2 is per window; TKE is the rel-L2
  of `0.5·(var_t u + var_t v)` over the 20 output frames, so phase-free; MVPE is the rel-L2 of the
  time-mean at probes x ∈ {13,21,29,37}, y ∈ {8,10,…,24}, per x-station, averaged.
* `time` → `100/(1 + sqrt(t/0.72896))`, t the mean board **seconds** per step (5.570 ms → 91.96);
  0 or non-finite ⇒ 0. **The private set is timed on 8× A800, so these constants need not carry.**
* `sps` (kit v9) → `100·w`. Per scored element (target inside `[lower, upper]` and non-zero):
  `(1−pm)·exp(−(upper−lower)/σ)`, `pm = e/(0.5+e)` per branch, branches DM 0.5 / TKE 0.3 / MVPE 0.2,
  σ = 0.0563870259. Always ship a band (the default ±5 %·|pred| is poor). **Malformed bounds (shape,
  non-finite, `lower > upper`) zero every subscore.**
* **SPS factorizes:** `sps = 100·mean_i(a_i)·containment`, `a_i = 0.5g(rel) + 0.3g(tke) + 0.2g(mvpe)`,
  `g(m) = 0.5/(0.5+m)`, so the optimal band is a sort (`Track_1/sps9.py`) and a round's `sps` follows
  from its other three subscores, with containment the residual (two recent negatives: L8→L4 −0.310,
  dropping the ViT pair −0.132). **No local instrument reads containment.**
* **The rule combining the subscores into `final_score` is unpublished. It is a free linear fit**
  (no Σw = 1) over **40** rounds pooled across both tracks: **rel_l2 0.4778 · tke 0.1010 · mvpe
  0.0921 · time 0.0965 · sps 0.2498** (Σw 1.0173, max residual 0.0032). `Track_1/final_v9.py` is
  the authority; work in deltas.
* **Exchange rate**, containment fixed, per 10 % of error removed: rel_l2 **+0.346**, tke **+0.290**,
  mvpe **+0.100**.
* **Noise:** the four error subscores are deterministic per archive; only `time` varies (same-shape
  drift 0.03–0.30 ms on T1, 0.4–0.7 ms on T2).
* **Both boards are one test set and one rule.** The same package lands within ~0.4 % on `rel_l2` and
  `sps` across tracks; T1's `mvpe` error is lower in every pair (~3–5 %) and, since w96, its `tke` by
  ~1.5 %. **A board-measured DELTA transfers across tracks at ≈ 0.85×** (R28) — a proxy one does not.

### Board latency — measured; move in deltas, never predict a level

Constants in `board_rounds.py` (`INV_MS_T1/T2`, `LOC_SLOPE_*`, `PREC_SLOPE_T1`, `board_ms`,
`time_score`, `invocation_cost`) and `predict_board.py` (`LAT_*`); derivations in `EVALUATION.md` §5.

* **Invocation:** +1.151 board ms on T1, +0.866 on T2 (R23, unfused); untested since.
* **FLOPs:** **1.75** board ms per 4060-eq local ms on T1 *within* a kernel family, **0.89 across one**
  (`LOC_SLOPE_T1_XFAM`, R28 — bracket [0, 1.75]). Track 2 has four readings (0.60 / 0.937 / 0.529 / ≈0.28)
  and ships **0.60**; **bracket it with 0.3 and 0.9.** A100 local ms × 1.850 = 4060-equivalent.
* **Precision** is charged less: fp16 kept 37 % on T1, ≈ 0.65 board ms per 4060-eq ms (`PREC_SLOPE_T1`).
* **A local read transfers only for a FLOPs cut, within ~2×, and the two local instruments disagree:**
  R28 vs R25 read **+0.797 ms** (kernel) and **+0.164** (package); the board gave **+0.404** — quote
  both as the bracket. Swin measured 0.63× and the board returned +1.17; R27's graph measured −0.2 %
  locally and cost +0.565 on the anchor's own kernels: **host-side launch accounting does not transfer**.
* At ~4.5 board ms on T1, 1 board ms ≈ −0.072 `final`; T2 carries a fixed ~+0.86 ms per-step handicap at
  one invocation. **The machine drifts by date — anchor on a recent round of the same shape.**

## What ships, and why

**Members.** Checkpoints in `Track_1/unet/` (untracked); read any recipe from
`torch.load(ckpt)["config"]`. **Each must be re-creatable from the release** (rules pivot).
* **The `cmb3sim` loss recipe**, shared by both families: `--stride 5 --epochs 400 --batch 32 --lr 2e-4
  --var-head --var-mlp --lift-mlp --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0
  --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 --alpha-train 0.0 --sim <train_sim> --lam-sim 0.3
  --sim-loss full --all-data`; `--lam-f 0.0 --lam-cov 2.0` swaps the MSE field term for a
  curvature-matched Welsch "coverage" loss that bends only the tail. Twins drop `--all-data`, add
  `--sim-exclude-held`. **FNO ×4** `fno_all_w96L8cmb3sim_s1..s4.pth` adds `--arch fno --width 96
  --n-layers 8 --modes-y 8 --modes-x 12 --dropout 0.45` (jobs `r19e`/`r19f`).
* **ViT ×2** (`ViT2d`, dim 384, 8 layers, 6 heads, dropout 0.3, no `--var-mlp`/`--lift-mlp`):
  `arch_vit_all_s1.pth` (plain MSE field, `--lam-grad 1.0 --lam-v 0.3`, no sim) and
  `arch_vit_all_cmb3sim_s3.pth` (cmb3sim losses and sim).
* **The shipped members are F-FNO** (`ffno_all_w40l8_s1..s4.pth`, w40 L8 modes 16×32, screen recipe +
  `--all-data`) on BOTH tracks. **Held-out twins** (the band's error field): `p2_ffno_l8_s1..s3.pth`,
  `fno_r18_cmb3sim{,_s2,_s3}.pth` for FNO work, `arch_vit_d384.pth`, `arch_vit_cmb3sim.pth`.
* **`train2.py`'s argparse defaults are NOT the recipe:** `--alpha-train` 1.0 (collapses the model),
  `--sim-loss shape`, `--lam-f` 1.0, `--lam-v/--lam-t` 1.0/0.5, `--loss rel`, `--width` 64,
  `--n-layers` 4, modes (12,16), `--epochs` 150.

**Inference** (`Track_1/submit_fno/submission.py` + `_fno.py`; `Track_2/submit/_fno.py` symlinks it):
1. Normalize → average the members' raw field and σ on the GPU → `compose`: rescale the fluctuation's
   per-pixel temporal std toward the σ head at **ALPHA 0.60** (0.55–0.70 is flat) → denormalize.
2. Band: `half = base · clip(σ/σ_ref, 0.3, 3.0)^p`, p 0.45. `base` is solved per element, box-smoothed
   (k 3) and fitted by `fit_band.py` on AoA 15–20 at a 1.39× stretch (the board's measured tail is
   1.36; refitting there buys 0.000).
3. `_fno.fuse()` folds same-architecture members into one invocation: `StackedFNO` (modes zero-padded,
   `w2` pads at the FRONT) and `StackedViT`, exact to ~1e-8 with TF32 off; `StackedSwin` and
   `StackedFFNO` (`FFNO_PATH` dense or fft) within 2e-5 in fp32.
4. Latency settings, each verified accuracy-neutral (sizes in `ROUNDS.md`): `HOST_WRITE = "torch"`
   (threaded output write, bit-identical); `SPEC_PATH = "bmm"` (real 2×2-block spectral matmul, 2e-6);
   int8 packing (free for FNO and ViT); `AMP` (autocast checked against fp32 at import, falling back on
   an exception, non-finite output or a diff above 2e-2; `AMP_ACTIVE` records the outcome; bf16 crashes
   in `torch.complex`). TF32 is off (`RPDE_TF32` opts in for drivers only). `_pick_batch` times batch
   sizes at import. Track 2 captures the whole device-side step in a CUDA graph at load, bit-exact
   against eager; **on Track 1 the same graph is closed** — R27 shipped it and paid +0.565 board ms.

**Why this shape** — the seven board findings behind every choice above (capacity pays through the
TAIL; depth at equal FLOPs is a wash; a second family pays through containment and a third does not;
member count is closed; sim and `cmb3sim` are real but small; `--lam-grad` is the only physics loss
that works; the σ head needs the shared trunk and `--var-mlp`) are stated with their sizes in
`ROUNDS.md` and `ROADMAP.md` Part 1.

## Instruments — a sign, not a size

**Catalogue and calibration constants: `EVALUATION.md` §4–§5.** The rules:

* **Every local number is a sign.** The proxy over-reads error CUTS ~2×; **a member REMOVAL is charged
  in full** (T2R15). Board/proxy ratios ≈ rel 1.07, tke 1.14, mvpe 1.45. **Across FAMILIES the proxy
  predicts neither size nor sign** (R16, R26); a *board* read of a family does, at 0.85× (R28).
* **Instrument defaults are not the operating point:** `--alpha` is 0.75 in `compare_v9`, `ens_v9`,
  `fit_band`, `predict_board` and 0.90 in `kfold`; `fit_band --band-p` is 0.5. **Pass `--alpha 0.60
  --band-p 0.45`.**
* **Screen at the ensemble size that ships** (`ens_v9.py --aoa 15 20` for member sets, `compare_v9.py`
  for single checkpoints, each with its own band; add the full split). Error an arm removes per member
  vanishes in the ensemble average.
* **Confirm** with the interior Re-band k-fold (`kfold.py --split re`, bands 1–3, `--paired`) —
  overstates `rel_l2` ~1 %, signs of tke/sps right. **For the PRIVATE set use the AoA-holdout screen
  instead** (`aoa_holdout_eval.py`): it is the only OOD instrument we have.
* **Predict** with `Track_1/predict_board.py`, anchored on a recent round of the same shape. Bracket
  `time` with the measured slopes. For `sps`, the per-sample factor is the estimate and 1:1 the
  optimistic end — and containment is board-only, so never bank it.
* **Control and replicate.** `train2.py --seed` bit-matches a control to its arm; replicate in the first
  job (single runs at 2–3σ have regressed). An arm that does not move its own training loss cannot move
  the held-out metric.
* **Refit everything downstream** (band, ALPHA) before comparing a changed member set. **Derive
  held-out lists** (`re_split.excluded()`, `kfold.py --folds`) — the grid is incomplete, so a typed
  `--exclude` leaks.
* **If an arm scores worse than the worst case for its own hypothesis, suspect the harness.**

## Closed — do not re-propose without new evidence

Evidence and files: `ROADMAP.md` Part 2, the latest handoff, `ROUNDS.md`, and the per-script
verdicts in `Track_1/closed/README.md` and `Track_2/closed/README.md`.
* **Band width and shape**, eight times (exact per-element solver, trajectory CV, spread-adaptive,
  asymmetric, online calibration) — but **not** the unscored-element fill, which is a different failure
  (above). Member weights (uniform is optimal). ALPHA (a plateau).
* **Architecture:** capacity past w96 L8 (L12, w128, modes (12,20)) and below it; ViT L12; member
  count; a third family (AFNO) or a third ViT; U-FNO; CNO, Transolver (latency); DeepONet; GNNs; KANO;
  DINOv2 init (no gain — and now **prohibited**).
* **σ head:** readout width/depth/inputs; loss weights (`lam_v` U-shaped); a detached head; a separate
  σ network; post-hoc calibration; FGN/CRPS generative σ. **σ-amplitude modelling is closed on PHYSICS**
  too: the TKE target is a 20-sample variance, our 0.538 already beats the hindsight long-run variance
  (0.570), and a 20 % cut would beat a 40-frame hindsight variance (0.456). No TKE-targeted arm.
* **Losses and regularizers:** the scorer's functionals in the loss (`--lam-p`, `--lam-t` up,
  `--lam-tail`, `--w-mean`, spectrum, Laplacian); ∇·u → 0 and momentum; `--wd`; SAM; `--aug-noise`;
  `--aug-scale`; `--phases 4`; 800 epochs; `--chan-pow`; SWA; `--point-mlp`. `--mode-drop` and
  `--aug-mask` sit at the seed floor.
* **Representation:** `--norm self` (−0.50 ensembled); phase/advection re-registration; oscillatory
  and temporal-basis heads; diffusion; analog forecasting; long context; denoise→predict→renoise.
* **Sim:** pretraining (+0.00); `lam_sim` dose; stride; per-pixel maps; `--sim-loss shape`;
  `--aux-fstride`; a pressure head. **New simulations are prohibited**; the release's sim is fine.
* **Track 2 adaptation:** residual EMA; online band calibration; gradient TTT; closed-form readout
  refit; trajectory history (worth nothing). Transductive pooled statistics are a leak.
* **Test-set range:** the gated mean blend (`BLEND_BETA = 0`) — **for the PUBLIC set only**; it wins out
  of distribution, so the private set re-opens it on the AoA-holdout reads.
* **Latency (board-measured, 09-17):** the **Track-1 CUDA graph** (R27, +0.565 board ms) and
  **F-FNO depth L4 on Track 2** (T2R14: errors a wash, containment −0.310).
* **Backbones (09-14 screen; `ROADMAP.md` Part 2):** ConvNeXt U-Net (w40, w72), PDE-Arena U-Net and
  SimVP; Swin capacity (w96); sim on Swin; Re/AoA oracle conditioning.
* **Dropping the ViT pair** (T2R15: −0.155 accuracy+containment against +0.105 of `time`).

## Build and gate a package

```bash
# Repo root. Builds and gates BOTH tracks from one member set; stages into ../sub_candidates/.
RPDE_A="<all-data members>" RPDE_M="<held-out twins>" RPDE_BAND=Track_1/bands/<band>.npz \
RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch \
RPDE_TAG=<tag>_a060 bash Track_1/submit_fno/build_round13.sh 0.60
```

* Without `RPDE_BAND` it refits the band (`fit_band.py --alpha ALPHA --band-p 0.45`). **Refit whenever
  the member set or ALPHA changes; always take the error field from held-out twins** — a band fitted on
  memorized errors scored sps 36.94 (R6).
* **Pass every knob**: unset ones fall back to older behaviour (round-13 width-24 members, ALPHA 0.90,
  fp32 packing, `AMP = None`). It `sed`s constants into both `submission.py` files and leaves them
  there, so **commit before building** (a build job restores the tree with `git checkout`) and **never
  edit `_fno.py`, `submission.py` or the build scripts while a build runs**.
* It refuses while other `rp_*` jobs run (`validate_strict.py` times the package; `RPDE_FORCE=1` skips
  that when timing is not recorded) and **does not propagate its gates' exit codes** — read the
  `SUBMITTABLE` line per track.
* **Gates, verifiers, provenance and the full checklist: `EVALUATION.md` §4/§6, `ROADMAP.md` Part 4.**
  Always run: `submit_fno/validate_strict.py` (T1, GPU) and `Track_2/validate_t2.py`.

## Submission contracts

**Track 1:** `submission.py` at the zip root. `predict(input_array, metadata=None)` takes
`(N,20,32,64,3)` raw physical and returns a float array of the same shape, or
`{"prediction", "lower", "upper"}`. Finite everywhere, `p` zero, bounds per element, `lower ≤ upper`.

**Track 2:** `get_ttt_model(submission_dir, device)` → an object with `reset_ttt_state()` and
`ttt_step(input_norm, prev_target_norm) → (prediction, info)`. `info["adapt_loss"]` is required; bounds
go in `info["lower"]`/`info["upper"]`, both or neither, for the whole run. Adapt only on the cached
previous input with the revealed previous target. **Return tensors the caller can keep:** a reused
staging buffer turns every stored prediction into the last one (rel-L2 0.67) and passes any harness
copying out on the same line.

### Evaluation containers — nothing is installed at eval time

Track 1 `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`, Track 2 `w3nhao/realpde-track2:v1`. **Neither
has `scipy`, `h5py`, `matplotlib` or `pandas`**; `einops` is vendored on T1, present on T2. Never
import h5py from anything in the zip, and vendor every other pure-Python dependency: the platform may
load `submission.py` with the extract dir off `sys.path`. Detail in `ROADMAP.md` Part 4.

## Engineering traps

* Never train through `compose` (`--alpha-train 0.0`); compose at inference only.
* Time `predict()` end to end, after warm-up, outside the loop; the forward pass is a small part of it,
  and the kernel and package timers disagree by 5× (see *Board latency*).
* `torch.amax(x, dim=())` reduces over all dims (it gave a whole Linear one int8 scale). Score the real
  code path, never a re-implementation of it.
* Preprocessing that reads real-data statistics (domain maps, masks, per-condition normalizers) must be
  told the split: `--exclude` only filters the loss, and round 14's sim map leaked 26 trajectories.
* A feature no local data exercises needs its own verifier (`verify/blend_path.py`): the strict
  gate passes such a package as if the feature were absent.
* Running jobs re-import the working tree per arm. Edit shared modules in a copy, install with an
  atomic `mv` after `ast.parse`, keep the default path bit-identical.
* Co-scheduled h5 readers need `HDF5_USE_FILE_LOCKING=FALSE` (`jobs/common.sh`); Slurm snapshots a
  batch script at submit time.
* A waiter's sentinel must be the job script's last line, and it must exit when the job leaves the queue.

## Environment

* **OSC Ascend.** A100-PCIe-40G via Slurm: `-A PAS1186 -M ascend -p nextgen --gpus-per-node=1`,
  `--chdir` the repo, `--output` on scratch, `--exclusive` only when timing (limits in the handoff).
  Login and OnDemand nodes have no GPU. The 12-core cpuset binds before the GPU: pin training and
  evaluation to disjoint cores, and do not judge liveness by `nvidia-smi` (~0 % when healthy).
  `debug-nextgen` (same A100, ≤ 1 h) is fast when nextgen jams, but needs the user's OK.
* Other projects share the allocation (`hd_*`, `p5_*`, `p6_*`, `tt26-*`, `m2_*`); `ondemand/*` is the
  user's desktop. Never touch them. Ours are `rp_*` on Ascend and `pz_*` on Pitzer.
* Repo `~/workspace/realpde/main` · data and kits `~/RealPDE/` (`official/{t1_v9,t2_v6}/`,
  `baseline_checkpoints/`) · jobs and logs `/fs/scratch/PAS1187/qwu/tmp/realpde/{jobs,logs}/`.
* Python: `source .pyenv.sh`, or `~/.conda/envs/realpde/bin/python`. `environment.yml` pins the eval
  stack: Python 3.10, torch 2.2.2+cu121, **numpy 1.26.4** (torch 2.2.2 is not numpy-2 compatible),
  **setuptools 69.5.1**. h5py, scipy, matplotlib and pandas are for local work only.

## Rules and dates

One shared Codabench account (≤ 3 members); 1 submission per track per **UTC** day (it rolls over
at 20:00 EDT), 100 per phase; Failed submissions are not counted.
**Main Development ends 2026-09-27; the Decision phase runs to 2026-10-25.**
Results 2026-11-10; workshop 2026-12-06. Winners open-source two weeks before presenting.
Prizes/track: $6k / $3k / $1.5k.

## Repo map

```
CLAUDE.md  EVALUATION.md  README.md  ROADMAP.md  ROUNDS.md  environment.yml  .pyenv.sh
Track_1/   top level: board_rounds, final_v9, predict_board, compare_v9, ens_v9, eval_v9, kfold,
            aoa_holdout_eval, sps9, screen_*, re_split, make_val, proxy_calib, ...  (roles: README.md)
  unet/       model.py (data), models_alt.py (FNO2d, ViT2d, ...), train2.py, simreal.py, *.pth (untracked)
  submit_fno/ submission.py, _fno.py, fit_band.py, build_zip.py, validate_strict.py, build_round13.sh
  verify/ bands/ report/ plan/   gates · scored bands · write-ups · plans and ledgers
  closed/     51 one-off experiments, question closed; README.md has the verdict for each
  submit/ submit_trick/ fno_deeponet/ deeponet/ diffusion/   historical
Track_2/   submit/, build_t2.py, validate_t2.py, stream_eval.py, verify_space.py, t2_common.py,
            step_breakdown.py, t2_lat17.py, t1_vs_t2_score.py, closed/ (ttt_*, the pre-graph probes)
leaderboard/  board.jsonl, runs.jsonl, screen.jsonl, lb.py
outside the repo: ../sub_candidates/ · ../handoffs/ · ../leaderboard.html
```
**A script the docs cite that is no longer at a track's top level is in that track's `closed/`.**
