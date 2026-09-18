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
3. `../sub_candidates/README.md`: staged and uploaded packages with their predictions.
4. The live record, above any table in prose: `Track_1/board_rounds.py` (every scored round),
   `final_v9.py` (the weight fit), `python leaderboard/lb.py --html`.

## Current state — 2026-09-18

| | `final` | package |
|---|---|---|
| Track 1 best | **81.401196** | **R28** P2-L8 on T1, F-FNO ×4 + ViT ×2 (09-18) |
| Track 2 best | **81.314785** | T2R13 P2-L8, the same members + the same ViT ×2 (09-16) |
| leader, read 09-17 | T1 81.9613 · T2 82.05275 | gaps 0.560 · 0.738 |
| **top-10 cut, 09-18** | **T1 81.624 (we are #21, +0.223 away) · T2 81.511 (#14, +0.197)** | the shortlist is the objective |

* **09-18: F-FNO took Track 1, and the ViT pair is worth keeping.** **R28** (R25's package with four
  fused F-FNO w40 L8 in place of the cmb3sim FNOs, T2R13's band): **+0.057, a NEW TRACK-1 BEST**, inside
  its bracket and 0.014 above the estimate. **The T2R13 − T2R12 board pair transferred at ≈ 0.85×** on
  every row (sps +0.272 = **containment +0.253** against T2's +0.302): **a BOARD read of a family
  transfers across tracks; a PROXY read does not** (R26). mvpe fell again (−0.142). **T2R15** (minus the
  ViT pair, 1 inv): **−0.050** — the pair is **+0.155 of accuracy+containment against −0.105 of `time`**
  and its containment −0.132 is **3× the −0.041 assumed**, so **it stays in every package**.
* **Two calibration corrections.** (a) **"Halve the local deltas" is for arms that REMOVE error**; a
  member REMOVAL is charged in full (T2R15's un-halved ratios were near-exact). (b) **`LOC_SLOPE_T2` is
  back to 0.60** (0.60 / 0.937 / 0.529 / ≈0.28, **bracket 0.3–0.9**) and T1 gains **`LOC_SLOPE_T1_XFAM`
  0.89** from R28. 09-18 was fast on both boards, so both are brackets.
* **THE RULES PIVOT (09-17):** slots are **learning** slots aimed at the **top-10 cut**, not the leader;
  every shipped checkpoint must be **re-creatable from the release** (`REPRODUCE.md`, `reproduce.sh`);
  and the private set is OOD in AoA and Re, so an **AoA-holdout screen** replaces R14's "hidden set in
  range" there. `plan/lit_20260917.md` §2, `plan_20260917.md` §6.
* **09-19, two proof slots, neither a score attempt** (`../sub_candidates/README.md` §NOW). T1 =
  `track1_r32_p2ffnoL8_tiled.zip`: R28's members, band and `submission.py` **byte-identical**, only the
  tiled `_fno.py` differs — it proves the code path the students ship on, errors must return identical,
  est. **81.40** (81.38–81.43). **The one permitted exception to "never resubmit for a timing draw"**: a
  new code path under identical weights, with a fingerprint. T2 = the **filled-band** package (the §4.9
  fix on T2R13's members) **if forge gates it by ~14:00 EDT, else HOLD**; `sps` up = the fill paid.
* **Packages of record — the same members on both boards.** 4 all-data F-FNO w40 L8 fused **dense** +
  2 ViTs · int8 `tensor` · band `t2r13_p2ffnoL8x4.npz` · ALPHA 0.60 · fp16, bmm (T1 adds HOST_WRITE
  torch) · 30 MB (R28) / 36 MB (T2R13).
* **Where the remaining score is.** The backbone campaign is **closed** (`plan/screen_20260914.md` 6z);
  09-17 closed the Track-1 graph and F-FNO depth, 09-18 the ViT-pair question (keep it). What is left is
  in `plan_20260917.md` §2 and is all **same-family**: **distilling the ensemble into one model** (+0.2 /
  +0.27 if the student loses < 0.05, pre-registered ship rule), the `StackedFFNO` batch-tiling fix, and
  the AoA-holdout screen for the private set.
* **THE BAND IS BLANKING-DEPENDENT — the largest private-set liability we know of**
  (`plan/lit_20260917.md` §4.9). `optimal_band` is fitted on **three recordings**, and at an element
  unscored in all of them `argmax` returns index 0 and **`base` collapses to ~0** (2,667 elements, 13.6×
  below the frame median). Bounds stay well-formed, so nothing is zeroed — the containment indicator just
  never fires there. **2,317 of them ARE scored somewhere in our own 82**, so a private recording that
  blanks less costs **−2.5 sps ≈ −0.62 `final`**. The mask-independent fill (`w = clip(n_ok/8,0,1)` toward
  a k=5 box mean of well-sampled neighbours) touches 4.4 % of elements, leaves the rest bit-identical,
  costs −0.0016 sps here and is worth ≈ +0.04 on the dev board.

## Standing rules

* **The user uploads; agents never do.** One submission per track per day.
* **The development board is a top-10 QUALIFIER, not the prize metric.** Organizers re-train each
  shortlisted method **from scratch** on their cluster (8× A800) and score a **private test set at novel
  AoA / Re**; those scores decide the top 3, an **irreproducible method is disqualified**, and external
  data, new simulations and outside pretrained weights are **prohibited**.
* **Predict before upload.** All five predicted subscores go into `../sub_candidates/README.md` (an
  estimate plus a bracket) **before** upload; one written afterwards is worthless. `board.jsonl` carries
  that `pred` in a PENDING row from upload day.
* **All-data checkpoints supply weights, never evidence** (rounds 7, 8 and 10 were each misled by
  one). Evidence comes from the held-out twins.
* **Compare packages by board-subscore deltas, never by `final`**; never resubmit for a timing draw.
* **One controlled lever per slot.** A slot where only one subscore can move (R14's gate, R23's unfused
  package) buys a number no local work can. **Never upload a package you expect to score worse just to
  fill a slot** — HOLD is a valid decision.
* **Only baked constants ship.** `submission.py` constants override the `RPDE_*` switches, which steer
  measurement drivers only. Anything decided at load time (`AMP`, `GRAPH`) needs a board fingerprint —
  and a fingerprint is a MAGNITUDE test, not a direction test (R27).
* **Local timing:** never while an `rp_*` job is alive; only interleaved arms count, and quote the
  kernel and package reads as a bracket.
* **No generated trees, caches or bulk data under `$HOME`** (1 M-inode quota). Jobs, logs and scratch
  go to `/fs/scratch/PAS1187/qwu/tmp/realpde/` or `/tmp`; `Track_1/unet/` and `../sub_candidates/` are
  the exceptions.
* **Wrap-up** = *Current state* here plus any changed fact, `ROUNDS.md`, `ROADMAP.md`, touched READMEs
  · board rows into `board_rounds.py` and `final_v9.py:BOARD` (self-checking at import),
  `predict_board.py` anchors, `board.jsonl`, `lb.py --html` · uploads marked in `../sub_candidates/`
  (archive → `submitted/`, §NOW → `HISTORY.md`, band → `Track_1/bands/`) · a dated handoff · a commit
  (≤ 7/day on `qwu/dev`).

## The problem

Both tracks: 20 frames of real PIV velocity on a 32×64 grid → the next 20, channels `[u, v, p]`
with `p` zero-filled and unscored. Same hidden targets and five subscores on both boards.
Simulation is training-side only.

**Track 1 (Sim2Real)** calls `predict()` once with all N windows batched, in **raw physical units** —
normalize and denormalize yourself — and times all of it. **Track 2 (LTTTA)** calls `ttt_step` once per
window at batch 1 in time order, **already normalized**, revealing the previous window's target; it times
every step plus `reset_ttt_state` (loading is untimed) and caps the run at 10 min. Both cap the extracted
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

Under `/users/PAS1187/qwu/RealPDE/` (not committed). Hugging Face mirror:
`AI4Science-WestlakeU/RealPDE-Competition-Data`. `RealPDEBench` is the separate full-resolution
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

The official program is `RealPDE/official/t1_v9/realpde_t1_starting_kit_v9/scoring.py` (Track 2:
`official/t2_v6/`). Our reconstruction, `RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py`
(what `validate_strict.py` runs), agrees with it to 2.4e-6
(`Track_1/verify/diff_scoring_{official,real}.py`). Score locally before every submission.

* `rel_l2`, `tke`, `mvpe` → `100/(1 + 0.5·e)`, on `u, v` only. rel-L2 is per window; TKE is the rel-L2
  of `0.5·(var_t u + var_t v)` over the 20 output frames, so phase-free; MVPE is the rel-L2 of the
  time-mean at probes x ∈ {13,21,29,37}, y ∈ {8,10,…,24}, per x-station, averaged.
* `time` → `100/(1 + sqrt(t/0.72896))`, t the mean board **seconds** per step (5.570 ms → 91.96);
  zero or non-finite ⇒ 0. **The private set is timed on 8× A800, so these constants need not carry.**
* `sps` (kit v9) → `100·w`. Per scored element (target inside `[lower, upper]` and non-zero):
  `(1−pm)·exp(−(upper−lower)/σ)`, `pm = e/(0.5+e)` per branch, branches DM 0.5 / TKE 0.3 / MVPE 0.2,
  σ = 0.0563870259. Always ship a band (the default ±5 %·|pred| is poor). **Malformed bounds (shape,
  non-finite, `lower > upper`) zero every subscore.**
* **SPS factorizes:** `sps = 100·mean_i(a_i)·containment`, `a_i = 0.5g(rel) + 0.3g(tke) + 0.2g(mvpe)`,
  `g(m) = 0.5/(0.5+m)`, so the optimal band is a sort (`Track_1/sps9.py`) and a round's `sps` follows
  from its other three subscores, with containment the residual. The steps that moved it are listed in
  `ROUNDS.md` (band, capacity, member count, family, and since 09-17 two negatives: L8→L4 −0.310 and
  dropping the ViT pair −0.132). **No local instrument reads containment.**
* **The rule combining the subscores into `final_score` is unpublished. It is a free linear fit**
  (no Σw = 1) over 32 rounds pooled across both tracks: **rel_l2 0.4775 · tke 0.1013 · mvpe
  0.0920 · time 0.0965 · sps 0.2501** (Σw 1.018, max residual 0.0025). `Track_1/final_v9.py` is
  the authority; work in deltas.
* **Exchange rate**, containment fixed, per 10 % of error removed: rel_l2 **+0.346**, tke **+0.290**,
  mvpe **+0.100**.
* **Noise:** the four error subscores are deterministic per archive; only `time` varies (same-shape
  drift 0.03–0.30 ms on T1, 0.4–0.7 ms on T2).
* **Both boards are one test set and one rule.** The same package lands within ~0.4 % on `rel_l2` and
  `sps` across tracks; T1's `mvpe` error is lower in every pair (~3–5 %) and, since w96, its `tke` by
  ~1.5 %. **A board-measured DELTA transfers across tracks at ≈ 0.85×** (R28 from the T2R13−T2R12 pair,
  every row the right sign) — a proxy-measured one does not.

### Board latency — measured; move in deltas, never predict a level

Constants and their derivations live in `board_rounds.py` (`INV_MS_T1/T2`, `LOC_SLOPE_*`, `PREC_SLOPE_T1`,
`board_ms`, `time_score`, `invocation_cost`) and `predict_board.py` (`LAT_*`). What they mean:

* **Invocation:** +1.151 board ms on T1, +0.866 on T2 (R23, unfused); untested since.
* **FLOPs:** **1.75** board ms per 4060-eq local ms on T1 *within* a kernel family, **0.89 across one**
  (`LOC_SLOPE_T1_XFAM`, R28 — bracket [0, 1.75]). Track 2 has four readings (0.60 / 0.937 / 0.529 / ≈0.28)
  and ships **0.60**; **bracket it with 0.3 and 0.9.** A100 local ms × 1.850 = 4060-equivalent.
* **Precision** is charged less: fp16 kept 37 % on T1, ≈ 0.65 board ms per 4060-eq ms (`PREC_SLOPE_T1`).
* **A local latency read transfers only for a FLOPs cut, within ~2×, and the two local instruments
  disagree with each other:** R28 vs R25 reads **+0.797 ms** at the kernel level and **+0.164** at the
  package level; the board gave **+0.404**, between them — quote both as the bracket. Swin's step measured
  0.63× and the board returned +1.17; R27's graph measured −0.2 % locally and cost +0.565 on the anchor's
  own kernels, so **host-side launch accounting does not transfer**.
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
  `--sim-loss shape`, `--lam-f` 1.0, `--lam-v/--lam-t` 1.0/0.5, `--loss rel`, `--width` 64, `--n-layers`
  4, modes (12,16), `--epochs` 150.

**Inference** (`Track_1/submit_fno/submission.py` + `_fno.py`; `Track_2/submit/_fno.py` is a
symlink to the same file):
1. Normalize → average the members' raw field and σ on the GPU → `compose`: rescale the
   fluctuation's per-pixel temporal std toward the σ head at **ALPHA 0.60** (0.55–0.70 is flat) →
   denormalize.
2. Band: `half = base · clip(σ/σ_ref, 0.3, 3.0)^p`, p 0.45. `base` is solved per element, box-smoothed
   (k 3) and fitted by `fit_band.py` on AoA 15–20 at a 1.39× error stretch (the board's measured tail
   is 1.36; refitting there buys 0.000).
3. `_fno.fuse()` folds same-architecture members into one invocation: `StackedFNO` (modes zero-padded,
   `w2` pads at the FRONT) and `StackedViT`, exact to ~1e-8 with TF32 off; `StackedSwin` and
   `StackedFFNO` (`FFNO_PATH` dense or fft) within 2e-5 in fp32.
4. Latency settings, each verified accuracy-neutral (sizes in `ROUNDS.md`): `HOST_WRITE = "torch"`
   (threaded output write, bit-identical); `SPEC_PATH = "bmm"` (real 2×2-block spectral matmul,
   2e-6); int8 packing (free for FNO and ViT); `AMP` (autocast checked against fp32 at import,
   falling back on an exception, non-finite output or a diff above 2e-2; `AMP_ACTIVE` records the
   outcome; bf16 crashes in `torch.complex`). TF32 is off (`RPDE_TF32` opts in for drivers only).
   `_pick_batch` times batch sizes at import. Track 2 captures the whole device-side step in a CUDA
   graph at load, verified bit-exact against eager; **on Track 1 the same graph is closed** — R27
   shipped it and paid +0.565 board ms.

**Why this shape** — the seven board findings behind every choice above (capacity pays through the
TAIL; depth at equal FLOPs is a wash; a second family pays through containment and a third does not;
member count is closed; sim and `cmb3sim` are real but small; `--lam-grad` is the only physics loss
that works; the σ head needs the shared trunk and `--var-mlp`) are stated with their sizes in
`ROUNDS.md`, "CLAUDE.md detail moved here", and in `ROADMAP.md` Part 1.

## Instruments — a sign, not a size

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
  instead**: it is the only OOD instrument we have.
* **Predict** with `Track_1/predict_board.py` (per-metric ratios, the 1.36 SPS tail, measured latency
  constants; `t2*` anchors select Track 2's), anchored on a recent round of the same shape. Bracket
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

Evidence and files: `ROADMAP.md` Part 2, the latest handoff, and `ROUNDS.md`.
* **Band width and shape**, eight times (exact per-element solver, trajectory CV, spread-adaptive,
  asymmetric, online calibration) — but **not** the unscored-element fill, which is a different failure
  (above). Member weights (uniform is optimal). ALPHA (a plateau).
* **Architecture:** capacity past w96 L8 (L12, w128, modes (12,20)) and below it; ViT L12; member
  count; a third family (AFNO) or a third ViT; U-FNO; CNO and Transolver (latency); DeepONet; GNNs;
  KANO; DINOv2 init (no gain — and now **prohibited**).
* **σ head:** readout width/depth/inputs; loss weights (`lam_v` U-shaped); a detached head; a separate
  σ network; post-hoc calibration; FGN/CRPS generative σ. **σ-amplitude modelling is closed on PHYSICS**
  too: the TKE target is a 20-sample variance, our 0.538 already beats the hindsight long-run variance
  (0.570), and a 20 % cut would beat a 40-frame hindsight variance (0.456). No TKE-targeted arm.
* **Losses and regularizers:** the scorer's functionals in the loss (`--lam-p`, `--lam-t` up,
  `--lam-tail`, `--w-mean`, spectrum, Laplacian); ∇·u → 0 and momentum; `--wd`; SAM; `--aug-noise`;
  `--aug-scale`; `--phases 4`; 800 epochs; `--chan-pow`; SWA; `--point-mlp`. `--mode-drop`/`--aug-mask`
  sit at the seed floor.
* **Representation:** `--norm self` (−0.50 ensembled); phase/advection re-registration; oscillatory and
  temporal-basis heads; diffusion; analog forecasting; long context; denoise→predict→renoise.
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

* Without `RPDE_BAND` the script refits the band (`fit_band.py --alpha ALPHA --band-p 0.45`). **Refit
  whenever the member set or ALPHA changes; always take the error field from held-out twins.** A band
  fitted on memorized errors scored sps 36.94 (R6).
* **Pass every knob** — unset ones fall back to older behaviour (round-13 width-24 members, ALPHA 0.90,
  fp32 packing, `AMP = None`). The script `sed`s constants into both `submission.py` files and leaves
  them there. **Never edit `_fno.py`, `submission.py` or the build scripts while a build runs**, and
  commit before building: a build job restores the tree with `git checkout`.
* It refuses while other `rp_*` jobs run (`validate_strict.py` times the package); `RPDE_FORCE=1` is for
  builds whose timing is not recorded. It also **does not propagate its gates' exit codes** — read the
  `SUBMITTABLE` line per track.
* **Gates, verifiers and provenance: `ROADMAP.md` Part 4.** The two that always run are
  `submit_fno/validate_strict.py` (Track 1, GPU) and `Track_2/validate_t2.py`; `verify/pkg_identity.py`
  is what proves a rebuild returns an earlier archive's arrays bit for bit.

## Submission contracts

**Track 1:** `submission.py` at the zip root. `predict(input_array, metadata=None)` takes
`(N,20,32,64,3)` raw physical and returns a float array of the same shape, or
`{"prediction", "lower", "upper"}`. Finite everywhere, `p` zero, bounds per element, `lower ≤ upper`.

**Track 2:** `get_ttt_model(submission_dir, device)` → an object with `reset_ttt_state()` and
`ttt_step(input_norm, prev_target_norm) → (prediction, info)`. `info["adapt_loss"]` is required; bounds
go in `info["lower"]`/`info["upper"]`, both or neither, for the whole run. Adapt only on the cached
previous input with the revealed previous target. **Return tensors the caller can keep:** a reused
staging buffer turns every stored prediction into the last one (rel-L2 0.67) and passes any harness that
copies out on the same line.

### Evaluation containers — nothing is installed at eval time

Track 1 `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`, Track 2 `w3nhao/realpde-track2:v1`. **Neither
has `scipy`, `h5py`, `matplotlib` or `pandas`**; `einops` is vendored on T1, present on T2. Never import
h5py from anything in the zip, and vendor every other pure-Python dependency: the platform may load
`submission.py` with the extract dir off `sys.path`. Detail in `ROADMAP.md` Part 4.

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
* Other projects share the allocation (`hd_*`, `p5_*`, `p6_*`, `tt26-*`); `ondemand/*` is the user's
  desktop. Never touch them. Ours are `rp_*`.
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
CLAUDE.md  README.md  ROADMAP.md  ROUNDS.md  environment.yml  .pyenv.sh
Track_1/   top level: board_rounds, final_v9, predict_board, compare_v9, ens_v9, kfold, sps9, ...
  unet/       model.py (data), models_alt.py (FNO2d, ViT2d, ...), train2.py, simreal.py, *.pth (untracked)
  submit_fno/ submission.py, _fno.py, fit_band.py, build_zip.py, validate_strict.py, build_round13.sh
  verify/ bands/ report/ plan/   gates · scored bands · write-ups · plans and ledgers
  submit/ submit_trick/ fno_deeponet/ deeponet/ diffusion/   historical
Track_2/   submit/, build_t2.py, validate_t2.py, stream_eval.py, verify_space.py, ttt_*.py (closed)
leaderboard/  board.jsonl, runs.jsonl, screen.jsonl, lb.py
outside the repo: ../sub_candidates/ · ../handoffs/ · ../leaderboard.html
```
