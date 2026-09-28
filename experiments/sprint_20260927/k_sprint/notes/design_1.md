# Design K1 — the `time` subscore (latency), both tracks

Author: design agent K1, 2026-09-27 16:35–17:00 EDT. Clone `K/wt_1` (K = `/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927`),
HEAD 55f0e88 + uncommitted edits (diff saved: `K/k1/k1_wt1.diff`, 4 files, all knobs default OFF = shipped path).
Harness + raw reads: `K/k1/*.py`, `K/logs/k1_t1lat*_*.jsonl`, `K/logs/k1_t2lat_*.jsonl`, `K/logs/rp_k271_*.log`.
All timings: torch 2.2.2+cu121 (`realpde` env), A100-PCIE-40GB (Ascend, = the A800's class) and H100 (Cardinal),
1 GPU / 12 cores, NOT exclusive nodes (shared with other users' jobs), evaluator-like processes (fresh process, import
untimed, first call = "cold", then 6–8 warm repeats per N), variants interleaved, 4–6 rounds.
Baseline = the F-sprint's `final0927/sub_candidates/track{1,2}_x27_a080.zip`.

## 0. What is timed (facts)
* The ingestion programs are NOT in the kits (only `scoring.py`, which reads `mean_t_neural_s` from predictions.npz). The kit's
  T2 `local_eval.py` times `ttt_step()` with `perf_counter` and charges `reset_ttt_state()` to the next step. Our mirror of
  the T1 ingestion (`Track_1/submit/validate_submission.py:58`) says outputs go through `np.asarray(.., dtype=float32)`.
* So everything below changes OUR code inside `predict()` / `ttt_step()` only. Nothing touches the timing environment.

## 1. Where the local time goes (A100, N = 56, fp16, ms/window)
| stage | ms/window | share |
|---|---|---|
| fused F-FNO x4 forward | 0.84 | 56 % |
| fused ViT x2 forward | 0.27 | 18 % |
| 3 output arrays device->host, pageable | 0.13 (pinned: 0.06) | 9 % |
| input upload f32 / f64 | 0.045 / 0.084 | 3–6 % |
| rest (alloc, compose, band, sigma) | ~0.2 | 13 % |
| **package, warm** | **1.51** (N 14: 1.53, N 28: 1.49, N 99: **1.79**) | |
| **package, cold first call, x27 as built** | **1.87–2.49** | |

F-FNO stack profile (5 forwards, 230 ms device): `native_layer_norm` **28 %** (1.44 ms per call, 9 calls/forward),
`copy_` (permute copies + dtype casts) 31 %, elementwise 23 %, the actual matmuls (baddbmm+bmm+mm) only 20 %.

## 2. Ranked tricks (measured)

### T1-a. `WARM_TAILS = True` — FLAG: the F-sprint's x27 T1 zips are built with it OFF
* BUILD_RUNBOOK §5 "chosen: off". Progress report 6.5 lists it as part of the shipped T1 path.
* Measured cold first call (ms/window, A100): x27 2.49 / 1.96 / 2.00 / 1.87 at N 56 / 99 / 14 / 28; with WARM_TAILS
  1.53 / 1.79 / 1.70 / 1.57. H100: 2.15 -> 1.10 at N 56. **-0.2 … -1.0 ms/window if the evaluator's process calls predict()
  once** (progress report: new subprocess per call). Bit-identical (md5).
* Every T1 zip built today must pass `RPDE_SHIP_WARM_TAILS=True`. Cost: import grows by ~20–30 s (untimed).
* Forcing the chunk WITHOUT the import-time probe/warm-up is a trap: cold N 14 went to 3.1 (A100) / 7.7 (H100) ms/window.

### T1-b. `PIN_OUT` — page-locked output arena (new, bit-identical)
* The three (N,20,32,64,3) outputs are consecutive slices of a page-locked arena allocated at import (1024 windows =
  1.5 GB); every call gets its own slice; falls back to the shipped allocation when used up / unavailable / warming.
* Removes fresh allocation + first-touch page faults + pageable staging of 1.47 MB per window from the timed call.
* Measured warm (ms/window): A100 1.51 -> 1.32 (N 56), 1.53 -> 1.39 (N 14), 1.79 -> 1.60 (N 99) = **-0.19 (-12.5 %)**;
  H100 1.06 -> 0.83 (-22 %). Cold with WARM_TAILS: 1.53 -> 1.31 (N 56). md5 identical at N 14/28/56/99, f32 and f64 input.
* ROADMAP 1.23 "rejected: pinned/async D2H, contiguous output buffer" was measured 09-05 on the host-write path before
  GPU_BAND/DIRECT_PRED existed; on today's path it pays.

### T1-c. `BATCH_CAP = 56` — chunk cap after `_pick_batch` (new, bit-identical)
* On the A100 `_pick_batch` picks 112 (7 of 8 processes), so every N <= 99 runs as ONE chunk; the shipped path is slower per
  window in one chunk of 99 than in 56 + 43: warm 1.60 -> 1.38 with PIN (**-0.22 ms/window at N 99**), N <= 56 unchanged.
  H100: no difference (0.88 vs 0.83). md5 identical at N 99.
* Applies to calls with N > 56 only (the board fingerprint read N in 14–99; share of such windows unknown, ~half).

### T1-d. `LN_MANUAL` — LayerNorm from `var_mean` + elementwise kernels (new, NOT bit-identical)
* F-FNO stack 0.834 -> 0.750 ms/window (-10 %); package **-0.09 ms/window** (1.309 -> 1.216 at N 56, 1.579 -> 1.489 at N 99).
* Prediction differs 6.6e-4 relative (max abs 1.7e-3 physical), lower/upper max abs 7e-4; rel_l2 and containment on 99
  windows unchanged to 5 digits (0.05841 / 0.68614, all-data package = not evidence, equality only).
* Track 1 only (adds kernels at batch 1): `_fno.LN_MANUAL` is set from T1's submission.py; T2 leaves it False.

### T2-a. `ARENA_STEPS = 1024` — page-locked per-step output arena (new, bit-identical)
* `_fetch` today: 3 pinned staging copies + sync + **3 fresh clones per step**; the evaluator keeps every prediction, so each
  clone is a new mapping with first-touch faults. Arena: each step's pred/lower/upper go straight into their own slot.
* Measured 336-step stream, 4 interleaved rounds, A100, graph CAPTURED: median **3.05 -> 2.82 ms/step (-0.23, -7.5 %)**,
  mean 3.19 -> 2.97. md5 over all 336 x 3 outputs identical. Returns numpy views (NUMPY_OUT path unchanged).
* Risk: 1.5 GB page-locked host memory at load; allocation failure falls back to the shipped path (try/except).

### Combined T1 stack (WARM + PIN 1024 + CAP 56 [+ LN]) — job rp_k271_t1lat3_a, see §6 for the final table
Expected from the parts, A100 warm: N 56 1.51 -> ~1.22, N 99 1.79 -> ~1.29 (-19 % … -28 %).

## 3. Predicted board delta (honest)
Board T1 5.70 ms/window vs ~1.5 local: a ~4 ms intercept none of this touches. Local deltas are host-memory-traffic and
kernel-time cuts, not FLOPs; CLAUDE.md: a local read transfers within ~2x for FLOPs cuts and "host-side launch accounting does
not transfer". Assumed transfer slope 1.0, bracket [0, 1.75].
| package | local delta (A100) | board ms (slope 1.0) [0, 1.75] | delta final |
|---|---|---|---|
| T1: PIN + CAP (+LN), vs a WARM_TAILS build | -0.19 … -0.50 ms/window | -0.3 [0, -0.9] | **+0.02 [0, +0.06]** |
| T1: same, vs x27 as built (no WARM_TAILS), if calls are cold | -0.5 … -1.2 | -0.8 [0, -2.0] | +0.05 [0, +0.13] |
| T2: ARENA | -0.23 ms/step | -0.23 [0, -0.4] | **+0.016 [0, +0.03]** |
Same-package board drift is 0.04 (T1) / 0.055 (T2) final, i.e. these gains are inside one timing draw. They are free
(bit-identical) add-ons to whatever package ships; they do not close the 0.2–0.26 gap by themselves.

## 4. Legality (brief §2)
No data, no weights, no cross-window statistic; outputs per window unchanged (bit-identical, LN 6.6e-4) and independent of
batch composition (md5 equal across chunkings). Nothing alters the platform's measurement: no keep-alive, no background
thread; the arenas are ordinary buffers allocated at load. Each call/step owns distinct storage (no overwrite of returned
arrays) — `validate_t2.py` asserts that for T2.

## 5. Executor task specs
Both executors work in `K/wt_1` (patches applied, `git diff` = `K/k1/k1_wt1.diff`). New build knobs in
`Track_1/submit_fno/build_round13.sh`: `RPDE_SHIP_PIN_OUT=<windows>`, `RPDE_SHIP_BATCH_CAP=<n>`, `RPDE_SHIP_LN_MANUAL=True`,
`RPDE_SHIP_ARENA_STEPS=<steps>` (plus the existing `RPDE_SHIP_WARM_TAILS=True`). One build per clone at a time: E2 must
`cp -r K/wt_1 K/wt_1b` first and build there.

**E1 — build + gate the latency package on the baseline members (both tracks).**
1. Copy `final0927/jobs/rp_f27x_build.sh` to `K/jobs/rp_k271_build.sh`; set `W=K/wt_1`, `RPDE_SUB=K/sub_candidates`,
   symlinks per BUILD_RUNBOOK §0.4 (already present in `K/wt_1/Track_1/`), band `band_r38_a080_c0_fill.npz` (`RPDE_BAND`).
2. Knobs = BUILD_RUNBOOK §1 common + T1 27lz + T2 asnp, plus
   `RPDE_SHIP_WARM_TAILS=True RPDE_SHIP_PIN_OUT=1024 RPDE_SHIP_BATCH_CAP=56 RPDE_SHIP_ARENA_STEPS=1024`, TAG `k27lat_a080`;
   second build adds `RPDE_SHIP_LN_MANUAL=True`, TAG `k27latln_a080`; third = TAG `k27lat_d1c_a080` with `RPDE_SHIP_D1C_N0=30.0`.
3. Gates: `validate_strict.py` (T1) SUBMITTABLE, `validate_t2.py --adaptive --extract /tmp/<unique>` SUBMITTABLE,
   `stream_eval.py --causal-check`. Check the `[rpde-t1] load:` / `[rpde-t2] load:` lines.
4. Equivalence against `track{1,2}_x27_a080.zip`: run `K/k1/k1_child.py` (T1) and `K/k1/k1_t2child.py` (T2) on both zips
   extracted to `$TMPDIR`; md5 must be EQUAL for k27lat (T1 at N 14/28/56/99, T2 over 336 steps); for k27latln report
   rel-diff (expect 6.6e-4) and the gate's scores vs the non-LN build.
5. Paired timing, template `K/jobs/rp_k271_t1lat3.sh` / `rp_k271_t2lat.sh`, >= 4 interleaved rounds, A100 (Ascend nextgen;
   `--exclusive` if it starts promptly). Report ms/window cold and warm per N, ms/step median and mean.
6. `K/notes/result_11.md`: zip paths + sha256, gate lines, equivalence, timing table, predicted `time` subscore bracket.

**E2 — port the same knobs onto whichever member set the commander picks** (K2/K3 or F-sprint winner): same build with that
`RPDE_A` / `RPDE_M` / band; if the winner adds a new stack, re-run `K/k1/k1_prof.py` on it (top device-time ops) and
`K/k1/k1_ln.py` (LN share). Report `K/notes/result_12.md`.

## 6. Final combined read
Job rp_k271_t1lat3_a (7566915), A100, torch 2.2.2, evaluator-like processes, interleaved, ms/window
(`K/logs/k1_t1lat3_ascend_7566915.jsonl`; first 2–3 rounds at the time of writing, the job runs 5):

| variant | warm N 14 | N 28 | N 56 | N 99 | cold first call N 99 | cold N 56 | bits vs x27 |
|---|---|---|---|---|---|---|---|
| x27 as built (WARM_TAILS off) | 1.557 | 1.506 | 1.520 | 1.788 | **2.356** | 1.676 | — |
| ship = x27 + WARM_TAILS | 1.567 | 1.492 | 1.527 | 1.789 | 1.779 | 1.535 | identical |
| **k1b** = WARM + PIN 1024 + CAP 56 | 1.376 | 1.327 | 1.316 | 1.365 | **1.377** | 1.326 | **identical (md5)** |
| **k1** = k1b + LN_MANUAL | 1.270 | 1.240 | 1.223 | 1.275 | **1.287** | 1.238 | 6.6e-4 relative |

k1b vs ship: -0.19 … -0.42 ms/window (-12 % … -24 %); k1 vs ship: -0.28 … -0.51 (-19 % … -29 %);
k1 vs x27 as built, cold: -0.44 … -1.07 ms/window.

## 7. Ruled out / not pursued
* **Member groups on parallel threads + CUDA streams (T1)**: warm 1.49 vs 1.51 (N 56), 1.82 vs 1.79 (N 99): no gain — the GPU,
  not the launch thread, is the limit at batch >= 14. Bit-identical, dropped.
* `FFNO_TILE`: 16 (shipped) 0.849, 28 0.833, 32 0.834, 8 0.888, 0 1.072, 4 0.960 ms/window (tile 4 not bit-identical): nothing to gain.
* F-FNO `fft` path at batch 56: 1.122 vs dense 0.844; fp32 + TF32 without autocast: 0.994. Shipped dense fp16 stays.
* Input f64 upload costs 0.084 vs 0.045 ms/window (f32) — 0.04 at stake, host-side cast not tried.
* **fp16 OUTPUT arrays** (halves the bytes handed back): not tried. Accuracy cost is nil (rounding 6e-5 on u ~ 0.15, order of
  lower <= pred <= upper preserved), but the unseen ingestion casts to float32 (`np.asarray(dtype=float32)`), and if that
  cast sits inside the timed region it ADDS ~1 ms/window. Sign unknown without the ingestion source: a lottery ticket, not a plan.
* FLOPs cuts without retraining (mode truncation, dropping the padding, ViT token/layer pruning): not screened — each changes
  the prediction and needs a held-out read; the forward is 74 % of local time but local time is ~26 % of board time.
* T2 kernel-count reduction / LN_MANUAL on T2: batch-1 graph is launch-bound; LN_MANUAL adds kernels there. Not pursued.
