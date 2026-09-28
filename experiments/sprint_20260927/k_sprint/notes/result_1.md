# Result E1 — K1 latency knobs built + gated (both tracks), and the integration clone with JOINT_GAIN

Executor E1, 2026-09-27, written 17:10–17:20 EDT. K = `/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927`.
All GPU work on Ascend A100-PCIE-40GB, torch 2.2.2+cu121 (`realpde` env). Harness: `K/e1/*.py`, raw reads `K/e1/*.jsonl`,
logs `K/logs/rp_k271_{build,bint,eqt1,eqt2,time,twin}_*.log`, `K/logs/build_<TAG>.log`.

## 0. Trees
* `K/wt_1`: HEAD **7fc7317** "K1 latency knobs" (local commit on top of 55f0e88; = `K/k1/k1_wt1.diff`). Never pushed.
* `K/wt_int`: clone of wt_1 + `K/pkg/joint_gain.patch` (md5 f4a722a5..., applied `git apply --3way`, **no conflicts**),
  HEAD **e6480f5** "K2 JOINT_GAIN knob". Untracked inputs symlinked (val_{seen,unseen}, 11 ckpts). Both trees clean after the builds.
* Band reused everywhere: `final0927/pkg/x_work/band_r38_a080_c0_fill.npz` (`$M0/Track_1/bands/band_r38_a080_c0_fill.npz` does
  not exist). Band/weights files in the zips are byte-identical to the x27 baseline zips (diff -r shows only code lines).

## 1. Packages (all in `K/sub_candidates/`)
Common knobs = BUILD_RUNBOOK §1 common + T1 27lz + T2 asnp, alpha 0.80, `RPDE_FORCE=1`, members 4x `ffno_all_w40l8_sim_s1..s4`
+ `arch_vit_all_s1` + `arch_vit_all_cmb3sim_s3`, twins `p2_ffno_l8_sim_s1..s3 arch_vit_d384 arch_vit_cmb3sim`.
LAT = `RPDE_SHIP_WARM_TAILS=True RPDE_SHIP_PIN_OUT=1024 RPDE_SHIP_BATCH_CAP=56 RPDE_SHIP_ARENA_STEPS=1024`.

| TAG | extra | T1 zip sha256 | T2 zip sha256 |
|---|---|---|---|
| **k27lat_a080** | LAT | b3141dfeeca05e612228ecc9afae40f62e7c45933d153d4ed759af6f109c0ff6 | 48b94ff4cdf3cff3b428deaf6a1cb85751297b38522fb2efd3605b1b7c0c8573 |
| k27latln_a080 | LAT + LN_MANUAL (T1) | 4ec55e8d8bb4715674f718e3acbc668b99b5a392401e0be27698923e352c5fa4 | fdfb6f00aafd5f8e81752157428099ea6c3b629681282afa3f63e522e79a1120 (= k27lat T2 functionally) |
| **k27lat_d1c_a080** | LAT + D1C_N0 30.0 | 4a2706cda592fcc6bae0ba02de0f1e1478858e2957000c01c0f835a8c37e559d (= k27lat T1 functionally) | 99d9e0c06dbff21a48c65cae15777674252da5e3cf8786d9ff25e40d9de92329 |
| k27twin_a080 / k27twinln_a080 | members = the 5 twins (held-out LN read only, NOT for upload) | 2c6ddba4... / 4bf87be2... | — |
| k27int_off_a080 (wt_int) | NO new knob (x27 knob set) — equivalence proof only | 719e3dcb... | a99919c5... |
| k27int_lat_a080 (wt_int) | LAT — equivalence proof only | bb6f336b... | 644f1214... |
| **k27all_a080** (wt_int) | LAT + JOINT_GAIN=True | 918b803f33f225359bbb4963e9598c866706c8170040355898d58bd18acf2b0b | 30472a589aff81c21352b2d5833d42791d40544360933a3b5c54dbf4c4b4c0c3 |
| **k27all_d1c_a080** (wt_int) | LAT + JOINT_GAIN + D1C_N0 30.0 | 7a5ec8e6bad2e541206ce447ac9136c786c26cfd5c23e60a08a4261ce5bff8d4 | e8f0d81096bf92c86343eb7676a87ba73bbadfb4951509854e3646846a5f30b6 |

(`track?_k27jg_*` in the same folder are NOT mine — another executor's.)

## 2. Gates (build job 7566965 / 7567155)
| TAG | T1 validate_strict | T1 load line | T2 in-build | T2 `--adaptive` | causal check | T2 load |
|---|---|---|---|---|---|---|
| k27lat | SUBMITTABLE | batch 56, amp float16, gpu_band, direct_pred, alpha 0.8, centre_alpha 0.0, small_graph lazy, **members 6** | SUBMITTABLE | SUBMITTABLE | CAUSALITY OK (176 steps, 0.0) | r1 True r1_asym True d1c_n0 0.0 alpha 0.8 **graph CAPTURED** numpy_out True **members 6** |
| k27latln | SUBMITTABLE | same | SUBMITTABLE | SUBMITTABLE | CAUSALITY OK | same |
| k27lat_d1c | SUBMITTABLE | same | NOT SUBMITTABLE (stateless — expected for D1c) | SUBMITTABLE | CAUSALITY OK | d1c_n0 30.0, graph CAPTURED, members 6 |
| k27all | SUBMITTABLE, `joint_gain True/True` | same | SUBMITTABLE | SUBMITTABLE | CAUSALITY OK | joint_gain True, graph CAPTURED, members 6 |
| k27all_d1c | SUBMITTABLE | same | NOT SUBMITTABLE (stateless, expected) | SUBMITTABLE | CAUSALITY OK | joint_gain True, d1c_n0 30.0, CAPTURED, members 6 |

Baked constants checked in the build logs AND by `diff -r` of the extracted zips vs `track{1,2}_x27_a080.zip`: the only code
differences are WARM_TAILS True, PIN_OUT 1024, BATCH_CAP 56, LN_MANUAL (False/True), ARENA_STEPS 1024 (+ the K1 code), and in
k27all* JOINT_GAIN True in all three files (`_fno.py`, both `submission.py`). ALPHA 0.80, AMP "fp16", SMALL_N 2, SMALL_GRAPH(_LAZY)
True, D2_DIRECT_PRED True, R1/R1_ASYM/NUMPY_OUT True as in x27.
(The validate scores printed by the gates are ALL-DATA on val_unseen = contaminated, equality checks only.)

## 3. Output equivalence vs the baseline zips (val_unseen)
### T1 (fresh process per variant; `K/e1/t1eq_7566967.jsonl`)
| input | N 14 | N 28 | N 56 | N 99 | cold first call N 56 |
|---|---|---|---|---|---|
| x27 base, f32 | bacdc5faf9 | 0949cf62bd | 6884ecacf7 | e8b387acf0 | 6884ecacf7 |
| **k27lat, f32** | **bacdc5faf9** | **0949cf62bd** | **6884ecacf7** | **e8b387acf0** | **6884ecacf7** |
| x27 base, f64 | bacdc5faf9 | 0949cf62bd | 6884ecacf7 | e8b387acf0 | 6884ecacf7 |
| **k27lat, f64** | identical | identical | identical | identical | identical |
| k27latln | 5f7997bd6c | f21e7a8a38 | eedc5f1759 | 5ff69bad2b | differs (expected) |
Plus every call of the edge-case runs (N 1, 13, 100, 1030, 336, f64 14, 9x 99) md5-identical base vs k27lat (§4).
N 336 full arrays: k27lat vs base max|diff| **0.0** on prediction / lower / upper.

### T2 (1100-step stream, val_unseen tiled, reset every 42; `K/e1/t2edge_7566968.jsonl`)
| | md5 first 336 steps | md5 all 1100 | md5 steps 1020–1030 (across the arena end) |
|---|---|---|---|
| x27 base | 2feb8fae18 | d9164f99c0 | 1b801b8425 |
| **k27lat** | **2feb8fae18** | **d9164f99c0** | **1b801b8425** |
| x27_d1c base | 9d893c8d6d | 68ae71e3ed | a0a65e598a |
| **k27lat_d1c** | **9d893c8d6d** | **68ae71e3ed** | **a0a65e598a** |

### LN_MANUAL (T1 only)
* All-data package, N 336: prediction max|diff| 4.0e-3 (physical), rel-L2 2.9e-4; lower 2.7e-3 / 8.1e-5; upper 2.4e-3 / 7.5e-5.
* **Held-out** (twin packages, members = the 5 twins, same band, `K/e1/twin_scores.jsonl`):

| | full rel | tke | mvpe | sps | AoA15-20 rel | tke | mvpe | sps |
|---|---|---|---|---|---|---|---|---|
| twin, LN off | 0.07505353 | 0.45459504 | 0.06580460 | 53.984175 | 0.10091022 | 0.44060928 | 0.08803739 | 47.859731 |
| twin, LN on | 0.07505349 | 0.45459547 | 0.06580571 | 53.984172 | 0.10091035 | 0.44061132 | 0.08803904 | 47.859642 |
Differences at 1e-6 relative: neutral (sign noise). Twin LN vs no-LN arrays: pred max 4.0e-3, rel 3.0e-4.

## 4. Edge cases (`K/e1/t1_edge.py`, `K/e1/t2_edge.py`) — all PASS
| case | result |
|---|---|
| T1 N=1 fresh process | PASS, md5 = base (6c898f66ec), pooled; 13.8 ms/window vs base 58.7 (WARM_TAILS) |
| T1 N=13 fresh | PASS, md5 = base, pooled |
| T1 N=100 fresh | PASS, md5 = base (eb05b85d2c), pooled, pool_i 100 |
| T1 N=1030 > PIN_OUT fresh | PASS, md5 = base (e5187e4c42), falls back to plain allocation (pool untouched) |
| T1 float64 input | PASS, md5 = base at N 14/28/56/99 and cold 56 |
| T1 repeated calls in one process (1, 13, 100, 1030, 14 f64, 56, 9x99, 1) | PASS: every call md5 = base; pool slices consumed in order (1,14,114, skip 1030, 128, 184, …, 976), the call that does not fit (99 at 976) and later calls fall back; **all 16 kept outputs re-hashed at the end unchanged, 0 memory overlaps** between any two returned predictions |
| T1 finiteness / shape / lower<=upper / p==0 | PASS every call |
| T2 keep all outputs, 1100 steps | PASS: all 1100x3 kept outputs re-hashed at the end unchanged (k27lat, k27lat_d1c) |
| T2 step 1025+ (arena end) | PASS: **no wrap-around** — `arena_i` stops at 1024 and later steps take the shipped clone path; md5 across 1020–1030 and all 1100 = base; step k == step k+336 outputs (periodic check) true |
| Arena reuse risk | none found: T1 `_pool_take` only moves forward, never wraps; T2 `arena_i` only moves forward (reset only once in `get_ttt_model` after the 3 untimed warm-up probes, whose outputs are never returned) |

Risk not testable here: T1 pins 1.5 GB and T2 1.5 GB page-locked host memory at load. A `pin_memory` exception falls back to the
shipped path, but a container memory limit that OOM-kills instead of raising would fail the run (Failed submissions are not counted).
The x27 packages already hold similar pinned staging; a smaller PIN_OUT (e.g. 128, enough for one 14–99 call per process) would
cut that to 190 MB if the commander wants a lower-risk variant.

## 5. Paired timing, package level (job 7566969, `--exclusive` A100 node a0355, only our job on it)
Fresh process per variant (import untimed), interleaved base / k27lat / k27latln, 2 rounds per cold N; warm = 3 repeats x 8 processes.

| ms/window | N 14 | N 28 | N 56 | N 99 |
|---|---|---|---|---|
| cold first call, x27 base | 2.589 | 2.198 | 2.014 | 2.211 |
| cold first call, **k27lat** | **1.605** | **1.482** | **1.361** | **1.439** |
| cold first call, k27latln | 1.494 | 1.389 | 1.320 | 1.340 |
| warm, x27 base | 1.551 | 1.478 | 1.519 | 1.788 |
| warm, **k27lat** | **1.399** | **1.346** | **1.335** | **1.389** |
| warm, k27latln | 1.295 | 1.264 | 1.237 | 1.303 |

Cold (= the evaluator's fresh-process-per-call case) k27lat vs x27: **-0.65 … -0.98 ms/window (-32 … -38 %)**, most of it
WARM_TAILS (which x27 lacks); warm (= what PIN_OUT + BATCH_CAP alone buy): -0.13 … -0.40. LN adds -0.04 … -0.10.

T2 ms/step, 336-step stream, 3 rounds interleaved:
| | median | mean | first 10 steps mean |
|---|---|---|---|
| x27 base | 3.21 / 3.31 / 3.28 | 9.67 / 5.14 / 3.57 | 37.0 / 36.9 / 5.4 |
| **k27lat** | **2.82 / 2.89 / 2.88** | **2.97 / 3.08 / 3.02** | 4.7 / 4.8 / 4.8 |
Median -0.40 ms/step; mean -0.5 … -6.7 (base had 37 ms/step first-touch stalls in its first steps in 2 of 3 rounds; the arena removes
them). The board scores the MEAN.

Board transfer (honest): these are host-memory / allocation / cold-start cuts, not FLOPs cuts; CLAUDE.md: host-side accounting has not
transferred reliably. Bracket with slope [0, 1.75]: T1 k27lat vs x27 as built -0.65 … -1.0 local -> **board 0 … -1.7 ms, +0 … +0.12
final** (slope 1: ~ -0.8 ms, +0.06); T2 -0.4 (median) … -1 (mean) local -> **0 … -0.7 board ms, +0 … +0.05 final**. Inside one timing
draw on either board; free because bit-identical.

## 6. Integration clone (phase 2)
* `joint_gain.patch` applied cleanly on top of the K1 commit (the two knobs touch disjoint hunks); no manual resolution needed.
* **Knobs OFF (k27int_off) = x27 baseline; LAT only (k27int_lat) = x27 baseline** — see §6a (job 7567244/7567245).
* k27all / k27all_d1c: gates green (§2). JOINT_GAIN changes the prediction by design (all-data gate rel 0.0608 -> 0.0604,
  contaminated); its held-out evidence is the other executor's (K2), not mine.

### 6a. Integration equivalence (jobs 7567244 T1, 7567245 T2; `K/e1/t1eq_7567244.jsonl`, `t1edge_7567244.jsonl`, `t2edge_7567245.jsonl`)
| package (wt_int) | T1 md5 N 14/28/56/99 + cold 56, f32 AND f64 | T1 edge calls (N 1,13,100,1030,336, seq of 16) | T1 N 336 max|diff| | T2 md5 336 / 1100 / 1020–1030 |
|---|---|---|---|---|
| k27int_off (all new knobs OFF) | **= x27 base** | **all = base** | **0.0** | **= x27 base** (2feb8fae / d9164f99 / 1b801b84) |
| k27int_lat (LAT only) | **= x27 base** | **all = base**, pooled | **0.0** | **= x27 base**, arena_i 1024 |
| k27all (LAT + JOINT_GAIN) | differs (by design) | all PASS (finite, shapes, l<=u, kept outputs unchanged, 0 overlaps) | pred 6.3e-2 max, rel 8.5e-3 | differs by design; kept_ok, periodic_equal |
| k27all_d1c | — | — | — | differs by design; kept_ok, periodic_equal |
Small-graph path under JOINT_GAIN (job 7567360): N=1 calls (1st eager, then the lazily captured batch-1 graph) vs the same window
in an N=56 call: k27all max|diff| pred 0.9–1.4e-3, same size as x27 base / k27lat (1.2–1.4e-3, the known batch-1 vs batch-56 fp16
difference) -> the graph's JOINT_GAIN branch is consistent with predict_fno's.
T2 medians in that job (not a timing job, shared node): base 3.03, int_off 2.99, int_lat 2.82, k27all 2.84, k27all_d1c 2.90 ms/step.

## 7. Rebuild recipe (new band and/or new members)
One build per clone at a time; `wt_int` carries both knobs. Copy `K/jobs/rp_k271_bint.sh` (it defines COMMON / T1K / T2K / LAT and
restores the three sed-ed files after each build), then:
```bash
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927; cd $K
# new band: BAND=<abs path>.  new members: edit A= (and M= twins) in the script, or pass absolute paths (script only checks [ -f ]).
env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES sbatch --exclude=a0130 \
  --export=ALL,BAND=/abs/path/new_band.npz,TAGS="k27all_a080 k27all_d1c_a080" \
  --chdir=$K/wt_int --output=$K/logs/rp_k271_bint_%j.log jobs/rp_k271_bint.sh
```
The script's cases: `k27all_a080` = LAT + JOINT_GAIN; `k27all_d1c_a080` = + D1C_N0 30; `k27all_ln_a080` = + LN_MANUAL;
`k27lat_a080` = LAT only; `k27int_off_a080` = no new knob. **Use a new TAG per new band** (edit the case labels, e.g. copy
`k27all_a080)` to `k27allB_a080)`), or the zip in `K/sub_candidates` is silently overwritten.
Manual equivalent (inside an sbatch in `$K/wt_int`):
```bash
env RPDE_FORCE=1 RPDE_SUB=$K/sub_candidates RPDE_BAND_FILL=1 RPDE_CENTRE_ALPHA=0 RPDE_INT8=1 RPDE_LINEAR=tensor 'RPDE_SHIP_AMP="fp16"' \
  RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch RPDE_FFNO_PATH=dense RPDE_BAND=<band.npz> \
  RPDE_SHIP_DIRECT_PRED=True RPDE_SHIP_SMALL_GRAPH=True RPDE_SHIP_SMALL_N=2 RPDE_SHIP_SMALL_GRAPH_LAZY=True \
  RPDE_SHIP_R1=True RPDE_SHIP_R1_ASYM=True RPDE_SHIP_NUMPY_OUT=True \
  RPDE_SHIP_WARM_TAILS=True RPDE_SHIP_PIN_OUT=1024 RPDE_SHIP_BATCH_CAP=56 RPDE_SHIP_ARENA_STEPS=1024 \
  RPDE_SHIP_JOINT_GAIN=True [RPDE_SHIP_D1C_N0=30.0] \
  RPDE_A="<members, space-separated>" RPDE_M="<held-out twins>" RPDE_TAG=<new_tag> \
  bash Track_1/submit_fno/build_round13.sh 0.80
git checkout -- Track_1/submit_fno/submission.py Track_2/submit/submission.py Track_1/submit_fno/_fno.py
python Track_2/validate_t2.py --adaptive --zip $K/sub_candidates/track2_<tag>.zip --extract /tmp/v_<tag>
python Track_2/stream_eval.py --submission /tmp/v_<tag> --causal-check
```
Rules: a band refit is needed when the members change (leave `RPDE_BAND` unset -> refit from `RPDE_M`); a new band with the same
members: `RPDE_BAND=<path>`. Build time ~45 s per tag with a reused band, gates +40 s per T2 zip. JOINT_GAIN asserts CENTRE_ALPHA 0
(centre0 band) at load. A new architecture member must load in `_fno.load_any`; BATCH_CAP 56 / PIN_OUT stay valid for any member set.

## 8. Verdict per knob
| knob | verdict | reason |
|---|---|---|
| WARM_TAILS (T1) | **SHIP** | bit-identical; cold first call -0.6 … -1.0 ms/window vs x27 as built (the x27 T1 zips lack it) |
| PIN_OUT 1024 (T1) | **SHIP** | bit-identical at every N/dtype tested; no aliasing; clean fallback at N > 1024 / exhaustion; -0.13 … -0.4 warm. Risk: 1.5 GB pinned at import |
| BATCH_CAP 56 (T1) | **SHIP** | bit-identical (N 99, 100, 336, 1030 md5 = base); -0.4 ms/window at N 99 warm |
| ARENA_STEPS 1024 (T2) | **SHIP** | bit-identical over 1100 steps incl. past the arena end; no wrap-around (falls back); kept outputs never overwritten; median -0.4, mean -0.5+ ms/step. Risk: 1.5 GB pinned at load |
| LN_MANUAL (T1) | optional / SHIP-able | not bit-identical (pred rel 3e-4), held-out rel/tke/mvpe/sps unchanged to 1e-6; extra -0.04 … -0.10 ms/window. Low value, tiny risk |
| JOINT_GAIN (both) | gates green, integrated | accuracy verdict belongs to K2's held-out read |
