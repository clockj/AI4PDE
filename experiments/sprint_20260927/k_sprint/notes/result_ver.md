# E-VER: independent verification of the COMBINED k27fin packages (written 17:48 EDT 09-27)

These are the packages checked, as built by the commander at 17:36 (clone `$K/wt_fin` @ b539609). The sha256 values were re-read inside the jobs.
* `track1_k27fin_a080.zip` 438cd3ff… · `track2_k27fin_a080.zip` 470f278d… · `track2_k27fin_d1c_a080.zip` eca654a2…
* The band.npz inside all three is byte-identical (md5 e8309d9f) to `$K/pkg/k0/band_k0_sigblend_lam0.5.npz`.

Jobs, all on Ascend nextgen A100 with a0130 excluded: rp_k27v_A 7567501, rp_k27v_B 7567503, rp_k27v_cbuild 7567497, rp_k27v_C 7567504, rp_k27v_D 7567506 (`--exclusive`).
* Logs: `$K/logs/rp_k27v_*`.
* Scripts: `$K/ver/{t1chunk,t2run,t1lat,cmpA,cmpB,scoreC,edge}.py` and `$K/ver/job*.sh`.
* Outputs: `$K/ver/out/{A,B,C,D}`.
* Twin packages: `$K/ver/sub/`.

Nothing was written into wt_fin, sub_candidates or final0927.

## A. T1 k27fin, real predict() on val_unseen — PASS
* **A1 PASS.** Prediction of k27fin vs k27all, chunked at N = 14 / 56 / 99 / 336: max|d| **0 / 0 / 0 / 0** (bitwise).
  * k27fin vs x27 is 6.27e-2. That is exactly JOINT_GAIN's known effect.
* **A2 PASS.**
  * Checked at every N: finite, lower <= pred <= upper everywhere, p channel exactly 0 in pred, lower and upper.
  * The returned value matches baseline x27: a dict {prediction, lower, upper} of numpy float32 arrays, (336,20,32,64,3).
  * Kept outputs of earlier calls were re-hashed after all later calls: 0 overwritten.
  * The mean band width of fin / all is 0.936.
* **A3 PASS.** `verify/sigblend.py` on k0a_dump dumps of fin (and k27all as the shipped-band reference):
  * The prediction is identical to the shipped-band package, and the centre c0 is identical.
  * lower vs the numpy formula: max|d| 2.98e-8. upper: 2.98e-8. width: max|d| 5.96e-8, max rel 9.9e-6.
  * Where contain-P is inactive (98.6 % of elements), the max rel |w - 2 half| is 5.0e-6.
  * The median width ratio SIGBLEND / shipped is 0.963 (u) / 0.948 (v). VERIFIER PASS.
* **A4 PASS.** Edge cases:
  * **One process:** N = 1, 2, 13, 14, 56, 99, 100, 129, 336, each called twice, in f32 and in f64. Every call passed, and the permutation of 336 windows is **bitwise** invariant (rel, tke, mvpe and sps identical).
  * **Fresh process:** each of those N in f32 and f64 (18 processes) PASS.
  * **Batch independence:** outputs for N >= 13 are bitwise equal to the 336-call values.
  * **N = 1 and 2 differ from batched by 2.04e-3 and 2.07e-3 abs (~2e-4 of rms).** This is the pre-existing fp16 small-batch kernel choice, the same one result_0a §4 reports (up to 4e-3). It is not cross-window information: permutation is bitwise. With SMALL_GRAPH off, N = 1 now runs eager. In k27all it ran the batch-1 graph, which was also about 1e-3 off batched.
* **A5 PASS.** `validate_strict` logic run on the EXISTING zip (my copy `ver/wt/Track_1/submit_fno/validate_zip.py`, with the rebuild step removed) prints **SUBMITTABLE**.
  * The load line reads `gpu_band True direct_pred True alpha 0.8 centre_alpha 0.0 joint_gain True/True small_graph None members 6`.
  * It also checks the import with the extract dir off sys.path, order independence, and 36.45 MB < 256 MB.

## B. T2 k27fin / k27fin_d1c, 336-step stream over val_unseen — PASS
* **B1 PASS.** The prediction equals k27all (and k27all_d1c) step by step: max|d| **0** for both.
* **B2 PASS, and the difference is explained.** lower and upper vs k27sbt2 (sbt2 has no JOINT_GAIN):
  * Max|dL| is 5.41e-2 and max|dU| 6.27e-2, which equals max|dP| (6.27e-2). 14.2 % of elements differ; 99.2 % of those also have P differing.
  * In the 8 trajectory-first windows (no R1 state yet), every one of the 10 843 differing elements is where P is at a bound (contain-P). Everywhere else in those windows the widths are identical (max|dW| 0).
  * In later steps the per-step mean width ratio fin / sbt2 is 0.9990–1.0032 (median 1.00026). That is R1's realized-error feed seeing the JOINT_GAIN prediction.
  * d1c: identical picture.
  * Checked too: finite, L <= P <= U, p channel 0.
* **B3 PASS.**
  * Kept outputs (raw returned objects, hashed at return and re-hashed at the end): 0 of 336 overwritten, on both.
  * `validate_t2 --adaptive`: **SUBMITTABLE** on both.
  * `stream_eval --causal-check`: **CAUSALITY OK** (176 steps, max|diff| 0) on both.
  * The load line has `graph CAPTURED`, `sigpow (0.2423…, 0.2454…)`, `joint_gain True`, `r1_asym True`, `numpy_out True`, `members 6`, with d1c_n0 0 / 30.
  * The in-build non-adaptive "NOT SUBMITTABLE: stateless" line on d1c is the expected adaptive-package message (BUILD_RUNBOOK §3).

## C. Held-out score of the COMBINATION (5 twins, val_unseen, kit scorer) — PASS, matches the sum of parts
* **Packages:** built in my clone `$K/ver/wt` (= b539609) with a copy of rp_k27_final.sh, A = M = the twins.
  * `vtwin_fin`: exactly the final constants.
  * `vtwin_base`: the shipped band `band_r38_a080_c0_fill.npz`, T1 27lz knobs (SMALL_GRAPH lazy), no JOINT_GAIN / EBAND / SIGPOW, with the same accuracy-neutral LAT knobs.
* **Same code as shipped:** submission.py and _fno.py in vtwin_fin are byte-identical to the shipped k27fin zips (T1, T2, T2 d1c).
* **Baselines reproduce baseline_ref:** T2 base 55.594 / 48.481, and d1c 55.646 / 48.484.
* **How it was run:** T1 through predict() in N = 56 chunks, T2 through the platform loop. `final4` = 0.4778 rel + 0.1010 tke + 0.0921 mvpe + 0.2498 sps (no time).

| read | split | rel_err base→fin | tke_err | mvpe_err | sps base→fin | cov base→fin | Δfinal4 | signs (final / sps / rel) |
|---|---|---|---|---|---|---|---|---|
| T1 predict | all 8 | 0.07505→0.07471 (**−0.462 %**) | 0.45460 (0.000 %) | 0.06580 (0.000 %) | 53.984→54.525 (**+0.541**) | 0.9200→0.9214 | **+0.143** | 8/8, 8/8, 8/8 |
| T1 predict | AoA 15-20 | 0.10091→0.10043 (−0.479 %) | 0.44061 (0) | 0.08804 (0) | 47.860→47.979 (**+0.119**) | 0.8766→0.8800 | **+0.040** | 3/3, 3/3, 3/3 |
| T2 asym R1 | all 8 | 0.07505→0.07471 (−0.462 %) | 0.45459 (0) | 0.06580 (0) | 55.594→55.823 (**+0.229**) | 0.9049→0.9050 | +0.065 | 8/8, 8/8, 8/8 |
| T2 asym R1 | AoA 15-20 | 0.10091→0.10043 (−0.479 %) | 0.44061 (0) | 0.08804 (0) | 48.481→48.564 (**+0.083**) | 0.8615→0.8623 | +0.031 | 3/3, 3/3, 3/3 |
| T2 asym R1 + D1c30 | all 8 | 0.07461→0.07427 (−0.466 %) | 0.45459 (0) | 0.06464 (0) | 55.646→55.874 (**+0.229**) | 0.9050→0.9051 | +0.065 | 8/8, 8/8, 8/8 |
| T2 asym R1 + D1c30 | AoA 15-20 | 0.10072→0.10023 (−0.481 %) | 0.44061 (0) | 0.08740 (0) | 48.484→48.563 (+0.078) | 0.8616→0.8622 | +0.030 | 3/3, 3/3, 3/3 |

Per-trajectory Δsps:
* **T1:** 3750_0 +2.127, 16500_5 +0.690, 26700_0 +0.546, 6300_10 +0.542, 8850_20 +0.194, 13950_10 +0.129, 11400_20 +0.094, 21600_15 +0.066.
* **T2 asym:** 3750_0 +0.881, 16500_5 +0.240, 26700_0 +0.208, 6300_10 +0.182, 8850_20 +0.115, 13950_10 +0.097, 11400_20 +0.090, 21600_15 +0.043.

Against the expected values:
* rel −0.46 %: matched.
* tke and mvpe unchanged: matched.
* T1 sps +0.5 / +0.1: read +0.54 / +0.12.
* T2 sps +0.23 / +0.08: read +0.23 / +0.08 (d1c +0.23 / +0.08).

**The combination is not worse than the sum of its parts.** It is fully additive, and every trajectory moves the right way on every read.

## D. Latency, package level — no material cost
Setup:
* One `--exclusive` job (node a0306, not shared), A100. Arms interleaved x27 → k27all → k27fin, 3 reps (T1) and 2 reps (T2).
* T1: a fresh process per arm, load untimed. "cold" = the first predict() of the process (the board regime: new subprocess per call); "warm" = the median of the next 5 calls. Values are ms/window.

T1:

| N | x27 cold | k27all cold | k27fin cold | fin − all cold | x27 warm | k27all warm | k27fin warm | fin − all warm |
|---|---|---|---|---|---|---|---|---|
| 14 | 1.99 / 2.37 / 2.68 | 1.55 / 1.49 / 1.48 | 1.52 / 1.60 / 1.48 | **+0.03** (−0.03…+0.11) | 1.61 | 1.45 | 1.50 | +0.04 |
| 56 | 1.63 / 1.76 / 1.70 | 1.34 / 1.34 / 1.34 | 1.37 / 1.37 / 1.38 | **+0.03** | 1.52 | 1.34 | 1.53 | **+0.18** |
| 99 | 1.92 / 1.70 / 1.77 | 1.39 / 1.40 / 1.40 | 1.41 / 1.42 / 1.42 | **+0.02** | 1.65 | 1.38 | 1.58 | **+0.20** |

T2, ms/step over the 336-step stream, 2 reps:

| | mean | median |
|---|---|---|
| x27_d1c | 3.296 / 3.271 | 3.195 / 3.103 |
| k27all_d1c | 3.130 / 3.071 | 2.958 / 2.957 |
| k27fin_d1c | 3.115 / 3.072 | 2.964 / 2.965 |
| fin − all (SIGPOW) | −0.01 | +0.007 |

**T2: SIGPOW costs nothing measurable.** The B and C jobs agree: fin 3.03 vs all 3.03 ms/step.

**T1: EBAND (plus SMALL_GRAPH off) costs about +0.03 ms/window cold, which is the board regime.** That is about +0.05 board ms at 1.75x, or −0.004 final.
* Caveat: the WARM repeat calls at N >= 56 in one process read +0.18…+0.20 ms/window slower for k27fin than k27all. Cold they are equal, and I did not investigate why.
* If the platform ever reuses a process across calls, the bracket is up to about +0.35 board ms, or about −0.025 final. That is still below the T1 held-out band gain on AoA 15-20 (Δfinal4 +0.040 incl. JOINT_GAIN).
* k27all/k27fin vs x27 keeps K1's gain: about −0.3 ms/window cold on T1 and −0.2 ms/step on T2.
* Local latency deltas transfer to the board only within about 2x.

VERDICT: T1 k27fin GO, T2 k27fin GO, T2 k27fin_d1c GO

## E. PIN_OUT 1024 rebuild (written 17:58 EDT)
Packages:
* `track1_k27fin2_a080.zip` bde1fe66…
* `track2_k27fin2_a080.zip` d487df0f…

Jobs: rp_k27v_E 7567602 (checks) and 7567603 (latency, `--exclusive`). Logs `$K/logs/rp_k27v_E{1,2}_*.log`, scripts `$K/ver/{eseq,t2mem}.py`, outputs `$K/ver/out/E`.

* **E1 T1 equality — PASS.**
  * **diff -r:** the extracted k27fin vs k27fin2 differ in exactly one line, `PIN_OUT = 128` → `PIN_OUT = 1024`. band.npz and the members are identical.
  * **Fresh processes:** N = 14, 56, 99, 336 × {f32, f64}. prediction, lower and upper are **bitwise equal** fin2 vs fin in 8/8 cases (and f32 = f64 hashes).
  * **One process, 16 calls** (99 ×12, then 56, 14, 1, 1030):
    * Every call's hash equals k27fin's for the same windows: 16/16.
    * Kept outputs re-hashed at the end: 0 overwritten.
    * np.shares_memory over all 48 returned arrays: 0 pairs.
  * **Pool fallback is clean.** Which calls are served from the pool (`.base` set):
    * fin2: calls 1–10 (990 rows); calls 11–12 (99) and the 56 fall back to plain allocation; the 14 and the 1 fit the remaining 34 rows again; the 1030 falls back.
    * fin (128): only the first call is served from the pool; later calls that fit the remaining rows are too.
    * No exception anywhere.
* **E2 T1 latency — PASS, the warm penalty is gone.** Exclusive A100, interleaved all → fin → fin2, 3 reps, fresh process per arm. Values are ms/window, means of 3.

| N | k27all cold / warm | k27fin cold / warm | k27fin2 cold / warm |
|---|---|---|---|
| 14 | 1.466 / 1.417 | 1.460 / 1.441 | 1.525 / 1.431 |
| 56 | 1.350 / 1.329 | 1.350 / **1.505** | 1.345 / **1.328** |
| 99 | 1.380 / 1.371 | 1.407 / **1.574** | 1.393 / **1.377** |

  * **Warm:** fin2 = k27all +0.00…+0.01, so the +0.18…+0.20 warm penalty was the exhausted 128-row pool.
  * **Cold:** equal at N = 56 and 99. At N = 14, fin2 is +0.06 ms/window over k27all and fin (1.478 / 1.540 / 1.557 vs about 1.46). That is about +0.9 ms per call, and I did not verify the cause.
    * The pool is allocated and pinned at import, so it is outside the timed call.
    * Board cost if real: at most about −0.005 final.
* **E3 T2 — PASS.**
  * `diff -r` of track2_k27fin2 vs track2_k27fin_d1c: **no difference at all**. The contents are identical; only the zip metadata differs.
  * 336-step stream: P, L and U are bitwise equal. 0 kept outputs overwritten. 3.01 vs 3.03 ms/step.
* **E4 host memory** (from `/proc/self/status`):
  * VmLck and VmPin read 0 kB in every case. CUDA pinned host memory (cudaHostAlloc) is not counted there. It shows in VmRSS.
  * **T1, after load:** VmRSS 2.25 GB (PIN_OUT 128) vs 3.63 GB (PIN_OUT 1024). The +1.38 GB matches 3 × 896 rows × 491 520 B = 1.32 GB, all page-locked.
  * **T1, after the 16-call sequence:** VmRSS 5.43 GB vs 5.53 GB, peak VmHWM 8.40 GB vs 8.49 GB (the peak is driven by the 1030-window call).
  * **T2 (ARENA_STEPS 1024), after load and 336 steps:** VmRSS 3.65 GB, VmHWM 3.97 GB (fin2 = fin_d1c).

VERDICT: T1 k27fin2 GO, T2 k27fin2 GO
