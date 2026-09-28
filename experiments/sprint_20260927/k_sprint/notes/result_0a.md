# Result E0a — SIGBLEND band, Track 1 (executor E0a, 2026-09-27, written 17:40 EDT)

**Verdict: SHIP `track1_k27sb05_a080.zip`** as the upside pick. `track1_k27sb03_a080.zip` is the lower-risk alternative.
The gain is small: point estimate about +0.025 final, bracket -0.005 to +0.09. It does not close the +0.2 gap by itself.
The code path is proven exact: the prediction is bit-identical to the baseline x27_a080, and the band equals the design
formula to within 6e-8 absolute (1e-5 relative). The packages survive every edge case tested.
**Only the Track-1 zips are valid.** Both `track2_k27sb0*` zips were deleted right after each build.

Clone `$K/wt_0a` (branch qwu/agent, local commits 1201ce4, c69d8ba, 9ea280b on top of 55f0e88; never pushed). Patch:
`$K/pkg/k0/sigblend_t1.patch`. Jobs: rp_k270_main 7567355, rp_k270_lat 7567366, rp_k270_fast 7567382, rp_k270_ship2 7567462
(Ascend nextgen; Cardinal stayed PENDING more than 3 min, so the job was cancelled and moved to Ascend).

## 1. Code path (read before building)
* `submission.py`: `EBAND` only acts on the GPU-band path. `use_gpu_band` needs no small graph, GPU_BAND, no graph, no err
  head, no blend, a sigma band, cuda, and CENTRE_ALPHA 0 or ALPHA. Any other path raises RuntimeError. `build_round13.sh:144`
  bakes `RPDE_SHIP_EBAND` with `ast.literal_eval` and a regex replace. The existing knobs are `RPDE_SHIP_SMALL_GRAPH` and
  `RPDE_SHIP_WARM_TAILS`.
* `_fno.predict_fno` band_gpu: `half = base * clip(sig/ref,.3,3)^p` (the ratio is computed on the host), then
  `half[..,:2] *= exp(yhat)`, then `lo = min(c0 - half, P)` and `hi = max(c0 + half, P)`, with c0 the alpha-0 field.
* **Change (commit 1201ce4 + 9ea280b): a logsig-only fast path.** It runs when the mode is "mult", clip is None and
  coefficients 1..4 are exactly zero. In that case `yhat = b0*log(sigma+1e-8) + b5` on the (B,1,H,W,2) sigma field. Any
  other EBAND constant takes the unchanged general path. The fast path is **bit-identical** to the general path: max|d| is
  0.0 on the full 336-window read of both lam packages, for P, lower, upper, sigma and c0.

## 2. Held-out read, real code path (twin members, val_unseen, one predict() call of 336 windows, kit v9 scorer)
The packages hold the 5 twins (`p2_ffno_l8_sim_s1..s3`, `arch_vit_d384`, `arch_vit_cmb3sim`). They were built by
`build_round13.sh` with the shipped T1 constants: alpha 0.80, centre 0, fp16, int8 tensor, bmm, dense, host-write torch,
direct pred, WARM_TAILS True and **SMALL_GRAPH False**. The three arms:
* **ref**: shipped band `band_r38_a080_c0_fill.npz`, EBAND None.
* **sb05**: `band_k0_sigblend_lam0.5.npz` plus its EBAND.
* **sb03**: the lam 0.3 npz plus its EBAND.

The prediction is bit-identical across the three packages, so the change in rel, tke and mvpe is exactly 0.
Stretch: `y_f = c0 + f (y - c0)`, with masked (zero) targets kept at 0. C = sps / (100 abar), shown x100.
Values are trajectory means.
Logs: `$K/logs/k0a_score.log` (script `Track_1/verify/k0a_score.py`), dumps in `$K/dumps/k0a/`.

**Reference (shipped band, twins):**

| f | split | rel | tke | mvpe | sps | cov % | C x100 |
|---|---|---|---|---|---|---|---|
| 1.00 | AoA 15-20 | 0.10091 | 0.44061 | 0.08804 | 47.785 | 87.58 | 64.004 |
| 1.00 | all 8 | 0.07505 | 0.45460 | 0.06580 | 54.046 | 92.03 | 69.831 |
| 1.29 | AoA 15-20 | 0.12753 | 0.55386 | 0.11364 | 42.149 | 81.22 | 59.826 |
| 1.29 | all 8 | 0.09454 | 0.56128 | 0.08483 | 49.243 | 87.35 | 66.655 |

**Paired deltas (arm - ref):**

| arm | f | split | dsps | dC x100 | sign (dC > 0) | cov |
|---|---|---|---|---|---|---|
| sb05 | 1.00 | AoA 15-20 | +0.103 | **+0.139** | 3/3 | 87.92 |
| sb05 | 1.00 | all 8 | +0.538 | **+0.674** | 8/8 | 92.16 |
| sb05 | 1.29 | AoA 15-20 | +0.031 | **+0.044** | 2/3 | 81.40 |
| sb05 | 1.29 | all 8 | +0.416 | **+0.538** | 7/8 | 87.29 |
| sb03 | 1.00 | AoA 15-20 | +0.105 | **+0.141** | 3/3 | 87.82 |
| sb03 | 1.00 | all 8 | +0.370 | **+0.464** | 8/8 | 92.14 |
| sb03 | 1.29 | AoA 15-20 | +0.065 | **+0.093** | 3/3 | 81.37 |
| sb03 | 1.29 | all 8 | +0.301 | **+0.393** | 8/8 | 87.35 |

**Per trajectory, dC x100:**

| arm, f | 3750_0 | 26700_0 | 16500_5 | 6300_10 | 13950_10 | 21600_15 | 8850_20 | 11400_20 |
|---|---|---|---|---|---|---|---|---|
| sb05, 1.00 | +2.534 | +0.751 | +0.852 | +0.678 | +0.157 | +0.092 | +0.223 | +0.101 |
| sb05, 1.29 | +2.374 | +0.573 | +0.697 | +0.439 | +0.092 | +0.085 | +0.087 | **-0.040** |
| sb03, 1.00 | +1.610 | +0.504 | +0.583 | +0.453 | +0.141 | +0.123 | +0.186 | +0.114 |
| sb03, 1.29 | +1.530 | +0.398 | +0.499 | +0.324 | +0.114 | +0.118 | +0.121 | +0.040 |

**In-sample caveat.** These package reads use the power-law constants fitted on all 8 trajectories: 42 numbers, fitted at
inflate 1.29. They are therefore in-sample. The honest leave-one-trajectory-out (LOTO) numpy numbers (design §4) are:

| arm | f | AoA 15-20 | all 8 |
|---|---|---|---|
| lam 0.5 | 1.00 | +0.146 | +0.626 |
| lam 0.5 | 1.29 | **+0.029** (2/3) | +0.482 (7/8) |
| lam 0.5 | 1.39 | **-0.010** (2/3) | not quoted |
| lam 0.3 | 1.29 | +0.083 (3/3) | +0.357 (8/8) |

**Code-path agreement.** The package read matches the numpy in-sample reference
(`$K/logs/k0a_insample.log`, `Track_1/probe_k0a/insample.py`) to within 0.003 in dC:
* lam 0.5, f 1.00: +0.674 against +0.674 (all 8) and +0.139 against +0.137 (AoA 15-20).
* lam 0.5, f 1.29: +0.538 against +0.538 (all 8) and +0.044 against +0.041 (AoA 15-20).
* Reference sps 54.046 against 54.051.

The package therefore evaluates exactly what the numpy study scored.

(An early run of my scorer let masked zero targets become non-zero under the stretch, which inflated the f=1.29 deltas
about 10x. It was fixed before any number above was recorded. Only the corrected numbers appear here.)

## 3. Verifier `Track_1/verify/sigblend.py` (package's own outputs, sigma and centre; `$K/logs/k0a_verifier{,_ship}.log`)
Run twice: on the twin packages against the twin ref, and on the **final ship zips** against the baseline
`final0927/sub_candidates/track1_x27_a080.zip`. That baseline carries the exact band the SIGBLEND npz is derived from
(md5 0cb37d82). All checks **PASS** for both lam and both member sets:
* The prediction is identical to the shipped-band package (max|d| 0.0), and so is the alpha-0 centre.
* All three arrays are finite. `lower <= P <= upper` holds everywhere. The p channel is exactly 0 in P, lower and upper.
* Package against the numpy formula `half = base' clip(sigma/sigma_ref,.3,3)^p' exp(lam q_c log(sigma+1e-8))` with
  `[min(c0-half,P), max(c0+half,P)]`:
  * lower and upper: **max|d| 2.98e-8**.
  * width: **max|d| 5.96e-8, max relative 1.1e-5**.
  * Where the contain-P rule is inactive (99 % of elements): max relative |w - 2 half| 5.4e-6.

  This is float32 noise, not a factor.
* The median width ratio SIGBLEND / shipped (u / v) is 0.963 / 0.947 for lam 0.5 and 0.978 / 0.968 for lam 0.3. For
  comparison, a band built from the SIGBLEND npz without the sigma-power factor would be about 3x wider.

## 4. Edge cases (`Track_1/verify/k0a_edge.py`; twin sb05 package AND both final ship zips; SMALL_GRAPH off)

| case | result |
|---|---|
| N = 1, 2, 13, 14, 56, 99, 100, 336, float32, one process, called twice each | PASS (no crash; shape, float32, finite, lo<=P<=hi, p = 0) |
| same N, float64 input, called twice each | PASS |
| fresh process per call: twin sb05, N in {1,2,13,14,56,99,100,336} x {f32, f64} (16 processes) | PASS 16/16 |
| fresh process per call: ship sb05 and sb03, N in {1,13,14,99}, f64 | PASS (see note) |
| permutation of 336 windows | PASS: P, lower and upper **bitwise** equal after un-permuting; rel, tke, mvpe and sps identical |
| batch composition | N >= 14: bitwise equal to the same windows in the 336 call |
| WARM_TAILS at import (predict for 1..112 rows, f32 and f64, EBAND active) | PASS |

Notes:
* **Fresh-process calls on the ship zips:** ship sb05 and sb03 each passed 4/4 (N = 1, 13, 14, 99, float64; job 7567462,
  finished 17:41).
* **Batch composition, N < 14:** outputs differ from the 336-call values by up to 4e-3 absolute (2.5e-4 of rms). This is the
  pre-existing fp16 kernel choice at small batch. It is not cross-window information: permutation is bitwise-invariant.

## 5. Latency (A100, `--exclusive` node, one process, interleaved arms, 20 reps; package batch forced equal)
Log: `$K/logs/rp_k270_fast_7567382.log`. Values are ms/window, median; the paired delta is against the shipped band.

| N | batch | ship | SIGBLEND general path | SIGBLEND fast path |
|---|---|---|---|---|
| 14 | 112 | 1.345 | +0.057 (20/20 slower) | **+0.025** (20/20) |
| 56 | 112 | 1.326 | +0.031 (20/20) | **+0.003** (10/20) |
| 99 | 112 | 1.486 | +0.031 (20/20) | **+0.002** (15/20) |
| 14 | 56 | 1.324 | +0.056 (18/20) | **+0.038** (19/20) |
| 56 | 56 | 1.340 | +0.031 (20/20) | **+0.004** (10/20) |
| 99 | 56 | 1.334 | +0.031 (20/20) | **+0.001** (14/20) |

* Kernel level, `exp(eband_yhat)` on the general path: 0.044 / 0.028 / 0.029 ms/window at N = 14 / 56 / 99.
* The general path cost 0.03 to 0.06 ms/window, which is at the 0.05 threshold, so the fast path was added. It costs
  0.001 to 0.004 ms/window at N >= 56 and 0.025 to 0.038 at N = 14.
* An earlier unforced read (`rp_k270_lat`) showed +0.19 at N=99. That was a confound: `_pick_batch` chose 56 for one
  package and 112 for the other.
* Board transfer: a FLOPs-type cost transfers at x0.9 to 1.75, and host-side launch accounting did not transfer
  historically. Board bracket: **0 to +0.07 board ms**, which is time 0 to -0.05 and final 0 to -0.005.

## 6. Packages (6 all-data members: 4x `ffno_all_w40l8_sim_s1..s4`, `arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3`)
Knobs:
* Common set: FORCE, BAND_FILL, CENTRE_ALPHA 0, INT8, LINEAR tensor, AMP "fp16", SPEC bmm, HOST_WRITE torch, FFNO dense,
  alpha 0.80.
* T1 27lz with `RPDE_SHIP_SMALL_GRAPH=False`: DIRECT_PRED True, SMALL_N 2, LAZY True (inert with SMALL_GRAPH False).
* `RPDE_SHIP_WARM_TAILS=True`: the knob exists. It moves untimed load only and outputs are bit-identical. Note that the
  runbook's 27lz baseline has it OFF, so it is a second (latency-only) difference from x27_a080.
* `RPDE_BAND` = the SIGBLEND npz, `RPDE_SHIP_EBAND` = the literal from the matching .json.

Built at HEAD 9ea280b (fast path with mode guard). Logs: `$K/logs/k0a_build_k27sb0{5,3}_a080.log`.

| zip | sha256 | gate |
|---|---|---|
| `$K/sub_candidates/track1_k27sb05_a080.zip` (30,284,409 B) | `380f86c2408fb871c18ba964e758a18d7434649bbe15e4c3ee6e97928c493692` | validate_strict **SUBMITTABLE** |
| `$K/sub_candidates/track1_k27sb03_a080.zip` (30,289,459 B) | `ed5df45984ebc245fc500481115a5a4daf8a66ff29b61d51f07e9dc636f518ad` | validate_strict **SUBMITTABLE** |

* The load line reads `gpu_band True direct_pred True alpha 0.8 centre_alpha 0.0 small_graph None members 6`.
* The baked constants read `EBAND = (((0.24232..,0,0,0,0,0),(0.24535..,0,0,0,0,0)),'mult',None)` for sb05 and
  `((0.14539..),(0.14721..))` for sb03, with `SMALL_GRAPH = False`, `WARM_TAILS = True` and `ALPHA = 0.80`.
* The zip's `_fno.py` carries the guarded fast path.
* `track2_k27sb05_a080.zip` and `track2_k27sb03_a080.zip` were **deleted** after each build. They were invalid: the npz
  without SIGPOW gives a band about 3x too wide.
* Earlier builds of the same tags, made at 17:28 without the mode guard, were overwritten. Their sha256 values
  (901f4293, 6fa731d7) are superseded; do not use them.

## 7. Prediction, `k27sb05_a080` vs the baseline `x27_a080` (T1 27lz), board scale
Board baseline: abar = sps/C = 43.06 / 59.94 = 0.718.

| subscore | delta | reason |
|---|---|---|
| rel | 0 | prediction bit-identical |
| tke | 0 | |
| mvpe | 0 | |
| time | 0 to -0.05 (point -0.01) | fast path, see §5; WARM_TAILS can only help the cold chunk |
| sps | **+0.02 to +0.35, point +0.10** | from dC x100 (below) times 0.718 |

dC x100 used for sps:
* +0.03 if the board looks like AoA 15-20 at f 1.29 (the honest LOTO number).
* +0.48 if it looks like the full split.
* Point +0.14, shaded low because band changes have historically over-read locally.

Final: **+0.00 to +0.09, point +0.025.**
For `k27sb03_a080`: sps +0.06 to +0.26, final +0.015 to +0.065, point +0.025. lam 0.3 is 3/3 on AoA 15-20 at f 1.00,
1.29 and 1.39 LOTO. lam 0.5 is 2/3 at 1.29 and -0.01 at 1.39, but has about 1.3x the full-split upside.

## 8. Verdict
**SHIP `track1_k27sb05_a080.zip`** (Track 1 only), because today is a free roll and we maximise upside.
* **Why it is safe:** the code path is exact, the prediction is unchanged, the bounds are valid everywhere and it
  survives every edge case tested. It is 8/8 and 7/8 positive on the 8 held-out trajectories (in-sample constants) and
  7/8 LOTO.
* **Risk:** on an AoA 15-20-like test set at the board's stretch, the gain is near zero (LOTO +0.03 / -0.01 C x100 at
  1.29 / 1.39).
* **Alternative:** if the commander prefers the variant that is never below the shipped band on the board proxy, pick
  `track1_k27sb03_a080.zip`. Its expected value is similar and its upside is smaller.
* **Not a big lever:** expect about +0.03 final, far from the +0.2 needed.
