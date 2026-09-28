# Design C — STREAM STACK for Track 2 (stream-native perspective)

Author: design agent C, 2026-09-27 16:1x–16:40 EDT. Worktree `/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_c` (uncommitted diff on 55f0e88).
**No training is launched; the design is training-free** (nothing new to reproduce: three scalar constants, no fitted tables).

## 1. Hypothesis

`tke` and `mvpe` are STATISTICS of one fixed flow condition, and the stream reveals, window after window, how OUR ensemble
mis-states those statistics on THIS trajectory. The team already corrects the first moment (D1c: time-mean bias) and the band (R1).
Nothing corrects the SECOND moment. Pure replacement of sigma by the revealed variance loses (physics caveat; my probe of SIGEMA as
committed reads **-0.55** in distribution, see 3), but a **ratio calibration in innovation form** does not replace the network's
per-window sigma: it keeps the head's window-to-window variation and only removes the trajectory-persistent multiplicative
mis-calibration of the composed amplitude. That is "D1c for the variance".

The shipped construction is the **stack**, in this order inside one step:
1. **FAMW** (family weights, per pixel, causal) on the pre-compose mean.
2. **D1c** (N0 30, exists) accumulating the residual of the field AFTER FAMW (Corin's order; automatic in the patch because `y0` is taken after FAMW).
3. **VARCAL** (new): causal per-pixel gain on the composed fluctuation amplitude.
4. **asym R1** (exists) scoring the corrected P / C, as today.

Subscores that move: `tke` up (VARCAL), `mvpe` + `rel` up slightly (D1c, FAMW), `rel` DOWN slightly (VARCAL: a larger amplitude costs
phase error), `sps` up (a-factor through tke, and containment because R1 sees better-centred fields).

## 2. Exact construction (all model space = input-normalised, per trajectory, reset in `reset_ttt_state`)

Notation: n = number of revealed windows of this trajectory (0 at the first step); `box3` = 3x3 replicate-padded box mean over (H, W)
(`_r1_box3`, the R1 operator); all sums run over PAST revealed windows only.

**FAMW** (`FW_N0`; NOT shipped today (see 8.0), Decision-phase value 20.0). Groups A = first fused group (F-FNO), B = second (ViT); `yA`, `yB` their member means; `w0 = n_B / (n_A + n_B)`
(= 2/6 in the package, so n = 0 reproduces the uniform mean up to rounding).
```
d = yB - yA                                  (1,20,H,W,2) of a revealed window
fA += sum_t d*d ;  fB += sum_t d*(y - yA)     y = that window's revealed target
w  = clip( w0 + (clip(box3(fB)/max(box3(fA),1e-12), 0, 1) - w0) * n/(n+FW_N0), 0, 1 )
y0 = yA + w*(yB - yA)                         replaces the uniform mean; sigma stays the uniform mean of the heads
```
**D1c** unchanged (`D1C_N0 = 30.0`); its `C` is now the post-FAMW `y0`.

**VARCAL** (`VC_N0`, ship 50.0; `VC_CLIP = (0.25, 4.0)`).
```
vpred_k = (ALPHA*sigma_head + (1-ALPHA)*s_)^2       the UNCALIBRATED composed variance of window k, (1,1,H,W,2)
cT += var_t(y_k) (population, 20 frames) ;  cP += vpred_k
g  = 1 + (clip(box3(cT)/max(box3(cP),1e-12), 0.25, 4) - 1) * n/(n+VC_N0)        n = 0 -> exactly 1
compose:  y = m_ + (y0 - m_) * ( (ALPHA*sigma + (1-ALPHA)*s_) * sqrt(g) ) / (s_ + 1e-6)
```
Band half-width, sigma_head in the band, and the centre (centre0 = y0 + D1c shift) are untouched by VARCAL.

Constants and what they are fitted on: `FW_N0 = 20`, `VC_N0 = 50`, `VC_CLIP` — three scalars chosen on the 8 `val_unseen` trajectories with the
held-out twins (the same evidence base as the band); the probe table shows the neighbours (VC 10/30/100, FW 3/8/20), so they sit on plateaus,
not on a fitted optimum. Nothing else is fitted. Re-creatable: no table ships.

## 3. Probe table (held-out twins `p2_ffno_l8_sim_s1..s3` + `arch_vit_d384` + `arch_vit_cmb3sim`, 8 `val_unseen` trajectories, 336 windows)

Instrument: `Track_2/stream_c/probe.py` on the cached dump `$S/ckpt/c/twins_unseen.npz` (CPU; official `scoring.py`; composite
`0.4778 rel + 0.1010 tke + 0.0921 mvpe + 0.2498 sps`, no time). Baseline = today's planned construction: family mean with w_ViT = 1/3,
ALPHA 0.80, band `r35_centre0_fill.npz` (centre0), asym R1 (N0 2, the submission's own `_CausalBandScalerT`). Paired, per trajectory.
**Calibration of the instrument:** it reproduces the team's D1c read (+0.026 full 7/8, +0.0056 AoA 15-20; team +0.028 / +0.0055).

SEE SECTION 8 (appended at hand-over) for the final numbers; raw logs `$S/logs/c_probe_*.log`, `c_ood_*.log`, json next to them.

Stretch convention used here: `Y_f = P_base + f*(Y - P_base)` on scored elements (the whole stream lives in the stretched world).
**WARNING: this convention inflates the target's fluctuation variance by construction, so every VARCAL read at f = 1.29 is biased UP
(it is an upper end, not an estimate).** The f = 1.0 reads are the honest ones for VARCAL.

## 4. Wiring into `Track_2/submit/submission.py` (done in wt_c; `git -C wt_c diff`)

* New constants `VC_N0 = 0.0`, `VC_CLIP = (0.25, 4.0)`, `FW_N0 = 0.0` next to `SIGEMA_*`. **0.0 = off = the code path of 55f0e88 exactly**
  (every new statement sits behind `self.vc` / `self.fw`, both False when the knob is 0 or R1 is off; FAMW additionally needs exactly 2 fused groups).
* Build knobs added to `Track_1/submit_fno/build_round13.sh`: `RPDE_SHIP_VC_N0`, `RPDE_SHIP_FW_N0` (same `sed` pattern as `RPDE_SHIP_D1C_N0`).
* State (`__init__`): `cT, cP, cn, gVp, gG` (VARCAL); `fA, fB, fn, gYA, gYB, gW, fw0` (FAMW). Reset in `reset_ttt_state` and after graph capture
  through one helper `_reset_c()`.
* Where it runs: **inside `_body`, i.e. inside the captured CUDA graph**, no host branch. The update block sits right after the SIGEMA block and
  before the forward: it reads the previous window's `gVp` / `gYA` / `gYB` (not yet overwritten) and the revealed target `gyp`, gated by the
  existing device flag `gflag` (0 on a trajectory's first step, so sums get exact zeros). `gG` / `gW` are written there and read after the forward.
  Trajectory boundaries: exactly R1's mechanism (`reset_ttt_state` clears state and `r1_prev`; the step after a reset runs with `gflag = 0`).
* Expected added latency: ~35 (VARCAL) + ~30 (FAMW) tiny batch-1 kernels on (32,64,2) / (1,20,32,64,2) tensors inside the graph; by analogy with
  asym R1 (+0.14 ms for a much heavier einsum) I expect **+0.05 to +0.12 ms/step local**, i.e. about **-0.004 to -0.010 final**.
  Measured by job 7566662 (section 8). If > 0.15 ms, move the two update blocks into R1's side stream (`with torch.cuda.stream(r1s)`) and move
  the `wait_stream(r1s)` to just before the FAMW mix.

## 5. Commands for the executor

```bash
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
# (a) CPU probe, any arm in probe.py:ARMS (about 40 s per arm, single process)
cd $S/wt_c/Track_2/stream_c && $PY -u probe.py --f 1.0 1.29 --arms "+D1c30" "+D1c30+VC30" "+FW20+D1c30+VC30"
# OOD cell (AoA-20 holdout members; F-FNO+sim x2 as family A, FNO as family B):
$PY -u probe.py --dump $S/ckpt/c/ood_aoa20.npz --fam-f 0 1 --fam-v 4 --f 1.0 --arms "+D1c30" "+D1c30+VC30" "+FW20+D1c30+VC30"
# (b) GPU read of the REAL code path on twin packages (already built: $S/pkg/c/{base,d1c,d1c_vc,stack}); job script $S/jobs/rp_f27c_smoke.sh
python Track_2/stream_eval.py --submission $S/pkg/c/stack --split unseen --repeat 3 --causal-check
# (c) ship: today's asnp+D1c build command plus
RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0 ... bash Track_1/submit_fno/build_round13.sh 0.80
#     run it FROM wt_c (the patch is only there), then Track_2/validate_t2.py --adaptive must print SUBMITTABLE.
```
`validate_t2.py`'s shuffled-order bit-identity rule must be the relaxed D1c rule (Emre's 06:00Z change): FAMW and VARCAL change P by design.

## 6. Kill criteria (decide per component, in this order)

1. `stream_eval.py --causal-check` on `$S/pkg/c/stack` is not "CAUSALITY OK" => do not ship any of it.
2. Knob-off identity: `$S/pkg/c/base` (patched file, knobs 0) must equal today's asnp twin read bit for bit (`verify/pkg_identity.py --tol 0`).
3. GPU read (b): `d1c_vc` minus `d1c` composite (no time) < +0.010 on the full split, or < 0 on AoA 15-20, or fewer than 6/8 trajectories positive
   => drop VARCAL (ship asnp + D1c as planned).
4. `stack` minus `d1c_vc` < 0 on the full split or on AoA 15-20 => drop FAMW (ship D1c + VARCAL).
5. Added local latency of the shipped combination > 0.15 ms/step after the side-stream move => drop FAMW first, then VARCAL.
6. Graph not CAPTURED in the load line of the package (`graph CAPTURED`) => do not ship.
7. Do NOT ship SIGEMA with this (my probe: -0.55 composite in distribution, tke -2.0; replacement, not calibration).

## 7. Reproducibility

Nothing is trained. The three constants are in `submission.py`; REPRODUCE needs one line: "T2 stream stack: `RPDE_SHIP_D1C_N0=30.0
RPDE_SHIP_VC_N0=50.0`". Causal by construction (only `prev_target_norm` of the same trajectory enters any sum).

## 8. Results at hand-over
(filled below)

### 8.0 FINAL RECOMMENDATION (overrides the ship constants quoted in sections 1, 2, 5 where they differ)

**Ship today: planned `asnp_a080` + D1c (N0 30) + VARCAL with `VC_N0 = 50.0`; FAMW OFF (`FW_N0 = 0.0`).**
`RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0` (build from wt_c). FAMW stays in the patch as a Decision-phase (private, OOD) knob:
it failed my own kill criterion 4 on the real code path (AoA 15-20 -0.004, and +0.12 ms/step local).

### 8.1 CPU probe, in distribution (8 val_unseen trajectories; delta composite without time vs asym-R1 baseline; f = 1.0)

| arm | full | traj > 0 | AoA 15-20 | traj > 0 | rel / tke / mvpe / sps (full) |
|---|---|---|---|---|---|
| D1c30 (= today's optional plan) | +0.0261 | 7/8 | +0.0056 | 2/3 | +0.019 / 0 / +0.049 / +0.049 |
| D1c10 | +0.0226 | 6/8 | -0.0100 | 1/3 | |
| D1c empirical-Bayes shrink (max with n/(n+30)) | +0.0226 | 5/8 | -0.0115 | 0/3 | |
| FAMW N0 8 / 20 | +0.0101 / +0.0122 | 5/8 / 7/8 | -0.0043 / +0.0024 | 0/3 / 2/3 | |
| **SIGEMA as committed (beta 0.8, N0 3)** | **-0.5527** | 1/8 | -0.5519 | 0/3 | -0.159 / -2.042 / 0 / -1.082 |
| VARCAL N0 30 alone | +0.0218 | 6/8 | +0.0072 | 2/3 | -0.053 / +0.340 / 0 / +0.050 |
| VARCAL N0 100 alone | +0.0220 | 8/8 | +0.0147 | 3/3 | -0.022 / +0.211 / 0 / +0.045 |
| VARCAL down-only / up-only (N0 30) | +0.0033 / +0.0166 | 4/8 / 6/8 | -0.0013 / +0.0074 | | the gain is the UP side |
| D1c30 + VARCAL 10 | +0.0107 | 3/8 | -0.0307 | 0/3 | too aggressive: the cliff |
| D1c30 + VARCAL 30 | +0.0476 | 8/8 | +0.0129 | 3/3 | -0.034 / +0.340 / +0.049 / +0.099 |
| **D1c30 + VARCAL 50 (ship)** | **+0.0517** | **8/8** | **+0.0202** | **3/3** | -0.019 / +0.297 / +0.049 / +0.104 |
| D1c30 + VARCAL 100 | +0.0479 | 8/8 | +0.0202 | 3/3 | -0.003 / +0.211 / +0.049 / +0.094 |
| FAMW 8 + D1c30 (after FAMW / raw C) | +0.0286 / +0.0280 | 7/8 | +0.0008 / -0.0002 | 2/3 | Corin's order is +0.0006..0.001 better |
| FAMW 20 + D1c30 + VARCAL 30 | +0.0538 | 8/8 | +0.0134 | 3/3 | |

**Increment of the shipped arm over today's planned asnp + D1c: +0.0256 full, +0.0146 AoA 15-20.**

f = 1.29 (UPPER END ONLY for VARCAL, see the warning in 3): D1c30 +0.0307 (7/8) / +0.0068; VARCAL30 +0.1107 (8/8) / +0.0724 (3/3);
D1c30+VARCAL30 +0.1407 (8/8) / +0.0790 (3/3); FAMW20+D1c30+VARCAL30 +0.1507 (8/8) / +0.0834 (3/3).

### 8.2 CPU probe, OUT of distribution (AoA-20 holdout members `screen/aoa/aoa20_ffnosim_s1,s2` = family A, `aoa20_fno_s1` = family B;
all 16 AoA-20 trajectories, 643 windows; band r35 (not refitted, paired); asym R1 baseline; f = 1.0)

| arm | delta composite | traj > 0 |
|---|---|---|
| D1c30 | +0.1864 | 16/16 |
| D1c10 / D1c empirical-Bayes | +0.2433 / +0.2625 | 16/16 |
| FAMW 8 | +0.0627 | 16/16 |
| VARCAL 30 | +0.0920 (rel -0.042, tke +0.696, sps +0.168) | 16/16 |
| D1c30 + VARCAL 30 | +0.2778 | 16/16 |
| FAMW 20 + D1c30 + VARCAL 30 | +0.3052 | 16/16 |
| FAMW 8 + D1c-EB + VARCAL 30 | +0.3609 | 16/16 |

Caveat: family B is an FNO stand-in (no AoA-holdout ViT exists), and the members are 3, not 6. The team's +0.38 D1c read used symmetric R1;
on asym R1 I read +0.19, consistent with the report's "part of it is already taken by asym R1".
**VARCAL's OOD increment over D1c: +0.091. Decision-phase candidates: FAMW and the empirical-Bayes D1c shrink (+0.03 and +0.06..0.08 more).**

### 8.3 GPU, the REAL code path (job 7566662, A100, twin packages `$S/pkg/c/*`, fp16, graph CAPTURED, `--causal-check` OK on all four)

| package | rel | tke | mvpe | sps | local ms/step | delta composite (full) | AoA 15-20 |
|---|---|---|---|---|---|---|---|
| base (asym R1) | 96.383 | 81.481 | 96.815 | 55.594 | 3.017 | 0 | 0 |
| d1c | 96.403 | 81.481 | 96.869 | 55.648 | 3.017 | +0.0280 | +0.0070 |
| d1c + VARCAL 30 | 96.351 | 81.813 | 96.869 | 55.697 | 3.056 | +0.0489 (**+0.0209 over d1c**) | +0.0126 (**+0.0056 over d1c**) |
| stack (+ FAMW 20) | 96.353 | 81.820 | 96.876 | 55.703 | 3.179 | +0.0527 (+0.0038 over d1c_vc) | +0.0083 (**-0.0043** over d1c_vc) |

The real path reproduces the probe (VARCAL30 increment: probe +0.0215, GPU +0.0209). Latency (separate processes, NOT interleaved; treat as
a bracket): VARCAL +0.04 ms/step, FAMW +0.12 ms/step. `VC_N0 = 50` on the real path: job **rp_f27c_smoke2** (log `$S/logs/rp_f27c_smoke2_*.log`,
package `$S/pkg/c/d1c_vc50`), two interleaved rounds of d1c vs d1c_vc50 for the latency delta.

### 8.4 Prediction (registered before any upload)

Delta T2 `final` of `asnp + D1c30 + VARCAL50` over `asnp + D1c30`: **+0.015, bracket [-0.01, +0.08]**.
Subscore deltas vs asnp + D1c: rel -0.03 [-0.08, 0] · tke +0.25 [+0.05, +0.8] · mvpe 0 · sps +0.04 [-0.02, +0.15] · time -0.04 [-0.10, -0.02]
(+0.03..0.07 board ms). Centre = the AoA 15-20 increment (+0.0146) with the full-split read (+0.0256) as the in-distribution optimistic end,
minus ~0.004 of latency; the upper end is the OOD increment (+0.09) if the board behaves like an unseen condition.
**This does not reach the top-10 cut by itself (gap ~0.17).** The trade is real: VARCAL spends `rel` (-0.02..-0.05 score) to buy `tke`.

### 8.5 Jobs
7566571 rp_f27c_dump (done) · 7566631 rp_f27c_dumpood (done) · 7566662 rp_f27c_smoke (done) · 7566742 rp_f27c_smoke2 (submitted 16:34, ~6 min).
No training job.

### 8.6 What I did NOT verify (executor: do these first)

* **Knob-off bit-identity** of the patched `submission.py` against 55f0e88 (kill criterion 2). By construction every new statement is behind
  `self.vc` / `self.fw`, and the patched `base` / `d1c` packages reproduce the team's D1c deltas, but no `pkg_identity.py --tol 0` run was made.
* **`validate_t2.py --adaptive`** on a real built archive with `VC_N0 = 50` (only `stream_eval.py --causal-check` was run, on twin packages).
* **Selection of N0 is on the same 8 trajectories it is read on** (no leave-one-trajectory-out refit was run). The arms 30 / 50 / 100 are all
  8/8 and within 0.004 of each other, so the selection bias is bounded by about that; N0 = 10 falls off a cliff (3/8), so do not go below 30.
* **Latency is from separate processes**, not interleaved arms; job 7566742 interleaves d1c and d1c_vc50 twice for a better delta.
* The OOD table uses a 3-member stand-in ensemble with an un-refitted band; it is a sign and an order of magnitude, not a size.
* The shipped all-data members were never run here; twin family weight is 2/5 on the GPU path and 1/3 in the CPU probe (package: 2/6).

### 8.7 GPU, real code path, `VC_N0 = 50` (job 7566742, read 16:38 EDT; AoA 15-20 line and round 2 of d1c_vc50 still to land in the log)

| package | rel | tke | mvpe | sps | local ms/step | causal-check |
|---|---|---|---|---|---|---|
| d1c (round 1 and round 2 identical) | 96.403 | 81.481 | 96.869 | 55.648 | 3.030 | OK |
| d1c + VARCAL 50 (round 1) | 96.366 | 81.773 | 96.869 | 55.701 | 3.072 | OK |

Increment over d1c, full split: 0.4778(-0.037) + 0.1010(+0.292) + 0.2498(+0.053) = **+0.0251** (CPU probe: +0.0256). Latency **+0.042 ms/step** local.
The AoA 15-20 increment on the real path is NOT read yet for N0 50 (probe: +0.0146); read it from the log before shipping (kill criterion 3).
