# Result A (executor EA) — ffvit cross-stitched F-FNO x ViT

**Updated 17:47 EDT. STATUS: FINAL VERDICT = KILL. No row of {Y, X, Y_CTL} x {wave A, wave L} comes within 0.2 of the +0.20 GO bar; best is Y (wave L) at -0.004 on AoA 15-20. No package built. All wave-L candref reads are in and agree: best T2 read (asym R1, AoA15-20) Y_L -0.000, asym+D1c -0.001; static (T1-equivalent) +0.024 — all noise. My GPU use ~1.3 GPU-h.**

## Verdict (design_a.md §6, applied literally)
Rule 1 candidate = best of {Y, X, Y_CTL} x {A, L} on ens_v9 AoA 15-20 final: **Y wave L, 79.423 vs BASE 79.427 = -0.004** (full split 81.867 vs 81.873 = -0.006; sps 38.58 vs 38.53 +0.05).
Rule 2 needs >= +0.20 (unfused: +0.35) -> **FAIL by ~0.2**. Rule 3: arm - control = +0.025 (L) / +0.013 (A), |d| < 0.05 -> the cross-stitch is not a cause of anything held-out; control fails rule 2 too (-0.029 / -0.030).
Rule 4 divergence check: every job's last logged `field` is below its epoch-0 value (no divergence) — the fine-tune fits the TRAINING set (wave L field -15 % arm, -9 % control) and does not transfer.
Z (BASE + 2 hybrid twins, 7 members, not a rule-1 candidate): +0.004 (A) / +0.017 (L) on AoA 15-20, +0.004 / +0.015 full — noise level, and it would cost +1-2 invocations.
Rule 5: no OOD evidence (twins not trained with an AoA held out).
**Why it failed:** the parents were already converged; 50-62 more epochs of the same recipe (joint or cross-stitched) raised held-out error of every single hybrid (compare_v9 AoA15-20: untrained merge relL2 0.1036 -> fine-tuned 0.1059) while training loss fell 7-15 %. The step-0 structure Y_0 was level with BASE (-0.010) and nothing afterwards moved it up.

## Wave L held-out read (evalL job 7566820, ens_v9 alpha 0.80, f 1.39, twins only)
| set | AoA15-20 relL2 | TKE | MVPE | sps | final | d vs BASE | FULL relL2 | TKE | MVPE | sps | final | d vs BASE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE (5 twins) | 0.1009 | 0.4404 | 0.0880 | 38.53 | 79.427 | — | 0.0750 | 0.4546 | 0.0658 | 44.99 | 81.873 | — |
| Y_0 (untrained merge) | 0.1010 | 0.4400 | 0.0881 | 38.50 | 79.417 | -0.010 | 0.0752 | 0.4545 | 0.0660 | 44.96 | 81.859 | -0.014 |
| **Y_L (arm)** | 0.1013 | 0.4408 | 0.0884 | 38.58 | 79.423 | **-0.004** | 0.0755 | 0.4566 | 0.0665 | 45.07 | 81.867 | -0.006 |
| Y_CTL_L | 0.1015 | 0.4408 | 0.0887 | 38.51 | 79.398 | -0.029 | 0.0755 | 0.4568 | 0.0665 | 45.01 | 81.850 | -0.023 |
| X_L (arm) | 0.1034 | 0.4488 | 0.0903 | 38.24 | 79.232 | -0.195 | 0.0772 | 0.4696 | 0.0689 | 44.73 | 81.664 | -0.209 |
| Z_L (BASE + ffvitL_t1,t2) | 0.1009 | 0.4403 | 0.0879 | 38.60 | 79.444 | +0.017 | 0.0751 | 0.4556 | 0.0660 | 45.08 | 81.888 | +0.015 |

## Wave L paired candref reads (vs baseline 7566633; stream at 1.00x, final excludes time)

### rp_f27a_candref_LY_7566824.log  (delta = cand - baseline; errors as relative %, sps/final absolute)
| read | d rel err | d tke err | d mvpe err | d sps | d final | cand final | base final |
|---|---|---|---|---|---|---|---|
| ens_v9 twins AoA15-20 f=1.39 | +0.40% | +0.09% | +0.45% | +0.05 | -0.004 | 79.423 | 79.427 |
| ens_v9 twins full f=1.39 | +0.67% | +0.44% | +1.06% | +0.08 | -0.006 | 81.867 | 81.873 |
| ens_v9 twins AoA15-20 f=1.29 | +0.40% | +0.09% | +0.45% | +0.04 | -0.004 | 80.238 | 80.242 |
| ens_v9 twins full f=1.29 | +0.67% | +0.44% | +1.06% | +0.07 | -0.007 | 82.607 | 82.614 |
| stream twin static all | +0.53% | +0.46% | +1.06% | +0.20 | +0.030 | 76.713 | 76.683 |
| stream twin static aoa15-20 | +0.40% | +0.02% | +0.45% | +0.14 | +0.024 | 74.562 | 74.539 |
| stream twin asym all | +0.53% | +0.46% | +1.06% | +0.05 | -0.008 | 77.077 | 77.085 |
| stream twin asym aoa15-20 | +0.40% | +0.02% | +0.45% | +0.04 | -0.000 | 74.694 | 74.694 |
| stream twin asym_d1c all | +0.54% | +0.46% | +0.77% | +0.05 | -0.006 | 77.107 | 77.113 |
| stream twin asym_d1c aoa15-20 | +0.40% | +0.02% | +0.23% | +0.04 | -0.001 | 74.701 | 74.702 |

### rp_f27a_candref_LYC_7566825.log  (delta = cand - baseline; errors as relative %, sps/final absolute)
| read | d rel err | d tke err | d mvpe err | d sps | d final | cand final | base final |
|---|---|---|---|---|---|---|---|
| ens_v9 twins AoA15-20 f=1.39 | +0.59% | +0.09% | +0.80% | -0.02 | -0.029 | 79.398 | 79.427 |
| ens_v9 twins full f=1.39 | +0.67% | +0.48% | +1.06% | +0.02 | -0.023 | 81.850 | 81.873 |
| ens_v9 twins AoA15-20 f=1.29 | +0.59% | +0.09% | +0.80% | -0.03 | -0.028 | 80.214 | 80.242 |
| ens_v9 twins full f=1.29 | +0.67% | +0.48% | +1.06% | +0.01 | -0.022 | 82.592 | 82.614 |
| stream twin static all | +0.53% | +0.48% | +1.06% | +0.12 | +0.010 | 76.693 | 76.683 |
| stream twin static aoa15-20 | +0.59% | +0.05% | +0.80% | +0.06 | -0.002 | 74.537 | 74.539 |
| stream twin asym all | +0.53% | +0.48% | +1.06% | +0.01 | -0.018 | 77.067 | 77.085 |
| stream twin asym aoa15-20 | +0.59% | +0.05% | +0.80% | -0.01 | -0.019 | 74.675 | 74.694 |
| stream twin asym_d1c all | +0.54% | +0.48% | +0.93% | +0.01 | -0.016 | 77.097 | 77.113 |
| stream twin asym_d1c aoa15-20 | +0.60% | +0.05% | +0.57% | -0.01 | -0.018 | 74.684 | 74.702 |

### rp_f27a_candref_LX_7566827.log  (delta = cand - baseline; errors as relative %, sps/final absolute)
| read | d rel err | d tke err | d mvpe err | d sps | d final | cand final | base final |
|---|---|---|---|---|---|---|---|
| ens_v9 twins AoA15-20 f=1.39 | +2.48% | +1.91% | +2.61% | -0.29 | -0.195 | 79.232 | 79.427 |
| ens_v9 twins full f=1.39 | +2.93% | +3.30% | +4.71% | -0.26 | -0.209 | 81.664 | 81.873 |
| ens_v9 twins AoA15-20 f=1.29 | +2.48% | +1.91% | +2.61% | -0.33 | -0.193 | 80.049 | 80.242 |
| ens_v9 twins full f=1.29 | +2.93% | +3.30% | +4.71% | -0.27 | -0.204 | 82.410 | 82.614 |
| stream twin static all | +2.93% | +3.30% | +4.71% | -0.08 | -0.132 | 76.551 | 76.683 |
| stream twin static aoa15-20 | +2.58% | +1.84% | +2.61% | -0.20 | -0.141 | 74.398 | 74.539 |
| stream twin asym all | +2.93% | +3.30% | +4.71% | -0.31 | -0.189 | 76.896 | 77.085 |
| stream twin asym aoa15-20 | +2.58% | +1.84% | +2.61% | -0.31 | -0.169 | 74.525 | 74.694 |
| stream twin asym_d1c all | +2.55% | +3.30% | +3.25% | -0.27 | -0.169 | 76.943 | 77.113 |
| stream twin asym_d1c aoa15-20 | +2.38% | +1.84% | +1.83% | -0.30 | -0.161 | 74.541 | 74.702 |


## Wave A held-out read (eval job 7566819, ens_v9 alpha 0.80, f 1.39, twins only)
| set | AoA15-20 relL2 | TKE | MVPE | sps | final | d vs BASE | FULL relL2 | TKE | MVPE | sps | final | d vs BASE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE (5 twins) | 0.1009 | 0.4404 | 0.0880 | 38.53 | 79.427 | — | 0.0750 | 0.4546 | 0.0658 | 44.99 | 81.873 | — |
| Y_0 (untrained merge) | 0.1010 | 0.4400 | 0.0881 | 38.50 | 79.417 | -0.010 | 0.0752 | 0.4545 | 0.0660 | 44.96 | 81.859 | -0.014 |
| **Y (arm)** | 0.1014 | 0.4409 | 0.0885 | 38.54 | 79.410 | **-0.017** | 0.0755 | 0.4568 | 0.0666 | 45.05 | 81.859 | -0.014 |
| Y_CTL | 0.1015 | 0.4406 | 0.0887 | 38.51 | 79.397 | -0.030 | 0.0755 | 0.4569 | 0.0665 | 45.00 | 81.849 | -0.024 |
| X_0 | 0.1021 | 0.4436 | 0.0887 | 38.24 | 79.300 | -0.127 | 0.0762 | 0.4599 | 0.0673 | 44.68 | 81.730 | -0.143 |
| X (arm) | 0.1036 | 0.4498 | 0.0904 | 38.19 | 79.213 | -0.214 | 0.0772 | 0.4698 | 0.0689 | 44.71 | 81.659 | -0.214 |
| X_CTL | 0.1037 | 0.4501 | 0.0905 | 38.12 | 79.191 | -0.236 | 0.0771 | 0.4703 | 0.0685 | 44.67 | 81.653 | -0.220 |
| Z (BASE + ffvit_t1,t2) | 0.1011 | 0.4405 | 0.0880 | 38.57 | 79.431 | +0.004 | | | | | | |

Readings: (1) joint fine-tune (Y_CTL - Y_0) = -0.020 AoA15-20: 50 more epochs made the members WORSE held-out although their
training loss fell 7-10 % — the parents were already converged; (2) cross-stitch (Y - Y_CTL) = +0.013 / +0.010 (|d| < 0.05,
inside noise) although it cut training field loss ~3 % vs control — it fits the training set, not the held-out one;
(3) X (ViT-heavy) is -0.21. No row is within 0.2 of the GO bar.

## Wave A paired candref reads (rp_f27x_candref.sh vs baseline job 7566633; stream = twin T2 package, 1.00x, final excludes time)

### rp_f27a_candref_aY_7566821.log  (delta = cand - baseline; errors as relative %, sps/final absolute)
| read | d rel err | d tke err | d mvpe err | d sps | d final | cand final | base final |
|---|---|---|---|---|---|---|---|
| ens_v9 twins AoA15-20 f=1.39 | +0.50% | +0.11% | +0.57% | +0.01 | -0.017 | 79.410 | 79.427 |
| ens_v9 twins full f=1.39 | +0.67% | +0.48% | +1.22% | +0.06 | -0.014 | 81.859 | 81.873 |
| ens_v9 twins AoA15-20 f=1.29 | +0.50% | +0.11% | +0.57% | +0.00 | -0.017 | 80.225 | 80.242 |
| ens_v9 twins full f=1.29 | +0.67% | +0.48% | +1.22% | +0.05 | -0.013 | 82.601 | 82.614 |
| stream twin static all | +0.53% | +0.51% | +1.22% | +0.15 | +0.017 | 76.700 | 76.683 |
| stream twin static aoa15-20 | +0.50% | +0.07% | +0.57% | +0.10 | +0.010 | 74.549 | 74.539 |
| stream twin asym all | +0.53% | +0.51% | +1.22% | +0.03 | -0.014 | 77.072 | 77.085 |
| stream twin asym aoa15-20 | +0.50% | +0.07% | +0.57% | +0.02 | -0.008 | 74.686 | 74.694 |
| stream twin asym_d1c all | +0.54% | +0.51% | +0.93% | +0.03 | -0.010 | 77.102 | 77.113 |
| stream twin asym_d1c aoa15-20 | +0.50% | +0.07% | +0.34% | +0.02 | -0.008 | 74.693 | 74.702 |

### rp_f27a_candref_aYC_7566822.log  (delta = cand - baseline; errors as relative %, sps/final absolute)
| read | d rel err | d tke err | d mvpe err | d sps | d final | cand final | base final |
|---|---|---|---|---|---|---|---|
| ens_v9 twins AoA15-20 f=1.39 | +0.59% | +0.05% | +0.80% | -0.02 | -0.030 | 79.397 | 79.427 |
| ens_v9 twins full f=1.39 | +0.67% | +0.51% | +1.06% | +0.01 | -0.024 | 81.849 | 81.873 |
| ens_v9 twins AoA15-20 f=1.29 | +0.59% | +0.05% | +0.80% | -0.04 | -0.029 | 80.213 | 80.242 |
| ens_v9 twins full f=1.29 | +0.67% | +0.51% | +1.06% | +0.00 | -0.024 | 82.590 | 82.614 |
| stream twin static all | +0.53% | +0.51% | +0.91% | +0.10 | +0.005 | 76.689 | 76.683 |
| stream twin static aoa15-20 | +0.59% | -0.02% | +0.80% | +0.05 | -0.002 | 74.536 | 74.539 |
| stream twin asym all | +0.53% | +0.51% | +0.91% | +0.01 | -0.019 | 77.067 | 77.085 |
| stream twin asym aoa15-20 | +0.59% | -0.02% | +0.80% | -0.00 | -0.016 | 74.678 | 74.694 |
| stream twin asym_d1c all | +0.54% | +0.51% | +0.77% | +0.01 | -0.016 | 77.097 | 77.113 |
| stream twin asym_d1c aoa15-20 | +0.60% | -0.02% | +0.57% | -0.00 | -0.015 | 74.686 | 74.702 |

### rp_f27a_candref_aX_7566823.log  (delta = cand - baseline; errors as relative %, sps/final absolute)
| read | d rel err | d tke err | d mvpe err | d sps | d final | cand final | base final |
|---|---|---|---|---|---|---|---|
| ens_v9 twins AoA15-20 f=1.39 | +2.68% | +2.13% | +2.73% | -0.34 | -0.214 | 79.213 | 79.427 |
| ens_v9 twins full f=1.39 | +2.93% | +3.34% | +4.71% | -0.28 | -0.214 | 81.659 | 81.873 |
| ens_v9 twins AoA15-20 f=1.29 | +2.68% | +2.13% | +2.73% | -0.37 | -0.211 | 80.031 | 80.242 |
| ens_v9 twins full f=1.29 | +2.93% | +3.34% | +4.71% | -0.30 | -0.209 | 82.405 | 82.614 |
| stream twin static all | +2.80% | +3.37% | +4.86% | -0.12 | -0.142 | 76.542 | 76.683 |
| stream twin static aoa15-20 | +2.68% | +2.11% | +2.73% | -0.24 | -0.159 | 74.380 | 74.539 |
| stream twin asym all | +2.80% | +3.37% | +4.86% | -0.32 | -0.192 | 76.893 | 77.085 |
| stream twin asym aoa15-20 | +2.68% | +2.11% | +2.73% | -0.34 | -0.184 | 74.510 | 74.694 |
| stream twin asym_d1c all | +2.55% | +3.37% | +3.41% | -0.29 | -0.173 | 76.940 | 77.113 |
| stream twin asym_d1c aoa15-20 | +2.48% | +2.11% | +1.83% | -0.33 | -0.174 | 74.527 | 74.702 |

Singles (compare_v9 AoA15-20, 1.39): untrained merge ffvit0Z_t1 relL2 0.1036 -> fine-tuned ffvit_t1 0.1059 / ffvit0_t1 0.1059 (+2.2 %); t2 0.1048 both; t3 0.1059/0.1060. Arm == control per member.

## Training monitor (16:55)
* All 23 training jobs RUNNING, none failed (the CANCELLED rows in sacct are the design agent's first wave L, 7566636-46).
* Training `field` loss, arm vs control at equal epoch — wave A: t1 ep30 0.03555 vs 0.03657 (-2.8 %), t2 ep30 0.03816 vs 0.03941 (-3.2 %).
  Wave L (lr 2e-4, at peak LR now): t1 ep20 0.03834 vs ctl 0.03961 (-3.2 %); ctl t1 ep20 is ABOVE its ep0 0.03916 (LR peak; the §6 kill
  rule reads the LAST logged value — will re-check at the end). The cross-stitch clearly moves its own training loss.

### earlier (16:35)
* Wave A (50 ep, lr 1e-4), start 16:19: arm twins 63.5 s/ep -> ~17:12; members 71 s/ep -> ~17:18; controls ~58-64 s/ep. Limit 17:49.
* Wave L (62 ep, lr 2e-4; re-submitted by the design agent 16:27 as 7566669-79): twins ~63 s/ep -> ~17:33; members ~70 s/ep -> ~17:40. Limit 17:57.
* Arm vs control, training loss at equal epoch (epoch 10, wave A): field t1 0.03685 vs 0.03735, t2 0.04007 vs 0.04032, t3 0.03679 vs 0.03721 (arm -0.6..-1.3 %); total loss arm lower on all three. Arm moves its own training loss. No job failed.

## Integration (§7)
1. **Packed round trip** (smoke.pth, 96 val_unseen windows, job 7566718): int8 tensor + fp16 pack 73.5 -> 19.7 MB; field rel diff packed vs fp32 **1.8e-3** (spec expected <1e-3; ViT Linears with per-tensor int8 scale), fp16 autocast 1.0e-3, both 2.1e-3; model rel-L2 vs target 0.08101 / 0.08098 / 0.08100 — i.e. no score cost. Finite. Batch-1 vs batched 9.6e-7.
2. **Rehearsal build, row Y with stand-ins** (smoke.pth under ffvit_m1/m2/t1/t2 names, real ffno_all s3/s4, p2 s3; job 7566718, worktree `$S/wt_a_bld`): T1 validate_strict **SUBMITTABLE** (43 MB extracted, members 4, amp float16, lazy small graph); T2 in-build **SUBMITTABLE**, `--adaptive` **SUBMITTABLE**, `graph CAPTURED`, members 4; +D1c variant `--adaptive` SUBMITTABLE (stateless FAIL expected); causal-check ran (rep line printed; OK line re-checked in 7566745).
3. **Fusion `StackedFFViT`** — DONE, installed in `wt_a/Track_1/submit_fno/_fno.py` (switch `FUSE_FFVIT = True`; design agent's original saved as `$S/pkg/a/_fno_designA_orig.py`). Verified vs the member loop, K=2 distinct members, arm and control, GPU: fp32 field max|d| 8-10e-7, logstd 6-8e-6 (fft and dense); predict_fno end to end 3e-7; fp16 autocast fused vs fp32 loop 3.6e-4.

## Latency — interleaved, same node, same job, 3 rounds (shared node, RPDE_FORCE; rough)
| package | T1 ms/window (336 win, median of 5) | T2 ms/step (336 steps, median) |
|---|---|---|
| baseline fallback (x27lz / x27asnp) | 1.474-1.476 | 2.870-2.877 |
| Y UNFUSED (3 invocations T1) | 1.532-1.537 (**+0.06, +4 %**) | 4.527-4.546 (**+1.66, +58 %**) |
| Y FUSED (StackedFFViT, 2 invocations T1) (job 7566745, other node; its baseline read 1.487-1.490 / 2.872-2.877) | 1.662-1.669 (**+0.18, +12 %**) | 5.073-5.078 (**+2.20, +77 %**) |
T2 unfused is a large local cost (batch-1 is launch-bound; the hybrids add 2 invocations of ~16 layers each).

Fused causal check: `CAUSALITY OK: 176 steps ... max |diff| 0.0e+00`. Eager micro-bench (job 7566777, fp16, smoke weights): B=112 dense
unfused 2 hybrids 76.0 ms vs fused 90.8 ms (links as grouped fp16 convs cost ~11.5 ms; even StackedFFNO+StackedViT without links is
79.3 vs 67.6 separate); B=1 fft eager fused 15.5 vs 23.5 ms (launch-bound, but T2 runs inside the CUDA graph where the package read
above has unfused faster). **Choice: unfused (`FUSE_FFVIT = False` baked at build) unless the board invocation tax argues otherwise:**
R23 measured +1.151 board ms per extra T1 invocation with identical arithmetic; local timing cannot see that tax, so for T1 the
bracket is unfused [+0.2, +1.3] board ms vs fused [+0.4, +0.6] (fused = +0.12 local ms/window x ~3.2 board/local, R24). I will
build T1 FUSED and T2 UNFUSED if GO (FUSE_FFVIT is one constant in the shared _fno.py -> set per zip after the build is not allowed
(never hand-edit a zip), so two builds with different FUSE_FFVIT).

## Jobs queued (afterok on training)
eval (wave A) 7566819, evalL 7566820; candref aY 7566821, aYC 7566822, aX 7566823, LY 7566824, LYC 7566825, LX 7566827.
Table tool: `$S/jobs/rp_f27a_table.py <baseline candref log 7566633> <cand logs...>`. Stream reads are at 1.00x (candref), ens_v9 at 1.39/1.29.

## Leftovers (for the commander)
* `wt_a/Track_1/submit_fno/_fno.py` now carries the verified `StackedFFViT` + `FUSE_FFVIT = True` (design agent's version: `$S/pkg/a/_fno_designA_orig.py`).
* `$S/wt_a_bld` = an extra git worktree (detached @55f0e88) used only for rehearsal builds; `$S/pkg/a/reh/` holds STAND-IN rehearsal zips (smoke weights) — NOT shippable.
* Tools: `$S/jobs/rp_f27a_build.sh` (build+gates+interleaved timing, env A_LIST/M_LIST/TAG/FUSE/TIMING/D1C), `rp_f27a_time.py`, `rp_f27a_table.py`.
