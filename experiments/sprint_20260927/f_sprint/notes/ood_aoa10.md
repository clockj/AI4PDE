# AoA-10 holdout (interpolation regime) — read of the T2 stream stack (executor EA)

**Updated 17:57 EDT. STATUS: DONE. In the interpolation regime the stream stack's gain is at (above) the AoA-20 OOD level, not the in-distribution level: C2 +0.521 vs asym R1 (18/18 trajectories), +0.275 over asym R1 + D1c30; first 14 windows +0.372 / +0.248. B2 adds +0.03 on top of either (18/18). Nothing tuned on this dump.**

## 1. Dump
`$S/ckpt/a/ood_aoa10.npz` (job rp_f27a_ood10 7567578, debug-nextgen, 1 min). `dump_twins.py --split aoa_10` copied unchanged from wt_c
(same construction as the AoA-20 dump 7566631): all 18 AoA-10 real trajectories, 42 windows each (756), streams in time order.
Members (all four from `screen/aoa/`, configs checked: every one excludes all 18 `*_10` trajectories plus the 8 val trajectories,
`--sim-exclude-held` where sim is used, no `--all-data`, norm real, 400 epochs):
| idx | member | arch | sim | per-member rel (model space) |
|---|---|---|---|---|
| 0 | aoa10_ffnosim_s1 | F-FNO w40 L8 | yes | 0.252 |
| 1 | aoa10_ffno_s1 | F-FNO w40 L8 | no | 0.255 |
| 2 | aoa10_fno_s1 | FNO w96 L8 | yes | 0.258 |
| 3 | aoa10_nosim_s1 | FNO w96 L8 | no | 0.259 |
**Family A = {0,1} (the two F-FNOs), family B = {2,3} (the two FNOs)**, weight 2/3 : 1/3 as in the probe — the F-FNO/other-family split
of the shipped ensemble, and the largest ensemble available (4 members vs AoA-20's 3). Sensitivity: the sim-only split A={0}, B={2}
(mirrors AoA-20's all-sim members) gives the same picture (§3, last line). FAMW is off in every arm.

## 2. Instrument
Copies of design C's `probe2.py` + arm file (`$S/ckpt/a/p2/stream_c/`, only change: a `--regime ood10` loader) and of wt_c2's
`submission.py` (for `_CausalBandScalerT`); band `r35_centre0_fill.npz` (probe2's default — the AoA-20 baseline it gives, 72.939,
equals EC's gauge). B2 = design B's `b2lab.stream` (copied) replayed on the probe2 prediction stream of the chosen arm, variant
`V(tsm=3, n0e=20, blk=4)` vs `R1ship`; its R1ship sps reproduces probe2's to 3 decimals (51.839 D1c30, 52.354 C2), so the two probes
combine exactly: B2 delta composite = 0.2498 x delta sps (it moves only the bounds). Jobs 7567588-93, logs `$S/logs/rp_f27a_p10_*.log`.

## 3. Results — delta composite (no time) and per-subscore deltas, (n/18) = trajectories with a positive delta

**f = 1.00, full streams (42 windows)**
| arm | vs asym R1 | [rel / tke / mvpe / sps] | vs asym R1 + D1c30 | [rel / tke / mvpe / sps] |
|---|---|---|---|---|
| D1c30 (today's planned) | +0.247 (18/18) | +0.141 / +0.000 / +0.631 / +0.485 | 0 | |
| D1c30 + VC50 = T2_CONS | +0.333 (18/18) | +0.112 / +0.596 / +0.631 / +0.645 | +0.086 (18/18) | -0.029 / +0.596 / 0 / +0.159 |
| D1c10 + VC50 = T2_AGGR | +0.421 (18/18) | +0.163 / +0.596 / +0.905 / +0.799 | +0.174 (18/18) | +0.022 / +0.596 / +0.274 / +0.313 |
| **C2** | **+0.521 (18/18)** | +0.209 / +0.762 / +1.028 / +1.001 | **+0.275 (18/18)** | +0.068 / +0.762 / +0.397 / +0.515 |
| B2 on D1c30 | +0.284 (= +0.247 +0.037) | sps +0.146 | +0.037 (18/18) | sps +0.146 |
| **C2 + B2 (T2_P2)** | **+0.553** (= +0.521 +0.031) | | **+0.306** | B2 part: +0.031 (18/18) |

**f = 1.00, first 14 windows of each stream**
| arm | vs asym R1 | vs asym R1 + D1c30 |
|---|---|---|
| D1c30 | +0.124 (18/18) | 0 |
| T2_CONS | +0.173 (18/18) | +0.048 (18/18) |
| T2_AGGR | +0.266 (18/18) | +0.142 (18/18) |
| **C2** | **+0.372 (18/18)** [rel +0.157 tke +0.498 mvpe +0.765 sps +0.707] | **+0.248 (18/18)** |
| B2 on D1c30 / on C2 | +0.031 (18/18) / +0.024 (18/18) on top | |
| **C2 + B2** | **+0.396** | **+0.272** |

Gain by revealed windows n (C2 vs asym R1, full): n 1-3 +0.152, 4-7 +0.394, 8-13 +0.530, 14-27 +0.586, 28+ +0.606.

**f = 1.29 (UPPER END for VARCAL arms — design C §5.3 bias; do not bank):** C2 +0.637 vs asym R1, +0.355 vs D1c30; CONS +0.432 / +0.150;
AGGR +0.532 / +0.249; D1c30 +0.282. All 18/18.
**Family-split sensitivity (A = ffnosim, B = fno-sim only, f 1.00):** C2 +0.479 vs asym R1, +0.252 vs D1c30; CONS +0.308; AGGR +0.388.

## 4. Gauge row (asym R1 baseline, no D1c) next to EC's table (result_c.md §B)
| regime | rel | tke | mvpe | sps | rel_e | tke_e | mvpe_e | cov | comp | mvpe/rel |
|---|---|---|---|---|---|---|---|---|---|---|
| in-dist full f1.00 | 96.385 | 81.484 | 96.818 | 55.586 | 0.0750 | 0.4545 | 0.0657 | 0.905 | 77.085 | 0.88 |
| in-dist AoA15-20 f1.00 | 95.197 | 81.942 | 95.779 | 48.459 | 0.1009 | 0.4408 | 0.0881 | 0.861 | 74.688 | 0.87 |
| OOD AoA-20 f1.00 | 94.286 | 79.843 | 94.247 | 44.616 | 0.1212 | 0.5049 | 0.1221 | 0.843 | 72.939 | 1.01 |
| **HOLD AoA-10 f1.00** | 95.803 | 79.904 | 95.161 | 51.353 | 0.0876 | 0.5030 | 0.1017 | 0.886 | 75.437 | **1.16** |
| HOLD AoA-10 f1.29 | 94.662 | 76.709 | 93.911 | 45.640 | 0.1128 | 0.6072 | 0.1297 | 0.853 | 73.027 | 1.15 |
| BOARD T2 09-25 | 94.771 | 79.516 | 94.132 | 43.556 | 0.1104 | 0.5152 | 0.1247 | — | 72.863 | 1.13 |

Where the board sits relative to AoA-10 holdout (f 1.00): rel error board 1.26x higher (0.110 vs 0.088 — AoA 10 is a mid-difficulty
angle; the board mixes all AoAs incl. 15-20); tke error board 1.02x (≈ same level); mvpe error board 1.23x; sps board 7.8 lower;
composite board 2.6 lower. **The error PATTERN matches best of all regimes**: mvpe/rel 1.16 vs board 1.13 (in-dist 0.87, AoA-20 1.01), and
tke/rel likewise (AoA-10 5.7, board 4.7, in-dist 6.1, AoA-20 4.2). At f 1.29 the AoA-10 LEVEL almost equals the board on every
error subscore (rel 0.113 vs 0.110, mvpe 0.130 vs 0.125, tke 0.607 vs 0.515 high) and on composite (73.03 vs 72.86), sps 45.6 vs 43.6.

## 5. Answer
**Closer to the AoA-20 read — in fact at or above it.** In the interpolation regime every arm's gain is OOD-sized: C2 +0.52 vs asym R1
(AoA-20 +0.40, in-dist full +0.06, in-dist AoA 15-20 +0.01); over asym R1 + D1c30 +0.27; 14-window streams keep ~70 % of it
(+0.37 / +0.25). Every arm is positive on 18/18 trajectories, and the ordering C2 > AGGR > CONS > D1c30 is the AoA-20 ordering.
What drives it: a whole flow condition (all of AoA 10) was unseen in training, so each trajectory carries a persistent mean-field
bias (D1c alone +0.25, mvpe +0.63) and a mis-scaled amplitude (VARCAL: tke +0.60) — the same two defects as AoA-20, with the board's
error pattern.
**Caveats (not tuned, but read honestly):** (i) weaker stand-in ensemble (4 single-seed holdout members, 2+2 families; band not
refitted); (ii) this regime removes a WHOLE AoA from training; whether the public hidden set is "unseen condition" (this regime)
or "unseen recording of a seen condition" (in-dist twins) is the open question — the gauge pattern (mvpe/rel) favours the former
but the in-dist AoA-10 trajectories also read 1.04-1.08 (design C2 §1); (iii) CPU probe only, no real-code-path run on this dump
(design C2 showed probe = real path to 0.001 in distribution).
