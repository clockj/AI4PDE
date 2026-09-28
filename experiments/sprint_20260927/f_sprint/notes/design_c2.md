# Design C, phase 2 — "C2": self-tuning, lead-time-resolved bias correction (T2 stream, prediction side)

Author: design agent C, 2026-09-27 16:40–17:35 EDT. Worktree `$S/wt_c2` (S = `/fs/scratch/PAS1187/qwu/tmp/realpde/final0927`).
Patch `$S/notes/c2.patch`. No training. No new fitted table. R1 tuning was handed to agent B and is not touched here.

## 0. Bottom line

**C2 = D1c with (a) an empirical-Bayes shrink that the trajectory tunes itself, (b) a rank-1 lead-time profile, plus VARCAL N0 30.**
It beats the best knob-only stack (`D1c10 + VC30 + FW8`) in every regime I can measure, and it does so WITHOUT family weights
(so without their +0.12 ms/step):

| composite without time, delta vs asym-R1 baseline | in-dist full (8 traj) | in-dist AoA 15-20 (3) | OOD AoA-20 holdout (16) | OOD, 14-window streams | in-dist, 14-window |
|---|---|---|---|---|---|
| CONS = D1c30 + VC50 (EC's T2_CONS) | +0.0517 (8/8) | +0.0202 (3/3) | +0.2677 (16/16) | +0.1369 | +0.0230 / a15 +0.0065 |
| K0 = D1c10 + VC30 | +0.0440 (6/8) | -0.0027 (1/3) | +0.3345 (16/16) | +0.2110 | +0.0146 / -0.0198 |
| KNOB = D1c10 + VC30 + FW8 (best knob-only) | +0.0433 (5/8) | -0.0093 (1/3) | +0.3480 (16/16) | +0.2263 | +0.0105 / -0.0309 |
| **C2 (ship)** | **+0.0622 (8/8)** | **+0.0090 (3/3)** | **+0.4008 (16/16)** | **+0.2616** | **+0.0315 / -0.0003** |
| C2 minus KNOB | +0.019 | +0.018 | **+0.053** | +0.035 | +0.021 / +0.031 |
| C2 minus CONS | +0.011 | -0.011 | **+0.133** | +0.125 | +0.009 / -0.007 |

**Against the commander's bar** (OOD gain at least +0.05 above the best knob-only stack, in-dist AoA 15-20 loss at most 0.03):
met on full-length streams (+0.053, marginal), NOT met on 14-window streams (+0.035). In-dist it is a gain over KNOB, and a loss of 0.011
on AoA 15-20 against CONS. It is the strongest thing I found; it is an incremental improvement of the mean-field correction, not a new lever.

## 1. Is the board OOD-like? (the thesis check you asked for)

| row | rel err | tke err | mvpe err | mvpe/rel | sps | composite (no time) |
|---|---|---|---|---|---|---|
| **board, T2 09-25** (from the subscores) | 0.1103 | 0.5152 | 0.1247 | **1.13** | 43.56 | 72.86 |
| local in-dist, full split | 0.0750 | 0.4545 | 0.0657 | 0.88 | 55.59 | 77.09 |
| local in-dist, AoA 15-20 | 0.1009 | 0.4408 | 0.0881 | 0.87 | 48.46 | 74.69 |
| same, my 1.29x error stretch | 0.1299 | 0.5480 | 0.1138 | 0.88 | 42.47 | 72.12 |
| local OOD (AoA-20 holdout members) | 0.1212 | 0.5049 | 0.1221 | 1.01 | 44.62 | 72.94 |

Honest reading:
* **In favour.** The board's mvpe/rel ratio (1.13) is above the pooled value of BOTH local regimes and above every single local trajectory
  (in-dist per-trajectory 0.54..1.08, OOD 0.85..1.08; `$S/logs/c2/ratio.log`). An error stretch cannot produce it (a stretch keeps the ratio at
  0.88 by construction). The share of window-mean error energy that is trajectory-persistent is 0.04-0.07 on in-dist AoA 15-20 and 0.17-0.34 OOD.
  So the board carries at least as much excess mean-field error as my OOD regime, which is what a persistent bias looks like.
* **Against / caveats.** (i) The composite-level match (72.86 vs 72.94) is partly a coincidence: my OOD dump is AoA 20 only, the hardest angle even
  in distribution (out-of-fold rel 0.121 there), on a weaker 3-member stand-in ensemble, with an un-refitted band. (ii) The ratio distributions
  overlap (in-dist AoA-10 trajectories read 1.04 and 1.08), so a hidden set with a different AoA/Re mix could read 1.13 without being OOD.
  (iii) Board rel is only 1.09x local AoA 15-20, while OOD rel is 1.20x: the board is not as far out as the holdout cell.
* **My number:** the board sits between the regimes; for the mean-field part I would weight lambda = 0.5, bracket [0.2, 0.8]. With that, C2 over
  CONS is worth about 0.5 x 0.133 - 0.5 x 0.011 = +0.06 [+0.02, +0.10] before the usual proxy discount.
* **R1 range (fact for agent B, not tuned by me).** Fraction of pixels whose final argmax sits on the grid's TOP multiplier (2.0): in-dist 0.6 %,
  in-dist at f = 1.29 1.1-1.3 %, OOD 3.2-3.6 %. On the BOTTOM multiplier (0.5): 15-25 % in every regime. The upper end is not binding, even OOD;
  the lower end is. If the grid is moved, it should extend down, not up.

## 2. The construction (exact)

All in the output (target-normalised) space, per trajectory, reset by `reset_ttt_state`. n = revealed windows (0 on the first step).
`C_k` = window k's pre-compose ensemble mean (after FAMW if that is on), `y_k` its revealed target, `e_k = C_k - y_k` (20,H,W,2).

```
raw(p)  = (1/n) sum_k mean_t e_k(t,p)                       per pixel, channel         [D1c's running mean]
s2(p)   = sum_k (mean_t e_k - raw)^2 / (n-1)                 window-to-window scatter
per channel, pooled over all 2048 pixels:
  se2   = mean_p(s2) / n ;  tau2 = max( mean_p(raw^2) - se2, 0 )
  EB    = [n >= 2] * tau2/(tau2 + se2) * n/(n + D1C_EB_N0)                              D1C_EB_N0 = 1
shr     = max( n/(n + D1C_N0), EB )                                                     D1C_N0 = 30 (the floor = today's D1c)
bt(t,p) = (1/n) sum_k e_k(t,p)
a(t)    = clip( 1 + ( <bt(t,.), raw>_p / <raw, raw>_p - 1 ) * n/(n + D1C_LEAD_N0), 0, 3 )    per channel, 20 numbers;  D1C_LEAD_N0 = 5
shift(t,p) = - a(t) * shr * raw(p)          added to P and to the band centre (exactly where D1c's shift goes; R1 scores the moved fields)
```
plus VARCAL as in phase 1 with `VC_N0 = 30`.

Why each piece (OOD increments inside the stack): the lead profile **+0.052** (the bias is small at lead 0, where the forecast is anchored
to the last input frame, and grows with lead time; a(t) is 20 numbers estimated from 2048 pixels, so it is stable from the first windows);
the EB shrink **+0.023** over a fixed N0 = 10 and +0.008 / +0.013 in distribution (it raises the shrink only on trajectories whose bias
stands out of their own scatter: OOD the u channel reads tau2/s2 = 0.25-0.48, i.e. an optimal N0 of 2-4, while OOD v reads 0.03-0.18 and
the in-dist AoA 15-20 trajectories 0.00-0.06, N0 17 and up; `$S/logs/c2/cfit.log`); VARCAL +0.10.

Constants: `D1C_N0 = 30` (unchanged), `D1C_EB_N0 = 1`, `D1C_LEAD_N0 = 5`, `VC_N0 = 30`. Four scalars, nothing fitted per pixel. Neighbours
measured: lead N0 2 / 5 / 12 within 0.007 OOD (read on the prior-variance variant); EB damping 0 / 1 within 0.001 OOD and in-dist; floor 10 / 15 / 20 / 30 within 0.002 OOD.

## 3. Gain as a function of n (revealed windows) — full-length streams

| windows | OOD: KNOB | OOD: **C2** | in-dist: KNOB | in-dist: **C2** |
|---|---|---|---|---|
| 0 (first window) | 0 | 0 | 0 | 0 |
| 1-3 | +0.110 | +0.097 | -0.013 | +0.013 |
| 4-7 | +0.240 | +0.291 | +0.010 | +0.032 |
| 8-13 | +0.314 | +0.369 | +0.024 | +0.046 |
| 14-27 | +0.402 | +0.474 | +0.043 | +0.068 |
| 28+ | +0.424 | +0.476 | +0.076 | +0.087 |

**Early windows: nothing I tried helps windows 1-3.** A fixed N0 = 10 is still the best there OOD (+0.110 vs +0.097). At n = 1 a single error
sample cannot separate bias from scatter; a prior on the scatter (c x sigma_head^2, c = 0.31 / 0.12 fitted on val_unseen) over-corrects and
LOSES in both regimes (windows 1-3: -0.09 in-dist, -0.05..-0.12 OOD). So on 14-window hidden trajectories every correction is worth about 60 %
of its full-length value (table in section 0), and the first window of every trajectory is uncorrected.

## 4. What I tried and closed (all in `$S/logs/c2/a1_*.log`, `a2_*.log`, `a3_*.log`; bias-only unless stated)

| idea | OOD | in-dist full / AoA 15-20 | verdict |
|---|---|---|---|
| D1c30 / D1c10 / D1c3 | +0.186 / +0.243 / +0.250 | +0.026 / +0.006 ; +0.023 / -0.010 ; +0.002 / -0.043 | the knob trade-off |
| EB shrink pooled per channel | +0.258 | +0.028 / +0.000 | kept |
| EB per pixel (phase 1) / regional 9x9 | +0.2625 / +0.261 | +0.023 / -0.012 ; +0.020 / -0.013 | same OOD, worse in-dist |
| **spatial smoothing of the bias, 3x3 / 5x5** | +0.155 / +0.084 (with EB) | +0.020 / +0.015 | **closed: the persistent bias is sharp, not smooth** |
| lead profile rank-1 (on EB) | **+0.305** | +0.033 / -0.002 | kept |
| lead: per-pixel linear ramp (on EB, which alone is +0.258) / per-pixel per-frame (on D1c30, alone +0.186) | +0.282 / +0.213 | +0.025 / -0.011 ; +0.016 / -0.021 | closed: no better than rank-1 OOD, tke and AoA 15-20 fall |
| multiplicative gain of the mean field | +0.108 | +0.018 / +0.004 | closed (additive is better) |
| prior-variance EB (n = 1), inside the stack | +0.393 against +0.401 without it; windows 1-3 -0.05 | +0.054 / -0.009 against +0.061 / +0.007 | closed |
| pooled amplitude factor in VARCAL (delta vs per-pixel only) | -0.035 | -0.061 full | closed |
| FW8 on top of the C2 stack (delta) | +0.003 | -0.003 / -0.006 | closed (and +0.12 ms) |
| floor N0 10 / 15 / 20 under EB (deltas vs floor 30) | -0.001..+0.000 | -0.008 / -0.014 .. -0.001 / -0.002 | floor stays 30 |
| VARCAL 20 / 50 inside C2 (deltas vs VC 30) | +0.002 / -0.012 | -0.007 / -0.011 ; +0.002 / +0.007 | VC 30 for upside; VC 50 is the in-dist-safe alternative |

In-dist at f = 1.29 (biased up for anything with VARCAL, see phase 1): section 5.

## 5. Real code path, latency, remaining numbers

### 5.1 Real code path on GPU (twin packages `$S/pkg/c2/*`, A100, fp16, graph CAPTURED, jobs 7567000 and 7567031)

| package | knobs | rel | tke | mvpe | sps | delta composite vs CONS: GPU | same, CPU probe | `--causal-check` |
|---|---|---|---|---|---|---|---|---|
| o_cons | the file as copied from wt_c (D1c30 + VC50) | 96.366 | 81.772 | 96.870 | 55.701 | 0 | 0 | OK |
| g_cons | patched file, new knobs off | 96.366 | 81.772 | 96.870 | 55.701 | **bit-identical** (pred / lower / upper `array_equal` True) | | OK |
| f_k0 | D1c10 + VC30 | 96.354 | 81.813 | 96.874 | 55.677 | -0.0072 | -0.0077 | OK |
| f_s1 | C2 without EB damping | 96.359 | 81.863 | 96.879 | 55.714 | +0.0099 | +0.0097 | OK |
| **g_s1b** | **C2 (ship)** | 96.359 | 81.862 | 96.880 | 55.716 | **+0.0104** | +0.0105 | OK |
| g_cl | D1c30 + lead 5 + VC50 | 96.368 | 81.799 | 96.870 | 55.711 | +0.0062 | +0.0064 | OK |

AoA 15-20 on the real path: CONS rel 95.157 / tke 82.235 / mvpe 95.814 / sps 48.518; **C2** 95.138 / 82.275 / 95.813 / 48.493 = **-0.0114** vs CONS
(probe -0.0112); g_cl 95.158 / 82.230 / 95.814 / 48.515 = -0.0008 (probe -0.0012). **The probe and the shipped code agree to 0.001 everywhere.**
**The exact file the patch produces** (`$S/wt_c2/Track_2/submit/submission.py`, byte-equal to patch side b) was run last, job 7567290:
knobs off = bit-identical to the original file; C2 = bit-identical to the `g_s1b` package above; CAUSALITY OK on both; 3.024 ms/step (C2)
against 3.049 (CONS) in the same job.
The OOD regime was NOT run through the real code path (no package exists for the AoA-holdout members); its evidence is the CPU probe only.

### 5.2 Latency

`stream_eval` local ms/step, packages run back to back on one node, two rounds (job 7567000): o_cons 3.046 / 3.023 · patched knobs-off 3.012 / 3.020 ·
C2 without damping 3.022 / 3.045 · D1c10+VC30 3.016. **The added cost of EB + lead profile is below this instrument's noise (about 0.03 ms), so well
inside the 0.1 ms budget**; it is ~20 tiny kernels and one (1,20,32,64,2) float64 accumulator inside the graph.
**My interleaved timer failed and I have no paired latency number:** `stream_c/interleave_time.py` loads five packages (five captured CUDA graphs)
into one process and died with "CUDA error: an illegal memory access" on the first replay of `f_base`, a package with every knob off. That is a
fault of loading several graph-holding models into one process, not of any one package (each package runs clean alone), but it means the
latency claim above rests on sequential runs only. Use `t2_lat17.py` / the build's own timing for the record.

### 5.3 In distribution at the 1.29 stretch (UPPER END for anything carrying VARCAL; do not bank)

CONS +0.1315 / a15 +0.0773 · KNOB +0.1426 / +0.0614 · **C2 +0.1522 / +0.0757** · C2 with VC50 +0.1392 / +0.0732 · D1c30+lead+VC50 +0.1360 / +0.0776.

### 5.4 Two alternatives EC can build from the same patch

| name | knobs | in-dist full / AoA 15-20 | OOD | 14-window OOD | use when |
|---|---|---|---|---|---|
| **C2** | D1C_N0 30, D1C_EB True, D1C_EB_N0 1, D1C_LEAD_N0 5, VC_N0 30 | +0.0622 / +0.0090 | +0.4008 | +0.2616 | the board is OOD-like (upside) |
| C2-v50 | same with VC_N0 50 | +0.0641 / +0.0155 | +0.3887 | +0.2461 | hedge: 0.012 less OOD, 0.007 more on AoA 15-20 |
| CONS+lead | D1C_N0 30, D1C_LEAD_N0 5, VC_N0 50 (EB off) | +0.0581 / +0.0190 | +0.2966 | +0.1476 | strictly safer than CONS: +0.006 in-dist, +0.029 OOD, AoA 15-20 -0.001 |

### 5.5 Prediction (registered before any upload)

Delta T2 `final` of C2 over EC's T2_CONS (D1c30 + VC50), same members, same band, same R1: **+0.03, bracket [-0.02, +0.12]**.
Centre = lambda 0.5 x (OOD +0.133) + 0.5 x (AoA 15-20 -0.011) = +0.06, halved for the proxy's over-read of error cuts. Lower end = the board is
in-distribution AoA 15-20 (-0.011) plus a latency draw; upper end = the board is my OOD regime at full length.
Subscores vs CONS: rel 0 [-0.03, +0.07] · tke +0.08 [+0.02, +0.15] · mvpe +0.10 [-0.01, +0.27] · sps +0.10 [-0.03, +0.25] · time 0 [-0.03, 0].
Over today's originally planned asnp + D1c30: **+0.05 [-0.02, +0.20]**. **This does not by itself reach the top-10 cut.**

### 5.6 Not verified

* Job 7567230 `rp_f27c_real` in the queue is NOT mine (I did not submit it; presumably EC's). My jobs: 7566953 (cancelled by me, superseded),
  7567000, 7567001 (cancelled by me before it started), 7567031, 7567290.

* OOD on the real code path (probe only); the all-data shipping members were never run by me (twins only).
* `validate_t2.py --adaptive` on a built archive; `pkg_identity.py`.
* Paired / interleaved latency (see 5.2).
* Constants were chosen on the same trajectories they are read on (8 in-dist + 16 OOD); no leave-one-trajectory-out refit. The neighbours in
  section 2 are flat to 0.001-0.006, so I expect little selection bias, but it is not measured.
* OOD stand-in: 3 members (2 F-FNO+sim, 1 FNO), AoA 20 only, band not refitted. A sign and an order of magnitude, not a size.

## 6. Instruction for EC (apply and build)

`cd $S/wt_c && patch -p1 < $S/notes/c2.patch` — it touches `Track_2/submit/submission.py` (5 hunks) and `Track_1/submit_fno/build_round13.sh`
(1 hunk); I dry-ran it at 17:04 against wt_c's files both in the baked state I copied at 16:41 and in their then-current un-baked state
(all hunks apply, offset -5 lines). If a hunk is rejected, do not force it: copy the two files from `$S/wt_c2/` instead, which also reverts
anything you changed in those two files after 16:41. With every new knob at its default the step is bit-identical to your current T2_CONS
(checked on the twin package). Build exactly as T2_CONS but with
`RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_D1C_EB=True RPDE_SHIP_D1C_EB_N0=1.0 RPDE_SHIP_D1C_LEAD_N0=5.0 RPDE_SHIP_VC_N0=30.0` and `RPDE_SHIP_FW_N0`
unset or 0.0. Then confirm the package's load line reads `d1c_eb True d1c_eb_n0 1.0 d1c_lead_n0 5.0 vc_n0 30.0 fw_n0 0.0 ... graph CAPTURED`,
run `Track_2/stream_eval.py --submission <pkg> --split unseen --causal-check` (must print CAUSALITY OK) and `Track_2/validate_t2.py --adaptive`
(must print SUBMITTABLE; its shuffled-order rule has to be the relaxed D1c one, because this changes P by design). Twin packages to compare
against are in `$S/pkg/c2/` (`o_cons`, `g_s1b`, `g_cl`, `f_k0`, `f_base`). Kill it if the graph is not captured, if the causal check fails, or
if the twin read of your built code differs from section 5.1 by more than 0.003 composite.
