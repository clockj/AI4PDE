# OOD instrument, chunk 5: arm T4 replicated and completed (2026-09-28, executor "gauge")

T4 is the Re-similarity augmentation (`--aug-resim 0.25`): a real window resampled in time and amplitude by a factor s.

Code, all in `experiments/decision_20260928/ood/`:
* New: `ood_jk.sh`, `ood_jkread.py`, `ood_split.py`, `ood_single.py`, `ood_poll5.sh`.
* New layouts in `ood_lib.py`: t4v, t4b, twins_t4.

Every job ran from a code snapshot. Runs are under `/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs/`.

## The T4 verdict (items a, b, e)

**T4 PASSES every term of its PLAN criterion. Headline: OOD mean over A20, A10, relo and rehi = +0.118.** Stream
(Track 2), each regime's T4 ensemble (four augmented F-FNO members + the control's two ViTs) against the mirror control of
its regime; bands refitted to each ensemble; real code path.

| regime | T4 − mirror, control (trajectories positive) | Δ rel / tke / mvpe / sps / C | first 3 / first 14 windows | frozen (Track 1) |
|---|---|---|---|---|
| A20 | **+0.0516** (16/16) | +0.041 / +0.026 / +0.000 / +0.117 / +0.093 | +0.078 / +0.057 | +0.095 (15/16) |
| A10 | **+0.0395** (17/18) | +0.019 / +0.068 / +0.018 / +0.088 / +0.050 | +0.065 / +0.044 | +0.125 (18/18) |
| relo | **+0.3335** (9/9) | +0.174 / +0.502 / +0.084 / +0.773 / +0.376 | +0.519 / +0.421 | +0.505 (9/9) |
| rehi | **+0.0458** (6/7) | +0.028 / −0.022 / −0.001 / +0.138 / +0.155 | +0.133 / +0.050 | +0.174 (6/7) |
| rein (reported) | +0.0245 (9/10) | +0.010 / +0.066 / +0.004 / +0.050 / +0.018 | +0.056 / +0.040 | +0.025 (9/10) |
| **in distribution, twins, all 8** | **+0.0332 (8/8)** | +0.015 / +0.080 / +0.006 / +0.071 / +0.028 | +0.012 / +0.029 | +0.027 (7/8) |
| in distribution, twins, AoA 15-20 | +0.0459 (3/3) | +0.021 / +0.106 / +0.007 / +0.099 / +0.054 | +0.037 / +0.055 | +0.041 (3/3) |

| criterion term | read | verdict |
|---|---|---|
| mean(relo, rehi) ≥ +0.10 | +0.190 | pass |
| A20 ≥ −0.02 | +0.052 | pass |
| A10 ≥ −0.02 | +0.040 | pass |
| in distribution (twins) ≥ −0.03 | +0.033 | pass |

**The development-board read (item b).** The three T4 twins plus the two ViT twins of record, against the five twins of
record. Both bands are refitted the shipping way (static on val_seen AoA 15-20, SIGBLEND on val_unseen), on the 8
`VAL_TRAJ` trajectories:

| | rel | tke | mvpe | sps | C | composite |
|---|---|---|---|---|---|---|
| five twins of record, all 8 | 96.376 | 81.837 | 96.879 | 56.008 | 72.154 | 77.2506 |
| T4 twins, all 8 | 96.391 | 81.917 | 96.885 | 56.079 | 72.181 | 77.2838 |
| five twins of record, AoA 15-20 | 95.160 | 82.260 | 95.812 | 48.630 | 65.042 | 74.7712 |
| T4 twins, AoA 15-20 | 95.181 | 82.366 | 95.819 | 48.729 | 65.095 | 74.8170 |

The shipping-way refit of the five twins of record reads 77.2506, against 77.2504 with the band of record: the refit
reproduces the package of record's operating point. T4 is not a trade: it gains in distribution too, and gains more on
AoA 15-20, the slice most like the board.

## T4v, the ViTs with the augmentation (item c)

**Single-member screen:** clears, 3 of 3. Each augmented ViT against its seed-matched control ViT, `wfinal` from the
trainings' own `re_split.py` reads:

| member | relo Δ | rehi Δ | mean(relo, rehi) Δ | rein Δ |
|---|---|---|---|---|
| ViT-C s3 | +4.541 | +0.092 | +2.316 | +0.079 |
| ViT-C s1 | +5.560 | +0.396 | +2.978 | +0.107 |
| ViT-P s1 | +6.719 | +0.404 | +3.562 | +0.060 |

**Ensemble at RE**, all six members augmented (four T4 F-FNO, ViT-P resim s1, ViT-C resim s3); band refitted; real path:

| group | vs mirror control (n+) | Δ rel / tke / mvpe / sps / C vs mirror | vs T4 with the control's ViTs (n+) | frozen vs mirror | frozen vs T4 |
|---|---|---|---|---|---|
| relo | **+0.7090** (9/9) | +0.373 / +0.736 / +0.210 / +1.757 / +1.062 | **+0.3755** (9/9) | +1.872 | +1.368 |
| rehi | +0.0247 (5/7) | +0.027 / −0.142 / −0.032 / +0.115 / +0.182 | −0.0211 (2/7) | +0.134 | −0.040 |
| rein | +0.0195 (8/10) | +0.009 / +0.038 / −0.008 / +0.049 / +0.037 | −0.0049 (4/10) | +0.020 | −0.004 |
| in distribution (REID; band in sample) | +0.0487 (6/6) | +0.022 / +0.091 / +0.007 / +0.115 / +0.068 | +0.0117 (4/6) | +0.044 | +0.005 |
| **mean(relo, rehi)** | **+0.367** | | **+0.177** | | |

**Against T4v's criterion at RE:** ≥ +0.10 over the control and ≥ +0.03 over T4, on the relo/rehi mean. It passes both,
but the whole margin is relo.
* At rehi the augmented ViTs lose slightly to the control's ViTs (−0.021, 2/7; tke −0.14).
* At rein they are level (−0.005).
* The A20 / A10 terms and the in-distribution twins term wait for the augmented A20 / A10 ViTs and the augmented ViT
  twins (items f-h).

## T4b, `--aug-resim 0.5` (item d)

**Not read yet.** The four members were at the end of training at 15:05, with no json yet. The poller screens them and
submits the ensemble read only if the screen clears. It goes into the second hand-back with items f-h.


## Part 1: property of the arm or of one seed set?  (no new training)

**At RE, T4's gain is outside what reseeding the control gives, and by a wide margin.**
* Every leave-one-out ensemble keeps it on every regime and on nearly every trajectory.
* Relo: the jackknife range is +0.301 to +0.315. Two control ensembles drawn from the six control seeds differ by at
  most 0.034 (about a ninth of the gain).
* Rehi: +0.039 to +0.055, against at most 0.011.
* The in-distribution split: +0.033 to +0.039, against at most 0.012.

How the ensembles were built:
* **Leave-one-out.** Three F-FNO members (drop s1, s2, s3 or s4) plus the control's two ViTs, T4 against control with
  the same seed dropped.
* **Control reseeding.** Four-member draws s1-4 (the mirror), s3-6, s1,2,5,6 and s1,3,4,6, plus the two ViTs.

Each ensemble has its own refitted band and build and is read on the real code path. In distribution means
`VAL_TRAJ` minus the RE trajectories (6 trajectories); the regime band is fitted on these, so the sps part is in sample.

**Control (Track 2):**

| group | T4 − control, full ensembles | jackknife T4 − control: mean [min, max], trajectories positive per drop | control − control across 4 reseeded draws: mean pairwise difference, max (6 pairs) | control leave-one-out range |
|---|---|---|---|---|
| relo | +0.3335 | **+0.309 [+0.301, +0.315]**, 9/9 in all 4 | 0.018, **0.034** | 0.009 |
| rehi | +0.0458 | **+0.045 [+0.039, +0.055]**, 6/7 in all 4 | 0.006, **0.011** | 0.018 |
| rein | +0.0245 | +0.024 [+0.021, +0.027], 9/10 in all 4 | 0.006, 0.010 | 0.009 |
| in distribution (REID) | +0.0370 | +0.036 [+0.033, +0.039], 5-6/6 | 0.006, 0.012 | 0.003 |

**Frozen (Track 1's case):**

| group | T4 − control, full ensembles | jackknife T4 − control | control − control, mean / max | control leave-one-out range |
|---|---|---|---|---|
| relo | +0.505 | +0.443 [+0.390, +0.519] | 0.041 / 0.076 | 0.115 |
| rehi | +0.174 | +0.175 [+0.158, +0.194] | 0.015 / 0.030 | 0.046 |
| rein | +0.025 | +0.024 [+0.024, +0.026] | 0.007 / 0.013 | 0.006 |
| in distribution (REID) | +0.040 | +0.039 [+0.035, +0.042] | 0.006 / 0.010 | 0.004 |

**Caveats.** The four T4 members still share one training recipe and commit, so this rules out one lucky seed set, not
one lucky configuration. The ViTs are the same two in every ensemble on both sides.


## Part 3: what T4 does not fix, and what is left after the stream

**Decomposition.** Per window, E = P − Y over u and v, in physical units. It splits exactly into three parts:
* the 20-frame time-mean error of the window, itself split into the part that persists over the whole trajectory (per
  pixel; what a per-trajectory correction such as D1c can learn) and the part that varies from window to window;
* the fluctuation error (E minus its time mean).

Shares are of Σ‖E‖², pooled over the group. The "rel without …" columns give the mean rel error if that part were
removed.

| run / variant / group | rel error | persistent time-mean | window-varying time-mean | fluctuation | rel without the time-mean error | rel without the fluctuation error |
|---|---|---|---|---|---|---|
| control (mirror) / frozen / rehi | 0.1084 | 6.9 % | 21.4 % | 71.7 % | 0.0916 | 0.0574 |
| control (mirror) / control / rehi | 0.1074 | **0.7 %** | 23.1 % | **76.2 %** | 0.0936 | 0.0521 |
| T4 / frozen / rehi | 0.1075 | 6.4 % | 21.8 % | 71.8 % | 0.0908 | 0.0568 |
| T4 / control / rehi | 0.1067 | 0.6 % | 23.4 % | 76.0 % | 0.0929 | 0.0521 |
| control / frozen / relo | 0.0993 | 28.3 % | 22.7 % | 49.0 % | 0.0682 | 0.0712 |
| control / control / relo | 0.0823 | 1.2 % | 34.1 % | 64.7 % | 0.0665 | 0.0479 |
| T4 / control / relo | 0.0785 | 1.2 % | 34.7 % | 64.1 % | 0.0631 | 0.0462 |
| control / frozen / A20 | 0.1220 | 12.4 % | 23.0 % | 64.6 % | 0.0941 | 0.0770 |
| control / control / A20 | 0.1147 | 1.0 % | 27.5 % | 71.5 % | 0.0939 | 0.0653 |
| control / control / rein | 0.0691 | 0.6 % | 25.5 % | 73.9 % | 0.0595 | 0.0348 |

Composite by subscore at rehi (scores; errors in brackets):

| | rel | tke | mvpe | sps | C | composite |
|---|---|---|---|---|---|---|
| control, frozen | 94.858 (0.1084) | 78.249 (0.5559) | 95.460 (0.0951) | 41.330 | 56.658 | 72.381 |
| control, stream | 94.906 (0.1074) | 79.157 (0.5266) | 96.049 (0.0823) | 43.070 | 58.443 | 72.981 |
| T4, frozen | 94.900 (0.1075) | 78.269 (0.5553) | 95.594 (0.0922) | 41.891 | 57.318 | 72.556 |
| T4, stream | 94.934 (0.1067) | 79.135 (0.5273) | 96.049 (0.0823) | 43.209 | 58.598 | 73.026 |

**Answer: what is left at rehi after the stream is not mean-field error the stream has not learned.**
* The trajectory-persistent time-mean error, which D1c targets, is down to 0.6-1.2 % of the squared error in every
  regime after the stream. It was 6.9 % at rehi, 12.4 % at A20 and 28.3 % at relo before it. D1c has learned it.
* What remains is **76 % fluctuation error** at rehi (65-74 % elsewhere) plus **23 % time-mean error that changes from
  window to window**. Neither is reachable by a per-trajectory bias correction:
  * the first needs a better forecast of the fluctuations;
  * the second needs a better per-window mean, i.e. the model's own forecast of the slow phase.
* T4 does not change this split at rehi: 76.0 % against 76.2 % fluctuation. It lowers rehi's error only slightly.
* The frozen split shows why rehi gains least:
  * the part a model change or a per-trajectory correction can move easily (the persistent mean) is 6.9 % at rehi,
    against 28 % at relo;
  * the rest at rehi is dominated by the fluctuations of the highest-Re flows, which neither T4 nor the stream
    addresses.

## Ledger, and what is verified

* **GPU:** 1.47 GPU-h in chunk 5 so far (18 Ascend jobs, all COMPLETED), plus a 15-second Pitzer CPU job. Budget 8.
* **Poller bug:** the chunk-5 poller tested the training sentinels with `-s`, and the sentinels are empty files, so T4
  A20 / A10 waited from 14:09 / 14:16 to 14:25. I stopped my poller, fixed the test to `-e`, and restarted it. No read
  was affected.

**Verified**
* Every read is on the real package code, with each ensemble's own refitted band. Every build ended `BUILD_FINAL OK`,
  with causality passing.
* The jackknife: 11 ensembles, each with its own band.
* The Part 3 decomposition is exact (the three shares sum to 1 to 1e-9).
* The shipping-way band refit of the five twins reproduces the band of record's composite to 2e-4.

**Not verified**
* T4's four members share one recipe and commit. The jackknife rules out a lucky seed set, not a lucky configuration.
* The T4v ensemble is read at RE only, with one seed per augmented ViT; its gain is carried by relo.
* The RE in-distribution split (REID) is in sample for the regime band. The twins read (item b) is the clean
  in-distribution read.
* T4b and items f-h are pending.

