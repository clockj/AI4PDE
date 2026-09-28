# OOD stream instrument: built and validated (2026-09-28, executor "gauge")

Code: `experiments/decision_20260928/ood/` (README has the commands). Runs: `/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs/<name>/`
(`report.md`, `summary.json`, `json/<variant>.json` with every window). GPU used: **0.35 GPU-h** (sacct, 7 jobs `rp_dp_ood_*`,
3 of them early failures).

## What it does

One Slurm job per ensemble (`bash ood_submit.sh NAME REGIME LAYOUT`) runs these steps:

1. **Members.** The member list is derived from the layout. The job refuses any member whose training set contains an
   evaluation or fit-split trajectory, any all-data model, and any sim co-training without `--sim-exclude-held`. The
   configs are read from `["config"]` and the train2.py exclusion rule is applied.
2. **Band.** The fit split is `model.VAL_TRAJ` minus the regime. The static band is fitted with `fit_band.py --ckpt E
   --err-ckpt E --alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored --fit-on unseen --fit-aoa <AoA 15-20 left in
   the split>` (inflate 1.39 and smoothing 3 are the defaults). The SIGBLEND refit then runs `dump_twins.py --val <fit
   split>` and `make_k0_band.py` (λ 0.5, 1.29, 16 bins). Every script runs unchanged, so no patch was needed.
3. **Build.** `build_final.sh` runs with `RPDE_A = RPDE_M = E` and the regime band plus its json. It must end in
   `BUILD_FINAL OK`, which requires both gates and the causality check. The job also checks two more things. The three
   `sed`-edited files must equal the commit before and after the build. The Track-2 gate's load line must show the
   shipped stream constants.
4. **Variants.** The archive is extracted once. Each variant is a copy of it with its constants rewritten by the
   build's own `sed` patterns. Each copy has its diff saved and checked.
   * `frozen`: every stream knob off.
   * `base`: asymmetric R1 only.
   * `control`: record + `PREV_SPACE "target"`.
   * `prev_input`: the record as built.
   * `statsmask`: control + `STATS_MASK`.
   * `famw`: control + `FW_N0 20.0` (design C's Decision-phase value, `design_c.md`).
5. **Stream.** Each variant runs in its own process through `Track_2/stream_eval.run`, with the revealed target
   normalised with the TARGET statistics. `--causal-check` runs on `control`. Every window's kit metrics are stored.
6. **Report.** The report gives, pooled per group and per trajectory:
   * rel / tke / mvpe / sps, C and coverage;
   * the composite (`W0` without the time term), on full streams, the first 3 windows and the first 14;
   * deltas to control and to base, with the number of trajectories positive;
   * first-window bit-identity of every variant to `frozen`;
   * ms/step, reported only.

   Subset scores are exact: each run's pooled result is checked against the kit's `aggregate_sps`, and the largest
   difference was 1e-8, which is float32 rounding.

Regimes are derived in code. A20 and A10 are 16 and 18 trajectories. RE is `re_split.SETS` (relo 9, rehi 7, rein 10).
ID is the eight `VAL_TRAJ` trajectories, reported as all8 and as its AoA 15-20 slice.

## Validation

Composite = W0 · (rel, tke, mvpe, sps). Δ values are pooled; the brackets count the trajectories whose own Δ is > 0.

### In distribution: the five twins of record, band of record (`id_twins`)

| variant | rel | tke | mvpe | sps | composite | Δ control | Δ base |
|---|---|---|---|---|---|---|---|
| frozen | 96.399 | 81.480 | 96.815 | 54.525 | 76.8514 | −0.3991 (0/8) | −0.3234 |
| base (asym R1) | 96.399 | 81.480 | 96.815 | 55.823 | 77.1748 | −0.0757 (0/8) | 0 |
| **control** | 96.376 | 81.837 | 96.879 | 56.007 | 77.2504 | 0 | **+0.0757 (8/8)** |
| prev_input | 96.376 | 81.837 | 96.879 | 55.982 | 77.2440 | **−0.0064 (0/8)** | +0.0693 (8/8) |
| statsmask | 96.376 | 81.836 | 96.877 | 56.005 | 77.2493 | −0.0012 (4/8) | +0.0745 (8/8) |
| famw | 96.375 | 81.838 | 96.880 | 56.005 | 77.2497 | −0.0007 (2/8) | +0.0750 (8/8) |

AoA 15-20 slice: control − base +0.0213 (3/3); prev_input −0.0027; statsmask +0.0002; famw −0.0048 (0/3).

**This agrees with `experiments/audit_20260927/refactor_pkg.md`.** Every subscore of arm (c) = control and arm (b) =
prev_input agrees to the digits printed there (96.3760 / 81.8367 / 96.8791 / 56.0075 and 55.9817). PREV_SPACE target
reads **+0.0064 with 8 of 8 positive**, the same as there; on AoA 15-20 it reads +0.0027 there and +0.0027 here. STATS_MASK
reads −0.0012 in both. The absolute composites are 0.0015 higher here (77.2504 against 77.2489) because that document
used rounded weights (0.4819 …) and this uses `W0` itself. For STATS_MASK, 4 of 8 trajectories are positive here and 3
of 8 there; one trajectory's delta is +0.0000 at four decimals.

### AoA 20 held out: stand-in `aoa20_ffnosim_s1, s2 + aoa20_fno_s1`, refitted band (`a20_standin`)

| variant | rel | tke | mvpe | sps | C | composite | first 3 | first 14 | Δ control | Δ base | Δ base, first 3 / 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | 94.322 | 79.845 | 94.246 | 42.758 | 59.815 | 72.5246 | 72.2998 | 72.3643 | −0.9083 (0/16) | −0.4554 | −0.146 / −0.362 |
| base | 94.322 | 79.845 | 94.246 | 44.586 | 62.372 | 72.9800 | 72.4454 | 72.7258 | −0.4529 (0/16) | 0 | 0 |
| **control** | 94.468 | 80.591 | 94.973 | 45.562 | 62.868 | 73.4329 | 72.5128 | 73.0304 | 0 | **+0.4529 (16/16)** | +0.067 / +0.305 |
| prev_input | 94.468 | 80.591 | 94.973 | 45.553 | 62.856 | 73.4306 | 72.5120 | 73.0286 | −0.0023 (0/16) | +0.4507 (16/16) | |
| statsmask | 94.467 | 80.589 | 94.973 | 45.561 | 62.868 | 73.4323 | 72.5127 | 73.0300 | −0.0006 (2/16) | +0.4524 (16/16) | |
| famw | 94.473 | 80.602 | 94.969 | 45.585 | 62.892 | 73.4417 | 72.5165 | 73.0388 | **+0.0088 (13/16)** | +0.4617 (16/16) | |

Control − base by subscore: rel +0.146, tke +0.747, mvpe +0.728, sps +0.976, ΔC ×100 +0.50.

### AoA 10 held out: stand-in `aoa10_ffnosim_s1, aoa10_ffno_s1 + aoa10_fno_s1, aoa10_nosim_s1` (`a10_standin`)

| variant | rel | tke | mvpe | sps | C | composite | first 3 | first 14 | Δ control | Δ base | Δ base, first 3 / 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | 95.816 | 79.665 | 95.152 | 49.570 | 66.653 | 75.0065 | 75.0406 | 74.9841 | −1.0035 (0/18) | −0.4362 | −0.166 / −0.349 |
| base | 95.816 | 79.665 | 95.152 | 51.321 | 69.012 | 75.4427 | 75.2063 | 75.3326 | −0.5673 (0/18) | 0 | 0 |
| **control** | 96.025 | 80.564 | 96.179 | 52.465 | 69.264 | 76.0100 | 75.2976 | 75.7398 | 0 | **+0.5673 (18/18)** | +0.091 / +0.407 |
| prev_input | 96.025 | 80.564 | 96.179 | 52.439 | 69.230 | 76.0036 | 75.2956 | 75.7347 | −0.0064 (0/18) | +0.5609 (18/18) | |
| statsmask | 96.024 | 80.563 | 96.178 | 52.466 | 69.267 | 76.0099 | 75.2981 | 75.7398 | −0.0001 (11/18) | +0.5672 (18/18) | |
| famw | 96.028 | 80.599 | 96.181 | 52.500 | 69.286 | 76.0241 | 75.3022 | 75.7496 | **+0.0141 (16/18)** | +0.5815 (18/18) | |

Control − base by subscore: rel +0.209, tke +0.899, mvpe +1.027, sps +1.144, ΔC ×100 +0.25.

### Against the sprint's reads

| regime | this instrument: control − base | sprint | Δ | base composite, here / sprint |
|---|---|---|---|---|
| AoA 20 | **+0.453 (16/16)** | +0.446 (T2_P2 vs asym R1, real path, job 7567448) | +0.007 | 72.980 / 72.939 |
| AoA 10 | **+0.567 (18/18)** | +0.553 (C2 +0.521 + B2 +0.031, CPU probe, `ood_aoa10.md`) | +0.014 | 75.443 / 75.437 |

No difference reaches 0.1. The member sets are the same as the sprint's. What changed since then:
* the band is refitted to each ensemble; the sprint used an in-distribution band;
* JOINT_GAIN is on, as in the package of record; T2_P2 was built before the merge;
* the revealed target is normalised with TARGET statistics on both sides;
* the path is real on both regimes; at AoA 10 the sprint used a probe.

Together these move the stack's gain by less than 0.015 and the base level by less than 0.05. On these stand-ins
**G0's criterion would pass** in both regimes: +0.45 and +0.57, positive on 100 % of trajectories. They are 3 to 4
single-seed members, not the mirror ensembles G0 is defined on.

### Checks

* **Causality.** `stream_eval --causal-check` passes on `control` in every run: ID 176 steps, A20 337, A10 396, all at
  max |diff| 0. `build_final.sh`'s own check passes as well.
* **Knob off equals frozen.** On the first window of every trajectory, `base`, `control`, `prev_input` and `statsmask`
  are bit-identical to `frozen` in pred, lower and upper: 8/8, 16/16 and 18/18. **FAMW breaks this property.** Its
  first window differs from `frozen` in every trajectory at ID (w0 2/5) and at A20 (w0 1/3), and in the prediction of
  1 of 18 at A10 (w0 1/2). The cause is that the mix yA + w0·(yB − yA) at n = 0 equals the plain member mean
  algebraically but not in floating point. This matters for arm I3: FAMW as written is not bit-identical to the package
  of record when it is switched on, even at n = 0. When it is off (`FW_N0 = 0`), the code path is untouched.
* Every variant's `VARIANT.diff` changes only its intended constant lines. The live state flags (r1 / d1c / vc / fw) and
  the load line match the knobs in every run.
* **RE plumbing.** A smoke job (`ood/smoke_re/`) streamed the ID package over the 26 RE trajectories and reported the
  relo / rehi / rein groups correctly. Its numbers are contaminated, because the twins trained on RE trajectories, and
  are not reported.

## Member layouts the package code accepts

`_fno.fuse` groups members by architecture and parameter shapes, and it accepts any count per family.
* **F-FNO w40 L8 + ViT d384 L8** gives 2 invocations. This is the package of record (4 + 2), and the ID run here (3 + 2)
  confirms it.
* **F-FNO + FNO w96 L8** gives 2 invocations: the stand-ins, 2 + 1 at A20 and 2 + 2 at A10. Both build, pass both
  gates, capture the graph and run FAMW with two groups.
* The full-size layouts are `mirror` (4 F-FNO + sim + VITP s1 + VITC s3), `six` (6 + 2) and `vitc2` (4 + VITC s1 + s3).
  Their paths come from `dec0928/ckpt/<REGIME>/<regime>_*.pth`, and they take the same F-FNO + ViT path as the package
  of record.
* **A third family**, for example F-FNO + FNO + ViT, would make three groups. FAMW then switches itself off silently:
  the json's `active.fw` is False and `ood_stream` now prints a WARNING.

## What could not be done as designed, and what I did instead

* **Build isolation.** The design called for one scratch worktree, `wt_ood`, for all builds. Two jobs on different
  nodes shared it behind a `flock`, which does not serialise across nodes on scratch. One Track-2 archive was therefore
  built from the unsubstituted defaults. Its gate load line read `r1 False … sigpow None`, the build ended
  `BUILD_FINAL OK` anyway, and my post-build check caught the problem. Each run now builds in its own `runs/<name>/src`,
  a `git archive` of the pinned commit 4e5ed37. `wt_ood` is restored and unused apart from supplying the commit. All
  results above come from the rebuilt runs. **The planner should know that the same hazard applies to any campaign job
  that builds or fits a band in a shared worktree such as `wt_base`.**
* **The "frozen package" check.** I read it as knob-off identity: every variant equals `frozen` on each trajectory's
  first window, and each variant's diff touches only its knob lines. I did not compare against a separately built
  knob-off archive, because `build_final.sh` hard-codes `R1=True` and the other stream knobs.
* **The static band's fit split.** It follows the shipping construction's `--fit-aoa 15 20`, restricted to the fit
  split. At A20 that leaves one trajectory (21600_15, 42 windows); at A10 and RE it leaves three. The SIGBLEND refit
  uses the whole split of six trajectories. The shipped static band was fitted on the twins' *in-sample* `val_seen`
  errors. The regime band is fitted on errors that are out of sample for the regime ensemble, as the design says, so it
  is somewhat wider than what an organiser rebuild would ship for the same members.
* **The in-distribution run** uses the band of record. A refit on `VAL_TRAJ` would be in-sample on the evaluation set.

## Verified, and not verified

**Verified**
* The pipeline reproduces the audited in-distribution read exactly.
* On both held-out angles it reproduces the sprint's reads of the stream stack within 0.015.
* It is causal, and knob-off is bit-identical on first windows.
* Subset scoring agrees with the kit.
* Stand-in ensembles with an FNO second family build and gate cleanly.

**Not verified**
* No full-size ensemble (4 F-FNO + 2 ViT, without a regime) exists yet. That layout is exercised only through the
  in-distribution 3 + 2 twins and the package of record.
* No RE ensemble exists, so the RE regime is checked for plumbing only.
* A20's static band rests on one fit trajectory.
* Local ms/step comes from shared nodes (`RPDE_FORCE=1`) and is not latency evidence.
* No number here was worse than the worst case of its own hypothesis. Every stream stack read is positive on every
  trajectory, and the knob deltas are small and signed as expected.
