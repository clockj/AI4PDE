# Closing gates on the candidate (chunk 6, 2026-09-28, executor "gauge")

**The candidate** is the from-scratch run of `reproduce.sh decision` from a clean clone of 467769d.
* **Commit.** My reads build from commit 4e5ed37. Between the two commits only `build_final.sh` differs (the lock and the
  baked-constant check); the package code, the band scripts and the harness are identical.
* **Read setup.** All reads use the real package code, the twins on the 8 `VAL_TRAJ` trajectories, and the band refitted
  the shipping way (static fit on val_seen AoA 15-20, SIGBLEND on val_unseen).
* **New code** in `ood/`: `ood_r1.py`, `ood_member.py`, `ood_expect.py`, and `r2/` (`r2_run.sh`, `r2_compare.py`,
  `t1_check.py`, and `t2_harness.py`, which is the sprint's QA harness plus one `fp32` mode).

## Summary

| gate | result |
|---|---|
| **R1** reproducibility (≤ 0.005) | **PASS**: from-scratch − campaign twins **−0.0003** composite (stream, all 8), −0.0006 on AoA 15-20; every trajectory within ±0.0012. Frozen −0.0017. |
| R1 one level down | The two ViT twins retrained **bit-identically** (0 of 14.7 M parameters differ). The three F-FNO twins differ at 0.42-0.49 % of weight norm. Every single-twin error difference on val_unseen is ≤ 2.2e-5. |
| **R1b** the band | The candidate barely depends on the refit. The from-scratch band, my shipping-way refit and the campaign twins' band give (a) the same composite to 1e-4. The band of record gives +0.0022 (stream) and +0.0201 (frozen) more, with a band about 2.4 % wider in its median base. |
| second candidate "T4 F-FNO only" | Built by the release's own path with `RPDE_DP_RESIM_VIT=` (empty): `BUILD_FINAL OK`, 2/2 gates, causality, 41/41 baked constants, clean tree after the build. Its twins: **+0.0329 (8/8)** over the twins of record; −0.0003 against the campaign's F-FNO-only twins (its own reproducibility read). |
| **R2** a different card and the fallbacks | **PASS** for all three archives (both candidates and the uploaded package). See below. |

**Two points about the release's empty-variable path (REPRODUCE.md §14).**
1. **The names are right; the tag is not what §14 says.**
   * The derived member names are correct: the four F-FNO members, the three F-FNO twins, and the record's
     `arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3`, `arch_vit_d384` and `arch_vit_cmb3sim`. The static fit's `--ckpt` and
     `--err-ckpt` confirm it.
   * **The tag becomes `dp_resim_v`**: the rule appends `_v`, then the empty value. §14 says the tag is `dp_resim` and does
     not mention this case.
   * So both archives are `track{1,2}_dp_resim_v.zip`. It works, but the name reads as "a ViT variant" rather than "the
     record's ViTs".
2. **A wrong closing message.** With `RPDE_REPRO_STAGE="dp_band dp_build"` the script ends with `STAGES DONE: dp_band
   dp_build (no build -- rerun without RPDE_REPRO_STAGE)`, although `dp_build` did build and gate both archives.

## R1 and R1b

### Gate R1: from-scratch twins (a) against the campaign's twins (b); acceptance |Δ composite| ≤ 0.005

#### (a) − (b): from-scratch run − campaign, five augmented twins each

| variant | slice | Δ rel | Δ tke | Δ mvpe | Δ sps | ΔC | **Δ composite** | per trajectory (composite) |
|---|---|---|---|---|---|---|---|---|
| control | all8 | -0.0002 | +0.0004 | -0.0002 | -0.0010 | -0.0011 | **-0.0003** | 11400_20 -0.0005, 13950_10 -0.0010, 16500_5 +0.0007, 21600_15 -0.0010, 26700_0 +0.0002, 3750_0 +0.0006, 6300_10 -0.0012, 8850_20 -0.0003 |
| control | a15_20 | -0.0004 | -0.0003 | -0.0007 | -0.0014 | -0.0009 | **-0.0006** |  |
| frozen | all8 | -0.0003 | -0.0004 | -0.0005 | -0.0058 | -0.0063 | **-0.0017** | 11400_20 -0.0006, 13950_10 -0.0010, 16500_5 -0.0010, 21600_15 -0.0039, 26700_0 -0.0066, 3750_0 +0.0007, 6300_10 -0.0016, 8850_20 +0.0001 |
| frozen | a15_20 | -0.0004 | -0.0003 | -0.0008 | -0.0046 | -0.0050 | **-0.0014** |  |

#### (a) − (c): from-scratch run − the five twins of record (the second, independent read of the candidate's in-distribution gain)

| variant | slice | Δ rel | Δ tke | Δ mvpe | Δ sps | ΔC | **Δ composite** | per trajectory (composite) |
|---|---|---|---|---|---|---|---|---|
| control | all8 | +0.0166 | +0.0712 | -0.0137 | +0.0698 | +0.0308 | **+0.0311** | 11400_20 +0.0389, 13950_10 +0.0136, 16500_5 +0.0019, 21600_15 +0.0432, 26700_0 -0.0220, 3750_0 +0.0576, 6300_10 +0.0468, 8850_20 +0.0702 |
| control | a15_20 | +0.0239 | +0.1292 | -0.0159 | +0.1127 | +0.0655 | **+0.0508** |  |
| frozen | all8 | +0.0130 | +0.0039 | -0.0195 | +0.0046 | -0.0124 | **+0.0060** | 11400_20 +0.0234, 13950_10 -0.0029, 16500_5 -0.0440, 21600_15 +0.0463, 26700_0 -0.1265, 3750_0 +0.0352, 6300_10 +0.0653, 8850_20 +0.0496 |
| frozen | a15_20 | +0.0282 | +0.0888 | -0.0051 | +0.0716 | +0.0212 | **+0.0397** |  |

**R1: PASS** — package composite (stream, all 8) of the from-scratch run 77.2817 against the campaign's 77.2821: Δ -0.0003 (bar 0.005)

### R1b: the band

| band | median base′ u / v | σ_ref median | q (u, v) | mean log κ u / v | (a) stream composite all 8 / AoA 15-20 | sps | C | mean width | (a) frozen composite all 8 |
|---|---|---|---|---|---|---|---|---|---|
| (a)'s own shipping-way refit | 0.03721 / 0.02669 | 0.00471 | 0.4741, 0.4813 | -2.128 / -2.287 | 77.2817 / 74.8220 | 56.078 | 72.184 | 0.01480 | 76.8576 |
| from-scratch run's band (reproduce.sh dp_band: members + twins) | 0.03719 / 0.02669 | 0.00472 | 0.4741, 0.4813 | -2.128 / -2.287 | 77.2817 / 74.8219 | 56.078 | 72.184 | 0.01480 | 76.8576 |
| campaign twins' band (id_t4all_ship) | 0.03718 / 0.02665 | 0.00471 | 0.4739, 0.4809 | -2.129 / -2.289 | 77.2818 / 74.8219 | 56.078 | 72.185 | 0.01480 | 76.8576 |
| band of record (k27) | 0.03808 / 0.02728 | 0.00481 | 0.4847, 0.4907 | -2.083 / -2.236 | 77.2839 / 74.8220 | 56.087 | 72.195 | 0.01478 | 76.8777 |

### The second candidate's twins (three from-scratch F-FNO twins + the two ViT twins of record)

#### second candidate − the five twins of record

| variant | slice | Δ rel | Δ tke | Δ mvpe | Δ sps | ΔC | **Δ composite** | per trajectory (composite) |
|---|---|---|---|---|---|---|---|---|
| control | all8 | +0.0144 | +0.0809 | +0.0062 | +0.0700 | +0.0263 | **+0.0329** | 11400_20 +0.0480, 13950_10 +0.0264, 16500_5 +0.0203, 21600_15 +0.0476, 26700_0 +0.0054, 3750_0 +0.0505, 6300_10 +0.0257, 8850_20 +0.0396 |
| control | a15_20 | +0.0208 | +0.1057 | +0.0069 | +0.0962 | +0.0509 | **+0.0450** |  |
| frozen | all8 | +0.0125 | +0.0710 | +0.0063 | +0.0496 | +0.0069 | **+0.0259** | 11400_20 +0.0414, 13950_10 +0.0195, 16500_5 +0.0022, 21600_15 +0.0430, 26700_0 -0.0211, 3750_0 +0.0577, 6300_10 +0.0317, 8850_20 +0.0326 |
| frozen | a15_20 | +0.0222 | +0.0889 | +0.0137 | +0.0733 | +0.0249 | **+0.0389** |  |

#### second candidate − the campaign's T4 F-FNO-only twins (id_t4_ship; its reproducibility read)

| variant | slice | Δ rel | Δ tke | Δ mvpe | Δ sps | ΔC | **Δ composite** | per trajectory (composite) |
|---|---|---|---|---|---|---|---|---|
| control | all8 | -0.0002 | +0.0006 | -0.0001 | -0.0012 | -0.0015 | **-0.0003** | 11400_20 -0.0010, 13950_10 +0.0000, 16500_5 +0.0003, 21600_15 -0.0011, 26700_0 +0.0004, 3750_0 +0.0011, 6300_10 -0.0018, 8850_20 -0.0005 |
| control | a15_20 | -0.0003 | -0.0000 | -0.0006 | -0.0027 | -0.0028 | **-0.0009** |  |
| frozen | all8 | -0.0003 | +0.0002 | -0.0003 | -0.0045 | -0.0052 | **-0.0013** | 11400_20 -0.0006, 13950_10 -0.0015, 16500_5 -0.0013, 21600_15 -0.0043, 26700_0 -0.0027, 3750_0 +0.0021, 6300_10 -0.0023, 8850_20 +0.0001 |
| frozen | a15_20 | -0.0004 | -0.0000 | -0.0008 | -0.0053 | -0.0062 | **-0.0016** |  |


**Why R1 is so close.**
* Seeded training reproduces the ViTs bit for bit on the same hardware class and stack. The F-FNO twins differ only
  through nondeterministic kernels.
* The organisers retrain on 8× A800, where the ViTs will not be bit identical either.
* So the relevant scale for their retrain is the control-to-control reseeding spread measured in chunk 5 (0.012 in
  distribution), not this −0.0003.

**R1 one level down.** Single twins at α 0.80 on val_unseen, from-scratch − campaign (rel / tke / mvpe):
* p2_ffno s1: −5.7e-7 / −8.4e-6 / +3.5e-6
* p2_ffno s2: −2.4e-7 / +4.9e-6 / −6.3e-7
* p2_ffno s3: −5.2e-7 / +2.2e-5 / −4.1e-6
* both ViTs: exactly 0

No member carries a miss (`runs/member_repro_vs_campaign.json`).

## R2: a different card, and the forced fallbacks

**Load-time choices.** The A100 (Ascend) and the V100S (Pitzer, `gpu-exp`) make the same choices for all three Track-2
archives:
* fp16 accepted by the package's own check;
* CUDA graph captured and replaying bit-identically (graph_worst 0.0);
* arena on.

The only card difference is Track 1's batch probe: 56 on the A100, 28 on the V100, for all three archives, as intended.

**Candidate against the uploaded package.** They load exactly alike. The only differences are the code changes of 09-28:
* the uploaded Track-1 package still prints its removed knobs;
* the uploaded Track-2 package lacks the non-finite guards. The harness's `D_g_nan_input_INFO` test (a NaN input) returns
  a non-finite prediction for the uploaded package on both cards and for neither candidate.

**The gates.**
* `validate_t2.py --adaptive` reads SUBMITTABLE for every archive on both cards.
* The Track-1 archive check reads SUBMITTABLE for every archive on both cards. That check is `t1_check.py`:
  `validate_strict.py`'s harsh load and contracts applied to the shipped archive, since `validate_strict.py` rebuilds its
  zip from checkpoints.

**The forced fallbacks.** On the A100, one at a time, through the sprint's harness: eager graph, fp32, eager + fp32, and no
pinned arena.
* **Eager and no-arena** are bit-identical to the unforced run.
* **fp32** differs by at most 2.9e-3 per-window relative L2 (the package's own fp16 bar is 2e-2).
* **V100 against A100** differs by at most 2.4e-3.
* The element tails are the same for the three archives: 0.03-0.06 % of prediction elements move by more than 1e-2.
* On the composite, every forced fallback and the other card move it by at most 0.0006.

**Criterion met:** every fallback returns the same numbers within the tolerance the package accepts, only slower, and
nothing differs between the candidate and the uploaded package in how it loads.

**Harness correction.** My first Track-1 fp32 run did not actually force fp32:
* It set `AMP` after the import, but Track 1 fixes its autocast dtype at import (`_AMP_DT`). That run read an exact 0
  difference, which I did not accept.
* The re-run sets `_AMP_DT` and does differ.
* Its load line still reports the fp16 chosen at import, because Track 1 loads at import.

Composite (W0, no time) on the whole val_unseen stream, Track 2 through the QA harness (denormalised as the kit does); differences to the unforced A100 run. The members are all-data models, so the level is in sample: only the differences mean something.

| archive | A100 graph (unforced) | V100 graph | A100 eager | A100 fp32 | A100 eager+fp32 | A100 no arena | Track 1: A100 / V100 − A100 / A100 fp32 − A100 |
|---|---|---|---|---|---|---|---|
| record | 79.0948 | -0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 78.6185 / -0.0000 / -0.0002 |
| cand_all | 78.7598 | -0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 78.2680 / -0.0001 / -0.0006 |
| cand_ffno | 78.9290 | -0.0000 | +0.0000 | -0.0002 | -0.0002 | +0.0000 | 78.4315 / -0.0001 / -0.0005 |

### Gate R2, detail: a different card, and the forced fallbacks

#### uploaded T2_MRGP2 / track1_k27fin_a080 (`record`)

* **a100**: validate_t2 --adaptive **SUBMITTABLE**; Track 1 archive check **SUBMITTABLE** (amp float16); harness load: {'mode': 'graph', 'load_s': 2.052581398980692, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3568988 kB', 'VmRSS': '3256684 kB'}}
  Track 1 load line: `no load line`
  QA harness tests reported RED: graph:D_g_nan_input_INFO, eager:D_g_nan_input_INFO, fp32:D_g_nan_input_INFO, eagerfp32:D_g_nan_input_INFO, pinfail:D_g_nan_input_INFO
* **v100**: validate_t2 --adaptive **SUBMITTABLE**; Track 1 archive check **SUBMITTABLE** (amp float16); harness load: {'mode': 'graph', 'load_s': 1.9038687171414495, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3465244 kB', 'VmRSS': '3153788 kB'}}
  Track 1 load line: `no load line`
  QA harness tests reported RED: graph:D_g_nan_input_INFO

Differences: max |diff| / max relative element difference / largest per-window relative L2 (package's fp16 bar 2e-2)

| comparison | pred; lower; upper |
|---|---|
| V100 − A100, Track 2, two-trajectory stream (stream_eval) | pred 3.16e-03 / 7.41e+00 / 1.25e-03; lower 3.07e-03 / 4.70e+00 / 6.98e-04; upper 2.40e-03 / 2.77e+00 / 6.80e-04 |
| V100 − A100, Track 2, whole val_unseen stream (harness, graph) | pred 4.54e-02 / 9.47e+00 / 9.64e-04; lower 8.74e-02 / 1.85e+01 / 2.42e-03; upper 7.54e-02 / 2.03e+01 / 2.27e-03 |
| V100 − A100, Track 1, val_unseen (336 windows) | pred 4.83e-03 / 9.61e+00 / 1.06e-03; lower 1.86e-03 / 3.73e+00 / 3.56e-04; upper 1.67e-03 / 2.04e+00 / 2.81e-04 |
| A100, forced eager − unforced, Track 2 whole stream | pred 0.00e+00 / 0.00e+00 / 0.00e+00; lower 0.00e+00 / 0.00e+00 / 0.00e+00; upper 0.00e+00 / 0.00e+00 / 0.00e+00 |
| A100, forced fp32 − unforced, Track 2 whole stream | pred 7.30e-02 / 1.48e+01 / 1.29e-03; lower 8.92e-02 / 2.54e+01 / 2.54e-03; upper 7.58e-02 / 2.41e+01 / 2.69e-03 |
| A100, forced eagerfp32 − unforced, Track 2 whole stream | pred 7.30e-02 / 1.48e+01 / 1.29e-03; lower 8.92e-02 / 2.54e+01 / 2.54e-03; upper 7.58e-02 / 2.41e+01 / 2.69e-03 |
| A100, forced pinfail − unforced, Track 2 whole stream | pred 0.00e+00 / 0.00e+00 / 0.00e+00; lower 0.00e+00 / 0.00e+00 / 0.00e+00; upper 0.00e+00 / 0.00e+00 / 0.00e+00 |
| A100, Track 1 forced fp32 − unforced | pred 3.33e-03 / 9.54e+00 / 1.42e-03; lower 3.33e-03 / 9.96e+00 / 5.71e-04; upper 2.71e-03 / 3.52e+00 / 4.56e-04 |

* A100 forced eager: {'mode': 'eager', 'load_s': 1.9732852729503065, 'graph': False, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': "RuntimeError('QA: capture disabled')", 'GRAPH_WORST': None, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 
* A100 forced fp32: {'mode': 'fp32', 'load_s': 1.9008902970235795, 'graph': True, 'AMP_ACTIVE': None, 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3063932 kB', 'VmRSS': '2977428 k
* A100 forced eagerfp32: {'mode': 'eagerfp32', 'load_s': 1.8957078469684348, 'graph': False, 'AMP_ACTIVE': None, 'GRAPH_ERR': "RuntimeError('QA: capture disabled')", 'GRAPH_WORST': None, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', '
* A100 forced pinfail: {'mode': 'pinfail', 'load_s': 1.3802164739463478, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': False, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '1998460 kB', 'VmRSS': '

#### candidate T4 all (from scratch) (`cand_all`)

* **a100**: validate_t2 --adaptive **SUBMITTABLE**; Track 1 archive check **SUBMITTABLE** (amp float16); harness load: {'mode': 'graph', 'load_s': 2.1552607259945944, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3570836 kB', 'VmRSS': '3272768 kB'}}
  Track 1 load line: `no load line`
* **v100**: validate_t2 --adaptive **SUBMITTABLE**; Track 1 archive check **SUBMITTABLE** (amp float16); harness load: {'mode': 'graph', 'load_s': 1.9266614679945633, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3470748 kB', 'VmRSS': '3174244 kB'}}
  Track 1 load line: `no load line`

Differences: max |diff| / max relative element difference / largest per-window relative L2 (package's fp16 bar 2e-2)

| comparison | pred; lower; upper |
|---|---|
| V100 − A100, Track 2, two-trajectory stream (stream_eval) | pred 4.46e-03 / 6.72e+00 / 1.17e-03; lower 4.05e-03 / 5.37e+00 / 7.31e-04; upper 2.88e-03 / 3.96e+00 / 7.37e-04 |
| V100 − A100, Track 2, whole val_unseen stream (harness, graph) | pred 4.61e-02 / 5.04e+00 / 1.02e-03; lower 5.76e-02 / 3.11e+01 / 2.32e-03; upper 8.95e-02 / 3.82e+01 / 2.06e-03 |
| V100 − A100, Track 1, val_unseen (336 windows) | pred 3.07e-03 / 7.37e+00 / 1.01e-03; lower 2.43e-03 / 2.88e+00 / 3.48e-04; upper 2.65e-03 / 1.41e+00 / 2.83e-04 |
| A100, forced eager − unforced, Track 2 whole stream | pred 0.00e+00 / 0.00e+00 / 0.00e+00; lower 0.00e+00 / 0.00e+00 / 0.00e+00; upper 0.00e+00 / 0.00e+00 / 0.00e+00 |
| A100, forced fp32 − unforced, Track 2 whole stream | pred 6.34e-02 / 1.45e+01 / 1.33e-03; lower 8.57e-02 / 3.52e+01 / 2.77e-03; upper 1.38e-01 / 4.08e+01 / 2.91e-03 |
| A100, forced eagerfp32 − unforced, Track 2 whole stream | pred 6.34e-02 / 1.45e+01 / 1.33e-03; lower 8.57e-02 / 3.52e+01 / 2.77e-03; upper 1.38e-01 / 4.08e+01 / 2.91e-03 |
| A100, forced pinfail − unforced, Track 2 whole stream | pred 0.00e+00 / 0.00e+00 / 0.00e+00; lower 0.00e+00 / 0.00e+00 / 0.00e+00; upper 0.00e+00 / 0.00e+00 / 0.00e+00 |
| A100, Track 1 forced fp32 − unforced | pred 5.02e-03 / 1.46e+01 / 1.24e-03; lower 4.71e-03 / 3.45e+00 / 5.77e-04; upper 3.19e-03 / 1.94e+00 / 4.57e-04 |

* A100 forced eager: {'mode': 'eager', 'load_s': 2.1168442550115287, 'graph': False, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': "RuntimeError('QA: capture disabled')", 'GRAPH_WORST': None, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 
* A100 forced fp32: {'mode': 'fp32', 'load_s': 2.177188728004694, 'graph': True, 'AMP_ACTIVE': None, 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3076172 kB', 'VmRSS': '2993684 kB
* A100 forced eagerfp32: {'mode': 'eagerfp32', 'load_s': 2.2405992520507425, 'graph': False, 'AMP_ACTIVE': None, 'GRAPH_ERR': "RuntimeError('QA: capture disabled')", 'GRAPH_WORST': None, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', '
* A100 forced pinfail: {'mode': 'pinfail', 'load_s': 1.5477286810055375, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': False, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '2007948 kB', 'VmRSS': '

#### candidate T4 F-FNO only (`cand_ffno`)

* **a100**: validate_t2 --adaptive **SUBMITTABLE**; Track 1 archive check **SUBMITTABLE** (amp float16); harness load: {'mode': 'graph', 'load_s': 2.0464543849229813, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3574124 kB', 'VmRSS': '3276904 kB'}}
  Track 1 load line: `no load line`
* **v100**: validate_t2 --adaptive **SUBMITTABLE**; Track 1 archive check **SUBMITTABLE** (amp float16); harness load: {'mode': 'graph', 'load_s': 1.9139632389415056, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3467928 kB', 'VmRSS': '3171456 kB'}}
  Track 1 load line: `no load line`

Differences: max |diff| / max relative element difference / largest per-window relative L2 (package's fp16 bar 2e-2)

| comparison | pred; lower; upper |
|---|---|
| V100 − A100, Track 2, two-trajectory stream (stream_eval) | pred 3.43e-03 / 6.46e+00 / 1.21e-03; lower 2.44e-03 / 7.48e+00 / 7.63e-04; upper 2.31e-03 / 4.31e+00 / 7.64e-04 |
| V100 − A100, Track 2, whole val_unseen stream (harness, graph) | pred 4.66e-02 / 9.09e+00 / 9.84e-04; lower 6.22e-02 / 2.52e+01 / 2.11e-03; upper 8.50e-02 / 2.78e+01 / 2.32e-03 |
| V100 − A100, Track 1, val_unseen (336 windows) | pred 3.38e-03 / 9.94e+00 / 1.06e-03; lower 2.23e-03 / 5.43e+00 / 3.47e-04; upper 1.83e-03 / 1.76e+00 / 2.81e-04 |
| A100, forced eager − unforced, Track 2 whole stream | pred 0.00e+00 / 0.00e+00 / 0.00e+00; lower 0.00e+00 / 0.00e+00 / 0.00e+00; upper 0.00e+00 / 0.00e+00 / 0.00e+00 |
| A100, forced fp32 − unforced, Track 2 whole stream | pred 4.93e-02 / 1.27e+01 / 1.31e-03; lower 1.31e-01 / 3.59e+01 / 2.43e-03; upper 1.01e-01 / 2.63e+01 / 2.75e-03 |
| A100, forced eagerfp32 − unforced, Track 2 whole stream | pred 4.93e-02 / 1.27e+01 / 1.31e-03; lower 1.31e-01 / 3.59e+01 / 2.43e-03; upper 1.01e-01 / 2.63e+01 / 2.75e-03 |
| A100, forced pinfail − unforced, Track 2 whole stream | pred 0.00e+00 / 0.00e+00 / 0.00e+00; lower 0.00e+00 / 0.00e+00 / 0.00e+00; upper 0.00e+00 / 0.00e+00 / 0.00e+00 |
| A100, Track 1 forced fp32 − unforced | pred 3.58e-03 / 1.02e+01 / 1.29e-03; lower 3.58e-03 / 6.59e+00 / 5.97e-04; upper 2.76e-03 / 1.76e+00 / 4.76e-04 |

* A100 forced eager: {'mode': 'eager', 'load_s': 2.0009805851150304, 'graph': False, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': "RuntimeError('QA: capture disabled')", 'GRAPH_WORST': None, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 
* A100 forced fp32: {'mode': 'fp32', 'load_s': 1.9457199180033058, 'graph': True, 'AMP_ACTIVE': None, 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '3070564 kB', 'VmRSS': '2992540 k
* A100 forced eagerfp32: {'mode': 'eagerfp32', 'load_s': 1.8995982469059527, 'graph': False, 'AMP_ACTIVE': None, 'GRAPH_ERR': "RuntimeError('QA: capture disabled')", 'GRAPH_WORST': None, 'arena': True, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', '
* A100 forced pinfail: {'mode': 'pinfail', 'load_s': 1.4107663908507675, 'graph': True, 'AMP_ACTIVE': 'float16', 'GRAPH_ERR': None, 'GRAPH_WORST': 0.0, 'arena': False, 'mem': {'VmLck': '0 kB', 'VmPin': '0 kB', 'VmHWM': '1996700 kB', 'VmRSS': '


## The expectation table for the owner

**In distribution: an expectation for the development board, not a registered prediction.** It is translated with the
evaluate-change skill's §4 rules:
* a local error cut is halved;
* the containment part of Δsps is never banked;
* time is unchanged;
* the bracket runs from a quarter of the accuracy parts (containment 0) to the local read in full.

Sources:
* **T4 all:** the candidate's own from-scratch twins (a) against the twins of record (c).
* **T4 F-FNO only:** the second candidate's twins against (c).
* **PREV_SPACE:** target against the uploaded package's `input`, on the twins with the band of record.

**Out of distribution:** the chunk-5 decision table, unchanged. These are the campaign's regime ensembles; the from-scratch
run has no held-out-regime ensembles.

| change | slice | local Δ rel / tke / mvpe / sps (containment part) | expected Δ rel / tke / mvpe / sps | expected Δ composite [low, high] |
|---|---|---|---|---|
| T4 all | all 8 | +0.017 / +0.071 / -0.014 / +0.070 (+0.025) | +0.008 / +0.036 / -0.007 / +0.022 | **+0.0125** [+0.0062, +0.0311] |
| T4 all | AoA 15-20 | +0.024 / +0.129 / -0.016 / +0.113 (+0.051) | +0.012 / +0.065 / -0.008 / +0.031 | **+0.0191** [+0.0095, +0.0508] |
| T4 F-FNO only | all 8 | +0.014 / +0.081 / +0.006 / +0.070 (+0.021) | +0.007 / +0.040 / +0.003 / +0.025 | **+0.0139** [+0.0069, +0.0329] |
| T4 F-FNO only | AoA 15-20 | +0.021 / +0.106 / +0.007 / +0.096 (+0.038) | +0.010 / +0.053 / +0.003 / +0.029 | **+0.0178** [+0.0089, +0.0450] |
| PREV_SPACE target vs the uploaded input | all 8 | -0.000 / +0.000 / -0.000 / +0.026 (+0.026) | -0.000 / +0.000 / -0.000 / -0.000 | **-0.0000** [-0.0000, +0.0064] |
| PREV_SPACE target vs the uploaded input | AoA 15-20 | -0.000 / +0.000 / -0.000 / +0.011 (+0.011) | -0.000 / +0.000 / -0.000 / -0.000 | **-0.0000** [-0.0000, +0.0027] |

Out of distribution, measured (stream, Δ composite to the mirror control; PREV_SPACE: control − prev_input):

| change | A20 | A10 | relo | rehi | rein | OOD mean |
|---|---|---|---|---|---|---|
| T4 all | +0.0527 | +0.0362 | +0.7090 | +0.0247 | +0.0195 | +0.2056 |
| T4 F-FNO only | +0.0516 | +0.0395 | +0.3335 | +0.0458 | +0.0245 | +0.1176 |
| PREV_SPACE | +0.0020 | +0.0066 | +0.0071 | +0.0040 | +0.0086 | +0.0049 |


**How to read it.** Against the package of record, a candidate carries its recipe's line plus the PREV_SPACE line.
* **In distribution** both recipes are expected at about +0.013 (all 8) to +0.019 (AoA 15-20), with every bracket above
  0. They are indistinguishable there, except that T4 all costs mvpe.
* **PREV_SPACE** moves only the interval. The rules therefore bank nothing for it (0 [0, +0.006]), although its
  mechanism is a certain fix: the board does normalise the revealed target with the target statistics.
* **Out of distribution** T4 all reads +0.206 and T4 F-FNO only +0.118. The difference is relo alone (+0.38); at rehi T4
  all is 0.02 lower.

## Ledger, verified, not verified

* **GPU:** chunk 6 used 0.62 GPU-h on Ascend (11 jobs, all COMPLETED) and 0.09 GPU-h on Pitzer V100 (3 jobs). All of the
  gauge's jobs in the campaign: 99 Ascend jobs, 10.20 GPU-h; 5 Pitzer jobs, 0.14 h (two of them CPU).
* **Verified:**
  * R1, R1b and the second candidate's read are on the real package code, with every band refitted as stated.
  * The second candidate was built by the release itself (a clean clone of 467769d, the stage names of §14). It ended
    `BUILD_FINAL OK` and left the clone clean.
  * R2 ran the three archives as shipped (sha256 in the job logs) on two card types, with the four fallbacks forced one at
    a time.
  * The from-scratch twins are genuinely retrained. The files differ; the F-FNO weights differ; the ViT weights are
    identical. The checkpoints record commit 467769d.
* **Not verified:**
  * R1 compares twins, not all-data members: the candidate's all-data members have no campaign counterpart. The
    member-level agreement of the twins is the evidence that the same recipe reproduces.
  * The organisers' 8× A800 and their stack: R2 covers the A100 and the V100 with torch 2.2.2 only.
  * The development-board figures are rule-based expectations.
  * The out-of-distribution rows come from the campaign's ensembles, one per recipe per regime.
  * Nothing was copied into `sub_candidates`, and nothing is staged.
