# Final-day sprint, 2026-09-27 (16:03 – 19:59 EDT)

Two Claude Code sessions worked in parallel on OSC Ascend during the last four hours of Main Development. This
directory keeps what each experiment asked, how it was run and what it returned. Code that shipped is in the main
tree (`Track_1/submit_fno/`, `Track_2/submit/`, `Track_1/sigblend/`); code that did not ship is kept here as patches
against `55f0e88`, never merged into the live modules.

| session | scratch area (disposable) | role | GPU-hours | jobs |
|---|---|---|---|---|
| F-sprint | `/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/` | commander, 3 design agents (A, B, C), 6 executors | 40.9 | 148 |
| K-sprint | `/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/` | commander, 4 designs (K0 – K3), 6 executors | 26.6 | 143 |

`jobs_sacct_20260927.txt` is the Slurm accounting record of every job of both sessions.

## Outcome

| track | package | board `final` | registered before upload |
|---|---|---|---|
| Track 1 | `track1_k27fin_a080.zip` (K-sprint) | **81.671149** | 81.67 [81.58, 81.79] |
| Track 2 | `T2_MRGP2_track2_MRGP2.zip` (both sessions, merged) | **82.055553** | 82.000 [81.885, 82.219] |

Both rebuild from the repository with `bash Track_1/submit_fno/build_final.sh`.

## Experiments

Every criterion below was written before its result. "Composite" is the fitted `final` without the `time` term.
Reads are on the held-out twins; "AoA-20" and "AoA-10" are models trained without that angle of attack.

| id | question | criterion | result | verdict | files |
|---|---|---|---|---|---|
| A | Does a cross-stitched F-FNO × ViT hybrid (`ffvit`), warm-started from both parents with zero-initialised links, beat the two-family ensemble? | ≥ +0.20 on AoA 15–20 | best row −0.004; training loss −7…−15 % while held-out rel-L2 rose 0.1036 → 0.1059 | **killed** | `f_sprint/design_a_ffvit/`, `notes/design_a.md`, `notes/result_a.md` |
| B | Does a band base cross-fitted on ~30 out-of-fold trajectories beat the base fitted on 3? | ΔC×100 ≥ +0.20 with 3/3 trajectories | +0.180, 1/3 trajectories | **killed** | `f_sprint/design_b_xfit/`, `notes/design_b.md`, `notes/result_b.md` |
| B2 | Track 2: smooth R1's running table before its argmax (TSM) and add a lead-time-block table (BLK)? | errors identical, sps up, ≤ 0.1 ms/step | composite +0.023 in distribution (8/8), +0.051…+0.062 AoA-20 (16/16); +0.23 ms/step measured by QA | **shipped**, latency rule overridden and charged | `f_sprint/design_b2_r1tables/`, `notes/design_b2.md`, `notes/b2.patch` |
| C | Track 2: causal per-trajectory corrections from revealed targets — D1c (mean-field bias), VARCAL (fluctuation amplitude), FAMW (family weights) | per component: ≥ 0 on both splits, ≥ 6/8 trajectories, ≤ 0.15 ms | D1c 30 + VARCAL 50: +0.052 in distribution, +0.268 AoA-20, +0.333 AoA-10; FAMW −0.004 on AoA 15–20; SIGEMA −0.553 | D1c + VARCAL **shipped**; FAMW, SIGEMA off | `f_sprint/design_c_stream/`, `notes/design_c.md`, `notes/result_c.md` |
| C2 | Replace D1c's fixed shrink by an empirical-Bayes factor and give the shift a rank-1 lead-time profile? | ≥ +0.05 out of distribution over the best knob-only stack, ≤ 0.03 lost on AoA 15–20 | +0.062 in distribution, +0.401 AoA-20, +0.521 AoA-10 (18/18) against asymmetric R1 | **shipped** | `notes/design_c2.md`, `notes/c2.patch`, `notes/ood_aoa10.md` |
| T1 probe | Can stacking or tree-gating the six members per window lower Track 1's error? | — | hindsight ceiling −0.67 % rel error; leave-one-out −0.06…−0.11 % | **closed** | `f_sprint/t1_stacking_probe/` |
| K0 | SIGBLEND: blend the per-element band with a σ power law fitted on held-out errors (λ 0.5) | sps up on all 8, not down on AoA 15–20, with R1 on | ΔC×100 +0.48 (7/8) leave-one-out; with R1 sps +0.218 (8/8) | **shipped** on both tracks | `k_sprint/notes/design_0.md`, `result_0a.md`, `result_0b.md`, `Track_1/sigblend/` |
| K1 | Host-side latency: WARM_TAILS, PIN_OUT, BATCH_CAP (T1), output arena (T2) | outputs md5-identical | T1 cold −0.65…−0.98 ms/window, T2 −0.40 ms/step for the first 1 024 steps | **shipped** | `k_sprint/k1_latency/`, `notes/design_1.md`, `result_1.md` |
| K2 | JOINT_GAIN: one amplitude gain for u and v in the compose | rel error −0.3 % with tke, mvpe within 0.01 % | rel error −0.462 % (8/8), tke and mvpe unchanged | **shipped** | `k_sprint/notes/design_2.md`, `result_2.md`, `Track_1/verify/joint_gain.py` |
| K3 | Warm-start polish: AoA 15/20 specialists, over-sampling, dropout off | ≥ +0.04 on AoA 15–20 and ≥ +0.02 over control | best arm +0.0125; specialists −0.22…−0.42 on the full split | **closed** | `k_sprint/notes/design_3.md`, `result_3.md` |
| merge | Do the two sessions' levers add? | every gate green and ≥ F's package in all three regimes | +0.057 / +0.029 / +0.034 (in distribution / AoA 15–20 / AoA-20) | **shipped** as MRGP2 | `notes/result_c.md` §T2_MRG_P2, `k_sprint/notes/result_mrg.md`, `f_sprint/qa/` |

## How to read the directories

* `f_sprint/notes/`, `k_sprint/notes/` — the designs and results as the agents wrote them, with their own time stamps.
  `BRIEF.md` / `BRIEF_K.md` are the rules both sessions worked under.
* `*/jobs/` — every Slurm script, as submitted. Paths point into the scratch areas above.
* `f_sprint/notes/t2_p2_vs_HEAD_55f0e88.patch`, `k_sprint/patches/k27fin_vs_55f0e88.patch` — what each session added
  to `55f0e88`. Their union is what this repository now holds.
* `f_sprint/qa/` — the adversarial harness of the independent QA executor: stream lengths 1 – 9 900, masked and
  constant fields, the arena transition at step 1 024, refused page-locked allocation, eager and fp32 fallbacks.

## Known defects carried into the Decision phase

* The official Track-2 runner normalises `prev_target` with the target statistics; R1's unscored-element test assumes
  the input statistics, so the test never fires (about −0.0015 `final`). `f_sprint/qa/qa_normcheck.py`.
* The output arena holds 1 024 steps. A longer run falls back to fresh copies, which is correct and slower.
* The SIGBLEND constants were fitted on the eight validation trajectories; their leave-one-out read is the evidence.

## Verifiers of the sprint

`k_sprint/verify/` holds the one-off verifiers the K-sprint wrote (`joint_gain.py`, `k0a_edge.py`, `k0a_lat.py`,
`k0a_score.py`). They rebuild their variants by patching the tree of `55f0e88`, so they run against that tree, not
against the current one. The two that stay useful, `k0a_dump.py` and `sigblend.py`, are in `Track_1/verify/`.
