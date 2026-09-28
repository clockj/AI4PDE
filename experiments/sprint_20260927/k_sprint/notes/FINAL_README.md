# K-sprint FINAL candidates — 2026-09-27 (predictions registered 17:50 EDT, BEFORE any upload)

Source tree: `/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_fin` (local commits on 55f0e88, never pushed);
full patch for the owner: `FINAL/k27fin_vs_55f0e88.patch`. Build job `jobs/rp_k27_final.sh`, log `logs/rp_k27_final_7567468.log`.
Members are UNCHANGED (recipe A: 4x `ffno_all_w40l8_sim_s1..s4` + `arch_vit_all_s1` + `arch_vit_all_cmb3sim_s3`), ALPHA 0.80.

## What changed against today's planned packages (T1 "27lz", T2 "asnp [+D1c]")
| lever | tracks | what | held-out evidence (twins, val_unseen) | predicted board d final |
|---|---|---|---|---|
| K0 SIGBLEND band (lam 0.5) | T1 (EBAND), T2 (SIGPOW) | `half = base' clip(s/s_ref)^p' (s+1e-8)^(lam q)`, a geometric blend of the shipped per-element band with a 42-number sigma power law fitted on held-out errors | T1 dC x100 LOTO: AoA 15-20 +0.03 (2/3), all 8 +0.48 (7/8) at f 1.29; T2 with R1 on: sps +0.22 (8/8) / +0.07 (3/3) | T1 +0.025 [0, +0.09]; T2 +0.017 [+0.005, +0.055] |
| K2 JOINT_GAIN | both | one amplitude gain for u and v in the ALPHA compose (same TKE map, same time-mean, prediction closer to y0) | rel error -0.46 % (8/8), tke / mvpe 0.0000 % | +0.014 [+0.005, +0.025] per track |
| K1 latency | T1: WARM_TAILS, PIN_OUT 128, BATCH_CAP 56; T2: ARENA_STEPS 1024 | page-locked output arenas, chunk cap, tail warm-up | outputs md5-identical; T1 warm -0.13..-0.40 ms/window, T2 median -0.40 ms/step (exclusive A100) | T1 +0.02 [0, +0.06]; T2 +0.016 [0, +0.05] |
| (T1 only) SMALL_GRAPH off | T1 | EBAND needs the GPU-band path at every N | N = 1..336 edge cases pass | 0 (evaluator N is 14-99) |
Closed by this sprint (nothing shipped): K3 warm-start polish arms (AoA-15/20 specialists, over-sampling, control) — best arm
+0.0125 local, specialists -0.22..-0.42 on the full split; zoo re-composition; 14 zero-training point-prediction tricks
(`notes/design_2.md` section 3); band centre gain > 1, spatial smoothing of the centre (`logs/k0_*`).

## Predicted subscores (estimate [bracket])
Anchors: T1 09-25 "A" (rel 94.7625 tke 79.5891 mvpe 94.2613 time 91.8773 sps 43.0631 -> 81.6224, 5.70 ms);
T2: the team's registered prediction for asnp + D1c, 81.783 [81.71, 81.86] (rel ~94.78 tke ~79.52 mvpe ~94.18 time ~92.1 sps ~43.63).

| package | rel | tke | mvpe | time | sps | final |
|---|---|---|---|---|---|---|
| **T1 `track1_k27fin_a080.zip`** | 94.786 [94.772, 94.798] | 79.589 [79.58, 79.60] | 94.261 [94.25, 94.27] | 92.01 [91.44, 92.35] | 43.18 [43.04, 43.48] | **81.67 [81.58, 81.79]** |
| **T2 `track2_k27fin_d1c_a080.zip`** | 94.80 [94.77, 94.83] | 79.52 [79.45, 79.60] | 94.18 [94.13, 94.24] | 92.25 [91.6, 92.8] | 43.72 [43.55, 44.0] | **81.83 [81.73, 81.93]** |
| T2 `track2_k27fin_a080.zip` (no D1c) | 94.79 | 79.52 | 94.13 | 92.25 | 43.71 | 81.82 [81.72, 81.92] |

| **T2 MERGED `track2_k27mrg_cons_a080.zip`** (F-sprint CONS = D1c30 + VARCAL 50, plus our ARENA + JOINT_GAIN + SIGPOW; registered 17:56 EDT) | 94.77 [94.73, 94.81] | 79.72 [79.55, 79.85] | 94.17 [94.13, 94.24] | 92.20 [91.6, 92.8] | 43.78 [43.58, 44.05] | **81.84 [81.74, 81.95]** |
| T1 `track1_k27fin2_a080.zip` (= k27fin with PIN_OUT 1024; outputs identical) | as k27fin | | | 92.03 [91.44, 92.40] | | 81.67 [81.58, 81.80] |

Merged-T2 held-out read (twins, asym R1, final without time, vs asnp + D1c30): CONS alone +0.025 (8/8) / +0.013 (3/3);
CONS + our three +0.088 (8/8) / +0.044 (3/3); ours alone (k27fin_d1c) +0.065 / +0.030. SIGPOW fully additive on VARCAL;
JOINT_GAIN keeps its rel gain and costs -0.014 tke score on top of VARCAL (net +0.009 / +0.014). `notes/result_mrg.md`.

Live cut (#10) at 21:39Z: T1 81.885, T2 81.988. **Neither package is predicted to reach the cut by itself**
(P(top 10) roughly 3 % T1, 3 % T2). Free roll: the board keeps the best.

## sha256
    438cd3ffaba930f35ecbc63f8b595d8fd192e331a6e6180e1e8d91e74d5106cd  track1_k27fin_a080.zip
    470f278dd981a8646047f2323d83d60a239e432417576d8400b0ee85c78a7dcd  track2_k27fin_a080.zip
    eca654a2b738d6218175a329e1a2fe44f1af4ed9afd9086fc8e4fa30062ded3f  track2_k27fin_d1c_a080.zip
(`track1_k27fin_d1c_a080.zip` = the same T1 content as `track1_k27fin_a080.zip`; D1c is T2-only.)

## Gates (build job 7567468)
T1: validate_strict SUBMITTABLE, members 6, gpu_band True, joint_gain True/True, EBAND baked, SMALL_GRAPH False.
T2 (both): validate_t2 --adaptive SUBMITTABLE, CAUSALITY OK (176 steps, 0.0), graph CAPTURED graph_worst 0.0, members 6,
sigpow (0.2423, 0.2454), joint_gain True, ARENA_STEPS 1024.
Independent verification of the COMBINATION: `notes/result_ver.md` (executor E-VER).

## FINAL PICKS (commander, 18:02 EDT)
| track | upload | why |
|---|---|---|
| **T1** | `FINAL/track1_k27fin_a080.zip` (438cd3ff...) | all three K levers; PIN_OUT 128 (0.17 GB page-locked); verified GO (E-VER A-D) |
| T1 alternative | `FINAL/track1_k27fin2_a080.zip` (bde1fe66...) | identical outputs, PIN_OUT 1024 (1.32 GB page-locked): only better if the platform REUSES one process for several predict() calls |
| **T2** | `FINAL/track2_k27mrg_cons_a080.zip` (68a53737...) | F-sprint CONS (D1c 30 + VARCAL 50) + ARENA + JOINT_GAIN + SIGPOW band; best held-out read of both sprints (+0.088 / +0.044 vs asnp + D1c) |
| T2 pure K-sprint | `FINAL/track2_k27fin_d1c_a080.zip` (eca654a2...) | asnp + D1c 30 + our three levers, no VARCAL |
| T2 fallback | `FINAL/track2_k27mrg_consnj_a080.zip` (7f226c7a...) | merged without JOINT_GAIN (if the JOINT_GAIN x VARCAL tke interference, -0.014 tke score locally, is judged too risky) |
Load time (A100, fresh process): import 10 s with WARM_TAILS (x27 baseline 4-5 s). Host RSS after load: T1 2.25 GB (PIN 128) /
3.63 GB (PIN 1024); T2 3.65 GB (ARENA_STEPS 1024).
