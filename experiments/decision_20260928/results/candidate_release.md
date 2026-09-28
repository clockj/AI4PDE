# Candidate release engineering (smith, 2026-09-28) — identity PASSED, smoke PASSED

Patch: `experiments/decision_20260928/patches/candidate_release.patch` = `git diff` of `dec0928/wt_cand` against a45a684,
limited to the seven files below. It contains the `train2.py` change of `t4_aug_resim.patch` again: that patch applied
cleanly, and `train2.py` is unchanged between 4e5ed37 and a45a684.

## Hunks
| file | change |
|---|---|
| `Track_1/unet/train2.py` | `--aug-resim` / `--aug-resim-p`: `t4_aug_resim.patch` as it was |
| `Track_1/verify/baked_constants.py` (new) | reads the module-level constants and `band.npz` out of both archives; checks them against what the build was asked for: every `RPDE_SHIP_*` knob (passed, or the tree default when unset), ALPHA, FUSE, AMP, BLEND_BETA, SPEC_PATH, HOST_WRITE, FFNO_PATH, and the band md5. Prints the table; exits 1 on any mismatch |
| `Track_1/submit_fno/build_final.sh` | lock file (host, pid, job; `noclobber`; removed by the exit trap); the knob list becomes one array `SHIP` so the same values are passed and then checked; `baked_constants.py` runs after the gates; a mismatch fails the build. With default arguments the build is otherwise unchanged |
| `reproduce.sh` | new target `decision`; `final` stays the default and every other target is untouched. Variables `RPDE_DP_RESIM_FFNO` / `RPDE_DP_RESIM_VIT` (default 0.25; empty = the record's recipe and names). Stages `dp_twins_vit_p`, `dp_twins_vit_c`, `dp_twins_ffno`, `dp_members_vit_p`, `dp_members_vit_c`, `dp_members_ffno`, `dp_band`, `dp_build`. Names get `_resim`. The stages train with `--ckpt-every 20 --resume` |
| `REPRODUCE.md` | §14, the `decision` target |
| `Track_1/bands/MANIFEST.md` | entry stub for `dp_resim_sigblend_lam0.5.npz`: the exact command, "to be completed when the band exists" |
| `.claude/rules/training.md` | two bullets on `--aug-resim` |

## Build details
* **Band refit (`dp_band`).** Static fit: `fit_band.py --alpha 0.80 --band-p 0.45 --fit-on seen --fit-aoa 15 20 --inflate 1.39 --fill-unscored --centre-alpha 0` (the fit flags are fit_band's defaults, written out). Then `dump_twins.py --val val_unseen` and `make_k0_band.py` (K0_INFL 1.29, K0_NB 16).
* **Build (`dp_build`).** `build_final.sh` with the refitted npz and json, `RPDE_SHIP_PREV_SPACE=target`, tag `dp_resim` (`_SMOKE` appended when epochs ≠ 400).

## Results so far
* **Verifier** (fixed source-order check), job 7575926 on an A100: **ALL CHECKS PASSED**. Offsets 1272 < 2161 < 5925 < 17139 < 17814.
* **Constant check, dry run on the uploads (CPU).** 39 of 41 checks OK. The two mismatches are `PREV_SPACE` and `STATS_MASK`: they are absent from the uploaded files because those knobs were added in the 09-28 refactor. A rebuild carries them, so this is expected. The result shows that the checker detects an absent or different constant.
* **Identity: PASS** (`rp_dp_cand_ident`, Ascend 7575926, A100, 4 min 14 s). `build_final.sh` with the record's checkpoints and both uploads as references:
  * exit 0, gates 2/2 SUBMITTABLE, `CAUSALITY OK` (176 steps, max |diff| 0);
  * `BAKED CONSTANTS OK: 41 checks, 0 mismatches`;
  * data BYTE-IDENTICAL on both tracks (the code files differ, as expected since the 09-28 refactor);
  * `FINAL_IDENTITY OK`: outputs IDENTICAL at tolerance 0 in all three comparisons;
  * `BUILD_FINAL OK`; the lock file was removed and the three sed-edited files were restored.
  * Negative tests: a held lock gave exit 1 with `REFUSING: another build holds …`; no build ran and the foreign lock was left in place. Checking the built archives against a request of `R1=False` + `PREV_SPACE=target` with the tree's defaults gave `BAKED CONSTANTS FAILED: 41 checks, 30 mismatches`, exit 1.
* **Smoke: PASS** (`rp_dp_cand_smoke`, Ascend 7575939, A100, 24 min 40 s ≈ 0.41 GPU-h). Ran `reproduce.sh decision` at 2 epochs from the copy `dec0928/cand/smoke/repo`, which has no `../sub_candidates` beside it.
  * All 11 `_resim` trainings finished, about 50 % of windows augmented.
  * `dp_band` wrote the static fit, the twin dump and the SIGBLEND files. The band md5 is dfaf5c63008a; the npz and json are in `cand/smoke/log/decision/band/k0/`.
  * `dp_build` passed both gates (2/2 SUBMITTABLE) and `CAUSALITY OK` (176 steps, max |diff| 0).
  * `BAKED CONSTANTS OK: 41 checks, 0 mismatches`, including `PREV_SPACE 'target'` and the refitted band's md5.
  * `BUILD_FINAL OK`; exit 0 in 1479 s. The tree's diff is unchanged after the build.
  * Archives: `track1_dp_resim_SMOKE.zip` (sha256 62e6c364…) and `track2_dp_resim_SMOKE.zip` (sha256 0ee1de9e…), in `cand/smoke/log/decision/pkg/`. **Not candidates.**
* **Seconds per epoch, A100, measured in the smoke:**

| model | all-data member | held-out twin |
|---|---|---|
| F-FNO + sim | 39 | 34–35 |
| ViT-C + sim | 39 | 35 |
| ViT-P | 22 | 20 |

## Job plan (prepared, not submitted; all pass `sbatch --test-only`)
* **Scripts.** `dec0928/cand/repro/jobs/rp_dp_repro_{twins_vit_p,twins_vit_c,twins_ffno_s1..3,members_vit_p,members_vit_c,members_ffno_s1..4}.sbatch` and `rp_dp_repro_build.sbatch`. Submit all with `CLONE=<clean clone> bash dec0928/cand/repro/jobs/submit_repro.sh`: the 11 trainings, then the build with afterok on all of them.
* **Job settings.** Ascend nextgen, `-c 8`, `--requeue`. Time limits: F-FNO / ViT-C members 6:00, their twins 5:30, ViT-P 3:30 / 3:00, build 1:30.
* **Expected GPU-hours: about 42.9 for the trainings + about 0.5 for the build ≈ 43 A100 GPU-h**, at the smoke's measured seconds per epoch. Every job fits its time limit: F-FNO / ViT-C members about 4.4 h against 6:00, ViT-P about 2.5 h against 3:30 / 3:00.

## Open points
1. `run()` in reproduce.sh overwrites a stage's trainlog. On a requeue the job scripts copy the previous log to `.attemptN` first.
2. `baked_constants.py` does not check the removed knobs `SMALL_GRAPH*`; build_round13 refuses them unless they hold their defaults.
3. If a sim-map cache is missing in the clone's environment, four jobs would build it concurrently. The two caches the plan needs exist (`simmap_p1_f0.npy` for all-data, `x476193e300` for VAL_TRAJ).
4. The static band's error field is in-sample for the twins (SEEN AoA 15–20), as the record's is (ood_chunk2 part A). This is kept deliberately.
5. Other `rp_*` jobs are running, so every build passes `RPDE_FORCE=1`, and ms/window from these builds is not latency evidence.

## GPU-hours used
Identity 0.07 + smoke 0.41 = **0.48 A100 GPU-h** (budget 4).
