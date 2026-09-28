# Repository layout audit — `realpde/main` @ `qwu/agent` (55f0e88 + dirty tree), 2026-09-27

Read-only audit. Method: `git ls-files` (359 tracked files) plus `git status -uall`; AST import scan of every
`.py` (repo-local imports, `importlib` targets, every `sys.path.insert`); grep of every shell and doc file for
invocations. Everything below is from those reads. Out of scope, as briefed: the internals of
`Track_1/unet/*.py`, `submit_fno/{submission,_fno}.py` and `Track_2/submit/submission.py`.

**Headline findings**

1. **The final recipe is not in git.** `Track_1/submit_fno/build_final.sh` (builds the 09-27 uploads: T1 81.671149,
   T2 82.055553), its defaults `Track_1/bands/k27_sigblend_lam0.5.npz` and `Track_1/sigblend/band_k0_sigblend_lam0.5.json`,
   the whole `Track_1/sigblend/` directory, `Track_1/bands/r38_a080_c0_fill.npz`, and the verifiers
   `Track_1/verify/{k0a_dump,k0a_edge,k0a_lat,k0a_score,joint_gain,sigblend}.py` are all **untracked**. The live files `submit_fno/{_fno.py,submission.py,build_round13.sh}`, `Track_2/submit/submission.py` and (since this
   audit started, concurrently edited) `board_rounds.py`, `final_v9.py`, `build_final.sh` are **modified and uncommitted**. Commit these first; every other step below depends on it.
2. **No doc describes the final operating point.** `CLAUDE.md` *Current state* (l.26) is still dated 09-20 (r34, ALPHA
   0.60, 81.418). `README.md` l.9–12 and l.22 and `Track_1/README.md` l.36 and l.136 are older still (81.401, w96 FNOs,
   ALPHA 0.60). No doc except `REPRODUCE.md` mentions ALPHA 0.80, r38, SIGBLEND, `qd_k/`, `sigblend/`, `r1_ref/` or
   `build_final.sh`. `reproduce.sh` gained a `final` target (SIGBLEND refit and `build_final.sh`, l.8-11 and l.384+) **while this audit
   ran**, so it is uncommitted. Re-check `sigblend/dump_twins.py`'s absolute paths against its new `--twins` argument. The ledgers are also stale: `leaderboard/board.jsonl` ends at 09-20 on a PENDING row.
3. **The instrument defaults are further off than CLAUDE.md says.** No local instrument can evaluate the band that
   shipped. `predict_board`, `aoa_holdout_eval`, `quant8` and the others read only `base/sigma_ref/p` from a band
   `.npz`. They ignore `centre_alpha`, and SIGBLEND's σ-power factor lives in a JSON sidecar rather than the npz. The
   SIGBLEND npz is therefore scored **about 3× too wide**, and nothing warns.
4. **`closed/` must stay where it is.** 49 of its 58 scripts (42 in T1, 7 in T2) carry a depth-1 `sys.path` shim, and
   the other 9 are self-contained. Nesting `closed/` deeper breaks all 49.
5. **A whole historical chain is already dead at import.** It loads the v7 kit by absolute path, and
   `~/RealPDE/realpde_t1_starting_kit_v7` no longer exists (only `official/t1_v9`, `official/t2_v6` and
   `realpde_t1_starting_kit_v9` remain). The chain: `eval_wp.py` → `eval_pair.py`, `calib_sigma.py`, `fno_deeponet/*`,
   `submit/fit_band*.py`, `submit/validate_submission.py`, `submit_trick/validate*.py`, `deeponet/eval_deeponet.py`,
   `closed/{diag_spectral,diag_analog,eval_all,tke_ceiling}.py`. So archiving it costs nothing runnable.

---

## 1. File classification

Classes: **LIVE** = pipeline of record · **INSTR** = instrument still useful · **VERIF** = gate or verifier · **LEDGER** ·
**CLOSED** = closed experiment · **HIST** = historical package or sweep · **DOC** = doc, plan or report · **JUNK**.
"U" means untracked.

### Top level
| path | class | evidence |
|---|---|---|
| `reproduce.sh` | LIVE | the end-to-end driver; calls train2, make_val, fit_band, build_round13, qd_k/{export,head_direct,h_ratio_check}, distill_pipeline, verify/pkg_identity |
| `environment.yml` | LIVE | the pinned eval stack (README Setup) |
| `.pyenv.sh` | LIVE (site-specific) | a hard-coded Ascend spack conda path, `.pyenv.sh:2` |
| `.gitignore` | LIVE | it ignores `Track_1/submit_fno/_package*/` and `*.bak`, yet both are tracked (see JUNK) |
| `CLAUDE.md` `README.md` `EVALUATION.md` `ROADMAP.md` `ROUNDS.md` `REPRODUCE.md` | DOC | the stale content is listed under finding 2 and §4 |
| `.claude/skills/{debug-workflow,karpathy-guidelines,rockfish-workflow}/SKILL.md` | JUNK for the release | generic copies of the user-level skills of the same names; `rockfish-workflow` is for a different cluster; nothing references them |

### `Track_1/` top-level `.py`
| file | class | evidence |
|---|---|---|
| `sps9.py` | LIVE | imported by fit_band, predict_board, sigblend/{bandlab,make_k0_band}, qd_k/export, r1_ref, and about 20 instruments and verifiers |
| `make_val.py` | LIVE | `reproduce.sh:232`, `qd_k/run_qdre_train.sh`; builds `val_{seen,unseen}.npz` |
| `board_rounds.py` `final_v9.py` | LIVE (ledger code) | final_v9 is imported by validate_strict, validate_t2, stream_eval, lb.py, predict_board and 11 closed scripts; board_rounds by final_v9, lb.py, predict_board, sub_value |
| `predict_board.py` | LIVE | the pre-upload prediction (CLAUDE *Instruments*); imported by `screen_eval.py` |
| `eval_v9.py` | LIVE (shared lib) | `infer()` is imported by 8 instruments and **20 closed scripts** |
| `re_split.py` | LIVE (lib) | `excluded()` is imported by aoa_holdout_eval, kfold, mean_blend_re, qd_k/re_readout and `qd_k/run_qdre_train.sh` |
| `compare_v9.py` `ens_v9.py` `kfold.py` `aoa_holdout_eval.py` | INSTR | CLAUDE *Instruments*, EVALUATION §4 |
| `mean_blend_re.py` | INSTR | imported by aoa_holdout_eval; re-opened for the private set |
| `alpha_board.py` `quant8.py` `calib_band13.py` `proxy_calib.py` `sub_value.py` | INSTR | Track_1/README; lb.py reads the calib_band13 and proxy_calib constants by text |
| `lat_r15.py` `lat_fast17.py` | INSTR | CLAUDE *Board latency*: "quote both as a bracket" |
| `screen_eval.py` `screen_sets.py` `screen_latency.py` | INSTR (dormant) | the 09-14 backbone screen; screen_eval upserts `leaderboard/screen.jsonl` and reads `plan/screen_*.json` |
| `exchange_v9.py` | CLOSED (not moved yet) | `HERE` is hard-coded absolute (l.7), it uses six round-9 checkpoints (`fno_grad*.pth`), and its result is frozen as CLAUDE *Exchange rate* |
| `eval_wp.py` `eval_pair.py` `calib_sigma.py` | CLOSED / dead | `eval_wp.py:23` loads the missing v7 kit at import; the other two import eval_wp; their only importers are HIST or CLOSED |

### `Track_1/submit_fno/`
| file | class | evidence |
|---|---|---|
| `submission.py` `_fno.py` | LIVE | the package itself (out of scope) |
| `build_round13.sh` | LIVE | the builder; called by build_final.sh, reproduce.sh and sigblend/rp_k27_final.sh |
| `build_final.sh` (U) | LIVE | the 09-27 recipe of record, which is on no doc path |
| `fit_band.py` `build_zip.py` `validate_strict.py` `bake_ffno_path.py` | LIVE | called from build_round13.sh (`fit_band` l.89, `validate_strict` l.235, `bake_ffno_path` l.203); build_zip is used through validate_strict/build_t2 via `_fno_pack` |
| `band_feats.py` | INSTR | the numpy reference for the EBAND features (`_fno.py:2395`, build_round13 l.162) |
| `README.md` | DOC | stale "As of R24/R25"; broken ref at l.67 |
| `build_round10.sh` `build_round11.sh` `build_round12.sh` `finalize_round11.sh` | HIST | superseded builders; round11 calls round10; they record how R10–R12 were built |
| `_package/_fno.py`, `_package/submission.py` | JUNK | a stale build stage (08-29); the path is in `.gitignore`; `build_zip.py:24` `STAGE_DEFAULT` writes here, so a default build dirties tracked files; the code is in git history |
| `band.npy` | JUNK, **removal blocked** | a legacy band; `submission.py:222` `_BAND_LEGACY` falls back to it when `band.npz` is absent (see §6) |

### `Track_1/unet/` (non-`.py` scope only; the `.py` files are classified by role)
| file | class | evidence |
|---|---|---|
| `train2.py` `model.py` `models_alt.py` `simreal.py` | LIVE | reproduce.sh |
| `distill_labels.py` `distill_pipeline.sh` | LIVE-optional | `reproduce.sh:388` (`STUDENT=1`); distillation itself is closed |
| `dinov2_init.py` | CLOSED | DINOv2 init: no gain, now prohibited; not imported by train2 |
| `train.py` | HIST | the v1 trainer; nothing invokes it |
| `runq.sh` `queue.txt` `queue_alldata.txt` `queue_round10.bak` `done.txt` `run_sim.sh` `run_sim2.sh` `run_ctx.sh` | HIST (job records) | the round 9–12 queue driver and arm lists; cited in ROUNDS.md and plan_2026080{8,9}.md; `.bak` is gitignored but tracked |
| `log_*.txt` (51 files, 372 KB) | HIST (records) | round 9–12 training logs; **not duplicated on scratch** (checked `/fs/scratch/.../realpde`), so they are archived, not deleted |

### `Track_1/verify/`
| file | class | evidence |
|---|---|---|
| `blend_path.py` `quant_identity.py` `graph_path.py` `fuse_swin.py` | VERIF (build-gated) | invoked or cited by build_round13.sh (l.239, l.219, l.108, l.230) |
| `pkg_identity.py` | VERIF | `reproduce.sh:421` |
| `cpu_smoke_submission` `amp_path` `ab_identical` `diff_scoring_{official,real}` `fuse_{ffno,vit,osc}` `diff_arch` `diff_ufno` `selfnorm_path` `simmap_modes` `phead_strip` `aug_mask_check` | VERIF | one-off bit-identity gates (EVALUATION §4, ROUNDS); some guard closed features but stay valid `_fno.py` regression gates |
| `k0a_dump.py` `k0a_edge.py` `k0a_lat.py` `k0a_score.py` `sigblend.py` `joint_gain.py` (all U) | VERIF | the gates for the 09-27 SIGBLEND and JOINT_GAIN package; not in git |

### Other Track_1 directories
| path | class | evidence |
|---|---|---|
| `sigblend/{bandlab,dump_twins,make_k0_band}.py`, `README.md`, `band_k0_sigblend_lam0.5.json`, `rp_k27_final.sh` (all U) | LIVE | this makes the shipped band; build_final.sh l.31 reads the json; rp_k27_final.sh is the job record |
| `qd_k/*` (14 files) | INSTR (reproduce `--recipe QD`) | reproduce.sh l.319–331; REPRODUCE §10; QD was not the final recipe, but its derivation is reproducible |
| `bands/*.npz` (17 tracked) and `MANIFEST.md` | LEDGER | reproduce.sh uses r24, t2r13, r34, r35 |
| `bands/k27_sigblend_lam0.5.npz`, `bands/r38_a080_c0_fill.npz` (U) | LEDGER | **the final bands**, not committed (`.gitignore` already whitelists `bands/*.npz`) |
| `closed/*.py` (51) and `README.md` | CLOSED | shimmed; the README has a verdict table |
| `plan/*.md` (25), `report/*` (23) | DOC | round write-ups and plans |
| `plan/screen_{kcurve,sigma,swin_profile}_20260914.json` | LEDGER (live data) | read by `screen_eval.py:67,78,101` and `unet/distill_labels.py` (kcurve); **must not move** |
| `plan/screen_oracle_20260914.json` | LEDGER | read by `closed/screen_oracle.py` |
| `plan/screen_kcurve_ffno_l8_20260914.json` | LEDGER (orphan) | no reader |
| `submit/` (July U-Net package): `README.md`, `plan_submission_20260729.md`, `submission.py`, `build_zip.py`, `fit_band.py`, `fit_band2.py`, `validate_submission.py`, `band.npy`, `_package/model.py` | HIST | fit_band, fit_band2 and validate_submission need the v7 kit (dead); `_package/model.py` is the shipped model snapshot |
| `submit/_package/{band.npy,submission.py}` | JUNK | byte-identical to `submit/band.npy` and `submit/submission.py` (md5 566c5f… / 718166…) |
| `submit/_validate_run/scores/{detailed_results.html,scores.json}` | JUNK (generated) | kit output from validate_submission; fold the numbers into the archive README first |
| `submit_trick/*` (7) | HIST | the disclosed exploit; `DISCLOSURE.md` is cited at CLAUDE.md l.119 and README.md "Two things" |
| `fno_deeponet/*` (28), `deeponet/*` (2), `diffusion/pde_refiner.py` | HIST | the round-4 sweeps and baselines; fno_deeponet's eval scripts import eval_wp (dead); `closed/eval_all.py:78,131` reads `deeponet/deeponet.pth` and `diffusion/unet_refiner.pth` (untracked checkpoints) |

### `Track_2/` and `leaderboard/`
| path | class | evidence |
|---|---|---|
| `build_t2.py` `validate_t2.py` | LIVE | build_round13.sh l.244–245 (build_final.sh gates through it) |
| `stream_eval.py` | LIVE (gate) | build_final.sh l.63 (`--causal-check`) |
| `verify_space.py` `t2_common.py` | VERIF, LIVE lib | t2_common is imported by validate_t2, stream_eval, pkg_identity and 7 closed scripts |
| `t2_lat17.py` `step_breakdown.py` `t1_vs_t2_score.py` | INSTR | Track_2/README l.141–144 |
| `submit/submission.py`, `submit/_fno.py` (symlink to `Track_1/submit_fno/_fno.py`) | LIVE | the T2 package |
| `submit_ho/`, `submit_ho5/` (4 symlinks) | INSTR fixture | the `verify_space.py:37` default `--submission submit_ho`; EVALUATION l.195 |
| `r1_ref/{causal_band,causal_band_t2}.py`, `README.md` | VERIF | fp64 references for R1; hard-coded `/home/william/...` paths |
| `closed/*` (7 files and README) | CLOSED | shimmed |
| `leaderboard/lb.py` | LIVE | the ledger renderer |
| `leaderboard/{board,runs,screen}.jsonl`, `README.md` | LEDGER / DOC | `board.jsonl` stops at 09-20 |

---

## 2. Import and invocation graph (LIVE, INSTR, VERIF)

**How modules are found.** No package `__init__` exists anywhere, so every import is by bare name after a
`sys.path.insert`. Five conventions account for all of them:

| convention | used by |
|---|---|
| **A.** `HERE = dirname(__file__)` (= `Track_1`), then insert `HERE` and `HERE/submit_fno` | compare_v9 (l.35), ens_v9 (l.37), kfold (l.54-55), aoa_holdout_eval (l.53-55, also `unet`), mean_blend_re (l.39-40), predict_board (l.66-67), proxy_calib (l.30-31), quant8 (l.37-38), re_split (l.36-37), screen_eval (l.62-63), screen_sets (l.25-26), lat_r15, lat_fast17, alpha_board (l.67), calib_band13 (l.41), eval_v9 (l.36-37) |
| **B.** `submit_fno/X.py` inserts `TRACK1` and `HERE` | fit_band.py (l.66-67), validate_strict.py (l.42) |
| **C.** `verify/X.py` inserts `T1 = dirname(dirname(__file__))` and `T1/{unet,submit_fno}` | amp_path (l.64), fuse_{ffno,swin,vit,osc}, graph_path (l.77), quant_identity (l.25-26), diff_arch, diff_ufno, selfnorm_path, phead_strip, simmap_modes, aug_mask_check; `pkg_identity` inserts `ROOT/Track_2` (l.29) for t2_common |
| **D.** `Track_2/X.py` inserts `HERE` and `HERE.parent/"Track_1"` | validate_t2 (l.44-45), stream_eval (l.37-38), t1_vs_t2_score, t2_lat17 (also `HERE/submit`), verify_space |
| **E.** `leaderboard/lb.py` inserts `dirname(HERE)/Track_1` | l.43 and l.958 → `board_rounds`, `final_v9` |

Edges (importer → repo module):

| file | imports (how found) | invoked by |
|---|---|---|
| `sps9.py` | kit v9 `scoring.py` via `importlib`, **absolute `sps9.py:30`** | everything below |
| `eval_v9.py` | `_fno` (A), kit v9 (**absolute l.34**) | compare_v9, ens_v9, kfold, alpha_board, proxy_calib, re_split, mean_blend_re, exchange_v9, and 20 closed scripts |
| `final_v9.py` | `board_rounds` (cwd or `Track_1.` package fallback, l.137-140); scipy | validate_strict, validate_t2, stream_eval, lb.py, predict_board, compare_v9, ens_v9, kfold, quant8, re_split, screen_eval, lat_fast17, aoa_holdout_eval, calib_band13, alpha_board |
| `predict_board.py` | sps9, `_fno`, board_rounds, final_v9 (A) | by hand; screen_eval `run_main`; finalize_round11 and build_round12 (HIST) |
| `make_val.py` | none; data at **absolute l.24** (overridable by `--data`) | reproduce.sh l.232 (runs `cd Track_1`); qd_k/run_qdre_train.sh |
| `re_split.py` | `_fno`, eval_v9, final_v9, sps9 (A); **absolute l.38** | kfold, aoa_holdout_eval, mean_blend_re, qd_k/re_readout |
| `kfold.py` | `_fno`, eval_v9, final_v9, sps9 (A), `compare_v9`; **absolute l.56-57** (data, scratch out) | by hand |
| `aoa_holdout_eval.py` | `_fno`, final_v9, mean_blend_re, model, re_split, sps9, kfold (A plus `unet`); **absolute l.57** | by hand |
| `mean_blend_re.py` | `_fno`, eval_v9, re_split, sps9 (A); **absolute l.43** | aoa_holdout_eval |
| `screen_eval.py` | `_fno`, final_v9, predict_board, sps9, ens_v9, fit_band (via `run_main`); **absolute `SCREEN` l.65**; reads `plan/*.json` and `../leaderboard/screen.jsonl` | screen_sets |
| `screen_latency.py` | `_fno`, models_alt (inserts `submit_fno` and `unet`) | by hand; cited by fuse_swin |
| `exchange_v9.py` | `_fno`, sps9, eval_v9; **`HERE` absolute l.7** | none |
| `submit_fno/fit_band.py` | `_fno`, sps9 (B) | build_round13.sh l.89, reproduce.sh l.314, screen_eval, sigblend/README step 1 |
| `submit_fno/validate_strict.py` | final_v9 (B); `submission` and kit v9 via `importlib`, **absolute l.37** | build_round13.sh l.235 |
| `submit_fno/build_zip.py` | `_fno_pack` (importlib of `_fno.py`) | validate_strict, build_t2 |
| `submit_fno/bake_ffno_path.py` | none | build_round13.sh l.203 |
| `submit_fno/build_round13.sh` | cwd = repo root (`cd "$(dirname "$0")/../.."`, l.18); `PY` defaults to **absolute l.21** | build_final.sh l.54, reproduce.sh l.239, rp_k27_final.sh l.47 |
| `submit_fno/build_final.sh` (U) | reads `bands/k27…npz`, `sigblend/…json`; **absolute `PY` fallback l.20** | by hand |
| `sigblend/dump_twins.py` (U) | `_fno` (inserts `T1` and `T1/submit_fno`); **absolute l.9, 11, 14** | by hand (README step 2) |
| `sigblend/bandlab.py` (U) | sps9 (inserts `T1`, l.14); **absolute l.17** | make_k0_band |
| `sigblend/make_k0_band.py` (U) | bandlab, sps9 (**needs `PYTHONPATH=Track_1`**, per README); **absolute l.20, 23** | by hand (README step 3) |
| `sigblend/rp_k27_final.sh` (U) | **absolute l.12, 17, 26** | Slurm (job record) |
| `verify/k0a_{dump,edge,lat,score}.py` (U) | `submission` via importlib; **absolute `val_unseen` at l.7 (dump, edge, lat) and l.8 (score); kit v9 at edge l.9, score l.6** | by hand |
| `verify/joint_gain.py`, `verify/sigblend.py` (U) | importlib of `submission` / none | by hand |
| `verify/diff_scoring_{official,real}.py` | kit files, **absolute l.16-17 / l.20-21** | by hand |
| `verify/fuse_ffno.py`, `verify/fuse_swin.py` | `_fno`, models_alt, sps9 (C); **absolute scratch l.62 / l.67** | by hand |
| `verify/simmap_modes.py`, `verify/aug_mask_check.py` | simreal / train2, model (C); **absolute data l.45-46 / l.40** | by hand |
| `qd_k/export.py` | `_fno`, sps9, make_val, load_export (inserts `REPO/Track_1{,/submit_fno}`, `REPO` from `RPDE_QDK_REPO`); **absolute defaults l.62, 63, 85** | reproduce.sh l.319 |
| `qd_k/head_direct.py` | centre_band, sweep_guards, load_export (its own dir), model and models_alt (`UNET` argv), `"."` | reproduce.sh l.322 |
| `qd_k/h_ratio_check.py` | `_fno` (`HERE/../submit_fno`) | reproduce.sh l.330 (runs `cd Track_1`) |
| `qd_k/re_readout.py` | `_fno`, centre_band, re_split (from `--code`); **absolute l.74** | qd_k/run_qdre_readout.sh |
| `qd_k/run_qdre_{train,readout}.sh` | **absolute `PY` and data defaults (train l.30-31, readout l.8-9)**; `pitzer` paths | by hand (Pitzer) |
| `Track_2/validate_t2.py` | t2_common, final_v9 (D); kit **absolute l.48** | build_round13.sh l.245 |
| `Track_2/stream_eval.py` | t2_common, final_v9 (D); kit **absolute l.35** | build_final.sh l.63 |
| `Track_2/t2_common.py` | data **absolute l.24** | validate_t2, stream_eval, verify_space, t1_vs_t2_score, pkg_identity, and 7 closed scripts |
| `Track_2/build_t2.py` | `_fno_pack` via importlib | build_round13.sh l.244 |
| `Track_2/t2_lat17.py`, `Track_2/t1_vs_t2_score.py` | (D); kit **absolute l.32 / l.36** | by hand |
| `Track_2/r1_ref/causal_band{,_t2}.py` | sps9 from **`/home/william/realpde-shared/fen`** (l.94 / l.69); twin export at `/home/william/pitzer-requests/...` (l.91-92 / l.70) | by hand; `causal_band_t2` imports its sibling via `__file__` (l.95) |
| `unet/distill_pipeline.sh` | **absolute `PY`, `REAL`, `SIM` defaults l.35-37** | reproduce.sh l.388 |
| `unet/simreal.py` | default cache **`/fs/scratch/...` l.98, 103** (`RPDE_SIMCACHE` overrides) | train2 |
| `unet/train2.py` | default data **absolute l.191** | reproduce.sh (runs `cd $U`) |
| `reproduce.sh` | **absolute l.46, 49, 57, 76, 81, 389** (all overridable) | by hand |
| `leaderboard/lb.py` | board_rounds, final_v9 (E); writes `../../leaderboard.html` (`RPDE_LB_HTML` overrides) | by hand |

**Hard-coded absolute paths.** Every live, instrument and verifier site is marked **absolute** with its line in the
table above: about 45 sites, of which roughly 22 are the OSC data or kit, 6 the repo's own path, 12 scratch, 2 files
`/home/william`, and 7 the interpreter. Outside the table (HIST and CLOSED): the v7 kit in `eval_wp.py:23`,
`submit/{fit_band.py:19, fit_band2.py:24, validate_submission.py:28}`, `submit_trick/{validate.py:28, validate_strict.py:32}`,
`deeponet/eval_deeponet.py:13`, `closed/{diag_analog:109, eval_all:27-28, tke_ceiling:29}`; `unet/dinov2_init.py:169`;
`unet/runq.sh:8`; `unet/run_{sim,sim2,ctx}.sh:11-13`; `build_round1{0,1,2}.sh`, `finalize_round11.sh:12`; and all 13
`fno_deeponet/*.sh`.

A single `RPDE_ROOT` / `RPDE_KIT` resolution (for example a `Track_1/paths.py`) would remove more than 20 of these. That
is a code change, so it is recommended but not part of the reorganisation below.

---

## 3. Instruments whose defaults are traps

The operating point of record: **ALPHA 0.80 · band-p 0.45 · centre_alpha 0 · fill-unscored · inflate 1.39 · SIGBLEND
(lam 0.5, q = (0.4847, 0.4907), p' = 0.225)**.

| file:line | default | at the operating point | note |
|---|---|---|---|
| `submit_fno/fit_band.py:108` | `--alpha 0.75` | 0.80 | |
| `submit_fno/fit_band.py:122` | `--band-p 0.5` | 0.45 | |
| `submit_fno/fit_band.py:109` | `--centre-alpha None` (centred on the prediction) | 0 | |
| `submit_fno/fit_band.py:142` | `--fill-unscored` off | on | |
| `submit_fno/build_round13.sh:19` | `ALPHA="${1:-0.90}"` | 0.80 | every other unset knob also falls back to old behaviour (CLAUDE) |
| `compare_v9.py:41`, `:42` | `--alpha 0.75`, **`--inflate 1.5`** | 0.80, 1.39 | the inflate default is off too |
| `ens_v9.py:46` | `--alpha 0.75`; band p fixed at 0 (`optimal_band` without the σ ratio, l.85); `--time-score 90.5066` (round 8) | 0.80 | |
| `eval_v9.py:113` | `infer(..., alpha=0.75)`; `--alphas [0.75]` (l.121) | 0.80 | every caller that omits alpha inherits 0.75 |
| `predict_board.py:398` | `--alpha 0.75` | 0.80 | |
| `predict_board.py:397` | `--band submit_fno/band.npz` | explicit band | the default is an untracked file holding whatever the last build wrote |
| `predict_board.py:445-446, 463-464` | reads only `base/sigma_ref/p` | centre and σ-power | ignores `centre_alpha`, `clip_lo/hi` and SIGBLEND |
| `kfold.py:299`, `re_split.py:92`, `mean_blend_re.py:89` | `--alpha 0.90` | 0.80 | |
| `aoa_holdout_eval.py:138` | `--alpha 0.60` (help: "the shipped compose alpha") | 0.80 | |
| `calib_band13.py:62`, `quant8.py:63`, `lat_r15.py:85`, `lat_fast17.py:119`, `verify/quant_identity.py:35` | `--alpha 0.60` | 0.80 | |
| `lat_r15.py:84`, `Track_2/t2_lat17.py:41`, `verify/quant_identity.py:36`, `Track_2/build_t2.py:46` | `--band` = an untracked build by-product | explicit | |
| `screen_eval.py:73` | `ALPHA, BAND_P, F_BOARD = 0.60, 0.45, 1.39` | 0.80 | used by screen_sets as `se.ALPHA` |
| `alpha_board.py` `--anchor 0.60`, `--time-score 93.1154` | | | |
| `qd_k/centre_band.py:22`, `qd_k/export.py:47` | `ALPHA = 0.60` | 0.80 | vendored verbatim; `h_ratio_check.py:41` and `head_direct.py:25` do use 0.80 |
| `verify/fuse_osc.py:66`, `fuse_vit.py:120`, `selfnorm_path.py:59` | `alpha=0.90` literal | none | fine: these are identity gates, where alpha only has to match on both sides |

**The SIGBLEND trap.** `bands/k27_sigblend_lam0.5.npz` stores `p = 0.225` and a blended `base`, but not `lam` or `q`.
Those sit in `sigblend/band_k0_sigblend_lam0.5.json` and are baked as `EBAND` / `SIGPOW`. Any instrument that loads
this npz computes a band about 3× too wide (the `sigblend/README.md` "Traps" paragraph), with no error raised. **No local
instrument (predict_board, aoa_holdout_eval, calib_band13, quant8, compare/ens) currently evaluates centre-0 or
SIGBLEND**, so their `sps` numbers are for a band that did not ship.

**One-line fix.** Add a constants module `Track_1/opoint.py`:
`ALPHA=0.80; BAND_P=0.45; CENTRE_ALPHA=0.0; INFLATE=1.39; F_TAIL=1.36; FILL_UNSCORED=True; SIGBLEND_JSON="sigblend/band_k0_sigblend_lam0.5.json"`.
Every instrument already puts `Track_1` on `sys.path`, so each argparse default becomes `default=opoint.ALPHA`.
Complement it with a guard in `sps9` (for example `load_band(path)`) that **refuses** an npz whose `p` is not 0.45
unless a SIGBLEND sidecar is given, and that honours `centre_alpha`.

**Would changing a default silently change a recorded number?**

- **Recorded numbers do not change.** `board_rounds.py`, `final_v9.py`, `predict_board.ANCHORS`, the `*.jsonl` ledgers
  and the bands hold literal values, and the anchors carry their own `--anchor-alpha`.
- **Invocations that relied on a default would re-run differently.** Three cases matter:
  - `screen_eval.run_main` passes `--alpha` explicitly, so it is safe. But its band cache key hashes alpha (l.194), so
    changing `screen_eval.ALPHA` invalidates the cache, which is harmless.
  - `build_round13.sh`'s `ALPHA` default is load-bearing only when called with no argument. `reproduce.sh`,
    `build_final.sh` and `rp_k27_final.sh` all pass it.
  - Instruments whose logged results (ROUNDS.md, `runs.jsonl`) were produced with defaults: kfold at 0.90, ens and
    compare at 0.75. After the change they re-run at 0.80.

  Record in `EVALUATION.md` §4: "results before 09-27 used the per-script defaults listed in `audit_layout` §3".
- **`fit_band.py` is the dangerous one.** It is the band-of-record producer, and `build_round13.sh:89` passes
  `--alpha "$ALPHA" --band-p 0.45` explicitly, so builds are unaffected. Changing its `--centre-alpha` or
  `--fill-unscored` defaults, however, would alter any refit that relies on build_round13 forwarding only when
  `RPDE_CENTRE_ALPHA` / `RPDE_BAND_FILL` are set. **Leave the build path's defaults alone**; change only the instruments.

---

## 4. Broken references

Checked every backticked or linked path with a file extension in CLAUDE.md, README.md, EVALUATION.md, ROADMAP.md
Part 4 (l.1457–end), REPRODUCE.md, Track_1/README.md, Track_2/README.md, leaderboard/README.md, and also
`submit_fno/README.md`, `closed/README.md`, `qd_k/README.md`, `sigblend/README.md` and `bands/MANIFEST.md`. Each was
resolved against the doc's directory, the repo root, `Track_1/`, `Track_2/`, `submit_fno/` and `unet/`. Paths into
`~/RealPDE`, `../sub_candidates`, `../handoffs` and the scratch `jobs/` and `logs/` are intentionally outside the repo
and are not listed.

**Genuinely broken (the target does not exist anywhere):**

| doc:line | reference | fix |
|---|---|---|
| `Track_1/submit_fno/README.md:67` | `../band_cv_v9.py` | `../closed/band_cv_v9.py` |
| `REPRODUCE.md:111` | `h5py_fix.sh` | not in the repo; name its location (Pitzer workspace) or vendor it |
| `REPRODUCE.md:663` | `qd_FINAL_verdict.txt` (Devi) | outside the repo; give its path, or copy it into `qd_k/` |
| `REPRODUCE.md:826` | `repro_score.py` (Ilse) | outside the repo; vendor it into `Track_1/verify/`, since it is the §12 acceptance instrument |
| `EVALUATION.md:304` | `HISTORY.md` | this is `../sub_candidates/HISTORY.md`; write the prefix |
| `bands/MANIFEST.md:189, 219, 220` | `expected_base_centre0_nofill.npz`, `f4v2a060_fill.npz`, `evidence/band_identity_r35.txt` | not in the repo; mark them "(sub_candidates evidence)" |
| `sigblend/README.md:7` (U) | `band_r38_a080_c0_fill.npz` | the committed name will be `Track_1/bands/r38_a080_c0_fill.npz`; step 2 also says "edit `U` / `out` paths" because `dump_twins.py` hard-codes them |
| `sigblend/README.md:16` (U) | `jobs/rp_k27_final.sh` | it is `Track_1/sigblend/rp_k27_final.sh` |
| `Track_2/r1_ref/README.md` | "paths at the top of `_load` / `self_test`" | they point at `/home/william/...` |

**Resolve locally but not in a fresh clone** (the target is untracked): `bands/k27_sigblend_lam0.5.npz` and
`sigblend/band_k0_sigblend_lam0.5.json` (`build_final.sh:30-31`), `bands/r38_a080_c0_fill.npz`, and
`Track_1/val_unseen.npz` (the k0a scripts; it is generated by make_val, which is by design).

**Stale content, not broken paths:**

- `README.md:9-10,12,22`: scores dated 09-19, "four width-96 FNOs".
- `Track_1/README.md:24-36`: the model paragraph says ALPHA 0.60. Its Layout table (l.60-72) omits `qd_k/`, `sigblend/`
  and `bands/*` finals, and it still says "`submit_fno/` … `build_round13.sh`" but not `build_final.sh`.
- `Track_1/README.md:136`: Status says 81.401.
- `CLAUDE.md:26-65`: *Current state* is dated 09-20. The Repo map (l.380-400) omits `qd_k/`, `sigblend/`, `r1_ref/` and
  `build_final.sh`. The *Build and gate* block (l.308) builds at 0.60.
- `submit_fno/README.md`: "As of R24/R25" and the constants table at ALPHA 0.60. No EBAND, JOINT_GAIN, SIGPOW, R1 or
  D1C constants appear.
- `Track_2/README.md:63-66, 120-136`: describes round-13 `submit_ho5`.
- `leaderboard/board.jsonl`: ends at the 09-20 PENDING row. The 09-21…09-27 rounds are not ledgered (check against
  `board_rounds.py`, which does mention 09-25…27).

---

## 5. Proposed target layout

**Principles.**

1. Every LIVE, INSTR and VERIF file keeps its path.
2. `Track_1/closed/` and `Track_2/closed/` keep their depth, because of the shim.
3. Historical packages and sweeps go under a **new `Track_1/archive/`**, with an index README. Most of these scripts
   are already dead through the v7 kit, so a one-level move breaks nothing runnable. The two that are runnable
   (`diffusion/pde_refiner.py`, `deeponet/deeponet_baseline.py`) get the closed/-style shim.
4. Only byte-duplicates and build by-products are removed.

```
.
├── CLAUDE.md README.md EVALUATION.md ROADMAP.md ROUNDS.md REPRODUCE.md reproduce.sh environment.yml .pyenv.sh .gitignore
├── Track_1/
│   ├── README.md
│   ├── sps9.py make_val.py re_split.py eval_v9.py board_rounds.py final_v9.py predict_board.py      # LIVE
│   ├── compare_v9.py ens_v9.py kfold.py aoa_holdout_eval.py mean_blend_re.py alpha_board.py quant8.py
│   │   calib_band13.py proxy_calib.py sub_value.py lat_r15.py lat_fast17.py
│   │   screen_eval.py screen_sets.py screen_latency.py                                              # INSTR
│   ├── opoint.py                          # NEW (optional, §3): operating-point constants
│   ├── unet/  train2.py model.py models_alt.py simreal.py distill_labels.py distill_pipeline.sh
│   ├── submit_fno/  submission.py _fno.py fit_band.py build_zip.py validate_strict.py bake_ffno_path.py
│   │                band_feats.py build_round13.sh build_final.sh README.md  [band.npy: see §6]
│   ├── sigblend/    bandlab.py dump_twins.py make_k0_band.py band_k0_sigblend_lam0.5.json rp_k27_final.sh README.md
│   ├── qd_k/        (unchanged, 14 files)
│   ├── bands/       *.npz (+ k27_sigblend_lam0.5.npz, r38_a080_c0_fill.npz) MANIFEST.md
│   ├── verify/      (19 tracked + k0a_*.py joint_gain.py sigblend.py)
│   ├── closed/      51 + exchange_v9.py eval_wp.py eval_pair.py calib_sigma.py dinov2_init.py; README.md
│   ├── plan/ report/                      (unchanged: screen_eval reads plan/*.json)
│   └── archive/                           NEW: README.md = index
│       ├── submit_unet_0730/  (was submit/, minus the two byte-duplicates and _validate_run/)
│       ├── submit_trick/      (was submit_trick/)
│       ├── fno_deeponet/      (was fno_deeponet/)
│       ├── deeponet/ diffusion/
│       ├── build_rounds/      build_round10.sh build_round11.sh build_round12.sh finalize_round11.sh
│       └── unet_r9_r12/       train.py runq.sh run_sim.sh run_sim2.sh run_ctx.sh queue.txt queue_alldata.txt
│                              queue_round10.txt done.txt logs/log_*.txt (51)
├── Track_2/  build_t2.py validate_t2.py stream_eval.py verify_space.py t2_common.py t2_lat17.py step_breakdown.py
│             t1_vs_t2_score.py submit/ submit_ho/ submit_ho5/ r1_ref/ closed/ README.md
└── leaderboard/  lb.py board.jsonl runs.jsonl screen.jsonl README.md
```

**The `archive/README.md` index.** One table row per item: *question asked · verdict · how it was run (command/job) ·
runnable today? (v7 kit gone → no)*. The content already exists and only needs collecting:

| archive item | source for the row |
|---|---|
| `submit_unet_0730` | `submit/README.md`; its `_validate_run/scores.json` numbers are pasted here before removal |
| `submit_trick` | `submit_trick/README.md` and `DISCLOSURE.md` |
| `fno_deeponet` | `plan_20260802.md`, `report_20260802.md`, `*_results_*.txt` |
| `deeponet`, `diffusion` | `report_20260729.md` |
| `build_rounds` | ROUNDS.md R10–R12 |
| `unet_r9_r12` | ROUNDS.md, `plan_2026080{8,9}.md` |

### Ordered operations

Each step leaves `reproduce.sh`, both gates and `lb.py` runnable. Per the build rule, commit before building.

**Step 0: commit the final recipe (no moves).**
```
git add Track_1/submit_fno/{_fno.py,submission.py,build_round13.sh,build_final.sh} Track_2/submit/submission.py
git add Track_1/sigblend/ Track_1/bands/k27_sigblend_lam0.5.npz Track_1/bands/r38_a080_c0_fill.npz
git add Track_1/verify/{k0a_dump,k0a_edge,k0a_lat,k0a_score,joint_gain,sigblend}.py
```
- **Edits:** add two `bands/MANIFEST.md` lines (k27 and r38 with their md5); fix `sigblend/README.md:7,16` (names above).
- **Check:** `bash Track_1/submit_fno/build_final.sh <tmpdir>` with `RPDE_REF_T1/T2` → `ALL … BYTE-IDENTICAL`.

**Step 1: remove junk (no importer affected).**
```
git rm -r --cached Track_1/submit_fno/_package          # already gitignored; build_zip default stage
git rm Track_1/submit/_package/band.npy Track_1/submit/_package/submission.py   # byte-duplicates
git rm -r Track_1/submit/_validate_run                  # generated kit output (numbers → archive README first)
git rm -r .claude/skills                                 # generic personal skills (owner's call; they are copies of ~/.claude skills)
```
- **Edits:** none. `submit/_package/model.py` stays (the shipped snapshot); it moves in step 3.

**Step 2: move dead or closed top-level scripts into `closed/` (same depth, so the shim pattern applies).**
```
git mv Track_1/exchange_v9.py Track_1/closed/     # HERE is absolute: runs unchanged
git mv Track_1/eval_wp.py Track_1/eval_pair.py Track_1/calib_sigma.py Track_1/closed/
git mv Track_1/unet/dinov2_init.py Track_1/closed/
```
- **Edits:**
  - Give `eval_wp.py` the closed/ 5-line shim, with its two `HERE` uses gaining a `dirname`. Its `sys.path.insert(HERE/"unet")`
    must still resolve to `Track_1/unet`.
  - `eval_pair.py` and `calib_sigma.py` import eval_wp as a sibling, so they need no shim beyond the standard one.
  - `dinov2_init.py` imports `models_alt` as a sibling: add `sys.path.insert(0, <Track_1>/unet)`.
  - `closed/diag_spectral.py` finds eval_wp as a sibling; nothing to do.
  - Historical importers of eval_wp (`fno_deeponet/eval_*.py`, `submit/fit_band2.py`) are handled in step 3.
  - **Docs:** `Track_1/README.md` l.91 (exchange_v9) and the eval_wp/eval_pair/calib_sigma row; its Layout `unet/` row
    (drop dinov2_init); `EVALUATION.md:124,178` (exchange_v9 → `closed/`); `closed/README.md` gains 5 rows (below);
    CLAUDE.md l.393 and Track_1/README l.70: "51" → 56.
- **New `closed/README.md` rows:**
  - `exchange_v9.py`: what an X% accuracy gain is worth → frozen as CLAUDE *Exchange rate*.
  - `eval_wp.py`, `eval_pair.py`, `calib_sigma.py`: the WP-era scorer (kit v7) → superseded by `eval_v9`/`sps9`; they
    fail at import since the v7 kit was removed.
  - `dinov2_init.py`: DINOv2 initialisation → no gain; now prohibited (outside pretrained weights).
- **Check:** `python -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" Track_1/closed/*.py`; and
  `grep -rn "eval_wp\|eval_pair\|calib_sigma\|exchange_v9\|dinov2_init" --include=*.py --include=*.sh` outside
  `closed/` and `archive/` → empty.

**Step 3: historical directories → `Track_1/archive/`.**
```
mkdir Track_1/archive   (via git mv targets)
git mv Track_1/submit        Track_1/archive/submit_unet_0730
git mv Track_1/submit_trick  Track_1/archive/submit_trick
git mv Track_1/fno_deeponet  Track_1/archive/fno_deeponet
git mv Track_1/deeponet      Track_1/archive/deeponet
git mv Track_1/diffusion     Track_1/archive/diffusion
git mv Track_1/submit_fno/build_round10.sh Track_1/submit_fno/build_round11.sh \
       Track_1/submit_fno/build_round12.sh Track_1/submit_fno/finalize_round11.sh  Track_1/archive/build_rounds/
```
- **Edits:**
  - `archive/diffusion/pde_refiner.py:22`: `"..", "unet"` → `"..", "..", "unet"` (it is runnable).
  - `archive/fno_deeponet/eval_{ens,phase3,sweep}.py:18-22` insert `dirname(dirname(__file__))`, which is now `archive/`.
    Change it to `Track_1` and also insert `Track_1/closed` for eval_wp. Optional: they are dead (v7) either way.
  - `archive/submit_unet_0730/fit_band2.py:25` and `fit_band.py:20` (TRACK1): the same one-dirname change. Optional, dead.
  - `closed/eval_all.py:78,131` reads `HERE/deeponet/deeponet.pth` and `HERE/diffusion/unet_refiner.pth`
    (untracked checkpoints): point it at `archive/…`. Already dead (v7).
  - The build_rounds scripts run from `Track_1/`, and `finalize_round11.sh:20` calls `bash submit_fno/build_round11.sh`.
    Leave them verbatim (they are records) and mark them "not runnable" in the index.
  - **Docs:**
    - `README.md` "Two things" and CLAUDE.md l.119 (`Track_1/submit_trick/DISCLOSURE.md` → `Track_1/archive/submit_trick/DISCLOSURE.md`).
    - `Track_1/README.md` Layout (l.72 historical row → `archive/`).
    - CLAUDE.md Repo map (l.395 `submit/ submit_trick/ fno_deeponet/ deeponet/ diffusion/ historical` → `archive/`).
    - ROADMAP Part 4 file map: grep for `submit_trick`, `fno_deeponet`, `Track_1/submit/`, `build_round1[012]`.
    - ROUNDS.md and report/*.md: leave as written. Historical prose refers to paths as they were, so add one line at the
      top of `archive/README.md`: "paths in ROUNDS.md/report/ before 09-27 refer to pre-archive locations".
- **Check:** `bash -n` on every `.sh`; `git grep -n "Track_1/submit/\|submit_trick/\|fno_deeponet\|build_round1[012]"`
  in `*.md` outside `ROUNDS.md` and `report/` → only archive-aware hits.

**Step 4: unet job records → `archive/unet_r9_r12/`.**
```
git mv Track_1/unet/{train.py,runq.sh,run_sim.sh,run_sim2.sh,run_ctx.sh,queue.txt,queue_alldata.txt,done.txt} Track_1/archive/unet_r9_r12/
git mv Track_1/unet/queue_round10.bak Track_1/archive/unet_r9_r12/queue_round10.txt   # it is a record, not an editor backup
git mv Track_1/unet/log_*.txt Track_1/archive/unet_r9_r12/logs/
```
- **Edits:** `runq.sh` does `cd "$(dirname "$0")"` and runs `train2.py` from there, so after the move it no longer finds
  train2 (a record, not runnable). `run_{sim,sim2,ctx}.sh` `cd` an absolute unet path, so they still work.
  `archive/unet_r9_r12/train.py` imports `model` as a sibling, so it needs `sys.path.insert(<Track_1>/unet)` if it is
  ever to run.
- **Docs:** `Track_1/README.md` Layout `unet/` row; CLAUDE Repo map `unet/` line.
- **Check:** `reproduce.sh` still finds `$U/train2.py` (l.69 preflight).

**Step 5: docs to match (no code moves).**

- **CLAUDE.md:** *Current state* (09-27 final: T1 81.671149, T2 82.055553, recipe `build_final.sh`); *Build and gate*
  → `build_final.sh`; the Repo map adds `qd_k/`, `sigblend/`, `archive/`, `Track_2/r1_ref/`; *Instruments* → "pass
  `--alpha 0.80 --band-p 0.45 --centre-alpha 0`; no instrument scores SIGBLEND".
- **READMEs:** `README.md` (table, Approach, build line); `Track_1/README.md` (model paragraph, Layout, Status);
  `submit_fno/README.md` (the constants table gains EBAND, JOINT_GAIN, D2_DIRECT_PRED, WARM_TAILS, PIN_OUT, BATCH_CAP,
  SMALL_GRAPH*; fix l.67); `Track_2/README.md` (R1, D1C, SIGEMA, SIGPOW, `--adaptive`, `r1_ref/`).
- **`REPRODUCE.md` and `reproduce.sh`:** the `final` target was being added concurrently (commit it with step 0);
  REPRODUCE §13 must match it; resolve the three missing files in §4.
- **`EVALUATION.md` §4:** the defaults note (§3). **Ledgers:** `board.jsonl` gets the 09-21…27 rows, then `lb.py --html`.

**Step 6 (optional code): `Track_1/opoint.py` plus the instrument defaults (§3).** This is one commit, and each
changed default is listed in the commit message.

---

## 6. Risks

| move or change | what breaks | mitigation |
|---|---|---|
| Nesting `closed/` deeper (for example `archive/closed/`) | 49 shims (`dirname(dirname(__file__))` must be `Track_1` or `Track_2`), so every `import sps9/eval_v9/final_v9/t2_common` and every `HERE/...` path fails | **Do not move `closed/`.** Rejected in §5 |
| Moving `eval_v9.py`, `final_v9.py`, `board_rounds.py`, `sps9.py`, `re_split.py` or `Track_2/t2_common.py` | 20+11 closed importers; validate_strict (l.43), validate_t2 (l.51), stream_eval, lb.py (l.43/958, bare `import final_v9`, `board_rounds`); `final_v9.py:137-140` imports `board_rounds` as a sibling, falling back to `Track_1.board_rounds` | keep them at `Track_1/` and `Track_2/` top level |
| Moving `make_val.py` | `reproduce.sh:232` (`cd Track_1 && make_val.py`), `qd_k/run_qdre_train.sh`, `qd_k/export.py` (imports it) | keep |
| Moving `plan/screen_*_20260914.json` | `screen_eval.py:67,78,101` and `unet/distill_labels.py` (kcurve) | keep `plan/` in place |
| Moving `Track_2/submit_ho*/` | `verify_space.py:37` default; EVALUATION l.195 | keep |
| Renaming `Track_2/submit/_fno.py` (a symlink into `Track_1/submit_fno/_fno.py`) or `submit_ho*/` (symlinks into `../submit/`) | `build_t2.py` packaging and T2 import; git stores them as mode 120000 | never `git mv` the symlink targets; `archive/` does not touch them |
| `git rm Track_1/submit_fno/band.npy` | `submission.py:222` `_BAND_LEGACY` fallback: when `band.npz` is absent (a fresh clone running `submit_fno/submission.py` from the tree, e.g. `verify/amp_path.py`, `graph_path.py`, `cpu_smoke_submission.py`), the band silently changes from this legacy file to the ±5 % kit default, or crashes | leave it until the `submission.py` auditor removes the fallback; then `git mv` it to `bands/legacy_r10_band.npy` (a record) |
| `git rm --cached submit_fno/_package/` | nothing imports it; `build_zip.py` recreates it; `finalize_round11.sh` and `build_round12.sh` glob `submit_fno/_package/fno*.pth` (HIST only) | safe |
| Build scripts during the reorganisation | `build_round13.sh` `sed`s constants into `submission.py` (both tracks) and `_fno.py`, and leaves them there; `build_final.sh` restores via a trap, and `rp_k27_final.sh:42` uses `git checkout`. A `git mv` or commit made while a build runs captures sed-edited constants | **no build or `rp_*` job may be running during any step**; run `git diff --stat` on the three files before each commit |
| `reproduce.sh` | depends on `Track_1/unet/{train2,model,models_alt,simreal}.py`, `unet/distill_pipeline.sh`, `make_val.py`, `submit_fno/{fit_band,build_round13,bake_ffno_path}`, `qd_k/{export,head_direct,h_ratio_check}.py`, `verify/pkg_identity.py`, `bands/{r24_xfam6,t2r13_p2ffnoL8x4,r34_p2ffnoL8sim_fill,r35_centre0_fill}.npz`, and `Track_2/submit/band8.npz` (build output) | none of these move in §5. Moving `distill_*` or `qd_k/` breaks it |
| Gates | `validate_strict.py` (kit v9 absolute l.37; imports `final_v9` via TRACK1); `validate_t2.py`/`stream_eval.py` (t2_common, final_v9, kit t2_v6 absolute); `verify/blend_path.py` (build_round13 l.239 `--dir Track_1/submit_fno/_package8`) | none move. The kit paths are absolute, so any machine but this one needs the kit at `~/RealPDE/official/...` (REPRODUCE §9.1) |
| `leaderboard/lb.py` | imports `board_rounds` and `final_v9` from `../Track_1` (l.43, l.958); reads `board/runs/screen.jsonl` next to itself; writes `../../leaderboard.html` (outside the repo) | keep `leaderboard/` beside `Track_1/`; moving the repo into a subdirectory changes the HTML location (set `RPDE_LB_HTML`) |
| Module-name shadowing | `build_zip.py`, `fit_band.py`, `validate_strict.py`, `_fno.py`, `submission.py`, `model.py` exist in several directories; imports resolve by `sys.path` order | the `archive/` move lowers the risk (the historical copies leave any `Track_1/<dir>` a script might insert); never add `archive/*` to a `sys.path` |
| Committing SIGBLEND bands without the json sidecar | `build_final.sh:34-35` needs both; any instrument reading the npz alone is 3× wide (§3) | commit the npz and json together; add the MANIFEST line saying "requires sigblend/band_k0_sigblend_lam0.5.json" |
