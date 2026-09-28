# Shipping-code refactor (audit tiers 1-3): report — 2026-09-28

Worktree `/fs/scratch/PAS1187/qwu/tmp/realpde/wrap0927/wt_pkg`, base commit `7b6ee8e`, nothing committed.
The patch is `notes/patch_pkg.diff`. It was made with `git diff HEAD --binary` because it deletes the binary `band.npy`, which plain `git diff` cannot apply.
It passes `git apply --check` on a pristine `git archive 7b6ee8e` and `git apply --check -R` on the worktree.
It contains no build by-products and no `unet_ckpt` symlink.

The uploaded references are `ref/t1` (board 81.671149) and `ref/t2` (board 82.055553).
Rebuilt packages are in `pkgP/final/{t1,t2}` and the twin packages in `pkgP/twin_{in,tgt,tgtsm}`.
Logs are in `logs/` (first-round logs in `logs/round1/`), and the job drivers in `jobs/rp_w27p_*.sh` and `jobs/w27p_*.py`.

## 1. What changed, per file (`git diff HEAD --numstat`: added / removed)

| file | + / − | what |
|---|---|---|
| `Track_1/submit_fno/submission.py` | 98 / 327 | **Pruned:** `GRAPH`, `GRAPH_FP`, `GRAPH_ACTIVE`, `GRAPH_PROBE_DIFF`, `_Graphed`, `_capture_graph`, `_capture_graph1`, `SMALL_GRAPH`, `SMALL_N`, `SMALL_GRAPH_LAZY` (+ `_G1_TRIED`, `_SMALL_SEEN`, `SMALL_GRAPH_DIFF`), `BATCH_FP`, `GPU_FP` (+ `_GPU_FP_DONE` and the `_fno._PRED_FP` plumbing), `LN_MANUAL`, and the `band.npy` fallback. `_load()` now returns 6 fields, not 8. **M4:** new `_eband_host_factor()`. On the host band path a logsig-only EBAND (`'mult'`, no clip, features 1..4 zero) is applied in the GPU path's order, `(base·r3)·exp(log(σ+1e-8)·b0+b5)`; any other EBAND still raises. **Docs:** new module docstring (6 members, the `predict()` steps); `GPU_BAND`, `EBAND`, band-check, `_pick_batch` and member-list comments refreshed; `HOST_WRITE` and `SPEC_PATH` marked inert in the final package. The DIAG line lost its graph/fingerprint fields and gained `eband`. |
| `Track_1/submit_fno/_fno.py` | 24 / 9 | **M3** in `predict_fno`'s GPU band block, before the host copies: `yp = nan_to_num(yp)`, then `lo = min(where(isfinite(lo), lo, p3), p3)` and `hi = max(where(isfinite(hi), hi, p3), p3)`. Comments refreshed per audit §7: the module docstring, `FFNO_PATH` and `StackedFFNO` ("dense on both tracks"), and the claim that the GPU band is bit-identical to the host path (it is not, once EBAND is on). `LN_MANUAL` and `_PRED_FP` were **left in place**: they are unreachable from the new `submission.py` but still defaults in `_fno.py`. |
| `Track_2/submit/submission.py` | 179 / 127 | **Pruned:** `FALLBACK_FP`, `STACKED_OUT`, `R1_SNAP` (the snap branch in `kfield`), `SIGEMA`, `SIGEMA_BETA`, `SIGEMA_N0`, `SIGEMA_DDOF` and all their state and branches. **M3:** `_body` is now a guard around `_step` (the old body). *Head:* `gflag *= isfinite(gyp).all() & gok`, then `gyp.nan_to_num_()`. *Tail:* `gok = all(isfinite(go))`, `go.nan_to_num_()`, and `gl/gh = where(isfinite, min/max(·, go), go)` computed in place. The four stored previous-window fields (`gC`, `gH`, `gC0`, `gVp`) are written with `nan_to_num(out=)` instead of `copy_`. **m1:** `torch.cuda.set_device` for `cuda:N` in `get_ttt_model`, and `get_device_name(dev)` in DIAG. **m2:** pageable fallback for the pinned staging. **New knob `PREV_SPACE = "input"`** (M1 arm): `_yt()` and `_ym()`, the `r1_z` choice, and VARCAL's `/A²`. **New knob `STATS_MASK = False`** (M2 arm). **Docs:** new docstring (the model and `ttt_step` steps 1-7), plus the R1 reference note, VARCAL, `R1_TSM`, `SPEC_PATH` (inert) and the `_fetch` docstring. The DIAG line lost `sigema`, `fallback_fp` and `stacked_out`, and gained `prev_space` and `stats_mask`. |
| `Track_1/submit_fno/build_round13.sh` | 32 / 26 | `_gone()` refuses (exit 1, naming `fdac00b`) any non-default `RPDE_SHIP_{GRAPH, GRAPH_FP, SMALL_GRAPH, SMALL_GRAPH_LAZY, SMALL_N, BATCH_FP, GPU_FP, LN_MANUAL, STACKED_OUT, FALLBACK_FP, SIGEMA, SIGEMA_BETA, SIGEMA_N0, SIGEMA_DDOF}`; the old defaults (e.g. `build_final.sh`'s `SMALL_GRAPH=False`) are accepted. Their seds and greps are removed. The `R1_ASYM` sed pattern is now `[^ ]*`, so the trailing comment survives (m10; checked in the built file). New `RPDE_SHIP_PREV_SPACE` (validated `input` or `target`) and `RPDE_SHIP_STATS_MASK` seds. |
| `Track_2/t2_common.py` | 23 / 0 | `normalize_tgt` computes in float32 the kit's way. It is **bit-identical to the kit's torch expression** (checked on 200,000 random elements); a float64 version differs in 12 % of elements by up to 2 ulp. Also `normalize_prev(y, space)`. |
| `Track_2/stream_eval.py`, `Track_2/validate_t2.py` | 17/9, 6/2 | `--prev-space {target,input}`, default `target`, printed in the header. |
| `Track_1/verify/final_identity.py` | new, 153 | The committed identity verifier (see §2). |
| `Track_1/verify/graph_path.py` → `Track_1/closed/graph_path.py` | 8 / 1 | `git mv`, closed/ path shim added, usage line updated. A row was added to `closed/README.md` (+9). |
| `Track_1/submit_fno/band.npy` | deleted | `git rm`. |
| `Track_1/verify/{k0a_edge,k0a_dump,fuse_swin,fuse_ffno}.py` | 1-2 lines each | `getattr` for the removed `SMALL_GRAPH` and `GRAPH_*` attributes, so these tools do not crash at a print. |
| `EVALUATION.md`, `Track_1/submit_fno/README.md`, `Track_1/archive/README.md` | small | The `graph_path` path; the `GRAPH` row marked removed; a note that the README's constants table predates the final package; the `submit_trick` path. |

**Changed files inside the rebuilt archives** (md5 against the uploads, job 7569409): `submission.py` (T1, T2) and `_fno.py` (both; the M3 guard and comments). `band.npz` and all six `.pth` are **byte-identical** on both tracks.

## 2. Output identity (requirement A): tolerance 0

Driver: `Track_1/verify/final_identity.py`. Each package is imported in its own subprocess with `PYTHONDONTWRITEBYTECODE=1`. The gate is `np.array_equal`; the verifier also counts elements whose bytes differ.

| check | job | result |
|---|---|---|
| T1 `predict()` on val_unseen at N = 14, 56, 99, 336, successive calls in one process (pool fits, fits, exhausted, allocated), prediction / lower / upper | 7569409 (after ALL edits) | **IDENTICAL**, 0 bytes differ in all 12 arrays |
| T2 full 336-step stream, reset at boundaries, target revealed every step, **INPUT**-statistics harness | 7569409 | **IDENTICAL**, 0 bytes differ |
| T2, same stream, **TARGET**-statistics harness (what the official runner does) | 7569409 | **IDENTICAL**, 0 bytes differ |
| same three checks, on the tree before the final T2 kernel trim | 7569355 | IDENTICAL |
| first attempt | 7569343 | the verifier had an argv bug (duplicate `--worker`); fixed, no result |

The graph was CAPTURED in all 4 T2 workers.

## 3. Gates (requirement B), job 7569409, after all edits

- `BUILD_FINAL OK`: **gates 2/2 SUBMITTABLE**, `CAUSALITY OK` (176 steps, max |diff| 0).
- T1 `validate_strict`: SUBMITTABLE. Local rel_l2 0.0604, tke 0.3260, mvpe 0.0476; 1.527 ms/window.
- T2 `validate_t2 --adaptive`, now with the default TARGET harness: SUBMITTABLE, every check `[ok]`. Scores: rel 97.031, tke 86.574, mvpe 97.681, sps **59.985**.
  - The same package under the old INPUT harness read sps 60.011 (the `fdac00b` rebuild log). That −0.026 is the same M1 effect the twin arms show below.
- Syntax: `ast.parse` passes on all 12 changed `.py` files; `bash -n` passes on `build_round13.sh`.
- The three twin builds also ended `BUILD_FINAL OK` (2/2 SUBMITTABLE, `CAUSALITY OK`), and the worktree was restored after every build.

## 4. M3 / M4 behaviour (`jobs/w27p_guards.py`, job 7569409)

- **T1, M3.** One N = 56 call, with window 3 holding one NaN and window 7 scaled by 1e30:
  - Uploaded package: 163,840 non-finite elements in each of prediction, lower and upper, and `lower <= pred <= upper` fails.
  - New package: all finite and ordered; windows 3 and 7 return u, v = 0; the other 54 windows are identical to a clean call.
- **T1, M4, host band path vs GPU band path** (same GPU members, `GPU_BAND=False`, 8 windows): the prediction is identical. Bounds differ by at most 7.5e-9 (≤ 9.7e-8 of the field max) in about 15 % of elements.
- **T1, M4, real CPU run** (`CUDA_VISIBLE_DEVICES=`): it no longer raises; output is finite and ordered.
  - Against the shipped GPU path (fp16 members): max |d| 1.7e-3, rel-L2 1.3e-3. This is dominated by fp16 vs fp32.
  - Against a GPU run with fp32 members on the host band path: max |d| 1.8e-6 on the prediction (rel-L2 9.5e-7) and 8.9e-7 on the bounds.
- **T2, M3**, 80 steps over 2 trajectories:
  - Uploaded package: a NaN in one revealed target made 3,031,040 elements non-finite for the rest of the trajectory; a NaN in one input made 5,734,400 non-finite.
  - New package: everything finite and ordered. A NaN target gives outputs **identical** to withholding that target (`prev=None`) at every step. A NaN input zeroes only that step's prediction (steps 5 and 47, i.e. step 5 of each trajectory).
  - The spec only guarded the target. An intermediate version with only the target guard zeroed **every later prediction of the trajectory** after one NaN input, because the stored `gC0` / `gC` / `gH` poisoned D1c. The `gok` flag and the `nan_to_num(out=)` writes of the stored fields fix that.

## 5. Latency (interleaved ABBA, a fresh process per read, same node; uploaded vs new)

The job ran on a0129 alongside other users' jobs (`m_prefixb_s0_s1`, `cuda_test`); interleaving controls for them.

| | uploaded | new | Δ | rounds new > uploaded |
|---|---|---|---|---|
| T1 N=56, median of 20 calls, median of 5 rounds (7569411) | 1.5431 ms/window | 1.5493 | **+0.006 (+0.4 %)** | 4/5 |
| T1 N=56, first call after import (pool slice) | 1.3407 | 1.3435 | +0.003 | — |
| T2 median ms/step over 336 steps, median of 5 rounds (7569411) | 3.3494 | 3.4310 | **+0.082 (+2.4 %)** | 5/5 |
| T2, earlier guard version with ~22 extra kernels (7569357) | 3.3304 | 3.4293 | +0.099 | 5/5 |

GPU-only time per T2 replay (CUDA events, 200 replays, job 7569470; each variant in its own process, 3 rounds, stable to ±0.001):

| uploaded | new, guards removed | head only | tail only | new (full) |
|---|---|---|---|---|
| 3.090 ms | 3.091 | 3.122 (+0.031) | 3.140 (+0.050) | 3.171 (+0.081) |

- The T2 step is GPU-bound: replay 3.09 ms out of 3.35 ms. The stored-field `nan_to_num` costs about 0.001 ms.
- Every guard op sits on the critical path. The head runs before the R1 fork and before D1c/VARCAL; the tail runs after the join.
- At the CLAUDE.md T2 slope bracket (0.3-0.9 board ms per local ms) and about −0.072 final per board ms, +0.08 local ms is roughly −0.002 to −0.005 `final`. That is an estimate, not a measurement.
- Moving the head, D1c and VARCAL onto the R1 side stream would take them off the critical path. That is not implemented: it is a scheduling change outside this refactor.

## 6. T2 revealed-target normalization (M1), and the M2 arm

- The official runner (`official/t2_v6/.../local_eval.py`) normalizes the target with `mean_tgt/std_tgt` in `Normalizer.preprocess` (l. 111-115, called at l. 198). The next step's `prev_target` is that `tgt_norm` (l. 199 and l. 205).
- Twin packages: the five held-out twins, with the band and knobs of `build_final.sh`. Arms (a) and (b) are the **same** package, read two ways, so two packages cover arms a-c.
- Package `twin_tgtsm` = (c) + `STATS_MASK`. Reads come from `stream_eval.py` on val_unseen (job 7569410, identical to the pre-trim job 7569356).
- Composite = 0.4819 rel + 0.0981 tke + 0.0911 mvpe + 0.2491 sps.

| arm | subset | rel | tke | mvpe | sps | composite |
|---|---|---|---|---|---|---|
| (a) PREV input + harness input (our earlier local read) | all 8 | 96.3760 | 81.8367 | 96.8791 | 56.0075 | 77.2489 |
| (b) PREV input + harness target (what the board ran) | all 8 | 96.3760 | 81.8367 | 96.8791 | 55.9817 | 77.2425 |
| (c) PREV target + harness target (the fix) | all 8 | 96.3760 | 81.8367 | 96.8791 | 56.0075 | 77.2489 |
| (d) (c) + STATS_MASK | all 8 | 96.3755 | 81.8359 | 96.8769 | 56.0048 | 77.2477 |
| (a) | AoA 15-20 | 95.1603 | 82.2599 | 95.8117 | 48.6290 | 74.7694 |
| (b) | AoA 15-20 | 95.1604 | 82.2598 | 95.8118 | 48.6178 | 74.7666 |
| (c) | AoA 15-20 | 95.1603 | 82.2599 | 95.8117 | 48.6289 | 74.7694 |
| (d) | AoA 15-20 | 95.1603 | 82.2598 | 95.8118 | 48.6297 | 74.7695 |

- **c − b:** sps +0.026 on all 8 and +0.011 on AoA 15-20; composite **+0.0064** and **+0.0027**. **8/8 trajectories positive**, ranging from +0.0017 (8850_20) to +0.0123 (3750_0). rel, tke and mvpe are unchanged to four decimals: M1 moves only the band, through R1's table and D1c's error field.
- **c vs a:** effectively the same read. Predictions agree to 7e-8, bounds to 8.6e-4 (argmax flips); composites are equal to four decimals.
- **d − c:** composite −0.0012 on all 8 and +0.0002 on AoA 15-20, 3/8 trajectories positive. The largest move is 3750_0 at −0.0097. **Null on this split.**

My reading, which is local evidence only:
- The M1 mismatch the board ran costs a small, consistent amount of sps. On val_unseen, 9.95 % of u, v target elements are exactly 0.
- `PREV_SPACE="target"` recovers our local read exactly and wins on every trajectory.
- The size (+0.006 composite locally) is of the same order as the guard's latency cost in §5.
- `STATS_MASK` shows no benefit here. The audit's concern was a *different* blanking pattern on the private set, which this split cannot test.
- Both knobs stay default-off. Nothing was uploaded or staged.

## 7. Deliberately left alone

- **Not implemented, as instructed:** m9 (auto-reset), m3 (`GRAPH_TOL`), m5 (`_pick_batch`), m7 (env reads in `_fno.py`), m8 (`FFNO_IMPL`).
- **Kept, as instructed:** `BLEND_BETA`, `HIST_BETA`, `FW_N0` (whose FAMW code got the `_ym` target-space conversion), the err-head band `_ERR`, symmetric R1, the per-channel compose and the host band path.
- **Kept in `_fno.py`:** `LN_MANUAL` and `_PRED_FP` code, and all non-shipped families. `HOST_WRITE` and `SPEC_PATH` were kept and documented as inert.
- **The T1 host band path has no M3 guard.** The spec put it in the GPU block only; the host path now serves only CPU runs and the QD err-head recipe, and its comment says so.
- **`build_final.sh` is unchanged.** It still passes `RPDE_SHIP_SMALL_GRAPH=False` and `_LAZY=False`, which the refusal accepts as defaults.
- **`Track_1/verify/pkg_identity.py` is unchanged.** It still uses `normalize_in` and never reveals targets; `final_identity.py` supersedes it for this purpose.
- **`Track_1/verify/joint_gain.py` is unchanged.** Its check (c) exercises the removed `SMALL_GRAPH_LAZY` path, so that check cannot pass on this tree (not run).
- **READMEs were not rewritten.** `Track_2/README.md`'s "ship frozen" and the stale `submit_fno/README.md` tables are outside this scope, apart from the notes listed in §1. CLAUDE.md is untouched.

## 8. Surprises

1. `normalize_in`-style float64 target normalization is **not** what the kit computes. The kit's float32 torch arithmetic differs in 12 % of elements by up to 2 ulp. The new `normalize_tgt` matches the kit bit for bit. On a physical 0 both give the same value.
2. Several `FrozenEnsemble` objects in one process crash with CUDA "illegal memory access".
   - This reproduces on the **uploaded** package too (job 7569470, with `ARENA_STEPS` patched to 0 in both).
   - Single-object processes are fine, and every gate and read here uses one object per process.
   - Not investigated further.
3. The NaN-input poisoning in T2 (§4) was not in the audit's M3 description. Only the target case was.
4. The T2 guard's cost is set by the kernels on the critical path, not by their arithmetic. Cutting about 22 extra kernels to about 14 only moved +0.099 to +0.082 ms.

GPU use: 9 jobs, about 34 minutes of wall time in total (`sacct`).
