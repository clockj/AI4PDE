# The candidate's bands committed; `reproduce.sh decision` reuses them (smith, 2026-09-28) — identity jobs RUNNING at hand-back

Patch: `experiments/decision_20260928/patches/candidate_band.patch`, made with `git diff --binary` of `dec0928/wt_cand2`
against 467769d. It contains the two npz files as binary hunks. It applies cleanly to 467769d with `git apply --binary`.
When committing, the two npz files need `git add -f`, because `.gitignore` excludes `*.npz`.

## Hunks
| file | change |
|---|---|
| `Track_1/bands/dp_resim_all_sigblend_lam0.5.npz` (new) | byte copy of `cand/repro/log/decision/band/k0/band_k0_sigblend_lam0.5.npz`, md5 4d1777c8bf826382ee9f8af3dcac071f (source = copy) |
| `Track_1/sigblend/band_dp_resim_all_sigblend_lam0.5.json` (new) | byte copy of its json, md5 ce4ef04b89106ad5bd332dda0a1355b9 (source = copy) |
| `Track_1/bands/dp_resim_ffno_sigblend_lam0.5.npz` (new) | byte copy of `ood/cand2/band/k0/band_k0_sigblend_lam0.5.npz`, md5 42026ad8158a5b84a7c9059b2f8f8837 (source = copy) |
| `Track_1/sigblend/band_dp_resim_ffno_sigblend_lam0.5.json` (new) | byte copy of its json, md5 30312f89f81dc8843d9dee0518575c46 (source = copy) |
| `reproduce.sh` | `decision` only. The tag names the recipe: `dp_resim_all`, `dp_resim_ffno`, else `dp_resim_f<S>_v<S>` with `none` for an empty variable. `dp_band` reuses the committed band of that recipe by default and refits only with `RPDE_REPRO_FITBAND=1`; a recipe with no committed band is refused, and the message names `RPDE_REPRO_FITBAND=1`. `dp_build` takes the band and json from the tree or from the refit, and checks the band before `ensure_val`. The closing line says that `dp_build` built and gated both archives when it is among the stages; otherwise the old message, unchanged for every existing target. One line added to the `RPDE_REPRO_FITBAND` knob doc. |
| `REPRODUCE.md` §14 | rewritten: the two recipes and how to select them, the default band behaviour (reuse), the archive names, the acceptance read (77.2817 vs 77.2821, largest per-trajectory 0.0012, bar 0.005), what is not verified |
| `Track_1/bands/MANIFEST.md` | the stub replaced by two complete entries |

## Checks done without a GPU
* **Tag and band selection** for four pairs of values:
  * 0.25 / 0.25 gives `dp_resim_all`, and its band exists.
  * 0.25 / empty gives `dp_resim_ffno`, and its band exists.
  * 0.5 / 0.25 gives `dp_resim_f05_v025`, with no band.
  * empty / empty gives `dp_resim_fnone_vnone`, with no band.
* **Refusal, login node:** `RPDE_DP_RESIM_FFNO=0.5` with stage `dp_band`, and again with stage `dp_build`, gave exit 1 and a message naming `RPDE_REPRO_FITBAND=1`.
* **Reuse message:** `RPDE_DP_RESIM_VIT=` with stage `dp_band` printed the committed band's md5; it ends with the unchanged "no build" line because `dp_build` was not among the stages.
* **Test side effect:** the `dp_build` refusal test ran `make_val.py` on the login node before the reorder. It created `val_{seen,unseen}.npz` in wt_cand2, identical by md5 to the repository's.

## Identity (jobs 7580924 `rp_dp_band_ident_all`, 7580925 `rp_dp_band_ident_ffno`): RUNNING at hand-back
* **Logs:** `dec0928/cand/band_commit/logs/rp_dp_band_ident_{all,ffno}_<id>.log`.
* **Setup:** each job builds in its own copy of 467769d with the patch applied, with stage `dp_build` only and the default band behaviour (reuse).
* **Comparison:** `cand/band_commit/jobs/cmp_zip.py` compares every archive file by file with the existing candidate.
* **The record:** the ffno job then runs `build_final.sh` with default arguments in a third copy (`band_commit/rec/repo`), against the two uploads.

## Open points
* The job results.
* The on-scratch archive names of the two existing candidates stay `dp_resim` / `dp_resim_v`, as they were built. The MANIFEST names them as "the tag before the rename".
