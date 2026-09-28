# Training-side audit: final-recipe closure, dead weight, reproducibility (2026-09-27)

Read-only audit of `~/workspace/realpde/main` on branch `qwu/agent`, HEAD `55f0e88`, with the working tree
**dirty**: modified `_fno.py`, both `submission.py` files and `build_round13.sh`; untracked `build_final.sh`,
`Track_1/sigblend/` and both final bands. Nothing was modified. The only thing run was CPU `torch.load(...)["config"]`
on the 11 final checkpoints. All line numbers refer to the working tree as it stood on 09-27.

**Bottom line.** The 11 trained artefacts are fully pinned by `reproduce.sh` stages that already exist, and they have
been re-trained to within the seed spread (REPRODUCE §9, §12). **The final packages themselves cannot be rebuilt from
any commit.** The code that bakes and evaluates the final constants is uncommitted. The SIGBLEND chain (dump →
`make_k0_band` → `build_final.sh`) is untracked and not wired into `reproduce.sh`. And `REPRODUCE.md`'s
`FINAL_RECIPE_T1/_T2` still reads `<TO BE DECIDED>`. Section 3 lists the fixes.

---

## 1. Final-recipe dependency closure

### 1.1 The 11 training commands

All 11 were read back from `config` / `config["train2_argv"]` and match `reproduce.sh`:

| artefact | reproduce.sh | flags (beyond `--data`) |
|---|---|---|
| `ffno_all_w40l8_sim_s{1..4}` | `members_ffno` :174-177 | `$FFNO $SIMF --seed s --all-data` |
| `p2_ffno_l8_sim_s{1..3}` | `twins_ffno` :160-165 | `$FFNO $SIMF --sim-exclude-held --seed s` |
| `arch_vit_all_s1` | `members_vit_p` :172 | `$VITP --seed 1 --all-data` |
| `arch_vit_all_cmb3sim_s3` | `members_vit_c` :173 | `$VITC --seed 3 --all-data $SIMF` |
| `arch_vit_d384` | `twins_vit_p` :158 | `$VITP --seed 1` |
| `arch_vit_cmb3sim` | `twins_vit_c` :159 | `$VITC --seed 1 $SIMF --sim-exclude-held` |

The fragments are at `reproduce.sh:115-126`: `LOSS`, `OPT`, `FFNO`, `VITC`, `VITP`, `SIMF`.

- The F-FNO checkpoints' `train2_argv` also carries `--ckpt-every 20 --resume`. These are inert (REPRODUCE §1.4).
- `reproduce.sh` passes these explicitly: `--wd 1e-5 --sim-map scalar --sim-stride 5`. The shipped F-FNO runs took
  them as defaults. The values are identical.
- The ViT checkpoints predate the `drop_path`, `oracle_cond`, `teacher*`, `ckpt_every` and `err_*` keys.
  `arch_vit_d384` also lacks `lam_cov`, `sim_map` and 10 others. Every missing key is "off" today (REPRODUCE §1.3).
- The F-FNO configs lack only `err_head`, `err_detach`, `lam_e`, `err_loss`, `qd_temp` and `pinball_tau`. The members
  were trained at `7ea0a86` / `d8926cb`, before the QD work.

### 1.2 `train2.py`: flags the recipe sets vs leaves at default

**Set by the recipe, with an argparse default that differs** (these are the traps):

| flag | line | default | recipe |
|---|---|---|---|
| `--epochs` | 192 | 150 | 400 |
| `--seed` | 193 | -1 (unseeded) | 1..4 |
| `--width` | 200 | 64 | 40 (F-FNO) / 384 (ViT) |
| `--var-head` | 203 | off | on |
| `--loss` | 291 | `rel` | `mse` |
| `--alpha-train` | 296 | 1.0 | 0.0 |
| `--lam-f` | 298 | 1.0 | 0.0 (`cmb3sim`) / 1.0 (VITP) |
| `--lam-v` / `--lam-t` | 305 / 306 | 1.0 / 0.5 | 0.15 / 0.15 (VITP: 0.3 / 0.15) |
| `--arch` | 407 | `unet` | `ffno` / `vit` |
| `--modes-y/-x` | 420-421 | 12 / 16 | 16 / 32 (F-FNO; the ViT ignores them) |
| `--n-layers` | 422 | 4 | 8 |
| `--lam-grad` | 435 | 0 | 1.0 |
| `--lam-tgrad` | 439 | 0 | 0.3 (VITP: 0) |
| `--dropout` | 525 | 0 | 0.3 |
| `--lam-cov` | 564 | 0 | 2.0 (VITP: 0) |
| `--sim-loss` | 664 | `shape` | `full` |
| `--all-data` | 744 | off | members only |

**Left at default, and the default is the recipe:**

- `--batch 32` (:198), `--stride 5` (:199), `--lr 2e-4` (:201)
- `--wd 1e-5` (:601), `--cov-c 0.25` (:576), `--var-hidden 128` (:239)
- `--norm real` (:382), `--ctx 20` (:388), `--phases 1` (:465), `--chan-pow 0` (:459), `--w-mean 1.0` (:340)
- `--lam-sim 0.3` (:661), `--sim-map scalar` (:675), `--sim-stride 5` (:687)
- `--exclude None`: falls back to `model.VAL_TRAJ` at :841 for twins, and to `[]` for all-data runs

**Not a flag at all:** the ViT drop-path schedule. It is fixed at 0.1 as a class default (`models_alt.py:891`).
`build()` does not pass it (:587-591), and `train2.py:816-818` refuses `--drop-path` for `vit`.

### 1.3 The active training path in `train2.py`

| step | lines |
|---|---|
| seeds: torch, CUDA, numpy only | 832-835 |
| `dev = "cuda"`, hard-coded | 837 |
| split `excl` | 841 |
| raw physical trajectories | 856-857 |
| `window_index` | 858 |
| sim load (`load_sim_mapped`; `real_exclude=excl` always, `exclude=excl` iff `--sim-exclude-held`) | 900-914 |
| `mean_v` / `std_v`; `cw = 1` at chan_pow 0 | 954-965 |
| `build(...)` | 981-995 |
| AdamW + OneCycleLR | 1059-1061 |
| `fetch` | 1063-1081 |
| resume | 1090-1131 |
| `randperm` epoch loop | 1134-1145 |
| normalization (s = 1) | 1152-1154 |

Loss terms active in the final recipe, in `compute_loss` (1215-1566):

- **Field MSE** (1286), weighted by `lam_f` (1288). The weight is **0** for the four `cmb3sim` recipes (F-FNO ×7,
  ViT-c ×2) and **1** for the two plain ViTs.
- **Welsch "coverage"** `l_cov` (1302-1312), with `cn = 0.25·SPS_SIGMA·cw/mul` (`SPS_SIGMA` :186). Active only in
  `cmb3sim`.
- **`l_grad`**: rel-L2 of `velocity_gradient` (1352-1358). Active in all 11.
- **`l_tgrad`**: rel-L2 of `time_derivative` (1359-1365). Active only in `cmb3sim`.
- **σ head**: log-std MSE `l_var` plus the TKE term `l_tke` (1373-1387). Active in all 11. `compose(...,
  alpha_train=0)` is the identity (`model.py:321-322`), so the field loss sees the raw field.
- **Sim `full` loss**: one sim batch per real step, mean/fluctuation split MSE, weight `lam_sim` 0.3
  (1457-1500, branch 1486-1490). Active in every run with `--sim` (9 of 11; not `arch_vit_all_s1` or `arch_vit_d384`).
  It is **not** Welsch-weighted.
- The final model is written at 1650-1686: `config` plus `git` and `train2_argv`.

Everything else in `compute_loss` is off in the final recipe: rel loss, tail, subpix, probe, div/vort/spec, err
head, FGN/CRPS, sim p-head, aux stride, mixup/aug, teacher, SAM, SWA.

### 1.4 Classes and functions exercised

**`models_alt.py`**

- **F-FNO path:** `build` 547-612 → `build_wave1` 1905-1940 (ffno branch 1925-1927, `ff=True`) → `FFNO2d` 1291-1368.
  - `FFNO2d` is built from `_GridIn` 684-694, `_FFNOLayer` 1184-1288, `_LayerNorm2d` 1003-1007 and `_heads` 649-681
    (field and log-std MLPs, no err head).
  - Training uses the FFT path `_FFNOLayer._spectral` (1268-1276). `_spectral_dense` (1240-1266) runs only at
    inference, via `set_spectral_path`.
- **ViT path:** `build` → `ViT2d` 869-940 (with `_heads`); heads = width // 64 = 6.
- **Physics helpers:** `velocity_gradient` 484-499, `time_derivative` 502-522.
- **Imported but unused:** `divergence`, `vorticity`, `radial_spectrum` (imported at `train2.py:974`).
  `radial_index` runs unconditionally at :976, with no effect on the model.

**`model.py`**

- Constants: `MEAN/STD` 34-35, `NORM_STATS` 48, `VAL_TRAJ` 51-52, `PHASES_ALL` 59.
- Functions: `load_trajectories` 62-90 (lazy h5py), `trajectory_names` 93-103, `window_index` 115-123,
  `to_ch/from_ch` 165-172, `std_t` 300-306, `compose` 309-329, `tke` 332-342, `self_norm` 356-365 (s = 1), `rel_l2` 374-378.
- `UNetTwoHead` 188-249 is imported but used only by `arch == "unet"`.

**`simreal.py`**

- `load_sim_mapped` 106-278, `mode="scalar"` only (speed match 223-224, scalar gain 228-230, mask 222 and 245).
- `_stats` 87-93, `C_AOA` 82, `TI_REF` 84, cache 144-160 and 272-277, `CACHE` default 103.
- Not used: the per-pixel `std`/`meanstd` branch (231-244), `with_p` (188, 246-250), `sim_names` 281-287.

**At inference / band time**, `_fno.load_any` (`Track_1/submit_fno/_fno.py:2300ff`):

- ViT: always the **vendored** `_fno.ViT2d` (:752, branch :2305-2321).
- F-FNO: the vendored `_fno.FFNO2d` (:1786) inside a package. **Inside the repo it is `models_alt.FFNO2d`**, because
  `FFNO_IMPL = "auto"` (:1669, :1859-1861) imports `../unet/models_alt.py` (:2343-2362).
- So `fit_band.py`, `sigblend/dump_twins.py` and every local instrument run **the `models_alt` definition**, and a
  refactor of `models_alt.py` changes what the band fit executes.

### 1.5 Band and SIGBLEND closure (`Track_1/sigblend/`, untracked)

1. **Static band.** `fit_band.py --alpha 0.80 --band-p 0.45 --fill-unscored --centre-alpha 0` (inflate defaults to
   1.39) gives `bands/r38_a080_c0_fill.npz`, md5 `0cb37d82`. This is identical to
   `/fs/scratch/.../final0927/pkg/x_work/band_r38_a080_c0_fill.npz`.
2. **Twin dump.** `dump_twins.py` runs the 5 twins at alpha 0 on `val_unseen` and writes F and S, the per-member
   fields and σ (:9-25).
3. **Power-law fit.** `make_k0_band.py` fits q_c and log κ_{t,c} on the 8 VAL_TRAJ trajectories (`fit_pl` :34-58). It
   writes `base' = base^(1-λ)·κ^λ` and `p' = (1-λ)·0.45` (:109-114), plus the EBAND literal and the T2 SIGPOW
   exponents in the JSON (:115-119).
   - It imports `bandlab.Lab` (`bandlab.py:52-73`: loads the dump, compose at 0.80), `sps9.SIGMA` and
     `sps9.optimal_band` (`sps9.py:75`, used by `bandlab.run`).
   - It needs `RPDE_KIT` (`sps9.py:30`) and `Track_1/val_unseen.npz` (built by `make_val.py`).
   - The committed `bands/k27_sigblend_lam0.5.npz` (md5 `e8309d9f`) equals the scratch output. It keeps `sigma_ref`
     and the p channel of `base` from the r38 band (checked).
4. **Build.** `build_final.sh` reuses that band with `RPDE_BAND`. `RPDE_BAND_FILL=1` is then ignored
   (`build_round13.sh:72-80`). The literals are read from `band_k0_sigblend_lam0.5.json` (:25-26).

---

## 2. Dead weight

"Importers" means tracked code outside `unet/`, from `git grep`. A checkpoint blob stores only a `state_dict` of
tensors plus numpy and config values, **no pickled classes**. So moving a class to another module never breaks
`torch.load`. It only breaks the code path that *rebuilds* the module: `models_alt.build(arch)`, which `_fno.load_any`
reaches by `importlib.import_module("models_alt")` (`_fno.py:2355`).

### 2.1 Model code (`models_alt.py`, 1955 lines)

The final recipe uses about 330 lines (§1.4).

| item | file:line | LoC | last used (evidence) | importers / checkpoints on disk | archive? |
|---|---|---|---|---|---|
| `FNO2d` (+ `SpectralConv2d`) | 81-400, 26-72 | 367 | R25 package of record (superseded; `reproduce.sh r25`); FNO was the member family through R28 | `screen_latency.py:108`, `closed/screen_oracle.py:75`, `verify/phead_strip.py:49`, `verify/aug_mask_check.py:124`, `closed/mcdrop_v9.py:70`, `eval_wp.py:100`; `_fno.py:152` has its own copy; ~250 fno_* checkpoints | **Keep in the main module**: `reproduce.sh r25` trains it |
| FNO2d sub-features: osc head, mean head, tbasis, FGN film, mode-drop, p-head, sub-head, cond | inside FNO2d, e.g. :135, :160-171, :185-192, :244-247, :270, :288 | ~150 | closed: osc (`ROADMAP.md:1271-1280`), FGN/CRPS (round 11-12, `ROADMAP.md:1070-1071`), oracle (09-14), sim p-head, subpix | `closed/osc_probe.py:44` imports `OSC_W_HI/LO` (:78) | leave in place: removing them changes `FNO2d.__init__`, and the vendored copy mirrors it |
| `DeepONetCh` | 403-451 | 49 | early rounds (`ROUNDS.md`, 13 mentions) | `build("deeponet")` only; no checkpoint in `unet/` | yes |
| `AFNO2d` | 697-775 | 79 | round 24 marginal-sps test (`ROADMAP.md:853`) | `verify/diff_arch.py:83`; vendored at `_fno.py:678`; `arch_afno_*` ×3 | yes, behind a shim |
| `DilResNet` | 778-818 | 41 | round-15 family screen | `arch_dil*` ×2 | yes |
| `UFNO2d` | 821-866 | 46 | R16 / T2R4 board (−0.45, withdrawn; `ROADMAP.md:611-641`) | `verify/diff_ufno.py:43`, `diff_arch.py`; vendored `_fno.py:615`; ufno/kfre_ufno ×12 | yes, behind a shim |
| `CNeXtUNet`, `_ConvNeXtBlock` | 1032-1090, 1013-1029 | 76 | 09-14 wave-1 screen, closed (`ROADMAP.md:1096`) | none; no checkpoint in `unet/` | yes |
| `UNetMod`, `_PreActResBlock` | 1115-1178, 1096-1112 | 81 | wave 1, closed | none | yes |
| `ffno_nff` variant | branch in 1197-1212, 1925-1927 | ~10 | wave 1 | `_fno.FFNO_VARIANTS` | leave (inside `_FFNOLayer`) |
| Swin family (`SwinUNet`, `_WindowAttnV2`, `_SwinV2Block`, `_Patch*`, `_TokenProj`, `_win_*`) | 1374-1656 | ~285 | R26 board (T1 Swin); capacity/sim closed 09-14 (`ROADMAP.md:1104-1109`) | `verify/fuse_swin.py`, `closed/screen_profile_swin.py:52`; vendored `_fno.py:1005-1330`; `swin_all_w64_*` ×4 | yes, behind a shim (`SWIN_IMPL="auto"` imports it in the repo) |
| `SimVP2` + gSTA blocks | 1662-1791 | 124 | wave 1, sps −2.0 (`ROADMAP.md:1102`) | none | yes |
| `STViT` + `_MHSA`, `_DividedSTBlock` | 1797-1902 | 108 | wave 1 | none | yes |
| `oracle_cond_features` | 1950-1955 | 6 | 09-14 oracle diagnostic | `train2.py:871`, `closed/screen_oracle.py` | yes |
| `divergence`, `vorticity`, `radial_spectrum`, `radial_index` | 454-481, 525-544 | 44 | `--lam-div/vort/spec` closed (CLAUDE.md "Closed") | imported unconditionally, `train2.py:974-976` | leave: the import line would have to change |
| `err_head` path | `_heads` 676-681; FFNO2d 1314-1319, 1339-1368 | ~45 | QD candidate (09-24/25, `REPRODUCE.md` §10), **not in the final package** | `reproduce.sh --recipe QD`, `qd_k/head_direct.py:71`, vendored `_fno.py:1811` | leave: off by default, adds no module |

About **1,000 of the 1,955 lines** are closed families that no final artefact needs.

### 2.2 `train2.py` (1691 lines): closed arms

The recipe exercises about 450 lines of `main()`. The closed arms together are about 700 lines of argparse and
branches:

- `aug_mask` 64-127; `--aug-scale/noise` 1178-1198
- SAM 130-161 and 1571-1581; SWA 1587-1598 and 1626-1628
- `aux_index` 164-180; aux loss 945-952 and 1525-1559
- subpix 842-853, 1233-1238, 1313-1320, 1631-1638
- oracle 819-830, 867-876, 1239-1243
- sim p-head 780-784, 915-943, 1467-1523 (p parts), 1639-1649
- teacher / distillation 1009-1057, 1204-1213
- FGN / CRPS 764-770, 1427-1455
- mixup 1168-1177; lam-tail 1289-1301; lam-p 1321-1337; div / vort / spec 1339-1372
- err head 1389-1425
- hist / ctx 967-971, 1066-1080
- `--loss rel` 1268-1277; `--sim-loss shape` 1491-1499
- aoa-boost / aoa-only 877-886; `--init` 998-1007 (DINOv2 transplant target)

**Do not delete any of this before the Decision phase** (§5). Every branch is guarded, so the default RNG stream is
untouched: REPRODUCE §12 records Corin's code review of the err-head commits, and `verify/aug_mask_check.py` checks
aug-mask.

### 2.3 Tracked helper files under `Track_1/unet/`

| file | lines | status | referenced by | action |
|---|---|---|---|---|
| `train.py` | 69 | original U-Net trainer; superseded by `train2.py` in WP1 (07-29) | docstring of `Track_1/submit/_package/model.py:4` | move to `Track_1/closed/unet/` |
| `dinov2_init.py` | 196 | DINOv2 transplant into ViT2d; closed and now **prohibited** (outside pretrained weights) | none (self only; default `--src` scratch :169) | move to archive; keeping it in the release invites the question |
| `distill_labels.py`, `distill_pipeline.sh` | 230, 103 | distillation closed (`ROADMAP.md:1172`) | **`reproduce.sh:386-391`** (`STUDENT=1`, default `STUDENT_OUT` on scratch :389) | remove the `student()` block from `reproduce.sh`, then archive both |
| `run_ctx.sh`, `run_sim.sh`, `run_sim2.sh` | 21-24 each | round-3/4 era sim arms; hard-coded `cd /users/PAS1187/...` (:11-14) | none | archive |
| `runq.sh`, `queue.txt`, `queue_alldata.txt`, `queue_round10.bak`, `done.txt` | 35 / 5 / 34 / 40 / 63 | round-10 queue driver and its state (last touched 08-29); `runq.sh:8` hard-codes PY | none | delete (history is in git) |
| `log_*.txt` ×49 | 372 KB | FNO w16-w64 / vmlp / ph4 logs of rounds 8-13 | none | delete, or move to scratch |
| `runq_nohup.log`, `__pycache__/` | — | untracked | — | ignore |

Also untracked and stale: `Track_2/submit/fno*.pth` are symlinks to `Track_1/unet/fno_all_vl_*` (the round-10 to 13
members). Loading `Track_2/submit/` as a submission dir loads those, not the final set. Deleting the `fno_all_vl_*`
checkpoints would leave them dangling.

---

## 3. Reproducibility defects

### BLOCKER

**B1. The final packages cannot be built from any commit.**
- `build_final.sh` bakes `SIGPOW`, `JOINT_GAIN`, `PIN_OUT`, `BATCH_CAP`, `ARENA_STEPS`, `R1_TSM`, `R1_BLK(_N0)`,
  `D1C_EB(_N0)`, `D1C_LEAD_N0` and `VC_N0`.
- None of these constants exists in HEAD's `Track_2/submit/submission.py`, `Track_1/submit_fno/submission.py` or
  `_fno.py`; they exist only in the working tree (checked by grep against `git show HEAD:`). The `sed` lines for
  them are also uncommitted (`git diff build_round13.sh`: +20 lines).
- `build_final.sh`, `Track_1/sigblend/` (5 files + json), `bands/k27_sigblend_lam0.5.npz` and
  `bands/r38_a080_c0_fill.npz` are all untracked. `.gitignore:12` whitelists `bands/*.npz`, so they can be committed.
- The byte-identity proof (`wrap0927/logs/rp_w27_verify_7569053.log`: "ALL 9 FILES BYTE-IDENTICAL" on both tracks)
  was run from `wrap0927/wt_verify`, a **non-git copy** of the dirty tree.
- **Fix:** commit all of the above in one commit. Re-run that verify job from a `git worktree` of that commit, and
  record the commit hash in `REPRODUCE.md` and `bands/MANIFEST.md`. MANIFEST has no `k27` or `r38_a080` entry today.

**B2. `REPRODUCE.md:10-11` still reads `FINAL_RECIPE_T1/_T2 = <TO BE DECIDED ...>`.**
- The packages of record are T1 `track1_k27fin_a080.zip` (board 81.671149) and T2 `T2_MRGP2_track2_MRGP2.zip`
  (82.055553), per `build_final.sh:6-7`.
- Neither is "A", "QD" or "QD_STATIC" as the header defines them. Both are recipe A's members and twins plus the
  SIGBLEND band, T1 EBAND and T2 SIGPOW, the stream stack (R1, D1C, VC) and the latency knobs.

**B3. `reproduce.sh` has no target that produces the final packages.**
- `both`, `t1` and `t2` build **r35** (ALPHA 0.60, `bands/r35_centre0_fill.npz`; :405, :203-220).
- `--recipe A` builds r38 at 0.80 with a refit static band (:354-367).
- No stage runs `dump_twins.py`, `make_k0_band.py` or `build_final.sh`.

**B4. `sigblend/dump_twins.py` hard-codes its inputs.**
- `U = "/users/PAS1187/.../Track_1/unet/"` (:10), the output path on scratch (:12) and
  `np.load("/users/PAS1187/.../Track_1/val_unseen.npz")` (:14).
- On any other machine the SIGBLEND refit dies at line 14. The README (:9) says "edit U / out paths"; a reproduction
  must not need editing.
- `bandlab.py:17` (`B_DUMP`) and `make_k0_band.py:20,23` (`K0_BAND`, out dir) default to scratch paths but can be
  overridden. The JSON records absolute scratch paths (`shipped_band`, `npz`), which is informational only.

### MAJOR

**M1. `reproduce.sh`'s gate check cannot pass for the final Track-2 package.**
- `build()` requires exactly two `^SUBMITTABLE$` lines (:240-242).
- `build_round13.sh` gates Track 2 as a *frozen* package. The adaptive stream stack fails its stateless replay:
  `rp_w27_verify` log line 106, "NOT SUBMITTABLE: stateless ... max diff 1.77e-01".
- `build_final.sh:57-60` re-gates with `validate_t2.py --adaptive` and `stream_eval.py --causal-check`. It only
  `grep`s their output: **its exit code depends only on the optional identity checks (:74-77)**.
- **Fix:** have `build_final.sh` fail unless all three hold:
  - T1 `SUBMITTABLE`
  - T2 `--adaptive` `SUBMITTABLE`
  - `CAUSALITY OK`

**M2. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is documented (REPRODUCE §0) but no script sets it.**
- `_fno.load_any` (`_fno.py:2300`) and the `--resume` load (`train2.py:1096`) use `torch.load` without
  `weights_only`. Checkpoints and resume files carry numpy (`mean/std`, `rng_np`), so under torch ≥ 2.6 the band fit,
  every gate and every resumed run fail.
- The verified build stack is torch 2.7.1 (REPRODUCE §0).
- **Fix:** export it at the top of `reproduce.sh` and `build_final.sh`. It is harmless on 2.2.2.

**M3. `PY` is not propagated to the child scripts.**
- `reproduce.sh:46-48` falls back to `python3` but does not `export PY`.
- `build_round13.sh:21` and `build_final.sh:18` default to `/users/PAS1187/qwu/.conda/envs/realpde/bin/python` with no
  fallback.
- A run that relies on the fallback trains fine and then fails at the build. **Fix:** `export PY` in `reproduce.sh`,
  and give the two build scripts the same `-x` fallback.

**M4. The SIGBLEND band has never been re-derived.**
- There is no record that `make_k0_band.py`, run on a fresh dump, reproduces md5 `e8309d9f`.
- The dump is written on GPU or CPU (`dump_twins.py:15`, `_fno.predict_fno`, amp None), and q comes from quantile
  bin edges (`make_k0_band.py:43`). A low-bit difference can move the constants.
- **Fix:** a CPU job doing dump (shipped twins) → `make_k0_band` → compare against the committed band's `base`, `p`
  and the JSON's q and logκ. Record the outcome. For a *reproduction*, reuse the committed band and JSON (the
  `band_env` rule, `reproduce.sh:248-250`).

**M5. Where the r38 static band comes from is not scripted for the final chain.**
- `build_A` keeps the 0.80 fit only when `RPDE_FINAL_ALPHA` is set (:360-361, copied from
  `Track_2/submit/band8.npz`).
- The SIGBLEND step needs it as `K0_BAND`. **Fix:** write the fit to `$OUT/band_r38_a080_c0_fill.npz` explicitly
  (call `fit_band.py` directly, as `qd_k_derive` does at :314-318), or reuse `bands/r38_a080_c0_fill.npz`.

**M6. `build_final.sh` defaults its output to `../sub_candidates`.**
- `OUT=${1:-../sub_candidates}` (:19). An organiser's clone has no such dir; ours holds the uploads.
- `reproduce.sh` should always pass an explicit scratch `OUTDIR`.

**M7. The sim-map cache key omits the data identity.**
- `simreal.py:146-151` keys on phases, frames, mode, `_p` and a hash of `real_exclude`, but not on `sim_dir`,
  `real_dir` or a data md5.
- `CACHE` defaults to `/fs/scratch/PAS1187/qwu/tmp/realpde/work` (:103), and `reproduce.sh:57-60` falls back to
  `$OUT/simcache` only when that path is not writable. A stale or foreign cache there is used silently.
- **Fix:** have `reproduce.sh` always set `RPDE_SIMCACHE=$OUT/simcache` unless the caller sets it (REPRODUCE §6.8
  already names this).

### MINOR

- **m1. Hard-coded defaults, all overridable:**
  - `train2.py:191` `--data`
  - `make_val.py:24` `DATA`
  - `re_split.py:38` `DATA`
  - `sps9.py:30` `RPDE_KIT` default
  - `reproduce.sh:49` `RPDE`, `:81` (Pitzer `mural_osc` hint), `:389` `STUDENT_OUT`
  - `distill_pipeline.sh:35-37`
  - `run_*.sh:11-14`
  - `runq.sh:8`
- **m2. Slurm and cluster assumptions** live only in non-pipeline job scripts: `sigblend/rp_k27_final.sh:2-8`
  (`-A PAS1186 -M ascend -p nextgen`, K-sprint scratch paths :13-24). It is documentation, not a pipeline step.
  `reproduce.sh` has no `sbatch` and needs one GPU per training stage.
- **m3. Non-determinism.** Seeds are set for torch, CUDA and numpy (`train2.py:832-835`). Nothing sets
  `cudnn.deterministic` or `use_deterministic_algorithms`. There is no DataLoader and so no worker RNG: batches come
  from `torch.randperm` and `torch.randint` on the CPU generator (:1138, :1462). The effect is measured: F-FNO
  retrains end ~0.5 % apart and ViT ~10 %, both inside the score seed spread (REPRODUCE §12). Nothing to fix; state it
  in the handover.
- **m4. `dev = "cuda"` is hard-coded** (`train2.py:837`), so no CPU smoke is possible. A 1-epoch GPU smoke is the
  cheapest real test.
- **m5. The held-out split is a typed constant,** `model.VAL_TRAJ` (`model.py:51-52`). It is the definition and not a
  derived list, and `make_val.py:32-60` asserts it still matches. `re_split.excluded()` is derived and no final
  artefact uses it. Acceptable as is.
- **m6. `environment.yml:22-26` has two `--index-url` lines in one pip block; pip keeps the last.** torch 2.2.2 is
  therefore resolved from PyPI, whose Linux wheel happens to be cu121. Use `--extra-index-url`. Neither this file nor
  REPRODUCE §0 pins the actual training stack (torch 2.7.1 / numpy 2.2.6). The CNO-baseline pins (`requests`,
  `setuptools`, :36-40) are dead.
- **m7. Code provenance.** The F-FNO members are `7ea0a86-dirty` / `d8926cb-dirty`, and the ViTs carry no hash at all
  (REPRODUCE §6.7). Retrains match on score, not on checksum; known and accepted.
- **m8. `reproduce.sh:115-126` sets `--wd 1e-5 --sim-map scalar --sim-stride 5` explicitly, and the shipped F-FNO
  argv did not.** Same values, so no effect, but key-by-key argv comparisons show a spurious difference.

### What to add so one command reproduces the final packages

In `reproduce.sh`, add a case `final)` (make it the default, replacing `both → r35`):

```bash
export PY TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 RPDE_SIMCACHE=${RPDE_SIMCACHE:-$OUT/simcache}
final) twins_vit; twins_ffno; members_vit; members_ffno; build_final_pkg ;;
build_final_pkg() {
  ensure_val
  local A="$(_A_R38)" M="$(_M_R38)" W="${RPDE_REPRO_LOG:-${TMPDIR:-/tmp}}/final"; mkdir -p "$W"
  if [ "${RPDE_REPRO_FITBAND:-0}" = "1" ]; then            # refit chain (new member set)
    $PY Track_1/submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha 0.80 --band-p 0.45 \
        --fill-unscored --centre-alpha 0 --out "$W/band_r38_a080_c0_fill.npz" || exit 1
    $PY Track_1/sigblend/dump_twins.py --unet "$P2" --val Track_1/val_unseen.npz "$W/dump_unseen.npz" || exit 1  # needs args (B4)
    ( cd Track_1/sigblend && B_DUMP="$W/dump_unseen.npz" K0_BAND="$W/band_r38_a080_c0_fill.npz" \
        K0_INFL=1.29 K0_NB=16 PYTHONPATH=.. $PY make_k0_band.py "$W/k0" ) || exit 1
    BAND="$W/k0/band_k0_sigblend_lam0.5.npz"; SB="$W/k0/band_k0_sigblend_lam0.5.json"
  else                                                     # reproduction of the scored package
    BAND=Track_1/bands/k27_sigblend_lam0.5.npz; SB=Track_1/sigblend/band_k0_sigblend_lam0.5.json
  fi
  RPDE_A="$A" RPDE_M="$M" RPDE_BAND="$BAND" RPDE_SIGBLEND="$SB" RPDE_FORCE=1 \
    bash Track_1/submit_fno/build_final.sh "${RPDE_REPRO_PKG:-$W/pkg}" || exit 1   # with M1's hard gate
}
```

Plus:

- `dump_twins.py`: take the twin list or dir, the val path and the output as arguments, with today's values as
  defaults.
- `build_final.sh`:
  - M1: fail on T1 `SUBMITTABLE`, T2 `--adaptive` `SUBMITTABLE` or `CAUSALITY OK` missing.
  - M2/M3: export the torch env var and give `PY` a fallback.
  - `RPDE_REF_T1/T2` identity stays optional.
- `REPRODUCE.md`: fill `FINAL_RECIPE_T1/_T2` with:
  - the commit,
  - the `reproduce.sh final` command,
  - band md5s `0cb37d82` (r38) and `e8309d9f` (k27),
  - the two uploaded archive names, and the byte-identity result.

  Move the A/QD/QD_STATIC block below it as history.
- `bands/MANIFEST.md`: add entries for `r38_a080_c0_fill` and `k27_sigblend_lam0.5`: provenance, the command,
  λ 0.5, infl 1.29, nb 16, q = (0.4847, 0.4907).

---

## 4. Checkpoint inventory (`Track_1/unet/*.pth`: 334 files, 18.64 GB, untracked)

**How classes were assigned.** Each file stem was matched against:

1. the 11 final names;
2. `reproduce.sh` and `build_final.sh` (seed loops expanded, s1-s4 or s1-s3 only);
3. any other tracked `.py`/`.sh` (instruments, verifiers, old build scripts);
4. `board_rounds.py`, `board.jsonl`, `bands/MANIFEST.md`, `final_v9.py` and `predict_board.py`;
5. any tracked `.md`.

The first match wins. The board files name packages, not checkpoints, so historical board members fall mostly into
class C (old `build_round1x.sh`) or class E (ROUNDS/ROADMAP).

| class | n | GB | name patterns |
|---|---|---|---|
| **A. FINAL** | 11 | 0.27 | `ffno_all_w40l8_sim_s1..4`, `p2_ffno_l8_sim_s1..3`, `arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3`, `arch_vit_d384`, `arch_vit_cmb3sim` (read-only copy also in `~/RealPDE/checkpoints_r34/`) |
| **B. `reproduce.sh` superseded recipes** (r25, r28/T2R13) | 14 | 0.83 | `fno_all_w96L8cmb3sim_s1..4`, `fno_r18_cmb3sim{,_s2,_s3}`, `ffno_all_w40l8_s1..4`, `p2_ffno_l8_s1..3` |
| **C. named by committed code** | 44 | 1.46 | `fno_all_vl_*`/`fno_vl_*` (rounds 10-13 members; `build_round10-13.sh`, `calib_band13.py`, `proxy_calib.py`, `alpha_board.py`, and the untracked `Track_2/submit/fno*.pth` symlinks); `unet_g1`, `unet_all_g1` (`_fno.py` docstring, `build_round12.sh`); `arch_vit_*cmb3sim_s{1,2,3,4}`, `_d2` (`verify/fuse_vit.py`); `arch_ufno_w96*`, `arch_afno_w96` (`verify/diff_ufno.py`, `diff_arch.py`); `fno_all_w96L8do45_s1` (`quant8.py`, `verify/quant_identity.py`); `fno_r17_simp` (`verify/phead_strip.py`); `fno_all_w96do30_s1` (`build_zip.py`); `fno_ozdo*`, `fno_fgn1`, `fno_simA_ctl`, `fno_cap_w96_do30*`, `fno_r16_w96_do45_L8*` (`closed/*`); `re_ctrl1/2` (`mean_blend_re.py`); `fno_all_do05_s1` (`lat_r15.py`) |
| **D. named only in board / manifest files** | 2 | 0.23 | `fno_all_w96L8do45sim_s1`, `fno_r17_sim` |
| **E. named only in docs** (ROUNDS / ROADMAP / READMEs; historical board members and round write-ups) | 69 | 5.86 | `fno_all_w96L8_s1..4`, `fno_all_w96L8do45_s2..4` (R2x members), `swin_all_w64_s1` (R26), `ufno_all_w96do30_s1` (R16/T2R4), `ffno_all_w40l6_s1,s4`, `arch_dil_w96`, `arch_unet_w96`, `fno_r16_*` / `fno_r17_*` round arms, `fno_cap_*`, `fno_sim_t1..3`, `kfre_*_f0`, `re_do15`, `re_self`, `fno_do05*`, `fno_osc2`, `fno_oscg1` |
| **F. unreferenced by exact name** | 194 | 9.99 | k-fold folds `kf_*_f0..4`, `kfre_*_f1..4` (38); round-18/19/20 arms `fno_r18_*`, `fno_r19_*`, `fno_r20_*`; `fno_sim{A,B,C,E}_*`; `fno_all_{do05,do15,sim,w96do30,w96do45,w96L8do45sim}_s*`; `fno_all_w96L8cmb3sim_s5..8` (R25 spares); `ffno_all_w40l4_s1..4` (T2R14 L4); `ffno_all_w40l6_s2,s3`; `arch_vit_{L12,d2L8,d2L12,d384_s2,d384_s3}`; `arch_afno_*`, `arch_dilres_cmb3sim`; `swin_all_w64_s2..4`, `ufno_all_w96do30_s2..4`; `fno_{mh,oz,osc,tb4,tg,wd,wm,self,rho,aug,m}*`; `re_*` |

Notes:

- **A + B (25 files, 1.1 GB)** is everything a `reproduce.sh` target reads.
- **C** is needed only to re-run historical verifiers.
- **E + F (263 files, 15.9 GB)** is archive material. Before deleting any of it:
  - move it to scratch or tape,
  - delete the `Track_2/submit/fno*.pth` symlinks first,
  - keep `fno_all_w96L8cmb3sim_s1..4` / `fno_r18_cmb3sim*` if `r25` stays in `reproduce.sh`.
- The QD artefacts (`ffno_all_w40l8_sim_qderr_*`, `qd_twin_*`) are **not** in `unet/`. So `reproduce.sh --recipe QD`
  would retrain them; that is fine, since QD is not final.
- Nothing was deleted.

---

## 5. Refactor proposal (conservative)

**Do before the Decision phase: only what B1-B4 and M1-M7 need.** That means committing the final code, adding the
`final` target, parameterising `dump_twins.py`, the gate and env fixes, and filling the docs. Then tag the commit
(`decision-final`) and hand that tag over.

**Do NOT do before the Decision phase:**

- **Any change to `train2.py`'s `main()`.** This covers deleting closed branches, reordering the loss, renaming flags
  and changing defaults. The organisers retrain from the released code. A changed RNG draw order, or a flag that
  disappears from `vars(args)`, breaks the key-by-key config check (REPRODUCE §8.3, §9) and forces a new 40-GPU-h
  equivalence run. It removes no risk the guarded branches carry today.
- **Changing argparse defaults to the recipe values.** It would silently change the meaning of every recorded
  `train2_argv` and older config.
- **Moving `FFNO2d` or `ViT2d` out of `models_alt.py`, or touching `_heads`, `_FFNOLayer` or `_GridIn`.** These are
  what the band fit and the SIGBLEND dump execute in the repo (`FFNO_IMPL="auto"`), and parameter creation order
  fixes the seeded initialisation.
- **De-duplicating `_fno.py`'s vendored classes against `models_alt`.** The archive must stay self-contained; the
  containers lack the repo.

**After the Decision phase, or on a separate branch with the checks below:**

1. **Split `models_alt.py` into a package**, keeping `models_alt.py` as a re-export shim:
   - `models/ffno.py`: `_FFNOLayer`, `FFNO2d`, `_LayerNorm2d`.
   - `models/vit.py`: `ViT2d`.
   - `models/common.py`: `_GridIn`, `_heads`, `SpectralConv2d`, and the physics helpers.
   - `models/fno.py`: `FNO2d`, for r25.
   - `models/archive.py`: DeepONet, AFNO, DilRes, UFNO, ConvNeXt, UNetMod, Swin, SimVP, STViT and
     `oracle_cond_features`.
   - `models/__init__.py`: `build`, `build_wave1`.

   The shim, `from models import *` plus the private names, is required: `_fno.py:2355` does
   `import_module("models_alt")`, and ~20 tracked instruments and verifiers import from it (§2.1). Checkpoints need no
   change, because they store no class pickles.
2. **Move files:**
   - `train.py`, `dinov2_init.py`, `distill_*`, `run_*.sh` → `Track_1/closed/unet/`, after removing `student()` from
     `reproduce.sh`.
   - Delete `runq.sh`, `queue*.txt`, `queue_round10.bak`, `done.txt` and the 49 `log_*.txt`.
   - Delete the untracked `Track_2/submit/fno*.pth` symlinks.
3. **`train2.py`:** add a banner comment marking the recipe-active flags; do not remove anything. If a trim is ever
   wanted, lift the closed branches into a `train2_arms.py` hook only after the checks below pass.

**Verification for each step.** All run on CPU except the smoke:

1. **Construction identity.** For each arch in {ffno w40 L8 16×32, vit 384 L8, fno w96 L8}:
   - under `torch.manual_seed(0)`, build with `models_alt.build` before and after;
   - assert identical `state_dict` keys, shapes and **bitwise** values (same RNG consumption), and equal parameter
     counts.
2. **Load and forward identity.** Load the 11 final checkpoints plus one of each archived family on disk (afno, ufno,
   swin, dil, unet, fno) through `_fno.load_any`. Run a fixed input `torch.randn(4, 40, 32, 64, generator=g)` and
   assert **bitwise** equal field and logstd, before and after, with TF32 off. For F-FNO also check the `dense` path.
3. **Run the existing verifiers:**
   - `verify/fuse_ffno.py` (vendored == `models_alt.FFNO2d`, bitwise), `verify/fuse_vit.py`, `verify/diff_arch.py`,
     `verify/diff_ufno.py`, `verify/fuse_swin.py`, `verify/phead_strip.py`, `verify/aug_mask_check.py`;
   - `validate_strict.py` and `validate_t2.py --adaptive` on a rebuilt package.
4. **Band identity.** Re-run `fit_band.py` on the shipped A/M at 0.80, and require the md5 of `base` and
   `sigma_ref` to equal `bands/r38_a080_c0_fill.npz`'s. Re-run `build_final.sh` with `RPDE_REF_T1/T2` and require
   "ALL FILES BYTE-IDENTICAL" on both tracks.
5. **Training smoke (GPU).** Run `train2.py $FFNO $SIMF --seed 1 --epochs 2`, and the ViT equivalent, before and
   after. Require the printed epoch-0 loss line to be identical; REPRODUCE §8.5 shows the first ~40 epochs are
   bit-identical on one card.

---

## 6. Doc drift in `REPRODUCE.md`

| where | says | reality |
|---|---|---|
| :10-11 | `FINAL_RECIPE_T1/_T2 = <TO BE DECIDED>` (A \| QD \| QD_STATIC) | Final = recipe-A members and twins plus the SIGBLEND band and the T1/T2 inference constants, built by the untracked `build_final.sh` (B2). |
| :19-22 | final rows are reproduced by `RPDE_FINAL_ALPHA=... bash reproduce.sh --recipe A`, reusing the 0.80 band | That path reuses the **static** r38 band. The final ships the **SIGBLEND** band (`k27`) plus EBAND/SIGPOW, which no `reproduce.sh` path creates (B3). |
| :25-27 | the three candidates' description ("everything below is history") | No mention of SIGBLEND, `JOINT_GAIN`, R1_TSM/BLK, D1C_EB/LEAD, VC_N0 or the latency knobs. |
| :70-77 | "ONE method of record … as of 2026-09-20: R30 r34" | Superseded twice (r35, then the 09-27 final). The same stale state is in CLAUDE.md *Current state*. |
| :93-97, §8.1 | "five knobs" | `reproduce.sh` now has more: `RPDE_REPRO_SMOKE`, `RPDE_REPRO_FORCE`, `RPDE_FINAL_ALPHA`, `RPDE_BAND_R34/R35/T1/T2`, `BUILD_ALPHA`, `RPDE_REPRO_LOG`, `STUDENT*`. |
| §2 :259-261 | `train2.py` passes `real_exclude=excl` "(line 803)" | Now `train2.py:908`. |
| §9.1 item 1 | PY default "line 40" | `reproduce.sh:46`. The fallback it asks for already exists (:46-48), but `PY` is not exported to the builds (M3). |
| §3 | "A reproduction must reuse the committed band (r35 …)" | The final band `k27_sigblend_lam0.5.npz` is not committed, and not in `bands/MANIFEST.md` (B1). |
| §4 | `build_round13.sh` "stages into `../sub_candidates` … `SUB` is not overridable" (also §8.2, §8.6 item 4) | `RPDE_SUB` exists now (`build_round13.sh:23`) and `build_final.sh` uses it. |
| §0 | "`environment.yml` pins the same versions for local parity" | True for the containers. The training and build stack actually verified (torch 2.7.1 / numpy 2.2.6, `mural_osc`) is not pinned anywhere in the repo (m6). |
| §1.2-1.5 | recipes | **Correct.** Byte-for-byte consistent with the checkpoints' `config` and with `reproduce.sh:115-177` (§1.1 here). |
| `sigblend/README.md` :13-17 | build knob list | Omits `R1`, `R1_ASYM`, `NUMPY_OUT`, `R1_TSM`, `R1_BLK(_N0)`, `D1C_*`, `VC_N0`, `DIRECT_PRED`, `SMALL_GRAPH_LAZY`. `build_final.sh:44-52` is the authority. The README points to `rp_k27_final.sh`, whose knob set differs again (no `R1_TSM/BLK`, `D1C_EB/LEAD`, `VC_N0`). |
| `sigblend/README.md` :9 | "edit U / out paths" | Should become arguments (B4). |
