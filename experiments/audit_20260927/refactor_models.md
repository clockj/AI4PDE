# Refactor: split the closed families out of `models_alt.py` (2026-09-28)

Worktree `/fs/scratch/PAS1187/qwu/tmp/realpde/wrap0927/wt_models` (detached at 7b6ee8e). Nothing committed.
Patch: `notes/patch_models.diff` (`git diff HEAD -- Track_1/unet` after `git add -N models_closed.py`;
two files only; checked to apply with `patch -p1` onto a `git archive HEAD` copy and to reproduce the
worktree files byte for byte: models_alt md5 `6f4677c3`, models_closed md5 `81979d27`).
Untouched: `train2.py`, `model.py`, `simreal.py`, `_fno.py`, and every verifier.

## Line counts

| file | before | after |
|---|---|---|
| `Track_1/unet/models_alt.py` | 1955 | 1041 |
| `Track_1/unet/models_closed.py` | – | 1003 |

## What moved (verbatim, into `models_closed.py`)

`DeepONetCh`, `AFNO2d`, `DilResNet`, `UFNO2d`, `_drop_path`, `_trunc_init`, `_var_mlp`,
`_ConvNeXtBlock`, `CNeXtUNet`, `_PreActResBlock`, `UNetMod`, `_win_part`, `_win_merge`,
`_WindowAttnV2`, `_SwinV2Block`, `_PatchMergeV2`, `_PatchExpand`, `_TokenProj`, `SwinUNet`,
`_ConvSC`, `_GASubBlock`, `_MetaBlock`, `SimVP2`, `_MHSA`, `_DividedSTBlock`, `STViT`,
`ORACLE_LOGRE_MU`, `ORACLE_LOGRE_SD`, `ORACLE_AOA_MU`, `ORACLE_AOA_SD`, `oracle_cond_features` (31 names).

`_drop_path`, `_trunc_init` and `_var_mlp` moved too: only the closed families use them (the
ViT's drop-path is its own method). `_ConvNeXtBlock` is also the Swin U-Net's `refine` block.

The move was done by a script (`modelsM/scripts/split.py`) that copies line slices of the HEAD
file. An AST check confirms that each of the 48 top-level definitions/assignments of the original
appears exactly once across the two new files, with identical source text.

## What stayed in `models_alt.py` (bodies byte-identical)

`SpectralConv2d`, `OSC_W_LO/HI`, `FNO2d`, the physics helpers (`divergence`, `vorticity`,
`velocity_gradient`, `time_derivative`, `radial_spectrum`, `radial_index`), `build`, `_heads`,
`_GridIn`, `ViT2d`, `F`, `WAVE1_ARCHS`, `PHASE2_ARCHS`, `_LayerNorm2d`, `_FFNOLayer`, `FFNO2d`,
`build_wave1`. The explanatory banners (round 15, wave 1) stay where they were.
New in this file: an added paragraph in the module docstring, plus the re-export block at the bottom
(`_CLOSED_NAMES`, `_reexport_closed`, `import models_closed`, and a guarded explicit
`from models_closed import (...)`).

## Import design (the circular dependency)

`models_closed` needs `_GridIn`, `_heads`, `_LayerNorm2d` and `SpectralConv2d` from `models_alt`
when its classes are defined. `models_alt.build`/`build_wave1` need the closed classes in their
own globals when they are called. A plain `from models_closed import X` at the bottom of
`models_alt` would break `import models_closed` from a clean interpreter, so the fix works like this:
- `models_alt` defines `_reexport_closed(mod)`, which binds every name in `_CLOSED_NAMES` into its
  globals, and then runs `import models_closed`.
- The last line of `models_closed` sets `_LOADED = True` and calls
  `models_alt._reexport_closed(sys.modules[__name__])`.
- If `_LOADED` is set, `models_alt` also re-binds the names itself, through an explicit
  `from models_closed import (...)` that linters and "go to definition" can follow, followed by
  `_reexport_closed`. This covers `verify/fuse_osc.py`, which runs `exec_module` on
  `models_alt.py` under a spec that is not registered in `sys.modules`.
- All three import orders were checked from a clean interpreter: `import models_alt` first,
  `import models_closed` first, and the unregistered spec exec. In each one, all 31 names resolve
  in `models_alt` to the same objects as in `models_closed`.
- An AST check confirms that `_CLOSED_NAMES`, the explicit import list, and the set of
  top-level names in `models_closed` are the same 31 names.
- pyflakes finds no undefined names in either file. It still reports F401 "imported but unused"
  for the re-exported private names. Those are intentional, and the import line carries
  `# noqa: F401` for flake8.

One visible difference: moved classes now report `__module__ == "models_closed"`, for example
`type(m).__module__` for a Swin or DilResNet member. `git grep "__module__"` finds no tracked code
that depends on this. Checkpoints hold only state_dicts, so `torch.load` is unaffected.

## Re-export list

Every moved name is re-exported. The repo imports these through `models_alt` (`git grep`):
`SwinUNet`, `_SwinV2Block`, `_WindowAttnV2`, `_win_part` (`closed/screen_profile_swin.py`),
`_ConvNeXtBlock`, `UFNO2d`, `oracle_cond_features` (`train2.py:871`, `closed/screen_oracle.py`),
`AFNO2d`/`DilResNet`/… via `build`, and `WAVE1_ARCHS`, which did not move.

## Verification

Python is `/users/PAS1187/qwu/.conda/envs/realpde/bin/python` (torch 2.2.2). The original module
comes from `git show HEAD:...` copies (`/tmp/w27m/orig_unet`, `/tmp/w27m/origT1`, and the
full-tree copy `/tmp/w27m/head`; on scratch, `modelsM/orig_unet`). Scripts are in `modelsM/scripts/`
and outputs in `modelsM/results/`. Every check below was re-run on the FINAL version of the files
(models_alt `6f4677c3`).

1. **Construction identity (CPU)**, via `ident_dump.py <dir> out.pt` for each module, then
   `ident_cmp.py`. **25/25 PASS.**
   - Each model is built under `torch.manual_seed(0)`. The check requires identical type,
     parameter count, state_dict keys, shapes and dtypes, bitwise-equal tensors, and a bitwise-equal
     global RNG state after the build.
   - The configurations cover every arch string `build` accepts, plus `build_wave1` called directly:
     - F-FNO: `ffno` w40 L8 16×32 (also with `err_head`), `ffno_nff`.
     - ViT: `vit` 384 L8 (with and without var-head).
     - FNO: `fno` w96 L8 8×12 var-mlp lift-mlp do0.45, and `fno` with mean/osc/tbasis/p/sub heads.
     - Closed round-15 families: `unet` w96; `afno` w96 (arch_afno_w96 config; also without
       var-head); `dilres` w96 (arch_dil_w96); `ufno` w96 (ufno_all_w96do30_s1); `deeponet`.
     - Closed wave-1 families: `cnext_unet`; `unet_mod`; `swin_unet` w64 L2 (swin_all_w64_s1),
       also with drop_path 0.2; `swin_unet_b6`, `swin_unet_win16`, `swin_unet_p1`; `simvp`; `stvit`.
     - `build_wave1` called directly for ffno, swin_unet and cnext_unet.
   - Control: the original compared with itself also gives 25/25.
2. **Forward identity (CPU)**, run by the same script. The input is
   `torch.randn(2,40,32,64, generator=seed 1)`. Every returned tensor matched bitwise in each of
   these modes:
   - `eval()` mode;
   - the F-FNO `dense` spectral path;
   - `train()` mode under `manual_seed(123)`, so dropout and drop-path draws are included.
3. **Checkpoint loading (CPU)**, via `load_dump.py <Track_1 dir>` and `load_cmp.py`.
   **19/19 PASS, bitwise.**
   - Checkpoints: the 11 final ones, plus arch_afno_w96, arch_dil_w96, arch_dilres_cmb3sim,
     arch_ufno_w96, ufno_all_w96do30_s1, swin_all_w64_s1, arch_unet_w96 and fno_all_w96L8cmb3sim_s1.
   - Each one goes through `_fno.load_any` (with the F-FNO dense path as well) and also through
     `models_alt.build` + strict `load_state_dict`, and one forward is run on `randn(4,40,32,64)`.
   - Through `load_any`, the F-FNO, DilResNet and Swin members resolve to the models_alt /
     models_closed definitions (`FFNO_IMPL`/`SWIN_IMPL="auto"`). ViT, AFNO and U-FNO go to the
     vendored `_fno` classes, so the direct `build` path is what exercises those three.
4. **Existing verifiers.**
   - **CPU** (`run_verifiers.sh`): `fuse_vit.py`, `diff_arch.py`, `diff_ufno.py`, `phead_strip.py`,
     `aug_mask_check.py`, `fuse_osc.py`, `fuse_ffno.py --quick` and `fuse_swin.py --quick` all
     **PASS** on the refactor (`results/verify_new2.txt`; logs in `logs/w27m_verify_new2/`).
   - **HEAD control:** all 8 also pass on a pristine HEAD tree (`results/verify_head.txt`). Their
     logs are identical to the refactor's except the timing tables of the two `--quick` runs.
   - **Harness gap:** the first HEAD attempt failed only because `Track_1/val_unseen.npz` is an
     untracked file missing from a `git archive` copy. It was symlinked in and the run repeated.
     This is not a defect in the code.
   - **GPU, full mode:** `fuse_ffno.py`, `fuse_swin.py`, `fuse_vit.py` and `diff_ufno.py` all
     **PASS**. Jobs 7569371 (first version) and 7569387 (final version); logs in
     `logs/w27m_verify_gpu{,2}/`.
   - **Not run:** `validate_strict.py`, `validate_t2.py`, and the band-identity/`build_final.sh`
     rebuild from audit §5. Those build packages from the shipped tree, and the package code
     (`_fno.py`) does not import `models_alt` inside an archive.
5. **Training smoke (GPU, A100)** with `train2.py`, 2 epochs, seed 1, using the given F-FNO(+sim)
   and ViT recipes, `RPDE_SIMCACHE=/fs/scratch/.../work`. Checkpoints and logs are in
   `modelsM/ckpt/`. Loss lines were compared with the `(Ns)` timing suffix stripped.
   - **ViT** (job 7569269, then 7569387): orig_a, orig_b and the refactor have identical loss
     lines and **bitwise-equal final state_dicts**.
   - **F-FNO, default settings** (job 7569268, then 7569387): the GPU run is **not
     deterministic**. Two runs of the ORIGINAL code differ by up to 2.0e-5 in the final weights,
     and the refactor differs from each original run by the same amount (1.7e-5 to 2.5e-5). All
     loss lines are identical at the printed precision in the final run. One earlier run showed
     `field 0.87943` against `0.87942`, which is within the orig-vs-orig spread.
   - **F-FNO, deterministic control** (job 7569336, then 7569387): the same recipe run through
     `modelsM/detrun.py`. This wrapper sets `use_deterministic_algorithms(True, warn_only)`,
     `cudnn.deterministic` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, then `runpy` runs the untouched
     `train2.py`. Result: orig_a, orig_b and the refactor have identical loss lines and
     **bitwise-equal state_dicts**. The refactor run was on a different node, a0157 against a0196.
   - **CPU supplement** (`step_dump.py`): seeded build plus 5 AdamW steps in train mode for ffno,
     vit and fno. Losses and state_dicts are bitwise equal.
6. **Parse check:** `ast.parse` passes on both files.

GPU time used: 5 jobs, about 35 GPU-minutes (7569268, 7569269, 7569336, 7569371, 7569387).

## Left undone / notes
- The audit (§5) proposed a `models/` package with a shim. This task specified the smaller
  two-file split instead, and that is what was done.
- Package-level gates (`validate_strict`, `validate_t2`, the rebuild with "ALL FILES
  BYTE-IDENTICAL") were not run, as explained under Verification item 4.
- The worktree index has `models_closed.py` intent-to-add (`git add -N`), as the deliverable
  command requires. The 334 `.pth` symlinks created for the verifiers were removed. The
  pre-existing untracked `Track_1/unet_ckpt` symlink was left in place.
