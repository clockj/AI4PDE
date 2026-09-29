# `submit_fno/` — the package code, the band, the build

Everything that ships on Track 1 lives here, and `_fno.py` ships on Track 2 as well (through the symlink
`Track_2/submit/_fno.py`). `build_final.sh` builds the two packages of record. Files named `submission*.zip`,
`_package*/` or `band8.npz` anywhere in the tree are by-products of the last build, never the package of record.
The specification is `../../REPRODUCE.md` §13; the procedure is the `build-package` skill.

## What is in an archive

| file | what |
|---|---|
| `submission.py` | `predict(input_array, metadata=None)`, with every shipped constant baked in |
| `_fno.py` | self-contained model code: `FFNO2d`, `ViT2d` (and the older families), `StackedFFNO`, `StackedViT`, `fuse()`, `load_any()`, `predict_fno()`, int8 / fp16 unpacking |
| `fno.pth` … `fno_4.pth` | the four all-data F-FNO members (`ffno_all_w40l8_sim_s1..s4`), int8, 1.5 MB each |
| `fno_5.pth`, `fno_6.pth` | the two ViTs (`arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3`), int8, 14.8 MB each |
| `band.npz` | `base`, `sigma_ref`, `p`, `centre_alpha` — `../bands/k27_sigblend_lam0.5.npz`, byte for byte |

36.3 MB extracted and 30.3 MB zipped, against the 256 MB cap. Only `torch` and `numpy` are imported. Member files are
named by position; `load_any` dispatches on each blob's own `arch` field. The Track-2 archive holds the same files with
its own `submission.py`.

## What `predict()` does

1. Normalize with the global normalizer (`_fno.MEAN` / `_fno.STD`), in chunks of at most `BATCH_CAP` windows.
2. Run the two fused groups under fp16 autocast and average the six members' field and σ on the GPU: y₀ and σ.
3. Compose at `ALPHA` with one gain per pixel for `u` and `v` (`JOINT_GAIN`): the prediction P.
4. Half-width on the GPU: `base′ · clip(σ/σ_ref, 0.3, 3)^p′ · (σ + 1e-8)^(λ q_c)`, the last factor through `EBAND`.
5. Interval centred on y₀ and widened to contain P: `lower = min(y₀ − h, P)`, `upper = max(y₀ + h, P)`.
6. Write `u, v` of the prediction and the bounds straight into page-locked output arrays; `p` stays zero.

Each window is forecast from its own input only; nothing is pooled across the windows of a call. At import, which is
untimed, the package checks fp16 against fp32, times batch sizes, allocates the output pool and runs every batch size
once.

## Baked constants

The evaluator's process never sees our shell, so anything that ships is a constant in `submission.py` (two of them in
`_fno.py`), written there by `build_round13.sh` with `sed`. **The values in the working tree are defaults, not the
package**: `build_final.sh` lists what ships.

| constant | shipped | tree default | what |
|---|---|---|---|
| `ALPHA` | 0.80 | 0.60 | fluctuation rescaling; changing it requires refitting the band |
| `JOINT_GAIN` | True | False | one compose gain for `u` and `v`; needs `centre_alpha` 0 in the band |
| `EBAND` | `((λq_u,0,0,0,0,0),(λq_v,0,0,0,0,0)), "mult", None` | None | the σ-power factor of the SIGBLEND band |
| `AMP` | `"fp16"` | None | autocast behind a load-time check against fp32; `AMP_ACTIVE` records what ran |
| `FFNO_PATH` | `"dense"` | as built | the F-FNO's spectral path |
| `FUSE` | True | True | False only for the invocation-cost measurement (R23) |
| `GPU_BAND`, `D2_DIRECT_PRED` | True, True | True, False | band and output write on the device side |
| `WARM_TAILS`, `PIN_OUT`, `BATCH_CAP` | True, 128, 56 | False, 0, 0 | host-side latency; outputs unchanged |
| `SPEC_PATH`, `HOST_WRITE` | `"bmm"`, `"torch"` | `"bmm"`, `"torch"` | inert for these members: they act on `FNO2d` and on the host write path |
| `BLEND_BETA` | 0.0 | 0.0 | the gated mean-blend: off on the development board, which is in range; open for the private set |

`RPDE_AMP`, `RPDE_TF32` and the other environment switches steer measurement drivers only. TF32 is off in the shipped
path.

## The band

```bash
python Track_1/submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha 0.80 --band-p 0.45 \
    --centre-alpha 0 --fill-unscored --out band_static.npz          # then ../sigblend/ (its README)
```

* The **error field** comes from `--err-ckpt`: the held-out twins. `sigma_ref` comes from `--ckpt`: the all-data members
  that emit σ at test time. Skipping `--err-ckpt` fits memorized errors: the July package did, and scored sps 36.94.
* **Refit whenever the member set or `ALPHA` changes.**
* `base` is the exact per-element optimum (`../sps9.py:optimal_band`) of the twins' errors on the AoA 15–20 trajectories
  of the fitting split (`--fit-aoa`), stretched by 1.39 (`--inflate`), box-smoothed (`--band-smooth 3`), with elements the
  fitting split never scored filled from their neighbours (`--fill-unscored`). The fitting split defaults to
  `--fit-on seen`, `val_seen`, whose three AoA 15–20 trajectories the twins trained on: the field is in sample
  (measured 0.002 – 0.012 composite against an out-of-sample fit; `../../ROADMAP.md` §1.47).
* `fit_band.py`'s own defaults (`--alpha 0.75 --band-p 0.5`) are older than the operating point; the build passes what it
  needs.
* Once a package scores, its band is versioned in `../bands/` (see `MANIFEST.md`), and `RPDE_BAND=` reuses it byte for
  byte. A SIGBLEND band travels with its json (`../sigblend/`).

## Build

```bash
bash Track_1/submit_fno/build_final.sh OUTDIR          # from the repo root, on a GPU node
```

`build_final.sh` calls `build_round13.sh 0.80` with every constant spelled out, puts the three files the build edits
back as they were, gates Track 2 as an adaptive package, and runs the causality check. With `RPDE_REF_T1` /
`RPDE_REF_T2` it compares the result with the uploaded archives: the band and the members byte for byte, the outputs bit
for bit. It exits 0 only if all of that passes. The code files differ from the uploads since 2026-09-28 (guards against
non-finite values, dead knobs removed); commit `fdac00b` rebuilds them byte for byte.

`build_round13.sh [ALPHA]` underneath runs five stages: the band (refit, or a copy of `RPDE_BAND`); the `sed` of every
`RPDE_*` / `RPDE_SHIP_*` constant; `validate_strict.py`, which builds the Track-1 archive through `build_zip.py` and
gates it; `Track_2/build_t2.py` and `Track_2/validate_t2.py`; the copy to `RPDE_SUB`.

* **Pass every knob.** Unset ones fall back to older behaviour: `RPDE_A` / `RPDE_M` to the round-13 width-24 members,
  `ALPHA` 0.90 with no argument, fp32 packing without `RPDE_INT8=1`, `AMP = None` without `RPDE_SHIP_AMP`.
* Called directly, `build_round13.sh` leaves the baked values in the working tree. Commit before building, and never
  edit `_fno.py`, `submission.py` or the build scripts while a build runs.
* It refuses while other `rp_*` Slurm jobs run, because `validate_strict.py` times the package. `RPDE_FORCE=1` builds
  anyway when the timing is not being recorded.

## Gates

* **`validate_strict.py`** — the authority, and it needs a GPU. It extracts to a clean directory, keeps the extract dir
  off `sys.path`, blocks `scipy` / `h5py` / `einops` / `matplotlib` / `pandas`, loads `submission.py` by file location,
  checks every contract the ingestion program asserts, asserts permutation invariance, scores through the kit scorer,
  and times `predict()` end to end.
* **`../verify/final_identity.py`** — a rebuilt package returns the same arrays as the uploaded one, bit for bit, at
  the batch sizes the evaluator uses.
* **`../verify/cpu_smoke_submission.py`** — the login-node stand-in: the same package blocking and load path, no timing.
* **`../verify/pkg_identity.py`** — whether two built archives return the same arrays.
* **`../verify/amp_path.py`**, **`fuse_ffno.py`**, **`fuse_vit.py`**, **`quant_identity.py`**, **`blend_path.py`**,
  **`sigblend.py`** (with `k0a_dump.py`) — precision, fusion, packing, blend and band gates.
* **`../verify/diff_scoring_{official,real}.py`** — the local scorer against the official kit v9 (2.4e-6).

## What this is not

Track 1 tiles windows at a stride equal to the output horizon, so window *k*'s target is window *k*+1's input. A
submission can therefore read the answers out of its own argument; doing so scores about 93.5 without modelling
anything. This package deliberately does not. See `../archive/submit_trick/DISCLOSURE.md` for the measurement and the
report sent to the organizers, who have confirmed such scores are not valid.
