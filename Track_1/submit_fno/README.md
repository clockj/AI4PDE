# `submit_fno/` — the package code, the band, the build

Everything that ships on Track 1 lives here, and `_fno.py` ships on Track 2 as well (through the
symlink `Track_2/submit/_fno.py`). Archives are built by `build_round13.sh` and staged in
`../../../sub_candidates/`. The `submission*.zip`, `_package*/` and `band*.np*` files in this
directory are build by-products and older rounds' artifacts, not the package of record. Scores,
the current package and the standing rules are in `../../CLAUDE.md`.

## What is in an archive

As of R24 / R25 (xfam6):

| file | what |
|---|---|
| `submission.py` | `predict(input_array, metadata=None)` |
| `_fno.py` | self-contained model code: `FNO2d`, `ViT2d`, `UNetTwoHead`, `UFNO2d`, `AFNO2d`, `StackedFNO`, `StackedViT`, `fuse()`, `load_any()`, fp16/int8 unpacking |
| `fno.pth` … `fno_4.pth` | 4 all-data w96 L8 `cmb3sim` FNOs, int8, 29.0 MB each |
| `fno_5.pth`, `fno_6.pth` | 2 ViTs (dim 384, 8 layers), int8, 14.8 MB each |
| `band.npz` | `base` / `sigma_ref` / `p` — here `../bands/r24_xfam6.npz`, byte for byte |

146 MB extracted, against the 256 MB cap. Only `torch` and `numpy` are imported. Member files
are named by position; `load_any` dispatches on each blob's own `arch` field.

## What `predict()` does

1. Normalize with the global GaussianNormalizer (`_fno.MEAN` / `_fno.STD`).
2. Run each fused group once (`FUSE = True`; xfam6 is two invocations) and average the members'
   raw field and σ on the GPU.
3. `compose`: pull the fluctuation's per-pixel temporal std toward σ at `ALPHA`.
4. Denormalize and write `u, v` into a pre-zeroed output; `p` stays zero.
5. Band: `half = base · clip(σ/σ_ref, 0.3, 3.0)^p`; return `{"prediction", "lower", "upper"}`.

`_pick_batch` chooses the batch size at import, which is untimed, on whatever GPU the evaluator
provides.

## Baked constants

The evaluator's process never sees our shell, so anything that ships is a constant in
`submission.py`, written there by `build_round13.sh` with `sed`.

| constant | package of record | what |
|---|---|---|
| `ALPHA` | `0.60` | fluctuation rescaling; changing it requires refitting the band |
| `FUSE` | `True` | `False` only for the invocation-cost measurement (R23) |
| `AMP` | `None` (R24) · `"fp16"` (R25, awaiting) · `None` in the tree | autocast behind a load-time check against fp32; `AMP_ACTIVE` and `AMP_PROBE_DIFF` record what ran |
| `SPEC_PATH` | `"bmm"` | real 2×2-block spectral matmul |
| `HOST_WRITE` | `"torch"` | threaded output write, bit-identical to the numpy one |
| `BLEND_BETA` | `0.0` | R14's gated mean-blend, off since the hidden set proved to be in range |

`RPDE_AMP`, `RPDE_TF32` and the other environment switches steer measurement drivers only. TF32
is disabled in the shipped path.

## The band

```bash
python Track_1/submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha 0.60 --band-p 0.45 --out band.npz
```

* The **error field** comes from `--err-ckpt`: the held-out twins, whose errors on the fitting
  windows are real predictions. `sigma_ref` comes from `--ckpt`: the all-data members that emit
  σ at test time. Skipping `--err-ckpt` fits memorized errors. The July 30 package did, got a
  median half-width of 0.0046 against board errors of 0.118, and scored sps 36.94.
* **Refit whenever the member set or `ALPHA` changes.**
* `base` is the exact per-element optimum (`../sps9.py:optimal_band`), box-smoothed
  (`--band-smooth 3`) and fitted on the AoA 15–20 subset (`--fit-aoa`) at a 1.39× error stretch
  (`--inflate`). p and k were chosen by leave-one-trajectory-out CV (`../band_cv_v9.py`). The
  band is closed; see `../../ROADMAP.md` Part 2.
* Once a package scores, its band is versioned in `../bands/` (see `MANIFEST.md`), and
  `RPDE_BAND=` reuses it byte for byte.

## Build

```bash
# from the repo root
RPDE_A="<all-data members>" RPDE_M="<held-out twins>" RPDE_BAND=Track_1/bands/<band>.npz \
RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch \
RPDE_TAG=r26_<tag>_a060 bash Track_1/submit_fno/build_round13.sh 0.60
```

`build_round13.sh [ALPHA]` runs five stages:
1. refit the band into `Track_2/submit/band8.npz`, or copy `RPDE_BAND` there;
2. `sed` `ALPHA`, `FUSE`, `AMP` and `SPEC_PATH` into both tracks' `submission.py`, and
   `HOST_WRITE` and `BLEND_BETA` into Track 1's;
3. `validate_strict.py` builds `submission8.zip` through `build_zip.py` and gates it;
4. `Track_2/build_t2.py`, then `Track_2/validate_t2.py`;
5. copy both archives to `../sub_candidates/track{1,2}_$RPDE_TAG.zip`.

* **Pass every knob.** Unset ones fall back to older behaviour: `RPDE_A`/`RPDE_M` to the
  round-13 width-24 members, `ALPHA` 0.90 with no argument, empty `SPEC_PATH`/`HOST_WRITE`, fp32
  packing without `RPDE_INT8=1`, `AMP = None` without `RPDE_SHIP_AMP`.
* The script leaves the baked values in the working tree; a job that sets `RPDE_SHIP_AMP`
  restores `AMP = None` itself. Never edit `_fno.py`, `submission.py` or the build scripts while
  a build runs.
* It refuses while other `rp_*` Slurm jobs run, because `validate_strict.py` times the package.
  `RPDE_FORCE=1` builds anyway when the timing is not being recorded.
* `build_round1{0,1,2}.sh` and `finalize_round11.sh` are earlier rounds' builds, kept for
  provenance.

## Gates

* **`validate_strict.py`** — the authority, and it needs a GPU. It extracts to a clean
  directory, chdirs elsewhere, keeps the extract dir off `sys.path`, blocks `scipy` / `h5py` /
  `einops` / `matplotlib` / `pandas`, loads `submission.py` by file location, checks every
  contract the ingestion program asserts, asserts permutation invariance, scores through the kit
  scorer (our reconstruction under `/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/`,
  identical to the official one), and times `predict()` end to end.
* **`../verify/cpu_smoke_submission.py`** — the login-node stand-in: the same package blocking
  and load path, no timing. `RPDE_ZIP=<archive>` selects the archive.
* **`../verify/pkg_identity.py`** — whether two built archives return the same arrays: the gate
  for a pure-latency A/B, and for "is this the archive on the board?". Give it the archive
  actually uploaded, from `../../../sub_candidates/submitted/`.
* **`../verify/amp_path.py`** — `AMP_ACTIVE` is what was baked, predictions match an fp32 twin of
  the same archive, and the two are timed interleaved in one process.
* **`../verify/fuse_vit.py`, `fuse_osc.py`, `quant_identity.py`, `blend_path.py`** — fusion,
  packing and blend gates.
* **`../verify/diff_scoring_{official,real}.py`** — the local scorer against the official kit
  v9 (worst disagreement 2.4e-6).

## What this is not

Track 1 tiles windows at a stride equal to the output horizon and hands `predict()` all of them
at once, so window *k*'s target is window *k+1*'s input. A submission can therefore read the
answers out of its own argument; doing so scores ~93.5 without modelling anything. This package
deliberately does not. See `../submit_trick/DISCLOSURE.md` for the measurement and the report
sent to the organizers, who have confirmed such scores are not valid.
