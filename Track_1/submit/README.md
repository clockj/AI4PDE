# Track 1 — Submission Package

**`submission.zip` is built and verified. Upload it to
<https://www.codabench.org/competitions/17363/>.**

Model: the small MSE U-Net (`Track_1/unet/unet_mse.pth`), plus SPS interval bounds.

## Verified result

Run end-to-end from a **clean extract of the zip**, through the **official** `scoring.py`, with
`scipy` / `h5py` / `matplotlib` / `pandas` blocked at import (the package set the Track 1
evaluation image lacks):

| subscore | value |
|---|---|
| rel_l2_score | 96.07 |
| tke_score | 75.05 |
| mvpe_score | 96.35 |
| time_score | 94.48 |
| sps_score | 61.66 |
| **final_score** | **84.72** |

on `val_unseen.npz` — 336 windows from 8 trajectories the model never trained on.

## Archive contents (3 files, all at the root)

```
submission.py     entry point: predict(input_array, metadata=None)
model.py          UNet class + normalization constants (vendored from ../unet/)
unet_mse.pth      checkpoint, 7.75 MB
```

Extracted size **7.75 MB** against the 256 MB cap — no fp16 packing needed. Imports only
`torch` and `numpy`, both present in `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`.

## Contract compliance

| requirement | status |
|---|---|
| `predict()` returns `(N, 20, 32, 64, 3)` float32 | verified |
| finite everywhere (else all subscores zero) | `nan_to_num` + asserted |
| `p` channel zero-filled | verified |
| bounds: keys exactly `{prediction, lower, upper}`, `lower <= prediction <= upper` | verified |
| `RETURN_BOUNDS = False` returns a **bare ndarray**, not a dict | verified |
| extracted archive < 256 MB | 7.75 MB |
| no scipy / h5py / matplotlib / pandas | verified under import blocker |

## Rebuild and re-verify

```bash
conda activate realpde
cd Track_1/submit
python build_zip.py                                   # -> submission.zip + _package/
python validate_submission.py --dir _package --blocked
python fit_band.py                                    # re-fit the SPS half-width
```

`validate_submission.py` runs `predict()` on `val_unseen.npz`, writes `predictions.npz` +
`ref/targets.npz` exactly as the platform would, and invokes the kit's own `scoring.py`.

## Things to know before reading the leaderboard number

**Expect the leaderboard to be lower than 84.72, and treat a large gap as information.**

- **Neither validation set tests Reynolds extrapolation.** Both sit inside the training range
  (Re 3750–26700); the hidden test set includes **Re 2968**, below everything the model has seen.
  This is the single most likely source of a shortfall. The model uses the official global
  `GaussianNormalizer`, *not* the per-window self-normalization that would help here — that is
  WP1/WP3 of `../plan/plan_20260729.md`, deliberately not in this submission.
- **`time_score` is uncertain.** We score 94.48 from 2.49 ms/window measured as
  *(whole `predict()` call) / N*, including numpy conversion and bound construction. The forward
  pass alone is 0.87 ms. The platform's ingestion program is not in the standalone kit, so we
  cannot see exactly what it times; our measure is the conservative one. The model is warmed up
  at import so checkpoint loading and CUDA setup are not charged to `predict()`.
- **`sps_score` 61.66 comes from a fitted band**, half-width `0.006` physical units, fitted on
  `val_seen` and verified on `val_unseen` (55.00 → 61.66, +1.33 final). The half-width is
  absolute, so at Re 2968 — where velocities are ~3× smaller — it should *gain* coverage, but
  this is untested. If the leaderboard SPS looks wrong, set `RETURN_BOUNDS = False` in
  `submission.py` for a one-line revert to the scorer's default band.

## Deliberate defensive choice

`predict()` falls back to persistence (repeat the last frame, ~79 final) if the input resolution
is not 32×64, rather than raising. The kit documents 32×64 in three places so this should be
unreachable, but a raised exception would zero the submission and cost a day.
