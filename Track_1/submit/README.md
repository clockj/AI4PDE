# Track 1 — Submission Package

**`submission.zip` is built and verified. Upload it to
<https://www.codabench.org/competitions/17363/>.**

Model: the small MSE U-Net (`Track_1/unet/unet_mse.pth`), plus SPS interval bounds.

## Which checkpoint ships

`unet_mse_alldata.pth` — the U-Net trained on **all 82 real trajectories** for **150 epochs**
(13,368 windows, final train MSE 0.02420, converged).

> **Its score cannot be measured honestly here.** Both local validation sets are training data
> for it. The 86.21 below is an **in-sample smoke test**, not a held-out result — it confirms the
> submission runs and complies with the contract, nothing more. Do not quote it as performance.

The honest performance estimate comes from the matched held-out model
(`unet_mse_e150.pth`, same recipe, 74 trajectories): **84.79** on `val_unseen.npz`.
**Expect the leaderboard nearer 84.8 than 86.2.**

| checkpoint | trained on | epochs | role | val_unseen |
|---|---|---|---|---|
| `unet_mse_alldata.pth` | 82 traj | 150 | **ships** | in-sample, not meaningful |
| `unet_mse_e150.pth` | 74 traj | 150 | measurement | **84.79** |
| `unet_mse.pth` | 74 traj | 40 | earlier baseline | 84.72 |

## Verified result

Run end-to-end from a **clean extract of the zip**, through the **official** `scoring.py`, with
`scipy` / `h5py` / `matplotlib` / `pandas` blocked at import (the package set the Track 1
evaluation image lacks):

| subscore | shipped model (in-sample) | held-out equivalent |
|---|---|---|
| rel_l2_score | 96.99 | 96.03 |
| tke_score | 79.09 | 75.95 |
| mvpe_score | 97.41 | 96.30 |
| time_score | 94.56 | 93.88 |
| sps_score | 62.99 | 61.79 |
| **final_score** | **86.21** | **84.79** |

The ~1.4-point spread between the columns is the memorization the shipped model gains from
having seen these trajectories — it is not skill that will appear on the leaderboard.

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
