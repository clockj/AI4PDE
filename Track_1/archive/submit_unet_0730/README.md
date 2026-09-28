# Track 1 — Submission Package

> **Historical — the 2026-07-30 U-Net package.** It shipped `wp2i_alldata.pth` and scored
> 83.019 on the pre-v9 board; under kit v9 the same package is board round R6, 79.733. The
> numbers below are pre-v9 and measured on the old RTX 4060. The current package code is in
> `../submit_fno/`, and the current state in `../../CLAUDE.md`.

**`submission.zip` is built and verified. Upload it to
<https://www.codabench.org/competitions/17363/>.**

Model: the two-head U-Net from WP1 + WP2 (`UNetTwoHead`), fluctuation rescaling at
`alpha = 0.75`, plus a per-element SPS interval band.
Full derivation: `../report/report_20260730.md`.

## Which checkpoint ships

`wp2i_alldata.pth` — the two-head U-Net trained on **all 82 real trajectories** for **150
epochs** with the WP1/WP2 recipe (`--var-head --alpha-train 0`).

> **Its score cannot be measured honestly here.** Both local validation sets are training data
> for it. Any number produced against them is an **in-sample smoke test** confirming the
> submission runs and complies with the contract — nothing more. Do not quote it as performance.

The honest estimate comes from the matched held-out model (`wp2i_s5.pth`, identical recipe,
74 trajectories): **85.33** end-to-end on `val_unseen.npz`.

| checkpoint | trained on | role | val_unseen (end-to-end) |
|---|---|---|---|
| `wp2i_alldata.pth` | 82 traj | **ships** | in-sample, not meaningful |
| `wp2i_s5.pth` | 74 traj | measurement | **85.33** |
| `unet_mse_e150.pth` | 74 traj | previous submission | 84.79 |

**Previous leaderboard result: 83.108**, against 84.79 locally — the hidden set is ~1.7 points
harder than `val_unseen`. Expect this one nearer **83.6** than 85.3.

## Verified result

Run end-to-end from a **clean extract of the zip**, through the **official** `scoring.py`, with
`scipy` / `h5py` / `matplotlib` / `pandas` blocked at import (the package set the Track 1
evaluation image lacks):

| subscore | this submission (held-out `wp2i_s5`) | previous submission |
|---|---|---|
| rel_l2_score | 95.75 | 96.03 |
| tke_score | **78.55** | 75.95 |
| mvpe_score | 96.20 | 96.30 |
| time_score | 94.41 | 93.88 |
| sps_score | 61.75 | 61.79 |
| **final_score** | **85.33** | **84.79** |

**+0.54.** The gain is TKE (+2.6 subscore) from the variance head, partly refunded on rel-L2
(−0.28) — the composition adds uncorrelated fluctuation, which is the trade `alpha` arbitrates.

`time_score` here uses the conservative end-to-end measure (2.55 ms/window = whole `predict()`
call ÷ N, including the one-time model load). The forward pass alone is ~0.4 ms; scored that way
the same model reaches 86.12.

## Archive contents (4 files, all at the root)

```
submission.py     entry point: predict(input_array, metadata=None)
model.py          UNetTwoHead + compose + normalization constants (vendored from ../unet/)
unet_mse.pth      checkpoint, 7.65 MB
band.npy          per-element SPS half-widths, (1,20,32,64,3), 0.49 MB
```

Extracted size **8.16 MB** against the 256 MB cap. Imports only `torch` and `numpy`, both
present in `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`.

## What changed from the previous submission

1. **`UNetTwoHead` instead of `UNet`.** No refinement `z` slot (40 dead input channels dropped,
   1.93M → 1.91M params and faster), plus a per-pixel log-std head branching off the same final
   decoder feature map as the field head.
2. **Fluctuation rescaling at `alpha = 0.75`** (`compose` in `model.py`): the output's per-pixel
   temporal std is pulled toward the variance head's prediction. TKE only sees per-pixel
   variance, so this is correct even though the fluctuation is phase-wrong.
3. **Per-element SPS band** from `band.npy` instead of a flat 0.006 half-width.
4. Trained with a **per-sample relative-L2 loss in physical units** rather than MSE.

## Contract compliance

| requirement | status |
|---|---|
| `predict()` returns `(N, 20, 32, 64, 3)` float32 | verified |
| finite everywhere (else all subscores zero) | `nan_to_num` + asserted |
| `p` channel zero-filled | verified |
| bounds: keys exactly `{prediction, lower, upper}`, `lower <= prediction <= upper` | verified |
| `RETURN_BOUNDS = False` returns a **bare ndarray**, not a dict | verified |
| extracted archive < 256 MB | 8.16 MB |
| no scipy / h5py / matplotlib / pandas | verified under import blocker |

## Rebuild and re-verify

```bash
conda activate realpde
cd Track_1/submit
python fit_band2.py --ckpt ../unet/wp2i_alldata.pth   # -> band.npy
python build_zip.py --ckpt ../unet/wp2i_alldata.pth   # -> submission.zip + _package/
python validate_submission.py --dir _package --blocked
```

`validate_submission.py` runs `predict()` on `val_unseen.npz`, writes `predictions.npz` +
`ref/targets.npz` exactly as the platform would, and invokes the kit's own `scoring.py`.

## Things to know before reading the leaderboard number

- **Neither validation set tests Reynolds extrapolation.** Both sit inside the training range
  (Re 3750–26700); the hidden test set includes **Re 2968**, below everything the model has seen.
  The model uses the official global `GaussianNormalizer`, *not* per-window self-normalization.
  Amplitude augmentation was built and tested for exactly this (`--aug-scale`) and scored flat
  locally — but locally is the wrong place to test it, so it remains an open question.
- **`sps_score` uses a band fitted on `val_seen`**, which is training data for the shipped model.
  The seen→unseen transfer was checked on the held-out model (63.06 → 61.75), so the procedure is
  validated, but expect ~1.3 points of optimism. If the leaderboard SPS looks wrong, set
  `RETURN_BOUNDS = False` in `submission.py` for a one-line revert to the scorer's default band.
- **`alpha = 0.75` was tuned on 336 unseen windows.** It is a shallow optimum (0.50 → 84.55,
  0.75 → 84.71, 1.00 → 84.57 before the band), so it should transfer, but it is one number
  fitted on one small set.

## Deliberate defensive choices

- `predict()` falls back to persistence (repeat the last frame, ~79 final) if the input
  resolution is not 32×64, rather than raising. A raised exception zeros the submission.
- `band.npy` is validated at import (shape, finiteness); anything unexpected drops the bounds and
  returns a bare prediction. Malformed bounds zero **every** subscore, not just SPS.
