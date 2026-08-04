# Track 1 submission — 0.90M FNO with velocity-gradient matching

An honest forecaster. Reads only its own input window; would score identically if the evaluator
called it one window at a time. `validate_strict.py` asserts that as a test.

## Build

```bash
M="unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth"
python submit_fno/fit_band.py       --ckpt $M   # writes band.npy
python submit_fno/validate_strict.py --ckpt $M   # builds the zip, must print SUBMITTABLE
```

Run from `Track_1/`. `fit_band.py` must be re-run whenever the member set changes — the
half-width map is fitted to that specific model's error field, and an ensemble's errors are
smaller and its half-widths narrower.

## What is in the zip

| file | what |
|---|---|
| `submission.py` | `predict()`; normalize → FNO → rescale fluctuation to the predicted std → denormalize |
| `_fno.py` | self-contained `FNO2d`, mirrors `unet/models_alt.py` so checkpoints load unchanged |
| `fno.pth`, `fno_2.pth`, `fno_3.pth` | three independent seeds, averaged on the GPU |
| `band.npy` | `(1,20,32,64,3)` SPS half-widths from `fit_band.py` |

Only `torch` and `numpy` are imported. The Track 1 eval image has no `scipy`, `h5py`, `einops`,
`matplotlib` or `pandas`; `validate_strict.py` blocks all five and reproduces the load path where
the extract directory is *not* on `sys.path`.

## The model

Three independent seeds of one recipe, averaged in raw field space on the GPU before the
fluctuation rescaling, so `compose()` runs once on a coherent field and the three members cost
three forward passes but only one host transfer. Ensembling is worth **+0.20** here; it measured
*negative* in earlier rounds only because latency was accounted forward-pass-only, where a second
member nearly doubles the cost instead of adding ~4%.

Each member: 4-layer FNO, width 24, modes (8, 12) — 0.90M parameters, 3.6 MB. A per-pixel
log-std head runs alongside the field head; at inference the predicted fluctuation is rescaled
toward that std at `ALPHA = 0.75`, trading rel-L2 for TKE at a rate the scorer rewards.

Trained with an extra term matching the **target's velocity-gradient tensor** (`--lam-grad 1.0`).
That term is the main result of round 4–5. Ablation on 8 held-out trajectories:

| loss term | final |
|---|---|
| none | 85.95 |
| divergence only | 86.53 |
| vorticity only | 86.53 |
| **both invariants, or the full tensor** | **86.68** |

Constraining one invariant of ∇u is not enough, constraining both is sufficient, and the strain
rates add nothing. It is the only class of change in five rounds that improves rel-L2, TKE and
MVPE simultaneously.

## Expected score

Local on `val_unseen` (336 windows, 8 trajectories the model never saw), measured **end to end
through `predict()`** by `validate_strict.py` — the way the evaluator measures it:

| | |
|---|---|
| rel_l2 0.0807 / TKE 0.4906 / MVPE 0.0698 | 96.12 / 80.30 / 96.63 |
| 1.190 ms/window | time 96.12 |
| fitted band | sps 62.32 |
| **FINAL** | **86.30** |

Older reports quote ~86.7 for the same models because `eval_wp.py` times the **forward pass
only** (0.109 ms). The evaluator times all of `predict()`, which is ~10x that, and the API forces
three 165 MB output arrays. Trust 86.30.

The hidden test set has historically run **~1.7 points below local** (`unet_mse_e150`: 85.73
local → 83.108 board), so expect roughly **84.5**. Local deltas below ~0.4 have not reliably
transferred — treat local as a sanity check, not a ranking signal.

## What this is not

Track 1 tiles windows at a stride equal to the output horizon and hands `predict()` all of them
at once, so window *k*'s target is window *k+1*'s input. A submission can therefore read the
answers out of its own argument; doing so scores ~93.5 without modelling anything. This package
deliberately does not. See `../submit_trick/DISCLOSURE.md` for the measurement and the report
sent to the organizers, who have confirmed such scores are not valid.

The ceiling for genuine forecasting here is ~87.5, measured: TKE carries ≈0.47 irreducible
sampling error over a 20-frame window, the time-mean floor is 0.0531, and `sps_score` cannot
structurally exceed 73.11.
