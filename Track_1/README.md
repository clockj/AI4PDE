# Track 1 — Sim2Real

Codabench: https://www.codabench.org/competitions/17363/
Starting kit: `~/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7/`

**Goal.** One-shot forecasting. Given 20 frames of real PIV velocity, predict the next 20.
No feedback, weights frozen at submission time. See `../CLAUDE.md` for the full contract,
metrics and measured baselines.

```python
# submission.py at the zip root
def predict(input_array, metadata=None):
    # (N, 20, 32, 64, 3) raw physical units, channels [u, v, p]
    # -> (N, 20, 32, 64, 3), or {"prediction":…, "lower":…, "upper":…}
```

Zip root also needs the checkpoint, `load_baseline.py`, `rpde_baselines/`, `_vendor/`.
Eval container has **no** `scipy` / `h5py` / `einops` / `pandas` — never `import h5py` here.

## Layout

```
unet/       model.py (shared backbone + data pipeline), train.py, unet_mse.pth
            ^ our best model: plain MSE regressor, 83.83 held-out. NOT a diffusion model.
diffusion/  pde_refiner.py, unet_refiner.pth   — PDE-Refiner experiment (negative result)
deeponet/   deeponet_baseline.py, eval_deeponet.py, deeponet.pth
make_val.py builds val_unseen.npz / val_seen.npz
eval_all.py scores every model on both sets and prints generalization gaps  <- start here
report/     report_<YYYYMMDD>.md
```

`unet/model.py` is the single source of truth for the backbone, normalization and windowing;
`diffusion/pde_refiner.py` imports it and adds only the refinement objective.

## Status (2026-07-29)

| model | final (held-out) | ms/win |
|---|---|---|
| **U-Net (mse)** ours | **83.83** | 0.87 |
| FNO sim_real_ft (organizers, scored on its own training data) | 83.06 | 11.6 |
| PDE-Refiner ours | 82.53 | 1.84 |
| DeepONet ours | 81.09 | 0.68 |

Full analysis, literature survey and caveats: `report/report_20260729.md`.

## Next

1. Tune the SPS interval band — +1.3 final measured, no model change.
2. Variance head for TKE — up to +5.0 final; the largest remaining block.
3. Self-normalize by window mean speed — insurance for the unseen test Re of 2968.
4. Submit the U-Net to Codabench. Nothing here is leaderboard-validated yet.
