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

## What to build

* `data/` — window builder (stride 20, `sub_s=2`), self-normalization, sim/real loaders.
* `models/` — mean head + per-pixel log-std head + fluctuation, composed as described in
  `../CLAUDE.md`. Backbone is secondary; the variance head is where the points are.
* `train/` — sim pretraining in non-dimensional space, then real fine-tuning.
* `submit/` — packaging to a <256 MB zip, plus a local `scoring.py` run before every upload.

## Status

Nothing built yet. First milestone: reproduce the organizers' FNO submission end to end and
score it locally, so the round trip is known-good before any modelling.
