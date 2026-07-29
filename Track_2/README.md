# Track 2 — LTTTA (Long-Term Test-Time Adaptation)

Codabench: https://www.codabench.org/competitions/17385/
Starting kit: `~/RealPDE/realpde_t2_starting_kit_v4/realpde_t2_starting_kit_v4/`

**Goal.** The same 20-in/20-out mapping, but streamed one window at a time in time order at
batch 1, with the *previous* window's ground truth revealed each step so the model can adapt
online. See `../CLAUDE.md` for the full contract, metrics and measured baselines.

```python
# submission.py at the zip root
def get_ttt_model(submission_dir, device):
    # -> object with reset_ttt_state() and ttt_step(input_norm, prev_target_norm)
    # tensors are (1, 20, 32, 64, 3), ALREADY NORMALIZED
```

Everything inside `ttt_step` is timed — adaptation, controller logic, LLM round-trips, forward.
`reset_ttt_state()` and construction are not. Hard 10-minute wall clock for the whole run.

## What to build

* `base/` — the frozen forecaster (share the Track 1 backbone; fp16-packed FNO to start).
* `adapt/` — low-capacity correction only: output affine/FiLM, low-rank adapter, or a
  backprop-free EMA/RLS bias correction on the prediction-vs-revealed-target residual.
* `variance/` — EMA calibration of the per-pixel std head from the revealed previous target.
  This is the highest-value Track 2 lever (see `../CLAUDE.md`).
* `policy/` — skip-update gating on previous-step error, to buy back time.
* `eval/` — the kit's `local_eval.py` streaming loop pointed at held-out real trajectories.

## Status

Nothing built yet. First milestone: run `local_eval.py` against the kit's persistence
submission, then against a wrapped fp16 FNO, to establish the streaming round trip.
