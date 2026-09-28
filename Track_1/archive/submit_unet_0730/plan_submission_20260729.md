# Track 1 — First Submission Plan — 2026-07-29

Goal: package `Track_1/unet/unet_mse.pth` (83.83 held-out) as a valid Codabench Track 1
submission, verified locally end-to-end with the official scorer before spending a submission.

This is WP5 of `Track_1/plan/plan_20260729.md`, pulled forward. No model training.

## Requirements, confirmed against the kit

| requirement | source | our status |
|---|---|---|
| `submission.py` at the **zip root** defining `predict(input_array, metadata=None)` | `submission_template.py` | to write |
| input `(N, 20, 32, 64, 3)`, **raw physical** units, channels `[u, v, p]` | template docstring | must normalize ourselves |
| output `(N, 20, 32, 64, 3)`, or dict `{prediction, lower, upper}` | template + `smoke_test_kit.py` (D) | to write |
| no-bounds path must return a **bare ndarray**, not a dict | `smoke_test_kit.py:271` | honour exactly |
| with bounds: keys exactly `{prediction, lower, upper}`, equal shapes, `lower <= prediction <= upper` | `smoke_test_kit.py:281-284` | assert before returning |
| predictions must be **finite everywhere** or all subscores are 0 | `scoring.py:301` | assert |
| extracted archive **< 256 MB** | kit README | checkpoint is **7.7 MB** — no fp16 packing needed |
| eval image lacks **scipy, h5py, matplotlib, pandas** | `smoke_test_kit.py:63` | see blocker below |
| eval image **does** ship requests, setuptools/pkg_resources | `smoke_test_kit.py:62` | (corrects an earlier note in `CLAUDE.md`) |
| `p` channel unmeasured on real data | kit README | zero-fill output channel 2 |

### Blocking issue found

`Track_1/unet/model.py` does `import h5py` at **module level** (line 20). h5py is **not in the
evaluation container**, so any submission importing `model.py` crashes on import. Fix: move the
h5py import inside `load_trajectories()`, which is the only function using it. One-line surgical
change; training code unaffected.

## Steps

| # | step | verify |
|---|---|---|
| 1 | Make `h5py` import lazy in `unet/model.py` | `import model` succeeds with h5py blocked; `train.py` still runs one epoch |
| 2 | Fit the SPS half-width on `val_seen.npz` | best `h` reported; honest (fit and eval on different sets) |
| 3 | Write `submit/submission.py` — loads `unet_mse.pth`, normalizes, forwards, denormalizes, zero-fills `p` | returns `(N,20,32,64,3)` float32, finite; bare ndarray when `RETURN_BOUNDS=False` |
| 4 | Write `submit/validate_submission.py` — runs `predict()` on `val_unseen.npz`, writes `predictions.npz` + `ref/targets.npz`, runs the **official** `scoring.py` | `final_score` within ±0.2 of `eval_all.py`'s 83.83 |
| 5 | Re-run step 4 with `scipy/h5py/matplotlib/pandas` blocked (the kit's `_BlockedTopLevel` technique) | identical score under the eval-image package set |
| 6 | Write `submit/build_zip.py` | zip < 256 MB, `submission.py` at root, loads and scores from a clean extract in a temp dir |
| 7 | Report results into `submit/` | — |

## Decision: bounds on or off for submission #1

`RETURN_BOUNDS = True` by default, using the half-width from step 2.

- Reward: +1.3 final measured (83.83 → 85.15).
- Risk: malformed bounds zero **every** subscore. But `pred ± h` with `h > 0` constant satisfies
  `lower <= pred <= upper` trivially, and finiteness follows from the prediction being finite,
  which we assert anyway. Step 4/5 verify the whole path through the real scorer.
- Residual unknown: `h` is absolute (physical units) and fitted at Re 3750–26700. The test set
  includes Re 2968, where velocities are ~3× smaller. A fixed absolute band should *gain*
  coverage there, but this is untested.
- The flag makes it a one-line revert if the leaderboard disagrees with local scoring.

## Out of scope

No training, no architecture change, no self-normalization. WP1/WP2 come after this submission,
per the main plan — the point of submitting now is to learn whether local numbers transfer at all.

## Success criterion for the whole task

A zip that, extracted into a clean directory with only torch+numpy importable, scores
**≥ 85.0** through the official `scoring.py` on `val_unseen.npz`.
