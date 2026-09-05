# Track 1 — Sim2Real

Codabench: https://www.codabench.org/competitions/17363/ ·
Official kit: `~/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9/`

**Goal.** One-shot forecasting. 20 frames of real PIV velocity in, the next 20 out, weights
frozen, no feedback. `../CLAUDE.md` has the contract, the metrics and every measured baseline;
`../ROADMAP.md` has the board record and the plan. **Start with the roadmap.**

```python
# submission.py at the zip root
def predict(input_array, metadata=None):
    # (N, 20, 32, 64, 3) raw PHYSICAL units, channels [u, v, p]
    # -> (N, 20, 32, 64, 3), or {"prediction": …, "lower": …, "upper": …}
```

The eval image has **no** `scipy` / `h5py` / `einops` / `matplotlib` / `pandas`. Never
`import h5py` from anything the zip contains — `unet/model.py` imports it lazily inside
`load_trajectories()` for exactly that reason.

## The model, in one paragraph

An ensemble of **0.90M-parameter FNOs** (width 24, modes (8,12), 4 layers) with `--var-mlp`
and `--lift-mlp`, trained with a **velocity-gradient matching loss** (`--lam-grad 1.0`) —
still the only loss term in the project's history whose effect cleared the local proxy's
measured resolution. Each member carries a per-pixel log-std head; at inference the predicted
fluctuation is rescaled toward that std at `ALPHA = 0.90`, and per-element SPS half-widths come
from `band.npz`. Same-architecture members are fused into one invocation by
`_fno.StackedFNO` / `_fno.fuse`.

**`train2.py`'s argparse defaults are NOT the shipped recipe** — `--alpha-train` defaults to
1.0 and collapses the model. Read the recipe off a checkpoint's stored `config`.

## Layout

| path | what |
|---|---|
| `unet/` | `model.py` (backbone + data pipeline), `models_alt.py` (FNO2d, DeepONet, the physics operators), `train2.py`, every checkpoint |
| `submit_fno/` | the shipped package: `submission.py`, `_fno.py`, `fit_band.py`, `build_zip.py`, `validate_strict.py` |
| `verify/` | `diff_scoring_{official,real}.py` (our scorer against the official kit, worst 2.4e-6), `cpu_smoke_submission.py` (login-node gate) |
| `report/`, `plan/` | per-round write-ups and plans; `plan/plan_20260831.md` is current |

## Instruments — what to trust, and how far

| tool | what it answers |
|---|---|
| `board_rounds.py` | every scored round, the subscore weight fit (8 rounds, max residual 0.025), the board-time inversion |
| `final_v9.py` | five subscores → `final`, with the admissible range. `delta()` is the one to use between packages |
| `proxy_calib.py` | **how far the local proxy can be trusted**: level ±1.3 %, deltas below ~2 % do not transfer and can invert |
| `predict_board.py` | the pre-submission prediction. 1-for-2; read its spread, not its midpoint |
| `compare_v9.py` | per-checkpoint scores, each with its own band. `--aoa 15 20` is the board-like subset |
| `ens_v9.py`, `alpha_v9.py`, `band_cv_v9.py`, `sps9.py` | ensemble/α/band machinery; the band is closed |
| `re_split.py` | Re-EXTRAPOLATION held-out sets plus an interior control — the one thing the standard split cannot see |

**The resolution limit is the important number here.** A change must move the proxy's error by
more than ~2 % before it is worth shipping (≈ +0.19 `final`). Every change made in rounds 9–12
was smaller than that.

## Building and gating a package

```bash
# 1. fit the band FOR THIS EXACT MEMBER SET, at the alpha it will deploy at.
#    --ckpt supplies sigma_ref (the all-data members that emit sigma at test time);
#    --err-ckpt supplies the ERROR FIELD (their held-out twins). Skipping --err-ckpt is the
#    single most expensive mistake available: the July 30 package did it and scored sps 36.94.
python submit_fno/fit_band.py --ckpt $A --err-ckpt $M --alpha 0.90 --band-p 0.45 --out band8.npz

# 2. build and gate in one step -- validate_strict forwards every argument to build_zip, so it
#    always validates an archive it just built from the members named here.
python submit_fno/validate_strict.py --band submit_fno/band8.npz \
       --stage submit_fno/_package8 --out submit_fno/submission8.zip --ckpt $A
```

`validate_strict.py` needs a GPU (it also times). `verify/cpu_smoke_submission.py` is the
login-node stand-in — point it at a specific archive with `RPDE_ZIP=... python
verify/cpu_smoke_submission.py`, since a round usually has more than one candidate staged. **Record the predicted board score before submitting** — it is the
only reason any round's post-mortem was possible.

## Status

Eight scored board rounds, best **80.60907** (R11). At constant `time_score` rounds 10–12 are
indistinguishable; the leader was **81.76136** on 2026-08-25. Staged for the next slot:
8 members / 1 invocation, in `../../sub_candidates/`.
