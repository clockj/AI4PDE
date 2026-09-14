# Track 1 — Sim2Real

Codabench: https://www.codabench.org/competitions/17363/ · Official kit:
`/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9/`

**Goal.** One-shot forecasting: 20 frames of real PIV velocity in raw physical units, the next 20
out, weights frozen, no feedback. `../CLAUDE.md` holds the contract, the scoring rule, the current
package and the standing rules; `../ROADMAP.md` what the board says and what is closed;
`../ROUNDS.md` every round's detail.

```python
# submission.py at the zip root
def predict(input_array, metadata=None):
    # (N, 20, 32, 64, 3) raw PHYSICAL units, channels [u, v, p]
    # -> (N, 20, 32, 64, 3), or {"prediction": …, "lower": …, "upper": …}
```

The eval image has **no** `scipy` / `h5py` / `matplotlib` / `pandas`, and `einops` only from the
kit's `_vendor/`. Never `import h5py` from anything the zip contains; `unet/model.py` imports it
lazily inside `load_trajectories()` for that reason.

## The model, in one paragraph

An ensemble of four **width-96, 8-layer FNOs** (modes (8, 12), Dropout2d 0.45 between spectral
blocks, an MLP lift, an MLP σ readout) and **two ViTs** (dim 384, 8 layers, dropout 0.3). Every
member has a field head and a per-pixel log-std head. The FNOs train their field with a
curvature-matched Welsch "coverage" loss in place of MSE (`--lam-f 0.0 --lam-cov 2.0`), which
bends only the error tail that SPS stops paying for, plus velocity-gradient (`--lam-grad 1.0`) and
temporal-derivative (`--lam-tgrad 0.3`) matching and co-training on CFD mapped into real's
measurement space (`--sim --lam-sim 0.3 --sim-loss full`). The second ViT uses the same losses and
sim; the first uses plain MSE with `--lam-grad 1.0` and no sim.
At inference the members' raw field and σ are averaged on the GPU, the fluctuation is rescaled
toward σ at `ALPHA = 0.60`, and per-element SPS half-widths come from `band.npz`.
Same-architecture members are fused into one invocation (`_fno.StackedFNO`, `_fno.StackedViT`).
The evidence behind each choice is in `../CLAUDE.md`, *What ships, and why*.

## Training

`train2.py`'s argparse defaults are **not** the shipped recipe: `--alpha-train` defaults to 1.0,
which trains through the composition and collapses the model. Read a recipe off a checkpoint's
stored `config` (`torch.load(path)["config"]`), or copy it from the job that trained it. The
shipped FNOs:

```bash
cd Track_1/unet          # checkpoints are written here and are not tracked
python train2.py --arch fno --width 96 --modes-y 8 --modes-x 12 --n-layers 8 --stride 5 --epochs 400 \
  --batch 32 --lr 2e-4 --dropout 0.45 --var-head --var-mlp --lift-mlp --loss mse \
  --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 \
  --alpha-train 0.0 --sim /users/PAS1187/qwu/RealPDE/train_sim/train_sim --lam-sim 0.3 \
  --sim-loss full --all-data --seed 1 --out fno_all_w96L8cmb3sim_s1.pth
```

A held-out twin — the only source of local evidence, and of the band's error field — drops
`--all-data` and adds `--sim-exclude-held`. `--seed` makes a control and its arm share
initialization and batch order. Job scripts for every run are under
`/fs/scratch/PAS1187/qwu/tmp/realpde/jobs/`.

## Layout

| path | what |
|---|---|
| `unet/` | `model.py` (data pipeline), `models_alt.py` (FNO2d, ViT2d, U-FNO, AFNO, DeepONet, the physics operators), `train2.py`, `simreal.py` (the sim → real domain map), `dinov2_init.py`; every checkpoint (untracked) |
| `submit_fno/` | the package: `submission.py`, `_fno.py`, `fit_band.py`, `build_zip.py`, `validate_strict.py`, `build_round13.sh` — see its `README.md` |
| `bands/` | every SPS band that has scored on a board, versioned, with `MANIFEST.md` |
| `verify/` | gates and bit-identity checks: scorer against the official kit, CPU smoke test, package identity, AMP, fusion, packing, blend |
| `report/`, `plan/` | per-round write-ups (`report_*`), plans (`plan_*`) and campaign ledgers (`campaign_*`) |
| `val_*.npz` | cached validation sets, untracked (`make_val.py`, `re_split.py`) |
| `submit/`, `submit_trick/`, `fno_deeponet/`, `deeponet/`, `diffusion/` | historical: the July U-Net package, the disclosed protocol exploit, the round-4 FNO/DeepONet sweeps, early baselines |

## Instruments

Every local number is a sign, not a size; `../CLAUDE.md`, *Instruments*, says how far each goes.

| tool | answers |
|---|---|
| `board_rounds.py` | every scored board round on both tracks; the latency constants and helpers (`board_ms`, `time_score`, `invocation_cost`) |
| `final_v9.py` | five subscores → `final` under the free 30-round fit, with the admissible range; self-checks every round at import |
| `predict_board.py` | a package's five board subscores, predicted before upload |
| `compare_v9.py` | single checkpoints, each with its own band (`--aoa 15 20` is the board-like subset) |
| `ens_v9.py` | member sets on the AoA 15–20 proxy — the screen at ensemble size |
| `kfold.py` | out-of-fold evaluation over all 82 trajectories; `--split re` bands 1–3 is the confirmation step, `--paired` the A/B |
| `proxy_calib.py` | board/proxy ratios per round, and the proxy's resolution |
| `sps9.py` | exact, fast SPS and the exact optimal band |
| `alpha_board.py`, `band_cv_v9.py` | ALPHA and band sweeps (both closed) |
| `quant8.py` | the scored cost of a packing scheme (`--score BAND.npz`) |
| `sub_value.py` | what a submission would resolve, as information |
| `re_split.py` | Re-extrapolation held-out sets (retired as decision instruments: the hidden set is interior) |

Most other top-level scripts are one-off diagnostics named after what they measured (`diag_*`,
`calib_*`, `tke_*`, `band_*`, `lat_*`); the round that used each is in `report/` or `../ROUNDS.md`.

## Build and gate

`bash submit_fno/build_round13.sh ALPHA` builds and gates both tracks' archives from one member
set; the full command, the knobs and the gates are in `submit_fno/README.md`. **Record the
predicted board subscores in `../../sub_candidates/README.md` before the user uploads.**

## Status

Live: `board_rounds.py`, and `python ../leaderboard/lb.py`. As of 2026-09-13: best **81.330636**
(R21, xfam5); last scored R24 (xfam6), 81.312041; R25 (xfam6 with fp16 autocast) awaiting its
result.
