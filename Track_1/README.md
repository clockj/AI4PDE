# Track 1 — Sim2Real

Codabench: https://www.codabench.org/competitions/17363/ · Official kit:
`/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9/`

**Goal.** One-shot forecasting: 20 frames of real PIV velocity in raw physical units, the next 20
out, weights frozen, no feedback. `../CLAUDE.md` holds the contract and the rules, `../STATUS.md` the
state, `../REPRODUCE.md` the method, `../ROADMAP.md` what the board says and what is closed,
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

Six members: four **factorized FNOs** (`ffno_all_w40l8_sim_s1..s4`: width 40, 8 layers, modes 16×32,
Dropout2d 0.3, co-trained on mapped CFD) and **two ViTs** (dimension 384, 8 layers, dropout 0.3).
Every member has a field head and a per-pixel log-std head. The F-FNOs and the second ViT train their
field with a curvature-matched Welsch "coverage" loss in place of MSE (`--lam-f 0.0 --lam-cov 2.0`),
which bends only the error tail, plus velocity-gradient (`--lam-grad 1.0`) and temporal-derivative
(`--lam-tgrad 0.3`) matching and co-training on CFD mapped into the measurement space
(`--sim --lam-sim 0.3 --sim-loss full`); the first ViT uses plain MSE with `--lam-grad 1.0` and no sim.
At inference the members are fused by family into two invocations (`_fno.StackedFFNO`,
`_fno.StackedViT`), their fields and σ are averaged on the GPU, and the fluctuation is rescaled toward
σ at `ALPHA = 0.80` with one gain for `u` and `v` (`JOINT_GAIN`). The interval is centred on the mean
field before that rescaling and widened to contain the prediction; its half-width is the SIGBLEND
band (`sigblend/`, `bands/k27_sigblend_lam0.5.npz`). `../REPRODUCE.md` §13 is the specification.

## Training

`train2.py`'s argparse defaults are **not** the recipe: `--alpha-train` defaults to 1.0, which trains
through the composition and collapses the model. The recipes are the fragments at the top of
`../reproduce.sh`, and every checkpoint stores its own (`torch.load(path, weights_only=False)["config"]`).
A shipped F-FNO member:

```bash
cd Track_1/unet          # checkpoints are written here and are not tracked
python train2.py --arch ffno --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 --var-head \
  --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 \
  --alpha-train 0.0 --stride 5 --batch 32 --lr 2e-4 --wd 1e-5 --epochs 400 \
  --sim /users/PAS1187/qwu/RealPDE/train_sim/train_sim --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 \
  --all-data --seed 1 --out ffno_all_w40l8_sim_s1.pth
```

A held-out twin — the only source of local evidence, and of the band's error field — drops
`--all-data` and adds `--sim-exclude-held`. `--seed` makes a control and its arm share
initialization and batch order.

## Layout

| path | what |
|---|---|
| `unet/` | `model.py` (data pipeline, `VAL_TRAJ`), `models_alt.py` (F-FNO, ViT, FNO, the heads, the physics operators, `build`), `models_closed.py` (the closed families), `train2.py`, `simreal.py` (the sim → real domain map); every checkpoint (untracked) |
| `submit_fno/` | the package: `submission.py`, `_fno.py`, `fit_band.py`, `bake_ffno_path.py`, `build_zip.py`, `validate_strict.py`, `build_round13.sh`, **`build_final.sh`** — see its `README.md` |
| `sigblend/` | the σ power law of the final band: twin dump, fit, the constants the packages bake |
| `bands/` | every SPS band that has scored on a board, versioned, with `MANIFEST.md` |
| `opoint.py`, `band_io.py` | the operating point every instrument defaults to; the band loader that reads a band as the package does |
| `qd_k/` | the derivation of the QD error-head band (09-24; not shipped, kept for `reproduce.sh --recipe QD`) |
| `verify/` | gates and bit-identity checks: scorer against the official kit, CPU smoke test, package identity, AMP, graph, fusion, packing, blend |
| `closed/` | one-off experiments whose question is closed, with a README table of each verdict |
| `archive/` | the first two months: the July U-Net package, the disclosed protocol exploit, the round-4 sweeps, old build scripts, the round 9–12 training queue and logs |
| `report/`, `plan/` | per-round write-ups (`report_*`), plans (`plan_*`) and campaign ledgers (`campaign_*`) |
| `val_*.npz` | cached validation sets, untracked (`make_val.py`, `re_split.py`) |

## Top-level files, by role

`../EVALUATION.md` is the full account: what each instrument reads, its exact invocation at the
operating point, what it writes and how the board corrects it. Every local number is a sign, not
a size.

**Scoring and calibration**

| file | what |
|---|---|
| `sps9.py` | loads the v9 scorer (`sps9.sc`) and adds exact, fast SPS plus `optimal_band` — every instrument scores through it |
| `board_rounds.py` | every scored board round on both tracks (`ROUNDS`, `ROUNDS_T2`), the measured latency constants (`INV_MS_*`, `LOC_SLOPE_*`, `PREC_SLOPE_T1`) and helpers (`board_ms`, `time_score`, `invocation_cost`) |
| `final_v9.py` | five subscores → `final` under the free pooled fit, with the admissible range; self-checks against `board_rounds` at import |
| `predict_board.py` | a package's five board subscores, predicted before upload; holds `RATIO`, `LAT_*` and the board anchors |
| `proxy_calib.py` | board/proxy ratios per round, and the proxy's resolution |
| `calib_band13.py` | the board's per-element error tail, inverted from a round whose band and `sps` are both known |
| `sub_value.py` | what a submission would resolve, as information |

**Accuracy instruments**

| file | what |
|---|---|
| `compare_v9.py` | single checkpoints, each with its own band (`--aoa 15 20` is the board-like subset) |
| `ens_v9.py` | member sets on the AoA 15–20 proxy — the screen at the size that ships |
| `eval_v9.py` | the shared `infer()` every scorer above calls |
| `kfold.py` | out-of-fold over all 82 trajectories; `--split re --bands 1 2 3` is the confirmation step, `--paired` the A/B |
| `aoa_holdout_eval.py` | a novel ANGLE with the shipped band machinery — the only OOD instrument, and the one that speaks to the private test set (`--inflate-sweep` prices the band width there) |
| `mean_blend_re.py` | the gated mean-blend out of distribution (`BLEND_BETA`; closed for the public set, re-opened by the pivot) |
| `alpha_board.py` | ALPHA against the board's error scale — refit it whenever the member set changes |
| `quant8.py` | the scored cost of a packing scheme (`--score BAND.npz`) |

**Screen ledger** (`plan/plan_20260914.md` §4, §7)

| file | what |
|---|---|
| `screen_eval.py` | a screen arm end to end; upserts `../leaderboard/screen.jsonl` and drives `predict_board`, `ens_v9` and `fit_band` through `run_main` |
| `screen_sets.py` | member-set evaluations for an arm, by token (`FNO3`, `VIT2`, …) |
| `screen_latency.py` | candidates and the shipped references interleaved in one process — the kernel-level latency read |
| `lat_r15.py`, `lat_fast17.py` | the package-level latency read, and what a fast path costs. The two disagree; quote both as a bracket |

**Splits**

| file | what |
|---|---|
| `make_val.py` | builds `val_seen.npz` / `val_unseen.npz` from `unet/model.py:VAL_TRAJ` |
| `re_split.py` | Re-extrapolation held-out sets; `excluded()` derives the list (retired as a decision instrument — the public hidden set is interior in Re) |

**Build and gates:** `submit_fno/` (package, band, `validate_strict.py`) and `verify/` (bit-identity
and feature gates). **Training:** `unet/`.

## Build and gate

`bash submit_fno/build_final.sh OUTDIR` builds and gates the two packages of record; `build_round13.sh`
underneath it takes every constant as an `RPDE_*` variable. The knobs and the gates are in
`submit_fno/README.md`, the procedure in the `build-package` skill. **Record the predicted board
subscores before the user uploads.**
