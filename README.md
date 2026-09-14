# AI4PDE — RealPDE Competition (NeurIPS 2026)

Our code for the [RealPDE competition](https://realpdecompetition.github.io/): forecast the next
20 frames of real PIV velocity around a NACA4418 airfoil from the previous 20, on a 32×64 grid.
CFD simulations of the same airfoil are available for training, never at test time.

| track | protocol | entry point | best board score |
|---|---|---|---|
| [1 — Sim2Real](https://www.codabench.org/competitions/17363/) | one batched call over every test window, raw physical units | `predict(input_array, metadata)` | 81.345 |
| [2 — LTTTA](https://www.codabench.org/competitions/17385/) | streaming at batch 1, normalized units, the previous window's truth revealed | `get_ttt_model(dir, device)` → `ttt_step(...)` | 81.164 |

Board scores as of 2026-09-14; the live record is `Track_1/board_rounds.py`, rendered by
`python leaderboard/lb.py --html`. A board score blends five subscores: relative L2 error,
turbulent kinetic energy, mean velocity at wake probes, latency, and an interval-coverage score
(SPS).

## Approach

Both tracks ship the same frozen model. Test-time adaptation was measured on Track 2, and it does
not pay for its latency.

* **Ensemble.** Four width-96, 8-layer Fourier neural operators with heavy channel dropout, plus
  two ViTs. The second architecture family earns its place through interval coverage rather than
  rms error.
* **Training.** A tail-robust (Welsch) field loss, matching of the target's velocity-gradient
  tensor and temporal derivative, a per-pixel log-std head, and co-training on CFD mapped into
  the PIV measurement space.
* **Inference.** Members are averaged on the GPU, the predicted fluctuation is rescaled toward
  the variance head, and per-element prediction intervals are fitted on held-out twins of the
  shipped members. Same-architecture members are fused into one batched invocation; weights ship
  as int8, with fp16 autocast verified against fp32 at load.

The measured reasons for each choice, and the long list of things that did not work, are in
`ROADMAP.md` and `ROUNDS.md`.

## Repository

| path | contents |
|---|---|
| `Track_1/` | models, training, evaluation instruments and the package code — see `Track_1/README.md` |
| `Track_1/unet/` | data pipeline (`model.py`), architectures (`models_alt.py`), training (`train2.py`), sim → real map (`simreal.py`) |
| `Track_1/submit_fno/` | the shipped `submission.py` and `_fno.py`, band fitting, build and gates — see its `README.md` |
| `Track_2/` | the streaming submission, its gate and evaluator, and the closed adaptation studies — see `Track_2/README.md` |
| `leaderboard/` | submission and experiment ledgers, and the calibration-log page — see `leaderboard/README.md` |
| `CLAUDE.md` | the working manual: contracts, scoring, the current package, standing rules |
| `ROADMAP.md` | what the board says, what is closed, what is next |
| `ROUNDS.md` | the detailed record, round by round |

Not in git: the data and official kits (`~/RealPDE/`), checkpoints (`*.pth`), validation caches
(`val_*.npz`) and built archives. Uploaded archives, with the predictions recorded before each
upload, are kept next to the repository in `../sub_candidates/`.

## Setup

```bash
conda env create -f environment.yml   # Python 3.10, torch 2.2.2+cu121, numpy 1.26.4: the evaluators' stack
conda activate realpde                # on OSC: source .pyenv.sh
```

The data is the competition release, mirrored on Hugging Face as
`AI4Science-WestlakeU/RealPDE-Competition-Data` (`train_real/`, `train_sim/`,
`baseline_checkpoints/`). The official starting kits, with the scoring program and smoke tests,
come from the Codabench pages. Paths default to the OSC machine; elsewhere, pass `--data` and
`--sim` to `Track_1/unet/train2.py`.

## Typical workflow

```bash
# 1. train a held-out twin with the shipped FNO recipe (swap --sim-exclude-held for --all-data to train a shipped member)
cd Track_1/unet
python train2.py --arch fno --width 96 --modes-y 8 --modes-x 12 --n-layers 8 --stride 5 --epochs 400 \
  --batch 32 --lr 2e-4 --dropout 0.45 --var-head --var-mlp --lift-mlp --loss mse \
  --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 \
  --alpha-train 0.0 --sim ~/RealPDE/train_sim/train_sim --lam-sim 0.3 --sim-loss full \
  --sim-exclude-held --seed 1 --out my_twin.pth

# 2. score it alone, then as part of an ensemble, on the board-like AoA 15-20 subset
cd ..
python compare_v9.py --ckpt unet/my_twin.pth --alpha 0.60 --aoa 15 20
python ens_v9.py --sets "unet/a.pth,unet/b.pth,unet/c.pth" --alpha 0.60 --aoa 15 20

# 3. build and gate both tracks' archives from the repo root; the member lists and packing
#    knobs are required (see Track_1/submit_fno/README.md)
cd ..
RPDE_A="..." RPDE_M="..." RPDE_INT8=1 RPDE_TAG=... bash Track_1/submit_fno/build_round13.sh 0.60
```

Every submission's five subscores are predicted and written down before it is uploaded (in
`../sub_candidates/README.md`, then ledgered in `leaderboard/board.jsonl`). That record is what
turns the leaderboard into a measurement instrument.

## Two things that shaped the code

* **Track 1's test windows tile**: window *k*'s target is window *k+1*'s input, so `predict()`
  could read the answers out of its own argument. We disclosed this to the organizers
  (`Track_1/submit_trick/DISCLOSURE.md`), and every package forecasts each window from that
  window's own input only.
* **The evaluation images lack `scipy`, `h5py`, `matplotlib` and `pandas`**, so the archives
  import only `torch` and `numpy`, and the model code is vendored into a single `_fno.py`.
