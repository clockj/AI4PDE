# AI4PDE — RealPDE Competition (NeurIPS 2026)

Our code for the [RealPDE competition](https://realpdecompetition.github.io/): forecast the next 20 frames of real PIV
velocity behind a NACA4418 airfoil from the previous 20, on a 32×64 grid, with a lower and an upper bound for every
element. CFD simulations of the same airfoil are available for training, never at test time.

| track | protocol | entry point | board score at the close of Main Development |
|---|---|---|---|
| [1 — Sim2Real](https://www.codabench.org/competitions/17363/) | batched calls over the test windows, raw physical units | `predict(input_array, metadata)` | **81.671** |
| [2 — LTTTA](https://www.codabench.org/competitions/17385/) | streaming at batch 1, normalized units, the previous window's target revealed at every step | `get_ttt_model(dir, device)` → `ttt_step(...)` | **82.056** |

Main Development closed on 2026-09-27; the release of that day is the tag `final-20260927`. Track 2 finished #8 and is
in the Decision phase (to 2026-10-25); Track 1 finished #20. The state is in [`STATUS.md`](STATUS.md); every scored
round is in `Track_1/board_rounds.py`.

## The method

Both tracks ship the same six trained networks. Track 2 adds three corrections that learn from the targets the stream
reveals; they train nothing.

* **Members.** Four factorized Fourier neural operators (width 40, 8 layers, 1.3 M parameters each) and two vision
  transformers (dimension 384, 8 layers, 14.7 M each). Every member also predicts, per pixel, the temporal standard
  deviation σ of the target window.
* **Training.** A bounded (Welsch) field loss, matching of the velocity gradient and of the time difference, a
  log-σ loss and a turbulent-kinetic-energy loss; co-training on CFD mapped into the units of the measurement.
* **Prediction.** The mean of the six fields, with the amplitude of its fluctuation pulled toward σ (one gain for
  `u` and `v`).
* **Interval.** Centred on the mean before that adjustment and widened to contain the prediction. The half-width is a
  per-element base, solved exactly on the errors of the members' twins (trained without eight validation trajectories)
  on three trajectories they trained on, at angles of attack 15 – 20, scaled per window by σ.
* **Track 2, online.** From the revealed targets of the current trajectory, strictly causally: the interval's lower
  and upper width are recalibrated per pixel; the time-mean bias of the prediction is removed; the fluctuation
  amplitude is recalibrated. On held-out angles of attack these corrections are worth ten times what they are worth in
  distribution.
* **Latency.** Members of a family are fused into one invocation; half precision, int8 weights, a dense spectral
  operator, and on Track 2 one CUDA graph for the whole step.

`REPRODUCE.md` specifies the method completely. The measured reason for each choice, and the list of what did not work,
are in `ROADMAP.md` and `ROUNDS.md`.

## Reproduce

```bash
conda env create -f environment.yml && conda activate realpde     # Python 3.10, torch 2.2.2+cu121, numpy 1.26.4
RPDE_REPRODUCE_GO=1 RPDE_REPRO_OUT=<dir> bash reproduce.sh final   # 5 twins + 6 members + band + both archives, ~37 A100-hours
bash Track_1/submit_fno/build_final.sh <outdir>                    # the two archives from existing checkpoints
RPDE_REPRODUCE_GO=1 RPDE_REPRO_OUT=<dir> bash reproduce.sh decision # the Decision-phase candidate (REPRODUCE.md §14)
```

The Decision-phase candidate is the same method with every network also trained on real windows resampled in time and
scaled in amplitude by a common factor (`--aug-resim 0.25`, to first order the recording at a neighbouring Reynolds
number), and with Track 2 reading the revealed target in the normalisation the official runner uses. It is built and
reproduced from scratch; it has not been uploaded.

The data is the competition release (`AI4Science-WestlakeU/RealPDE-Competition-Data` on Hugging Face); the official
starting kits with the scoring program come from the Codabench pages. Paths default to our cluster: set `RPDE` for
`reproduce.sh`, or pass `--data` and `--sim` to `Track_1/unet/train2.py`. No external data, no new simulations and no
outside pretrained weights are used.

## Repository

| path | contents |
|---|---|
| `REPRODUCE.md`, `reproduce.sh` | the method, stage by stage, and the script that runs it end to end |
| `Track_1/unet/` | data pipeline (`model.py`), architectures (`models_alt.py`), training (`train2.py`), the sim → real map (`simreal.py`) |
| `Track_1/submit_fno/` | the shipped `submission.py` and `_fno.py`, band fitting, the build (`build_final.sh`) and the gate |
| `Track_1/sigblend/`, `Track_1/bands/` | the σ power law of the band and its constants; every band that has scored |
| `Track_2/` | the streaming `submission.py`, its gates (`validate_t2.py`, `stream_eval.py`) |
| `Track_1/*.py` | the local instruments; `opoint.py` holds the operating point they default to |
| `EVALUATION.md` | how local evaluation works and how each board result corrected it |
| `ROADMAP.md`, `ROUNDS.md` | what the board taught us, what is closed, and the record round by round |
| `experiments/`, `Track_1/closed/`, `Track_2/closed/`, `Track_1/archive/` | every experiment that did not ship: scripts, criteria, verdicts |
| `leaderboard/` | the ledgers of uploads and experiments |
| `CLAUDE.md`, `.claude/` | instructions for coding agents: rules, path-scoped notes, procedures |

Not in git: the data and official kits, checkpoints (`*.pth`), validation caches (`val_*.npz`) and built archives.

## Two things that shaped the code

* **The test windows tile**: window *k*'s target is window *k*+1's input, so a Track-1 `predict()` could read the
  answers out of its own argument. We disclosed this to the organizers
  (`Track_1/archive/submit_trick/DISCLOSURE.md`), and every Track-1 package forecasts each window from that window's
  own input only. The online corrections of Track 2 use targets the protocol has already revealed, and only those.
* **The evaluation images lack `scipy`, `h5py`, `matplotlib` and `pandas`**, so the archives import only `torch` and
  `numpy`, and the model code is vendored into a single `_fno.py`.
