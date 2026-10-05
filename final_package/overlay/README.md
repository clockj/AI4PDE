# RealPDE Track 2 (LTTTA) — training package

| | |
|---|---|
| team | `@@TEAM@@` |
| leaderboard row | Main Development Phase, 2026-09-27: `final_score` **82.055553** (rel 94.871355, tke 80.665340, mvpe 94.394560, time 91.243742, sps 44.381923) |
| submitted archive | `T2_MRGP2_track2_MRGP2.zip`, sha256 `1aa2896f7f98c3c775cba926c96a9972ea81dc66a3003090b34aaaf1a68f54bf` |
| this package | training and build code of that submission, at the commit it was built from (`fdac00b`) |

The package re-creates the submitted archive from the competition's released training data and nothing else. It holds
code only: **no model weights, no fitted tables, no data, no external or pretrained model.** The eleven networks, the
validation windows, the interval band and its constants are all produced by the one command below.

## Run it

```bash
conda env create -f environment.yml && conda activate realpde    # or any Python 3.10 with torch 2.2.2, numpy 1.26, scipy and h5py
RPDE=<release>  RPDE_KIT=<kit>  WORK=<output dir>  bash run_track2.sh
```

| variable | meaning |
|---|---|
| `RPDE` | the competition release: the directory holding `train_real/train_real/*.h5` and `train_sim/train_sim/*.h5` |
| `RPDE_KIT` | the directory of the **Track 1 starting kit v9** that holds the official `scoring.py` (the band fit and the gates score with it) |
| `WORK` | a writable directory; checkpoints, caches, logs and the archive go here |
| `PY` | optional: the Python interpreter (default `python3` on `PATH`) |

It ends with `RUN_TRACK2 OK` and writes **`$WORK/submission_track2.zip`**, the Track 2 submission: `submission.py`,
`_fno.py`, `band.npz` and six member files (`fno.pth`, `fno_2.pth` … `fno_6.pth`), 35 MB extracted. Any failure stops
the run with a line that names the log to read. The run is restartable: a stage whose checkpoint exists is skipped.

Requirements: one CUDA GPU (we used an A100 40 GB and a V100 32 GB), 32 GB of host memory (training holds the data in
memory and peaks at 21 GB), about 5 GB of disk under `$WORK` (each sim-map cache is 1.6 GB) and 0.5 GB inside the
package directory, which must be writable: the build writes its intermediate files into it, so run one build per copy
of the package. No network access is needed.

## What it does

| stage (`RPDE_REPRO_STAGE`) | what | seeds | A100 hours |
|---|---|---|---|
| `twins_vit_p` | ViT twin, trained without the eight validation trajectories | 1 | 1.8 |
| `twins_vit_c` | ViT twin with CFD co-training | 1 | 3.4 |
| `twins_ffno` | three F-FNO twins with CFD co-training | 1 2 3 | 3 × 3.4 |
| `members_vit_p` | ViT member, all data | 1 | 2.0 |
| `members_vit_c` | ViT member, all data, CFD co-training | 3 | 3.8 |
| `members_ffno` | four F-FNO members, all data, CFD co-training | 1 2 3 4 | 4 × 3.75 |
| (build) | validation windows → static band → twin dump → σ power law → archive → gates | | 0.2 |

About **37 GPU-hours** in sequence. The six members are what the archive ships. The five twins are the same recipes
trained without eight validation trajectories (`model.VAL_TRAJ`); they exist only to measure the error the interval
band is fitted to. Every training command is in `reproduce.sh` (`FFNO`, `VITC`, `VITP`, `SIMF`), every constant baked
into the archive is listed in `Track_1/submit_fno/build_final.sh`.

### In parallel

The eleven trainings are independent. One job per stage and seed, each with a sim-map cache of its own (the cache is
written without a lock; 1.6 GB per job that co-trains on CFD), then the build:

```bash
RPDE_REPRO_STAGE=twins_vit_p                        RPDE_SIMCACHE=$WORK/sc_1  bash run_track2.sh
RPDE_REPRO_STAGE=twins_vit_c                        RPDE_SIMCACHE=$WORK/sc_2  bash run_track2.sh
RPDE_REPRO_STAGE=twins_ffno    RPDE_REPRO_SEEDS=1   RPDE_SIMCACHE=$WORK/sc_3  bash run_track2.sh     # also seeds 2, 3
RPDE_REPRO_STAGE=members_vit_p                      RPDE_SIMCACHE=$WORK/sc_6  bash run_track2.sh
RPDE_REPRO_STAGE=members_vit_c                      RPDE_SIMCACHE=$WORK/sc_7  bash run_track2.sh
RPDE_REPRO_STAGE=members_ffno  RPDE_REPRO_SEEDS=1   RPDE_SIMCACHE=$WORK/sc_8  bash run_track2.sh     # also seeds 2, 3, 4
bash run_track2.sh                                  # all eleven are skipped as existing; fits the band, builds, gates
```

(`RPDE`, `RPDE_KIT` and `WORK` as above on every line; the training jobs may share one copy of the package, the
build must not overlap another build in the same copy.) The longest single run is 3.8 hours.

## What to expect from a re-training

* **The weights are not bit-identical.** Seeds fix initialisation and batch order, not cuDNN's kernels: on other
  hardware the F-FNO weights come back about 0.5 % apart and the ViT weights about 10 %. Everything after training is
  deterministic.
* **The four error subscores reproduce at the seed level.** Our own from-scratch re-training of these eleven networks
  (2026-09-25, V100 cards, PyTorch 2.7.1) moved our local estimate of `final_score` by +0.0006 on the validation
  trajectories at angles of attack 15 – 20 and by +0.0040 on all eight. The seed-to-seed spread of one member is about
  0.017 on that scale, of the ensemble about 0.01.
* **`time` is measured on your hardware.** On the development board the same archive family drew 5.0 to 6.7 ms per
  step; the row above was the slowest draw we recorded (6.7 ms).
* **The built `_fno.py` is byte-identical to the submitted one** (sha256 `6d72a3c6ef306a6c…`), and `submission.py`
  differs from the submitted one (sha256 `10fd96d2a47a89cb…`) in one line: the literal `SIGPOW`, two exponents fitted
  on the twins trained here (submitted: `(0.24232558823017827, 0.24535682217400656)`).
* **From our own eleven checkpoints this package rebuilds the submitted archive file by file**, all nine files
  byte-identical, when it is given the band of the submission instead of refitting one (checked 2026-10-05).

## Data

* Training reads every `*.h5` it finds in `train_real/train_real` and `train_sim/train_sim`: 82 and 100 files in the
  release as published. **Our submitted networks were trained with `train_real/7575_0.h5` present**, the file the
  data page asks participants to leave out because it is a copy of `6300_0`; we had not excluded it. With the file in
  place this package repeats the submitted training. Without it the code runs unchanged on 81 trajectories
  (two training stages checked) and trains a slightly different model.
* CFD trajectories are used in training only, mapped into the units of the measurement by `Track_1/unet/simreal.py`
  from statistics of the released real data (a twin never sees the statistics of a trajectory it holds out).
* No generated data, no external data, no pretrained weights, no augmentation.

## The method

Both tracks of our entry ship the same six networks; Track 2 adds three corrections that learn from the targets the
stream reveals and train nothing.

* **Members.** Four factorized Fourier neural operators (width 40, 8 layers, 1.3 M parameters each) and two vision
  transformers (dimension 384, 8 layers, 14.7 M each). Every member also predicts, per pixel, the temporal standard
  deviation σ of the target window.
* **Training.** 400 epochs over windows at stride 5, batch 32, AdamW with a one-cycle schedule peaking at 2e-4,
  weight decay 1e-5. A bounded (Welsch) field loss, matching of the velocity gradient and of the time difference, a
  log-σ loss and a turbulent-kinetic-energy loss; five of the six members and four of the five twins co-train on the
  mapped CFD. Every augmentation switch of `train2.py` is off.
* **Prediction.** The mean of the six fields, with the amplitude of its fluctuation moved toward σ (`ALPHA` 0.80, one
  gain for `u` and `v`).
* **Interval.** Centred on the mean before that adjustment and widened to contain the prediction. The half-width is a
  per-element base, solved on the twins' errors, scaled per window by a power of σ (`fit_band.py`, then
  `Track_1/sigblend/make_k0_band.py`).
* **Online, strictly causal.** From the revealed targets of the current trajectory only, and cleared by
  `reset_ttt_state`: the interval's lower and upper width are recalibrated per pixel; the time-mean bias of the
  prediction is removed; the fluctuation amplitude is recalibrated. `Track_2/stream_eval.py --causal-check` replaces
  every later window by noise and requires all earlier outputs to stay bit-identical; the build fails otherwise.
* **Latency.** Members of a family are fused into one invocation; half precision, int8 weights, a dense spectral
  operator, one CUDA graph for the whole step.

## What differs from commit `fdac00b`

| file | change | why |
|---|---|---|
| `Track_2/t2_common.py` | one line: the gates read the released data from `$RPDE` | it was a path of our cluster |
| `run_track2.sh`, `README.md`, `MANIFEST.sha256` | new | the entry point, this manual, the file list |
| `Track_2/submit/_fno.py` | not shipped; `run_track2.sh` creates it | it is a symbolic link to `Track_1/submit_fno/_fno.py`, and a transport can break one |

Everything else is byte-identical to the commit (`MANIFEST.sha256` lists every file), and only the files this
pipeline reads are included. None of the three changes touches a file that enters the archive.

## Checked before hand-over

On an A100 with PyTorch 2.2.2, inside a sandbox in which none of our own directories exist (the release, the kit and
the Python environment at neutral paths, the environment variables cleared), 2026-10-05:

* this package, extracted from its archive, ran `run_track2.sh` from the released data to `RUN_TRACK2 OK` at 2 epochs
  per network: every training stage, the band refit, the build, both gates and the causality check;
* given our eleven checkpoints and the band of the submission, it rebuilt the submitted archive with all nine files
  byte-identical;
* two training stages ran on a copy of the release without `train_real/7575_0.h5`.

Not run by us: the full 400 epochs with this exact package. The recipes, the seeds and the training code are those of
our re-training of 2026-09-25.

Contact: `@@CONTACT@@`
