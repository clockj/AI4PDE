# xtgg233: Track 2 training package

## Codabench
- Username: `xtgg233`
- Submission id this package reproduces: **948792** (Main Development Phase, 2026-09-27 23:48 UTC; archive
  `T2_MRGP2_track2_MRGP2.zip`, sha256 `1aa2896f7f98c3c775cba926c96a9972ea81dc66a3003090b34aaaf1a68f54bf`)
- final_score of that submission: **82.0555530033** (rel_l2 94.8713545063, tke 80.6653398510, mvpe 94.3945599963,
  time 91.2437415498, sps 44.3819226434)

## How to reproduce
- Build command, if different from `docker build -t <name> .`: the same. The Dockerfile is the kit's: the base is
  your Track 2 evaluation image, nothing is installed, the package is copied to `/workspace`.
- Train command, if different from the default: the same,
  `python -c "import solution; solution.train('/data', '/output', 'cuda')"`.

`train()` trains the eleven networks of the submission (six members that ship, five twins that only serve to fit the
interval band), fits the band on the twins' held-out errors, bakes the six members and the band into the submission
archive, runs the archive through the platform's contract and a causality check, and unpacks it into `/output`:
`submission.py`, `_fno.py`, `band.npz`, `fno.pth`, `fno_2.pth` … `fno_6.pth` (35 MB), next to the archive itself
(`submission_track2.zip`) and a `work/` directory with every checkpoint and log. `get_ttt_model('/output', 'cuda')`
is the archive's own `get_ttt_model(submission_dir, device)` on that directory.

**Time.** The eleven trainings take **about 37 GPU-hours on an A100** (measured inside this image on an
A100-PCIE-40GB: 33.4 s per epoch for an F-FNO and 33.6 s for a ViT with the CFD term, i.e. 3.7 h each at 400 epochs;
about 1.8 h for a ViT without it; the step is GPU-bound, so running two trainings on one card gains nothing — we
measured it). `train()` runs them one at a time per GPU, **spread over every GPU the container sees**, longest first:
on one GPU the run takes about 37 – 38 hours, on two about 18 – 19. **We therefore ask, in the submission e-mail, for
two GPUs (`--gpus '"device=0,1"'`) or for 48 hours on one**; nothing else differs. The CFD map, the band fit, the
build and the gates add about 15 minutes.

The run is restartable: a checkpoint that exists is skipped, a training that was interrupted resumes from its last
20-epoch boundary. Resources used (measured inside this image on an A100-PCIE-40GB): 2.5 – 2.6 GiB of GPU memory per training (3.5 GB as `nvidia-smi` shows it), at most 5.3 GB of host memory per process (the CFD map; a training holds about 4 GB), about 6 GB of disk under `/output` while running (the two CFD-map tables, 3.3 GB, are deleted at the end). No network access is needed at any point. Nothing outside `/output`
is written.

A log line `[train <seconds>s] ...` marks every stage; the per-network logs are `/output/work/ckpt/<name>.log`. The
run ends with `TRAIN OK`.

## Hardware your submitted model was trained on
- GPU model and count: the four F-FNO members and the three F-FNO twins on one **NVIDIA V100S-PCIE-32GB** each
  (Pitzer, Ohio Supercomputer Center); the two ViT members and the two ViT twins on one **NVIDIA A100-PCIE-40GB**
  each (Ascend, OSC). One GPU per network; eleven separate jobs.
- CPU: 4 – 12 cores of an Intel Xeon per job (Pitzer: Xeon 6148 / 8268; Ascend: AMD EPYC 7643).
- Memory: 32 – 64 GB requested per job; a training holds the data in memory and peaks at 21 GB.
- Training time on this hardware: F-FNO member 4.85 h each (17 439 – 17 488 s), F-FNO twin 4.37 h each
  (15 743 – 15 748 s), ViT member with the CFD term 3.81 h (13 703 s), ViT member without 2.00 h (7 198 s), ViT twin
  with the CFD term 3.43 h (12 362 s), ViT twin without 1.82 h (6 536 s): 43.6 GPU-hours in all. The band fit, the
  build and the gates: about 10 minutes. The logs of these eleven runs are in `logs/` (see "Notes").
- Peak GPU memory on this hardware: not recorded by those runs. Measured by this package on an A100: 2.5 GiB
  allocated by an F-FNO training, 2.6 GiB by a ViT training (`torch.cuda.max_memory_allocated`; about 3.5 GB per
  process as `nvidia-smi` reports it).

## Environment
- Base image: `w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972`
  (your Track 2 evaluation image: Python 3.10.14, PyTorch 2.2.2 + CUDA 12.1 + cuDNN 8.9.2, numpy 1.26.4, h5py 3.16.0).
- Added packages and versions (pinned the same way in the Dockerfile): **none.** The training uses torch, numpy and
  h5py only; `band/scoring.py` is your Track 1 starting kit v9 `scoring.py`, unchanged (sha256
  `a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39`), which the band fit scores with.

  The submitted networks were trained with the same versions (PyTorch 2.2.2 + CUDA 12.1, numpy 1.26.4, Python
  3.10) in a conda environment.

## Data, weights and code
- Training data used: every `*.h5` of `/data/train_real` and `/data/train_sim` as mounted (81 real and 100 CFD
  trajectories), sub-sampled to 32 × 64. `train_real/7575_0.h5` is skipped by the code (`unet/data.py: DROP`) whether
  or not it is present. **The submitted networks had been trained with that file present** (82 real trajectories; it
  is a copy of `6300_0.h5`, which the data page asks participants to leave out, and we had not excluded it); with it
  absent this package trains on the 81 others, which the retrained networks will differ by at the seed level.
  The CFD trajectories enter training only, mapped into the units and support of the real recordings by
  `unet/simreal.py` from statistics of the released real data (a twin never uses the statistics of a trajectory it
  holds out). No `mean_std_real.pt` is read: its input statistics are constants in the code.
- Augmentation: none.
- Official baseline checkpoints used, if any: none. Every network starts from a random initialisation.
- Third-party code and its license: none beyond PyTorch and numpy. The F-FNO and the ViT are our own implementations
  (`unet/models_alt.py`: factorised Fourier layers after Tran et al., ICLR 2023; a plain pre-norm ViT over 4 × 4
  patches); `band/scoring.py` is the organizers' file named above.

## Randomness
- Seed(s): `torch.manual_seed` / `torch.cuda.manual_seed_all` / `numpy.random.seed` with the seed of each recipe
  (`solution.py: RECIPES`): F-FNO members 1, 2, 3, 4; ViT member without the CFD term 1; ViT member with it 3;
  F-FNO twins 1, 2, 3; ViT twins 1 and 1. The seed fixes the initialisation and the batch order. cuDNN is left in its
  default (non-deterministic) mode, as it was for the submitted runs: on the same card the weights come back the
  same for many epochs, across cards the F-FNO weights differ by about 0.5 % and the ViT weights by about 10 % after
  400 epochs, and the ensemble's subscores by about 0.01 of `final_score` (seed-to-seed spread of one member about
  0.017). Everything after training — the band fit, the build — is deterministic.
- How runs or checkpoints are selected, if at all: not at all. Every recipe is trained once, its last epoch is the
  checkpoint, all six members ship. The five twins are not selected either: they are the same recipes trained
  without the eight validation trajectories, and their errors on those trajectories are what the band is fitted to.

## Track 2 only
- Does the method call the LLM gateway: **no.** Nothing is called; the container needs no network.

## Contact
- Name and email, reachable during the Decision Phase: team xtgg233, **zclock33@outlook.com**

## Notes

**What to expect from the retraining.** Our own from-scratch re-training of these eleven recipes (2026-09-25, V100
cards) moved our local estimate of `final_score` on the eight validation trajectories by +0.0006 (angles of attack
15 – 20) and +0.0040 (all eight). The `time` subscore is measured on your hardware: the development board drew 5.0 to
6.7 ms per step for this archive family; the row above was the slowest draw (6.7 ms). The band's two `SIGPOW`
exponents are refitted on the twins trained here (submitted: `(0.24232558823017827, 0.24535682217400656)`); every
other constant of the archive is a literal in `build/build.py`. From our own eleven checkpoints this package rebuilds
the submitted archive file by file, all nine files byte-identical, when it is given the submitted band instead of
refitting one (checked 2026-10-06, and again inside this image on 2026-10-07).

**The logs of the training runs behind the submission** are in `logs/`, one per network, named as in `RECIPES`
(`member_ffno_s1.log` … `twin_vit_p.log`). They were written by the previous layout of this code (`train2.py`, whose
flags `--loss mse --alpha-train 0.0 --var-head --sim-loss full --sim-map scalar` are now the only behaviour; the
`config:` line of each log is the recipe). `twin_vit_p.log` holds only the last three lines of its run (the job
piped the training through `tail -3`); the run is the `twin_vit_p` recipe, 6 536 s on an A100-PCIE-40GB, job 7142610
of 2026-09-03.

**Layout.**

```
Dockerfile           the kit's: your evaluation image, the package copied to /workspace
solution.py          train(data_dir, output_dir, device) and get_ttt_model(output_dir, device); the eleven recipes
unet/                training: data.py (windows, normalisation), models_alt.py (F-FNO, ViT), simreal.py (the CFD map),
                     train.py (one network)
band/                the interval band: make_val.py, fit_band.py, dump_twins.py, sigma_band.py, sps.py, scoring.py (yours)
submission/          the two files that ship, with their default constants; the build bakes the constants into copies
build/               build.py (constants, int8 packing, the archive), validate.py (the contract), causal_check.py, stream.py
logs/                the eleven training logs behind submission 948792
METHOD.md            the method
```

`submission/_fno.py` rebuilds an F-FNO checkpoint through `unet/models_alt.py` when that file exists next to it (the
band fit runs the training definition) and through its own vendored copy otherwise (in `/output`); the names of that
directory and module are fixed by `_fno.py`, which is why the training code lives under `unet/`.
