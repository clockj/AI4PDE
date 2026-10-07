# The final submission: Track 2's training package (2026-10-07)

What goes to the organizers for the final evaluation, where it lives, and what was checked. The owner sends it;
agents never do (`CLAUDE.md` rule 1).

## What the organizers ask for

The invitation e-mail (2026-09-29, "Dear xtgg233 … invited to the final evaluation … in Track 2") and
`FINAL_SUBMISSION.md` of the Final-phase starting kit (`../realpde_final_starting_kit_track2.zip`, downloaded 10-07;
sha256 `f09a888b…`), which also holds the `Dockerfile`, `README_TEMPLATE.md`, `METHOD_TEMPLATE.md` and a toy
`solution.py`. By **2026-10-09 24:00 UTC**:

1. **A Docker image** (linux/amd64) built from the kit's Dockerfile (`FROM w3nhao/realpde-track2@sha256:f07d1d68…`,
   `WORKDIR /workspace`, `COPY . /workspace`), pushed to a **private Docker Hub repository with read access for the
   account `w3nhao`**; named with its digest in the e-mail.
2. **A zip on Google Drive** with the same Dockerfile, `solution.py`, the code, a `README.md` and a `METHOD.md` (their
   templates); they review the code from it and rebuild the image from it if the pull fails. The README is to be
   accompanied by the log of the training run behind the Codabench submission.
3. **An e-mail** to realpde-competition@googlegroups.com: team name, Codabench username, track, image name and
   digest, **the Codabench submission id the package reproduces**, the Drive link. A request for more resources, with
   its reason, goes in the same e-mail.

The interface: `/workspace/solution.py` with `train(data_dir, output_dir, device)` and
`get_ttt_model(output_dir, device)` (the Codabench contract with `output_dir` in place of the submission directory).
They run `docker run --rm --network none --gpus '"device=0"' --shm-size 32g -v …:/data:ro -v …:/output IMAGE python
-c "import solution; solution.train('/data', '/output', 'cuda')"`, then their own script imports `solution` from
`/workspace`, loads the model from `/output` and runs the private test set, both without network. `/data` holds
`train_sim/`, **`train_real/` without `7575_0.h5`**, and `mean_std_real.pt`. **One A800 80 GB, 16 cores, 128 GB,
24 hours** (more on request). The earlier facts (10-05, public API): the submitted package is also scored on the
development set and must reproduce the leaderboard row.

**The Codabench submission id is 948792** (`https://www.codabench.org/api/leaderboards/19282/`, owner `xtgg233`,
created 2026-09-27 23:48 UTC, final_score 82.0555530033 = the row of record). The Codabench username is `xtgg233`.

## What is sent: branch `final-submission`

The method of the leaderboard row (T2R24, `T2_MRGP2_track2_MRGP2.zip`, sha256 `1aa2896f…`), as training code in
the kit's shape, on the orphan branch **`final-submission`** (worktree **`../code-finalsub`**), reorganised from
`fdac00b`'s code on 10-06 and fitted to the kit on 10-07:

| | |
|---|---|
| `Dockerfile` | the kit's, verbatim apart from a comment: the base image already holds torch 2.2.2 / numpy 1.26.4 / h5py 3.16 / scipy, nothing is installed |
| `solution.py` | `train`: the eleven recipes (`RECIPES`), the CFD map once per table, the trainings **`JOBS` at a time on the one GPU** (longest first, `--ckpt-every 20 --resume`), the band, the archive, the two gates, the archive unpacked into `output_dir`; `get_ttt_model`: the archive's own on that directory |
| `unet/` | the training code of 10-06; `data.py` now resolves the organizers' flat layout and the archives' nested one (`release_dirs`) and always skips `7575_0` (`DROP`) |
| `band/` | the band fit; `scoring.py` is the organizers' Track 1 kit v9 file, vendored unchanged (`RPDE_KIT` is gone) |
| `submission/` | `submission.py`, `_fno.py`: byte-identical to `fdac00b`'s tree files, as before |
| `build/` | `build.py`, `validate.py`, `causal_check.py`, `stream.py` (`RPDE_REAL`) |
| `logs/` | the eleven training logs behind submission 948792 (found 10-07; see below) |
| `README.md`, `METHOD.md` | the kit's templates filled in |

`run_track2.sh` and `environment.yml` are gone (the container is the environment). The branch's `README.md` says
what differs from the submission: `7575_0` absent (their mount), the `SIGPOW` exponents refitted, everything else a
literal of `build/build.py`.

**The training runs of record** (the checkpoints `Track_1/unet/{ffno_all_w40l8_sim_s1..4, arch_vit_all_s1,
arch_vit_all_cmb3sim_s3, p2_ffno_l8_sim_s1..3, arch_vit_d384, arch_vit_cmb3sim}.pth`): the seven F-FNOs on Pitzer
V100S-32GB (jobs 54405441/2, 54408222, 54408272; 54403827/8, 54405127; 4.85 h per member, 4.37 h per twin), the four
ViTs on Ascend A100-40GB (jobs 7199859, 7252310, 7204136, 7142610; 2.00 / 3.81 / 3.43 / 1.82 h), all with
`~/.conda/envs/realpde` (torch 2.2.2+cu121); 43.6 GPU-hours. Logs: scratch `members/ffno_sim/*.log`,
`screen/p2sim/*.log`, `logs/r20/r20d_vit.log`, `logs/r24/r24b_vit_s3.log`, `logs/r21/r21c_vit.log`; `arch_vit_d384`'s
job piped the training through `tail -3`, so only its last three lines exist (`logs/slurm-7142610.out`). Every
checkpoint's `config` matches its log's; the recipes are the branch's.

## What was checked

10-06 (branch of that day, `run_track2.sh`): old vs new training code bit-identical on the CPU (`verify/cpu_equiv.sh`,
still valid: it runs `unet/train.py`); the band outputs and the archive of record 9/9 byte-identical from the
checkpoints of record (job 55454190); the branch end to end at 2 epochs in a bubblewrap sandbox (jobs 55454191,
55454608).

10-07 (the kit's shape), `verify/job_container.sh`: the image built with podman from the branch's Dockerfile,
converted to a SIF, run with apptainer the way the organizers run docker (`--containall --cleanenv`, `/data` read-only
in their layout without `7575_0`, `/output` writable, 1 GPU, 16 cores, 128 GB):

| check | job | result |
|---|---|---|
| the image: `podman build` from the branch's Dockerfile on a0108, `python -c "import solution"` inside it | login node | builds; base image holds torch 2.2.2+cu121, numpy 1.26.4, h5py 3.16.0, scipy 1.15.3, Python 3.10.14 |
| **the time budget**: one training alone, 6 epochs, inside the image on an A100-PCIE-40GB | 7711451 (B) | F-FNO **33.4 s/epoch** (3.7 h at 400), ViT with the CFD term **33.6 s/epoch** (3.7 h); peak GPU memory 2.5 / 2.6 GiB; the first epoch's loss identical on the V100 (Pitzer 55496932: 46 s/epoch) |
| four trainings at once on the one GPU | 7711451 (A) | 133 s/epoch each = **no throughput gain** (the step is GPU-bound: 73 ms of 84 ms per step is the forward/backward, profiling job 7711527); six at once (7711492) the same. Hence `JOBS = 1` per GPU, the trainings spread over the GPUs seen, and **the e-mail asks for two GPUs or 48 h**: ~37 h on one A100, ~18 h on two |
| `solution.train('/data', '/output', 'cuda')` end to end at 6 epochs: map, 11 trainings, windows, band, archive, both gates, unpack | 7711451 (A) | **`TRAIN OK`** in 39 min; `/output` holds the nine files + the archive; SUBMITTABLE, CAUSALITY OK; peak host RSS 5.3 GB (the map); 81 real trajectories (no `7575_0`) |
| the band and the archive from the eleven checkpoints of record, inside the image | 7711451 (D) | **all 9 files byte-identical to the upload** with the band of record; the band refitted in the container differs from the committed one by ≤ 2.2e-4 in `base` (sigma_ref, p, clips bit-identical; SIGPOW 0.2427/0.2455 vs 0.2423/0.2454 — GPU kernels, as on 10-06) |
| `solution.get_ttt_model('/output', 'cuda')` through the organizers' evaluation loop (`verify/eval_output.py`, the T2 kit's `local_eval` stream + its `scoring.py`) | 7711492 (C) | **`EVAL_OUTPUT OK`** on the 6-epoch output of 7711492 (A100) and of Pitzer 55496932 (V100): `FrozenEnsemble` loads in 3.5 s, 336 steps over the 8 held-out trajectories, bounds on every step, the kit's scoring gives the same subscores as the package's own `validate.py` (rel 95.39, tke 78.91, mvpe 96.28, sps 49.8 for that smoke model), 3.6 ms/step |
| the full run, 400 epochs, from the final SIF (commit 952d1e1): 2 × A100 (7712670), 1 × A100 (7712671), 2 × V100 (Pitzer 55498053), then the evaluation step on `/output` and on the archive of record. (`--cleanenv` drops Slurm's `CUDA_VISIBLE_DEVICES` and `--nv` binds every GPU of the node; the scripts pass the allocation in explicitly since 09:00) | pending | resubmitted 10-07 09:00 EDT (the first three were cancelled: see the trap below); expected 10-08 ~04:00 (2 GPUs) / ~23:00 (1 GPU) |

## What the owner does

1. **Build and push the image** (on a0108, where `docker` is rootless podman with its storage on `/tmp`):
   ```bash
   cd ~/workspace/realpde/code-finalsub
   podman build --format docker --platform linux/amd64 -t docker.io/<dockerhub-user>/realpde-t2:final .
   podman login docker.io
   podman push --digestfile /tmp/realpde_t2_digest.txt docker.io/<dockerhub-user>/realpde-t2:final
   cat /tmp/realpde_t2_digest.txt        # sha256:… — the digest that goes into the e-mail
   ```
   On hub.docker.com: make the repository **private** (Settings → Visibility) and add **`w3nhao`** as a collaborator
   with read access (Collaborators). If the plan does not allow collaborators on a private repository, say so in the
   e-mail: they rebuild from the zip.
2. **The zip**: `git -C ~/workspace/realpde/code-finalsub archive --format=zip -o ~/workspace/realpde/artifacts/xtgg233_track2_training_package.zip HEAD`,
   upload to Google Drive, share "anyone with the link can view".
3. **The e-mail** (reply to the invitation thread, to realpde-competition@googlegroups.com):

   > Subject: RealPDE final evaluation — training package of team xtgg233 (Track 2)
   >
   > Team name: xtgg233. Codabench username: xtgg233. Track: 2 (LTTTA).
   > Image: `<dockerhub-user>/realpde-t2@sha256:<digest>` (private; `w3nhao` has read access).
   > Codabench submission id this package reproduces: 948792 (final_score 82.0555530033, 2026-09-27).
   > Google Drive: `<link>` — the Dockerfile, solution.py, the code, README.md, METHOD.md, and under `logs/` the
   > eleven training logs of the run behind submission 948792.
   >
   > Resources: we ask for **two GPUs, or 48 hours on one**. The package trains eleven networks (six members and
   > five twins for the interval band); they take about 37 GPU-hours on an A100 (3.7 h each for nine of them,
   > measured inside this image), and the training step is GPU-bound, so they cannot share one card. `train()`
   > spreads them over every GPU the container sees: with `--gpus '"device=0,1"'` it finishes in about 18 – 19
   > hours; on one GPU it needs about 37 – 38 hours. 16 cores and 128 GB are enough either way (peak host memory
   > 5.3 GB per process, 3.5 GB of GPU memory per training).
   > One difference from the submitted run is unavoidable and is stated in the README: our networks had been trained
   > with `train_real/7575_0.h5` present (the duplicate of `6300_0`), which your mount does not contain.
   >
   > Contact: zclock33@outlook.com (xtgg233).

4. **Push** `final-submission` and `qwu/dev`; agents never push.

## Files

```
final_package/
  README.md                      this file
  verify/job_container.sh        the organizers' run (apptainer from the podman-built image): timing, pipeline, evaluation, identity
  verify/eval_output.py          the organizers' evaluation step: solution.get_ttt_model('/output') through the kit's local_eval loop
  verify/cpu_equiv.sh            old code vs new, bit for bit, on the CPU
  verify/job_equiv.sh            old vs new on a GPU: training, CFD map, band, archive, gates (10-06)
```
