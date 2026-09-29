---
name: osc-jobs
description: Submit, monitor and time Slurm jobs for this project on the OSC clusters (Ascend A100, Pitzer V100), within the storage and etiquette rules of the shared allocation. Use whenever a task needs a GPU.
---

# Jobs on OSC

The login node has no GPU. Everything that loads a model runs in a Slurm job.

## Template

```bash
#!/bin/bash
#SBATCH -J rp_<what>            # ours are rp_* on Ascend, pz_* on Pitzer
#SBATCH -A PAS1186
#SBATCH -M ascend
#SBATCH -p nextgen
#SBATCH --gpus-per-node=1
#SBATCH -c 12
#SBATCH -t 00:40:00
set -uo pipefail
cd <a scratch worktree, or the repo for read-only work>
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
...
echo RP_<WHAT>_END               # the last line: a waiter looks for it
```

```bash
DESK=$(squeue -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
sbatch ${DESK:+--exclude="$DESK"} --output=/fs/scratch/PAS1187/qwu/tmp/realpde/logs/%x_%j.log job.sh
```

## Rules

* **Every submission excludes the node of the owner's interactive desktop.** Look it up at submit time as above; it
  moves when the desktop restarts.
* **Never touch jobs that are not ours**: `ondemand/*`, `ttf_*`, `hd_*`, `p5_*`, `p6_*`, `tt26-*`, `m2_*`.
* **Nothing generated goes under `$HOME`**: job scripts, logs, worktrees, caches and dumps live in
  `/fs/scratch/PAS1187/qwu/tmp/realpde/`. `quota -s` shows the inode count; `df -i` does not.
* `debug-nextgen` (≤ 1 h) needs the owner's OK. Cardinal queues are slow: cancel after 3 min pending and use Ascend.
* Slurm copies the script at submit time; editing it afterwards changes nothing.

## Waiting and reading

* Wait on the queue, not on a guess: `squeue -M ascend -u "$USER" -h -n <name>`; a waiter must also stop when the job
  leaves the queue without its sentinel.
* Logs can contain NUL bytes: read them through `tr -d '\000'`.
* `nvidia-smi` near 0 % is normal for this workload; the 12-core cpuset binds before the GPU.

## Timing

* Never time while another `rp_*` job runs on the node; use `--exclusive` and set `OMP_NUM_THREADS=12` explicitly (an
  exclusive node otherwise gives 128 threads and an invalid read).
* Only interleaved arms count: A, B, A, B on the same node, medians of at least four rounds.
* V100 wall clock is never latency evidence. A100 ms × 1.850 is the 4060-equivalent the old tables use.
* A local latency delta transfers to the board only for a cut in compute within one kernel family, and within a factor
  of two. Quote a bracket.

## Pitzer

**Ascend comes first whenever it is idle** (owner, 2026-09-28: the two cost the same and the A100 is faster), training
included. Pitzer is the overflow: send work there only when a real job on Ascend is still pending after about 15
minutes. **`sbatch --test-only` does not tell**: on 09-28 it printed a start 24 hours away while real `nextgen` jobs
started within 4 – 21 seconds; submit one job as a probe instead. Slurm dependencies do not cross clusters.
V100, partition `gpu-exp`, waits under a minute; the A100 is about 1.3× faster for the F-FNO. Interpreter
`/users/PAS1187/qwu/envs/mural_osc/bin/python` (torch 2.7.1); the `realpde` conda env also runs there.
