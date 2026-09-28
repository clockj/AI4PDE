# Training throughput probe (X2) — FINAL, 16:27 EDT 2026-09-27 (all arms done on both clusters; probe jobs finished)

Jobs: Ascend 7566570 (a0278, A100-PCIE-40GB), Cardinal 14942370 (c0817, H100, ran in `debug`). Logs:
`$S/logs/rp_f27p_{asc,car}_*.log` (every line timestamped). Probe checkpoints in `$S/ckpt/probe/{asc,car}/` (junk, 3 epochs).

## Table (steady s/epoch = mean of epochs after epoch 0; epochs in 75 min = (4500 - startup - epoch0)/steady + 1, rounded down ~3 %)

| cluster | recipe | batch | epoch 0 | **steady s/epoch** | startup before epoch 0 | **epochs in 75 min** |
|---|---|---|---|---|---|---|
| Ascend A100 | (a) F-FNO w40 L8 twin +sim | 32 | 39 s | **30.5** | 83 s cold (7 s import + 76 s h5 load) / ~44 s warm | **~140** |
| Ascend A100 | (b) same, `--init p2_ffno_l8_sim_s1` | 32 | 31 s | 31.0 | 44 s | ~140 |
| Ascend A100 | (d) F-FNO | 64 | 31 s | 30.5 | 44 s | ~140 |
| Ascend A100 | (d) F-FNO | 128 | 32 s | 31.5 | 43 s | ~135 |
| Ascend A100 | (c) ViT d384 L8 twin +sim | 32 | 31 s | 30.6 | 43 s | ~140 |
| Cardinal H100 | (a) F-FNO w40 L8 twin +sim | 32 | 23 s | **18.3** | 74 s cold (67 s h5 load) / ~44 s warm | **~235** |
| Cardinal H100 | (b) same, `--init` | 32 | 19 s | 18.5 | 45 s | ~235 |
| Cardinal H100 | (d) F-FNO | 64 | 21 s | 20.5 | 45 s | ~210 |
| Cardinal H100 | (d) F-FNO | 128 | 22 s | 21.5 | 45 s | ~200 |
| Cardinal H100 | (c) ViT d384 L8 twin +sim | 32 | 16 s | **15.0** | 45 s | **~290** |

Startup: the "cold" number is the first train2.py in a job (h5 read from $HOME NFS); the sim-map cache HIT on every arm
(`/fs/scratch/.../work/simmap_p1_f0_x476193e300.npy`, 92/100 sim trajectories, 17756 sim + 12040 real windows) and costs < 0.1 s.
Budget ~90 s startup to be safe. Final save is < 1 s.

**Throughput is CPU/host-bound, not GPU-bound**: on A100 b32 = b64 = b128 = ViT(14.7M params) = F-FNO(1.3M) ≈ 30.5 s/epoch.
`fetch()` stacks every window from CPU tensors in a Python list comprehension on the main thread, then `.to(dev)`.
Bigger batch does NOT help (and on H100 is slightly slower; with lr unchanged it also takes fewer optimizer steps:
ep2 loss 1.188 (b32) / 1.372 (b64) / 2.064 (b128)). The H100 advantage (1.67x) is mostly Cardinal's faster CPU.
Losses agree across clusters to 1e-5 (ep2 1.18826 vs 1.18827) — same numerics, results interchangeable.

**No AMP / channels-last / torch.compile / TF32 / num-workers switch exists in train2.py** (grepped argparse and code).

## --init, LR schedule, resume
* `--init CKPT`: **non-strict**: copies only tensors present in both with equal shape, prints
  `init: ... -> k/N tensors copied (m left at their random init)`. For the F-FNO twin: **79/79 copied**, both clusters.
  Always check that line (a flag mismatch silently gives a partial load, reported only by the count).
* It loads **weights only**: fresh AdamW (no optimizer state), fresh OneCycleLR, epoch counter 0.
* Epoch-0 loss: warm start **0.767** vs from scratch **3.480** (ep2: 0.764 vs 1.188).
* **LR schedule** = `OneCycleLR(max_lr=--lr, total_steps=epochs*steps_per_epoch)` with torch defaults:
  **30 % warmup** from lr/25 (8e-6) up to --lr (2e-4) by cosine, then cosine anneal to lr/25/1e4 (~8e-10) at the last step.
  It always spans the given --epochs, so a short run still anneals fully to ~0; a warm start with the default --lr re-heats to 2e-4
  (for fine-tuning consider a smaller `--lr`; that is a design choice, not measured here).
* `--ckpt-every N` + `--resume`: writes `<out>.resume.pt` (model, opt, sched, epoch, RNG); `--resume` refuses a different --epochs.
  **Not needed** for a non-preemptible 80-min job with `-t 01:25:00`; cheap insurance (`--ckpt-every 10`) is in the template.

## Queue waits observed (16:13 submit)
* Ascend `nextgen`: **19 s** (16:13:28 → 16:13:47). At 16:26: 33 idle nextgen nodes × 3 A100 ≈ 100 free GPUs.
* Cardinal `gpu,debug`: **75 s** (→ 16:14:43), landed in **debug**. At 16:26 Cardinal gpu nodes had **96/104 H100 allocated (8 free)
  and 50 pending jobs**. `debug` caps at **1 h** — an 80-min job must go to `-p gpu` and will queue behind those 50.
* Cardinal env OK: `/users/PAS1187/qwu/.conda/envs/realpde/bin/python` → torch 2.2.2+cu121, cuda True, NVIDIA H100.

## Recommendation
* **Bulk (20+ jobs): Ascend nextgen, `--epochs 130`** for F-FNO or ViT at batch 32 (140 fit in 75 min; 130 leaves ~5 min margin),
  `-t 01:25:00`. Keep batch 32.
* **Optional, ≤ ~6 jobs: Cardinal H100.** An 80-min job needs `-p gpu` (queue risk). A **≤ 60-min `-p debug` job fits ~170 F-FNO epochs
  (≈ 200 ViT)** — more than 80 min on A100 — with a 75 s wait observed. Use `--epochs 160`, `-t 01:00:00`, `-p debug,gpu`.
* Template: `$S/jobs/TEMPLATE_train80.sh` (common.sh preamble, REPO=worktree, HDF5_USE_FILE_LOCKING=FALSE, output on scratch,
  desktop exclusion in the sbatch line in its header).
