# T4 `--aug-resim`: implementation, verification, prepared jobs (smith, 2026-09-28)

## Patch
`experiments/decision_20260928/patches/t4_aug_resim.patch` (sha1 7a1578a0…, `git diff` of `dec0928/wt_t4` = 4e5ed37 + patch). Only `Track_1/unet/train2.py` changes: +132 / −8 lines.
* New module functions `resim_frames(d, t0, s)` and `aug_resim(trajs_t, sel, S, p, gen)`. Frame j comes from source time t0 + s·j by linear interpolation (`a + w(b−a)`), and the velocities are multiplied by s **in physical units**. An element is set to 0 when it is 0 in a neighbour that carries weight. A sample with t0 + 39s > T−1 is left as it is (s = 1) and counted.
* New flags `--aug-resim` (default 0.0) and `--aug-resim-p` (default 0.5). The only change to existing defaults is that `config` gains these two keys.
* Guards:
  * Validation runs under `args.aug_resim != 0.0`. It refuses `--hist`, `--ctx≠20`, `--teacher`, `--subpix`, `--mixup`, `--norm self` and `--aug-scale`, and it also refuses S < 0 and p outside (0, 1].
  * Every other new statement runs under `args.aug_resim > 0.0`: creating the generator, the batch block right after `fetch()` (before `self_norm`), the per-epoch `resim:` log line, and saving and restoring the generator in the resume file.
  * The one default-path change: `save_resume` now builds its dict under a name (`ck_out`) before `torch.save`. The saved content is the same.
* RNG: the new draws come from a **dedicated CPU `torch.Generator`** seeded with `--seed + 7000003` (unseeded if `--seed < 0`). Each batch draws 2·B float64 uniforms: one for the hit decision and one for log s. The default CPU stream (randperm, sim randint), the CUDA stream and numpy are not touched. The only thing that differs from the control is the data, and so the gradients.

## Verification
* **Default path unchanged: VERIFIED.** The source scan found 19 uses, all guarded. CPU job pz_dp_t4_cpucheck (Pitzer 54995764) ran a tiny seeded run (ffno w8 L2, stride 40, 2 epochs, sim, `--ckpt-every 1`, CPU) four times:
  * `wt_base`: losses 5.11194 / 4.21785.
  * `wt_t4` without the flag, and `wt_t4` with `--aug-resim 0.0`: identical printed losses; all 31 weight tensors bit-identical to `wt_base`; the final global CPU RNG state is equal; the resume file has no extra keys.
  * `wt_t4` with `--aug-resim 0.25`: losses 5.21111 / 4.28139; 30 of 31 tensors differ. The final global CPU RNG state is still equal to `wt_base`, which confirms the dedicated generator shifts no existing draw. The resume file gains the key `rng_resim`.
  * Log: `dec0928/t4/logs/pz_dp_t4_cpucheck_54995764.log`.
* **Verifier** (`patches/t4_aug_resim_check.py`): every check PASSED except one, the source-order check. That check was faulty (it matched a comment), has been fixed since, and has not been re-run. What it checks holds by the line numbers of train2.py in wt_t4: the aug block comes before self_norm (line 1272), std_t(y) (line 1497) and tke(y, mul) (line 1505).
  * The augmentation draws nothing from the global RNG, the generator is reproducible, and s lies in [0.8, 1.25].
  * At s = 1 the stored window comes back bit for bit on real data.
  * Travelling wave: the augmented sample equals the analytic U→sU field to max relative error 9.9e-4 and rms 5.3e-4 (s = 1.25, 0.8, 1.1). An amplitude-only scale gives rms 0.08–0.19.
  * A field linear in time is reproduced exactly (≤1.4e-7), and the frame step across the 19→20 boundary equals s²·slope: input and target are contiguous.
  * Zero elements stay exactly 0 and normalise to exactly −MEAN/STD.
  * Windows past the end are left un-augmented and counted.
  * Refusals verified: all 7 combinations, plus bad p and negative S.
  * Variance / TKE targets: `std_t(y)` and `tke(y, mul)` are formed inside `compute_loss` from `y` after the replacement, so they come from the augmented target. The code line order is 1272 `self_norm` < 1497 `std_t(y)` < 1505 `tke`. Numerically, std_t scales by s (diff 4e-9).
* **Zero set is NOT static (the brief expected it would be).** In all 82 trajectories some pixels are zero in only some frames: up to 16 % of pixels, with 25–48 on/off transitions. Joint u=v=0 is about 9–12 % of elements per frame, of which only 1.2–6 % of pixels are static. v-only zeros are 1–3 % of elements. The brief's rule (0 if either weighted neighbour is 0) grows the zero fraction slightly: on 4 trajectories, 0.1098→0.1135, 0.1144→0.1161, 0.1263→0.1273 and 0.0973→0.0985. Kept as specified; a nearest-frame rule is the alternative if this matters.
* **Windows that cannot be augmented** at s = 1.25, stride 5: 1.23 % of 13 368 (all 82 trajectories) and 1.24 % of 8 056 (RE training list). None at s = 0.8. In the smoke: 15 and 5 per epoch.
* **Smoke** (pz_dp_t4_smoke, Pitzer 54995776, V100S): 2 epochs, RE list with `--aug-resim 0.25`, run as `--max-epoch 1` then `--resume`, which restored the generator state and completed. The exclusion list was derived in code (26 real + 30 sim + 8 VAL_TRAJ = 36) and is equal to `ckpt/RE/exclude.txt`. The RE sim cache x1ff76a3af6 had been built by forge before my run. Replaced windows: 0.492 and 0.497 of 8 056.
  * **With the flag:** epoch 0 took 32 s and epoch 1 took 29 s.
  * **Without the flag:** the same job and node, 2 epochs, took 27 s and 26 s (53 s cumulative). The flag costs about +3 s per epoch (≈ +10 %, CPU-side resampling). Verified.
* The smoke job wrote its checkpoint; its config records `aug_resim 0.25` and git `4e5ed37-dirty`, and the resume file carries `rng_resim`.

## Prepared job scripts (not submitted)
All pass `sbatch --test-only`. They run on Ascend nextgen, `-c 8`, 6 h, `--requeue`, with `--ckpt-every 20 --resume`, the same single-member read and `.done` sentinel as wave 1, and they were generated by `dec0928/t4/jobs/gen_t34.py`.
* T3 (from `wt_base`, which must be clean at 4e5ed37): `dec0928/t4/jobs/rp_dp_t3_{a20,re}_s{1..4}.sbatch`. Checkpoints go to `dec0928/t3/ckpt/<REG>/<reg>_ffnosim_lam10_s<k>.pth`.
* T4 RE: `dec0928/t4/jobs/rp_dp_t4_re_s{1..4}.sbatch`. T4 A20/A10: `dec0928/t4/jobs/later/rp_dp_t4_{a20,a10}_s{1..4}.sbatch`. Checkpoints go to `dec0928/t4/ckpt/<REG>/<reg>_ffnosim_resim_s<k>.pth`. Each job refuses to start unless `git diff` of `wt_t4` hashes to 7a1578a0…
* Submit with: `DESK=$(squeue -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,); sbatch ${DESK:+--exclude="$DESK"} <script>`

## GPU-hours
One Pitzer GPU job (smoke, 4 min 03 s elapsed), about 0.07 GPU-h. The CPU job used no GPU.

## Deviations
1. The scaling is applied in physical units before normalisation, not with the normalised affine form. `fetch` returns raw physical frames, and this keeps the zero exactly 0; the affine form only reaches it up to rounding.
2. Validation is guarded by `!= 0.0` so that negative S is refused.
3. The CPU identity run executes train2.py with its `dev = "cuda"` line swapped for `"cpu"` (a runner inside the verifier), because train2 hard-codes cuda.
4. My Pitzer jobs were submitted without `--exclude`: the desktop node a0130 is on Ascend, and Pitzer rejects the name. All prepared scripts are Ascend and take the exclude.
5. Checkpoint file names use a lowercase regime (`re_…`, `a20_…`), as wave 1 does.
6. The T3 scripts live in `t4/jobs/` as the brief asked; their logs, sentinels and checkpoints go to `dec0928/t3/`.
