# RealPDE final-day sprint — common brief (2026-09-27, all times EDT)

You are one agent in a commander-led sprint. Read this file fully, then your own mandate (in your prompt).
Be fast: the clock below is hard. Write results to files; keep your final report short and factual.

## 0. Clock (hard)
| time (EDT) | what |
|---|---|
| 16:10 | sprint starts |
| **16:40** | design specs + code due (design agents) |
| **16:50** | last moment to launch a training job |
| **18:20** | every training job must have finished (use `-t 01:30:00` or less) |
| 18:20–18:55 | evaluate, refit band, build + gate candidate packages |
| **19:00** | commander picks the final package(s) |
| 19:59 | Codabench cutoff (23:59Z). The USER uploads. Agents NEVER upload. |

Check `date` often. A result that arrives after 18:55 is worth nothing.

## 1. Where we stand
* Task: 20 frames of real PIV (u,v) on 32×64 → next 20 frames. Two tracks, same hidden test set, same 5 subscores.
  `final = 0.4778·rel + 0.1010·tke + 0.0921·mvpe + 0.0965·time + 0.2498·sps`.
  `sps = 100·ā·C` (ā from rel/tke/mvpe, C = containment: mean of exp(−width/0.05639) over elements whose target is inside the band, 0 outside).
* Our best: **T1 81.6224 (#22)**, **T2 81.7396 (#15)**. **Top-10 cut (09-27 01Z): T1 81.885, T2 81.954, still rising.**
  Today's already-planned packages predict T1 81.60, T2 81.78. We need roughly **+0.2 to +0.3 more**.
* Gap decomposition vs top-10: our rel is top-2 of the board; **the gap is containment C** (ours 0.599 T1 / 0.607 T2 vs 0.611–0.624)
  plus ~0.07 of `time` (ours ~5.0–5.9 ms/step, top-10 median ~4.7). Prices: rel error −1 % ≈ +0.034; tke error −1 % ≈ +0.028;
  ΔC +0.01 ≈ +0.18; 1 board ms ≈ 0.07.
* **Today is a free roll**: the board keeps each team's best, so a worse submission costs nothing.
  => **maximise the probability of a large gain (upside), not the expected value.** A high-variance design is welcome.
* Shipping system ("recipe A / r38"): 4× F-FNO w40 L8 (+sim co-training) fused in one invocation + 2× ViT d384 L8 fused in a second;
  mean of 6 fields = y0; ALPHA 0.80 amplitude rescale toward the σ head; band centred on y0 (centre0),
  half = base·clip(σ/σ_ref,0.3,3)^0.45, base fitted per element on held-out twins at stretch 1.39. T2 adds causal online
  band calibration (R1, asymmetric) and optionally D1c (causal per-pixel bias removal) and SIGEMA.
* MUST READ (skim, 5 min max): `/users/PAS1187/qwu/workspace/realpde/artifacts/20260927_progress_report.md`
  sections 零, 4.4–4.6, 五, 六 (what was tried and closed, backbone configs). Project rules: `CLAUDE.md` in the repo
  (its "Current state" is stale at 09-20; the progress report wins). **Do not re-propose a closed arm unchanged.**

## 2. Hard rules (violations disqualify us)
1. No external data, no new simulations, no outside pretrained weights. Everything must be re-creatable from the released
   train_real/train_sim by a from-scratch re-train. Warm-starting from OUR OWN checkpoints is allowed (it is a two-stage recipe),
   but you must write down the full recipe (stage 1 = REPRODUCE.md §1.4/§1.5, stage 2 = yours).
2. No leakage. Track 1: the prediction for window k may depend only on window k's own input — no statistic pooled across
   the windows of a call. Track 2: only already-revealed windows (strictly causal). `stream_eval.py --causal-check` must pass.
3. Held-out evidence only: all-data checkpoints supply weights, never evidence. Evidence comes from twins trained without
   the 8 validation trajectories (`model.VAL_TRAJ`; twins drop `--all-data`, add `--sim-exclude-held`).
4. Agents never upload to Codabench and never `git push`. Do not commit to the main repo; work in your worktree.
5. Never touch jobs that are not ours (`ondemand/*`, `ttf_*`, `hd_*`, `p5_*`, `p6_*`, `tt26-*`, `m2_*`). Name ours `rp_f27<dir>_*`.

## 3. Resources
* Repo (read-only for you): `/users/PAS1187/qwu/workspace/realpde/main` (branch qwu/agent @ 55f0e88).
* **Your worktree** (edit here): `/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/wt_<dir>` (a, b, c, x). Untracked checkpoints are
  NOT in the worktree: reference them by absolute path `/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet/<name>.pth`
  (or symlink the ones you need into the worktree's `Track_1/unet/`).
  Members: `ffno_all_w40l8_sim_s1..s4`, `arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3`. Twins: `p2_ffno_l8_sim_s1..s3`,
  `arch_vit_d384`, `arch_vit_cmb3sim`. Bands: `Track_1/bands/`. Val splits: `Track_1/val_unseen.npz` etc.
* Sprint scratch: `S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927` → `$S/{jobs,logs,ckpt,pkg,notes}`. New checkpoints go to
  `$S/ckpt/<dir>/`. **Nothing generated goes under $HOME** (1M-inode quota; hook refuses at 95 %). `/tmp` is node-local.
* Data: `/users/PAS1187/qwu/RealPDE/train_real/train_real`, `/users/PAS1187/qwu/RealPDE/train_sim/train_sim`.
  Sim-map caches: `/fs/scratch/PAS1187/qwu/tmp/realpde/work/simmap_p1_f0*.npy` (do not delete).
* Python: `/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u` (torch 2.2.2+cu121). Job preamble: see
  `/fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh` (copy it; set `REPO` to YOUR worktree; keep `HDF5_USE_FILE_LOCKING=FALSE`).
* Slurm (one multi-cluster; $HOME and /fs/scratch shared): account `-A PAS1186`, `--gpus-per-node=1`, `-c 12`.
  * Ascend A100: `-M ascend -p nextgen` (many idle nodes). `debug-nextgen` (≤1 h) is allowed today for smokes.
  * Cardinal H100: `-M cardinal -p gpu` (or `-p debug`, ≤1 h).  Do NOT use Pitzer.
  * **Every sbatch must exclude the desktop node**:
    `DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)` then
    `sbatch ${DESK:+--exclude="$DESK"} ...` (Ascend only; currently a0130). `--output` on scratch, pass `--chdir`.
  * Submitting from inside another job: `env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES sbatch ...`.
  * Login nodes have no GPU. Wait on jobs with `squeue`/`sacct` polling + log tails, not blind sleeps.
* GPU budget: **≤ 150 GPU-hours per direction** (one GPU per member; train members in parallel jobs, never sequentially).
* Throughput facts: the shipped F-FNO recipe (400 epochs, stride 5, +sim) took ~3.9 h on A100; ViT 2.0–3.8 h.
  A throughput probe is running; its result will appear in `$S/notes/throughput.md` — read it when it exists.
  `train2.py` has `--init <ckpt>` (warm start) and `--epochs`. **train2.py argparse defaults are NOT the recipe** — copy the
  full flag list from REPRODUCE.md §1.4/§1.5 and change only what your design changes.

## 4. Instruments (every local number is a sign, not a size)
* Operating point: **`--alpha 0.80 --band-p 0.45`**, stretch/inflate 1.39, centre0 band. Instrument defaults differ — pass them.
* Ensemble-level screen at the size that ships: `Track_1/ens_v9.py --aoa 15 20` (board proxy) and the full split; single
  checkpoints: `Track_1/compare_v9.py`. Always refit the band for a changed member set (`submit_fno/fit_band.py`).
* OOD (private set): `Track_1/aoa_holdout_eval.py`. Track 2 stream: `Track_2/stream_eval.py` (+ `--causal-check`).
* Reference numbers for the current system will be written by the pipeline agent to `$S/notes/baseline_ref.md`.
* History says: the proxy over-reads error cuts ~2×; across model FAMILIES it predicts neither size nor sign; "tighter band"
  changes over-read or flip on the board; a +1.15 ms extra invocation costs ≈ −0.08 on T1. Report honestly with brackets.

## 5. Build + gates (pipeline agent owns the runbook: `$S/notes/BUILD_RUNBOOK.md`)
`Track_1/submit_fno/build_round13.sh` builds both tracks from a member list (`RPDE_A`), twins (`RPDE_M`), band, and
`RPDE_SHIP_*` knobs; gates are `submit_fno/validate_strict.py` (T1) and `Track_2/validate_t2.py [--adaptive]` (T2) and must
print SUBMITTABLE. Archive cap 256 MB; eval containers have torch 2.2.2, no scipy/h5py/pandas. A NEW architecture needs
load + forward support inside `Track_1/submit_fno/_fno.py` (it is self-contained in the zip) — budget time for that.

## 6. What a deliverable looks like
* Design agents: `$S/notes/design_<dir>.md` = hypothesis · which subscore moves and why · predicted Δfinal with bracket ·
  exact training / eval commands · inference integration plan · reproducibility recipe · kill criteria · job IDs launched.
* Executors: `$S/notes/result_<dir>.md` = table of held-out reads vs the baseline reference (AoA 15–20 and full split,
  paired, same instrument), package path + sha256 + gate lines, predicted five subscores with a bracket.
