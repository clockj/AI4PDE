# RealPDE final-day "small tricks" sprint (K-sprint) — common brief (2026-09-27, all times EDT)

You are one agent in a commander-led sprint. Read this fully, then your own mandate (in your prompt).
Be fast. Write results to files; keep your final report short and factual. Check `date` often.

A SECOND, independent sprint (the "F-sprint", `/fs/scratch/PAS1187/qwu/tmp/realpde/final0927`, jobs `rp_f27*`) runs in
parallel. **Never write into its directories, never touch its jobs.** You may READ its notes, dumps and job templates.
It covers: A = F-FNO x ViT hybrid backbone (fine-tune waves), B = band / centre probes (containment), C = Track-2 online
corrections (D1c, family weights, variance correction). **Do not duplicate A, B, C.** We are the small-tricks sprint.

## 0. Clock (hard)
| time (EDT) | what |
|---|---|
| 16:35 | K-sprint starts |
| **17:05** | design specs due (`$K/notes/design_<dir>.md`), probe jobs already launched |
| **17:20** | last moment to launch a training job (must finish by 18:25; use `-t 01:00:00` or less) |
| **18:30** | every experiment has its held-out read written down |
| 18:30-19:00 | refit band, build + gate candidate packages |
| **19:05** | commander picks; hands the zip(s) to the user |
| 19:59 | Codabench cutoff (23:59Z). The USER uploads. Agents NEVER upload, never `git push`. |

A result that arrives after 18:55 is worth nothing.

## 1. Where we stand
* Task: 20 frames of real PIV (u,v) on 32x64 -> next 20 frames. Two tracks, same hidden test set, same 5 subscores.
  `final = 0.4778 rel + 0.1010 tke + 0.0921 mvpe + 0.0965 time + 0.2498 sps` (each subscore 0-100).
  rel/tke/mvpe = `100/(1+0.5 e)`; time = `100/(1+sqrt(t/0.72896))`, t = board seconds per window/step;
  `sps = 100 * abar * C` (abar from rel/tke/mvpe per window; C = containment: mean over scored elements of
  exp(-(upper-lower)/0.05639) if the target is inside [lower, upper], else 0).
* Our best: **T1 81.6224 (#22)**, **T2 81.7396 (#15)**. **Top-10 cut (09-27 01Z): T1 81.885, T2 81.954, rising.**
  Today's already-planned packages predict T1 81.60, T2 81.78. We need **+0.2 to +0.3 more**.
* Board subscores of our best: T1 rel 94.7625 tke 79.5891 mvpe 94.2613 time 91.8773 (5.70 ms/window) sps 43.0631;
  T2 rel 94.7044 tke 79.6087 mvpe 94.1319 time 92.3393 (5.0 ms/step) sps 43.5168.
* Prices: rel error -1 % = +0.034 final; tke error -1 % = +0.028; mvpe error -1 % = +0.010; dC +0.01 = +0.18;
  1 board ms = 0.07 (at 5.7 ms; time 3.0 ms would be +0.20, 2.0 ms +0.30).
* **Today is a free roll**: the board keeps each team's best, so a worse submission costs nothing.
  => maximise the probability of a LARGE gain (upside), not the expected value.
* Shipping system ("recipe A / r38"): 4x F-FNO w40 L8 (+sim co-training) fused into one invocation + 2x ViT d384 L8 fused into
  a second; mean of the 6 fields = y0; ALPHA 0.80 rescales the per-pixel temporal fluctuation amplitude toward the sigma head
  (P = m + (y0-m)(a*sigma + (1-a)*s)/s); band centred on y0 ("centre0"), half = base*clip(sigma/sigma_ref,0.3,3)^0.45,
  base fitted per element on held-out twins at stretch 1.39. fp16 AMP, int8-packed weights, dense F-FNO path.
  T1: one predict() call per 14-99 windows, new subprocess per call, chunked. T2: batch-1 steps, CUDA graph, two streams,
  asymmetric online band calibration R1 (+ optional D1c / SIGEMA knobs).
* MUST SKIM (5 min max): `/users/PAS1187/qwu/workspace/realpde/artifacts/20260927_progress_report.md` sections
  0 (零), 4.3-4.6, 5 (五), 6 (六) = what was tried and CLOSED. Project rules: `CLAUDE.md` in the repo ("Current state" there
  is stale at 09-20; the progress report wins). **Do not re-propose a closed arm unchanged.**

## 2. Hard rules (violations disqualify us)
1. No external data, no new simulations, no outside pretrained weights. Everything re-creatable from the released
   train_real/train_sim by a from-scratch re-train. Warm-starting from OUR OWN checkpoints is allowed (two-stage recipe) —
   write the full recipe down.
2. No leakage. Track 1: the prediction for window k may depend ONLY on window k's own input — no statistic pooled across the
   windows of a call (windows tile: window k's target IS window k+1's input). Track 2: only already-revealed windows,
   strictly causal; `stream_eval.py --causal-check` must pass. Output for a window must not depend on batch composition.
3. Held-out evidence only: all-data checkpoints supply weights, never evidence. Evidence comes from the 5 twins
   (`p2_ffno_l8_sim_s1..s3`, `arch_vit_d384`, `arch_vit_cmb3sim`) on `val_unseen` (8 trajectories never trained on).
   Report AoA 15-20 (board proxy, 126 windows) AND the full split (336 windows), PAIRED against the baseline, per trajectory
   sign counts (x/3, x/8).
4. Do not tamper with the timing environment (no background GPU keep-alive threads etc.). Latency tricks must change OUR
   compute, not the platform's measurement.
5. Never touch jobs that are not ours (`ondemand/*`, `ttf_*`, `hd_*`, `p5_*`, `p6_*`, `tt26-*`, `m2_*`, and the F-sprint's
   `rp_f27*`). Name ours **`rp_k27<dir>_*`** (dir = 1, 2, 3).
6. Nothing generated under `$HOME` (1M-inode quota). No commits to the main repo, no pushes.

## 3. Resources
* `K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927` -> `$K/{jobs,logs,ckpt,pkg,notes,dumps,sub_candidates}`.
* Code: make YOUR OWN clone, edit only there:
  `git clone -q /users/PAS1187/qwu/workspace/realpde/workingtree.2/codebase $K/wt_<dir>` (HEAD 55f0e88, branch qwu/agent).
  Untracked inputs are NOT in the clone; symlink them from `M0=/users/PAS1187/qwu/workspace/realpde/main`:
  `Track_1/val_seen.npz`, `Track_1/val_unseen.npz`, and every `Track_1/unet/<ckpt>.pth` you need (334 checkpoints live in
  `$M0/Track_1/unet/`; read a recipe with `torch.load(ckpt, map_location="cpu")["config"]`).
  Members: `ffno_all_w40l8_sim_s1..s4`, `arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3`. Twins: see rule 3.
  Bands: `$M0/Track_1/bands/` (shipped: `band_r38_a080_c0_fill.npz`).
* Ready-made held-out dumps (READ-ONLY, from the F-sprint): `/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/b/dump_unseen.npz`,
  `.../ckpt/c/twins_unseen.npz`, `.../ckpt/c/twins_all.npz`, `.../ckpt/c/ood_aoa20.npz` (inspect keys with numpy; the scripts that
  wrote them are in `final0927/wt_b/Track_1/probe_b/` and `final0927/wt_c/Track_2/stream_c/`).
* F-sprint infrastructure you may COPY (never edit in place): `final0927/notes/BUILD_RUNBOOK.md` (how to build + gate a package;
  `RPDE_FORCE=1` is mandatory today; always pass an absolute `RPDE_SUB=$K/sub_candidates` and a unique TAG),
  `final0927/notes/baseline_ref.md` (reference reads of the current system), `final0927/jobs/rp_f27x_candref.sh` (one job: fit band,
  pack twins, stream reads), `final0927/jobs/rp_f27x_build.sh`, `final0927/notes/throughput.md`.
* Data: `/users/PAS1187/qwu/RealPDE/train_real/train_real`, `/users/PAS1187/qwu/RealPDE/train_sim/train_sim`.
  Official scorers + ingestion programs: `/users/PAS1187/qwu/RealPDE/official/{t1_v9,t2_v6}/`, kits under `/users/PAS1187/qwu/RealPDE/`.
* Python: `/users/PAS1187/qwu/.conda/envs/realpde/bin/python -u` (torch 2.2.2+cu121 = the eval containers' version).
  Job preamble: copy `/fs/scratch/PAS1187/qwu/tmp/realpde/jobs/common.sh`, set `REPO` to YOUR clone, keep
  `HDF5_USE_FILE_LOCKING=FALSE`.
* **The login/desktop node (a0130) has NO GPU.** CPU-only numpy probes there are fine: set `OMP_NUM_THREADS=8`, keep them
  short, never start thread-heavy things (per-user limit 4096 threads on the node).
* Slurm (one multi-cluster; $HOME and /fs/scratch shared). Account `-A PAS1186`, `--gpus-per-node=1`, `-c 12`,
  `--output` on scratch, pass `--chdir`. Submit with `env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES sbatch ...`.
  **The owner asked us to PREFER CARDINAL** (Ascend already carries ~25 F-sprint jobs):
  * Cardinal H100 (1.67x faster than A100): `-M cardinal -p gpu` (measured 16:30: a 1-GPU 1-h job started in ~30 s; ignore
    the pessimistic `--test-only` estimate). **Per-user cap on Cardinal = 24 GPUs, shared by everybody in both sprints.**
    K-sprint allowance on Cardinal: direction 1 <= 3 GPUs, direction 2 <= 3, direction 3 <= 14 at any one time; overflow goes to
    Ascend. If a Cardinal job is still PENDING after 3 min, `scancel` it and resubmit on Ascend.
    `-M cardinal -p debug` (<= 1 h, 1 GPU per job, max 2 jobs per user) is a fallback for short probes only.
  * Ascend A100: `-M ascend -p nextgen` (measured 16:30: starts within seconds; per-user cap 96 GPUs, the F-sprint holds
    ~25; K-sprint keeps <= 20 there so the F-sprint's build/eval jobs are never starved). Must exclude the desktop node:
    `DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)`;
    `sbatch ${DESK:+--exclude="$DESK"} ...`.
  * Do NOT use Pitzer. Wait on jobs with `squeue`/`sacct` polling + log tails (poll every 20-30 s), not blind sleeps.
* Latency measurements: only on a node where no other job of ours runs (`--exclusive` is allowed for a <= 20 min timing job),
  interleaved arms (A B A B ...), report both kernel-level and package-level reads. Board transfer of a local latency delta is
  unreliable (FLOPs cuts transfer within ~2x; host-side launch accounting did not transfer) — say so in your bracket.
* GPU budget: <= 100 GPU-hours per direction.

## 4. Instruments (every local number is a sign, not a size)
* Operating point: `--alpha 0.80 --band-p 0.45`, inflate/stretch 1.39 (board-measured 1.29: report both), centre0 band.
  Instrument argparse defaults differ — pass them explicitly.
* Ensemble-level screen: `Track_1/ens_v9.py --aoa 15 20` and the full split; single checkpoints: `Track_1/compare_v9.py`;
  band: `Track_1/submit_fno/fit_band.py`; T2 stream: `Track_2/stream_eval.py` (+ `--causal-check`);
  OOD: `Track_1/aoa_holdout_eval.py`. Scorer reconstruction: `Track_1/sps9.py`, kit `scoring.py`.
* History: the proxy over-reads error cuts ~2x; "tighter band" changes over-read or flip on the board; board error is ~1.29x
  the local val error; a +1.15 ms extra invocation costs about -0.08 on T1. Report honestly with brackets.

## 5. Build + gates
`Track_1/submit_fno/build_round13.sh` builds both tracks from members (`RPDE_A`), twins (`RPDE_M`), a band and `RPDE_SHIP_*`
knobs (see BUILD_RUNBOOK.md for the exact knob sets of today's baseline packages "27lz" (T1) and "asnp" (T2)). Gates:
`submit_fno/validate_strict.py` (T1) and `Track_2/validate_t2.py [--adaptive]` (T2) must print SUBMITTABLE. Archive cap
256 MB; eval containers: torch 2.2.2, no scipy/h5py/pandas/matplotlib. Run a build only inside your own clone (it seds
constants into submission.py), one build per clone at a time.
Baseline zips already built today by the F-sprint (same knobs as the planned uploads): `final0927/sub_candidates/track{1,2}_x27_a080.zip`
and `track{1,2}_x27_d1c_a080.zip`.

## 6. Deliverables
* Design agents: `$K/notes/design_<dir>.md` = ranked tricks · for each: hypothesis, which subscore moves and why, evidence from
  your own quick probes, predicted delta-final with bracket, legality check against section 2, kill criterion · exact executor
  task specs (commands, files, job scripts) · job IDs already launched.
* Executors: `$K/notes/result_<dir><n>.md` = held-out reads paired against the baseline (AoA 15-20 and full split), package
  path + sha256 + gate lines if built, predicted five subscores with a bracket, and an honest verdict (SHIP / NO-SHIP).
