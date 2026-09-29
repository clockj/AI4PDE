# AI4PDE — RealPDE (NeurIPS 2026)

Forecast real PIV flow past a NACA4418 airfoil: 20 frames of velocity on a 32×64 grid → the next 20, with a lower and an
upper bound per element. CFD of the same airfoil is a training-side asset only.
[Site](https://realpdecompetition.github.io/) · [Track 1 Sim2Real](https://www.codabench.org/competitions/17363/) ·
[Track 2 LTTTA](https://www.codabench.org/competitions/17385/) · remote `github.com/clockj/AI4PDE` · branches `main`
(release) and `qwu/dev` (working; merges to `main` go through a pull request).

@STATUS.md

This file holds what stays true. State lives in `STATUS.md`, history in `ROUNDS.md`, procedures in `.claude/skills/`,
and rules that only matter inside one directory in `.claude/rules/` (loaded when a file there is read).

## Where things are

| need | read |
|---|---|
| re-create the method from the release | `REPRODUCE.md`, `reproduce.sh` (default target `final`; release tag `final-20260927`) |
| how local evaluation works and how the board corrects it | `EVALUATION.md` |
| out-of-distribution reads (whole AoA or Re band held out, real package path) | `experiments/decision_20260928/ood/README.md` |
| the Decision-phase campaign: criteria, every table, decisions | `experiments/decision_20260928/` (`PLAN.md`, `STATE.md`) |
| what is closed, with the evidence | `ROADMAP.md` Part 2 — **check it before proposing an experiment** |
| what a round did and why | `ROUNDS.md` (append-only, indexed at the top) |
| every scored round, and the fitted `final` rule | `Track_1/board_rounds.py`, `Track_1/final_v9.py` (print `W0`; never quote weights from prose) |
| experiments: scripts, criteria, verdicts | `experiments/` (campaigns), `Track_1/closed/`, `Track_2/closed/`, `Track_1/archive/` |
| uploads and their registered predictions | `../sub_candidates/` (`final/`, `MANIFEST.md`, `HISTORY.md`) |
| the newest dated handoff | `../handoffs/` — it wins over a repo doc only where it is newer than that doc's last commit |

## Rules

1. **The user uploads; agents never do.** The Decision phase (to 2026-10-25) allows **one submission per track in
   total**. Never stage an upload without the owner's explicit go.
2. **Predict before upload.** All five subscores, each with a bracket, go into `../sub_candidates/<date>/README.md` and a
   PENDING row of `leaderboard/board.jsonl` before the upload. A prediction written afterwards is worthless.
3. **Evidence comes only from held-out twins.** All-data checkpoints supply weights, never numbers, and never the band's
   error field (that field is the twins' error on `val_seen` AoA 15–20, which they trained on, ×1.39, `--fit-on seen`;
   SIGBLEND's constants are out of sample). Derive held-out lists in code (`re_split.excluded()`, `kfold.py --folds`);
   never type an `--exclude`.
4. **Track 1: each window is forecast from its own input only.** The windows tile (window *k*'s target is window
   *k*+1's input), so any statistic pooled across the windows of one `predict()` call contains targets, even when it
   passes a permutation check. The disclosed exploit (`Track_1/archive/submit_trick/`) is never shipped, and nothing
   that needs history belongs on Track 1.
5. **Track 2: strictly causal.** `ttt_step` may use the current input and targets already revealed, of the current
   trajectory only; all state is cleared in `reset_ttt_state`; returned tensors belong to the caller. Every Track-2
   package passes `validate_t2.py --adaptive` and `stream_eval.py --causal-check`.
6. **Nothing external.** No external data, no new simulations, no outside pretrained weights.
7. **Everything that ships is re-creatable from the release.** An irreproducible method is disqualified. A change to
   members, band or a baked constant is not done until `reproduce.sh` and `build_final.sh` produce it and its band is in
   `Track_1/bands/` with a `MANIFEST.md` entry.
8. **Only baked constants ship.** `submission.py` in the tree holds defaults (ALPHA 0.60, every stream knob off); the
   shipped values are the `RPDE_SHIP_*` list in `Track_1/submit_fno/build_final.sh`. A knob turned off must reproduce
   the previous output bit for bit.
9. **Compare packages by board-subscore deltas, never by `final`.** One controlled lever per slot. Never resubmit for a
   timing draw. HOLD is a valid decision.
10. **No generated trees, caches or bulk data under `$HOME`** (1 M-inode quota). Jobs, logs, worktrees and scratch go to
    `/fs/scratch/PAS1187/qwu/tmp/realpde/` or `/tmp`. `Track_1/unet/*.pth` and `../sub_candidates/` are the exceptions.
11. **Never touch other projects' Slurm jobs** (`ondemand/*`, `ttf_*`, `hd_*`, `p5_*`, `p6_*`, `tt26-*`, `m2_*`). Ours are
    `rp_*` (Ascend) and `pz_*` (Pitzer). Every `sbatch` excludes the owner's interactive desktop node.
12. **Commits:** on `qwu/dev`, when a piece of work is done, at most 10 a day and never split without need; never pushed
    by an agent. Never commit checkpoints, `val_*.npz` or archives. Commit before building: a build edits tracked files.

## The problem and the contracts

Both tracks score the same hidden windows with the same five subscores. Channels are `[u, v, p]`; `p` is zero-filled and
unscored. Neither track tells the model Re or AoA.

* **Track 1** calls `predict(input_array, metadata=None)` with `(N, 20, 32, 64, 3)` in **raw physical units** and times
  all of it; the evaluator sends 14–99 windows per call. Return `{"prediction", "lower", "upper"}` of the same shape.
  Normalize and denormalize yourself.
* **Track 2** calls `get_ttt_model(submission_dir, device)` once (untimed), then `reset_ttt_state()` per trajectory and
  `ttt_step(input_norm, prev_target_norm) → (prediction, info)` per window at batch 1, **already normalized**, 10 min in
  all. `info["adapt_loss"]` is required; bounds go in `info["lower"]` / `info["upper"]`.
* **The revealed target is normalised with the TARGET statistics, the input with the input statistics.** They differ in
  the fifth decimal, which is enough to break an exact-zero test.
* **Malformed bounds (shape, non-finite, `lower > upper`) zero every subscore.**
* **Containers:** Track 1 `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`, Track 2 `w3nhao/realpde-track2:v1`. Neither has
  `scipy`, `h5py`, `matplotlib` or `pandas`, and nothing is installed at evaluation time. An archive holds exactly
  `submission.py`, `_fno.py`, `band.npz` and the members' `.pth`, under 256 MB.

## Data

Under `/users/PAS1187/qwu/RealPDE/` (not in git); mirror `AI4Science-WestlakeU/RealPDE-Competition-Data`.

* `train_real/train_real/{Re}_{AoA}.h5`: 82 trajectories (18 Re × 5 AoA, 8 cells missing), `u`, `v` as top-level datasets,
  `(T, 64, 128)` float64, T = 868 for 79 of them (607, 492 and 282 for the other three).
* `train_sim/train_sim/{Re}_{AoA}.h5`: 100 trajectories, 1000 frames, `u, v, p`, **non-dimensional**; real data is in m/s.
  Never mix raw arrays: `Track_1/unet/simreal.py` maps sim into the real measurement space.
* Windows: subsample `[::2, ::2]`, 20 in + 20 out, stride 20, tail padded by repeating the last frame; 42 windows per
  full trajectory. Held-out trajectories: `model.VAL_TRAJ` (8), the authority for every split.
* About 10 % of `u, v` targets are exactly 0 (outside the field of view, the airfoil's shadow), 3–16 % of pixels in some
  frames only. Only the interval average and the coverage of `sps` skip them; `rel`, `tke`, `mvpe` and `sps`'s accuracy
  factor are computed without a mask, zeros included (`scoring.py`: `rel_l2_per_sample`, `aggregate_sps`).

## Scoring

`rel`, `tke`, `mvpe` are `100/(1 + 0.5·error)`; `time` is `100/(1 + sqrt(t/0.72896))`, t in seconds per window or step;
`sps = 100 · ā · C`, the accuracy factor times containment. **The rule combining them into `final` is unpublished and is
fitted** from our own board rounds (`final_v9.py`). Only `time` is random for a given archive. No local instrument reads
containment as the board does: state every band change in ΔC as well as in `final`.

## What ships

Six members, fused by family into two invocations → mean field y₀ and mean σ head → compose at ALPHA with one gain for
`u` and `v` → prediction P. Interval: centre y₀, half-width from `band.npz` (SIGBLEND), widened to contain P. Track 2 runs
the same path inside one CUDA graph and corrects the interval width, the time-mean bias and the fluctuation amplitude
from revealed targets. `Track_1/submit_fno/build_final.sh` is the authority for every constant. Release: tag
`final-20260927` (4e5ed37); for archives byte-identical to the uploads, code included, build from `fdac00b`.
**The Decision-phase candidate (not uploaded)** trains all six members and five twins with `--aug-resim 0.25` and adds
`PREV_SPACE = "target"`: `RPDE_REPRODUCE_GO=1 bash reproduce.sh decision` (`REPRODUCE.md` §14; `STATUS.md`).

## Traps

* **`train2.py`'s argparse defaults are not the recipe** (`--alpha-train 1.0` collapses the model). Recipes are in
  `reproduce.sh` and in each checkpoint's `torch.load(p)["config"]`.
* **Never train through `compose`**; compose at inference only.
* **A SIGBLEND band needs its sidecar** (`Track_1/sigblend/*.json`). Read without it the band is about 4× too wide and
  nothing fails. Load bands with `Track_1/band_io.py`.
* **Score the real code path**, never a re-implementation of it.
* **Running jobs re-import the working tree.** Edit shared modules in a scratch worktree; never edit `_fno.py`,
  `submission.py` or a build script while a build runs.
* **torch ≥ 2.6 refuses the checkpoints** unless `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (the scripts export it).
* **If an arm scores worse than the worst case of its own hypothesis, suspect the harness.**
* **One build per tree.** Two builds in one tree overwrote each other's baked constants and still ended `BUILD_FINAL OK`;
  `build_final.sh` now takes a lock file and checks the archive's constants (`Track_1/verify/baked_constants.py`).
* **`sbatch --test-only` is not evidence that Ascend is busy** (it said 24 h; jobs started in seconds). Probe with a job.
* **`STATS_MASK` stays off.** The scorer counts zero targets in rel, tke and mvpe, and unmasked D1c and VARCAL learn a
  recording's own blank; masked, a re-blanked recording lost on every trajectory.

## Environment

* **OSC.** Ascend (A100): `-A PAS1186 -M ascend -p nextgen --gpus-per-node=1`, first whenever it is idle, training
  included. Pitzer (V100): the overflow. Login nodes have no GPU. `--output` on scratch. Templates: the `osc-jobs` skill.
* Python: `~/.conda/envs/realpde/bin/python` (torch 2.2.2, numpy 1.26.4 — the evaluators' stack); `environment.yml`.
* Repo `~/workspace/realpde/main` · data and kits `~/RealPDE/` · scratch `/fs/scratch/PAS1187/qwu/tmp/realpde/`.
* User-facing summaries are written in Chinese; repository documents in English.

## Dates

Main Development closed 2026-09-27. Decision phase to 2026-10-25. Results 2026-11-10. Workshop 2026-12-06; winners
open-source two weeks before presenting.

## Repo map

```
CLAUDE.md  STATUS.md  README.md  REPRODUCE.md  reproduce.sh  EVALUATION.md  ROADMAP.md  ROUNDS.md  environment.yml
.claude/     rules/ (path-scoped)   skills/ (build-package, record-board-result, reproduce-from-release,
             osc-jobs, evaluate-change)
Track_1/     instruments at top level: predict_board, compare_v9, ens_v9, kfold, aoa_holdout_eval, ...
             opoint.py (the operating point), band_io.py (band loader), sps9.py, board_rounds.py, final_v9.py
  unet/        model.py, models_alt.py, train2.py, simreal.py; *.pth (untracked)
  submit_fno/  submission.py, _fno.py, fit_band.py, build_zip.py, validate_strict.py, build_round13.sh, build_final.sh
  sigblend/    the SIGBLEND band recipe and its constants      bands/   scored bands + MANIFEST.md
  verify/      gates, identity checks, baked_constants.py      qd_k/    the QD error-head derivation (not shipped)
  closed/      closed experiments, one verdict each            archive/ the first two months
  plan/ report/  dated plans and write-ups
Track_2/     submit/submission.py, build_t2.py, validate_t2.py, stream_eval.py, t2_common.py, r1_ref/, closed/
experiments/ cross-track campaigns (sprint_20260927/, audit_20260927/, decision_20260928/ with ood/ the instrument)
leaderboard/ board.jsonl, runs.jsonl, screen.jsonl, lb.py
outside the repo: ../sub_candidates/ · ../handoffs/ · ../artifacts/
```
