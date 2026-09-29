# State of the Decision-phase campaign

Plan and criteria: `PLAN.md`. Update this file at every wave; keep job ids, paths and the GPU-hour ledger here.

**Closed 2026-09-29.** The owner's decisions and the closing steps are at the end of this file. Scratch paths below
(`R/…`) are as they were when each section was written; after the cleanup of 09-29 what remains is listed in
`/fs/scratch/PAS1187/qwu/tmp/realpde/README.md`, and what was deleted in `CLEANUP_20260929.md` beside it.

## Ledger

| date | wave | GPU-h used (sacct) | cumulative | of 300 |
|---|---|---|---|---|
| 2026-09-28 | opened | 0 | 0 | 0 % |
| 2026-09-28 05:15 | wave 1 moved to Ascend; smokes and the cancelled Pitzer start | 1.4 (Pitzer 1.1 cancelled, smokes 0.3) | 1.4 | 0.5 % |

| 2026-09-28 10:57 | every training finished (24 wave 1, 8 T3, 4 T4), instrument chunks 1 – 4 so far | wave 1 60.3 · T3 20.9 · T4 10.6 · instrument 5.7 · other 0.2 | **97.7** | 33 % |
| 2026-09-28 15:08 | wave 3 (T4 on A20 / A10, twins, T4v on RE done; T4b, T4v second stage running) and instrument chunks 4 – 5 | sacct over `rp_dp_*` / `pz_dp_*`, running jobs counted to 15:08 | **156.5** | 52 % |
| 2026-09-28 18:02 | every training of wave 3 finished (24 jobs COMPLETED); instrument chunk 5 | sacct, 170 GPU jobs | **171.6** | 57 % |
| 2026-09-28 22:38 | the from-scratch run of the candidate (41.8) and the last instrument reads | sacct, 185 GPU jobs | **213.7** | 71 % |
| 2026-09-28 23:03 | the closing gates (R1, R1b, the second candidate, R2) | sacct, 199 GPU jobs | **214.4** | 71 % |
| 2026-09-29 00:45 | the identity rebuilds from the committed bands; campaign closed | sacct, 201 GPU jobs | **214.5** | 72 % |

Estimates made at 05:15 (A100): wave 1 about 60, arms T3 + T4(RE) about 31; both held. Read the ledger with
`sacct -M ascend,pitzer -X -n -S 2026-09-28 -o jobname%26,elapsed,state,alloctres%40 | grep -E 'rp_dp_|pz_dp_'`.

## Wave 1 — instrument

Submitted 2026-09-28 by forge; running on Ascend since 05:07 EDT. Root `/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/` (`R` below).

* **Code:** worktree `R/wt_base`, detached at **`4e5ed37725b53ff975ae20c7b453fc94a8aa7f49`** (HEAD of `qwu/dev`); every
  job checks the hash and refuses otherwise. `R/wt_base/Track_1/val_{relo,rehi,rein}.npz` copied from the working tree
  (trajectory sets checked equal to `re_split.names_for`, 42 windows each).
* **Recipes:** `reproduce.sh` fragments `FFNO SIMF VITC VITP` inlined verbatim by `R/jobs/gen_wave1.py`; no `--all-data`;
  `--exclude <list> --ckpt-every 20 --resume`, Slurm `--requeue`. Scripts source nothing.
* **Exclusions** (`R/jobs/derive_exclude.py` -> `R/ckpt/<REG>/exclude.txt`, in code from `model.VAL_TRAJ`, file names,
  `re_split.excluded()`):

| regime | held-out stems | names excluded | real left for training | sim left (`--sim-exclude-held`) | sim-map cache |
|---|---|---|---|---|---|
| A20 | 16 real + 20 sim at AoA 20, + 8 VAL_TRAJ | 26 | 60 of 82 (9833 windows) | 74 | `simmap_p1_f0_x3363c136b1` (existed, 09-19) |
| A10 | 18 real + 20 sim at AoA 10, + 8 VAL_TRAJ | 26 | 58 (9384 windows) | 74 | `simmap_p1_f0_x6b8c7e1d61` (existed) |
| RE | 26 real (`re_split.excluded()`) + 30 sim at the 6 Re, + 8 VAL_TRAJ | 36 | 50 (8056 windows) | 64 | `simmap_p1_f0_x1ff76a3af6` (built once by the smoke) |

* **Reused (copied, byte-identical):** `screen/aoa/aoa20_ffnosim_s1,s2.pth` -> `R/ckpt/A20/a20_ffnosim_s1,s2.pth`,
  `aoa10_ffnosim_s1.pth` -> `R/ckpt/A10/a10_ffnosim_s1.pth`. Evidence: stored `config` equals every flag of
  `FFNO SIMF --sim-exclude-held` (epochs 400, seeds as named, `all_data` False), and the sorted `exclude` equals the derived
  list; logs end at epoch 399 + `saved`. Trained from `d7492c7-dirty` (s1s) and `cddecd9-dirty` (A20 s2), not 4e5ed37.
  Their single-member reads are re-run with today's defaults by `rp_dp_read_reused`.
* **Smoke** `pz_dp_smoke` 54995736 (Pitzer gpudebug-exp, V100S-32GB, RE, 2 epochs, into `R/smoke/`): all three recipes
  parse and save; log shows `50 trajectory-phases -> 8056 windows, 36 held out`; the RE cache was built once here;
  `re_split.py` and `aoa_holdout_eval.py` both run (33 s and 115 s). Steady epoch, **V100**: F-FNO+sim 27 s, ViT-C+sim
  29 s, ViT-P 15 s.
* **Cluster.** `sbatch --test-only -M ascend -p nextgen -c 8 --gpus-per-node=1 --exclude=a0130` printed
  `Job 7571453 to start at 2026-09-29T04:31:16` (04:54 EDT) and `Job 7571490 ... 2026-09-29T04:40:29` (05:03 EDT) for
  -t 1:00 to 8:00, so the wave first went to Pitzer (pz_dp_*, ids 54995916-54995933 and 54995937-54995943, submitted 05:03).
  **That estimate was wrong:** real `nextgen` jobs start in seconds (instrument executor's 7571463 / 7571494: 21 s and
  12 s; the probe 7571497: submitted 05:06:47, started 05:06:51; the other 17 started within about 10 s). On the owner's
  rule (Ascend first whenever it is idle) the Pitzer jobs were **cancelled after 3 min 47 s (18 running, 7 never started),
  moved to Ascend**; cost about 1.1 V100 GPU-h. Their partial output (18 training logs, no `.resume.pt` or `.pth` yet)
  was deleted so `--resume` cannot pick up a V100 fragment. Their scripts and ids are kept in `R/jobs/pz_dp_*.sbatch`,
  `R/jobs/submitted_pitzer_cancelled.tsv`.
* **Measured on Ascend, A100-PCIE-40GB, seconds per epoch (epochs 0 to 5):** F-FNO+sim A20 25.4, A10 24.2, RE 20.8;
  ViT-C+sim A20 25.4, A10 24.0, RE 20.6; ViT-P A20 13.2, A10 12.7, RE 10.8. (The F-FNO A100/V100 ratio is about 1.3.)
* **Extrapolated, A100:** 400 epochs = F-FNO+sim 2.8 / 2.7 / 2.3 h (A20 / A10 / RE), ViT-C+sim the same, ViT-P 1.5 /
  1.4 / 1.2 h. **24 trainings about 58.3 GPU-h, plus reads about 1.3, = about 60 A100 GPU-h**, plus about 1.3 already
  spent (Pitzer 1.1 cancelled, smoke 0.2). **Last job expected to end around 11:00 EDT 2026-09-28** (the chain
  a20_ffnosim_s3 then a20_vitc_s1, about 5.7 h from 05:08).
* **Jobs** (all Ascend `nextgen`, `-c 8`, 1 A100, `--requeue`, desktop node a0130 excluded; F-FNO -t 6:00:00, ViT
  8:00:00, read 1:30:00; scripts `R/jobs/<name>.sbatch`, ids in `R/jobs/submitted.tsv`; checkpoint
  `R/ckpt/<REG>/<reg>_<member>.pth`, training log `.pth.log`, read `.eval.txt` + `.json` next to it; Slurm log
  `R/logs/<name>_<id>.log`; sentinel `R/logs/<name>.done`; last line `RP_DP_END`). 18 leaders run; 7 followers wait on
  `afterany` of one leader each, so at most 18 of these run at once.

| job | id | dependency |
|---|---|---|
| rp_dp_re_ffnosim_s5 (the probe) | 7571497 | - |
| rp_dp_a20_ffnosim_s3 / s4 / s5 / s6 | 7571499 / 500 / 501 / 502 | - |
| rp_dp_a20_vitp_s1 | 7571503 | - |
| rp_dp_a20_vitc_s3 | 7571504 | - |
| rp_dp_a10_ffnosim_s2 / s3 / s4 / s5 | 7571505 / 506 / 507 / 508 | - |
| rp_dp_a10_vitp_s1 | 7571509 | - |
| rp_dp_a10_vitc_s3 | 7571510 | - |
| rp_dp_re_ffnosim_s1 / s2 / s3 | 7571511 / 512 / 513 | - |
| rp_dp_re_vitp_s1 | 7571514 | - |
| rp_dp_re_vitc_s3 | 7571515 | - |
| rp_dp_re_ffnosim_s4 | 7571516 | afterany:7571514 (re vitp) |
| rp_dp_re_ffnosim_s6 | 7571517 | afterany:7571509 (a10 vitp) |
| rp_dp_a10_ffnosim_s6 | 7571518 | afterany:7571497 (re s5) |
| rp_dp_re_vitc_s1 | 7571519 | afterany:7571511 (re s1) |
| rp_dp_a20_vitc_s1 | 7571520 | afterany:7571499 (a20 s3) |
| rp_dp_a10_vitc_s1 | 7571521 | afterany:7571505 (a10 s2) |
| rp_dp_read_reused | 7571522 | afterany:7571503 (a20 vitp) |

Reads: A20/A10 `aoa_holdout_eval.py --holdout-aoa <20|10> --fit-set val_traj --json` (opoint defaults); RE
`re_split.py --ckpt` (it has no json option: stdout to `.eval.txt`, parsed into `.json` by the job). A timeout is not
requeued by Slurm; resubmitting the same script resumes from `.resume.pt`.

Check: `squeue -M ascend -u $USER -o '%.9i %.24j %.9T %.8M %.9l %E' | grep rp_dp_` and
`ls /fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/logs/*.done`; ledger
`sacct -M ascend -X -n -S 2026-09-28 -o jobname%24,elapsed,state | grep rp_dp_` (plus the Pitzer rows above).

## Wave 2 — training arms T3 and T4 (submitted 2026-09-28 05:15 EDT, waiting on wave 1)

Scripts and generator: `R/t4/jobs/` (`gen_t34.py`; `later/` holds T4 for A20 and A10, not submitted). Ids in
`R/t4/jobs/submitted_t34.tsv`. Each job waits on `afterany` of one wave-1 job that has no follower, so the campaign
never runs more than 20 jobs. All on Ascend `nextgen`, `-c 8`, 6:00:00, desktop node excluded.

| arm | what differs from the control | regimes, seeds | runs from | ids |
|---|---|---|---|---|
| T3 | `--lam-sim 1.0` (control 0.3) | RE and A20, seeds 1 – 4 | `R/wt_base` (must be clean at 4e5ed37) | 7571545 – 7571552 |
| T4 | `--aug-resim 0.25` (p 0.5) | RE, seeds 1 – 4 | `R/wt_t4` = 4e5ed37 + `patches/t4_aug_resim.patch` (the job checks the diff's sha1 7a1578a0) | 7571541 – 7571544 |

* T4's implementation and its checks: `results/t4_impl.md`. With the flag absent or at 0 the run is bit-identical to the
  unmodified tree (31 weight tensors, the global CPU random state); with it the augmentation draws from its own
  generator. It costs about 10 % per epoch.
* The generator wrote the read step's outputs to `./.eval.txt` and `./.json` for every job (an empty base name); fixed
  in `gen_t34.py` before anything was submitted, scripts regenerated, `wt_t4`'s diff hash unchanged.
* T4 on A20 and A10 (8 jobs in `later/`) is submitted only if the RE arm clears its single-member screen (`PLAN.md` §5).

## Single-member reads (every training finished 2026-09-28 10:55 EDT, all 36 jobs COMPLETED)

`wfinal` of one checkpoint with its own band, `aoa_holdout_eval.py` (A20, A10) and `re_split.py` (RE); json next to each
checkpoint. **Signs before sizes: the decisions are taken on the ensemble reads of the instrument.**

| | in distribution (fit split) | A20 | A10 | relo | rehi | rein |
|---|---|---|---|---|---|---|
| F-FNO + sim, 6 seeds, mean ± sd | 85.98 ± 0.03 (A20 run) | 81.06 ± 0.08 | 83.90 ± 0.08 | 85.58 ± 0.07 | 80.42 ± 0.04 | 85.14 ± 0.02 |
| ViT-C + sim, s1 / s3 | 85.76 / 85.72 | 79.53 / 79.67 | 83.18 / 83.17 | 79.94 / 81.20 | 79.89 / 79.99 | 84.96 / 84.91 |
| ViT-P, no sim, s1 | 85.54 | 78.30 | 83.05 | 78.51 | 79.50 | 84.70 |
| T3 (`--lam-sim 1.0`) − control, seeds 1 – 4 | +0.022 | **+0.171** (+0.007, +0.289, +0.161, +0.228) | not trained | −0.031 (2/4) | −0.002 (2/4) | −0.022 (1/4) |

* **The ViT family is the weak link out of distribution.** A single ViT is 0.2 – 0.4 behind a single F-FNO in
  distribution, 0.7 – 0.9 behind at A10, 0.4 – 0.9 at rehi, **1.4 – 2.8 at A20 and 4.4 – 7.1 at relo** (rel error 0.146 –
  0.192 against 0.086). Sim co-training is worth +1.2 … +2.7 to a ViT at A20 and relo; the ViT that ships without sim
  is the worst member in every held-out regime. Ensemble layouts `vitc2`, `ffno4`, `ffno6`, `six_vitc2` and FAMW's
  learned weights were added to the instrument's wave to price it.
* **The seed spread of F-FNO + sim is small out of distribution too** (0.04 – 0.08), not the 12 – 58× of `ROADMAP.md`
  §1.12, which was measured on the unseeded FNO w24 of August. Arm T1 has less to average away than its premise said.
* **Low-Re extrapolation is not hard for F-FNO + sim** (relo 85.58, above rein's 85.14); high-Re extrapolation is
  (rehi 80.42, rel error 0.112, sps 37).
* **T3 cannot pass as registered**: its criterion needs the mean of relo and rehi at +0.10 and the single-member deltas
  there are null, inside the seed spread. At A20 it moves a member by two seed spreads (+0.17, 4/4). The ensemble read at
  A20 is taken for the record.
* **T4's own read step failed** in all four jobs: `wt_t4` has no `val_{relo,rehi,rein}.npz` (they were copied into
  `wt_base` only). The checkpoints are complete; job 7572732 (`rp_dp_t4_read`) redoes the four reads from `wt_base`,
  whose `re_split.py` is identical.

### T4, the single-member screen on RE (read by `rp_dp_t4_read` 7572732, seed-matched to the control)

| set | T4 (`--aug-resim 0.25`), seeds 1 – 4 | delta to the control | rel error, T4 / control | sps, T4 / control |
|---|---|---|---|---|
| rein | 85.245 85.265 85.255 85.241 | **+0.120** (+0.136 +0.118 +0.120 +0.108; 4/4) | 0.0693 / 0.0706 | 50.57 / 50.30 |
| relo | 86.115 86.110 86.124 86.097 | **+0.523** (+0.622 +0.450 +0.571 +0.451; 4/4) | 0.0781 / 0.0863 | 54.97 / 53.93 |
| rehi | 80.660 80.649 80.613 80.595 | **+0.208** (+0.210 +0.261 +0.179 +0.181; 4/4) | 0.1100 / 0.1123 | 37.42 / 36.93 |

The control's seed spread is 0.02 / 0.08 / 0.03, so every delta is five to seven spreads. The interior control moves
too, which no training-side arm since sim co-training had done. **The screen is cleared; this is a single member with
its own band, not the ensemble read the criterion asks for.**

### Wave 3 — T4's continuation and extensions (submitted 2026-09-28 11:04 EDT; ids in `R/t4/jobs/submitted_t4x.tsv`)

Registered in `PLAN.md` at 11:03 (commit 43322eb, one minute after the T4 read job ended at 11:01), before submission.
Clock times in this file are read from `git log`, `sacct` and file stamps, not from memory. Generator `R/t4/jobs/gen_t4x.py`; every recipe
was checked flag by flag against its control's job script, the only difference being `--aug-resim`. All from `R/wt_t4`.

| arm | runs | ids | estimate (A100) |
|---|---|---|---|
| T4 on A20 and A10, F-FNO + sim, seeds 1 – 4 | 8 | 7572995 – 7573003 | 25 GPU-h |
| T4 in-distribution twins (hold out `VAL_TRAJ`), seeds 1 – 3 | 3 | 7573007 – 7573009 | 11 |
| T4v, ViT-C s3, s1 and ViT-P s1 with the augmentation, RE | 3 | 7573004 – 7573006 | 6 |
| T4b, `--aug-resim 0.5`, RE seeds 1 – 4 (each behind one leader) | 4 | 7573010 – 7573013 | 10 |

### T4v, the single-member screen on RE (finished 13:44 EDT; seed-matched to the control ViTs)

| member | rein | relo | rehi | delta rein / relo / rehi | mean of relo, rehi | rel error at relo, arm / control |
|---|---|---|---|---|---|---|
| ViT-P s1 + `--aug-resim 0.25` | 84.758 | 85.227 | 79.899 | +0.060 / **+6.719** / +0.404 | +3.562 | 0.0859 / 0.1921 |
| ViT-C s3 + the same | 84.989 | 85.738 | 80.082 | +0.079 / **+4.541** / +0.092 | +2.316 | 0.0803 / 0.1456 |
| ViT-C s1 + the same | 85.062 | 85.501 | 80.281 | +0.107 / **+5.560** / +0.396 | +2.978 | 0.0851 / 0.1648 |

**The screen is cleared** (bar: +0.20 on the mean of relo and rehi, rein ≥ −0.05). With the augmentation a ViT at low Re
is where an F-FNO is (85.2 – 85.7 against 85.6 for the control F-FNO and 86.1 for the augmented one): the family's
weakness below the training range was the amplitude it had never seen. At rehi its rel error does not improve
(0.1164 – 0.1202 against 0.1156 – 0.1174); the gain there is in the other subscores.

**Second stage submitted 13:45 EDT** (`R/t4/jobs/gen_t4y.py`, ids in `R/t4/jobs/submitted_t4y.tsv`, each behind one
running T4 job): the mirror's two ViTs with the augmentation on A20 and A10 (7575504 – 7575507) and the two ViT twins on
`VAL_TRAJ` (7575508, 7575509; recipes checked against the stored config of `arch_vit_d384` and `arch_vit_cmb3sim`).
About 15 GPU-h.

### T4b, the dose (`--aug-resim 0.5` against 0.25), single-member screen on RE (read 16:57 EDT)

| seed | rein | relo | rehi | mean of relo, rehi |
|---|---|---|---|---|
| 1 | +0.010 | +0.094 | +0.004 | +0.049 |
| 2 | +0.002 | +0.110 | −0.070 | +0.020 |
| 3 | +0.000 | +0.073 | −0.066 | +0.004 |
| 4 | +0.008 | +0.103 | −0.040 | +0.032 |
| mean | +0.005 | +0.095 | −0.043 | **+0.026** |

**T4b fails its criterion** (+0.05 on the mean of relo and rehi with 4 of 4 seeds; none reaches it). The wider range
buys low Re (+0.095, 4/4) and pays at high Re (−0.043, 3 of 4 negative). **0.25 stands.** No ensemble read is taken.

Single-member reads of wave 3 on the held-out angles, against the seed-matched control: F-FNO with the augmentation
A20 +0.152 (+0.095 +0.213 +0.163 +0.137), A10 +0.259 (+0.219 +0.289 +0.349 +0.178), fit split +0.12 in both; ViT-P with
the augmentation A20 −0.061, A10 +0.113, fit split +0.097 in both (one seed).

## Instrument (gauge)

Code in `ood/` (`README.md` has the commands), generated files in `R/ood/`, note `results/ood_instrument.md`. One run =
one Ascend job of 3 – 7 minutes: member check, band refit on `VAL_TRAJ` minus the regime, `build_final.sh` with the
ensemble as members and twins, variants by rewriting the baked constants of the extracted package, streams through
`stream_eval.run` with the revealed target normalised as the runner does, causal check, report. 0.35 GPU-h so far.

**Validated 2026-09-28 on what existed** (composite, trajectories with a positive delta in brackets):

| read | in distribution, 8 twins' trajectories | AoA 20 stand-in (3 members) | AoA 10 stand-in (4 members) |
|---|---|---|---|
| stream stack − asymmetric R1 alone | +0.0757 (8/8) | +0.453 (16/16) | +0.567 (18/18) |
| the same, first 3 / first 14 windows | — | +0.067 / +0.305 | +0.091 / +0.407 |
| `PREV_SPACE` target − input | +0.0064 (8/8) | +0.0023 (16/16) | +0.0064 (18/18) |
| `STATS_MASK` on − off | −0.0012 (4/8) | −0.0006 (2/16) | −0.0001 (11/18) |
| FAMW (`FW_N0` 20) on − off | −0.0007 (2/8) | +0.0088 (13/16) | +0.0141 (16/18) |

* The in-distribution row equals `audit_20260927/refactor_pkg.md` to the printed digits; the held-out-angle rows are
  within 0.015 of the sprint's CPU probes (+0.446, +0.553). Causal check: largest difference 0 in every run.
* **These are not the reads of gate G0 or of any arm**: the stand-ins have 3 – 4 single-seed members and FNO w96 as the
  second family. The decisions wait for the mirror ensembles of wave 1.
* FAMW on top of the full stack reads +0.01 on the stand-ins, against the +0.15 it read on symmetric R1 without D1c.
  It also breaks the first-window identity (its mix equals the mean algebraically, not in floating point).
* A20's static band rests on one fit trajectory (`--fit-aoa 15 20` within `VAL_TRAJ` minus AoA 20); A10 and RE on three.

**Hazard found by the instrument, open for the closing gates.** Two jobs on different nodes built in one scratch
worktree behind a `flock`; `flock` does not serialise across nodes on scratch. One Track-2 archive was built from the
unsubstituted defaults (`r1 False … sigpow None` in the gate's load line) **and the build still ended
`BUILD_FINAL OK`**. Every build of this campaign therefore runs in its own tree (`git archive` of the commit) and checks
the rewritten files before and after. `build_final.sh` itself does not verify that the constants it baked are in the
archive it gated: to be fixed before the closing gates.

### Chunk 2 of the instrument (dry runs on the stand-ins, 0.78 GPU-h; `results/ood_chunk2.md`)

* **The shipped static band is fitted in sample.** `build_round13.sh:105` calls `fit_band.py` without `--fit-on`, whose
  default is `seen` with `--fit-aoa 15 20`: the error field is the twins' error on the three AoA 15 – 20 trajectories of
  `val_seen` (10125_20, 12675_20, 22875_15; 126 windows), **which the twins trained on**, times 1.39. `MANIFEST.md`
  records the flag; `CLAUDE.md` rule 3 and the method report read as if the field were held out. The SIGBLEND constants
  are fitted out of sample (8 trajectories, 336 windows). Measured on the stand-ins, a band fitted the shipping way and
  one fitted out of sample differ by 0.002 – 0.012 composite, so this is a fact to state correctly, not a lever.
* **Arm I2 (`STATS_MASK`) fails, and the mechanism reverses the audit's recommendation.** Plain data: −0.0012 / −0.0006
  / −0.0001. Under a synthetic re-blank (border band, dilated zero set, near-wake rectangle) it loses on every
  trajectory: in distribution −4.77 / −0.10 / −2.40, A20 −0.85 / −0.02 / −0.60, A10 −1.12 / −0.11 / −2.34. Containment
  does not move; the loss is in rel, tke and mvpe, which the scorer takes over the whole field. Unmasked, D1c and
  VARCAL learn the recording's own blanked region from the revealed targets and pull the prediction to zero there;
  the mask switches that off. **`STATS_MASK` stays off.**
* **Arm I1 holds on every read so far**, re-blanked streams included: `PREV_SPACE = "input"` loses on every trajectory
  (−0.002 … −0.012).
* **Arm I5, dry run:** every constant sits on a plateau within ±0.01 of the shipped point, except `VC_N0`, which wants
  to be lower out of distribution (A10 +0.007 at 15) and higher in distribution (−0.021 at 15, 0/8). Decision on the
  mirror ensembles, leave-one-regime-out; nothing here reaches the +0.03 bar.
* **Arm I4, bound:** if windows with at most two revealed targets had the gain of windows 8 – 13, the full stream would
  gain +0.026 (A20) and +0.035 (A10); on 14-window streams +0.075 and +0.104. That is the whole early deficit, of which
  an interval prior can recover only the interval's part. Nothing is built for I4 unless the mirror ensembles' bound is
  clearly above +0.03.
* **The zero set is not static**: 3 – 16 % of pixels are zero in some frames only (mean 8.5 %), against 1.2 – 6 % that
  are always zero. Arm I7 (added to `PLAN.md` before any read) asks what the members predict at blanked elements.

### Chunk 3: the screen of arm I7, zero propagation (0.30 GPU-h; `results/ood_i7_screen.md`)

Post hoc on dumps of the real package path (they reproduce the stream records exactly), rescored through the kit.

* **On the real recordings there is nothing to remove.** Where the input is zero in all 20 frames the members already
  predict 0.0001 – 0.0004 m/s against a flow of about 0.15 m/s; elements whose target is zero carry 0.01 – 0.06 % of the
  squared error. Rule Z1 (zero where the input is zero in all 20 frames) reads **+0.0001 / +0.0002 / +0.0001** (in
  distribution, A20, A10; 7/8, 16/16, 18/18), against a bar of +0.03. **I7 is closed as a score lever.**
* **Under a synthetic re-blank it is large.** Z1 against the control on Track 2: in distribution +0.878 / +0.020 / +0.574
  (border band, dilated zero set, near-wake rectangle), A20 +0.156 / +0.006 / +0.288, A10 +0.150 / +0.022 / +0.475,
  every trajectory positive; on the frozen package (Track 1's case) +0.8 … +5.5. The members have learned where the
  training recordings are blanked, not the rule; the stream stack relearns a new blank in about ten windows and
  removes three quarters of its cost, Z1 removes it from the first window.
* Rules that read only the last input frames lose in distribution (−0.011 at one frame, 0/8): transient zeros.
  Zeroing only part of the output window creates a spurious fluctuation (tke −0.1 … −1.4).
* **What is left is a question of insurance, not of score**: the same premise as the band fill that ships
  (`--fill-unscored`, dev-board price −0.0016). Z1's measured price on our data is zero. Its cost is a new branch in the
  one package that can be uploaded. Not built; the owner decides.

### Chunk 4: the ensemble reads on the mirror ensembles (6.42 GPU-h; `results/ood_chunk4.md`)

Real package path, band refitted to every ensemble, full stream stack unless "frozen"; composite, trajectories with a
positive delta in brackets. Every build passed both gates and the causal check.

**Gate G0 passes.** The shipped stream stack against asymmetric R1 alone, at the member count that ships: **A20 +0.692
(16/16), A10 +0.539 (18/18)**; reported, not gated: relo +1.146 (9/9), rehi +0.292 (6/7), rein +0.079 (9/10). The
out-of-distribution claim of the phase report stands, and the stand-ins had under-read it at A20 (+0.453).

| arm | read against the control | verdict by its criterion |
|---|---|---|
| I1 `PREV_SPACE = "target"` | A20 +0.0020, A10 +0.0066, relo +0.0071, rehi +0.0040, rein +0.0086, in distribution +0.0064 (8/8); never worse on any trajectory | **pass** |
| I2 `STATS_MASK` | see chunk 2 | fail |
| I3 FAMW | A20 +0.020, A10 −0.003, relo +0.011, rehi −0.001; OOD mean +0.007; in distribution −0.001 | fail |
| I4 early-window prior | not built. Hindsight bound A20 +0.039, A10 +0.032, relo +0.053, rehi +0.013 (mean +0.034); +0.10 … +0.16 on 14-window streams | open, low: the bound is the whole early deficit, an interval prior reaches part of it |
| I5 stream constants, leave one regime out | left-out A20 +0.017, A10 +0.004, relo +0.035, rehi −0.020 (1/7); in distribution −0.007 … −0.027 | fail |
| I6 band from held-out-condition errors | ΔC×100 A20 +0.151, A10 −0.068, relo −0.152, rehi −0.003 (mean −0.018); in-distribution sps −0.103 (0/8) | fail |
| I7 zero propagation | plain +0.0001 … +0.0003; synthetic re-blanks +0.008 … +0.85 | fail as a score lever |
| T1 six F-FNO members | A20 +0.033, A10 +0.024, relo +0.087, rehi +0.023; **OOD mean +0.042**; in distribution +0.004; +0.357 ms per step (local A100, interleaved) | fail, narrowly (bar +0.05) |
| T2 both ViTs with sim | A20 −0.019, A10 +0.008, relo +0.054, rehi +0.019; OOD mean +0.015; in distribution +0.008 | fail |
| T3 sim dose 1.0 | A20 +0.013 (9/16); relo −0.082 (0/9), rehi −0.050 (0/7); in distribution −0.005 | fail |
| **T4 Re-similarity augmentation**, F-FNO members, RE | **relo +0.334 (9/9), rehi +0.046 (6/7), mean +0.190**; rein +0.025; in distribution +0.037 (6/6); frozen: relo +0.505, rehi +0.174 | **passes every term measured**; A20, A10 and the twins' in-distribution read are training (wave 3); one seed set |

**The ViT question.** A single ViT is far behind out of distribution, but in the ensemble the stream stack repairs most
of it. Layout minus mirror: F-FNO only (`ffno4`) A20 −0.176, A10 −0.061, relo +0.187, rehi −0.097, in distribution
−0.110 (0/18); on the twins of record −0.140 (0/8), which is T2R15's −0.155 again. Without the stream (Track 1's case)
dropping the ViTs is worth +1.12 at relo and about nothing elsewhere. `six_vitc2` is the best fixed layout (OOD mean
+0.050, every regime positive, in distribution +0.009) and costs the latency of two more members. FAMW does move its
weight towards the F-FNO where the ViTs are weak (ViT weight 0.20 at relo, 0.26 at A20, prior 0.333), too slowly to pay.
**No fixed composition beats the mirror everywhere; the composition stays.**

**Blanked regions are a property of the angle** (the airfoil's shadow): 6 – 7 pixels are static-zero in all 82
recordings, the union is 232 (u) / 311 (v). A held-out AoA-20 recording carries 5 – 10 static-zero pixels that no
training recording has, and the members predict 0.0005 m/s there (0.0003 on the shared ones, flow scale 0.08). **The
members generalise natural blanking; the synthetic masks of I2 / I7 overstate the exposure.**

Two I6 jobs died because their script was edited while they ran (bash reads a script as it executes); re-run, and
every instrument job now runs from a snapshot of the code.

### Chunk 5, first part: T4's verdict (1.47 GPU-h; `results/ood_chunk5.md`)

**T4 passes every term of its criterion.** Four augmented F-FNO members with the control's two ViTs, against the mirror
control of the same regime; each ensemble with its own refitted band; real package path.

| regime | stream (Track 2) | per subscore: rel / tke / mvpe / sps | frozen (Track 1's case) | criterion |
|---|---|---|---|---|
| A20 | **+0.052 (16/16)** | +0.041 / +0.026 / +0.000 / +0.117 | +0.095 (15/16) | ≥ −0.02, pass |
| A10 | **+0.040 (17/18)** | +0.019 / +0.068 / +0.018 / +0.088 | +0.125 (18/18) | ≥ −0.02, pass |
| relo | **+0.334 (9/9)** | +0.174 / +0.502 / +0.084 / +0.773 | +0.505 (9/9) | mean of relo, rehi |
| rehi | **+0.046 (6/7)** | +0.028 / −0.022 / −0.001 / +0.138 | +0.174 (6/7) | +0.190 ≥ +0.10, pass |
| rein | +0.025 (9/10) | | +0.025 | reported |
| **OOD mean** (A20, A10, relo, rehi) | **+0.118** | | +0.225 | headline |
| **in distribution, twins, all 8** | **+0.033 (8/8)** | +0.015 / +0.080 / +0.006 / +0.071 | +0.027 (7/8) | ≥ −0.03, pass |
| in distribution, twins, AoA 15 – 20 | +0.046 (3/3) | +0.021 / +0.106 / +0.007 / +0.099 | +0.041 (3/3) | |

* The twins read is the one that stands for the development board: three T4 twins with the two ViT twins of record
  against the five twins of record, both bands refitted the shipping way. Composite 77.2838 against 77.2506; the refit
  of the record's twins reproduces the band of record's composite to 2e-4.
* **Not a seed effect.** Leave-one-seed-out at RE, eleven ensembles: relo +0.309 [+0.301, +0.315], rehi +0.045
  [+0.039, +0.055], rein +0.024, in distribution +0.036, against a control-to-control spread over four reseeded draws
  of at most 0.034 / 0.011 / 0.010 / 0.012. The four members share one recipe and one commit: this rules out a lucky
  seed set, not a lucky configuration.
* **T4v at RE, all six members augmented:** relo **+0.709 (9/9)** against the control and +0.376 against T4 with the
  control's ViTs; rehi +0.025 (5/7) and −0.021 (2/7); rein +0.020 and −0.005. Frozen, relo +1.872. All of its margin
  over T4 is at relo. A20, A10 and the twins are training.
* **What is left after the stream is not reachable by a per-trajectory correction.** Share of the squared error,
  control, frozen → stream: the mean error that persists over a trajectory falls 6.9 → 0.7 % (rehi), 28.3 → 1.2 % (relo),
  12.4 → 1.0 % (A20); what remains is 65 – 76 % fluctuation (relo 65, A20 72, rehi 76) and 23 – 34 % mean error that changes from
  window to window.
  rehi gains least from everything because it had little persistent error to begin with.

### Chunk 5, second part: the decision table (0.25 GPU-h; `results/ood_chunk5.md`, `ood/ood_decision.py`)

Delta composite to the control (trajectories with a positive delta); the control row is absolute. Real package path,
every ensemble with its own refitted band; the twins columns are the 8 `VAL_TRAJ` trajectories with the band refitted
the shipping way. "T4 F-FNO only" = four augmented F-FNO with the control's two ViTs; "T4 all" = all six augmented.

**Stream (Track 2)**

| | twins, all 8 | twins, AoA 15 – 20 | A20 | A10 | relo | rehi | rein | OOD mean |
|---|---|---|---|---|---|---|---|---|
| control | 77.2506 | 74.7712 | 73.6581 | 76.1447 | 78.3785 | 72.9806 | 77.1478 | 75.2905 |
| T4 F-FNO only | +0.0332 (8/8) | +0.0459 (3/3) | +0.0516 (16/16) | +0.0395 (17/18) | +0.3335 (9/9) | +0.0458 (6/7) | +0.0245 (9/10) | **+0.1176** |
| T4 all | +0.0315 (7/8) | +0.0514 (3/3) | +0.0527 (14/16) | +0.0362 (14/18) | +0.7090 (9/9) | +0.0247 (5/7) | +0.0195 (8/10) | **+0.2056** |

**Frozen (Track 1's case)**

| | twins, all 8 | twins, AoA 15 – 20 | A20 | A10 | relo | rehi | rein | OOD mean |
|---|---|---|---|---|---|---|---|---|
| control | 76.8516 | 74.6041 | 72.4290 | 75.1865 | 76.4400 | 72.3811 | 76.7728 | 74.1092 |
| T4 F-FNO only | +0.0272 (7/8) | +0.0405 (3/3) | +0.0947 (15/16) | +0.1247 (18/18) | +0.5049 (9/9) | +0.1743 (6/7) | +0.0246 (9/10) | +0.2246 |
| T4 all | +0.0077 (5/8) | +0.0411 (3/3) | +0.1050 (14/16) | +0.1354 (17/18) | +1.8724 (9/9) | +0.1339 (4/7) | +0.0204 (9/10) | +0.5617 |

* **T4v passes every term of its criterion** (OOD mean +0.206 over the control; +0.088 over T4 F-FNO only; in
  distribution +0.032), **and its whole margin over T4 F-FNO only is one regime.** "T4 all" minus "T4 F-FNO only":
  relo **+0.376 (9/9)**, rehi **−0.021 (2/7)**, both outside the reseeding spread; rein −0.005, twins −0.002, A20 +0.001,
  A10 −0.003, all inside it. The augmented ViTs lose at rehi through tke (−0.12) and mvpe (−0.03), and they cost mvpe
  in distribution (−0.020).
* **The two recipes cannot be told apart in distribution.** Translated with the project's correction rules (an error
  cut halved, containment never banked, time unchanged): about **+0.013 … +0.019 composite on a board-like set**,
  brackets [+0.006, +0.051], for either. This is an expectation from the rules, not a board read: nothing can be
  uploaded to the development board any more.
* **So the choice is a bet on the private set's Reynolds numbers.** Below the training range the augmented ViTs are
  worth +0.38; above it they cost 0.02; at a novel angle and in distribution they change nothing. The release's own
  hints point both ways: the Track-2 example file is Re 2968 (below), the sim-only Reynolds numbers are 15225 (inside)
  and 27975 (above).
* Each recipe is one ensemble per regime; "T4 all" rests on one seed per augmented ViT; the reseeding spread was
  measured at RE and in distribution only.

## The candidate (2026-09-28)

| | value |
|---|---|
| recipe | the method of record, every member and twin trained with `--aug-resim 0.25` (p 0.5); **the planner's recommendation is "T4 all"**, on the asymmetry of the bet; "T4 F-FNO only" is the conservative alternative and needs no further training once "T4 all" is trained (it reuses the ViTs of record) |
| constants | those of `build_final.sh`, plus `PREV_SPACE = "target"` |
| unchanged | member composition, ALPHA, JOINT_GAIN, the band's construction, every stream constant, `STATS_MASK` off, FAMW off |
| code | `patches/candidate_release.patch` applied to the tree: `train2.py` (the flag, inert at 0), `reproduce.sh decision`, `build_final.sh` (lock, constant check), `Track_1/verify/baked_constants.py`, documents. `results/candidate_release.md` |
| verified | the candidate code rebuilds both packages of record, data byte for byte and outputs bit for bit (job 7575926); the constant check passes 41 of 41 and fails when it should; `reproduce.sh decision` runs end to end at 2 epochs (job 7575939) |
| not verified at the time (467769d) | the 400-epoch run from a clean clone, the candidate band, the band's reuse path: all three done since, see the three sections below |

## The from-scratch run of the candidate (2026-09-28 18:12 → 22:38 EDT, 41.8 GPU-h)

`reproduce.sh decision` from a clean clone of commit `467769d` (`R/cand/repro/repo`), eleven trainings as eleven jobs
(7577300 – 7577310), then the band refit and the build (7577311). Recipe "T4 all": `RPDE_DP_RESIM_FFNO` and
`RPDE_DP_RESIM_VIT` at their default 0.25.

| | result |
|---|---|
| trainings | 11 of 11 COMPLETED; F-FNO members 4.3 h, F-FNO and ViT-C twins 3.9 h, ViT-C member 4.3 h, ViT-P 2.2 / 2.4 h (A100) |
| band | refitted from the new twins with the record's flags: `R/cand/repro/log/decision/band/k0/band_k0_sigblend_lam0.5.npz`, md5 `4d1777c8bf82…`; SIGBLEND q = (0.4741, 0.4813) against the record's (0.4847, 0.4907), log κ lower by 0.03 – 0.05: a slightly narrower band, as the twins' errors are smaller |
| build | `BUILD_FINAL OK`: gates 2 of 2 SUBMITTABLE, causality OK (176 steps, largest difference 0), baked constants 41 of 41 with `PREV_SPACE 'target'` |
| archives | `R/cand/repro/pkg/track1_dp_resim.zip` (sha256 `99b843c6…4a588d`), `track2_dp_resim.zip` (sha256 `909c6cd5…b8e58b`), 30 MB each. **On scratch only; nothing is staged.** |

Still to do before the candidate can be proposed: the reproducibility read (these twins against the campaign's twins,
within 0.005), the "T4 F-FNO only" build from the same checkpoints, the package on a different card, the band committed
with its `MANIFEST.md` entry, the registered predictions.

## The closing gates (2026-09-28 22:40 → 23:00 EDT, 0.71 GPU-h; `results/ood_chunk6.md`)

(a) = the five twins of the from-scratch run, (b) = the campaign's five augmented twins, (c) = the five twins of record;
8 `VAL_TRAJ` trajectories, band refitted the shipping way.

| gate | read | verdict |
|---|---|---|
| **R1 reproducibility** | (a) 77.2817 against (b) 77.2821: **0.0003**, bar 0.005; per trajectory within ±0.0012; frozen −0.0017. The two ViT twins retrained bit-identically on this hardware, the F-FNO twins differ by 0.4 – 0.5 % of weight norm and by at most 2.2e-5 in rel / tke / mvpe | **pass** |
| the candidate's gain, read a second time | (a) − (c): **+0.0311** all 8 (7 of 8; 26700_0 −0.022), +0.0508 on AoA 15 – 20; frozen +0.0060 / +0.0397 | agrees with the campaign's +0.0315 |
| **R1b the band** | three refits (from-scratch, shipping-way on (a), campaign twins) agree to 1e-4 composite; the band of record is 2.4 % wider and reads +0.002 (stream), +0.020 (frozen) on (a) | an organizer's own refit changes nothing measurable |
| **the second candidate**, "T4 F-FNO only" | built by `reproduce.sh decision` with `RPDE_DP_RESIM_VIT=` from a clean clone: `BUILD_FINAL OK`, 41 of 41. Its twins +0.0329 (8/8) against the record, +0.0450 on AoA 15 – 20. Archives `R/ood/cand2/pkg/track{1,2}_dp_resim_v.zip` (sha256 `5463cc12…`, `42e18eb3…`) | built |
| **R2 a different card** | A100 and V100S: the same load-time choices for all three archives (fp16 accepted, graph captured, arena on; Track 1's batch probe 56 / 28 as intended); gates SUBMITTABLE on both; V100 − A100 at most 2.4e-3 relative L2 per window, forced fp32 2.9e-3 (the package's own bar is 2e-2), forced eager and no arena bit-identical; composite moves by at most 0.0006 | **pass** |

* The reproducibility read is on one hardware and stack; for the organizers' retrain the scale to expect is the
  reseeding spread (0.012 in distribution, 0.034 at relo).
* R1 compares twins. The candidate's all-data members have no second training to compare with.
* The uploaded Track-2 package returns a non-finite prediction on a NaN input (it predates the guards); both
  candidates do not.
* Two defects of `reproduce.sh decision` found by the second build, in repair: the tag of the empty-variable recipe
  (`dp_resim_v`), and a closing message that says "no build" after a build.

**In distribution, expected on a board-like set** (the project's correction rules; not a registered prediction):

| change | all 8 | AoA 15 – 20 |
|---|---|---|
| T4 all | +0.0125 [+0.0062, +0.0311] | +0.0191 [+0.0095, +0.0508] |
| T4 F-FNO only | +0.0139 [+0.0069, +0.0329] | +0.0178 [+0.0089, +0.0450] |
| `PREV_SPACE` target against the uploaded `input` | 0.0000 [0, +0.0064] | 0.0000 [0, +0.0027] |

`PREV_SPACE` banks nothing under the rules because it moves only the interval; its mechanism is a certain fix.

## The candidate bands committed (2026-09-28 23:13 EDT; `patches/candidate_band.patch`, `results/candidate_band.md`)

| recipe | selected by | tag | committed band | npz md5 | q (u, v) |
|---|---|---|---|---|---|
| T4 all | both variables at their default 0.25 | `dp_resim_all` | `Track_1/bands/dp_resim_all_sigblend_lam0.5.npz` + `Track_1/sigblend/band_dp_resim_all_sigblend_lam0.5.json` | `4d1777c8bf82…` | 0.4741, 0.4813 |
| T4 F-FNO only | `RPDE_DP_RESIM_VIT=` (empty) | `dp_resim_ffno` | `…/dp_resim_ffno_sigblend_lam0.5.npz` + `…/band_dp_resim_ffno_sigblend_lam0.5.json` | `42026ad8158a…` | 0.4786, 0.4881 |

* `reproduce.sh decision` now reuses the committed band of the selected recipe, as `final` reuses its own, and refits
  only with `RPDE_REPRO_FITBAND=1`; a recipe without a committed band is refused. The tags name the recipe; the closing
  message after a build is true.
* **Identity (jobs 7580924, 7580925):** rebuilt from the committed bands through the reuse path, both candidates'
  archives equal the existing ones **file by file, byte for byte** (9 of 9 files, both tracks, both recipes), with
  `BUILD_FINAL OK` and 41 of 41 baked constants. `build_final.sh` with default arguments still rebuilds the packages of
  record: data byte-identical to the uploads, outputs identical at tolerance 0.
* The candidates on scratch keep the tags they were built with (`dp_resim`, `dp_resim_v`); the rebuilt, identical
  archives are `R/cand/band_commit/{all,ffno}/pkg/track{1,2}_dp_resim_{all,ffno}.zip`.

## Decisions

* **2026-09-28:** `STATS_MASK` stays off (arm I2 closed by its own criterion; see above).
* **2026-09-28:** arms I3, I5, I6, T1, T2, T3 closed by their criteria on the mirror ensembles. I1 passes.
* **2026-09-28:** the member composition (4 F-FNO + ViT-P + ViT-C) stays.
* **2026-09-28:** T4 passes; T4b fails (0.25 stands); T4v passes. The recipe is the owner's choice between "T4 all"
  and "T4 F-FNO only"; the from-scratch run trains what both need.
* **2026-09-28:** arm I7 closed as a score lever (+0.0002 against +0.03). Open as insurance, for the owner.

* **2026-09-28, owner:** Ascend first whenever it is idle (same price, faster); Pitzer is the overflow. `sbatch --test-only`
  is not evidence that Ascend is busy: it printed a start 24 h away while real jobs started in seconds. Probe with one
  real job instead.
* **2026-09-28, owner:** the working branch is `qwu/dev`; commit when a piece of work is done, at most 10 a day.

## The owner's decisions (2026-09-29)

| question | decision |
|---|---|
| the recipe of the candidate | **"T4 all"**: all six members and five twins with `--aug-resim 0.25`, plus `PREV_SPACE = "target"`. `reproduce.sh decision` with its defaults, tag `dp_resim_all` |
| zero propagation (I7) as insurance | **not added** |
| the documents that state what the campaign overturned | **to be updated** (`results/docs_update.md`; the two reports in `../artifacts/`) |
| the release of 09-27 | **tagged** `final-20260927` on `4e5ed37` (main after pull request #3; annotated; local, the owner pushes it) |
| upload | **not decided.** Nothing is staged; no registered prediction is written into `../sub_candidates/` |

## Closing steps (2026-09-29, 00:45 – 01:00 EDT)

* **Tag** `final-20260927` → `4e5ed37`. `fdac00b` (the tree that rebuilds the uploads byte for byte, code included) is
  an ancestor of it.
* **Promotion.** The candidate's eleven checkpoints were copied from `R/cand/repro/ckpt/` into `Track_1/unet/`, md5
  checked, no name clash (261 MB, 11 files; list with md5 in `R/cand/repro/ckpt/PROMOTED_20260929.md5`). The scratch
  copies stay as the second copy.
* **The candidate's archives** are kept in `R/candidate/` (`track1_dp_resim_all.zip` sha256 `53291df1…790e96`,
  `track2_dp_resim_all.zip` sha256 `bd0a3fb4…68028f`). Their contents equal the from-scratch run's archives file by
  file (md5 of all nine files, both tracks); the sha256 of the zip differs because the archive was rebuilt under its
  new tag.
* **Worktrees.** Nine detached worktrees under `R/` removed with `git worktree remove` after checking that each one's
  changes are in the repository (`wt_t4` = `patches/t4_aug_resim.patch`, `wt_cand` = `patches/candidate_release.patch`,
  `wt_cand2` = commit `6ad14ea` file by file; `wt_base` and `ood/wt_ood` were clean). `git worktree list` shows the
  working tree only.
* **Scratch.** `R/` went from 84 GB to 2.1 GB: 77.2 GB and 48 956 entries in 392 paths deleted by
  `R/jobs/cleanup_20260929.py` (dry run first), every path with its size and reason in
  `/fs/scratch/PAS1187/qwu/tmp/realpde/CLEANUP_20260929.md`. Kept: every final checkpoint, training log, json and eval
  text, the instrument's reports and json, every band, job scripts, Slurm logs. Nothing outside `R/` was touched.
* **The older scratch areas, done later on 09-29 at the owner's request:** the sprint and the wrap-up of 09-27
  (`final0927` 50 GB, `trick0927` 38 GB, `wrap0927` 5.4 GB) and `distill` (28 GB) went from 120.2 GB to 61 MB:
  working trees and clones, the checkpoints of closed arms (194 distinct, none of them in `Track_1/unet/`), dumps and
  archives deleted; notes, job scripts, logs, json and named bands kept in place. Nothing read those areas; the
  shipped bands and the two uploads have byte-identical copies in the repository and in `../sub_candidates/final/`.
  The eight heads of the K-sprint's clones that the repository does not hold are kept as git bundles
  (`trick0927/clones/`). Every path, with md5 and training arguments of each checkpoint:
  `/fs/scratch/PAS1187/qwu/tmp/realpde/CLEANUP_20260929_sprint.md`.

