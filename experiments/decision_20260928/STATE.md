# State of the Decision-phase campaign

Plan and criteria: `PLAN.md`. Update this file at every wave; keep job ids, paths and the GPU-hour ledger here.

## Ledger

| date | wave | GPU-h used (sacct) | cumulative | of 300 |
|---|---|---|---|---|
| 2026-09-28 | opened | 0 | 0 | 0 % |
| 2026-09-28 05:15 | wave 1 moved to Ascend; smokes and the cancelled Pitzer start | 1.4 (Pitzer 1.1 cancelled, smokes 0.3) | 1.4 | 0.5 % |

Committed, not yet spent (estimates, A100): wave 1 about 60, arms T3 + T4(RE) about 31. Read the ledger with
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

## Decisions

* **2026-09-28:** `STATS_MASK` stays off (arm I2 closed by its own criterion; see above).

* **2026-09-28, owner:** Ascend first whenever it is idle (same price, faster); Pitzer is the overflow. `sbatch --test-only`
  is not evidence that Ascend is busy: it printed a start 24 h away while real jobs started in seconds. Probe with one
  real job instead.
* **2026-09-28, owner:** the working branch is `qwu/dev`; commit when a piece of work is done, at most 10 a day.
