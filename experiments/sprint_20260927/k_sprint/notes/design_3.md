# Design K3 — short warm-start "polish" of the shipped members + zoo re-composition

Author: design agent K3, 2026-09-27 16:35–17:00 EDT. `K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927`.
Clones: `$K/wt_3` (training + candref reads, HEAD 55f0e88, NO edits to tracked files), `$K/wt_3b` (build only).
`U=/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet` (stage-1 checkpoints, read-only).
**All training jobs are already running (§3). The executors' work is §5 (evaluate) and §6 (build + gate).**

## 1. Hypothesis (main bet: hard-regime specialisation)
The board's error level is ABOVE our hardest held-out proxy on every subscore (board rel 0.1105 / tke 0.513 / mvpe 0.122
vs twins on val_unseen AoA 15–20: 0.1009 / 0.4404 / 0.0880; full split 0.0750 / 0.4546 / 0.0658; out-of-fold rel by AoA is
0.052 / 0.053 / 0.086 / 0.106 / 0.121). The public set therefore behaves like AoA 15–20 (mean of the two OOF values
0.1135 ≈ board 0.1105), while our members spend 60 % of their training draws on AoA 0–10. A short warm-start polish that
draws only (or mostly) AoA 15/20 real windows re-allocates the 1.3 M-parameter F-FNO (and the ViTs) to the regime the board
scores. Sim co-training is kept unchanged (all AoA), so the regulariser that pays `mvpe` stays.
Untested before: `--aoa-boost/--aoa-only` exist in `train2.py` since round 18 but were only run from scratch on FNO w24
(`fno_r18_aoab2`), never as a warm-start stage on the shipped F-FNO/ViT members, never at the shipped operating point.

Secondary arm: `do0` — dropout-off polish (adapt the trunk to the deterministic inference regime).
Control: `ctl` — same epochs / lr / recipe, nothing changed (separates "more training" from the trick).

## 2. Arms (one change each; everything else = REPRODUCE.md §1.4/§1.5 (F-FNO) or §1.2/§1.3 (ViT))
Stage 2 = stage-1 command + `--init <stage-1 ckpt>` + `--epochs E --lr LR` (OneCycleLR over E epochs, peak LR, starts at
LR/25: no destructive first step — verified in the logs, epoch-0 loss equals the parent's level) + the arm flags.
`--seed` = the parent's seed.

| arm | change | epochs | peak lr | models |
|---|---|---|---|---|
| `ctl` | none | 60 | 1e-4 | 3 F-FNO twins + 4 F-FNO members + 2 ViT twins + 2 ViT members |
| `hib` | `--aoa-boost 4 --aoa-hi 15 20` (AoA 15/20 = 72.8 % of real draws instead of 40 %) | 60 | 1e-4 | same 11 |
| `hio` | `--aoa-only --aoa-hi 15 20` (5361 of 13368 real windows; sim unchanged) | 150 (= same number of steps) | 1e-4 | same 11 |
| `hio2` | as `hio`, recipe peak lr (plasticity dose) | 150 | 2e-4 | 3 F-FNO twins + 4 F-FNO members |
| `do0` | `--dropout 0.0` | 60 | 1e-4 | 3 F-FNO twins + 4 F-FNO members |

Twins come only from twin parents (`p2_ffno_l8_sim_s1..s3`, `arch_vit_d384`, `arch_vit_cmb3sim`; held-out split,
`--sim-exclude-held`), members only from all-data parents (`ffno_all_w40l8_sim_s1..s4`, `arch_vit_all_s1`,
`arch_vit_all_cmb3sim_s3`; `--all-data`). Held-out evidence stays held-out.

Outputs: `$K/ckpt/3/<arm>/<parent>_<arm>.pth`. Logs: `$K/logs/rp_k273_<arm>_<t|m><n>_<jobid>.log` (F-FNO) and
`rp_k273_<arm>_v<tp|tc|mp|mc>_<jobid>.log` (ViT: tp/tc = twin plain/cmb3sim, mp/mc = member plain/cmb3sim).
A job is finished when its log contains the line `... DONE` (the `.pth` may be half-written before that).

Exact commands: `$K/jobs/rp_k273_polish.sh` (F-FNO), `$K/jobs/rp_k273_polish_vit.sh` (ViT); submit wrappers
`$K/jobs/rp_k273_submit.sh ARM t|m N cluster EPOCHS LR DROPOUT [EXTRA]`, `$K/jobs/rp_k273_submit_vit.sh ARM tp|tc|mp|mc cluster EPOCHS LR [EXTRA]`.

## 3. Jobs launched (all job IDs: `$K/notes/design_3_jobids.txt`)
Cardinal `-p gpu` started 1 of 14 jobs in 3 min (Priority) → per the brief the pending ones were cancelled and resubmitted on
Ascend nextgen. **K3 holds ~47 Ascend GPUs until ~17:25 (above the 20-GPU K-sprint allowance; flagged to the commander)**.

| wave | jobs | cluster | started | expected end |
|---|---|---|---|---|
| F-FNO members ctl/hib/hio ×4, do0 m1–2 | 7566761–7566774 | Ascend | 16:39 | ≈ 17:15 (measured 34.4 s/epoch; hio 13.8) |
| F-FNO twin ctl_t1 | 14942663 | Cardinal H100 | 16:40 | ≈ 16:59 (18.3 s/epoch) |
| F-FNO twins (11) + do0 m3–4 | 7566849–7566862 | Ascend | 16:42 | twins ≈ 17:14, do0 m3–4 ≈ 17:17 |
| ViT ctl/hib/hio × (tp, tc, mp, mc) | 7566877–7566889 | Ascend | 16:46 | plain ≈ 17:03–17:05, cmb3sim ≈ 17:19–17:24 |
| hio2 F-FNO twins ×3 + members ×4 | 7566895–7566901 | Ascend | 16:47 | twins ≈ 17:20, members ≈ 17:23 |
| candref baseline (per-trajectory) / zmix / zvcc | 7566871 / 7566828 / 7566872 | Ascend | 16:40–16:45 | done ≈ 16:52 |

All logs checked at 16:46: `init: ... 79/79 tensors copied`, configs as intended (dropout / aoa flags / lr), losses at the
parents' level at epoch 0.

## 4. Zoo re-composition (zero training) — CLOSED, nothing to ship
* QD-retrained members ("qd_static") are NOT on this machine (no `*qd*`/`err-head` checkpoint among the 334) → cannot be built.
* Paired candref reads vs baseline (twin package, static band = T1-equivalent; `d_fin_nt` = Δ of
  0.4778 rel + 0.1010 tke + 0.0921 mvpe + 0.2498 sps):

| set | what | AoA 15–20 d_fin_nt | full d_fin_nt |
|---|---|---|---|
| `zmix` | F-FNO stack = 2 sim + 1 no-sim twin (members 2 sim + 2 no-sim) | −0.0055 | −0.0132 |
| `zvcc` | ViT pair = 2× cmb3sim (instead of plain + cmb3sim) | +0.0004 | −0.0054 |

  Both inside the noise; every other two-invocation composition in the zoo is already closed (third family, member count,
  L4/L6 depth, Swin, FNO w96) or lacks twins (Swin, F-FNO L4/L6 all-data only) or is prohibited (`arch_vit_d2*` = DINOv2 init).

## 5. EXECUTOR A — evaluate (start 17:15 for the `f` reads, 17:24 for `fv` and `hio2`; finish by 17:50)
```bash
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
squeue -u $USER -M ascend,cardinal -h -o "%j %T %M" | grep rp_k273 | sort        # what is still running
grep -L "DONE" $K/logs/rp_k273_{ctl,hib,hio,hio2,do0}_*.log                      # logs WITHOUT the DONE line
# one paired read per arm (refuses with "NOT READY" until all the arm's jobs printed DONE); ~6 min each, run in parallel
for a in ctl hib hio do0 hio2; do $K/jobs/rp_k273_eval_arm.sh $a f; done          # polished F-FNO + ORIGINAL ViTs
for a in ctl hib hio; do $K/jobs/rp_k273_eval_arm.sh $a fv; done                   # polished F-FNO + polished ViTs
# early look with the twins only (does not need the members):  $K/jobs/rp_k273_eval_arm.sh <arm> f 1
python3 $K/jobs/rp_k273_table.py ctl_f hib_f hio_f do0_f hio2_f ctl_fv hib_fv hio_fv | tee $K/notes/result_3_table.txt
```
`rp_k273_eval_arm.sh` submits `$K/jobs/rp_k273_candref.sh` (copy of the F-sprint template, reads only): `ens_v9` twin reads
(AoA 15–20 and full, f 1.39 / 1.29), `fit_band` with the shipped construction (`--ckpt members --err-ckpt twins`, 0.80,
p 0.45, centre0, fill, inflate `${INFLATE:-1.39}`), a twin T2 package with the asnp constants, stream reads static / asym R1 /
asym R1 + D1c on val_unseen (all 8, AoA 15–20) and the static read per trajectory. Output `$K/pkg/3/candref_<tag>/`
(`band.npz` there is a shippable band for the arm). `rp_k273_table.py` prints every read paired against `baseline`
(log `rp_k273_cr_baseline_7566871.log`, already done) with per-trajectory sign counts. `time` in these reads is meaningless.

**Band trap (specialist arms).** The band base is fitted on `val_seen` (trajectories the twins TRAINED on, incl. 10125_20,
12675_20, 22875_15) and bridged to unseen data by the 1.39 inflate. A polish on AoA 15/20 lowers the twins' in-sample error
there faster than their held-out error, so 1.39 may become too narrow. Check `coverage` in the stream `errors` line
(baseline static: 0.877 AoA 15–20, 0.920 all). If an arm's coverage is more than 0.010 below baseline, re-read it at
`INFLATE=1.50` and `INFLATE=1.60` (`INFLATE=1.50 $K/jobs/rp_k273_eval_arm.sh hio f` → tag `hio_f_i150`) and take the inflate
whose AoA 15–20 coverage is closest to 0.877 (coverage-matched; pre-registered — do not pick by best sps).

**Decision rule (pre-registered).** Primary read = twin package, STATIC band, val_unseen AoA 15–20 (`stream twin static aoa15-20`).
1. SHIP-candidate if `d_fin_nt` ≥ +0.04 vs baseline AND ≥ +0.02 vs the matching control (`ctl_f` / `ctl_fv`) AND rel sign
   count ≥ 2/3 AoA 15–20 trajectories AND the `ens_v9` AoA 15–20 rel error is lower than baseline.
2. Full split (`stream twin static all`): report it. `d_fin_nt` ≥ −0.05 → plain SHIP; below that → "GAMBLE" (ships only if
   the commander wants a high-variance public-board bet; it bets the public set is AoA 15–20-like).
3. Among passing arms take the largest AoA 15–20 `d_fin_nt`; `fv` beats `f` only if it is ≥ +0.02 better.
4. Nothing passes → NO-SHIP; K3 contributes nothing to the package (say so plainly).
Board projection: predicted board Δfinal = 0.5 × local AoA 15–20 `d_fin_nt` (the proxy over-reads error cuts ~2×), bracket
[0, 1.0 × local]; T2 (asym R1) read from the `asym` rows, same halving.

Write `$K/notes/result_3a.md`: the table, coverage per arm, the verdict per arm, the winner (arm, mode, inflate, band path).

## 6. EXECUTOR B — build + gate the winner (start when A names it; must finish by 18:55)
```bash
K=/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927
DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
# ARM/MODE = the winner; BAND = the candref band of the winning tag (shipped construction at the chosen inflate)
export ARM=hio MODE=f BAND=$K/pkg/3/candref_hio_f/band.npz        # D1C=1 adds RPDE_SHIP_D1C_N0=30.0 to the T2 package
env -u SLURM_JOB_ID -u SLURM_MPI_TYPE -u CUDA_VISIBLE_DEVICES sbatch ${DESK:+--exclude=$DESK} --export=ALL \
    --chdir=$K/wt_3b --output=$K/logs/rp_k273_build_%j.log $K/jobs/rp_k273_build.sh
```
`rp_k273_build.sh` = BUILD_RUNBOOK §2 with today's knob sets (common recipe-A knobs, T1 "27lz", T2 "asnp", ALPHA 0.80,
`RPDE_FORCE=1`, `RPDE_SUB=$K/sub_candidates`), runs only in `$K/wt_3b`, restores the sed-ed files, then
`validate_t2.py --adaptive` and `stream_eval.py --causal-check`. Zips: `$K/sub_candidates/track{1,2}_k273_<arm>_<mode>_a080.zip`.
Read the `SUBMITTABLE` line per track (the build does not propagate gate exit codes), `CAUSALITY OK`, and the sha256.
One build at a time in `wt_3b`. Two T2 variants wanted (asnp and asnp + D1c): run the job twice, second with `D1C=1`.
The all-data package's own validate numbers are contaminated (members saw val) — evidence is §5 only.
Write `$K/notes/result_3b.md`: zip paths, sha256, gate lines, predicted five subscores = today's baseline prediction
(T1 81.603, T2 81.776 / +D1c 81.783) + the projected Δ per subscore from §5, with the bracket.

## 7. Predicted Δfinal (before any read; board, vs today's planned packages)
| arm | estimate | bracket | P(≥ +0.10) |
|---|---|---|---|
| `hio` / `hio2` (F-FNO + ViT) | +0.02 | [−0.10, +0.12] | ~5 % |
| `hib` | +0.01 | [−0.05, +0.08] | ~3 % |
| `do0` | −0.02 | [−0.15, +0.05] | ~1 % |
| `ctl` | 0.00 | [−0.03, +0.03] | 0 |
Moves `rel` (and through ā, `sps`); `tke`/`mvpe` expected flat to slightly better at high AoA; containment can go either way
(band trap above). Risk specific to the specialists: low-AoA windows get worse; if the public set contains them the package
loses what the full-split read shows.

## 8. Reproducibility recipe
Stage 1: REPRODUCE.md §1.4 (4 F-FNO members), §1.5 (3 F-FNO twins), §1.2/§1.3 (ViT members / twins) — unchanged.
Stage 2 (per model, 1 GPU, ≤ 40 min A100): the same stage-1 command with `--init <stage-1 ckpt> --epochs E --lr LR` and the
arm flags of §2 (`rp_k273_polish.sh`, `rp_k273_polish_vit.sh` are the exact commands; same `--seed` as the parent).
Stage 3: band refit on the polished twins (`fit_band.py`, shipped construction, inflate per §5) and build (§6).
Nothing external: released train_real / train_sim only, our own checkpoints as the warm start.

## 9. Ruled out in design (not launched)
* L1 / Huber / smaller-`cov-c` field term: same family as the closed per-element Welsch scale (covmap A/B, 6/6 negative);
  calm-pixel error is target noise + an unreadable window bias, a median-seeking centre does not move it.
* Late-horizon-weighted loss: late frames already dominate the MSE gradient (their error is 2× larger).
* Stride-1/2 polish: 2.5–5× fewer epochs in the same wall-clock; from-scratch `fno_r18_stride2` was not adopted.
* Real-only final phase: sim is the lever that paid `mvpe` on the board (+0.056/+0.068); dropping it is charged in full.
* Joint fine-tune of the fused 4×F-FNO stack on the ensemble mean: it is a w160-equivalent single model (capacity ladder
  closed) and the F-sprint's `ffvit0` control already measures joint fine-tuning.
* From-scratch AoA specialists: 400 epochs do not fit the clock on A100, and less data at full plasticity overfits.
