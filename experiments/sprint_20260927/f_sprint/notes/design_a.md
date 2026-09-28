# Design A — `ffvit`: a cross-stitched F-FNO x ViT hybrid (architecture composition)

Author: design agent A, 2026-09-27 16:30 EDT. Worktree `S/wt_a` (S = `/fs/scratch/PAS1187/qwu/tmp/realpde/final0927`),
HEAD 55f0e88 + uncommitted edits (listed in §8). U = `/users/PAS1187/qwu/workspace/realpde/main/Track_1/unet`, K = `$S/ckpt/a`.
**All training jobs are already running (§5). The executor's work is §6 (evaluate) and §7 (integration / package).**

## 1. Hypothesis
Today the two families only meet at the output (mean of 4 F-FNO + 2 ViT fields). `ffvit` puts one trained F-FNO
(per-pixel, separable spectral mixing) and one trained ViT (128 tokens, input-dependent global attention) into ONE network as
two streams that run layer for layer and **exchange features after layers 1, 3, 5**:

```
f2v[i]: LayerNorm2d(40) -> Conv(40,384,k4,s4) -> GELU -> Conv1x1(384,384)      added to the ViT tokens
v2f[i]: LayerNorm(384)  -> Linear(384,384)    -> GELU -> Linear(384,40*16)     unpatchified, added to the F-FNO map
output: field = s*field_f + (1-s)*field_v, sigma = s*sigma_f + (1-s)*sigma_v, s = sigmoid(mix), mix learned, init s = 1/3
```

The LAST layer of every link is zero-initialised, so with `f.*`/`v.*` warm-started from our own trained parents the network
at step 0 **is exactly** the (1/3 F-FNO + 2/3 ViT) ensemble of its two parents (checked: max |diff| 4.8e-7 field, 2.4e-7 logstd).
Stage 2 is a short fine-tune with the unchanged F-FNO loss recipe (REPRODUCE.md §1.4/§1.5). The ViT stream gets the
full-resolution spectral features it lacks (its 4x4 unpatchify is its weak point); the F-FNO stream gets input-dependent
global context. 18.34 M params (F-FNO 1.31 M + ViT 14.66 M + links 2.37 M).

**Control `ffvit0`** = the same network with NO links (15.97 M): same parents, same mixed output, same loss, epochs, lr,
seed. Arm minus control = the cross-stitch; control minus parents = "joint fine-tune / more training".

## 2. Which subscore moves, predicted delta
* Intended: `rel`/`tke` of the hybrid members (the ViT stream's error falls) and, through a different error field in the
  tail, containment C via the refitted band. `mvpe` expected flat.
* Risk: joint training through a mixed output co-adapts the two streams and REDUCES the diversity that pays containment.
  That is exactly what the twins ensemble read (not the single-member read) decides.
* **Predicted delta final vs today's package, same knobs: +0.02, bracket [-0.15, +0.20]** (T1, fused; unfused costs a
  further -0.08 on T1 per extra invocation). Honest P(>= +0.10 on the board) ~ 7 %.

## 3. Members, parents, and the two waves
| name | arch | F parent (`f.*`) | ViT parent (`v.*`) | seed | split |
|---|---|---|---|---|---|
| `*_t1` | twin | `p2_ffno_l8_sim_s1` | `arch_vit_d384` | 1 | held-out (`--sim-exclude-held`, no `--all-data`) |
| `*_t2` | twin | `p2_ffno_l8_sim_s2` | `arch_vit_cmb3sim` | 2 | held-out |
| `*_t3` | twin | `p2_ffno_l8_sim_s3` | `arch_vit_d384` | 3 | held-out |
| `*_m1` | member | `ffno_all_w40l8_sim_s1` | `arch_vit_all_s1` | 1 | `--all-data` |
| `*_m2` | member | `ffno_all_w40l8_sim_s2` | `arch_vit_all_cmb3sim_s3` | 2 | `--all-data` |
| `*_m3` | member | `ffno_all_w40l8_sim_s3` | `arch_vit_all_s1` | 3 | `--all-data` |
| `*_m4` | member | `ffno_all_w40l8_sim_s4` | `arch_vit_all_cmb3sim_s3` | 4 | `--all-data` |

Twins come only from twin parents, members only from all-data parents, so held-out evidence stays held-out.

* **Wave A** (prefix `ffvit_` arm / `ffvit0_` control): `--epochs 50 --lr 1e-4`, started 16:19.
  Expected finish: twins **~17:13**, members **~17:20 EDT**.
* **Wave L** (prefix `ffvitL_` arm / `ffvit0L_` control): `--epochs 62 --lr 2e-4` (the recipe's own lr; the high-variance
  arm), started 16:27. Expected finish: twins **~17:34**, members **~17:42 EDT**. Every job has `-t 01:30:00`, so wave L
  is killed at 17:57 at the latest. (A first wave L at 75 epochs, jobs 7566636-7566646, was CANCELLED at 16:27: at the
  measured 71 s/epoch its all-data members would have hit the time limit. Its logs are in `$S/logs/a_cancelled/`.)
* Measured on A100 (wave A, epoch 5): control twin 58 s/epoch; arm twin ~63; arm all-data member ~71.

## 4. Exact training command (stage 2)
Job script `$S/jobs/rp_f27a_train.sh`, submitted through `$S/jobs/rp_f27a_submit.sh NAME ARCH FPAR VPAR SEED ALLDATA EPOCHS LR`.
Per job:
```bash
cd $S/wt_a/Track_1/unet
$PY make_ffvit_init.py $U/$FPAR.pth $U/$VPAR.pth $K/init/init_$NAME.pth      # keys f.<ffno key>, v.<vit key>
$PY train2.py --data $REAL --arch $ARCH --width 40 --n-layers 8 --modes-y 16 --modes-x 32 --dropout 0.3 --var-head \
   --loss mse --lam-f 0.0 --lam-cov 2.0 --cov-c 0.25 --lam-grad 1.0 --lam-tgrad 0.3 --lam-v 0.15 --lam-t 0.15 \
   --alpha-train 0.0 --stride 5 --batch 32 --lr $LR --epochs $EPOCHS --wd 1e-5 --seed $SEED \
   --sim $SIM --lam-sim 0.3 --sim-loss full --sim-map scalar --sim-stride 5 \
   {--sim-exclude-held | --all-data} --init $K/init/init_$NAME.pth --out $K/$NAME.pth
```
`--init` is non-strict by design (copies same-name same-shape tensors and prints the count): logs must say
`191/228 tensors copied` (arm) or `191/192` (control). All 23 jobs printed exactly that.
`train2.py` uses OneCycleLR over `--epochs` (warm-up to `--lr` over the first 30 %, anneal to ~0); it logs every 5 epochs.

## 5. Jobs launched (all Ascend A100 `nextgen`, a0130 excluded, 1 GPU, 12 cores, `-t 01:30:00`; all RUNNING at 16:28)
| wave | name | job id | | wave | name | job id |
|---|---|---|---|---|---|---|
| A | ffvit_t1 | 7566615 | | L | ffvitL_t1 | 7566669 |
| A | ffvit_t2 | 7566616 | | L | ffvitL_t2 | 7566670 |
| A | ffvit_t3 | 7566617 | | L | ffvitL_t3 | 7566671 |
| A | ffvit0_t1 | 7566618 | | L | ffvit0L_t1 | 7566672 |
| A | ffvit0_t2 | 7566619 | | L | ffvit0L_t2 | 7566673 |
| A | ffvit0_t3 | 7566620 | | L | ffvitL_m1 | 7566674 |
| A | ffvit_m1 | 7566621 | | L | ffvitL_m2 | 7566675 |
| A | ffvit_m2 | 7566622 | | L | ffvitL_m3 | 7566676 |
| A | ffvit_m3 | 7566623 | | L | ffvitL_m4 | 7566677 |
| A | ffvit_m4 | 7566624 | | L | ffvit0L_m1 | 7566678 |
| A | ffvit0_m1 | 7566625 | | L | ffvit0L_m2 | 7566679 |
| A | ffvit0_m2 | 7566626 | | smoke | smoke (1 epoch, done, OK) | 7566614 |

Logs `$S/logs/rp_f27a_<name>_<jobid>.log`; a finished job ends with `rp_f27a_<name> DONE` and `saved .../<name>.pth`.
**Checkpoints written: `$K/<name>.pth` for the 23 names above** (73 MB each, fp32); init files in `$K/init/`.
GPU budget: 23 jobs x <= 1.5 h = <= 34.5 GPU-h (of 150).
Watch: `grep -E "epoch +[0-9]+ " <log> | tail -2` (logged every 5 epochs; the number in parentheses is cumulative
seconds). A job killed by the time limit leaves NO checkpoint — then use the wave-A member of the same name.
Job-id list as submitted: `$S/notes/design_a_jobids.txt`.

## 6. Evaluation (executor) — held-out twins only
Instrument: `$S/jobs/rp_f27a_eval.sh` (ens_v9 at `--alpha 0.80`, AoA 15-20 and the full split, each set with its own band;
plus compare_v9 singles). It skips sets with a missing checkpoint. The wave is chosen by env `WAVE` ("" = wave A,
"L" = wave L; wave L has control twins t1, t2 only, so X_CTL is skipped there). Submit once the wave-A twins have
finished (~17:15) and again for wave L (~17:36):
```bash
DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
for WV in "" L; do   # one at a time, when that wave's twins exist
sbatch -A PAS1186 -M ascend -p nextgen ${DESK:+--exclude="$DESK"} --gpus-per-node=1 -c 12 -t 00:40:00 -J rp_f27a_eval$WV \
  --export=ALL,WAVE=$WV --chdir=$S/wt_a --output=$S/logs/rp_f27a_eval${WV}_%j.log $S/jobs/rp_f27a_eval.sh
done
```
A rehearsal of this script on the 1-epoch smoke checkpoint (WAVE=S, job 7566683) is in `$S/logs/rp_f27a_evalS_7566683.log`.
Twins ensembles, and the shipped ensemble each one stands for:
| tag | twins ensemble (evidence, `RPDE_M`) | shipped ensemble (`RPDE_A`) | invocations if hybrids unfused / fused |
|---|---|---|---|
| BASE | p2_ffno_l8_sim_s1,s2,s3 + arch_vit_d384 + arch_vit_cmb3sim | the current 6 members | 2 |
| Y_0 / X_0 | `ffvit0Z_t1, ffvit0Z_t2, p2_ffno_l8_sim_s3` / `ffvit0Z_t1..t3` — **epoch-0 references**: `$K/ffvit0Z_t*.pth` hold the UNTRAINED merge of the parents (= 1/3 F-FNO + 2/3 ViT), so Y_0 is Y's ensemble before any fine-tuning | — | — |
| **Y (primary)** | `ffvit_t1, ffvit_t2, p2_ffno_l8_sim_s3` | `ffvit_m1, ffvit_m2, ffno_all_w40l8_sim_s3, ffno_all_w40l8_sim_s4` | 3 / 2 — same FLOPs as today |
| Y_CTL | `ffvit0_t1, ffvit0_t2, p2_ffno_l8_sim_s3` | `ffvit0_m1, ffvit0_m2, ffno_all_w40l8_sim_s3, ..._s4` | 3 / 2 |
| X | `ffvit_t1, ffvit_t2, ffvit_t3` | `ffvit_m1..m4` | 4 / 1 — two more ViT trunks of FLOPs (~ +1 board ms) |
| X_CTL | `ffvit0_t1..t3` | (no control members m3, m4 were trained; X_CTL is evidence only) | — |
| Z | BASE + `ffvit_t1, ffvit_t2` | current 6 + `ffvit_m1, ffvit_m2` | 4 / 3 — only if the gain is large |
At step 0, Y's shipped ensemble weights are F-FNO 2/3, ViT 1/3 in total — the same family balance as today's package.
Uniform member averaging (what `predict_fno` does; each unfused hybrid has weight 1) is assumed everywhere.

Read three differences for each structure: `Y_0 - BASE` (what the re-weighted structure alone costs or buys),
`Y_CTL - Y_0` (joint fine-tuning), `Y - Y_CTL` (the cross-stitch). The epoch-0 references are being scored now by job
`rp_f27a_evalZ` (`$S/logs/rp_f27a_evalZ_*.log`, WAVE=NONE: only BASE, Y_0, X_0 run).
**Epoch-0 references, measured 16:31 (job 7566719, ens_v9 alpha 0.80, held-out twins; compare deltas only):**

| set | AoA 15-20: relL2 | TKE | MVPE | sps | mean4 | final-proxy | FULL: relL2 | TKE | MVPE | sps | final-proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE (5 twins) | 0.1009 | 0.4404 | 0.0880 | 38.53 | 75.695 | 79.427 | 0.0750 | 0.4546 | 0.0658 | 44.99 | 81.873 |
| Y_0 | 0.1010 | 0.4400 | 0.0881 | 38.50 | 75.688 | 79.417 | 0.0752 | 0.4545 | 0.0660 | 44.96 | 81.859 |
| X_0 | 0.1021 | 0.4436 | 0.0887 | 38.24 | 75.559 | 79.300 | 0.0762 | 0.4599 | 0.0673 | 44.68 | 81.730 |

So **structure Y starts level with BASE (-0.010 / -0.014)**: everything Y gains or loses after fine-tuning is the
fine-tune. Structure X starts 0.13 behind (it is ViT-heavy) and also costs FLOPs — X needs a much larger gain.

**Decision rule (write the numbers into `result_a.md` before deciding):**
1. Candidate = the best of {Y, X, Y_CTL} x {wave A, wave L} on ens_v9 **AoA 15-20** `final`/`mean4`, tie-broken by the full split.
2. **GO** only if candidate - BASE >= **+0.20** on AoA 15-20 (the proxy over-reads ~2x, so this is ~+0.10 board) **and**
   candidate - BASE >= 0 on the full split **and** its containment proxy / `sps` column is not below BASE's.
   If the hybrids must ship unfused, add the invocation cost: require >= +0.35 (T1); T2 runs inside the CUDA graph, measure it.
3. If the control matches or beats the arm (|arm - ctl| < 0.05), the cross-stitch is not the cause: ship the control
   variant if it passes rule 2 (it is simpler and its members exist for Y), and say so in the result.
4. **KILL** (report and stop, do not build): no candidate passes rule 2 by 18:25; or any member's last logged `field`
   loss is above its epoch-0 value (diverged fine-tune — drop that wave); or a gate does not print SUBMITTABLE;
   or the packed (int8/fp16) package differs from the fp32 twins read by more than 0.02 in any subscore.
5. Optional OOD read for the private set if time allows: `Track_1/aoa_holdout_eval.py` is NOT applicable (twins were not
   trained with an AoA held out) — skip it; say "no OOD evidence" in the result.

## 7. Inference integration
**Done in `wt_a` (tested on CPU with the smoke checkpoint):**
* `Track_1/submit_fno/_fno.py`: vendored `FFViT2d` (built from the vendored `FFNO2d` and `ViT2d`, placed just above
  `fuse`), `FFVIT_ARCHS`, and a `load_any` branch (`arch in FFVIT_ARCHS`) that always builds the vendored class.
  Vendored vs training forward: **0.0** difference; `set_spectral_path("dense")` vs fft 2.0e-6. `fuse()` needs no change
  to run: hybrids fall into the "unknown family" branch and come back **unfused, one invocation each**.
* So every instrument (`ens_v9`, `compare_v9`, `fit_band`) and `build_round13.sh` already load `ffvit`/`ffvit0` checkpoints.

**To do by the executor while training runs (in this order; stop at the first step time does not allow):**
1. **Packed round trip**: `pack_state_dict(sd, fp16=True, int8=True, linear="tensor")` -> `dequant8` -> `load_any` on
   `$K/smoke.pth`; compare the field with the fp32 model on 96 val windows (expect rel diff < 1e-3, as for ViT). Key names
   are prefixed `f.`/`v.` but the packer keys on shape/size only, so no rule should change. Then fp16 autocast forward
   (`AMP="fp16"`), and a T2 batch-1 CUDA-graph capture (`Track_2/validate_t2.py` on a Y package does both).
2. **Rehearse the build with wave-A names as soon as they exist** (or with `smoke.pth` copied to four names) following
   `$S/notes/BUILD_RUNBOOK.md` §2 with `RPDE_A` / `RPDE_M` of row Y and absolute paths; band is refitted (no `RPDE_BAND`).
3. **Fusion — `StackedFFViT` (one invocation for K hybrids)**, new class in `_fno.py`, composed of existing parts:
   * `self.sf = StackedFFNO([m.f for m in members])`, `self.sv = StackedViT([m.v for m in members])`.
   * Refactor `StackedViT.forward`'s loop body into `def _block(self, i, t)` (pure move, bit-identical) and reuse
     `sf.lift0/lift2/layers[i]/norm_out/hf/hl`, `sv.embed/pos/_lin/no_w/no_b/pj_w/pj_b/head_*`.
   * State: `h` member-major `(B, K*40, 36, 72)`, `t` `(K, B, 128, 384)`. After layer i in links:
     `f2v`: per-member LayerNorm over channels of `h[..., :32, :64]` (`_StkLN` on `(K,B,H,W,40)`), then ONE grouped conv
     `Conv2d(K*40, K*384, 4, stride=4, groups=K)`, GELU, grouped 1x1 conv -> reshape `(B,K,384,128)` -> `(K,B,128,384)`;
     `v2f`: `F.layer_norm` + per-member affine, two `sv._lin` einsums with stacked `(K,out,in)` weights, unpatchify to
     `(B, K*40, 32, 64)` member-major, `F.pad` to 36x72. Compute both deltas from the PRE-link `h`, `t`, then add (as the
     member does).
   * Heads: take the PER-MEMBER fields before the group mean: `field = mean_k(s_k*field_f_k + (1-s_k)*field_v_k)`,
     `sigma = mean_k(s_k*exp(clamp(ls_f_k,-8,2)) + (1-s_k)*exp(clamp(ls_v_k,-8,2)))`, return `(field, log sigma)`,
     set `n_members = K` (the `StackedFFNO` contract, so `predict_fno` weights the group by K).
   * In `fuse()` add `elif type(g[0]).__name__ == "FFViT2d": out.append(StackedFFViT(g).to(dev).eval())` BEFORE the
     generic `elif not isinstance(g[0], FNO2d)` branch. Verify against the member loop: fp32 field diff <= 2e-5 on CPU and GPU.
   * If fusion is not ready by 18:10, ship unfused and charge it in the prediction (-0.08 T1 per extra invocation).
4. Gates: `submit_fno/validate_strict.py` (T1) and `Track_2/validate_t2.py --adaptive` + `stream_eval.py --causal-check`
   must print SUBMITTABLE / CAUSALITY OK. Archive size of Y: 2 x ~20 MB + 2 x ~1.5 MB int8 — far below 256 MB.

The model is a pure function of window k's own input (no state, no cross-window statistic): leak rules are untouched.

## 8. Code changed in `wt_a` (nothing committed; existing architectures bit-identical — only additions)
* `Track_1/unet/models_alt.py`: `FFVIT_ARCHS`, class `FFViT2d` (before `build_wave1`), dispatch in `build()`.
* `Track_1/unet/train2.py`: `--arch` choices gain `ffvit`, `ffvit0`.
* `Track_1/unet/make_ffvit_init.py` (new): merges two parent checkpoints into an `--init` file.
* `Track_1/submit_fno/_fno.py`: §7.
* `Track_1/val_seen.npz`, `val_unseen.npz`: symlinks to the main repo (for the instruments).
Smokes passed: shapes (B,40,32,64)/(B,2,32,64); params 18.339 M / 15.972 M; step-0 parent equivalence 4.8e-7;
gradients reach the zero-init link layers and `mix`; strict state-dict round trip; 1-epoch GPU job trains, saves and
reloads through `_fno.load_any` (links non-zero after one epoch, s = 0.335).

## 9. Reproducibility recipe (from the release, no external data or weights)
1. Stage 1: train the parents exactly as REPRODUCE.md — F-FNO members §1.4, F-FNO twins §1.5, the two ViTs (members
   `arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3`; twins `arch_vit_d384`, `arch_vit_cmb3sim`) per their §1.x entries.
2. Stage 2: for each row of §3, `make_ffvit_init.py <F parent> <ViT parent> init.pth`, then the §4 command with that
   row's seed and split, `--arch ffvit --epochs 50 --lr 1e-4` (wave A) or `--epochs 62 --lr 2e-4` (wave L).
3. Band: `fit_band.py` on the twins ensemble of the shipped row (§6), ALPHA 0.80, `--band-p 0.45`, as the build does.
Stage 2 costs ~0.9-1.25 A100-hours per member. Both stages use only train_real + train_sim.

## 10. Notes for the executor
* Do not edit `Track_1/unet/models_alt.py` or `train2.py` in `wt_a` (the training recipe of record is what is on disk
  now). Edit `_fno.py` in a copy and install it with an atomic `mv` after `ast.parse`: eval jobs import it at start.
* Early training signal (epoch 5 of 50, training loss with dropout, arm vs control, same seed and parents):
  field 0.03689 vs 0.03711 (t1), 0.03971 vs 0.03991 (t2), 0.03658 vs 0.03679 (m1) — the links move the loss by ~0.5 %,
  consistently. Small; the ensemble read decides.
* `s = sigmoid(mix)` is stored in each checkpoint (`model_state_dict["mix"]`); report it per member (init 0.3333).
* ens_v9's `final` column uses round-9 weights and its own `time_score`: use DELTAS against BASE in the same table only,
  and convert with the BRIEF prices (rel error -1 % = +0.034, tke -1 % = +0.028, dC +0.01 = +0.18) for the prediction.
* If only wave A passes and wave L is not finished by 18:05, decide on wave A; do not wait.
* Job `rp_f27a_reh` (7566718, worktree `wt_a_bld`, script `rp_f27a_build.sh`) and the files `rp_f27a_rt.py`,
  `rp_f27a_time.py` were NOT created by the design agent (they appeared at 16:28-16:29, presumably the executor's build
  rehearsal). The design agent did not touch them.
* Pipeline rehearsal on the 1-epoch smoke hybrid (compare_v9, AoA 15-20, single model): hybrid relL2 0.1055 / wfinal
  82.857 vs F-FNO twin s1 0.1046 / 82.878 vs ViT twin 0.1080 / 82.582 — one hybrid is already a 2-member ensemble.
