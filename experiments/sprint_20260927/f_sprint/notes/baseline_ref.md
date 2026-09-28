# Baseline reference reads — current system (recipe A / r38), 09-27 (X1)

Jobs: rp_f27x_ref 7566573 (ens_v9, fit_band), rp_f27x_build 7566572 (twin T2 stream reads), rp_f27x_candref 7566633 (template
re-run on the baseline, see bottom). Logs: `$S/logs/rp_f27x_{ref,build,candref}_*.log`. All reads use the **5 held-out twins**
`p2_ffno_l8_sim_s1..s3, arch_vit_d384, arch_vit_cmb3sim` (val_unseen = 8 trajectories never trained on). Every local number is a
sign, not a size. **Compare a candidate PAIRED: same script, same split, same flags.**

## 1. T1 ensemble screen — `Track_1/ens_v9.py` (per-set optimal band, CENTRED ON THE PREDICTION, errors ×f)
ens_v9 has no `--band-p` / centre0 option: its band is the per-element optimal band fit on val_seen AoA 15–20 at the stretch
`--f`, centred on the alpha-0.80 prediction. Errors columns are raw (unstretched); sps and final use the ×f stretch;
`final` uses `final_v9` weights with time 90.507 fixed. Use it for rel/tke/mvpe deltas and a relative sps sign.

```bash
cd $W/Track_1
TW=unet/p2_ffno_l8_sim_s1.pth,unet/p2_ffno_l8_sim_s2.pth,unet/p2_ffno_l8_sim_s3.pth,unet/arch_vit_d384.pth,unet/arch_vit_cmb3sim.pth
$PY ens_v9.py --sets "$TW" --alpha 0.80 --f 1.39 --aoa 15 20            # and --f 1.29; full split: --aoa 0 5 10 15 20
```
| split | f | rel_l2 err | tke err | mvpe err | sps | mean4 | final (proxy) |
|---|---|---|---|---|---|---|---|
| AoA 15–20 (126 win) | 1.39 | 0.1009 | 0.4404 | 0.0880 | 38.53 | 75.695 | 79.427 |
| AoA 15–20 | 1.29 | 0.1009 | 0.4404 | 0.0880 | 40.28 | 76.668 | 80.242 |
| full (336 win) | 1.39 | 0.0750 | 0.4546 | 0.0658 | 44.99 | 77.913 | 81.873 |
| full | 1.29 | 0.0750 | 0.4546 | 0.0658 | 46.66 | 78.824 | 82.614 |

## 2. Shipped band construction on twins — `fit_band.py` (σ-scaled, p 0.45, centre0, fill; A = twins too)
```bash
$PY Track_1/submit_fno/fit_band.py --ckpt $M --err-ckpt $M --alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored --inflate 1.39 --out <scratch>.npz
```
| inflate | sps val_seen (fit) | board proxy val_unseen AoA 15–20 @1.00× / 1.05× / 1.15× / 1.30× |
|---|---|---|
| 1.39 | 52.16 | 47.03 / 45.91 / 43.74 / 40.65 |
| 1.29 | 52.41 | 47.11 / 45.94 / 43.68 / 40.46 |
Caveat printed by the tool: these sps lines score the band centred on the prediction, not the re-centred (centre0) band. For
the true shipped band, use §3 (the package path, static = T1-equivalent compute + band).

## 3. T2 stream (and the centre0 band as shipped) — `Track_2/stream_eval.py` on a TWIN package
Package = the 5 twins packed with `build_t2.py`, band = the shipped band (`band_r38_a080_c0_fill.npz`, fit `--ckpt A(all-data)
--err-ckpt twins`, 0.80, p 0.45, centre0, fill, 1.39), baked asnp constants (ALPHA 0.80, fp16, bmm, dense, numpy out), then
`R1/R1_ASYM/D1C_N0` switched by `sed` in the extracted copy. Not stretched (1.00×). `time` column meaningless (FORCE, shared).

| variant | split | rel err | tke err | mvpe err | coverage | rel | tke | mvpe | sps | final (local pred) |
|---|---|---|---|---|---|---|---|---|---|---|
| static band (R1 off) | all 8 | 0.0751 | 0.4546 | 0.0658 | 0.920 | 96.383 | 81.480 | 96.815 | 53.984 | 85.725 |
| static band | AoA 15–20 | 0.1009 | 0.4406 | 0.0880 | 0.877 | 95.197 | 81.947 | 95.784 | 47.860 | 83.580 |
| asym R1 (asnp) | all 8 | 0.0751 | 0.4546 | 0.0658 | 0.905 | 96.383 | 81.480 | 96.815 | 55.594 | 86.120 |
| asym R1 | AoA 15–20 | 0.1009 | 0.4406 | 0.0880 | 0.862 | 95.197 | 81.947 | 95.784 | 48.481 | 83.732 |
| asym R1 + D1c (N0 30) | all 8 | 0.0746 | 0.4546 | 0.0646 | 0.905 | 96.403 | 81.480 | 96.869 | 55.646 | 86.149 |
| asym R1 + D1c | AoA 15–20 | 0.1007 | 0.4406 | 0.0874 | 0.862 | 95.206 | 81.947 | 95.813 | 48.484 | 83.738 |

Deltas (all 8): asym R1 vs static sps +1.61 (final +0.395 local); D1c on top +0.029 final (matches Jonas' +0.028);
AoA 15–20: R1 +0.62 sps, D1c +0.006 final. The static row is the T1-equivalent read of the shipped centre0 band.

## 4. How to read a CANDIDATE paired against this
One job does all of §1–§3 for any member set (fits its own band with the shipped construction, packs its twins with the asnp
constants, runs static / asym / asym+D1c streams):
```bash
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927
DESK=$(squeue -M ascend -u "$USER" -h -o '%N %j' | awk '$2 ~ /bc_desktop|ondemand/ {print $1}' | paste -sd,)
sbatch --export=ALL,CAND_A="<all-data members>",CAND_M="<held-out twins>",TAG=<tag>,W=$S/wt_<dir> ${DESK:+--exclude=$DESK} \
   --chdir=$S/wt_<dir> --output=$S/logs/rp_f27<dir>_candref_%j.log $S/jobs/rp_f27x_candref.sh
```
~4–5 min on A100. Your worktree W needs the ckpts (absolute paths are fine) and `Track_1/val_{seen,unseen}.npz` symlinks
(BUILD_RUNBOOK §0.4). Requires the candidate's members to load through `_fno.load_any` in W. Outputs land in `$S/pkg/candref_<tag>/`.
Compare against the baseline run of the same script (TAG=baseline, log `rp_f27x_candref_<id>.log`), not only the tables above.
**Evidence = twins only**: the build's validate numbers on all-data packages (rel 0.0608, sps ~59) are contaminated.

**Template verified (16:25):** `rp_f27x_candref.sh` with TAG=baseline (job 7566633, log `$S/logs/rp_f27x_candref_7566633.log`)
reproduced every number in §1 and §3 exactly (same band construction; the fit_band board-proxy line with A=all-data reads
51.44 / 50.39 / 48.34 / 45.33, contaminated centre). Wall time 216 s.
