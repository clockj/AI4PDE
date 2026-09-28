# Design B — containment by CROSS-FITTED band base (XF).  2026-09-27, written 16:35 EDT

Owner: design agent B (containment / uncertainty-native).  Worktree `$S/wt_b` (`S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927`).
Probe code: `$S/wt_b/Track_1/probe_b/{dump_twins,bandlab,probe1_centre,probe2_fitset,make_folds,probe3_xfit}.py`.
Probe logs: `$S/logs/b_probe1.log`, `$S/logs/b_probe2.log`.  Twin dump: `$S/ckpt/b/dump_unseen.npz`.

## 0. Verdict in four lines
* **Frame questions answered NO** (zero-GPU, leave-one-trajectory-out, paired): the band CENTRE, a speed/Re scaling of the
  width, the inflate value and the contain-P rule are all at their optimum. Nothing there to ship.
* **One thing survived**: the per-element `base` is fitted on THREE trajectories and loses 0.36–0.57 C×100 going from in-sample
  to a trajectory it has not seen; adding only two more (off-AoA) fit trajectories already returns +0.06…+0.64 C×100.
* **Arm launched**: 10 fold models (5 folds × F-FNO+sim, ViT) trained FROM SCRATCH → out-of-fold errors on ~30 AoA 15–20
  trajectories the models never saw → `base` refitted on those, level-matched to the twins by one scalar, blended in log space.
* **Inference code is unchanged. Only `band.npz['base']` changes.** No new knob, no latency, no malformed-bound risk.

## 1. Scorer read (facts that constrain any design)
`scoring.py:aggregate_sps`: per element on u,v only, `a_i * exp(-(upper-lower)/0.0563870259)` if `lower <= y <= upper` and
`y != 0`; denominator = number of non-zero-target elements; everything in RAW m/s; `p` never read; the prediction does NOT
have to lie inside the interval (the scorer never compares pred with the bounds; only `lower <= upper` is policed). `a_i` is
per window, from the scored prediction only. Hence interval and prediction are independent objects, and width is charged in
absolute m/s, so fast (high-Re) windows dominate the loss of C. No legitimate structural loophole found.

## 2. Probe table (all LOTO, twins `p2_ffno_l8_sim_s1..s3 + arch_vit_d384 + arch_vit_cmb3sim`, val_unseen, alpha 0.80, p 0.45)
Baseline V0 = shipped construction (centre y0, base on AoA 15–20, inflate 1.39, k3 box, contain-P). For an AoA 15–20 eval
trajectory the base is fitted on the OTHER TWO; for the other five on all three. C×100 = mean containment; `hi` = AoA 15–20
(3 trajectories), `all` = 8. Deltas are paired vs V0; (n/m) = trajectories with ΔC > 0.

V0 level: hi C 63.56 / 59.51 / 58.08 and sps 47.46 / 41.74 / 39.92 at f = 1.00 / 1.29 / 1.39; all C 69.42 / 66.41 / 65.28.
(Board C is 0.599: the LOTO read at f ≈ 1.29 reproduces it, so this instrument is at the right operating point.)

| variant | hi ΔC f=1.00 | hi ΔC f=1.29 | hi ΔC f=1.39 | all ΔC f=1.29 | verdict |
|---|---|---|---|---|---|
| no contain-P | +0.019 (3/3) | −0.003 (2/3) | −0.009 (2/3) | −0.010 (2/8) | null |
| centre m+0.9(y0−m) (shrink fluctuation) | −0.176 (0/3) | −0.123 (0/3) | −0.102 (0/3) | −0.110 (1/8) | closed |
| centre m+0.8 / 0.65 / 0.5 / 0.0 | −0.57 / −1.37 / −2.29 / −5.34 | −0.40 / −1.00 / −1.71 / −4.22 | −0.33 / −0.87 / −1.52 / −3.85 | −0.37 … −3.99 | closed, monotone |
| centre shrink growing with lead time (1→0.8, 1→0.6) | −0.09 / −0.32 | −0.07 / −0.21 | −0.05 / −0.17 | −0.06 / −0.21 | closed |
| centre 0.9·y0+0.1·mean_t(input) ; 0.75/0.25 | −0.43 / −2.08 | −0.28 / −1.49 | −0.23 / −1.29 | −0.27 / −1.39 | closed |
| centre = F-FNO-only mean / ViT-only mean | −0.30 / −0.68 (0/3) | −0.18 / −0.43 | −0.14 / −0.35 | −0.21 / −0.51 (0/8) | closed |
| centre = member median | −0.071 (0/3) | −0.033 (0/3) | −0.019 (0/3) | −0.047 (0/8) | closed |
| centre 3-frame temporal smooth | −0.053 (0/3) | −0.042 (0/3) | −0.035 (0/3) | −0.036 (0/8) | closed |
| width × (U_in/U_ref)^q on non-dim σ, q 0.45 / 0.7 / 1.0 | +0.01 / −0.07 / −1.10 | −0.02 / −0.11 / −1.21 | −0.03 / −0.14 / −1.25 | +0.21 (3/8) / +0.06 / −1.51 | noise, closed |
| shipped ratio × U^0.3 / U^0.55 | −0.05 / −0.87 | −0.07 / −0.96 | −0.10 / −1.00 | +0.22 (4/8) / −0.71 | noise, closed |
| inflate 1.2 / 1.6 / 1.8 | +0.39 / −0.67 / −1.41 | −0.24 / −0.13 / −0.47 | −0.42 / +0.05 / −0.16 | +0.09 / −0.40 / −0.93 | 1.39 is the robust value |
| **base fitted IN-SAMPLE** (the 3 eval trajectories themselves) | +0.357 (3/3) | **+0.535 (3/3)** | **+0.566 (3/3)** | — | = what 3-trajectory fitting loses out of sample |
| **LOTO, fit on the other AoA ≥ 10 trajectories** (4 instead of 2) | +0.283 (3/3) | +0.060 (2/3) | +0.002 (1/3) | **+0.431 (7/8)** | more fit data helps |
| LOTO, fit on all 7 others (any AoA) | +0.354 (3/3) | −0.049 (1/3) | −0.156 (1/3) | +0.464 (5/8) | helps, but off-AoA data narrows the band |

Reading: the ensemble mean y0 is a local optimum of the centre in every direction tried (more members averaged = better
centre: 3 F-FNO −0.30, 2 ViT −0.68, all 5 best). The one live variable is the SAMPLE the base is estimated from: 81 920 free
parameters from 126 windows of 3 recordings. The team's "15 trajectories are worse than 3" read was (as far as I can tell)
scored on the same 3 trajectories the 3-trajectory base was fitted on; under LOTO the sign is the other way.

## 3. Hypothesis
The board's hidden trajectories are not our three. A base estimated from ~30 AoA-matched trajectories, each scored by models
that never saw it (cross-fitting), has a lower-variance shape than one from 3, and recovers part of the 0.36–0.57 C×100
in-sample→LOTO gap. **Subscore moved: sps only** (rel/tke/mvpe/time bit-identical: prediction and code unchanged).
**Predicted ΔC +0.001…+0.004 (board), Δsps +0.07…+0.3, Δfinal +0.02…+0.07; bracket [−0.03, +0.09].** Ceiling +0.005 C.

## 4. Construction to ship
```
half(i,e) = base_xf(e) * clip(sigma_i(e)/sigma_ref(e), 0.3, 3)^0.45          # unchanged inference, centre y0, contain-P
base_xf   = exp((1-w) * log base_tw + w * log(k * base_oof))
```
* `base_tw` = the shipped base (`band.npz['base']` of the package being built: twins' |y0 − y| on val_unseen AoA 15–20).
* `base_oof` = `sps9` exact per-element solver on the out-of-fold |y0_oof − y| of the fold models on the AoA 15–20
  trajectories among the 74 non-validation recordings, ×1.39, deflated by the fold models' own
  `clip(σ/σ_ref_oof, 0.3, 3)^0.45`, box-smoothed k = 3 (same code path as `fit_band.py`, a_i from the OOF field).
* `k` = one scalar (grid 0.4…1.6) that maximises the twins' fit-split sps of `k·base_oof·R` at inflate 1.39. It absorbs the
  level difference between a 2-member 150-epoch fold pair and the 5-member 400-epoch twin set.
* `w` ∈ {0.25, 0.5, 0.75, 1.0}: picked by the LOTO table of `probe3_xfit.py eval`. Constants fitted offline on training
  data only; window k's band depends only on window k's input (σ head) — rule-compliant on Track 1.
* Bounds cannot be malformed: `base_xf > 0`, finite (asserted in `ship`), same shape, and the inference path is untouched.

## 5. Wiring
* `_fno.py`, `submission.py`, `fit_band.py`: **no change** (so "knob off = bit-identical" holds trivially: ship the old band).
* Build: fit the package's band exactly as planned, then
  `python Track_1/probe_b/probe3_xfit.py ship --w <W> --band-in <shipped band.npz> --out $S/pkg/band_xf_w<W>.npz`
  and build with `RPDE_BAND=$S/pkg/band_xf_w<W>.npz` (BUILD_RUNBOOK). All other keys of the npz (sigma_ref, p, clip,
  centre_alpha) are copied through. T2 may use the same band under R1 (R1 rescales online, so expect less there).

## 6. Executor runbook (after the jobs finish, ~17:40–17:50 EDT; total < 20 min, CPU is enough)
```bash
S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927; PY=/users/PAS1187/qwu/.conda/envs/realpde/bin/python
cd $S/wt_b/Track_1/probe_b
ls $S/ckpt/b/xf_*_f?.pth                      # expect 10; a fold with one model still works, a fold with none is skipped
$PY -u probe3_xfit.py dump  2>&1 | tee $S/logs/b_xf_dump.log     # writes $S/ckpt/b/oof.npz (~1 GB); prints OOF rel per fold
$PY -u probe3_xfit.py eval  2>&1 | tee $S/logs/b_xf_eval.log     # V0 + w = 0.25/0.5/0.75/1.0, paired, LOTO
$PY -u probe3_xfit.py ship --w <W> --band-in <band of the chosen package> --out $S/pkg/band_xf_w<W>.npz
```
`eval` is slow in one place only (the V0 fits use `sps9.optimal_band`; ~5 min in total). Run `dump` on a GPU node if one is
free (`srun -M ascend -p debug-nextgen ...`), otherwise on the login node (the twin dump of 336 windows took 20 s on CPU).
Sanity: OOF rel (AoA 15–20, 2 members, 150 epochs) should be 0.10–0.14; if > 0.2 a fold model is broken — drop that fold.

## 7. Kill criteria (written before any result)
* SHIP only if, for the chosen w, `hi` ΔC×100 ≥ **+0.20 at f = 1.29 AND ≥ +0.15 at f = 1.39 with 3/3 trajectories positive at
  f = 1.29**, and `all` ΔC ≥ 0 at f = 1.29 with ≥ 5/8. Pick the smallest w that passes (least departure from the shipped band).
* If k lands on the grid edge (0.4 or 1.6) for any trajectory: the level match failed → kill.
* If no w passes: ship the unchanged band. Do not tune anything else on these 8 trajectories.
* Jobs not finished by 18:20, or fewer than 3 folds usable: kill (the arm needs the trajectory count).

## 8. Jobs launched (16:31 EDT, Ascend A100, `-t 01:25:00`, desktop node a0130 excluded) — list also in `design_b_jobids.txt`
| job | name | what |
|---|---|---|
| 7566725 / 27 / 29 / 31 / 33 | rp_f27b_xf_ffno_f0..f4 | F-FNO w40 L8 + sim, twin recipe, from scratch, 150 epochs, `--exclude fold_k ∪ VAL_TRAJ` |
| 7566726 / 28 / 30 / 32 / 34 | rp_f27b_xf_vit_f0..f4 | ViT d384 L8 plain (VITP recipe), from scratch, 150 epochs, same exclusion |
Script `$S/jobs/rp_f27b_xfit.sh`; folds `$S/wt_b/Track_1/probe_b/folds.json` (74 trajectories, stratified by (AoA, Re),
round-robin, 15/15/15/15/14); outputs `$S/ckpt/b/xf_{ffno,vit}_f{k}.pth`; logs `$S/logs/rp_f27b_xf_*`. ~12 GPU-hours total.
No warm start on purpose: a twin has already seen every non-validation trajectory, so a warm-started fold model's
"held-out" errors would be memorised ones (the R6 failure).

## 9. Reproducibility recipe (for REPRODUCE.md if this ships)
Stage 1 = REPRODUCE.md §1.4/§1.5 unchanged (members, twins, band). Stage 2 = `make_folds.py` → 10 × `rp_f27b_xfit.sh`
(flags in the script; seeds 1; 150 epochs) → `probe3_xfit.py dump` → `probe3_xfit.py ship --w W`. Uses only train_real and
train_sim; no external data, no pretrained weights. The fold models ship no weights — they only supply the error field.

## 10. Honest odds
P(board C moves ≥ +0.005) ≈ **5 %** (that is the ceiling of this lever). P(ΔC ≥ +0.002) ≈ 25 %. P(negative) ≈ 30 %.
This arm cannot by itself close the 0.012–0.025 C gap to the top 10; nothing I probed can.

## 11. Addendum 16:38 EDT — status of the code path, and one warning
* `probe3_xfit.py` was run end to end (`dump` → `eval`) on STAND-IN checkpoints: `$S/ckpt/b_smoke/xf_{ffno,vit}_f0.pth` are
  symlinks to the twins `p2_ffno_l8_sim_s1` / `arch_vit_d384` (logs `$S/logs/b_xf_smoke_{dump,eval}.log`). It runs: 252
  windows / 6 trajectories for fold 0, k = 1.01–1.20 (not at the grid edge), `optimal_band_fast == sps9.optimal_band` bitwise.
* **The smoke rows (w 0.25: hi ΔC +0.12 at f 1.29; w 0.5: +0.16) are NOT evidence and must not be quoted.** The stand-in
  twins TRAINED on fold 0's trajectories, so that "OOF" error field is a memorised one (exactly the R6 failure mode; k > 1 is
  the solver scaling too-small errors back up). Only the run on the real `xf_*_f{k}.pth` in `$S/ckpt/b/` counts.
* `eval` takes ~70 s for V0 plus ~35 s per w on the login node (20 cores); `dump` ~1 min per fold on CPU.
* Measured training rate (16:36): ViT folds ~13 s/epoch (no sim) → done ≈ 17:05; F-FNO folds ~25 s/epoch → done ≈ 17:35–17:40.
  Both well inside the 01:25 limit. The final `.pth` is written only at the END of training (`.resume.pt` is not loadable by
  `_fno.load_any`), so a job killed early leaves no usable checkpoint for that fold/arch.
* Side effect to clean up after the sprint: the five F-FNO fold jobs each built a sim-map cache for their exclusion list,
  `/fs/scratch/PAS1187/qwu/tmp/realpde/work/simmap_p1_f0_x{a672202964,e1d4fbca9b,fac906014d,5500a38dde,b26ceb856c}.npy`
  (1.64 GB each, scratch, 5 inodes). Do not delete while the jobs run; the pre-existing `simmap_p1_f0_x476193e300.npy` is
  the twins' and must stay.
* Optional extension, only if the XF table passes and time remains (NOT implemented, lower priority than packaging): the
  LOTO-7 row says off-AoA fit data narrows the band for AoA 15–20. `dump` currently keeps AoA 15–20 only (`FIT_AOA`);
  widening it to (10, 15, 20) is a one-line change and the AoA ≥ 10 probe row (+0.43 all, 7/8) suggests it is the safer
  fit set for a board whose AoA mix we do not know. Judge it with the same kill criteria.
