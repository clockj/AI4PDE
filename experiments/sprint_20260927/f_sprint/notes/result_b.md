# Result B — cross-fitted band base (XF), executor EB

**Updated 17:43 EDT — STATUS: VERDICT = KILL. No w passes design_b §7 (hi at f=1.29: best ΔC×100 +0.180 with only 1/3 trajectories positive; criterion ≥ +0.20 and 3/3). Nothing shipped; keep the fallback band.**

## VERDICT: KILL (design_b §7 applied literally; thresholds not moved, nothing else tuned)
Real run: job 7567362 (`$S/logs/rp_f27b_realprobe_7567362.log`, copy `$S/logs/b_xf_eval.log`), 17:37–17:40. All 10 fold models, 150 epochs,
provenance guard OK on all 10 (fold k + 8 val in config['exclude'], all_data False). OOF rel per fold (F-FNO+ViT pair, AoA 15–20):
0.092 / 0.093 / 0.095 / 0.104 / 0.099 (expected 0.10–0.14: sane, no fold dropped). 1231 OOF windows / 30 trajectories.
k per LOTO eval trajectory: 0.951 ×5, 0.848 (21600_15), 1.008 (8850_20, 11400_20) — never on the grid edge (0.4 / 1.6).

Paired LOTO vs V0 (ΔC×100, (n/m) = trajectories with ΔC > 0). V0: hi C 63.562 / 59.510 / 58.075, all 69.423 / 66.412 / 65.276 at f 1.00 / 1.29 / 1.39.
| w | hi f1.00 | hi f1.29 | hi f1.39 | all f1.00 | all f1.29 | all f1.39 | §7 |
|---|---|---|---|---|---|---|---|
| 0.25 | +0.048 (1/3) | +0.129 (1/3) | +0.152 (2/3) | +0.042 (6/8) | +0.102 (6/8) | +0.121 (7/8) | FAIL (hi 1.29 < 0.20, 1/3) |
| 0.5  | +0.025 (1/3) | +0.180 (1/3) | +0.229 (2/3) | +0.037 (6/8) | +0.150 (6/8) | +0.189 (7/8) | FAIL (hi 1.29 < 0.20, 1/3) |
| 0.75 | −0.069 (1/3) | +0.161 (1/3) | +0.235 (1/3) | −0.015 (4/8) | +0.146 (6/8) | +0.204 (6/8) | FAIL |
| 1.0  | −0.230 (1/3) | +0.073 (1/3) | +0.173 (1/3) | −0.113 (3/8) | +0.093 (5/8) | +0.167 (6/8) | FAIL |
Δsps (hi, f 1.29): +0.092 / +0.129 / +0.116 / +0.054. Reading: the mean gain on the AoA 15–20 split is carried by ONE of the three
trajectories (1/3 positive at f 1.29 for every w); on all 8 the sign is mostly positive (6/8) but small (+0.10…+0.15 C×100 ≈ board
ΔC +0.001–0.0015 at best ≈ +0.02–0.03 final, inside the noise the rule was written to reject). Per the pre-registered rule: **KILL,
ship the unchanged band**. No band_xf / packages were produced for the real run (the rehearsal artefacts under `$S/pkg/b/rehearsal/`
are plumbing tests with 2-epoch models — DO NOT SHIP them).
If the commander nevertheless wants the band (against §7): `cd $S/wt_b/Track_1/probe_b && B_CKPT=$S/ckpt/b $PY probe3_xfit.py ship --w 0.5
--band-in <band.npz> --out $S/pkg/b/band_xf_w0.5.npz` (38 s, CPU), then `sbatch --export=ALL,BAND=...,TAG=f27b_xf,OUTD=$S/pkg/b
$S/jobs/rp_f27b_build.sh` (~5 min incl. gates and paired stream reads). Not recommended.

## Monitoring (real fold jobs 7566725–34, limit 01:25:00)
* 16:38: all 10 RUNNING. **Exclusion verified** in logs: "train: 59 trajectory-phases -> 9550 windows, 23 held out" = fold_k (15) ∪ VAL_TRAJ (8); the logged exclude list = folds.json fold k + val. train2.py applies `--exclude` at load_trajectories (not only the loss) and passes it to the sim map (`real_exclude=excl`, `--sim-exclude-held`). The checkpoint's `config['exclude']` carries the list (checked on rehearsal ckpt: fold 2 = its 15 + 8 val, all_data False).
* 16:40: F-FNO epoch 10, 27 s/epoch → 150 ep ≈ 67 min → ends ≈ 17:39 (limit 17:56). ViT epoch 30, 13.3 s/epoch → ends ≈ 17:05. Losses falling on all 10.

* 16:53: F-FNO epoch 45 (loss 0.86–0.87, 24.9 s/ep → end ≈ 17:34); ViT epoch 95–100 (loss 0.56–0.57, 12.9 s/ep → end ≈ 17:04). All healthy, all well inside 01:25.
* 17:17: F-FNO epoch 100–105, ETA ≈17:37. Real dump+eval job queued with afterany dependency on the 5 F-FNO jobs (rp_f27b_realprobe).
* 17:05: all 5 ViT fold models finished (150 epochs, final loss 0.539–0.546). F-FNO epoch ~80.

## Early read — ViT-only OOF (informational; NOT the spec, which needs both families; job 7567231, `$S/logs/rp_f27b_earlyvit_7567231.log`)
Provenance guard OK on all 5 (fold k + 8 val excluded, 150 epochs). OOF rel per fold 0.098 / 0.097 / 0.099 / 0.109 / 0.104 (sane, ~0.10). k 0.951 (0.848 for 21600_15; no edge).
| w | hi ΔC f1.00 | hi ΔC f1.29 (n/3) | hi ΔC f1.39 | all ΔC f1.29 (n/8) |
|---|---|---|---|---|
| 0.25 | +0.057 | +0.137 (1/3) | +0.159 | +0.069 (5/8) |
| 0.5 | +0.028 | +0.183 (1/3) | +0.234 | +0.069 (2/8) |
| 0.75 | −0.084 | +0.151 (1/3) | +0.220 | +0.003 (2/8) |
| 1.0 | −0.276 | +0.031 (1/3) | +0.130 | −0.129 (2/8) |
Mean hi gain is carried by one trajectory; would FAIL §7 (needs 3/3 at f 1.29). The decision is made on the full 10-model run only.

## Rehearsal (stand-ins: same script, --epochs 2, out $S/ckpt/b/rehearsal/, jobs 7566746–55 + 7566757)
* `dump`: 5 folds × 2 models, 1231 OOF windows over 30 AoA 15–20 trajectories (252/252/223/252/252 per fold). OOF rel 0.22–0.32 (2-epoch models; meaningless). ~1 min on A100.
* `eval`: V0 row reproduces design_b's table exactly (hi C 63.562 / 59.510 / 58.075; all 69.423 / 66.412 / 65.276). ~1.5 min total after the fix below.
* `ship` (login CPU, 38 s): keys/shapes/dtypes identical to input band (asserted), base finite > 0; 960/81920 elements have no scored OOF sample (kept = shipped base).
* Build with rehearsal band (job 7566860, 4.5 min, log `$S/logs/rp_f27b_rehbuild_7566860.log`): band in package byte-identical to the shipped band file;
  T1 `[rpde-t1] load: ... gpu_band True direct_pred True alpha 0.8 centre_alpha 0.0 small_graph lazy members 6` → validate_strict **SUBMITTABLE**;
  T2 `[rpde-t2] load: ... r1 True r1_asym True d1c_n0 30.0 ... graph CAPTURED ... numpy_out True members 6` → in-build gate NOT SUBMITTABLE on "stateless" only (expected with D1c), `--adaptive` **SUBMITTABLE**, **CAUSALITY OK** (176 steps, 0 diff).
  Worktree restored (`git status`: only `?? Track_1/probe_b/`).
* Paired twin stream reads: the fallback-band rows reproduce baseline_ref §3 exactly (static 53.984 / 47.860, asym 55.594 / 48.481, asym+D1c 55.646 / 48.484), so the instrument is the same.
  **Caveat for the real run:** the fallback band's base was FITTED on the twins' errors of these very val_unseen AoA 15–20 trajectories, so this stream read is in-sample for the fallback and biased AGAINST XF on `hi`; only the LOTO table (`eval`) is a fair comparison. The stream read is reported for the T2 R1 interaction only.

## Probe-code fixes / additions (probe3_xfit.py; original saved as notes/probe3_xfit.orig.py)
1. `eval`: bt (twin LOTO base) and k do not depend on w — fitted once per trajectory instead of 4× (speed); per-trajectory k printed with an explicit "K ON GRID EDGE -> KILL" flag (design_b §7 criterion was not checked by the code).
2. `ship`: k grid-edge warning; elements with NO scored OOF sample keep the shipped (fill-unscored) base instead of the solver's arbitrary value (the solver returns the smallest error there → log-blend would shrink the fill); channel 2 copied from input; post-write assertion that keys/shapes/dtypes equal the input band and every non-base key is bit-identical; prints the new/shipped ratio percentiles.
3. `dump`: provenance guard — for every fold checkpoint, assert `config['all_data']` is False and fold k's trajectories AND VAL_TRAJ ⊆ `config['exclude']` before its OOF prediction (makes an in-sample leak impossible, not just unlikely).
4. New job scripts: `$S/jobs/rp_f27b_xfreh.sh` (xfit with OUTDIR), `rp_f27b_probe.sh` (dump/eval on GPU via sbatch; srun -M fails "Hostlist is empty"), `rp_f27b_build.sh` (build with a given band + gates + causal + paired twin stream reads XF vs fallback band).
