# Design B2 — Track 2 band under the stream, OOD-like regime: TSM + BLK on top of asym R1.  2026-09-27

Owner: design agent B, phase 2.  `S=/fs/scratch/PAS1187/qwu/tmp/realpde/final0927`.  Worktree `$S/wt_b2` (detached 55f0e88 +
wt_c's `submission.py` / `build_round13.sh` / `stream_c/` as copied 16:41 EDT; the as-copied files are kept in
`$S/ckpt/b2/{submission_c_asof.py,build_round13_c_asof.sh}`).  Patch: `$S/notes/b2.patch`.  Probe code:
`$S/wt_b2/Track_2/stream_b2/{b2prep,b2lab,b2centre,b2report,b2table,b2unit}.py`.  Results: `$S/ckpt/b2/res_*.json`.

## 0. Verdict
* **Re-tuning R1 itself is a null, in all three regimes.** Multiplier range, N0, output smoothing, a global prior, per-frame
  structure: out of distribution no variant is worth more than +0.010 composite; in distribution they read between -0.02 and
  +0.035 with split trajectory signs (table §3). The team's in-distribution choice of R1's constants is also the right one
  out of distribution and at f = 1.29.
* **A faster / separate shift of the interval CENTRE is negative everywhere** (0/16 OOD, 0/3 AoA 15–20). Asym R1 already absorbs
  the centre bias better than a direct shift does. Closed; the patch I wrote for it is withdrawn (kept only as
  `$S/ckpt/b2/cshift_DEAD_do_not_apply.patch`).
* **One construction clears the bar, narrowly: TSM + BLK** (§1). OOD **+0.051 / +0.062** composite at f 1.00 / 1.29, **16/16**
  trajectories both; in distribution **+0.023 / +0.023** (8/8), AoA 15–20 **+0.016 / +0.016** (3/3). No regime loses. It changes
  only the bounds; rel / tke / mvpe are bit-identical.
* **Size, honestly: small.** It is +0.02 where we validate and +0.05–0.06 where the board may be. R1's own hindsight ceiling
  (the whole trajectory's table, no shrinkage, not causal) is only +0.06…+0.10 above shipped R1, so no re-parametrisation of a
  per-pixel multiplier can be large; TSM + BLK takes about half of the element-resolution ceiling's causal part.

## 1. Construction (strictly causal, per trajectory, reset at trajectory boundaries)
Shipped asym R1 keeps, per pixel/channel, a K×K table (K = 9 multipliers 0.5…2 of the static half-width h, lower × upper) of the
sps contribution each pair would have earned on the revealed windows; this window's (k_lo, k_hi) = argmax, 3×3 box, shrunk to 1
by n/(n+2). Two additions, same statistic, same revealed windows, nothing else read:

1. **TSM (`R1_TSM = 3`)**: the per-pixel TABLE is 3×3 box-smoothed (replicate-padded, over H, W) **before** the argmax. The
   sufficient statistic is pooled over the 9 neighbours; the argmax is then taken on 9× the sample. Output smoothing and
   shrinkage unchanged.
2. **BLK (`R1_BLK_N0 = 20.0`, `R1_BLK = 4`)**: a second table at lead-time-block resolution, K×K×G×H×W×C with G = 20/4 = 5
   blocks of 4 consecutive output frames (the per-pixel table is its sum over blocks, so there is still ONE einsum per step).
   Per block and pixel: 3×3 box of the table → argmax → 3×3 box of the multiplier = k_blk. Then
   `k = k_pix + (k_blk − k_pix) · n/(n + 20)`, clamped to [0.5, 2.0], constant within a block; n = revealed windows.
   n = 0 → k = 1 exactly (the static band), as today.

Interval, unchanged: `lower = min(C − h·k_lo, P)`, `upper = max(C + h·k_hi, P)`, C = y0 + D1c shift, so `lower ≤ P ≤ upper`
by construction, and k ∈ [0.5, 2] is finite and positive ⇒ bounds cannot be malformed. **Order of operations:** R1's table
scores the centre AFTER the D1c shift and the prediction AFTER D1c and VARCAL (it reads the previous step's emitted `go`, `gC`,
`gH`), which is what the probe does too. Constants: three scalars (3, 20.0, 4), nothing fitted on data files.

Why it helps more out of distribution: there the static base (fitted on 3 in-distribution trajectories) is wrong in a way that
depends on lead time and on position, and the revealed windows of the same flow condition are the right sample for it. The
gain is there from the second window and grows slowly with n (§4).

## 2. What the knobs are NOT
Lead-time resolution itself is worth little: 2 blocks of 10 frames give +0.045…+0.052 OOD against +0.051…+0.062 for 5 blocks
and +0.048…+0.060 for 20 single frames. Most of the gain is the second, table-smoothed, separately-shrunk estimate. G = 5 is
chosen because it costs no more than the shipped update.

## 3. Three-regime table — Δ composite (no time) = 0.2498·Δsps, paired against asym R1 + D1c (N0 30), CPU probe
Dumps: in distribution = 5 held-out twins on val_unseen (8 trajectories, 336 windows); OOD = design C's AoA-20 holdout dump
(16 trajectories, 643 windows, 3 stand-in members, band NOT refitted). Stretch f as in design C's probe. (n/m) = trajectories
with a positive delta. `first 14` = only the first 14 windows of every stream. Baseline sps reproduces `probe.run` to 4 decimals.

| variant | in-dist full f=1.00 | full f=1.29 | AoA 15–20 f=1.00 | AoA 15–20 f=1.29 | OOD f=1.00 | OOD f=1.29 | OOD first 14, f=1.29 |
|---|---|---|---|---|---|---|---|
| asym R1 + D1c30 (sps level) | 55.64 | 50.12 | 48.46 | 42.47 | 44.98 | 39.05 | 38.09 |
| **TSM 3 + BLK 4, N0 20 (proposed)** | **+0.023 (8/8)** | **+0.023 (8/8)** | **+0.016 (3/3)** | **+0.016 (3/3)** | **+0.051 (16/16)** | **+0.062 (16/16)** | **+0.051 (16/16)** |
| TSM 3 + BLK 4, N0 10 | +0.025 (8/8) | +0.023 (8/8) | +0.013 (3/3) | +0.012 (3/3) | +0.055 (16/16) | +0.067 (16/16) | +0.057 (16/16) |
| TSM 3 + BLK 4, N0 30 | +0.021 (8/8) | +0.022 (8/8) | +0.016 (3/3) | +0.017 (3/3) | +0.046 (16/16) | +0.057 (16/16) | +0.047 (16/16) |
| BLK 4 N0 20 without TSM | +0.022 (8/8) | +0.021 (8/8) | +0.013 (3/3) | +0.012 (3/3) | +0.044 (16/16) | +0.053 (16/16) | +0.031 (16/16) |
| TSM 3 alone | +0.004 (7/8) | +0.007 (8/8) | +0.006 (3/3) | +0.010 (3/3) | +0.015 (16/16) | +0.021 (16/16) | +0.031 (16/16) |
| TSM 3 + per-frame table (20 frames), N0 30 | +0.021 (8/8) | +0.022 (8/8) | +0.017 (3/3) | +0.017 (3/3) | +0.048 (16/16) | +0.060 (16/16) | +0.047 (16/16) |
| TSM 3 + global prior (N0g 0.25) | +0.025 (8/8) | +0.013 (6/8) | +0.007 (3/3) | +0.010 (3/3) | +0.017 (16/16) | +0.022 (16/16) | +0.034 (16/16) |
| R1 N0 1 (shipped 2) | n/a | +0.002 (3/8) | n/a | −0.005 (0/3) | +0.005 (15/16) | +0.002 (11/16) | +0.005 (11/16) |
| R1 N0 4 | n/a | −0.009 (5/8) | n/a | +0.004 (3/3) | −0.014 (0/16) | −0.008 (4/16) | −0.018 (3/16) |
| R1 no output smoothing | n/a | −0.019 (1/8) | n/a | −0.022 (0/3) | +0.008 (13/16) | +0.005 (12/16) | −0.006 (5/16) |
| R1 range 0.5…4 (K 13) | n/a | +0.002 (3/8) | n/a | −0.001 (0/3) | +0.002 (16/16) | +0.008 (16/16) | +0.013 (16/16) |
| R1 range 0.25…4 (K 17) + global prior | n/a | +0.014 (4/8) | n/a | −0.003 (1/3) | +0.007 (12/16) | +0.007 (12/16) | +0.017 (11/16) |
| R1 × per-frame factor (pooled over pixels) | n/a | −0.004 (2/8) | n/a | +0.001 (2/3) | +0.005 (14/16) | +0.001 (9/16) | +0.005 (12/16) |
| centre shift N0 10 (P keeps 30) | −0.006 (2/8) | −0.007 (1/8) | −0.010 (0/3) | −0.011 (0/3) | −0.003 (1/16) | −0.007 (1/16) | −0.007 (2/16) |
| centre shift N0 5 | −0.012 (1/8) | −0.015 (1/8) | −0.018 (0/3) | −0.020 (0/3) | −0.010 (0/16) | −0.015 (0/16) | −0.024 (0/16) |
| centre shift N0 2 | −0.022 (1/8) | −0.026 (0/8) | −0.031 (0/3) | −0.034 (0/3) | −0.023 (0/16) | −0.031 (0/16) | −0.061 (0/16) |
| centre shift N0 5 + per-frame bias N0 30 | −0.014 (1/8) | −0.012 (2/8) | −0.023 (0/3) | −0.021 (0/3) | +0.005 (11/16) | +0.023 (14/16) | −0.002 (8/16) |
| *hindsight ceiling, per-pixel multiplier (NOT causal)* | +0.081 | +0.065 | +0.058 | +0.055 | +0.095 | +0.086 | +0.156 |
| *hindsight ceiling, per-element (NOT causal)* | +0.147 | +0.135 | +0.136 | +0.128 | +0.226 | +0.231 | +0.302 |

On top of **D1c N0 10** (same probe): proposed variant +0.022 (8/8) in-dist f 1.29, +0.015 (3/3) AoA 15–20, **+0.049 (16/16)**
OOD f 1.00. The gain does not depend on D1c's N0. "n/a" = that wave was not run on that cache (the in-distribution f 1.00
read of the R1-constant wave exists for D1c N0 10 only; the variants listed above read between -0.021 and +0.035 there,
none with more than 4/8 positive trajectories).

## 4. Gain versus n (proposed variant, Δ composite by index of the window in its stream)
| regime | n = 0 | 1 | 2–3 | 4–7 | 8–13 | 14–27 | 28–41 |
|---|---|---|---|---|---|---|---|
| in-dist f 1.00 | 0 | +0.001 | +0.018 | +0.025 | +0.025 | +0.024 | +0.025 |
| in-dist f 1.29 | 0 | +0.009 | +0.021 | +0.024 | +0.022 | +0.023 | +0.026 |
| OOD f 1.00 | 0 | +0.037 | +0.043 | +0.046 | +0.045 | +0.051 | +0.060 |
| OOD f 1.29 | 0 | +0.057 | +0.057 | +0.055 | +0.054 | +0.063 | +0.073 |

Never negative in any bin; a 14-window stream gets 80–85 % of the full-stream gain. The first window is the static band.

## 5. Real path (twin T2 package, A100, `stream_eval.py`) and latency
See §8 (filled from jobs rp_f27b2_realA 7567309 / realB 7567310, logs `$S/logs/rp_f27b2_real{A,B}_*.log`).
CPU-side checks already done (`b2unit.py`): knobs OFF → the patched scaler returns bit-identical multipliers to the as-copied
class over 30 windows; knobs ON → patched scaler vs the lab's reference: 6e-8 on almost every element, worst 2.1e-2 on a few
elements where a near-tie in the argmax flips with the summation order (block sums vs direct sums).

## 6. Patch and build knobs
`$S/notes/b2.patch` (unified, `-p1`, 7 hunks; applies cleanly to wt_c's CURRENT `Track_2/submit/submission.py` and
`Track_1/submit_fno/build_round13.sh`, checked 17:13 with `patch -p1`; it does NOT apply to repo HEAD, which lacks design C's
VARCAL/FAMW block). New constants default OFF (`R1_TSM = 1`, `R1_BLK_N0 = 0.0`, `R1_BLK = 4`): with them off every changed line
reduces to the old expression. Build variables: `RPDE_SHIP_R1_TSM=3 RPDE_SHIP_R1_BLK_N0=20.0` (`RPDE_SHIP_R1_BLK=4` is the default).
State added: one float32 tensor 9×9×5×32×64×2 = 6.6 MB on the GPU; the multiplier buffers `gk`/`gk2` become (1,20,H,W,2).

## 7. Instruction for executor EC (one paragraph)
In the worktree you build from, run `patch -p1 --dry-run < $S/notes/b2.patch` then `patch -p1 < $S/notes/b2.patch`; build the T2
candidate exactly as you already do for D1c + VARCAL and add `RPDE_SHIP_R1_TSM=3 RPDE_SHIP_R1_BLK_N0=20.0` (R1 and R1_ASYM must be
True, which they are in the asnp stack). Build the SAME stack once more without the two variables as the paired control. Gate both
with `validate_t2.py --adaptive` and `stream_eval.py --causal-check`; then on the TWIN package compare the two with
`stream_eval.py` (all 8 and `--aoa 15 20`): rel / tke / mvpe must be identical to the last printed digit, sps must rise by
about +0.09 (all 8) and +0.06 (AoA 15–20), and the step time must not rise by more than 0.1 ms in an interleaved read. Ship the
patched package only if all of that holds; if the step time rises by more than 0.1 ms, or sps does not rise on the twins, ship
the control. Do not change R1_N0, R1_KS or the smoothing: they were re-tested here and are already at their optimum.

## 8. Real-path results (twin T2 package = the 5 held-out twins + the shipped band, asnp constants, A100, CUDA graph CAPTURED)
Jobs rp_f27b2_realA 7567309, rp_f27b2_realB 7567310 (script `$S/jobs/rp_f27b2_real.sh`), val_unseen, unstretched (f = 1.00).
In distribution only: there is no OOD twin package, so the OOD columns of section 3 exist on the CPU probe alone.

| stack | split | sps OFF | sps ON (TSM 3, BLK 4, N0 20) | delta sps | delta composite | rel / tke / mvpe |
|---|---|---|---|---|---|---|
| asym R1 + D1c 30 | all 8 | 55.646 | 55.737 | +0.091 | +0.023 | identical (96.403 / 81.480 / 96.869) |
| asym R1 + D1c 30 + VARCAL 50 | all 8 | 55.699 | 55.790 | +0.091 | +0.023 | identical (96.366 / 81.772 / 96.869) |
| asym R1 + D1c 30 + VARCAL 50 | AoA 15-20 | 48.516 | 48.576 | +0.060 | +0.015 | identical (95.157 / 82.230 / 95.813) |
| asym R1 + D1c 10 + VARCAL 50 | all 8 | 55.681 | 55.771 | +0.090 | +0.022 | identical (96.368 / 81.772 / 96.873) |

The real path reproduces the CPU probe (+0.023 full, +0.016 AoA 15-20) and the OFF rows reproduce `baseline_ref.md` (55.646).

* **Knobs OFF == the as-copied submission.py, bitwise**: `pred`, `lower`, `upper` IDENTICAL over all 336 windows.
* **Knobs ON**: prediction bit-identical to OFF (band-only change); bounds finite, `lower <= upper`, `lower <= pred <= upper`
  on every element of every window.
* **`stream_eval.py --causal-check`: CAUSALITY OK** on both ON stacks (D1c 30 + VARCAL 50, D1c 10 + VARCAL 50): 176 steps
  bit-identical with every later window replaced by noise, max |diff| 0.
* **Latency: no measurable increase.** Interleaved on one node (shared, so a bracket), mean ms/step of the settled repeats:
  OFF 3.437 / 3.323 / 3.178 / 3.145, ON 3.164 / 3.144 / 3.186 / 3.138. The last interleaved pair reads 3.145 vs 3.138.
  Within the noise of the instrument (about 0.05 ms), so under the 0.1 ms allowance. Not a board number: the board charges
  launches differently (R27), and this adds a handful of kernels on R1's side stream. Bracket for the board: 0 to +0.1 ms,
  i.e. 0 to -0.007 final.
* Caveat on provenance: jobs 7567309/7567310 ran the patched file BEFORE the three new constants were moved to a different
  line (done so that the patch applies to wt_c's current file whatever constants a build has baked). Code is otherwise the
  same. Job rp_f27b2_realA2 7567361 repeats part A on the final file (as-copied + b2.patch == wt_b2's file, byte-identical,
  sha256 31d5b4b6...); its outcome is in the addendum at the end of this file.

## 9. What this is worth, and what it is not
* Predicted board effect: sps +0.09 (if the board behaves like our validation) to +0.25 (if it behaves like the OOD dump at
  f = 1.29), i.e. **final +0.02 to +0.06**, minus 0 to 0.007 for latency. Bracket [0.00, +0.07].
* It does NOT close the gap to the top 10 (about 0.2 on Track 2). It is one small, all-positive, band-only increment that
  stacks on D1c and VARCAL without touching them.
* Evidence limits: the OOD dump is a 3-member stand-in ensemble with a band that was not refitted for it, so its static band
  is worse than ours would be and any online calibration looks better there; the in-distribution reads are 8 trajectories.
  Signs are unanimous (8/8, 3/3, 16/16); sizes are a sign, not a size.
* N0 20 was chosen as the middle of a flat row (N0 10 / 20 / 30: OOD +0.055 / +0.051 / +0.046 at f 1.00; AoA 15-20
  +0.013 / +0.016 / +0.016). It was chosen on these same dumps, so its own selection is in-sample; the row is flat enough
  that this does not change the sign.

## Addendum 17:22 EDT — confirmation on the FINAL file
Job rp_f27b2_realA2 7567361 (log `$S/logs/rp_f27b2_realA2_7567361.log`), twin package, the file that `b2.patch` produces:
* knobs OFF vs the as-copied submission.py: `pred`, `lower`, `upper` IDENTICAL (336 windows); sps 55.646 both.
* knobs ON (TSM 3, BLK 4, N0 20), D1c 30: sps 55.737 (+0.091), rel / tke / mvpe unchanged; prediction bit-identical to OFF;
  bounds finite, `lower <= upper`, `lower <= pred <= upper` everywhere; CUDA graph CAPTURED (graph_worst 0.0).
* knobs ON, D1c 30 + VARCAL 50: `--causal-check` CAUSALITY OK (176 steps bit-identical, max |diff| 0); sps 55.790.
Same numbers as jobs 7567309 / 7567310, so moving the constants changed nothing.

## Files
| what | where |
|---|---|
| patch (apply with `patch -p1` in the build worktree) | `$S/notes/b2.patch` (sha256 6fdb8802...) |
| patched files as tested | `$S/wt_b2/Track_2/submit/submission.py` (sha256 31d5b4b6...), `$S/wt_b2/Track_1/submit_fno/build_round13.sh` |
| as-copied originals (wt_c, 16:41 EDT) | `$S/ckpt/b2/submission_c_asof.py`, `$S/ckpt/b2/build_round13_c_asof.sh` |
| probe code | `$S/wt_b2/Track_2/stream_b2/` (`b2prep.py` cache, `b2lab.py` variants, `b2centre.py`, `b2report.py`, `b2table.py`, `b2unit.py`) |
| cached streams and per-window results | `$S/ckpt/b2/{in,ood}_d{30,10}_f{100,129}.npz`, `$S/ckpt/b2/res_*.json` |
| job scripts | `$S/jobs/rp_f27b2_lab.sh` (CPU labs on compute nodes), `$S/jobs/rp_f27b2_real.sh` (real path) |
| withdrawn centre-shift patch (do not apply) | `$S/ckpt/b2/cshift_DEAD_do_not_apply.patch` |

Re-run the three-regime table: `cd $S/wt_b2/Track_2/stream_b2 && python b2table.py d30` (and `d10`); one variant in detail:
`python b2report.py ood_d30_f129 --set wave4`. Scratch use of this phase: 7.2 GB in `$S/ckpt/b2/` (8 stream caches);
delete after the sprint.
