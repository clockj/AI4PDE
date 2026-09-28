# Result E0b: SIGBLEND band on Track 2 (SIGPOW knob), 2026-09-27 17:32 EDT

**Verdict: SHIP, T2 only, lam 0.5**: `track2_k27sbt2_a080.zip` (asnp + SIGPOW) and `track2_k27sbt2_d1c_a080.zip` (asnp + D1c30 + SIGPOW).
The ship criterion was "the asym-R1 row gains sps on all 8 and does not lose on AoA 15-20". It is met at both lam values: lam 0.5 gives +0.218 (8/8) on all 8 and +0.067 (3/3) on AoA 15-20.
The gain is small. R1 already recovers about 60 % of the static gain online.

## Code
* Clone `$K/wt_0b`, local commit `beeedbc "T2 SIGPOW knob"` on top of 55f0e88. Nothing was pushed. Patch: `$K/pkg/k0/sigpow_t2.patch`.
* `Track_2/submit/submission.py`:
  * Adds the module constant `SIGPOW = None`.
  * In `__init__`, sets `self.sigpow` to a float32 tensor of shape (2,) when SIGPOW is set. It refuses (RuntimeError) together with the err-head band (`_ERR`) or when there is no band.
  * In `_body`, in the static/drv branch right after the `clip(sig*std/sref)^p` line and before the R1 gains, it adds `half = half * ((sig * self.std).float() + 1e-8) ** self.sigpow`. It raises if `sig` is None.
  * The load line prints `sigpow`.
  * The centre (centre0 / D1c shift) and the contain-P `min/max` are untouched. `_body` is the only band path: graph and eager both run it.
* Build knob in `build_round13.sh`: `RPDE_SHIP_SIGPOW='(<u>, <v>)'`, which runs `sed "s|^SIGPOW = None |SIGPOW = ... |"`.

## Held-out T2 stream reads (TWIN package: 5 twins, val_unseen, asnp constants, `Track_2/stream_eval.py --save` + offline scoring with the same official scorer functions)
The (a) rows reproduce `baseline_ref.md` §3 exactly: static 53.984 / 47.860, asym 55.594 / 48.481, asym + D1c 55.646 / 48.484.
The prediction is bit-identical across bands, so rel, tke and mvpe do not move. "fx" is final without the time term, and dfx = 0.2498 × dsps.

Stretch 1.00 (stream_eval's own read):

| variant | band | all 8 sps | cov | d sps | traj + | AoA 15-20 sps | cov | d sps | traj + |
|---|---|---|---|---|---|---|---|---|---|
| static | (a) shipped | 53.984 | 0.920 | — | | 47.860 | 0.877 | — | |
| static | (b) lam 0.3 | 54.349 | 0.921 | +0.364 | 8/8 | 47.965 | 0.879 | +0.105 | 3/3 |
| static | (c) lam 0.5 | 54.514 | 0.921 | +0.530 | 8/8 | 47.964 | 0.880 | +0.104 | 3/3 |
| asym R1 | (a) | 55.594 | 0.905 | — | | 48.481 | 0.862 | — | |
| asym R1 | (b) lam 0.3 | 55.751 | 0.905 | +0.157 | 8/8 | 48.534 | 0.862 | +0.053 | 3/3 |
| asym R1 | (c) lam 0.5 | 55.812 | 0.905 | **+0.218** | 8/8 | 48.548 | 0.862 | **+0.067** | 3/3 |
| asym R1 + D1c | (a) | 55.646 | 0.905 | — | | 48.484 | 0.862 | — | |
| asym R1 + D1c | (b) lam 0.3 | 55.802 | 0.905 | +0.156 | 8/8 | 48.534 | 0.862 | +0.050 | 3/3 |
| asym R1 + D1c | (c) lam 0.5 | 55.863 | 0.905 | +0.217 | 8/8 | 48.547 | 0.862 | +0.063 | 3/3 |

Rows common to every variant:

| split | rel err | tke err | mvpe err | rel | tke | mvpe |
|---|---|---|---|---|---|---|
| all 8 | 0.0751 | 0.4546 | 0.0658 | 96.383 | 81.480 | 96.815 |
| AoA 15-20 | 0.1009 | 0.4406 | 0.0880 | 95.197 | 81.947 | 95.784 |

With D1c on, the errors are 0.0746 / 0.4546 / 0.0646 on all 8 and 0.1007 / 0.4406 / 0.0874 on AoA 15-20.

Local final (fx) deltas vs (a):

| variant | lam 0.3, all 8 | lam 0.5, all 8 | lam 0.3, AoA 15-20 | lam 0.5, AoA 15-20 |
|---|---|---|---|---|
| static | +0.091 | +0.132 | +0.026 | +0.026 |
| asym R1 | +0.039 | +0.054 | +0.013 | +0.017 |
| asym R1 + D1c | +0.039 | +0.054 | +0.012 | +0.016 |

Stretch 1.29 is an offline read, target = P + 1.29 (Y − P) with Y = 0 kept at 0. stream_eval itself has no stretch option.

| variant | lam 0.3, all 8 | lam 0.5, all 8 | lam 0.3, AoA 15-20 | lam 0.5, AoA 15-20 |
|---|---|---|---|---|
| static | +0.292 (8/8) | +0.402 (7/8) | +0.059 (3/3) | +0.021 (2/3) |
| asym R1 | +0.086 (8/8) | +0.096 (8/8) | +0.029 (3/3) | +0.023 (3/3) |
| asym R1 + D1c | +0.085 (8/8) | +0.100 (8/8) | +0.031 (3/3) | +0.032 (3/3) |

Per trajectory, asym R1, lam 0.5, stretch 1.0: 3750_0 +0.85, 26700_0 +0.24, 16500_5 +0.23, 6300_10 +0.16, 13950_10 +0.09, 21600_15 +0.04, 8850_20 +0.08, 11400_20 +0.07.
The gain is concentrated at low Re and low AoA, as design_0 predicted.

Caveats:
* The static rows land in the expected range (+0.36 / +0.53 on all 8, +0.10 on AoA 15-20).
* The SIGBLEND constants (q and 40 log kappa values: 42 numbers) were fitted on all 8 of these held-out trajectories, so every read above is **in-sample** for those constants. design_0's leave-one-trajectory-out reads are the honest version, and they are of the same size.
* With R1 on, the gain shrinks to about 40 % at stretch 1.0 and about 25 % at stretch 1.29, because R1 already corrects part of the same miscalibration online.

## Verifier
Dumps are in `$K/dumps/k0b/`. Log: `$K/logs/rp_k270b_reads_7567357.log`.
* **Knob off vs unpatched code** (`unpatch.py` strips the patch; checked byte-identical to `git show 55f0e88:...`): pred, lower and upper are all `array_equal` True over the 336-step stream, for both static and asym R1.
* **Knob on** (lam 0.3 and 0.5 × static / asym / asym + D1c):
  * The prediction equals the knob-off prediction exactly.
  * All outputs are finite, and `L <= P <= U` holds everywhere.
  * Mean width: static 0.01673 → 0.01567 (lam 0.5); asym 0.01413 → 0.01354.
* **Width vs the numpy formula** (design_0 §3, with centre0 and contain-P, from the recorded y0 and sigma): max |d| is 5.4e-08 physical, max relative 9.9e-06 (lam 0.5); 5.5e-08 / 9.1e-06 (lam 0.3). That is float32 noise.
* **Graph vs eager**, over 336 steps, static + SIGPOW: bit-exact for pred, lower and upper (max |d| 0). Every load line shows `graph CAPTURED graph_worst 0.0`.
* **Causal check**: `CAUSALITY OK` (176 steps, max |d| 0) on the twin asym + SIGPOW package, on the twin asym + D1c + SIGPOW package, and on all 4 all-data builds.
* **`validate_t2.py --adaptive`**: SUBMITTABLE on all 4 all-data builds. The in-build stateless gate prints NOT SUBMITTABLE for the D1c builds; that is expected (D1c changes P).
* **Not tested**: the refusal path (SIGPOW with an err-head band raises RuntimeError at construction) was not exercised, because no shipped config uses `_ERR`.

## Packages (all-data members: 4× ffno_all_w40l8_sim_s1..s4 + arch_vit_all_s1 + arch_vit_all_cmb3sim_s3)
All four packages share these settings: asnp knobs, ALPHA 0.80, RPDE_FORCE=1, `RPDE_BAND=$K/pkg/k0/band_k0_sigblend_lam0.5.npz` (for lam 0.5; the `band.npz` inside each zip was checked with `cmp` against it), and `members 6`.

Built in `$K/wt_0b2` (job 7567358, log `$K/logs/rp_k270b_build_7567358.log`) and copied, byte-identical, from `$K/pkg/k0b/stage_sub/`:

| zip | tag at build | sha256 |
|---|---|---|
| `$K/sub_candidates/track2_k27sbt2_a080.zip` | `k27sbt2_l5_a080` | `f6303f6132607fefbeb0c2b6cc0f3a1c6a727f0e4dc1d58be2236dbb2897e5f2` |
| `$K/sub_candidates/track2_k27sbt2_d1c_a080.zip` | `k27sbt2_l5_d1c_a080` | `ed3a8243f4640d7ad7d7f14602657f258e91b8ed76c859215e1d12341e1370d2` |

* Load lines: `r1 True r1_asym True d1c_n0 {0.0 | 30.0} sigpow (0.24232558823017827, 0.24535682217400656) alpha 0.8 graph CAPTURED graph_worst 0.0 numpy_out True members 6`.
* Gates: `--adaptive` SUBMITTABLE and CAUSALITY OK, for both zips.
* The lam 0.3 alternatives are staged only (not in sub_candidates), also gated SUBMITTABLE and CAUSALITY OK:
  * `$K/pkg/k0b/stage_sub/track2_k27sbt2_l3_a080.zip`: `50ce54ff…`
  * `$K/pkg/k0b/stage_sub/track2_k27sbt2_l3_d1c_a080.zip`: `da217532…`
* **No Track-1 zip exists**: every build's `track1_*` was deleted immediately, because it would carry the SIGBLEND npz without EBAND.
* Contaminated all-data validate reads (not evidence): sps 59.765 (lam 0.5) against 59.696 (lam 0.3).

## Predicted board subscores vs the baseline T2 package (asnp, resp. asnp + D1c; board 09-27: rel 94.7044, tke 79.6087, mvpe 94.1319, time 92.3393, sps 43.5168)
| subscore | predicted change | why |
|---|---|---|
| rel | 0 | the prediction is bit-identical |
| tke | 0 | the prediction is bit-identical |
| mvpe | 0 | the prediction is bit-identical |
| time | 0 (−0.01 … 0) | 2 elementwise ops added inside the captured graph |
| sps | **+0.07**, bracket +0.02 … +0.22 | local asym-R1 reads: AoA 15-20 +0.067 / +0.023 at stretch 1.29, all 8 +0.218 / +0.096 |

* **final: +0.017, bracket +0.005 … +0.055.**
* History says band changes over-read locally; this one is a robustness change, positive on 8/8 trajectories in every R1 row.
* It is small against the +0.2 the sprint needs. It is a free roll: it never lost on any held-out trajectory with R1 on.

## Files
* Job scripts: `$K/jobs/rp_k270b_reads.sh`, `$K/jobs/rp_k270b_build.sh`
* Helpers: `$K/jobs/k0b/{score_saves.py, verify_sigpow.py, unpatch.py}`
* Twin packages: `$K/pkg/k0b/t2twin_{a,l3,l5}.zip`
