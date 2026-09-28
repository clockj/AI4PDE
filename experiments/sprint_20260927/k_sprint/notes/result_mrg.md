# Result E1 phase 3 — MERGED Track-2 candidate (F-sprint VARCAL stack + K1 ARENA + K2 JOINT_GAIN + K0 SIGPOW)

E1, 2026-09-27 17:39–17:55 EDT. K = `/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927`, S = `final0927` (read only).
Logs: `K/logs/rp_k271_mb_*`, `rp_k271_mgate_*`, `rp_k271_mread_*`, `K/logs/build_<tag>.log`. Raw reads `K/e1/mread.jsonl`, `K/e1/mgate_*.jsonl`.

## 1. Merge
* `K/wt_mrg` = clone of `K/wt_fin` (b539609) + `git -C S/wt_c diff` saved as `K/pkg/fsprint_c.diff` (md5 04480ea5; 2 files:
  build_round13.sh + Track_2/submit/submission.py; `Track_2/submit/` gained no new files; `stream_c/` not taken; nothing from
  c2/b2/wt_c_p2 taken). Local commit **f159924**. `git apply --3way`: build script clean; submission.py 2 conflicts, hand-resolved:
  1. the load print line: all fields kept (`d1c_n0 vc_n0 fw_n0 sigpow ... joint_gain centre_alpha`).
  2. **the compose (ORDER):** JOINT_GAIN branch now computes `gj_ = |t|_w / (|s|_w + 1e-7)` (same ops/order as K2, bit-identical when
     VC off) and, when VARCAL is on, records `gVp = (s_ * gj_)^2` — the per-channel variance of the field JOINT_GAIN actually composes
     (uncalibrated = sigma_head, as in their per-channel branch where gVp = (ALPHA*sig + (1-ALPHA)*s_)^2 = composed variance) —
     then applies `y = m_ + (y - m_) * (gj_ * gG)`. With JOINT_GAIN off, their VC branch is taken verbatim. VARCAL's update of
     cT/cP/gG (start of step, from the previous window) is unchanged. SIGPOW multiplies the static band before the R1 gains (unchanged K0 code).
     FAMW code present, OFF (FW_N0 0.0).
* Clones `wt_mrg2`, `wt_mrg3` (same commit) were used only for parallel twin builds.

## 2. Equivalence proofs (336-step stream over val_unseen, md5 of all pred/lower/upper)
| | md5 | vs | md5 | result |
|---|---|---|---|---|
| (a) merged, OUR knobs off, CONS knobs (asnp + D1C 30 + VC 50, ship band) `track2_k27mrg_eqA` | 7ec2d3602ac0c81843464adb7d77ffa6 | their `T2_CONS_track2_c27cons_d1c30vc50.zip` | 7ec2d3602ac0c81843464adb7d77ffa6 | **BIT-IDENTICAL** |
| (b) merged, THEIR knobs off (VC 0, D1c 30), ours on `track2_k27mrg_eqB` | 55b3ec7b17882943d550fd17985c404f | `K/sub_candidates/track2_k27fin_d1c_a080.zip` | 55b3ec7b17882943d550fd17985c404f | **BIT-IDENTICAL** |

## 3. Held-out reads (TWIN T2 packages, members = the 5 twins, built through build_round13.sh with the same knobs; stream_eval, asym R1 on)
Scores are subscores (100/(1+0.5e)); final-no-time = 0.4778 rel + 0.1010 tke + 0.0921 mvpe + 0.2498 sps. Sign count = per trajectory final-no-time.
Cross-check: R0/R1 reproduce the F-sprint's own twin reads (result_c.md: CONS full 96.366/81.773/96.869/55.701).

### all
| variant | vs | rel | tke | mvpe | sps | cov | final-no-time | d rel d tke d mvpe d sps d fnt | traj sign (fnt) |
|---|---|---|---|---|---|---|---|---|---|
| R0 asnp+D1c30 | - | 96.403 | 81.480 | 96.869 | 55.646 | 0.9050 | 77.113 |
| R1 CONS | R0 asnp+D1c30 | 96.366 | 81.772 | 96.869 | 55.699 | 0.9051 | 77.138 | -0.038 +0.292 +0.000 +0.053 +0.025 | 8+/0- of 8 |
| R1+JG | R1 CONS | 96.382 | 81.758 | 96.869 | 55.708 | 0.9051 | 77.146 | +0.017 -0.014 +0.000 +0.008 +0.009 | 7+/1- of 8 |
| R2 CONS+ARENA+JG | R1 CONS | 96.382 | 81.758 | 96.869 | 55.708 | 0.9051 | 77.146 | +0.017 -0.014 +0.000 +0.008 +0.009 | 7+/1- of 8 |
| R3 CONS+ARENA+JG+SIGPOW | R1 CONS | 96.382 | 81.758 | 96.869 | 55.925 | 0.9052 | 77.201 | +0.017 -0.014 +0.000 +0.226 +0.063 | 8+/0- of 8 |
| R3nj CONS+ARENA+SIGPOW (no JG) | R1 CONS | 96.366 | 81.772 | 96.869 | 55.917 | 0.9052 | 77.192 | +0.000 +0.000 +0.000 +0.217 +0.054 | 8+/0- of 8 |
| R3 CONS+ARENA+JG+SIGPOW | R0 asnp+D1c30 | 96.382 | 81.758 | 96.869 | 55.925 | 0.9052 | 77.201 | -0.021 +0.278 +0.000 +0.279 +0.088 | 8+/0- of 8 |
| R0+all3 (no VC) = k27fin_d1c twin | R0 asnp+D1c30 | 96.420 | 81.480 | 96.869 | 55.874 | 0.9051 | 77.178 | +0.016 +0.000 +0.000 +0.228 +0.065 | 8+/0- of 8 |
| R4 AGGR | R0 asnp+D1c30 | 96.368 | 81.772 | 96.873 | 55.681 | 0.9051 | 77.135 | -0.035 +0.292 +0.004 +0.035 +0.022 | 6+/2- of 8 |
| R5 AGGR+all3 | R4 AGGR | 96.385 | 81.758 | 96.873 | 55.906 | 0.9052 | 77.198 | +0.017 -0.014 +0.000 +0.226 +0.063 | 8+/0- of 8 |
| R5 AGGR+all3 | R0 asnp+D1c30 | 96.385 | 81.758 | 96.873 | 55.906 | 0.9052 | 77.198 | -0.018 +0.278 +0.004 +0.260 +0.085 | 8+/0- of 8 |
| R3 CONS+ARENA+JG+SIGPOW | R3nj CONS+ARENA+SIGPOW (no JG) | 96.382 | 81.758 | 96.869 | 55.925 | 0.9052 | 77.201 | +0.017 -0.014 +0.000 +0.008 +0.009 | 7+/1- of 8 |

### aoa15_20
| variant | vs | rel | tke | mvpe | sps | cov | final-no-time | d rel d tke d mvpe d sps d fnt | traj sign (fnt) |
|---|---|---|---|---|---|---|---|---|---|
| R0 asnp+D1c30 | - | 95.206 | 81.947 | 95.813 | 48.484 | 0.8616 | 74.702 |
| R1 CONS | R0 asnp+D1c30 | 95.157 | 82.230 | 95.813 | 48.516 | 0.8618 | 74.715 | -0.049 +0.283 +0.000 +0.032 +0.013 | 3+/0- of 3 |
| R1+JG | R1 CONS | 95.179 | 82.227 | 95.813 | 48.531 | 0.8618 | 74.729 | +0.022 -0.003 +0.000 +0.015 +0.014 | 3+/0- of 3 |
| R2 CONS+ARENA+JG | R1 CONS | 95.179 | 82.227 | 95.813 | 48.531 | 0.8618 | 74.729 | +0.022 -0.003 +0.000 +0.015 +0.014 | 3+/0- of 3 |
| R3 CONS+ARENA+JG+SIGPOW | R1 CONS | 95.179 | 82.227 | 95.813 | 48.596 | 0.8624 | 74.745 | +0.022 -0.003 +0.000 +0.079 +0.030 | 3+/0- of 3 |
| R3nj CONS+ARENA+SIGPOW (no JG) | R1 CONS | 95.157 | 82.230 | 95.813 | 48.580 | 0.8624 | 74.731 | +0.000 +0.000 +0.000 +0.064 +0.016 | 3+/0- of 3 |
| R3 CONS+ARENA+JG+SIGPOW | R0 asnp+D1c30 | 95.179 | 82.227 | 95.813 | 48.596 | 0.8624 | 74.745 | -0.026 +0.280 +0.000 +0.111 +0.044 | 3+/0- of 3 |
| R0+all3 (no VC) = k27fin_d1c twin | R0 asnp+D1c30 | 95.227 | 81.947 | 95.813 | 48.563 | 0.8622 | 74.732 | +0.022 -0.000 +0.000 +0.078 +0.030 | 3+/0- of 3 |
| R4 AGGR | R0 asnp+D1c30 | 95.153 | 82.230 | 95.799 | 48.465 | 0.8617 | 74.699 | -0.052 +0.283 -0.013 -0.019 -0.002 | 1+/2- of 3 |
| R5 AGGR+all3 | R4 AGGR | 95.176 | 82.227 | 95.799 | 48.549 | 0.8625 | 74.730 | +0.022 -0.003 +0.000 +0.083 +0.031 | 3+/0- of 3 |
| R5 AGGR+all3 | R0 asnp+D1c30 | 95.176 | 82.227 | 95.799 | 48.549 | 0.8625 | 74.730 | -0.030 +0.280 -0.013 +0.064 +0.029 | 3+/0- of 3 |
| R3 CONS+ARENA+JG+SIGPOW | R3nj CONS+ARENA+SIGPOW (no JG) | 95.179 | 82.227 | 95.813 | 48.596 | 0.8624 | 74.745 | +0.022 -0.003 +0.000 +0.015 +0.014 | 3+/0- of 3 |

**Additivity on top of VC:**
* SIGPOW: **fully additive** — sps +0.217 all / +0.064 AoA15-20 on CONS (R3nj vs R1) vs +0.2 / +0.06 alone; 8/8, 3/3.
* JOINT_GAIN: rel gain survives (+0.017 score all, +0.022 AoA; ~ -0.45 % rel error, as alone), but **mild interference on tke**:
  -0.014 (all) / -0.003 (AoA) tke score on top of VC, where alone it moved tke 0.000. Net still positive: +0.009 all (7/8),
  +0.014 AoA (3/3) (R3 vs R3nj). Small; not a reason to drop it.
* ARENA: no output change (R2 == R1+JG exactly).
* CONS vs AGGR: with all add-ons R3 (CONS) beats R5 (AGGR) by +0.003 all and +0.015 AoA15-20; AGGR alone is 1+/2- vs R0 at AoA 15-20.

## 4. All-data merged packages (`K/sub_candidates/`, RPDE_FORCE=1, SIGBLEND band lam 0.5, EBAND baked, SMALL_GRAPH off, PIN_OUT 128, ARENA 1024)
| TAG | knobs | T2 sha256 | T1 sha256 (same build) |
|---|---|---|---|
| **k27mrg_cons_a080** (= R3) | asnp + D1C 30 + VC 50 + ARENA + JOINT_GAIN + SIGPOW | 68a53737ce44bdedcd07290828fd87a404ba8c4abd1ed41511f5c9da48c43eff | 9126c833b36cfbb4749b6272e53f9750e8a995df5eb2fca1914cfb70c202415d |
| k27mrg_aggr_a080 (= R5) | asnp + D1C 10 + VC 50 + ARENA + JOINT_GAIN + SIGPOW | d3f89c4fff9095f6efee0e4462144b33b77dac5c1748bc9755d73dbfeb610914 | 81363e0957aa778159d2893456568f215e617f43cee76803287d7257faba8d29 |
| k27mrg_consnj_a080 (= R3nj, fallback) | CONS + ARENA + SIGPOW, no JOINT_GAIN | 7f226c7a71550b640cfd033380b7694fa00a3aa08106e51d3197bebdd7a145b9 | e5b35140b5c380b90edbf51ee21e41b30456de9ffeb931588a274a9429a49e89 (T1 without JOINT_GAIN) |

Gates (job 7567514): both T2 zips: in-build stateless NOT SUBMITTABLE (expected with D1c); `validate_t2 --adaptive` **SUBMITTABLE**;
`stream_eval --causal-check` **CAUSALITY OK** (176 steps, max diff 0); load line `d1c_n0 30.0/10.0 vc_n0 50.0 fw_n0 0.0 sigpow (0.2423, 0.2454)
joint_gain True graph CAPTURED ... members 6`; kept-output safety over 1100 steps: all 3300 kept arrays unchanged, arena_i 1024 then
fallback, step k == step k+336 (periodic) true.
Track-1 zips of the same builds: **valid** — EBAND baked with the SIGBLEND band, SMALL_GRAPH False, validate_strict SUBMITTABLE,
joint_gain True/True, members 6 (T1 is untouched by D1c/VC, so they equal k27fin's T1 functionally).
k27mrg_consnj_a080 (job 7567606): `--adaptive` SUBMITTABLE, CAUSALITY OK, joint_gain False, graph CAPTURED, members 6, 1100-step kept outputs unchanged.

## 5. Verdict
| package | verdict | reason |
|---|---|---|
| **track2_k27mrg_cons_a080** | **SHIP (preferred T2)** | best held-out read: vs R0 +0.088 final-no-time all 8 (8/8), +0.044 AoA15-20 (3/3); vs their CONS +0.063 (8/8) / +0.030 (3/3). Both equivalences bit-exact; all gates green. Plus ARENA's -0.2…-0.4 ms/step (time, not in these numbers) |
| track2_k27mrg_aggr_a080 | NO-SHIP (vs cons) | read below CONS on both splits (-0.003 all, -0.015 AoA15-20); AGGR alone is 1+/2- at AoA15-20. Its OOD edge (result_c: +0.09 sps OOD vs CONS) is the only argument for it |
| track2_k27mrg_consnj_a080 | fallback | use only if JOINT_GAIN's small tke interference with VC (-0.014 tke score) is judged risky; it gives up +0.009 / +0.014 |
Local numbers are signs, not sizes (proxy over-reads error cuts ~2x; sps "tighter band" gains have over-read on the board before).
