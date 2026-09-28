# result_c — EXECUTOR EC (final T2 package)

**19:08 EDT STATUS: T2_MRGP2 (K-merge + C2 + B2, WITH JOINT_GAIN) BUILT, ALL GATES GREEN, X3 GREEN; paired read >= T2_P2 in all 3 regimes (+0.057 / +0.029 / +0.034) and -0.09 ms faster -> per the commander's adoption rule it REPLACES T2_P2 as the upload. pkg/c_final/T2_MRGP2_track2_MRGP2.zip sha256 1aa2896f7f98c3c775cba926c96a9972ea81dc66a3003090b34aaaf1a68f54bf.**

Edits to wt_c submission.py / build_round13.sh: NONE. wt_c_p2 = cp of wt_c (no .git) + c2.patch + b2.patch; b2 hunk 6 (the load print line) rejected -> hand-merged (adds r1_tsm/r1_blk_n0/r1_blk to the print), nothing else (design C's patch untouched). Added untracked `Track_2/stream_c/ec_matrix.py`, `ec_report.py` only.

## B. Board-likeness gauge (CPU probe, band r38 fallback, baseline = asym R1 without D1c; composite = no-time 4-term)
```
== GAUGE (baseline = asym R1, no D1c; band r38)
regime                         rel     tke    mvpe     sps |   rel_e   tke_e  mvpe_e    cov | comp
in full f1.00               96.385  81.484  96.818  55.586 |  0.0750  0.4545  0.0657  0.905 | 77.085
in AoA15-20 f1.00           95.197  81.942  95.779  48.459 |  0.1009  0.4408  0.0881  0.861 | 74.688
OOD AoA20 f1.00             94.286  79.843  94.247  44.616 |  0.1212  0.5049  0.1221  0.843 | 72.939
in full f1.29               95.392  78.256  95.936  50.067 |  0.0966  0.5557  0.0847  0.876 | 74.825
in AoA15-20 f1.29           93.899  78.492  94.617  42.471 |  0.1299  0.5480  0.1138  0.825 | 72.116
OOD AoA20 f1.29             92.785  76.717  92.714  38.680 |  0.1555  0.6070  0.1572  0.800 | 70.282
BOARD T2 09-25              94.771  79.516  94.132  43.556 |  0.1104  0.5152  0.1247      - | 72.863
HOLD AoA10 f1.00 (EA)       95.803  79.904  95.161  51.353 |  0.0876  0.5030  0.1017  0.886 | 75.437
HOLD AoA10 f1.29 (EA)       94.662  76.709  93.911  45.640 |  0.1128  0.6072  0.1297  0.853 | 73.027
  (EA 17:55: AoA-10 holdout = INTERPOLATION regime, 18 traj x 42 windows, members aoa10_{ffnosim,ffno}_s1 (fam A) + aoa10_{fno,nosim}_s1 (fam B),
   probe2 band r35_centre0_fill; mvpe/rel 1.16 (board 1.13, in-dist 0.87, OOD20 1.01). Detail: notes/ood_aoa10.md)
lambda f1.00: rel_e +0.47  tke_e +1.16  mvpe_e +1.08  sps +1.28  comp +1.04
lambda f1.29: rel_e -0.77  tke_e -0.56  mvpe_e +0.25  sps -0.29  comp -0.41
```
Error ratios board / in-dist AoA15-20 (f=1.0): rel 1.09, tke 1.17, mvpe 1.42. OOD / in-dist AoA15-20: rel 1.20, tke 1.15, mvpe 1.39. A pure stretch (f=1.29): 1.29 / 1.24 / 1.29.
Reading: at f=1.0 the board sits AT the OOD level on tke, mvpe, sps and composite (lambda ~1.0-1.3) and halfway on rel (0.47).
At f=1.29 (the team's "board = in-distribution with a 1.29-1.39 tail stretch" model) the board is BETTER than in-dist AoA15-20 (lambda < 0).
The two hypotheses explain the LEVEL equally; the error PATTERN (mvpe disproportionately high, rel least) matches OOD (1.20/1.15/1.39) better than a uniform stretch (1.29/1.24/1.29).
**lambda = 0.5, bracket [0.25, 1.0]. Trust: LOW** — OOD dump = 3 weaker stand-in members (FNO as family B), band not refitted; the level coincidence could be member weakness; OOD rel is inflated by that.

## C. Candidate matrix (delta composite vs asym-R1 baseline; E(lambda) = (1-l)*d_in_a15 + l*d_OOD; latency charge not yet included)
```
d10_v30_f20       +0.0480   6/8   -0.0024  1/3   +0.0654  +0.3512  16/16  0.000 | +0.0860 +0.1744 +0.2628
d10_v30_f8        +0.0436   5/8   -0.0090  1/3   +0.0610  +0.3482  16/16  0.000 | +0.0803 +0.1696 +0.2589
d10_v50_f20       +0.0521   7/8   +0.0048  2/3   +0.0637  +0.3412  16/16  0.000 | +0.0889 +0.1730 +0.2571
d10_v50_f8        +0.0475   6/8   -0.0018  1/3   +0.0591  +0.3381  16/16  0.000 | +0.0832 +0.1682 +0.2532
d10_v30_f0        +0.0440   6/8   -0.0027  1/3   +0.0617  +0.3345  16/16  0.000 | +0.0816 +0.1659 +0.2502
d10_v50_f0        +0.0481   6/8   +0.0045  1/3   +0.0600  +0.3245  16/16  0.000 | +0.0845 +0.1645 +0.2445
d30_v30_f20       +0.0537   8/8   +0.0135  3/3   +0.0834  +0.3052  16/16  0.000 | +0.0864 +0.1593 +0.2322
d30_v30_f8        +0.0501   7/8   +0.0070  2/3   +0.0789  +0.3058  16/16  0.000 | +0.0817 +0.1564 +0.2311
d30_v50_f20       +0.0577   8/8   +0.0206  3/3   +0.0816  +0.2950  16/16  0.000 | +0.0892 +0.1578 +0.2264
d30_v50_f8        +0.0541   8/8   +0.0144  3/3   +0.0771  +0.2958  16/16  0.000 | +0.0848 +0.1551 +0.2255
d30_v30_f0        +0.0477   8/8   +0.0129  3/3   +0.0791  +0.2777  16/16  0.000 | +0.0791 +0.1453 +0.2115
d30_v50_f0        +0.0518   8/8   +0.0202  3/3   +0.0775  +0.2676  16/16  0.000 | +0.0820 +0.1439 +0.2057
d10_v0_f0         +0.0226   6/8   -0.0099  1/3   -0.0107  +0.2433  16/16  0.000 | +0.0534 +0.1167 +0.1800
d30_v0_f0         +0.0261   7/8   +0.0056  2/3   +0.0069  +0.1863  16/16  0.000 | +0.0508 +0.0960 +0.1411
```

## A. Verification (job 7566775, 16:39-16:54)
* Knob-off identity: wt_c build with asnp+D1c30 (VC/FW unset) vs fallback `track2_x27asnp_d1c30_a080.zip` (built from 55f0e88): every file md5-equal except submission.py (diff = only design C's added lines); stream_eval on val_unseen: pred / lower / upper **bit-equal** (336x20x32x64x3).
* All 9 builds (off, cons d30v50, d10v50, d10v30, d30v30, d10v30fw20, d10v30fw8, d30v50fw20, d10v50fw20): `validate_t2 --adaptive` **SUBMITTABLE**, `stream_eval --causal-check` **CAUSALITY OK (176 steps, max|diff| 0)**, load line `graph CAPTURED members 6`, knobs as intended.
* Boundaries: VARCAL/FAMW sums are gated by R1's `gflag` (0 on a trajectory's first step) and zeroed in `reset_ttt_state` via `_reset_c()` = R1's mechanism. Nothing depends on trajectory length (only n/(n+N0) shrink); on 14-window streams the effect is simply smaller (see e14 rows).
* Zips + sha256: `$S/pkg/c_work/track2_c27*.zip`.

## C2. Per-subscore deltas (probe, f=1.0) incl. FIRST-14-WINDOWS subset (hidden streams can be as short as 14)
```
in a15:  d30 +0.0056 | d30v50 +0.0202 | d15v50 +0.0118 | d10v50 +0.0045 | d10v50f20 +0.0048 | d10v30f20 -0.0024
in e14a15 (first 14 windows, AoA15-20): d30 -0.0070 | d30v50 +0.0068 | d15v50 -0.0063 | d10v50 -0.0199 | d10v50f20 -0.0227
OOD full: d30 +0.186 | d30v50 +0.268 | d15v50 +0.309 | d10v50 +0.325 | d10v50f20 +0.341 | d10v30f20 +0.351
   (d10v50: rel +0.117 tke +0.583 mvpe +0.661 sps +0.595)
OOD e14: d30 +0.094 | d30v50 +0.137 | d15v50 +0.178 | d10v50 +0.197 | d10v50f20 +0.211
```
Latency from job 7566775 was INVALID (exclusive node -> OMP 128 threads, 3.5-4.9 ms noise); re-measuring in 7567225 (-c 12, 4 interleaved rounds, medians).

## Latency (job 7567225, -c 12, 4 interleaved rounds, rep-3 medians, ms/step local, all-data packages)
ref asnp+D1c30 3.05-3.13 (one spike 3.36) | CONS d30v50 3.09-3.14 | d10v50 3.08-3.11 (one 3.35) | d10v50fw20 3.21-3.22 | d10v30fw20 3.22.
=> VARCAL +0.03..0.04 ms, FAMW +0.12..0.13 ms more (board at T2 slope 0.6 [0.3,0.9]: FAMW -0.003..-0.008 final).

## Real code path (job 7567230): twins in-dist + OOD package (aoa20_ffnosim_s1,s2 + aoa20_fno_s1, 3 members, same band) — reproduces the probe
```
OOD  base  rel 94.285 tke 79.845 mvpe 94.248 sps 44.617
OOD  d30   rel 94.396 tke 79.845 mvpe 94.719 sps 44.976
OOD  d30v50 (CONS) rel 94.367 tke 80.427 mvpe 94.719 sps 45.122
OOD  d10v50 (AGGR) rel 94.403 tke 80.427 mvpe 94.909 sps 45.210
OOD  d10v50f20 rel 94.414 tke 80.433 mvpe 94.917 sps 45.250 | d10v30f20 94.400/80.546/94.917/45.271 | d15v50 94.392/80.427/94.851/45.190
IN a15 base 95.197/81.955/95.785/48.484 | d30 95.205/81.955/95.814/48.486 | CONS 95.157/82.236/95.814/48.518 | AGGR 95.153/82.236/95.800/48.467 | d10v50f20 95.151/82.236/95.794/48.457
IN full base 96.383/81.481/96.815/55.594 | CONS 96.366/81.773/96.869/55.701 | AGGR 96.368/81.773/96.874/55.682
```

## D. Packages (knob-only)
* **T2_CONS** = `$S/pkg/c_final/T2_CONS_track2_c27cons_d1c30vc50.zip` sha256 d88c5d214b3ce967a1fbc332dfab7cf4a70934ba38bee807634559fafb43ad5c — asnp_a080 knobs + RPDE_SHIP_D1C_N0=30.0 RPDE_SHIP_VC_N0=50.0, band = fallback band_r38. validate_t2 --adaptive SUBMITTABLE; CAUSALITY OK; graph CAPTURED members 6.
* **T2_AGGR** = `$S/pkg/c_final/T2_AGGR_track2_c27_d1c10vc50.zip` sha256 bad3f25a161d45811617868529d50cf16dd72653fc47fd3385d800aa194d2dae — asnp_a080 + D1C_N0=10.0 VC_N0=50.0 (FAMW OFF). Same gates all green.
  Why not FAMW: +0.013 OOD (probe, with an FNO stand-in for the ViT family) against -0.005..-0.008 of board time and a real-path AoA15-20 read of -0.001..-0.004: net ~0. Why D1c10 not 5: OOD saturates at 10 (d5 +0.347 vs d10 +0.341 with FW) while in-dist AoA15-20 falls to -0.012..-0.02.
  AGGR in-dist a15 +0.0045 (>= -0.03 OK), first-14-window AoA15-20 -0.020 (the cost of a fast D1c on short streams); OOD +0.325 (16/16).

## P2: c2.patch + b2.patch merged (wt_c_p2), verified (job 7567447) and read on the REAL code path (job 7567448)
* Merge: c2 applied clean (offset -5); b2 hunk 6 (load print line only) rejected and hand-merged. Knobs off: pred/lower/upper **bit-equal** to fallback asnp+D1c30.
* Builds p2_c2, p2_c2b2, p2_consb2, p2_aggrb2: all validate_t2 --adaptive SUBMITTABLE (includes finite + lower<=upper checks), CAUSALITY OK, graph CAPTURED, load lines show intended knobs.
* C2 REPRODUCES the designer exactly (+0.063 / +0.009 / +0.401 vs designer +0.062 / +0.009 / +0.401). B2 reproduces its sps gain (+0.09 full, +0.06 a15, rel/tke/mvpe identical) and adds +0.045 OOD.
* Paired real-path table (twins in-dist = 5 held-out twins; OOD = AoA-20 holdout package, 3 members; no-time composite):
```
-- IN full: base comp 77.086
   base    rel 96.383 tke 81.480 mvpe 96.815 sps 55.595 | d vs base +0.0000 | d vs d30(plan) -0.0283
   d30     rel 96.404 tke 81.480 mvpe 96.870 sps 55.648 | d vs base +0.0283 | d vs d30(plan) +0.0000
   cons    rel 96.366 tke 81.772 mvpe 96.870 sps 55.701 | d vs base +0.0529 | d vs d30(plan) +0.0246
   aggr    rel 96.369 tke 81.772 mvpe 96.874 sps 55.682 | d vs base +0.0500 | d vs d30(plan) +0.0216
   c2      rel 96.359 tke 81.862 mvpe 96.880 sps 55.716 | d vs base +0.0633 | d vs d30(plan) +0.0350
   c2b2    rel 96.359 tke 81.862 mvpe 96.880 sps 55.804 | d vs base +0.0853 | d vs d30(plan) +0.0570
   consb2  rel 96.366 tke 81.772 mvpe 96.870 sps 55.792 | d vs base +0.0756 | d vs d30(plan) +0.0473
   aggrb2  rel 96.369 tke 81.772 mvpe 96.874 sps 55.772 | d vs base +0.0725 | d vs d30(plan) +0.0441
-- IN a15: base comp 74.696
   base    rel 95.197 tke 81.954 mvpe 95.785 sps 48.484 | d vs base +0.0000 | d vs d30(plan) -0.0075
   d30     rel 95.206 tke 81.954 mvpe 95.814 sps 48.486 | d vs base +0.0075 | d vs d30(plan) +0.0000
   cons    rel 95.157 tke 82.235 mvpe 95.814 sps 48.518 | d vs base +0.0204 | d vs d30(plan) +0.0130
   aggr    rel 95.153 tke 82.235 mvpe 95.800 sps 48.467 | d vs base +0.0045 | d vs d30(plan) -0.0030
   c2      rel 95.138 tke 82.275 mvpe 95.813 sps 48.493 | d vs base +0.0091 | d vs d30(plan) +0.0016
   c2b2    rel 95.138 tke 82.275 mvpe 95.813 sps 48.555 | d vs base +0.0245 | d vs d30(plan) +0.0171
   consb2  rel 95.157 tke 82.235 mvpe 95.814 sps 48.579 | d vs base +0.0357 | d vs d30(plan) +0.0282
   aggrb2  rel 95.153 tke 82.235 mvpe 95.800 sps 48.528 | d vs base +0.0197 | d vs d30(plan) +0.0123
-- OOD: base comp 72.939
   base    rel 94.285 tke 79.844 mvpe 94.248 sps 44.617 | d vs base +0.0000 | d vs d30(plan) -0.1861
   d30     rel 94.396 tke 79.844 mvpe 94.719 sps 44.976 | d vs base +0.1861 | d vs d30(plan) +0.0000
   cons    rel 94.367 tke 80.426 mvpe 94.719 sps 45.122 | d vs base +0.2675 | d vs d30(plan) +0.0814
   aggr    rel 94.403 tke 80.426 mvpe 94.909 sps 45.210 | d vs base +0.3242 | d vs d30(plan) +0.1381
   c2      rel 94.438 tke 80.556 mvpe 94.975 sps 45.372 | d vs base +0.4006 | d vs d30(plan) +0.2145
   c2b2    rel 94.438 tke 80.556 mvpe 94.975 sps 45.553 | d vs base +0.4458 | d vs d30(plan) +0.2597
   consb2  rel 94.367 tke 80.426 mvpe 94.719 sps 45.325 | d vs base +0.3182 | d vs d30(plan) +0.1321
   aggrb2  rel 94.403 tke 80.426 mvpe 94.909 sps 45.409 | d vs base +0.3739 | d vs d30(plan) +0.1878
```
* **Latency (job 7567500, 4 interleaved rounds, rep-3 medians; X3 QA job 7567576 agrees):** ref asnp+D1c30 3.00-3.09 | CONS 3.09-3.12 | AGGR 3.09-3.10 | C2 3.23 | consb2/aggrb2 3.32 | **C2+B2 3.45**.
  => C2 +0.18 ms, B2 +0.22 ms more, C2+B2 +0.40 ms vs asnp+D1c30. **B2 FAILS its designer's own ship rule (<= +0.1 ms) and C2+B2 fails the <= 0.2 ms AGGR constraint.**
  Charged anyway (board = 0.6 x local, bracket 0.3-0.9; -0.065 final per board ms) C2+B2 still has the highest E at every lambda (table below): B2's +0.015..+0.045 composite beats its -0.006..-0.014 time cost.

## F. Registered predictions (BEFORE upload). Anchor = 09-25 T2 row (alpha 0.80): rel 94.771 / tke 79.516 / mvpe 94.132 / time 91.758 / sps 43.556 = 81.7156,
treated as asym-R1 without D1c; subscore = anchor + (1-l)*d_inAoA15-20 + l*d_OOD (real-path deltas vs base), l = 0.5 central, bracket [l 0.25 & slope 0.9, l 1.0 & slope 0.3].
time centre 92.05 = mid of 09-25 / 09-26 draws, minus latency; **time is a draw: add +-0.4 score (+-0.6 ms) -> +-0.04 final on top of every bracket.**
```
d30    rel 94.831 [94.805,94.882]  tke 79.516 [79.516,79.516]  mvpe 94.382 [94.271,94.603]  sps 43.737 [43.647,43.915]  time 92.050 [92.050,92.050]  final 81.842 [81.797,81.931]
cons   rel 94.792 [94.761,94.853]  tke 79.947 [79.872,80.098]  mvpe 94.382 [94.271,94.603]  sps 43.826 [43.708,44.061]  time 92.034 [92.026,92.042]  final 81.888 [81.825,82.012]
aggr   rel 94.808 [94.767,94.889]  tke 79.947 [79.872,80.098]  mvpe 94.470 [94.308,94.793]  sps 43.844 [43.692,44.149]  time 92.034 [92.026,92.042]  final 81.908 [81.827,82.069]
c2     rel 94.818 [94.765,94.924]  tke 80.032 [79.934,80.228]  mvpe 94.509 [94.335,94.859]  sps 43.938 [43.752,44.311]  time 91.977 [91.941,92.014]  final 81.943 [81.842,82.142]
c2b2   rel 94.818 [94.765,94.924]  tke 80.032 [79.934,80.228]  mvpe 94.509 [94.335,94.859]  sps 44.060 [43.843,44.492]  time 91.889 [91.808,91.969]  final 81.965 [81.852,82.183]
aggrb2 rel 94.808 [94.767,94.889]  tke 79.947 [79.872,80.098]  mvpe 94.470 [94.308,94.793]  sps 43.974 [43.787,44.348]  time 91.941 [91.887,91.996]  final 81.932 [81.838,82.114]
final vs plan d30 at lam .25/.5/.75/1.0 (slope .6):
  cons   +0.029  +0.046  +0.063  +0.080
  aggr   +0.031  +0.066  +0.101  +0.137
  c2     +0.048  +0.101  +0.154  +0.207
  c2b2   +0.062  +0.123  +0.183  +0.244
  aggrb2 +0.046  +0.090  +0.133  +0.177
  consb2 +0.044  +0.070  +0.096  +0.122
```
(d30 = today's planned asnp+D1c30, for reference.) Caveat on the lambda: the whole spread between packages is the OOD term; at lambda = 0 (board behaves like our
in-distribution AoA15-20 validation) every candidate is within +-0.03 of the plan and none reaches the cut.

## Recommendation
Upload **T2_P2 (C2+B2)** — highest expected gain at every lambda (+0.06 / +0.12 / +0.18 / +0.24 over the plan at lambda .25/.5/.75/1), in-dist AoA15-20 +0.017 vs plan
(no in-distribution loss), all gates green, stress-tested by X3. Its only violation is latency (+0.40 local ms, ~-0.017 final, already charged). If the commander enforces the
0.2 ms rule: **T2_C2** (+0.10 at lambda .5). T2_AGGR is dominated by C2; T2_CONS is the safe choice (+0.05).

## T2_MRG_P2 (18:35-19:07) = K-sprint merge (trick0927/wt_mrg @ f159924, copied to $S/wt_mrgp2 without .git) + c2.patch + b2.patch
* Merge: c2 and b2 applied with offsets; only the load print line was rejected (both patches) and hand-merged (keeps every K field + r1_tsm/r1_blk/d1c_eb/lead). The lines added are identical to wt_c_p2's. c2 touches only the D1c block (not the compose); VARCAL 30 goes through the K-merge's JOINT_GAIN*gG compose branch unchanged.
* Build = rp_k271_mb.sh "k27mrg_cons_a080" recipe (OURS = SMALL_GRAPH off + WARM_TAILS + PIN_OUT 128 + BATCH_CAP 56 + ARENA_STEPS 1024 + JOINT_GAIN + EBAND + SIGPOW (0.24232558823017827, 0.24535682217400656), band SIGBLEND lam0.5 md5 e8309d9f, checked inside the zips) + P2 knobs (D1C_N0 30, D1C_EB True, D1C_EB_N0 1.0, D1C_LEAD_N0 5.0, VC_N0 30, R1_TSM 3, R1_BLK_N0 20.0, R1_BLK 4; FW 0, SIGEMA off). Job 7568186.
* (a) C2/B2 off + VC 50 in the merged file == trick0927/FINAL/track2_k27mrg_cons_a080.zip: pred/lower/upper BIT-EQUAL.
* (b) K knobs off (SIGPOW None + r38 band, JOINT_GAIN off) but ARENA 1024 ON + P2 knobs == our T2_P2: pred/lower/upper BIT-EQUAL (the arena and the merge do not change values; D1c-lead shift sits where wt_c_p2 has it).
* (c) MRGP2 and MRGP2nj: validate_t2 --adaptive SUBMITTABLE (finite 0 bad, lower<=upper 0 bad), CAUSALITY OK (176 steps, 0 diff), graph CAPTURED members 6; 1100-step kept-outputs test (K's t2_edge.py): kept_ok, 0 bad, arena_i 1024, periodic_equal. X3 round 2: GREEN on both (9900-step run, arena-refused path, eager/fp32).
* Host RSS right after load (VmRSS): 2.93 GB (X3: +1.9 GB vs P2). The arena covers only the first 1024 steps of a process (then clone path, 3.42 -> 3.57 ms).
* Latency (job 7568361, 4 interleaved rounds, rep-3 medians, local ms/step): T2_P2 3.44 | K merged cons 3.08-3.15 | **MRGP2 3.35** | MRGP2nj 3.33. MRGP2 = -0.09 vs P2, +0.25 vs K cons.
### Paired real-path reads (twin packages built from the merged tree; OOD = same zip with the 3 AoA-20 holdout members)
```
-- IN full (d = no-time composite vs our T2_P2 twin)
   tw_c2           rel 96.359 tke 81.862 mvpe 96.879 sps 55.715 | d -0.0217
   tw_p2           rel 96.359 tke 81.862 mvpe 96.879 sps 55.802 | d +0.0000
   tw_mcons        rel 96.382 tke 81.758 mvpe 96.869 sps 55.925 | d +0.0303
   tw_mc2          rel 96.376 tke 81.837 mvpe 96.879 sps 55.937 | d +0.0393
   tw_mp2nosp_nj   rel 96.359 tke 81.862 mvpe 96.879 sps 55.802 | d +0.0000
   tw_mp2nosp      rel 96.376 tke 81.837 mvpe 96.879 sps 55.807 | d +0.0068
   tw_mp2nj        rel 96.359 tke 81.862 mvpe 96.879 sps 56.002 | d +0.0500
   tw_mp2          rel 96.376 tke 81.837 mvpe 96.879 sps 56.008 | d +0.0571
-- IN a15 (d = no-time composite vs our T2_P2 twin)
   tw_c2           rel 95.138 tke 82.270 mvpe 95.812 sps 48.491 | d -0.0150
   tw_p2           rel 95.138 tke 82.270 mvpe 95.812 sps 48.551 | d +0.0000
   tw_mcons        rel 95.179 tke 82.227 mvpe 95.813 sps 48.596 | d +0.0266
   tw_mc2          rel 95.160 tke 82.260 mvpe 95.812 sps 48.569 | d +0.0140
   tw_mp2nosp_nj   rel 95.138 tke 82.270 mvpe 95.812 sps 48.551 | d +0.0000
   tw_mp2nosp      rel 95.160 tke 82.260 mvpe 95.812 sps 48.565 | d +0.0130
   tw_mp2nj        rel 95.138 tke 82.270 mvpe 95.812 sps 48.615 | d +0.0160
   tw_mp2          rel 95.160 tke 82.260 mvpe 95.812 sps 48.629 | d +0.0290
-- OOD (d = no-time composite vs our T2_P2 twin)
   tw_c2           rel 94.438 tke 80.556 mvpe 94.975 sps 45.372 | d -0.0455
   tw_p2           rel 94.438 tke 80.556 mvpe 94.975 sps 45.554 | d +0.0000
   tw_mcons        rel 94.399 tke 80.452 mvpe 94.719 sps 45.159 | d -0.1514
   tw_mc2          rel 94.469 tke 80.592 mvpe 94.975 sps 45.407 | d -0.0183
   tw_mp2nosp_nj   rel 94.438 tke 80.556 mvpe 94.975 sps 45.554 | d +0.0000
   tw_mp2nosp      rel 94.469 tke 80.592 mvpe 94.975 sps 45.582 | d +0.0254
   tw_mp2nj        rel 94.438 tke 80.556 mvpe 94.975 sps 45.587 | d +0.0082
   tw_mp2          rel 94.469 tke 80.592 mvpe 94.975 sps 45.616 | d +0.0339
```
* tw_mp2nosp_nj == tw_p2 in every digit (merge check). JOINT_GAIN on top of C2+B2: +0.007 / +0.013 / **+0.026 OOD** -> positive, ship WITH it (tke -0.025 in-dist, +0.036 OOD).
* Band-side additivity: SIGPOW alone (mc2 - c2) +0.061 / +0.029 / +0.027; SIGPOW on top of B2 (mp2 - mp2nosp) +0.050 / +0.016 / +0.009. B2 alone (p2 - c2) +0.022 / +0.015 / +0.046; B2 on top of SIGPOW (mp2 - mc2) +0.018 / +0.015 / +0.052. Mostly additive; they overlap on OOD (SIGPOW's OOD gain shrinks 0.027 -> 0.009), never cancel.
* SIGBLEND caveat: its 42 constants are fitted on the same 8 val_unseen trajectories (in-sample); K's leave-one-trajectory-out read is C x100 +0.03 (AoA 15-20) / +0.48 (all 8) at f 1.29.
* vs K merged cons: +0.027 full, +0.002 AoA15-20, +0.185 OOD.

### Registered prediction T2_MRGP2 (same convention as P2: anchor 09-25 row, lambda 0.5, bracket [l .25 & slope .9, l 1.0 & slope .3], +-0.04 time draw on top)
rel 94.845 [94.79, 94.955] · tke 80.045 [79.93, 80.26] · mvpe 94.509 [94.335, 94.859] · time 91.925 [91.86, 91.99] · sps 44.130 [43.90, 44.55] · **final 82.000 [81.885, 82.219]**.
At lambda 0 (board = in-distribution AoA15-20): final 81.787 (P2 at lambda 0: 81.755). Live cut 81.988: reached at lambda >= ~0.5.
