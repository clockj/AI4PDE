"""Every scored Codabench round, and what the five subscores imply about `final_score`.

v9 stopped publishing the combination rule.  Round 9 inferred a weight family from three
rounds; round 10 falsified the no-intercept form with a fourth.  With EIGHTEEN scored rounds
the linear no-intercept model is comfortably OVERDETERMINED and is tested, not assumed.

**ROUND 17: THE `sum(w) = 1` CONSTRAINT IS DROPPED, AND IT WAS NEVER TESTED.**  Every fit from
round 9 to round 16 imposed it -- not because anything measured it, but because a weighted mean
was the natural guess and nobody removed the row to see what happened.  Removing it:

    fit (18 rounds, both tracks)  rel_l2    tke   mvpe   time    sps   sum w  max resid
    sum(w) = 1  (rounds 9-16)      0.570  0.066  0.047  0.085  0.232   1.000     0.049
    FREE linear  (round 17+)       0.475  0.102  0.094  0.097  0.250   1.018    0.0023

**Twenty times tighter with one fewer assumption**, and it is confirmed by the only weight ever
read off the board WITHOUT a fit: `W_TIME_DIRECT` below is 0.0956 from the R13/R14
single-subscore A/B, and the free fit says **0.0967** where the constrained fit said 0.085.
Leave-one-out predicts a held-out round's `final` to +-0.004 (rms 0.002), so this is now an
instrument rather than a regression.  Why `sum(w)` lands at 1.018 rather than 1 is unknown -- a
sixth term, or a subscore transformed before the average -- and the ~0.002 residual says
whatever it is, it is small.  It does not matter for DELTAS, which is how the fit is used.

`fit()` returns the FREE fit by default; pass `constrained=True` for the historical one.

**The fit POOLS BOTH TRACKS.**  Round 13 shipped the identical package to Track 1 and Track 2
on the same day and the two boards returned the same four error subscores to <=0.3 % of the
error (mvpe excepted, and even that is one window's worth), and Track 1's fitted weights
reprice Track 2's subscores to 80.650 against a published 80.653 -- so the two tracks are one
test set under one scoring rule (ROADMAP 1.16).  Pooling costs nothing and buys five rounds.
Set `POOL_TRACKS = False` to recover the Track-1-only fit.

    python board_rounds.py
"""
from __future__ import annotations

import itertools

import numpy as np

NAMES = ["rel_l2", "tke", "mvpe", "time", "sps"]

# (label, date, submission id, rel, tke, mvpe, time, sps, final).
# Submission ids come from the Codabench history read 2026-08-31.  Two rounds -- 883908 and
# 886137 -- were submitted by a collaborator and have no package recorded in this repo; their
# subscores are unknown, so they are listed for the record but excluded from the fit.
ROUNDS = [
    ("R6  U-Net 1 inv ", "2026-08-04", 878641, 94.4342,   78.1660,   93.7922,   91.6973,   36.9374,   79.733),
    ("R7  FNO x3 3 inv", "2026-08-05", 879768, 94.5754,   78.5278,   93.9100,   91.7264,   37.4680,   79.984883),
    ("R9  FNO x8 8 inv", "2026-08-06", 881164, 94.6342,   78.8825,   93.8030,   90.5066,   38.2338,   80.115065),
    ("R10 7 memb/3 inv", "2026-08-08", 883017, 94.5764,   79.1528,   93.6812,   91.1209,   39.6263,   80.512025),
    ("X1  collaborator", "2026-08-09", 883908, 94.461888, 79.593999, 93.655155, 92.102517, 39.328139, 80.519055),
    ("R11 8 memb/3 inv", "2026-08-10", 885003, 94.570878, 79.080459, 93.350128, 92.777073, 39.545007, 80.609074),
    ("X2  collaborator", "2026-08-11", 886137, 94.461620, 78.843945, 92.926545, 93.263011, 38.878534, 80.372179),
    ("R12 11 memb/4inv", "2026-08-31", 909571, 94.490331, 79.167748, 93.418065, 91.803630, 39.637266, 80.515842),
    ("R13 8memb/1inv a60", "2026-08-31", 910641, 94.679785, 77.983720, 93.613336, 93.115432, 39.689258, 80.643823),
    # R14 is R13's members, R13's band and R13's ALPHA plus the GATED MEAN-BLEND (beta 0.15,
    # fired only for windows whose mean speed leaves the training range).  All four error
    # subscores came back BIT-IDENTICAL to R13's, so the gate never fired on a single hidden
    # window: the hidden set is INSIDE the training speed range and phi = 0.  The only moving
    # part is `time`, which makes the pair a direct measurement of `w_time` (see main()).
    ("R14 blend8/1inv a60", "2026-09-02", 912521, 94.679785, 77.983720, 93.613336, 92.978725, 39.689258, 80.630755),
    # R15 is T2R2's EXACT member set on the Track-1 board -- the third cross-track identity
    # check, and it passes: rel +0.15 %, tke -0.46 %, sps -0.02 % of the error against 912522,
    # with mvpe the outlier again at -4.17 % (36 scored probe points per window, the smallest
    # effective sample of the five, so the first place a one-window pool difference shows).
    ("R15 dropout8/1inv a60", "2026-09-03", 913176, 94.780094, 77.747643, 94.122329, 93.180177, 40.140948, 80.835124),
    # R16 is T2R3's EXACT member set on the Track-1 board -- the FOURTH cross-track identity
    # check, and the account's best score on either board.  In error terms against 913187:
    # rel -0.17 %, tke -1.52 %, mvpe -4.94 %, sps +0.098.  Same shape as the three earlier
    # checks (mvpe the outlier); tke's 1.5 % is the largest tke offset of the four and is
    # worth watching, not acting on.  Board latency 4.52 ms against T2R3's 4.78.
    ("R16 w96x8/1inv a60 fp16", "2026-09-04", 914506, 94.764049, 78.433653, 94.030389, 92.699846, 41.259088, 81.119780),
    # R17 = DEPTH at dropout 0.45: 4 all-data width-96 EIGHT-layer members, one fused
    # invocation, fp16, ALPHA 0.60.  The account's best score on either board.  Against R16
    # (8 members at FOUR layers, same width, same invocation count) it is a WASH in accuracy:
    # rel +0.37 % of error WORSE, tke -1.19 %, mvpe +1.00 % worse, sps +0.066, which the free
    # weights price at +0.023.  The rest of the +0.050 is a better time draw (4.52 -> 4.15
    # board ms).  Equal FLOPs, equal score -- 4xL8 and 8xL4 sit on one accuracy-per-FLOP
    # frontier.  The prediction (81.188) over-read tke and sps, as the FNO-family proxy has
    # now done twice at this operating point.
    ("R17 L8do45x4/1inv a60", "2026-09-04", 915380, 94.745531, 78.635462, 93.974234, 92.988361, 41.324574, 81.169843),
    ("R18 L8do45x4 hostw   ", "2026-09-06", 917177, 94.745531, 78.635462, 93.974234, 93.058602, 41.324574, 81.176572),
    # R19 = the SIM members: four all-data w96 L8 do45 FNOs CO-TRAINED on mapped sim
    # (--lam-sim 0.3 --sim-loss full), R18's exact shape so `time` is a pure draw.  It reads
    # sim's ACCURACY, and the answer is: real, half the size the proxy said, and TKE goes the
    # WRONG WAY.  rel_l2 -0.63 % of error (proxy said -1.19 %), mvpe -1.02 % (proxy -0.99 %,
    # nearly exact), tke **+0.61 % WORSE** where both the proxy (-1.04 %) and the interior
    # k-fold (+0.03 %, null) said otherwise, and `sps` +0.008 against a predicted +0.361.
    ("R19 simx4/1inv a60 fp16", "2026-09-08", 918587, 94.776881, 78.532564, 94.031877, 93.030936, 41.332982, 81.186041),
    # R20 = cmb3sim: R19's recipe plus round 18's three wave-1 winners (--lam-tgrad 0.3,
    # --lam-v 0.15, --lam-cov 2.0/--cov-c 0.25).  Same shape, member count, invocation count and
    # fast paths, so it is a single-variable A/B and the latency is R19's by construction.
    # THE ACCOUNT'S BEST SCORE ON EITHER BOARD.  Recorded prediction 81.195 (per-sample factor)
    # / 81.226 (local sps transferred 1:1); actual 81.240, i.e. above the optimistic end -- but
    # +0.023 of that is a favourable `time` draw (93.031 predicted, 93.270 delivered).  At
    # constant latency the accuracy part is +0.031 on this board and +0.049 on Track 2.
    ("R20 cmb3simx4/1inv a60", "2026-09-09", 920369, 94.794521, 78.518609, 94.081696, 93.270089, 41.410372, 81.239831),
    ("R21 xfam5 cmb3x4+ViT/2inv", "2026-09-10", 921000, 94.847008, 78.684381, 94.298686, 92.676031, 41.755710, 81.330636),
    # R23 = R20's FOUR MEMBERS, UNFUSED.  The same checkpoints, the same fp16 packing, R20's own
    # band byte for byte (md5 2eed9e6b...), `FUSE = False` the only difference -- 1 invocation
    # becomes 4.  Deliberately NOT a score attempt: it is the only way to measure the invocation
    # term, which had been FITTED since round 10 and never tested.
    #   THE DESIGN HELD.  rel/tke/mvpe came back bit-identical to six decimals and `sps` moved
    #   1.8e-05, so every point of the -0.225 is `time`.  3.795 -> 7.249 board ms for +3
    #   invocations = **1.151 ms each**, against the fitted 0.435: the model was 2.65x LIGHT.
    ("R23 cmb3simx4 UNFUSED/4inv", "2026-09-11", 921706, 94.794522, 78.518609, 94.081696, 90.932390, 41.410354, 81.015201),
    # R24 = xfam6: R21's UPLOADED archive plus one cmb3sim-recipe ViT fused into the ViT group by
    # StackedViT (fno.pth-fno_5.pth tensor-identical to R21; 2 invocations). Submission id not
    # reported. ACCURACY +0.050 -- rel -0.30 %, tke -0.65 %, mvpe -0.91 % of error, sps +0.111 =
    # forced +0.058 + CONTAINMENT +0.053, a third of the first ViT's, as the local curve said.
    # TIME -0.069: 4.553 -> 5.570 board ms for +0.316 A100 ms/window of local compute. Net
    # -0.019. Both this and T2R11 missed their estimates in the `time` row -- see the FLOPs block.
    ("R24 xfam6 R21+ViT/2inv", "2026-09-12", 0, 94.861827, 78.793104, 94.347768, 91.961259, 41.866674, 81.312041),
    # R25 = R24's EXACT archive with `AMP = "fp16"` baked in (autocast, checked against fp32 at load
    # time and dropped on any failure). Submission id not reported. THE FINGERPRINT SAYS fp16 RAN:
    # against R24, rel -0.000037, tke +0.000276, mvpe -0.000260, sps +0.001013 -- not identical to six
    # decimals, and none near the 0.01 void line. So `time` reads the precision change alone: 5.570 ->
    # 5.076 board ms (-0.494), 37 % of the -1.321 that the A100's interleaved -21 % (1.918 -> 1.510
    # ms/window) would give at LOC_SLOPE_T1 (15-60 % across Track 1's 0.03-0.30 ms drift). Estimate of
    # record 81.356 assumed 50 %; actual 81.345, the miss all `time`. NEW TRACK-1 BEST, +0.014 over R21.
    ("R25 xfam6 fp16/2inv", "2026-09-13", 0, 94.861790, 78.793380, 94.347508, 92.297891, 41.867687, 81.344644),
    # R26 = P1: the FIRST non-FNO primary family to ship. Four all-data Swin U-Net w64 L2 fused by
    # StackedSwin + R25's two ViTs, 2 invocations; band refit from the Swin twins; archive tagged
    # r27_p1swin4_a060 (the r26-tagged graph archive was never uploaded). Submission id not reported.
    #   IT LOST ON ACCURACY AND ON TIME. Against R25: rel -0.114, tke -0.126, mvpe -0.174, sps +0.128,
    #   `time` -0.770, net -0.126. Estimate of record 81.453 (rel 94.855, tke 78.871, mvpe 94.399,
    #   time 93.240, sps 41.900): every error row landed on the WRONG SIDE of R25.
    #   THE PROXY INVERTED. It read Swin as equal-or-better on all three error metrics; the board moved
    #   all three DOWN. That is the U-FNO failure a second time -- across model FAMILIES the proxy
    #   predicts neither size nor sign, only that something changed.
    #   TIME WENT BACKWARDS. The fused Swin step measured 0.63x R25's interleaved (1.019 vs 1.404
    #   ms/window), which predicted -1.245 board ms at LOC_SLOPE_T1; the board came back at 6.245 ms
    #   against R25's 5.076, i.e. +1.17. Even the fp32-fallback band (time 91.79-92.66) is above the
    #   91.528 actual, so a load-time fallback alone cannot explain it: Swin's kernels transfer badly
    #   to the eval GPU. RULE: a local latency cut measured on kernels UNLIKE the anchor's does not
    #   transfer, and LOC_SLOPE / PREC_SLOPE hold only WITHIN a kernel family.
    ("R26 p1swin4 swinx4+ViT2/2inv", "2026-09-16", 0, 94.747724, 78.667443, 94.173298, 91.528039, 41.995785, 81.218444),
    # R27 = R25's ARCHIVE WITH `GRAPH = True`, nothing else: predict()'s device-side chunk replays
    # from a CUDA graph captured at import and verified bit-exact against eager, and the band's `base`
    # is scaled by 1 + GRAPH_FP (1e-4) on the graph path so the run leaves a fingerprint. Archive
    # `track1_r26_xfam6fp16graph_a060.zip` (md5 4f83127266...), staged 09-14. Submission id not reported.
    #   THE FINGERPRINT SAYS THE GRAPH RAN. rel, tke and mvpe came back BIT-IDENTICAL to R25 to six
    #   decimals, and sps moved +8.2e-05 (41.867687 -> 41.867769): not identical, so the capture did not
    #   fall back to eager. The predicted sign was -1.35e-04 at the LOCAL 1.36 error stretch; the board's
    #   own tail gives the derivative the other sign. The MAGNITUDE (~1e-4) is what identifies the path.
    #   AND THE GRAPH MADE IT SLOWER. 5.076 -> 5.641 board ms, +0.565, where the estimate of record
    #   (81.428, bracket 81.37-81.52) assumed one invocation's launch overhead came OFF (-1.151 ms,
    #   R23). +0.565 is outside Track 1's 0.03-0.30 ms same-shape drift in the wrong direction, so
    #   either the eval GPU charges the capture/replay path more than the launches it removes, or the
    #   day's drift is larger than anything recorded. Net -0.037 against R25, ALL of it `time`.
    #   THE TRACK-1 GRAPH IS CLOSED. With R26's Swin step (+1.17 where -1.245 was predicted) and R25's
    #   fp16 (37 % of the local cut), this is the THIRD consecutive local latency read that did not
    #   transfer to the board -- and the first one on the SAME kernels as its anchor, which is what
    #   makes it the strongest of the three: host-side launch accounting does not transfer either.
    ("R27 xfam6fp16 GRAPH/2inv", "2026-09-17", 0, 94.861790, 78.793380, 94.347508, 91.914674, 41.867769, 81.307815),
    # R28 = THE F-FNO L8 TWIN ON TRACK 1, and a NEW TRACK-1 BEST: R25's package with the primary family
    # swapped -- four all-data F-FNO w40 L8 members fused by StackedFFNO on the dense path + R25's two
    # ViTs, 2 invocations, band `t2r13_p2ffnoL8x4.npz` byte for byte. The exact cross-track twin of
    # T2R13. Archive `track1_r31_p2ffnoL8_a060.zip` (md5 467a7bad...). Submission id not reported.
    #   81.401196, +0.056552 over R25, INSIDE its 81.25-81.44 bracket and 0.014 ABOVE the 81.387
    #   estimate of record -- the first Track-1 slot this campaign has both predicted and won.
    #   1. THE FAMILY TRANSFERRED ACROSS BOARDS AT 0.85x. Predicting from the T2R13 - T2R12 board pair
    #      gave rel +0.029 / tke +0.194 / mvpe -0.161 / sps +0.302; the board delivered +0.029 / +0.159
    #      / -0.142 / +0.272. Every row has the right SIGN and ~85 % of the size. `sps` +0.272 is
    #      forced only +0.019, so CONTAINMENT MOVED +0.253 here against +0.302 on Track 2 -- the same
    #      mechanism, the same fraction. A board read of a family DOES transfer; a proxy read does not.
    #   2. MVPE IS STILL THE EXCEPTION (-0.142, +2.66 % of error). It has now fallen on every package
    #      that replaced the cmb3sim FNO: Swin -0.174, F-FNO -0.161 (T2) and -0.142 (T1).
    #   3. THE FIRST CROSS-FAMILY LATENCY READ THAT LANDED ON THE RIGHT SIDE. 5.076 -> 5.480 board ms,
    #      +0.404, for the fused stack's +0.246 A100 ms/window (+0.455 4060-eq) = 0.89 board ms per
    #      4060-eq ms -- HALF of LOC_SLOPE_T1. Swin inverted outright; this one is merely cheap. Kept
    #      as LOC_SLOPE_T1_XFAM below, NOT as a change to LOC_SLOPE_T1. The estimate assumed the full
    #      1.75 and so under-read `time` by 0.26 subscore, which is where the +0.014 beat came from.
    #      Caveat: the eval machine ran fast on BOTH boards today (see T2R15), so part of the +0.40 vs
    #      +0.80 is same-day drift.
    #   R28 IS THE TRACK-1 ANCHOR FOR F-FNO PACKAGES; `r25` stays the anchor for FNO packages.
    ("R28 p2ffnoL8 ffnox4+ViT2/2inv", "2026-09-18", 0, 94.890524, 78.952686, 94.205730, 92.021555, 42.139290, 81.401196),
]

UNSCORED = []   # every round on both tracks now has full subscores

# ---------------------------------------------------------------------------------------
# TRACK 2.  Same shape:  (label, date, submission id, rel, tke, mvpe, time, sps, final)
#
# Round 1 shipped the SAME 8 members, band and ALPHA as Track 1's round 13, so its four error
# subscores are a direct test of whether the two tracks score the same hidden trajectories.
# ANSWERED 2026-08-31, and the answer is YES for three of the four:
#
#     subscore     T1 R13      T2 R1      diff     rel. diff in the ERROR
#     rel_l2      94.679785  94.694512   +0.015          -0.28 %
#     tke         77.983720  77.942224   -0.041          +0.24 %
#     mvpe        93.613336  93.399401   -0.214          +3.59 %
#     sps         39.689258  39.730331   +0.041          -0.10 %
#
# Three of four agree to <=0.3 % of the error -- far inside anything a different test set
# could produce, and far outside float32 noise being the explanation for mvpe. So the sets
# are the same or near-identical, and MVPE, which is read at 4 probe COLUMNS (36 points of
# 2048), is simply the metric with the smallest effective sample size: it is the first place
# a small difference in the scored window set shows up.  Working conclusion: **Track-1
# instruments transfer to Track 2**; treat mvpe as the one number to re-measure per track.
#
# Track 2's `final_score` combination is separately unpublished. Do not assume it is Track
# 1's: with one round it is not identified at all.  (For the record, Track 1's fitted weights
# applied to T2 R1's subscores give 80.667 against the published 80.653.)
#
# CONFIRMED 2026-09-02, and again 2026-09-03: the pooled fit's max residual is 0.038 at twelve
# rounds and 0.044 at fourteen, no worse than the nine Track-1 rounds alone (0.035) -- the
# direct evidence that pooling is legitimate, since a different combination rule on Track 2
# would have shown up as a residual.  Round 15 tightens it further: R15 shipped T2R2's EXACT
# members to the Track-1 board and three of four error subscores came back within 0.5 % of the
# error (rel +0.15 %, tke -0.46 %, sps -0.02 %; mvpe -4.17 %, the usual outlier).
ROUNDS_T2 = [
    ("T2R1 8memb/1inv a60", "2026-08-31", 910642, 94.694512, 77.942224, 93.399401, 93.287797, 39.730331, 80.652944),
    # 8 all-data DROPOUT members (4x0.05, 4x0.15), 1 invocation, ALPHA 0.60.  +0.200 over T2R1
    # and the account's best score on either board.
    ("T2R2 dropout8/1inv a60", "2026-09-02", 912522, 94.787655, 77.667926, 93.882222, 93.633350, 40.148056, 80.852826),
    # 8 all-data WIDTH-96 FNOs (4x dropout 0.30, 4x 0.45), 400 epochs, fp16-packed, ALPHA 0.60.
    # NEW BEST on either board.  The gain is `sps` (+1.013) and `tke` (-2.9 % of error) while
    # `rel_l2` got fractionally WORSE (+0.64 %): capacity x dropout buys the per-element TAIL,
    # not the rms.  It paid 1.41 board ms for it (3.370 -> 4.780).
    ("T2R3 w96x8/1inv a60 fp16", "2026-09-03", 913187, 94.755824, 78.174302, 93.739707, 92.508902, 41.160630, 81.019213),
    # 3 all-data U-FNO members (width 96, dropout 0.30, 400 ep), 3 invocations, ALPHA 0.60.
    # MISSED its prediction by -0.45 (predicted 81.33), and most of the miss is ACCURACY, not
    # latency: against the T2R3 anchor rel_l2 came back +1.90 % WORSE where the 3-vs-3 said
    # -4.4 %, a 4.0-pp instrument residual and the largest on record.  Latency was the other
    # half: 8.69 board ms against T2R3's 4.78 (the U-Net branch is conv-heavy at batch 1 and
    # `allow_tf32 = False` makes 3x3 convs ~50 % slower -- 5.457 ms/step measured, not the
    # 1.28 the staging README claimed).  At T2R3's latency it would have scored 81.086, +0.07.
    # Working hypothesis: U-FNO's local advantage is in-distribution memorization; the
    # interior Re-band k-fold, the one instrument that called T2R3's shape, was never run on
    # it.  Do NOT submit the Track-1 twin of this package.
    ("T2R4 ufno3/3inv a60 fp16", "2026-09-04", 914508, 94.661471, 78.413683, 93.727402, 90.154359, 41.600131, 80.879200),
    # T2R5 = THE SAME FOUR MEMBERS AS R17 on the Track-2 board -- the FIFTH cross-track
    # identity check.  (The staged recommendation for this slot was the do30 twin
    # `track2_r16_L8x4_a060.zip`; the user uploaded the do45 archive, so the pair became an
    # identity check instead of the dropout-at-depth A/B it was designed as.)  R17 vs T2R5 in
    # error terms: rel -0.36 %, tke -1.64 %, mvpe -4.76 %, sps +0.133 -- Track 1 better on
    # every error subscore.  Since w96 the T1 package has beaten its T2 twin on tke by ~1.5 %
    # three times running; at w24 the two tracks agreed to +-0.3 %.  Something in the Track-2
    # path costs ~0.03 `final` at width 96 (plan_20260905.md Q1e).
    # Against its own T2R3 anchor the depth change is a wash, exactly as on Track 1: rel
    # +0.57 % worse, tke -1.07 %, mvpe +0.81 % worse, sps +0.031 -> fitted +0.0003, actual
    # +0.0001.  The user's leaderboard LIST line read 80.97843 while the detail read
    # 81.019298; 81.019298 is the consistent value -- hold it out of the fit and the other 17
    # rounds predict 81.02032, which is 0.8 sigma from 81.019298 and 35 sigma from 80.97843.
    ("T2R5 L8do45x4/1inv a60", "2026-09-04", 915381, 94.727329, 78.357043, 93.692366, 92.427051, 41.191134, 81.019298),
    ("T2R6 L8do45x4 bmm     ", "2026-09-06", 917178, 94.727329, 78.357043, 93.692366, 92.399134, 41.191137, 81.016621),
    # T2R7 = R19's IDENTICAL members and band on the other board -- the SIXTH cross-track
    # identity check, and the second reading of sim.  Every error subscore moves in the same
    # direction as R19's and by a similar amount (rel -0.71 % vs -0.63 %, tke +0.83 % WORSE vs
    # +0.61 %, mvpe -2.61 % vs -1.02 %, sps +0.020 vs +0.008), so the sim reading is not a
    # single-board accident.  Its larger `final` is a `time` draw (+0.497 score), not accuracy.
    ("T2R7 simx4/1inv a60 fp16", "2026-09-08", 918588, 94.762818, 78.220131, 93.843944, 92.896212, 41.210644, 81.086639),
    # T2R8 = R20's IDENTICAL members and band -- the SEVENTH cross-track identity check, and the
    # second reading of cmb3sim.  Every error subscore moves in the same direction as R20's and
    # by a similar amount (rel -0.39 % vs -0.36 %, tke -0.34 % vs +0.08 %, mvpe -1.54 % vs
    # -0.89 %, sps +0.100 vs +0.077), so cmb3sim is not a single-board accident.  Its SMALLER
    # `final` gain is a `time` draw (-0.299 score against R20's +0.239): at constant latency
    # this board's accuracy gain is the LARGER of the two, +0.049 against +0.031.
    ("T2R8 cmb3simx4/1inv a60", "2026-09-09", 920370, 94.782311, 78.278425, 93.933159, 92.597588, 41.310588, 81.106430),
    ("T2R9 xfam5 cmb3x4+ViT/2inv", "2026-09-10", 921001, 94.841673, 78.469938, 94.157190, 91.521812, 41.649497, 81.155762),
    # T2R10 = the same unfused experiment on the streaming board, run the same day.  Identity
    # held here too (rel/tke/mvpe to six decimals, sps 3e-06).  4.659 -> 7.255 board ms =
    # **0.866 ms per invocation**, 1.99x the fitted 0.435.
    #   AND THE TWO TRACKS MEET.  At 1 invocation T2 costs +0.864 ms more than T1; at 4 they are
    #   7.249 vs 7.255.  So Track 2's streaming penalty is a FIXED per-step overhead, not a
    #   per-invocation one, and Track 1 actually pays MORE per invocation (1.151 vs 0.866).
    #   Round 22's reading -- "the invocation tax is 2.4x heavier on Track 2" -- is withdrawn:
    #   what was heavier on Track 2 that round was the ViT member, not the invocation.
    ("T2R10 cmb3simx4 UNFUSED/4inv", "2026-09-11", 921707, 94.782311, 78.278425, 93.933159, 90.928559, 41.310591, 80.945935),
    # T2R11 = x8v: R21's UPLOADED archive plus FNO s5-s8 fused into the FNO group, with fno.pth-
    # fno_5.pth AND the band byte-identical to T2R9's. Submission id not reported. MEMBER COUNT
    # IS CLOSED: accuracy -0.020 (rel -0.09 %, tke +0.44 % worse, mvpe +1.64 % worse; sps -0.023 =
    # forced -0.046 + containment +0.023, inside noise), exactly the null the local K-curve
    # extrapolation gave once the ViT is in the set. TIME -0.077: 6.255 -> 7.628 board ms. Net -0.097.
    ("T2R11 x8v R21+4FNO/2inv", "2026-09-12", 0, 94.845935, 78.395628, 94.066988, 90.720061, 41.626260, 81.058649),
    # T2R12 = R25's archive on the streaming board, read against T2R9: the second ViT AND fp16 at once.
    # Submission id not reported. The errors landed on the cross-track prediction (rel +0.004, tke
    # +0.008, mvpe +0.017, sps -0.0003) and sps +0.111 against T2R9 = forced +0.070 + containment
    # +0.041, the second ViT as on R24 (+0.053). `time` 91.040 is below the 91.52 line: fp16 did NOT
    # pay for the second ViT's FLOPs here -- 6.255 -> 7.061 board ms, against 7.795 with no fp16
    # transfer, so ~0.73 of a possible 3.61 ms came back (inside Track 2's drift). Net +0.009 over T2R9:
    # NEW TRACK-2 BEST. Estimate of record 81.223; the miss is the `time` row (-0.063).
    ("T2R12 xfam6 fp16/2inv", "2026-09-13", 0, 94.861391, 78.587250, 94.224322, 91.039757, 41.760694, 81.164374),
    # T2R13 = P2-L8: four all-data F-FNO w40 L8 fused by StackedFFNO on the DENSE spectral path +
    # T2R12's two ViTs (tensor-identical), 2 invocations; band refit from the L8 twins; archive tagged
    # r27_p2ffno4_a060. Submission id not reported. NEW TRACK-2 BEST, +0.150 over T2R12, board 6.069 ms.
    #   Against T2R12: rel +0.029, tke +0.194, mvpe -0.161, `time` +0.599, sps +0.302, net +0.150.
    #   It landed at the OPTIMISTIC end of its bracket (estimate of record 81.237, optimistic 81.329,
    #   actual 81.315), the first package this campaign has UNDER-predicted.
    #   THE FAMILY TRANSFERRED ON TRACK 2 where it inverted on Track 1's Swin: rel and tke both up,
    #   and sps +0.302 is far above the per-sample factor alone, so the F-FNO family moved CONTAINMENT.
    #   `time` +0.599 for a step measured at 0.861x T2R12's -- the first latency cut this project has
    #   predicted and collected inside its bracket, and the pair that re-measures LOC_SLOPE_T2 below.
    #   MVPE IS THE EXCEPTION: -0.161 here and -0.174 on Swin the same day, the only metric to fall on
    #   BOTH tracks. FNO holds mvpe better than either new family.
    ("T2R13 p2ffnoL8 ffnox4+ViT2/2inv", "2026-09-16", 0, 94.890753, 78.781089, 94.063576, 91.638537, 42.063113, 81.314785),
    # T2R14 = P2-L4: T2R13 with ONE lever, DEPTH. Four all-data F-FNO w40 **L4** members fused by
    # StackedFFNO on the dense path + T2R13's two ViTs, band refit from the held-out L4 twins, ALPHA
    # 0.60, fp16 / bmm / dense, 2 invocations. Archive `track2_r30_p2ffnoL4_a060.zip` (md5 38c15783...,
    # build job 7332924). Submission id not reported. Estimate of record 81.371, bracket 81.34-81.41.
    #   IT LANDED AT 81.308, BELOW ITS OWN BRACKET, for -0.006 against T2R13: rel +0.024, tke -0.030,
    #   mvpe +0.073, `time` +0.473, sps -0.273.
    #   1. DEPTH IS A WASH ON THE THREE ERRORS, exactly as the L8->L4 screen said it would be. The
    #      predicted tke cost (-0.166) came in at -0.030, rel rose 0.024 and mvpe 0.073. Halving the
    #      cost of the stack costs nothing measurable in rms error.
    #   2. THE SLOT WAS LOST IN CONTAINMENT. With the a-factor computed from the three actual errors
    #      (a = 0.5g(rel) + 0.3g(tke) + 0.2g(mvpe), g(m) = 0.5/(0.5+m)), the FORCED part of sps is
    #      +0.038; the board returned -0.273, so CONTAINMENT FELL -0.310. The L4 sigma head is worse
    #      calibrated than L8's, and NO local instrument reads containment across depth -- the band is
    #      refit per package, and every refit scores well on its own twins.
    #   3. THE 0.937 SLOPE WAS A HIGH DRAW. The interleaved build read -0.738 A100 ms/step (2.6325 vs
    #      P2-L8's 3.3707) = -1.366 4060-eq; the board saved 0.723 ms (6.069 -> 5.346), i.e. **0.529
    #      board ms per 4060-eq ms**. Predicted `time` was 92.502 at 0.937 and 92.178 at the old 0.60;
    #      the actual 92.112 is below BOTH. See LOC_SLOPE_T2 -- it is now the mean of the two reads.
    #   VERDICT: F-FNO DEPTH IS CLOSED ON THE BOARD TOO. L8 stays the Track-2 package and T2R13
    #   (81.314785) remains the Track-2 best; L4 buys latency and hands it straight back in containment.
    ("T2R14 p2ffnoL4 ffnoL4x4+ViT2/2inv", "2026-09-17", 0, 94.914425, 78.750833, 94.136582, 92.111693, 41.790524, 81.308359),
    # T2R15 = T2R13 WITH THE ViT PAIR REMOVED: the same four all-data F-FNO w40 L8 members fused on the
    # dense path, ALONE, at ONE invocation, band refit from the L8 twins only. Archive
    # `track2_r28_l8x4_a060.zip` (md5 db1808de..., built and gated 09-15). Submission id not reported.
    #   81.265138, -0.049647 against T2R13, AT THE FLOOR of its 81.26-81.48 bracket (estimate 81.431).
    #   1. THE ViT PAIR STAYS IN EVERY PACKAGE. Priced at the fitted weights, dropping it costs
    #      -0.155 of accuracy+containment and returns +0.105 of `time`, i.e. NET -0.050. sps -0.321 is
    #      forced -0.188 plus CONTAINMENT -0.132 -- three times the -0.041 the estimate assumed from
    #      T2R12. Two ViTs at 14.8 MB each buy containment no F-FNO member reproduces.
    #   2. "HALVE THE LOCAL DELTAS" UNDER-READ THIS ONE. The un-halved AoA 15-20 ratios (rel x1.0165,
    #      tke x1.0195, mvpe x1.0297) predict rel 94.811 / tke 78.457 / mvpe 93.886 against the actual
    #      94.823 / 78.519 / 93.886 -- essentially exact, where the halved version missed all three.
    #      The 2x over-read rule is for an arm that REMOVES error; a member REMOVAL is charged in full.
    #   3. A FOURTH TRACK-2 LATENCY READING, AND THE LOWEST YET. 6.069 -> 4.480 board ms, -1.589,
    #      against -2.736 predicted (58 % realized). Charging the dropped invocation at INV_MS_T2
    #      0.866 leaves -0.723 board ms for -2.551 4060-eq = 0.28 -- and if the invocation term is
    #      itself smaller today, both are. 4.480 ms is also BELOW T2R8's 4.659 ms floor for a
    #      one-invocation FNO x4 package, so the machine was fast: same-day drift on both boards.
    #   LOC_SLOPE_T2 goes back to 0.60 with an explicit 0.3-0.9 bracket (see below). T2R13
    #   (81.314785) remains the Track-2 best.
    ("T2R15 l8x4 ffnox4 noViT/1inv", "2026-09-18", 0, 94.823177, 78.519336, 93.885731, 92.730627, 41.742474, 81.265138),
]

# The fit pools both tracks -- see the module docstring.  ROUNDS first, then ROUNDS_T2;
# `final_v9.BOARD` lists the same rounds by submission date instead, which is harmless (least
# squares does not see row order) and is checked as a multiset by `final_v9._check_sync`.
POOL_TRACKS = True

FIT_ROUNDS = ROUNDS + ROUNDS_T2 if POOL_TRACKS else ROUNDS
X = np.array([r[3:8] for r in FIT_ROUNDS], float)
Y = np.array([r[8] for r in FIT_ROUNDS], float)

# `w_time`, MEASURED DIRECTLY rather than fitted -- now FIVE TIMES, by five single-subscore
# A/Bs in which the four error subscores came back bit-identical and only `time` moved:
#   R13   -> R14   the gated mean-blend fired on ZERO hidden windows
#   R17   -> R18   HOST_WRITE="torch": the same bytes to the same place, threaded
#   T2R5  -> T2R6  SPEC_PATH="bmm": the spectral product as one batched real matmul (2e-06)
#   R20   -> R23   FUSE=False.  dtime -2.338, the largest single-subscore move ever submitted
#   T2R8  -> T2R10 the same, on the other board
# The first three had `dtime` of 0.03-0.14 score points, so each divided two small numbers; the
# round-23 pair moved `time` by 1.7-2.3 points and lands at 0.09609 / 0.09616 -- 0.07 % apart,
# and the free fit's 0.0965 sits between them.  This weight is now MEASURED, not estimated, and
# the constrained fit's 0.085 is excluded by 15 %.
W_TIME_AB = [
    ("R13->R14  ", 92.978725 - 93.115432, 80.630755 - 80.643823),
    ("R17->R18  ", 93.058602 - 92.988361, 81.176572 - 81.169843),
    ("T2R5->T2R6", 92.399134 - 92.427051, 81.016621 - 81.019298),
    ("R20->R23  ", 90.932390 - 93.270089, 81.015201 - 81.239831),
    ("T2R8->T2R10", 90.928559 - 92.597588, 80.945935 - 81.106430),
]
W_TIME_DIRECT = float(np.mean([f / t for _, t, f in W_TIME_AB]))   # = 0.0958
# R24 -> R25 (fp16 autocast) is a sixth, NEAR-single-subscore pair, kept OUT of the mean: the error
# subscores moved by <= 3e-4 and sps by 1.0e-3, so it is not bit-identical. Raw dfinal/dtime 0.0969;
# with those small moves priced out at the fitted weights, 0.0961 -- inside the five readings above.


# ---------------------------------------------------------------------------------------
# THE INVOCATION, MEASURED (round 23, 2026-09-11).
#
# `board ms = a + b*invocations + c*local ms` had been FITTED since round 10 and never tested.
# Round 23 tested it the only way it can be tested: ship a package that differs from one
# already on the board in the invocation count and in NOTHING else.  `track{1,2}_r23_nofuse4
# fp16_a060.zip` are round 20's four checkpoints, round 20's fp16 packing and round 20's band
# byte for byte, with `FUSE = False` -- so `_fno.fuse()` does not collapse the four width-96
# L8 members into one grouped StackedFNO and the shipped path runs four sequential forwards.
#
#   the gate:  rel_l2, tke and mvpe came back IDENTICAL TO SIX DECIMALS on both boards and
#              `sps` moved 1.8e-05 / 3e-06.  Nothing but `time` moved, which is what makes the
#              rest of this block a measurement instead of an inference.
#   the read:  T1  3.795 -> 7.249 ms over +3 invocations = 1.151 ms each  (fitted 0.435: x2.65)
#              T2  4.659 -> 7.255 ms over +3 invocations = 0.866 ms each  (fitted 0.435: x1.99)
#
# Two things fall out that no fit could have given:
#
#   1. AN INVOCATION IS EXPENSIVE.  Priced through `w_time` at round 20's operating point, the
#      FIRST extra invocation costs -0.085 `final` on Track 1 and -0.058 on Track 2 (the curve
#      is convex, so the fourth costs -0.066 / -0.049).  That is the size of a good accuracy
#      arm.  Any package with more than one architecture family has to earn it back.
#   2. TRACK 2'S PENALTY IS FIXED, NOT PER-INVOCATION -- and round 22's reading is WITHDRAWN.
#      At 1 invocation Track 2 costs +0.864 ms more than Track 1; at 4 the two boards land at
#      7.249 and 7.255.  So the streaming path's cost is a per-STEP overhead that the member
#      set does not multiply, and Track 1 actually pays MORE per invocation.  "The invocation
#      tax is 2.4x heavier on Track 2" (round 22, from R21/T2R9) was reading the VIT's forward
#      pass, which is heavier at batch 1, not the invocation.
#
# Use these as DELTAS against a recent anchor of the same shape.  The level model is withdrawn:
# its residuals reach -2.5 ms and are ordered by date.
INV_MS_T1 = (0.72896 * (100.0 / 90.932390 - 1.0) ** 2
             - 0.72896 * (100.0 / 93.270089 - 1.0) ** 2) * 1000.0 / 3.0     # = 1.151
INV_MS_T2 = (0.72896 * (100.0 / 90.928559 - 1.0) ** 2
             - 0.72896 * (100.0 / 92.597588 - 1.0) ** 2) * 1000.0 / 3.0     # = 0.866


def board_ms(time_score: float) -> float:
    """Invert `time_score = 100/(1+sqrt(t/0.72896))` back to board milliseconds."""
    return 0.72896 * (100.0 / float(time_score) - 1.0) ** 2 * 1000.0


def time_score(ms: float) -> float:
    """Forward: board milliseconds -> `time_score`."""
    return 100.0 / (1.0 + (float(ms) / 1000.0 / 0.72896) ** 0.5)


def invocation_cost(anchor_time_score: float, n: int = 1, track: int = 1) -> tuple:
    """(d time_score, d final) of adding `n` sequential forwards to a package scoring `anchor`.

    The time curve is convex, so this depends on where you already are -- a second invocation
    on a 3.8 ms package costs more than a fifth one on a 7 ms package.  Always pass the
    anchor's own `time_score`.
    """
    per = INV_MS_T1 if track == 1 else INV_MS_T2
    d = time_score(board_ms(anchor_time_score) + n * per) - anchor_time_score
    return d, d * W_TIME_DIRECT


# ---------------------------------------------------------------------------------------
# ROUND 24 CORRECTS ROUND 23: FLOPs ARE CHARGED TOO (2026-09-12).
#
# Round 23 read "the board is launch-bound, not FLOP-bound" off the unfused package. What it
# measured was that LAUNCHES are expensive; it could not show that FLOPs are free, and they are
# not. Round 24 added compute INSIDE fused groups, on both boards, the same day:
#   R21  -> R24     + one ViT in the ViT group    local +0.316 A100 ms/window   4.553 -> 5.570 ms
#   T2R9 -> T2R11   + four FNOs in the FNO group  local +1.234 A100 ms/step     6.255 -> 7.628 ms
# Both boards +22 %. On Track 1 that is 1.74 board ms per 4060-equivalent local ms, and the only
# SAME-DAY FLOPs pair ever submitted (R16 -> R17, w96 L4 x8 -> L8 x4) reads 1.76. Track 1's
# board/local ratio has sat at 1.48-1.66 across the four rounds with a trusted local timing, and
# its identical-shape day-to-day drift has been 0.03-0.30 ms, so +1.02 ms is not drift. Track 2
# reads 0.60, one reading at ~2.7 sigma of its 0.4-0.7 ms drift. The 0.336 "local slope" was a
# cross-day refit over residuals ordered by date, and on Track 1 it was 5x light.
# At R21's operating point, +0.1 A100 ms/window of local compute is ~ -0.023 `final` on Track 1,
# and a board ms saved is ~ +0.072.
LOC_SLOPE_T1 = 1.75     # board ms per 4060-equivalent local ms, Track 1 (R16->R17 1.76, R21->R24 1.74)
# ...but only WITHIN a kernel family. R28 (2026-09-18) is the first CROSS-FAMILY Track-1 read that did
# not invert: swapping the four fused cmb3sim FNOs for four fused F-FNO w40 L8 cost +0.246 A100
# ms/window (+0.455 4060-eq) and the board charged +0.404 ms, i.e. 0.89 -- HALF of LOC_SLOPE_T1. Swin
# (R26) inverted outright and the CUDA graph (R27) inverted on the SAME kernels, so the honest reading
# is that a cross-family Track-1 estimate belongs in a bracket [0, 1.75] with 0.89 as its only anchor.
# One reading, and the eval machine ran fast on both boards that day (T2R15 came in under T2R8's floor).
LOC_SLOPE_T1_XFAM = 0.89
# AND THE TWO LOCAL INSTRUMENTS DISAGREE WITH EACH OTHER, not just with the board (rp_latCs, 09-18).
# The SAME pair, R28's stack against R25's, reads +0.797 board ms at the KERNEL level (verify/fuse_ffno
# §7, the fused stacks timed on their own) and only **+0.164** at the PACKAGE level (rp_latCs, whole
# `predict()` interleaved in one process). The board delivered **+0.404**, between them. So the kernel
# read is 2.0x heavy, the package read is 2.5x light, and NEITHER instrument alone is trustworthy for a
# cross-family prediction -- quote both ends as the bracket and expect the board in the middle. (The
# gap is real work the kernel timer does not see and the package timer amortizes: normalization, the
# compose step, the band and the host write are common to both packages and dilute the stack's share.)
# Track 2's FLOPs slope, FOUR READINGS, all single cross-date pairs on a machine that drifts by date:
#   T2R9  -> T2R11  (09-10 -> 09-12)  +1.234 A100 ms/step  -> +1.373 board ms   = 0.60
#   T2R12 -> T2R13  (09-13 -> 09-16)  -0.573 A100 ms/step  -> -0.992 board ms   = 0.937
#   T2R13 -> T2R14  (09-16 -> 09-17)  -0.738 A100 ms/step  -> -0.723 board ms   = 0.529
#   T2R13 -> T2R15  (09-16 -> 09-18)  -1.379 A100 ms/step  -> -1.589 board ms, of which -0.866 is the
#                                     dropped invocation, leaving -0.723 over -2.551 4060-eq = ~0.28
#                                     (and if INV_MS_T2 is smaller today, both terms are)
# 0.937 was a HIGH DRAW: three of the four readings sit at or below 0.60, and the two most recent are
# the lowest. The shipped constant goes BACK TO 0.60, which is the median and the only value with two
# readings near it; bracket every Track-2 latency lever with **0.3 and 0.9** rather than pricing it at
# a point. T2R15 also came in at 4.480 board ms, BELOW T2R8's 4.659 one-invocation floor, so part of
# its cut is same-day machine speed and the ~0.28 is the low end of what a slope can mean.
LOC_SLOPE_T2 = 0.60     # 0.60 -> 0.937 (09-16) -> 0.733 (09-17) -> back to 0.60 (09-18); bracket 0.3-0.9

# ROUND 25: A PRECISION CUT IS NOT CHARGED LIKE A FLOPs CUT (2026-09-14). R25 is R24's archive with
# fp16 autocast baked in, and its error subscores moved by <= 3e-4 (the fingerprint that fp16 ran),
# so `time` reads the precision change alone: 5.570 -> 5.076 board ms for -0.408 A100 ms/window
# measured interleaved (-0.755 4060-eq). That is 0.65 board ms per 4060-eq local ms, 37 % of
# LOC_SLOPE_T1: the board realises much less of fp16 than the A100's tensor cores do. One reading;
# Track 1's 0.03-0.30 ms drift brackets it at 0.26-1.05. On Track 2, T2R12 moved fp16 and the second
# ViT together; charging the ViT at LOC_SLOPE_T2, fp16 recovered ~0.73 of 3.61 board ms (~0.12 per
# 4060-eq ms), which is inside that board's 0.4-0.7 ms drift of zero.
PREC_SLOPE_T1 = 0.65    # board ms per 4060-eq local ms saved by a PRECISION change, Track 1 (R24 -> R25)


def fit(constrained: bool = False, nonneg: bool = True):
    """Least squares for `final = w . subscores`, no intercept.

    Default (round 17 onward) is the FREE fit: eighteen pooled rounds against five weights,
    13 dof, max |residual| 0.0023.  `constrained=True` adds the historical `sum(w) = 1` row,
    which costs a factor of twenty in residual (0.049) and is what rounds 9-16 reported.  The
    constraint was assumed, never tested; the residual is the test and it fails it.
    """
    if not constrained:
        w, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return w, X @ w - Y
    from scipy.optimize import nnls
    A = np.vstack([X, np.ones(5) * 100.0])       # scale the constraint row into range
    b = np.concatenate([Y, [100.0]])
    if nonneg:
        w, _ = nnls(A, b)
    else:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return w, X @ w - Y


def loo():
    """Leave-one-round-out: weight spread, and how well a held-out round's `final` is predicted.

    The honest test of a fit with no held-out data of its own.  Refit on 17 rounds, predict
    the 18th; the spread of the weights says how much any single round is carrying.
    """
    n = len(Y)
    ws, errs = [], []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        wi, *_ = np.linalg.lstsq(X[m], Y[m], rcond=None)
        ws.append(wi)
        errs.append(float(X[i] @ wi - Y[i]))
    return np.array(ws), np.array(errs)


def weight_bounds(tol_mult: float = 2.0, tol_floor: float = 0.005,
                  constrained: bool = False):
    """Range of each weight over every non-negative weighting that fits nearly as well.

    Linear program: minimise/maximise w_i subject to |X w - Y| <= tol and w >= 0, plus
    sum(w) = 1 only when `constrained`.  `tol` is whichever is larger -- a multiple of the
    best achievable max-residual, or a floor -- so a knife-edge best fit does not collapse the
    reported interval to a point.  With the constraint dropped the best fit is 20x tighter, so
    the floor drops with it (0.01 -> 0.005); the resulting family is far narrower than the one
    rounds 9-16 reported, which is the point.
    """
    from scipy.optimize import linprog
    w_best, r_best = fit(constrained=constrained)
    tol = max(tol_mult * np.abs(r_best).max(), tol_floor)
    n = 5
    A_ub = np.vstack([X, -X])
    b_ub = np.concatenate([Y + tol, -(Y - tol)])
    eq = dict(A_eq=np.ones((1, n)), b_eq=np.array([1.0])) if constrained else {}
    out = {}
    for i in range(n):
        c = np.zeros(n); c[i] = 1.0
        lo = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, **eq)
        hi = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, **eq)
        out[NAMES[i]] = (lo.x[i] if lo.success else np.nan,
                         hi.x[i] if hi.success else np.nan)
    return w_best, np.abs(r_best).max(), tol, out


def compare(a_label, b_label):
    """Subscore-by-subscore delta between two rounds, priced with the fitted weights."""
    ia = [r[0].startswith(a_label) for r in FIT_ROUNDS].index(True)
    ib = [r[0].startswith(b_label) for r in FIT_ROUNDS].index(True)
    w, _ = fit()
    d = X[ib] - X[ia]
    print(f"\n=== {FIT_ROUNDS[ib][0].strip()} vs {FIT_ROUNDS[ia][0].strip()} ===")
    print(f"  {'subscore':<10s} {'delta':>9s} {'x weight':>9s} {'= final':>9s}")
    for n, dv, wv in zip(NAMES, d, w):
        print(f"  {n:<10s} {dv:>+9.3f} {wv:>9.3f} {dv * wv:>+9.3f}")
    print(f"  {'TOTAL':<10s} {'':>9s} {'':>9s} {float(d @ w):>+9.3f}"
          f"   (actual {Y[ib] - Y[ia]:+.3f})")


def main():
    print("=== Scored rounds ===")
    print(f"{'round':<24s} {'tr':>2s} {'date':<11s} {'id':>7s}  " +
          "  ".join(f"{n:>9s}" for n in NAMES) + "      final")
    for r in FIT_ROUNDS:
        tr = 2 if r in ROUNDS_T2 else 1
        print(f"{r[0]:<24s} {tr:>2d} {r[1]:<11s} {r[2]:>7d}  " +
              "  ".join(f"{v:9.4f}" for v in r[3:8]) + f"  {r[8]:9.5f}")

    wc, resc = fit(constrained=True)
    _, res_free = fit()
    w, res, tol, bounds = weight_bounds()
    ndof = len(FIT_ROUNDS) - 5
    print("\n=== final_score = w . subscores,  no intercept ===")
    print(f"  {len(FIT_ROUNDS)} rounds"
          f"{' POOLED across both tracks' if POOL_TRACKS else ' (Track 1 only)'}"
          f", five weights -> {ndof} degrees of freedom free / {ndof - 1} constrained")
    print("\n  --- (a) the historical fit: sum(w) = 1, w >= 0  [rounds 9-16] ---")
    print(f"  {'subscore':<10s} {'fitted':>8s}")
    for n, v in zip(NAMES, wc):
        print(f"  {n:<10s} {v:>8.3f}")
    print(f"  {'sum':<10s} {wc.sum():>8.3f}      max |residual| {np.abs(resc).max():.4f}")

    print("\n  --- (b) THE FREE FIT: no constraint at all  [round 17 onward, the default] ---")
    print(f"  max |residual| {res:.4f}   rms {np.sqrt((res_free ** 2).mean()):.4f}"
          f"   -- {np.abs(resc).max() / max(res, 1e-12):.0f}x tighter than (a)")
    print(f"  weight interval = every non-negative weighting fitting to within {tol:.4f}\n")
    print(f"  {'subscore':<10s} {'fitted':>8s}   {'admissible':>18s}   {'sum=1 fit':>9s}"
          f"   round-9 claim")
    r9 = dict(rel_l2="0.266", tke="0.244", mvpe="0.241", time="0.069", sps="0.181")
    for n, v, vc in zip(NAMES, w, wc):
        lo, hi = bounds[n]
        print(f"  {n:<10s} {v:>8.4f}   {lo:>8.4f} - {hi:<7.4f}   {vc:>9.3f}   {r9[n]}")
    print(f"  {'sum':<10s} {w.sum():>8.4f}"
          f"   -- NOT 1, and nothing forces it to be; see the module docstring")

    print("\n  --- and one weight is MEASURED, not fitted: THREE single-subscore A/Bs ---")
    print("  Each pair shipped the SAME prediction array by a different route, so all four")
    print("  error subscores came back identical and only `time` moved.")
    print(f"    {'pair':<11s} {'d time':>10s} {'d final':>10s} {'w_time':>8s}")
    for nm, dt, df in W_TIME_AB:
        print(f"    {nm:<11s} {dt:>+10.6f} {df:>+10.6f} {df / dt:>8.5f}")
    print(f"    => w_time = {W_TIME_DIRECT:.4f}   (FREE fit {w[3]:.4f}, "
          f"admissible {bounds['time'][0]:.4f}-{bounds['time'][1]:.4f}; "
          f"sum=1 fit {wc[3]:.4f})")
    print("  A single-subscore A/B is the only way to read a weight without the fit.  It lands")
    print("  on the FREE fit and is 12 % away from the constrained one: the direct measurement")
    print("  picks the model, and it picks the one without the assumed constraint.")

    print("\n=== Leave-one-round-out (free fit) ===")
    ws, errs = loo()
    print("  Refit on 17 rounds, predict the 18th.  This is the only out-of-sample number the")
    print("  weight family has, and it is what licenses calling it an instrument.")
    print(f"    prediction error   max {np.abs(errs).max():.4f}   rms "
          f"{np.sqrt((errs ** 2).mean()):.4f}")
    print(f"\n  {'subscore':<10s} {'full fit':>9s}   {'LOO min':>8s} - {'LOO max':<8s}")
    for i, n in enumerate(NAMES):
        print(f"  {n:<10s} {w[i]:>9.4f}   {ws[:, i].min():>8.4f} - {ws[:, i].max():<8.4f}")
    worst = int(np.argmax(np.abs(errs)))
    print(f"\n  worst held-out round: {FIT_ROUNDS[worst][0].strip()} "
          f"({errs[worst]:+.4f})")

    print("\n=== Deltas, R12 vs the two rounds that matter ===")
    compare("R10", "R12")
    compare("R11", "R12")

    print("\n=== Time-normalised final: what each round scores at a COMMON board latency ===")
    print("  Board `time_score` swings 3.5 points across the eighteen rounds and is only partly")
    print("  under our control, so the raw ranking conflates model quality with eval-machine")
    print("  luck. This holds time_score fixed at the median and re-prices the other four.")
    ref = float(np.median(X[:, 3]))
    lo_t, hi_t = bounds["time"]
    print(f"\n  reference time_score = {ref:.3f}\n")
    print(f"  {'round':<24s} {'final':>9s} {'time adj':>9s} {'normalised':>11s}"
          f" {'(range)':>16s}")
    rows = []
    for r, x, y in zip(FIT_ROUNDS, X, Y):
        adj = (ref - x[3]) * w[3]
        rng = sorted([(ref - x[3]) * lo_t, (ref - x[3]) * hi_t])
        rows.append((r[0], y, adj, y + adj, y + rng[0], y + rng[1]))
    for lab, y, adj, norm, a, b in rows:
        print(f"  {lab:<24s} {y:>9.3f} {adj:>+9.3f} {norm:>11.3f}"
              f" {a:>7.3f}-{b:<7.3f}")
    best = max(rows, key=lambda t: t[3])
    print(f"\n  best at constant latency: {best[0].strip()} ({best[3]:.3f})")

    print("\n=== Board latency, inverted from time_score ===")
    print("  THE LEVEL MODEL IS WITHDRAWN (round 23).  `4.447 + 0.324*inv + 0.431*local` is")
    print("  kept in the `model` column only to show HOW wrong it is: its residuals reach")
    print("  -2.5 ms and are ordered by date, i.e. the eval machine drifts and the intercept")
    print("  is not a constant.  Never predict a board LEVEL with it.")
    print("  What IS measured, from the round-23 unfused pair (same checkpoints, same fp16")
    print("  packing, same band bytes, FUSE=False the only change, both tracks the same day):")
    print(f"     one extra sequential model forward = +{INV_MS_T1:.3f} board ms on Track 1,")
    print(f"                                          +{INV_MS_T2:.3f} board ms on Track 2.")
    print("  Anchor on a RECENT round of the same shape and move only in deltas.")
    print(f"  {'round':<24s} {'inv':>4s} {'local ms':>9s} {'board ms':>9s} "
          f"{'model':>7s} {'miss':>7s}")
    # local ms are 4060-EQUIVALENT (the A100 runs 1.850x faster; CLAUDE.md round 12).
    known = {"R6": (1, 2.648), "R7": (3, 1.282), "R9": (8, 2.211),
             "R10": (3, 3.598), "R11": (3, 2.002), "R12": (4, 2.235),
             "R13": (1, 1.900),
             # R14: R13's package + the (never-firing) blend gate. 1.033 ms/window measured on
             # the A100; x1.850 = 1.911 4060-equivalent.
             "R14": (1, 1.911),
             # R15: T2R2's members on Track 1, 1.023 ms A100 x1.850 = 1.893.
             "R15": (1, 1.893),
             # R16: T2R3's package on the Track-1 board, so the same 3.017.
             "R16": (1, 3.017),
             "T2R1": (1, 1.900), "T2R2": (1, 2.546),
             # T2R3: 8 fused width-96 members, 1.631 ms A100 x1.850 = 3.017.
             "T2R3": (1, 3.017),
             # T2R4: 3 U-FNO members, 3 invocations. 5.457 ms/step measured in the build log
             # (slurm-7144661) x1.850 = 10.10 4060-equivalent.  The README's "1.28 ms/step"
             # was wrong and is withdrawn.
             "T2R4": (3, 10.10),
             # R17: 4 all-data w96 L8 members, one fused invocation. 1.515 ms A100 x1.850 =
             # 2.803 4060-equivalent (lat_r15 on an idle node, log slurm-7152736.out); the
             # do45 build itself could not be timed (rp_q4b held the cpuset) but is the
             # identical architecture, so the do30 timing carries.
             "R17": (1, 2.803),
             # T2R5: the same four members through the Track-2 streaming path, batch 1.
             # 2.904 ms/step measured (validate_t2, log slurm-7156902.out) x1.850 = 5.37.
             "T2R5": (1, 5.37),
             # R18 / T2R6: R17 / T2R5's IDENTICAL members and band, with only the shipped
             # path's arithmetic changed -- HOST_WRITE="torch" on Track 1 (bit-identical) and
             # SPEC_PATH="bmm" on Track 2 (2e-06 relative).  The four error subscores came back
             # EXACTLY equal on Track 1 and equal to 3e-06 of sps on Track 2, which is the gate
             # the pair was designed around, so each round is a single-subscore A/B on `time`.
             # Both local figures are RATIO-scaled from a contended build job (x0.834 measured
             # back to back in one process), not measured idle.  THE BOARD FALSIFIED BOTH
             # PREDICTIONS: Track 1 predicted -0.283 board ms and returned -0.089 (31 %), and
             # Track 2 predicted -0.81 and returned +0.039 (a small move the WRONG way).
             "R18": (1, 2.451), "T2R6": (1, 4.480),
             # R19 / T2R7: R18 / T2R6's shape with different weights.
             "R19": (1, 2.451), "T2R7": (1, 4.480),
             # R20 / T2R8: cmb3sim, the same shape again.
             "R20": (1, 2.450), "T2R8": (1, 4.480),
             # R21 / T2R9: xfam5 -- R20's four FNOs (one fused group) plus a ViT (a second
             # group), and int8 instead of fp16 to fit the cap.  Two variables at once, so its
             # +0.758 / +1.596 ms cannot be split between the ViT's forward and the packing.
             "R21": (2, None), "T2R9": (2, None),
             # R23 / T2R10: THE MEASUREMENT.  R20 / T2R8's own checkpoints, own fp16 packing
             # and own band (md5 2eed9e6b...), `FUSE = False` the only difference, uploaded to
             # both boards on one day.  Verified before upload by verify/pkg_identity.py at
             # 4.66e-06 / 4.27e-06 against the archives actually uploaded in round 20; verified
             # after by the board, which returned rel/tke/mvpe bit-identical to six decimals
             # and sps to 2e-05.  Local ms is NOT recorded: every reading available (1.139,
             # 1.355, 1.406 A100 ms) came from a contended node and they span 23 %, which is
             # larger than the whole effect the local term is supposed to carry.
             "R23": (4, None), "T2R10": (4, None),
             # R24 / T2R11: timed side by side with R21 in ONE process on one node (rp_r24h /
             # rp_r24i): xfam6 1.812 A100 ms/window (x1.850 = 3.352), x8v's Track-2 step 4.661.
             "R24": (2, 3.352), "T2R11": (2, None),
             # R25 / T2R12: R24's archive with fp16 autocast. rp_r25b timed it interleaved with its
             # own fp32 twin in one process: 1.510 against 1.918 A100 ms/window (x1.850 = 2.794).
             "R25": (2, 2.794), "T2R12": (2, None)}
    for r in FIT_ROUNDS:
        key = r[0].split()[0]
        t = 0.72896 * (100.0 / r[6] - 1.0) ** 2 * 1000.0
        if key in known:
            inv, loc = known[key]
            if loc is None:
                print(f"  {r[0]:<24s} {inv:>4d} {'-':>9s} {t:>9.3f}")
            else:
                m = 4.447 + 0.324 * inv + 0.431 * loc
                print(f"  {r[0]:<24s} {inv:>4d} {loc:>9.3f} {t:>9.3f} {m:>7.3f} {t - m:>+7.3f}")
        else:
            print(f"  {r[0]:<24s} {'?':>4s} {'?':>9s} {t:>9.3f}")


if __name__ == "__main__":
    main()
