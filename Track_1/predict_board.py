"""Predict a package's five board subscores before submitting it.

    python predict_board.py --err-ckpt <held-out members> --ckpt <shipped all-data members> \
                            --invocations 3 --local-ms 2.74

Round 10 turned three loose proxies into calibrated instruments, and this file is where they
are used together. Nothing here is fitted on round 11 -- every constant comes from the round-10
board result -- so the output is a genuine out-of-sample prediction, which is the only kind
worth recording before a submission.

The three instruments
---------------------
1. **Per-metric board/proxy ratios.** The AoA 15-20 held-out ensemble's metrics times a fixed
   ratio. Anchored on round 10: rel x1.083, tke x1.117, mvpe x1.452. The ratios absorb both
   "the board is harder than the proxy" and "the shipped all-data models are better than the
   held-out ones", which is legitimate only because the two member sets move together --
   whenever a member is added to the package, its held-out counterpart is added to `--err-ckpt`.
2. **The SPS tail instrument.** `calib_board11.py` inverted the round-10 board SPS to a
   per-element stretch f_hat = 1.360 and, at that stretch, reproduced the board's 39.626 to
   0.05. So: stretch the proxy error field by 1.360, take the accuracy factors from the
   PREDICTED board metrics, score the package's actual band.
3. **Board time.** `board ms = 3.274 + 1.151*invocations (0.866 on Track 2) + 1.75*local`
   (0.60 on Track 2). Both slopes are MEASURED on the board -- see the LAT_LOC / LAT_INV block
   below. The
   0.803 slope this file used through round 17 was FALSIFIED by R18 and T2R6, two rounds that
   shipped a bit-identical prediction array through a faster path and returned -0.089 and
   +0.039 board ms for local cuts of -0.352 and -0.890. At ~4 board ms, 1 board ms = -0.070
   final, so the practical consequence is that latency work is finished and EXTRA MEMBERS ARE
   NEARLY FREE -- the binding constraint on the member count is the 256 MB cap.

ANCHOR-RELATIVE MODE (`--anchor`), added round 15 and now the default way to use this file
-----------------------------------------------------------------------------------------
The absolute path above multiplies the proxy by fixed board/proxy RATIOs. Those ratios have a
2.5-6.3 % spread across rounds, which is +-0.15 `final` -- larger than any change since round 8.
The fix is the one round 12's post-mortem already drew for `final`: **work in deltas.**

Give the file the ANCHOR package as well (its all-data members, its held-out twins, its band
and its alpha) and it scores both packages on the same instrument, then predicts

    board_error(candidate) = board_error(anchor) * proxy(candidate) / proxy(anchor)
    board_sps(candidate)   = board_sps(anchor)   + [sps_proxy(cand) - sps_proxy(anchor)]

The board/proxy ratio cancels into the anchor, so what is being extrapolated is only the
RATIO OF TWO PROXIES -- and the anchor's own recorded board result carries the level. `final`
is then priced with `final_v9`'s FREE eighteen-round weights on the DELTA, where the
intercept cancels too.  (Round 17 dropped the `sum(w) = 1` constraint the weights had carried
since round 9: max residual 0.049 -> 0.0023.  Deltas priced before 2026-09-04 used the
constrained weights and are worth re-pricing before they are cited.)

The anchors are real board rounds with recorded subscores; T2R2 (2026-09-02, the 8 all-data
dropout members at ALPHA 0.60) is the default because it is the most recent, it is the package
any round-15 candidate is a delta against, and its members' held-out twins are on disk.

`--local-ms` must come from `validate_strict.py` on an otherwise IDLE machine. Timing it
against a training run inflates it by ~1.6x and the error goes straight into the prediction.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import sps9  # noqa: E402
from sps9 import prep, sps_score, sc  # noqa: E402

F_TAIL = 1.360                       # calib_board11.py, inverted from round 10
# Board/proxy error ratios, on the AoA 15-20 held-out proxy.  These were anchored on round 10
# ALONE until 2026-08-31; `proxy_calib.py` now reconstructs the proxy for R10, R11 and R12 and
# measures them, so they are a mean of three rather than a reading of one:
#
#     rel   1.081 / 1.063 / 1.090     mean 1.078   spread 2.5 %
#     tke   1.116 / 1.167 / 1.167     mean 1.150   spread 4.5 %
#     mvpe  1.451 / 1.537 / 1.543     mean 1.511   spread 6.3 %
#
# The round-12 miss was `tke`, whose single-round anchor of 1.117 was the LOWEST of the three.
#
# READ THE SPREAD, NOT THE MEAN.  +-2.5 % on rel-L2 is about +-0.12 `final`, and the proxy is
# worse than that as a DELTA instrument -- between R11 and R12 it got the SIGN of the rel-L2
# move wrong.  This file predicts a level to about +-0.15; it cannot resolve a change smaller
# than ~2 % of error, and nothing since round 8 has been bigger.
RATIO = dict(rel=1.078, tke=1.150, mvpe=1.511)
R10 = dict(rel_l2=94.576394, tke=79.152828, mvpe=93.681162, time=91.120851, sps=39.626333,
           final=80.512025)

# ROUND 18: THE LOCAL-MS SLOPE IS 0.336, NOT 0.803, AND IT WAS MEASURED RATHER THAN FITTED.
# R18 and T2R6 each shipped the SAME prediction array as its anchor through a faster path, so
# all four error subscores came back identical and `time` was the only thing that moved:
#   R17  -> R18    local -0.352 4060-eq ms   board -0.089 ms   slope +0.25
#   T2R5 -> T2R6   local -0.890              board +0.039      slope -0.04
#   T2R3 -> T2R5   local +2.353 (8xL4 -> 4xL8, the batch-1 step nearly doubled)  +0.114  +0.05
# Refitting the ELEVEN one-invocation rounds -- all of the modern era -- gives
# `board = 3.274 + 0.336*local` at rms 0.337 ms, where the seven-round model below missed every
# one of them by 1.3-2.5 ms.
#
# ROUND 23: THE INVOCATION COEFFICIENT IS 1.151 (T1) / 0.866 (T2), NOT 0.435 -- AND IT IS NOW
# MEASURED THE SAME WAY. `track{1,2}_r23_nofuse4fp16_a060.zip` were round 20's own checkpoints,
# round 20's own fp16 packing and round 20's own band byte for byte, with `FUSE = False` the
# only difference: 1 fused invocation became 4 sequential ones. rel_l2/tke/mvpe came back
# IDENTICAL TO SIX DECIMALS on both boards and sps moved 2e-05, so `time` is the whole delta.
#   R20  -> R23     3.795 -> 7.249 board ms over +3 invocations = 1.151 each  (x2.65 the fit)
#   T2R8 -> T2R10   4.659 -> 7.255                              = 0.866 each  (x1.99)
# Track 2 is CHEAPER per invocation, not dearer: its +0.86 ms handicap at one invocation is a
# fixed per-step cost of the streaming path, and at four invocations the two boards meet
# (7.249 vs 7.255). Round 22's "the invocation tax is 2.4x heavier on Track 2" is withdrawn --
# that was R21's ViT forward, which is heavy at batch 1, not the invocation.
# Priced through w_time at round 20's operating point, the FIRST extra invocation is
# **-0.085 `final` on Track 1 and -0.058 on Track 2**: the size of a good accuracy arm.
# NOTE that in --anchor mode (the only mode anyone should use) LAT_C cancels in the difference
# and ONLY THE SLOPES matter. The per-track split is why `--track` now reaches this constant.
LAT_C = 3.274
LAT_INV_T1, LAT_INV_T2 = 1.151, 0.866
LAT_INV = LAT_INV_T1                 # back-compat default; pass track=2 to lat_ms() instead
#
# ROUND 24: THE LOCAL-MS SLOPE IS 1.75 ON TRACK 1, NOT 0.336 -- FLOPs ARE CHARGED. Both round-24
# estimates of record landed BELOW their floors, and both misses were the `time` row: xfam6
# added one ViT inside the fused ViT group (+0.316 A100 ms/window, one process) and the board
# went 4.553 -> 5.570 ms; x8v added four FNOs inside the FNO group (+1.234 ms/step) and Track 2
# went 6.255 -> 7.628. On Track 1 that is 1.74 board ms per 4060-equivalent ms, matching the only
# same-day FLOPs pair (R16 -> R17, 1.76); Track 2 reads 0.60 on one noisier reading. The 0.336
# above was refit ACROSS days, and Track 1's R17 -> R18 "0.25" was a host-write change -- host
# work is cheap on the board, GPU compute is not.
# LAT_LOC_T2 HAS FOUR READINGS AND GOES BACK TO 0.60 (2026-09-18):
#   T2R9 -> T2R11 0.60 | T2R12 -> T2R13 0.937 | T2R13 -> T2R14 0.529 | T2R13 -> T2R15 ~0.28 (after
#   charging the dropped invocation at LAT_INV_T2). Three of the four sit at or below 0.60 and the
#   two most recent are the lowest, so 0.937 was a high draw. Shipped 0.60, the median; **bracket
#   every Track-2 latency lever with 0.3 and 0.9**, never a point. T2R15 landed at 4.480 board ms,
#   below T2R8's 4.659 one-invocation floor, so that day was fast on BOTH boards.
# LAT_LOC_T1 is 1.75 WITHIN a kernel family. R28 is the first CROSS-family Track-1 read that did not
#   invert: +0.455 4060-eq bought +0.404 board ms = 0.89, half of 1.75 (board_rounds.LOC_SLOPE_T1_XFAM).
#   Price a cross-family Track-1 candidate in the bracket [0, 1.75] with 0.89 as its only anchor.
LAT_LOC_T1, LAT_LOC_T2 = 1.75, 0.60
LAT_LOC_T1_XFAM = 0.89
LAT_LOC = LAT_LOC_T1


def lat_inv_of(track: int = 1) -> float:
    """Board ms per extra SEQUENTIAL model forward, measured in round 23. See above."""
    return LAT_INV_T1 if int(track) == 1 else LAT_INV_T2


def lat_loc_of(track: int = 1) -> float:
    """Board ms per 4060-equivalent local ms of GPU compute, measured in round 24. See above."""
    return LAT_LOC_T1 if int(track) == 1 else LAT_LOC_T2

# Recorded board rounds usable as anchors: the five actual subscores, and `final`.
ANCHORS = {
    # 2026-09-03, 8 all-data width-96 FNOs at ALPHA 0.60, one fused invocation. The account's
    # best on either board, and the right anchor for anything built on top of it.
    "t2r15w96": dict(rel_l2=94.755824, tke=78.174302, mvpe=93.739707, time=92.508902,
                     sps=41.160630, final=81.019213),
    # 2026-09-04, the SAME eight width-96 members on the TRACK-1 board (id 914506) -- the
    # fourth cross-track identity check, and the account's best score on either board. Use
    # this one to anchor a Track-1 candidate and `t2r15w96` to anchor its Track-2 twin: the
    # two differ only in which board's level the delta is added to (1 invocation, 3.017 ms
    # 4060-equivalent in both cases, since it is the identical package).
    "r16w96": dict(rel_l2=94.764049, tke=78.433653, mvpe=94.030389, time=92.699846,
                   sps=41.259088, final=81.119780),
    # 2026-09-04 22:06, 4 all-data width-96 EIGHT-layer members at dropout 0.45, one fused
    # invocation, fp16, ALPHA 0.60 (id 915380). The account's best score on either board and
    # the DEFAULT anchor for any Track-1 candidate built on the 4xL8 do45 package. Depth at 4
    # members turned out to be a wash against `r16w96`'s 8 members at 4 layers (+0.023 of
    # accuracy; the rest of its +0.050 was a better time draw), so the two are interchangeable
    # as anchors on accuracy -- but this one carries the newer, faster latency level:
    # 1 invocation, 2.803 ms 4060-equivalent, 4.15 board ms against r16w96's 4.52.
    "r17L8": dict(rel_l2=94.745531, tke=78.635462, mvpe=93.974234, time=92.988361,
                  sps=41.324574, final=81.169843),
    # 2026-09-04 22:06, the SAME four members on the TRACK-2 board (id 915381) -- the fifth
    # cross-track identity check. Anchor a Track-2 twin here and its Track-1 sibling on
    # `r17L8`; the packages are identical, so the pair differs only in which board's level the
    # delta is added to. Note the two are NOT as close as the earlier identity checks: T1 beat
    # T2 on every error subscore (rel -0.36 %, tke -1.64 %, mvpe -4.76 %, sps +0.133), the
    # third ~1.5 % tke gap in a row at width 96, so a Track-2 prediction anchored here already
    # has that path difference folded into its level.
    "t2r16L8": dict(rel_l2=94.727329, tke=78.357043, mvpe=93.692366, time=92.427051,
                    sps=41.191134, final=81.019298),
    "t1r15drop": dict(rel_l2=94.780094, tke=77.747643, mvpe=94.122329, time=93.180177,
                      sps=40.140948, final=80.835124),
    "t2r2": dict(rel_l2=94.787655, tke=77.667926, mvpe=93.882222, time=93.633350,
                 sps=40.148056, final=80.852826),
    "t1r14": dict(rel_l2=94.679785, tke=77.983720, mvpe=93.613336, time=92.978725,
                  sps=39.689258, final=80.630755),
    # 2026-09-06 18:17. R17's and T2R5's packages again, with only the shipped path's
    # arithmetic changed (HOST_WRITE="torch" / SPEC_PATH="bmm"). The four error subscores came
    # back bit-identical on Track 1 and equal to 3e-06 of sps on Track 2, so these anchors carry
    # the SAME accuracy level as r17L8 / t2r16L8 and differ only in `time`. Anchor here when the
    # candidate is built on top of the fast paths, which everything from round 18 on is.
    "r18": dict(rel_l2=94.745531, tke=78.635462, mvpe=93.974234, time=93.058602,
                sps=41.324574, final=81.176572),
    "t2r6": dict(rel_l2=94.727329, tke=78.357043, mvpe=93.692366, time=92.399134,
                 sps=41.191137, final=81.016621),
    # 2026-09-08 03:44. Sim co-training on top of the R18 fast paths: 4 all-data w96 L8 do45
    # `--sim --lam-sim 0.3 --sim-loss full` members, one fused invocation, fp16, ALPHA 0.60
    # (918587 / 918588). THE DEFAULT ANCHOR for anything built on the sim recipe, which is
    # every candidate from round 19 on. Against R18: rel -0.63 %, mvpe -1.02 %, tke +0.61 %
    # (worse), sps +0.008 -- the round that showed a local sps delta does not transfer.
    "r19": dict(rel_l2=94.776881, tke=78.532564, mvpe=94.031877, time=93.030936,
                sps=41.332982, final=81.186041),
    "t2r7": dict(rel_l2=94.762818, tke=78.220131, mvpe=93.843944, time=92.896212,
                 sps=41.210644, final=81.086639),
    # 2026-09-09 12:42. cmb3sim = R19's recipe plus round 18's three wave-1 winners
    # (--lam-tgrad 0.3, --lam-v 0.15, --lam-cov 2.0 / --cov-c 0.25), four all-data members, one
    # fused invocation, fp16, ALPHA 0.60 (920369 / 920370). R20 IS THE ACCOUNT'S BEST SCORE ON
    # EITHER BOARD and the DEFAULT anchor for anything built on cmb3sim. Recorded prediction
    # 81.195 middle / 81.226 optimistic; actual 81.240, of which +0.023 was a favourable `time`
    # draw -- at constant latency the accuracy part is +0.031 (T1) and +0.049 (T2), and the
    # containment residual is +0.049 / +0.039, the first nonzero one outside capacity and band.
    "r20": dict(rel_l2=94.794521, tke=78.518609, mvpe=94.081696, time=93.270089,
                sps=41.410372, final=81.239831),
    "t2r8": dict(rel_l2=94.782311, tke=78.278425, mvpe=93.933159, time=92.597588,
                 sps=41.310588, final=81.106430),
    # ROUND 21 / T2R9, 2026-09-10 -- cmb3sim x4 + ONE ViT d384 L8, 2 invocations. The first
    # heterogeneous package this account has shipped, and THE DEFAULT ANCHOR for anything
    # cross-family. Recorded 81.303 / 81.176 middle; actual 81.331 / 81.156 (+0.028 / -0.020).
    # Its sps splits +0.162 forced / +0.184 CONTAINMENT on T1 and +0.176 / +0.163 on T2, the
    # largest containment move since dropout and replicated across boards. Its latency is the
    # warning: board ms went 3.795 -> 4.553 (T1) and 4.659 -> 6.255 (T2) for the SAME change,
    # against +0.653 predicted.
    #   ROUND 23 RE-READS THIS. An invocation costs 1.151 (T1) / 0.866 (T2) ms, so R21's ViT
    #   group came in BELOW one FNO invocation on Track 1 (+0.758) and at nearly TWICE one on
    #   Track 2 (+1.596). An "invocation" is really one model's kernel launches, and a ViT's
    #   are few and large: cheap when a batch of 64 amortises them, dear at batch 1. R21 also
    #   moved fp16 -> int8, which this cannot separate.
    "r21": dict(rel_l2=94.847008, tke=78.684381, mvpe=94.298686, time=92.676031,
                sps=41.755710, final=81.330636),
    "t2r9": dict(rel_l2=94.841673, tke=78.469938, mvpe=94.157190, time=91.521812,
                 sps=41.649497, final=81.155762),
    # ROUND 23 / T2R10, 2026-09-11 -- R20 / T2R8 UNFUSED, 4 invocations. NOT an accuracy anchor
    # (its error subscores ARE R20's, by construction); it is here so the invocation
    # coefficient can be re-derived from the table, and as the anchor for any future 4-group
    # package. See the LAT_INV block above.
    "r23": dict(rel_l2=94.794522, tke=78.518609, mvpe=94.081696, time=90.932390,
                sps=41.410354, final=81.015201),
    "t2r10": dict(rel_l2=94.782311, tke=78.278425, mvpe=93.933159, time=90.928559,
                  sps=41.310591, final=80.945935),
    # ROUND 24 / T2R11, 2026-09-12 -- each R21's uploaded archive plus ONE thing. r24 = xfam6
    # (+1 ViT fused, accuracy +0.050, time -0.069); t2r11 = x8v (+4 FNO fused, accuracy -0.020,
    # time -0.077). Anchor anything built on xfam6 at r24 -- WITH local ms given on both sides,
    # because FLOPs are charged.
    "r24": dict(rel_l2=94.861827, tke=78.793104, mvpe=94.347768, time=91.961259,
                sps=41.866674, final=81.312041),
    "t2r11": dict(rel_l2=94.845935, tke=78.395628, mvpe=94.066988, time=90.720061,
                  sps=41.626260, final=81.058649),
    # ROUND 25 / T2R12, 2026-09-13 -- R24's archive with `AMP = "fp16"` baked in, on both boards. The
    # error subscores moved by <= 3e-4 against R24: the fingerprint that fp16 RAN on the board GPU.
    # Track 1 realised 37 % of the A100's interleaved -21 % (5.570 -> 5.076 board ms, i.e. ~0.65 board
    # ms per 4060-eq ms for a precision cut, board_rounds.PREC_SLOPE_T1); on Track 2 the same archive
    # (T2R9 + the second ViT + fp16) came in at 7.061 ms against T2R9's 6.255. Anchor anything built on
    # xfam6fp16 here -- `r25` for Track 1, `t2r12` for Track 2 -- with local ms measured in fp16 on
    # BOTH sides, interleaved: fp16 cuts each package's compute by a package-dependent fraction.
    "r25": dict(rel_l2=94.861790, tke=78.793380, mvpe=94.347508, time=92.297891,
                sps=41.867687, final=81.344644),
    "t2r12": dict(rel_l2=94.861391, tke=78.587250, mvpe=94.224322, time=91.039757,
                  sps=41.760694, final=81.164374),
    # ROUND 26 / T2R13, 2026-09-16 -- the first non-FNO primary families on the board, one per track.
    # `r26` is P1 (Swin U-Net w64 L2 x4 + ViT x2): it LOST -0.126, with all three error subscores below
    # R25's and `time` 91.528 where 93.240 was predicted, so anchor Track-1 Swin work here and nothing
    # else -- R25 remains Track 1's best and its accuracy reference.
    # `t2r13` is P2-L8 (F-FNO w40 L8 x4 dense + ViT x2) and is THE DEFAULT TRACK-2 ANCHOR from now on:
    # it is the Track-2 best (81.314785), its errors and its containment both moved, and every future
    # Track-2 package is a delta against it. Board step 6.069 ms at 2 invocations.
    "r26": dict(rel_l2=94.747724, tke=78.667443, mvpe=94.173298, time=91.528039,
                sps=41.995785, final=81.218444),
    "t2r13": dict(rel_l2=94.890753, tke=78.781089, mvpe=94.063576, time=91.638537,
                  sps=42.063113, final=81.314785),
    # ROUND 27 / T2R14, 2026-09-17 -- both slots lost, and NEITHER is the anchor to use.
    # `r27` is R25's archive with the CUDA graph on: errors BIT-IDENTICAL to R25, sps +8.2e-05 (the
    # GRAPH_FP fingerprint, so the graph ran) and `time` 91.915 for 5.641 board ms against R25's
    # 5.076 -- the graph made the package SLOWER. It exists here only to price the graph path; anchor
    # every Track-1 candidate on **`r25`**, which is still Track 1's best and its accuracy reference.
    # `t2r14` is P2-L4 (F-FNO w40 L4 x4 dense + ViT x2), T2R13 with depth as the only lever: the three
    # errors are a wash (rel +0.024, tke -0.030, mvpe +0.073) and the -0.273 sps is CONTAINMENT
    # (-0.310 after the +0.038 forced by the errors), so L8 stays. Anchor Track-2 work on **`t2r13`**;
    # use `t2r14` only to price L4's latency (board step 5.346 ms at 2 invocations) or its containment.
    "r27": dict(rel_l2=94.861790, tke=78.793380, mvpe=94.347508, time=91.914674,
                sps=41.867769, final=81.307815),
    "t2r14": dict(rel_l2=94.914425, tke=78.750833, mvpe=94.136582, time=92.111693,
                  sps=41.790524, final=81.308359),
    # ROUND 28 / T2R15, 2026-09-18 -- the F-FNO family on Track 1, and the ViT pair priced.
    # `r28` is the NEW TRACK-1 BEST (81.401196): R25's package with four fused F-FNO w40 L8 members in
    # place of the four cmb3sim FNOs, T2R13's band, 2 invocations, board step 5.480 ms. The family's
    # board-measured accuracy and containment transferred from Track 2 at ~0.85x (sps +0.272, of which
    # +0.253 is containment). **THE DEFAULT ANCHOR FOR ANY F-FNO PACKAGE ON TRACK 1**; anchor an FNO
    # package on `r25`, which is still the reference for that family's accuracy.
    # `t2r15` is T2R13 with the ViT pair dropped (L8 x4 alone, ONE invocation, board step 4.480 ms):
    # -0.050, so the pair stays. Use it ONLY to price a one-invocation Track-2 package or the ViT
    # pair's containment (-0.132, three times the -0.041 assumed from T2R12); T2R13 is still the
    # Track-2 accuracy anchor and the Track-2 best.
    "r28": dict(rel_l2=94.890524, tke=78.952686, mvpe=94.205730, time=92.021555,
                sps=42.139290, final=81.401196),
    "t2r15": dict(rel_l2=94.823177, tke=78.519336, mvpe=93.885731, time=92.730627,
                  sps=41.742474, final=81.265138),
    # ROUND 29 / T2R16, 2026-09-19 -- TWO NULL LEVERS, both landing exactly on their pre-registered
    # fingerprints. `r29` is R28's package on the batch-tiled `_fno.py`: all four error subscores
    # BIT-IDENTICAL to R28, `time` +0.016 board ms (inside drift), net -0.001. The tiled path is proven
    # exact on the eval GPU and is the base of every later package, but it is NOT a latency lever.
    # `t2r16` is T2R13's package with the `--fill-unscored` band: rel/tke/mvpe bit-identical, sps
    # -0.0066 (the fill bought nothing -- the hidden set's blanking matches the fit split's), and the
    # +0.009 net is a -0.177 ms TIME DRAW. It is the Track-2 BEST BY SCORE (81.323990) and NOT a method
    # improvement: **anchor Track-2 accuracy work on `t2r13`**, whose package is still the method of
    # record. Anchor Track-1 work on `r28` (F-FNO) or `r25` (FNO), not on `r29`.
    "r29": dict(rel_l2=94.890524, tke=78.952686, mvpe=94.205730, time=92.010936,
                sps=42.139290, final=81.400174),
    # ROUND 30 / T2R17, 2026-09-20 -- r34, the SIM-CO-TRAINED members, and the new best on BOTH boards.
    # R28's/T2R13's package with the four F-FNO w40 L8 members and their three band twins retrained with
    # sim co-training and the band refit from the sim twins (`r34_p2ffnoL8sim_fill.npz`). **THESE ARE THE
    # DEFAULT ANCHORS FROM NOW ON**, `r30` for Track 1 and `t2r17` for Track 2: they are the method of
    # record on both boards. Note when pricing from them: the sim lever moved the MEAN FIELD (mvpe
    # +0.056 / +0.068, the un-halved twin delta) and NOT rel or tke, and its `sps` gain is containment
    # (+0.134 / +0.160) from the band refit. `t2r17`'s `time` is a BAD DRAW -- 6.596 board ms, +0.704 on
    # the anchor, the top of Track 2's drift range -- so price Track-2 latency from `t2r16`'s 5.892 ms
    # or bracket it, never from this row's level.
    "t2r16": dict(rel_l2=94.890753, tke=78.781089, mvpe=94.063576, time=91.750947,
                  sps=42.056535, final=81.323990),
    "r30": dict(rel_l2=94.878882, tke=78.883397, mvpe=94.261277, time=91.959093,
                sps=42.263813, final=81.418569),
    "t2r17": dict(rel_l2=94.885309, tke=78.806628, mvpe=94.131921, time=91.313614,
                  sps=42.241165, final=81.333776),
    "r10": R10,
}


def errors_of(subs):
    """Invert `100/(1+0.5e)` to recover the board's own error for the three error subscores."""
    return {k: 2.0 * (100.0 / subs[v] - 1.0)
            for k, v in (("rel", "rel_l2"), ("tke", "tke"), ("mvpe", "mvpe"))}


def score_time(ms):
    return 100.0 / (1.0 + np.sqrt((ms / 1000.0) / 0.72896))


def sps_selftest() -> None:
    """Every consecutive board step, priced by the PER-SAMPLE FACTOR alone.

    `sps = 100 * mean_i(a_i) * (weighted containment)` with `a_i = 0.5g(rel)+0.3g(tke)+0.2g(mvpe)`
    and `g(m) = 0.5/(0.5+m)`, so a step can be priced from published subscores alone:

        sps_pred(B) = sps(A) * a(B's board errors) / a(A's board errors)

    There is no free parameter, so the RESIDUAL IS THE CONTAINMENT TERM -- exactly the part of
    `sps` that a better band or a better model has to buy. Sorted by |residual|, the eighteen
    consecutive pairs on the two boards give the complete list of what has ever moved it: the
    BAND, regularized CAPACITY, MEMBER COUNT, and the architecture FAMILY. Everything rounds
    16-19 tried -- sim co-training, depth at equal FLOPs, the coverage and temporal-gradient
    losses, `lam_v`, both latency rebuilds, the gated blend -- lands inside +-0.05.

    Read it that way and not as a goodness-of-fit: the model is exact from R13 on because the
    band, the member count and the family have been frozen since R13, not because it is right
    about everything.
    """
    import board_rounds as br

    def a_of(r):
        er = lambda v: 2.0 * (100.0 / v - 1.0)  # noqa: E731
        return float(sps9.weights_from_metrics(np.float64(er(r[3])), np.float64(er(r[4])),
                                               np.float64(er(r[5]))))

    out = []
    for rows in (br.ROUNDS, br.ROUNDS_T2):
        rs = sorted([r for r in rows if not r[0].strip().startswith("X")], key=lambda r: r[1])
        for x, y in zip(rs, rs[1:]):
            q = x[7] * a_of(y) / a_of(x)
            out.append((y[7] - q, x[0].strip(), y[0].strip(), q, y[7], y[7] - x[7]))

    print(f"{'step':>46s} {'pred':>7s} {'actual':>7s} {'d sps':>7s} {'RESID':>7s}")
    for d, a, b, q, act, ds in sorted(out, key=lambda t: -abs(t[0])):
        print(f"{a[:21]:>22s} -> {b[:21]:<22s} {q:7.3f} {act:7.3f} {ds:+7.3f} {d:+7.3f}")
    r = np.array([abs(t[0]) for t in out])
    print(f"\n  {len(r)} pairs: median |resid| {np.median(r):.3f}   rms {np.sqrt((r ** 2).mean()):.3f}"
          f"   inside 0.1: {int((r < 0.1).sum())}/{len(r)}")
    print("  From R13 on (band, members and family frozen): median 0.026, i.e. +-0.007 `final`.")
    print("  THE RESIDUAL IS CONTAINMENT. Everything that has ever moved it:")
    print("    +1.410  the band refit (R9->R10, p 0.45 / blur k 3)")
    print("    +0.992 / +0.961  width 24 -> 96, replicated on both boards")
    print("    +0.668  members 3 -> 8 at w24")
    print("    +0.464 / -0.440  U-FNO x3 in, and back out again -- a FAMILY effect")
    print("    +0.262 / +0.253  dropout 0.05/0.15 -> 0.30/0.45, replicated on both boards")
    print("  Nothing else clears 0.15. Round 16 filed U-FNO as a -0.45 miss and closed the")
    print("  family; that was true of the PACKAGE (time -0.207 at batch 1, rel_l2 +1.9 %) and")
    print("  false of the MEMBER, whose +0.44 sps was read as noise and was containment.")


def main():
    if "--sps-selftest" in sys.argv:
        return sps_selftest()
    if "--sps-selftest" in sys.argv:
        return sps_selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="shipped all-data members (sigma)")
    ap.add_argument("--err-ckpt", nargs="+", required=True, help="held-out counterparts")
    ap.add_argument("--band", default=os.path.join(HERE, "submit_fno", "band.npz"))
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--invocations", type=int, required=True)
    ap.add_argument("--local-ms", type=float, required=True,
                    help="validate_strict's ms/window, measured on an IDLE machine")
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--anchor", choices=sorted(ANCHORS), default=None,
                    help="predict RELATIVE to this recorded board round instead of "
                         "multiplying the proxy by fixed ratios. Needs --anchor-err-ckpt "
                         "(and --anchor-ckpt / --anchor-band / --anchor-alpha if the anchor "
                         "package differs from the candidate in those).")
    ap.add_argument("--anchor-err-ckpt", nargs="+", default=None,
                    help="the ANCHOR package's held-out twins")
    ap.add_argument("--anchor-ckpt", nargs="+", default=None,
                    help="the ANCHOR package's shipped all-data members (sigma for its band)")
    ap.add_argument("--anchor-band", default=None)
    ap.add_argument("--anchor-alpha", type=float, default=None)
    ap.add_argument("--anchor-local-ms", type=float, default=None,
                    help="the ANCHOR package's 4060-equivalent local ms. Given, board latency "
                         "is predicted as the anchor's MEASURED board ms plus the model's "
                         "difference, which cancels the eval machine's drift.")
    ap.add_argument("--anchor-invocations", type=int, default=None)
    args = ap.parse_args()
    if args.anchor and not args.anchor_err_ckpt:
        raise SystemExit("--anchor needs --anchor-err-ckpt (the anchor package's twins)")

    import torch
    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load_any(c, dev) for c in args.ckpt]
    m_err = [_fno.load_any(c, dev) for c in args.err_ckpt]

    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hu = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    X, Y = du["x"][hu], du["y"][hu]

    def infer(models):
        sig = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=sig)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32), sig

    def infer_at(models, alpha):
        sig_ = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(models, X, dev, alpha=alpha, sigma_out=sig_)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32), sig_

    def sps_of(P_, sig_, band_path, metrics):
        """This package's own band, scored on its own held-out error field at the board tail."""
        z_ = np.load(band_path)
        base_, ref_, p_ = z_["base"][..., :2], z_["sigma_ref"][..., :2], float(z_["p"])
        half_ = (base_ * np.clip(sig_ / ref_, 0.3, 3.0) ** np.float32(p_)).astype(np.float32)
        A_, _, scored_, n_ = prep(P_, Y)
        a_ = float(sps9.weights_from_metrics(np.float64(metrics["rel"]),
                                             np.float64(metrics["tke"]),
                                             np.float64(metrics["mvpe"])))
        return float(sps_score(np.float32(F_TAIL) * A_, np.full(len(A_), a_), scored_, n_,
                               half_))

    P, _ = infer(m_err)                  # error field: held-out, honest
    _, sig = infer(m)                    # sigma: the shipped models', as submission.py uses

    prox = dict(rel=float(np.mean(sc.rel_l2_per_sample(P, Y, 2))),
                tke=float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))),
                mvpe=float(sc.mvpe_rel_l2(P, Y)))
    pred = {k: prox[k] * RATIO[k] for k in prox}

    z = np.load(args.band)
    base, ref, p = z["base"][..., :2], z["sigma_ref"][..., :2], float(z["p"])
    half = (base * np.clip(sig / ref, 0.3, 3.0) ** np.float32(p)).astype(np.float32)

    A, _, scored, n = prep(P, Y)
    a0 = float(sps9.weights_from_metrics(np.float64(pred["rel"]), np.float64(pred["tke"]),
                                         np.float64(pred["mvpe"])))
    sps = sps_score(np.float32(F_TAIL) * A, np.full(len(A), a0), scored, n, half)

    # Which board's invocation coefficient applies. Track 2 is CHEAPER per invocation (0.866
    # vs 1.151, round 23) but pays a fixed per-step overhead on top, which the anchor's own
    # measured latency already carries. Anchors named t2* are Track 2; without an anchor there
    # is no level anyway, so Track 1's is the harmless default.
    track = 2 if (args.anchor or "").startswith("t2") else 1
    lat_inv = lat_inv_of(track)
    lat_loc = lat_loc_of(track)
    board_ms = LAT_C + lat_inv * args.invocations + lat_loc * args.local_ms

    anc = anc_board = None
    if args.anchor:
        # --- anchor-relative: what is extrapolated is the RATIO OF TWO PROXIES -------------
        anc = ANCHORS[args.anchor]
        anc_board = errors_of(anc)
        a_err = [_fno.load_any(c, dev) for c in args.anchor_err_ckpt]
        a_alpha = args.anchor_alpha if args.anchor_alpha is not None else args.alpha
        Pa, _ = infer_at(a_err, a_alpha)
        a_prox = dict(rel=float(np.mean(sc.rel_l2_per_sample(Pa, Y, 2))),
                      tke=float(np.mean(sc.tke_rel_l2_per_sample(Pa, Y, 2))),
                      mvpe=float(sc.mvpe_rel_l2(Pa, Y)))
        pred = {k: anc_board[k] * prox[k] / a_prox[k] for k in prox}
        # SPS: the anchor's own band and sigma, scored on the anchor's own twins, so the
        # difference is the package change and the level comes from the board.
        a_ck = args.anchor_ckpt or args.anchor_err_ckpt
        a_mod = [_fno.load_any(c, dev) for c in a_ck]
        _, a_sig = infer_at(a_mod, a_alpha)
        # The anchor's SPS must be scored at the anchor's own BOARD metrics, not its proxy
        # metrics. sps carries a per-sample factor a_i = 0.5g(rel)+0.3g(tke)+0.2g(mvpe) with
        # g(m) = 0.5/(0.5+m), so scoring one side at proxy error and the other at board error
        # multiplies the two by different constants (0.730 against 0.701 here) and invents a
        # -1.2 point difference between a package and ITSELF. The identity control below is
        # what caught it, and is why this mode runs one.
        a_sps = sps_of(Pa, a_sig, args.anchor_band or args.band, anc_board)
        c_sps = sps_of(P, sig, args.band, pred)
        sps = anc["sps"] + (c_sps - a_sps)
        # TIME, delta-consistent for the same reason `final` is. The seven-round latency fit
        # has rms 0.690 ms and residuals ORDERED BY DATE -- the eval machine drifts -- so its
        # LEVEL is worth about +-1 `time_score` point. Here it predicts 4.571 ms for a package
        # whose own board round returned 3.371. Using the model for the DIFFERENCE and the
        # anchor's measured latency for the level cancels the drift, and makes an unchanged
        # package predict its own time_score exactly.
        anc_ms_actual = 728.96 * (100.0 / anc["time"] - 1.0) ** 2
        anc_ms_model = (LAT_C + lat_inv * (args.anchor_invocations or args.invocations)
                        + lat_loc * (args.anchor_local_ms
                                     if args.anchor_local_ms is not None else args.local_ms))
        board_ms = anc_ms_actual + (board_ms - anc_ms_model)
        print(f"  board ms: anchor measured {anc_ms_actual:.3f}, model {anc_ms_model:.3f}; "
              f"candidate model {LAT_C + lat_inv*args.invocations + lat_loc*args.local_ms:.3f}"
              f" -> anchored {board_ms:.3f}")
        print(f"anchor {args.anchor}: proxy rel {a_prox['rel']:.5f} tke {a_prox['tke']:.5f} "
              f"mvpe {a_prox['mvpe']:.5f}  -> board rel {anc_board['rel']:.5f} "
              f"tke {anc_board['tke']:.5f} mvpe {anc_board['mvpe']:.5f}, sps {anc['sps']:.3f}")
        print(f"  proxy ratio candidate/anchor: rel {prox['rel']/a_prox['rel']:.4f}  "
              f"tke {prox['tke']/a_prox['tke']:.4f}  mvpe {prox['mvpe']/a_prox['mvpe']:.4f}  "
              f"| sps on the shared instrument {a_sps:.3f} -> {c_sps:.3f}")

    subs = dict(rel_l2=sc.score_error(pred["rel"]), tke=sc.score_error(pred["tke"]),
                mvpe=sc.score_error(pred["mvpe"]), time=float(score_time(board_ms)), sps=sps)

    print(f"proxy (AoA {args.aoa}, {len(X)} windows, held-out ensemble of {len(m_err)}): "
          f"rel {prox['rel']:.5f}  tke {prox['tke']:.5f}  mvpe {prox['mvpe']:.5f}")
    print(f"predicted board errors:  rel {pred['rel']:.5f}  tke {pred['tke']:.5f}  "
          f"mvpe {pred['mvpe']:.5f}   (a = {a0:.4f})")
    print(f"board ms = {LAT_C} + {lat_inv}*{args.invocations} (track {track}, MEASURED "
          f"round 23) + {lat_loc}*{args.local_ms:.3f} (MEASURED round 24) = {board_ms:.3f}\n")
    base = ANCHORS[args.anchor] if args.anchor else R10
    label = args.anchor if args.anchor else "round 10"
    print(f"{'subscore':>10s} {label:>10s} {'predicted':>10s} {'delta':>8s}")
    print("-" * 42)
    order = ("rel_l2", "tke", "mvpe", "time", "sps")
    for k in order:
        print(f"{k:>10s} {base[k]:10.3f} {subs[k]:10.3f} {subs[k] - base[k]:+8.3f}")
    # Priced with final_v9's FREE eighteen-round pooled weights, on the DELTA -- the level fit's
    # common-mode error cancels in a difference, which is round 12's post-mortem in one line.
    import final_v9
    mid, lo, hi = final_v9.delta([base[k] for k in order], [subs[k] for k in order])
    print(f"\npredicted final = {base['final']:.3f} {mid:+.3f} = "
          f"**{base['final'] + mid:.3f}**   (admissible {base['final'] + lo:.3f}"
          f"-{base['final'] + hi:.3f})")
    print(f"  delta-priced with final_v9's {len(final_v9.BOARD)}-round pooled weights; "
          f"the intercept cancels.")
    print("  time_score is the soft number: the board-latency fit's residuals are ordered by "
          "date\n  (the eval machine drifts) and it does not predict better than ~+-1 point, "
          "i.e. +-0.07 final.")

    # `sps` IS THE OTHER SOFT NUMBER, and unlike `time` it has no stable correction. Three
    # arms have now been read against a local sps delta and the transfer ratio is not even the
    # same SIGN of error:
    #     R15  capacity x dropout   local +0.49  ->  board +1.013   (UNDER-read 2.1x)
    #     R17  depth L4 -> L8       k-fold +0.31 ->  board +0.066 / +0.031   (over 5-10x)
    #     R19  sim co-training      k-fold +0.10 ->  board +0.008 / +0.020   (over 5-12x)
    # The pattern that fits all three: board `sps` responds to the far ERROR TAIL, local `sps`
    # responds to the bulk. R15 made rel_l2 0.64 % WORSE and sps +1.013 better; R19 made
    # rel_l2 0.63 % better and sps +0.008. An arm that cuts the rms moves local containment
    # (local errors sit near the band edge) and not board containment (board errors sit far
    # outside it, the tail being 1.35x an rms-matched stretch -- round 13's calib_band13).
    # So the conservative number below DROPS the sps row entirely. Quote both: the full number
    # is the optimistic end and the ex-sps number is the floor, and say which is which.
    b2 = [base[k] for k in order]
    s2 = [subs[k] if k != "sps" else base["sps"] for k in order]
    mid2, _, _ = final_v9.delta(b2, s2)

    # THE MECHANISTIC MIDDLE ESTIMATE, and R19 is what forced it. sps factorizes exactly as
    #     sps = 100 * (weighted mean over samples of a_i) * (weighted containment term)
    # with a_i = 0.5g(rel) + 0.3g(tke) + 0.2g(mvpe), g(m) = 0.5/(0.5+m) -- and the two factors
    # move for different reasons. R19 decomposed cleanly: rel contributed +0.000468 to a, mvpe
    # +0.000330, tke -0.000459, net +0.00034, worth about +0.020 sps -- and the board returned
    # +0.008/+0.020. THE ENTIRE BOARD MOVE WAS THE PER-SAMPLE FACTOR; containment contributed
    # nothing. So the honest central estimate holds containment FIXED and prices only what the
    # other three subscores imply, which needs no local sps at all:
    #     sps_pred = sps_anchor * a(predicted errors) / a(anchor's board errors)
    # This is not a scaled local delta and cannot inherit local sps's 5-12x over-read. It is
    # also not a floor: R15 moved containment (rel_l2 0.64 % WORSE, sps +1.013), so an arm that
    # genuinely cuts the far tail will beat it. Quote the middle number and name the bracket.
    if anc is not None:
        aa = float(sps9.weights_from_metrics(np.float64(anc_board["rel"]),
                                             np.float64(anc_board["tke"]),
                                             np.float64(anc_board["mvpe"])))
        ac = float(sps9.weights_from_metrics(np.float64(pred["rel"]), np.float64(pred["tke"]),
                                             np.float64(pred["mvpe"])))
        sps_da = anc["sps"] * ac / aa
        s3 = [subs[k] if k != "sps" else sps_da for k in order]
        mid3, _, _ = final_v9.delta(b2, s3)
        print(f"\n  THREE sps ASSUMPTIONS, and the middle one is the estimate to quote:")
        print(f"    local sps transferred 1:1   sps {subs['sps']:7.3f}   "
              f"final {base['final'] + mid:.3f}   (optimistic; over-read 5-12x on R17/R19)")
        print(f"    per-sample factor ONLY      sps {sps_da:7.3f}   "
              f"final {base['final'] + mid3:.3f}   <-- containment held fixed, R19's mechanism")
        print(f"    sps unchanged               sps {base['sps']:7.3f}   "
              f"final {base['final'] + mid2:.3f}   (floor)")
        print(f"    a_i moves {aa:.5f} -> {ac:.5f}; the sps row is worth {mid - mid2:+.3f} at "
              f"the top of that bracket and {mid3 - mid2:+.3f} in the middle.")
    else:
        print(f"\n  WITHOUT the sps row (the conservative floor): {base['final']:.3f} "
              f"{mid2:+.3f} = **{base['final'] + mid2:.3f}**")
    print("  Local sps has over-read the board by 5-12x on the two arms that cut the rms and\n"
          "  UNDER-read it 2.1x on the one arm that cut the tail. It is a sign, not a size.")


if __name__ == "__main__":
    main()
