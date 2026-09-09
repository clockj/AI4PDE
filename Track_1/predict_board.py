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
3. **Board time.** `board ms = 3.274 + 0.336*invocations... ` -- see LAT_LOC below. The
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
# one of them by 1.3-2.5 ms. The invocation coefficient cannot be re-measured from a set that
# is all one invocation, so it is left where round 13 put it. NOTE that in --anchor mode (the
# only mode anyone should use) LAT_C cancels in the difference and ONLY THE SLOPES matter.
LAT_C, LAT_INV, LAT_LOC = 3.274, 0.435, 0.336

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

    board_ms = LAT_C + LAT_INV * args.invocations + LAT_LOC * args.local_ms

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
        anc_ms_model = (LAT_C + LAT_INV * (args.anchor_invocations or args.invocations)
                        + LAT_LOC * (args.anchor_local_ms
                                     if args.anchor_local_ms is not None else args.local_ms))
        board_ms = anc_ms_actual + (board_ms - anc_ms_model)
        print(f"  board ms: anchor measured {anc_ms_actual:.3f}, model {anc_ms_model:.3f}; "
              f"candidate model {LAT_C + LAT_INV*args.invocations + LAT_LOC*args.local_ms:.3f}"
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
    print(f"board ms = {LAT_C} + {LAT_INV}*{args.invocations} + "
          f"{LAT_LOC}*{args.local_ms:.3f} = {board_ms:.3f}\n")
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
