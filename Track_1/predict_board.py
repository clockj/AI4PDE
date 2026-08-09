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
3. **Board time.** `board ms = 4.447 + 0.324*invocations + 0.431*local ms`, four rounds, 0.064 ms.

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
RATIO = dict(rel=1.0831, tke=1.1174, mvpe=1.4522)
R10 = dict(rel_l2=94.576394, tke=79.152828, mvpe=93.681162, time=91.120851, sps=39.626333,
           final=80.512025)


def score_time(ms):
    return 100.0 / (1.0 + np.sqrt((ms / 1000.0) / 0.72896))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="shipped all-data members (sigma)")
    ap.add_argument("--err-ckpt", nargs="+", required=True, help="held-out counterparts")
    ap.add_argument("--band", default=os.path.join(HERE, "submit_fno", "band.npz"))
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--invocations", type=int, required=True)
    ap.add_argument("--local-ms", type=float, required=True,
                    help="validate_strict's ms/window, measured on an IDLE machine")
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    args = ap.parse_args()

    import torch
    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load(c, dev) for c in args.ckpt]
    m_err = [_fno.load(c, dev) for c in args.err_ckpt]

    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hu = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    X, Y = du["x"][hu], du["y"][hu]

    def infer(models):
        sig = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=sig)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32), sig

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

    board_ms = 4.447 + 0.324 * args.invocations + 0.431 * args.local_ms
    subs = dict(rel_l2=sc.score_error(pred["rel"]), tke=sc.score_error(pred["tke"]),
                mvpe=sc.score_error(pred["mvpe"]), time=float(score_time(board_ms)), sps=sps)

    print(f"proxy (AoA {args.aoa}, {len(X)} windows, held-out ensemble of {len(m_err)}): "
          f"rel {prox['rel']:.5f}  tke {prox['tke']:.5f}  mvpe {prox['mvpe']:.5f}")
    print(f"predicted board errors:  rel {pred['rel']:.5f}  tke {pred['tke']:.5f}  "
          f"mvpe {pred['mvpe']:.5f}   (a = {a0:.4f})")
    print(f"board ms = 4.447 + 0.324*{args.invocations} + 0.431*{args.local_ms:.3f} "
          f"= {board_ms:.3f}\n")
    print(f"{'subscore':>10s} {'round 10':>10s} {'predicted':>10s} {'delta':>8s}")
    print("-" * 42)
    # Mid "what one subscore point buys" from the four-round non-negative fits. Deltas only:
    # the level fit is falsified (it needs a negative mvpe weight) but a difference between two
    # packages cancels the intercept, which is the whole reason to work this way.
    W = dict(rel_l2=0.36, tke=0.24, mvpe=0.32, time=0.08, sps=0.23)
    for k in ("rel_l2", "tke", "mvpe", "time", "sps"):
        print(f"{k:>10s} {R10[k]:10.3f} {subs[k]:10.3f} {subs[k] - R10[k]:+8.3f}")
    tot = sum(W[k] * (subs[k] - R10[k]) for k in W)
    print(f"\npredicted final = {R10['final']:.3f} {tot:+.3f} = **{R10['final'] + tot:.3f}**")
    print("  (delta-based; the level fit of final_v9 is falsified, the intercept cancels here)")


if __name__ == "__main__":
    main()
