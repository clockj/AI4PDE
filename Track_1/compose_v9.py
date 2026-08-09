"""Price every candidate ensemble COMPOSITION in board `final`, end to end.

    python compose_v9.py

Round 8 chose the member set from "architecture diversity is worth +0.05..0.07, an extra seed
+0.01..0.03", round 10 cut it on an AoA-proxy margin of 0.001, and the board disagreed with
both. Neither was an end-to-end price: a member changes accuracy AND invocations AND local
time, and until round 11 the last two could not be converted into `final`. Now they can:

  * accuracy  -> per-metric board/proxy ratios anchored on round 10
  * SPS       -> band refit per composition, scored at the calibrated tail stretch f = 1.360
  * time      -> board ms = 4.447 + 0.324*invocations + 0.431*local ms, measured per composition

Everything is computed on the HELD-OUT members, so the accuracy numbers are predictions rather
than memorization; the shipped package uses the all-data counterparts of whatever wins.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import sps9  # noqa: E402
from sps9 import optimal_band, prep, sps_score, sc  # noqa: E402

F_TAIL = 1.360
RATIO = dict(rel=1.0831, tke=1.1174, mvpe=1.4522)
R10 = dict(rel_l2=94.576394, tke=79.152828, mvpe=93.681162, time=91.120851, sps=39.626333,
           final=80.512025)
W = dict(rel_l2=0.36, tke=0.24, mvpe=0.32, time=0.08, sps=0.23)
BAND_P, BAND_K, INFLATE = 0.5, 3, 1.39

POOL = {
    "s1": "unet/fno_vmlp_lift.pth", "s2": "unet/fno_vmlp_lift_s2.pth",
    "s3": "unet/fno_vl_s3.pth", "s4": "unet/fno_vl_s4.pth",
    "m1220": "unet/fno_vl_m1220.pth", "w48": "unet/fno_vl_w48.pth",
    "w64": "unet/fno_vl_w64.pth",
}
SETS = [
    ("s1 s2", None), ("s1 s2 s3", None), ("s1 s2 s3 s4", None),
    ("s1 s2 s3 s4 m1220", None),
    ("s1 s2 s3 s4 m1220 w48", None),
    ("s1 s2 s3 s4 m1220 w64", None),
    ("s1 s2 s3 s4 m1220 w48 w64", "round-11 candidate"),
    ("s1 s2 s3 s4 w64", None),
    ("s1 s2 s3 s4 w48", None),
    ("s1 s2 s3 s4 m1220 w64 w48", None),
]


def score_time(ms):
    return 100.0 / (1.0 + np.sqrt((ms / 1000.0) / 0.72896))


def smooth(b, k):
    o = b.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * 5
        pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(k)], 0)
    return o.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--time-reps", type=int, default=5)
    args = ap.parse_args()

    import torch
    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {k: _fno.load(os.path.join(HERE, v), dev) for k, v in POOL.items()}

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    hs = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]], args.aoa)
    hu = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]], args.aoa)
    Xs, Ys = ds["x"][hs], ds["y"][hs]
    Xu, Yu = du["x"][hu], du["y"][hu]
    # Timing needs the full 336-window call the evaluator makes, not the 126-window proxy.
    Xt = np.ascontiguousarray(du["x"])

    def infer(models, X):
        sig = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=sig)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32), sig

    def local_ms(models):
        out = np.zeros((len(Xt), 20, 32, 64, 3), np.float32)
        sig = np.empty((len(Xt), 1, 32, 64, 2), np.float32)
        ts = []
        for _ in range(args.time_reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _fno.predict_fno(models, Xt, dev, alpha=args.alpha, out=out, sigma_out=sig)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) / len(Xt) * 1e3)
        return float(np.min(ts))

    # The band-and-bounds stage is composition-independent; measured once at 0.13 ms/window.
    BAND_MS = 0.13

    print(f"{'composition':30s} {'inv':>3s} {'fwd ms':>7s} | {'rel':>6s} {'tke':>6s} "
          f"{'mvpe':>6s} {'time':>6s} {'sps':>6s} | {'d final':>8s}")
    print("-" * 92)
    rows = []
    for names, note in SETS:
        keys = names.split()
        models = _fno.fuse([loaded[k] for k in keys])
        inv = len(models)
        Pu, sig_u = infer(models, Xu)
        Ps, sig_s = infer(models, Xs)
        prox = dict(rel=float(np.mean(sc.rel_l2_per_sample(Pu, Yu, 2))),
                    tke=float(np.mean(sc.tke_rel_l2_per_sample(Pu, Yu, 2))),
                    mvpe=float(sc.mvpe_rel_l2(Pu, Yu)))
        pred = {k: prox[k] * RATIO[k] for k in prox}

        # Band refit on val_seen AoA-hi exactly as fit_band.py does it, for THIS composition.
        ref = np.maximum(sig_s.mean(0, keepdims=True), 1e-8).astype(np.float32)
        rs = np.clip(sig_s / ref, 0.3, 3.0)
        As, as_, sc_s, _ = prep(Ps, Ys)
        base = smooth(optimal_band((np.float32(INFLATE) * As / rs ** np.float32(BAND_P)
                                    ).astype(np.float32), as_, sc_s), BAND_K)
        half = (base * np.clip(sig_u / ref, 0.3, 3.0) ** np.float32(BAND_P)).astype(np.float32)
        A, _, scored, n = prep(Pu, Yu)
        a0 = float(sps9.weights_from_metrics(np.float64(pred["rel"]), np.float64(pred["tke"]),
                                             np.float64(pred["mvpe"])))
        sps = sps_score(np.float32(F_TAIL) * A, np.full(len(A), a0), scored, n, half)

        fwd = local_ms(models)
        board_ms = 4.447 + 0.324 * inv + 0.431 * (fwd + BAND_MS)
        subs = dict(rel_l2=sc.score_error(pred["rel"]), tke=sc.score_error(pred["tke"]),
                    mvpe=sc.score_error(pred["mvpe"]), time=float(score_time(board_ms)),
                    sps=sps)
        d = sum(W[k] * (subs[k] - R10[k]) for k in W)
        rows.append((d, names, subs, inv, fwd))
        print(f"{names:30s} {inv:3d} {fwd:7.3f} | {subs['rel_l2']:6.2f} {subs['tke']:6.2f} "
              f"{subs['mvpe']:6.2f} {subs['time']:6.2f} {subs['sps']:6.2f} | {d:+8.3f}"
              + (f"  <- {note}" if note else ""))

    rows.sort(reverse=True)
    print(f"\nbest: {rows[0][1]}   predicted board final "
          f"{R10['final'] + rows[0][0]:.3f}  ({rows[0][0]:+.3f})")
    print("NOTE: fwd ms is measured on THIS machine; if anything else is using the GPU every")
    print("      row is inflated and the time column is not trustworthy. Accuracy columns are.")


if __name__ == "__main__":
    main()
