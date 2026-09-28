"""End-to-end local ms/window for a MEMBER SET, priced into board ms and `final`.

WHY THIS EXISTS.  Round 15 bets on capacity (width 96 against the shipped width 24), and the
only thing that can kill it before the science does is latency: `board ms = 2.616 + 0.435*inv
+ 0.803*local` and 1 board ms is -0.070 `final` at the ~4 ms operating point.  A w96 member is
16x a w24 member in 1x1-conv FLOPs, but the shipped package is LAUNCH-bound, not FLOP-bound, so
the real cost is unknown until it is measured.

WHAT IT MEASURES.  Exactly what `predict()` does -- `predict_fno` into a pre-zeroed output plus
the sigma-scaled band arithmetic -- not the forward pass alone.  Rounds 1-5 quoted a
forward-only 0.109 ms against a real 1.19, and every `time_score` in those reports is inflated
by it.  The band stage is member-INDEPENDENT, so it is included to make the absolute number
comparable to `validate_strict.py`'s and to the recorded board rounds.

    python lat_r15.py --sets "unet/fno_all_do05_s1.pth,..." --rep 5

`--rep N` takes the MEDIAN of N timed passes after a warm-up, because a single pass on a shared
cluster node is a draw from a heavy-tailed distribution.  Never run it while another job of
yours holds the same node: the 12-core cpuset binds before the GPU does.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import opoint  # noqa: E402

# Measured conversions.  A100 -> the RTX 4060 every historical local ms was measured on
# (round 12: 1.082 A100 against 2.002 recorded 4060 for the identical package), then the
# seven-round board-latency fit (round 13, rms 0.690 ms), then the local slope of the
# time_score curve at ~4 board ms.
R_4060 = 1.850
BOARD_C, BOARD_INV, BOARD_LOC = 2.616, 0.435, 0.803
FINAL_PER_BOARD_MS = -0.070


def timed_predict(models, x, band, ref, p, alpha, batch, dev):
    """A byte-for-byte replica of submission.predict()'s timed section."""
    pred = np.zeros((len(x), 20, 32, 64, 3), dtype=np.float32)
    sig = np.empty((len(x), 1, 32, 64, 2), dtype=np.float32) if p else None
    import _fno
    _fno.predict_fno(models, x, dev, alpha=alpha, batch=batch, out=pred, sigma_out=sig)
    if p and sig is not None and band is not None:
        r3 = np.zeros((len(x), 1, 32, 64, 3), dtype=np.float32)
        np.power(np.clip(sig / ref[..., :2], 0.3, 3.0), np.float32(p), out=r3[..., :2])
        lower, upper = np.empty_like(pred), np.empty_like(pred)
        t_pred, t_lo, t_hi = (torch.from_numpy(a) for a in (pred, lower, upper))
        torch.mul(torch.from_numpy(band), torch.from_numpy(r3), out=t_hi)
        torch.sub(t_pred, t_hi, out=t_lo)
        torch.add(t_pred, t_hi, out=t_hi)
        return pred, lower, upper
    return pred, pred, pred


def pick_batch(models, band, ref, p, alpha, dev):
    """submission._pick_batch, on the same candidates, against the same work."""
    probe = np.zeros((112, 20, 32, 64, 3), dtype=np.float32)
    best_t, best_b = None, 64
    for b in (14, 28, 56, 112):
        timed_predict(models, probe[:b], band, ref, p, alpha, b, dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timed_predict(models, probe, band, ref, p, alpha, b, dev)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        if best_t is None or t < best_t:
            best_t, best_b = t, b
    return best_b, best_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True,
                    help="comma-separated checkpoint lists; a `NAME=` prefix labels the row. "
                         "A `*K` suffix on a path repeats that member K times, which is legal "
                         "for TIMING (cost does not depend on the weights) and nothing else.")
    # 2026-09-28: defaults from opoint (were Track_2/submit/band8.npz, an untracked build by-product,
    # and alpha 0.60). The timed section is the pre-09-21 host band path (pred -/+ base*ratio^p); for
    # the SIGBLEND band it omits the sigma-power factor, one extra (N,1,H,W,2) exp on the GPU.
    ap.add_argument("--band", default=opoint.path(opoint.FINAL_BAND),
                    help=f"band npz, read through band_io.load_band (default {opoint.FINAL_BAND})")
    ap.add_argument("--alpha", type=float, default=opoint.ALPHA,
                    help=f"compose alpha (default {opoint.ALPHA}, opoint.ALPHA)")
    ap.add_argument("--joint-gain", action=argparse.BooleanOptionalAction, default=None,
                    help="compose with _fno.JOINT_GAIN (same cost either way). Neither flag (default) = "
                         f"opoint.JOINT_GAIN ({opoint.JOINT_GAIN}) when --band is centred on y0, else off")
    ap.add_argument("--rep", type=int, default=5)
    ap.add_argument("--batch", type=int, default=0, help="0 = measure it, as submission.py does")
    args = ap.parse_args()

    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import band_io
    bd = band_io.load_band(args.band)
    band, ref, p = bd.base, bd.sigma_ref, bd.p
    _fno.JOINT_GAIN = band_io.joint_gain_for(bd, args.joint_gain)
    x = np.load(os.path.join(HERE, "val_unseen.npz"))["x"].astype(np.float32)
    print(f"device={dev} windows={len(x)} band={os.path.basename(args.band)} "
          f"alpha={args.alpha} p={p} joint_gain={_fno.JOINT_GAIN} ({bd.kind})", flush=True)
    print(f"{'set':<26} {'memb':>4} {'inv':>3} {'params':>9} {'bs':>4} "
          f"{'A100 ms':>8} {'4060 ms':>8} {'board ms':>8} {'d_final':>8}")

    base_board = None
    for spec in args.sets:
        name, _, lst = spec.partition("=")
        if not lst:
            name, lst = spec.split(",")[0].split("/")[-1][:24], spec
        paths = []
        for q in lst.split(","):
            q, _, k = q.partition("*")
            paths += [q] * int(k or 1)
        cache = {}
        models = []
        for q in paths:
            full = q if os.path.isabs(q) else os.path.join(HERE, q)
            # Repeats share weights but must be DISTINCT modules: fuse() stacks parameters.
            models.append(_fno.load_any(full, dev))
        fused = _fno.fuse(models)
        npar = sum(sum(t.numel() for t in m.parameters()) for m in fused)
        b = args.batch or pick_batch(fused, band, ref, p, args.alpha, dev)[0]
        timed_predict(fused, x[:64], band, ref, p, args.alpha, b, dev)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.rep):
            t0 = time.perf_counter()
            timed_predict(fused, x, band, ref, p, args.alpha, b, dev)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) / len(x) * 1e3)
        a100 = float(np.median(ts))
        loc = a100 * R_4060
        boardms = BOARD_C + BOARD_INV * len(fused) + BOARD_LOC * loc
        if base_board is None:
            base_board = boardms
        print(f"{name:<26} {len(models):>4} {len(fused):>3} {npar/1e6:>8.2f}M {b:>4} "
              f"{a100:>8.3f} {loc:>8.3f} {boardms:>8.3f} "
              f"{(boardms - base_board) * FINAL_PER_BOARD_MS:>+8.3f}", flush=True)
        del models, fused, cache
        torch.cuda.empty_cache()
    print("\nd_final is relative to the FIRST row, at "
          f"{FINAL_PER_BOARD_MS} `final` per board ms.")


if __name__ == "__main__":
    main()
