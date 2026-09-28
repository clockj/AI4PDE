"""Blend the predicted time-mean toward the INPUT window's own mean -- out of distribution.

Round 6 measured this and closed it: blending made MVPE monotonically worse, 0.0698 -> 0.1240,
because "the model's mean beats persistence by a lot".  That was measured IN DISTRIBUTION,
where it is true -- the model's time-mean error on re_split's interior control is 0.0382.

Round 13 found something that reopens it.  Decomposing two IDENTICAL-recipe controls
(`diag_spectral.py --set`):

    set    ckpt     rel-L2   mean part   fluc part
    rein   ctrl1    0.0754      0.0382      0.0648
    rein   ctrl2    0.0754      0.0384      0.0647
    relo   ctrl1    0.1147      0.0764      0.0850
    relo   ctrl2    0.1404      0.1126      0.0815

**The whole of the extrapolation seed variance is in the TIME-MEAN.**  The mean part differs by
47% between two runs of the same recipe; the fluctuation part differs by 4%, and the worse-mean
seed is the BETTER one there.  So out of distribution the fluctuation transfers and the mean
field does not -- and the input window's own time-mean is a MEASUREMENT, not a prediction, so
it cannot be wrong about the Reynolds number in the way a learned mean can.

That is a mechanism, and it predicts a sign flip: the blend should lose on `rein` (round 6's
result, reproduced as the control) and win on `relo`/`rehi`.  If it does not, the mechanism is
wrong and the idea stays closed -- this time with an out-of-distribution measurement behind it.

No training: the blend is post-hoc on a finished prediction.

    python mean_blend_re.py --ckpt unet/re_ctrl1.pth unet/re_ctrl2.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import opoint  # noqa: E402


def train_speed_range(exclude, data="/users/PAS1187/qwu/RealPDE/train_real/train_real"):
    """[min, max] mean streamwise speed over the TRAINING trajectories.

    The observable at test time is the window's own mean speed, not Re -- `predict()` is given
    neither Re nor AoA. Mean speed is proportional to Re (mean_u/Re = 1.02-1.23e-5, constant to
    +-10% over the whole range), so it is the right proxy and it is measurable from the input.
    """
    import h5py
    lo, hi = np.inf, -np.inf
    for f in sorted(os.listdir(data)):
        if not f.endswith(".h5") or f[:-3] in exclude:
            continue
        with h5py.File(os.path.join(data, f), "r") as h:
            m = float(h["u"][::50, ::2, ::2].mean())
        lo, hi = min(lo, m), max(hi, m)
    return lo, hi


def gate(X: np.ndarray, lo: float, hi: float, ref: float = 0.15) -> np.ndarray:
    """Per-window blend weight in [0, 1]: 0 inside the training speed range, ramping outside.

    The blend is a correction for a mean field the model had no data to learn -- so applying it
    inside the training range is pure cost (round 6's null, reproduced), and applying it outside
    is the whole gain. `ref` is the ramp width as a fraction of the training range.
    """
    s = X[..., 0].mean(axis=(1, 2, 3))                       # (N,) window mean speed
    d = np.maximum(np.maximum(lo - s, s - hi), 0.0) / max(hi - lo, 1e-9)
    return np.clip(d / ref, 0.0, 1.0).astype(np.float32)


def blend(P: np.ndarray, X: np.ndarray, beta) -> np.ndarray:
    """Replace beta of the predicted time-mean with the input window's measured time-mean."""
    if np.isscalar(beta) and beta == 0.0:
        return P
    beta = np.asarray(beta, np.float32).reshape(-1, 1, 1, 1, 1) if not np.isscalar(beta) \
        else np.float32(beta)
    mp = P.mean(1, keepdims=True)
    mx = X.mean(1, keepdims=True)
    out = P.copy()
    out[..., :2] = (P - mp + (1.0 - beta) * mp + beta * mx)[..., :2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    # 2026-09-28: default from opoint (was 0.90).
    ap.add_argument("--alpha", type=float, default=opoint.ALPHA,
                    help=f"compose alpha (default {opoint.ALPHA}, opoint.ALPHA)")
    ap.add_argument("--joint-gain", action=argparse.BooleanOptionalAction, default=opoint.JOINT_GAIN,
                    help="compose with _fno.JOINT_GAIN (sets the module flag predict_fno reads; default "
                         f"{opoint.JOINT_GAIN}, opoint.JOINT_GAIN)")
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--sets", nargs="+", default=["rein", "relo", "rehi"])
    ap.add_argument("--gate", action="store_true",
                    help="scale beta per window by how far its own mean speed falls OUTSIDE "
                         "the training range, instead of applying it everywhere. The blend is "
                         "a correction for a mean field the model had no data to learn, so "
                         "in-range windows should get none of it.")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="the trajectories held out of training, to define that range. "
                         "Defaults to re_split.py's 26.")
    ap.add_argument("--ref", type=float, nargs="+", default=[0.15],
                    help="gate ramp width, as a fraction of the training speed range: the "
                         "blend reaches full beta at `ref` outside it. NEVER calibrated -- "
                         "0.15 was a guess, and it is what decides how much of the measured "
                         "gain a DEPLOYED package sees, since an all-data model's speed range "
                         "is wide enough that the hidden set's lowest Re sits only ~2.7%% "
                         "outside it (report_20260831_round13.md 7.3).")
    ap.add_argument("--ens", action="store_true",
                    help="score every --ckpt as ONE ensemble. This is the deciding "
                         "measurement: the blend works by reducing the variance of the "
                         "predicted MEAN FIELD across draws, and an ensemble already averages "
                         "that variance down, so the single-model gain is an upper bound on "
                         "what a shipped 8-member package would see.")
    a = ap.parse_args()

    import torch
    import _fno
    import sps9
    from eval_v9 import infer
    sc = sps9.sc
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _fno.JOINT_GAIN = bool(a.joint_gain)

    rng_ = (0.0, 0.0)
    if a.gate:
        import re_split
        rng_ = train_speed_range(set(a.exclude or re_split.excluded()))
        print(f"training mean-speed range {rng_[0]:.4f} - {rng_[1]:.4f} m/s")
    groups = ([a.ckpt] if a.ens else [[c] for c in a.ckpt])
    for grp in groups:
        paths = [c if os.path.isabs(c) else os.path.join(HERE, c) for c in grp]
        m = [_fno.load_any(p, dev) for p in paths]
        print(f"\n=== {'ens x%d' % len(m) if a.ens else os.path.basename(grp[0])} ===")
        print(f"{'set':6s} {'beta':>5s} {'rel-L2':>8s} {'mean part':>10s} {'fluc part':>10s} "
              f"{'MVPE':>8s}")
        for st in a.sets:
            d = np.load(os.path.join(HERE, f"val_{st}.npz"))
            X, Y = d["x"], d["y"]
            P0 = infer(m, X, dev, a.alpha)
            nrm = np.sqrt((Y[..., :2].astype(np.float64) ** 2).sum((1, 2, 3, 4)))
            first = True
            for rf in (a.ref if a.gate else [0.0]):
                g = gate(X, *rng_, ref=rf) if a.gate else None
                if a.gate:
                    print(f"{'':6s} ref={rf:.3f}  gate fires on "
                          f"{float((g > 0).mean())*100:5.1f}% of windows, mean strength "
                          f"{float(g.mean()):.3f}")
                for b in a.betas:
                    P = blend(P0, X, b * g if a.gate else b)
                    dd = (P[..., :2] - Y[..., :2]).astype(np.float64)
                    dm = dd.mean(1)
                    mean_p = float((np.sqrt(20 * (dm ** 2).sum((1, 2, 3))) / nrm).mean())
                    rel = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
                    fluc = float(np.sqrt(max(rel ** 2 - mean_p ** 2, 0.0)))
                    print(f"{st if first else '':6s} {b:5.2f} {rel:8.4f} {mean_p:10.4f} "
                          f"{fluc:10.4f} {sc.mvpe_rel_l2(P, Y):8.4f}")
                    first = False
        del m


if __name__ == "__main__":
    main()
