"""Is the model's 20-frame time-mean beaten by simply carrying the input window's mean forward?

    python mvpe_probe.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

MVPE averages the prediction over all 20 output frames before taking the error, at 9x4 probe
points in the wake.  It is therefore not a test of the forecast at all -- the fluctuating part
integrates away and what is left is the predicted MEAN flow at those probes.

That matters because MVPE is the subscore that transfers worst.  Across submissions it inflates
1.9-2.4x from local to board, against 1.46x for rel-L2, and its local value (96.63) already sits
at the measured ceiling while its board value (93.79) is three points below.  So the entire gap
is generalization, and the question is whether a lower-variance estimator of the same quantity
would carry better.

The candidate is the input window's own time-mean.  The wake mean evolves on a much slower
timescale than the vortex shedding, so the previous 20 frames are a nearly unbiased estimate of
the next 20 frames' mean, and one that involves no extrapolation and so has nothing to
generalize.  If it is competitive locally it is very likely better on the board, and the two can
be blended:  mean_used = w*model_mean + (1-w)*input_mean, leaving the fluctuation alone so
rel-L2 and TKE are untouched.

The blend is reported at the level of the whole field, not just the probes, because the
submission cannot special-case the probe locations without also moving rel-L2.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import sys

import numpy as np

import sps9
from sps9 import sc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def remean(P, new_mean):
    """Replace P's time-mean with `new_mean`, keeping every frame's fluctuation about it."""
    return (P - P.mean(axis=1, keepdims=True) + new_mean).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]

    for tag in ("seen", "unseen"):
        d = np.load(os.path.join(HERE, f"val_{tag}.npz"))
        X, Y = d["x"], d["y"]
        P = infer(models, X, dev, args.alpha)
        in_mean = X.mean(axis=1, keepdims=True).astype(np.float32)   # (N,1,H,W,3)
        in_mean[..., 2] = 0.0

        print(f"\nval_{tag}")
        print(f"  {'blend w (model mean weight)':32s} {'mvpe':>7s} {'rel_l2':>8s} "
              f"{'tke':>7s} {'sps':>7s}")
        for w in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0):
            Q = remean(P, w * P.mean(axis=1, keepdims=True) + (1 - w) * in_mean)
            A, a, scored, n = sps9.prep(Q, Y)
            band = sps9.optimal_band(np.float32(1.5) * A, a, scored)
            print(f"  {w:32.2f} {sc.mvpe_rel_l2(Q, Y):7.4f} "
                  f"{float(np.mean(sc.rel_l2_per_sample(Q, Y, 2))):8.4f} "
                  f"{float(np.mean(sc.tke_rel_l2_per_sample(Q, Y, 2))):7.4f} "
                  f"{sps9.sps_score(A, a, scored, n, band):7.2f}")

        # How much of the MVPE error is in the mean at all? Substituting the TRUE mean bounds
        # what any mean-side change can do; the remainder is the fluctuation, which MVPE
        # averages over but does not fully cancel at 20 frames.
        Q = remean(P, Y.mean(axis=1, keepdims=True))
        print(f"  {'oracle: true time-mean':32s} {sc.mvpe_rel_l2(Q, Y):7.4f} "
              f"{float(np.mean(sc.rel_l2_per_sample(Q, Y, 2))):8.4f} "
              f"{float(np.mean(sc.tke_rel_l2_per_sample(Q, Y, 2))):7.4f}")


if __name__ == "__main__":
    main()
