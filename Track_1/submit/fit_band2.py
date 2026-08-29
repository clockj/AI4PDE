"""Fit the per-element SPS half-width map that ships with the two-head submission.

    python fit_band2.py --ckpt ../unet/wp2i_s5.pth

Fits on `val_seen.npz`, verifies on `val_unseen.npz`, writes `band.npy` -- a single
(1, 20, 32, 64, 3) array of half-widths that `submission.py` broadcasts over the batch.

The map is precomputed offline on purpose. The best band measured in eval_wp.py also rescales
by each window's own magnitude, but that is worth +0.03 final and would put a nonlinear
per-element solve inside the timed predict() call. A static array costs one broadcast.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
KIT = "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7"
sys.path.insert(0, TRACK1)

spec = importlib.util.spec_from_file_location("sc", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

from eval_wp import optimal_half, run  # noqa: E402


def sps(P, Y, half):
    return sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=P - half, upper=P + half)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(TRACK1, "unet", "wp2i_s5.pth"))
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--out", default=os.path.join(HERE, "band.npy"))
    args = ap.parse_args()

    sets = {}
    for tag in ("seen", "unseen"):
        d = np.load(os.path.join(TRACK1, f"val_{tag}.npz"))
        P, _, _ = run(args.ckpt, d["x"], args.alpha)
        sets[tag] = (P, d["y"])

    Ps, Ys = sets["seen"]
    Pu, Yu = sets["unseen"]

    rmse = np.sqrt(np.mean((Ps - Ys) ** 2, axis=0, keepdims=True)).astype(np.float32)
    base = optimal_half(rmse)

    best = max(((sps(Ps, Ys, c * base), c) for c in np.linspace(0.1, 3.0, 30)))
    scale, fitted = best[1], best[0]
    band = (scale * base).astype(np.float32)

    print(f"checkpoint {args.ckpt}  alpha {args.alpha}")
    print(f"fitted scale {scale:.3f} on seen -> sps {fitted:.2f}")
    print(f"                       on unseen -> sps {sps(Pu, Yu, band):.2f}")
    print(f"  default +-5% band  on unseen -> sps "
          f"{sc.score_sps(sc.aggregate_sps(Pu, Yu, 2)[0]):.2f}")
    print(f"half-width range {band[..., :2].min():.5f} .. {band[..., :2].max():.5f}")

    np.save(args.out, band)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
