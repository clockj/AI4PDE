"""Fit the per-element SPS half-width map that ships with the FNO submission.

    python submit_fno/fit_band.py --ckpt ../unet/fno_grad.pth

Fits on `val_seen.npz`, reports on `val_unseen.npz`, writes `band.npy` -- one
(1, 20, 32, 64, 3) array of half-widths that `submission.py` broadcasts over the batch.

Why a static array rather than the band eval_wp.py actually scores with: the best measured band
also rescales per element by the window's own magnitude (`amp`). That is worth about +0.03 final
and would put a per-window nonlinear solve inside the TIMED predict() call, where it costs
time_score. A precomputed array costs one broadcast.

Why fit on val_seen and only REPORT on val_unseen: the half-width map is a fitted quantity. The
scale `c` is chosen where the model is in-sample, exactly as every other comparison in this
project does it, so the val_unseen number stays an honest estimate rather than a tuned one.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
sys.path.insert(0, TRACK1)
sys.path.insert(0, HERE)

import _fno  # noqa: E402
from eval_wp import amp, err_map, fit_scale, optimal_half, sps_of  # noqa: E402

KIT = "/home/clockj/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7"
spec = importlib.util.spec_from_file_location("sc", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=[os.path.join(TRACK1, "unet", "fno_grad.pth")],
                    help="one checkpoint, or several to fit the band for that ensemble. "
                         "The band is derived from the model's OWN error field, so it "
                         "must be refitted whenever the member set changes -- an "
                         "ensemble's errors are smaller and its half-widths narrower.")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--out", default=os.path.join(HERE, "band.npy"))
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = [_fno.load(c, dev) for c in args.ckpt]

    ds = np.load(os.path.join(TRACK1, "val_seen.npz"))
    du = np.load(os.path.join(TRACK1, "val_unseen.npz"))

    def infer(X):
        uv = _fno.predict_fno(m, X, dev, alpha=args.alpha)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    Ps, Ys = infer(ds["x"]), ds["y"]
    Pu, Yu = infer(du["x"]), du["y"]

    # The half-width field is derived from val_seen's per-element RMS error, which is a
    # legitimate offline quantity: it uses no test target and no per-window information.
    w = optimal_half(err_map(Ps, Ys) * amp(Ps))
    c, sps_seen = fit_scale(Ps, Ys, w, np.linspace(0.1, 3.0, 30))
    band = (c * optimal_half(err_map(Ps, Ys))).astype(np.float32)

    sps_unseen = sps_of(Pu, Yu, band)
    sps_default = sps_of(Pu, Yu, None)

    np.save(args.out, band)
    print(f"ckpt        {', '.join(args.ckpt)}")
    print(f"scale c     {c:.4f}")
    print(f"band        {band.shape}  half-width min {band.min():.5f} "
          f"median {np.median(band):.5f} max {band.max():.5f}")
    print(f"sps  val_seen   (fitted)   {sps_seen:.2f}")
    print(f"sps  val_unseen (reported) {sps_unseen:.2f}   vs {sps_default:.2f} "
          f"for the scorer's default +-5% band  -> {(sps_unseen-sps_default)/5:+.2f} final")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
