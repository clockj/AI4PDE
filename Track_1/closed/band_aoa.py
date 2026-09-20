"""Fit the SPS band on the regime the board actually resembles.

    python band_aoa.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

`inflate_v9.py` sized the band by scaling the whole local error field by 1.46, the ratio of
board rel-L2 (0.1179) to local val_unseen rel-L2 (0.0807).  That is a crude model of the shift:
it assumes the board's errors are the local ones stretched uniformly.

Per-trajectory error says something much more specific.  Error is governed by angle of attack
and by almost nothing else, and restricting val_unseen to AoA 15-20 deg gives rel-L2 **0.112**
against the board's 0.1179 -- 5% apart, where the full split is 46% apart.  So the board is far
better modelled as "the high-AoA regime, slightly harder" than as "everything, 1.46x harder",
and the extrapolation needed shrinks from 1.46x to about 1.05x.

That matters because the two produce different bands: a uniform 1.46x stretch widens the
interval everywhere, including the low-incidence pixels where errors are genuinely small and a
narrow interval is worth a lot, whereas fitting on high-AoA windows widens it only where high
incidence actually puts the error.

The table scores each candidate band on three views of val_unseen: everything, the AoA 15-20
subset, and that subset stretched a further 1.05x.  A band is chosen for the middle two without
collapsing on the first, since the true AoA composition of the hidden set is unknown.
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
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def aoa_mask(ids, keep):
    return np.isin(np.array([int(str(i).split(":")[0].split("_")[1]) for i in ids]), keep)


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
    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    Ps, Ys = infer(models, ds["x"], dev, args.alpha), ds["y"]
    Pu, Yu = infer(models, du["x"], dev, args.alpha), du["y"]

    hi_s = aoa_mask(ds["ids"], (15, 20))
    hi_u = aoa_mask(du["ids"], (15, 20))
    print(f"fit split: {hi_s.sum()}/{len(hi_s)} windows at AoA 15-20   "
          f"report split: {hi_u.sum()}/{len(hi_u)}")
    print(f"rel-L2  val_seen all {np.mean(sc.rel_l2_per_sample(Ps, Ys, 2)):.4f}  "
          f"hi {np.mean(sc.rel_l2_per_sample(Ps[hi_s], Ys[hi_s], 2)):.4f}   "
          f"val_unseen all {np.mean(sc.rel_l2_per_sample(Pu, Yu, 2)):.4f}  "
          f"hi {np.mean(sc.rel_l2_per_sample(Pu[hi_u], Yu[hi_u], 2)):.4f}   board 0.1179")

    views = {}
    for nm, P, Y, f in (("unseen all", Pu, Yu, 1.0),
                        ("unseen hi-AoA", Pu[hi_u], Yu[hi_u], 1.0),
                        ("unseen hi-AoA x1.05", Pu[hi_u], Yu[hi_u], 1.05)):
        A, a, sc_, n = prep(P, Y)
        dm, tk, mv = sps9.sample_metrics(P, Y)
        views[nm] = (np.float32(f) * A, sps9.weights_from_metrics(f * dm, f * tk, f * mv),
                     sc_, n)

    cands = {}
    S_all = prep(Ps, Ys)
    for f in (1.0, 1.2, 1.5):
        cands[f"all val_seen, inflate {f}"] = optimal_band(np.float32(f) * S_all[0],
                                                           S_all[1], S_all[2])
    S_hi = prep(Ps[hi_s], Ys[hi_s])
    for f in (1.0, 1.1, 1.25):
        cands[f"hi-AoA val_seen, inflate {f}"] = optimal_band(np.float32(f) * S_hi[0],
                                                              S_hi[1], S_hi[2])

    print(f"\n{'band':32s} {'median h':>9s} | " +
          " ".join(f"{k:>21s}" for k in views))
    print("-" * (44 + 22 * len(views)))
    for nm, b in cands.items():
        print(f"{nm:32s} {np.median(b):9.5f} | " +
              " ".join(f"{sps_score(*v, b):21.2f}" for v in views.values()))


if __name__ == "__main__":
    main()
