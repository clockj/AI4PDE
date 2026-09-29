"""Arm I6: dump what a band fit needs on a regime's HELD-OUT windows, from that regime's own ensemble (GPU).

    python ood_i6dump.py --regime A20 --members <paths> --band <regime band.npz> --out RUN/i6/heldout.npz

Computed exactly the way `fit_band.py` computes its fit split (the repository's `_fno.predict_fno`, JOINT_GAIN at the tree
default, fp32): P = the compose at alpha 0.80 (for the accuracy factors a_i), F = the alpha-0 ensemble mean (the interval
centre y0, whose |y0 - y| is the error the band is solved on), S = the members' mean sigma head (physical). Written in
`dump_twins.py`'s format with ONE pseudo-member (F = y0[None], S = sigma[None]), so `sigblend/make_k0_band.py` reads it
unchanged (bandlab's y0 = F.mean(0), sig = S.mean(0)), plus P and `sref`, the sigma_ref of the regime's own band (the
scale that normalises THIS ensemble's sigma). Windows: `t2_common.build_stream` of the regime's evaluation trajectories
(the same windows as make_val.py and the stream).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_lib  # noqa: E402

WT = ood_lib.WT
sys.path.insert(0, os.path.join(WT, "Track_1"))
sys.path.insert(0, os.path.join(WT, "Track_1", "submit_fno"))
sys.path.insert(0, os.path.join(WT, "Track_2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--band", required=True, help="the regime's own band npz (its sigma_ref)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="heldout", choices=["heldout", "static-fit"],
                    help="static-fit = the windows fit_band.py fitted this regime's static band on (val_unseen at the fit "
                         "AoA, inferred over all of val_unseen and then selected, exactly as fit_band.py does): the input "
                         "of the identity check that the --from-dump path reproduces the regime band")
    a = ap.parse_args()
    import torch
    import _fno
    import t2_common as t2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    keep = None
    if a.windows == "heldout":
        ev, _, _ = ood_lib.regime(a.regime)
        stream = t2.build_stream(ev)
        X = np.stack([s[2] for s in stream]).astype(np.float32)
        Y = np.stack([s[3][..., :2] for s in stream]).astype(np.float32)
        ids = np.array([f"{s[0]}:{s[1]}" for s in stream])
    else:
        _, _, fit_aoa, fit_on = ood_lib.fit_split(a.regime)
        d = np.load(os.path.join(WT, "Track_1", f"val_{fit_on}.npz"))
        X, Y, ids = d["x"], d["y"][..., :2], d["ids"]
        keep = np.isin(np.array([int(str(i).split(":")[0].split("_")[1]) for i in ids]), fit_aoa)
    models = [_fno.load_any(p, dev) for p in a.members]
    P = _fno.predict_fno(models, X, dev, alpha=0.80)[..., :2].astype(np.float32)
    S = np.empty((len(X), 1, 32, 64, 2), np.float32)
    C = _fno.predict_fno(models, X, dev, alpha=0.0, sigma_out=S)[..., :2].astype(np.float32)
    if keep is not None:
        P, S, C, Y, ids = P[keep], S[keep], C[keep], Y[keep], ids[keep]
    sref = np.load(a.band)["sigma_ref"].astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, F=C[None], S=S[None], x=np.zeros((len(Y), 1), np.float32), y=Y, ids=ids,
             names=np.array(["ensemble"]), P=P, sref=sref, regime=np.array(a.regime.upper()),
             members=np.array([os.path.basename(p) for p in a.members]), joint_gain=np.array(bool(_fno.JOINT_GAIN)))
    print(f"[ood_i6dump] {a.regime} ({a.windows}): {len(Y)} windows, {len(models)} members, JOINT_GAIN {_fno.JOINT_GAIN}, "
          f"{time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
