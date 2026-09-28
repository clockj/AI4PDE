"""Score the Phase-1 size sweep. Run on an idle GPU -- time_score needs it.

    python fno_deeponet/eval_sweep.py --ckpt fno_s fno_m ... > fno_deeponet/sweep_results.txt

One forward pass per model per set (alpha applied afterwards on the cached raw field and
sigma), so the alpha sweep costs no extra GPU. SPS band fitted on val_seen and reported on
val_unseen, exactly as every other comparison in this project.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calib_sigma import compose_np  # noqa: E402
from eval_pair import cross_fit  # noqa: E402
from eval_wp import (HERE, amp, err_map, fit_scale, optimal_half, run, sc,  # noqa: E402
                     subscores, val_history)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    du, ds = np.load(f"{HERE}/val_unseen.npz"), np.load(f"{HERE}/val_seen.npz")
    Xu, Yu, idu = du["x"], du["y"], du["ids"]
    Xs, Ys = ds["x"], ds["y"]
    Hu, Hs = val_history(Xu, idu), val_history(Xs, ds["ids"])
    traju = np.array([str(i).split(":")[0] for i in idu])
    grid = np.linspace(0.1, 3.0, 30)

    print(f"{'model':16s} {'params':>8s} {'ms/win':>7s} {'a':>5s} {'relL2':>7s} {'TKE':>7s} "
          f"{'MVPE':>7s} | {'rel':>5s} {'tke':>5s} {'mvpe':>5s} {'time':>5s} {'sps':>5s} "
          f"| {'FINAL':>6s}", flush=True)

    for name in args.ckpt:
        path = name if os.path.isabs(name) else os.path.join(HERE, "unet", f"{name}.pth")
        if not os.path.exists(path):
            print(f"{name:16s} (missing)")
            continue
        blob = torch.load(path, map_location="cpu")
        npar = sum(v.numel() for v in blob["model_state_dict"].values())
        hu = Hu if blob.get("hist") else None
        hs = Hs if blob.get("hist") else None

        # One pass per set at alpha=0: gives the raw field and sigma, both alpha-independent.
        rawu, sigu, tu = run(path, Xu, 0.0, hist=hu)
        raws, sigs, _ = run(path, Xs, 0.0, hist=hs)
        if args.calibrate and sigu is not None:
            sigu = cross_fit(Xu, sigu, Hu, Yu, traju)

        best = None
        for a in (0.0, 0.5, 0.75, 1.0):
            if sigu is None and a > 0:
                break
            Pu = compose_np(rawu, sigu, a) if sigu is not None else rawu
            Ps = compose_np(raws, sigs, a) if sigs is not None else raws
            emap = err_map(Ps, Ys) * amp(Ps)
            c, _ = fit_scale(Ps, Ys, optimal_half(emap), grid)
            v = subscores(Pu, Yu, tu, half=c * optimal_half(err_map(Ps, Ys) * amp(Pu)))
            if best is None or v["final"] > best[1]["final"]:
                best = (a, v)
        a, v = best
        print(f"{name:16s} {npar/1e6:7.2f}M {tu*1000:7.3f} {a:5.2f} {v['rel_err']:7.4f} "
              f"{v['tke_err']:7.4f} {v['mvpe_err']:7.4f} | {v['rel']:5.1f} {v['tke']:5.1f} "
              f"{v['mvpe']:5.1f} {v['time']:5.1f} {v['sps']:5.1f} | {v['final']:6.2f}",
              flush=True)


if __name__ == "__main__":
    main()
