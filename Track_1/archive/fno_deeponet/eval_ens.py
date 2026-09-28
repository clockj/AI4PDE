"""Score ensembles of already-trained checkpoints. No training required.

    python fno_deeponet/eval_ens.py --set fno_div_hi+fno_vort fno_div_hi+fno_grad+fno_dv ...

Members are averaged in raw field space BEFORE the alpha/compose step, so the variance
re-inflation still sees a single coherent field. Sigma is averaged too. Timing is the SUM of
the members' per-window times, which is what the evaluator would actually pay -- ensembling is
only worth it if the accuracy gain beats the time_score it costs.

The SPS band is fitted on val_seen and reported on val_unseen, as in every other comparison
here.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calib_sigma import compose_np  # noqa: E402
from eval_wp import (HERE, amp, err_map, fit_scale, optimal_half, run, subscores,  # noqa: E402
                     val_history)


def cached(name, Xu, Xs, Hu, Hs, store):
    """One forward pass per model per set, reused across every ensemble that names it."""
    if name in store:
        return store[name]
    path = os.path.join(HERE, "unet", f"{name}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu")
    hu = Hu if blob.get("hist") else None
    hs = Hs if blob.get("hist") else None
    rawu, sigu, tu = run(path, Xu, 0.0, hist=hu)
    raws, sigs, _ = run(path, Xs, 0.0, hist=hs)
    store[name] = (rawu, sigu, raws, sigs, tu)
    return store[name]


def mean_or_none(parts):
    return None if any(p is None for p in parts) else np.mean(parts, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", required=True,
                    help="each entry is member names joined by '+', e.g. fno_div_hi+fno_vort")
    args = ap.parse_args()

    du, ds = np.load(f"{HERE}/val_unseen.npz"), np.load(f"{HERE}/val_seen.npz")
    Xu, Yu = du["x"], du["y"]
    Xs, Ys = ds["x"], ds["y"]
    Hu, Hs = val_history(Xu, du["ids"]), val_history(Xs, ds["ids"])
    grid = np.linspace(0.1, 3.0, 30)
    store = {}

    print(f"{'ensemble':44s} {'ms/win':>7s} {'a':>5s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} "
          f"| {'rel':>5s} {'tke':>5s} {'mvpe':>5s} {'time':>5s} {'sps':>5s} | {'FINAL':>6s}",
          flush=True)

    for spec in args.set:
        members = spec.split("+")
        try:
            got = [cached(m, Xu, Xs, Hu, Hs, store) for m in members]
        except FileNotFoundError as e:
            print(f"{spec:44s} (missing {os.path.basename(str(e))})", flush=True)
            continue
        rawu = np.mean([g[0] for g in got], axis=0)
        raws = np.mean([g[2] for g in got], axis=0)
        sigu = mean_or_none([g[1] for g in got])
        sigs = mean_or_none([g[3] for g in got])
        tu = sum(g[4] for g in got)          # the evaluator pays for every member

        best = None
        for a in (0.0, 0.5, 0.75, 1.0):
            if sigu is None and a > 0:
                break
            Pu = compose_np(rawu, sigu, a) if sigu is not None else rawu
            Ps = compose_np(raws, sigs, a) if sigs is not None else raws
            c, _ = fit_scale(Ps, Ys, optimal_half(err_map(Ps, Ys) * amp(Ps)), grid)
            v = subscores(Pu, Yu, tu, half=c * optimal_half(err_map(Ps, Ys) * amp(Pu)))
            if best is None or v["final"] > best[1]["final"]:
                best = (a, v)
        a, v = best
        print(f"{spec:44s} {tu*1000:7.3f} {a:5.2f} {v['rel_err']:7.4f} {v['tke_err']:7.4f} "
              f"{v['mvpe_err']:7.4f} | {v['rel']:5.1f} {v['tke']:5.1f} {v['mvpe']:5.1f} "
              f"{v['time']:5.1f} {v['sps']:5.1f} | {v['final']:6.2f}", flush=True)


if __name__ == "__main__":
    main()
