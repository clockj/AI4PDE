"""Phase 3: pair field-only FNOs with a separate sigma net, calibrate, and ensemble.

    python fno_deeponet/eval_phase3.py > fno_deeponet/phase3_results.txt

Mirrors the decoupled-pair recipe that won round 3 on the U-Net (report_20260731.md section
2.5), applied to FNO: the field network carries no variance head at all, so the variance
objective cannot reshape its trunk and its MVPE is whatever the field net alone achieves.
Sigma comes from `wpF1_varonly` (width-32 U-Net variance net, history channels), then the
cross-fitted post-hoc calibration.

Latency is the sum of the networks actually run, measured on an idle GPU.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calib_sigma import compose_np  # noqa: E402
from eval_pair import cross_fit  # noqa: E402
from eval_wp import (HERE, amp, err_map, fit_scale, optimal_half, run,  # noqa: E402
                     subscores, val_history)

GRID = np.linspace(0.1, 3.0, 30)


def band_score(Pu, Yu, Ps, Ys, tstep):
    c, _ = fit_scale(Ps, Ys, optimal_half(err_map(Ps, Ys) * amp(Ps)), GRID)
    return subscores(Pu, Yu, tstep, half=c * optimal_half(err_map(Ps, Ys) * amp(Pu)))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="+", default=["fno_field1", "fno_field2"])
    A = ap.parse_args()
    du, ds = np.load(f"{HERE}/val_unseen.npz"), np.load(f"{HERE}/val_seen.npz")
    Xu, Yu, idu = du["x"], du["y"], du["ids"]
    Xs, Ys = ds["x"], ds["y"]
    Hu, Hs = val_history(Xu, idu), val_history(Xs, ds["ids"])
    traju = np.array([str(i).split(":")[0] for i in idu])

    sig = os.path.join(HERE, "unet/wpF1_varonly.pth")
    _, sigu, t_sig = run(sig, Xu, 0.0, hist=Hu)
    _, sigs, _ = run(sig, Xs, 0.0, hist=Hs)
    calu = cross_fit(Xu, sigu, Hu, Yu, traju)

    fields = {}
    for name in A.fields:
        p = os.path.join(HERE, f"unet/{name}.pth")
        fields[name] = (*run(p, Xu, 0.0), *run(p, Xs, 0.0)[:1])

    print(f"{'configuration':44s} {'ms/win':>7s} {'a':>5s} {'relL2':>7s} {'TKE':>7s} "
          f"{'MVPE':>7s} {'sps':>5s} | {'FINAL':>6s}", flush=True)

    names = list(fields)
    combos = [(f"{names[0]} + sigma", [fields[names[0]]], sigu),
              (f"{names[0]} + sigma + calib", [fields[names[0]]], calu)]
    for k in range(2, len(names) + 1):
        sub = [fields[n] for n in names[:k]]
        combos.append((f"ENSEMBLE x{k} + sigma", sub, sigu))
        combos.append((f"ENSEMBLE x{k} + sigma + calib", sub, calu))
    for label, fs, S in combos:
        rawu = np.mean([f[0] for f in fs], axis=0)
        raws = np.mean([f[3] for f in fs], axis=0)
        tstep = sum(f[2] for f in fs) + t_sig
        best = None
        for a in (0.0, 0.5, 0.75, 1.0):
            v = band_score(compose_np(rawu, S, a), Yu,
                           compose_np(raws, sigs, a), Ys, tstep)
            if best is None or v["final"] > best[1]["final"]:
                best = (a, v)
        a, v = best
        print(f"{label:44s} {tstep*1000:7.3f} {a:5.2f} {v['rel_err']:7.4f} {v['tke_err']:7.4f} "
              f"{v['mvpe_err']:7.4f} {v['sps']:5.1f} | {v['final']:6.2f}", flush=True)


if __name__ == "__main__":
    main()
