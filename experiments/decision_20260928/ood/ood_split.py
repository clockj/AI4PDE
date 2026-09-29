"""Where the error left after the stream sits: time-mean against fluctuation (chunk 5, part 3; CPU, from dumps).

    python ood_split.py RUN:VARIANT:GROUP ...        # e.g. runs/re_mirror:control:rehi runs/re_t4:control:rehi

Per window, E = P - Y over u and v (physical; the numerator of the kit's rel is ||E|| per window). E splits exactly into
the window's 20-frame time-mean error Ebar (per pixel and channel) and the fluctuation error E' = E - Ebar:
    ||E||^2 = 20 ||Ebar||^2 + ||E'||^2.
The time-mean part splits further into what persists over the whole trajectory (the trajectory's mean of Ebar over its
windows, per pixel -- the bias a per-trajectory correction such as D1c can learn) and what varies from window to window:
    sum_k ||Ebar_k||^2 = K ||Ebar_traj||^2 + sum_k ||Ebar_k - Ebar_traj||^2.
Reported as shares of the squared-error sum, pooled over the group's windows; plus the rel error itself (mean over windows
of ||E|| / ||Y||) and what it would be with each part removed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ood_lib  # noqa: E402


def split(run, variant, group):
    d = os.path.join(run, "dump", variant)
    traj = np.load(os.path.join(d, "traj.npy"))
    reg = json.load(open(os.path.join(run, "json", "control.json")))["regime"]
    trajs = ood_lib.regime(reg)[1][group]
    P = np.load(os.path.join(d, "P.npy"), mmap_mode="r"); Y = np.load(os.path.join(d, "Y.npy"), mmap_mode="r")
    tot = mean_w = mean_p = fl = 0.0
    rel, rel_nomean, rel_nofl = [], [], []
    for t in trajs:
        idx = np.where(traj == t)[0]
        E = (np.asarray(P[idx], np.float64) - np.asarray(Y[idx], np.float64))          # (K, 20, H, W, 2)
        Yn = np.sqrt((np.asarray(Y[idx], np.float64) ** 2).sum((1, 2, 3, 4)))
        Eb = E.mean(1, keepdims=True)                                                      # (K, 1, H, W, 2)
        Ef = E - Eb
        Et = Eb.mean(0, keepdims=True)                                                     # trajectory-persistent mean error
        K = len(idx)
        tot += float((E ** 2).sum())
        mean_p += float(20 * K * (Et ** 2).sum())
        mean_w += float(20 * ((Eb - Et) ** 2).sum())
        fl += float((Ef ** 2).sum())
        rel += list(np.sqrt((E ** 2).sum((1, 2, 3, 4))) / np.maximum(Yn, 1e-8))
        rel_nomean += list(np.sqrt((Ef ** 2).sum((1, 2, 3, 4))) / np.maximum(Yn, 1e-8))
        rel_nofl += list(np.sqrt(20 * (Eb ** 2).sum((1, 2, 3, 4))) / np.maximum(Yn, 1e-8))
    return dict(se_total=tot, share_mean_persistent=mean_p / tot, share_mean_window=mean_w / tot, share_fluct=fl / tot,
                check=(mean_p + mean_w + fl) / tot, rel=float(np.mean(rel)), rel_if_no_mean_error=float(np.mean(rel_nomean)),
                rel_if_no_fluct_error=float(np.mean(rel_nofl)), n_traj=len(trajs))


def main():
    L = ["| run / variant / group | rel error | SE share: persistent time-mean | window-varying time-mean | fluctuation | "
         "rel without the time-mean error | rel without the fluctuation error |", "|---|---|---|---|---|---|---|"]
    out = {}
    for spec in sys.argv[1:]:
        run, variant, group = spec.split(":")
        r = split(run, variant, group)
        assert abs(r["check"] - 1) < 1e-9, r["check"]
        out[spec] = r
        L.append(f"| {os.path.basename(run)} / {variant} / {group} | {r['rel']:.4f} | {100 * r['share_mean_persistent']:.1f} % | "
                 f"{100 * r['share_mean_window']:.1f} % | {100 * r['share_fluct']:.1f} % | {r['rel_if_no_mean_error']:.4f} | "
                 f"{r['rel_if_no_fluct_error']:.4f} |")
    print("\n".join(L))
    json.dump(out, open("/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs/split.json", "w"), indent=1)


if __name__ == "__main__":
    main()
