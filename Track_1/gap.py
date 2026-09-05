"""Where does the error live, and what does meeting a NEW trajectory actually cost?

CLAUDE.md's binding-constraint claim -- "74 trajectories, not the model" -- rests on one pair
of numbers measured in round 1 (seen rel-L2 0.049 against unseen 0.084) and never decomposed.
Under the nine-round weights `rel_l2` is 0.582 and a 10% error cut is +0.948 `final`, so the
2.1x seen/unseen gap is the largest single quantity in the project. This says WHICH PART of the
prediction fails to generalize.

The split is exact and orthogonal. For each window, in physical units over `u, v` only:

    y = ym + yf      ym = mean_t(y)   (the time-mean field)
    p = pm + pf
    ||p - y||^2  =  T*||pm - ym||^2  +  ||pf - yf||^2       (mean part + fluctuation part)

and the fluctuation part splits again by asking what an ORACLE per-pixel amplitude would buy:

    Es = || pf * (s_y / s_p) - yf ||^2      amplitude corrected, so this is SHAPE error
    Ea = Ef - Es                            what amplitude calibration can still recover

with `rho = <pf, yf> / (||pf|| ||yf||)` the decorrelation measure the round-12 phase work
implicated but never quantified: with optimal GLOBAL rescaling the fluctuation error cannot go
below ||yf||*sqrt(1 - rho^2), so rho bounds every amplitude-side fix at once.

    python gap.py --ckpt unet/fno_simA_ctl.pth              # one model, seen vs unseen
    python gap.py --ckpt a.pth b.pth --aoa 15 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


def decompose(P: np.ndarray, Y: np.ndarray) -> dict:
    """Error energy of (N,20,32,64,3) prediction against target, split into parts.

    Every number is returned as a FRACTION of ||Y||^2 over the measured channels, so its
    square root is directly comparable to a rel-L2 and the parts add up.
    """
    p, y = P[..., :2].astype(np.float64), Y[..., :2].astype(np.float64)
    n = np.sum(y ** 2)
    pm, ym = p.mean(1, keepdims=True), y.mean(1, keepdims=True)
    pf, yf = p - pm, y - ym
    e_mean = np.sum((pm - ym) ** 2) * p.shape[1]
    e_fluc = np.sum((pf - yf) ** 2)

    s_p = np.sqrt((pf ** 2).mean(1, keepdims=True))
    s_y = np.sqrt((yf ** 2).mean(1, keepdims=True))
    e_shape = np.sum((pf * (s_y / np.maximum(s_p, 1e-9)) - yf) ** 2)

    rho = float(np.sum(pf * yf) / max(np.sqrt(np.sum(pf ** 2) * np.sum(yf ** 2)), 1e-12))
    # per-window relative L2, the scored quantity, for cross-checking against compare_v9
    rel = float(np.mean(np.linalg.norm((p - y).reshape(len(p), -1), axis=1)
                        / np.linalg.norm(y.reshape(len(y), -1), axis=1)))
    return dict(rel=rel, tot=np.sqrt((e_mean + e_fluc) / n), mean=np.sqrt(e_mean / n),
                fluc=np.sqrt(e_fluc / n), shape=np.sqrt(e_shape / n),
                amp=(e_fluc - e_shape) / n, rho=rho,
                floor=np.sqrt(np.sum(yf ** 2) * (1 - rho ** 2) / n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--aoa", type=int, nargs="+", default=None)
    ap.add_argument("--sets", nargs="+", default=["val_seen", "val_unseen"])
    args = ap.parse_args()

    import torch

    import _fno
    from eval_v9 import infer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = {}
    for s in args.sets:
        d = np.load(os.path.join(HERE, f"{s}.npz"))
        sel = slice(None)
        if args.aoa:
            aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in d["ids"]])
            sel = np.isin(aoa, args.aoa)
        data[s] = (d["x"], d["y"][sel], sel)

    hdr = (f"{'checkpoint':22s} {'set':10s} {'rel':>6s} | {'total':>6s} {'mean':>6s} "
           f"{'fluct':>6s} {'shape':>6s} | {'rho':>6s} {'amp%':>6s} {'floor':>6s}")
    print(hdr + "\n" + "-" * len(hdr))
    rows = {}
    for c in args.ckpt:
        path = c if os.path.isabs(c) else os.path.join(HERE, c)
        if not os.path.exists(path):
            print(f"{os.path.basename(c):22s} (missing)")
            continue
        m = [_fno.load_any(path, dev)] if hasattr(_fno, "load_any") else [_fno.load(path, dev)]
        for s, (x, y, sel) in data.items():
            r = decompose(infer(m, x, dev, args.alpha)[sel], y)
            rows[(c, s)] = r
            print(f"{os.path.basename(c):22s} {s:10s} {r['rel']:6.4f} | {r['tot']:6.4f} "
                  f"{r['mean']:6.4f} {r['fluc']:6.4f} {r['shape']:6.4f} | {r['rho']:6.3f} "
                  f"{100*r['amp']/max(r['fluc']**2,1e-12):5.1f}% {r['floor']:6.4f}")
        if ("val_seen" in data) and ("val_unseen" in data):
            a, b = rows[(c, "val_seen")], rows[(c, "val_unseen")]
            d_tot = b["tot"] ** 2 - a["tot"] ** 2
            print(f"{'':22s} {'GAP':10s} {b['rel']/a['rel']:5.2f}x | "
                  f"energy {d_tot:.5f} = mean {100*(b['mean']**2-a['mean']**2)/d_tot:4.0f}% + "
                  f"fluct {100*(b['fluc']**2-a['fluc']**2)/d_tot:4.0f}%"
                  f"   (of which shape {100*(b['shape']**2-a['shape']**2)/max(b['fluc']**2-a['fluc']**2,1e-12):4.0f}%)"
                  f"   rho {a['rho']:.3f} -> {b['rho']:.3f}")
    print("\nmean/fluct/shape are sqrt of energy fractions, so they combine as "
          "total^2 = mean^2 + fluct^2 and are on the rel-L2 scale.\n"
          "floor = the fluctuation error left after an OPTIMAL global rescaling, i.e. what "
          "decorrelation alone costs.")


if __name__ == "__main__":
    main()
