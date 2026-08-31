"""A Reynolds-EXTRAPOLATION validation split, and the control that makes it readable.

WHY.  Every local split this project has ever used holds out trajectories drawn from the
INTERIOR of the training Re range, and the hidden test set does not: the Track-2 example
trajectory is Re 2968, below the entire training range of 3750-26700.  So no local number we
have can see distribution-shifted behaviour, and five times a held-out local gain has failed
to transfer.  `ROADMAP.md` P1.

WHAT.  Three disjoint held-out sets from `train_real`:

    val_relo   Re in {3750, 5025}      9 traj   extrapolation BELOW the training range
    val_rehi   Re in {25425, 26700}    7 traj   extrapolation ABOVE it
    val_rein   Re in {12675, 19050}   10 traj   INTERIOR control, same protocol

and a training set of the remaining 56.  The interior control is the whole point: absolute
scores on 56 trajectories are not comparable to anything, but the *difference* between how an
arm does on `rein` and on `relo`/`rehi` is exactly the extrapolation effect, with training-set
size, recipe and seed all held fixed.

PRE-REGISTERED PASS CONDITION (`ROADMAP.md` P1).  This split is only worth keeping if it
REORDERS something.  If every arm ranks the same on `rein` as on `relo`/`rehi`, it is one more
local number that does not predict the board and should be abandoned, not trusted.

    python re_split.py --build                       # write the three npz files
    python re_split.py --ckpt a.pth b.pth            # score arms on all three
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
T_IN = T_OUT = 20
INTERVAL = 20
SUB = 2
HORIZON = T_IN + T_OUT

RE_LO = (3750, 5025)
RE_HI = (25425, 26700)
RE_IN = (12675, 19050)
SETS = {"relo": RE_LO, "rehi": RE_HI, "rein": RE_IN}


def names_for(res, data=DATA):
    return sorted(f[:-3] for f in os.listdir(data)
                  if f.endswith(".h5") and int(f.split("_")[0]) in res)


def excluded(data=DATA):
    """Every trajectory held out of training for this instrument: 26 of 82."""
    return sorted(n for s in SETS.values() for n in names_for(s, data))


def build(data=DATA, out_dir=HERE):
    import h5py
    for tag, res in SETS.items():
        xs, ys, ids = [], [], []
        for name in names_for(res, data):
            with h5py.File(os.path.join(data, f"{name}.h5"), "r") as f:
                u = f["u"][:, ::SUB, ::SUB].astype(np.float32)
                v = f["v"][:, ::SUB, ::SUB].astype(np.float32)
            d = np.stack([u, v, np.zeros_like(u)], -1)
            for t in range(0, len(d) - HORIZON + 1, INTERVAL):
                xs.append(d[t:t + T_IN]); ys.append(d[t + T_IN:t + HORIZON])
                ids.append(f"{name}:{t}")
        p = os.path.join(out_dir, f"val_{tag}.npz")
        np.savez_compressed(p, x=np.asarray(xs, np.float32), y=np.asarray(ys, np.float32),
                            ids=np.array(ids))
        print(f"{p}: {len(xs)} windows from {len(names_for(res, data))} trajectories "
              f"(Re {res})")
    ex = excluded(data)
    print(f"\ntrain2.py --exclude {' '.join(ex)}")
    print(f"  ({len(ex)} held out, {82 - len(ex)} left for training)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ckpt", nargs="+", default=None)
    ap.add_argument("--ens", action="store_true",
                    help="score every --ckpt as ONE ensemble instead of one row each. The "
                         "point is not accuracy: two runs of the identical recipe agree to "
                         "0.001 wfinal on the interior control and differ by 1.42 on relo "
                         "(round 13), so an ensemble's real job out of distribution is to "
                         "average that draw away, and this measures whether it does.")
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--inflate", type=float, default=1.39)
    ap.add_argument("--ms", type=float, default=1.02)
    a = ap.parse_args()
    if a.build:
        build()
        return
    if not a.ckpt:
        raise SystemExit("pass --build or --ckpt")

    import torch
    import _fno
    import final_v9
    import sps9
    from sps9 import optimal_band, prep, sc, sps_score
    from eval_v9 import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = {t: np.load(os.path.join(HERE, f"val_{t}.npz")) for t in SETS}
    t_score = sc.score_time(a.ms / 1000.0)
    f = np.float32(a.inflate)

    hdr = (f"{'checkpoint':22s} {'set':6s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} {'sps':>6s} "
           f"| {'wfinal':>7s}")
    print(hdr); print("-" * len(hdr))
    groups = ([a.ckpt] if a.ens else [[c] for c in a.ckpt])
    for grp in groups:
        paths = [c if os.path.isabs(c) else os.path.join(HERE, c) for c in grp]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"{os.path.basename(missing[0]):22s} (missing)"); continue
        label = (f"ens x{len(paths)}" if a.ens else os.path.basename(grp[0]))
        m = [_fno.load_any(p, dev) for p in paths]
        # The band is fitted ONCE, on the interior control, and reused on the extrapolation
        # sets -- the same seen -> unseen structure the main pipeline uses, so an arm cannot
        # win by having its band refitted to the set it is judged on.
        Pi = infer(m, d["rein"]["x"], dev, a.alpha)
        S = prep(Pi, d["rein"]["y"])
        band = optimal_band(f * S[0], S[1], S[2])
        for tag in ("rein", "relo", "rehi"):
            P, Y = infer(m, d[tag]["x"], dev, a.alpha), d[tag]["y"]
            r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
            k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
            mv = sc.mvpe_rel_l2(P, Y)
            sp = sps_score(*prep(P, Y), band)
            five = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), t_score, sp]
            print(f"{label if tag == 'rein' else '':22s} {tag:6s} {r:7.4f} "
                  f"{k:7.4f} {mv:7.4f} {sp:6.2f} | {final_v9.predict(five)[0]:7.3f}")
        print()


if __name__ == "__main__":
    main()
