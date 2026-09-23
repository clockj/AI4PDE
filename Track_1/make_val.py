"""Build the seen/unseen validation sets used by every Track 1 comparison.

UNSEEN: 8 trajectories excluded from our models' training data (VAL_TRAJ).
SEEN  : 8 trajectories that WERE in our models' training data, matched in Re/AoA spread.

Both are training data for the organizers' sim_real_ft checkpoints, so scoring those on both
sets acts as a difficulty control: if FNO scores the same on each, the sets are equally hard and
any gap our models show is genuine generalization error.

    python make_val.py            # writes val_unseen.npz and val_seen.npz
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

T_IN = T_OUT = 20
INTERVAL = 20
SUB = 2
HORIZON = T_IN + T_OUT
DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"  # overridden by --data

UNSEEN = ["3750_0", "6300_10", "8850_20", "11400_20",
          "13950_10", "16500_5", "21600_15", "26700_0"]
SEEN = ["5025_0", "7575_10", "10125_20", "12675_20",
        "17775_10", "19050_5", "22875_15", "25425_0"]


def _check_against_authority() -> None:
    """UNSEEN is typed out above, and MANIFEST.md warns that a hand-typed split leaks:

        "do not type a --exclude list by hand, the (Re, AoA) grid is incomplete and a typed
         list leaks"

    The authority is `unet/model.py:VAL_TRAJ`. Today the two agree, so nothing here is wrong --
    but nothing ENFORCES that they keep agreeing. If VAL_TRAJ is ever edited and this list is
    not, every "held-out" number built from val_unseen.npz silently becomes contaminated, with
    no error raised anywhere. That is the failure the warning describes.

    Parsed rather than imported: model.py pulls in torch, and this script needs only h5py and
    numpy. Parsing keeps the check free of that dependency.
    """
    import re

    mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet", "model.py")
    with open(mp, encoding="utf-8") as f:
        m = re.search(r"VAL_TRAJ\s*=\s*\[(.*?)\]", f.read(), re.S)
    if m is None:
        raise SystemExit(f"make_val.py: cannot find VAL_TRAJ in {mp}. Refusing to build a split "
                         "that cannot be checked against its authority.")
    val = set(re.findall(r'"([^"]+)"', m.group(1)))
    if set(UNSEEN) != val:
        raise SystemExit(
            "make_val.py: UNSEEN has drifted from unet/model.py:VAL_TRAJ.\n"
            f"  in UNSEEN but NOT held out (THESE LEAK): {sorted(set(UNSEEN) - val)}\n"
            f"  held out but missing from UNSEEN:        {sorted(val - set(UNSEEN))}\n"
            "Fix UNSEEN to match VAL_TRAJ; do not edit VAL_TRAJ to match UNSEEN.")
    if set(SEEN) & val:
        raise SystemExit(f"make_val.py: SEEN overlaps the held-out split: "
                         f"{sorted(set(SEEN) & val)} -- val_seen would not be 'seen'.")


_check_against_authority()


def build(names: list[str], out: str) -> None:
    xs, ys, ids = [], [], []
    for name in names:
        with h5py.File(os.path.join(DATA, f"{name}.h5"), "r") as f:
            u = f["u"][:, ::SUB, ::SUB].astype(np.float32)
            v = f["v"][:, ::SUB, ::SUB].astype(np.float32)
        d = np.stack([u, v, np.zeros_like(u)], -1)
        for t in range(0, len(d) - HORIZON + 1, INTERVAL):
            xs.append(d[t:t + T_IN])
            ys.append(d[t + T_IN:t + HORIZON])
            ids.append(f"{name}:{t}")
    np.savez_compressed(out, x=np.asarray(xs, np.float32), y=np.asarray(ys, np.float32),
                        ids=np.array(ids))
    print(f"{out}: {len(xs)} windows from {len(names)} trajectories")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--suffix", default="",
                    help="written as val_unseen<suffix>.npz / val_seen<suffix>.npz; use "
                         "'_sim' for the WP-D sim diagnostic")
    a = ap.parse_args()
    DATA = a.data
    build(UNSEEN, f"val_unseen{a.suffix}.npz")
    build(SEEN, f"val_seen{a.suffix}.npz")
