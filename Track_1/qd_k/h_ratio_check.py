"""Like-for-like H-level check for the QD band's k (REPRODUCE.md §10.3; dry-run gap #9).

The shipped band is half = k * H, with H from the ALL-DATA QD members and k chosen on the held-out
QD TWINS (§10.3). If retraining made the all-data heads systematically smaller than the twins' heads,
the twin-chosen k would be too narrow for them. So compare the two on the SAME inputs, where BOTH are
in-sample: val_seen (the twins trained on it too; only VAL_TRAJ = val_unseen is held out of the twins).

    median H (all-data members, val_seen) / median H (twins, val_seen) = r
    0.95 <= r <= 1.05 -> k = k_base                         (the 09-24 build: r = 0.9884)
    r < 0.95          -> k = k_base * GRID_STEP            (one step wider on the k grid)
    r > 1.05          -> k = k_base, and the README says so (wider heads: the band is already wider)
    |r - 1| > 0.10    -> WARNING: outside what the rule was written for; a human reviews before shipping

GRID_STEP = (2.5/0.4)^(1/21) = 1.091187..., the ratio of head_direct.py's k grid (the docs said 1.089;
dry-run gap #20). k_base = 1.1399 is a grid point, so k_base * GRID_STEP = 1.2438, its upper neighbour.

H is exactly what the package multiplies by k: `_fno.predict_fno(..., err_out=E)` at the shipped ALPHA,
the member-average of exp(clamp(logerr)) in PHYSICAL units; the median is over every element of E.
This is Emre's inline check from the qd_build job (09-24, logs/hlevel.txt: all-data 0.005961, twins
0.006031, ratio 0.9884), moved into the repo with the grid step corrected; the arithmetic is unchanged.

    cd Track_1 && python qd_k/h_ratio_check.py \
        --members unet/ffno_all_w40l8_sim_qderr_s{1,2,3,4}.pth --twins qd_twin_s{1..6}.pth \
        --val val_seen.npz --k-base 1.1399 [--device cuda]
Prints `K_USE=<k>` as its last line (build scripts grep it). Exit 0 on success, 2 if |r-1| > 0.10.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submit_fno"))
import _fno  # noqa: E402

GRID_STEP = (2.5 / 0.4) ** (1.0 / 21.0)          # 1.091187...; head_direct.py's KGRID ratio
ALPHA_SHIP = 0.80


def median_h(paths, x, device, alpha):
    ms = [_fno.load_any(p, device) for p in paths]
    for p, m in zip(paths, ms):
        if getattr(m, "head_logerr", None) is None:
            sys.exit(f"FAIL: {p} has no error head (need a --err-head --err-loss qd checkpoint)")
    E = np.zeros((len(x), 20, 32, 64, 2), np.float32)
    _fno.predict_fno(ms, x, device, alpha=alpha, err_out=E, amp=None)
    if not np.isfinite(E).all():
        sys.exit("FAIL: non-finite H")
    return float(np.median(E))


def rule(r, k_base):
    if r < 0.95:
        return k_base * GRID_STEP, "r < 0.95: widened one grid step"
    if r <= 1.05:
        return k_base, "0.95 <= r <= 1.05: base kept"
    return k_base, "r > 1.05: base kept, note it in the package README"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--members", nargs="+", help="the all-data QD members (the package's H)")
    ap.add_argument("--twins", nargs="+", help="the held-out QD twins k was chosen on")
    ap.add_argument("--val", default="val_seen.npz", help="val_seen.npz from make_val.py (in-sample for both sets)")
    ap.add_argument("--k-base", type=float, default=1.1399, help="k chosen on the twins (§10.3)")
    ap.add_argument("--alpha", type=float, default=ALPHA_SHIP, help="compose ALPHA of the package (0.80)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true", help="check the rule and the grid step, no models")
    a = ap.parse_args()

    assert abs(GRID_STEP - 1.0911867) < 1e-6, GRID_STEP
    assert abs(1.1399 * GRID_STEP - 1.2438) < 1e-3
    if a.selftest:
        for r, want in ((0.9884, 1.1399), (0.94, 1.1399 * GRID_STEP), (1.06, 1.1399), (0.95, 1.1399), (1.05, 1.1399)):
            got = rule(r, 1.1399)[0]
            assert math.isclose(got, want, rel_tol=1e-9), (r, got, want)
        print("selftest OK: grid step %.6f, rule boundaries as documented" % GRID_STEP)
        return 0

    if not (a.members and a.twins):
        ap.error("--members and --twins are required (except with --selftest)")
    d = np.load(a.val)
    x = d["x"]
    print(f"val {a.val}: {len(x)} windows   alpha {a.alpha}   device {a.device}")
    ha = median_h(a.members, x, a.device, a.alpha)
    ht = median_h(a.twins, x, a.device, a.alpha)
    r = ha / ht
    k_use, why = rule(r, a.k_base)
    print(f"median H val_seen: all-data {ha:.6f}  twins {ht:.6f}  ratio {r:.4f}")
    print(f"base k {a.k_base:.4f} -> K_USE {k_use:.4f}  ({why})   grid step {GRID_STEP:.4f}")
    rc = 0
    if abs(r - 1.0) > 0.10:
        print("WARNING: |ratio-1| > 10% -- outside what the rule was written for; review before anything ships")
        rc = 2
    print(f"K_USE={k_use:.6f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
