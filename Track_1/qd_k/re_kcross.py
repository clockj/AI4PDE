"""Local k readouts from qdre_tables_<seed>.npz (no GPU). Jonas's finals-k criterion 1 (via Devi 12:22Z):
CROSS-SELECTION inside rehi: pick QD k* on Re 25425 (whole KGRID_D, in-sample there), evaluate on 26700; and the
reverse. Also prints, per group and seed, the QD sps curve around 1.1399 and QD(k) - QD(1.1399) at the selected k.
usage: python re_kcross.py qdre_tables_1.npz qdre_tables_2.npz
"""
import sys

import numpy as np

for p in sys.argv[1:]:
    z = np.load(p)
    KD, KQ, W = z["kgrid_d"], float(z["kq"]), float(z["W"])
    iq = int(np.argmin(np.abs(KD - KQ)))

    def S(g, i, sel):
        a, n, t = z[g + "_a"], z[g + "_nsc"], z[g + "_tq"][i]
        return 100.0 * float(a[sel] @ t[sel]) / int(n[sel].sum())
    print(f"\n==== {p}  (QD k grid {len(KD)} pts, shipped {KQ:.4f})")
    g = "rehi"
    tr = z[g + "_traj"]; re_ = np.array([int(t.split("_")[0]) for t in tr])
    for fit, ev in ((25425, 26700), (26700, 25425)):
        sf, se = re_ == fit, re_ == ev
        if not sf.any() or not se.any():
            print(f"  rehi cross {fit}->{ev}: no trajectories at one of these Re in this file (fake/test data?)"); continue
        ks = int(np.argmax([S(g, i, sf) for i in range(len(KD))]))
        d_ev = W * (S(g, ks, se) - S(g, iq, se))
        print(f"  rehi cross: k* on Re {fit} = {KD[ks]:.4f} ({int(round(np.log(KD[ks] / KQ) / np.log(KD[1] / KD[0]))):+d} steps)"
              f"  -> on Re {ev}: QD(k*) - QD({KQ:.4f}) = {d_ev:+.5f} final")
    for g in ("rein", "rehi", "relo"):
        allg = np.ones(len(z[g + "_traj"]), bool)
        curve = {round(float(KD[i]), 4): round(W * (S(g, i, allg) - S(g, iq, allg)), 5) for i in range(iq - 2, iq + 3)}
        print(f"  {g}: in-sample QD(k) - QD({KQ:.4f}) final, k around shipped: {curve}")
