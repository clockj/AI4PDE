"""Chunk 5, part 1: T4 − control over the leave-one-out ensembles, next to the control-against-control reseeding spread.

    python ood_jkread.py [RUNS_DIR]
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs"
GROUPS = [("relo", False), ("rehi", False), ("rein", False), ("REID", True)]


def comp(run, g, idread, v="control"):
    p = os.path.join(RUNS, run, "idread" if idread else "", "summary.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["sets"]["plain"]
    if g not in s or v not in s[g]:
        return None
    return s[g][v]


def main():
    L = []
    for v in ("control", "frozen"):
        L += [f"### {v}", "", "| group | T4 − control, full 4-member ensembles | jackknife T4 − control, mean [min, max] (n+ trajectories per drop) | "
              "control − control over reseeded 4-member draws: pairwise |Δ| mean, max (range of levels) | "
              "control leave-one-out spread (range of levels) |", "|---|---|---|---|---|"]
        for g, idr in GROUPS:
            full = None
            a, b = comp("re_mirror", g, idr, v), comp("re_t4", g, idr, v)
            if a and b:
                full = b["full"]["composite"] - a["full"]["composite"]
            jk, npos = [], []
            for d in (1, 2, 3, 4):
                c, t = comp(f"re_jk_ctl_d{d}", g, idr, v), comp(f"re_jk_t4_d{d}", g, idr, v)
                if c and t:
                    jk.append(t["full"]["composite"] - c["full"]["composite"])
                    npos.append(sum(t["traj"][x]["full"] > c["traj"][x]["full"] for x in c["traj"]))
            draws = [comp(r, g, idr, v) for r in ("re_mirror", "re_draw_s3456", "re_draw_s1256", "re_draw_s1346")]
            lv = [x["full"]["composite"] for x in draws if x]
            pw = [abs(p - q) for p, q in itertools.combinations(lv, 2)]
            loo = [comp(f"re_jk_ctl_d{d}", g, idr, v) for d in (1, 2, 3, 4)]
            ll = [x["full"]["composite"] for x in loo if x]
            n = len(next(iter([x for x in (a, b) if x]))["traj"]) if (a or b) else 0
            L.append(f"| {g}{' (in distribution: VAL_TRAJ minus RE)' if idr else ''} | {'' if full is None else '%+.4f' % full} | "
                     + (f"{np.mean(jk):+.4f} [{min(jk):+.4f}, {max(jk):+.4f}] ({', '.join(f'{p}/{n}' for p in npos)})" if jk else "pending")
                     + " | " + (f"{np.mean(pw):.4f}, {max(pw):.4f} ({max(lv) - min(lv):.4f}; {len(lv)} draws)" if len(lv) > 1 else "pending")
                     + " | " + (f"{max(ll) - min(ll):.4f} ({len(ll)})" if len(ll) > 1 else "pending") + " |")
        L.append("")
    print("\n".join(L))


if __name__ == "__main__":
    main()
