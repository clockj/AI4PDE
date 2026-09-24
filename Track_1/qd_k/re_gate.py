"""qdre GATE (pre-registered, Ilse 08:03Z / 08:06Z 09-24; Devi's wording). Combines the per-seed CSVs.

  per trajectory: delta = mean over seeds of delta_final (QD k=1.1399 - control)
  PASS iff  (1) rehi one-sided 95% t lower bound of the per-trajectory deltas > 0   (7 traj, df 6, t 1.943180)
       and  (2) median(rehi per-traj delta) >= 0.5 * median(rein per-traj delta)
  Also printed (not gating): per-Re means inside each group (trajectories at one Re are not independent),
  relo the same way, and each seed on its own.
usage: python re_gate.py qdre_delta_1.csv qdre_delta_2.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np

T95 = {1: 6.313752, 2: 2.919986, 3: 2.353363, 4: 2.131847, 5: 2.015048, 6: 1.943180, 7: 1.894579, 8: 1.859548, 9: 1.833113}
print("INPUT CSVs: " + ", ".join(sys.argv[1:]) + "   (qdre_delta_<seed>.csv = OFFICIAL symmetric control k, Ilse 08:22Z; "
      "qdre_delta_diag_<seed>.csv = her 08:06Z rule, diagnostic)")
rows = [r for p in sys.argv[1:] for r in csv.DictReader(open(p))]
seeds = sorted({r["seed"] for r in rows})
per = defaultdict(list); meta = {}
for r in rows:
    per[r["trajectory"]].append(float(r["delta_final"])); meta[r["trajectory"]] = (r["group"], int(r["re"]))
assert all(len(v) == len(seeds) for v in per.values()), "a trajectory is missing a seed"
print(f"seeds {seeds}; per-trajectory delta = seed mean")
G = defaultdict(dict)
for t, v in per.items():
    G[meta[t][0]][t] = float(np.mean(v))
for g in ("rein", "rehi", "relo"):
    d = np.array(list(G[g].values())); n = len(d)
    lb = d.mean() - T95[n - 1] * d.std(ddof=1) / np.sqrt(n)
    print(f"\n[{g}] n={n}  mean {d.mean():+.5f}  median {np.median(d):+.5f}  one-sided 95% LB {lb:+.5f}  n+ {(d > 0).sum()}/{n}")
    for re_ in sorted({meta[t][1] for t in G[g]}):
        dd = [G[g][t] for t in G[g] if meta[t][1] == re_]
        print(f"   Re {re_:6d}: mean {np.mean(dd):+.5f} over {len(dd)} traj")
    for s in seeds:
        ds = [float(r["delta_final"]) for r in rows if r["group"] == g and r["seed"] == s]
        print(f"   seed {s}: mean {np.mean(ds):+.5f}  median {np.median(ds):+.5f}")
NT = {g: len(G[g]) for g in ("rein", "rehi", "relo")}
print(f"\ntrajectories per group {NT}")
if not any("fake" in p or "t1" in p for p in sys.argv[1:]):
    assert NT == {"rein": 10, "rehi": 7, "relo": 9}, NT
hi, ri = np.array(list(G["rehi"].values())), np.array(list(G["rein"].values()))
# Devi 08:23Z diagnostic (not a gate): Re-level units only -- mean per Re (2 Re values), df = 1, t = 6.313752
rem = [np.mean([G["rehi"][t] for t in G["rehi"] if meta[t][1] == re_]) for re_ in sorted({meta[t][1] for t in G["rehi"]})]
if len(rem) >= 2:
    lb_re = np.mean(rem) - T95[len(rem) - 1] * np.std(rem, ddof=1) / np.sqrt(len(rem))
    print(f"[diag] rehi per-Re means {[round(float(x), 5) for x in rem]}  Re-level one-sided 95% LB (df {len(rem) - 1}) {lb_re:+.5f}")
lb = hi.mean() - T95[len(hi) - 1] * hi.std(ddof=1) / np.sqrt(len(hi))
c1 = lb > 0
c2 = None if np.median(ri) <= 0 else bool(np.median(hi) >= 0.5 * np.median(ri))
print(f"\nGATE (1) rehi LB {lb:+.5f} > 0: {'PASS' if c1 else 'FAIL'}")
if c2 is None:   # Devi 08:23Z: with rein median <= 0 the ratio is meaningless (it would pass automatically)
    print(f"GATE (2) UNDEFINED: rein median {np.median(ri):+.5f} <= 0 -- needs a leader ruling, NOT a pass")
else:
    print(f"GATE (2) median rehi {np.median(hi):+.5f} >= 0.5 x median rein {0.5 * np.median(ri):+.5f}: {'PASS' if c2 else 'FAIL'}")
print(f"VERDICT: {'UNDEFINED (gate 2)' if (c2 is None and c1) else ('PASS' if (c1 and c2) else 'FAIL')}")
