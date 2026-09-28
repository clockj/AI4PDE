"""Design B2: the three-regime markdown table. d = delta of the no-time composite (0.2498 * delta sps) vs shipped asym R1 on the
same prediction-side stream; (n/m) = trajectories with a positive delta.  python b2table.py <d30|d10>"""
import sys, json, os
import numpy as np
S = "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927"
d = sys.argv[1] if len(sys.argv) > 1 else "d30"
cols = [("in", "100", "full"), ("in", "129", "full"), ("in", "100", "hi"), ("in", "129", "hi"), ("ood", "100", "full"), ("ood", "129", "full"), ("ood", "129", "t14")]
rows = {}
base_lv = {}
for which, f, sel in cols:
    tag = f"{which}_{d}_f{f}"
    for sfx, base in (("", "R1ship"), ("_wave2", "R1ship"), ("_wave3", "R1ship"), ("_wave4", "R1ship"), ("_centre", "c_ship")):
        p = f"{S}/ckpt/b2/res_{tag}{sfx}.json"
        if not os.path.exists(p):
            continue
        R = json.load(open(p)); names = np.array(R["names"]); a = np.array(R["a"]); nsc = np.array(R["nsc"]); idx = np.array(R["idx"])
        aoa = np.array([int(t.split("_")[1]) for t in names]); trajs = list(dict.fromkeys(names))
        m = np.ones(len(names), bool)
        if sel == "hi":
            m = aoa >= 15
        if sel == "t14":
            m = idx < 14
        sps = lambda c, mm: 100.0 * float((a[mm] * c[mm]).sum()) / float(nsc[mm].sum())
        b = np.array(R["var"][base]["con"])
        base_lv[(which, f, sel)] = (sps(b, m), R["ref_comp"])
        for nm, v in R["var"].items():
            if nm in ("R1ship", "R1ship_", "c_ship"):
                continue
            c = np.array(v["con"])
            tr = [t for t in trajs if (names == t)[m].any() or True]
            per = [sps(c, (names == t) & m) - sps(b, (names == t) & m) for t in trajs if ((names == t) & m).any()]
            rows.setdefault(nm, {})[(which, f, sel)] = (0.2498 * (sps(c, m) - sps(b, m)), sum(x > 0 for x in per), len(per))
print("| variant | " + " | ".join(f"{w} {s} f={int(f) / 100:.2f}" for w, f, s in cols) + " |")
print("|---|" + "---|" * len(cols))
print("| asym R1 + D1c (level: sps) | " + " | ".join(f"{base_lv[c][0]:.2f}" if c in base_lv else "n/a" for c in cols) + " |")
for nm, r in rows.items():
    print(f"| {nm} | " + " | ".join((f"{r[c][0]:+.3f} ({r[c][1]}/{r[c][2]})" if c in r else "n/a") for c in cols) + " |")
