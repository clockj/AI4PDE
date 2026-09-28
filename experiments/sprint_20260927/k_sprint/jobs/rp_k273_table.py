#!/usr/bin/env python3
"""Parse rp_k273_cr_<tag>_*.log (candref) into one paired table. usage: rp_k273_table.py [tag ...]  (baseline is always row 0)
Columns: ens_v9 twin reads (errors; sps, final proxy at f) and twin-package stream reads (scores). d_fin_nt = delta of
0.4778 rel + 0.1010 tke + 0.0921 mvpe + 0.2498 sps (no time) vs baseline. Per-trajectory sign counts from the static stream."""
import glob, re, sys, os
K = "/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927"
W = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)
def parse(tag):
    fs = sorted(glob.glob(f"{K}/logs/rp_k273_cr_{tag}_[0-9]*.log"), key=os.path.getmtime)
    if not fs: return None
    out, stage = {}, None
    for ln in open(fs[-1], errors="replace"):
        m = re.match(r"\[STAGE [^\]]+\] (.*)", ln)
        if m: stage = m.group(1).strip(); continue
        if stage and stage.startswith("ens_v9") and "|" in ln and re.search(r"\|\s+0\.\d+", ln):
            v = [float(x) for x in re.findall(r"-?\d+\.\d+", ln.split("|", 1)[1])]
            out[stage] = dict(rel_e=v[0], tke_e=v[1], mvpe_e=v[2], sps=v[3], final=v[5])
        if stage and stage.startswith("stream") and "scores" in ln:
            d = dict(zip(re.findall(r"(rel|tke|mvpe|time|sps)", ln), map(float, re.findall(r"-?\d+\.\d+", ln))))
            d["fin_nt"] = sum(W[k] * d[k] for k in W); out[stage] = d
        if stage and stage.startswith("stream") and "errors" in ln:
            out[stage + " ERR"] = [float(x) for x in re.findall(r"-?\d+\.\d+", ln)]
    out["_done"] = "RP_K273_CANDREF_END" in open(fs[-1], errors="replace").read(); out["_log"] = fs[-1]
    return out
REF = os.environ.get("REF", "baseline")
tags = [REF] + [t for t in sys.argv[1:] if t != REF]
R = {t: parse(t) for t in tags}; B = R[REF]
print(f"# paired reference: {REF}")
TR = "3750_0 26700_0 16500_5 6300_10 13950_10 21600_15 8850_20 11400_20".split()
for t in tags:
    r = R[t]
    if r is None: print(f"## {t}: no log"); continue
    print(f"\n## {t}   (complete={r['_done']})  {r['_log']}")
    for k in sorted(k for k in r if k.startswith("ens_v9")):
        v = r[k]; b = B.get(k) if B else None
        s = f"  {k:34s} rel {v['rel_e']:.4f} tke {v['tke_e']:.4f} mvpe {v['mvpe_e']:.4f} sps {v['sps']:.2f} final {v['final']:.3f}"
        if b and t != REF:
            s += f" | d rel {100*(v['rel_e']/b['rel_e']-1):+.2f}% tke {100*(v['tke_e']/b['tke_e']-1):+.2f}% mvpe {100*(v['mvpe_e']/b['mvpe_e']-1):+.2f}% sps {v['sps']-b['sps']:+.2f} final {v['final']-b['final']:+.3f}"
        print(s)
    for k in [k for k in r if k.startswith("stream") and not k.endswith("ERR") and " traj " not in k]:
        v = r[k]; b = B.get(k) if B else None
        cov = r.get(k + " ERR", [None]*4)[-1]; cb = B.get(k + " ERR", [None]*4)[-1] if B else None
        s = f"  {k:34s} rel {v['rel']:.3f} tke {v['tke']:.3f} mvpe {v['mvpe']:.3f} sps {v['sps']:.3f} cov {cov}"
        if b and t != REF:
            s += f" | d rel {v['rel']-b['rel']:+.3f} tke {v['tke']-b['tke']:+.3f} mvpe {v['mvpe']-b['mvpe']:+.3f} sps {v['sps']-b['sps']:+.3f} d_fin_nt {v['fin_nt']-b['fin_nt']:+.4f}"
        print(s)
    if t != REF and B:
        ds = []
        for tr in TR:
            k = f"stream twin static traj {tr}"
            if k in r and k in B: ds.append((tr, r[k]["fin_nt"] - B[k]["fin_nt"], r[k]["rel"] - B[k]["rel"], r[k]["sps"] - B[k]["sps"]))
        if ds:
            hi = [d for d in ds if d[0].split("_")[1] in ("15", "20")]
            print("  per-trajectory static d_fin_nt: " + "  ".join(f"{a}:{b:+.3f}" for a, b, _, _ in ds))
            print(f"  sign counts d_fin_nt>0: AoA15-20 {sum(d[1]>0 for d in hi)}/{len(hi)}  all {sum(d[1]>0 for d in ds)}/{len(ds)};  rel>0: {sum(d[2]>0 for d in hi)}/{len(hi)}, {sum(d[2]>0 for d in ds)}/{len(ds)};  sps>0: {sum(d[3]>0 for d in hi)}/{len(hi)}, {sum(d[3]>0 for d in ds)}/{len(ds)}")
