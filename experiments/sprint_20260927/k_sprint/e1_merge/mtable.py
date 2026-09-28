import json, sys
R = {}
for l in open("/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/e1/mread.jsonl"):
    r = json.loads(l); R[r["tag"]] = r
names = {"tw_R0": "R0 asnp+D1c30", "tw_R1": "R1 CONS", "tw_R1j": "R1+JG", "tw_R2": "R2 CONS+ARENA+JG", "tw_R3": "R3 CONS+ARENA+JG+SIGPOW",
         "tw_R3nj": "R3nj CONS+ARENA+SIGPOW (no JG)", "tw_R4": "R4 AGGR", "tw_R5": "R5 AGGR+all3", "tw_R0add": "R0+all3 (no VC) = k27fin_d1c twin"}
keys = ["rel", "tke", "mvpe", "sps", "coverage", "fnt"]
def row(t, sp, ref=None):
    d = R[t][sp]; s = " | ".join("%.4f" % d[k] if k == "coverage" else "%.3f" % d[k] for k in keys)
    if ref:
        b = R[ref][sp]; s += " | " + " ".join("%+.3f" % (d[k] - b[k]) for k in ("rel", "tke", "mvpe", "sps", "fnt"))
        tr = R[t]["traj"]; tb = R[ref]["traj"]
        pos = sum(tr[x]["fnt"] > tb[x]["fnt"] for x in tr); neg = sum(tr[x]["fnt"] < tb[x]["fnt"] for x in tr)
        if sp == "all": s += " | %d+/%d- of %d" % (pos, neg, len(tr))
        else:
            a = [x for x in tr if int(x.split("_")[1]) >= 15]
            s += " | %d+/%d- of %d" % (sum(tr[x]["fnt"] > tb[x]["fnt"] for x in a), sum(tr[x]["fnt"] < tb[x]["fnt"] for x in a), len(a))
    return s
pairs = [("tw_R0", None), ("tw_R1", "tw_R0"), ("tw_R1j", "tw_R1"), ("tw_R2", "tw_R1"), ("tw_R3", "tw_R1"), ("tw_R3nj", "tw_R1"), ("tw_R3", "tw_R0"),
         ("tw_R0add", "tw_R0"), ("tw_R4", "tw_R0"), ("tw_R5", "tw_R4"), ("tw_R5", "tw_R0"), ("tw_R3", "tw_R3nj")]
for sp in ("all", "aoa15_20"):
    print(f"\n### {sp}\n| variant | vs | rel | tke | mvpe | sps | cov | final-no-time | d rel d tke d mvpe d sps d fnt | traj sign (fnt) |\n|---|---|---|---|---|---|---|---|---|---|")
    for t, ref in pairs:
        if t in R and (ref is None or ref in R):
            print(f"| {names[t]} | {names.get(ref, '-') if ref else '-'} | {row(t, sp, ref)} |")
