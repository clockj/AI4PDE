# EA: paired delta table from rp_f27x_candref logs (candidate vs baseline). usage: table.py base.log cand1.log [cand2.log ...]
import re, sys
WT = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)


def parse(path):
    out, stage = {}, None
    for ln in open(path):
        m = re.search(r"\[STAGE [^\]]*\] (.*)", ln)
        if m:
            stage = m.group(1).strip(); continue
        if stage and stage.startswith("ens_v9"):
            m = re.search(r"\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)", ln)
            if m:
                v = list(map(float, m.groups()))
                out[stage] = dict(rel_e=v[0], tke_e=v[1], mvpe_e=v[2], sps=v[3], mean4=v[4], final=v[5])
        if stage and stage.startswith("stream"):
            m = re.search(r"errors\s+rel ([\d.]+)\s+tke ([\d.]+)\s+mvpe ([\d.]+)\s+coverage ([\d.]+)", ln)
            if m:
                v = list(map(float, m.groups())); out.setdefault(stage, {}).update(rel_e=v[0], tke_e=v[1], mvpe_e=v[2], cov=v[3])
            m = re.search(r"scores\s+rel ([\d.]+)\s+tke ([\d.]+)\s+mvpe ([\d.]+)\s+time ([\d.]+)\s+sps ([\d.]+)", ln)
            if m:
                v = list(map(float, m.groups()))
                d = out.setdefault(stage, {}); d.update(rel=v[0], tke=v[1], mvpe=v[2], sps=v[4])
                d["final"] = sum(WT[k] * d[k] for k in WT)      # time excluded (not valid locally)
    return out


base = parse(sys.argv[1])
for p in sys.argv[2:]:
    c = parse(p)
    print(f"\n### {p.split('/')[-1]}  (delta = cand - baseline; errors as relative %, sps/final absolute)")
    print("| read | d rel err | d tke err | d mvpe err | d sps | d final | cand final | base final |")
    print("|---|---|---|---|---|---|---|---|")
    for k in base:
        if k not in c or "final" not in c[k]:
            print(f"| {k} | MISSING |"); continue
        b, a = base[k], c[k]
        r = lambda n: 100 * (a[n] / b[n] - 1)
        print(f"| {k} | {r('rel_e'):+.2f}% | {r('tke_e'):+.2f}% | {r('mvpe_e'):+.2f}% | {a['sps'] - b['sps']:+.2f} | "
              f"{a['final'] - b['final']:+.3f} | {a['final']:.3f} | {b['final']:.3f} |")
