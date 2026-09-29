"""Gate G0 and every arm against its PLAN.md criterion, from the finished runs (no GPU).

    python ood_arms.py [RUNS_DIR]      # default /fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs

Reads runs/{a20,a10,re}_{mirror,six,vitc2}/summary.json (+ idread/summary.json), runs/id_twins/summary.json (in
distribution, twins of record), runs/*_i6*/ (I6), runs/*_mirror/i7.json (I7), runs/timing_*.json (T1). A regime whose run
is missing is reported as missing, never substituted. OOD regimes = A20, A10, relo, rehi (rein is reported, not counted).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs"
OOD = [("A20", "a20", "A20"), ("A10", "a10", "A10"), ("relo", "re", "relo"), ("rehi", "re", "rehi")]
ALLG = OOD + [("rein", "re", "rein")]


def S(run, sub="plain", idread=False):
    p = os.path.join(RUNS, run, "idread" if idread else "", "summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["sets"].get(sub)


def dv(run, group, v, ref="control", sub="plain"):
    s = S(run, sub)
    if not s or group not in s or v not in s[group] or ref not in s[group][v].get("delta", {}):
        return None
    return s[group][v]["delta"][ref]


def fmt(d, key="full"):
    return "missing" if d is None else f"{d[key]:+.4f} ({d['npos']}/{d['n']})"


def verdict(ok):
    return "**PASS**" if ok else "**FAIL**"


def cross(run_a, run_b, group, v="control", idread=False):
    """composite of run_b minus run_a (variant v), pooled and per trajectory."""
    a, b = S(run_a, idread=idread), S(run_b, idread=idread)
    if not a or not b or group not in a or group not in b:
        return None
    A, B = a[group][v], b[group][v]
    d = B["full"]["composite"] - A["full"]["composite"]
    per = [B["traj"][t]["full"] - A["traj"][t]["full"] for t in A["traj"]]
    return dict(full=d, npos=int(sum(x > 0 for x in per)), n=len(per),
                first3=B["first3"]["composite"] - A["first3"]["composite"],
                first14=B["first14"]["composite"] - A["first14"]["composite"],
                subs={k: B["full"][k] - A["full"][k] for k in ("rel", "tke", "mvpe", "sps", "C")})


def main():
    L = []
    # ---------------- G0
    L += ["## Gate G0: control − base on the mirror ensembles (≥ +0.15 on A20 and A10, ≥ 80 % positive)", "",
          "| regime | control − base (n+) | first 3 / 14 | rel / tke / mvpe / sps | verdict |", "|---|---|---|---|---|"]
    g0 = []
    for lab, r, g in ALLG:
        d = dv(f"{r}_mirror", g, "control", "base")
        if d is None:
            L.append(f"| {lab} | missing | | | |"); continue
        ok = d["full"] >= 0.15 and d["npos"] >= 0.8 * d["n"]
        if lab in ("A20", "A10"):
            g0.append(ok)
        L.append(f"| {lab} | {fmt(d)} | {d['first3']:+.4f} / {d['first14']:+.4f} | {d['d_rel']:+.3f} / {d['d_tke']:+.3f} / "
                 f"{d['d_mvpe']:+.3f} / {d['d_sps']:+.3f} | {verdict(ok) if lab in ('A20', 'A10') else '(reported)'} |")
    L += ["", f"G0: {verdict(len(g0) == 2 and all(g0)) if len(g0) == 2 else 'incomplete'}", ""]

    # ---------------- I1
    L += ["## I1: PREV_SPACE target (control − prev_input ≥ 0 in every regime)", "", "| regime | control − prev_input (n+) |", "|---|---|"]
    ok1 = []
    for lab, r, g in ALLG:
        d = dv(f"{r}_mirror", g, "prev_input")
        if d is None:
            L.append(f"| {lab} | missing |"); continue
        x = -d["full"]; n_ = d["n"] - d["npos"] - 0
        ok1.append(x >= 0)
        L.append(f"| {lab} | {x:+.4f} ({d['n'] - d['npos']}/{d['n']} not negative for control) |")
    di = dv("id_twins", "all8", "prev_input")
    L += ["| in distribution (twins) | %s |" % ("missing" if di is None else "%+.4f" % -di["full"]), "",
          f"I1: {verdict(all(ok1)) if len(ok1) == 5 else 'incomplete'}", ""]

    # ---------------- I3 FAMW
    L += ["## I3: FAMW (FW_N0 20) − control (OOD mean ≥ +0.05, A20 and A10 > 0, in distribution ≥ −0.01, ≤ +0.15 ms)", "",
          "| regime | famw − control (n+) | first 3 / 14 | local ms/step famw − control |", "|---|---|---|---|"]
    vals = {}
    for lab, r, g in OOD:
        d = dv(f"{r}_mirror", g, "famw")
        ms = None
        try:
            jf = json.load(open(os.path.join(RUNS, f"{r}_mirror", "json", "famw.json")))
            jc = json.load(open(os.path.join(RUNS, f"{r}_mirror", "json", "control.json")))
            ms = jf["step_ms_median"] - jc["step_ms_median"]
        except Exception:
            pass
        if d is not None:
            vals[lab] = d["full"]
        early = "" if d is None else "%+.4f / %+.4f" % (d["first3"], d["first14"])
        L.append(f"| {lab} | {fmt(d)} | {early} | {'' if ms is None else '%+.3f' % ms} |")
    di = dv("id_twins", "all8", "famw")
    if len(vals) == 4:
        m = float(np.mean(list(vals.values())))
        ok = m >= 0.05 and vals["A20"] > 0 and vals["A10"] > 0 and di is not None and di["full"] >= -0.01
        L += [f"| OOD mean | {m:+.4f} | | |", f"| in distribution | {fmt(di)} | | |", "", f"I3: {verdict(ok)} (latency: local, shared nodes)", ""]
    else:
        L += ["", "I3: incomplete", ""]

    # ---------------- I4 bound
    L += ["## I4: hindsight bound (windows n ≤ 2 given the 8-13 gain), full streams / first 14 windows", "",
          "| regime | bound full | bound first 14 |", "|---|---|---|"]
    for lab, r, g in ALLG:
        s = S(f"{r}_mirror")
        nb = (s or {}).get(g, {}).get("_nbins")
        L.append("| %s | %s | %s |" % (lab, "missing" if not nb else "%+.4f" % nb["bound_full"],
                                        "" if not nb else "%+.4f" % nb["bound_first14"]))
    L.append("")

    # ---------------- I7
    L += ["## I7: zero propagation Z1 (post hoc), vs control", "",
          "| regime | plain (n+ / n−) | m1 | m2 | m3 | Track 1 plain |", "|---|---|---|---|---|---|"]
    for lab, r, g in ALLG:
        p = os.path.join(RUNS, f"{r}_mirror", "i7.json")
        if not os.path.exists(p):
            L.append(f"| {lab} | missing | | | | |"); continue
        B = json.load(open(p))["B"]
        c = lambda m: B[m][g]["Z1"]["vs_control"]  # noqa: E731
        f = lambda m: f"{c(m)['full']['composite']:+.4f} ({c(m)['npos']}/{c(m)['n']})" if m in B else "n/a"  # noqa: E731
        L.append(f"| {lab} | {c('plain')['full']['composite']:+.4f} ({c('plain')['npos']}/{c('plain')['n']} / {c('plain')['nneg']}) | "
                 f"{f('m1')} | {f('m2')} | {f('m3')} | {B['plain'][g]['Z1']['frozen_vs_frozen']['full']['composite']:+.4f} |")
    L.append("")

    # ---------------- T1 / T2
    for arm, lay, bar_ood, bar_id in (("T1", "six", 0.05, 0.0), ("T2", "vitc2", 0.05, -0.02)):
        L += [f"## {arm}: {lay} − mirror, control (OOD mean ≥ +{bar_ood}, in distribution ≥ {bar_id:+.2f})", "",
              "| regime | Δ composite (n+) | first 3 / 14 | rel / tke / mvpe / sps |", "|---|---|---|---|"]
        vals = {}
        for lab, r, g in ALLG:
            d = cross(f"{r}_mirror", f"{r}_{lay}", g)
            if d is None:
                L.append(f"| {lab} | missing | | |"); continue
            if lab != "rein":
                vals[lab] = d["full"]
            s_ = d["subs"]
            L.append(f"| {lab} | {d['full']:+.4f} ({d['npos']}/{d['n']}) | {d['first3']:+.4f} / {d['first14']:+.4f} | "
                     f"{s_['rel']:+.3f} / {s_['tke']:+.3f} / {s_['mvpe']:+.3f} / {s_['sps']:+.3f} |")
        ids = [cross(f"{r}_mirror", f"{r}_{lay}", f"{r.upper()}ID", idread=True) for r in ("a20", "a10", "re")]
        idv = [x["full"] for x in ids if x]
        for r, x in zip(("A20ID", "A10ID", "REID"), ids):
            L.append("| in distribution %s (VAL_TRAJ minus the regime; band in sample) | %s | | %s |" % (
                r, "missing" if not x else "%+.4f (%d/%d)" % (x["full"], x["npos"], x["n"]),
                "" if not x else " / ".join("%+.3f" % x["subs"][k] for k in ("rel", "tke", "mvpe", "sps"))))
        if len(vals) == 4 and idv:
            m = float(np.mean(list(vals.values()))); mi = float(np.mean(idv))
            L += [f"| OOD mean | {m:+.4f} | | |", f"| in-distribution mean | {mi:+.4f} | | |", "",
                  f"{arm}: {verdict(m >= bar_ood and mi >= bar_id)}", ""]
        else:
            L += ["", f"{arm}: incomplete", ""]
        if arm == "T1":
            tp = os.path.join(RUNS, "timing_a20_six_vs_mirror.json")
            if os.path.exists(tp):
                t = json.load(open(tp))
                L += [f"T1 local A100 step time (same job, interleaved, {len(t['rounds_a'])} rounds): mirror {t['median_a']:.3f} ms, "
                      f"six {t['median_b']:.3f} ms, six − mirror {t['delta_median']:+.3f} ms [{t['delta_min']:+.3f}, {t['delta_max']:+.3f}] "
                      f"(node {t['host']}; a price tag, not board latency evidence)", ""]
    print("\n".join(L))


if __name__ == "__main__":
    main()
