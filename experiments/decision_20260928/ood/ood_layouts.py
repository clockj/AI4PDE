"""Ensemble-composition layouts against `mirror`, per regime, with the in-distribution read next to each (no GPU).

    python ood_layouts.py [RUNS_DIR]

For every layout L and regime group: composite of L minus composite of mirror (same regime, same variant), pooled, with the
count of trajectories where L is better, the four subscores, for `control` (Track 2) and `frozen` (Track 1's case); the
in-distribution read of the same two ensembles (idread: VAL_TRAJ minus the regime; the band is fitted on those
trajectories, so the sps part there is in sample); and FAMW: famw − control and the learned weight of family B (the second
fused group, the ViTs), mean over each stream's windows k >= 28 and over its first 14, against its prior w0.
In distribution on the twins: F-FNO-only twins against the five twins of record, both with the band refitted the
shipping way (id_ffno3_ship vs id_twins_ship).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs"
GROUPS = [("A20", "a20", "A20"), ("A10", "a10", "A10"), ("relo", "re", "relo"), ("rehi", "re", "rehi"), ("rein", "re", "rein")]
LAYOUTS = ["ffno4", "ffno6", "six", "vitc2", "six_vitc2", "t3", "t4"]
OOD = ("A20", "A10", "relo", "rehi")


def summ(run, idread=False):
    p = os.path.join(RUNS, run, "idread" if idread else "", "summary.json")
    return json.load(open(p))["sets"]["plain"] if os.path.exists(p) else None


def cross(a, b, g, v):
    if not a or not b or g not in a or g not in b or v not in a[g] or v not in b[g]:
        return None
    A, B = a[g][v], b[g][v]
    per = [B["traj"][t]["full"] - A["traj"][t]["full"] for t in A["traj"]]
    return dict(d=B["full"]["composite"] - A["full"]["composite"], npos=int(sum(x > 0 for x in per)), n=len(per),
                sub={k: B["full"][k] - A["full"][k] for k in ("rel", "tke", "mvpe", "sps")},
                f14=B["first14"]["composite"] - A["first14"]["composite"])


def fmt(x):
    return "—" if x is None else f"{x['d']:+.4f} ({x['npos']}/{x['n']})"


def subs(x):
    return "" if x is None else " / ".join(f"{x['sub'][k]:+.3f}" for k in ("rel", "tke", "mvpe", "sps"))


def famw_weight(run, trajs):
    p = os.path.join(RUNS, run, "json", "famw.json")
    if not os.path.exists(p):
        return None
    w = json.load(open(p))["windows"]
    if "fw_w" not in w:
        return None
    W = np.asarray(w["fw_w"]); k = np.asarray(w["k"]); tr = np.asarray(w["traj"])
    s = np.isin(tr, trajs)
    return dict(late=W[s & (k >= 28)].mean(0).tolist(), early=W[s & (k < 14)].mean(0).tolist(), w0=w["fw_w0"])


def main():
    L = []
    for lay in LAYOUTS:
        L += [f"## {lay} − mirror", "",
              "| regime | control Δ (n+) | control rel / tke / mvpe / sps | control first 14 | frozen Δ (n+) | "
              "in distribution (regime ensemble) control Δ (n+) | frozen Δ (n+) |", "|---|---|---|---|---|---|---|"]
        vals, idv = [], {}
        for lab, r, g in GROUPS:
            m, x = summ(f"{r}_mirror"), summ(f"{r}_{lay}")
            c, f = cross(m, x, g, "control"), cross(m, x, g, "frozen")
            mi, xi = summ(f"{r}_mirror", True), summ(f"{r}_{lay}", True)
            ci = cross(mi, xi, f"{r.upper()}ID", "control"); fi = cross(mi, xi, f"{r.upper()}ID", "frozen")
            if c is None and f is None:
                L.append(f"| {lab} | missing | | | | | |"); continue
            if lab in OOD and c is not None:
                vals.append(c["d"])
            if ci is not None:
                idv[r] = ci["d"]
            L.append(f"| {lab} | {fmt(c)} | {subs(c)} | {'' if c is None else '%+.4f' % c['f14']} | {fmt(f)} | "
                     f"{fmt(ci) if lab in ('A20', 'A10', 'relo') else '(RE: see relo row)'} | "
                     f"{fmt(fi) if lab in ('A20', 'A10', 'relo') else ''} |")
        if vals:
            L += [f"| OOD mean ({len(vals)} of 4) | {np.mean(vals):+.4f} | | | | in-distribution mean "
                  f"{np.mean(list(idv.values())):+.4f} ({len(idv)} ensembles) | |" if idv else
                  f"| OOD mean ({len(vals)} of 4) | {np.mean(vals):+.4f} | | | | | |"]
        L.append("")
    # FAMW per layout
    L += ["## FAMW (FW_N0 20): famw − control and the learned weight of family B (the ViTs)", "",
          "| layout | regime | famw − control (n+) | weight of family B, windows k ≥ 28 (u / v) | first 14 windows (u / v) | prior w0 |",
          "|---|---|---|---|---|---|"]
    for lay in ["mirror"] + LAYOUTS:
        for lab, r, g in GROUPS:
            s = summ(f"{r}_{lay}")
            if not s or g not in s or "famw" not in s[g]:
                continue
            d = s[g]["famw"]["delta"].get("control")
            trajs = list(s[g]["control"]["traj"])
            w = famw_weight(f"{r}_{lay}", trajs)
            L.append(f"| {lay} | {lab} | {'' if d is None else '%+.4f (%d/%d)' % (d['full'], d['npos'], d['n'])} | "
                     f"{'inactive (one family)' if w is None else '%.3f / %.3f' % tuple(w['late'])} | "
                     f"{'' if w is None else '%.3f / %.3f' % tuple(w['early'])} | {'' if w is None else '%.3f' % w['w0']} |")
    L.append("")
    # twins in distribution
    a, b = summ("id_twins_ship"), summ("id_ffno3_ship")
    L += ["## In distribution, twins (band refitted the shipping way for both): F-FNO-only twins − the five twins of record", "",
          "| group | control Δ (n+) | rel / tke / mvpe / sps | frozen Δ (n+) |", "|---|---|---|---|"]
    for g in ("all8", "a15_20"):
        c, f = cross(a, b, g, "control"), cross(a, b, g, "frozen")
        L.append(f"| {g} | {fmt(c)} | {subs(c)} | {fmt(f)} |")
    L.append("")
    print("\n".join(L))


if __name__ == "__main__":
    main()
