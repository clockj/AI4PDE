"""Chunk 5: the complete T4 verdict (and T4v / T4b when read), each against the mirror control of its regime. No GPU.

    python ood_t4.py [RUNS_DIR]

T4 criterion (PLAN.md): mean(relo, rehi) >= +0.10; A20, A10 >= -0.02; in distribution >= -0.03 (the three T4 twins + the
two ViT twins of record against the five twins of record, both bands refitted the shipping way, 8 VAL_TRAJ trajectories).
Headline: the OOD mean over A20, A10, relo, rehi.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs"


def S(run, idread=False):
    p = os.path.join(RUNS, run, "idread" if idread else "", "summary.json")
    return json.load(open(p))["sets"]["plain"] if os.path.exists(p) else None


def cross(a, b, g, v):
    A, B = S(a), S(b)
    if not A or not B or g not in A or g not in B or v not in A[g] or v not in B[g]:
        return None
    x, y = A[g][v], B[g][v]
    per = [y["traj"][t]["full"] - x["traj"][t]["full"] for t in x["traj"]]
    return dict(d=y["full"]["composite"] - x["full"]["composite"], npos=int(sum(p > 0 for p in per)), n=len(per),
                sub={k: y["full"][k] - x["full"][k] for k in ("rel", "tke", "mvpe", "sps", "C")},
                f3=y["first3"]["composite"] - x["first3"]["composite"], f14=y["first14"]["composite"] - x["first14"]["composite"],
                lev=(x["full"], y["full"]))


def row(lab, c, f):
    if c is None:
        return f"| {lab} | pending | | | |"
    s = c["sub"]
    return (f"| {lab} | **{c['d']:+.4f}** ({c['npos']}/{c['n']}) | {s['rel']:+.3f} / {s['tke']:+.3f} / {s['mvpe']:+.3f} / "
            f"{s['sps']:+.3f} / {s['C']:+.3f} | {c['f3']:+.4f} / {c['f14']:+.4f} | "
            f"{'' if f is None else '%+.4f (%d/%d)' % (f['d'], f['npos'], f['n'])} |")


def main():
    L = ["## T4 against its criterion", "",
         "| regime | T4 − mirror, control (n+) | rel / tke / mvpe / sps / C | first 3 / 14 | frozen (Track 1) |", "|---|---|---|---|---|"]
    reads = {}
    for lab, a, b, g in (("A20", "a20_mirror", "a20_t4", "A20"), ("A10", "a10_mirror", "a10_t4", "A10"),
                         ("relo", "re_mirror", "re_t4", "relo"), ("rehi", "re_mirror", "re_t4", "rehi"),
                         ("rein", "re_mirror", "re_t4", "rein")):
        c, f = cross(a, b, g, "control"), cross(a, b, g, "frozen")
        reads[lab] = c
        L.append(row(lab, c, f))
    for g, lab in (("all8", "in distribution, twins, all 8"), ("a15_20", "in distribution, twins, AoA 15-20")):
        c, f = cross("id_twins_ship", "id_t4_ship", g, "control"), cross("id_twins_ship", "id_t4_ship", g, "frozen")
        reads[g] = c
        L.append(row(lab, c, f))
    L.append("")
    have = {k: v for k, v in reads.items() if v is not None}
    terms = []
    if "relo" in have and "rehi" in have:
        m = (have["relo"]["d"] + have["rehi"]["d"]) / 2
        terms.append(("mean(relo, rehi) ≥ +0.10", f"{m:+.4f}", m >= 0.10))
    for r in ("A20", "A10"):
        terms.append((f"{r} ≥ −0.02", "pending" if r not in have else f"{have[r]['d']:+.4f}", None if r not in have else have[r]["d"] >= -0.02))
    terms.append(("in distribution (twins, all 8) ≥ −0.03", "pending" if "all8" not in have else f"{have['all8']['d']:+.4f}",
                  None if "all8" not in have else have["all8"]["d"] >= -0.03))
    L += ["| criterion term | read | verdict |", "|---|---|---|"]
    for t, r, ok in terms:
        L.append(f"| {t} | {r} | {'pending' if ok is None else ('pass' if ok else '**fail**')} |")
    ood = [have[r]["d"] for r in ("A20", "A10", "relo", "rehi") if r in have]
    L += ["", f"**Headline: OOD mean over {len(ood)} of 4 regimes = {np.mean(ood):+.4f}**" if ood else "", ""]
    if all(ok is not None for _, _, ok in terms):
        L.append(f"**T4: {'PASS' if all(ok for _, _, ok in terms) else 'FAIL'}**")
    # twins levels
    for g in ("all8", "a15_20"):
        c = reads.get(g)
        if c:
            a, b = c["lev"]
            L += ["", f"Twins, {g}: five twins of record rel {a['rel']:.3f} tke {a['tke']:.3f} mvpe {a['mvpe']:.3f} sps {a['sps']:.3f} "
                  f"C {a['C']:.3f} composite {a['composite']:.4f}; T4 twins rel {b['rel']:.3f} tke {b['tke']:.3f} mvpe {b['mvpe']:.3f} "
                  f"sps {b['sps']:.3f} C {b['C']:.3f} composite {b['composite']:.4f}"]
    # T4v / T4b
    for arm, run in (("T4v", "re_t4v"), ("T4b", "re_t4b")):
        if not S(run):
            continue
        L += ["", f"## {arm} ensemble at RE", "", "| group | vs mirror control (n+) | vs T4 with control ViTs (n+) | frozen vs mirror |", "|---|---|---|---|"]
        vals_m, vals_t = [], []
        for g in ("relo", "rehi", "rein"):
            cm, ct, fm = cross("re_mirror", run, g, "control"), cross("re_t4", run, g, "control"), cross("re_mirror", run, g, "frozen")
            if g != "rein":
                vals_m.append(cm["d"]); vals_t.append(ct["d"])
            L.append(f"| {g} | {cm['d']:+.4f} ({cm['npos']}/{cm['n']}) | {ct['d']:+.4f} ({ct['npos']}/{ct['n']}) | {fm['d']:+.4f} |")
        L.append(f"| mean(relo, rehi) | {np.mean(vals_m):+.4f} | {np.mean(vals_t):+.4f} | |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
