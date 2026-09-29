"""Chunk 5 item h: the decision table for control / T4 F-FNO only / T4 all, the recipe comparison, and the in-distribution
twins deltas translated with the project's correction rules (evaluate-change skill, section 4). No GPU.

    python ood_decision.py [RUNS_DIR]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ood_score  # noqa: E402

W = ood_score.W0          # rel, tke, mvpe, time, sps
COLS = [("twins, all 8", "id", "all8"), ("twins, AoA 15-20", "id", "a15_20"), ("A20", "a20", "A20"), ("A10", "a10", "A10"),
        ("relo", "re", "relo"), ("rehi", "re", "rehi"), ("rein", "re", "rein")]
RUN = {"control": {"id": "id_twins_ship", "a20": "a20_mirror", "a10": "a10_mirror", "re": "re_mirror"},
       "T4 F-FNO only": {"id": "id_t4_ship", "a20": "a20_t4", "a10": "a10_t4", "re": "re_t4"},
       "T4 all": {"id": "id_t4all_ship", "a20": "a20_t4v", "a10": "a10_t4v", "re": "re_t4v"}}
OOD = ("A20", "A10", "relo", "rehi")
SPREAD = {"relo": 0.034, "rehi": 0.011, "rein": 0.010, "twins, all 8": 0.012, "twins, AoA 15-20": 0.012}   # jackknife (REID for ID)


def get(run, g, v):
    p = os.path.join(RUNS, run, "summary.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["sets"]["plain"]
    return s.get(g, {}).get(v)


def diff(a, b):
    if a is None or b is None:
        return None
    per = [b["traj"][t]["full"] - a["traj"][t]["full"] for t in a["traj"]]
    return dict(d=b["full"]["composite"] - a["full"]["composite"], npos=int(sum(p > 0 for p in per)), n=len(per),
                sub={k: b["full"][k] - a["full"][k] for k in ("rel", "tke", "mvpe", "sps", "C", "abar")})


def main():
    L = []
    for v, track in (("control", "stream (Track 2)"), ("frozen", "frozen (Track 1's case)")):
        L += [f"### Decision table, {track}: Δ composite to control (trajectories positive); control row = absolute", "",
              "| row | " + " | ".join(c[0] for c in COLS) + " | OOD mean |", "|---|" + "---|" * (len(COLS) + 1)]
        ctl = {c[0]: get(RUN["control"][c[1]], c[2], v) for c in COLS}
        L.append("| control (absolute) | " + " | ".join("—" if ctl[c[0]] is None else "%.4f" % ctl[c[0]]["full"]["composite"] for c in COLS)
                 + " | %.4f |" % np.mean([ctl[k]["full"]["composite"] for k in OOD if ctl[k]]))
        for row in ("T4 F-FNO only", "T4 all"):
            cells, ood = [], []
            for lab, key, g in COLS:
                d = diff(ctl[lab], get(RUN[row][key], g, v))
                cells.append("pending" if d is None else "%+.4f (%d/%d)" % (d["d"], d["npos"], d["n"]))
                if d and lab in OOD:
                    ood.append(d["d"])
            L.append(f"| {row} | " + " | ".join(cells) + f" | {'%+.4f' % np.mean(ood) if len(ood) == 4 else 'pending'} |")
        L.append("")
    # recipe comparison
    L += ["### T4 all − T4 F-FNO only (stream), against the control-to-control reseeding spread", "",
          "| column | Δ (n+) | rel / tke / mvpe / sps / C | reseeding spread (max) | outside the spread |", "|---|---|---|---|---|"]
    for lab, key, g in COLS:
        d = diff(get(RUN["T4 F-FNO only"][key], g, "control"), get(RUN["T4 all"][key], g, "control"))
        if d is None:
            L.append(f"| {lab} | pending | | | |"); continue
        sp = SPREAD.get(lab)
        s = d["sub"]
        L.append(f"| {lab} | {d['d']:+.4f} ({d['npos']}/{d['n']}) | {s['rel']:+.3f} / {s['tke']:+.3f} / {s['mvpe']:+.3f} / {s['sps']:+.3f} / "
                 f"{s['C']:+.3f} | {'not measured' if sp is None else '%.3f' % sp} | "
                 f"{'n/a' if sp is None else ('yes, ' + ('gain' if d['d'] > 0 else 'loss')) if abs(d['d']) > sp else 'no'} |")
    L.append("")
    # development-board translation from the in-distribution twins (stream)
    L += ["### Development board, translated from the in-distribution twins (stream; evaluate-change skill, section 4)", "",
          "Rules: an error cut read locally is halved; containment C is never banked, so the part of Δsps that comes from C is "
          "set to 0 and only the accuracy-factor part (Δā) is kept, halved like any error cut. Bracket: low = the accuracy "
          "parts at a quarter, high = the local read in full (C included). Composite = W0 without time; time does not move "
          "(same members, same architectures, same package code).", "",
          "| recipe | slice | local Δ rel / tke / mvpe / sps (C part of sps) | expected Δ rel / tke / mvpe / sps | expected Δ composite [low, high] |",
          "|---|---|---|---|---|"]
    for row in ("T4 F-FNO only", "T4 all"):
        for lab, g in (("all 8", "all8"), ("AoA 15-20", "a15_20")):
            a, b = get(RUN["control"]["id"], g, "control"), get(RUN[row]["id"], g, "control")
            if a is None or b is None:
                L.append(f"| {row} | {lab} | pending | | |"); continue
            fa, fb = a["full"], b["full"]
            d = {k: fb[k] - fa[k] for k in ("rel", "tke", "mvpe", "sps")}
            # split Δsps (= 100 abar C in first order) into the accuracy part and the containment part
            s0 = fa["sps"]
            acc = s0 * (fb["abar"] / fa["abar"] - 1.0)
            cpart = d["sps"] - acc
            exp = dict(rel=0.5 * d["rel"], tke=0.5 * d["tke"], mvpe=0.5 * d["mvpe"], sps=0.5 * acc)
            lo = dict(rel=0.25 * d["rel"], tke=0.25 * d["tke"], mvpe=0.25 * d["mvpe"], sps=0.25 * acc)
            comp = lambda x: float(W[0] * x["rel"] + W[1] * x["tke"] + W[2] * x["mvpe"] + W[4] * x["sps"])  # noqa: E731
            L.append(f"| {row} | {lab} | {d['rel']:+.3f} / {d['tke']:+.3f} / {d['mvpe']:+.3f} / {d['sps']:+.3f} ({cpart:+.3f}) | "
                     f"{exp['rel']:+.3f} / {exp['tke']:+.3f} / {exp['mvpe']:+.3f} / {exp['sps']:+.3f} | "
                     f"**{comp(exp):+.4f}** [{comp(lo):+.4f}, {comp(d):+.4f}] |")
    L.append("")
    print("\n".join(L))


if __name__ == "__main__":
    main()
