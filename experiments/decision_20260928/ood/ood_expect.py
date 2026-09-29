"""Chunk 6: the expectation table for the owner. In distribution, translated with the evaluate-change skill's section-4
rules; out of distribution, the measured decision-table deltas unchanged. No GPU.

    python ood_expect.py [RUNS_DIR]

In distribution = the from-scratch twins (stream), bands refitted the shipping way, 8 VAL_TRAJ trajectories:
  T4 all          id_repro_ship  − id_twins_ship     (the candidate's own from-scratch twins against the twins of record)
  T4 F-FNO only   id_cand2_ship  − id_twins_ship     (three from-scratch F-FNO twins + the two ViT twins of record)
  PREV_SPACE      id_twins: control (target) − prev_input (the uploaded package's "input"), band of record
Rules: a local error cut is halved; the containment part of Δsps (Δsps − sps·Δā/ā) is never banked; the accuracy part of
Δsps is halved; time unchanged; composite = W0 without time. Bracket: low = accuracy parts at a quarter (containment 0),
high = the local read in full.
Out of distribution = the decision table (ood_decision.py, chunk 5), each recipe's ensemble − the mirror control, stream;
PREV_SPACE: control − prev_input on the mirror ensembles (chunk 4).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ood_score  # noqa: E402

W = ood_score.W0


def S(run):
    return json.load(open(os.path.join(RUNS, run, "summary.json")))["sets"]["plain"]


def translate(fa, fb):
    d = {k: fb[k] - fa[k] for k in ("rel", "tke", "mvpe", "sps")}
    acc = fa["sps"] * (fb["abar"] / fa["abar"] - 1.0)
    comp = lambda x: float(W[0] * x["rel"] + W[1] * x["tke"] + W[2] * x["mvpe"] + W[4] * x["sps"])  # noqa: E731
    exp = dict(rel=0.5 * d["rel"], tke=0.5 * d["tke"], mvpe=0.5 * d["mvpe"], sps=0.5 * acc)
    lo = dict(rel=0.25 * d["rel"], tke=0.25 * d["tke"], mvpe=0.25 * d["mvpe"], sps=0.25 * acc)
    return d, d["sps"] - acc, exp, comp(exp), comp(lo), comp(d)


def ood(row):
    out = {}
    for lab, reg, g in (("A20", "a20", "A20"), ("A10", "a10", "A10"), ("relo", "re", "relo"), ("rehi", "re", "rehi"), ("rein", "re", "rein")):
        a = S(f"{reg}_mirror")[g]
        if row == "PREV_SPACE":
            x, y = a["prev_input"]["full"], a["control"]["full"]
        else:
            b = S(f"{reg}_{'t4' if row == 'T4 F-FNO only' else 't4v'}")[g]
            x, y = a["control"]["full"], b["control"]["full"]
        out[lab] = y["composite"] - x["composite"]
    out["OOD mean"] = float(np.mean([out[k] for k in ("A20", "A10", "relo", "rehi")]))
    return out


def main():
    rows = [("T4 all", "id_twins_ship", "id_repro_ship", "control", "control"),
            ("T4 F-FNO only", "id_twins_ship", "id_cand2_ship", "control", "control"),
            ("PREV_SPACE target vs the uploaded input", "id_twins", "id_twins", "prev_input", "control")]
    L = ["| change | slice | local Δ rel / tke / mvpe / sps (containment part) | expected Δ rel / tke / mvpe / sps | "
         "expected Δ composite [low, high] |", "|---|---|---|---|---|"]
    for lab, a, b, va, vb in rows:
        for g, gl in (("all8", "all 8"), ("a15_20", "AoA 15-20")):
            fa, fb = S(a)[g][va]["full"], S(b)[g][vb]["full"]
            d, cp, e, ce, cl, ch = translate(fa, fb)
            L.append(f"| {lab} | {gl} | {d['rel']:+.3f} / {d['tke']:+.3f} / {d['mvpe']:+.3f} / {d['sps']:+.3f} ({cp:+.3f}) | "
                     f"{e['rel']:+.3f} / {e['tke']:+.3f} / {e['mvpe']:+.3f} / {e['sps']:+.3f} | **{ce:+.4f}** [{cl:+.4f}, {ch:+.4f}] |")
    L += ["", "Out of distribution, measured (stream, Δ composite to the mirror control; PREV_SPACE: control − prev_input):", "",
          "| change | A20 | A10 | relo | rehi | rein | OOD mean |", "|---|---|---|---|---|---|---|"]
    for lab in ("T4 all", "T4 F-FNO only", "PREV_SPACE"):
        o = ood(lab)
        L.append(f"| {lab} | " + " | ".join(f"{o[k]:+.4f}" for k in ("A20", "A10", "relo", "rehi", "rein", "OOD mean")) + " |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
