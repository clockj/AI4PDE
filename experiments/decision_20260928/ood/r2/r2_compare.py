"""Gate R2: compare the cards and the forced fallbacks from r2_run.sh outputs. No GPU.

    python r2_compare.py [R2ROOT]

Per archive: the load-time choices each card made (from the QA harness LOAD line and the package's own load line), the gates
(validate_t2 --adaptive; t1_check contracts and permutation stability), the V100 - A100 difference of pred / lower / upper on
the two-trajectory stream (stream_eval) and on the whole val_unseen stream (harness mode graph), Track 1's whole-split
output, and on the A100 each forced fallback (eager, fp32, eager+fp32, no pinned arena) against the unforced run.
Reported: max |diff|; max relative element difference (|diff| / max(|ref|, 1e-3 of the field's rms)); and the largest
per-window relative L2 difference ||P' - P|| / ||P||, the measure the package's own fp16 check accepts below 2e-2.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/r2"
NAMES = [("record", "uploaded T2_MRGP2 / track1_k27fin_a080"), ("cand_all", "candidate T4 all (from scratch)"),
         ("cand_ffno", "candidate T4 F-FNO only")]


def load_npz(p):
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k][..., :2].astype(np.float64) for k in ("pred", "lower", "upper") if k in z.files}


def diff(a, b):
    if a is None or b is None:
        return None
    out = {}
    for k in ("pred", "lower", "upper"):
        x, y = a[k], b[k]
        d = np.abs(x - y)
        scale = 1e-3 * np.sqrt((x ** 2).mean())
        n = len(x)
        wl2 = np.sqrt(((x - y) ** 2).reshape(n, -1).sum(1)) / np.maximum(np.sqrt((x ** 2).reshape(n, -1).sum(1)), 1e-12)
        out[k] = dict(max_abs=float(d.max()), max_rel=float((d / np.maximum(np.abs(x), scale)).max()), max_window_relL2=float(wl2.max()))
    return out


def fmt(d):
    if d is None:
        return "missing"
    return "; ".join(f"{k} {v['max_abs']:.2e} / {v['max_rel']:.2e} / {v['max_window_relL2']:.2e}" for k, v in d.items())


def load_info(card, name):
    o = os.path.join(ROOT, card, name)
    info = {}
    lg = os.path.join(o, "qa_graph.log")
    if os.path.exists(lg):
        t = open(lg, errors="replace").read().replace("\x00", "")
        m = re.search(r"^LOAD (.*)$", t, re.M)
        if m:
            info["harness"] = m.group(1)[:260]
        m = re.search(r"\[rpde-t2\] load: (.*)$", t, re.M)
        if m:
            s = m.group(1)
            info["t2"] = {k: re.search(k + r" (\S+)", s).group(1) for k in ("gpu '[^']*'", "amp", "graph", "graph_worst", "numpy_out", "members")
                          if re.search(k + r" (\S+)", s)}
            info["t2_gpu"] = re.search(r"gpu '([^']*)'", s).group(1)
    vt = os.path.join(o, "validate_t2.log")
    if os.path.exists(vt):
        t = open(vt, errors="replace").read()
        info["validate_t2"] = "SUBMITTABLE" if re.search(r"^SUBMITTABLE", t, re.M) else "NOT SUBMITTABLE"
    j = os.path.join(o, "t1.json")
    if os.path.exists(j):
        t1 = json.load(open(j))
        info["t1"] = dict(submittable=t1["submittable"], amp=t1["amp_active"],
                          load=next((re.sub(r"^\[rpde-t1\] load: ", "", l) for l in t1["load_line"].splitlines() if "[rpde-t1] load:" in l), "no load line")[:260])
    reds = []
    for m in ("graph", "eager", "fp32", "eagerfp32", "pinfail"):
        p = os.path.join(o, f"qa_{m}.json")
        if os.path.exists(p):
            R = json.load(open(p))
            reds += [f"{m}:{k}" for k, v in R["tests"].items() if str(v.get("status", "")).startswith("RED")]
    info["qa_red"] = reds
    return info


def main():
    L = ["## Gate R2: a different card, and the forced fallbacks", ""]
    for name, lab in NAMES:
        L += [f"### {lab} (`{name}`)", ""]
        for card in ("a100", "v100"):
            i = load_info(card, name)
            L.append(f"* **{card}**: validate_t2 --adaptive **{i.get('validate_t2', 'missing')}**; Track 1 archive check "
                     f"**{'SUBMITTABLE' if i.get('t1', {}).get('submittable') else i.get('t1', 'missing')}** (amp {i.get('t1', {}).get('amp')}); "
                     f"harness load: {i.get('harness', 'missing')}")
            if "t1" in i:
                L.append(f"  Track 1 load line: `{i['t1']['load']}`")
            if i.get("qa_red"):
                L.append(f"  QA harness tests reported RED: {', '.join(i['qa_red'])}")
        L.append("")
        L += ["Differences: max |diff| / max relative element difference / largest per-window relative L2 (package's fp16 bar 2e-2)", "",
              "| comparison | pred; lower; upper |", "|---|---|"]
        a, v = os.path.join(ROOT, "a100", name), os.path.join(ROOT, "v100", name)
        L.append(f"| V100 − A100, Track 2, two-trajectory stream (stream_eval) | {fmt(diff(load_npz(a + '/stream.npz'), load_npz(v + '/stream.npz')))} |")
        L.append(f"| V100 − A100, Track 2, whole val_unseen stream (harness, graph) | {fmt(diff(load_npz(a + '/qa_graph_A.npz'), load_npz(v + '/qa_graph_A.npz')))} |")
        L.append(f"| V100 − A100, Track 1, val_unseen (336 windows) | {fmt(diff(load_npz(a + '/t1.npz'), load_npz(v + '/t1.npz')))} |")
        for m in ("eager", "fp32", "eagerfp32", "pinfail"):
            L.append(f"| A100, forced {m} − unforced, Track 2 whole stream | {fmt(diff(load_npz(a + '/qa_graph_A.npz'), load_npz(a + f'/qa_{m}_A.npz')))} |")
        L.append(f"| A100, Track 1 forced fp32 − unforced | {fmt(diff(load_npz(a + '/t1.npz'), load_npz(a + '/t1_fp32.npz')))} |")
        L.append("")
        # fallback load lines
        for m in ("eager", "fp32", "eagerfp32", "pinfail"):
            lg = os.path.join(a, f"qa_{m}.log")
            if os.path.exists(lg):
                mm = re.search(r"^LOAD (.*)$", open(lg, errors="replace").read(), re.M)
                L.append(f"* A100 forced {m}: {mm.group(1)[:220] if mm else 'no LOAD line'}")
        L.append("")
    print("\n".join(L))


if __name__ == "__main__":
    main()
