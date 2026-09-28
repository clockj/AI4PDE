"""Arm I5: stream constants chosen leave-one-regime-out (post hoc on the c_* reads of the mirror runs).

    python ood_i5.py pick REG=RUN:GROUP ...            # e.g. A20=runs/a20_mirror:A20 A10=runs/a10_mirror:A10 \
                                                          #      relo=runs/re_mirror:relo rehi=runs/re_mirror:rehi
    python ood_i5.py read REG=RUN:GROUP ... --id RUN:GROUP

pick: for each of the four rotations (one regime left out) and each constant, the value (shipped, half, double, and the
  extra D1C_N0 / VC_N0 points) with the largest MEAN pooled composite delta to control over the three regimes kept;
  prints the picks, the additive prediction of their gain on the left-out regime (sum of the one-at-a-time deltas) and the
  OOD_VARIANT_SPECS string of the combined variant `i5_lo_<left-out>` to be streamed in the left-out regime's run and in
  the in-distribution run (the combined read is the evidence; the additive sum is only a prediction).
read: the combined variants' delta to control on the left-out regime and in distribution, with counts (after streaming).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ood_variants import CSET, SHIPPED  # noqa: E402


def deltas(spec):
    reg, _, rg = spec.partition("=")
    run, _, group = rg.partition(":")
    S = json.load(open(os.path.join(run, "summary.json")))["sets"]["plain"][group]
    return reg, run, group, S


def pick(a):
    R = [deltas(s) for s in a.regs]
    regs = [r[0] for r in R]
    D = {r[0]: {v: r[3][v]["delta"]["control"] for v in r[3] if v.startswith("c_")} for r in R}
    out = {}
    for lo in regs:
        keep = [r for r in regs if r != lo]
        picks, pred, spec = {}, 0.0, []
        for knob, shipped in SHIPPED.items():
            cands = [(None, shipped, 0.0)] + [(n, x, float(np.mean([D[r][n]["full"] for r in keep])))
                                              for n, k, x in CSET if k == knob and all(n in D[r] for r in keep)]
            best = max(cands, key=lambda c: c[2])
            picks[knob] = best[1]
            if best[0] is not None:
                pred += D[lo][best[0]]["full"] if best[0] in D[lo] else float("nan")
                spec.append(f"{knob}={float(best[1])!r}")
        name = f"i5_lo_{lo.lower()}"
        out[lo] = dict(picks=picks, additive_prediction_on_left_out=pred, variant=name,
                       spec=f"{name}:{','.join(spec)}" if spec else None)
        print(f"left out {lo:5s}: " + ", ".join(f"{k} {v:g}" for k, v in picks.items())
              + f"   additive prediction on {lo}: {pred:+.4f}   spec {out[lo]['spec']}")
    specs = " ".join(o["spec"] for o in out.values() if o["spec"])
    print(f"\nOOD_VARIANT_SPECS=\"{specs}\"")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def read(a):
    picks = json.load(open(a.picks))
    rows = ["| left out | picks | Δ on left-out (n+) | first 3 / 14 | in distribution Δ (n+) | additive prediction |",
            "|---|---|---|---|---|---|"]
    idr, _, idg = a.id.partition(":")
    SI = json.load(open(os.path.join(idr, "summary.json")))["sets"]["plain"][idg]
    for s in a.regs:
        reg, run, group, S = deltas(s)
        p = picks[reg]
        v = p["variant"]
        if p["spec"] is None:
            rows.append(f"| {reg} | shipped point (no change) | 0 | | 0 | 0 |"); continue
        d = S.get(v, {}).get("delta", {}).get("control")
        di = SI.get(v, {}).get("delta", {}).get("control")
        f = lambda x: "n/a" if x is None else f"{x['full']:+.4f} ({x['npos']}/{x['n']})"  # noqa: E731
        early = "" if d is None else "%+.4f / %+.4f" % (d["first3"], d["first14"])
        rows.append(f"| {reg} | {', '.join(f'{k} {x:g}' for k, x in p['picks'].items())} | {f(d)} | {early} | "
                    f"{f(di)} | {p['additive_prediction_on_left_out']:+.4f} |")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "read"])
    ap.add_argument("regs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--picks", default=None)
    ap.add_argument("--id", default=None)
    a = ap.parse_args()
    pick(a) if a.cmd == "pick" else read(a)


if __name__ == "__main__":
    main()
