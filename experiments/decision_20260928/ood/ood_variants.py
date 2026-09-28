"""Knob variants of ONE built Track-2 package, on the real code path.

    python ood_variants.py PKGDIR            # PKGDIR/record = the extracted archive build_final.sh produced

Each variant is a copy of `record` whose baked module-level constants in submission.py are rewritten with the SAME
patterns `Track_1/submit_fno/build_round13.sh` uses (`^NAME = [^ ]*`, `^R1 = .*`), exactly one hit each. Members and
band.npz are symlinked, `_fno.py` is copied. Every variant's diff against `record` is written next to it and checked to
touch only the intended constant lines.

    record      what build_final.sh ships (the package of record's constants, PREV_SPACE "input")
    frozen      every stream knob off (R1 False, D1c / VARCAL / FAMW 0, TSM 1, BLK 0): the frozen ensemble
    base        asymmetric R1 only: D1C_N0 0, VC_N0 0, R1_TSM 1, R1_BLK_N0 0 (the sprint's "asym R1" baseline)
    control     record + PREV_SPACE "target"
    prev_input  record (PREV_SPACE "input"), named for the report
    statsmask   control + STATS_MASK True
    famw        control + FW_N0 20.0 (design C's Decision-phase value, design_c.md)
    c_<knob>_<value>   control with ONE stream constant moved (CSET: half / double the shipped value, and D1C_N0, VC_N0 10 / 100)
Every variant except prev_input has PREV_SPACE "target" (G0: target on both sides).
"""
from __future__ import annotations

import difflib
import os
import re
import shutil
import sys

# (name, regex exactly as build_round13.sh's sed, replacement)
def _pat(name, value):
    rx = r"^R1 = .*" if name == "R1" else rf"^{name} = [^ \n]*"      # sed works per line: [^ ]* never crosses one
    return name, re.compile(rx, re.M), f"{name} = {value}"


TARGET = [_pat("PREV_SPACE", '"target"')]
OFF_STREAM = [_pat("D1C_N0", "0.0"), _pat("VC_N0", "0.0"), _pat("FW_N0", "0.0"),
              _pat("R1_TSM", "1"), _pat("R1_BLK_N0", "0.0")]
# ARM I5 dry run: one stream constant at a time around the shipped point (build_final.sh), on top of `control`.
SHIPPED = {"R1_N0": 2.0, "R1_BLK_N0": 20.0, "D1C_N0": 30.0, "D1C_EB_N0": 1.0, "D1C_LEAD_N0": 5.0, "VC_N0": 30.0}
EXTRA = {"D1C_N0": (10.0, 100.0), "VC_N0": (10.0, 100.0)}


def _fmt(v):
    return ("%g" % v)


CSET = []
for _k, _v in SHIPPED.items():
    for _x in sorted({_v / 2, _v * 2, *EXTRA.get(_k, ())}):
        CSET.append((f"c_{_k.lower()}_{_fmt(_x)}", _k, _x))

VARIANTS = {
    "frozen": [_pat("R1", "False")] + OFF_STREAM + TARGET,
    "base": OFF_STREAM + TARGET,
    "control": TARGET,
    "prev_input": [_pat("PREV_SPACE", '"input"')],
    "statsmask": TARGET + [_pat("STATS_MASK", "True")],
    "famw": TARGET + [_pat("FW_N0", "20.0")],
}
for _name, _k, _x in CSET:
    VARIANTS[_name] = TARGET + [_pat(_k, repr(float(_x)))]


def make(pkg, name, edits):
    src = os.path.join(pkg, "record")
    dst = os.path.join(pkg, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in sorted(os.listdir(src)):
        if f == "submission.py":
            continue
        if f.endswith(".py"):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        else:
            os.symlink(os.path.realpath(os.path.join(src, f)), os.path.join(dst, f))
    s0 = open(os.path.join(src, "submission.py")).read()
    s = s0
    for nm, rx, rep in edits:
        s, n = rx.subn(rep, s)
        if n != 1:
            raise SystemExit(f"{name}: pattern for {nm} matched {n} times (expected 1)")
    if len(s.splitlines()) != len(s0.splitlines()):
        raise SystemExit(f"{name}: an edit changed the line count")
    open(os.path.join(dst, "submission.py"), "w").write(s)
    diff = list(difflib.unified_diff(s0.splitlines(), s.splitlines(), "record/submission.py",
                                     f"{name}/submission.py", n=0, lineterm=""))
    open(os.path.join(dst, "VARIANT.diff"), "w").write("\n".join(diff) + "\n")
    changed = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    names = {nm for nm, _, _ in edits}
    stray = [l for l in changed if l.split(" = ")[0] not in names]
    if stray:
        raise SystemExit(f"{name}: unexpected changed lines {stray}")
    return [l.split("#")[0].strip() for l in changed]


def main():
    if sys.argv[1:] == ["--list-cset"]:
        print(" ".join(n for n, _, _ in CSET))
        return
    pkg = sys.argv[1]
    if not os.path.isfile(os.path.join(pkg, "record", "submission.py")):
        raise SystemExit(f"{pkg}/record/submission.py missing: extract the built track2 archive there first")
    # extra variants from OOD_VARIANT_SPECS="name:KNOB=VALUE,KNOB=VALUE name2:..." (on top of control: PREV_SPACE target)
    for spec in os.environ.get("OOD_VARIANT_SPECS", "").split():
        name, _, kv = spec.partition(":")
        VARIANTS[name] = TARGET + [_pat(k, v) for k, v in (x.split("=", 1) for x in kv.split(",") if x)]
    only = sys.argv[2:] or list(VARIANTS)
    for name in only:
        ch = make(pkg, name, VARIANTS[name])
        print(f"{name:11s} " + ("; ".join(ch) if ch else "(no change: identical to record)"))


if __name__ == "__main__":
    main()
