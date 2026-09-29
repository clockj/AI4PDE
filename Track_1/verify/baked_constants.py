"""Build gate: do the two archives carry the constants the build was asked to bake?

    python Track_1/verify/baked_constants.py --t1 track1.zip --t2 track2.zip --alpha 0.80 --band band.npz \
        --tree-t1 <submission.py> --tree-fno <_fno.py> --tree-t2 <submission.py>

Called by `build_final.sh` after the gates, in the environment `build_round13.sh` ran in (the RPDE_* knobs). Why it
exists: two builds sharing one worktree once restored each other's files mid-build, and an archive of the UNSUBSTITUTED
defaults still ended `BUILD_FINAL OK` (experiments/decision_20260928/results/ood_instrument.md, "Build isolation"). The
gates score a package; nothing checked that it was the package asked for.

For every constant `build_round13.sh` can write, the expected value is what it writes: the knob's value when the knob
is set, otherwise the value the script falls back to (ALPHA, FUSE, AMP, BLEND_BETA, SPEC_PATH, HOST_WRITE are always
written) or the tree's own default (the `[ -n ... ] && sed` knobs; pass the tree's files as they were BEFORE the
build). Values are compared as Python literals read from the module-level assignment of the extracted file. Also checks
that band.npz in both archives is the band asked for, byte for byte. Prints the table; exits 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import sys
import tempfile
import zipfile

# knob -> (constant, files, kind). kind "lit": the knob's value is a Python literal; "str": a bare word the build quotes.
# Files: t1s / t1f = Track 1 submission.py / _fno.py, t2s / t2f = the same for Track 2 (t2f is the same _fno.py).
KNOBS = [
    ("RPDE_SHIP_JOINT_GAIN", "JOINT_GAIN", ("t1f", "t1s", "t2s", "t2f"), "lit"),
    ("RPDE_SHIP_DIRECT_PRED", "D2_DIRECT_PRED", ("t1f", "t2f"), "lit"),
    ("RPDE_SHIP_EBAND", "EBAND", ("t1s",), "lit"),
    ("RPDE_SHIP_WARM_TAILS", "WARM_TAILS", ("t1s",), "lit"),
    ("RPDE_SHIP_PIN_OUT", "PIN_OUT", ("t1s",), "lit"),
    ("RPDE_SHIP_BATCH_CAP", "BATCH_CAP", ("t1s",), "lit"),
    ("RPDE_SHIP_SIGPOW", "SIGPOW", ("t2s",), "lit"),
    ("RPDE_SHIP_GRAPH_TOL", "GRAPH_TOL", ("t2s",), "lit"),
    ("RPDE_SHIP_ARENA_STEPS", "ARENA_STEPS", ("t2s",), "lit"),
    ("RPDE_SHIP_NUMPY_OUT", "NUMPY_OUT", ("t2s",), "lit"),
    ("RPDE_SHIP_R1", "R1", ("t2s",), "lit"),
    ("RPDE_SHIP_R1_ASYM", "R1_ASYM", ("t2s",), "lit"),
    ("RPDE_SHIP_R1_TSM", "R1_TSM", ("t2s",), "lit"),
    ("RPDE_SHIP_R1_BLK_N0", "R1_BLK_N0", ("t2s",), "lit"),
    ("RPDE_SHIP_R1_BLK", "R1_BLK", ("t2s",), "lit"),
    ("RPDE_SHIP_D1C_N0", "D1C_N0", ("t2s",), "lit"),
    ("RPDE_SHIP_D1C_EB", "D1C_EB", ("t2s",), "lit"),
    ("RPDE_SHIP_D1C_EB_N0", "D1C_EB_N0", ("t2s",), "lit"),
    ("RPDE_SHIP_D1C_LEAD_N0", "D1C_LEAD_N0", ("t2s",), "lit"),
    ("RPDE_SHIP_VC_N0", "VC_N0", ("t2s",), "lit"),
    ("RPDE_SHIP_FW_N0", "FW_N0", ("t2s",), "lit"),
    ("RPDE_SHIP_STATS_MASK", "STATS_MASK", ("t2s",), "lit"),
    ("RPDE_SHIP_PREV_SPACE", "PREV_SPACE", ("t2s",), "str"),
]
# Always written by build_round13.sh, with these fallbacks when the knob is unset.
ALWAYS = [
    ("RPDE_FUSE", "FUSE", ("t1s", "t2s"), "lit", "True"),
    ("RPDE_SHIP_AMP", "AMP", ("t1s", "t2s"), "lit", "None"),
    ("RPDE_BETA", "BLEND_BETA", ("t1s",), "lit", "0.0"),
    ("RPDE_SPEC_PATH", "SPEC_PATH", ("t1s", "t2s"), "str", ""),
    ("RPDE_HOST_WRITE", "HOST_WRITE", ("t1s",), "str", ""),
]
# Knobs removed on 2026-09-28: build_round13.sh refuses any value but the old default, and there is nothing to bake.
REMOVED = ("RPDE_SHIP_SMALL_GRAPH", "RPDE_SHIP_SMALL_GRAPH_LAZY")

MISSING = object()


def consts(path: str) -> dict:
    out = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--band", required=True)
    ap.add_argument("--tree-t1", required=True, help="Track_1/submit_fno/submission.py as it was before the build")
    ap.add_argument("--tree-fno", required=True, help="Track_1/submit_fno/_fno.py as it was before the build")
    ap.add_argument("--tree-t2", required=True, help="Track_2/submit/submission.py as it was before the build")
    a = ap.parse_args()
    env = os.environ

    tmp = tempfile.mkdtemp()
    got, band_md5 = {}, {}
    for tr, z in (("t1", a.t1), ("t2", a.t2)):
        zf = zipfile.ZipFile(z)
        names = {os.path.basename(n): n for n in zf.namelist()}
        for short, base in ((tr + "s", "submission.py"), (tr + "f", "_fno.py")):
            p = os.path.join(tmp, short + ".py")
            open(p, "wb").write(zf.read(names[base]))
            got[short] = consts(p)
        band_md5[tr] = hashlib.md5(zf.read(names["band.npz"])).hexdigest()
    tree = {"t1s": consts(a.tree_t1), "t1f": consts(a.tree_fno), "t2s": consts(a.tree_t2), "t2f": consts(a.tree_fno)}

    rows, bad = [], 0

    def row(name, files, exp, src):
        nonlocal bad
        for f in files:
            v = got[f].get(name, MISSING)
            ok = (v is not MISSING and v == exp and type(v) is type(exp)) or (v is MISSING and exp is MISSING)
            bad += not ok
            rows.append((name, f, "(absent)" if exp is MISSING else repr(exp),
                         "(absent)" if v is MISSING else repr(v), src, "OK" if ok else "MISMATCH"))

    def parse(raw, kind):
        return raw if kind == "str" else ast.literal_eval(raw)

    row("ALPHA", ("t1s", "t2s"), ast.literal_eval(a.alpha), "argument")
    for var, name, files, kind, fb in ALWAYS:
        raw = env.get(var)
        row(name, files, parse(raw if raw is not None else fb, kind), f"{var}" if raw is not None else "fallback")
    for var, name, files, kind in KNOBS:
        raw = env.get(var, "")
        if raw:
            row(name, files, parse(raw, kind), var)
        else:
            for f in files:
                row(name, (f,), tree[f].get(name, MISSING), "tree default")
    ffno = env.get("RPDE_FFNO_PATH", "")
    if ffno:
        p = ffno.split()
        for f, v in (("t1s", p[0]), ("t2s", p[1] if len(p) > 1 else p[0])):
            row("FFNO_PATH", (f,), None if v == "none" else v, "RPDE_FFNO_PATH")
    else:
        for f in ("t1s", "t2s"):
            row("FFNO_PATH", (f,), tree[f].get("FFNO_PATH", MISSING), "tree default")

    want = hashlib.md5(open(a.band, "rb").read()).hexdigest()
    for tr in ("t1", "t2"):
        ok = band_md5[tr] == want
        bad += not ok
        rows.append(("band.npz md5", tr, want[:12], band_md5[tr][:12], "RPDE_BAND", "OK" if ok else "MISMATCH"))

    w = [max(len(str(r[i])) for r in rows + [("constant", "file", "expected", "found", "source", "")]) for i in range(6)]
    w = [min(x, 60) for x in w]
    fmt = "  ".join("{:<%d}" % x for x in w)
    print("BAKED CONSTANTS (the archives against what the build was asked for)")
    print(fmt.format("constant", "file", "expected", "found", "source", ""))
    for r in rows:
        print(fmt.format(*[str(x)[:60] for x in r]))
    for var in REMOVED:
        if env.get(var):
            print(f"  {var}={env[var]}: removed knob (2026-09-28), refused by build_round13.sh unless the old default; not baked")
    print(f"BAKED CONSTANTS {'OK' if not bad else 'FAILED'}: {len(rows)} checks, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
