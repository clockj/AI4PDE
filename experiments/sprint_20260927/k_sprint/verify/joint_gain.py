"""JOINT_GAIN verifier (K2, 2026-09-27). GPU.

Builds, from an already-built baseline package (zip), three variants that differ ONLY in code:
  base : the zip as built (old code)
  off  : this tree's submission.py / _fno.py with the zip's baked constants re-applied, JOINT_GAIN False
  on   : the same with JOINT_GAIN True
  y0   : `off` with ALPHA 0.0 (compose skipped -> the ensemble mean y0; used as the reference for "closer to y0")
and checks
  (a) off == base bit for bit (prediction, lower, upper)                         [T1 batched call, T2 full stream]
  (b) on: per-pixel k = 0.5 (var_t u + var_t v) (physical) == off's; time mean == off's; ||P_on - y0|| <= ||P_off - y0||
      for every window; lower <= pred <= upper
  (c) T1: permutation of the windows leaves the output unchanged (drift vs rms); the batch-1 lazy-graph path
      (SMALL_GRAPH) also runs the joint compose (on_N1 near on_batched, far from off)
  (d) T2: graph CAPTURED in the on package (load line), stream_eval --causal-check passes on `on`.

Usage (repo root):  python Track_1/verify/joint_gain.py --t1zip <track1 zip> --t2zip <track2 zip> --work /tmp/jg_verify
Each variant runs in its own subprocess (the packages import modules named submission / _fno).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
STD_IN = np.array([0.0968056545, 0.015960684], np.float64)
STD_TGT = np.array([0.0968104079, 0.0159636438], np.float64)
MEAN_TGT = np.array([0.154962569, -0.000517793698], np.float64)
CONST = re.compile(r"^([A-Z][A-Z0-9_]*) = ")


def _set_consts(text, vals):
    out, seen = [], set()
    for ln in text.splitlines(keepends=True):
        m = CONST.match(ln)
        if m and m.group(1) in vals and m.group(1) not in seen:
            seen.add(m.group(1))
            ln = vals[m.group(1)] + "\n"
        out.append(ln)
    assert seen == set(vals), f"constants not found: {set(vals) - seen}"
    return "".join(out)


def rebake(rel: str, built: str, head: str, dst: str, overrides: dict):
    """dst = the BUILT file with this tree's diff (vs BASEREV) for `rel` applied, then `overrides` set.
    Constants the build sed'ed are reset to HEAD values for the patch (context) and restored after it."""
    hl = {m.group(1): ln for ln in open(head).read().splitlines() if (m := CONST.match(ln))}
    bl = {m.group(1): ln for ln in open(built).read().splitlines() if (m := CONST.match(ln))}
    baked = {k: v for k, v in bl.items() if k in hl and hl[k] != v}
    tmp = dst + ".pre"
    open(tmp, "w").write(_set_consts(open(built).read(), {k: hl[k] for k in baked}))
    diff = subprocess.check_output(["git", "-C", ROOT, "diff", BASEREV, "HEAD", "--", rel])   # committed patch only
    subprocess.run(["patch", "-s", "-F0", "-o", dst, tmp], input=diff, check=True)
    os.remove(tmp)
    vals = dict(baked)
    vals.update({k: f"{k} = {v}" for k, v in overrides.items()})
    txt = _set_consts(open(dst).read(), vals)
    open(dst, "w").write(txt)
    return sorted(baked)


def stage(zpath, work, track, variants):
    base = os.path.join(work, f"t{track}_base")
    shutil.rmtree(base, ignore_errors=True)
    zipfile.ZipFile(zpath).extractall(base)
    sub_rel = "Track_1/submit_fno/submission.py" if track == 1 else "Track_2/submit/submission.py"
    fno_rel = "Track_1/submit_fno/_fno.py"
    heads = {}
    for rel in (sub_rel, fno_rel):
        h = os.path.join(work, f"head_t{track}_" + os.path.basename(rel))
        with open(h, "w") as f:
            f.write(subprocess.check_output(["git", "-C", ROOT, "show", f"{BASEREV}:{rel}"], text=True))
        heads[rel] = h
    dirs = {"base": base}
    for name, ov in variants.items():
        d = os.path.join(work, f"t{track}_{name}")
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(base, d)
        k1 = rebake(sub_rel, os.path.join(base, "submission.py"), heads[sub_rel],
                    os.path.join(d, "submission.py"), ov.get("sub", {}))
        k2 = rebake(fno_rel, os.path.join(base, "_fno.py"), heads[fno_rel],
                    os.path.join(d, "_fno.py"), ov.get("fno", {}))
        print(f"[stage] t{track}_{name}: baked submission {k1} | _fno {k2}", flush=True)
        dirs[name] = d
    return dirs


# ------------------------------------------------------------------------------ T1 runner (subprocess)
def run_t1(d, out):
    os.chdir(d)
    sys.path.insert(0, d)
    import importlib.util
    spec = importlib.util.spec_from_file_location("submission", os.path.join(d, "submission.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = mod
    spec.loader.exec_module(mod)
    x = np.load(os.path.join(ROOT, "Track_1", "val_unseen.npz"))["x"]
    r = mod.predict(x.copy())
    perm = np.random.default_rng(0).permutation(len(x))
    rp = mod.predict(x[perm].copy())
    inv = np.argsort(perm)
    n1 = [mod.predict(x[i:i + 1].copy()) for i in (0, 1, 2, 3)]     # call 1 eager, 2.. lazy batch-1 graph
    np.savez(out, pred=r["prediction"][..., :2], lower=r["lower"][..., :2], upper=r["upper"][..., :2],
             pred_perm=rp["prediction"][inv][..., :2],
             pred_n1=np.concatenate([o["prediction"][..., :2] for o in n1]),
             lower_n1=np.concatenate([o["lower"][..., :2] for o in n1]),
             upper_n1=np.concatenate([o["upper"][..., :2] for o in n1]),
             jg_sub=str(getattr(mod, "JOINT_GAIN", None)), jg_fno=str(getattr(mod._fno, "JOINT_GAIN", None)),
             small_graph=str(getattr(mod, "SMALL_GRAPH_ACTIVE", None)))


def tke(p):             # physical (N,20,H,W,2) -> (N,H,W)
    p = p.astype(np.float64)
    return 0.5 * (p[..., 0].var(1) + p[..., 1].var(1))


def check_pair(tag, off, on, y0, lo=None, hi=None, tol_k=1e-4):
    ok = True
    k0, k1 = tke(off), tke(on)
    rel_k = np.abs(k1 - k0) / np.maximum(k0, 1e-12)
    big = k0 > 1e-6 * k0.max()
    tm_d = np.abs(on.astype(np.float64).mean(1) - off.astype(np.float64).mean(1))
    tm_rel = tm_d.max() / np.abs(off).max()
    d_on = np.sqrt(((on.astype(np.float64) - y0) ** 2).sum(axis=(1, 2, 3, 4)))
    d_off = np.sqrt(((off.astype(np.float64) - y0) ** 2).sum(axis=(1, 2, 3, 4)))
    closer = int((d_on <= d_off).sum())
    tke_ok = rel_k[big].max() < tol_k
    tm_ok = tm_rel < 1e-6
    ok &= bool(tke_ok and tm_ok and closer == len(d_on))
    print(f"[{tag}] TKE map rel diff max {rel_k[big].max():.2e} (p99.9 {np.quantile(rel_k[big], 0.999):.2e}) "
          f"{'OK' if tke_ok else 'FAIL'} | time-mean max|d| {tm_d.max():.2e} (rel to max|P| {tm_rel:.2e}) "
          f"{'OK' if tm_ok else 'FAIL'} | closer to y0 {closer}/{len(d_on)} "
          f"(mean ||P-y0|| off {d_off.mean():.4f} on {d_on.mean():.4f}, ratio {(d_on / d_off).mean():.4f})", flush=True)
    if lo is not None:
        cont = bool(np.all(lo <= on) and np.all(on <= hi) and np.all(lo <= hi))
        ok &= cont
        print(f"[{tag}] lower <= pred <= upper: {'OK' if cont else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1zip")
    ap.add_argument("--t2zip")
    ap.add_argument("--work", default="/tmp/jg_verify")
    ap.add_argument("--baserev", default="55f0e88", help="commit the baseline zips were built from")
    ap.add_argument("--run-t1", nargs=2, metavar=("DIR", "OUT"))
    a = ap.parse_args()
    if a.run_t1:
        return run_t1(*a.run_t1)
    global BASEREV
    BASEREV = a.baserev
    os.makedirs(a.work, exist_ok=True)
    allok = True
    V = {"off": {"sub": {"JOINT_GAIN": "False"}, "fno": {"JOINT_GAIN": "False"}},
         "on": {"sub": {"JOINT_GAIN": "True"}, "fno": {"JOINT_GAIN": "True"}},
         "y0": {"sub": {"JOINT_GAIN": "False", "ALPHA": "0.0"}, "fno": {"JOINT_GAIN": "False"}}}
    if a.t1zip:
        dirs = stage(a.t1zip, a.work, 1, V)
        R = {}
        for n, d in dirs.items():
            o = os.path.join(a.work, f"t1_{n}.npz")
            print(f"[T1 {n}] running ...", flush=True)
            subprocess.run([PY, "-u", __file__, "--run-t1", d, o], check=True)
            R[n] = np.load(o)
        b, f, g, y = R["base"], R["off"], R["on"], R["y0"]
        print(f"[T1] on package flags: submission.JOINT_GAIN {g['jg_sub']} _fno.JOINT_GAIN {g['jg_fno']} "
              f"small_graph {g['small_graph']}", flush=True)
        same = all(np.array_equal(b[k], f[k]) for k in ("pred", "lower", "upper", "pred_n1", "lower_n1", "upper_n1"))
        allok &= same
        print(f"[T1] (a) knob OFF vs baseline zip, batched + batch-1 graph calls: "
              f"{'BIT-IDENTICAL' if same else 'DIFFERENT'}", flush=True)
        dg = np.abs(g["pred"] - f["pred"]).max()
        print(f"[T1] on vs off max|dP| {dg:.3e} (must be > 0: the shipped batched path runs the patched compose)", flush=True)
        allok &= bool(dg > 0)
        allok &= check_pair("T1 (b) batched", f["pred"], g["pred"], y["pred"].astype(np.float64), g["lower"], g["upper"])
        rms = np.sqrt((g["pred"].astype(np.float64) ** 2).mean())
        dr = np.abs(g["pred_perm"] - g["pred"]).max() / rms
        dr0 = np.abs(f["pred_perm"] - f["pred"]).max() / rms
        allok &= bool(dr < 1e-4)
        print(f"[T1] (c) permutation drift on {dr:.2e} of rms (off {dr0:.2e}) {'OK' if dr < 1e-4 else 'FAIL'}", flush=True)
        # batch-1 calls: [0] eager, [1..3] lazy graph (SMALL_GRAPH_LAZY, SMALL_N 2)
        e_on = np.abs(g["pred_n1"] - g["pred"][:4]).max(axis=(1, 2, 3, 4))
        e_off = np.abs(g["pred_n1"] - f["pred"][:4]).max(axis=(1, 2, 3, 4))
        gr_ok = bool(np.all(e_on < 0.1 * e_off))
        allok &= gr_ok
        print(f"[T1] (c) batch-1 calls (0 eager, 1-3 graph): max|on_N1 - on_batched| {np.array2string(e_on, precision=2)} "
              f"vs max|on_N1 - off_batched| {np.array2string(e_off, precision=2)} -> graph site patched: "
              f"{'OK' if gr_ok else 'FAIL'}", flush=True)
        allok &= check_pair("T1 (b) batch-1 graph", f["pred_n1"], g["pred_n1"], y["pred_n1"].astype(np.float64),
                            g["lower_n1"], g["upper_n1"])
    if a.t2zip:
        dirs = stage(a.t2zip, a.work, 2, V)
        R = {}
        for n, d in dirs.items():
            o = os.path.join(a.work, f"t2_{n}.npz")
            print(f"[T2 {n}] stream_eval ...", flush=True)
            r = subprocess.run([PY, "-u", os.path.join(ROOT, "Track_2", "stream_eval.py"), "--submission", d, "--save", o],
                               capture_output=True, text=True)
            for ln in r.stdout.splitlines():
                if re.search(r"load:|errors|scores|final|saved", ln):
                    print(f"[T2 {n}] {ln.strip()}", flush=True)
            if r.returncode:
                print(r.stderr[-3000:], flush=True)
                raise SystemExit(f"T2 {n} failed")
            if n == "on":
                allok &= "graph CAPTURED" in r.stdout and "joint_gain True" in r.stdout
                print(f"[T2 on] graph CAPTURED + joint_gain True in load line: "
                      f"{'OK' if ('graph CAPTURED' in r.stdout and 'joint_gain True' in r.stdout) else 'FAIL'}", flush=True)
            R[n] = np.load(o)
        phys = lambda p: p[..., :2].astype(np.float64)       # stream_eval --save stores PHYSICAL (denormalized) arrays
        b, f, g, y = R["base"], R["off"], R["on"], R["y0"]
        same = all(np.array_equal(b[k], f[k]) for k in ("pred", "lower", "upper"))
        allok &= same
        print(f"[T2] (a) knob OFF vs baseline zip, full stream: {'BIT-IDENTICAL' if same else 'DIFFERENT'}", flush=True)
        allok &= check_pair("T2 (b) stream", phys(f["pred"]), phys(g["pred"]), phys(y["pred"]),
                            phys(g["lower"]), phys(g["upper"]))
        r = subprocess.run([PY, "-u", os.path.join(ROOT, "Track_2", "stream_eval.py"), "--submission", dirs["on"],
                            "--causal-check"], capture_output=True, text=True)
        cz = "CAUSALITY OK" in r.stdout
        allok &= cz
        print(f"[T2] (d) causal check on `on`: {'CAUSALITY OK' if cz else 'FAIL'} "
              f"{[ln for ln in r.stdout.splitlines() if 'CAUSALITY' in ln][:1]}", flush=True)
    print("JOINT_GAIN VERIFY: " + ("ALL OK" if allok else "FAILED"), flush=True)
    sys.exit(0 if allok else 1)


BASEREV = "55f0e88"
if __name__ == "__main__":
    main()
