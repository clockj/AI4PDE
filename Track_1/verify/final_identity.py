"""Prove a REBUILT pair of packages returns the uploaded packages' outputs bit for bit (tolerance 0). GPU.

    python Track_1/verify/final_identity.py --t1-a <uploaded T1 dir> --t1-b <rebuilt T1 dir> \
                                            --t2-a <uploaded T2 dir> --t2-b <rebuilt T2 dir>

WHY (2026-09-28). `build_final.sh` + RPDE_REF_T1/T2 compares the FILES inside two archives. That is the right gate
for a rebuild of an unchanged tree and the wrong one for a refactor of `submission.py` / `_fno.py`, whose files
differ by construction: what has to be shown is that the OUTPUTS do not. `verify/pkg_identity.py` compares outputs,
but at a tolerance, on Track 1 in a single N = 336 call (the pinned output pool, tail chunks and pool exhaustion are
never exercised), and on Track 2 without ever revealing a target (so R1 / D1c / VARCAL never move). This does both
properly:

  * Track 1: `predict()` on the 336 val_unseen windows as SUCCESSIVE calls of N = 14, 56, 99 and 336 in ONE process
    (the first two fit the 128-window pinned pool, the third exhausts it, the fourth allocates; tails of 14, 43, 0).
  * Track 2: the full 8-trajectory stream, `reset_ttt_state()` at each boundary, the previous window's target revealed
    at every step, once with the LEGACY harness normalization of that target (input statistics -- what our local reads
    used before 09-28) and once with the TARGET statistics the official runner uses (`t2_common.normalize_tgt`).

Each package runs in its own subprocess (both import modules named `submission` and `_fno`), with bytecode writing
off so an extracted reference directory stays untouched. Comparison: `np.array_equal` on prediction / lower / upper
(the gate), plus the count of elements whose BYTES differ (+0.0 vs -0.0 is the only way the two can disagree), the
max |difference| and the first differing index. Exit 0 only if every array is equal.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VAL = os.path.join(ROOT, "Track_1", "val_unseen.npz")
T1_CALLS = (14, 56, 99, 336)              # successive predict() calls: [0:14], [14:70], [70:169], [0:336]


def _import(pkg):
    import importlib.util
    sys.path.insert(0, pkg)
    spec = importlib.util.spec_from_file_location("submission", os.path.join(pkg, "submission.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = m
    spec.loader.exec_module(m)
    return m


def worker_t1(pkg, val, out):
    X = np.load(val)["x"].astype(np.float32)
    m = _import(pkg)
    res, i = {}, 0
    for k, n in enumerate(T1_CALLS):
        sl = slice(0, n) if n == len(X) else slice(i, i + n)
        o = m.predict(np.ascontiguousarray(X[sl]))
        for key in ("prediction", "lower", "upper"):
            res[f"c{k}_N{n}_{key}"] = np.array(o[key], dtype=np.float32, copy=True)   # copy: pool slices are reused
        if n != len(X):
            i += n
    np.savez(out, **res)


def worker_t2(pkg, val, out, space):
    import torch
    sys.path.insert(0, os.path.join(ROOT, "Track_2"))
    import t2_common as t2
    d = np.load(val)
    X, Y, ids = d["x"], d["y"], [str(s).split(":")[0] for s in d["ids"]]
    m = _import(pkg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obj = m.get_ttt_model(pkg, dev)
    print(f"  [{os.path.basename(pkg.rstrip('/'))}] graph {'CAPTURED' if getattr(obj, 'graph', None) is not None else 'NONE'}",
          flush=True)
    P, L, U = [], [], []
    prev, name = None, None
    for i in range(len(X)):
        if ids[i] != name:
            obj.reset_ttt_state()
            prev, name = None, ids[i]
        pn, info = obj.ttt_step(torch.from_numpy(t2.normalize_in(X[i])[None]), prev)
        prev = torch.from_numpy(t2.normalize_prev(Y[i], space)[None])
        P.append(np.array(np.asarray(pn), dtype=np.float32, copy=True))
        L.append(np.array(np.asarray(info["lower"]), dtype=np.float32, copy=True))
        U.append(np.array(np.asarray(info["upper"]), dtype=np.float32, copy=True))
    np.savez(out, prediction=np.concatenate(P), lower=np.concatenate(L), upper=np.concatenate(U))


def run_worker(args, work, tag):
    out = os.path.join(work, tag + ".npz")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, os.path.abspath(__file__)] + args + ["--out", out], env=env)
    if r.returncode != 0:
        raise SystemExit(f"worker {tag} failed (rc {r.returncode})")
    return np.load(out)


def compare(label, A, B):
    ok = True
    print(f"\n{label}")
    print(f"  {'array':<24} {'equal':>6} {'bytes differ':>13} {'max|A-B|':>11}  first diff")
    for k in sorted(A.files):
        a, b = A[k], B[k]
        if a.shape != b.shape:
            print(f"  {k:<24} SHAPE {a.shape} vs {b.shape}"); ok = False; continue
        eq = bool(np.array_equal(a, b))
        nb = int(np.count_nonzero(a.view(np.int32) != b.view(np.int32)))
        with np.errstate(invalid="ignore"):
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        dm = float(np.nanmax(d)) if d.size else 0.0
        first = "" if eq else str(np.unravel_index(int(np.argmax(np.where(np.isnan(d), np.inf, d))), a.shape))
        print(f"  {k:<24} {str(eq):>6} {nb:>13} {dm:>11.3e}  {first}")
        ok &= eq
    print(f"  -> {'IDENTICAL (tolerance 0)' if ok else '*** DIFFERS ***'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-a"); ap.add_argument("--t1-b")
    ap.add_argument("--t2-a"); ap.add_argument("--t2-b")
    ap.add_argument("--t2-space", nargs="+", choices=["input", "target"], default=["input", "target"])
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--work", default=None, help="scratch dir for the workers' npz (default: a fresh /tmp dir)")
    ap.add_argument("--worker", choices=["t1", "t2"], default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pkg", help=argparse.SUPPRESS); ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--space", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker == "t1":
        return worker_t1(os.path.abspath(a.pkg), a.val, a.out)
    if a.worker == "t2":
        return worker_t2(os.path.abspath(a.pkg), a.val, a.out, a.space)

    work = a.work or tempfile.mkdtemp(prefix="final_identity_", dir="/tmp")
    os.makedirs(work, exist_ok=True)
    ok = True
    if a.t1_a and a.t1_b:
        R = [run_worker(["--worker", "t1", "--pkg", p, "--val", a.val], work, f"t1_{t}")
             for t, p in (("a", a.t1_a), ("b", a.t1_b))]
        ok &= compare(f"TRACK 1  predict() at N = {', '.join(map(str, T1_CALLS))} (successive calls, one process)\n"
                      f"  A = {a.t1_a}\n  B = {a.t1_b}", *R)
    if a.t2_a and a.t2_b:
        for sp in a.t2_space:
            R = [run_worker(["--worker", "t2", "--pkg", p, "--val", a.val, "--space", sp], work, f"t2_{sp}_{t}")
                 for t, p in (("a", a.t2_a), ("b", a.t2_b))]
            ok &= compare(f"TRACK 2  full stream, targets revealed, revealed target normalized with {sp.upper()} statistics\n"
                          f"  A = {a.t2_a}\n  B = {a.t2_b}", *R)
    print(f"\nFINAL_IDENTITY {'OK' if ok else 'FAILED'}   (workers' arrays in {work})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
