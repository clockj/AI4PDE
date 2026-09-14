"""Gate: a Track-1 package with a baked GRAPH runs the graph, returns the eager path's arrays,
and carries its fingerprint.

    python Track_1/verify/graph_path.py --t1 EXTRACTED_T1_DIR        # GPU

WHY THIS EXISTS.  The CUDA graph on Track 1's `predict` path (round 26) is decided at IMPORT
behind a verification against eager, so a package can silently fall back -- and a replayed graph
is bit-exact, so the board's error subscores cannot say which path ran.  The graph path therefore
scales the band's `base` by (1 + GRAPH_FP).  This checks, on the archive itself:

  * `GRAPH_ACTIVE` is True when GRAPH is baked True;
  * the prediction equals an eager twin of the SAME archive (`GRAPH = False`) EXACTLY, including
    a tail shorter than a chunk, which runs eager (exercised separately);
  * the bounds carry exactly the fingerprint, and rel / tke / mvpe do not move;
  * what the fingerprint does to sps, locally and at the board's 1.36 error stretch, for several
    GRAPH_FP values -- the band is fitted near its optimum, so a small widening is close to
    second order in sps, and the value has to be chosen from this table, not assumed;
  * permutation invariance of the graph path;
  * graph against eager ms/window, interleaved in ONE process (evidence only on an idle node).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
F_TAIL = 1.36          # the board's per-element error stretch (calib_band13.py)


def _twin(d: str) -> str:
    """A copy of the extracted package with GRAPH forced to False."""
    out = tempfile.mkdtemp(prefix="graph_twin_")
    shutil.copytree(d, out, dirs_exist_ok=True)
    src = os.path.join(out, "submission.py")
    txt, n = re.subn(r"(?m)^GRAPH = .*$", "GRAPH = False", open(src).read())
    if n != 1:
        raise SystemExit(f"{d}: expected exactly one `GRAPH = ` line, found {n}")
    open(src, "w").write(txt)
    return out


def _load(d: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, "submission.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2])
    a = ap.parse_args()
    ok = True
    txt = open(os.path.join(a.t1, "submission.py")).read()
    baked = re.search(r"(?m)^GRAPH = (.*)$", txt).group(1).strip()
    fp = float(re.search(r"(?m)^GRAPH_FP = (.*)$", txt).group(1))
    ship = _load(a.t1, "graph_ship")
    ref = _load(_twin(a.t1), "graph_eager")
    batch = ship._state[5] if ship._state is not None else None
    print(f"GRAPH = {baked}   GRAPH_FP = {fp:g}   GRAPH_ACTIVE {ship.GRAPH_ACTIVE}   probe max|d| "
          f"{ship.GRAPH_PROBE_DIFF}   AMP_ACTIVE {ship.AMP_ACTIVE}   chunk {batch}   "
          f"(eager twin: GRAPH_ACTIVE {ref.GRAPH_ACTIVE}, AMP_ACTIVE {ref.AMP_ACTIVE})")
    if baked == "True" and ship.GRAPH_ACTIVE is not True:
        print("  [FAIL] GRAPH was baked but the capture was not kept"); ok = False

    sys.path.insert(0, T1)
    import sps9                                    # noqa: E402
    from sps9 import prep, sc, sps_score           # noqa: E402
    du = np.load(os.path.join(T1, "val_unseen.npz"))
    X, Y = du["x"], du["y"]

    def errors(P):
        dm, tk, mv = sps9.sample_metrics(P, Y)
        return (float(np.mean(sc.rel_l2_per_sample(P, Y, 2))),
                float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))), float(sc.mvpe_rel_l2(P, Y)),
                sps9.weights_from_metrics(dm, tk, mv))

    def sps_of(P, half, w, stretch=1.0):
        A_, _, s_, n_ = prep(P, Y)
        return float(sps_score(np.float32(stretch) * A_, w, s_, n_, half.astype(np.float32)))

    o1, o0 = ship.predict(X), ref.predict(X)
    P1, P0 = o1["prediction"], o0["prediction"]
    dpred = float(np.abs(P1 - P0).max())
    h1 = (o1["upper"] - P1)[..., :2]
    h0 = (o0["upper"] - P0)[..., :2]
    sym = float(np.abs((P1 - o1["lower"])[..., :2] - h1).max())
    m = h0 > 1e-3
    ratio = float(np.median(h1[m] / h0[m])) if m.any() else float("nan")
    print(f"  prediction, graph vs eager twin: max|d| {dpred:.3e}   (n = {len(X)}, chunk {batch})")
    print(f"  band half-width ratio graph/eager: median {ratio:.8f} (expect {1 + fp:.8f})   "
          f"lower/upper symmetric to {sym:.1e}")
    if dpred > 0.0:
        print("  [FAIL] the graph path returns different predictions"); ok = False
    if baked == "True" and ship.GRAPH_ACTIVE and abs(ratio - (1 + fp)) > max(fp / 4, 2e-6):
        print("  [FAIL] the bounds do not carry the fingerprint"); ok = False

    e1, e0 = errors(P1), errors(P0)
    s1, s0 = sps_of(P1, h1, e1[3]), sps_of(P0, h0, e0[3])
    t1, t0 = sps_of(P1, h1, e1[3], F_TAIL), sps_of(P0, h0, e0[3], F_TAIL)
    print(f"  eager twin  rel {e0[0]:.8f}  tke {e0[1]:.8f}  mvpe {e0[2]:.8f}  sps {s0:.6f}  (x{F_TAIL} tail: {t0:.6f})")
    print(f"  as shipped  rel {e1[0]:.8f}  tke {e1[1]:.8f}  mvpe {e1[2]:.8f}  sps {s1:.6f}  (x{F_TAIL} tail: {t1:.6f})")
    worst = max(abs(e1[i] - e0[i]) / max(abs(e0[i]), 1e-12) for i in range(3))
    print(f"  error metrics, worst relative move {worst:.2e};   sps fingerprint {s1 - s0:+.3e} "
          f"(x{F_TAIL} tail {t1 - t0:+.3e}) = {0.25 * (t1 - t0):+.1e} final at w_sps 0.25")
    if worst > 1e-6:
        print("  [FAIL] an error metric moved"); ok = False

    print("  fingerprint sweep on the eager twin's arrays: sps with base x (1 + eps)")
    for stretch in (1.0, F_TAIL):
        b0 = sps_of(P0, h0, e0[3], stretch)
        cells = []
        for eps in a.eps:
            s = sps_of(P0, h0 * np.float32(1.0 + eps), e0[3], stretch)
            cells.append(f"eps {eps:g}: {s - b0:+.2e}")
        print(f"    stretch {stretch:.2f}  sps {b0:.6f}   " + "   ".join(cells))

    if batch:
        k = min(2 * batch + 7, len(X))
        dtail = float(np.abs(ship.predict(X[:k])["prediction"] - ref.predict(X[:k])["prediction"]).max())
        print(f"  tail ({k} windows = 2 chunks + {k - 2 * batch}): graph vs eager max|d| {dtail:.3e}")
        if dtail > 0.0:
            print("  [FAIL] the tail moves the prediction"); ok = False

    perm = np.random.default_rng(1).permutation(len(X))
    inv = float(np.abs(ship.predict(X[perm])["prediction"] - P1[perm]).max())
    print(f"  permutation invariance of the shipped path: max|d| {inv:.2e}")
    if inv > 1e-4:
        print("  [FAIL] order-dependent"); ok = False

    if torch.cuda.is_available():
        te, tg = [], []
        for _ in range(6):
            for mdl, acc in ((ref, te), (ship, tg)):
                for _ in range(3):
                    torch.cuda.synchronize(); t = time.perf_counter(); mdl.predict(X)
                    torch.cuda.synchronize(); acc.append(time.perf_counter() - t)
        ae, ag = 1e3 * np.median(te) / len(X), 1e3 * np.median(tg) / len(X)
        print(f"  interleaved, one process: eager twin {ae:.3f} ms/window   graph {ag:.3f}   ({100 * (ag / ae - 1):+.1f} %)")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
