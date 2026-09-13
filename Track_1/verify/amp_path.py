"""Gate: a package with a baked AMP runs what it claims, scores what fp32 scores, and is faster.

    python Track_1/verify/amp_path.py --t1 EXTRACTED_T1_DIR [--t2 EXTRACTED_T2_DIR]    # GPU

WHY THIS EXISTS.  Round 24 found the board charging GPU compute (~1.75 board ms per
4060-equivalent local ms on Track 1), which makes a cheaper precision an accuracy-free latency
lever -- and r25a found fp16 autocast accuracy-neutral and permutation-exact on the A100.  But
AMP is decided at LOAD time behind a self-check, so a package can silently run fp32; and a
board result read as "fp16 did nothing" would then be a misreading.  This checks, on the
archive itself, that `AMP_ACTIVE` is what was baked, that the prediction matches an fp32 twin
of the SAME archive, and it times the two interleaved in ONE process -- the only local timing
that has been reproducible in this project.

Each package is loaded twice: as shipped, and as a copy whose `AMP = ...` line reads `AMP = None`.
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


def _twin(d: str, tag: str) -> str:
    """A copy of the extracted package with AMP forced to None."""
    out = tempfile.mkdtemp(prefix=f"amp_{tag}_")
    shutil.copytree(d, out, dirs_exist_ok=True)
    src = os.path.join(out, "submission.py")
    txt = open(src).read()
    txt2, n = re.subn(r"(?m)^AMP = .*$", "AMP = None", txt)
    if n != 1:
        raise SystemExit(f"{d}: expected exactly one `AMP = ` line, found {n}")
    open(src, "w").write(txt2)
    return out


def _load(d: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, "submission.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def track1(d: str) -> bool:
    ok = True
    baked = re.search(r'(?m)^AMP = (.*)$', open(os.path.join(d, "submission.py")).read()).group(1)
    ship = _load(d, "t1_ship")
    ref = _load(_twin(d, "t1"), "t1_fp32")
    print(f"TRACK 1  baked AMP = {baked}   AMP_ACTIVE {ship.AMP_ACTIVE}   probe diff "
          f"{ship.AMP_PROBE_DIFF if ship.AMP_PROBE_DIFF is None else f'{ship.AMP_PROBE_DIFF:.2e}'}"
          f"   (fp32 twin: {ref.AMP_ACTIVE})")
    if baked.strip() != "None" and ship.AMP_ACTIVE is None:
        print("  [FAIL] AMP was baked but the self-check turned it off"); ok = False
    sys.path.insert(0, T1)
    import sps9                                    # noqa: E402
    from sps9 import prep, sc, sps_score           # noqa: E402
    du = np.load(os.path.join(T1, "val_unseen.npz"))
    X, Y = du["x"], du["y"]

    def score(out):
        P, hi = out["prediction"], out["upper"]
        half = (hi - P)[..., :2].astype(np.float32)
        A_, _, s_, n_ = prep(P, Y)
        dm, tk, mvv = sps9.sample_metrics(P, Y)
        return (float(np.mean(sc.rel_l2_per_sample(P, Y, 2))),
                float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))), float(sc.mvpe_rel_l2(P, Y)),
                float(sps_score(A_, sps9.weights_from_metrics(dm, tk, mvv), s_, n_, half)))

    o1, o0 = ship.predict(X), ref.predict(X)
    s1, s0 = score(o1), score(o0)
    d_rel = float(np.linalg.norm(o1["prediction"] - o0["prediction"]) / np.linalg.norm(o0["prediction"]))
    print(f"  fp32 twin  rel {s0[0]:.5f} tke {s0[1]:.5f} mvpe {s0[2]:.5f} sps {s0[3]:.3f}")
    print(f"  as shipped rel {s1[0]:.5f} tke {s1[1]:.5f} mvpe {s1[2]:.5f} sps {s1[3]:.3f}   "
          f"prediction rel-L2 vs fp32 {d_rel:.2e}")
    worst = max(abs(s1[i] - s0[i]) / max(abs(s0[i]), 1e-12) for i in range(3))
    if worst > 1e-3 or abs(s1[3] - s0[3]) > 0.02:
        print(f"  [FAIL] error metrics moved (worst relative {worst:.2e}, sps {s1[3]-s0[3]:+.3f})"); ok = False
    perm = np.random.default_rng(1).permutation(len(X))
    inv = float(np.abs(ship.predict(X[perm])["prediction"] - o1["prediction"][perm]).max())
    print(f"  permutation invariance of the shipped path: max|d| {inv:.2e}")
    if inv > 1e-4:
        print("  [FAIL] order-dependent"); ok = False
    t0, t1 = [], []
    for _ in range(6):
        for m, acc in ((ref, t0), (ship, t1)):
            torch.cuda.synchronize(); t = time.perf_counter(); m.predict(X)
            torch.cuda.synchronize(); acc.append(time.perf_counter() - t)
    a, b = 1e3 * np.median(t0) / len(X), 1e3 * np.median(t1) / len(X)
    print(f"  interleaved, one process: fp32 twin {a:.3f} ms/window   shipped {b:.3f}   ({100*(b/a-1):+.1f} %)")
    return ok


def track2(d: str) -> bool:
    ok = True
    baked = re.search(r'(?m)^AMP = (.*)$', open(os.path.join(d, "submission.py")).read()).group(1)
    ship_m, ref_m = _load(d, "t2_ship"), None
    tw = _twin(d, "t2")
    ref_m = _load(tw, "t2_fp32")
    ship = ship_m.get_ttt_model(d, "cuda")
    ref = ref_m.get_ttt_model(tw, "cuda")
    print(f"TRACK 2  baked AMP = {baked}   AMP_ACTIVE {ship_m.AMP_ACTIVE}   probe diff "
          f"{ship_m.AMP_PROBE_DIFF if ship_m.AMP_PROBE_DIFF is None else f'{ship_m.AMP_PROBE_DIFF:.2e}'}"
          f"   graph {'ACTIVE' if ship.graph is not None else 'DROPPED'}   (fp32 twin graph "
          f"{'ACTIVE' if ref.graph is not None else 'DROPPED'})")
    if baked.strip() != "None" and ship_m.AMP_ACTIVE is None:
        print("  [FAIL] AMP was baked but the self-check turned it off"); ok = False
    if ship.graph is None:
        print("  [FAIL] the shipped Track-2 path lost its CUDA graph"); ok = False
    g = torch.Generator(device="cpu").manual_seed(2)
    xs = [(torch.randn((1, 20, 32, 64, 3), generator=g) * 0.5).cuda() for _ in range(24)]
    ship.reset_ttt_state(); ref.reset_ttt_state()
    worst = 0.0
    for x in xs:
        p1 = ship.ttt_step(x, x)[0].float(); p0 = ref.ttt_step(x, x)[0].float()
        worst = max(worst, float((p1 - p0).norm() / p0.norm().clamp(min=1e-12)))
    print(f"  ttt_step prediction rel-L2 vs fp32 twin, worst of 24 steps: {worst:.2e}")
    if worst > 5e-3:
        print("  [FAIL] Track-2 prediction moved"); ok = False
    x = xs[0]
    t0, t1 = [], []
    for _ in range(40):
        for mdl, acc in ((ref, t0), (ship, t1)):
            for _ in range(5):
                torch.cuda.synchronize(); t = time.perf_counter(); mdl.ttt_step(x, x)
                torch.cuda.synchronize(); acc.append(time.perf_counter() - t)
    a, b = 1e3 * np.median(t0[25:]), 1e3 * np.median(t1[25:])
    print(f"  interleaved, one process: fp32 twin {a:.3f} ms/step   shipped {b:.3f}   ({100*(b/a-1):+.1f} %)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1")
    ap.add_argument("--t2")
    a = ap.parse_args()
    ok = True
    if a.t1:
        ok &= track1(a.t1)
    if a.t2:
        ok &= track2(a.t2)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
