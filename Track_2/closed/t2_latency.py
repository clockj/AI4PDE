"""What a Track-2 step costs, per member set, at batch 1 on this GPU.

Track 1 hands `predict()` every window at once, so an extra member costs a fraction of a
millisecond spread over N windows.  Track 2 pays the whole launch cost every step, so the
member/latency trade has to be re-measured here rather than inherited.

    python t2_latency.py                # every configuration below
    python t2_latency.py --only fused8

Reports mean ms/step and the `time_score` it implies, plus the score cost of each extra
invocation, so a member set can be chosen against the accuracy it buys.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_2/.  Putting the PARENT of closed/ on the
# path keeps `import t2_common / sps9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_2/, so every submit/ and ../Track_1/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Track_1"))
import t2_common as t2  # noqa: E402

KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)

# name -> members, taken from the shipped all-data package (build_round12.sh order).
U = str(HERE.parent / "Track_1" / "unet")
ALL = [f"{U}/fno_all_vl_s{i}.pth" for i in range(1, 8)] + [
    f"{U}/fno_all_vl_m1220.pth", f"{U}/fno_all_vl_w48.pth",
    f"{U}/fno_all_vl_w64.pth", f"{U}/unet_all_g1.pth"]

CONFIGS = {
    "r12_11memb": ALL,                       # exactly what Track 1 ships: 4 invocations
    "fused10":    ALL[:8] + ALL[9:],         # drop w48 -> 3 invocations (ROADMAP P2)
    "fused9":     ALL[:8] + ALL[10:],        # w24 group + U-Net -> 2 invocations
    "fused8":     ALL[:8],                   # w24 group only -> 1 invocation
    "fused4":     ALL[:4],                   # 4 w24 seeds -> 1 invocation
    "single":     ALL[:1],                   # 1 member -> 1 invocation
}


def bench(models, band, ref, p, dev, n=300, warm=60):
    import _fno  # noqa
    sub = importlib.util.spec_from_file_location("t2sub", HERE / "submit" / "submission.py")
    m = importlib.util.module_from_spec(sub); sub.loader.exec_module(m)
    obj = m.FrozenEnsemble(models, dev, band, ref, p)
    x = torch.randn(1, 20, 32, 64, 3)
    for _ in range(warm):
        obj.ttt_step(x, None)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = perf_counter()
        obj.ttt_step(x, None)
        torch.cuda.synchronize()
        ts.append(perf_counter() - t0)
    return np.asarray(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE / "submit"))
    import _fno

    dev = "cuda"
    z = np.load(HERE / "submit" / "band.npz")
    band, ref, p = z["base"].astype(np.float32), z["sigma_ref"].astype(np.float32), float(z["p"])

    print(f"{'config':12s} {'memb':>5s} {'inv':>4s} {'ms/step':>9s} {'p50':>8s} {'p90':>8s} "
          f"{'time_score':>11s}")
    base = None
    for name, ckpts in CONFIGS.items():
        if a.only and name not in a.only:
            continue
        models = _fno.fuse([_fno.load_any(c, dev) for c in ckpts])
        ts = bench(models, band, ref, p, dev, n=a.n)
        mean = float(ts.mean())
        s = sc.score_time(mean)
        if base is None:
            base = s
        print(f"{name:12s} {len(ckpts):5d} {len(models):4d} {mean*1e3:9.3f} "
              f"{np.median(ts)*1e3:8.3f} {np.percentile(ts,90)*1e3:8.3f} {s:11.3f}")
        del models
        torch.cuda.empty_cache()

    print("\nreference: score_time at a few budgets")
    for ms in (0.5, 1, 2, 3, 5, 8, 12, 20):
        print(f"  {ms:5.1f} ms -> {sc.score_time(ms/1000):.3f}")


if __name__ == "__main__":
    main()
