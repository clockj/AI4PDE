"""Round 17 Q1a: Track-2 ms/step for a member set under one fast-path variant.

Track 1 amortises the launch cost over a batch of 64+; Track 2 pays it every step at batch 1,
so the latency of a variant has to be measured here rather than converted from there.  The
variant is chosen by the SAME environment variables the shipped `submission.py` reads
(`RPDE_TF32`, `RPDE_AMP`), and both are read at import, so one process measures one variant --
the driver script loops over them by re-invoking python.

    RPDE_TF32=1 python t2_lat17.py --sets "L8x4=a.pth,b.pth,..."

The CUDA graph is left ON, exactly as the package ships it.  `_capture()` verifies its replay
against the eager path and silently falls back on ANY disagreement, so a variant that breaks
capture shows up here as a jump in ms/step, not as a wrong answer -- `graph=` in the output row
says which path actually ran.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "submit"))
sys.path.insert(0, str(HERE.parent / "Track_1"))

KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="NAME=a.pth,b.pth ...")
    ap.add_argument("--band", default=str(HERE / "submit" / "band.npz"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warm", type=int, default=60)
    a = ap.parse_args()

    import _fno
    sub = importlib.util.spec_from_file_location("t2sub", HERE / "submit" / "submission.py")
    m = importlib.util.module_from_spec(sub)
    sub.loader.exec_module(m)
    dev = "cuda"
    z = np.load(a.band)
    band, ref, p = (z["base"].astype(np.float32), z["sigma_ref"].astype(np.float32),
                    float(z["p"]))
    tag = (f"TF32={os.environ.get('RPDE_TF32', '0') not in ('', '0')} "
           f"AMP={os.environ.get('RPDE_AMP', 'off') or 'off'}")
    print(f"variant: {tag}   band={os.path.basename(a.band)}", flush=True)
    print(f"{'set':<14} {'memb':>5} {'inv':>4} {'ms/step':>9} {'p50':>8} {'p90':>8} "
          f"{'time_sc':>8} {'graph':>6}")
    for spec in a.sets:
        name, _, lst = spec.partition("=")
        paths = [q if os.path.isabs(q) else str(HERE.parent / "Track_1" / q)
                 for q in lst.split(",")]
        models = _fno.fuse([_fno.load_any(q, dev) for q in paths])
        obj = m.FrozenEnsemble(models, dev, band, ref, p)
        x = torch.randn(1, 20, 32, 64, 3)
        for _ in range(a.warm):
            obj.ttt_step(x, None)
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.n):
            t0 = perf_counter()
            obj.ttt_step(x, None)
            torch.cuda.synchronize()
            ts.append(perf_counter() - t0)
        ts = np.asarray(ts)
        print(f"{name:<14} {len(paths):>5} {len(models):>4} {ts.mean()*1e3:>9.3f} "
              f"{np.median(ts)*1e3:>8.3f} {np.percentile(ts, 90)*1e3:>8.3f} "
              f"{sc.score_time(float(ts.mean())):>8.3f} "
              f"{'yes' if obj.graph is not None else 'NO':>6}", flush=True)
        del models, obj
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
