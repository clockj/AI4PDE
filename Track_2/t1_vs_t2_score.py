"""Round 17 Q1e: score the two SUBMITTED archives side by side on the same windows.

The board has now put the same member set on both tracks five times, and since width 96 the
Track-1 package has beaten its Track-2 twin on `tke` by ~1.5 % three rounds running (R16 -1.52,
R17 -1.64) and on `sps` by 0.10-0.13 twice, where at width 24 the two agreed to +-0.3 %.  Either
the two code paths compute different numbers -- fp16 unpacking, the CUDA graph, the
normalized-space affine, band handling -- or they compute the same numbers and the difference
belongs to the eval machine.  This decides which, locally.

    python t1_vs_t2_score.py --t1 /path/to/extracted_t1 --t2 /path/to/extracted_t2

Both archives are run on the SAME val_unseen windows -- Track 1 through `predict()` in raw
physical units, Track 2 through `get_ttt_model` / `ttt_step` in normalized space, one window at
a time in the evaluator's order with `reset_ttt_state()` at trajectory boundaries -- and both
are then scored by the official kit's `scoring.py`.  A difference here is ours; no difference
here means the board's is the machine's.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Track_1"))
import t2_common as t2  # noqa: E402

VAL = ["11400_20", "13950_10", "16500_5", "21600_15", "26700_0", "3750_0", "6300_10", "8850_20"]

KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sc)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def unload(*names):
    for k in list(sys.modules):
        if k in names:
            del sys.modules[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True, help="extracted Track-1 archive directory")
    ap.add_argument("--t2", required=True, help="extracted Track-2 archive directory")
    ap.add_argument("--traj", nargs="+", default=VAL)
    a = ap.parse_args()

    stream = t2.build_stream(a.traj)
    X = np.stack([s[2] for s in stream]).astype(np.float32)
    Y = np.stack([s[3] for s in stream]).astype(np.float32)
    print(f"{len(X)} windows from {len(a.traj)} trajectories", flush=True)

    d1 = Path(a.t1).resolve()
    sys.path.insert(0, str(d1))
    m1 = load(d1 / "submission.py", "t1_submission")
    o1 = m1.predict(X)
    P1, L1, U1 = (np.asarray(o1[k], dtype=np.float32) for k in ("prediction", "lower", "upper"))
    unload("t1_submission", "_fno")
    sys.path.remove(str(d1))

    d2 = Path(a.t2).resolve()
    sys.path.insert(0, str(d2))
    m2 = load(d2 / "submission.py", "t2_submission")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obj = m2.get_ttt_model(str(d2), dev)
    print(f"  T2 CUDA graph: {'ACTIVE' if getattr(obj, 'graph', None) is not None else 'off'}",
          flush=True)
    P2, L2, U2 = [], [], []
    for name, tid, x_raw, y_raw, first in stream:
        if first:
            obj.reset_ttt_state()
        pn, info = obj.ttt_step(torch.from_numpy(t2.normalize_in(x_raw)[None]), None)
        # .copy() -- round 13 found a gate that handed back a reused staging buffer and every
        # stored prediction became the last one.  Never store what the model returns by
        # reference.
        P2.append(t2.denorm_pred(np.asarray(pn)[0]).copy())
        L2.append(t2.denorm_pred(np.asarray(info["lower"])[0]).copy())
        U2.append(t2.denorm_pred(np.asarray(info["upper"])[0]).copy())
    P2, L2, U2 = (np.stack(z).astype(np.float32) for z in (P2, L2, U2))
    unload("t2_submission", "_fno")

    scale = float(np.abs(P1[..., :2]).max())
    print("\n--- the two paths against each other (physical units) ---")
    for nm, x1, x2 in (("prediction", P1, P2), ("lower", L1, L2), ("upper", U1, U2)):
        d = np.abs(x1[..., :2] - x2[..., :2])
        rl = float(np.linalg.norm((x1[..., :2] - x2[..., :2]).ravel())
                   / np.linalg.norm(x1[..., :2].ravel()))
        print(f"  {nm:11s} max|T1-T2| {float(d.max()):.3e} ({float(d.max())/scale:.2e} of "
              f"field max)   rel-L2 {rl:.3e}")
    h1, h2 = U1 - P1, U2 - P2
    print(f"  half-width  mean T1 {float(h1[..., :2].mean()):.6f}  T2 "
          f"{float(h2[..., :2].mean()):.6f}   max|dh| {float(np.abs(h1-h2)[..., :2].max()):.3e}")

    print("\n--- official subscores on these windows ---")
    print(f"{'path':6s} {'relL2':>8s} {'TKE':>8s} {'MVPE':>8s} {'sps':>8s} | "
          f"{'rel':>7s} {'tke':>7s} {'mvpe':>7s}")
    for nm, P, L, U in (("T1", P1, L1, U1), ("T2", P2, L2, U2)):
        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        mv = float(sc.mvpe_rel_l2(P, Y))
        s = sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=L, upper=U)[0])
        print(f"{nm:6s} {r:8.5f} {k:8.5f} {mv:8.5f} {s:8.4f} | {sc.score_error(r):7.3f} "
              f"{sc.score_error(k):7.3f} {sc.score_error(mv):7.3f}", flush=True)


if __name__ == "__main__":
    main()
