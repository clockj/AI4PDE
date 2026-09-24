"""Shared twin export for the band / interval / alpha arms (Halvor s_in, Corin alpha-map, Bram centre).

Writes, per split (val_seen, val_unseen; ALL AoA) -- read it with load_export.load(), which rebuilds
the float32 fields exactly as below:
    ids              (N,)                 "<Re>_<AoA>:<t0>"
    x                (N,20,32,64,2) f32   raw input (m/s), exact
    y                (N,20,32,64,2) f32   raw target, exact (scorer drops elements with y == 0)
    e_a0, e_a1       (N,20,32,64,2) f16   twin-ensemble ERROR (pred - y) at ALPHA=0 and ALPHA=1.
                                          |E| ~ 0.01 m/s -> f16 step ~8e-6 (raw preds ~0.3 m/s
                                          would be ~2e-4 and scramble optimal_band's sort).
                                          compose is linear in ALPHA: e(a) = (1-a)*e_a0 + a*e_a1,
                                          prediction = y + e.
    sig_twin         (N,1,32,64,2)  f32   twin sigma head (physical units)
    sig_full         (N,1,32,64,2)  f32   full-member sigma head (sigma_ref comes from these)
Sized to fit the 500 MB fetch limit for both splits.

Plus, per split, a SEPARATE file members_<split>.npz (fetch each on its own):
    e_mem            (5,N,20,32,64,2) f16  each twin's own ERROR before the member average, no
                                           compose (ALPHA=0 per member), same order as `twin`.
    sig_mem          (5,N,1,32,64,2)  f32  each twin's own sigma head.
For error-shape / member-median work (Jonas, Bram). load_export.load_members() reads it.

Twins give the honest error field on val_unseen (VAL_TRAJ, never trained on). Full members are only
used for sigma, never for error (research notes 6.3).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Track_1", "submit_fno"))
sys.path.insert(0, os.path.join(HERE, "Track_1"))

FULL = ["ffno_all_w40l8_sim_s1.pth", "ffno_all_w40l8_sim_s2.pth", "ffno_all_w40l8_sim_s3.pth",
        "ffno_all_w40l8_sim_s4.pth", "arch_vit_all_s1.pth", "arch_vit_all_cmb3sim_s3.pth"]
TWIN = ["p2_ffno_l8_sim_s1.pth", "p2_ffno_l8_sim_s2.pth", "p2_ffno_l8_sim_s3.pth",
        "arch_vit_d384.pth", "arch_vit_cmb3sim.pth"]
ALPHA = 0.60


def locate(roots, names):
    found = {}
    for root in roots:
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f in names and f not in found:
                    found[f] = os.path.join(dp, f)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", nargs="+", default=["/data", "/users/PAS1187/qwu/RealPDE"])
    ap.add_argument("--data", default="/users/PAS1187/qwu/RealPDE/train_real/train_real")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--tmp", default="/tmp/emre_export")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)

    found = locate(a.ckpt_root, set(FULL + TWIN))
    missing = [n for n in FULL + TWIN if n not in found]
    if missing:
        print("MISSING checkpoints:", missing)
        r = subprocess.run(f"find {' '.join(a.ckpt_root)} -maxdepth 6 -name '*.pth' | head -100",
                           shell=True, capture_output=True, text=True)
        print("== *.pth under ckpt roots ==\n" + r.stdout)
        return 3
    for n in FULL + TWIN:
        print(f"ckpt {n:34s} {found[n]}")

    import torch
    import _fno
    import make_val
    try:
        import sps9                       # loads the v9 kit from /users/PAS1187/qwu/RealPDE
    except Exception as e:                # local smoke test has no kit; metrics are sanity-only
        print("sps9 unavailable, skipping sanity metrics:", e)
        sps9 = None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, torch.__version__, "AMP", _fno._AMP)
    full = [_fno.load_any(found[n], dev) for n in FULL]
    twin = [_fno.load_any(found[n], dev) for n in TWIN]

    make_val.DATA = a.data
    for split, names in (("unseen", make_val.UNSEEN), ("seen", make_val.SEEN)):
        src = os.path.join(a.tmp, f"val_{split}.npz")
        make_val.build(names, src)
        d = np.load(src)
        x, y, ids = d["x"], d["y"], d["ids"]
        n = len(x)
        t0 = time.time()
        sig_t = np.empty((n, 1, 32, 64, 2), np.float32)
        sig_f = np.empty((n, 1, 32, 64, 2), np.float32)
        y1 = _fno.predict_fno(twin, x, dev, alpha=1.0, sigma_out=sig_t)[..., :2].copy()
        y0 = _fno.predict_fno(twin, x, dev, alpha=0.0)[..., :2].copy()
        y6 = _fno.predict_fno(twin, x, dev, alpha=ALPHA)[..., :2].copy()
        _fno.predict_fno(full, x, dev, alpha=ALPHA, sigma_out=sig_f)
        # compose linear in alpha: the reconstruction must match the direct ALPHA call.
        lin = float(np.abs((1 - ALPHA) * y0 + ALPHA * y1 - y6).max())
        print(f"[{split}] N={n} infer {time.time() - t0:.1f}s  linearity max|diff| {lin:.2e}")
        if not lin < 1e-3:
            print("LINEARITY CHECK FAILED")
            return 4

        # Sanity numbers against the notes (twin, ALPHA 0.60): rel/tke/mvpe on AoA 15-20 and all.
        P = np.concatenate([y6, np.zeros_like(y6[..., :1])], -1)
        aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in ids])
        for tag, k in (("all", np.ones(n, bool)), ("aoa15-20", np.isin(aoa, [15, 20]))):
            if k.any() and sps9 is not None:
                dm, tk, mv = sps9.sample_metrics(P[k], y[k])
                print(f"[{split}] {tag:9s} n={int(k.sum()):4d} rel {np.mean(dm):.5f} "
                      f"tke {np.mean(tk):.5f} mvpe {np.mean(mv):.5f}")

        x2, y2 = x[..., :2].astype(np.float32), y[..., :2].astype(np.float32)
        # x and y in full f32: real PIV fluctuations reach ~0.2 m/s, where a f16 fluctuation
        # split loses ~1e-4 (job 20260921-052807 failed that round trip). Exact is simpler.
        np.savez_compressed(
            os.path.join(a.out, f"export_{split}.npz"), ids=ids, x=x2, y=y2,
            e_a0=(y0 - y2).astype(np.float16), e_a1=(y1 - y2).astype(np.float16),
            sig_twin=sig_t, sig_full=sig_f, sig_full_is_twin=np.int8(0),
            full=np.array(FULL), twin=np.array(TWIN))
        # Round-trip check: the loader must rebuild y's zero set exactly and y/x to <1e-4 m/s.
        import load_export
        r = load_export.load(os.path.join(a.out, f"export_{split}.npz"))
        ok = np.array_equal(r["y"], y2) and np.array_equal(r["x"], x2)
        print(f"[{split}] round-trip x,y exact; |dy|,|dx| max "
              f"{np.abs(r['y'] - y2).max():.2e}, {np.abs(r['x'] - x2).max():.2e}  ok={ok}")
        if not ok:
            return 5
        if True:                          # both splits: seen is where bands are fitted (Bram)
            e_mem = np.empty((len(twin),) + y2.shape, np.float16)
            sig_mem = np.empty((len(twin), n, 1, 32, 64, 2), np.float32)
            for j, m in enumerate(twin):
                pj = _fno.predict_fno([m], x, dev, alpha=0.0, sigma_out=sig_mem[j])[..., :2]
                e_mem[j] = (pj - y2).astype(np.float16)
            # The fused alpha=0 path averages members in raw space, so the member mean must match.
            mm = float(np.abs(e_mem.astype(np.float32).mean(0) - (y0 - y2)).max())
            print(f"[{split}] member-mean vs ensemble alpha=0 max|diff| {mm:.2e}")
            np.savez_compressed(os.path.join(a.out, f"members_{split}.npz"), ids=ids,
                                e_mem=e_mem, sig_mem=sig_mem, twin=np.array(TWIN))
            print(f"[{split}] wrote members_{split}.npz "
                  f"{os.path.getsize(os.path.join(a.out, f'members_{split}.npz')) / 1e6:.1f} MB")
        print(f"[{split}] wrote export_{split}.npz "
              f"{os.path.getsize(os.path.join(a.out, f'export_{split}.npz')) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
