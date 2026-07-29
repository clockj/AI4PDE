"""Score every Track 1 model on both validation sets and report the generalization gap.

    python eval_all.py

Prints one table per set (unseen / seen) plus the per-model delta. The organizers'
sim_real_ft checkpoints trained on BOTH sets, so their delta is the difficulty control:
if it is near zero the sets are equally hard, and our models' deltas are real
generalization error.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = "/home/clockj/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7"
CKPT = "/home/clockj/RealPDE/baseline_checkpoints/sim_real_ft"
for p in (KIT, os.path.join(HERE, "deeponet"), os.path.join(HERE, "diffusion")):
    sys.path.insert(0, p)

spec = importlib.util.spec_from_file_location("sc", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

MEAN3 = np.array([0.154960856, -0.000513992854, 0.0], np.float32)
STD3 = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
MEAN3T = np.array([0.154962569, -0.000517793698, 0.0], np.float32)
STD3T = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)


def subscores(P, Y, tstep):
    c = 2
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, c)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, c)))
    m = sc.mvpe_rel_l2(P, Y)
    sps = sc.score_sps(sc.aggregate_sps(P, Y, c)[0])
    s = [sc.score_error(r), sc.score_error(k), sc.score_error(m), sc.score_time(tstep), sps]
    return dict(rel_err=r, tke_err=k, mvpe_err=m, rel=s[0], tke=s[1], mvpe=s[2],
                time=s[3], sps=sps, final=float(np.mean(s)), ms=tstep * 1000)


def run_official(name, ckpt, X, B=16):
    from load_baseline import load_baseline
    model, _ = load_baseline(os.path.join(CKPT, ckpt), device="cuda")
    model = model.cuda().eval()
    mi, si = torch.from_numpy(MEAN3).cuda(), torch.from_numpy(STD3).cuda()
    mt, st = torch.from_numpy(MEAN3T).cuda(), torch.from_numpy(STD3T).cuda()
    outs, t = [], 0.0
    with torch.no_grad():
        for i in range(0, len(X), B):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + B])).cuda()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            yb = model((xb - mi) / si) * st + mt
            torch.cuda.synchronize(); t += time.perf_counter() - t0
            outs.append(yb.float().cpu().numpy())
    P = np.concatenate(outs).astype(np.float32)
    P[..., 2] = 0.0
    del model
    torch.cuda.empty_cache()
    return P, t / len(X)


def run_deeponet(X, B=16):
    from deeponet_baseline import MEAN, STD, DeepONet
    m = DeepONet().cuda()
    m.load_state_dict(torch.load(os.path.join(HERE, "deeponet/deeponet.pth"),
                                 map_location="cuda")["model_state_dict"])
    m.eval()
    mean, std = torch.from_numpy(MEAN).cuda(), torch.from_numpy(STD).cuda()
    outs, t = [], 0.0
    with torch.no_grad():
        for i in range(0, len(X), B):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + B, ..., :2])).cuda()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            yb = m((xb - mean) / std) * std + mean
            torch.cuda.synchronize(); t += time.perf_counter() - t0
            outs.append(yb.float().cpu().numpy())
    uv = np.concatenate(outs)
    del m
    torch.cuda.empty_cache()
    return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32), t / len(X)


def run_unet(ckpt, X, B=16):
    from pde_refiner import MEAN, STD, UNet, from_ch, refiner_predict, to_ch
    blob = torch.load(ckpt, map_location="cuda")
    mode = blob.get("mode", "mse")
    m = UNet().cuda()
    m.load_state_dict(blob["model_state_dict"])
    m.eval()
    mean, std = torch.from_numpy(MEAN).cuda(), torch.from_numpy(STD).cuda()
    outs, t = [], 0.0
    with torch.no_grad():
        for i in range(0, len(X), B):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + B, ..., :2])).cuda()
            prev = to_ch((xb - mean) / std)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if mode == "refiner":
                y = refiner_predict(m, prev)
            else:
                k0 = torch.zeros(len(prev), dtype=torch.long, device="cuda")
                y = m(prev, torch.zeros_like(prev), k0)
            torch.cuda.synchronize(); t += time.perf_counter() - t0
            outs.append((from_ch(y) * std + mean).float().cpu().numpy())
    uv = np.concatenate(outs)
    del m
    torch.cuda.empty_cache()
    return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32), t / len(X)


def main() -> None:
    sets = {}
    for tag, path in [("unseen", "val_unseen.npz"), ("seen", "val_seen.npz")]:
        d = np.load(os.path.join(HERE, path))
        sets[tag] = (d["x"], d["y"])

    models = [
        ("FNO sim_real_ft", lambda X: run_official("fno", "sim_real_fno_fp16.pth", X), "train"),
        ("CNO sim_real_ft", lambda X: run_official("cno", "sim_real_cno.pth", X), "train"),
        ("DeepONet", lambda X: run_deeponet(X), "ours"),
        ("U-Net (mse)", lambda X: run_unet(os.path.join(HERE, "diffusion/unet_mse.pth"), X), "ours"),
        ("PDE-Refiner", lambda X: run_unet(os.path.join(HERE, "diffusion/unet_refiner.pth"), X), "ours"),
    ]

    res = {}
    for name, fn, _kind in models:
        res[name] = {}
        for tag, (X, Y) in sets.items():
            try:
                P, t = fn(X)
            except FileNotFoundError:
                print(f"(skipped {name}: checkpoint missing)")
                break
            res[name][tag] = subscores(P, Y, t)

    hdr = f"{'model':18s} {'set':7s} {'relL2':>6s} {'TKE':>6s} {'MVPE':>6s} | {'rel':>5s} {'tke':>5s} {'mvpe':>5s} {'time':>5s} {'sps':>5s} | {'FINAL':>6s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, fn, kind in models:
        r = res.get(name, {})
        if not r:
            continue
        for tag in ("unseen", "seen"):
            if tag in r:
                v = r[tag]
                print(f"{name:18s} {tag:7s} {v['rel_err']:6.3f} {v['tke_err']:6.3f} {v['mvpe_err']:6.3f} | "
                      f"{v['rel']:5.1f} {v['tke']:5.1f} {v['mvpe']:5.1f} {v['time']:5.1f} {v['sps']:5.1f} | {v['final']:6.2f}")
        if "unseen" in r and "seen" in r:
            gap = r["seen"]["final"] - r["unseen"]["final"]
            note = "  <- both sets are TRAINING data (difficulty control)" if kind == "train" else "  <- generalization gap"
            print(f"{'':18s} {'delta':7s} {'':>6s} {'':>6s} {'':>6s} | {'':>5s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} | {gap:+6.2f}{note}")
        print()


if __name__ == "__main__":
    main()
