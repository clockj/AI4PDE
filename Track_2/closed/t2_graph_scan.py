"""Does CUDA-graph capture make extra INVOCATIONS free at batch 1?

`t2_latency.py` measured +2.4 ms per invocation eagerly, which is what made the Track-1
11-member / 4-invocation package cost 10.6 ms a step against a fused 8-member group's 3.8.
`t2_profile.py` then showed the whole eager step is launch overhead: capturing it in a CUDA
graph took 3.70 ms to 1.04 ms, bit-exact.  If a graph also flattens the per-invocation cost,
the member/latency trade Track 2 inherited from Track 1 stops existing and the accuracy-optimal
member set is simply the biggest one.

    python t2_graph_scan.py
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
sys.path.insert(0, str(HERE / "submit"))
import _fno  # noqa: E402

KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)

T_OUT, H, W, CH = 20, 32, 64, 40
ALPHA = 0.90
U = str(HERE.parent / "Track_1" / "unet")
ALL = [f"{U}/fno_all_vl_s{i}.pth" for i in range(1, 8)] + [
    f"{U}/fno_all_vl_m1220.pth", f"{U}/fno_all_vl_w48.pth",
    f"{U}/fno_all_vl_w64.pth", f"{U}/unet_all_g1.pth"]
CONFIGS = {
    "r12_11memb": ALL,
    "fused10": ALL[:8] + ALL[9:],
    "fused9": ALL[:8] + ALL[10:],
    "fused8": ALL[:8],
    "single": ALL[:1],
}


def build(models, band, sref, p, std, dev):
    gx = torch.zeros(1, 20, H, W, 3, device=dev)
    go = torch.zeros(1, T_OUT, H, W, 3, device=dev)
    gl, gh = torch.zeros_like(go), torch.zeros_like(go)

    def body():
        pv = gx[..., :2].permute(0, 1, 4, 2, 3).reshape(1, CH, H, W).contiguous()
        y = sg = None
        tot = 0
        for m in models:
            nm = getattr(m, "n_members", 1); tot += nm
            raw, ls = m(pv)
            yi = raw.reshape(1, T_OUT, 2, H, W).permute(0, 1, 3, 4, 2) * nm
            y = yi if y is None else y + yi
            si = ls.clamp(-8., 2.).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
            sg = si if sg is None else sg + si
        y, sg = y / tot, sg / tot
        m_ = y.mean(1, keepdim=True); s_ = y.std(1, keepdim=True, unbiased=False)
        y = m_ + (y - m_) * ((ALPHA * sg + (1 - ALPHA) * s_) / (s_ + 1e-6))
        hf = band * torch.clamp((sg * std) / sref, 0.3, 3.0) ** p
        go[..., :2] = y; gl[..., :2] = y - hf; gh[..., :2] = y + hf
    return gx, go, gl, gh, body


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    dev = "cuda"
    z = np.load(HERE / "submit" / "band.npz")
    band = torch.from_numpy(z["base"][..., :2].astype(np.float32)).to(dev)
    sref = torch.from_numpy(z["sigma_ref"][..., :2].astype(np.float32)).to(dev)
    p = float(z["p"]); std = torch.from_numpy(_fno.STD).to(dev)
    x_host = torch.randn(1, 20, H, W, 3)
    pins = [torch.empty(1, T_OUT, H, W, 3, pin_memory=True) for _ in range(3)]

    print(f"{'config':12s} {'memb':>5s} {'inv':>4s} {'eager ms':>9s} {'graph ms':>9s} "
          f"{'t_eager':>8s} {'t_graph':>8s} {'exact':>10s}")
    for name, cks in CONFIGS.items():
        models = _fno.fuse([_fno.load_any(c, dev) for c in cks])
        with torch.no_grad():
            gx, go, gl, gh, body = build(models, band, sref, p, std, dev)
            gx.copy_(x_host.to(dev))
            for _ in range(20):
                body()
            torch.cuda.synchronize()
            ref = go.clone()

            def eager():
                gx.copy_(x_host, non_blocking=True)
                body()
                for pn, src in zip(pins, (go, gl, gh)):
                    pn.copy_(src, non_blocking=True)
                torch.cuda.synchronize()
            for _ in range(40):
                eager()
            torch.cuda.synchronize(); t0 = perf_counter()
            for _ in range(a.n):
                eager()
            te = (perf_counter() - t0) / a.n

            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    body()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()

            def graphed():
                gx.copy_(x_host, non_blocking=True)
                g.replay()
                for pn, src in zip(pins, (go, gl, gh)):
                    pn.copy_(src, non_blocking=True)
                torch.cuda.synchronize()
            for _ in range(40):
                graphed()
            torch.cuda.synchronize(); t0 = perf_counter()
            for _ in range(a.n):
                graphed()
            tg = (perf_counter() - t0) / a.n
            gx.copy_(x_host.to(dev)); g.replay(); torch.cuda.synchronize()
            err = float((go - ref).abs().max())
        print(f"{name:12s} {len(cks):5d} {len(models):4d} {te*1e3:9.3f} {tg*1e3:9.3f} "
              f"{sc.score_time(te):8.3f} {sc.score_time(tg):8.3f} {err:10.1e}")
        del models, g
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
