"""Where the 3.2 ms floor of a batch-1 Track-2 step goes, and what removes it.

`t2_latency.py` measured a floor of ~3.18 ms/step for a SINGLE member and +2.4 ms per extra
invocation, with extra members inside a fused group costing ~0.09 ms.  A single 0.9M FNO
forward on 32x64 is a few hundred microseconds of arithmetic, so most of that floor is
launch overhead and host transfers -- i.e. it is engineering, not model cost, and it is worth
about 0.08 `final` per millisecond removed.

This decomposes the step and then tries the fixes, each on top of the last:
  1. as-is
  2. pinned staging buffers for the three device->host copies
  3. transfer `pred` and `half` only, and form lower/upper on the host
  4. CUDA graph capture of the whole device-side step

    python t2_profile.py [--ckpt ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "submit"))
import _fno  # noqa: E402

T_OUT, H, W, CH = 20, 32, 64, 40
U = str(HERE.parent / "Track_1" / "unet")
DEFAULT = [f"{U}/fno_all_vl_s{i}.pth" for i in range(1, 8)] + [f"{U}/fno_all_vl_m1220.pth"]
ALPHA = 0.90


def timeit(fn, n=300, warm=60):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=DEFAULT)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    dev = "cuda"
    models = _fno.fuse([_fno.load_any(c, dev) for c in a.ckpt])
    z = np.load(HERE / "submit" / "band.npz")
    band = torch.from_numpy(z["base"][..., :2].astype(np.float32)).to(dev)
    sref = torch.from_numpy(z["sigma_ref"][..., :2].astype(np.float32)).to(dev)
    p = float(z["p"])
    std = torch.from_numpy(_fno.STD).to(dev)
    print(f"{len(a.ckpt)} members -> {len(models)} invocations, band p={p}")

    x_host = torch.randn(1, 20, H, W, 3)
    x_dev = x_host.to(dev)
    prev = x_dev[..., :2].permute(0, 1, 4, 2, 3).reshape(1, CH, H, W).contiguous()

    def fwd():
        y = sig = None
        tot = 0
        for m in models:
            nm = getattr(m, "n_members", 1); tot += nm
            raw, ls = m(prev)
            yi = raw.reshape(1, T_OUT, 2, H, W).permute(0, 1, 3, 4, 2) * nm
            y = yi if y is None else y + yi
            si = ls.clamp(-8., 2.).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
            sig = si if sig is None else sig + si
        return y / tot, sig / tot

    with torch.no_grad():
        print("\n--- decomposition (each measured alone) ---")
        print(f"  H2D input                 {timeit(lambda: x_host.to(dev, non_blocking=True), a.n):7.3f} ms")
        print(f"  permute/reshape/contig    {timeit(lambda: x_dev[..., :2].permute(0,1,4,2,3).reshape(1,CH,H,W).contiguous(), a.n):7.3f} ms")
        print(f"  ensemble forward          {timeit(fwd, a.n):7.3f} ms")
        y0, s0 = fwd()

        def compose():
            m_ = y0.mean(1, keepdim=True)
            s_ = y0.std(1, keepdim=True, unbiased=False)
            return m_ + (y0 - m_) * ((ALPHA * s0 + (1 - ALPHA) * s_) / (s_ + 1e-6))
        print(f"  compose                   {timeit(compose, a.n):7.3f} ms")
        yc = compose()

        def bandop():
            r = torch.clamp((s0 * std) / sref, 0.3, 3.0) ** p
            return band * r
        print(f"  band                      {timeit(bandop, a.n):7.3f} ms")
        half = bandop()
        out = torch.zeros(1, T_OUT, H, W, 3, device=dev)
        out[..., :2] = yc
        print(f"  1x D2H (1,20,32,64,3)     {timeit(lambda: out.to('cpu', copy=True), a.n):7.3f} ms")
        pin = torch.empty(1, T_OUT, H, W, 3, pin_memory=True)
        print(f"  1x D2H pinned             {timeit(lambda: pin.copy_(out, non_blocking=False), a.n):7.3f} ms")

        # ---------------- whole-step variants -------------------------------------
        print("\n--- whole step ---")
        o = torch.zeros(1, T_OUT, H, W, 3, device=dev)
        lo = torch.zeros_like(o); hi = torch.zeros_like(o)

        def step_asis():
            xx = x_host.to(dev, non_blocking=True)
            pv = xx[..., :2].permute(0, 1, 4, 2, 3).reshape(1, CH, H, W).contiguous()
            y = sg = None; tot = 0
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
            o[..., :2] = y; lo[..., :2] = y - hf; hi[..., :2] = y + hf
            return o.to("cpu", copy=True), lo.to("cpu", copy=True), hi.to("cpu", copy=True)
        print(f"  1 as-is                   {timeit(step_asis, a.n):7.3f} ms")

        pins = [torch.empty(1, T_OUT, H, W, 3, pin_memory=True) for _ in range(3)]

        def step_pinned():
            xx = x_host.to(dev, non_blocking=True)
            pv = xx[..., :2].permute(0, 1, 4, 2, 3).reshape(1, CH, H, W).contiguous()
            y = sg = None; tot = 0
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
            o[..., :2] = y; lo[..., :2] = y - hf; hi[..., :2] = y + hf
            for pn, src in zip(pins, (o, lo, hi)):
                pn.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            return pins
        print(f"  2 + pinned D2H            {timeit(step_pinned, a.n):7.3f} ms")

        pp = torch.empty(1, T_OUT, H, W, 3, pin_memory=True)
        ph = torch.empty(1, T_OUT, H, W, 2, pin_memory=True)

        def step_2xfer():
            xx = x_host.to(dev, non_blocking=True)
            pv = xx[..., :2].permute(0, 1, 4, 2, 3).reshape(1, CH, H, W).contiguous()
            y = sg = None; tot = 0
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
            o[..., :2] = y
            pp.copy_(o, non_blocking=True); ph.copy_(hf, non_blocking=True)
            torch.cuda.synchronize()
            l = pp.clone(); u = pp.clone()
            l[..., :2] -= ph; u[..., :2] += ph
            return pp, l, u
        print(f"  3 + 2 transfers, host +-  {timeit(step_2xfer, a.n):7.3f} ms")

        # ---- 4. CUDA graph -------------------------------------------------------
        try:
            gx = torch.zeros(1, 20, H, W, 3, device=dev)
            go = torch.zeros(1, T_OUT, H, W, 3, device=dev)
            gl = torch.zeros_like(go); gh = torch.zeros_like(go)

            def body():
                pv = gx[..., :2].permute(0, 1, 4, 2, 3).reshape(1, CH, H, W).contiguous()
                y = sg = None; tot = 0
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

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    body()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            gx.copy_(x_host.to(dev)); g.replay(); torch.cuda.synchronize()
            body_out = go.clone()

            def step_graph():
                gx.copy_(x_host, non_blocking=True)
                g.replay()
                for pn, src in zip(pins, (go, gl, gh)):
                    pn.copy_(src, non_blocking=True)
                torch.cuda.synchronize()
                return pins
            t = timeit(step_graph, a.n)
            # correctness: graph output vs eager
            gx.copy_(x_host.to(dev)); g.replay(); torch.cuda.synchronize()
            eager = step_asis()[0].to(dev)
            err = float((go - eager).abs().max())
            print(f"  4 + CUDA graph            {t:7.3f} ms   (max|graph-eager| {err:.2e})")
        except Exception as e:                                  # noqa: BLE001
            print(f"  4 CUDA graph FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
