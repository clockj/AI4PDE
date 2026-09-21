"""Track 2 per-step latency, taken apart at batch 1 on a shipped archive.

    python Track_2/step_breakdown.py --dir EXTRACTED_T2_DIR        # GPU; an idle node for evidence

WHAT IT MEASURES.  One `ttt_step` is (i) a host->device copy of the input into the graph's static
buffer, (ii) one `graph.replay()` of the whole device step, (iii) handing three tensors back.
Those three are timed end to end and separately.  Then the device step is cut into its stages --
input preparation, each fused member group's forward (StackedFNO, StackedViT) under the shipped
AMP, and the post stage (member average, compose, target affine, band) -- and every stage is
timed on its own, eager and as its own captured CUDA graph, with:

  kernels   CUDA kernel launches in one eager call (torch.profiler)
  eager-graph  what the graph removes: host-side dispatch, which scales with the kernel count
  b4/b1     graph ms at batch 4 over batch 1: ~4 means the stage is FLOP-bound on this GPU,
            ~1 means its cost is per launch, not per element
  fp32      graph ms with AMP off: a FLOP-bound stage on tensor cores speeds up under fp16

Numbers are A100 milliseconds; only same-process comparisons mean anything, and none is evidence
while other jobs share the node.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time

import numpy as np
import torch

T, H, W = 20, 32, 64


def _load(d: str):
    spec = importlib.util.spec_from_file_location("t2_breakdown", os.path.join(d, "submission.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def t_ms(fn, n=300, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


def kernels(fn):
    try:
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity, profile
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return sum(1 for e in prof.events() if getattr(e, "device_type", None) == DeviceType.CUDA)
    except Exception:
        return -1


def graphed(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def stages(sub, fe, b, amp):
    """Stage callables over static buffers at batch b; outputs are written in place."""
    dev = fe.gx.device
    x = torch.randn((b, T, H, W, 3), device=dev) * 0.5
    pv = torch.zeros((b, 2 * T, H, W), device=dev)
    models = fe.models
    with torch.no_grad():
        pv.copy_(x[..., :2].permute(0, 1, 4, 2, 3).reshape(b, 2 * T, H, W))
        shapes = [tuple(t.shape for t in m(pv)) for m in models]
    raw = [torch.zeros(s[0], device=dev) for s in shapes]
    ls = [torch.zeros(s[1], device=dev) for s in shapes]
    go = torch.zeros((b, T, H, W, 2), device=dev)
    gl, gh = torch.zeros_like(go), torch.zeros_like(go)

    def s_prep():
        pv.copy_(x[..., :2].permute(0, 1, 4, 2, 3).reshape(b, 2 * T, H, W))

    def s_group(i):
        def f():
            if amp is None:
                r, l_ = models[i](pv)
            else:
                with torch.autocast("cuda", dtype=amp):
                    r, l_ = models[i](pv)
            raw[i].copy_(r.float())
            ls[i].copy_(l_.float())
        return f

    def s_post():
        y = sig = None
        total = 0
        for i, m in enumerate(models):
            nm = getattr(m, "n_members", 1)
            total += nm
            yi = raw[i].reshape(b, T, 2, H, W).permute(0, 1, 3, 4, 2) * nm
            y = yi if y is None else y + yi
            si = ls[i].clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
            sig = si if sig is None else sig + si
        y, sig = y / total, sig / total
        m_ = y.mean(1, keepdim=True)
        s_ = y.std(1, keepdim=True, unbiased=False)
        y = m_ + (y - m_) * ((sub.ALPHA * sig + (1 - sub.ALPHA) * s_) / (s_ + 1e-6))
        go.copy_(y * fe.Ad + fe.Bd)
        if fe.half0 is not None:
            half = fe.half0
            if fe.band_p and fe.sref is not None:
                half = half * torch.clamp((sig * fe.std) / fe.sref, 0.3, 3.0) ** fe.band_p
            torch.sub(go, half, out=gl)
            torch.add(go, half, out=gh)

    names = ["input prep"] + [f"group {i}: {type(m).__name__}({getattr(m, 'n_members', 1)})"
                              for i, m in enumerate(models)] + ["average+compose+band"]
    return names, [s_prep] + [s_group(i) for i in range(len(models))] + [s_post]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    sub = _load(a.dir)
    fe = sub.get_ttt_model(a.dir, "cuda")
    amp = sub._AMP
    print(f"archive {a.dir}\n  AMP_ACTIVE {sub.AMP_ACTIVE}   graph {'ACTIVE' if fe.graph is not None else 'DROPPED'}   "
          f"groups {[type(m).__name__ + '(' + str(getattr(m, 'n_members', 1)) + ')' for m in fe.models]}")

    xc = (torch.randn((1, T, H, W, 3)) * 0.5).float()
    xg = xc.cuda()
    fe.reset_ttt_state()
    print("\nwhole step (median ms)")
    rows = [("ttt_step, CPU input (host copy out)", lambda: fe.ttt_step(xc)),
            ("ttt_step, CUDA input (device clone out)", lambda: fe.ttt_step(xg)),
            ("  input copy, CPU -> static buffer", lambda: fe.gx.copy_(xc)),
            ("  input copy, CUDA -> static buffer", lambda: fe.gx.copy_(xg))]
    if fe.graph is not None:
        rows.append(("  graph.replay (the whole device step)", fe.graph.replay))
    rows += [("  eager device step (_body)", fe._body),
             ("  outputs back, CPU (pinned copy + clone)", lambda: fe._fetch((fe.go, fe.gl, fe.gh), False)),
             ("  outputs back, CUDA (clone)", lambda: fe._fetch((fe.go, fe.gl, fe.gh), True))]
    for nm, fn in rows:
        print(f"  {nm:44s} {t_ms(fn):8.3f}")
    print(f"  eager device step kernels: {kernels(fe._body)}")

    print("\nstages of the device step (median ms)")
    print(f"  {'stage':34s} {'eager':>8s} {'graph':>8s} {'eager-graph':>11s} {'kernels':>8s} "
          f"{'graph b4/b1':>11s} {'graph fp32':>10s}")
    n1, st1 = stages(sub, fe, 1, amp)
    _, st4 = stages(sub, fe, 4, amp)
    _, st32 = stages(sub, fe, 1, None)
    tot = [0.0, 0.0]
    for nm, f1, f4, f32 in zip(n1, st1, st4, st32):
        e, g = t_ms(f1), t_ms(graphed(f1))
        g4, g32 = t_ms(graphed(f4)), t_ms(graphed(f32))
        tot[0] += e
        tot[1] += g
        print(f"  {nm:34s} {e:8.3f} {g:8.3f} {e - g:11.3f} {kernels(f1):8d} {g4 / max(g, 1e-9):11.2f} {g32:10.3f}")
    print(f"  {'sum of stages':34s} {tot[0]:8.3f} {tot[1]:8.3f} {tot[0] - tot[1]:11.3f}")


if __name__ == "__main__":
    main()
