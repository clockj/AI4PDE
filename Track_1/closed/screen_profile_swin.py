"""Where Swin's Track-2 step goes (main 09-14, chunk 2b). A profile and cheap EXACT rewrites,
measured without changing the model file.

    python Track_1/screen_profile_swin.py --ckpt SCREEN/w1/w1_swin_unet_s1.pth [--out JSON]

Track 2 ships a CUDA-graph-captured batch-1 step, so launch overhead is gone and what remains is
GPU kernel time -- which, for tensors this small, is dominated by the NUMBER of kernels (each
costs a fixed few-hundred-microsecond floor or more on the device), not their FLOPs. Rows,
interleaved in one process like screen_latency.py:
  fno_ref       the vendored FNO w96 L8 (ratio base)
  swin          SwinUNet w64 L2 exactly as trained
  swin_cache    EXACT: logit scale, continuous position bias and the shifted-window mask + bias
                sum precomputed once per (batch, dtype) instead of on every call
  swin_gather   EXACT: swin_cache + roll -> partition and merge -> unroll each done as ONE
                precomputed index_select
  swin_math     swin_gather with SDPA pinned to the math kernel
  abl_noskip    ABLATION (changes the output): the 4 ConvNeXt skip blocks removed
  abl_norefine  ABLATION: the final residual 3x3 refine removed
  abl_noshift   ABLATION: every block unshifted (no roll, no mask)
Per row: CUDA-graph replay at batch 1 (fp16, fp32), eager batch 64 fp16, eager batch 1 fp16; the
CUDA kernels one batch-1 forward launches (torch.profiler); max |output - swin| for the exact
rewrites; and the top operators by device time for swin and fno_ref.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import contextlib
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
sys.path.insert(0, os.path.join(HERE, "unet"))

import _fno  # noqa: E402
import models_alt  # noqa: E402
from models_alt import _SwinV2Block, _WindowAttnV2, _win_part  # noqa: E402

OUT = os.path.join(HERE, "plan", "screen_swin_profile_20260914.json")
SETTINGS = [("b1g", True), ("b1g", False), ("b64", True), ("b1", True)]


# ------------------------------------------------------------------ exact rewrites (bound per module)
def _attn_cached(self: _WindowAttnV2, x, mask=None):
    bw, n, c = x.shape
    q, k, v = (self.qkv(x).reshape(bw, n, 3, self.heads, c // self.heads)
               .permute(2, 0, 3, 1, 4).unbind(0))
    dt = v.dtype
    key = (bw, dt, mask is not None)
    if key not in self._pc:
        with torch.no_grad():
            scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp().float()
            bias = (self.cpb(self.coords)[self.rel_index]
                    .reshape(n, n, self.heads).permute(2, 0, 1)).float()
            if mask is None:
                am = bias.unsqueeze(0)
            else:
                nw = mask.shape[0]
                am = ((bias[None, None] + mask[None]).expand(bw // nw, nw, self.heads, n, n)
                      .reshape(bw, self.heads, n, n))
            self._pc[key] = (scale, am.to(dt).contiguous())
    scale, am = self._pc[key]
    q = F.normalize(q, dim=-1) * scale
    k = F.normalize(k, dim=-1)
    o = F.scaled_dot_product_attention(q.to(dt), k.to(dt), v, attn_mask=am, dropout_p=0.0,
                                       scale=1.0)
    return self.proj_drop(self.proj(o.transpose(1, 2).reshape(bw, n, c)))


def _block_gather(self: _SwinV2Block, x):
    b, h, w, c = x.shape
    if self._gi is None:
        sy, sx = self.shift
        ar = torch.arange(h * w, device=x.device, dtype=torch.float32).reshape(1, h, w, 1)
        r = torch.roll(ar, (-sy, -sx), (1, 2)) if (sy or sx) else ar
        self._gi = _win_part(r, *self.win).reshape(-1).long()
        self._ginv = torch.argsort(self._gi)
    wh, ww = self.win
    win = x.reshape(b, h * w, c).index_select(1, self._gi).reshape(-1, wh * ww, c)
    y = self.attn(win, self.mask)
    y = y.reshape(b, h * w, c).index_select(1, self._ginv).reshape(b, h, w, c)
    x = x + self.norm1(y)
    return x + self.norm2(self.mlp(x))


def _swin_forward(self, prev, detach_var: bool = False):
    """SwinUNet.forward, verbatim, with two ablation switches (checked equal to the class when off)."""
    x = self.embed_norm(self.embed(self.cat(prev)).permute(0, 2, 3, 1))
    skips = []
    for blocks, merge, sblk in zip(self.enc, self.merge, self.skip_blocks):
        for blk in blocks:
            x = blk(x)
        skips.append(x if self._noskip else sblk(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        x = merge(x)
    for blk in self.bottleneck:
        x = blk(x)
    for expand, skip, blocks, s in zip(self.expand, self.skip, self.dec, reversed(skips)):
        x = skip(torch.cat([expand(x), s], -1))
        for blk in blocks:
            x = blk(x)
    h = self.final(x).permute(0, 3, 1, 2)
    if not self._norefine:
        h = h + self.refine(h)
    logstd = self.head_logstd(h) if self.head_logstd is not None else None
    return self.head_field(h), logstd


class _MathSDPA(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                            enable_mem_efficient=False):
            return self.m(x)


def variants(swin):
    import types
    out = {"swin": swin}

    def cached(m):
        for mod in m.modules():
            if isinstance(mod, _WindowAttnV2):
                mod._pc = {}
                mod.forward = types.MethodType(_attn_cached, mod)
        return m

    def gathered(m):
        for mod in m.modules():
            if isinstance(mod, _SwinV2Block):
                mod._gi = None
                mod.forward = types.MethodType(_block_gather, mod)
        return m

    def ablate(m, noskip=False, norefine=False, noshift=False):
        m._noskip, m._norefine = noskip, norefine
        m.forward = types.MethodType(_swin_forward, m)
        if noshift:
            for mod in m.modules():
                if isinstance(mod, _SwinV2Block):
                    mod.shift, mod.mask = (0, 0), None
        return m

    out["swin_cache"] = cached(copy.deepcopy(swin))
    out["swin_gather"] = gathered(cached(copy.deepcopy(swin)))
    out["swin_math"] = _MathSDPA(gathered(cached(copy.deepcopy(swin))))
    if hasattr(swin, "prepare_inference"):
        # the model-level speedups as shipped in models_alt (exact by construction; checked below)
        out["swin_prep16"] = copy.deepcopy(swin).prepare_inference(fp32_max_batch=0)
        out["swin_prep32b1"] = copy.deepcopy(swin).prepare_inference(fp32_max_batch=1)
        out["swin_prep32"] = copy.deepcopy(swin).prepare_inference(fp32_max_batch=1 << 20)
    out["_swin_replica"] = ablate(copy.deepcopy(swin))
    out["abl_noskip"] = ablate(copy.deepcopy(swin), noskip=True)
    out["abl_norefine"] = ablate(copy.deepcopy(swin), norefine=True)
    out["abl_noshift"] = ablate(copy.deepcopy(swin), noshift=True)
    return out


# ------------------------------------------------------------------ timing / profiling
def _amp(on):
    return (torch.autocast("cuda", dtype=torch.float16, cache_enabled=False) if on
            else contextlib.nullcontext())


@torch.no_grad()
def fwd(m, x, amp):
    with _amp(amp):
        return m(x)


@torch.no_grad()
def capture(m, amp):
    sx = torch.randn(1, 40, 32, 64, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fwd(m, sx, amp)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fwd(m, sx, amp)
    return g


@torch.no_grad()
def kernels(m, x, amp):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fwd(m, x, amp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fwd(m, x, amp)
        torch.cuda.synchronize()
    n_k = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    top = sorted(((e.key, e.count, e.self_cuda_time_total / 1e3) for e in prof.key_averages()
                  if e.self_cuda_time_total > 0), key=lambda t: -t[2])[:18]
    dev_ms = sum(e.self_cuda_time_total for e in prof.key_averages()) / 1e3
    return n_k, round(dev_ms, 3), [dict(op=k, count=c, self_cuda_ms=round(t, 4)) for k, c, t in top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    swin = _fno.load_any(args.ckpt, dev).eval()
    rows = {"fno_ref": _fno.load_any(os.path.join(HERE, "unet", "fno_r18_cmb3sim.pth"), dev).eval()}
    rows.update(variants(swin))
    x1 = torch.randn(1, 40, 32, 64, device=dev)
    x64 = torch.randn(64, 40, 32, 64, device=dev)

    exact = {}
    with torch.no_grad():
        ref32, ref16 = fwd(swin, x64, False)[0], fwd(swin, x64, True)[0].float()
        rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
        # context for the fp16 column: how far fp16 itself moves the as-trained model
        exact["swin_fp16_vs_fp32"] = dict(max=float((ref16 - ref32).abs().max()), rel=rel(ref16, ref32))
        for name in [n_ for n_ in ("swin_cache", "swin_gather", "swin_math", "swin_prep16",
                                   "swin_prep32b1", "swin_prep32", "_swin_replica") if n_ in rows]:
            o32, o16 = fwd(rows[name], x64, False)[0], fwd(rows[name], x64, True)[0].float()
            exact[name] = dict(fp32=float((o32 - ref32).abs().max()), fp32_rel=rel(o32, ref32),
                               fp16=float((o16 - ref16).abs().max()), fp16_rel=rel(o16, ref16))
        if "swin_prep32" in rows:
            # fp32_max_batch forces fp32 under autocast: must equal the plain fp32 forward bitwise
            exact["swin_prep32_autocast_vs_swin_fp32"] = float((fwd(rows["swin_prep32"], x64, True)[0].float() - ref32).abs().max())
            one = x64[:1]
            exact["swin_prep32b1_b1_autocast_vs_swin_fp32"] = float((fwd(rows["swin_prep32b1"], one, True)[0].float() - fwd(swin, one, False)[0]).abs().max())
    replica = rows.pop("_swin_replica")
    del replica
    print("exactness vs swin (max |d field|):", json.dumps(exact))
    assert exact["_swin_replica"]["fp32"] == 0.0, "the ablation forward is not the class forward"

    runs, niter = {}, {}
    for mode, amp in SETTINGS:
        for name, m in rows.items():
            if mode == "b1g":
                g = capture(m, amp)
                run = g.replay
            else:
                x = x1 if mode == "b1" else x64
                run = (lambda m=m, x=x, amp=amp: fwd(m, x, amp))
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            t = time.perf_counter()
            run()
            torch.cuda.synchronize()
            niter[(name, mode, amp)] = int(min(1000, max(3, math.ceil(0.08 / (time.perf_counter() - t)))))
            runs[(name, mode, amp)] = run
    acc = defaultdict(list)
    for r in range(args.rounds):
        for mode, amp in SETTINGS:
            for name in rows:
                k = (name, mode, amp)
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(niter[k]):
                    runs[k]()
                torch.cuda.synchronize()
                acc[k].append((time.perf_counter() - t) / niter[k])
        print(f"round {r + 1}/{args.rounds}", flush=True)

    out_rows = {}
    for name, m in rows.items():
        rec = {}
        for mode, amp in SETTINGS:
            v = acc[(name, mode, amp)][1:]
            b = 64 if mode == "b64" else 1
            rec[f"ms_{mode}_{'fp16' if amp else 'fp32'}"] = round(1e3 * float(np.median(v)) / b, 5)
        rec["kernels_b1_fp16"], rec["device_ms_b1_fp16_eager"], top16 = kernels(m, x1, True)
        rec["kernels_b1_fp32"], _, _ = kernels(m, x1, False)
        if name in ("swin", "fno_ref"):
            rec["top_ops_b1_fp16"] = top16
        out_rows[name] = rec
    # STACK PROXY for P1 on Track 2: one Swin at batch 4 (CUDA graph) stands in for 4 fused members
    # at batch 1 (a StackedSwin runs the same kernels on 4x the tokens, plus bmm overhead), against
    # the shipped FNO x4 fused group at batch 1 -- interleaved in this process.
    stack = {}
    if "swin_prep16" in rows:
        load = lambda n_: _fno.load_any(os.path.join(HERE, "unet", n_), dev)  # noqa: E731
        _fno._SPEC = "bmm"
        f4 = _fno.fuse([load(f"fno_all_w96L8cmb3sim_s{i}.pth") for i in (1, 2, 3, 4)])[0]
        cand = {"swin_b4": (rows["swin"], True), "swin_prep16_b4": (rows["swin_prep16"], True),
                "swin_prep32_b4": (rows["swin_prep32"], True)}
        x4 = torch.randn(4, 40, 32, 64, device=dev)

        @torch.no_grad()
        def cap4(m, amp):
            s_ = torch.cuda.Stream()
            with torch.cuda.stream(s_):
                for _ in range(3):
                    fwd(m, x4, amp)
            torch.cuda.current_stream().wait_stream(s_)
            g_ = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_):
                fwd(m, x4, amp)
            return g_

        graphs = {k: cap4(m, amp) for k, (m, amp) in cand.items()}
        graphs["pkg_fno4_fused_b1"] = capture(f4, True)
        acc4 = defaultdict(list)
        for r in range(args.rounds):
            for k, g_ in graphs.items():
                for _ in range(3):
                    g_.replay()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(200):
                    g_.replay()
                torch.cuda.synchronize()
                acc4[k].append((time.perf_counter() - t) / 200)
        stack = {k: round(1e3 * float(np.median(v[1:])), 4) for k, v in acc4.items()}
        best = min(("swin_prep16_b4", "swin_prep32_b4"), key=lambda k: stack[k])
        stack.update(swin_prep_best_b4g_ms=stack[best], swin_prep_best_name=best,
                     pkg_fno4_fused_b1g_fp16_ms=stack["pkg_fno4_fused_b1"],
                     note="per replay of the whole batch-4 step (4 windows) / FNO x4 fused step; ms")
        print("stack proxy (ms per step):", json.dumps(stack))
    base = out_rows["fno_ref"]
    for rec in out_rows.values():
        for k in [k for k in rec if k.startswith("ms_")]:
            rec["ratio_" + k[3:]] = round(rec[k] / base[k], 3)
    sw = out_rows["swin"]
    struct = dict(
        blocks=sum(isinstance(m_, _SwinV2Block) for m_ in swin.modules()),
        convnext_skip_blocks=sum(len(s) for s in swin.skip_blocks),
        stages=[dict(tokens=f"{16 >> i}x{32 >> i}", dim=64 * 2 ** i,
                     window=f"{min(8, 16 >> i)}x{min(8, 32 >> i)}",
                     windows=(16 >> i) // min(8, 16 >> i) * ((32 >> i) // min(8, 32 >> i)))
                for i in range(3)])
    savings = {name: dict(b1g_fp16_ms=round(sw["ms_b1g_fp16"] - rec["ms_b1g_fp16"], 4),
                          b1g_fp32_ms=round(sw["ms_b1g_fp32"] - rec["ms_b1g_fp32"], 4),
                          kernels_fp16=sw["kernels_b1_fp16"] - rec["kernels_b1_fp16"])
               for name, rec in out_rows.items() if name not in ("swin", "fno_ref")}
    result = dict(date=time.strftime("%Y-%m-%d %H:%M"), host=os.uname().nodename,
                  job=os.environ.get("SLURM_JOB_ID"), ckpt=os.path.abspath(args.ckpt),
                  structure=struct, rows=out_rows, exactness=exact, savings_vs_swin=savings,
                  stack_proxy=stack,
                  note="b1g = CUDA-graph replay (Track 2's shipped form); kernels = CUDA kernels "
                       "launched by one eager batch-1 forward (torch.profiler)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)

    print(f"\nstructure: {json.dumps(struct)}")
    print(f"{'row':14s} {'b1g fp16':>9s} {'x FNO':>6s} {'b1g fp32':>9s} {'b64 fp16':>9s} {'b1 eager':>9s} "
          f"{'kern16':>7s} {'kern32':>7s}")
    for name, rec in out_rows.items():
        print(f"{name:14s} {rec['ms_b1g_fp16']:9.3f} {rec['ratio_b1g_fp16']:6.2f} {rec['ms_b1g_fp32']:9.3f} "
              f"{rec['ms_b64_fp16']:9.4f} {rec['ms_b1_fp16']:9.2f} {rec['kernels_b1_fp16']:7d} "
              f"{rec['kernels_b1_fp32']:7d}")
    print("\nsavings vs swin:", json.dumps(savings, indent=1))
    for name in ("swin", "fno_ref"):
        print(f"\ntop device ops, {name}, eager b1 fp16:")
        for t in out_rows[name]["top_ops_b1_fp16"]:
            print(f"  {t['self_cuda_ms']:8.3f} ms  x{t['count']:4d}  {t['op']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
