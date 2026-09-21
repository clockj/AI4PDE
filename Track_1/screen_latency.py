"""Backbone-screen latency: every candidate and the shipped references INTERLEAVED in one process.

    python Track_1/screen_latency.py --out LAT.json --cand cnext_unet:40:2 ffno:40:12:16:32 ...
    python Track_1/screen_latency.py --pick LAT.json          # the wave-1 size rule, per family

Plan 2026-09-14 section 4. These are FLOP-class SCREENING numbers, not board predictions: the
network forward only (normalization, compose, band and host write are identical for every family
and are left out), eval mode under no_grad, fp32 with TF32 off (as both submissions ship) and fp16
autocast, at
  b64   batch 64, eager -- Track-1-like (predict() is eager), A100 ms per WINDOW;
  b1    batch 1, eager -- launch-bound: every model costs ~5-14 ms here, whatever its FLOPs;
  b1g   batch 1 replayed from a CUDA graph -- Track-2-like (the Track-2 step ships graph-captured),
        A100 ms per STEP.
Only the RATIO to `ref_fno_w96L8` measured in the same process is evidence (the same package has
read 3.37 and 3.87 ms/step in two jobs). FLOPs come from torch.utils.flop_counter on CPU at batch
1, FFTs uncounted -- scout's method, which reads FNO w96 L8 at 0.525 GFLOPs and the ViT at 4.89x.
A100 latency and FLOPs disagree for overhead-bound families; the board's latency slope was
calibrated in A100 ms on FNO and ViT deltas, so both are reported.

Rows
  ref_fno_w96L8    _fno.load_any(fno_r18_cmb3sim.pth): the vendored, UNFUSED FNO2d -- ratio base
  ref_vit_d384L8   _fno.load_any(arch_vit_cmb3sim.pth): the vendored ViT2d
  pkg_fno4_fused   _fno.fuse of the four shipped all-data FNOs, SPEC_PATH "bmm" (xfam6's FNO group)
  pkg_vit2_fused   _fno.fuse of the two shipped all-data ViTs (xfam6's ViT group)
  train_fno_w96L8  models_alt.FNO2d exactly as the screen's FNO control trains (information)
  <arch>_w..L..    models_alt.build(arch, width, ..., dropout 0.3), random init
Params / MB of a fused row are summed over its members (a StackedViT holds weights as buffers).

Timing: one untimed warm-up per (row, setting) that also sizes the iteration count to ~80 ms;
then `--rounds` rounds, each looping setting -> row, so rows of one setting are adjacent in time;
the first round is discarded and the median of the rest is reported. cudnn.benchmark on.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
sys.path.insert(0, os.path.join(HERE, "unet"))

import _fno  # noqa: E402
import models_alt  # noqa: E402

A100_TO_4060 = 1.850          # CLAUDE.md: A100 local ms x 1.850 = 4060-equivalent
UNET = os.path.join(HERE, "unet")
SETTINGS = [("b64", False), ("b64", True), ("b1", False), ("b1", True), ("b1g", False),
            ("b1g", True)]
STACK4 = [("b4g", False), ("b4g", True)]      # --stack4: batch-4 CUDA graph, ms per STEP
BATCH = {"b64": 64, "b1": 1, "b1g": 1, "b4g": 4}
FNO4 = [f"fno_all_w96L8cmb3sim_s{i}.pth" for i in (1, 2, 3, 4)]
VIT2 = ["arch_vit_all_s1.pth", "arch_vit_all_cmb3sim_s3.pth"]


def _key(mode: str, amp: bool) -> str:
    return f"ms_{mode}_{'fp16' if amp else 'fp32'}"


def _int8_mb(state: dict) -> float:
    """Archive MB through the REAL packer (int8, --linear tensor), as build_zip ships members."""
    sd = _fno.pack_state_dict({k: v.detach().cpu() for k, v in state.items()},
                              int8=True, linear="tensor")
    return sum(v.numel() * v.element_size() for v in sd.values()) / 1e6


def _gflops(model_cpu: torch.nn.Module) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    model_cpu.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model_cpu(torch.randn(1, 40, 32, 64))
    return fc.get_total_flops() / 1e9


def _meta_of(members_cpu: list) -> dict:
    n = sum(sum(p.numel() for p in m.parameters()) for m in members_cpu)
    return dict(params_M=round(n / 1e6, 4), fp16_MB=round(2.0 * n / 1e6, 2),
                int8_MB=round(sum(_int8_mb(m.state_dict()) for m in members_cpu), 2),
                gflops_b1=round(sum(_gflops(m) for m in members_cpu), 4))


def references(dev: str):
    rows = []
    load = lambda n, d: _fno.load_any(os.path.join(UNET, n), d)  # noqa: E731
    rows.append(("ref_fno_w96L8", load("fno_r18_cmb3sim.pth", dev),
                 dict(arch="fno", width=96, n_layers=8, members=1,
                      **_meta_of([load("fno_r18_cmb3sim.pth", "cpu")]))))
    rows.append(("ref_vit_d384L8", load("arch_vit_cmb3sim.pth", dev),
                 dict(arch="vit", width=384, n_layers=8, members=1,
                      **_meta_of([load("arch_vit_cmb3sim.pth", "cpu")]))))
    _fno._SPEC = "bmm"
    g = _fno.fuse([load(n, dev) for n in FNO4])
    assert len(g) == 1, f"four FNOs fused into {len(g)} groups"
    rows.append(("pkg_fno4_fused", g[0], dict(arch="fno", width=96, n_layers=8, members=4,
                                              **_meta_of([load(n, "cpu") for n in FNO4]))))
    g = _fno.fuse([load(n, dev) for n in VIT2])
    assert len(g) == 1, f"two ViTs fused into {len(g)} groups"
    rows.append(("pkg_vit2_fused", g[0], dict(arch="vit", width=384, n_layers=8, members=2,
                                              **_meta_of([load(n, "cpu") for n in VIT2]))))
    build = lambda: models_alt.build("fno", 96, True, 0, 8, 12, 8, var_mlp=True,  # noqa: E731
                                     lift_mlp=True, dropout=0.45)
    torch.manual_seed(0)
    rows.append(("train_fno_w96L8", build().to(dev).eval(),
                 dict(arch="fno", width=96, n_layers=8, members=1, **_meta_of([build()]))))
    return rows


def candidates(specs: list[str], dev: str):
    """arch:width:layers[:modes_y:modes_x][:variant], variant in
    dense   FFNO2d.set_spectral_path("dense")                         (row name suffix _dense)
    prep    SwinUNet.prepare_inference(fp32_max_batch=1): b1 fp32, larger batches fp16   (_prep)
    prep32  SwinUNet.prepare_inference(fp32_max_batch=2**20): always fp32             (_prep32)"""
    rows = []
    for s in specs:
        p = s.split(":")
        arch, width, nl = p[0], int(p[1]), int(p[2])
        nums = [int(t) for t in p[3:] if t.isdigit()]
        opts = [t for t in p[3:] if not t.isdigit()]
        my, mx = (nums[0], nums[1]) if len(nums) >= 2 else (8, 12)
        torch.manual_seed(0)
        m = models_alt.build(arch, width, True, 0, my, mx, nl, dropout=0.3)
        meta = _meta_of([m])
        name = f"{arch}_w{width}L{nl}" + (f"_m{my}x{mx}" if arch.startswith("ffno") else "")
        m = m.to(dev).eval()
        for opt in opts:
            if opt == "dense":
                m.set_spectral_path("dense")
            elif opt == "prep":
                m.prepare_inference(fp32_max_batch=1)
            elif opt == "prep32":
                m.prepare_inference(fp32_max_batch=1 << 20)
            else:
                raise SystemExit(f"unknown variant {opt!r} in {s}")
            name += f"_{opt}"
        rows.append((name, m, dict(arch=arch, width=width, n_layers=nl, modes_y=my, modes_x=mx,
                                   members=1, variant="+".join(opts) or None, **meta)))
    return rows


def _amp_ctx(amp: bool):
    # cache_enabled=False: required for autocast inside torch.cuda.graph capture, harmless eager
    return (torch.autocast("cuda", dtype=torch.float16, cache_enabled=False) if amp
            else contextlib.nullcontext())


@torch.no_grad()
def forward(m, x, amp: bool):
    with _amp_ctx(amp):
        return m(x)


@torch.no_grad()
def capture(m, amp: bool, batch: int = 1):
    """A CUDA graph of one forward, warmed up on a side stream (the Track-2 recipe)."""
    static_x = torch.randn(batch, 40, 32, 64, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            forward(m, static_x, amp)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = forward(m, static_x, amp)
    g.replay()
    torch.cuda.synchronize()
    ref = forward(m, static_x, amp)
    diff = float((out[0].float() - ref[0].float()).abs().max())
    return g, diff


WAVE1 = ("cnext_unet", "swin_unet", "simvp", "ffno", "unet_mod")
# Revised plan 09-14: FLOPs targets per family (x FNO w96 L8), plus the original A100 band.
FLOPS_BAND = {"cnext_unet": (0.6, 1.9), "swin_unet": (0.7, 2.8), "simvp": (0.0, 2.0),
              "ffno": (0.0, 2.0), "unet_mod": (0.0, 2.0)}


def pick(path: str, lat_hi: float = 2.0, b1g_max: float = 3.0) -> None:
    """The wave-1 size rule on a finished table: per family, the LARGEST-FLOPs size inside the
    family's FLOPs target with ratio_b64_fp16 <= lat_hi and ratio_b1g_fp16 <= b1g_max; if none, the
    cheapest in latency, commented out (IMPRACTICAL above 3x)."""
    with open(path) as fh:
        rows = json.load(fh)["rows"]
    for fam in WAVE1:
        cand = [r for r in rows if r.get("arch") == fam and r.get("ratio_b64_fp16")]
        if not cand:
            print(f"# {fam} MISSING from {path}")
            continue
        lo, hi = FLOPS_BAND[fam]
        ok = [r for r in cand if lo <= r["flops_ratio_fno"] <= hi
              and r["ratio_b64_fp16"] <= lat_hi
              and (r.get("ratio_b1g_fp16") or r.get("ratio_b1_fp16") or 0.0) <= b1g_max]
        best = (max(ok, key=lambda r: (r["flops_ratio_fno"], r["params_M"])) if ok
                else min(cand, key=lambda r: r["ratio_b64_fp16"]))
        spec = f"{fam}:{best['width']}:{best['n_layers']}"
        if fam.startswith("ffno"):
            spec += f":{best['modes_y']}:{best['modes_x']}"
        flag = "ok" if ok else ("IMPRACTICAL" if best["ratio_b64_fp16"] > 3.0 else "outside-band")
        print(f"{'' if ok else '# '}{spec} {flag} flops={best['flops_ratio_fno']} "
              f"b64fp16={best['ratio_b64_fp16']} b1g_fp16={best.get('ratio_b1g_fp16')} "
              f"params_M={best['params_M']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--cand", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--target-s", type=float, default=0.08)
    ap.add_argument("--pick", default=None, help="print wave-1 sizes from a finished table")
    ap.add_argument("--stack4", action="store_true",
                    help="also time a batch-4 CUDA graph per row (ms per STEP): the stand-in for "
                         "four fused members at batch 1 in the P1 / P2 / P3 Track-2 projections")
    args = ap.parse_args()
    if args.pick:
        return pick(args.pick)
    if not args.out:
        raise SystemExit("--out is required")

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    rows = references(dev) + candidates(args.cand, dev)
    settings = SETTINGS + (STACK4 if args.stack4 else [])
    xs = {"b64": torch.randn(64, 40, 32, 64, device=dev),
          "b1": torch.randn(1, 40, 32, 64, device=dev)}

    graphs, gdiff, niter, ok = {}, {}, {}, {}
    for mode, amp in settings:
        for name, m, _ in rows:
            k = (name, mode, amp)
            try:
                if mode in ("b1g", "b4g"):
                    graphs[k], gdiff[k] = capture(m, amp, BATCH[mode])
                    run = graphs[k].replay
                else:
                    x = xs[mode]
                    f, s = forward(m, x, amp)
                    b = BATCH[mode]
                    assert f.shape == (b, 40, 32, 64) and (s is None or s.shape == (b, 2, 32, 64))
                    assert torch.isfinite(f.float()).all(), "non-finite output"
                    run = (lambda m=m, x=x, amp=amp: forward(m, x, amp))
                for _ in range(2):
                    run()
                torch.cuda.synchronize()
                t = time.perf_counter()
                run()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                niter[k] = int(min(1000, max(3, math.ceil(args.target_s / dt))))
                ok[k] = run
            except Exception as e:                       # e.g. an FFT that refuses fp16
                ok[k] = None
                print(f"[skip] {name} {_key(mode, amp)}: {type(e).__name__}: {str(e)[:160]}")
    acc = defaultdict(list)
    for r in range(args.rounds):
        for mode, amp in settings:
            for name, _, _ in rows:
                k = (name, mode, amp)
                if ok[k] is None:
                    continue
                n, run = niter[k], ok[k]
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(n):
                    run()
                torch.cuda.synchronize()
                acc[k].append((time.perf_counter() - t) / n)
        print(f"round {r + 1}/{args.rounds} done", flush=True)

    base_flops = next(meta for name, _, meta in rows if name == "ref_fno_w96L8")["gflops_b1"]
    out = []
    for name, _, meta in rows:
        rec = dict(name=name, **meta)
        rec["flops_ratio_fno"] = round(meta["gflops_b1"] / base_flops, 3)
        for mode, amp in settings:
            v = acc.get((name, mode, amp), [])
            v = v[1:] if len(v) > 1 else v
            b = BATCH[mode] if mode == "b64" else 1          # b64: per window; b1 / b1g / b4g: per step
            rec[_key(mode, amp)] = round(1e3 * float(np.median(v)) / b, 5) if v else None
            rec[_key(mode, amp) + "_iqr"] = (round(1e3 * float(np.subtract(*np.percentile(v, [75, 25]))) / b, 5)
                                             if len(v) > 2 else None)
            if mode == "b1g":
                rec[f"graph_maxdiff_{'fp16' if amp else 'fp32'}"] = gdiff.get((name, mode, amp))
        out.append(rec)
    base = next(r for r in out if r["name"] == "ref_fno_w96L8")
    for r in out:
        for mode, amp in settings:
            k = _key(mode, amp)
            r["ratio_" + k[3:]] = round(r[k] / base[k], 3) if r[k] and base[k] else None
        r["ms_ratio_fno"] = r["ratio_b64_fp16"]
    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.uname().nodename,
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                cudnn_benchmark=True, tf32=False, rounds=args.rounds,
                a100_to_4060=A100_TO_4060, job=os.environ.get("SLURM_JOB_ID"),
                ref_gflops=base_flops,
                note="forward only, eval, no_grad; ratio base ref_fno_w96L8 in the same process; "
                     "b1g = CUDA-graph replay; FLOPs on CPU, batch 1, FFT uncounted")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(meta=meta, rows=out), fh, indent=1)

    fmt = lambda v, n=3: "n/a" if v is None else f"{v:.{n}f}"  # noqa: E731
    print("\n| row | params M | int8 MB | FLOPs x | b64 fp32 | b64 fp16 | b1 fp16 | b1g fp32 | "
          "b1g fp16 | ratio b64 fp16 | ratio b1g fp16 |\n" + "|---" * 11 + "|")
    for r in out:
        print(f"| {r['name']} | {r['params_M']:.2f} | {fmt(r['int8_MB'], 1)} | "
              f"{fmt(r['flops_ratio_fno'], 2)} | {fmt(r['ms_b64_fp32'], 4)} | "
              f"{fmt(r['ms_b64_fp16'], 4)} | {fmt(r['ms_b1_fp16'], 2)} | {fmt(r['ms_b1g_fp32'], 3)} | "
              f"{fmt(r['ms_b1g_fp16'], 3)} | {fmt(r['ratio_b64_fp16'], 2)} | "
              f"{fmt(r['ratio_b1g_fp16'], 2)} |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
