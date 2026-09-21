"""Round 17 Q1g: where the shipped inference path actually spends its time.

Q1 established that the package is NOT FLOP-bound -- seven precision variants span 2.7 %, and
doubling the members costs 1.59x rather than 2x. At 4x w96 L8 the forward is ~1.7 GFLOP/window
against an A100's 19.5 TFLOP/s, i.e. ~6 % utilization, and the activation traffic alone bounds
it near 0.2 ms/window against a measured 1.41. So the time is in kernels that are neither
matmul nor bandwidth-optimal, or in host work.

    python prof_q1g.py --sets "L8do45x4=a.pth,b.pth,c.pth,d.pth" --band ... [--profile]
                       [--sweep] [--variants] [--t2]

`--profile` dumps the torch.profiler table; `--sweep` the batch-size curve; `--variants` times
the opt-in rewrites in `_fno` (`RPDE_SPEC=bmm`, `RPDE_TAIL=jit`) and checks each against git
HEAD; `--t2` profiles the Track-2 eager step and times the graph separately.

CONTENTION GATE: the fp32 4-member reference must reproduce 1.41 A100 ms/window to within 3 %,
or every ms in the output is labelled contended and only the kernel SHARES are trustworthy.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

R_4060 = 1.850
BOARD_LOC = 0.803
FINAL_PER_BOARD_MS = -0.070
REF_MS = 1.412          # lat_fast17 job 7161154, fp32, 4xL8 do45, batch 112
T_OUT_, H_, W_ = 20, 32, 64


def band_of(path):
    z = np.load(path)
    return z["base"].astype(np.float32), z["sigma_ref"].astype(np.float32), float(z["p"])


def full_step(models, x, band, ref, p, alpha, batch):
    """submission.predict()'s timed section, byte for byte (as in lat_r15/lat_fast17)."""
    import _fno
    pred = np.zeros((len(x), 20, 32, 64, 3), dtype=np.float32)
    sig = np.empty((len(x), 1, 32, 64, 2), dtype=np.float32) if p else None
    _fno.predict_fno(models, x, "cuda", alpha=alpha, batch=batch, out=pred, sigma_out=sig)
    if p and sig is not None:
        r3 = np.zeros((len(x), 1, 32, 64, 3), dtype=np.float32)
        np.power(np.clip(sig / ref[..., :2], 0.3, 3.0), np.float32(p), out=r3[..., :2])
        lower, upper = np.empty_like(pred), np.empty_like(pred)
        tp, tl, th = (torch.from_numpy(a) for a in (pred, lower, upper))
        torch.mul(torch.from_numpy(band), torch.from_numpy(r3), out=th)
        torch.sub(tp, th, out=tl)
        torch.add(tp, th, out=th)
    return pred


def timeit(fn, rep=5):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def phases(models, x, band, ref, p, alpha, batch, rep=5):
    """Split the timed section into its stages, each with its own synchronize.

    Syncing between stages costs a little and shifts the total up slightly; the point is the
    SPLIT, not the total. `lat_r15`'s two member counts already imply a member-independent
    floor (t = a + b*K over K = 4 and 8 gives a = 0.573 ms/window), and this says what a is.
    """
    import _fno
    dev = "cuda"
    n = len(x)
    out = {k: 0.0 for k in ("upload", "forward", "compose", "d2h", "hostwrite", "band")}
    dst = np.zeros((n, 20, 32, 64, 3), dtype=np.float32)
    sig = np.empty((n, 1, 32, 64, 2), dtype=np.float32)
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    std = torch.from_numpy(_fno.STD).to(dev)
    for r in range(rep + 1):
        acc = {k: 0.0 for k in out}
        for i in range(0, n, batch):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            xb = torch.from_numpy(x[i:i + batch]).to(dev, non_blocking=True)[..., :2].float()
            prev = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), 40, 32, 64)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            y = s = None
            tot = 0
            with torch.no_grad():
                for m in models:
                    nm = getattr(m, "n_members", 1)
                    tot += nm
                    raw, ls = m(prev)
                    yi = raw.reshape(len(xb), 20, 2, 32, 64).permute(0, 1, 3, 4, 2) * nm
                    y = yi if y is None else y + yi
                    si = ls.clamp(-8.0, 2.0).exp().permute(0, 2, 3, 1).unsqueeze(1) * nm
                    s = si if s is None else s + si
            if tot > 1:
                y, s = y / tot, s / tot
            torch.cuda.synchronize(); t2 = time.perf_counter()
            m_ = y.mean(1, keepdim=True)
            s_ = y.std(1, keepdim=True, unbiased=False)
            y = m_ + (y - m_) * ((alpha * s + (1 - alpha) * s_) / (s_ + 1e-6))
            yd = y * std + mean
            torch.cuda.synchronize(); t3 = time.perf_counter()
            host = yd.float().cpu().numpy()
            sig[i:i + batch] = (s * std).float().cpu().numpy()
            torch.cuda.synchronize(); t4 = time.perf_counter()
            dst[i:i + batch, ..., :2] = host
            t5 = time.perf_counter()
            for k, dt in (("upload", t1 - t0), ("forward", t2 - t1), ("compose", t3 - t2),
                          ("d2h", t4 - t3), ("hostwrite", t5 - t4)):
                acc[k] += dt
        t0 = time.perf_counter()
        r3 = np.zeros((n, 1, 32, 64, 3), dtype=np.float32)
        np.power(np.clip(sig / ref[..., :2], 0.3, 3.0), np.float32(p), out=r3[..., :2])
        lo, hi = np.empty_like(dst), np.empty_like(dst)
        tp, tl, th = (torch.from_numpy(a) for a in (dst, lo, hi))
        torch.mul(torch.from_numpy(band), torch.from_numpy(r3), out=th)
        torch.sub(tp, th, out=tl)
        torch.add(tp, th, out=th)
        acc["band"] += time.perf_counter() - t0
        if r:                                    # r == 0 is the warm-up
            for k in out:
                out[k] += acc[k] / rep
    return {k: v / n * 1e3 for k, v in out.items()}


def host_probe(x, band, ref, p, batch, rep=5):
    """The MEMBER-INDEPENDENT half, component by component.

    `lat_r15`'s two member counts give t = 0.573 + 0.2098*K ms/window, so 0.573 -- 41 % of the
    shipped path -- is host work that no model change touches. This says what it is made of and
    what the three obvious repairs are worth:

      * a PINNED, asynchronous D2H instead of `.cpu()` on pageable memory (and, since the copy
        can then overlap the next batch's forward, whether there is anything left to overlap);
      * writing a CONTIGUOUS (N, 20, 32, 64, 2) buffer instead of the strided
        `dst[i:i+b, ..., :2] = ...`, with the zero-filled 3-channel array assembled once at the
        end in a single pass;
      * the band stage, which round 11 rewrote to run through torch (numpy's ufuncs are
        single-threaded and this is bandwidth-bound) -- re-checked at THIS N and this size.

    Everything is wrapped by the caller in try/except: this is a diagnostic and must never be
    the reason the profiling job returns nothing.
    """
    n, B = len(x), batch
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    y = torch.randn(B, T_OUT_, H_, W_, 2, device=dev)
    mb = y.numel() * 4 / 1e6
    dst3 = np.zeros((n, T_OUT_, H_, W_, 3), dtype=np.float32)
    buf2 = np.empty((n, T_OUT_, H_, W_, 2), dtype=np.float32)
    pin = torch.empty(B, T_OUT_, H_, W_, 2, dtype=torch.float32, pin_memory=cuda)

    def t(fn, r=rep):
        fn()
        sync()
        ts = []
        for _ in range(r):
            t0 = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    def d2h_pinned():
        pin.copy_(y, non_blocking=True)
        sync()
        return pin.numpy()

    h = y.cpu().numpy()
    t_dst3, t_buf2, t_h = (torch.from_numpy(a) for a in (dst3, buf2, h))
    nbatch = int(np.ceil(n / B))
    rows = [
        ("D2H .cpu() pageable  (per batch)", t(lambda: y.cpu().numpy()), mb, nbatch),
        ("D2H pinned + copy_   (per batch)", t(d2h_pinned), mb, nbatch),
        ("host write STRIDED dst[...,:2] (per batch)",
         t(lambda: dst3.__setitem__((slice(0, B), Ellipsis, slice(0, 2)), h)), mb, nbatch),
        ("host write CONTIGUOUS buf2[i:i+b] (per batch)",
         t(lambda: buf2.__setitem__(slice(0, B), h)), mb, nbatch),
        ("np.zeros((N,20,32,64,3)) once",
         t(lambda: np.zeros((n, T_OUT_, H_, W_, 3), dtype=np.float32)),
         n * T_OUT_ * H_ * W_ * 3 * 4 / 1e6, 1),
        ("final assembly dst3[...,:2] = buf2 once (numpy)",
         t(lambda: dst3.__setitem__((Ellipsis, slice(0, 2)), buf2)),
         n * T_OUT_ * H_ * W_ * 2 * 4 / 1e6, 1),
        # Round 11's lesson, applied to the OUTPUT write rather than the band: numpy's strided
        # assignment is single-threaded, torch's `copy_` is not, and this is bandwidth-bound.
        # If this row is fast, the contiguous-buffer rewrite is unnecessary -- the strided
        # write was never the problem, the single thread was.
        ("host write STRIDED via torch copy_  (per batch)",
         t(lambda: t_dst3[:B, ..., :2].copy_(t_h)), mb, nbatch),
        ("final assembly via torch copy_ once",
         t(lambda: t_dst3[..., :2].copy_(t_buf2)),
         n * T_OUT_ * H_ * W_ * 2 * 4 / 1e6, 1),
    ]

    sig = np.random.rand(n, 1, H_, W_, 2).astype(np.float32) * 0.01 + 0.005

    def band_stage():
        r3 = np.zeros((n, 1, H_, W_, 3), dtype=np.float32)
        np.power(np.clip(sig / ref[..., :2], 0.3, 3.0), np.float32(p), out=r3[..., :2])
        lo, hi = np.empty_like(dst3), np.empty_like(dst3)
        tp, tl, th = (torch.from_numpy(a) for a in (dst3, lo, hi))
        torch.mul(torch.from_numpy(band), torch.from_numpy(r3), out=th)
        torch.sub(tp, th, out=tl)
        torch.add(tp, th, out=th)

    rows.append(("band stage (torch-threaded, round 11) once", t(band_stage),
                 n * T_OUT_ * H_ * W_ * 3 * 4 * 3 / 1e6, 1))

    print(f"\n--- host-side components (batch {B}, N = {n}, {nbatch} batches) ---")
    print(f"{'component':<46} {'ms':>8} {'MB':>8} {'GB/s':>7} {'ms/window total':>16}")
    for name, sec, mbytes, count in rows:
        gbs = mbytes / 1e3 / sec if sec > 0 else float("nan")
        print(f"{name:<46} {sec*1e3:>8.3f} {mbytes:>8.1f} {gbs:>7.1f} "
              f"{sec * count / n * 1e3:>16.4f}")
    d_pag, d_pin = rows[0][1], rows[1][1]
    w_str, w_con, asm_np, w_torch = rows[2][1], rows[3][1], rows[5][1], rows[6][1]
    optA = (w_str - w_con) * nbatch - asm_np           # contiguous buffer + one assembly
    optB = (w_str - w_torch) * nbatch                  # same strided write, threaded
    asm_t = rows[7][1]
    optC = (w_str * nbatch) - (w_con * nbatch + asm_t)   # contiguous buffer + THREADED assembly

    def fin(sec):
        """`final` gained by SAVING `sec` of A100 wall time over the whole N."""
        return -sec / n * 1e3 * R_4060 * BOARD_LOC * FINAL_PER_BOARD_MS
    print(f"\n  pinned D2H saves {(d_pag-d_pin)*nbatch/n*1e3:+.4f} A100 ms/window "
          f"({fin((d_pag-d_pin)*nbatch):+.4f} `final`)")
    print(f"  option A, contiguous (N,...,2) buffer + one 3-channel assembly: "
          f"{optA/n*1e3:+.4f} ms/window ({fin(optA):+.4f} `final`) -- the assembly is ITSELF a "
          f"strided write of the same bytes, so it gives most of it back")
    print(f"  option B, the SAME strided write through torch (threaded): "
          f"{optB/n*1e3:+.4f} ms/window ({fin(optB):+.4f} `final`) -- no buffer, no assembly")
    print(f"  option C, contiguous buffer + THREADED assembly: "
          f"{optC/n*1e3:+.4f} ms/window ({fin(optC):+.4f} `final`) -- both repairs together")
    print(f"  overlap ceiling: an async D2H can hide at most the pinned copy itself, "
          f"{d_pin*nbatch/n*1e3:.4f} ms/window, and only if the next batch's forward is "
          f"longer than it.")


def do_profile(fn, label, warm=3):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=False) as pr:
        fn()
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"\n----- profile: {label} -----")
    print(pr.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=15))
    ka = pr.key_averages()
    cuda_us = sum(getattr(e, "self_cuda_time_total", 0.0) for e in ka)
    try:
        nk = sum(len(e.kernels) for e in pr.events())
    except Exception:
        nk = -1
    print(f"  wall (incl. profiler overhead) {wall*1e3:8.2f} ms")
    print(f"  summed self CUDA time          {cuda_us/1e3:8.2f} ms  "
          f"({100*cuda_us/1e3/(wall*1e3):.1f} % of wall -> the rest is GPU IDLE or host work)")
    print(f"  raw CUDA kernel launches       {nk}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--band", required=True)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rep", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--t2", action="store_true")
    a = ap.parse_args()

    import _fno
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    band, ref, p = band_of(a.band)
    x = np.load(os.path.join(HERE, "val_unseen.npz"))["x"].astype(np.float32)

    for spec in a.sets:
        name, _, lst = spec.partition("=")
        paths = [q if os.path.isabs(q) else os.path.join(HERE, q) for q in lst.split(",")]
        models = _fno.fuse([_fno.load_any(q, "cuda") for q in paths])
        print(f"\n================ {name}: {len(paths)} members -> {len(models)} invocation(s)"
              f" ================", flush=True)

        # --- CONTENTION GATE -------------------------------------------------------------
        _fno._SPEC, _fno._TAIL, _fno._HOSTW = "", "", ""
        base112 = timeit(lambda: full_step(models, x, band, ref, p, a.alpha, 112), a.rep)
        ms112 = base112 / len(x) * 1e3
        ok = abs(ms112 - REF_MS) / REF_MS <= 0.03
        verdict = "CLEAN" if ok else "*** CONTENDED -- kernel shares stand, ms values do not"
        print(f"CONTENTION GATE: fp32 batch-112 reference {ms112:.3f} A100 ms/window against "
              f"the recorded {REF_MS:.3f} ({100*(ms112-REF_MS)/REF_MS:+.1f} %) -> {verdict}",
              flush=True)

        if a.sweep:
            print(f"\n--- batch-size sweep (fp32, shipped path) ---")
            print(f"{'batch':>6} {'A100 ms/win':>12} {'batches':>8}")
            for b in (16, 32, 64, 112, 168, 336):
                try:
                    t = timeit(lambda: full_step(models, x, band, ref, p, a.alpha, b), 3)
                except RuntimeError as e:
                    print(f"{b:>6} {type(e).__name__}")
                    torch.cuda.empty_cache()
                    continue
                print(f"{b:>6} {t/len(x)*1e3:>12.3f} {int(np.ceil(len(x)/b)):>8}", flush=True)

        print(f"\n--- stage split at batch {a.batch} (A100 ms/window; syncs between stages) ---")
        ph = phases(models, x, band, ref, p, a.alpha, a.batch, a.rep)
        tot = sum(ph.values())
        for k, v in ph.items():
            print(f"  {k:<10} {v:7.4f}  ({100*v/tot:5.1f} %)")
        print(f"  {'TOTAL':<10} {tot:7.4f}   (member-independent = everything but `forward`: "
              f"{tot - ph['forward']:.4f})", flush=True)

        try:
            host_probe(x, band, ref, p, a.batch, a.rep)
        except Exception as e:                      # never lose the rest of the job to this
            print(f"  host_probe FAILED: {type(e).__name__}: {e}", flush=True)

        if a.profile:
            do_profile(lambda: full_step(models, x[:a.batch], band, ref, p, a.alpha, a.batch),
                       f"T1 predict_fno, one batch of {a.batch}")

        if a.variants:
            print(f"\n--- rewrite variants (batch {a.batch} and 112) ---")
            ref_out = None
            print(f"{'spec/tail/hostwrite':<24} {'ms/win@'+str(a.batch):>12} "
                  f"{'ms/win@112':>11} "
                  f"{'max|d| vs shipped':>18} {'d board ms':>11} {'d final':>8}")
            base_b, base_112 = None, None
            for vs, vt, vh in (("", "", ""), ("bmm", "", ""), ("", "jit", ""),
                               ("", "", "torch"), ("bmm", "jit", "torch")):
                _fno._SPEC, _fno._TAIL, _fno._HOSTW = vs, vt, vh
                lab = f"{vs or 'einsum'}/{vt or 'eager'}/{vh or 'numpy'}"
                try:
                    tb = timeit(lambda: full_step(models, x, band, ref, p, a.alpha, a.batch),
                                a.rep) / len(x) * 1e3
                    t112 = timeit(lambda: full_step(models, x, band, ref, p, a.alpha, 112),
                                  a.rep) / len(x) * 1e3
                    o = full_step(models, x, band, ref, p, a.alpha, a.batch)
                except RuntimeError as e:
                    print(f"{lab:<24} FAILED {type(e).__name__}: {str(e)[:60]}")
                    torch.cuda.empty_cache()
                    continue
                if ref_out is None:
                    ref_out, base_b, base_112 = o.copy(), tb, t112
                d = float(np.abs(o - ref_out).max())
                dms = (t112 - base_112) * R_4060 * BOARD_LOC
                print(f"{lab:<24} {tb:>12.3f} {t112:>11.3f} {d:>18.2e} {dms:>+11.3f} "
                      f"{dms*FINAL_PER_BOARD_MS:>+8.3f}", flush=True)
            _fno._SPEC, _fno._TAIL, _fno._HOSTW = "", "", ""

        del models
        torch.cuda.empty_cache()

    if a.t2:
        t2_section(a, band, ref, p)


def t2_section(a, band, ref, p):
    import importlib.util
    import _fno
    T2 = os.path.join(os.path.dirname(HERE), "Track_2")
    sys.path.insert(0, os.path.join(T2, "submit"))
    sp = importlib.util.spec_from_file_location("t2sub", os.path.join(T2, "submit",
                                                                     "submission.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    for spec in a.sets:
        name, _, lst = spec.partition("=")
        paths = [q if os.path.isabs(q) else os.path.join(HERE, q) for q in lst.split(",")]
        models = _fno.fuse([_fno.load_any(q, "cuda") for q in paths])
        xt = torch.randn(1, 20, 32, 64, 3)
        print(f"\n================ TRACK 2: {name} ================", flush=True)
        for vs, vt in (("", ""), ("bmm", ""), ("", "jit"), ("bmm", "jit")):
            _fno._SPEC, _fno._TAIL = vs, vt
            lab = f"spec={vs or 'einsum'},tail={vt or 'eager'}"
            for use_graph in (True, False):
                obj = m.FrozenEnsemble(models, "cuda", band, ref, p, use_graph=use_graph)
                for _ in range(60):
                    obj.ttt_step(xt, None)
                torch.cuda.synchronize()
                ts = []
                for _ in range(300):
                    t0 = time.perf_counter()
                    obj.ttt_step(xt, None)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                tag = "graph" if (use_graph and obj.graph is not None) else "eager"
                print(f"  {lab:<18} {tag:<6} {np.mean(ts)*1e3:7.3f} ms/step "
                      f"(p50 {np.median(ts)*1e3:.3f})", flush=True)
                if vs == "" and vt == "" and not use_graph and a.profile:
                    do_profile(lambda: obj.ttt_step(xt, None), "T2 eager step, batch 1")
                del obj
        _fno._SPEC, _fno._TAIL = "", ""
        del models
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
