"""Gate the built archive the way the platform runs it, and refuse anything malformed.

    python validate.py --zip <archive> --extract <scratch dir>

Checks: `submission.py` at the archive root and the extracted size under 256 MB; the module
imports with the extract directory NOT on sys.path (as the platform loads it) and with scipy,
h5py, matplotlib and pandas blocked; per step, a (1, 20, 32, 64, 3) float32 finite prediction
with the pressure channel exactly zero, `info["adapt_loss"]` present, finite bounds with
lower <= upper on every step; no aliasing of the returned buffers; a CUDA input and a CPU
input get the same answer. Then the five subscores on the eight held-out trajectories through
the official `scoring.py`, with the step time measured as the metric measures it.
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import os
import shutil
import sys
import zipfile
from time import perf_counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stream as t2  # noqa: E402

# The organizers' scoring.py (Track 1 starting kit v9), vendored unchanged under band/.
_s = importlib.util.spec_from_file_location("official_scoring", os.path.join(HERE, "..", "band", "scoring.py"))
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)

BLOCKED = {"scipy", "h5py", "matplotlib", "pandas"}
FAIL = []


def check(ok, msg):
    print(f"  [{'ok ' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAIL.append(msg)


class Blocker:
    def __enter__(self):
        self._real = builtins.__import__
        for m in list(sys.modules):
            if m.split(".")[0] in BLOCKED:
                del sys.modules[m]

        def guard(name, *a, **k):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"No module named {name!r} (blocked: not in the eval image)")
            return self._real(name, *a, **k)
        builtins.__import__ = guard
        return self

    def __exit__(self, *e):
        builtins.__import__ = self._real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--extract", required=True, help="scratch directory the archive is unpacked into")
    ap.add_argument("--traj", nargs="+", default=t2.UNSEEN)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    print(f"=== archive {a.zip}")
    if os.path.isdir(a.extract):
        shutil.rmtree(a.extract)
    os.makedirs(a.extract)
    with zipfile.ZipFile(a.zip) as z:
        members = z.namelist()
        z.extractall(a.extract)
    size = sum(os.path.getsize(os.path.join(a.extract, m)) for m in members) / 1e6
    check("submission.py" in members, "submission.py is at the archive root")
    check(size < 256, f"extracted {size:.1f} MB < 256 MB cap")
    check(not any("/" in m for m in members), "no nesting")

    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stream = t2.build_stream(a.traj)          # h5py is read before the blocker goes up
    print(f"=== import, packages blocked {sorted(BLOCKED)}, device {dev}")
    keep = list(sys.path)
    with Blocker():
        spec = importlib.util.spec_from_file_location(
            "participant_submission", os.path.join(a.extract, "submission.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        check(hasattr(mod, "get_ttt_model"), "defines get_ttt_model")
        t_load = perf_counter()
        model = mod.get_ttt_model(a.extract, dev)
        t_load = perf_counter() - t_load
        check(hasattr(model, "reset_ttt_state") and hasattr(model, "ttt_step"),
              "returns an object with reset_ttt_state and ttt_step")

        print(f"=== stream {len(stream)} steps over {len(a.traj)} trajectories")
        preds, lows, ups, tgts, times = [], [], [], [], []
        bounds = None
        prev_t = None
        bad_shape = bad_dtype = bad_p = bad_order = bad_finite = 0
        for name, tid, x_raw, y_raw, first in stream:
            xn = torch.from_numpy(t2.normalize_in(x_raw)[None])
            yn = torch.from_numpy(t2.normalize_in(y_raw)[None])
            rt = 0.0
            if first:
                t0 = perf_counter(); model.reset_ttt_state()
                if dev == "cuda":
                    torch.cuda.synchronize()
                rt = perf_counter() - t0
                prev_t = None
            t0 = perf_counter()
            pn, info = model.ttt_step(xn, prev_t)
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(perf_counter() - t0 + rt)
            prev_t = yn
            p = torch.as_tensor(pn)
            bad_shape += tuple(p.shape) != (1, 20, 32, 64, 3)
            bad_dtype += p.dtype != torch.float32
            pa = p.detach().cpu().numpy()
            bad_finite += not np.all(np.isfinite(pa))
            bad_p += float(np.abs(pa[..., 2]).max()) != 0.0
            if not isinstance(info, dict) or "adapt_loss" not in info:
                FAIL.append('info must be a dict containing "adapt_loss"')
            has = ("lower" in info) or ("upper" in info)
            if bounds is None:
                bounds = has
            elif has != bounds:
                FAIL.append("bounds started/stopped mid-run")
            preds.append(pa)                                  # kept, not copied: the aliasing probe
            tgts.append(y_raw)
            if has:
                lo = torch.as_tensor(info["lower"]).detach().cpu().numpy()
                hi = torch.as_tensor(info["upper"]).detach().cpu().numpy()
                bad_order += int(np.any(lo > hi))
                bad_finite += int(not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))))
                lows.append(lo); ups.append(hi)
        check(bad_shape == 0, f"every prediction is (1,20,32,64,3)  [{bad_shape} bad]")
        check(bad_dtype == 0, f"every prediction is float32  [{bad_dtype} bad]")
        check(bad_finite == 0, f"predictions and bounds are finite  [{bad_finite} bad]")
        check(bad_p == 0, f"pressure channel is exactly zero  [{bad_p} bad]")
        check(bad_order == 0, f"lower <= upper everywhere  [{bad_order} bad]")
        check(bounds is True, "SPS bounds supplied on every step")
        uniq = len({p.tobytes() for p in preds[:12]})
        check(uniq > 1, f"no aliasing: first 12 predictions are distinct  [{uniq} unique]")

        if dev == "cuda":
            xc = torch.from_numpy(t2.normalize_in(stream[0][2])[None]).cuda()
            model.reset_ttt_state()
            pc, ic = model.ttt_step(xc, None)
            pc = torch.as_tensor(pc)
            model.reset_ttt_state()
            ph, _ = model.ttt_step(xc.cpu(), None)
            check(pc.is_cuda, "a CUDA input gets a CUDA prediction back")
            check("lower" in ic and torch.as_tensor(ic["lower"]).is_cuda,
                  "a CUDA input gets CUDA bounds back")
            d = float((pc.cpu() - torch.as_tensor(ph)).abs().max())
            check(d == 0.0, f"CUDA-input and CPU-input paths agree bit for bit ({d:.2e})")

    sys.path[:] = keep
    Y = np.stack(tgts).astype(np.float32)
    Praw = np.stack([t2.denorm_pred(p[0]) for p in preds]).astype(np.float32)
    Lraw = np.stack([t2.denorm_pred(l[0]) for l in lows]).astype(np.float32) if lows else None
    Uraw = np.stack([t2.denorm_pred(u[0]) for u in ups]).astype(np.float32) if ups else None
    t_step = float(np.mean(times))
    r = float(np.mean(sc.rel_l2_per_sample(Praw, Y, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(Praw, Y, 2)))
    m = sc.mvpe_rel_l2(Praw, Y)
    s, cov = sc.aggregate_sps(Praw, Y, 2, lower=Lraw, upper=Uraw)
    five = [sc.score_error(r), sc.score_error(k), sc.score_error(m),
            sc.score_time(t_step), sc.score_sps(s)]
    print(f"\n=== scores on {len(a.traj)} held-out trajectories (local; the board's set is another)")
    print(f"  errors  rel {r:.4f}  tke {k:.4f}  mvpe {m:.4f}   coverage {cov:.3f}")
    print(f"  scores  rel {five[0]:.3f}  tke {five[1]:.3f}  mvpe {five[2]:.3f}  "
          f"time {five[3]:.3f}  sps {five[4]:.3f}")
    print(f"  {t_step*1e3:.3f} ms/step local  ({np.median(times)*1e3:.3f} median, "
          f"{np.percentile(times, 99)*1e3:.3f} p99)")
    print(f"  wall clock: {t_load:.2f} s to construct + {np.sum(times):.2f} s for "
          f"{len(times)} steps = {t_load + np.sum(times):.2f} s   (600 s hard limit)")

    print()
    if FAIL:
        print("NOT SUBMITTABLE:")
        for f in FAIL:
            print(f"  - {f}")
        raise SystemExit(1)
    print("SUBMITTABLE")


if __name__ == "__main__":
    main()
