"""Gate a Track-2 archive the way the platform will, and refuse to pass anything sloppy.

    python validate_t2.py --zip submission_t2.zip           # GPU if there is one
    python validate_t2.py --zip submission_t2.zip --device cpu   # login-node stand-in

What it checks, and why each one is here:

* **Archive shape and size.**  ``submission.py`` at the ROOT, extracted size under 256 MB.
* **Import isolation.**  The extract directory is deliberately NOT put on ``sys.path`` and the
  file is loaded through ``spec_from_file_location``, which is how the platform loads it -- a
  bare ``import _fno`` fails that way unless ``submission.py`` fixes ``sys.path`` itself.
* **Missing packages.**  ``scipy``/``h5py``/``matplotlib``/``pandas`` are blocked during the
  import and the run.  (The Track-2 image does ship ``einops``, unlike Track 1's, but nothing
  here should need it.)
* **The output contract**, per step: shape ``(1,20,32,64,3)``, float32, finite, ``p`` exactly
  zero, ``info["adapt_loss"]`` present, ``lower <= upper`` everywhere, bounds all-or-nothing
  and CONSISTENT ACROSS THE WHOLE RUN.  A malformed bound zeroes every subscore, not just SPS.
* **No aliasing.**  Predictions are kept, not copied, and compared at the end: if ``ttt_step``
  hands back a reused staging buffer, every stored prediction becomes the last one.  This is a
  real bug that local harnesses hide, because they copy out on the same line.
* **Statelessness, for a FROZEN submission.**  Re-running one trajectory's windows in isolation
  must reproduce the streamed answers bit for bit.  That is the Track-2 analogue of Track 1's
  permutation-invariance assertion, and it is what says the model is not reading its stream
  position.  Pass ``--adaptive`` to skip it once a submission really does adapt.
* **Timing**, measured the way the metric does: mean over every step, with ``reset_ttt_state``
  charged to the step after the boundary.
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Track_1"))
import t2_common as t2  # noqa: E402

KIT = os.environ.get("RPDE_KIT", "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9")   # dir holding the v9 kit's scoring.py; default = the Pitzer mount (REPRODUCE §9.1)
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)
import final_v9  # noqa: E402

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
    ap.add_argument("--zip", default=str(HERE / "submission_t2.zip"))
    ap.add_argument("--extract", default="/tmp/rp_t2_validate")
    ap.add_argument("--traj", nargs="+", default=t2.UNSEEN)
    ap.add_argument("--device", default=None)
    ap.add_argument("--adaptive", action="store_true",
                    help="skip the statelessness assertion (a real TTT model is stateful)")
    ap.add_argument("--repeat", type=int, default=2)
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
    check(not any("/" in m and not m.endswith("/") and m.count("/") > 1 for m in members),
          "no unexpected nesting")

    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Build the stream BEFORE the import blocker goes up: reading h5 is the harness's job and
    # h5py is one of the packages the submission is forbidden to touch.
    stream = t2.build_stream(a.traj)
    print(f"=== import, packages blocked {sorted(BLOCKED)}, device {dev}")
    keep = list(sys.path)
    with Blocker():
        spec = importlib.util.spec_from_file_location(
            "participant_submission", os.path.join(a.extract, "submission.py"))
        mod = importlib.util.module_from_spec(spec)
        # deliberately do NOT add the extract dir to sys.path
        spec.loader.exec_module(mod)
        check(hasattr(mod, "get_ttt_model"), "defines get_ttt_model")
        # Construction is not TIMED by the metric, but it does count against the run's hard
        # 10-minute wall clock -- and this one loads eight checkpoints, fuses them and captures
        # (and verifies) a CUDA graph, so it is worth knowing rather than assuming.
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
            preds.append(pa)                                  # kept, NOT copied: aliasing probe
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

        P = np.concatenate(preds, 0).astype(np.float32)
        uniq = len({p.tobytes() for p in preds[:12]})
        check(uniq > 1, f"no aliasing: first 12 predictions are distinct  [{uniq} unique]")

        # Exercise BOTH return paths.  `ttt_step` mirrors the device of the input it is
        # handed, and the platform's choice is not documented -- the kit's local_eval passes
        # CPU tensors while its reference model returns CUDA ones.  Whichever the platform
        # does, one of these two is the path that will run.
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

        if not a.adaptive:
            model.reset_ttt_state()
            g = [i for i, s in enumerate(stream) if s[0] == stream[0][0]]
            solo = []
            for i in reversed(g[:8]):                          # reverse order, no prev target
                pn, _ = model.ttt_step(
                    torch.from_numpy(t2.normalize_in(stream[i][2])[None]), None)
                solo.append(torch.as_tensor(pn).detach().cpu().numpy())
            d = max(float(np.abs(solo[k] - preds[g[7 - k]]).max()) for k in range(8))
            check(d == 0.0, f"stateless: re-running windows out of order is bit-identical "
                            f"(max diff {d:.2e})")

    sys.path[:] = keep
    Y = np.stack(tgts).astype(np.float32)
    L = np.stack(lows).astype(np.float32) if lows else None
    U = np.stack(ups).astype(np.float32) if ups else None
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
    print(f"\n=== scores on {a.traj if len(a.traj) < 4 else str(len(a.traj)) + ' trajectories'}")
    print(f"  errors  rel {r:.4f}  tke {k:.4f}  mvpe {m:.4f}   coverage {cov:.3f}")
    print(f"  scores  rel {five[0]:.3f}  tke {five[1]:.3f}  mvpe {five[2]:.3f}  "
          f"time {five[3]:.3f}  sps {five[4]:.3f}")
    print(f"  mean5 {np.mean(five):.3f}   " + final_v9.report(five))
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
