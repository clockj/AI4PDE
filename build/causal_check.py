"""The causality assertion for the online corrections: nothing an output depends on may come
from a later window.

    python causal_check.py --submission <extracted archive dir>

The eight held-out trajectories are streamed twice: once as they are, once with every window
after each trajectory's middle one replaced by noise (its input, and the target the platform
would reveal). Every prediction, lower and upper bound up to and including the cut must be
bit-identical between the two runs. The build fails if they are not.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from time import perf_counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stream as t2  # noqa: E402


def import_submission(d: str):
    f = os.path.join(d, "submission.py")
    if not os.path.exists(f):
        raise SystemExit(f"no submission.py under {d}")
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("t2_submission", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(model, stream, device):
    """Drive the platform's loop. Returns (pred, lower, upper, target) in physical units."""
    import torch

    preds, lows, ups, tgts = [], [], [], []
    bounds_active = None
    for name, tid, x_raw, y_raw, is_first in stream:
        xn = torch.from_numpy(t2.normalize_in(x_raw)[None])
        yn = torch.from_numpy(t2.normalize_in(y_raw)[None])
        if is_first:
            model.reset_ttt_state()
            if device == "cuda":
                torch.cuda.synchronize()
            prev_target = None
        pred_norm, info = model.ttt_step(xn, prev_target)
        if device == "cuda":
            torch.cuda.synchronize()
        prev_target = yn
        pred_norm = torch.as_tensor(pred_norm).detach().cpu().numpy()
        has = "lower" in info
        if bounds_active is None:
            bounds_active = has
        elif has != bounds_active:
            raise SystemExit("submission changed its mind about SPS bounds mid-run")
        preds.append(t2.denorm_pred(pred_norm[0]))
        tgts.append(y_raw)
        if has:
            lows.append(t2.denorm_pred(torch.as_tensor(info["lower"]).detach().cpu().numpy()[0]))
            ups.append(t2.denorm_pred(torch.as_tensor(info["upper"]).detach().cpu().numpy()[0]))
    P = np.stack(preds).astype(np.float32)
    Y = np.stack(tgts).astype(np.float32)
    L = np.stack(lows).astype(np.float32) if bounds_active else None
    U = np.stack(ups).astype(np.float32) if bounds_active else None
    return P, L, U, Y


def causal_check(model, stream, device, seed=0):
    rng = np.random.default_rng(seed)
    names = [s[0] for s in stream]
    cut = {}
    for nm in dict.fromkeys(names):
        idx = [i for i, n in enumerate(names) if n == nm]
        cut[nm] = idx[len(idx) // 2]
    noisy = []
    for i, (nm, tid, x, y, first) in enumerate(stream):
        if i > cut[nm]:
            x = rng.standard_normal(x.shape).astype(np.float32) * x.std() + x.mean()
        if i >= cut[nm]:        # window c's own target is revealed only at step c+1: noise it too
            y = rng.standard_normal(y.shape).astype(np.float32) * y.std() + y.mean()
        noisy.append((nm, tid, x, y, first))
    A = run(model, stream, device)
    B = run(model, noisy, device)
    keep = np.array([i <= cut[n] for i, n in enumerate(names)])
    worst = 0.0
    for k, (a, b) in enumerate(zip(A[:3], B[:3])):
        if a is None:
            continue
        d = np.abs(a[keep] - b[keep])
        worst = max(worst, float(d.max()))
        if not np.array_equal(a[keep], b[keep]):
            raise SystemExit(f"CAUSALITY FAIL: {['pred', 'lower', 'upper'][k]} at steps <= cut changed when only "
                             f"FUTURE windows were replaced by noise (max |diff| {d.max():.3e})")
    return int(keep.sum()), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="directory holding the extracted archive")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stream = t2.build_stream(t2.UNSEEN)
    d = os.path.abspath(a.submission)
    mod = import_submission(d)
    model = mod.get_ttt_model(d, dev)
    t0 = perf_counter()
    n, worst = causal_check(model, stream, dev)
    print(f"CAUSALITY OK: {n} steps up to each trajectory's cut are bit-identical with every later "
          f"window replaced by noise (max |diff| {worst:.1e}; {perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
