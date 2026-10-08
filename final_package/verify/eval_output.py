"""The organizers' evaluation step, as FINAL_SUBMISSION.md describes it: import `solution` from
/workspace, `get_ttt_model(output_dir, "cuda")`, stream the trajectories the way the official
`local_eval.py` of the Track 2 kit does (reset at trajectory boundaries, the previous target passed,
each step timed), and score through the official scoring.py.

    python eval_output.py --kit <t2 kit dir> --data <dir with test_real/*.h5 and mean_std_real.pt> \
        --output /output [--workspace /workspace]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--workspace", default="/workspace")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

sys.path.insert(0, a.kit)
import local_eval as le  # noqa: E402

sc_spec = importlib.util.spec_from_file_location("official_scoring", os.path.join(a.kit, "scoring.py"))
sc = importlib.util.module_from_spec(sc_spec)
sc_spec.loader.exec_module(sc)

os.chdir(a.workspace)
sys.path.insert(0, a.workspace)
import solution  # noqa: E402

t = perf_counter()
model = solution.get_ttt_model(a.output, a.device)
print(f"get_ttt_model({a.output!r}, {a.device!r}): {type(model).__name__} in {perf_counter() - t:.1f}s", flush=True)
for attr in ("reset_ttt_state", "ttt_step"):
    assert callable(getattr(model, attr, None)), attr

norm = le.Normalizer(Path(a.data) / "mean_std_real.pt")
stream = le.build_stream(Path(a.data))
print(f"stream: {len(stream)} steps over {len({s['sim_id'] for s in stream})} trajectories", flush=True)
times, preds, tgts, lows, ups = [], [], [], [], []
prev = None
for step in stream:
    inp, tgt = step["input"].unsqueeze(0), step["target"].unsqueeze(0)
    r = 0.0
    if step["is_first"]:
        t0 = perf_counter()
        model.reset_ttt_state()
        r = perf_counter() - t0
        prev = None
    xn, yn = norm.preprocess(inp, tgt)
    t0 = perf_counter()
    pn, info = model.ttt_step(xn, prev[1] if prev is not None else None)
    times.append(perf_counter() - t0 + r)
    prev = (xn.detach(), yn.detach())
    pn = torch.as_tensor(pn)
    assert tuple(pn.shape) == tuple(yn.shape), pn.shape
    assert isinstance(info, dict) and "adapt_loss" in info and "lower" in info and "upper" in info
    preds.append(norm.postprocess_pred(pn.detach()).squeeze(0).cpu().numpy().astype(np.float32))
    tgts.append(tgt.squeeze(0).numpy().astype(np.float32))
    lows.append(norm.postprocess_pred(torch.as_tensor(info["lower"])).squeeze(0).cpu().numpy())
    ups.append(norm.postprocess_pred(torch.as_tensor(info["upper"])).squeeze(0).cpu().numpy())
P, Y, L, U = (np.stack(x) for x in (preds, tgts, lows, ups))
c = 2
rel = float(np.mean(sc.rel_l2_per_sample(P, Y, c)))
tke = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, c)))
mv = float(np.mean(sc.mvpe_rel_l2_per_sample(P, Y)))
sps = sc.aggregate_sps(P, Y, c, L, U)
sps_v = sps[0] if isinstance(sps, tuple) else sps
cov = float(np.mean(((Y[..., :2] >= L[..., :2]) & (Y[..., :2] <= U[..., :2]))[Y[..., :2] != 0]))
t_mean = float(np.mean(times))
print(f"errors  rel {rel:.4f}  tke {tke:.4f}  mvpe {mv:.4f}   coverage {cov:.3f}")
print(f"scores  rel {sc.score_error(rel):.3f}  tke {sc.score_error(tke):.3f}  mvpe {sc.score_error(mv):.3f}  "
      f"time {sc.score_time(t_mean):.3f}  sps {sc.score_sps(sps_v):.3f}")
print(f"time    {t_mean * 1e3:.3f} ms/step mean, {np.median(times) * 1e3:.3f} median over {len(times)} steps")
print("EVAL_OUTPUT OK")
