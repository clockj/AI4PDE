"""Evaluator-like T2 stream: get_ttt_model (untimed), then timed ttt_step per window, predictions KEPT until the end.
usage: k1_t2child.py <pkgdir> <tag> <outjson>"""
import sys, os, json, time, hashlib
import numpy as np
pkg, tag, outp = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")
x, y = z["x"].astype(np.float32), z["y"].astype(np.float32)
mi = np.array([0.154960856, -0.000513992854, 0.0], np.float32); si = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
mt = np.array([0.154962569, -0.000517793698, 0.0], np.float32); st = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
os.chdir(pkg); sys.path.insert(0, pkg)
import torch
import submission
xn = torch.from_numpy((x - mi) / si).unsqueeze(1)      # (N,1,20,32,64,3) cpu
yn = torch.from_numpy((y - mt) / st).unsqueeze(1)
m = submission.get_ttt_model(pkg, "cuda")
keep, ts = [], []
n, per = len(xn), 42
for k in range(n):
    reset = 0.0
    if k % per == 0:
        t0 = time.perf_counter(); m.reset_ttt_state(); reset = time.perf_counter() - t0
    prev = None if k % per == 0 else yn[k - 1]
    inp = xn[k]
    t0 = time.perf_counter()
    pred, info = m.ttt_step(inp, prev)
    ts.append(time.perf_counter() - t0 + reset)
    keep.append((pred, info.get("lower"), info.get("upper")))
h = hashlib.md5()
for p_, l_, u_ in keep:
    for a in (p_, l_, u_):
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
ts = np.array(ts) * 1e3
res = {"tag": tag, "steps": n, "mean_ms": float(ts.mean()), "median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)),
       "first10_mean": float(ts[:10].mean()), "md5": h.hexdigest(), "gpu": torch.cuda.get_device_name(0),
       "graph": m.graph is not None, "arena": getattr(submission, "ARENA_STEPS", None), "type": type(keep[0][0]).__name__}
open(outp, "a").write(json.dumps(res) + "\n"); print(json.dumps(res))
