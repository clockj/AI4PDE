"""E1 T1 edge cases. usage: t1_edge.py <pkgdir> <tag> <mode> <outjson> [savenpz]
mode 'seq'  : one process, calls N = 1,13,100,1030(>PIN_OUT),14(f64),56, 99 x 9 (exhausts the arena), 1; md5 of each
              output at return; at the end re-hash every KEPT output -> 'kept_ok'.
mode 'single:<N>:<dt>' : fresh process, one call.
"""
import sys, os, json, time, hashlib
import numpy as np
pkg, tag, mode, outp = sys.argv[1:5]
save = sys.argv[5] if len(sys.argv) > 5 else None
X = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["x"].astype(np.float32)
XX = np.concatenate([X] * 4)            # 1344 windows (tiled) for N > 336
os.chdir(pkg); sys.path.insert(0, pkg)
import submission
import torch

def md5(o):
    h = hashlib.md5()
    for k in ("prediction", "lower", "upper"):
        h.update(np.ascontiguousarray(o[k]).tobytes())
    return h.hexdigest()

def check(o, x):
    p, l, u = o["prediction"], o["lower"], o["upper"]
    ok = p.shape == x.shape and l.shape == x.shape and u.shape == x.shape
    ok = ok and all(np.isfinite(a).all() for a in (p, l, u)) and bool((l <= u).all()) and bool((p[..., 2] == 0).all())
    return bool(ok)

res = {"tag": tag, "mode": mode, "pin": getattr(submission, "PIN_OUT", None), "cap": getattr(submission, "BATCH_CAP", None),
       "ln": getattr(submission, "LN_MANUAL", None), "warm": getattr(submission, "WARM_TAILS", None),
       "batch": int(submission._state[5]) if submission._state is not None else None, "gpu": torch.cuda.get_device_name(0)}
calls = []
if mode == "seq":
    plan = [(1, "f32"), (13, "f32"), (100, "f32"), (1030, "f32"), (14, "f64"), (56, "f32")] + [(99, "f32")] * 9 + [(1, "f32")]
else:
    _, n, dt = mode.split(":"); plan = [(int(n), dt)]
kept = []
off = 0
for n, dt in plan:
    x = XX[off % 336: off % 336 + n] if n <= 336 else XX[:n]
    off += 7
    if dt == "f64":
        x = x.astype(np.float64)
    x = np.ascontiguousarray(x)
    t0 = time.perf_counter(); o = submission.predict(x); t = time.perf_counter() - t0
    pooled = bool(getattr(submission, "_POOL", None) is not None and
                  any(np.shares_memory(o["prediction"], p.numpy()) for p in submission._POOL))
    h = md5(o)
    calls.append({"n": n, "dt": dt, "md5": h, "ok": check(o, x), "ms_per_win": 1e3 * t / n, "pooled": pooled,
                  "pool_i": getattr(submission, "_POOL_I", None)})
    kept.append((o, h))
    if save and n >= 100 and mode != "seq":
        np.savez(save, **{k: o[k] for k in ("prediction", "lower", "upper")})
res["calls"] = calls
res["kept_ok"] = all(md5(o) == h for o, h in kept)
res["kept_bad"] = [i for i, (o, h) in enumerate(kept) if md5(o) != h]
# overlap check: no two kept outputs share memory
ov = 0
for i in range(len(kept)):
    for j in range(i + 1, len(kept)):
        if np.shares_memory(kept[i][0]["prediction"], kept[j][0]["prediction"]):
            ov += 1
res["overlaps"] = ov
open(outp, "a").write(json.dumps(res) + "\n"); print(json.dumps(res))
