"""E1 T2 edge: stream NSTEPS steps (val_unseen tiled, reset every 42), KEEP every output, hash at return and re-hash at end.
usage: t2_edge.py <pkgdir> <tag> <nsteps> <outjson>"""
import sys, os, json, time, hashlib
import numpy as np
pkg, tag, nsteps, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
z = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")
x, y = z["x"].astype(np.float32), z["y"].astype(np.float32)
mi = np.array([0.154960856, -0.000513992854, 0.0], np.float32); si = np.array([0.0968056545, 0.015960684, 1.0], np.float32)
mt = np.array([0.154962569, -0.000517793698, 0.0], np.float32); st = np.array([0.0968104079, 0.0159636438, 1.0], np.float32)
os.chdir(pkg); sys.path.insert(0, pkg)
import torch
import submission
xn = torch.from_numpy((x - mi) / si).unsqueeze(1)
yn = torch.from_numpy((y - mt) / st).unsqueeze(1)
m = submission.get_ttt_model(pkg, "cuda")
def hh(t):
    return hashlib.md5(np.ascontiguousarray(np.asarray(t)).tobytes()).hexdigest()
keep, H, ts = [], [], []
per, N = 42, len(xn)
for k in range(nsteps):
    kk = k % N
    reset = 0.0
    if kk % per == 0:
        t0 = time.perf_counter(); m.reset_ttt_state(); reset = time.perf_counter() - t0
    prev = None if kk % per == 0 else yn[kk - 1]
    t0 = time.perf_counter()
    pred, info = m.ttt_step(xn[kk], prev)
    ts.append(time.perf_counter() - t0 + reset)
    trip = (pred, info.get("lower"), info.get("upper"))
    keep.append(trip); H.append(tuple(hh(a) for a in trip))
bad = [k for k in range(nsteps) if tuple(hh(a) for a in keep[k]) != H[k]]
def tot(a, b):
    h = hashlib.md5()
    for p_, l_, u_ in keep[a:b]:
        for q in (p_, l_, u_):
            h.update(np.ascontiguousarray(np.asarray(q)).tobytes())
    return h.hexdigest()
# periodic check: step k and k+336 see the same inputs and the same reset pattern -> outputs must be equal
per_eq = all(H[k] == H[k + N] for k in range(min(nsteps - N, 3 * N))) if nsteps > N else None
ts = np.array(ts) * 1e3
res = {"tag": tag, "steps": nsteps, "arena": getattr(submission, "ARENA_STEPS", None), "arena_i": getattr(m, "arena_i", None),
       "graph": m.graph is not None, "kept_ok": not bad, "kept_bad_n": len(bad), "kept_bad_first": bad[:5],
       "md5_336": tot(0, min(336, nsteps)), "md5_all": tot(0, nsteps), "periodic_equal": per_eq,
       "md5_1020_1030": tot(1020, min(1030, nsteps)) if nsteps > 1020 else None,
       "median_ms": float(np.median(ts)), "mean_ms": float(ts.mean()), "type": type(keep[0][0]).__name__}
open(outp, "a").write(json.dumps(res) + "\n"); print(json.dumps(res))
