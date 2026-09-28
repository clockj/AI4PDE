"""One evaluator-like process: import the package (untimed), one COLD predict(), then warm repeats per N.

usage: k1_child.py <pkgdir> <tag> <cold_N> <dtype> <reps> <outjson>
"""
import sys, os, json, time, hashlib
import numpy as np

pkg, tag, cold_n, dt, reps, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
VAL = "/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz"
x_all = np.load(VAL)["x"][:112].astype(np.float64 if dt == "f64" else np.float32)
os.chdir(pkg)
sys.path.insert(0, pkg)
t0 = time.perf_counter()
import submission  # noqa: E402
import torch  # noqa: E402
t_import = time.perf_counter() - t0


def md5(o):
    h = hashlib.md5()
    for k in ("prediction", "lower", "upper"):
        h.update(np.ascontiguousarray(o[k]).tobytes())
    return h.hexdigest()


def reset_pool():
    if hasattr(submission, "_POOL_I"):
        submission._POOL_I = 0
    if hasattr(submission, "_POOL_USED"):
        submission._POOL_USED = False


res = {"tag": tag, "dtype": dt, "import_s": t_import, "cold_n": cold_n, "amp": submission.AMP_ACTIVE,
       "batch": int(submission._state[5]) if submission._state is not None else None,
       "par": getattr(submission._fno, "PAR_MEMBERS", None), "pin": getattr(submission, "PIN_OUT", None),
       "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
x = np.ascontiguousarray(x_all[:cold_n])
t0 = time.perf_counter()
o = submission.predict(x)
res["cold_ms_per_win"] = 1e3 * (time.perf_counter() - t0) / cold_n
res["cold_md5"] = md5(o)
del o
res["warm"] = {}
res["md5"] = {}
for n in (14, 28, 56, 99):
    x = np.ascontiguousarray(x_all[:n])
    ts = []
    for r in range(reps + 2):
        reset_pool()
        t0 = time.perf_counter()
        o = submission.predict(x)
        ts.append(1e3 * (time.perf_counter() - t0) / n)
        if r == 0:
            res["md5"][str(n)] = md5(o)
        del o
    ts = ts[2:]
    res["warm"][str(n)] = {"median": float(np.median(ts)), "min": float(np.min(ts)), "mean": float(np.mean(ts))}
# stage breakdown at N = 56 (device-synchronised), shipped arithmetic only
try:
    models, dev = submission._state[0], submission._state[1]
    xb = torch.from_numpy(np.ascontiguousarray(x_all[:56])).to(dev)[..., :2].float()
    prev = ((xb - torch.from_numpy(submission._fno.MEAN).to(dev)) / torch.from_numpy(submission._fno.STD).to(dev)) \
        .permute(0, 1, 4, 2, 3).reshape(56, 40, 32, 64)
    bd = {}
    with torch.no_grad():
        for i, m in enumerate(models):
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.float16):
                    m(prev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                with torch.autocast("cuda", dtype=torch.float16):
                    m(prev)
            torch.cuda.synchronize()
            bd[f"group{i}_{type(m).__name__}"] = 1e3 * (time.perf_counter() - t0) / 10 / 56
        # upload
        for dtt in (np.float32, np.float64):
            xh = np.ascontiguousarray(x_all[:56].astype(dtt))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                torch.from_numpy(xh).to(dev)[..., :2].float()
            torch.cuda.synchronize()
            bd[f"upload_{np.dtype(dtt).name}"] = 1e3 * (time.perf_counter() - t0) / 10 / 56
        # device -> host of three full-size arrays, pageable fresh vs pageable reused vs pinned
        g = torch.randn(56, 20, 32, 64, 3, device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            for _k in range(3):
                h = np.empty((56, 20, 32, 64, 3), dtype=np.float32)
                torch.from_numpy(h).copy_(g)
                del h
        bd["d2h3_fresh"] = 1e3 * (time.perf_counter() - t0) / 5 / 56
        hs = [np.zeros((56, 20, 32, 64, 3), dtype=np.float32) for _ in range(3)]
        for h in hs:
            h += 1
        t0 = time.perf_counter()
        for _ in range(5):
            for h in hs:
                torch.from_numpy(h).copy_(g)
        bd["d2h3_reused"] = 1e3 * (time.perf_counter() - t0) / 5 / 56
        ps = [torch.zeros((56, 20, 32, 64, 3)).pin_memory() for _ in range(3)]
        t0 = time.perf_counter()
        for _ in range(5):
            for h in ps:
                h.copy_(g)
        bd["d2h3_pinned"] = 1e3 * (time.perf_counter() - t0) / 5 / 56
    res["breakdown_ms_per_win_N56"] = bd
except Exception as e:
    res["breakdown_error"] = repr(e)
with open(outp, "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps(res))
