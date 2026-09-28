"""E0a latency: shipped band vs SIGBLEND (EBAND) packages, package level, paired and interleaved in ONE process (both
packages share the same _fno.py; only submission.py's EBAND and band.npz differ), plus a kernel-level read of eband_yhat.
usage: k0a_lat.py <pkgA> <pkgB> [reps]"""
import importlib.util, os, sys, time
import numpy as np, torch
a, b = sys.argv[1], sys.argv[2]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
X = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["x"]
def load(pkg, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg, "submission.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A, B = load(a, "subA"), load(b, "subB")
print("A EBAND", A.EBAND, "| B EBAND", B.EBAND, flush=True)
for m in (A, B):
    for n in (14, 56, 99):
        m.predict(X[:n])
res = {}
for n in (14, 56, 99):
    ta, tb = [], []
    for r in range(reps):
        for m, acc in ((A, ta), (B, tb)) if r % 2 == 0 else ((B, tb), (A, ta)):
            torch.cuda.synchronize(); t0 = time.perf_counter(); m.predict(X[:n]); torch.cuda.synchronize()
            acc.append((time.perf_counter() - t0) * 1e3 / n)
    ta, tb = np.array(ta), np.array(tb); d = tb - ta
    print(f"PACKAGE N={n:<3d} ship {np.median(ta):.4f}  sigblend {np.median(tb):.4f} ms/window   paired d median {np.median(d):+.4f} "
          f"mean {d.mean():+.4f} (sd {d.std(ddof=1)/np.sqrt(len(d)):.4f})  B slower in {int((d>0).sum())}/{len(d)}", flush=True)
# kernel level: eband_yhat on device tensors of chunk size
fno = B._fno; dev = "cuda"
beta = torch.tensor(B.EBAND[0], dtype=torch.float32, device=dev)
for n in (14, 56, 99):
    c0 = torch.randn(n, 20, 32, 64, 2, device=dev) * 0.1 + 0.2; sg = torch.rand(n, 1, 32, 64, 2, device=dev) * 0.02 + 1e-3
    x = torch.randn(n, 20, 32, 64, 3, device=dev)
    for _ in range(5): torch.exp(fno.eband_yhat(c0, sg, x, beta))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50): torch.exp(fno.eband_yhat(c0, sg, x, beta))
    torch.cuda.synchronize(); t = (time.perf_counter() - t0) / 50 * 1e3
    print(f"KERNEL  N={n:<3d} exp(eband_yhat) {t:.4f} ms/call = {t/n:.4f} ms/window", flush=True)
