"""E0a: run an EXTRACTED Track-1 package's predict() on val_unseen (all 336 windows, one call) through the real code path,
and also dump the package's own sigma head and alpha-0 centre (same models, same predict_fno, same batch/amp) so the band
formula can be verified.  usage: k0a_dump.py <pkgdir> <out.npz>"""
import importlib.util, os, sys, time
import numpy as np
pkg, out = sys.argv[1], sys.argv[2]
d = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")
X, ids = d["x"], d["ids"]
spec = importlib.util.spec_from_file_location("submission", os.path.join(pkg, "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
print("EBAND", sub.EBAND, "SMALL_GRAPH", sub.SMALL_GRAPH, "GPU_BAND", sub.GPU_BAND, "ALPHA", sub.ALPHA,
      "CENTRE_ALPHA", sub.CENTRE_ALPHA, "p", sub._state[4], flush=True)
sub.predict(X[:32])
t0 = time.perf_counter(); o = sub.predict(X); print(f"predict N=336 {1e3*(time.perf_counter()-t0)/336:.3f} ms/window (not latency evidence)")
P, lo, hi = o["prediction"], o["lower"], o["upper"]
models, dev, band, ref, p, batch = sub._state[:6]
pred2 = np.zeros_like(P); sig = np.empty((len(X), 1, 32, 64, 2), np.float32); c0 = np.zeros_like(P)
sub._fno.predict_fno(models, np.ascontiguousarray(X, dtype=np.float32), dev, alpha=sub.ALPHA, batch=batch, out=pred2,
                     sigma_out=sig, amp=sub._AMP_DT, centre_out=c0)
print("pred re-run vs predict(): max|d| %.3e" % np.abs(pred2 - P).max())
np.savez(out, P=P, lo=lo, hi=hi, sig=sig, c0=c0[..., :2], ids=ids, p3max=np.array([np.abs(a[..., 2]).max() for a in (P, lo, hi)]))
print("wrote", out, flush=True)
