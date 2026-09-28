import sys, time, os, zipfile, importlib.util, numpy as np
z, tag = sys.argv[1], sys.argv[2]
d = f"/tmp/k27_load_{tag}_{os.getpid()}"; os.makedirs(d, exist_ok=True)
zipfile.ZipFile(z).extractall(d)
os.chdir("/tmp")
t0 = time.time()
spec = importlib.util.spec_from_file_location("submission", os.path.join(d, "submission.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t1 = time.time()
x = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["x"][:56]
r = m.predict(x, {"sample_id": list(range(56))})
t2 = time.time()
r = m.predict(x, {"sample_id": list(range(56))})
t3 = time.time()
p = r["prediction"] if isinstance(r, dict) else r
print(f"[{tag}] import {t1-t0:.1f}s  first predict(56) {1000*(t2-t1)/56:.2f} ms/window (incl. any lazy load {t2-t1:.1f}s)  second {1000*(t3-t2)/56:.2f} ms/window  finite {bool(np.isfinite(p).all())}", flush=True)
