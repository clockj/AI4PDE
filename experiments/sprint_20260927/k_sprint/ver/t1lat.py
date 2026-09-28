"""t1lat.py <pkgdir> <N> : fresh process; load (untimed), cold first predict() on N windows, then 5 warm calls on other windows."""
import importlib.util, os, sys, time
import numpy as np
pkg, N = sys.argv[1], int(sys.argv[2])
X = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["x"]
os.chdir("/tmp")
spec = importlib.util.spec_from_file_location("submission", os.path.join(pkg, "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
t = []
for k in range(6):
    x = X[(k * 37) % 200:(k * 37) % 200 + N].copy()
    t0 = time.perf_counter(); sub.predict(x); t.append((time.perf_counter() - t0) * 1e3 / N)
print(f"LAT {os.path.basename(pkg)} N={N} cold {t[0]:.3f} warm_med {np.median(t[1:]):.3f} ms/window")
