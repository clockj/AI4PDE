"""eseq.py <pkgdir> <out.npz> seq|one <N> <dt> : E checks. seq = one process: 99 x 12, 56, 14, 1, 1030; keep all outputs,
hash, re-hash at end, shares_memory across all returned arrays; /proc/self/status memory after load and at end."""
import hashlib, importlib.util, os, sys
import numpy as np
pkg, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
X = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["x"]
XX = np.concatenate([X] * 4)  # 1344 windows for the 1030 call
os.chdir("/tmp")
def mem(tag):
    d = {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/self/status") if l.split(":")[0] in ("VmRSS", "VmLck", "VmPin", "VmHWM")}
    print(f"MEM {os.path.basename(pkg)} {tag} {d}", flush=True)
spec = importlib.util.spec_from_file_location("submission", os.path.join(pkg, "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
mem("after-load")
H = lambda o: hashlib.sha256(b"".join(np.ascontiguousarray(o[k]).tobytes() for k in ("prediction", "lower", "upper"))).hexdigest()
if mode == "one":
    n, dt = int(sys.argv[4]), sys.argv[5]
    o = sub.predict(X[:n].astype(dt)); np.savez(out, P=o["prediction"], lo=o["lower"], hi=o["upper"]); print("ONE", n, dt, H(o)); sys.exit()
calls = [(k * 20 % 237, 99) for k in range(12)] + [(5, 56), (40, 14), (7, 1), (0, 1030)]
outs, hs = [], []
for s, n in calls:
    o = sub.predict(XX[s:s + n].astype(np.float32)); outs.append(o); hs.append(H(o))
bad = sum(H(o) != h for o, h in zip(outs, hs))
arrs = [o[k] for o in outs for k in ("prediction", "lower", "upper")]
share = sum(np.shares_memory(arrs[i], arrs[j]) for i in range(len(arrs)) for j in range(i + 1, len(arrs)))
pinned = [bool(getattr(o["prediction"], "base", None) is not None) for o in outs]
print(f"SEQ {os.path.basename(pkg)} calls {len(outs)} overwritten {bad} shared-memory pairs {share} has-base {pinned}", flush=True)
mem("after-seq")
np.savez(out, hashes=np.array(hs))
