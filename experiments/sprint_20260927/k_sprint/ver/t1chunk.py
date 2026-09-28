"""E-VER: t1chunk.py <pkgdir> <N> <out.npz>  -- one fresh process: predict() over val_unseen in chunks of N (the
evaluator's convention), keep every returned dict, hash it right after the call, re-hash all at the end (kept-output safety)."""
import hashlib, importlib.util, os, sys
import numpy as np
pkg, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
X = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["x"]
os.chdir("/tmp")
spec = importlib.util.spec_from_file_location("submission", os.path.join(pkg, "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
H = lambda o: hashlib.sha256(b"".join(np.ascontiguousarray(o[k]).tobytes() for k in ("prediction", "lower", "upper"))).hexdigest()
outs, hs = [], []
for s in range(0, len(X), N):
    o = sub.predict(X[s:s + N].copy())
    outs.append(o); hs.append(H(o))
bad = sum(H(o) != h for o, h in zip(outs, hs))
print(f"[t1chunk] {pkg} N={N} calls {len(outs)} kept-output overwritten: {bad}", flush=True)
cat = lambda k: np.concatenate([o[k] for o in outs])
o0 = outs[0]
np.savez(out, P=cat("prediction"), lo=cat("lower"), hi=cat("upper"), overwritten=bad,
         dtype=str(o0["prediction"].dtype), keys=str(sorted(o0)), typ=str(type(o0["prediction"])))
