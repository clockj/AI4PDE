"""E-VER: t2run.py <pkgdir> <out.npz> : 336-step stream over val_unseen, platform loop; keeps the RAW returned objects,
hashes each right after the step and re-hashes all at the end (kept-output safety); saves raw-unit P/L/U/Y + times."""
import hashlib, sys
from time import perf_counter
import numpy as np, torch
sys.path.insert(0, "/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/ver/wt/Track_2")
import stream_eval as se
t2 = se.t2
from pathlib import Path
pkg, out = sys.argv[1], sys.argv[2]
stream = t2.build_stream(t2.UNSEEN)
mod = se.import_submission(Path(pkg).resolve())
model = mod.get_ttt_model(str(Path(pkg).resolve()), "cuda")
def h(*a):
    return hashlib.sha256(b"".join(np.ascontiguousarray(torch.as_tensor(x).detach().cpu().numpy()).tobytes() for x in a)).hexdigest()
keep, hs, T, Y, prev = [], [], [], [], None
for name, tid, x_raw, y_raw, first in stream:
    xn = torch.from_numpy(t2.normalize_in(x_raw)[None]); yn = torch.from_numpy(t2.normalize_in(y_raw)[None])
    rt = 0.0
    if first:
        t0 = perf_counter(); model.reset_ttt_state(); torch.cuda.synchronize(); rt = perf_counter() - t0; prev = None
    t0 = perf_counter(); p, info = model.ttt_step(xn, prev); torch.cuda.synchronize(); T.append(perf_counter() - t0 + rt)
    prev = yn
    keep.append((p, info["lower"], info["upper"])); hs.append(h(p, info["lower"], info["upper"])); Y.append(y_raw)
bad = sum(h(*k) != x for k, x in zip(keep, hs))
T = np.asarray(T)
print(f"[t2run] {pkg} steps {len(keep)} kept-output overwritten {bad} | {1e3*T.mean():.3f} ms/step mean, {1e3*np.median(T):.3f} median, "
      f"adapt_loss present {'adapt_loss' in info} ptype {type(keep[0][0]).__name__}", flush=True)
dn = lambda a: t2.denorm_pred(torch.as_tensor(a).detach().cpu().numpy()[0])
P = np.stack([dn(k[0]) for k in keep]).astype(np.float32); L = np.stack([dn(k[1]) for k in keep]).astype(np.float32)
U = np.stack([dn(k[2]) for k in keep]).astype(np.float32)
np.savez(out, P=P, L=L, U=U, Y=np.stack(Y).astype(np.float32), T=T, overwritten=bad, ids=np.array([f"{s[0]}:{s[1]}" for s in stream]))
