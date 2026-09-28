"""Interleaved latency: several T2 packages loaded in ONE process, streams alternated A B C / C B A ..., mean ms/step per pass."""
import sys, importlib.util, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import t2_common as t2

def load(d, tag):
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("sub_" + tag, d + "/submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.get_ttt_model(d, "cuda"); sys.path.remove(d)
    return m

def one_pass(model, stream):
    ts = []; prev = None
    for name, tid, xn, yn, first in stream:
        rt = 0.0
        if first:
            t0 = time.perf_counter(); model.reset_ttt_state(); torch.cuda.synchronize(); rt = time.perf_counter() - t0; prev = None
        t0 = time.perf_counter(); model.ttt_step(xn, prev); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0 + rt)
        prev = yn
    return 1e3 * float(np.mean(ts)), 1e3 * float(np.median(ts))

dirs = sys.argv[1:]
st = [(s[0], s[1], torch.from_numpy(t2.normalize_in(s[2])[None]), torch.from_numpy(t2.normalize_in(s[3])[None]), s[4]) for s in t2.build_stream(t2.UNSEEN)]
models = [load(d, str(i)) for i, d in enumerate(dirs)]
for m in models:
    one_pass(m, st)
R = {d: [] for d in dirs}
for rep in range(8):
    order = list(range(len(dirs))); order = order if rep % 2 == 0 else order[::-1]
    for i in order:
        R[dirs[i]].append(one_pass(models[i], st))
ref = np.array(R[dirs[0]])
for d in dirs:
    a = np.array(R[d]); dd = a[:, 0] - ref[:, 0]
    print(f"TIMING {Path(d).name:10s} mean/step {a[:, 0].mean():.3f} ms (min {a[:, 0].min():.3f}, median-of-steps {a[:, 1].mean():.3f}) | paired delta vs {Path(dirs[0]).name}: {dd.mean():+.3f} +- {dd.std(ddof=1) / np.sqrt(len(dd)):.3f} ms", flush=True)
