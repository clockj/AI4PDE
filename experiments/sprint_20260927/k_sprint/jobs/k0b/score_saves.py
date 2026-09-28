"""Offline scoring of stream_eval --save dumps (same official scorer funcs as stream_eval.score), paired vs a baseline.
usage: score_saves.py BASE_LABEL label=path.npz ..."""
import sys, os, numpy as np
W = "/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_0b"
sys.path.insert(0, W + "/Track_2"); sys.path.insert(0, W + "/Track_1")
sys.argv, args = sys.argv[:1], sys.argv[1:]
import stream_eval as se, t2_common as t2, final_v9
base = args[0]; runs = dict(a.split("=", 1) for a in args[1:])
D = {k: np.load(v) for k, v in runs.items()}
def sc(P, L, U, Y):
    d = se.score(P, L, U, Y, 0.005)
    d["fx"] = float(final_v9.W0 @ np.array([d["rel"], d["tke"], d["mvpe"], 0.0, d["sps"]]))   # final ex-time
    return d
res = {}
for k, d in D.items():
    names = np.array([s.split(":")[0] for s in d["ids"]]); aoa = np.array([t2.aoa_of(n) for n in names])
    P, L, U, Y = d["pred"], d["lower"], d["upper"], d["target"]
    for f in (1.0, 1.29):
        Ys = Y if f == 1.0 else np.where(Y == 0, 0, P + f * (Y - P)).astype(np.float32)
        for sp, sel in (("all8", np.ones(len(P), bool)), ("aoa15-20", np.isin(aoa, [15, 20]))):
            res[(k, f, sp)] = sc(P[sel], L[sel], U[sel], Ys[sel])
        for n in dict.fromkeys(names):
            s = names == n
            res[(k, f, n)] = sc(P[s], L[s], U[s], Ys[s])
    res[(k, "names")] = list(dict.fromkeys(names))
names = res[(base, "names")]
for f in (1.0, 1.29):
    print(f"\n=== stretch {f} (target deviation from P scaled by f; 1.0 = stream_eval's own read) ===")
    print(f"{'variant':34s} {'split':9s} {'relE':>7s} {'tkeE':>7s} {'mvpeE':>7s} {'cov':>6s} {'rel':>7s} {'tke':>7s} {'mvpe':>7s} {'sps':>7s} {'fx':>8s} | {'dsps':>7s} {'dfx':>7s} traj+")
    for k in runs:
        for sp in ("all8", "aoa15-20"):
            r = res[(k, f, sp)]; b = res[(base, f, sp)]
            tr = [n for n in names if (sp == "all8" or t2.aoa_of(n) in (15, 20))]
            pos = sum(res[(k, f, n)]["sps"] > res[(base, f, n)]["sps"] for n in tr)
            print(f"{k:34s} {sp:9s} {r['rel_err']:7.4f} {r['tke_err']:7.4f} {r['mvpe_err']:7.4f} {r['coverage']:6.3f} {r['rel']:7.3f} {r['tke']:7.3f} {r['mvpe']:7.3f} {r['sps']:7.3f} {r['fx']:8.3f} | {r['sps']-b['sps']:+7.3f} {r['fx']-b['fx']:+7.3f} {pos}/{len(tr)}")
    print("per-trajectory sps delta vs", base)
    for k in runs:
        print(f"  {k:34s} " + " ".join(f"{n}:{res[(k,f,n)]['sps']-res[(base,f,n)]['sps']:+.2f}" for n in names))
