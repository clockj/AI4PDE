"""Score saved stream_eval dumps: all 8, AoA 15-20, per trajectory. usage: mscore.py <out.jsonl> tag=npz ..."""
import sys, json, importlib.util, numpy as np
KIT = "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py"
spec = importlib.util.spec_from_file_location("sc", KIT); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
W = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)
def score(P, L, U, Y):
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2))); k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))); m = float(sc.mvpe_rel_l2(P, Y))
    s, cov = sc.aggregate_sps(P, Y, 2, lower=L, upper=U)
    d = dict(rel_err=r, tke_err=k, mvpe_err=m, coverage=float(cov), rel=sc.score_error(r), tke=sc.score_error(k), mvpe=sc.score_error(m), sps=float(sc.score_sps(s)))
    d["fnt"] = sum(W[x] * d[x] for x in W)
    return d
out = sys.argv[1]
for a in sys.argv[2:]:
    tag, f = a.split("=", 1)
    z = np.load(f); P, Y, L, U, ids = z["pred"], z["target"], z["lower"], z["upper"], z["ids"]
    tr = np.array([s.split(":")[0] for s in ids]); aoa = np.array([int(t.split("_")[1]) for t in tr])
    res = {"tag": tag, "all": score(P, L, U, Y), "aoa15_20": score(P[aoa >= 15], L[aoa >= 15], U[aoa >= 15], Y[aoa >= 15]), "traj": {}}
    for t in dict.fromkeys(tr):
        m = tr == t; res["traj"][t] = score(P[m], L[m], U[m], Y[m])
    open(out, "a").write(json.dumps(res) + "\n")
    print(tag, "all", {k: round(v, 5) for k, v in res["all"].items()}, flush=True)
