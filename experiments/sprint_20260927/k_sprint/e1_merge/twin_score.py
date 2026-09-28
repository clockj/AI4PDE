"""Held-out score of a TWIN T1 package (members = held-out twins) on val_unseen, full + AoA 15-20. usage: twin_score.py <pkgdir> <tag> <savenpz>"""
import sys, os, importlib.util, json
import numpy as np
pkg, tag, save = sys.argv[1:4]
KIT = "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9"
d = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz"); X, Y, ids = d["x"], d["y"], d["ids"]
spec = importlib.util.spec_from_file_location("sc", os.path.join(KIT, "scoring.py")); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
os.chdir(pkg); sys.path.insert(0, pkg)
import submission
o = submission.predict(X)
P, lo, hi = o["prediction"].copy(), o["lower"].copy(), o["upper"].copy()
np.savez(save, prediction=P, lower=lo, upper=hi)
aoa = np.array([int(s.split(":")[0].split("_")[1]) for s in ids])
res = {"tag": tag}
for nm, m in (("full", np.ones(len(X), bool)), ("aoa15_20", aoa >= 15)):
    p, y, l, u = P[m], Y[m], lo[m], hi[m]
    r = float(np.mean(sc.rel_l2_per_sample(p, y, 2))); k = float(np.mean(sc.tke_rel_l2_per_sample(p, y, 2))); mv = float(sc.mvpe_rel_l2(p, y))
    sps = float(sc.score_sps(sc.aggregate_sps(p, y, 2, lower=l, upper=u)[0]))
    res[nm] = {"n": int(m.sum()), "rel": r, "tke": k, "mvpe": mv, "sps": sps}
print(json.dumps(res)); open(os.path.join(os.path.dirname(save), "twin_scores.jsonl"), "a").write(json.dumps(res) + "\n")
