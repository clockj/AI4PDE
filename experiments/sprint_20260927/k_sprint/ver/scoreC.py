import sys, numpy as np
sys.path.insert(0, "/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/ver/wt/Track_2")
import stream_eval as se
D = sys.argv[1]
WT = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)
fin = lambda q: sum(WT[k] * q[k] for k in WT)
pad = lambda a: np.concatenate([a[..., :2], np.zeros(a.shape[:-1] + (1,), np.float32)], -1).astype(np.float32)
vu = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")
def load(n):
    z = np.load(f"{D}/{n}.npz")
    if "lo" in z:
        return pad(z["P"]), pad(z["lo"]), pad(z["hi"]), pad(vu["y"]), np.array([s.split(":")[0] for s in vu["ids"]])
    return pad(z["P"]), pad(z["L"]), pad(z["U"]), pad(z["Y"]), np.array([s.split(":")[0] for s in z["ids"]])
for lab, b, f in (("T1 predict N=56", "t1_base", "t1_fin"), ("T2 asym R1", "t2_base", "t2_fin"), ("T2 asym R1 + D1c30", "t2_base_d1c", "t2_fin_d1c")):
    try:
        B, F = load(b), load(f)
    except Exception as e:
        print(lab, "MISSING", e); continue
    names = B[4]; assert (names == F[4]).all() and np.abs(B[3] - F[3]).max() == 0
    aoa = np.array([int(n.split("_")[1]) for n in names])
    print(f"\n### {lab}: base={b} fin={f}")
    for sn, sel in (("all 8", np.ones(len(names), bool)), ("AoA 15-20", np.isin(aoa, [15, 20]))):
        q = {k: se.score(*(a[sel] for a in Z[:4]), 1e-3) for k, Z in (("base", B), ("fin", F))}
        for k in ("base", "fin"):
            x = q[k]; print(f"| {sn} | {k} | rel_err {x['rel_err']:.5f} | tke_err {x['tke_err']:.5f} | mvpe_err {x['mvpe_err']:.5f} | sps {x['sps']:.4f} | cov {x['coverage']:.4f} | final-no-time {fin(x):.4f} |")
        a, c = q["base"], q["fin"]
        tr = list(dict.fromkeys(names[sel])); nf = ns = nr = 0; per = []
        for n in tr:
            m = names == n
            p = {k: se.score(*(a_[m] for a_ in Z[:4]), 1e-3) for k, Z in (("base", B), ("fin", F))}
            nf += fin(p["fin"]) > fin(p["base"]); ns += p["fin"]["sps"] > p["base"]["sps"]; nr += p["fin"]["rel_err"] < p["base"]["rel_err"]
            per.append(f"{n} {p['fin']['sps']-p['base']['sps']:+.3f}")
        print(f"| {sn} | d | rel_err {100*(c['rel_err']/a['rel_err']-1):+.3f}% | tke_err {100*(c['tke_err']/a['tke_err']-1):+.3f}% | mvpe_err "
              f"{100*(c['mvpe_err']/a['mvpe_err']-1):+.3f}% | sps {c['sps']-a['sps']:+.4f} | cov {c['coverage']-a['coverage']:+.4f} | final-no-time "
              f"{fin(c)-fin(a):+.4f} | final up {nf}/{len(tr)}, sps up {ns}/{len(tr)}, rel_err down {nr}/{len(tr)} |")
        print("   per-traj d sps:", ", ".join(per))
