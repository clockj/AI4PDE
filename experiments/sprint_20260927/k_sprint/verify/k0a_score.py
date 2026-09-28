"""E0a: score package dumps (k0a_dump.py) on val_unseen with the kit v9 scorer, per trajectory, at target stretch f
(y_f = c0 + f (y - c0), c0 = the package's alpha-0 centre), paired against the first dump.
usage: k0a_score.py <ref_dump> <arm_dump> [<arm_dump> ...]"""
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("sc", "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
Y = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/val_unseen.npz")["y"]
HI = ["21600_15", "8850_20", "11400_20"]


def read(path, f):
    D = np.load(path); traj = np.array([str(i).split(":")[0] for i in D["ids"]])
    names = sorted(set(traj), key=lambda t: (int(t.split("_")[1]), int(t.split("_")[0])))
    P, lo, hi, c0 = D["P"], D["lo"], D["hi"], D["c0"]
    res = {}
    for t in names:
        m = traj == t
        y = Y[m].astype(np.float32).copy()
        y[..., :2] = c0[m] + np.float32(f) * (y[..., :2] - c0[m])
        r = float(np.mean(sc.rel_l2_per_sample(P[m], y, 2))); k = float(np.mean(sc.tke_rel_l2_per_sample(P[m], y, 2)))
        mv = sc.mvpe_rel_l2(P[m], y)
        g = lambda e: 0.5 / (0.5 + e)
        abar = float(np.mean(0.5 * g(sc.rel_l2_per_sample(P[m], y, 2)) + 0.3 * g(sc.tke_rel_l2_per_sample(P[m], y, 2))
                             + 0.2 * g(sc.mvpe_rel_l2_per_sample(P[m], y))))
        w, cov = sc.aggregate_sps(P[m], y, 2, lower=lo[m], upper=hi[m])
        res[t] = dict(rel=r, tke=k, mvpe=mv, sps=100 * w, cov=100 * cov, C=100 * w / abar, n=int(m.sum()))
    return names, res


def agg(res, ts, k):
    return float(np.mean([res[t][k] for t in ts]))   # trajectory-mean (equal weight per trajectory)


paths = sys.argv[1:]
for f in (1.00, 1.29):
    names, R = read(paths[0], f)
    print(f"\n######## stretch f = {f:.2f}   (trajectory-mean; C = sps/(100 abar) x100; cov %)")
    print(f"{'arm':<10s} {'traj':<9s} {'rel':>8s} {'tke':>8s} {'mvpe':>8s} {'sps':>8s} {'cov':>6s} {'C':>8s}")
    for t in names:
        r = R[t]; print(f"{'ref':<10s} {t:<9s} {r['rel']:8.5f} {r['tke']:8.5f} {r['mvpe']:8.5f} {r['sps']:8.4f} {r['cov']:6.2f} {r['C']:8.4f}")
    for lab, ts in (("AoA15-20", HI), ("all8", names)):
        print(f"{'ref':<10s} {lab:<9s} " + " ".join(f"{agg(R, ts, k):8.5f}" for k in ("rel", "tke", "mvpe")) +
              f" {agg(R, ts, 'sps'):8.4f} {agg(R, ts, 'cov'):6.2f} {agg(R, ts, 'C'):8.4f}")
    for pth in paths[1:]:
        arm = pth.split("/")[-1].replace(".npz", "")
        _, S = read(pth, f)
        for t in names:
            d = {k: S[t][k] - R[t][k] for k in ("rel", "tke", "mvpe", "sps", "cov", "C")}
            print(f"{arm:<10s} {t:<9s} d: rel {d['rel']:+.1e} tke {d['tke']:+.1e} mvpe {d['mvpe']:+.1e} sps {d['sps']:+.4f} cov {d['cov']:+.2f} C {d['C']:+.4f}")
        for lab, ts in (("AoA15-20", HI), ("all8", names)):
            dC = [S[t]["C"] - R[t]["C"] for t in ts]; ds = [S[t]["sps"] - R[t]["sps"] for t in ts]
            print(f"{arm:<10s} {lab:<9s} PAIRED dsps {np.mean(ds):+.4f} dC {np.mean(dC):+.4f} ({sum(x > 0 for x in dC)}/{len(ts)} up)  "
                  f"cov {agg(S, ts, 'cov'):.2f} (ref {agg(R, ts, 'cov'):.2f})  d-rel/tke/mvpe max|.| "
                  f"{max(abs(S[t][k]-R[t][k]) for t in ts for k in ('rel','tke','mvpe')):.1e}")
