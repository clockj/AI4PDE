"""Paired table from stream_eval --save npz files: off vs on for static / asym / asymd1c; all 8 and AoA 15-20; per-trajectory signs."""
import sys, os
import numpy as np
W = "/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/wt_2xr"
sys.path.insert(0, W + "/Track_2"); sys.path.insert(0, W + "/Track_1")
sys.argv = [sys.argv[0]] + sys.argv[1:]
import stream_eval as se
import t2_common as t2
O = sys.argv[1]
WT = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)
fin = lambda d: sum(WT[k] * d[k] for k in WT)
for mode in ("static", "asym", "asymd1c"):
    R = {k: np.load(f"{O}/{mode}_{k}.npz") for k in ("off", "on")}
    ids = R["off"]["ids"]; names = np.array([s.split(":")[0] for s in ids])
    aoa = np.array([t2.aoa_of(n) for n in names])
    for sel_name, sel in (("all8", np.ones(len(ids), bool)), ("aoa15-20", np.isin(aoa, [15, 20]))):
        ds = {}
        for k, z in R.items():
            ds[k] = se.score(z["pred"][sel], z["lower"][sel], z["upper"][sel], z["target"][sel], 1e-3)
        a, b = ds["off"], ds["on"]
        print(f"{mode:8s} {sel_name:9s} OFF rel {a['rel_err']:.5f} tke {a['tke_err']:.5f} mvpe {a['mvpe_err']:.5f} "
              f"| scores rel {a['rel']:.4f} tke {a['tke']:.4f} mvpe {a['mvpe']:.4f} sps {a['sps']:.4f} final4 {fin(a):.4f}")
        print(f"{'':8s} {'':9s} ON  rel {b['rel_err']:.5f} tke {b['tke_err']:.5f} mvpe {b['mvpe_err']:.5f} "
              f"| scores rel {b['rel']:.4f} tke {b['tke']:.4f} mvpe {b['mvpe']:.4f} sps {b['sps']:.4f} final4 {fin(b):.4f}")
        print(f"{'':8s} {'':9s} d   rel_err {100*(b['rel_err']/a['rel_err']-1):+.3f}% tke_err {100*(b['tke_err']/a['tke_err']-1):+.4f}% "
              f"mvpe_err {100*(b['mvpe_err']/a['mvpe_err']-1):+.4f}% | d rel {b['rel']-a['rel']:+.4f} tke {b['tke']-a['tke']:+.4f} "
              f"mvpe {b['mvpe']-a['mvpe']:+.4f} sps {b['sps']-a['sps']:+.4f} final4 {fin(b)-fin(a):+.4f}")
        # per trajectory
        tr = [n for n in dict.fromkeys(names[sel])]
        nrel = nfin = 0
        for n in tr:
            m = names == n
            p = {k: se.score(z["pred"][m], z["lower"][m], z["upper"][m], z["target"][m], 1e-3) for k, z in R.items()}
            nrel += p["on"]["rel_err"] < p["off"]["rel_err"]
            nfin += fin(p["on"]) > fin(p["off"])
        print(f"{'':8s} {'':9s} per-trajectory: rel err down {nrel}/{len(tr)}, final4 up {nfin}/{len(tr)}")
