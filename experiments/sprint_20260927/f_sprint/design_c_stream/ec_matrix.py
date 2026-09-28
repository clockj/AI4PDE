"""EC (executor, 09-27): candidate matrix + board-likeness gauge on design C's cached dumps (CPU; probe.run is the construction).
One process = one (dump, f, arm). Writes json: levels (full / aoa>=15) with errors + coverage, and per-trajectory composites."""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as pr

def arm_cfg(name):
    if name == "base":
        return {}
    d, v, f = name.split("_")            # d30_v50_f0
    c = dict(d1c_n0=float(d[1:]), d1c_after_fw=True)
    if float(v[1:]): c["vc_n0"] = float(v[1:])
    if float(f[1:]): c["fw_n0"] = float(f[1:])
    return c

def lev(P, L, U, Y):
    d = pr.score(P, L, U, Y)
    sc = Y != 0
    d["cov"] = float(np.mean(((Y >= L) & (Y <= U))[sc]))
    return d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump"); ap.add_argument("--band"); ap.add_argument("--f", type=float, default=1.0)
    ap.add_argument("--arm"); ap.add_argument("--out")
    ap.add_argument("--fam-f", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--fam-v", type=int, nargs="+", default=[3, 4])
    a = ap.parse_args()
    D = pr.load(a.dump, a.fam_f, a.fam_v); band = np.load(a.band)
    D["Pref"] = pr.run(D, band, {}, 1.0)[0] if a.f != 1.0 else None
    P, L, U, Yf = pr.run(D, band, arm_cfg(a.arm), a.f)
    hi = D["aoa"] >= 15
    res = dict(arm=a.arm, f=a.f, full=lev(P, L, U, Yf), a15=lev(P[hi], L[hi], U[hi], Yf[hi]),
               per={t: pr.score(P[D["names"] == t], L[D["names"] == t], U[D["names"] == t], Yf[D["names"] == t])["comp"]
                    for t in dict.fromkeys(D["names"].tolist())},
               aoa={t: int(t.split("_")[1]) for t in dict.fromkeys(D["names"].tolist())},
               nwin={t: int((D["names"] == t).sum()) for t in dict.fromkeys(D["names"].tolist())})
    json.dump(res, open(a.out, "w"), default=float)
    print(a.arm, a.f, "full comp %.4f a15 comp %.4f" % (res["full"]["comp"], res["a15"]["comp"]))
