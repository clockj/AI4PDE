"""Design B2, CENTRE-side probe: the interval centre gets its OWN causal bias shift, decoupled from the prediction's D1c.

Shipped (D1c): b = sum over revealed windows of mean_t(C0 - y) / (n + D1C_N0); P and the centre C both move by -b (C0 = y0, the
alpha-0 ensemble mean). Here P keeps D1C_N0 (cached, 30 or 10) and the CENTRE uses
    C = y0 - dsum / (n + N0C)  [- (dsumT - dsum) / (n + N0F)   per lead-time residual, optional]
with N0C < D1C_N0: the scored prediction has to be robust in rel/mvpe, the centre only has to be where the target is.
Asym R1 (shipped constants) is rebuilt on top of each centre (its table scores the moved centre), so order of operations =
shift first, calibrate the residual after.
    python b2centre.py <cache tag>   -> $S/ckpt/b2/res_<tag>_centre.json   (report: b2report.py <tag> --set centre --base c_ship)
"""
import os, sys, json, time
import numpy as np, torch
import b2lab as L
S = L.S
N0 = {"d30": 30.0, "d10": 10.0}


def CV(n0c, n0f=0.0, sm=1, **r1):
    return dict(n0c=n0c, n0f=n0f, csm=sm, r1=L.V(**r1))


def variants(n0p):
    v = {"c_ship": CV(n0p)}
    for n0c in (10.0, 5.0, 2.0, 1.0, 0.3):
        if n0c != n0p:
            v[f"c_n{n0c:g}"] = CV(n0c)
    v["c_n2_sm3"] = CV(2.0, sm=3); v["c_n5_sm3"] = CV(5.0, sm=3)
    v["c_n2_f10"] = CV(2.0, n0f=10.0); v["c_n2_f30"] = CV(2.0, n0f=30.0); v["c_n5_f30"] = CV(5.0, n0f=30.0)
    v["c_n2_glob"] = CV(2.0, prior="glob", n0g=0.25); v["c_n5_glob"] = CV(5.0, prior="glob", n0g=0.25)
    return v


def main():
    tag = sys.argv[1]; n0p = N0[tag.split("_")[1]]
    Z = np.load(f"{S}/ckpt/b2/{tag}.npz"); a = Z["a"]; names = Z["names"]
    P, C, H, Y = (torch.from_numpy(Z[k]) for k in ("P", "C", "h", "Y"))
    vs = variants(n0p); N = len(names)
    j0, j1 = L.GRIDS["g9"]; ks = torch.tensor([2.0 ** (j / 4.0) for j in range(j0, j1 + 1)], dtype=torch.float64); K = len(ks); ks32 = ks.float()
    con = {k: np.zeros(N) for k in vs}; wid = {k: np.zeros(N) for k in vs}; nsc = np.zeros(N); idx = np.zeros(N, int)
    t0 = time.time()
    for i in range(N):
        if i == 0 or names[i] != names[i - 1]:
            n = 0; dsum = torch.zeros((1, 32, 64, 2), dtype=torch.float64); dsumT = torch.zeros((20, 32, 64, 2), dtype=torch.float64)
            st = {k: dict(n=0, cum2=torch.zeros((K, K, 32, 64, 2))) for k in vs}
        idx[i] = n
        p, h, y = P[i], H[i], Y[i]
        y0 = (C[i].double() + dsum / (n + n0p)) if n > 0 else C[i].double()        # undo the cached D1c shift: y0 = C - D1, D1 = -dsum/(n+N0)
        scored = y != 0; nsc[i] = float(scored.sum())
        for k, v in vs.items():
            c = y0
            if n > 0:
                b = dsum / (n + v["n0c"])
                if v["n0f"] > 0:
                    b = b + (dsumT - dsum) / (n + v["n0f"])
                if v["csm"] > 1:
                    b = L.box(b, v["csm"], (1, 2))
                c = y0 - b
            c = c.float()
            klo, khi = L.kfields(st[k], v["r1"], ks, K)
            lo = torch.minimum(c - h * klo.float(), p); hi = torch.maximum(c + h * khi.float(), p)
            assert bool(torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo <= hi).all())
            w = torch.exp(-((hi - lo).double()) / L.SIG)
            ins = (y >= lo) & (y <= hi) & scored
            con[k][i] = float((w * ins).sum()); wid[k][i] = float(((hi - lo) * scored).sum())
            A, B = L.tables(p, c, h, y, ks32)
            st[k]["cum2"] += torch.einsum("athwc,bthwc->abhwc", A, B); st[k]["n"] += 1
        e = y0 - y.double(); dsum += e.mean(0, keepdim=True); dsumT += e; n += 1
        if i % 100 == 0:
            print(tag, "window", i, "%.0fs" % (time.time() - t0), flush=True)
    out = dict(tag=tag, names=[str(x) for x in names], a=a.tolist(), ref_sps=float(Z["ref_sps"]), ref_comp=float(Z["ref_comp"]),
               nsc=nsc.tolist(), idx=idx.tolist(), var={k: dict(con=con[k].tolist(), wid=wid[k].tolist(), cfg=str(vs[k])) for k in vs})
    for k in vs:
        print(f"{tag} {k:12s} sps {100.0 * float(a @ con[k]) / nsc.sum():.4f}", flush=True)
    print(f"(ref sps from probe.run {float(Z['ref_sps']):.4f})")
    json.dump(out, open(f"{S}/ckpt/b2/res_{tag}_centre.json", "w"))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("B2_THREADS", "3")))
    main()
