import sys, numpy as np
D = sys.argv[1]
for N in (14, 56, 99, 336):
    f, a = np.load(f"{D}/fin_{N}.npz"), np.load(f"{D}/all_{N}.npz")
    P, lo, hi = f["P"], f["lo"], f["hi"]
    d = np.abs(P - a["P"]).max()
    ok = np.isfinite(P).all() and np.isfinite(lo).all() and np.isfinite(hi).all() and (lo <= P).all() and (P <= hi).all()
    p0 = max(np.abs(z[..., 2]).max() for z in (P, lo, hi))
    print(f"A1 N={N:3d} pred fin vs k27all max|d| {d:.3e} | A2 finite&lo<=P<=hi {ok} p-chan max {p0} shape {P.shape} dtype {f['dtype']} "
          f"(all dtype {a['dtype']}) keys {f['keys']} typ {f['typ']} overwritten fin {int(f['overwritten'])} all {int(a['overwritten'])}"
          f" | band width fin/all mean {(hi-lo)[...,:2].mean()/(a['hi']-a['lo'])[...,:2].mean():.4f}")
f = np.load(f"{D}/fin_56.npz")
for N in (14, 99, 336):
    g = np.load(f"{D}/fin_{N}.npz")
    print(f"A4 fin N={N} vs N=56: pred {np.abs(g['P']-f['P']).max():.3e} lo {np.abs(g['lo']-f['lo']).max():.3e} hi {np.abs(g['hi']-f['hi']).max():.3e}")
try:
    x = np.load(f"{D}/x27_56.npz"); print(f"A2 x27 baseline: shape {x['P'].shape} dtype {x['dtype']} keys {x['keys']} typ {x['typ']}; pred fin vs x27 max|d| {np.abs(x['P']-f['P']).max():.3e}")
except Exception as e: print("x27 cmp fail", e)
