import sys, numpy as np
D = sys.argv[1]
for v in ("", "_d1c"):
    f, a, s = (np.load(f"{D}/{n}{v}.npz") for n in ("fin", "all", "sbt2"))
    P, L, U = f["P"], f["L"], f["U"]
    ok = all(np.isfinite(z).all() for z in (P, L, U)) and (L <= P).all() and (P <= U).all()
    print(f"B{v or ' asnp'}: B1 pred fin vs k27all max|d| {np.abs(P-a['P']).max():.3e} | B2 lower fin vs sbt2 max|d| {np.abs(L-s['L']).max():.3e} "
          f"upper {np.abs(U-s['U']).max():.3e}; pred fin vs sbt2 {np.abs(P-s['P']).max():.3e}; half-width (U-L) fin vs sbt2 max|d| "
          f"{np.abs((U-L)-(s['U']-s['L'])).max():.3e} mean rel {np.abs((U-L)-(s['U']-s['L']))[...,:2].mean()/(s['U']-s['L'])[...,:2].mean():.2e}"
          f" | finite&L<=P<=U {ok} p-chan max {max(np.abs(z[...,2]).max() for z in (P,L,U))} | overwritten {int(f['overwritten'])}")
    # is the L/U diff explained by the prediction diff? centre0 band: c0 = y0 unchanged; contain-P binds where P moves
    dP = np.abs(P - s["P"]).max(axis=None)
    m = (np.abs(L - s["L"]) > 1e-6) | (np.abs(U - s["U"]) > 1e-6)
    print(f"   elements with |dL|or|dU| > 1e-6: {m.mean()*100:.3f}%  ; of those, where P differs by > 1e-6: "
          f"{(m & (np.abs(P-s['P'])>1e-6)).sum()/max(m.sum(),1)*100:.1f}%; where P is at a bound (fin): {(m & ((np.abs(P-L)<1e-6)|(np.abs(P-U)<1e-6))).sum()/max(m.sum(),1)*100:.1f}%")
