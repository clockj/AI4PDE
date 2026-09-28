"""K0 'SIGBLEND' band (commander, 2026-09-27): geometric blend of the shipped per-element band with a low-dimensional
sigma power law, in a form the EXISTING package code evaluates:

    half = base' * clip(sigma/sigma_ref, .3, 3)^p' * exp(lam * q_c * log(sigma + 1e-8))
    base' = base^(1-lam) * kappa_{t,c}^lam          p' = (1-lam) * p          (npz only)
    EBAND = (((lam*q_u, 0,0,0,0, 0), (lam*q_v, 0,0,0,0, 0)), "mult", None)    (T1 build knob RPDE_SHIP_EBAND)

kappa_{t,c} (per frame, per channel) and q_c (per channel) come from the exact per-bin optimum of the SPS interval term on
the HELD-OUT twins' errors (val_unseen, 8 trajectories never trained on), |y0 - y| * infl, binned by the twins' own sigma.
Evaluation: leave-one-trajectory-out over the 8 (the power law never sees the trajectory it is scored on); the shipped
band itself was fitted on val_seen and is honest on val_unseen.  Final constants: fitted on all 8.

usage: B_DUMP=<dump_twins.py output> python make_k0_band.py OUTDIR
env: K0_BAND (the static band, default Track_1/bands/r38_a080_c0_fill.npz), K0_INFL (default 1.29), K0_NB (default 16)
"""
import json, os, sys
import numpy as np
from bandlab import Lab, FS
from sps9 import SIGMA

BAND = os.environ.get("K0_BAND", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bands", "r38_a080_c0_fill.npz"))
INFL = float(os.environ.get("K0_INFL", "1.29")); NB = int(os.environ.get("K0_NB", "16"))
LAMS = (0.3, 0.5, 0.7)
if len(sys.argv) < 2:
    sys.exit(__doc__)
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

L = Lab(); y0, sig, y, P, scored, traj = L.y0, L.sig, L.y, L.P, L.scored, L.traj
E = np.abs(y0 - y).astype(np.float32); a1 = L.a(1.0)
names, hi = L.names, L.hi
lo_names = [t for t in names if t not in hi]
z = np.load(BAND)
base3 = z["base"].astype(np.float32); sref3 = z["sigma_ref"].astype(np.float32); p_s = float(z["p"])
base_s, sref = base3[..., :2], sref3[..., :2]
clipr = np.clip(sig / sref, 0.3, 3.0).astype(np.float32)             # (N,1,32,64,2)
logsig = np.log(sig.astype(np.float32) + np.float32(1e-8))           # the EBAND 'logsig' feature, (N,1,32,64,2)


def fit_pl(fit, infl=INFL, nb=NB):
    """-> q (2,), logk (20, 2)"""
    W = (a1[fit][:, None, None, None, None] * scored[fit]).astype(np.float64)
    Ef, Ls = E[fit], np.broadcast_to(logsig, y0.shape)[fit]
    q = np.zeros(2); logk = np.zeros((20, 2))
    for c in range(2):
        X, Y, T = [], [], []
        for t in range(20):
            A = (Ef[:, t, :, :, c] * np.float32(infl)).ravel(); s = Ls[:, t, :, :, c].ravel(); w = W[:, t, :, :, c].ravel()
            ok = w > 0; A, s, w = A[ok], s[ok], w[ok]
            edges = np.quantile(s, np.linspace(0, 1, nb + 1)); edges[0], edges[-1] = -np.inf, np.inf
            bi = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, nb - 1)
            for j in range(nb):
                m = bi == j
                Aj, wj = A[m], w[m]
                o = np.argsort(Aj, kind="stable"); Aj, wj = Aj[o], wj[o]
                gain = np.cumsum(wj) * np.exp(-2.0 * Aj.astype(np.float64) / SIGMA)
                X.append(np.median(s[m])); Y.append(np.log(max(float(Aj[gain.argmax()]), 1e-6))); T.append(t)
        X, Y, T = np.array(X), np.array(Y), np.array(T)
        xm = np.array([X[T == t].mean() for t in range(20)]); ym = np.array([Y[T == t].mean() for t in range(20)])
        q[c] = float(((X - xm[T]) * (Y - ym[T])).sum() / ((X - xm[T]) ** 2).sum())      # pooled slope, per-frame intercepts
        logk[:, c] = ym - q[c] * xm
    return q, logk


def half(lam, q, logk, ev):
    """exactly the package formula, numpy"""
    bp = (base_s ** np.float32(1 - lam)) * np.exp(lam * logk).astype(np.float32)[None, :, None, None, :]
    return (bp * clipr[ev] ** np.float32((1 - lam) * p_s) * np.exp(np.float32(lam) * q.astype(np.float32) * logsig[ev])).astype(np.float32)


def score(hfun):
    res = {f: {} for f in FS}
    for t in names:
        ev = traj == t; fit = ~ev
        h = hfun(fit, ev)
        lo_, up_ = np.minimum(y0[ev] - h, P[ev]), np.maximum(y0[ev] + h, P[ev])
        sc_ = scored[ev]; n = int(sc_.sum())
        for f in FS:
            yf = y0[ev] + np.float32(f) * (y[ev] - y0[ev])
            ins = (yf >= lo_) & (yf <= up_) & sc_
            con = np.where(ins, np.exp(-(up_ - lo_).astype(np.float64) / SIGMA), 0.0)
            af = L.a(f)[ev]
            res[f][t] = (100.0 * float(af @ con.reshape(len(con), -1).sum(1)) / n, float(af.mean()), 100.0 * float(con.sum()) / n,
                         100.0 * float(ins.sum()) / n, float((up_ - lo_)[sc_].mean()))
    return res


def table(name, res, ref=None):
    for f in FS:
        row = [f"{name:<34s} f={f:4.2f}"]
        for lab, ts in (("hi", hi), ("lo", lo_names), ("all", names)):
            s = np.mean([res[f][t][0] for t in ts]); c = np.mean([res[f][t][2] for t in ts])
            cov = np.mean([res[f][t][3] for t in ts]); w = np.mean([res[f][t][4] for t in ts])
            txt = f"{lab}: sps {s:6.3f} C {c:6.3f} cov {cov:5.1f} w {w:.4f}"
            if ref is not None:
                dc = [res[f][t][2] - ref[f][t][2] for t in ts]; ds = [res[f][t][0] - ref[f][t][0] for t in ts]
                txt += f" dsps {np.mean(ds):+.3f} dC {np.mean(dc):+.3f} ({sum(d > 0 for d in dc)}/{len(ts)})"
            row.append(txt)
        print(" | ".join(row), flush=True)
    if ref is not None:
        print("    per trajectory dC (f=1.29): " + "  ".join(f"{t} {res[1.29][t][2] - ref[1.29][t][2]:+.3f}" for t in names), flush=True)


print(f"shipped band {BAND}  p {p_s}  infl {INFL}  nb {NB}", flush=True)
ref = score(lambda fit, ev: (base_s * clipr[ev] ** np.float32(p_s)).astype(np.float32)); table("A shipped band (real npz)", ref)
memo = {}
def pl(fit):
    k = fit.tobytes()
    if k not in memo:
        memo[k] = fit_pl(fit)
    return memo[k]
for lam in LAMS + (1.0,):
    table(f"SIGBLEND lam {lam} (LOTO-7)", score(lambda fit, ev, lam=lam: half(lam, *pl(fit), ev)), ref)

q, logk = fit_pl(np.ones(len(traj), bool))
print("FINAL fit on all 8:  q (u, v) =", np.round(q, 4).tolist(), flush=True)
print("FINAL log kappa per frame (u):", np.round(logk[:, 0], 3).tolist())
print("FINAL log kappa per frame (v):", np.round(logk[:, 1], 3).tolist(), flush=True)
for lam in LAMS:
    bp = np.zeros_like(base3)
    bp[..., :2] = (base_s ** np.float32(1 - lam)) * np.exp(lam * logk).astype(np.float32)[None, :, None, None, :]
    keys = {k: z[k] for k in z.files}
    keys.update(base=bp.astype(np.float32), p=np.float32((1 - lam) * p_s))
    out = os.path.join(OUT, f"band_k0_sigblend_lam{lam:.1f}.npz")
    np.savez(out, **keys)
    eb = (tuple([float(np.float32(lam * q[0])), 0.0, 0.0, 0.0, 0.0, 0.0]), tuple([float(np.float32(lam * q[1])), 0.0, 0.0, 0.0, 0.0, 0.0]))
    lit = repr((eb, "mult", None))
    json.dump({"lam": lam, "q": q.tolist(), "logkappa": logk.tolist(), "infl": INFL, "p_shipped": p_s, "p_new": (1 - lam) * p_s,
               "EBAND": lit, "T2_sigpow": [lam * q[0], lam * q[1]], "shipped_band": BAND, "npz": out},
              open(os.path.join(OUT, f"band_k0_sigblend_lam{lam:.1f}.json"), "w"), indent=1)
    # in-sample sanity of the written constants (NOT evidence)
    h = half(lam, q, logk, np.ones(len(traj), bool))
    print(f"wrote {out}  p' {(1 - lam) * p_s:.4f}  RPDE_SHIP_EBAND='{lit}'   median half {np.median(h):.5f} (shipped {np.median(base_s * clipr ** np.float32(p_s)):.5f})", flush=True)
