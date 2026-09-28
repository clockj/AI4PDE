"""Design C phase 2 probe: stronger causal corrections of the trajectory-persistent error. CPU, cached dumps, physical units.
Every statistic at window k uses only windows < k of the same trajectory. Arms run in a fork pool."""
import argparse, importlib.util, os, sys, json, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "submit"))
KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py")); sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)
_m = importlib.util.spec_from_file_location("sub", str(HERE.parent / "submit" / "submission.py")); sub = importlib.util.module_from_spec(_m); _m.loader.exec_module(sub)
sub.R1_ASYM = True
MEAN = np.array([0.15496085584163666, -0.0005139928543940187], np.float32); STD = np.array([0.09680565446615219, 0.015960684046149254], np.float32)
W = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)
KS9 = tuple(0.5 * 2 ** (j / 4) for j in range(9))

def boxk(a, k):   # (...,H,W,C) replicate-padded kxk mean over H,W
    if k <= 1:
        return a
    r = k // 2
    p = np.pad(a, [(0, 0)] * (a.ndim - 3) + [(r, r), (r, r), (0, 0)], mode="edge")
    H, Wd = a.shape[-3], a.shape[-2]
    c = np.cumsum(np.cumsum(np.pad(p, [(0, 0)] * (a.ndim - 3) + [(1, 0), (1, 0), (0, 0)]), axis=-3), axis=-2)
    return (c[..., k:k + H, k:k + Wd, :] - c[..., :H, k:k + Wd, :] - c[..., k:k + H, :Wd, :] + c[..., :H, :Wd, :]) / float(k * k)

def score(P, L, U, Y):
    pad = lambda a: np.concatenate([a, np.zeros_like(a[..., :1])], -1).astype(np.float32)
    P, L, U, Y = pad(P), pad(L), pad(U), pad(Y)
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2))); k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))); m = float(sc.mvpe_rel_l2(P, Y))
    s, cov = sc.aggregate_sps(P, Y, 2, lower=L, upper=U)
    d = dict(rel=sc.score_error(r), tke=sc.score_error(k), mvpe=sc.score_error(m), sps=sc.score_sps(s), rel_e=r, tke_e=k, mvpe_e=m, cov=float(cov))
    d["comp"] = sum(W[q] * d[q] for q in W)
    return d

def run(D, band, cfg, f=1.0):
    yf, yv, sf, sv, Y, names = D["yf"], D["yv"], D["sf"], D["sv"], D["Y"], D["names"]
    N = len(Y); al = cfg.get("alpha", 0.8); w0 = 1.0 / 3.0
    base, sref, bp = band["base"][0, ..., :2], band["sigma_ref"][0, ..., :2], float(band["p"])
    Pn = np.zeros_like(Y); Ln = np.zeros_like(Y); Un = np.zeros_like(Y); Yf = np.zeros_like(Y)
    r1 = sub._CausalBandScalerT(cfg.get("r1_ks", KS9), cfg.get("r1_n0", 2.0), [1.0, 1.0], sub._SIGMA_G, "cpu", (32, 64, 2))
    bc = cfg.get("bias"); sat = []
    for i in range(N):
        if i == 0 or names[i] != names[i - 1]:
            r1.reset(); n = 0; prev = None
            E1 = np.zeros((20, 32, 64, 2)); E2 = np.zeros((1, 32, 64, 2)); SG = np.zeros((1, 32, 64, 2))
            G1 = np.zeros((1, 32, 64, 2)); G2 = np.zeros((1, 32, 64, 2))
            fA = np.zeros((32, 64, 2)); fB = np.zeros((32, 64, 2)); vT = np.zeros((1, 32, 64, 2)); vP = np.zeros((1, 32, 64, 2))
        if prev is not None:
            y = prev["Y"]; scored = torch.from_numpy(y != 0)
            r1.update_gated_asym(torch.from_numpy(prev["P"]), torch.from_numpy(prev["C"]), torch.from_numpy(prev["h"])[None].expand(1, 20, 32, 64, 2),
                                 torch.from_numpy(y), scored, torch.ones((), dtype=torch.float64))
            e = prev["Cd"] - y[0]; E1 += e; em = e.mean(0, keepdims=True); E2 += em * em; SG += (prev["sig"] * STD) ** 2
            cm = prev["Cd"].mean(0, keepdims=True); ym = y[0].mean(0, keepdims=True); G1 += cm * ym; G2 += cm * cm
            d = prev["yv"] - prev["yf"]; fA += (d * d).sum(0); fB += (d * (y[0] - prev["yf"])).sum(0)
            v = ((y[0] - MEAN) / STD).var(0, keepdims=True); vT += v; vP += prev["vpred"]
            n += 1
        w = w0
        if cfg.get("fw_n0", 0) and n > 0:
            ols = boxk(fB, 3) / np.maximum(boxk(fA, 3), 1e-12)
            w = np.clip(w0 + (np.clip(ols, 0.0, 1.0) - w0) * n / (n + cfg["fw_n0"]), 0.0, 1.0)
        y0 = (1 - w) * yf[i] + w * yv[i]
        sig = (1 - w0) * sf[i] + w0 * sv[i]
        yn = (y0 - MEAN) / STD
        m_ = yn.mean(0, keepdims=True); s_ = yn.std(0, keepdims=True)
        tgt = al * sig + (1 - al) * s_
        vpred = tgt ** 2
        if cfg.get("vc_n0", 0) and n > 0:
            g = boxk(vT, 3) / np.maximum(boxk(vP, 3), 1e-12); g = 1.0 + (np.clip(g, *cfg.get("vc_clip", (0.25, 4.0))) - 1.0) * n / (n + cfg["vc_n0"])
            if cfg.get("vc_g_n0", 0):        # pooled (per channel) ratio trusted early, per-pixel residual ratio trusted late
                Gp = vT.sum((1, 2), keepdims=True) / np.maximum(vP.sum((1, 2), keepdims=True), 1e-12)
                gl = boxk(vT, 3) / np.maximum(boxk(vP, 3), 1e-12) / Gp
                g = (1.0 + (np.clip(Gp, 0.25, 4.0) - 1.0) * n / (n + cfg["vc_g_n0"])) * (1.0 + (np.clip(gl, 0.25, 4.0) - 1.0) * n / (n + cfg["vc_n0"]))
            tgt = tgt * np.sqrt(g)
        P = (m_ + (yn - m_) * tgt / (s_ + 1e-6)) * STD + MEAN
        D1 = 0.0
        if bc and n > 0:
            n0 = bc.get("n0", 30.0); ks = bc.get("smooth", 1)
            bm = boxk(E1.mean(0, keepdims=True) / n, ks)                       # (1,H,W,2) time-mean bias estimate (smoothed)
            shr = n / (n + n0)
            mode = bc.get("mode", "d1c")
            if mode in ("eb", "ebg", "ebr"):
                c0 = np.asarray(bc.get("c", 0.0), dtype=np.float64); n0s = bc.get("n0s", 0.0)
                raw = E1.mean(0, keepdims=True) / n
                ss = np.maximum(E2 - n * raw * raw, 0.0)                       # sum of squared deviations of the window-mean errors
                if np.any(c0 > 0):
                    s2 = (n0s * c0 * (SG / n) + ss) / (n0s + n - 1.0)          # prior: s2 = c * sigma_head^2 (physical), n0s pseudo-windows
                else:
                    s2 = ss / max(n - 1.0, 1.0) if n > 1 else None
                if s2 is not None:
                    se2 = s2 / n / float(ks * ks if bc.get("white", False) else 1.0)
                    if mode == "ebg":
                        b2 = (bm * bm).mean((1, 2), keepdims=True); se2 = se2.mean((1, 2), keepdims=True) + 0 * se2
                        b2 = b2 + 0 * bm
                    elif mode == "ebr":
                        kr = bc.get("kr", 9); b2 = boxk(bm * bm, kr); se2 = boxk(se2, kr)
                    else:
                        b2 = boxk(bm * bm, 3); se2 = boxk(se2, 3)
                    tau2 = np.maximum(b2 - bc.get("kse", 1.0) * se2, 0.0)
                    sh_eb = tau2 / (tau2 + se2 + 1e-20) * (n / (n + bc.get("eb_n0", 0.0)))
                    shr = np.maximum(sh_eb * bc.get("eb_cap", 1.0), shr)
            b = bm * shr
            lead = bc.get("lead", "none")
            if lead == "rank1":
                bt = boxk(E1 / n, ks)                                           # (20,H,W,2)
                a = (bt * bm).sum((1, 2), keepdims=True) / np.maximum((bm * bm).sum((1, 2), keepdims=True), 1e-20)   # (20,1,1,2)
                a = 1.0 + (a - 1.0) * n / (n + bc.get("n0l", 5.0))
                b = b * np.clip(a, 0.0, 3.0)
            elif lead == "full":
                bt = boxk(E1 / n, ks)
                b = b + (bt - bm) * (n / (n + bc.get("n0l", 30.0)))
            elif lead == "lin":                                                # pooled 2-parameter ramp in lead time, per pixel slope shrunk hard
                bt = boxk(E1 / n, ks); tt = (np.arange(20) - 9.5)[:, None, None, None] / 9.5
                sl = (bt * tt).sum(0, keepdims=True) / (tt * tt).sum()
                b = b + sl * tt * (n / (n + bc.get("n0l", 30.0)))
            D1 = -b
            if bc.get("gain", 0) :
                gg = boxk(G1, 3) / np.maximum(boxk(G2, 3), 1e-20); gg = 1.0 + (np.clip(gg, 0.5, 2.0) - 1.0) * n / (n + bc["gain"])
                D1 = (gg - 1.0) * y0.mean(0, keepdims=True) + 0.0 * b
        P = (P + D1).astype(np.float32); C = (y0 + D1).astype(np.float32)
        h = (base * np.clip(sig * STD / sref, 0.3, 3.0) ** bp).astype(np.float32)
        if n > 0:
            klo, khi = r1.kfield_asym(); klo, khi = klo[0].numpy(), khi[0].numpy()
        else:
            klo = khi = np.ones((1, 32, 64, 2), np.float32)
        lo = np.minimum(C - h * klo, P); hi = np.maximum(C + h * khi, P)
        Pn[i], Ln[i], Un[i] = P, lo, hi
        yt = Y[i]
        if f != 1.0:
            ref = D["Pref"][i]; yt = np.where(Y[i] != 0, ref + f * (Y[i] - ref), 0.0).astype(np.float32)
        Yf[i] = yt
        prev = dict(P=P[None], C=C[None], h=h, Y=yt[None], Cd=y0, yf=yf[i], yv=yv[i], vpred=vpred, sig=sig)
        if n > 0 and (i + 1 == N or names[i + 1] != names[i]):
            kmax = max(cfg.get("r1_ks", KS9)); kmin = min(cfg.get("r1_ks", KS9))
            flat = r1.cum2.flatten(0, 1).argmax(0); K = len(r1.ks); al_, ah_ = (flat // K).numpy(), (flat % K).numpy()
            sat.append([float((al_ == K - 1).mean()), float((ah_ == K - 1).mean()), float((al_ == 0).mean()), float((ah_ == 0).mean())])
    return Pn, Ln, Un, Yf, np.mean(sat, 0) if sat else None

def load(path, fam_f, fam_v, trunc=0):
    z = np.load(path)
    names = z["names"]; keep = np.ones(len(names), bool)
    idx = np.zeros(len(names), int)
    for i in range(len(names)):
        idx[i] = 0 if (i == 0 or names[i] != names[i - 1]) else idx[i - 1] + 1
    if trunc:
        keep = idx < trunc
    y = z["y"][:, keep].astype(np.float32) * STD + MEAN; sg = z["sig"][:, keep].astype(np.float32)
    Yn = z["Y"][keep].astype(np.float32)
    D = dict(yf=y[list(fam_f)].mean(0), yv=y[list(fam_v)].mean(0), sf=sg[list(fam_f)].mean(0), sv=sg[list(fam_v)].mean(0), names=names[keep], idx=idx[keep])
    Yp = Yn * STD + MEAN
    D["Y"] = np.where(np.abs(Yp) < 1e-9, 0.0, Yp).astype(np.float32)
    D["aoa"] = np.array([int(s.split("_")[1]) for s in D["names"]])
    return D

BINS = [(0, 0), (1, 3), (4, 7), (8, 13), (14, 27), (28, 999)]
G = {}

def evaluate(job):
    nm, cfg, f = job
    D, band = G["D"], G["band"]
    P, L, U, Yf, sat = run(D, band, cfg, f)
    out = dict(full=score(P, L, U, Yf))
    hi = D["aoa"] >= 15
    out["a15"] = score(P[hi], L[hi], U[hi], Yf[hi]) if hi.any() else None
    out["per"] = {t: score(P[D["names"] == t], L[D["names"] == t], U[D["names"] == t], Yf[D["names"] == t])["comp"] for t in dict.fromkeys(D["names"])}
    out["bins"] = {}
    for a_, b_ in BINS:
        s = (D["idx"] >= a_) & (D["idx"] <= b_)
        if s.sum() >= 4:
            out["bins"][f"{a_}-{b_}"] = score(P[s], L[s], U[s], Yf[s])["comp"]
    out["sat"] = None if sat is None else [float(x) for x in sat]
    return nm, f, out

B30 = dict(mode="d1c", n0=30.0)
ARMS = {
    "base": dict(),
    "D1c30": dict(bias=B30),
    "D1c10": dict(bias=dict(mode="d1c", n0=10.0)),
    "D1c3": dict(bias=dict(mode="d1c", n0=3.0)),
    "KNOB(D1c10+VC30+FW8)": dict(bias=dict(mode="d1c", n0=10.0), vc_n0=30.0, fw_n0=8.0),
    "CONS(D1c30+VC50)": dict(bias=B30, vc_n0=50.0),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="in", choices=["in", "ood"])
    ap.add_argument("--band", default="/users/PAS1187/qwu/workspace/realpde/main/Track_1/bands/r35_centre0_fill.npz")
    ap.add_argument("--f", type=float, nargs="+", default=[1.0]); ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--armfile", default=None, help="python file defining ARMS2 = {name: cfg}")
    ap.add_argument("--ref", default="base"); ap.add_argument("--trunc", type=int, default=0)
    ap.add_argument("--json", default=None); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    S = "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/"
    if a.regime == "in":
        D = load(S + "twins_unseen.npz", (0, 1, 2), (3, 4), a.trunc)
    else:
        D = load(S + "ood_aoa20.npz", (0, 1), (4,), a.trunc)
    band = np.load(a.band); band = {k: band[k] for k in band.files}
    if a.armfile:
        ns = {}; exec(open(a.armfile).read(), ns); ARMS.update(ns["ARMS2"])
    sel = [k for k in ARMS if (a.arms is None or k in a.arms or k in ("base", a.ref))]
    if any(f != 1.0 for f in a.f):
        D["Pref"] = run(D, band, ARMS["base"], 1.0)[0]
    G["D"], G["band"] = D, band
    jobs = [(k, ARMS[k], f) for f in a.f for k in sel]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("fork").Pool(min(a.procs, len(jobs))) as pool:
        res = pool.map(evaluate, jobs, chunksize=1)
    R = {(nm, f): o for nm, f, o in res}
    for f in a.f:
        for ref in dict.fromkeys(["base", a.ref]):
            b = R[(ref, f)]
            print(f"--- regime {a.regime} trunc {a.trunc} f={f:.2f}  reference = {ref}: comp {b['full']['comp']:.4f} rel_e {b['full']['rel_e']:.4f} tke_e {b['full']['tke_e']:.4f} mvpe_e {b['full']['mvpe_e']:.4f} "
                  f"| rel {b['full']['rel']:.3f} tke {b['full']['tke']:.3f} mvpe {b['full']['mvpe']:.3f} sps {b['full']['sps']:.3f} cov {b['full']['cov']:.3f} sat {b['sat']}")
            if b["a15"] and a.regime == "in":
                x = b["a15"]; print(f"    a15-20: comp {x['comp']:.4f} rel_e {x['rel_e']:.4f} tke_e {x['tke_e']:.4f} mvpe_e {x['mvpe_e']:.4f} | rel {x['rel']:.3f} tke {x['tke']:.3f} mvpe {x['mvpe']:.3f} sps {x['sps']:.3f} cov {x['cov']:.3f}")
            for k in sel:
                o = R[(k, f)]
                dp = [o["per"][t] - b["per"][t] for t in o["per"]]; d15 = [o["per"][t] - b["per"][t] for t in o["per"] if int(t.split("_")[1]) >= 15]
                q = lambda s, key: o[s][key] - b[s][key]
                bins = " ".join(f"{kk}:{o['bins'][kk] - b['bins'][kk]:+.3f}" for kk in o["bins"])
                line = f"{k:34s} d {q('full','comp'):+.4f} ({sum(x > 0 for x in dp)}/{len(dp)}) [rel {q('full','rel'):+.3f} tke {q('full','tke'):+.3f} mvpe {q('full','mvpe'):+.3f} sps {q('full','sps'):+.3f}]"
                if a.regime == "in":
                    line += f" | a15 {q('a15','comp'):+.4f} ({sum(x > 0 for x in d15)}/{len(d15)}) [rel {q('a15','rel'):+.3f} tke {q('a15','tke'):+.3f} mvpe {q('a15','mvpe'):+.3f} sps {q('a15','sps'):+.3f}]"
                print(line + f" | n-bins {bins}", flush=True)
    print(f"elapsed {time.time() - t0:.0f}s")
    if a.json:
        json.dump({f"{k[0]}@{k[1]}": v for k, v in R.items()}, open(a.json, "w"), indent=1, default=float)
