"""Design C probe: causal stream stack on cached held-out twin outputs (CPU). Physical units throughout.
Baseline = today's planned construction: mean of families (w_vit = 1/3), ALPHA 0.80, centre0 band, asym R1 (N0 2).
Arms add causal corrections. Every statistic at window k uses only windows < k of the same trajectory."""
import argparse, importlib.util, os, sys, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "submit"))
KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py")); sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)
_m = importlib.util.spec_from_file_location("sub", str(HERE.parent / "submit" / "submission.py")); sub = importlib.util.module_from_spec(_m); _m.loader.exec_module(sub)
sub.R1_ASYM = True
MEAN = np.array([0.15496085584163666, -0.0005139928543940187], np.float32); STD = np.array([0.09680565446615219, 0.015960684046149254], np.float32)
W = dict(rel=0.4778, tke=0.1010, mvpe=0.0921, sps=0.2498)

def box3(a):   # (...,H,W,C) replicate-padded 3x3 mean over H,W
    p = np.pad(a, [(0, 0)] * (a.ndim - 3) + [(1, 1), (1, 1), (0, 0)], mode="edge")
    H, Wd = a.shape[-3], a.shape[-2]
    return sum(p[..., i:i + H, j:j + Wd, :] for i in range(3) for j in range(3)) / 9.0

def score(P, L, U, Y):
    pad = lambda a: np.concatenate([a, np.zeros_like(a[..., :1])], -1).astype(np.float32)
    P, L, U, Y = pad(P), pad(L), pad(U), pad(Y)
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2))); k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))); m = float(sc.mvpe_rel_l2(P, Y))
    s, cov = sc.aggregate_sps(P, Y, 2, lower=L, upper=U)
    d = dict(rel=sc.score_error(r), tke=sc.score_error(k), mvpe=sc.score_error(m), sps=sc.score_sps(s), rel_e=r, tke_e=k, mvpe_e=m)
    d["comp"] = sum(W[q] * d[q] for q in W)
    return d

def run(D, band, cfg, f=1.0):
    """cfg keys: alpha, fw_n0 (0=off), fw_mode, d1c_n0 (0=off), d1c_eb (bool), d1c_after_fw, sigema, sig_beta, sig_n0, vc_n0 (0=off), r1_n0"""
    yf, yv, sf, sv, Y, names = D["yf"], D["yv"], D["sf"], D["sv"], D["Y"], D["names"]
    N = len(Y); al = cfg.get("alpha", 0.8); w0 = 1.0 / 3.0
    base, sref, bp = band["base"][0, ..., :2], band["sigma_ref"][0, ..., :2], float(band["p"])
    Pn = np.zeros_like(Y); Ln = np.zeros_like(Y); Un = np.zeros_like(Y); Yf = np.zeros_like(Y)
    r1 = sub._CausalBandScalerT(sub.R1_KS, cfg.get("r1_n0", 2.0), [1.0, 1.0], sub._SIGMA_G, "cpu", (32, 64, 2))
    prev = None
    for i in range(N):
        if i == 0 or names[i] != names[i - 1]:
            r1.reset(); n = 0; prev = None
            dsum = np.zeros((1, 32, 64, 2)); dsq = np.zeros((1, 32, 64, 2))
            fA = np.zeros((32, 64, 2)); fB = np.zeros((32, 64, 2))
            vS = np.zeros((1, 32, 64, 2)); vW = 0.0; vT = np.zeros((1, 32, 64, 2)); vP = np.zeros((1, 32, 64, 2))
        if prev is not None:     # reveal window i-1's target
            y = prev["Y"]; scored = torch.from_numpy(y != 0)
            r1.update_gated_asym(torch.from_numpy(prev["P"]), torch.from_numpy(prev["C"]), torch.from_numpy(prev["h"])[None].expand(1, 20, 32, 64, 2),
                                 torch.from_numpy(y), scored, torch.ones((), dtype=torch.float64))
            e = (prev["Cd"] - y[0]).mean(0, keepdims=True); dsum += e; dsq += e * e
            d = prev["yv"] - prev["yf"]; fA += (d * d).sum(0); fB += (d * (y[0] - prev["yf"])).sum(0)
            v = (y[0] - MEAN) / STD; v = v.var(0, keepdims=True)
            b = cfg.get("sig_beta", 0.8); vS = b * vS + v; vW = b * vW + 1.0
            vT += v; vP += prev["vpred"]
            n += 1
        # ---- this window, from past only
        w = w0
        if cfg.get("fw_n0", 0) and n > 0:
            ols = box3(fB) / np.maximum(box3(fA), 1e-12)
            w = np.clip(w0 + (np.clip(ols, cfg.get("fw_lo", 0.0), cfg.get("fw_hi", 1.0)) - w0) * n / (n + cfg["fw_n0"]), 0.0, 1.0)
        y0 = (1 - w) * yf[i] + w * yv[i]            # physical, (20,H,W,2)
        y0raw = (1 - w0) * yf[i] + w0 * yv[i]
        sig = (1 - w0) * sf[i] + w0 * sv[i]         # model-space sigma (1,H,W,2)
        yn = (y0 - MEAN) / STD
        m_ = yn.mean(0, keepdims=True); s_ = yn.std(0, keepdims=True)
        sa = sig
        if cfg.get("sigema") and n > 0:
            wt = n / (n + cfg.get("sig_n0", 3.0)); sa = np.sqrt(wt * vS / vW + (1 - wt) * sig * sig)
        tgt = al * sa + (1 - al) * s_
        if cfg.get("vc_n0", 0) and n > 0:           # causal variance-ratio calibration of the composed amplitude
            g = box3(vT) / np.maximum(box3(vP), 1e-12); g = 1.0 + (np.clip(g, *cfg.get("vc_clip", (0.25, 4.0))) - 1.0) * n / (n + cfg["vc_n0"])
            tgt = tgt * np.sqrt(g)
        vpred = (al * sig + (1 - al) * s_) ** 2     # uncalibrated composed variance (for the ratio)
        P = (m_ + (yn - m_) * tgt / (s_ + 1e-6)) * STD + MEAN
        Cd = y0 if cfg.get("d1c_after_fw", True) else y0raw      # the field whose bias D1c accumulates
        D1 = 0.0
        if cfg.get("d1c_n0", 0) and n > 0:
            bm = dsum / n
            if cfg.get("d1c_eb"):
                var = np.maximum(dsq / n - bm * bm, 0.0) * (n / max(n - 1, 1)) if n > 1 else None
                if var is None:
                    shr = n / (n + cfg["d1c_n0"])
                else:
                    se2 = box3(var) / n; b2 = box3(bm * bm); tau2 = np.maximum(b2 - se2, 0.0)
                    shr_eb = tau2 / (tau2 + se2 + 1e-20)
                    shr = np.maximum(shr_eb * n / (n + cfg.get("eb_n0", 2.0)), n / (n + cfg["d1c_n0"])) if cfg.get("eb_max", True) else shr_eb * n / (n + cfg.get("eb_n0", 2.0))
                D1 = -bm * shr
            else:
                D1 = -dsum / (n + cfg["d1c_n0"])
        P = (P + D1).astype(np.float32); C = (y0 + D1).astype(np.float32)
        h = (base * np.clip(sig * STD / sref, 0.3, 3.0) ** bp).astype(np.float32)        # (20,H,W,2)
        klo, khi = (r1.kfield_asym() if n > 0 else (torch.ones(1, 1, 32, 64, 2), torch.ones(1, 1, 32, 64, 2)))
        klo, khi = klo[0].numpy(), khi[0].numpy()
        lo = np.minimum(C - h * klo, P); hi = np.maximum(C + h * khi, P)
        Pn[i], Ln[i], Un[i] = P, lo, hi
        yt = Y[i]
        if f != 1.0:
            ref = D["Pref"][i]; yt = np.where(Y[i] != 0, ref + f * (Y[i] - ref), 0.0).astype(np.float32)
        Yf[i] = yt
        prev = dict(P=P[None], C=C[None], h=h, Y=yt[None], Cd=Cd, yf=yf[i], yv=yv[i], vpred=vpred)
    return Pn, Ln, Un, Yf

def load(path, fam_f=(0, 1, 2), fam_v=(3, 4)):
    z = np.load(path)
    y = z["y"].astype(np.float32) * STD + MEAN; sg = z["sig"].astype(np.float32)
    D = dict(yf=y[list(fam_f)].mean(0), yv=y[list(fam_v)].mean(0), sf=sg[list(fam_f)].mean(0), sv=sg[list(fam_v)].mean(0),
             Y=(z["Y"].astype(np.float32) * STD + MEAN), names=z["names"])
    D["Y"] = np.where(np.abs(z["Y"] * STD + MEAN) < 1e-9, 0.0, D["Y"]).astype(np.float32)
    D["aoa"] = np.array([int(s.split("_")[1]) for s in D["names"]])
    return D

ARMS = {
    "base(asymR1)": dict(),
    "+D1c30": dict(d1c_n0=30.0, d1c_after_fw=False),
    "+D1c10": dict(d1c_n0=10.0, d1c_after_fw=False),
    "+D1cEB": dict(d1c_n0=30.0, d1c_eb=True, d1c_after_fw=False),
    "+D1cEBonly": dict(d1c_n0=30.0, d1c_eb=True, eb_max=False, d1c_after_fw=False),
    "+FW8": dict(fw_n0=8.0),
    "+FW3": dict(fw_n0=3.0),
    "+FW20": dict(fw_n0=20.0),
    "+SIGEMA": dict(sigema=True),
    "+VC10": dict(vc_n0=10.0),
    "+VC30": dict(vc_n0=30.0),
    "+FW8+D1c30(afterFW)": dict(fw_n0=8.0, d1c_n0=30.0),
    "+FW8+D1c30(raw)": dict(fw_n0=8.0, d1c_n0=30.0, d1c_after_fw=False),
    "+FW8+D1cEB": dict(fw_n0=8.0, d1c_n0=30.0, d1c_eb=True),
    "+FW8+D1c30+SIGEMA": dict(fw_n0=8.0, d1c_n0=30.0, sigema=True),
    "+FW8+D1c30+VC30": dict(fw_n0=8.0, d1c_n0=30.0, vc_n0=30.0),
    "+FW8+D1cEB+VC30": dict(fw_n0=8.0, d1c_n0=30.0, d1c_eb=True, vc_n0=30.0),
    "+VC30dn": dict(vc_n0=30.0, vc_clip=(0.25, 1.0)),
    "+VC30up": dict(vc_n0=30.0, vc_clip=(1.0, 4.0)),
    "+VC10dn": dict(vc_n0=10.0, vc_clip=(0.25, 1.0)),
    "+VC100": dict(vc_n0=100.0),
    "+FW20+D1c30+VC30": dict(fw_n0=20.0, d1c_n0=30.0, vc_n0=30.0),
    "+D1c30+VC30": dict(d1c_n0=30.0, vc_n0=30.0),
    "+D1c30+VC10": dict(d1c_n0=30.0, vc_n0=10.0),
    "+D1c30+VC50": dict(d1c_n0=30.0, vc_n0=50.0),
    "+D1c30+VC100": dict(d1c_n0=30.0, vc_n0=100.0),
    "+FW20+D1c30+VC50": dict(fw_n0=20.0, d1c_n0=30.0, vc_n0=50.0),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/twins_unseen.npz")
    ap.add_argument("--band", default="/users/PAS1187/qwu/workspace/realpde/main/Track_1/bands/r35_centre0_fill.npz")
    ap.add_argument("--f", type=float, nargs="+", default=[1.0, 1.29]); ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--fam-f", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--fam-v", type=int, nargs="+", default=[3, 4])
    a = ap.parse_args()
    D = load(a.dump, a.fam_f, a.fam_v); band = np.load(a.band)
    out = {}
    for f in a.f:
        D["Pref"] = run(D, band, ARMS["base(asymR1)"], 1.0)[0]
        res = {}
        for nm, cfg in ARMS.items():
            if a.arms and nm not in a.arms and nm != "base(asymR1)":
                continue
            P, L, U, Yf = run(D, band, cfg, f)
            full = score(P, L, U, Yf); hi = D["aoa"] >= 15
            a15 = score(P[hi], L[hi], U[hi], Yf[hi])
            per = {t: score(P[D["names"] == t], L[D["names"] == t], U[D["names"] == t], Yf[D["names"] == t])["comp"] for t in dict.fromkeys(D["names"])}
            res[nm] = dict(full=full, a15=a15, per=per)
            b = res["base(asymR1)"]
            dp = [per[t] - b["per"][t] for t in per]; d15 = [per[t] - b["per"][t] for t in per if int(t.split("_")[1]) >= 15]
            print(f"f={f:.2f} {nm:24s} full comp {full['comp']:.4f} (d {full['comp']-b['full']['comp']:+.4f}; rel {full['rel']-b['full']['rel']:+.3f} tke {full['tke']-b['full']['tke']:+.3f} mvpe {full['mvpe']-b['full']['mvpe']:+.3f} sps {full['sps']-b['full']['sps']:+.3f}) "
                  f"pos {sum(x > 0 for x in dp)}/{len(dp)} | a15-20 d {a15['comp']-b['a15']['comp']:+.4f} (rel {a15['rel']-b['a15']['rel']:+.3f} tke {a15['tke']-b['a15']['tke']:+.3f} mvpe {a15['mvpe']-b['a15']['mvpe']:+.3f} sps {a15['sps']-b['a15']['sps']:+.3f}) pos {sum(x > 0 for x in d15)}/{len(d15)}", flush=True)
        out[str(f)] = res
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)
