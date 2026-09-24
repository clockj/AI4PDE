"""Decoupled interval CENTRE for the SPS band (Bram, 2026-09-21).

Point prediction stays p = y(ALPHA=0.60), so rel/tke/mvpe and a_i are untouched. The interval is
centred on c = y(alpha_c) and stretched to contain p (kit smoke test: lower <= pred <= upper):
    lower = min(c - h, p),  upper = max(c + h, p),  width = 2h + max(0, |p-c| - h)
h = base * clip(sig/sig_ref, .3, 3)**P, base = optimal_band(inflate*|t-c| / r**P) on the FIT split,
k=3 box smoothing -- the shipped fit_band recipe with only the centre changed. alpha_c = 0.60
reproduces the shipped band exactly (the paired control).

compose is linear in alpha: y(a) = (1-a)*y(0) + a*y(1).
Scoring is re-implemented here (no kit on this VM); a_i are shared by every arm, so only the
interval term differs between arms and the comparison is exact up to a common weighting.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

SIGMA = 0.0563870259
ALPHA = 0.60
PX = [13, 21, 29, 37]
PY = list(range(8, 25, 2))


# ---------------------------------------------------------------- scorer pieces (u,v only)
def _rel(a, b, m):
    return np.sqrt(((a - b) ** 2 * m).sum()) / max(np.sqrt((b ** 2 * m).sum()), 1e-12)


def sample_weights(P, Y):
    g = lambda e: 0.5 / (0.5 + e)  # noqa: E731
    out = np.empty(len(P))
    for i in range(len(P)):
        p, t = P[i, ..., :2].astype(np.float64), Y[i, ..., :2].astype(np.float64)
        dm = _rel(p, t, np.ones_like(t))   # official scorer keeps y==0 in rel/tke/mvpe (ROUNDS:1194)
        kp = 0.5 * (p[..., 0].var(0) + p[..., 1].var(0))
        kt = 0.5 * (t[..., 0].var(0) + t[..., 1].var(0))
        tk = _rel(kp, kt, np.ones_like(kt))
        pm, tm = p.mean(0), t.mean(0)
        mv = np.mean([_rel(pm[PY, x], tm[PY, x], np.ones_like(tm[PY, x])) for x in PX])
        out[i] = 0.5 * g(dm) + 0.3 * g(tk) + 0.2 * g(mv)
    return out


def optimal_band(A, a, scored, cap=0.08):
    n = len(A)
    flat, msk = A.reshape(n, -1), scored.reshape(n, -1)
    order = np.argsort(flat, axis=0, kind="stable")
    srt = np.take_along_axis(flat, order, axis=0)
    wts = np.take_along_axis(a[:, None] * msk, order, axis=0)
    gain = np.cumsum(wts, axis=0) * np.exp(-2.0 * srt.astype(np.float64) / SIGMA)
    h = np.take_along_axis(srt, gain.argmax(axis=0)[None], axis=0)[0]
    return np.minimum(h, cap).reshape((1,) + A.shape[1:]).astype(np.float32)


def smooth3(b, k=3):
    o = b.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * o.ndim
        pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(k)], axis=0)
    return o.astype(np.float32)


def sps_num(lo, hi, t, a):
    """Per-window sps numerator a_i * sum_e 1[in] exp(-width/sigma), and scored counts."""
    sc = t != 0
    inside = (t >= lo) & (t <= hi) & sc
    contrib = np.where(inside, np.exp(-(hi - lo) / SIGMA), 0.0)
    return a * contrib.reshape(len(t), -1).sum(1), sc.reshape(len(t), -1).sum(1).astype(np.float64)


def sps(p, lo, hi, t, a):
    """100 * sum_i a_i * sum_e 1[lo<=t<=hi] exp(-(hi-lo)/sigma) / n_scored, scored = t != 0."""
    sc = t != 0
    inside = (t >= lo) & (t <= hi) & sc
    contrib = np.where(inside, np.exp(-(hi - lo) / SIGMA), 0.0)
    per = contrib.reshape(len(t), -1).sum(1)
    return float(100.0 * (a @ per) / sc.sum())


# ---------------------------------------------------------------- the arm
def fit_base(c_fit, t_fit, r_fit, a_fit, inflate, P):
    A = (inflate * np.abs(t_fit - c_fit) / r_fit ** P).astype(np.float32)
    return smooth3(optimal_band(A, a_fit, t_fit != 0))


def bounds(p, c, base, r, P):
    h = base * r ** P
    return np.minimum(c - h, p), np.maximum(c + h, p)


def compose(y0, y1, al):
    return (1.0 - al) * y0 + al * y1


def run(fit, ev, alphas, inflate=1.39, P=0.45, stretches=(1.0, 1.15), extra=None):
    """fit/ev: dicts with y0,y1 (N,20,32,64,2) raw, t (target), sig (twin sigma, N,1,32,64,2),
    sig_ref (1,1,32,64,2) from the FULL members, traj (N,) trajectory labels."""
    rf = np.clip(fit["sig"] / fit["sig_ref"], 0.3, 3.0)
    re_ = np.clip(ev["sig"] / ev["sig_ref"], 0.3, 3.0)
    pf, pe = compose(fit["y0"], fit["y1"], ALPHA), compose(ev["y0"], ev["y1"], ALPHA)
    af, ae = sample_weights(pf, fit["t"]), sample_weights(pe, ev["t"])
    res = {}
    arms = [(al, compose(fit["y0"], fit["y1"], al), compose(ev["y0"], ev["y1"], al)) for al in alphas]
    arms += [(k, cf_, ce_) for k, (cf_, ce_) in (extra or {}).items()]
    for al, cf, ce in arms:
        base = fit_base(cf, fit["t"], rf, af, inflate, P)
        lo, hi = bounds(pe, ce, base, re_, P)
        row = {"base_med": float(np.median(base)),
               "ext_frac": float(np.mean(np.abs(pe - ce) > base * re_ ** P))}
        for s in stretches:          # board proxy: stretch the eval error about the point pred
            ts = pe + s * (ev["t"] - pe)
            ts = np.where(ev["t"] == 0, 0, ts)
            row[f"sps@{s}"] = sps(pe, lo, hi, ts, ae)
        row["per_traj"] = {str(k): sps(pe[ev["traj"] == k], lo[ev["traj"] == k], hi[ev["traj"] == k],
                                       ev["t"][ev["traj"] == k], ae[ev["traj"] == k])
                           for k in np.unique(ev["traj"])}
        ts = np.where(ev["t"] == 0, 0, pe + 1.15 * (ev["t"] - pe))
        row["_num"], row["_nsc"] = sps_num(lo, hi, ts, ae)
        res[al] = row
        print(f"centre {str(al):>6}  base_med {row['base_med']:.5f}  ext {row['ext_frac']:.3f}  "
              + "  ".join(f"sps@{s} {row[f'sps@{s}']:.3f}" for s in stretches), flush=True)
    return res


def decide(res, traj, hi_mask, arm=0.0, ctl=0.6, zmin=2.0):
    """The pre-registered rule (amended 09:03, #battleground-0921:ac7fc4f3 afed4b9f), @1.15 stretch."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # vendored beside this file
    from band_paired import paired_sps, loo_traj
    na, nb, nsc = res[arm]["_num"], res[ctl]["_num"], res[ctl]["_nsc"]
    d_hi = 100 * (na[hi_mask].sum() - nb[hi_mask].sum()) / nsc[hi_mask].sum()
    per = {str(t): 100 * (na[traj == t].sum() - nb[traj == t].sum()) / nsc[traj == t].sum()
           for t in np.unique(traj)}
    hi_tr = np.unique(traj[hi_mask])
    d, se, lo, hi = paired_sps(traj, na, nb, nsc)
    loo = {str(k): v for k, v in loo_traj(traj, na, nb, nsc).items()}
    c1 = d_hi >= 0.30 and all(per[str(t)] > 0 for t in hi_tr)
    c2 = se > 0 and d / se >= zmin
    c3 = sum(v > 0 for v in per.values()) >= 6 and all(v > 0 for v in loo.values())
    out = {"d_hi15-20": d_hi, "d_all8": d, "se": se, "ci95": [lo, hi], "z": d / se if se else None,
           "per_traj": per, "loo": loo, "mde80": 2.8 * se, "C1": bool(c1), "C2": bool(c2), "C3": bool(c3), "PASS": bool(c1 and c2 and c3)}
    print(json.dumps(out, indent=1, default=float))
    return out


def synthetic(n=60, seed=0):
    """Truth fluctuation f_t; prediction knows it with correlation rho; y1 inflated, y0 shrunk."""
    g = np.random.default_rng(seed)
    sh = (n, 20, 32, 64, 2)
    m = 0.2 + 0.05 * g.standard_normal((n, 1, 32, 64, 2))
    ft = 0.02 * g.standard_normal(sh)
    known = 0.6 * ft + 0.8 * 0.02 * g.standard_normal(sh)      # corr ~0.6
    y0 = m + 0.6 * known + 0.003 * g.standard_normal(sh)
    y1 = m + 1.0 * known
    t = m + ft + 0.004 * g.standard_normal((n, 1, 32, 64, 2))
    t[:, :, :2] = 0.0                                          # a blanked strip
    sig = np.full((n, 1, 32, 64, 2), 0.02, np.float32) * (1 + 0.2 * g.random((n, 1, 1, 1, 1)))
    return {"y0": y0.astype(np.float32), "y1": y1.astype(np.float32), "t": t.astype(np.float32),
            "sig": sig.astype(np.float32), "sig_ref": sig.mean(0, keepdims=True),
            "traj": np.repeat(np.arange(n // 20), 20)}


# ---------------------------------------------------------------- real run on Emre's export
def kit_check(kit_path, p, lo, hi, t, a_mine):
    """Harness check: my sps vs the kit's own aggregate_sps on the same bounds. Must agree < 1e-3."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("kit_sc", kit_path)
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    P = np.concatenate([p, np.zeros_like(p[..., :1])], -1)
    Y = np.concatenate([t, np.zeros_like(t[..., :1])], -1)
    L = np.concatenate([lo, np.zeros_like(lo[..., :1])], -1)
    U = np.concatenate([hi, np.zeros_like(hi[..., :1])], -1)
    ref = sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=L, upper=U)[0])
    mine = sps(p, lo, hi, t, a_mine)
    print(f"SCORER CHECK (vs RECONSTRUCTION; official agrees to 2.4e-6, ROUNDS:1194): mine {mine:.5f}  kit {ref:.5f}  diff {mine - ref:+.2e}  "
          f"{'OK' if abs(mine - ref) < 1e-3 else 'MISMATCH -- do not read the arm'}")
    return mine, ref


def from_export(seen_path, unseen_path, alphas, band_check=None, out_json=None, kit=None, sigma="twin",
                members=None):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # vendored beside this file
    from load_export import load
    S, U = load(seen_path), load(unseen_path)
    traj = lambda r: np.array([str(i).split(":")[0] for i in r["ids"]])        # noqa: E731
    aoa = lambda r: np.array([int(t.split("_")[1]) for t in traj(r)])          # noqa: E731
    keep = np.isin(aoa(S), [15, 20])                                           # fit_band default
    sk = "sig_twin" if sigma == "twin" else "sig_full"     # primary: twin (OOS on unseen)
    sig_ref = np.maximum(S[sk][keep].mean(0, keepdims=True), 1e-8)
    fit = {"y0": (S["y"] + S["e_a0"])[keep], "y1": (S["y"] + S["e_a1"])[keep], "t": S["y"][keep],
           "sig": S[sk][keep], "sig_ref": sig_ref, "traj": traj(S)[keep]}
    ev = {"y0": U["y"] + U["e_a0"], "y1": U["y"] + U["e_a1"], "t": U["y"],
          "sig": U[sk], "sig_ref": sig_ref, "traj": traj(U)}
    print(f"sigma = {sk}")
    print(f"fit {keep.sum()} windows ({sorted(set(fit['traj']))}); eval {len(ev['t'])} windows")
    extra = None
    if members:        # secondary arm "med": centre = y + per-element median of member raw errors
        from load_export import load_members
        Ms, Mu = load_members(members[0]), load_members(members[1])
        assert list(Ms["ids"][keep]) == list(S["ids"][keep]) and list(Mu["ids"]) == list(U["ids"])
        extra = {"med": (S["y"][keep] + np.median(Ms["e_mem"][:, keep], axis=0),
                         U["y"] + np.median(Mu["e_mem"], axis=0))}
        del Ms, Mu
    res = run(fit, ev, alphas, extra=extra)
    if band_check:     # GATE: shipped recipe (sig_full) at alpha 0.60 must reproduce r34 base
        ship = np.load(band_check)["base"][..., :2]
        sref_f = np.maximum(S["sig_full"][keep].mean(0, keepdims=True), 1e-8)
        rf = np.clip(S["sig_full"][keep] / sref_f, 0.3, 3.0)
        pf = compose(fit["y0"], fit["y1"], ALPHA)
        mine = fit_base(pf, fit["t"], rf, sample_weights(pf, fit["t"]), 1.39, 0.45)
        rel = np.abs(mine - ship) / np.maximum(ship, 1e-6)
        print(f"band check vs {os.path.basename(band_check)}: median rel diff {np.median(rel):.4f}, "
              f"p90 {np.percentile(rel, 90):.4f}, frac>1% {np.mean(rel > 0.01):.3f} "
              f"(fill-unscored touches ~4.5 %)  GATE {'OK' if np.median(rel) < 0.01 else 'FAIL'}")
    if kit:
        rf = np.clip(fit["sig"] / sig_ref, 0.3, 3.0); re_ = np.clip(ev["sig"] / sig_ref, 0.3, 3.0)
        pf, pe = compose(fit["y0"], fit["y1"], ALPHA), compose(ev["y0"], ev["y1"], ALPHA)
        base = fit_base(pf, fit["t"], rf, sample_weights(pf, fit["t"]), 1.39, 0.45)
        lo, hi = bounds(pe, pe, base, re_, 0.45)
        kit_check(kit, pe, lo, hi, ev["t"], sample_weights(pe, ev["t"]))
    dec = decide(res, ev["traj"], np.isin(aoa(U), [15, 20]))
    # DESCRIPTIVE ONLY: arm-vs-control delta split by pixels whose own INPUT window is part-zero
    xz = U["x"][..., :2] == 0
    inter = (xz.any(1) & ~xz.all(1))[:, None]                       # (N,1,32,64,2)
    pe = compose(ev["y0"], ev["y1"], ALPHA); re_ = np.clip(ev["sig"] / sig_ref, 0.3, 3.0)
    ae = sample_weights(pe, ev["t"]); ts = np.where(ev["t"] == 0, 0, pe + 1.15 * (ev["t"] - pe))
    for name in [0.0] + (["med"] if extra else []):
        cA = compose(ev["y0"], ev["y1"], 0.0) if name == 0.0 else extra["med"][1]
        cA_f = compose(fit["y0"], fit["y1"], 0.0) if name == 0.0 else extra["med"][0]
        rf = np.clip(fit["sig"] / sig_ref, 0.3, 3.0); pf = compose(fit["y0"], fit["y1"], ALPHA)
        af = sample_weights(pf, fit["t"])
        bA, bC = fit_base(cA_f, fit["t"], rf, af, 1.39, 0.45), fit_base(pf, fit["t"], rf, af, 1.39, 0.45)
        (lA, hA), (lC, hC) = bounds(pe, cA, bA, re_, 0.45), bounds(pe, pe, bC, re_, 0.45)
        n_sc = (ts != 0).sum()
        for lab, m in (("intermittent", inter), ("other", ~inter)):
            mm = np.broadcast_to(m, ts.shape)
            dA = sps(pe, np.where(mm, lA, 0), np.where(mm, hA, -1), ts, ae)   # -1 upper => never inside
            dC = sps(pe, np.where(mm, lC, 0), np.where(mm, hC, -1), ts, ae)
            print(f"[descriptive] centre {name}: {lab:12s} share of scored {(mm & (ts != 0)).sum() / n_sc:.3f}"
                  f"  dsps@1.15 {dA - dC:+.3f}")
    if extra:
        print("SECONDARY arm 'med' -- read only if the primary FAILS (prereg):")
        dec_med = decide(res, ev["traj"], np.isin(aoa(U), [15, 20]), arm="med", zmin=2.3)
        dec = {"primary": dec, "med": dec_med}
    if out_json:
        json.dump({"arms": {str(k): {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                            for k, v in res.items()}, "decision": dec},
                  open(out_json, "w"), indent=1, default=float)
    return res, dec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--json", default=None)
    ap.add_argument("--seen"); ap.add_argument("--unseen")
    ap.add_argument("--band-check", default=None, help="Track_1/bands/r34_p2ffnoL8sim_fill.npz")
    ap.add_argument("--kit", default=None, help="kit scoring.py copied into Emre's /out")
    ap.add_argument("--sigma", choices=["twin", "full"], default="twin")
    ap.add_argument("--members", nargs=2, default=None, help="members_seen.npz members_unseen.npz")
    args = ap.parse_args()
    if args.seen:
        from_export(args.seen, args.unseen, args.alphas, args.band_check, args.json, args.kit, args.sigma, args.members)
    if args.synthetic:
        fit, ev = synthetic(60, 0), synthetic(60, 1)
        res = run(fit, ev, args.alphas)
        assert res[0.6]["ext_frac"] == 0.0, "alpha_c = ALPHA must never extend the interval"
        decide(res, ev["traj"], ev["traj"] >= 1)
        if args.json:
            json.dump({str(k): {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}, open(args.json, "w"), indent=1)


