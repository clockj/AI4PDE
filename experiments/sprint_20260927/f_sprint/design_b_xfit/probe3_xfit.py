"""Design B, cross-fitted band base (XF).  Two stages, both runnable on CPU (GPU used if present).

  python probe3_xfit.py dump   # OOF fields of the fold models xf_{ffno,vit}_f{k}.pth on THEIR held-out fold -> oof.npz
  python probe3_xfit.py eval   # LOTO on val_unseen (twin fields): blended base vs the shipped construction (V0)
  python probe3_xfit.py ship --w W --band-in <shipped band.npz> --out <new band.npz>   # writes the band to ship

Construction:  base_xf(e) = exp((1-w) * log base_tw(e) + w * log(k * base_oof(e)))
  base_tw  = the shipped per-element base (twins' held-out |y0 - y| on val_unseen AoA 15-20, x1.39, / sigma-ratio^0.45, k3 box)
  base_oof = the same solver on the OUT-OF-FOLD errors |y0_oof - y| of the fold models over the AoA 15-20 trajectories of the
             74 non-validation recordings (each scored by a model pair that never saw it), deflated by the fold models' own
             clip(sigma/sigma_ref_oof, .3, 3)^0.45, x1.39, k3 box
  k        = ONE scalar matching the OOF error level to the twins': argmax of the twins' fit-split sps of k*base_oof*R
  w        = blend weight, chosen by LOTO below (kill if no w > 0 wins).
Inference is unchanged: half = base * clip(sigma/sigma_ref, .3, 3)^0.45, centre y0.  Only band.npz['base'] changes.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); T1 = os.path.dirname(HERE)
sys.path.insert(0, T1); sys.path.insert(0, os.path.join(T1, "submit_fno"))
import sps9
from sps9 import SIGMA, sps_score
from bandlab import Lab, smooth, sigma_ratio, FS, INFL, P_EXP
CK = os.environ.get("B_CKPT", "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/b")
OOF = os.path.join(CK, "oof.npz")
DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
FIT_AOA = (15, 20)


def optimal_band_fast(A, a, scored, cap=0.08):
    """sps9.optimal_band with the sample axis made contiguous (same math; ~20x faster on > 300 windows)."""
    n = len(A)
    flat = np.ascontiguousarray(A.reshape(n, -1).T)
    w = np.ascontiguousarray((a[:, None] * scored.reshape(n, -1)).T)
    order = np.argsort(flat, axis=1, kind="stable")
    srt = np.take_along_axis(flat, order, 1); wts = np.take_along_axis(w, order, 1)
    gain = np.cumsum(wts, 1) * np.exp(-2.0 * srt.astype(np.float64) / SIGMA)
    h = srt[np.arange(len(srt)), gain.argmax(1)]
    return np.minimum(h, cap).reshape((1,) + A.shape[1:]).astype(np.float32)


def dump():
    import h5py, torch, _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fj = json.load(open(os.path.join(HERE, "folds.json"))); fd = fj["folds"]; fd_val = fj["val"]
    E, Sg, Y, ids, am = [], [], [], [], []
    for k, names in fd.items():
        cks = [p for p in (os.path.join(CK, f"xf_{a}_f{k}.pth") for a in ("ffno", "vit")) if os.path.exists(p)]
        if not cks:
            print("fold", k, "has no checkpoint -- skipped"); continue
        for c in cks:                                           # EB: provenance guard -- OOF only if the model never saw it
            cfg = torch.load(c, map_location="cpu")["config"]
            ex = set(cfg.get("exclude") or [])
            assert not cfg.get("all_data") and set(names) <= ex and set(fd_val) <= ex, (c, sorted(set(names) - ex))
            print(f"  {os.path.basename(c)}: epochs {cfg.get('epochs')}, excludes fold {k} ({len(names)}) + val ({len(fd_val)}): OK", flush=True)
        ms = [_fno.load_any(c, dev) for c in cks]
        xs, ys, ii = [], [], []
        for name in names:
            if int(name.split("_")[1]) not in FIT_AOA:
                continue
            with h5py.File(os.path.join(DATA, f"{name}.h5"), "r") as f:
                d = np.stack([f["u"][:, ::2, ::2], f["v"][:, ::2, ::2]], -1).astype(np.float32)
            for t in range(0, len(d) - 40 + 1, 20):
                xs.append(d[t:t + 20]); ys.append(d[t + 20:t + 40]); ii.append(f"{name}:{t}")
        X = np.concatenate([np.asarray(xs), np.zeros((len(xs), 20, 32, 64, 1), np.float32)], -1)
        y = np.asarray(ys)
        f0 = np.zeros_like(y); s0 = np.zeros((len(y), 1, 32, 64, 2), np.float32)
        for m in ms:
            s = np.empty_like(s0)
            f0 += _fno.predict_fno([m], X, dev, alpha=0.0, sigma_out=s, batch=16, amp=None)[..., :2] / len(ms)
            s0 += s / len(ms)
        P = f0                                                  # a_i of the OOF field itself (alpha 0)
        P3 = np.concatenate([P, np.zeros_like(P[..., :1])], -1); Y3 = np.concatenate([y, np.zeros_like(y[..., :1])], -1)
        am.append(sps9.sample_weights(P3, Y3))
        E.append(np.abs(f0 - y).astype(np.float32)); Sg.append(s0); Y.append(y != 0.0); ids += ii
        print("fold", k, [os.path.basename(c) for c in cks], len(y), "windows  rel", float(sps9.sample_metrics(P3, Y3)[0].mean()), flush=True)
    np.savez(OOF, A=np.concatenate(E), sig=np.concatenate(Sg), scored=np.concatenate(Y), a=np.concatenate(am), ids=np.array(ids))
    print("wrote", OOF)


def base_oof():
    d = np.load(OOF)
    sig = d["sig"]; ref = np.maximum(sig.mean(0, keepdims=True), 1e-8)
    R = np.clip(sig / ref, 0.3, 3.0) ** P_EXP
    A = (d["A"] * np.float32(INFL) / R).astype(np.float32)
    print("OOF fit windows", len(A), "trajectories", len({str(i).split(':')[0] for i in d["ids"]}))
    return smooth(optimal_band_fast(A, d["a"], d["scored"]))


def evaluate():
    L = Lab(); bo = base_oof(); a1 = L.a(1.0)
    ks = np.exp(np.linspace(np.log(0.4), np.log(1.6), 25))
    ref = L.run(L.y0); L.table("V0 shipped LOTO", ref)
    cache = {}                                   # EB fix: bt and k do not depend on w -> fit once per trajectory
    for t, fit in L.fits().items():
        R = np.broadcast_to(sigma_ratio(L.sig, fit), L.y0.shape)
        Af = np.abs(L.y0[fit] - L.y[fit]) * np.float32(INFL)
        bt = smooth(sps9.optimal_band((Af / R[fit]).astype(np.float32), a1[fit], L.scored[fit]))
        nf = int(L.scored[fit].sum())
        ki = int(np.argmax([sps_score(Af, a1[fit], L.scored[fit], nf, (kk * bo * R[fit]).astype(np.float32)) for kk in ks]))
        cache[t] = (R, bt, ks[ki])
        print(f"  eval {t}: k {ks[ki]:.3f}" + ("  ** K ON GRID EDGE -> KILL (design_b §7) **" if ki in (0, len(ks) - 1) else ""), flush=True)
    for w in (0.25, 0.5, 0.75, 1.0):
        res = {f: {} for f in FS}
        for t, fit in L.fits().items():
            R, bt, k = cache[t]
            base = np.exp((1 - w) * np.log(np.maximum(bt, 1e-9)) + w * np.log(np.maximum(k * bo, 1e-9))).astype(np.float32)
            ev = L.traj == t
            h = base * R[ev]
            lo = np.minimum(L.y0[ev] - h, L.P[ev]); up = np.maximum(L.y0[ev] + h, L.P[ev])
            sc_ = L.scored[ev]; n = int(sc_.sum())
            for f in FS:
                yf = L.y0[ev] + np.float32(f) * (L.y[ev] - L.y0[ev])
                con = np.where((yf >= lo) & (yf <= up) & sc_, np.exp(-(up - lo).astype(np.float64) / SIGMA), 0.0)
                af = L.a(f)[ev]
                res[f][t] = (100.0 * float(af @ con.reshape(len(con), -1).sum(1)) / n, float(af.mean()), 100.0 * float(con.sum()) / n)
        L.table(f"XF blend w={w}", res, ref)


def ship():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--band-in", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = Lab(); bo = base_oof(); b = dict(np.load(a.band_in)); bt = b["base"][..., :2]
    fit = np.isin(L.traj, L.hi); a1 = L.a(1.0)
    R = np.broadcast_to(sigma_ratio(L.sig, fit), L.y0.shape)[fit]
    Af = np.abs(L.y0[fit] - L.y[fit]) * np.float32(INFL); nf = int(L.scored[fit].sum())
    ks = np.exp(np.linspace(np.log(0.4), np.log(1.6), 25))
    ki = int(np.argmax([sps_score(Af, a1[fit], L.scored[fit], nf, (kk * bo * R).astype(np.float32)) for kk in ks]))
    k = ks[ki]
    if ki in (0, len(ks) - 1):
        print(f"** WARNING: k {k:.3f} on the grid edge -> level match failed (design_b §7 kill) **")
    new = np.exp((1 - a.w) * np.log(np.maximum(bt, 1e-9)) + a.w * np.log(np.maximum(k * bo, 1e-9))).astype(np.float32)
    # EB fix: an element no OOF window scores (target exactly 0 everywhere) has no OOF information -> keep the shipped
    # (fill-unscored) base there instead of blending with the solver's arbitrary value.
    unsc = ~np.load(OOF)["scored"].any(0, keepdims=True)
    print("elements without any scored OOF sample (kept = shipped base):", int(unsc.sum()), "of", unsc.size)
    new = np.where(unsc, bt, new).astype(np.float32)
    assert new.shape == bt.shape and np.isfinite(new).all() and (new > 0).all()
    b0 = dict(np.load(a.band_in))
    b["base"] = np.concatenate([new, b0["base"][..., 2:]], -1).astype(b0["base"].dtype)
    np.savez(a.out, **b)
    chk = np.load(a.out)                                        # EB: same keys, shapes, dtypes as the input band
    assert sorted(chk.files) == sorted(b0.keys()), (chk.files, list(b0))
    for kk in b0:
        assert chk[kk].shape == b0[kk].shape and chk[kk].dtype == b0[kk].dtype, kk
        if kk != "base":
            assert np.array_equal(chk[kk], b0[kk]), kk
    r = new / np.maximum(bt, 1e-12)
    print(f"ratio new/shipped base: p5 {np.percentile(r, 5):.3f} median {np.median(r):.3f} p95 {np.percentile(r, 95):.3f}")
    print(f"k {k:.4f} w {a.w}; base median {np.median(bt):.5f} -> {np.median(new):.5f}; wrote {a.out}")


if __name__ == "__main__":
    {"dump": dump, "eval": evaluate, "ship": ship}[sys.argv[1]]()
