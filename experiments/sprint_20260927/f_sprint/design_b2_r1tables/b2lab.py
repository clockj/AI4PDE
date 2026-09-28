"""Design B2 band lab: causal band-side variants replayed on a cached prediction-side stream (b2prep.py). CPU, physical units.

All variants share R1's statistic: for a grid ks of multipliers of the STATIC half-width h, the running K x K table of the
sps contribution the interval [min(C - ka*h, P), max(C + kb*h, P)] would have earned on the revealed windows of THIS
trajectory (submission.py:_CausalBandScalerT.update_gated_asym, verbatim).  They differ in how the table is pooled and shrunk:
  pixel  table pooled over the 20 frames, per (H,W,C)   -> k_pix   (shipped R1: grid 0.5..2, box3, n/(n+N0), prior 1)
  glob   the same table summed over pixels               -> k_g     (prior of k_pix instead of 1; learnt from window 1)
  frame  table pooled over pixels, per lead time t       -> k_t/k_g (multiplies k_pix)
  elem   table per element (T,H,W,C)                     -> k_e     (k_pix shrunk toward it with n/(n+N0e))
    python b2lab.py <cache tag> [variant-set]     -> $S/ckpt/b2/res_<tag>.json
"""
import os, sys, json, time
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("B2_THREADS", "2")))
SIG = 0.0563870259
S = "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927"


def box(a, k, dims):
    o = a
    if k <= 1:
        return o
    r = k // 2
    for ax in dims:
        q = o.movedim(ax, 0); L = q.shape[0]
        q = torch.cat([q[:1]] * r + [q] + [q[-1:]] * r, 0)
        o = (sum(q[j:j + L] for j in range(k)) / float(k)).movedim(0, ax)
    return o


def V(**kw):
    d = dict(n0=2.0, sm=3, prior="one", n0g=1.0, n0t=0.0, n0e=0.0, sym=False, tsm=1, te=1, esm=3, oracle=False, blk=1)
    d.update(kw); return d


GRIDS = {"g9": (-4, 4), "g9e": (-4, 4), "g13": (-6, 6), "g13u": (-4, 8), "g17": (-8, 8)}       # ks = 2^(j/4): 0.5..2 | 0.354..2.83 | 0.5..4 | 0.25..4
SETS = {
    "wave1": {
        "g9": {"R1ship": V(), "n0_0.5": V(n0=0.5), "n0_1": V(n0=1.0), "n0_4": V(n0=4.0), "n0_8": V(n0=8.0),
               "sm1": V(sm=1), "sm5": V(sm=5),
               "glob_g0.25": V(prior="glob", n0g=0.25), "glob_g1": V(prior="glob", n0g=1.0), "glob_g1_n4": V(prior="glob", n0g=1.0, n0=4.0),
               "glob_g0.25_n4": V(prior="glob", n0g=0.25, n0=4.0), "globonly": V(prior="glob", n0g=0.25, n0=1e9),
               "frame_t1": V(n0t=1.0), "frame_t4": V(n0t=4.0), "glob_frame": V(prior="glob", n0g=0.25, n0t=2.0),
               "elem_e10": V(n0e=10.0), "elem_e30": V(n0e=30.0)},
        "g13": {"g13": V(), "g13_glob_g0.25": V(prior="glob", n0g=0.25)},
        "g13u": {"g13u": V(), "g13u_n1": V(n0=1.0), "g13u_glob_g0.25": V(prior="glob", n0g=0.25), "g13u_glob_g1_n4": V(prior="glob", n0g=1.0, n0=4.0)},
        "g17": {"g17": V(), "g17_glob_g0.25": V(prior="glob", n0g=0.25)},
    },
    "wave3": {       # combinations of the three small positives of waves 1-2 (table smoothing, global prior, element residual)
        "g9": {"R1ship": V(), "tsm3": V(tsm=3), "tsm3_glob": V(tsm=3, prior="glob", n0g=0.25),
               "tsm3_e30": V(tsm=3, n0e=30.0), "tsm3_glob_e30": V(tsm=3, prior="glob", n0g=0.25, n0e=30.0), "tsm3_glob_e60": V(tsm=3, prior="glob", n0g=0.25, n0e=60.0)},
        "g13u": {"u_tsm3_glob": V(tsm=3, prior="glob", n0g=0.25), "u_tsm3_glob_e30": V(tsm=3, prior="glob", n0g=0.25, n0e=30.0)},
    },
    "wave4": {       # the element residual at LEAD-TIME BLOCK resolution (blk frames per block): the shippable, cheap form
        "g9": {"R1ship": V(), "tsm3_e30": V(tsm=3, n0e=30.0), "tsm3_blk4_e10": V(tsm=3, n0e=10.0, blk=4), "tsm3_blk4_e20": V(tsm=3, n0e=20.0, blk=4),
               "tsm3_blk4_e30": V(tsm=3, n0e=30.0, blk=4), "tsm3_blk5_e20": V(tsm=3, n0e=20.0, blk=5), "tsm3_blk2_e30": V(tsm=3, n0e=30.0, blk=2),
               "blk4_e20": V(n0e=20.0, blk=4), "tsm3_blk10_e10": V(tsm=3, n0e=10.0, blk=10)},
    },
    "wave2": {
        "g9": {"R1ship": V(), "tsm3": V(tsm=3), "tsm3_sm1": V(tsm=3, sm=1), "tsm5_sm1": V(tsm=5, sm=1), "tsm3_sm1_n1": V(tsm=3, sm=1, n0=1.0),
               "ORACLE_pix": V(oracle=True), "ORACLE_pix_tsm3": V(oracle=True, tsm=3, sm=1)},
        "g9e": {"R1ship_": V(), "elem_e30": V(n0e=30.0), "elem_e100": V(n0e=100.0), "elem_te5_e10": V(n0e=10.0, te=5), "elem_te5_e30": V(n0e=30.0, te=5),
                "elem_te5_e3": V(n0e=3.0, te=5), "elem_te9_e10": V(n0e=10.0, te=9), "ORACLE_elem_te5": V(n0e=1e-9, te=5, oracle=True)},
    },
}


def kfields(st, v, ks, K):
    n = st["n"]
    one = torch.ones((1, 32, 64, 2), dtype=torch.float64)
    if v["oracle"]:                       # NOT causal: the whole trajectory's table, no shrinkage -- a ceiling, never shippable
        st = dict(st, n=1e12, cum2=st["full2"], cumE=st.get("fullE"))
        n = st["n"]
    if n == 0:
        return one, one
    flat = box(st["cum2"], v["tsm"], (2, 3)).flatten(0, 1).argmax(0)
    kl = box(ks[torch.div(flat, K, rounding_mode="floor")], v["sm"], (0, 1)); kh = box(ks[flat % K], v["sm"], (0, 1))
    pl = ph = 1.0
    g = int(st["cum2"].sum((2, 3, 4), dtype=torch.float64).flatten().argmax()); gl, gh = float(ks[g // K]), float(ks[g % K])
    if v["prior"] == "glob":
        sg = n / (n + v["n0g"]); pl = 1.0 + (gl - 1.0) * sg; ph = 1.0 + (gh - 1.0) * sg
    sh = n / (n + v["n0"])
    klo = (pl + (kl - pl) * sh)[None]; khi = (ph + (kh - ph) * sh)[None]
    if v["n0t"] > 0:
        ft = st["cumT"].flatten(0, 1).argmax(0)
        gt = int(st["cumT"].sum(2, dtype=torch.float64).flatten().argmax())
        rl = box(ks[torch.div(ft, K, rounding_mode="floor")] / float(ks[gt // K]), 3, (0,)); rh = box(ks[ft % K] / float(ks[gt % K]), 3, (0,))
        s = n / (n + v["n0t"])
        klo = klo * (1.0 + (rl - 1.0) * s)[:, None, None, None]; khi = khi * (1.0 + (rh - 1.0) * s)[:, None, None, None]
    if v["n0e"] > 0:
        tabE = st["cumE"]
        if v["blk"] > 1:
            tabE = tabE.reshape(K, K, 20 // v["blk"], v["blk"], 32, 64, 2).sum(3)
        fe = box(box(tabE, v["te"], (2,)), v["esm"], (3, 4)).flatten(0, 1).argmax(0)
        el = ks[torch.div(fe, K, rounding_mode="floor")]; eh = ks[fe % K]
        if v["blk"] > 1:
            el = el.repeat_interleave(v["blk"], 0); eh = eh.repeat_interleave(v["blk"], 0)
        if v["te"] <= 1:
            el = box(el, 3, (1, 2)); eh = box(eh, 3, (1, 2))
        s = n / (n + v["n0e"])
        klo = klo + (el - klo) * s; khi = khi + (eh - khi) * s
    lo_, hi_ = float(ks[0]), float(ks[-1])
    return klo.clamp(lo_, hi_), khi.clamp(lo_, hi_)


def tables(p, c, h, y, ks32):
    """update_gated_asym's two factors, (K,T,H,W,C) each, for multipliers ks of the static half-width."""
    scored = y != 0
    hk = h[None] * ks32[:, None, None, None, None]
    lo = torch.minimum(c - hk, p); hi = torch.maximum(c + hk, p)
    below = y < c
    A = torch.exp(-((c - lo) / SIG)) * torch.where(below, y >= lo, torch.tensor(True)) * scored
    B = torch.exp(-((hi - c) / SIG)) * torch.where(below, torch.tensor(True), y <= hi)
    return A, B


def stream(Z, gname, variants):
    j0, j1 = GRIDS[gname]; ks = torch.tensor([2.0 ** (j / 4.0) for j in range(j0, j1 + 1)], dtype=torch.float64); K = len(ks); ks32 = ks.float()
    P, C, H, Y, names = (torch.from_numpy(Z[k]) for k in ("P", "C", "h", "Y")), None, None, None, Z["names"]
    P, C, H, Y = list(P)
    N = len(names); need_t = any(v["n0t"] > 0 for v in variants.values()); need_e = any(v["n0e"] > 0 for v in variants.values())
    con = {nm: np.zeros(N) for nm in variants}; nsc = np.zeros(N); idx = np.zeros(N, int); wid = {nm: np.zeros(N) for nm in variants}
    st = None
    full = {}
    if any(v["oracle"] for v in variants.values()):
        for i in range(N):
            nm_ = str(names[i])
            if nm_ not in full:
                full[nm_] = [torch.zeros((K, K, 32, 64, 2)), torch.zeros((K, K, 20, 32, 64, 2)) if need_e else None]
            A, B = tables(P[i], C[i], H[i], Y[i], ks32)
            full[nm_][0] += torch.einsum("athwc,bthwc->abhwc", A, B)
            if need_e:
                full[nm_][1] += A[:, None] * B[None]
    for i in range(N):
        if i == 0 or names[i] != names[i - 1]:
            st = dict(n=0, cum2=torch.zeros((K, K, 32, 64, 2)), cumT=torch.zeros((K, K, 20)) if need_t else None,
                      cumE=torch.zeros((K, K, 20, 32, 64, 2)) if need_e else None)
            if full:
                st["full2"], st["fullE"] = full[str(names[i])]
        idx[i] = st["n"]
        p, c, h, y = P[i], C[i], H[i], Y[i]
        scored = y != 0; nsc[i] = float(scored.sum())
        for nm, v in variants.items():
            klo, khi = kfields(st, v, ks, K)
            lo = torch.minimum(c - h * klo.float(), p); hi = torch.maximum(c + h * khi.float(), p)
            assert bool(torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo <= hi).all())
            w = torch.exp(-((hi - lo).double()) / SIG)
            ins = (y >= lo) & (y <= hi) & scored
            con[nm][i] = float((w * ins).sum()); wid[nm][i] = float(((hi - lo) * scored).sum())
        # reveal window i's target: update the tables (static-h multipliers, exactly update_gated_asym)
        A, B = tables(p, c, h, y, ks32)
        st["cum2"] += torch.einsum("athwc,bthwc->abhwc", A, B)
        if need_t:
            st["cumT"] += torch.einsum("athwc,bthwc->abt", A, B)
        if need_e:
            st["cumE"] += A[:, None] * B[None]
        st["n"] += 1
    return con, nsc, idx, wid


def main():
    tag = sys.argv[1]; vs = SETS[sys.argv[2] if len(sys.argv) > 2 else "wave1"]
    Z = np.load(f"{S}/ckpt/b2/{tag}.npz"); a = Z["a"]; names = Z["names"]
    out = dict(tag=tag, names=[str(x) for x in names], a=a.tolist(), ref_sps=float(Z["ref_sps"]), ref_comp=float(Z["ref_comp"]), var={})
    for g, variants in vs.items():
        t0 = time.time()
        con, nsc, idx, wid = stream(Z, g, variants)
        out["nsc"] = nsc.tolist(); out["idx"] = idx.tolist()
        for nm in variants:
            out["var"][nm] = dict(con=con[nm].tolist(), wid=wid[nm].tolist(), cfg=variants[nm], grid=g)
            print(f"{tag} {nm:18s} sps {100.0 * float(a @ con[nm]) / nsc.sum():.4f}", flush=True)
        print(f"{tag} grid {g} done {time.time() - t0:.0f}s   (ref sps from probe.run {float(Z['ref_sps']):.4f})", flush=True)
    sfx = "" if len(sys.argv) <= 2 else "_" + sys.argv[2]
    json.dump(out, open(f"{S}/ckpt/b2/res_{tag}{sfx}.json", "w"))


if __name__ == "__main__":
    main()
