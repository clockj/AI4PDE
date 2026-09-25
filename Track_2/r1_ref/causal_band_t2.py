"""R1 for Track 2 — Emre's interface (DM 00:51Z 09-25). Tensors (1,T,32,64,2) on device in TARGET-normalised
space: physical = v*std_tgt[c] + mean_tgt[c]; half physical = h*std_tgt[c]. Width term uses physical width:
exp(-(hi-lo)*std_tgt[c]/sigma). Coverage test is affine-invariant per channel. State on device, no host sync,
no Python branch on data -> CUDA-graph capturable. cum float64, argmax/argmin first-occurrence, box3 =
replicate-padded 3x3 mean in the numpy reference's op order, snap to grid. n is a device tensor.
Weighting: UNWEIGHTED cumulative (no a_i). SHIPPING (Jonas 01:41Z): continuous k (snap=False), K=9 log grid on [0.5, 2.0], n0 = 2, smooth on.
"""
import torch
KS16 = [round(0.5 + 0.1 * i, 3) for i in range(16)]        # legacy linear grid
KS = [0.5 * 2 ** (j / 4) for j in range(9)]                 # SHIPPING: K=9 log grid, 0.5..2.0, contains 1.0


def _box3(a):                                # (H,W,C) float64, same op order as causal_band.box3
    o = a
    for ax in (0, 1):
        q = o.movedim(ax, 0); L = q.shape[0]
        q = torch.cat([q[:1], q, q[-1:]], 0)
        o = (((q[0:L] + q[1:L + 1]) + q[2:L + 2]) / 3.0).movedim(0, ax)
    return o


class CausalBandScalerT:
    def __init__(self, ks=KS, n0=2.0, std_tgt=(1.0, 1.0), sigma=0.0563870259, device="cpu",
                 shape=(32, 64, 2), smooth=True, snap=False, acc_dtype=torch.float64):
        """acc_dtype: dtype of cum and of the width term. float64 = bitwise vs numpy reference;
        float32 = free latency lever (gains identical to 3 decimals on 4 stretches; verify with self_test)."""
        self.device, self.shape, self.smooth, self.sigma, self.snap = device, tuple(shape), smooth, float(sigma), snap
        self.acc_dtype = acc_dtype
        self.ks = torch.tensor(list(ks), dtype=torch.float64, device=device)
        self.ks32 = self.ks.to(torch.float32)
        self.n0 = torch.tensor(float(n0), dtype=torch.float64, device=device)
        self.std_tgt = torch.tensor(list(std_tgt), dtype=acc_dtype, device=device)   # (2,)
        self.cum = torch.zeros((len(ks),) + self.shape, dtype=acc_dtype, device=device)
        self.n = torch.zeros((), dtype=torch.float64, device=device)
        self.one = torch.ones((), dtype=torch.float64, device=device)

    @torch.no_grad()
    def reset(self):
        self.cum.zero_(); self.n.zero_()

    @torch.no_grad()
    def kfield(self):                        # -> (1,1,H,W,C) float32, from PAST windows only
        kh = self.ks[self.cum.argmax(0)]                                    # ks float64 -> k math in float64
        if self.smooth:
            kh = _box3(kh)
        k = 1.0 + (kh - 1.0) * self.n / (self.n + self.n0)          # n = 0 -> exactly 1.0
        if self.snap:                                                # legacy: snap to nearest grid point
            j = (self.ks[:, None, None, None] - k[None]).abs().argmin(0)
            return self.ks32[j][None, None]
        return k.to(torch.float32)[None, None]                       # continuous k (default since 01:4xZ 09-25)

    @torch.no_grad()
    def update(self, P, C, h, y_tgt, scored):
        """(1,T,H,W,C) target-normalised; scored: bool (1,T,H,W,C). Call after the window's target is revealed."""
        hk = h * self.ks32[:, None, None, None, None]                # (K,T,H,W,C)
        lo = torch.minimum(C - hk, P); hi = torch.maximum(C + hk, P)
        w = torch.exp(-((hi - lo).to(self.acc_dtype) * self.std_tgt) / self.sigma)
        cov = (y_tgt >= lo) & (y_tgt <= hi) & scored
        self.cum += torch.where(cov, w, torch.zeros_like(w)).sum(1)
        self.n += self.one


# ----------------------------------------------------------------------------------------------- self-test
MEAN_TGT = (0.154962569, -0.000517793698); STD_TGT = (0.0968104079, 0.0159636438)   # CLAUDE.md normaliser


def _load(alpha=0.80):
    import sys, numpy as np
    sys.path.insert(0, "/home/william/realpde-shared/fen"); from sps9 import optimal_band, prep
    R = "/home/william/pitzer-requests/emre/twin_export/results/"
    d = np.load(R + "export_unseen.npz", allow_pickle=True); ids, y, sig = d["ids"], d["y"], d["sig_twin"]
    e0 = d["e_a0"].astype(np.float32); e1 = d["e_a1"].astype(np.float32)
    P = (y + (1 - alpha) * e0 + alpha * e1).astype(np.float32); C = (y + e0).astype(np.float32)
    pad3 = lambda a: np.concatenate([a, np.zeros_like(a[..., :1])], -1)
    traj = np.array([str(i).split(":")[0] for i in ids]); tid = np.array([int(str(i).split(":")[1]) for i in ids])
    ds = np.load(R + "export_seen.npz", allow_pickle=True)
    aoas = np.array([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]]); fs = aoas >= 15
    e0s, e1s, ys, sigs = ds["e_a0"].astype(np.float32)[fs], ds["e_a1"].astype(np.float32)[fs], ds["y"][fs], ds["sig_twin"][fs]
    Pf, Yf = pad3((ys + (1 - alpha) * e0s + alpha * e1s).astype(np.float32)), pad3(ys)
    sref = np.maximum(sigs.mean(0, keepdims=True), 1e-8).astype(np.float32)
    As, as_, ms_, _ = prep(Pf, Yf)
    b = optimal_band((np.abs(e0s) * np.float32(1.39) / (np.clip(sigs / sref, 0.3, 3.0)[..., :2] ** 0.45)).astype(np.float32), as_, ms_)
    o = b.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * 5; pad[ax] = (1, 1); q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(3)], axis=0)
    b = o.astype(np.float32)
    h_all = np.stack([(b * np.clip(sig[i:i+1] / sref, 0.3, 3.0)[..., :2] ** 0.45)[0] for i in range(len(ids))]).astype(np.float32)
    _, a_all, msk, _ = prep(pad3(P), pad3(y)); nsc = msk[..., :2].reshape(len(ids), -1).sum(1)
    return ids, traj, tid, P[..., :2], C[..., :2], y[..., :2], h_all, a_all, nsc


def self_test(device="cpu", expect_hi=None, snap=False, ks=KS, acc_dtype=torch.float64):
    import sys, numpy as np
    sys.path.insert(0, __file__.rsplit("/", 1)[0]); import causal_band as ref
    ids, traj, tid, P, C, y, h, a_all, nsc = _load()
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)[None]
    mean = np.array(MEAN_TGT, np.float32); std = np.array(STD_TGT, np.float32)
    for mode in ("identity", "normalised"):
        if mode == "identity":
            sc = CausalBandScalerT(ks=ks, std_tgt=(1.0, 1.0), device=device, snap=snap, acc_dtype=acc_dtype); f = lambda v: v; fh = lambda v: v
        else:
            sc = CausalBandScalerT(ks=ks, std_tgt=STD_TGT, device=device, snap=snap, acc_dtype=acc_dtype); f = lambda v: ((v - mean) / std).astype(np.float32); fh = lambda v: (v / std).astype(np.float32)
        ref_sc = ref.CausalBandScaler(ks=ks, snap=snap); mism = 0; nel = 0; num_t = num_n = num_0 = 0.0; n_hi = 0.0
        for t in sorted(set(traj)):
            idx = np.where(traj == t)[0][np.argsort(tid[traj == t])]; sc.reset(); ref_sc.reset(); hi_t = int(t.split("_")[1]) >= 15
            for n, i in enumerate(idx):
                kt = sc.kfield()[0, 0].cpu().numpy(); kn = ref_sc.kfield(); mism += int((kt != kn).sum()); nel += kt.size
                if n == 0: assert np.all(kt == np.float32(1.0)), "n=0 must give exactly 1.0"
                if hi_t:
                    for kk, acc in ((kt, "t"), (kn, "n"), (np.ones_like(kt), "0")):
                        lo, hi = ref.interval(P[i], C[i], h[i], kk); v = a_all[i] * ref.sps_numerator(lo, hi, y[i])
                        if acc == "t": num_t += v
                        elif acc == "n": num_n += v
                        else: num_0 += v
                scored = T(y[i] != 0)
                sc.update(T(f(P[i])), T(f(C[i])), T(fh(h[i])), T(f(y[i])), scored); ref_sc.update(P[i], C[i], h[i], y[i])
            if hi_t: n_hi += nsc[idx].sum()
        g_t, g_n = 100 * (num_t - num_0) / n_hi, 100 * (num_n - num_0) / n_hi
        print(f"[{device}] snap={snap} K={len(ks)} acc={str(acc_dtype).split('.')[-1]} {mode:>10s}: k-field mismatches {mism}/{nel} ({100*mism/nel:.4f}%); AoA>=15 gain torch {g_t:+.3f} vs numpy {g_n:+.3f} sps" + (f" (expected {expect_hi:+.3f})" if expect_hi else ""))
        if mode == "identity" and acc_dtype == torch.float64:
            assert mism == 0 and (expect_hi is None or abs(g_t - expect_hi) < 0.002)
        else:   # Jonas's fallback criterion: <0.1% pixels differ and gain within 0.005 sps
            assert mism / nel < 1e-3 and abs(g_t - g_n) < 0.005
    # causality assertion (normalised mode): shuffle the future and replace its targets by noise
    sc = CausalBandScalerT(ks=ks, std_tgt=STD_TGT, device=device, snap=snap, acc_dtype=acc_dtype); rng = np.random.default_rng(0)
    for t in sorted(set(traj)):
        idx = np.where(traj == t)[0][np.argsort(tid[traj == t])]; m = len(idx) // 2
        idx2 = np.concatenate([idx[:m], rng.permutation(idx[m:])])
        outs = []
        for order, corrupt in ((idx, False), (idx2, True)):
            sc.reset(); kf = []
            for n, i in enumerate(order):
                kf.append(sc.kfield()[0, 0].cpu().numpy().copy())
                yy = y[i] if not (corrupt and n >= m) else rng.normal(size=y[i].shape).astype(np.float32)
                sc.update(T(f(P[i])), T(f(C[i])), T(fh(h[i])), T(f(yy)), T(yy != 0))
            outs.append(kf)
        for n in range(m + 1):
            assert np.array_equal(outs[0][n], outs[1][n]), f"causality violated at window {n} of {t}"
    print(f"[{device}] snap={snap} K={len(ks)}: causality assertion passed on 8 trajectories (future shuffled + noised -> past k-fields bitwise identical)")


KS8 = [0.5 * 2 ** (j / 4) for j in range(8)]          # asymmetric log-8 containing 1.0: 0.5 .. 1.68

if __name__ == "__main__":
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devs:
        self_test(dev, expect_hi=0.489, snap=True, ks=KS16)  # legacy reproduction
        self_test(dev, snap=False, ks=KS)                     # SHIPPING: continuous k, K=9 log
        self_test(dev, snap=False, ks=KS, acc_dtype=torch.float32)   # fp32 accumulator (latency lever; fallback criterion)
    if not torch.cuda.is_available():
        print("no CUDA here: run `python causal_band_t2.py` on the node for the CUDA pass")
