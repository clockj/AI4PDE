"""R1 reference implementation: causal per-pixel band multiplier from a trajectory's own realized errors.
Track-2 only. numpy. Physical units (same units as band.npz half-widths and targets).

Deployed interval (unchanged): lo = min(c - k*h, P), hi = max(c + k*h, P); sps element score
= 1[lo <= y <= hi] * exp(-(hi - lo)/SIGMA) on scored (y != 0) elements.

State per trajectory: cum (K, H, W, C) = running sum over PAST windows of, for each candidate k,
sum_t [ 1[lo_k <= y <= hi_k] * exp(-(hi_k - lo_k)/SIGMA) * scored ]    (unweighted; a_i weighting
verified to change nothing to 3 decimals on 8/8 trajectories).

Rule for the CURRENT window (n windows already seen):
  n == 0  -> k = 1 everywhere
  n >= 1  -> k_hat = KS[argmax_k cum]; k_hat = box3x3(k_hat) (edge-padded mean, per channel);
             k = 1 + (k_hat - 1) * n / (n + N0);  snap to grid KS.
Shipping config (01:4xZ 09-25): continuous k (no snap), K=9 log grid 2^(j/4) j=-4..4, N0 = 2, box 3x3.
Legacy reference (+0.489): snap=True with KS16 (linear 0.5..2.0 step 0.1).

Causality: update(...) is called AFTER the current window's interval was produced, with the revealed
target; it never sees the current or future windows before they are scored.  self_test() asserts it.
"""
import numpy as np
SIGMA = 0.0563870259
KS16 = np.round(np.arange(0.5, 2.0 + 1e-9, 0.1), 3)        # legacy linear grid (snap version, +0.489)
KS = [0.5 * 2 ** (j / 4) for j in range(9)]                 # SHIPPING grid (Jonas 01:41Z): K=9 log, 0.5..2.0, contains 1.0
N0 = 2


def box3(a):
    """3x3 edge-padded box mean over the two spatial axes of an (H, W, C) array."""
    o = a.astype(np.float64)
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim; pad[ax] = (1, 1)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + a.shape[ax]), axis=ax) for i in range(3)], axis=0)
    return o


class CausalBandScaler:
    def __init__(self, shape=(32, 64, 2), ks=KS, n0=N0, smooth=True, snap=False):
        self.ks, self.n0, self.smooth, self.shape, self.snap = np.asarray(ks, np.float64), float(n0), smooth, shape, snap
        self.k1 = int(np.abs(self.ks - 1.0).argmin()); assert abs(self.ks[self.k1] - 1.0) < 1e-9
        self.reset()

    def reset(self):                                   # trajectory boundary (reset_ttt_state)
        self.cum = np.zeros((len(self.ks),) + self.shape, np.float64); self.n = 0

    def kfield(self):
        """Multiplier field (H, W, C) for the CURRENT window, from past windows only."""
        if self.n == 0:
            return np.ones(self.shape, np.float32)
        kh = self.ks[self.cum.argmax(0)]
        if self.smooth:
            kh = box3(kh)
        k = 1.0 + (kh - 1.0) * self.n / (self.n + self.n0)
        if self.snap:   # legacy (09-25 01:xxZ reference): snap to nearest grid point
            return self.ks[np.abs(self.ks[:, None, None, None] - k[None]).argmin(0)].astype(np.float32)
        return k.astype(np.float32)   # continuous k (Jeff-Dean 01:37Z): grid only indexes the cumulative

    def update(self, P, C, h, y):
        """After the window is scored: P, C, h, y are (T, H, W, C) physical; h = UNSCALED shipped
        half-width for that window (base * sigma-ratio^p), i.e. before this scaler's multiplier."""
        scored = (y != 0)
        for j, k in enumerate(self.ks):
            hk = h * np.float32(k)
            lo, hi = np.minimum(C - hk, P), np.maximum(C + hk, P)
            w = np.exp(-(hi - lo).astype(np.float64) / SIGMA)
            self.cum[j] += np.where((y >= lo) & (y <= hi) & scored, w, 0.0).sum(0)
        self.n += 1


def interval(P, C, h, k):
    hk = h * k[None]
    return np.minimum(C - hk, P), np.maximum(C + hk, P)


def sps_numerator(lo, hi, y):
    """Per-window unweighted sps numerator (sum over scored elements of cov * width term)."""
    scored = (y != 0)
    return float(np.where((y >= lo) & (y <= hi) & scored, np.exp(-(hi - lo).astype(np.float64) / SIGMA), 0.0).sum())


def run_trajectory(P, C, h, y, scaler=None, **kw):
    """P, C, h, y: (n_win, T, H, W, C) in time order. Returns list of k-fields used per window."""
    sc = scaler or CausalBandScaler(P.shape[2:], **kw); sc.reset(); ks = []
    for n in range(len(P)):
        k = sc.kfield(); ks.append(k)              # 1) interval for window n from PAST only
        sc.update(P[n], C[n], h[n], y[n])         # 2) then learn from window n's revealed target
    return ks


def self_test(export="/home/william/pitzer-requests/emre/twin_export/results/export_unseen.npz",
              export_seen="/home/william/pitzer-requests/emre/twin_export/results/export_seen.npz",
              alpha=0.80, expect_hi=None, snap=False, ks=KS):
    import sys; sys.path.insert(0, "/home/william/realpde-shared/fen")
    from sps9 import optimal_band, prep
    d = np.load(export, allow_pickle=True); ids, y, sig = d["ids"], d["y"], d["sig_twin"]
    e0 = d["e_a0"].astype(np.float32); e1 = d["e_a1"].astype(np.float32)
    P = (y + (1 - alpha) * e0 + alpha * e1).astype(np.float32); C = (y + e0).astype(np.float32)
    pad3 = lambda a: np.concatenate([a, np.zeros_like(a[..., :1])], -1)
    traj = np.array([str(i).split(":")[0] for i in ids]); tid = np.array([int(str(i).split(":")[1]) for i in ids])
    ds = np.load(export_seen, allow_pickle=True)
    aoas = np.array([int(str(i).split(":")[0].split("_")[1]) for i in ds["ids"]]); fs = aoas >= 15
    e0s, e1s, ys, sigs = ds["e_a0"].astype(np.float32)[fs], ds["e_a1"].astype(np.float32)[fs], ds["y"][fs], ds["sig_twin"][fs]
    Pf, Yf = pad3((ys + (1 - alpha) * e0s + alpha * e1s).astype(np.float32)), pad3(ys)
    sref = np.maximum(sigs.mean(0, keepdims=True), 1e-8).astype(np.float32)
    rat_f = np.clip(sigs / sref, 0.3, 3.0)[..., :2] ** 0.45
    As, as_, ms_, _ = prep(Pf, Yf)
    b = optimal_band((np.abs(e0s) * np.float32(1.39) / rat_f).astype(np.float32), as_, ms_)
    o = b.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * 5; pad[ax] = (1, 1); q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(3)], axis=0)
    b = o.astype(np.float32)
    h_all = np.stack([(b * np.clip(sig[i:i+1] / sref, 0.3, 3.0)[..., :2] ** 0.45)[0] for i in range(len(ids))]).astype(np.float32)
    _, a_all, msk, _ = prep(pad3(P), pad3(y)); nsc = msk[..., :2].reshape(len(ids), -1).sum(1)
    Pu, Cu, yu = P[..., :2], C[..., :2], y[..., :2]
    num1 = num0 = 0.0; n_hi = 0.0
    for t in sorted(set(traj)):
        idx = np.where(traj == t)[0][np.argsort(tid[traj == t])]
        ks_ = run_trajectory(Pu[idx], Cu[idx], h_all[idx], yu[idx], snap=snap, ks=ks)
        assert np.all(ks_[0] == np.float32(1.0)), 'n=0 must give exactly 1.0'
        if int(t.split("_")[1]) >= 15:
            for n, i in enumerate(idx):
                lo, hi = interval(Pu[i], Cu[i], h_all[i], ks_[n]); num1 += a_all[i] * sps_numerator(lo, hi, yu[i])
                lo, hi = interval(Pu[i], Cu[i], h_all[i], np.ones(ks_[n].shape, np.float32)); num0 += a_all[i] * sps_numerator(lo, hi, yu[i])
            n_hi += nsc[idx].sum()
        # ---- CAUSALITY ASSERTION on this trajectory: corrupt / shuffle the FUTURE, outputs unchanged
        rng = np.random.default_rng(0); m = len(idx) // 2
        idx2 = np.concatenate([idx[:m], rng.permutation(idx[m:])])                   # shuffle future order
        y_bad = yu[idx2].copy(); y_bad[m:] = rng.normal(size=y_bad[m:].shape).astype(np.float32)  # noise future targets
        ks2 = run_trajectory(Pu[idx2], Cu[idx2], h_all[idx2], y_bad, snap=snap, ks=ks)
        for n in range(m + 1):   # windows 0..m see identical pasts -> identical k (window m's k uses windows < m only)
            assert np.array_equal(ks_[n], ks2[n]), f"causality violated at window {n} of {t}"
    gain = 100 * (num1 - num0) / n_hi
    print(f"self_test(snap={snap}, K={len(ks)}): AoA>=15 causal gain {gain:+.3f} sps" + (f" (expected {expect_hi:+.3f})" if expect_hi else "") + "; n=0 -> exactly 1.0; causality assertions passed on 8 trajectories")
    if expect_hi is not None: assert abs(gain - expect_hi) < 0.002, gain
    return gain


KS8 = [0.5 * 2 ** (j / 4) for j in range(8)]          # asymmetric log-8, contains 1.0: 0.5 .. 1.68

if __name__ == "__main__":
    self_test(snap=True, ks=KS16, expect_hi=0.489)    # legacy reproduction
    self_test(snap=False, ks=KS)                       # SHIPPING: continuous k, K=9 log
