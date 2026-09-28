# per-trajectory mvpe/rel error ratio of the baseline prediction (no corrections), both regimes: is the board's ratio inside the in-distribution spread?
import numpy as np, sys
sys.argv = ["x"]
import probe2 as p
S = "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/"
band = np.load("/users/PAS1187/qwu/workspace/realpde/main/Track_1/bands/r35_centre0_fill.npz"); band = {k: band[k] for k in band.files}
pad = lambda a: np.concatenate([a, np.zeros_like(a[..., :1])], -1).astype(np.float32)
for reg, D in (("in", p.load(S + "twins_unseen.npz", (0, 1, 2), (3, 4))), ("ood", p.load(S + "ood_aoa20.npz", (0, 1), (4,)))):
    w0 = 1 / 3; y0 = (1 - w0) * D["yf"] + w0 * D["yv"]; sig = (1 - w0) * D["sf"] + w0 * D["sv"]
    yn = (y0 - p.MEAN) / p.STD; m_ = yn.mean(1, keepdims=True); s_ = yn.std(1, keepdims=True)
    P = ((m_ + (yn - m_) * (0.8 * sig + 0.2 * s_) / (s_ + 1e-6)) * p.STD + p.MEAN).astype(np.float32)
    Y = D["Y"]
    for t in dict.fromkeys(D["names"]):
        s = D["names"] == t
        r = float(np.mean(p.sc.rel_l2_per_sample(pad(P[s]), pad(Y[s]), 2))); k = float(np.mean(p.sc.tke_rel_l2_per_sample(pad(P[s]), pad(Y[s]), 2))); m = float(p.sc.mvpe_rel_l2(pad(P[s]), pad(Y[s])))
        # persistent share: error of the trajectory-mean of (P - Y) time-means relative to the rms window-mean error
        e = (y0[s] - Y[s]).mean(1); pers = float((e.mean(0) ** 2).sum() / (e ** 2).mean(0).sum())
        print(f"{reg:4s} {t:10s} n {int(s.sum()):3d} rel_e {r:.4f} tke_e {k:.4f} mvpe_e {m:.4f} mvpe/rel {m / r:.2f} tke/rel {k / r:.2f} persistent share of window-mean error energy {pers:.3f}")
