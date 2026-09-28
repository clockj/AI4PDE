# c = (window-to-window variance of the time-mean error of y0) / (sigma_head^2), pooled over pixels, per channel, per trajectory.
import numpy as np, sys
sys.argv = ["x"]
import probe2 as p
S = "/fs/scratch/PAS1187/qwu/tmp/realpde/final0927/ckpt/c/"
for reg, D in (("in", p.load(S + "twins_unseen.npz", (0, 1, 2), (3, 4))), ("ood", p.load(S + "ood_aoa20.npz", (0, 1), (4,)))):
    w0 = 1 / 3; y0 = (1 - w0) * D["yf"] + w0 * D["yv"]; sig = ((1 - w0) * D["sf"] + w0 * D["sv"]) * p.STD
    rows = []
    for t in dict.fromkeys(D["names"]):
        s = D["names"] == t
        e = (y0[s] - D["Y"][s]).mean(1)                    # (n,H,W,2)
        v = e.var(0, ddof=1); sg2 = (sig[s][:, 0] ** 2).mean(0); b2 = e.mean(0) ** 2
        c = v.mean((0, 1)) / sg2.mean((0, 1)); tau = np.maximum(b2.mean((0, 1)) - v.mean((0, 1)) / len(e), 0) / v.mean((0, 1))
        rows.append(np.concatenate([c, tau]))
        print(f"{reg} {t:10s} c(u,v) {c[0]:.3f} {c[1]:.3f}   tau2/s2 (u,v) {tau[0]:.3f} {tau[1]:.3f}  -> optimal N0 {1 / max(tau[0], 1e-9):.1f} {1 / max(tau[1], 1e-9):.1f}")
    r = np.array(rows); print(reg, "mean c", r[:, :2].mean(0), "median c", np.median(r[:, :2], 0), "mean tau2/s2", r[:, 2:].mean(0))
