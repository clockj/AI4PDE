"""Probe 1: the band CENTRE (prediction P and a_i fixed).  Probe 2: Re / speed scaling of the width.  LOTO, paired."""
import numpy as np
from bandlab import Lab, sigma_ratio, compose, P_EXP
L = Lab(); y0, sig = L.y0, L.sig
print("trajectories", L.names, "hi", L.hi)
print("local metrics rel/tke/mvpe hi:", [float(m[np.isin(L.traj, L.hi)].mean()) for m in L.met], "all:", [float(m.mean()) for m in L.met])
ref = L.run(y0); L.table("V0 shipped (centre y0)", ref)
L.table("V0b no contain-P", L.run(y0, contain=False), ref)
m = y0.mean(1, keepdims=True)
for b in (0.9, 0.8, 0.65, 0.5, 0.0):
    L.table(f"V1 centre m+{b}(y0-m)", L.run(m + b * (y0 - m)), ref)
t = np.arange(20, dtype=np.float32)[None, :, None, None, None] / 19.0
for b in (0.8, 0.6):
    L.table(f"V1t beta_t 1->{b}", L.run(m + (1 - (1 - b) * t) * (y0 - m)), ref)
xm = L.x.mean(1, keepdims=True).astype(np.float32)
for g in (0.1, 0.25):
    L.table(f"V2 centre (1-{g})y0+{g}*mean_t(x)", L.run((1 - g) * y0 + g * xm), ref)
L.table("V3 centre FFNO-only mean", L.run(L.F[:3].mean(0)), ref)
L.table("V3 centre ViT-only mean", L.run(L.F[3:].mean(0)), ref)
L.table("V3 centre member median", L.run(np.median(L.F, 0)), ref)
ys = y0.copy(); ys[:, 1:-1] = (y0[:, :-2] + y0[:, 1:-1] + y0[:, 2:]) / 3
L.table("V4 centre t-smooth3", L.run(ys), ref)
# ---- probe 2: speed scaling.  U = mean_t,space of input u (own window only)
U = np.abs(L.x[..., 0]).mean((1, 2, 3)).astype(np.float32)[:, None, None, None, None]
for q in (0.45, 0.7, 1.0):
    def rf(fit, q=q):
        sn = sig / U
        r = np.clip(sn / np.maximum(sn[fit].mean(0, keepdims=True), 1e-8), 0.3, 3.0) ** P_EXP
        return (r * (U / U[fit].mean()) ** q).astype(np.float32)
    L.table(f"V5 nd-sigma^.45 * U^{q}", L.run(y0, rf), ref)
for q in (0.3, 0.55):
    L.table(f"V5b shipped ratio * U^{q}", L.run(y0, lambda fit, q=q: (sigma_ratio(sig, fit) * (U / U[fit].mean()) ** q).astype(np.float32)), ref)
for infl in (1.2, 1.6, 1.8):
    L.table(f"V6 inflate {infl}", L.run(y0, infl=infl), ref)
