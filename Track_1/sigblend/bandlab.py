"""Design B band lab: leave-one-trajectory-out (LOTO) containment probes on held-out twin fields (dump_twins.py).

Every variant = (centre field c, per-window width factor R).  half = base * R, base solved per element by the exact
sps9 solver on the FIT trajectories' |c - y| * inflate / R, box-smoothed k=3 (exactly fit_band.py), interval
[min(c-h, P), max(c+h, P)] (shipped 'contain P' rule) unless contain=False.
LOTO: an AoA 15-20 eval trajectory is scored with a base fitted on the OTHER AoA 15-20 trajectories; any other eval
trajectory with the base fitted on all three AoA 15-20 trajectories (= the shipped fit set).  Nothing is scored in-sample.
Stretch f: target moved to y_f = c0 + f*(y - c0) (c0 = the shipped centre y0, the same stretched target for every variant)
and the three metrics in a_i multiplied by f.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); T1 = os.path.dirname(HERE)
sys.path.insert(0, T1)
import sps9
from sps9 import optimal_band, SIGMA
DUMP = os.environ.get("B_DUMP") or sys.exit("bandlab: set B_DUMP to the npz written by dump_twins.py")
ALPHA, P_EXP, INFL = 0.80, 0.45, 1.39
FS = (1.0, 1.29, 1.39)


def load():
    d = np.load(DUMP)
    F, S, x, y = d["F"].astype(np.float32), d["S"].astype(np.float32), d["x"], d["y"]
    traj = np.array([str(i).split(":")[0] for i in d["ids"]])
    aoa = np.array([int(t.split("_")[1]) for t in traj]); re = np.array([int(t.split("_")[0]) for t in traj])
    return F, S, x, y, traj, aoa, re


def compose(y0, sig, alpha=ALPHA):
    m = y0.mean(1, keepdims=True); s = np.maximum(y0.std(1, keepdims=True), 1e-8)
    return (m + (y0 - m) * (alpha * sig + (1 - alpha) * s) / s).astype(np.float32)


def smooth(b, k=3):
    o = b.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * 5; pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(k)], axis=0)
    return o.astype(np.float32)


def sigma_ratio(sig, fit, p=P_EXP):
    ref = np.maximum(sig[fit].mean(0, keepdims=True), 1e-8)
    return (np.clip(sig / ref, 0.3, 3.0) ** p).astype(np.float32)


class Lab:
    def __init__(self):
        self.F, self.S, self.x, self.y, self.traj, self.aoa, self.re = load()
        self.y0 = self.F.mean(0); self.sig = self.S.mean(0)
        self.P = compose(self.y0, self.sig)
        P3 = np.concatenate([self.P, np.zeros_like(self.P[..., :1])], -1)
        Y3 = np.concatenate([self.y, np.zeros_like(self.y[..., :1])], -1)
        self.met = sps9.sample_metrics(P3, Y3)            # local (f = 1) metrics of the scored prediction
        self.scored = self.y != 0.0
        self.names = sorted(set(self.traj), key=lambda t: (int(t.split("_")[1]), int(t.split("_")[0])))
        self.hi = [t for t in self.names if int(t.split("_")[1]) in (15, 20)]

    def a(self, f=1.0):
        return sps9.weights_from_metrics(*(f * m for m in self.met))

    def fits(self):
        """eval trajectory -> boolean mask of its FIT windows (LOTO inside AoA 15-20; shipped fit set otherwise)."""
        out = {}
        for t in self.names:
            out[t] = np.isin(self.traj, [h for h in self.hi if h != t])
        return out

    def run(self, centre, rfun=None, infl=INFL, contain=True, fitmask=None, basefun=None):
        """centre (N,20,H,W,2); rfun(fit_mask) -> R broadcastable to centre (default: shipped sigma ratio).
        returns {f: {traj: (sps, abar)}}"""
        res = {f: {} for f in FS}
        a1 = self.a(1.0)
        for t, fit in (fitmask or self.fits()).items():
            R = sigma_ratio(self.sig, fit) if rfun is None else rfun(fit)
            R = np.broadcast_to(R, centre.shape)
            A = np.abs(centre[fit] - self.y[fit]) * np.float32(infl) / R[fit]
            base = smooth(optimal_band(A.astype(np.float32), a1[fit], self.scored[fit]))
            if basefun is not None:
                base = basefun(base)
            ev = self.traj == t
            h = base * R[ev]
            lo, up = centre[ev] - h, centre[ev] + h
            if contain:
                lo, up = np.minimum(lo, self.P[ev]), np.maximum(up, self.P[ev])
            sc_ = self.scored[ev]; n = int(sc_.sum())
            for f in FS:
                yf = self.y0[ev] + np.float32(f) * (self.y[ev] - self.y0[ev])
                con = np.where((yf >= lo) & (yf <= up) & sc_, np.exp(-(up - lo).astype(np.float64) / SIGMA), 0.0)
                af = self.a(f)[ev]
                sps = 100.0 * float(af @ con.reshape(len(con), -1).sum(1)) / n
                res[f][t] = (sps, float(af.mean()), 100.0 * float(con.sum()) / n)
        return res

    def table(self, name, res, ref=None):
        """one line per stretch: sps / C (x100) on AoA 15-20 and the full split, paired delta vs ref and sign counts."""
        lines = []
        for f in FS:
            row = [f"{name:<28s} f={f:4.2f}"]
            for lab, ts in (("hi", self.hi), ("all", self.names)):
                s = np.mean([res[f][t][0] for t in ts]); c = np.mean([res[f][t][2] for t in ts])
                txt = f"{lab}: sps {s:6.3f} C {c:6.3f}"
                if ref is not None:
                    ds = [res[f][t][0] - ref[f][t][0] for t in ts]; dc = [res[f][t][2] - ref[f][t][2] for t in ts]
                    txt += f"  dsps {np.mean(ds):+.3f} dC {np.mean(dc):+.3f} ({sum(d > 0 for d in dc)}/{len(ts)})"
                row.append(txt)
            lines.append(" | ".join(row))
        print("\n".join(lines), flush=True)
