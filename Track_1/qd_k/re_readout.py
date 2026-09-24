"""qdre readout (Fen; Ilse 08:02-08:06Z 09-24, #Track-1-W-2 282746e6). ONE checkpoint (one seed) per call.

Question: does the QD direct band (half = k*H, k = 1.1399 FIXED, the shipped constant) still beat a recipe band when
the Reynolds number is outside the training range? Model = a QD twin trained on the 56 trajectories re_split.py
leaves (qdre_arm_sN.pth). Everything below comes from THAT ONE MODEL (Ilse 1: per seed self-contained):
  y0 = alpha-0 prediction (band centre E0), y1 = alpha-1, pe = compose(y0, y1, 0.80), sigma head, H = err head
  (physical: _fno.predict_fno err_out = exp(clamp(logerr)) * std, exactly what the package ships).
Scoring = head_direct.py's construction (Bram's centre band): t = pe + 1.39*(y - pe) where y != 0,
  X = t - E0, D = pe - E0, lo = min(-h, D), up = max(h, D), v = exp(-(up-lo)/SIGMA) if lo <= X <= up and scored,
  sps(traj) = 100 * sum_w a_w * sum v / n_scored,  a_w = sample_weights(pe, y).
CONTROL = recipe band: base, sigma_ref fitted on the 56 TRAINING trajectories, all AoA (Ilse 2), fit_base(y0, y, clip
  (sig/sigma_ref,.3,3), a, 1.39, 0.45); half = k0 * base * clip(sig/sigma_ref,.3,3)^0.45.
  k0 (Ilse 2): rehi, relo -> chosen on ALL of rein, carried over; rein -> leave-one-trajectory-out inside rein.
  Never chosen on training windows (memorised errors would pick a narrow control and flatter QD).
QD: k = 1.1399 fixed on every group (chosen on other data).
OFFICIAL control k (Ilse 08:22Z): SYMMETRIC -- each group's k0 chosen on the other two groups pooled.
Outputs (--out): qdre_delta_<tag>.csv  seed,trajectory,re,group,delta_final  OFFICIAL (symmetric control k)
                 qdre_delta_diag_<tag>.csv  same columns, Ilse's 08:06Z control-k rule (diagnostic)
                 qdre_detail_<tag>.csv the same + sps_qd, sps_ctrl, k_ctrl, n_scored
                 qdre_summary_<tag>.txt per group pooled / median / per-Re means; k* diagnostics
The GATE combines seeds and is applied later (re_gate.py), not here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = 0.249752                        # sps -> final, as head_direct.py
ALPHA, STRETCH, P_SHIP = 0.80, 1.39, 0.45
K_QD = 1.14
KGRID = np.unique(np.concatenate([np.exp(np.linspace(np.log(0.4), np.log(2.5), 22)), [1.0]])).astype(np.float32)
_step = float(np.log(KGRID[1] / KGRID[0]))
KGRID_D = np.unique(np.concatenate([KGRID, np.exp(np.arange(np.log(2.5) + _step, np.log(20.0) + 1e-9, _step))])
                    ).astype(np.float32)
KQ = float(KGRID[np.argmin(np.abs(np.log(KGRID / K_QD)))])      # 1.1399 (the shipped grid point)


def windows(names, data, sub, interval, horizon=40, t_in=20):
    import h5py
    xs, ys, ids = [], [], []
    for name in names:
        with h5py.File(os.path.join(data, f"{name}.h5"), "r") as f:
            u = f["u"][:, ::sub, ::sub].astype(np.float32)
            v = f["v"][:, ::sub, ::sub].astype(np.float32)
        d = np.stack([u, v, np.zeros_like(u)], -1)
        for t in range(0, len(d) - horizon + 1, interval):
            xs.append(d[t:t + t_in]); ys.append(d[t + t_in:t + horizon]); ids.append(f"{name}:{t}")
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)[..., :2].copy(), np.array(ids)


def infer(m, x, dev):
    import _fno
    n = len(x)
    sig = np.empty((n, 1, 32, 64, 2), np.float32)
    H = np.zeros((n, 20, 32, 64, 2), np.float32)
    y0 = _fno.predict_fno([m], x, dev, alpha=0.0, err_out=H)[..., :2].copy()
    y1 = _fno.predict_fno([m], x, dev, alpha=1.0, sigma_out=sig)[..., :2].copy()
    y8 = _fno.predict_fno([m], x[:16], dev, alpha=ALPHA)[..., :2]
    lin = float(np.abs(((1 - ALPHA) * y0[:16] + ALPHA * y1[:16]) - y8).max())
    assert lin < 1e-3, f"compose linearity failed: {lin}"
    assert np.any(H), "err_out is all zero: the checkpoint carries no err head (or _fno did not read it)"
    return y0, y1, sig, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="seed label for the CSV, e.g. 1")
    ap.add_argument("--data", default="/users/PAS1187/qwu/RealPDE/train_real/train_real")
    ap.add_argument("--code", required=True, help="repo root (has Track_1/re_split.py, Track_1/submit_fno/_fno.py)")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--fit-interval", type=int, default=20, help="window interval on the 56 training trajectories")
    ap.add_argument("--fake-export", default=None, help="LOCAL TEST ONLY: build groups from a twin export instead of h5")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.code, "Track_1", "submit_fno"))
    sys.path.insert(0, os.path.join(a.code, "Track_1"))
    sys.path.insert(0, HERE)
    import torch
    import _fno
    import re_split
    from centre_band import SIGMA, compose, fit_base, sample_weights
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    rep = open(os.path.join(a.out, f"qdre_summary_{a.tag}.txt"), "w")

    def say(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True); rep.write(msg + "\n"); rep.flush()

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = blob.get("config", {})
    say(f"ckpt {a.ckpt}  seed={cfg.get('seed')} epochs={cfg.get('epochs')} err_loss={blob.get('err_loss')} "
        f"qd_temp={blob.get('qd_temp')}  device {dev}")
    assert blob.get("err_head") and blob.get("err_loss") == "qd", "not a QD err-head checkpoint"
    assert np.allclose(np.asarray(blob["std"], np.float32).reshape(-1)[:2], _fno.STD[:2]), "normaliser mismatch"
    m = _fno.load_any(a.ckpt, dev)

    if a.fake_export:                  # local plumbing test: 8 unseen twin trajectories -> 3 fake groups
        sys.path.insert(0, os.path.dirname(a.fake_export))
        d = np.load(a.fake_export)
        tr = np.array([str(i).split(":")[0] for i in d["ids"]])
        u = sorted(set(tr))
        G = {"rein": u[:3], "rehi": u[3:6], "relo": u[6:]}
        X_all = np.concatenate([d["x"], np.zeros_like(d["x"][..., :1])], -1)
        Y_all = d["y"]

        def load_set(names, interval=None):
            s = np.isin(tr, names)
            return X_all[s], Y_all[s], d["ids"][s]
        train_names = u[:3]            # fake: fit side = rein trajectories (plumbing only)
    else:
        G = {g: re_split.names_for(res, a.data) for g, res in re_split.SETS.items()}
        held = sorted(n for v in G.values() for n in v)
        allnames = sorted(f[:-3] for f in os.listdir(a.data) if f.endswith(".h5"))
        train_names = [n for n in allnames if n not in set(held)]
        say(f"groups: " + ", ".join(f"{g} {len(v)}" for g, v in G.items()) + f"; train {len(train_names)}")
        assert (len(G["relo"]), len(G["rehi"]), len(G["rein"]), len(train_names)) == (9, 7, 10, 56)
        ex = cfg.get("exclude")
        if ex is not None:
            assert sorted(ex) == held, f"checkpoint was trained with a different exclude set: {sorted(ex)}"
            say("checkpoint exclude set == re_split.excluded()  (26)")
        else:
            say("WARNING: checkpoint config records no 'exclude'; cannot verify the split from the blob")

        def load_set(names, interval=20):
            return windows(names, a.data, re_split.SUB, interval)

    # ---- fit side: recipe base + sigma_ref on the 56 TRAINING trajectories (all AoA) -------------------------
    xf, yf, idf = load_set(train_names, a.fit_interval)
    y0f, y1f, sgf, _ = infer(m, xf, dev)
    del xf
    sref = np.maximum(sgf.mean(0, keepdims=True), 1e-8).astype(np.float32)
    pef = compose(y0f, y1f, ALPHA)
    af = sample_weights(pef, yf)
    base = fit_base(y0f, yf, np.clip(sgf / sref, 0.3, 3.0), af, STRETCH, P_SHIP)[0].astype(np.float32)
    say(f"fit side: {len(yf)} windows  base median {float(np.median(base)):.6f}  sigma_ref median "
        f"{float(np.median(sref)):.6f}  [{time.time() - t0:.0f}s]")
    del y0f, y1f, sgf, pef, yf

    # ---- held side: per group, per trajectory tables over k ------------------------------------------------
    TAB = {}                           # group -> dict(traj ids, a, nsc, tq[k], tc[k])
    for g in ("rein", "rehi", "relo"):
        x, y, ids = load_set(G[g])
        y0, y1, sg, H = infer(m, x, dev)
        del x
        pe = compose(y0, y1, ALPHA)
        t = np.where(y == 0, 0, pe + STRETCH * (y - pe)).astype(np.float32)
        X, D = (t - y0).astype(np.float32), (pe - y0).astype(np.float32)
        sc = t != 0
        aw = sample_weights(pe, y)
        n = len(y)
        ship = (base[None] * np.clip(sg / sref, 0.3, 3.0) ** np.float32(P_SHIP)).astype(np.float32)
        ship = np.ascontiguousarray(np.broadcast_to(ship, X.shape))
        SIG = np.float32(SIGMA)

        def vv_of(h):
            lo, up = np.minimum(-h, D), np.maximum(h, D)
            v = np.where((X >= lo) & (X <= up) & sc, np.exp(-(up - lo) / SIG), np.float32(0))
            return v.reshape(n, -1).sum(1, dtype=np.float64)
        TAB[g] = dict(traj=np.array([str(i).split(":")[0] for i in ids]), a=aw,
                      nsc=sc.reshape(n, -1).sum(1),
                      tq={float(k): vv_of(H * np.float32(k)) for k in KGRID_D},
                      tc={float(k): vv_of(ship * np.float32(k)) for k in KGRID},
                      Hmed=float(np.median(H[sc])), abar=float(aw.mean()))
        say(f"[{g}] {n} windows, {len(set(TAB[g]['traj']))} traj, H median {TAB[g]['Hmed']:.6f}, "
            f"a_bar {TAB[g]['abar']:.4f}  [{time.time() - t0:.0f}s]")
        del y0, y1, sg, H, pe, t, X, D, sc, ship, y

    # Devi 12:22Z: save the per-window tables so any later k readout (cross-selection, oracle, ...) runs locally.
    np.savez_compressed(os.path.join(a.out, f"qdre_tables_{a.tag}.npz"),
                        kgrid=KGRID, kgrid_d=KGRID_D, kq=np.float32(KQ), W=np.float64(W),
                        **{f"{g}_{f_}": v for g in TAB for f_, v in (
                            ("traj", TAB[g]["traj"]), ("a", TAB[g]["a"]), ("nsc", TAB[g]["nsc"]),
                            ("tq", np.stack([TAB[g]["tq"][float(k)] for k in KGRID_D])),
                            ("tc", np.stack([TAB[g]["tc"][float(k)] for k in KGRID])))})
    say(f"saved qdre_tables_{a.tag}.npz (per-window v-sums: tq [kgrid_d x windows], tc [kgrid x windows], a, nsc, traj)")

    def S(g, vv, sel):
        T_ = TAB[g]
        return 100.0 * float(T_["a"][sel] @ vv[sel]) / max(int(T_["nsc"][sel].sum()), 1)

    rein_all = np.ones(len(TAB["rein"]["traj"]), bool)
    k_rein = max(KGRID, key=lambda k: S("rein", TAB["rein"]["tc"][float(k)], rein_all))
    say(f"control k chosen on ALL rein (for rehi, relo): {float(k_rein):.4f}")
    # SYMMETRIC = OFFICIAL (Devi 08:19Z; Ilse 08:22Z, fixed before any number): qdre_delta_<tag>.csv.
    # Ilse's 08:06Z rule (rein LOTO / rehi,relo from rein) is DIAGNOSTIC only: qdre_delta_diag_<tag>.csv.: every group's control k0 is chosen on the
    # OTHER TWO held-out groups pooled (all cross-cell, like QD's k). Pooled sps over groups = n_scored-weighted.
    def k_sym(g):
        others = [h for h in ("rein", "rehi", "relo") if h != g]
        def tot(k):
            num = sum(float(TAB[h]["a"] @ TAB[h]["tc"][float(k)]) for h in others)
            return 100.0 * num / sum(int(TAB[h]["nsc"].sum()) for h in others)
        return max(KGRID, key=tot)
    KS = {g: float(k_sym(g)) for g in ("rein", "rehi", "relo")}
    say("SYMMETRIC control k (chosen on the other two groups): " + ", ".join(f"{g} {k:.4f}" for g, k in KS.items()))
    rows, rows_sym = [], []
    for g in ("rein", "rehi", "relo"):
        T_ = TAB[g]
        for tj in sorted(set(T_["traj"])):
            sel = T_["traj"] == tj
            if g == "rein":            # LOTO inside rein
                keep = ~sel
                k0 = max(KGRID, key=lambda k: S("rein", T_["tc"][float(k)], keep))
            else:
                k0 = k_rein
            sq, s0 = S(g, T_["tq"][KQ], sel), S(g, T_["tc"][float(k0)], sel)
            rows.append((a.tag, tj, int(tj.split("_")[0]), g, W * (sq - s0), sq, s0, float(k0), int(T_["nsc"][sel].sum())))
            s1_ = S(g, T_["tc"][KS[g]], sel)
            rows_sym.append((a.tag, tj, int(tj.split("_")[0]), g, W * (sq - s1_), sq, s1_, KS[g], int(T_["nsc"][sel].sum())))
    with open(os.path.join(a.out, f"qdre_delta_{a.tag}.csv"), "w") as f:
        f.write("seed,trajectory,re,group,delta_final\n")
        for r in rows_sym:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f}\n")
    for g in ("rein", "rehi", "relo"):
        d = np.array([r[4] for r in rows_sym if r[3] == g])
        say(f"[OFFICIAL symmetric {g}] mean {d.mean():+.5f}  median {float(np.median(d)):+.5f}  n+ {int((d > 0).sum())}/{len(d)}")
    # ORACLE control (Ilse 12:16Z): control at its IN-SAMPLE best k inside each group (it sees the group's own targets);
    # QD still fixed 1.1399. Can only UNDERSTATE QD's advantage: a lower bound, not an estimate.
    KO = {}
    for g in ("rein", "rehi", "relo"):
        allg = np.ones(len(TAB[g]["traj"]), bool)
        KO[g] = float(max(KGRID, key=lambda k: S(g, TAB[g]["tc"][float(k)], allg)))
    rows_or = []
    for g in ("rein", "rehi", "relo"):
        T_ = TAB[g]
        for tj in sorted(set(T_["traj"])):
            sel = T_["traj"] == tj
            sq, so = S(g, T_["tq"][KQ], sel), S(g, T_["tc"][KO[g]], sel)
            rows_or.append((a.tag, tj, int(tj.split("_")[0]), g, W * (sq - so), sq, so, KO[g], int(T_["nsc"][sel].sum())))
    with open(os.path.join(a.out, f"qdre_delta_oracle_{a.tag}.csv"), "w") as f:
        f.write("seed,trajectory,re,group,delta_final\n")
        for r in rows_or:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f}\n")
    for g in ("rein", "rehi", "relo"):
        d = np.array([r[4] for r in rows_or if r[3] == g])
        say(f"[ORACLE-control lower bound {g}] control k {KO[g]:.4f} (in-sample best)  mean {d.mean():+.5f}  "
            f"median {float(np.median(d)):+.5f}  n+ {int((d > 0).sum())}/{len(d)}")
    # QD WIDE vs NARROW (Jonas via Ilse 12:17Z): QD at the next grid point up minus QD at 1.1399, per trajectory.
    # On rein it is the price of widening; on rehi the gain from widening at high Re. No control involved.
    KW = float(KGRID[np.searchsorted(KGRID, KQ) + 1])
    with open(os.path.join(a.out, f"qdre_delta_qdwide_{a.tag}.csv"), "w") as f:
        f.write("seed,trajectory,re,group,delta_final\n")
        for g in ("rein", "rehi", "relo"):
            T_ = TAB[g]
            dd = []
            for tj in sorted(set(T_["traj"])):
                sel = T_["traj"] == tj
                dv = W * (S(g, T_["tq"][KW], sel) - S(g, T_["tq"][KQ], sel)); dd.append(dv)
                f.write(f"{a.tag},{tj},{int(tj.split('_')[0])},{g},{dv:.6f}\n")
            dd = np.array(dd)
            say(f"[QD k {KW:.4f} - QD k {KQ:.4f}, {g}] mean {dd.mean():+.5f}  median {float(np.median(dd)):+.5f}  "
                f"n+ {int((dd > 0).sum())}/{len(dd)}")
    with open(os.path.join(a.out, f"qdre_delta_diag_{a.tag}.csv"), "w") as f:
        f.write("seed,trajectory,re,group,delta_final\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f}\n")
    # Devi 08:51Z: every detail line says which rule it is. rule=official (symmetric) first, then rule=diag.
    with open(os.path.join(a.out, f"qdre_detail_{a.tag}.csv"), "w") as f:
        f.write("rule,seed,trajectory,re,group,delta_final,sps_qd,sps_ctrl,k_ctrl,n_scored\n")
        for lab, RR in (("official", rows_sym), ("diag", rows), ("oracle", rows_or)):
            for r in RR:
                f.write(lab + "," + ",".join(str(x) if not isinstance(x, float) else f"{x:.6f}" for x in r) + "\n")
    for g in ("rein", "rehi", "relo"):
        d = np.array([r[4] for r in rows if r[3] == g])
        say(f"[DIAG (Ilse 08:06Z rule) {g}] mean {d.mean():+.5f}  median {float(np.median(d)):+.5f}  n+ {int((d > 0).sum())}/{len(d)}")
    say("\n##### OFFICIAL detail below = SYMMETRIC control k (Ilse 08:22Z). DIAG rule appears only in the lines above "
        "and in qdre_delta_diag / rule=diag rows. #####")

    for g in ("rein", "rehi", "relo"):
        R = [r for r in rows_sym if r[3] == g]
        d = np.array([r[4] for r in R]); nh = np.array([r[8] for r in R])
        say(f"\n[OFFICIAL {g}] per-traj delta final (QD k={KQ:.4f} - control at symmetric k0 {KS[g]:.4f}):")
        for r in R:
            say(f"    {r[1]:10s} Re {r[2]:6d}  delta {r[4]:+.5f}  sps QD {r[5]:.4f}  ctrl {r[6]:.4f} (k0 {r[7]:.4f})")
        say(f"  pooled (n_scored-weighted) {float((d * nh).sum() / nh.sum()):+.5f}  mean {d.mean():+.5f}  "
            f"median {float(np.median(d)):+.5f}  n+ {int((d > 0).sum())}/{len(d)}")
        for re_ in sorted(set(r[2] for r in R)):
            dd = np.array([r[4] for r in R if r[2] == re_])
            say(f"  Re {re_:6d}: mean {dd.mean():+.5f} over {len(dd)} traj")
        # k* diagnostics (IN-SAMPLE over the group, optimistic; Emre / Ilse): QD k* and control k*
        allg = np.ones(len(TAB[g]["traj"]), bool)
        kq = max(KGRID_D, key=lambda k: S(g, TAB[g]["tq"][float(k)], allg))
        kc = max(KGRID, key=lambda k: S(g, TAB[g]["tc"][float(k)], allg))
        steps = int(round(np.log(float(kq) / KQ) / _step))
        say(f"  [diag, in-sample] QD k* {float(kq):.4f} ({steps:+d} grid steps from {KQ:.4f}); control k* {float(kc):.4f}"
            + ("   <-- >=2 steps: WARNING only if the other seed drifts the same way" if abs(steps) >= 2 else ""))
    say(f"== DONE qdre readout [{time.time() - t0:.0f}s]")
    rep.close()


if __name__ == "__main__":
    main()
