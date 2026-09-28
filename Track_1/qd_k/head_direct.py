"""[Fen copy of Bram's combo_perm20.py + DIRECT family, --K, p-value gate; Ilse 21:11Z]
K=20 PERMUTATION GATE (Devi; Lior gate 0-prime; Corin asked 20:11Z) on the MSE s1 err head.

Statistic T(hd) = the NESTED OOS pooled held diff (q AND k chosen on kept trajectories, held scored
once; q=0 = shipped real band) exactly as combo_nested_realband.py. T is computed for the real
head and for K=20 copies of the head permuted ACROSS WINDOWS (rng seeds 0..19; each copy keeps the
marginal and each field's spatial/frame structure, destroys alignment with that window's error).
Gate: real T > 95th percentile of the 20 permuted T (rank 21/21 <=> p <= 1/21). Reported as rank.
"""

import os
import sys

import numpy as np
import torch

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("arm"); _ap.add_argument("unet")
_ap.add_argument("--K", type=int, default=20, help="number of window permutations for the gate")
_ap.add_argument("--real-only", action="store_true")
_ap.add_argument("--family", choices=["driver", "direct", "both"], default="both")
_ap.add_argument("--direct-kmax", type=float, default=2.5,
                 help="extend the k grid UPWARD for the direct family only (superset of KGRID, same log step, 1.0 kept)")
_ap.add_argument("--alpha", type=float, default=0.80,
                 help="compose ALPHA. Default 0.80 = what SHIPS (r38). 0.60 only to reproduce the voided old reads.")
_ap.add_argument("--band", default=os.environ.get("RPDE_QDK_BAND"),
                 help="control band file (--base band). Build it in-repo, the r38 recipe on the same members: "
                      "fit_band.py --ckpt <6 all-data> --err-ckpt <5 twins> --alpha 0.80 --band-p 0.45 --fill-unscored "
                      "--centre-alpha 0 (REPRODUCE.md 10.3). The 09-24 run used r38's band, md5 02f473e9. "
                      "Not needed with --base recipe. k* does not depend on the control: it only sets the reported delta.")
_ap.add_argument("--pred-k", type=float, default=None,
                 help="predicted k* for this head (final set: MSE 4.0 | KL 2.1 | pinball 1.5 | QD 1.25)")
_ap.add_argument("--pred-k-calibration", action="store_true",
                 help="the prediction was CALIBRATED on this head's own measurement (MSE): label it so, not a self-check")
_ap.add_argument("--all-unseen", action="store_true",
                 help="Ilse check e: reporting cell = ALL 8 unseen VAL_TRAJ (leave-one-trajectory-out), not AoA>=15 only")
_ap.add_argument("--base", choices=["band", "recipe"], default="band",
                 help="control band: 'band' = --band file (r38, default); 'recipe' = base/sigma_ref refit on twin val_seen AoA 15/20")
_ap.add_argument("--fixed-k", type=float, default=None,
                 help="also report direct band at a FIXED k (nearest grid point, NOT nested) vs the same nested-k control")
_ap.add_argument("--csv", default=None, help="append REAL direct per-trajectory rows: seed,trajectory,delta_final,k")
_args = _ap.parse_args()
ARM, UNET, KPERM = _args.arm, _args.unet, _args.K
BRAMW = os.path.dirname(os.path.abspath(__file__))   # vendored: centre_band / sweep_guards / load_export live beside this file

sys.path.insert(0, BRAMW)
sys.path.insert(0, BRAMW)
sys.path.insert(0, UNET)
sys.path.insert(0, ".")
sys.path.insert(0, BRAMW)
from centre_band import SIGMA, compose, fit_base, sample_weights  # noqa: E402
from load_export import load  # noqa: E402
from sweep_guards import check_grid, check_not_edge  # noqa: E402
import models_alt  # noqa: E402
from model import from_ch  # noqa: E402

R = os.environ.get("RPDE_QDK_EXPORT")
if not R:
    sys.exit("set RPDE_QDK_EXPORT to the twin export dir (export_{seen,unseen}.npz from Track_1/qd_k/export.py; REPRODUCE.md 10.3)")
R = R.rstrip("/") + "/"
ALPHA, STRETCH, P_SHIP = _args.alpha, 1.39, 0.45   # Corin 23:54Z: ALPHA/band were hard-coded to the voided config
QGRID = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60])   # 0.0 == the shipped band exactly
KGRID = np.unique(np.concatenate([
    np.exp(np.linspace(np.log(0.4), np.log(2.5), 22)), [1.0]])).astype(np.float32)
check_grid(KGRID, 1.0, "deployed global scale")
check_grid(QGRID, 0.0, "q=0 must be in the grid: it IS the shipped band")

blob = torch.load(ARM, map_location="cpu", weights_only=False)
cfg = blob.get("config", {})
m = models_alt.build(blob["arch"], blob["width"], blob.get("var_head", True), 0,
                     blob.get("modes_y", 8), blob.get("modes_x", 12), blob.get("n_layers", 4),
                     dropout=cfg.get("dropout", 0.0), err_head=True)
m.load_state_dict(blob["model_state_dict"])
m.eval()
mean = torch.as_tensor(np.asarray(blob["mean"], np.float32))
std = torch.as_tensor(np.asarray(blob["std"], np.float32))
print(f"ARM {ARM}\n  epochs={cfg.get('epochs')} lam_e={cfg.get('lam_e')} seed={cfg.get('seed')}")

S, U = load(R + "export_seen.npz"), load(R + "export_unseen.npz")
aoa = lambda D: np.array([int(str(i).split(":")[0].split("_")[1]) for i in D["ids"]])  # noqa: E731
kS, kU = np.isin(aoa(S), [15, 20]), (aoa(U) >= 0 if _args.all_unseen else aoa(U) >= 15)
print(f"CELL {'ALL 8 unseen trajectories (--all-unseen)' if _args.all_unseen else 'AoA>=15 unseen (default)'}   "
      f"CONTROL BASE {_args.base}")
trajU = np.array([str(i).split(":")[0] for i in U["ids"]])[kU]

F0, F1, Ft = (S["y"] + S["e_a0"])[kS], (S["y"] + S["e_a1"])[kS], S["y"][kS]
af = sample_weights(compose(F0, F1, ALPHA), Ft)
sref = np.maximum(S["sig_full"][kS].mean(0, keepdims=True), 1e-8)
base = fit_base(F0, Ft, np.clip(S["sig_full"][kS] / sref, 0.3, 3.0), af, STRETCH,
                P_SHIP)[0].astype(np.float32)

# ---- REAL SHIPPED BAND (Ilse/Emre: md5 3327298f..., extracted from track1_f4v1a060.zip) ------------
# My recipe base differs from the shipped one by up to 24% relative on ~2/3 of u/v elements, and
# containment is the MOST base-shape-sensitive quantity. This rerun swaps in the shipped base and the
# shipped sigma_ref (u, v channels only; p is never scored) and changes nothing else.
import hashlib
if _args.base == "band":
    if not _args.band:
        sys.exit("--base band needs --band (or RPDE_QDK_BAND): the control band file; or use --base recipe")
    _Z = np.load(_args.band)
    print(f"BAND {_args.band}\n  md5 {hashlib.md5(open(_args.band, 'rb').read()).hexdigest()[:8]}   ALPHA {ALPHA}   STRETCH {STRETCH}   p {P_SHIP}")
else:
    _Z = {"p": np.float32(P_SHIP), "centre_alpha": np.float32(0.0)}   # the recipe control's own conventions
    print(f"BAND (none: --base recipe, control refit from the twin export)   ALPHA {ALPHA}   STRETCH {STRETCH}   p {P_SHIP}")
print(f"recipe base median {float(np.median(base)):.6f}  recipe sigma_ref median {float(np.median(sref)):.6f}")
if _args.base == "band":
    base = _Z["base"][0][..., :2].astype(np.float32)
    sref = np.maximum(_Z["sigma_ref"][..., :2].astype(np.float32), 1e-8)
else:
    print("  --base recipe: control uses the RECIPE base/sigma_ref above, NOT the band file")
print(f"CONTROL base median {float(np.median(base)):.6f}  SHIPPED sigma_ref median {float(np.median(sref)):.6f}")
assert float(_Z["p"]) == np.float32(0.45) and float(_Z["centre_alpha"]) == 0.0


E0 = (U["y"] + U["e_a0"])[kU].astype(np.float32)
pe = compose(U["y"] + U["e_a0"], U["y"] + U["e_a1"], ALPHA)[kU].astype(np.float32)
Et = U["y"][kU]
t = np.where(Et == 0, 0, pe + STRETCH * (Et - pe)).astype(np.float32)
X, D = (t - E0).astype(np.float32), (pe - E0).astype(np.float32)
sc = t != 0
a = sample_weights(compose(U["y"] + U["e_a0"], U["y"] + U["e_a1"], ALPHA), U["y"])[kU].astype(np.float64)
n, nsc, SIG = len(t), int(sc.sum()), np.float32(SIGMA)


def sps_sub(h, sel):
    lo, up = np.minimum(-h, D), np.maximum(h, D)
    v = np.where((X >= lo) & (X <= up) & sc, np.exp(-(up - lo) / SIG), np.float32(0))
    vv = v.reshape(n, -1).sum(1, dtype=np.float64)
    return 100.0 * float(a[sel] @ vv[sel]) / max(int(sc[sel].sum()), 1)


ALL = np.ones(n, bool)
with torch.no_grad():
    out = []
    for i in range(0, int(kU.sum()), 8):
        xb = torch.from_numpy(U["x"][kU][i:i + 8].astype(np.float32))
        xb = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), 40, 32, 64)
        out.append(from_ch(m(xb, detach_var=False)[2]).numpy())
hd = np.exp(np.concatenate(out, 0).astype(np.float32))
rat = np.ascontiguousarray(np.broadcast_to(np.clip(U["sig_full"] / sref, 0.3, 3.0)[kU], X.shape))
ship = base[None] * rat ** np.float32(P_SHIP)          # the shipped band, exactly
medh = np.float32(np.median(hd[sc]))
W = 0.249752
print(f"\nreporting cell: {n} windows, {len(np.unique(trajU))} trajectories")


def vv_of(u):
    lo, up = np.minimum(-u, D), np.maximum(u, D)
    v = np.where((X >= lo) & (X <= up) & sc, np.exp(-(up - lo) / SIG), np.float32(0))
    return v.reshape(n, -1).sum(1, dtype=np.float64)


NSC = sc.reshape(n, -1).sum(1)
TRAJS = np.unique(trajU)


def T(h, tag):
    """Same numbers as the per-fold sps_sub loop: per-window sums vv are computed ONCE per (q, k)
    over all windows (they do not depend on the fold), then each fold's kept/held score is
    100 * a[sel] @ vv[sel] / sc[sel].sum() exactly as sps_sub does."""
    mh = np.float32(np.median(h[sc]))
    tab = {}
    for q in QGRID:
        uq = ship * (h / mh) ** np.float32(q)
        for k in KGRID:
            tab[(float(q), float(k))] = vv_of(uq * np.float32(k))
    s = lambda vv, sel: 100.0 * float(a[sel] @ vv[sel]) / max(int(NSC[sel].sum()), 1)  # noqa: E731
    num, tot, qs, folds = 0.0, 0, [], []
    for held in TRAJS:
        keep = trajU != held
        best = None
        for q in QGRID:
            sk, kk = max((s(tab[(float(q), float(k))], keep), float(k)) for k in KGRID)
            if best is None or sk > best[0]:
                best = (sk, float(q), kk)
        _, qsel, ksel = best
        h_sel = s(tab[(qsel, ksel)], ~keep)
        k0 = max((s(tab[(0.0, float(k))], keep), float(k)) for k in KGRID)[1]
        h_0 = s(tab[(0.0, k0)], ~keep)
        nh = int(sc[~keep].sum())
        num += (h_sel - h_0) * nh
        tot += nh
        qs.append(qsel)
        folds.append((str(held), h_sel - h_0, nh))
    d = num / tot
    if tag == "REAL":
        # Corin's rule (20:51Z, fixed before pinball lands): report each fold, and the pooled diff
        # with the LARGEST-contributing fold removed -- a +0.03 that lives in one fold does not pass.
        for f, df, nh in folds:
            print(f"    fold {f:10s} diff {df:+.4f} sps  n_scored {nh}  contribution {df * nh / tot:+.4f}")
        big = max(folds, key=lambda r: r[1] * r[2])
        rest = [r for r in folds if r is not big]
        dr = sum(r[1] * r[2] for r in rest) / sum(r[2] for r in rest)
        print(f"    drop largest-contribution fold ({big[0]}): {dr:+.4f} sps = {W * dr:+.5f} final")
        # Corin's fixed gates (20:54Z). NOTE: T is the BAND-SIDE part only (head on top of the shipped
        # band). pinball - no-head in final ALSO includes the arm's own field change vs its control,
        # measured by aux_price.py; add that before judging gate 1 and gate 2.
        print(f"    gate 2 band-side (drop-largest >= +0.015 final): {W * dr:+.5f}  "
              f"{'PASS' if W * dr >= 0.015 else 'FAIL'}   [add aux_price field delta before judging]")
        print(f"    gate 3 (each remaining fold > 0): {[f'{r[1]:+.4f}' for r in rest]}  "
              f"{'PASS' if all(r[1] > 0 for r in rest) else 'FAIL'}")
    print(f"  {tag:10s} T = {d:+.4f} sps = {W * d:+.5f} final   q per fold {qs}", flush=True)
    return d



# ======================= Fen: DIRECT family (Ilse 21:11Z) ==================================
# The head outputs log|e| in NORMALISED units (loss sees |pred_norm - y_norm|). As a direct half-width
# it must be in the physical units of X, so multiply by the checkpoint's per-channel std. Probe
# (probe_units.py, MSE s1): median |E|/(hd) = 0.111 (u) / 0.020 (v) raw vs std 0.0968 / 0.0160 => after
# the conversion H is within 1.1-1.25x of the median physical error on both channels. Without it the v
# channel would need k ~ 0.02, far below KGRID, and the direct column would fail on UNITS, not merit.
STDC = np.asarray(blob["std"], np.float32).reshape(-1)[: hd.shape[-1]]
hd_phys = (hd * STDC).astype(np.float32)
print(f"direct family: H converted to physical units with std {STDC.tolist()}")


_step = float(np.log(KGRID[1] / KGRID[0]))
if _args.direct_kmax > float(KGRID[-1]) + 1e-6:
    _ext = np.exp(np.arange(np.log(float(KGRID[-1])) + _step, np.log(_args.direct_kmax) + 1e-9, _step))
    KGRID_D = np.unique(np.concatenate([KGRID, _ext])).astype(np.float32)
else:
    KGRID_D = KGRID
check_grid(KGRID_D, 1.0, "direct family: 1.0 must stay an exact key")
assert set(np.round(KGRID, 6)).issubset(set(np.round(KGRID_D, 6))), "direct grid must be a SUPERSET of KGRID"
print(f"direct k grid: {len(KGRID_D)} points, [{KGRID_D[0]:.3f}, {KGRID_D[-1]:.3f}] (spec KGRID ends at {KGRID[-1]:.3f})")
_TAB0 = None


def T_direct(h, tag):
    """h = k * H (H already physical), ONLY k scanned (same KGRID, 1.0 exact). k picked on the kept
    trajectories, the held one scored once. Baseline = the shipped band with its own nested k
    (identical to T's q=0 baseline), so driver and direct columns share one reference."""
    global _TAB0
    tabd = {float(k): vv_of(h * np.float32(k)) for k in KGRID_D}
    if _TAB0 is None:                       # shipped-band baseline: same for every permutation, compute once
        _TAB0 = {float(k): vv_of(ship * np.float32(k)) for k in KGRID}
    tab0 = _TAB0
    s = lambda vv, sel: 100.0 * float(a[sel] @ vv[sel]) / max(int(NSC[sel].sum()), 1)  # noqa: E731
    num, tot, ks, folds = 0.0, 0, [], []
    vd_all, v0_all = np.zeros(n), np.zeros(n)      # held-window sums under the selected k (each window held once)
    for held in TRAJS:
        keep = trajU != held
        ksel = max((s(tabd[float(k)], keep), float(k)) for k in KGRID_D)[1]
        k0 = max((s(tab0[float(k)], keep), float(k)) for k in KGRID)[1]
        vd_all[~keep], v0_all[~keep] = tabd[ksel][~keep], tab0[k0][~keep]
        dsel = s(tabd[ksel], ~keep) - s(tab0[k0], ~keep)
        nh = int(sc[~keep].sum())
        num += dsel * nh; tot += nh; ks.append(ksel); folds.append((str(held), dsel, nh))
        if tag == "REAL":
            print(f"    [recon] held {held}: shipped k0 {k0:.4f}  QD k* {ksel:.4f}  sps held direct {s(tabd[ksel], ~keep):.4f}"
                  f"  shipped {s(tab0[k0], ~keep):.4f}  (shipped at k=1: {s(tab0[1.0], ~keep):.4f})")
    d = num / tot
    if tag == "REAL" and _args.fixed_k is not None:
        kf = float(KGRID_D[np.argmin(np.abs(np.log(KGRID_D / _args.fixed_k)))])
        fnum, frows = 0.0, []
        for held in TRAJS:
            keep = trajU != held
            k0 = max((s(tab0[float(k)], keep), float(k)) for k in KGRID)[1]
            df = s(tabd[kf], ~keep) - s(tab0[k0], ~keep); nh = int(sc[~keep].sum())
            fnum += df * nh; frows.append((str(held), df))
        print(f"    FIXED k {kf:.4f} (requested {_args.fixed_k}, nearest grid point; NOT nested; control = shipped band nested k):"
              f"  pooled {fnum / tot:+.4f} sps = {W * fnum / tot:+.5f} final  |  per fold sps "
              + " ".join(f"{f} {x:+.4f}" for f, x in frows))
        if _args.csv:
            with open(_args.csv.replace(".csv", "_fixedk.csv"), "a") as fh:
                for f, x in frows:
                    fh.write(f"{cfg.get('seed')},{f},{W * x:.6f},{kf:.4f}\n")
    if tag == "REAL" and _args.csv:
        with open(_args.csv, "a") as fh:
            for (f, df, nh), kk in zip(folds, ks):
                fh.write(f"{cfg.get('seed')},{f},{W * df:.6f},{kk:.4f}\n")
    if tag == "REAL":
        for f, df, nh in folds:
            print(f"    fold {f:10s} diff {df:+.4f} sps  n_scored {nh}  contribution {df * nh / tot:+.4f}")
        big = max(folds, key=lambda r: r[1] * r[2]); rest = [r for r in folds if r is not big]
        dr = sum(r[1] * r[2] for r in rest) / sum(r[2] for r in rest)
        print(f"    drop largest-contribution fold ({big[0]}): {dr:+.4f} sps = {W * dr:+.5f} final")
        print(f"    gate 2 band-side (drop-largest >= +0.015 final): {W * dr:+.5f}  "
              f"{'PASS' if W * dr >= 0.015 else 'FAIL'}   [add aux_price field delta before judging]")
        print(f"    gate 3 (each remaining fold > 0): {[f'{r[1]:+.4f}' for r in rest]}  "
              f"{'PASS' if all(r[1] > 0 for r in rest) else 'FAIL'}")
        # Ilse 22:3xZ: containment C and a_bar columns, DIAGNOSTIC ONLY (judge in final units).
        # Band-side only: the predicted field is the same for both bands, so a_bar is IDENTICAL by
        # construction and dC = dsps/(100*a_bar). The arm's own field change (aux_price) is not in here.
        abar = float(a.mean()); NT = float(NSC.sum())
        sps_d = 100.0 * float(a @ vd_all) / NT; sps_0 = 100.0 * float(a @ v0_all) / NT
        C_d, C_0 = sps_d / (100 * abar), sps_0 / (100 * abar)
        print(f"    [diagnostic] a_bar {abar:.5f} (identical for both bands: same field)  |  "
              f"C direct {C_d:.5f}  C shipped {C_0:.5f}  dC {C_d - C_0:+.5f}")
        edge = [k for k in ks if k in (float(KGRID_D[0]), float(KGRID_D[-1]))]
        print(f"    **k* per fold (Jonas self-check; QD should be ~1): {[round(k, 3) for k in ks]}**"
              f"{'   <-- GRID EDGE: k* pinned, reading not trustworthy' if edge else ''}")
        if _args.pred_k is not None:
            rat = [k / _args.pred_k for k in ks]
            off = [r for r in rat if not (0.7 <= r <= 1.4)]
            _lab = "CALIBRATION POINT (not a test)" if _args.pred_k_calibration else "PRE-REGISTERED (self-check)"
            print(f"    **k* {_lab} {_args.pred_k:.3g}  vs  MEASURED {[round(k, 3) for k in ks]}"
                  f"  ratio {[round(r, 2) for r in rat]}**"
                  f"{'   <-- OUTSIDE 0.7-1.4x: CHECK UNITS FIRST (std conversion / loss mul, sigma_g) before any conclusion' if off else '   (within 0.7-1.4x)'}")
    print(f"  {tag:10s} T_direct = {d:+.4f} sps = {W * d:+.5f} final   k* per fold {[round(k, 3) for k in ks]}", flush=True)
    return d


FAMS = {"driver": lambda h, tag: T(h, tag), "direct": T_direct}
use = ["driver", "direct"] if _args.family == "both" else [_args.family]
real = {}
for f in use:
    print(f"\n######## FAMILY {f.upper()} — REAL head")
    real[f] = FAMS[f](hd if f == "driver" else hd_phys, "REAL")
if _args.real_only:
    sys.exit(0)
ALPHA_P = 0.05 if KPERM < 100 else 0.01
for f in use:
    print(f"\n######## FAMILY {f.upper()} — permutation null, K={KPERM}")
    null = []
    for s_ in range(KPERM):
        p = np.random.default_rng(s_).permutation(n)
        assert (p != np.arange(n)).mean() > 0.9
        hh = hd[p] if f == "driver" else hd_phys[p]
        null.append(FAMS[f](hh, f"perm s{s_}"))
    null = np.array(null)
    pval = (1 + int((null >= real[f]).sum())) / (KPERM + 1)
    rank = 1 + int((null < real[f]).sum())
    p5, p50, p95 = np.percentile(null, [5, 50, 95])
    print(f"  null (final): p5 {W*p5:+.5f}  p50 {W*p50:+.5f}  p95 {W*p95:+.5f}  sd {W*null.std(ddof=1):.5f}")
    print(f"  REAL {W*real[f]:+.5f} final; rank {rank}/{KPERM + 1}; p = {pval:.4f}; "
          f"GATE 0-prime (p <= {ALPHA_P}): {'PASS' if pval <= ALPHA_P else 'FAIL'}"
          f"{'   -> rerun with --K 100 (Corin: p <= 0.01 required for any claimed pass)' if (pval <= ALPHA_P and KPERM < 100) else ''}")
print("== DONE head_direct ==")
