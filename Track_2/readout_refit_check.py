"""Offline gate for a closed-form Bayesian readout refit on Track 2 (an ALPaCA-style last layer).

    python readout_refit_check.py --ens auto --check-stream        # the job: twin, then 3 twins if positive
    python readout_refit_check.py --device cpu --max-traj 1 --max-windows 3 --bench-reps 3 --ens on

WHAT IS NEW RELATIVE TO THE CLOSED NULLS (`README.md`, "What is closed").
  * `ttt_offline.py` corrected the OUTPUT from lagged residuals, feature-blind and additive. Its
    oracle, a per-trajectory mean residual field, removes 1.75 % of rel-L2.
  * `ttt_grad.py` took SGD/Adam steps on parameter subsets through a backward pass: -1.6 % rel-L2
    on a single member for ~9.7 ms/step.
  * This file SOLVES the last linear map of `head_field` (and of `head_logstd`) in closed form from
    the revealed pair's per-pixel 128-d penultimate features, with no backward pass. The solution is
    the ridge estimate, i.e. the posterior mean under an isotropic Gaussian prior centred on the
    trained weights. It is the un-meta-trained ceiling check for the layer proposed in
    `Track_1/plan/lit_backbones_20260914.md`, C1.

PROTOCOL.
  * Members: held-out twins only. Each checkpoint's config must read all_data False and exclude
    None, which is how `train2.py` holds out `model.VAL_TRAJ`.
  * Windows: `Track_1/val_unseen.npz`, the set `ens_v9.py` / `compare_v9.py` score, re-ordered to
    the evaluator's (filename, time_id) order. Its trajectory names are asserted equal to
    `model.VAL_TRAJ`, and the stride-20 tiling (window j's target IS window j+1's input) is
    asserted bit for bit. `--check-stream` also matches the first trajectory against
    `t2_common.build_stream`.
  * Causality: window j is corrected only from windows < j of its own trajectory; the statistics
    reset at every trajectory boundary, where the platform calls `reset_ttt_state`.
  * Frozen path: compose (ALPHA 0.60), member averaging and denormalization are
    `_fno.predict_fno`'s, with TF32 OFF as in both shipped `submission.py` files. The batch-1 frozen
    arm is checked against `predict_fno` itself at batch 1 (and, for information, at batch 64) on
    the fields and on the official metrics. Job 7281131 aborted on a field-level max |diff| of
    2.07e-3 with cuDNN's default TF32 ON; the same check read 4.5e-7 on CPU.
  * Scoring: the official kit's per-sample rel / TKE / MVPE on windows j >= 1 (every window with a
    revealed predecessor, i.e. the brief's 1-based k >= 2), pooled and per AoA.

ARMS: readout x memory x prior strength.
  field   ridge on head_field's last 1x1 conv (129 -> 40 per pixel, bias included); target y
  sigma   ridge on head_logstd's last 1x1 conv (129 -> 2); target log(std_t(y) + 1e-4), which is
          train2.py's own log-std target
  both    field and sigma together
  affine  per-output-channel gain and bias on the raw field (2 -> 1 per channel): the low-rank control
  last    statistics from window j-1 only.   cum  every revealed window of the trajectory (RLS).
  rho     lambda = rho * trace(G) / (n_windows * d): a prior of fixed strength per window of data.
Rows are pixels whose revealed target is not identically zero. Corrections are applied on the same
pixel set (the field of view does not move within a trajectory). Every arm, including an in-batch
zero-correction copy of the frozen arm, is composed in ONE batch, and the percentages are taken
against that copy, so every arm shares the same numerics.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

# Both shipped submission.py files turn TF32 OFF (RPDE_TF32=1 opts in, for drivers only). cuDNN's
# own default is ON, and batch-size-dependent TF32 kernels are what made job 7281131's frozen arm
# differ from predict_fno by 2.07e-3 at one field element.
_TF32 = os.environ.get("RPDE_TF32", "") not in ("", "0")
torch.backends.cuda.matmul.allow_tf32 = _TF32
torch.backends.cudnn.allow_tf32 = _TF32

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T1 = ROOT / "Track_1"
U = T1 / "unet"
sys.path.insert(0, str(T1 / "submit_fno"))
import _fno  # noqa: E402


def _load_module(name: str, path) -> object:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
sc = _load_module("sc9", os.path.join(KIT, "scoring.py"))

T, H, W, C = _fno.T_OUT, _fno.H, _fno.W, _fno.C
CH = T * C
SINGLE = [str(U / "fno_r18_cmb3sim.pth")]
TWINS = [str(U / f"fno_r18_cmb3sim{s}.pth") for s in ("", "_s2", "_s3")]
HEADS = ("field", "sigma", "both", "affine")
MEMS = ("last", "cum")
METRICS = ("rel", "tke", "mvpe")
EXCH = {"rel": 0.346, "tke": 0.290, "mvpe": 0.100}   # CLAUDE.md: +final per 10 % of error removed
BOARD_PER_A100_MS = 1.850 * 0.60                     # A100 -> 4060-eq -> Track-2 board ms (one reading)
FINAL_PER_BOARD_MS_T2 = -0.056                       # plan_20260914.md section 2, at ~7 ms/step


# --------------------------------------------------------------------------------------------- data
def load_heldout(max_traj=None, max_windows=None):
    """val_unseen.npz in evaluator order, grouped by trajectory, with the Track-2 protocol asserted."""
    z = np.load(T1 / "val_unseen.npz")
    ids = [str(i) for i in z["ids"]]
    zx, zy = z["x"], z["y"]                               # decompress each member ONCE
    names = sorted({s.split(":")[0] for s in ids})
    unet_model = _load_module("unet_model", U / "model.py")
    if names != sorted(unet_model.VAL_TRAJ):
        raise SystemExit(f"val_unseen trajectories {names} != model.VAL_TRAJ")
    X, Y, sel, groups = [], [], [], []
    for name in names[: max_traj or len(names)]:
        idx = sorted((i for i, s in enumerate(ids) if s.split(":")[0] == name),
                     key=lambda i: int(ids[i].split(":")[1]))
        ts = [int(ids[i].split(":")[1]) for i in idx]
        if ts[0] != 0 or any(b - a != 20 for a, b in zip(ts, ts[1:])):
            raise SystemExit(f"{name}: time ids do not start at 0 and tile at stride 20")
        g = []
        for i in idx[: max_windows or len(idx)]:
            g.append(len(X))
            X.append(zx[i, ..., :2]); Y.append(zy[i, ..., :2]); sel.append(ids[i])
        groups.append(g)
    X = np.ascontiguousarray(np.stack(X).astype(np.float32))
    Y = np.ascontiguousarray(np.stack(Y).astype(np.float32))
    for g in groups:
        for a, b in zip(g, g[1:]):
            if not np.array_equal(Y[a], X[b]):
                raise SystemExit(f"{sel[a]}: target != next input -- not the Track-2 tiling")
    return X, Y, sel, groups


def check_stream(X, Y, ids, groups) -> str:
    t2 = _load_module("t2_common", HERE / "t2_common.py")
    g = groups[0]
    name = ids[g[0]].split(":")[0]
    st = t2.build_stream([name])
    for k, i in enumerate(g):
        nm, t, x, y, first = st[k]
        if f"{nm}:{t}" != ids[i] or first != (k == 0):
            raise SystemExit(f"stream order mismatch at {ids[i]}")
        if not (np.array_equal(x[..., :2], X[i]) and np.array_equal(y[..., :2], Y[i])):
            raise SystemExit(f"stream window mismatch at {ids[i]}")
    return f"{name}: {len(g)} windows bit-identical to t2_common.build_stream"


# ------------------------------------------------------------------------------------------ members
class Tapped:
    """A frozen member with both penultimate readout activations exposed by forward hooks, so the
    features are the shipped forward's own numbers, not a re-implementation of it."""

    def __init__(self, path: str, dev: str):
        cfg = torch.load(path, map_location="cpu")["config"]
        if cfg.get("all_data") or cfg.get("exclude") is not None:
            raise SystemExit(f"{path}: not a VAL_TRAJ held-out twin "
                             f"(all_data={cfg.get('all_data')}, exclude={cfg.get('exclude')})")
        self.path = path
        self.m = _fno.load(path, dev)
        m = self.m
        if not (isinstance(m.head_field, torch.nn.Sequential)
                and isinstance(m.head_logstd, torch.nn.Sequential)):
            raise SystemExit(f"{path}: expected MLP readouts on both heads (--var-mlp)")
        if m.head_osc is not None or m.head_tb is not None or m.head_mean is not None:
            raise SystemExit(f"{path}: an additive head would bypass the refit readout")
        self.pf = self.ps = None
        m.head_field[1].register_forward_hook(lambda mod, i, o: setattr(self, "pf", o))
        m.head_logstd[1].register_forward_hook(lambda mod, i, o: setattr(self, "ps", o))

    @torch.no_grad()
    def __call__(self, prev):
        raw, ls = self.m(prev)
        return raw, ls, self.pf, self.ps


def to_prev(xw, mean, std):
    """(B,T,H,W,2) physical -> (B,40,H,W) normalized, exactly predict_fno's layout."""
    return ((xw - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xw), CH, H, W)


def compose_batch(raws, lss, alpha, mean, std):
    """raws (A,M,40,H,W), lss (A,M,2,H,W) -> (A,T,H,W,2) physical, as predict_fno composes."""
    a, m = raws.shape[:2]
    y = raws.reshape(a, m, T, C, H, W).permute(0, 1, 2, 4, 5, 3)
    sig = lss.clamp(-8.0, 2.0).exp().permute(0, 1, 3, 4, 2).unsqueeze(2)
    if m > 1:
        y, sig = y.sum(1) / m, sig.sum(1) / m
    else:
        y, sig = y[:, 0], sig[:, 0]
    m_ = y.mean(1, keepdim=True)
    f_ = y - m_
    s_ = y.std(1, keepdim=True, unbiased=False)
    y = m_ + f_ * ((alpha * sig + (1 - alpha) * s_) / (s_ + 1e-6))
    return y * std + mean


# --------------------------------------------------------------------------------- the closed form
def feat(p, dtype=torch.float64):
    """(1,128,H,W) activation -> (HW,129) rows with a bias column."""
    f = p[0].to(dtype).reshape(p.shape[1], -1)
    return torch.cat([f, torch.ones(1, f.shape[1], dtype=dtype, device=f.device)], 0).T


def window_stats(raw, ls, pf, ps, yw, mean, std):
    """Sufficient statistics (G, C) of the three readouts from ONE revealed window."""
    valid = (yw != 0).any(dim=4).any(dim=1)[0]                      # (H,W)
    v = valid.reshape(-1, 1).double()
    yn = (yw - mean) / std                                          # (1,T,H,W,2)
    yn_ch = yn.permute(0, 1, 4, 2, 3).reshape(CH, H, W)
    tgt_s = torch.log(yn.std(1, unbiased=False)[0].permute(2, 0, 1) + 1e-4)   # (2,H,W)
    rf = (yn_ch - raw[0]).double().reshape(CH, -1)                  # (40,HW)
    rs = (tgt_s - ls[0]).double().reshape(C, -1)                    # (2,HW)
    Pf, Ps = feat(pf) * v, feat(ps) * v
    out = {"field": (Pf.T @ Pf, Pf.T @ (rf.T * v)),
           "sigma": (Ps.T @ Ps, Ps.T @ (rs.T * v))}
    A = torch.stack([raw[0].double().reshape(CH, -1),
                     torch.ones(CH, H * W, dtype=torch.float64, device=raw.device)], -1)
    A = A * v.reshape(1, -1, 1)                                     # (40,HW,2)
    out["affine"] = (A.transpose(1, 2) @ A, A.transpose(1, 2) @ (rf * v.T).unsqueeze(-1))
    return out, valid


def ridge(G, Cm, n, rho):
    d = G.shape[-1]
    lam = rho * torch.diagonal(G, dim1=-2, dim2=-1).sum(-1) / (max(n, 1) * d)
    A = G + lam[..., None, None] * torch.eye(d, dtype=G.dtype, device=G.device)
    L, info = torch.linalg.cholesky_ex(A)
    if bool((info != 0).any()):
        L = torch.linalg.cholesky(A + 1e-8 * torch.eye(d, dtype=G.dtype, device=G.device))
    return torch.cholesky_solve(Cm, L)


def corrections(st, raw, pf, ps, mask, mem, rho):
    """Readout corrections for the CURRENT window from statistics of earlier windows only."""
    m = mask.reshape(-1, 1).double()
    gf, cf, n = st[("field", mem)]
    cor_f = ((feat(pf) @ ridge(gf, cf, n, rho)) * m).T.reshape(CH, H, W).float()
    gs, cs, n = st[("sigma", mem)]
    cor_s = ((feat(ps) @ ridge(gs, cs, n, rho)) * m).T.reshape(C, H, W).float()
    ga, ca, n = st[("affine", mem)]
    th = ridge(ga, ca, n, rho)[..., 0]                              # (40,2)
    r = raw[0].double().reshape(CH, -1)
    cor_a = ((th[:, :1] * r + th[:, 1:]) * m.T).reshape(CH, H, W).float()
    return cor_f, cor_s, cor_a


# ----------------------------------------------------------------------------------------- stream
def arm_names(rhos):
    return ["frozen"] + [f"{h}/{m}/rho={r:g}" for h in HEADS for m in MEMS for r in rhos]


def _metrics(P, Yw):
    return (sc.rel_l2_per_sample(P, Yw, 2), sc.tke_rel_l2_per_sample(P, Yw, 2),
            sc.mvpe_rel_l2_per_sample(P, Yw))


def run_stream(members, X, Y, ids, groups, dev, alpha, rhos):
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    std = torch.from_numpy(_fno.STD).to(dev)
    names = arm_names(rhos)
    A, M, N = len(names), len(members), len(X)
    res = {k: np.full((A, N), np.nan, dtype=np.float64) for k in METRICS}
    res_b1 = {k: np.full(N, np.nan, dtype=np.float64) for k in METRICS}
    jpos = np.zeros(N, dtype=int)
    P_frozen = np.empty((N, T, H, W, C), dtype=np.float32)
    first = True
    for g in groups:
        st = [None] * M
        mask = None
        for j, idx in enumerate(g):
            jpos[idx] = j
            xw = torch.from_numpy(X[idx:idx + 1]).to(dev)
            yw = torch.from_numpy(Y[idx:idx + 1]).to(dev)
            prev = to_prev(xw, mean, std)
            outs = [mb(prev) for mb in members]
            if first:
                raw, ls, pf, ps = outs[0]                         # the hooks ARE the readout's input
                assert torch.equal(members[0].m.head_field[2](pf), raw)
                assert torch.equal(members[0].m.head_logstd[2](ps), ls)
                first = False
            na = A if j >= 1 else 1
            raws = torch.empty(na, M, CH, H, W, device=dev)
            lss = torch.empty(na, M, C, H, W, device=dev)
            for i, (raw, ls, pf, ps) in enumerate(outs):
                raws[:, i] = raw[0]
                lss[:, i] = ls[0]
                if j == 0:
                    continue
                cache = {(mem, rho): corrections(st[i], raw, pf, ps, mask, mem, rho)
                         for mem in MEMS for rho in rhos}
                a = 1
                for h in HEADS:
                    for mem in MEMS:
                        for rho in rhos:
                            cf, cs, ca = cache[(mem, rho)]
                            if h in ("field", "both"):
                                raws[a, i] += cf
                            if h in ("sigma", "both"):
                                lss[a, i] += cs
                            if h == "affine":
                                raws[a, i] += ca
                            a += 1
            Yw1 = Y[idx:idx + 1]
            # The frozen arm ALONE at batch 1 -- predict_fno's shapes -- is what the equivalence
            # check reads. The arms, including an in-batch copy of the frozen arm, share one batch.
            P1 = compose_batch(raws[:1].contiguous(), lss[:1].contiguous(),
                               alpha, mean, std).float().cpu().numpy()
            P_frozen[idx] = P1[0]
            for k, v in zip(METRICS, _metrics(P1, Yw1)):
                res_b1[k][idx] = v[0]
            P = compose_batch(raws, lss, alpha, mean, std).float().cpu().numpy()
            for k, v in zip(METRICS, _metrics(P, np.broadcast_to(Yw1, P.shape))):
                res[k][:na, idx] = v
            # reveal window j's target: update every member's statistics for window j+1
            for i, (raw, ls, pf, ps) in enumerate(outs):
                s, valid = window_stats(raw, ls, pf, ps, yw, mean, std)
                if st[i] is None:
                    st[i] = {}
                    for key, (G, Cm) in s.items():
                        st[i][(key, "last")] = (G, Cm, 1)
                        st[i][(key, "cum")] = (G.clone(), Cm.clone(), 1)
                else:
                    for key, (G, Cm) in s.items():
                        st[i][(key, "last")] = (G, Cm, 1)
                        g0, c0, n0 = st[i][(key, "cum")]
                        st[i][(key, "cum")] = (g0 + G, c0 + Cm, n0 + 1)
                mask = valid
    return names, res, res_b1, jpos, P_frozen


def equivalence(ref, P, Y, ids):
    """Field- and metric-level agreement of the batch-1 frozen arm with predict_fno's output."""
    d = np.abs(ref - P)
    n, t, y, x, c = np.unravel_index(int(np.argmax(d)), d.shape)
    info = {"max_abs": float(d.max()), "p99.99_abs": float(np.quantile(d, 0.9999)),
            "frac_gt_1e-4": float((d > 1e-4).mean()),
            "argmax": f"{ids[n]} t={t} y={y} x={x} c={c}",
            "argmax_pred_temporal_std": float(P[n, :, y, x, c].std()),
            "argmax_target_identically_zero": bool((Y[n, :, y, x, c] == 0).all())}
    for k, a_, b_ in zip(METRICS, _metrics(ref, Y), _metrics(P, Y)):
        info[f"{k}_pooled_reldiff"] = float(abs(a_.mean() - b_.mean()) / max(float(b_.mean()), 1e-12))
        info[f"{k}_max_window_reldiff"] = float(np.max(np.abs(a_ - b_) / np.maximum(b_, 1e-12)))
    return info


# ---------------------------------------------------------------------------------------- report
def aoa_of(window_id: str) -> int:
    """'11400_20:40' (or a bare trajectory name) -> 20."""
    return int(window_id.split(":")[0].split("_")[1])


def summarize(names, res, jpos, ids, tag):
    aoa = np.array([aoa_of(s) for s in ids])
    subsets = {"pooled": jpos >= 1, "pooled_j>=2": jpos >= 2, "aoa15-20": (jpos >= 1) & (aoa >= 15)}
    for a_ in sorted(set(aoa.tolist())):
        subsets[f"aoa{a_}"] = (jpos >= 1) & (aoa == a_)
    table = {}
    for sub, sel in subsets.items():
        if not sel.any():
            continue
        base = {k: float(res[k][0, sel].mean()) for k in METRICS}
        rows = {}
        for a, nm in enumerate(names):
            err = {k: float(res[k][a, sel].mean()) for k in METRICS}
            pct = {k: 100.0 * (base[k] - err[k]) / base[k] for k in METRICS}
            rows[nm] = {"err": err, "pct": pct, "n": int(sel.sum()),
                        "dfinal_acc": sum(EXCH[k] * pct[k] / 10.0 for k in METRICS)}
        table[sub] = rows
    pooled = table["pooled"]
    best = max(names[1:], key=lambda nm: pooled[nm]["dfinal_acc"])
    print(f"\n=== {tag}: {int(subsets['pooled'].sum())} scored windows (j >= 1), "
          f"{len(set(s.split(':')[0] for s in ids))} trajectories ===")
    fz = pooled["frozen"]["err"]
    print(f"frozen (j>=1)  rel {fz['rel']:.5f}  tke {fz['tke']:.5f}  mvpe {fz['mvpe']:.5f}")
    hdr = (f"{'arm':26s} | {'rel':>8s} {'tke':>8s} {'mvpe':>8s} | {'%rel':>6s} {'%tke':>6s} "
           f"{'%mvpe':>6s} | {'15-20 %rel':>10s} {'%tke':>6s} | {'dfin_acc':>8s}")
    print(hdr)
    print("-" * len(hdr))
    hi = table.get("aoa15-20", pooled)
    for nm in names:
        r, q = pooled[nm], hi[nm]
        print(f"{nm:26s} | {r['err']['rel']:8.5f} {r['err']['tke']:8.5f} {r['err']['mvpe']:8.5f} | "
              f"{r['pct']['rel']:+6.2f} {r['pct']['tke']:+6.2f} {r['pct']['mvpe']:+6.2f} | "
              f"{q['pct']['rel']:+10.2f} {q['pct']['tke']:+6.2f} | {r['dfinal_acc']:+8.3f}")
    print(f"\nbest arm by pooled accuracy-only dfinal: {best}")
    for sub in [s for s in table if s.startswith("aoa") or s == "pooled_j>=2"]:
        r, f0 = table[sub][best], table[sub]["frozen"]
        print(f"  {sub:12s} n={r['n']:3d}  rel {f0['err']['rel']:.5f}->{r['err']['rel']:.5f} "
              f"({r['pct']['rel']:+.2f}%)  tke {f0['err']['tke']:.5f}->{r['err']['tke']:.5f} "
              f"({r['pct']['tke']:+.2f}%)  mvpe {f0['err']['mvpe']:.5f}->{r['err']['mvpe']:.5f} "
              f"({r['pct']['mvpe']:+.2f}%)")
    for h in HEADS:
        bh = max((nm for nm in names[1:] if nm.startswith(h + "/")),
                 key=lambda nm: pooled[nm]["dfinal_acc"])
        print(f"  best {h:6s}: {bh:26s} pooled %rel {pooled[bh]['pct']['rel']:+.2f} "
              f"%tke {pooled[bh]['pct']['tke']:+.2f} %mvpe {pooled[bh]['pct']['mvpe']:+.2f}")
    return table, best


# ----------------------------------------------------------------------------------------- timing
def bench(members, X, Y, dev, reps, rho):
    """Batch-1 wall time of ONE closed-form update + application, eager, synchronized.

    One step = build (G, C) from the revealed window, Cholesky-solve, apply the correction to the
    next window's features. The two forwards are the frozen path's and are not charged here.
    """
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    std = torch.from_numpy(_fno.STD).to(dev)
    cuda = str(dev).startswith("cuda")
    x0, y0 = torch.from_numpy(X[:1]).to(dev), torch.from_numpy(Y[:1]).to(dev)
    x1 = torch.from_numpy(X[1:2]).to(dev)
    o0 = [tuple(t.clone() for t in mb(to_prev(x0, mean, std))) for mb in members]
    o1 = [tuple(t.clone() for t in mb(to_prev(x1, mean, std))) for mb in members]

    def step(heads, dtype, nmem):
        for i in range(nmem):
            raw, ls, pf, ps = o0[i]
            _, _, pf1, ps1 = o1[i]
            v = (y0 != 0).any(dim=4).any(dim=1).reshape(-1, 1).to(dtype)
            yn = (y0 - mean) / std
            eye = torch.eye(129, dtype=dtype, device=raw.device)
            if "field" in heads:
                r = (yn.permute(0, 1, 4, 2, 3).reshape(1, CH, H, W) - raw).to(dtype).reshape(CH, -1).T * v
                Pm = feat(pf, dtype) * v
                G = Pm.T @ Pm
                L, _ = torch.linalg.cholesky_ex(G + rho * torch.diagonal(G).sum() / 129 * eye)
                corr = (feat(pf1, dtype) @ torch.cholesky_solve(Pm.T @ r, L)) * v
                _ = raw + corr.T.reshape(1, CH, H, W).float()
            if "sigma" in heads:
                ts = torch.log(yn.std(1, unbiased=False).permute(0, 3, 1, 2) + 1e-4)
                r = (ts - ls).to(dtype).reshape(C, -1).T * v
                Pm = feat(ps, dtype) * v
                G = Pm.T @ Pm
                L, _ = torch.linalg.cholesky_ex(G + rho * torch.diagonal(G).sum() / 129 * eye)
                corr = (feat(ps1, dtype) @ torch.cholesky_solve(Pm.T @ r, L)) * v
                _ = ls + corr.T.reshape(1, C, H, W).float()
        if cuda:
            torch.cuda.synchronize()

    def fwd(nmem):
        p1 = to_prev(x1, mean, std)
        for i in range(nmem):
            members[i](p1)
        if cuda:
            torch.cuda.synchronize()

    out = {}
    cases = [(f"forward x{n} (frozen, reference)", lambda n=n: fwd(n)) for n in sorted({1, len(members)})]
    for n in sorted({1, len(members)}):
        for heads in (("field",), ("sigma",), ("field", "sigma")):
            for dt in (torch.float32, torch.float64):
                cases.append((f"update {'+'.join(heads)} x{n} {str(dt)[6:]}",
                              lambda h=heads, d=dt, n=n: step(h, d, n)))
    for _ in range(max(3, reps // 10)):                   # warm every kernel first
        for _, fn in cases:
            fn()
    times = {nm: [] for nm, _ in cases}
    for _ in range(reps):                                 # interleaved, so drift hits every case
        for nm, fn in cases:
            t0 = perf_counter()
            fn()
            times[nm].append((perf_counter() - t0) * 1e3)
    print(f"\n=== batch-1 wall time, {dev}, eager, TF32 {'on' if _TF32 else 'off'}, {reps} interleaved reps (ms) ===")
    for nm, _ in cases:
        t = np.asarray(times[nm])
        med = float(np.median(t))
        out[nm] = {"median_ms": med, "p90_ms": float(np.percentile(t, 90))}
        extra = ""
        if nm.startswith("update"):
            bms = med * BOARD_PER_A100_MS
            extra = f"   ~{bms:.2f} T2 board ms ~ {bms * FINAL_PER_BOARD_MS_T2:+.3f} final (if A100)"
        print(f"  {nm:34s} median {med:7.3f}  p90 {out[nm]['p90_ms']:7.3f}{extra}")
    return out


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=SINGLE)
    ap.add_argument("--ens-ckpt", nargs="+", default=TWINS)
    ap.add_argument("--ens", choices=["auto", "on", "off"], default="auto",
                    help="auto: run the twin ensemble only if the single member's best arm is positive")
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--rhos", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0],
                    help="prior strengths; the CPU smoke run showed rho <= 0.1 over-fits one window "
                         "(rel-L2 -5 to -16 %%), so the grid starts at 1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-traj", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--bench-reps", type=int, default=300)
    ap.add_argument("--check-stream", action="store_true")
    ap.add_argument("--out", default=None, help="json summary")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        torch.set_num_threads(4)
    X, Y, ids, groups = load_heldout(a.max_traj, a.max_windows)
    print(f"[t2rr] {len(X)} windows, {len(groups)} held-out trajectories (derived from val_unseen.npz "
          f"== model.VAL_TRAJ), tiling asserted; alpha {a.alpha}; rhos {a.rhos}; device {dev}; "
          f"TF32 {'on' if _TF32 else 'off'}")
    summary = {"alpha": a.alpha, "rhos": a.rhos, "n_windows": len(X), "ids": ids, "tf32": _TF32}
    if a.check_stream:
        msg = check_stream(X, Y, ids, groups)
        print(f"[t2rr] {msg}")
        summary["stream_check"] = msg

    def evaluate(paths, tag):
        members = [Tapped(p, dev) for p in paths]
        names, res, res_b1, jpos, Pf = run_stream(members, X, Y, ids, groups, dev, a.alpha, a.rhos)
        mods = [mb.m for mb in members]
        eq = {}
        for b in (1, 64):
            ref = _fno.predict_fno(mods, X, dev, alpha=a.alpha, batch=b, amp=None)
            eq[f"batch{b}"] = e = equivalence(ref, Pf, Y, ids)
            del ref
            worst_w = max(e[f"{k}_max_window_reldiff"] for k in METRICS)
            print(f"[t2rr] {tag}: frozen@b1 vs predict_fno(batch={b}): field max|d| {e['max_abs']:.2e} "
                  f"p99.99 {e['p99.99_abs']:.2e} frac>1e-4 {e['frac_gt_1e-4']:.1e} at {e['argmax']} "
                  f"(pred temporal std {e['argmax_pred_temporal_std']:.2e}, target zero "
                  f"{e['argmax_target_identically_zero']}); metric rel-diff pooled rel "
                  f"{e['rel_pooled_reldiff']:.1e} tke {e['tke_pooled_reldiff']:.1e} mvpe "
                  f"{e['mvpe_pooled_reldiff']:.1e}, worst window {worst_w:.1e}")
        sel = jpos >= 1
        noise = (max(float(np.max(np.abs(res[k][0, sel] - res_b1[k][sel])
                                  / np.maximum(res_b1[k][sel], 1e-12))) for k in METRICS)
                 if sel.any() else 0.0)
        print(f"[t2rr] {tag}: in-batch frozen copy vs batch-1 frozen, worst per-window metric "
              f"rel-diff {noise:.1e}")
        worst = max(eq["batch1"][f"{k}_pooled_reldiff"] for k in METRICS)
        if worst > 1e-3:
            raise SystemExit(f"frozen arm does not reproduce predict_fno at the metric level "
                             f"(pooled rel-diff {worst:.1e} > 1e-3)")
        if worst > 1e-4:
            print(f"[t2rr] WARNING: metric-level agreement {worst:.1e} is above 1e-4 (abort at 1e-3)")
        table, best = summarize(names, res, jpos, ids, tag)
        return members, {"members": [os.path.basename(p) for p in paths], "table": table,
                         "best": best, "equivalence": eq, "batch_noise_metric_reldiff": noise}

    members, summary["single"] = evaluate(a.ckpt, "single held-out twin")
    best = summary["single"]["best"]
    pos = summary["single"]["table"]["pooled"][best]["dfinal_acc"] > 0
    if a.ens == "on" or (a.ens == "auto" and pos):
        members, summary["ensemble"] = evaluate(a.ens_ckpt, f"{len(a.ens_ckpt)}-twin ensemble")
    else:
        print(f"\n[t2rr] ensemble skipped (--ens {a.ens}, single-member best arm positive: {pos})")

    def dump():
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(summary, f, indent=1)
            print(f"[t2rr] wrote {a.out}")

    dump()                                                # the tables survive a late failure
    summary["bench"] = bench(members, X, Y, dev, a.bench_reps, a.rhos[len(a.rhos) // 2])
    dump()
    for key in [k for k in ("single", "ensemble") if k in summary]:
        best = summary[key]["best"]
        r = summary[key]["table"]["pooled"][best]
        q = summary[key]["table"].get("aoa15-20", summary[key]["table"]["pooled"])[best]
        nmem = len(summary[key]["members"])
        bkey = {"field": "field", "affine": "field", "sigma": "sigma",
                "both": "field+sigma"}[best.split("/")[0]]
        upd = summary["bench"].get(f"update {bkey} x{nmem} float32", {}).get("median_ms", float("nan"))
        lat = upd * BOARD_PER_A100_MS * FINAL_PER_BOARD_MS_T2
        net = 0.5 * r["dfinal_acc"] + lat               # the proxy over-reads error cuts ~2x
        gate = (max(r["pct"]["rel"], r["pct"]["tke"]) > 3.0
                and min(r["pct"]["rel"], r["pct"]["tke"]) > -0.5)
        verdict = "BUILD" if gate and net > 0 else "CLOSE"
        print(f"\n[t2rr] VERDICT {key} (best {best}): pooled %rel {r['pct']['rel']:+.2f} "
              f"%tke {r['pct']['tke']:+.2f} %mvpe {r['pct']['mvpe']:+.2f}; AoA 15-20 %rel "
              f"{q['pct']['rel']:+.2f} %tke {q['pct']['tke']:+.2f} %mvpe {q['pct']['mvpe']:+.2f}; "
              f"update {upd:.3f} ms (fp32, x{nmem}) ~ {lat:+.3f} final; "
              f"net 0.5*acc + latency = {net:+.3f} -> {verdict}")
        print("        rule: gate = > 3 % on rel or tke pooled with neither worse than -0.5 %, "
              "AND net > 0")
        summary[f"verdict_{key}"] = {"verdict": verdict, "gate": gate, "net_final": net,
                                     "update_ms": upd, "best": best}
    dump()


if __name__ == "__main__":
    main()
