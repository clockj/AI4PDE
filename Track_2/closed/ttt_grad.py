"""Gradient test-time training on the streamed trajectory: the last open Track-2 lever.

WHY THIS IS THE ONLY ONE LEFT.  Windows tile at stride 20 with T_in = T_out = 20, so window
k-1's revealed target IS window k's input, bit for bit.  Therefore **anything computable from
`prev_target_norm` alone is already computable from `input_norm`, and Track 1 could compute it
too**.  Exactly two things are genuinely new here:

  (a) the PAIR (x_{k-1}, y_{k-1}) -- in-distribution supervision from the very trajectory being
      forecast, which is the one thing that attacks the project's binding constraint (74
      training trajectories);
  (b) the REALIZED ERROR of predictions already made.

(b) is priced in `ttt_band.py` (a scale-only online band is worth <= +0.67 sps in hindsight and
a causal estimator recovers ~0 of it) and in `ttt_offline.py` (lag-1 residual correlation
+0.05; the ORACLE per-trajectory additive correction is worth 1.75% of rel-L2).  This file
prices (a), which is the only form of adaptation those two do not bound.

    python ttt_grad.py --sweep                     # the grid
    python ttt_grad.py --arm all --lr 1e-4         # one arm

Everything is scored by the official kit scoring on the same 336 held-out windows the frozen
baseline is scored on, and the per-step wall time is measured, because a gradient step has to
pay for its own latency at `time` weight 0.081 before it buys anything.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_2/.  Putting the PARENT of closed/ on the
# path keeps `import t2_common / sps9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_2/, so every submit/ and ../Track_1/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "submit"))
sys.path.insert(0, str(HERE.parent / "Track_1"))
import t2_common as t2  # noqa: E402
import _fno  # noqa: E402
import final_v9  # noqa: E402

KIT = "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9"
_s = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(_s); _s.loader.exec_module(sc)

T, H, W, CH = 20, 32, 64, 40
ALPHA = 0.90
U = str(HERE.parent / "Track_1" / "unet")
HO8 = [f"{U}/fno_vl_s{i}.pth" for i in (1, 2, 3, 4)] + [
    f"{U}/fno_vl_m1220.pth", f"{U}/fno_vl_w48.pth", f"{U}/fno_vl_w64.pth", f"{U}/unet_g1.pth"]

# Which parameters an arm is allowed to move.  Ordered from most to least capacity; the
# project's standing result is that capacity at the BOUNDARIES helps and capacity in the TRUNK
# hurts, so the boundary-only arms are the ones with a prior.
ARMS = {
    "all":   lambda m: [p for p in m.parameters() if p.requires_grad],
    "heads": lambda m: [p for n, p in m.named_parameters() if n.startswith("head")],
    "field": lambda m: [p for n, p in m.named_parameters() if n.startswith("head_field")],
    "lift":  lambda m: [p for n, p in m.named_parameters() if n.startswith("lift")],
    "norm":  lambda m: [p for n, p in m.named_parameters() if ".norm." in n or n.startswith("norm.")],
    "last":  lambda m: [p for n, p in m.named_parameters()
                        if n.startswith("head") or ".3." in n],
}


def to_prev(xn):
    """(B,20,32,64,>=2) normalized -> (B,40,32,64) trunk input."""
    return xn[..., :2].permute(0, 1, 4, 2, 3).reshape(len(xn), CH, H, W)


def compose(raw, logstd, alpha=ALPHA):
    y = raw.reshape(-1, T, 2, H, W).permute(0, 1, 3, 4, 2)
    if logstd is None or alpha <= 0:
        return y
    sig = logstd.clamp(-8., 2.).exp().permute(0, 2, 3, 1).unsqueeze(1)
    m_ = y.mean(1, keepdim=True)
    s_ = y.std(1, keepdim=True, unbiased=False)
    return m_ + (y - m_) * ((alpha * sig + (1 - alpha) * s_) / (s_ + 1e-6))


def load_stream(names, dev):
    st = t2.build_stream(names)
    X = torch.from_numpy(np.stack([t2.normalize_in(s[2]) for s in st])).to(dev)
    Yn = torch.from_numpy(np.stack([t2.normalize_in(s[3]) for s in st])).to(dev)
    Yr = np.stack([s[3] for s in st]).astype(np.float32)
    grp, cur = [], [0]
    for i in range(1, len(st)):
        if st[i][0] != st[i - 1][0]:
            grp.append(cur); cur = []
        cur.append(i)
    grp.append(cur)
    return X, Yn, Yr, grp, [f"{s[0]}:{s[1]}" for s in st]


def score(P_raw, Y_raw, half, t_step):
    r = float(np.mean(sc.rel_l2_per_sample(P_raw, Y_raw, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P_raw, Y_raw, 2)))
    m = sc.mvpe_rel_l2(P_raw, Y_raw)
    p2 = P_raw[..., :2]
    s = sc.aggregate_sps(P_raw, Y_raw, 2, lower=p2 - half, upper=p2 + half)[0]
    five = [sc.score_error(r), sc.score_error(k), sc.score_error(m),
            sc.score_time(t_step), sc.score_sps(s)]
    return r, k, m, five, final_v9.predict(five)[0]


def run(model, X, Yn, Yr, grp, arm, lr, opt_name, steps, buf, loss_name, dev, alpha=ALPHA):
    """Stream with adaptation; returns (pred_raw, mean seconds/step)."""
    init = copy.deepcopy(model.state_dict())
    params = ARMS[arm](model) if arm != "none" else []
    P = np.zeros((len(X), T, H, W, 3), dtype=np.float32)
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    std = torch.from_numpy(_fno.STD).to(dev)
    times = []
    for g in grp:
        t0 = perf_counter()
        model.load_state_dict(init)
        if params:
            params = ARMS[arm](model)
            opt = (torch.optim.SGD(params, lr=lr, momentum=0.9) if opt_name == "sgd"
                   else torch.optim.Adam(params, lr=lr))
        reset_t = perf_counter() - t0
        hist = []
        for j, i in enumerate(g):
            t0 = perf_counter()
            if params and hist:
                model.train()
                for _ in range(steps):
                    idx = [hist[-1]] if buf == "last" else (
                        [hist[np.random.randint(len(hist))]] if buf == "rand" else hist[-min(len(hist), 4):])
                    xb = X[idx]
                    yb = Yn[idx][..., :2]
                    raw, ls = model(to_prev(xb))
                    pr = raw.reshape(len(xb), T, 2, H, W).permute(0, 1, 3, 4, 2)
                    if loss_name == "mse":
                        loss = ((pr - yb) ** 2).mean()
                    else:                                    # per-sample relative L2, physical
                        d = ((pr - yb) * std).reshape(len(xb), -1)
                        t_ = ((yb) * std + mean).reshape(len(xb), -1)
                        loss = (d.norm(dim=1) / t_.norm(dim=1)).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                raw, ls = model(to_prev(X[i:i + 1]))
                y = compose(raw, ls, alpha)
                P[i, ..., :2] = (y * std + mean).float().cpu().numpy()[0]
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(perf_counter() - t0 + (reset_t if j == 0 else 0.0))
            hist.append(i)
    model.load_state_dict(init)
    return P, float(np.mean(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[f"{U}/fno_vl_s1.pth"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--arm", default="all")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--opt", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--buf", default="last", choices=["last", "rand", "recent"])
    ap.add_argument("--loss", default="mse", choices=["mse", "rel"])
    ap.add_argument("--band", default=str(HERE / "submit" / "band.npz"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = _fno.fuse([_fno.load_any(c, dev) for c in a.ckpt])
    assert len(models) == 1, "adapt a single fused group; pass same-architecture members"
    model = models[0]
    X, Yn, Yr, grp, ids = load_stream(t2.UNSEEN, dev)
    z = np.load(a.band)
    half = z["base"][..., :2].astype(np.float32)      # fixed band: ranking, not a package
    print(f"{len(a.ckpt)} member(s), {len(X)} windows, {len(grp)} trajectories, dev={dev}")

    P0, t0 = run(model, X, Yn, Yr, grp, "none", 0, "sgd", 0, "last", "mse", dev)
    r, k, m, five, wf = score(P0, Yr, half, t0)
    print(f"\n{'arm':7s} {'opt':5s} {'lr':>8s} {'buf':7s} {'loss':5s} | {'rel':>7s} {'tke':>7s} "
          f"{'mvpe':>7s} {'ms':>7s} | {'wfinal':>7s} {'d':>7s}")
    print(f"{'FROZEN':7s} {'-':5s} {'-':>8s} {'-':7s} {'-':5s} | {r:7.4f} {k:7.4f} {m:7.4f} "
          f"{t0*1e3:7.2f} | {wf:7.3f} {0.0:7.3f}")
    base = wf

    grid = ([(arm, o, lr, b, ls) for arm in ("all", "heads", "lift", "norm")
             for o, lr in (("sgd", 1e-4), ("sgd", 1e-3), ("sgd", 1e-2),
                           ("adam", 1e-5), ("adam", 1e-4))
             for b in ("last",) for ls in ("mse",)]
            + [("all", "adam", 1e-5, b, "mse") for b in ("rand", "recent")]
            + [("all", "adam", 1e-5, "last", "rel")]) if a.sweep else \
           [(a.arm, a.opt, a.lr, a.buf, a.loss)]

    for arm, o, lr, b, ls in grid:
        P, t = run(model, X, Yn, Yr, grp, arm, lr, o, a.steps, b, ls, dev)
        r, k, m, five, wf = score(P, Yr, half, t)
        print(f"{arm:7s} {o:5s} {lr:8.0e} {b:7s} {ls:5s} | {r:7.4f} {k:7.4f} {m:7.4f} "
              f"{t*1e3:7.2f} | {wf:7.3f} {wf - base:+7.3f}", flush=True)


if __name__ == "__main__":
    main()
