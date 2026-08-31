"""K-fold cross-validation over all 82 real trajectories, with error bars.

WHY.  Every local number this project has ever quoted comes from ONE held-out set of 8
trajectories (336 windows), and `proxy_calib.py` measures its resolution at ~2% of error --
which is larger than every per-round improvement since 2026-08-08, and large enough that
R11 -> R12 the proxy got the SIGN of the rel-L2 move wrong.  Three rounds of work are
indistinguishable from noise at the board (`ROADMAP.md` 1.7), and part of that is the board
and part of it is the instrument.  This fixes the part that is the instrument.

WHAT.  Partition the 82 trajectories into K folds, train one model per fold with that fold
excluded, and evaluate each model on its own fold.  Every trajectory is then scored by a model
that never saw it, and the pooled sample is all 3 444 windows instead of 336 -- 10x, with the
Re x AoA grid fully covered instead of 8 arbitrary points on it.

    python kfold.py --build                              # write the folds (once)
    python kfold.py --score --ckpt f0.pth ... f4.pth      # pooled, with bootstrap error bars
    python kfold.py --paired --a a0.pth ... --b b0.pth    # A vs B, paired over trajectories

WHAT IT DOES AND DOES NOT FIX.  It removes the LEVEL noise -- which trajectories happened to
be held out -- and it gives every arm the same full coverage, so a per-AoA or per-Re breakdown
finally has a usable sample size in each cell.  It does NOT remove the board's distribution
shift: the hidden set has unseen Reynolds numbers and reads ~1.078x the AoA 15-20 proxy on
rel-L2.  For that, `re_split.py` is the instrument and this one is the control.

THE NUMBER THAT MATTERS is the paired bootstrap.  Two arms are scored on identical windows, so
the trajectory effect -- which dominates the level, rel-L2 0.044 at AoA 5 against 0.105 at 20
-- cancels in the difference.  `--paired` resamples TRAJECTORIES (not windows: windows within a
trajectory are neither independent nor exchangeable, they tile it) and reports the delta's own
standard error.  That is the honest answer to "is this arm real", and it replaces comparing a
delta to a borrowed seed sigma.

COST.  K trainings per configuration instead of one, ~18 min each on an A100.  At K = 5 that
is 1.5 GPU-hours per arm -- affordable for CONFIRMING an arm that already cleared the cheap
proxy, not for screening.  Screen on `compare_v9.py --aoa 15 20`, confirm here.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
OUT_DIR = os.environ.get("RPDE_KFOLD_DIR", "/fs/scratch/PAS1187/qwu/tmp/realpde/kfold")
T_IN = T_OUT = INTERVAL = 20
SUB = 2
HORIZON = T_IN + T_OUT
K = 5


def all_names(data=DATA):
    return sorted(f[:-3] for f in os.listdir(data) if f.endswith(".h5"))


def folds(k=K, data=DATA):
    """K folds, stratified so each spans the AoA range and the Re range.

    Sorting by (AoA, Re) and dealing round-robin puts consecutive Re at the same AoA into
    DIFFERENT folds, so no fold is missing an angle -- which matters because error is governed
    by AoA and by almost nothing else, and an unbalanced fold would just be measuring its own
    AoA mix.
    """
    def key(n):
        re_, aoa = n.split("_")
        return (int(aoa), int(re_))
    ordered = sorted(all_names(data), key=key)
    out = [[] for _ in range(k)]
    for i, n in enumerate(ordered):
        out[i % k].append(n)
    return [sorted(f) for f in out]


def build(k=K, data=DATA, out_dir=OUT_DIR):
    import h5py
    os.makedirs(out_dir, exist_ok=True)
    fs = folds(k, data)
    for i, names in enumerate(fs):
        xs, ys, ids = [], [], []
        for name in names:
            with h5py.File(os.path.join(data, f"{name}.h5"), "r") as f:
                u = f["u"][:, ::SUB, ::SUB].astype(np.float32)
                v = f["v"][:, ::SUB, ::SUB].astype(np.float32)
            d = np.stack([u, v, np.zeros_like(u)], -1)
            for t in range(0, len(d) - HORIZON + 1, INTERVAL):
                xs.append(d[t:t + T_IN]); ys.append(d[t + T_IN:t + HORIZON])
                ids.append(f"{name}:{t}")
        p = os.path.join(out_dir, f"fold{i}.npz")
        np.savez(p, x=np.asarray(xs, np.float32), y=np.asarray(ys, np.float32),
                 ids=np.array(ids))
        print(f"{p}: {len(xs)} windows / {len(names)} trajectories")
    print()
    for i, names in enumerate(fs):
        print(f"# fold {i}\nEX{i}=\"{' '.join(names)}\"")
    print(f"\n{sum(len(f) for f in fs)} trajectories over {k} folds")


LIMIT = 0        # >0 truncates every fold, for exercising the code path on a CPU


def load_folds(k=K, out_dir=OUT_DIR):
    ds = [np.load(os.path.join(out_dir, f"fold{i}.npz")) for i in range(k)]
    if LIMIT:
        ds = [{"x": d["x"][:LIMIT], "y": d["y"][:LIMIT], "ids": d["ids"][:LIMIT]} for d in ds]
    return ds


def traj_of(ids):
    return np.array([str(i).split(":")[0] for i in ids])


def aoa_of(ids):
    return np.array([int(str(i).split(":")[0].split("_")[1]) for i in ids])


def predict_all(ckpts, alpha, dev, k=K, out_dir=OUT_DIR):
    """Per-fold inference: fold i is scored ONLY by the checkpoint that excluded it."""
    import _fno
    from eval_v9 import infer
    ds = load_folds(k, out_dir)
    P, Y, I = [], [], []
    for i, (c, d) in enumerate(zip(ckpts, ds)):
        path = c if os.path.isabs(c) else os.path.join(HERE, c)
        m = [_fno.load_any(path, dev)]
        P.append(infer(m, d["x"], dev, alpha)); Y.append(d["y"]); I.append(d["ids"])
        del m
    return np.concatenate(P), np.concatenate(Y), np.concatenate(I)


def metrics(P, Y):
    """Per-window rel-L2 and TKE, and the MVPE pieces, kept UNPOOLED so they can be resampled."""
    import sps9
    sc = sps9.sc
    return (np.asarray(sc.rel_l2_per_sample(P, Y, 2), float),
            np.asarray(sc.tke_rel_l2_per_sample(P, Y, 2), float))


def boot(vals, traj, n=2000, seed=0):
    """Bootstrap the mean of a per-window quantity by resampling TRAJECTORIES.

    Windows tile a trajectory at stride 20, so they are neither independent nor exchangeable;
    resampling them would understate the error by roughly sqrt(42).
    """
    rng = np.random.default_rng(seed)
    names = np.unique(traj)
    idx = {t: np.flatnonzero(traj == t) for t in names}
    out = np.empty(n)
    for b in range(n):
        pick = rng.choice(names, size=len(names), replace=True)
        out[b] = float(np.mean(np.concatenate([vals[idx[t]] for t in pick])))
    return float(out.mean()), float(out.std(ddof=1))


def score(ckpts, alpha, inflate, ms, dev, k=K, out_dir=OUT_DIR, nboot=2000):
    import final_v9
    import sps9
    from sps9 import optimal_band, prep, sc, sps_score
    P, Y, I = predict_all(ckpts, alpha, dev, k, out_dir)
    tr, ao = traj_of(I), aoa_of(I)
    rel, tke = metrics(P, Y)
    mv = sc.mvpe_rel_l2(P, Y)

    # Leave-one-fold-out band: an arm must not be judged by a band fitted to the same errors.
    # `prep` returns (|E|, a_i, scored mask, n_scored), and n_scored is a COUNT OVER THE SET --
    # so a fold's score must be computed with that fold's own count, not the pooled one. And
    # sps_score is a sum over samples divided by that count, so folds pool by re-weighting with
    # n_scored, not with window count. Getting either wrong silently rescales SPS.
    #
    # The band is fitted on INFLATED errors and scored on uninflated ones, matching
    # compare_v9.py exactly, so the two are directly comparable. (alpha_board.py inflates both
    # sides because it is projecting to the board rather than ranking locally.)
    sizes = [len(d["ids"]) for d in load_folds(k, out_dir)]
    edges = np.cumsum([0] + sizes)
    A, aw, scored, n_all = prep(P, Y)
    f = np.float32(inflate)
    num = 0.0
    for i in range(k):
        hold = np.zeros(len(I), bool); hold[edges[i]:edges[i + 1]] = True
        fit = ~hold
        band = optimal_band(f * A[fit], aw[fit], scored[fit])
        n_hold = int(np.count_nonzero(scored[hold]))
        num += n_hold * sps_score(A[hold], aw[hold], scored[hold], n_hold, band)
    sp = num / n_all

    r_m, r_s = boot(rel, tr, nboot)
    k_m, k_s = boot(tke, tr, nboot)
    five = [sc.score_error(rel.mean()), sc.score_error(tke.mean()), sc.score_error(mv),
            sc.score_time(ms / 1000.0), sp]
    print(f"pooled over {len(I)} windows / {len(np.unique(tr))} trajectories, {k} folds")
    print(f"  rel-L2 {rel.mean():.5f} +- {r_s:.5f}   TKE {tke.mean():.5f} +- {k_s:.5f}   "
          f"MVPE {mv:.5f}")
    print(f"  scores rel {five[0]:.3f}  tke {five[1]:.3f}  mvpe {five[2]:.3f}  "
          f"time {five[3]:.3f}  sps {five[4]:.3f}   (sps LOFO over folds)")
    print(f"  {final_v9.report(five, 'wfinal ')}")
    print(f"\n  {'AoA':>5s} {'n':>5s} {'rel-L2':>9s} {'TKE':>9s}")
    for a in sorted(np.unique(ao)):
        s = ao == a
        print(f"  {a:5d} {int(s.sum()):5d} {rel[s].mean():9.5f} {tke[s].mean():9.5f}")
    return dict(rel=rel, tke=tke, traj=tr, mv=mv, sps=sp, five=five)


def paired(a_ck, b_ck, alpha, inflate, ms, dev, k=K, out_dir=OUT_DIR, nboot=4000, seed=0):
    """A vs B on identical windows, with the delta's own bootstrap standard error."""
    import sps9
    sc = sps9.sc
    Pa, Y, I = predict_all(a_ck, alpha, dev, k, out_dir)
    Pb, _, _ = predict_all(b_ck, alpha, dev, k, out_dir)
    tr = traj_of(I)
    ra, ka = metrics(Pa, Y)
    rb, kb = metrics(Pb, Y)
    print(f"paired over {len(I)} windows / {len(np.unique(tr))} trajectories")
    print(f"  {'metric':8s} {'A':>9s} {'B':>9s} {'B - A':>10s} {'SE':>9s} {'z':>7s} {'%':>7s}")
    for name, va, vb in (("rel-L2", ra, rb), ("TKE", ka, kb)):
        d = vb - va
        m, s = boot(d, tr, nboot, seed)
        z = m / s if s > 0 else np.nan
        print(f"  {name:8s} {va.mean():9.5f} {vb.mean():9.5f} {m:+10.5f} {s:9.5f} "
              f"{z:+7.2f} {100 * m / va.mean():+7.2f}")
    mva, mvb = sc.mvpe_rel_l2(Pa, Y), sc.mvpe_rel_l2(Pb, Y)
    print(f"  {'MVPE':8s} {mva:9.5f} {mvb:9.5f} {mvb - mva:+10.5f} {'':9s} {'':7s} "
          f"{100 * (mvb - mva) / mva:+7.2f}")
    print("\n  A delta is a result when |z| clears ~2 AND the % clears the board's ~2% "
          "transfer resolution\n  (proxy_calib.py). The first is necessary; only the second "
          "is sufficient.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--folds", action="store_true",
                    help="print the fold exclude-lists as shell assignments and exit; a job "
                         "script evals this so the training splits cannot drift from the "
                         "scoring splits")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--ckpt", nargs="+", default=None)
    ap.add_argument("--a", nargs="+", default=None)
    ap.add_argument("--b", nargs="+", default=None)
    ap.add_argument("-k", type=int, default=K)
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--inflate", type=float, default=1.39)
    ap.add_argument("--ms", type=float, default=1.02)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0,
                    help="truncate every fold to this many windows. For exercising the "
                         "scoring path on a CPU before a job spends an hour training the "
                         "models it will be handed -- NOT for reporting numbers.")
    a = ap.parse_args()
    if a.folds:
        for i, names in enumerate(folds(a.k)):
            print(f'EX{i}="{" ".join(names)}"')
        return
    if a.build:
        build(a.k, out_dir=a.out_dir); return
    global LIMIT
    LIMIT = a.limit
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.paired:
        if not (a.a and a.b) or len(a.a) != a.k or len(a.b) != a.k:
            raise SystemExit(f"--paired needs --a and --b with exactly {a.k} checkpoints each")
        paired(a.a, a.b, a.alpha, a.inflate, a.ms, dev, a.k, a.out_dir); return
    if not a.ckpt or len(a.ckpt) != a.k:
        raise SystemExit(f"--score needs exactly {a.k} checkpoints, fold-ordered")
    score(a.ckpt, a.alpha, a.inflate, a.ms, dev, a.k, a.out_dir)


if __name__ == "__main__":
    main()
