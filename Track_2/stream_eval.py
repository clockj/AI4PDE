"""Score a Track-2 submission by STREAMING real held-out trajectories through it.

The kit's ``local_eval.py`` is a plumbing check on synthetic data.  This is the real thing:
the same streaming protocol (batch 1, time order, ``reset_ttt_state`` at boundaries, previous
target revealed and timed exactly the way the platform charges it), driven by ``train_real``
trajectories, and scored with the OFFICIAL ``scoring.py`` from the kit.

    python stream_eval.py --submission submit --split unseen
    python stream_eval.py --submission submit --split unseen --aoa 15 20
    python stream_eval.py --submission submit --traj 3750_0 26700_0 --repeat 3

Two numbers it prints are not comparable to anything else without care:

* ``time_score`` is LOCAL wall time.  Track 1's board runs ~2-5x slower than our local timing
  and the mapping is noisy (``ROADMAP.md`` 1.5); Track 2's board overhead is entirely unknown
  until we have a round.  Treat it as a budget, not a prediction.
* the error subscores come from held-out ``train_real`` trajectories, which sit INSIDE the
  training Re range.  The board's per-element error runs ~1.39x ours (``inflate_v9.py``).
  ``--aoa 15 20`` is the subset that most resembles the board.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KIT = os.environ.get("RPDE_KIT", "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9")   # REPRODUCE §9.1

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Track_1"))

import t2_common as t2  # noqa: E402

_spec = importlib.util.spec_from_file_location("sc9", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def import_submission(d: Path):
    f = d / "submission.py"
    if not f.exists():
        raise SystemExit(f"no submission.py under {d}")
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("t2_submission", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(model, stream, device, warm=0, prev_space="target"):
    """Drive the platform's loop.  Returns (pred_raw, lower_raw, upper_raw, target, times).

    ``prev_space`` is how the REVEALED target is normalized: "target" (default since 2026-09-28) is what the official
    runner does (``t2_common.normalize_tgt``); "input" is this harness before 09-28 (``normalize_in``), kept so an old
    local read can be reproduced exactly.
    """
    import torch

    preds, lows, ups, tgts, times = [], [], [], [], []
    bounds_active = None
    for name, tid, x_raw, y_raw, is_first in stream:
        xn = torch.from_numpy(t2.normalize_in(x_raw)[None])
        yn = torch.from_numpy(t2.normalize_prev(y_raw, prev_space)[None])   # only used as prev_target
        reset_t = 0.0
        if is_first:
            t0 = perf_counter()
            model.reset_ttt_state()
            if device == "cuda":
                torch.cuda.synchronize()
            reset_t = perf_counter() - t0
            prev_target = None
        t0 = perf_counter()
        pred_norm, info = model.ttt_step(xn, prev_target)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(perf_counter() - t0 + reset_t)
        prev_target = yn                      # cached AFTER the timed call, as the kit does

        pred_norm = torch.as_tensor(pred_norm).detach().cpu().numpy()
        has = "lower" in info
        if bounds_active is None:
            bounds_active = has
        elif has != bounds_active:
            raise SystemExit("submission changed its mind about SPS bounds mid-run")
        preds.append(t2.denorm_pred(pred_norm[0]))
        tgts.append(y_raw)
        if has:
            lows.append(t2.denorm_pred(torch.as_tensor(info["lower"]).detach().cpu().numpy()[0]))
            ups.append(t2.denorm_pred(torch.as_tensor(info["upper"]).detach().cpu().numpy()[0]))
    P = np.stack(preds).astype(np.float32)
    Y = np.stack(tgts).astype(np.float32)
    L = np.stack(lows).astype(np.float32) if bounds_active else None
    U = np.stack(ups).astype(np.float32) if bounds_active else None
    return P, L, U, Y, np.asarray(times)


def causal_check(model, stream, device, seed=0, prev_space="target"):
    """CAUSALITY ASSERTION for adaptive (ttt) packages, 2026-09-25 (Ilse / Cindy, R1).

    In each trajectory pick a cut step c (its middle window). Re-run the stream with every window AFTER c replaced by
    noise -- both its input and its target, so the truth the platform would reveal at steps > c is garbage -- and
    require the outputs (pred / lower / upper) at every step <= c to be BIT-IDENTICAL to the clean run. A package that
    reads anything from the future (a trajectory-pooled statistic, a look-ahead buffer) fails this. Returns
    (n_steps_compared, max_abs_diff); raises SystemExit on any difference.
    """
    rng = np.random.default_rng(seed)
    names = [s[0] for s in stream]
    cut = {}
    for nm in dict.fromkeys(names):
        idx = [i for i, n in enumerate(names) if n == nm]
        cut[nm] = idx[len(idx) // 2]
    noisy = []
    for i, (nm, tid, x, y, first) in enumerate(stream):
        if i > cut[nm]:
            x = rng.standard_normal(x.shape).astype(np.float32) * x.std() + x.mean()
        if i >= cut[nm]:        # window c's own target is revealed only at step c+1: noise it too
            y = rng.standard_normal(y.shape).astype(np.float32) * y.std() + y.mean()
        noisy.append((nm, tid, x, y, first))
    A = run(model, stream, device, prev_space=prev_space)
    B = run(model, noisy, device, prev_space=prev_space)
    keep = np.array([i <= cut[n] for i, n in enumerate(names)])
    worst = 0.0
    for k, (a, b) in enumerate(zip(A[:3], B[:3])):
        if a is None:
            continue
        d = np.abs(a[keep] - b[keep])
        worst = max(worst, float(d.max()))
        if not np.array_equal(a[keep], b[keep]):
            raise SystemExit(f"CAUSALITY FAIL: {['pred', 'lower', 'upper'][k]} at steps <= cut changed when only "
                             f"FUTURE windows were replaced by noise (max |diff| {d.max():.3e})")
    return int(keep.sum()), worst


def score(P, L, U, Y, t_step):
    c = 2
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, c)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, c)))
    m = sc.mvpe_rel_l2(P, Y)
    s, cov = sc.aggregate_sps(P, Y, c, lower=L, upper=U)
    return dict(rel_err=r, tke_err=k, mvpe_err=m, coverage=cov,
                rel=sc.score_error(r), tke=sc.score_error(k), mvpe=sc.score_error(m),
                time=sc.score_time(t_step), sps=sc.score_sps(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=str(HERE / "submit"))
    ap.add_argument("--split", choices=["unseen", "seen", "all"], default="unseen")
    ap.add_argument("--traj", nargs="+", default=None)
    ap.add_argument("--aoa", type=int, nargs="+", default=None,
                    help="restrict the SCORED windows (the stream is still run in full, so "
                         "adaptation sees the trajectory it would see on the board)")
    ap.add_argument("--repeat", type=int, default=1, help="re-run for a stable timing median")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None, help="npz of predictions, for band fitting")
    ap.add_argument("--causal-check", action="store_true",
                    help="assert that noising every window after each trajectory's middle leaves the earlier outputs "
                         "bit-identical (required for any adaptive T2 package; REPRODUCE §9.1)")
    ap.add_argument("--prev-space", choices=["target", "input"], default="target",
                    help="normalization of the revealed previous target: 'target' = the official runner (kit t2_v6 "
                         "local_eval, mean_tgt/std_tgt; default since 2026-09-28), 'input' = this harness before 09-28")
    a = ap.parse_args()

    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = a.traj or {"unseen": t2.UNSEEN, "seen": t2.SEEN, "all": t2.all_names()}[a.split]
    stream = t2.build_stream(names)
    print(f"[stream_eval] {len(stream)} steps over {len(set(s[0] for s in stream))} "
          f"trajectories, device={dev}, revealed target normalized with {a.prev_space.upper()} statistics")

    mod = import_submission(Path(a.submission).resolve())
    model = mod.get_ttt_model(str(Path(a.submission).resolve()), dev)
    if a.causal_check:
        n, worst = causal_check(model, stream, dev, prev_space=a.prev_space)
        print(f"[stream_eval] CAUSALITY OK: {n} steps up to each trajectory's cut are bit-identical with every later "
              f"window replaced by noise (max |diff| {worst:.1e})")

    best = None
    for rep in range(a.repeat):
        P, L, U, Y, T = run(model, stream, dev, prev_space=a.prev_space)
        gc.collect()
        med = float(np.mean(T))
        if best is None or med < best[0]:
            best = (med, P, L, U, Y, T)
        print(f"  rep {rep}: mean/step {med*1e3:.3f} ms  (median {np.median(T)*1e3:.3f}, "
              f"p90 {np.percentile(T,90)*1e3:.3f})")
    t_step, P, L, U, Y, T = best

    sel = np.ones(len(P), bool)
    if a.aoa:
        aoa = np.array([t2.aoa_of(s[0]) for s in stream])
        sel = np.isin(aoa, a.aoa)
        print(f"[stream_eval] scoring AoA {a.aoa}: {int(sel.sum())}/{len(sel)} windows")
    d = score(P[sel], None if L is None else L[sel], None if U is None else U[sel],
              Y[sel], t_step)

    import final_v9
    five = [d["rel"], d["tke"], d["mvpe"], d["time"], d["sps"]]
    print(f"\n  errors   rel {d['rel_err']:.4f}  tke {d['tke_err']:.4f}  mvpe {d['mvpe_err']:.4f}"
          f"   coverage {d['coverage']:.3f}")
    print(f"  scores   rel {d['rel']:.3f}  tke {d['tke']:.3f}  mvpe {d['mvpe']:.3f}  "
          f"time {d['time']:.3f}  sps {d['sps']:.3f}")
    print(f"  mean5 {np.mean(five):.3f}   {final_v9.report(five, 'w')}")
    print(f"  local {t_step*1e3:.3f} ms/step   (bounds: "
          f"{'supplied' if L is not None else 'DEFAULT +-5%'})")
    if a.save:
        np.savez_compressed(a.save, pred=P, target=Y,
                            lower=np.zeros(0) if L is None else L,
                            upper=np.zeros(0) if U is None else U,
                            ids=np.array([f"{s[0]}:{s[1]}" for s in stream]), times=T)
        print(f"  saved {a.save}")


if __name__ == "__main__":
    main()
