"""Does TRAJECTORY HISTORY carry information the trained model does not already have?

`ctx_probe.py` established that a parameter-free predictor improves 0.1424 -> 0.1059 as its
input grows from the protocol's 20 frames to 200, and that the target's time-mean is estimated
23% better by a running average than by the input window's own mean (0.1022 -> 0.0783). But
the model is not a parameter-free predictor: CLAUDE.md records its own time-mean error at
0.0438, well below every history estimator. So "history is informative" and "history helps THIS
model" are different claims, and only the second one is worth anything.

This measures the second. Three arms, all scored through the v9 scorer with each arm's own
refitted SPS band, because a band fitted to one arm flatters it:

    A  the model alone                                   (the control)
    B  time-mean blended toward the running history mean (beta sweep)
    C  the whole field blended toward a long-context DMD (alpha sweep)

LEGITIMACY. This is a TRACK 2 instrument. Track 2's official `local_eval.py` holds one model
object, calls `ttt_step` in strict time order and calls `reset_ttt_state()` at trajectory
boundaries -- so frames the stream has already delivered are the model's to keep, and using
them is what a streaming forecaster does. It reads only frames BEFORE the forecast time, which
is what separates it from the leak CLAUDE.md records (window k+1's input IS window k's target).
Track 1 gets one window and nothing else; none of this is available there.
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401  -- moved into closed/ 2026-09-19
# This file used to sit at the top level of Track_1/.  Putting the PARENT of closed/ on the
# path keeps `import sps9 / eval_v9 / final_v9 / <sibling>` working, and `HERE` below still
# resolves to Track_1/, so every val_*.npz, unet/*.pth and submit_fno/ path is unchanged.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))

import final_v9  # noqa: E402
from sps9 import optimal_band, prep, sc, sps_score  # noqa: E402


def score(P, Y, band_fit, ms, inflate):
    S = prep(*band_fit)
    band = optimal_band(np.float32(inflate) * S[0], S[1], S[2])
    U = prep(P, Y)
    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
    mv = sc.mvpe_rel_l2(P, Y)
    five = [sc.score_error(r), sc.score_error(k), sc.score_error(mv),
            sc.score_time(ms / 1000.0), sps_score(*U, band)]
    return r, k, mv, five, final_v9.predict(five)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--inflate", type=float, default=1.39)
    ap.add_argument("--ms", type=float, default=1.90)
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.15, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--dmd", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--ctx", type=int, default=200)
    ap.add_argument("--aoa", type=int, nargs="+", default=None)
    ap.add_argument("--gain", type=float, nargs="+",
                    default=[0.7, 0.85, 0.95, 1.05, 1.2],
                    help="arm D: a global gain on the composed fluctuation, y = m + g*(y-m). "
                         "Round 6 swept this and called it noise, under weights that priced "
                         "`tke` at 0.244; the nine-round fit prices it at 0.097 against "
                         "`rel_l2` 0.582, so the rel/tke trade this knob rides has moved by a "
                         "factor of 2.5 and the old optimum need not survive. A pure-mean "
                         "predictor scores fluctuation error ||yf||/||y|| = 0.0813 on this "
                         "split, so a model whose fluctuation error exceeds that is being "
                         "paid for by TKE alone.")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    import torch

    import _fno
    from ctx_probe import history_for
    from dmd_probe import dmd_forecast
    from eval_v9 import infer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    x, y, ids = du["x"], du["y"], du["ids"]
    sel = np.ones(len(x), bool)
    if args.aoa:
        sel = np.isin([int(str(i).split(":")[0].split("_")[1]) for i in ids], args.aoa)

    m = [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
         for c in args.ckpt]
    P = infer(m, x, dev, args.alpha)
    Ps = infer(m, ds["x"], dev, args.alpha)
    if args.save:
        np.savez_compressed(args.save, p=P, ids=ids)
        print(f"saved predictions -> {args.save}")

    def hist_for(idlist):
        tr, wh, tt = history_for(idlist)
        hm = np.stack([tr[w][:t + 20].mean(0) for w, t in zip(wh, tt)])
        dm = np.stack([dmd_forecast(tr[w][max(t + 20 - args.ctx, 0):t + 20], 20,
                                    min(8, max(2, min(t + 20, args.ctx) - 2)))
                       for w, t in zip(wh, tt)])
        return hm, dm, tt

    hist_mean, dmd, t0 = hist_for(ids)
    # The band is refit for EVERY arm on that arm's own val_seen predictions. Fitting it once
    # on the baseline is exactly the mistake that made round 6 declare the band closed: a band
    # fitted at one operating point makes every other point look worse in-sample by
    # construction (ROADMAP 1.10, "refit everything downstream before comparing").
    hist_mean_s, dmd_s, _ = hist_for(ds["ids"])
    hdr = (f"{'arm':30s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} {'tke':>6s} "
           f"{'mvpe':>6s} {'sps':>6s} | {'wfinal':>7s}")
    print(f"\n{len(x[sel])} windows"
          + (f", AoA {args.aoa}" if args.aoa else "") + f", history median "
          f"{int(np.median(t0 + 20))} frames\n{hdr}\n" + "-" * len(hdr))

    def row(tag, Q, Qs):
        r, k, mv, five, wf = score(Q[sel], y[sel], (Qs, ds["y"]), args.ms, args.inflate)
        print(f"{tag:30s} {r:7.4f} {k:7.4f} {mv:7.4f} | {five[0]:6.2f} {five[1]:6.2f} "
              f"{five[2]:6.2f} {five[4]:6.2f} | {wf:7.3f}")
        return wf

    base = row("A  model alone", P, Ps)
    pm = P[..., :2].mean(1, keepdims=True)
    pms = Ps[..., :2].mean(1, keepdims=True)

    def blend_mean(Q, H, M, b):
        R = Q.copy()
        R[..., :2] += np.float32(b) * (H[:, None] - M)
        return R

    for b in args.beta:
        if b == 0.0:
            continue
        row(f"B  mean -> history, beta {b:.2f}",
            blend_mean(P, hist_mean, pm, b), blend_mean(Ps, hist_mean_s, pms, b))
    for a in args.dmd:
        if a == 0.0:
            continue
        def mix(Q, D):
            R = Q.copy()
            R[..., :2] = (1 - np.float32(a)) * Q[..., :2] + np.float32(a) * D
            return R
        row(f"C  field -> DMD{args.ctx}, a {a:.2f}", mix(P, dmd), mix(Ps, dmd_s))

    Pm = P[..., :2].mean(1, keepdims=True); Pms = Ps[..., :2].mean(1, keepdims=True)
    for g in args.gain:
        Q = P.copy();  Q[..., :2] = Pm + np.float32(g) * (P[..., :2] - Pm)
        R = Ps.copy(); R[..., :2] = Pms + np.float32(g) * (Ps[..., :2] - Pms)
        row(f"D  fluctuation gain {g:.2f}", Q, R)

    em = np.linalg.norm((P[sel, ..., :2] - y[sel, ..., :2]).reshape(int(sel.sum()), -1), axis=1)
    ed = np.linalg.norm((dmd[sel] - y[sel, ..., :2]).reshape(int(sel.sum()), -1), axis=1)
    print(f"\nper-window error correlation, model vs DMD{args.ctx}: "
          f"{np.corrcoef(em, ed)[0, 1]:+.3f}   (1.0 = DMD adds nothing)")
    print(f"model time-mean rel-L2 {np.mean(np.linalg.norm((pm[sel,0]-y[sel,...,:2].mean(1)).reshape(int(sel.sum()),-1),axis=1)/np.linalg.norm(y[sel,...,:2].mean(1).reshape(int(sel.sum()),-1),axis=1)):.4f}"
          f"   history mean {np.mean(np.linalg.norm((hist_mean[sel]-y[sel,...,:2].mean(1)).reshape(int(sel.sum()),-1),axis=1)/np.linalg.norm(y[sel,...,:2].mean(1).reshape(int(sel.sum()),-1),axis=1)):.4f}")
    print(f"baseline wfinal {base:.3f}")


if __name__ == "__main__":
    main()
