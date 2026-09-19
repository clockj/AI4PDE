"""Score a checkpoint on an ANGLE-OF-ATTACK it never saw, with the shipped band machinery.

    python aoa_holdout_eval.py --ckpt unet/x.pth --holdout-aoa 10

WHY THIS EXISTS.  Every local instrument in this project holds out trajectories drawn from the
Reynolds axis (`re_split.py`, `kfold.py --split re`).  The competition's PRIVATE test set is
described by the organizers as "unseen parameters, novel angles-of-attack / Reynolds numbers"
(Codabench 17363 / 17385, and the site's evaluation line), and the top 3 are decided on it after
the organizers re-train the shortlisted methods.  R14's finding -- the hidden set is inside the
training speed range, the gate fired on none -- is a statement about the PUBLIC validation set
and does not cover the private one.  Nothing we have measures behaviour at an AoA the model has
never seen, and error follows AoA more strongly than anything else (out-of-fold rel_l2
0.052 / 0.053 / 0.086 / 0.106 / 0.121 at AoA 0..20).

WHAT IT MEASURES.  Two splits, both built from `train_real` filenames, never typed:

  eval  every trajectory at `--holdout-aoa`          the arm was trained without these
  fit   the arm's in-distribution held-out set,      the arm was trained without these too
        minus the held-out AoA

and it reports the four error subscores plus `sps` on BOTH, so the AoA effect is the difference
between them with training-set size, recipe and seed held fixed -- the same construction
`re_split.py` uses for the Re axis.

THE BAND.  A shipped band takes its error field from HELD-OUT TWINS of the shipped members
(`fit_band.py --err-ckpt`); a single screen run has no twins, so the band here is fitted on the
run's OWN in-distribution held-out windows (`--fit-set`) and reported on the held-out-AoA
windows.  That is honest -- neither split was trained on -- but it is NOT the shipped
construction, so `sps` here is comparable ACROSS ARMS of this screen and not against a board
round.  Everything else matches what ships: ALPHA 0.60, `--band-p` 0.45, inflate 1.39,
box-smooth 3, sigma_ref from the scored models.

THE MEAN-BLEND.  `submission.py`'s gated mean-blend (BLEND_BETA, `mean_blend_re.py`) replaces
beta of the predicted 20-frame time-mean with the INPUT window's measured one, gated on windows
whose mean speed falls outside the training range.  It was measured to win out of distribution
(relo rel-L2 0.1080 -> 0.1035) and to lose in it, and R14 shipped it gated and it fired on
nothing.  A novel AoA is out of distribution in a way the SPEED gate cannot see, so this script
reports three rows: no blend, the gate as shipped (which on an AoA holdout is expected to fire
on ~no window, and that is the finding), and the blend applied UNGATED at the same beta.

    python aoa_holdout_eval.py --ckpt unet/a.pth unet/b.pth --holdout-aoa 20 \
        --fit-set val_traj --alpha 0.60 --band-p 0.45 --beta 0.15
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
sys.path.insert(0, os.path.join(HERE, "unet"))

DATA = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
T_IN = T_OUT = 20
INTERVAL = 20
SUB = 2
HORIZON = T_IN + T_OUT


def names_at_aoa(aoa: int, data: str = DATA) -> list[str]:
    """Every train_real stem at this AoA, DERIVED FROM FILENAMES (never typed)."""
    return sorted(f[:-3] for f in os.listdir(data)
                  if f.endswith(".h5") and int(f[:-3].split("_")[1]) == aoa)


def all_names(data: str = DATA) -> list[str]:
    return sorted(f[:-3] for f in os.listdir(data) if f.endswith(".h5"))


def fit_names(which: str, held: list[str], data: str = DATA) -> list[str]:
    """The arm's in-distribution held-out trajectories, minus the held-out AoA."""
    if which == "val_traj":
        from model import VAL_TRAJ
        base = list(VAL_TRAJ)
    elif which == "re_excluded":
        import re_split
        base = re_split.excluded(data)
    else:
        raise SystemExit(f"unknown --fit-set {which}")
    h = set(held)
    return sorted(n for n in base if n not in h)


def windows(names: list[str], data: str = DATA, limit: int | None = None):
    """(x, y, ids) at 32x64, T_in = T_out = 20, stride 20 -- re_split.build's construction.

    `limit` is windows PER TRAJECTORY, so a smoke test still samples every trajectory in the
    split rather than reading only the first one.
    """
    import h5py
    xs, ys, ids = [], [], []
    for name in names:
        with h5py.File(os.path.join(data, f"{name}.h5"), "r") as f:
            u = f["u"][:, ::SUB, ::SUB].astype(np.float32)
            v = f["v"][:, ::SUB, ::SUB].astype(np.float32)
        d = np.stack([u, v, np.zeros_like(u)], -1)
        n = 0
        for t in range(0, len(d) - HORIZON + 1, INTERVAL):
            xs.append(d[t:t + T_IN]); ys.append(d[t + T_IN:t + HORIZON])
            ids.append(f"{name}:{t}"); n += 1
            if limit and n >= limit:
                break
    return (np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.array(ids))


def speed_range(names: list[str], data: str = DATA) -> tuple[float, float]:
    """Per-WINDOW mean streamwise speed range over the TRAINING trajectories.

    Per window, not per trajectory: the gate is evaluated per window and windows fluctuate
    around their trajectory's mean, so the per-trajectory range under-states it and the gate
    then fires on in-range windows (`verify/blend_path.py` caught exactly that in R14).
    """
    import h5py
    lo, hi = np.inf, -np.inf
    for name in names:
        with h5py.File(os.path.join(data, f"{name}.h5"), "r") as f:
            u = f["u"][:, ::SUB, ::SUB].astype(np.float32)
        n = (len(u) - HORIZON + 1)
        for t in range(0, max(n, 0), INTERVAL):
            s = float(u[t:t + T_IN].mean())
            lo, hi = min(lo, s), max(hi, s)
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one checkpoint, or several to score as ONE ensemble")
    ap.add_argument("--holdout-aoa", type=int, required=True, choices=[0, 5, 10, 15, 20])
    ap.add_argument("--fit-set", default="val_traj", choices=["val_traj", "re_excluded"],
                    help="which in-distribution held-out list the arm ALSO excluded; the band "
                         "is fitted on it. val_traj = model.VAL_TRAJ (the 09-14 screen "
                         "protocol, 8 trajectories); re_excluded = re_split.excluded() (26).")
    ap.add_argument("--alpha", type=float, default=0.60, help="the shipped compose alpha")
    ap.add_argument("--band-p", type=float, default=0.45)
    ap.add_argument("--inflate", type=float, default=1.39)
    ap.add_argument("--band-smooth", type=int, default=3)
    ap.add_argument("--inflate-sweep", type=float, nargs="*", default=[],
                    help="extra band stretches to report after the main table: the band is refitted "
                         "on the fit split at each and `sps` is read on BOTH splits, so the width "
                         "that is right for a novel angle arrives with its in-distribution price")
    ap.add_argument("--ms", type=float, default=5.08,
                    help="board ms/window used only for `time`; identical across rows, so it "
                         "shifts wfinal rather than reordering the arms")
    ap.add_argument("--beta", type=float, default=0.15, help="mean-blend beta (R14's value)")
    ap.add_argument("--ref", type=float, default=0.02,
                    help="gate ramp width as a fraction of the training speed range "
                         "(submission.py's BLEND_REF)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the windows PER TRAJECTORY -- for a CPU smoke test only")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    import _fno
    import final_v9
    import sps9
    from sps9 import optimal_band, prep, sc, sps_score
    from mean_blend_re import blend, gate

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    held = names_at_aoa(a.holdout_aoa)
    fit = fit_names(a.fit_set, held)
    train = [n for n in all_names() if n not in set(held) | set(fit)]
    print(f"holdout AoA {a.holdout_aoa}: {len(held)} trajectories  |  fit split "
          f"({a.fit_set} minus AoA {a.holdout_aoa}): {len(fit)}  |  training: {len(train)}")
    print(f"  held: {' '.join(held)}")
    print(f"  fit : {' '.join(fit)}")

    Xh, Yh, idh = windows(held, limit=a.limit)
    Xf, Yf, idf = windows(fit, limit=a.limit)
    print(f"  windows: held-out AoA {len(Xh)}   fit {len(Xf)}   device {dev}")

    m = [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev) for c in a.ckpt]

    def infer(X):
        uv = _fno.predict_fno(m, X, dev, alpha=a.alpha)
        return np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)

    def sigma(X):
        s = np.empty((len(X), 1, 32, 64, 2), dtype=np.float32)
        _fno.predict_fno(m, X, dev, alpha=a.alpha, sigma_out=s)
        return np.concatenate([s, np.zeros_like(s[..., :1])], -1).astype(np.float32)

    Pf, Ph = infer(Xf), infer(Xh)

    # --- the band, exactly fit_band.py's construction, on the fit split ---------------------
    sig_f = sigma(Xf)
    sigma_ref = np.maximum(sig_f.mean(0, keepdims=True), 1e-8).astype(np.float32)
    ratio_f = np.clip(sig_f / sigma_ref, 0.3, 3.0)
    As, as_, scored_s, ns = prep(Pf, Yf)

    def fit_base(inflate):
        Adef = np.float32(inflate) * As
        if a.band_p:
            Adef = (Adef / ratio_f[..., :2] ** a.band_p).astype(np.float32)
        base = optimal_band(Adef, as_, scored_s)
        if a.band_smooth > 1:
            k, o = a.band_smooth, base.astype(np.float64)
            for ax in (1, 2, 3):
                pad = [(0, 0)] * 5
                pad[ax] = (k // 2, k // 2)
                q = np.pad(o, pad, mode="edge")
                o = np.mean([np.take(q, range(i, i + base.shape[ax]), axis=ax) for i in range(k)],
                            axis=0)
            base = o.astype(np.float32)
        return base

    base = fit_base(a.inflate)
    ratio_h = np.clip(sigma(Xh) / sigma_ref, 0.3, 3.0)
    half_h = (base * ratio_h[..., :2] ** a.band_p).astype(np.float32)
    half_f = (base * ratio_f[..., :2] ** a.band_p).astype(np.float32)

    t_score = sc.score_time(a.ms / 1000.0)

    def row(tag, P, Y, half):
        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        mv = float(sc.mvpe_rel_l2(P, Y))
        U = prep(P, Y)
        five = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), t_score,
                sps_score(U[0], U[1], U[2], U[3], half)]
        wf = float(final_v9.predict(five)[0])
        print(f"{tag:34s} {r:7.4f} {k:7.4f} {mv:7.4f} | {five[0]:6.2f} {five[1]:6.2f} "
              f"{five[2]:6.2f} {five[3]:6.2f} {five[4]:6.2f} | {wf:7.3f}")
        return dict(tag=tag, rel=r, tke=k, mvpe=mv, sub=five, wfinal=wf)

    hdr = (f"{'split':34s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} | {'rel':>6s} {'tke':>6s} "
           f"{'mvpe':>6s} {'time':>6s} {'sps':>6s} | {'wfinal':>7s}")
    print(); print(hdr); print("-" * len(hdr))
    out = [row(f"fit ({a.fit_set}, in-distribution)", Pf, Yf, half_f),
           row(f"HELD-OUT AoA {a.holdout_aoa}", Ph, Yh, half_h)]

    # --- the gated mean-blend on the same windows ------------------------------------------
    lo, hi = speed_range(train)
    g = gate(Xh, lo, hi, a.ref)
    print(f"\ntraining per-window speed range [{lo:.6f}, {hi:.6f}] (ref {a.ref}); "
          f"gate fires on {int((g > 0).sum())}/{len(g)} held-out-AoA windows, "
          f"mean weight {float(g.mean()):.4f}")
    out.append(row(f"  + blend beta={a.beta} GATED", blend(Ph, Xh, a.beta * g), Yh, half_h))
    out.append(row(f"  + blend beta={a.beta} ungated", blend(Ph, Xh, a.beta), Yh, half_h))
    # THE IN-DISTRIBUTION PRICE of shipping the blend ungated (forge 09-19, main). The gate fires on
    # 0/756 and 42/643 held-out-AoA windows -- a novel ANGLE is invisible to a SPEED gate -- so if the
    # blend is to help the private set it has to ship ungated, and then it also applies to every
    # in-distribution window. This row is what that costs, on the fit split, at the same beta.
    out.append(row(f"  fit + blend beta={a.beta} ungated", blend(Pf, Xf, a.beta), Yf, half_f))
    # THE BAND WIDTH A NOVEL ANGLE WANTS (main 09-19). `--inflate` 1.39 is the PUBLIC hidden set's
    # measured tail; the private set is out of distribution, where the error field is larger than
    # the fit split's by more than that, and `sps` is the subscore that pays for it (48 -> 38-40 on
    # the AoA-20 holdout). Same predictions, band refitted per stretch, both splits.
    sweep = []
    if a.inflate_sweep:
        Uf, Uh = prep(Pf, Yf), prep(Ph, Yh)
        print(f"\n{'inflate':>8s} {'sps fit':>8s} {'sps HELD-OUT':>13s}")
        for f_ in a.inflate_sweep:
            b = fit_base(f_)
            sf = sps_score(Uf[0], Uf[1], Uf[2], Uf[3],
                           (b * ratio_f[..., :2] ** a.band_p).astype(np.float32))
            sh = sps_score(Uh[0], Uh[1], Uh[2], Uh[3],
                           (b * ratio_h[..., :2] ** a.band_p).astype(np.float32))
            print(f"{f_:8.2f} {sf:8.3f} {sh:13.3f}")
            sweep.append(dict(inflate=f_, sps_fit=float(sf), sps_held=float(sh)))
    print("\nThe band is fitted on this run's own in-distribution held-out windows, not on "
          "held-out twins\nof the scored models, so `sps` compares ACROSS ARMS of this screen "
          "and not against a board round.")

    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump(dict(ckpt=a.ckpt, holdout_aoa=a.holdout_aoa, fit_set=a.fit_set,
                           alpha=a.alpha, band_p=a.band_p, inflate=a.inflate, ms=a.ms,
                           beta=a.beta, ref=a.ref, n_held=len(Xh), n_fit=len(Xf),
                           speed_range=[lo, hi], gate_fired=int((g > 0).sum()),
                           rows=out, inflate_sweep=sweep), f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
