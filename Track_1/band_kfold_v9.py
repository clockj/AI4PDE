"""Fit the SPS band's shape on k-fold out-of-fold errors instead of on 126 windows.

    python band_kfold_v9.py --fold-ckpt unet/kfre_w96L8do45sim_f{0..4}.pth \
        --ckpt <the 4 all-data members> --err-ckpt <the 3 held-out twins> \
        --incumbent ../sub_candidates/band_r17_simx4.npz --alpha 0.60

WHY. The shipped band is `base`, an (1, 20, 32, 64, 3) array of per-element half-widths, and
`optimal_band` sets each element to one of its OWN observed |E| values -- an order statistic
over the fitting windows, element by element. The fitting split is val_seen restricted to
AoA 15-20: **126 windows for 81 920 free parameters.** That is why round 10b needed a k=3 box
blur, chosen by leave-one-trajectory-out CV, just to keep the estimate from overfitting, and
why five separate "the band is closed" verdicts were all about band FORM -- the exponent, the
smoothing, off-centre intervals, the exact asymmetric ceiling. **None of them was about the
SIZE of the fitting sample**, and sps is 0.250 of the board weight sitting at 41.33 against a
ceiling of 70.61, so it is 7.3 of the ~13 points still available.

The five Re-split k-fold models of the shipped recipe already exist. Each was trained on 4/5 of
the trajectories, so pooling their out-of-fold predictions gives an honest error field over
every trajectory -- ~10x the fitting sample at the same AoA, with no training.

TWO THINGS THIS MUST NOT DO, and how each is handled:

  * **Contaminate the reporting split.** The k-fold covers all 82 trajectories, 8 of which are
    val_unseen. Those 8 are DROPPED from the fit, so the challenger is fitted on the same 74
    trajectories as the incumbent and reported on the same disjoint 8.
  * **Import the wrong scale.** A fold model trained on 4/5 of the data has larger errors than
    the shipped 4-member ensemble, so the k-fold band would be uniformly too wide. Only the
    SHAPE is taken from it; a single scalar `c` is fitted on the incumbent's own 126-window
    split, where one parameter is well determined. `sigma_ref` and `p` are the shipped models'
    and are carried over untouched, so the deployed rule `half = c * shape * (sigma/sigma_ref)^p`
    differs from the incumbent in exactly one factor.

The smoothing width is chosen by leave-one-TRAJECTORY-out CV inside the fitting pool, never on
the reporting split -- the unsmoothed band IS the in-sample optimum by construction, so it
cannot be selected in-sample (round 10b).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import sps9
from sps9 import optimal_band, prep, sc, sps_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
KFOLD_DIR = os.environ.get("RPDE_KFOLD_DIR", "/fs/scratch/PAS1187/qwu/tmp/realpde/kfold_re")


def box_blur(b: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur over (T, H, W); channels stay independent. k=1 is the identity."""
    if k <= 1:
        return b.astype(np.float32)
    o = b.astype(np.float64)
    for ax in (1, 2, 3):
        pad = [(0, 0)] * 5
        pad[ax] = (k // 2, k // 2)
        q = np.pad(o, pad, mode="edge")
        o = np.mean([np.take(q, range(i, i + b.shape[ax]), axis=ax) for i in range(k)], axis=0)
    return o.astype(np.float32)


def traj_of(ids) -> np.ndarray:
    return np.array([str(i).split(":")[0] for i in ids])


def aoa_of(ids) -> np.ndarray:
    return np.array([int(str(i).split(":")[0].split("_")[1]) for i in ids])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-ckpt", nargs="+", required=True, help="the K k-fold checkpoints")
    ap.add_argument("--ckpt", nargs="+", required=True, help="shipped all-data members (sigma)")
    ap.add_argument("--err-ckpt", nargs="+", required=True, help="their held-out twins")
    ap.add_argument("--incumbent", required=True, help="the band .npz this is judged against")
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--aoa", type=int, nargs="+", default=[15, 20])
    ap.add_argument("--inflate", type=float, default=1.39)
    ap.add_argument("--smooth", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--kfold-dir", default=KFOLD_DIR)
    ap.add_argument("--out", default=None, help="write the winning band here (optional)")
    args = ap.parse_args()

    import torch
    import _fno

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = np.float32(args.inflate)

    def load(cs):
        return [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
                for c in cs]

    def infer(models, X, want_sigma=False):
        s = np.empty((len(X), 1, 32, 64, 2), np.float32) if want_sigma else None
        uv = _fno.predict_fno(models, X, dev, alpha=args.alpha, sigma_out=s)
        P = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1).astype(np.float32)
        if not want_sigma:
            return P
        return P, np.concatenate([s, np.zeros_like(s[..., :1])], -1).astype(np.float32)

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    unseen_traj = set(traj_of(du["ids"]).tolist())

    # ---- the shipped models: sigma_ref and p come from the incumbent, untouched -------------
    inc = np.load(args.incumbent if os.path.isabs(args.incumbent)
                  else os.path.join(HERE, args.incumbent))
    base_inc, sigma_ref, p = inc["base"], inc["sigma_ref"], float(inc["p"])
    print(f"incumbent {os.path.basename(args.incumbent)}: p {p:.2f}  "
          f"median half {np.median(base_inc[..., :2]):.5f}")

    m = load(args.ckpt)
    m_err = load(args.err_ckpt)

    # ---- the incumbent's fitting split, which also fits the challenger's single scalar ------
    keep_s = np.isin(aoa_of(ds["ids"]), args.aoa)
    Pf, Yf = infer(m_err, ds["x"])[keep_s], ds["y"][keep_s]
    _, sig_f = infer(m, ds["x"], want_sigma=True)
    ratio_f = np.clip(sig_f[keep_s] / sigma_ref, 0.3, 3.0)
    Af, af, sf, nf = prep(Pf, Yf)
    Adef_f = (f * Af / ratio_f[..., :2] ** p).astype(np.float32)

    # ---- the reporting split, untouched by any fit ------------------------------------------
    Pu, sig_u = infer(m, du["x"], want_sigma=True)
    keep_u = np.isin(aoa_of(du["ids"]), args.aoa)
    Pu, Yu = Pu[keep_u], du["y"][keep_u]
    ratio_u = np.clip(sig_u[keep_u] / sigma_ref, 0.3, 3.0)
    Au, au, su, nu = prep(Pu, Yu)
    dm, tk, mv = sps9.sample_metrics(Pu, Yu)
    a_b = sps9.weights_from_metrics(f * dm, f * tk, f * mv)

    def report(band: np.ndarray) -> float:
        """sps on val_unseen at the chosen AoA, with the deployed per-window rule."""
        half = (band[..., :2] * ratio_u[..., :2] ** p).astype(np.float32)
        return sps_score(f * Au, a_b, su, nu, half)

    print(f"\nreporting on val_unseen AoA {args.aoa}: {int(keep_u.sum())} windows")
    print(f"  INCUMBENT                                     sps {report(base_inc):7.3f}")

    # ---- the challenger: pooled out-of-fold errors, val_unseen trajectories DROPPED ---------
    A_l, a_l, s_l, r_l, t_l = [], [], [], [], []
    for i, c in enumerate(args.fold_ckpt):
        d = np.load(os.path.join(args.kfold_dir, f"fold{i}.npz"))
        tr, ao = traj_of(d["ids"]), aoa_of(d["ids"])
        sel = np.isin(ao, args.aoa) & ~np.isin(tr, list(unseen_traj))
        if not sel.any():
            continue
        mk = load([c])
        P, sg = infer(mk, d["x"][sel], want_sigma=True)
        del mk
        A_i, a_i, s_i, _ = prep(P, d["y"][sel])
        A_l.append(A_i); a_l.append(a_i); s_l.append(s_i)
        r_l.append(np.clip(sg / sigma_ref, 0.3, 3.0))
        t_l.append(tr[sel])
        print(f"  fold {i}: {int(sel.sum())} windows kept")
    A_k = np.concatenate(A_l); a_k = np.concatenate(a_l)
    s_k = np.concatenate(s_l); r_k = np.concatenate(r_l); t_k = np.concatenate(t_l)
    Adef_k = (f * A_k / r_k[..., :2] ** p).astype(np.float32)
    print(f"  pooled {len(A_k)} windows / {len(np.unique(t_k))} trajectories "
          f"({len(A_k)/max(len(Af),1):.1f}x the incumbent's fitting sample)")

    # ---- smoothing width by leave-one-TRAJECTORY-out CV *inside the pool* -------------------
    names = np.unique(t_k)
    # Grouped CV, not leave-one-out: `optimal_band` sorts every element's observed |E| across
    # the fitting windows, so on a 10x pool it is the expensive step and 74 x 4 refits would
    # dominate the job. Five trajectory groups give the same answer for a one-dimensional
    # choice at 1/15th the cost, and the grouping is by TRAJECTORY because windows tile one at
    # stride 20 and are not exchangeable.
    rng = np.random.default_rng(0)
    grp = {nm: int(g) for nm, g in zip(names, rng.permutation(len(names)) % 5)}
    gid = np.array([grp[t] for t in t_k])
    print("\n  grouped-trajectory CV inside the k-fold pool (picks the blur, never the "
          "reporting split):")
    cv = {}
    for k in args.smooth:
        tot = num = 0.0
        for g in range(5):
            hold = gid == g
            if not hold.any() or hold.all():
                continue
            b = box_blur(optimal_band(Adef_k[~hold], a_k[~hold], s_k[~hold]), k)
            nh = int(np.count_nonzero(s_k[hold]))
            num += nh * sps_score(f * A_k[hold], a_k[hold], s_k[hold], nh,
                                  (b[..., :2] * r_k[hold][..., :2] ** p).astype(np.float32))
            tot += nh
        cv[k] = num / tot
        print(f"    smooth k={k}: CV sps {cv[k]:7.3f}", flush=True)
    k_best = max(cv, key=cv.get)
    print(f"    -> k = {k_best}")
    shape = box_blur(optimal_band(Adef_k, a_k, s_k), k_best)
    print(f"    -> k = {k_best}")

    # ---- one scalar, fitted where the incumbent is fitted ------------------------------------
    print("\n  scalar c on the incumbent's own 126-window split (one parameter, so it is "
          "well determined there):")
    grid = np.linspace(0.6, 1.8, 25)
    best_c, best_v = 1.0, -1.0
    for c in grid:
        h = (c * shape[..., :2] * ratio_f[..., :2] ** p).astype(np.float32)
        v = sps_score(f * Af, af, sf, nf, h)
        if v > best_v:
            best_c, best_v = float(c), v
    print(f"    c = {best_c:.3f}   (in-sample sps {best_v:.3f})")

    band = np.concatenate([best_c * shape[..., :2],
                           np.zeros_like(shape[..., :1])], -1).astype(np.float32)
    r_inc, r_new = report(base_inc), report(band)
    print(f"\n  CHALLENGER  k-fold shape, k={k_best}, c={best_c:.3f}   sps {r_new:7.3f}"
          f"   ({r_new - r_inc:+.3f} vs the incumbent)")
    print(f"  median half-width  incumbent {np.median(base_inc[..., :2]):.5f}   "
          f"challenger {np.median(band[..., :2]):.5f}")
    if args.out and r_new > r_inc:
        np.savez(args.out, base=band, sigma_ref=sigma_ref, p=np.float32(p))
        print(f"  wrote {args.out}")
    elif args.out:
        print("  NOT written: the challenger did not beat the incumbent out of sample")


if __name__ == "__main__":
    main()
