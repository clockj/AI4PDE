"""Three diagnostics that gate the remaining Track 1 design space.

    python diag_spectral.py --ckpt unet/wp2i_s5.pth

(A) Decompose the unseen rel-L2 error into its time-mean and fluctuation parts. They are
    orthogonal under the time inner product, so the split is exact. This says whether the
    remaining error lives in the mean field (which MVPE also sees, and which is learnable)
    or in the fluctuation (which is phase prediction, and may be unreachable).

(B) Spectral coherence between predicted and target fluctuation, per radial wavenumber.
    Coherence measures PHASE agreement independently of amplitude; the gain ratio measures
    amplitude independently of phase. Together they say whether a second network is needed
    at all: high coherence with gain < 1 means the model already knows where the eddies are
    and merely under-predicts their strength, which a handful of scalars can fix.

(C) The score reachable from the stride-20 tiling alone. Windows tile each trajectory with
    INTERVAL == OUT_STEP == 20, so window k's target IS window k+1's input. Track 1's
    predict() receives every window at once, so this is a property of the protocol, not of
    any model. Measured here to size the gap between modelling and the leaderboard.

The grid is uniform with dx == dy (0.00171 m native, 0.00342 m at sub_s=2) and the 32x64
domain is twice as long as it is tall, so the isotropic radial index is sqrt(kx^2 + 4 ky^2)
with kx the FFT index along W=64 and ky along H=32. Both directions reach Nyquist at r = 32.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'submit_fno'))

T = 20


# --------------------------------------------------------------------------- (A)


def decompose(P: np.ndarray, Y: np.ndarray) -> dict:
    """Exact orthogonal split of the per-sample relative L2 into mean and fluctuation.

    ||P - Y||^2 = T * ||mean_t(P) - mean_t(Y)||^2 + ||f_P - f_Y||^2, so the two relative
    contributions add in quadrature to the reported rel-L2. Scored channels only.
    """
    P, Y = P[..., :2], Y[..., :2]
    ax = (1, 2, 3, 4)
    mP, mY = P.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    fP, fY = P - mP, Y - mY

    norm_Y = np.sqrt((Y ** 2).sum(ax))
    e_mean = np.sqrt(T * ((mP - mY) ** 2).sum(ax))
    e_fluc = np.sqrt(((fP - fY) ** 2).sum(ax))

    return dict(
        rel_total=float(np.mean(np.sqrt(e_mean ** 2 + e_fluc ** 2) / norm_Y)),
        rel_mean=float(np.mean(e_mean / norm_Y)),
        rel_fluc=float(np.mean(e_fluc / norm_Y)),
        # How much of the target's own energy is fluctuation. A predictor with a perfect
        # time-mean and zero fluctuation scores exactly this.
        fluc_frac=float(np.mean(np.sqrt((fY ** 2).sum(ax)) / norm_Y)),
        # How much of the target fluctuation the model captures coherently, in energy terms.
        # 0 = the model predicts no fluctuation or an uncorrelated one; 1 = exact.
        fluc_captured=float(np.mean(1.0 - ((fP - fY) ** 2).sum(ax) / (fY ** 2).sum(ax))),
    )


# --------------------------------------------------------------------------- (B)


def radial_index() -> tuple[np.ndarray, int]:
    """Isotropic radial wavenumber bin for each (ky, kx) of a 32x64 rfft2 grid."""
    ky = np.fft.fftfreq(32) * 32          # -16 .. 15
    kx = np.fft.rfftfreq(64) * 64         #   0 .. 32
    r = np.sqrt(ky[:, None] ** 2 * 4 + kx[None, :] ** 2)
    return np.rint(r).astype(int), 33     # bins 0..32, beyond is corner-only aliasing


def coherence(P: np.ndarray, Y: np.ndarray) -> dict:
    """Per-radial-k phase coherence and amplitude ratio of the FLUCTUATION field.

    gamma(k) = |<F_P conj(F_Y)>| / sqrt(<|F_P|^2> <|F_Y|^2>)  in [0, 1], phase only.
    gain(k)  = sqrt(<|F_P|^2> / <|F_Y|^2>)                     amplitude only.

    Averaged over windows, frames and the two scored channels. A model that predicted the
    conditional mean would show gamma -> 0 and gain -> 0 at every k it cannot resolve.
    """
    P, Y = P[..., :2], Y[..., :2]
    fP = P - P.mean(1, keepdims=True)
    fY = Y - Y.mean(1, keepdims=True)
    # (N, T, H, W, C) -> fft over the two spatial axes
    FP = np.fft.rfft2(fP, axes=(2, 3))
    FY = np.fft.rfft2(fY, axes=(2, 3))

    cross = (FP * np.conj(FY)).mean(axis=(0, 1, 4))   # (H, Wr) complex
    pp = (np.abs(FP) ** 2).mean(axis=(0, 1, 4))
    yy = (np.abs(FY) ** 2).mean(axis=(0, 1, 4))

    idx, nb = radial_index()
    g, gain, energy = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    for b in range(nb):
        m = idx == b
        if not m.any():
            continue
        c, a, y = cross[m].sum(), pp[m].sum(), yy[m].sum()
        g[b] = np.abs(c) / np.sqrt(a * y + 1e-30)
        gain[b] = np.sqrt(a / (y + 1e-30))
        energy[b] = y
    return dict(gamma=g, gain=gain, energy=energy / energy.sum())


# --------------------------------------------------------------------------- (C)


def transductive(X: np.ndarray, ids: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Window k's target is window k+1's input whenever both lie in the same trajectory.

    INTERVAL == OUT_STEP == 20 in the official windowing (local_eval.py:48-51), so this is
    exact, not approximate. The last window of each trajectory has no successor and keeps the
    model's prediction.
    """
    P = fallback.copy()
    traj = np.array([str(i).split(":")[0] for i in ids])
    hit = 0
    for k in range(len(X) - 1):
        if traj[k] == traj[k + 1]:
            P[k] = X[k + 1]
            hit += 1
    print(f"  windows whose target is present in the input array: {hit}/{len(X)} "
          f"({100 * hit / len(X):.1f}%)")
    return P


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=["unet/fno_vl_s1.pth"],
                    help="one or more members; they are ensembled exactly as the package does, "
                         "which matters here: gamma is a property of THE PREDICTION, so a "
                         "single member and an ensemble are different measurements")
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--set", default="unseen", choices=["unseen", "seen", "unseen_sim", "rein", "relo", "rehi"],
                    help="`rein`/`relo`/`rehi` are re_split.py's interior control and the two "
                         "Re-EXTRAPOLATION sets, and they only mean anything for a checkpoint "
                         "trained with re_split's --exclude. "
                         "`seen` is the trajectories the model TRAINED on. Comparing the "
                         "mean/fluctuation split across seen and unseen localizes the "
                         "generalization gap -- and 68%% of the squared error lives in the "
                         "fluctuation, so if the gap does too, then the fluctuation's SHAPE "
                         "and the generalization problem are the same problem.")
    ap.add_argument("--tiling", action="store_true",
                    help="also print (C), the score reachable from the stride-20 tiling. That "
                         "route is disclosed and unused; it is a property of the protocol.")
    args = ap.parse_args()

    import torch
    import _fno
    from eval_v9 import infer
    import sps9
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(os.path.join(HERE, f"val_{args.set}.npz"))
    X, Y, ids = d["x"], d["y"], d["ids"]
    m = [_fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
         for c in args.ckpt]
    P = infer(m, X, dev, args.alpha)
    tag = f"{len(m)} member(s), alpha={args.alpha}, val_{args.set}"

    print(f"\n{'='*72}\n(A) rel-L2 decomposition  --  {tag}\n{'='*72}")
    for label, pred in (("model", P), ("input time-mean", np.repeat(X.mean(1, keepdims=True),
                                                                    T, axis=1))):
        st = decompose(pred, Y)
        print(f"  {label:16s} rel-L2 {st['rel_total']:.4f} = mean {st['rel_mean']:.4f} "
              f"(+) fluc {st['rel_fluc']:.4f}   [quadrature]")
        print(f"  {'':16s} target fluctuation is {100*st['fluc_frac']:.1f}% of field norm; "
              f"model captures {100*st['fluc_captured']:.1f}% of its energy")

    print(f"\n{'='*72}\n(B) spectral coherence of the fluctuation, per radial wavenumber\n"
          f"{'='*72}")
    print("  gamma(k) = |<F_P conj(F_Y)>| / sqrt(<|F_P|^2><|F_Y|^2>) is computed from OUR")
    print("  PREDICTION, so it is what this model achieves, NOT what is achievable. A perfect")
    print("  predictor scores 1 here. The energy-weighted Gamma^2 below bounds a post-hoc")
    print("  per-wavenumber filter on this prediction and nothing else.")
    c = coherence(P, Y)
    print(f"  {'k':>3s} {'energy%':>8s} {'coherence':>10s} {'gain':>7s}   "
          f"(coherence = phase agreement, gain = |pred|/|target|)")
    for b in range(1, 33):
        if c["energy"][b] < 1e-5:
            continue
        print(f"  {b:3d} {100*c['energy'][b]:8.2f} {c['gamma'][b]:10.3f} {c['gain'][b]:7.3f}")
    g2 = float((c["energy"] * c["gamma"] ** 2).sum() / c["energy"].sum())
    print(f"  energy-weighted Gamma^2 = {g2:.4f}   ({tag})")

    if args.tiling:
        print(f"\n{'='*72}\n(C) transductive bound from the stride-20 tiling\n{'='*72}")
        from eval_wp import subscores   # pulls in the v7 kit path; only needed here
        Pt = transductive(X, ids, P)
        base, v = subscores(P, Y, 1.0), subscores(Pt, Y, 1.0)
        print(f"  {'':22s} {'relL2':>7s} {'TKE':>7s} {'MVPE':>7s} {'sps':>6s} {'FINAL':>7s}")
        for label, st in (("model alone", base), ("+ tiling structure", v)):
            print(f"  {label:22s} {st['rel_err']:7.4f} {st['tke_err']:7.4f} "
                  f"{st['mvpe_err']:7.4f} {st['sps']:6.2f} {st['final']:7.2f}")

if __name__ == "__main__":
    main()
