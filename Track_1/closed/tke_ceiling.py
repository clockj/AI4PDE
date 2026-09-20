"""How much of the TKE error is predictable, and how much is 20-frame sampling noise?

WP2 rescales the predicted fluctuation so that TKE(output) = 0.5*sum_c sigma_hat_c^2 exactly.
The TKE subscore then depends on ONE thing: how well a per-pixel variance field over 20 frames
can be predicted. This script bounds that from data alone, with no model involved.

    python tke_ceiling.py                 # writes report/tke_ceiling_<date>.txt

Estimators, weakest to strongest:
  input      TKE of the 20 INPUT frames -- available at inference, the trivial baseline
  traj       per-trajectory mean TKE field over all its windows -- a strong smooth prior,
             still legal at inference only if the trajectory could be identified
  oracle-lp  the target's own TKE, 3x3 box-smoothed -- not legal, measures how much of the
             field is spatially coherent
  oracle     the target's own TKE -- error 0 by definition, sanity check

`traj` is the number that matters: it is roughly the best any conditional-mean predictor of
the variance field can do, so the gap between it and 0 is irreducible sampling noise.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7"
spec = importlib.util.spec_from_file_location("sc", os.path.join(KIT, "scoring.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def rel(p, t):
    p = p.reshape(len(p), -1)
    t = t.reshape(len(t), -1)
    return float(np.mean(np.linalg.norm(p - t, axis=1) / np.linalg.norm(t, axis=1).clip(1e-8)))


def box3(a):
    """3x3 box filter over (N, H, W), edge-replicated. No scipy in this project's eval path."""
    p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return sum(p[:, i:i + a.shape[1], j:j + a.shape[2]]
               for i in range(3) for j in range(3)) / 9.0


def main() -> None:
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say(f"{'set':8s} {'estimator':12s} {'TKE relL2':>10s} {'tke_score':>10s}")
    say("-" * 44)
    for tag in ("unseen", "seen"):
        d = np.load(os.path.join(HERE, f"val_{tag}.npz"))
        X, Y, ids = d["x"], d["y"], d["ids"]
        traj = np.array([i.split(":")[0] for i in ids])

        ke_t = sc.kinetic_energy(Y[..., :2])
        ke_i = sc.kinetic_energy(X[..., :2])

        ke_traj = np.empty_like(ke_t)
        for name in np.unique(traj):
            m = traj == name
            ke_traj[m] = ke_t[m].mean(0)

        for label, est in [("input", ke_i), ("traj", ke_traj),
                           ("oracle-lp", box3(ke_t)), ("oracle", ke_t)]:
            e = rel(est, ke_t)
            say(f"{tag:8s} {label:12s} {e:10.3f} {sc.score_error(e):10.2f}")
        say()

    # How much does the 20-frame window itself limit things? Split each target window into two
    # 10-frame halves: their TKE fields estimate the same underlying statistics, so the
    # disagreement between them is pure sampling noise at half the sample count.
    d = np.load(os.path.join(HERE, "val_unseen.npz"))
    Y = d["y"]
    a = sc.kinetic_energy(Y[:, :10, ..., :2])
    b = sc.kinetic_energy(Y[:, 10:, ..., :2])
    say(f"10-frame half-window disagreement (unseen): relL2 {rel(a, b):.3f}")
    say("  -> two independent estimates of the SAME field differ by this much;")
    say("     the 20-frame version of this number is the floor no predictor can beat.")
    say()

    # PIV noise fraction. Real turbulence is temporally correlated, measurement noise is white,
    # so lag-1 autocovariance recovers the correlated part and the shortfall is the noise.
    for tag in ("unseen", "seen"):
        Y = np.load(os.path.join(HERE, f"val_{tag}.npz"))["y"][..., :2]
        d = Y - Y.mean(1, keepdims=True)
        var = np.mean(d ** 2, axis=1)
        cov1 = np.mean(d[:, :-1] * d[:, 1:], axis=1)
        frac = 1.0 - cov1.sum((0, 1, 2)) / var.sum((0, 1, 2))
        say(f"PIV white-noise fraction ({tag}): u {frac[0]:.3f}  v {frac[1]:.3f}")

        # Same estimate per element, turned into a floor on the FIELD metric: no predictor can
        # reproduce white measurement noise, so its norm is a lower bound on rel-L2.
        nvar = np.clip(var - cov1, 0.0, None)
        floor = float(np.sqrt(Y.shape[1] * nvar.sum() / (Y ** 2).sum()))
        say(f"  rel-L2 floor from that noise alone ({tag}): {floor:.3f} "
            f"-> rel_l2_score <= {sc.score_error(floor):.1f}")
    say("  -> fraction of the scored variance that is measurement noise, not flow.")

    path = os.path.join(HERE, "report", "tke_ceiling_20260729.txt")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
