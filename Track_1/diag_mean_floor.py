"""Is the time-mean error noise-limited, the way tke_score is?

    python diag_mean_floor.py

`diag_spectral.py` splits the unseen rel-L2 0.0888 into a 0.0503 time-mean part and a 0.0729
fluctuation part, and the fluctuation part is provably near its floor (coherent fraction
Gamma^2 = 0.308). That makes the time-mean the only remaining target -- but only if it is
actually predictable.

It might not be. The scorer compares against the target's *realized* 20-frame mean, and 20
frames is a few shedding periods, so that realized mean is the true mean flow plus a partially
averaged fluctuation. A predictor that knew the true mean flow exactly would still be wrong by
that sampling term. This measures it with the same half-window trick that produced the TKE
floor in report_20260730.md section 1.

Split each 20-frame target into two 10-frame halves with means m1, m2. For a stationary series
the two halves are near-independent, so Var(m1 - m2) = 2*Var(m_half) and the full 20-frame mean
has Var(m) = Var(m_half)/2 = Var(m1 - m2)/4. The sampling error of the target's own mean is
therefore 0.5*(m1 - m2) in rms, and no predictor can do better than that.

Frames are positively autocorrelated, so the halves are not fully independent and this
UNDERSTATES the floor: the true irreducible error is at least what is printed.
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = 20
PROBE_Y = [8, 10, 12, 14, 16, 18, 20, 22, 24]
PROBE_X = [13, 21, 29, 37]


def mvpe_of(mean_pred: np.ndarray, mean_tgt: np.ndarray) -> float:
    """scoring.mvpe_rel_l2 on already-time-averaged fields (N, H, W, C)."""
    cols = []
    for px in PROBE_X:
        pp = mean_pred[:, PROBE_Y, px, :2].reshape(len(mean_pred), -1)
        tt = mean_tgt[:, PROBE_Y, px, :2].reshape(len(mean_tgt), -1)
        cols.append(np.linalg.norm(pp - tt, axis=1) / np.linalg.norm(tt, axis=1).clip(1e-8))
    return float(np.mean(np.stack(cols)))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="'_sim' for the simulation validation set")
    a = ap.parse_args()
    d = np.load(os.path.join(HERE, f"val_unseen{a.suffix}.npz"))
    Y = d["y"][..., :2].astype(np.float64)

    m1, m2 = Y[:, :10].mean(1), Y[:, 10:].mean(1)      # (N, H, W, C)
    m = Y.mean(1)
    err = 0.5 * (m1 - m2)                               # rms sampling error of the 20-frame mean

    norm_Y = np.sqrt((Y ** 2).sum((1, 2, 3, 4)))
    rel_mean_floor = float(np.mean(np.sqrt(T * (err ** 2).sum((1, 2, 3))) / norm_Y))

    tag = f"val_unseen{a.suffix}"
    print(f"{'='*74}\ntime-mean sampling floor, {tag} ({len(Y)} windows)\n{'='*74}")
    print(f"  irreducible sampling floor of the 20-frame mean   {rel_mean_floor:.4f}")
    if not a.suffix:
        print(f"  measured mean-part rel-L2 of wp2i_s5             0.0503 -> reachable "
              f"{np.sqrt(max(0.0503**2 - rel_mean_floor**2, 0)):.4f}")

    print(f"\n{'='*74}\nMVPE floor at the wake probes\n{'='*74}")
    print(f"  MVPE between the two 10-frame half-means         {mvpe_of(m1, m2):.4f}")
    print(f"  MVPE of a perfect-true-mean predictor            {mvpe_of(m + err, m):.4f}"
          f"   <- the floor")

    # What the whole rel-L2 would be if the mean part were driven to its floor and the
    # fluctuation part stayed where it is.
    tot = float(np.hypot(rel_mean_floor, 0.0729))  # 0.0729 is the REAL model fluctuation part
    print(f"\n{'='*74}\nimplication\n{'='*74}")
    print(f"  rel-L2 if the mean part hits its floor    {tot:.4f}  (now 0.0888)")
    print(f"  rel_l2_score                              {100/(1+0.5*tot):.2f}  (now 95.76)")
    print(f"  final delta from rel_l2 alone             "
          f"{(100/(1+0.5*tot) - 100/(1+0.5*0.0888))/5:+.3f}")


if __name__ == "__main__":
    main()
