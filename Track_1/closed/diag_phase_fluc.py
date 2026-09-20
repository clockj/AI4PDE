"""WP-2 diagnostic, second instrument: phase error of the FLUCTUATION field only.

diag_phase.py's whole-field shift came back null on the model AND on its positive control
(persistence, which carries a real ~0.44 cells/frame advection lag, recovered 0.7%): the
time-mean field dominates the norm and is damaged by any shift, so it pins the registration
at zero and the instrument cannot see fluctuation phase at all.

This version shifts ONLY the fluctuation part -- P' = mean_t(P) + roll(P - mean_t(P)) -- and
scores the fluctuation-only interior rel-L2, so a shedding-phase misregistration has nothing
anchoring it. Controls:

  * synthetic: prediction := target with its own fluctuation rolled by (dy, dx) = (1, 3).
    The instrument MUST report ~100% recovery at exactly (-1, -3), or it is broken.
  * persistence has zero fluctuation, so it drops out of this variant entirely.

For an advecting frozen pattern a temporal phase shift IS a streamwise spatial shift
(~0.44 cells/frame at U~0.15 m/s), so the dx range +-6 spans ~+-13 frames of shedding phase.

    python diag_phase_fluc.py --ckpt unet/fno_vl_s1.pth
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

from diag_phase import DXS, DYS, RY, RX  # noqa: E402  (same grid and interior)


def fluc(P: np.ndarray) -> np.ndarray:
    return P - P.mean(axis=1, keepdims=True)


def fluc_err(Pf: np.ndarray, Yf: np.ndarray) -> np.ndarray:
    d = Pf[:, :, RY, RX, :2] - Yf[:, :, RY, RX, :2]
    t = Yf[:, :, RY, RX, :2]
    return (np.linalg.norm(d.reshape(len(Pf), -1), axis=1)
            / np.linalg.norm(t.reshape(len(Pf), -1), axis=1))


def best_fluc_shift(Pf: np.ndarray, Yf: np.ndarray):
    best = np.full(len(Pf), np.inf)
    bdy = np.zeros(len(Pf), np.int64)
    bdx = np.zeros(len(Pf), np.int64)
    for dy in DYS:
        for dx in DXS:
            e = fluc_err(np.roll(Pf, (dy, dx), axis=(2, 3)), Yf)
            m = e < best
            best[m], bdy[m], bdx[m] = e[m], dy, dx
    return best, bdy, bdx


def report(tag: str, P: np.ndarray, Y: np.ndarray) -> None:
    Pf, Yf = fluc(P), fluc(Y)
    e0 = fluc_err(Pf, Yf)
    eA, dy, dx = best_fluc_shift(Pf, Yf)
    rec = 1.0 - eA.mean() / e0.mean()
    nz = (dy != 0) | (dx != 0)
    print(f"== {tag} ==")
    print(f"  fluc-only interior rel-L2  e0 {e0.mean():.4f} -> best-shift {eA.mean():.4f}   "
          f"recovered {100*rec:5.1f}%")
    print(f"  median (dy,dx) ({int(np.median(dy))},{int(np.median(dx))})   nonzero shift in "
          f"{100*nz.mean():.0f}% of windows   mean |dx| {np.abs(dx).mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    args = ap.parse_args()

    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    X, Y = du["x"], du["y"]

    # Instrument calibration: a pure (1, 3) fluctuation shift must be recovered ~fully.
    Ysyn = Y.mean(axis=1, keepdims=True) + np.roll(fluc(Y), (1, 3), axis=(2, 3))
    report("synthetic control: target with fluc rolled by (1,3) -- must recover ~100% at (-1,-3)",
           Ysyn, Y)

    import torch
    import _fno
    from eval_v9 import infer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    P = infer(models, X, dev, alpha=args.alpha)
    report(f"model ({', '.join(os.path.basename(c) for c in args.ckpt)})", P, Y)
    print("\nReading: recovered% here is the ceiling a shedding-phase/advection fix could buy "
          "on the fluctuation error (which carries ~68% of the squared field error). Small "
          "recovered% + calibrated control = the error is decorrelation/amplitude, and WP-2's "
          "phase-modality premise is dead.")


if __name__ == "__main__":
    main()
