"""WP-2 diagnostic: how much of the per-window error is phase/advection, not amplitude?

The wake is quasi-periodic shedding with a 60-75 frame period, and a 20-frame window fixes
neither the phase nor the convection speed. Before spending any training on a phase modality
(plan_20260810 WP-2), measure the ceiling: if a spatial re-registration of the PREDICTION
recovers little error, the branch dies here.

    python diag_phase.py --ckpt unet/fno_vl_s1.pth [--aoa 15 20]

Method. For each val_unseen window, roll the prediction by (dy, dx) and score it against the
target on a fixed interior region (rows 2..29, cols 4..59 -- the union margin of the shift
grid, so np.roll's wrap-around never enters the comparison and every shift sees the same
denominator ||Y_interior||):

  A) one best shift per WINDOW (all 20 frames together) -- static registration error;
  B) one best shift per OUTPUT FRAME -- advection: if the best dx grows with t, the model's
     convection speed is off, and the slope IS the speed error in cells/frame.

`recovered = 1 - e_shifted/e_0` is the fraction of interior rel-L2 a phase fix could buy at
most. Two controls calibrate the instrument:

  * persistence (repeat the last input frame): by output frame t the flow has really advanced
    t+1 frames, so B MUST recover a lot here -- if it does not, the diagnostic is too blunt
    to say anything about the model (positive control);
  * the model itself under A with dx,dy = 0 recovers 0 by construction (null control).

Shift grid: dx in [-6, 6] (streamwise; mean convection is ~1 cell/frame at U~0.15 m/s,
dx=0.00342 m, dt=10 ms), dy in [-2, 2].
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

DXS = np.arange(-6, 7)
DYS = np.arange(-2, 3)
MY, MX = 2, 6                      # interior margins = max |shift| per axis
RY, RX = slice(MY, 32 - MY), slice(MX, 64 - MX)


def interior_err(P: np.ndarray, Y: np.ndarray, per_frame: bool) -> np.ndarray:
    """rel-L2 on the interior. per_frame=False: (N,); True: (N, T)."""
    d, t = P[:, :, RY, RX, :2] - Y[:, :, RY, RX, :2], Y[:, :, RY, RX, :2]
    if per_frame:
        num = np.sqrt((d ** 2).sum((2, 3, 4)))
        den = np.sqrt((t ** 2).sum((2, 3, 4)))
    else:
        num = np.linalg.norm(d.reshape(len(P), -1), axis=1)
        den = np.linalg.norm(t.reshape(len(P), -1), axis=1)
    return num / den


def best_shift(P: np.ndarray, Y: np.ndarray, per_frame: bool):
    """Minimize interior error over the shift grid. Returns (err, dy, dx) arrays."""
    n, T = P.shape[:2]
    shape = (n, T) if per_frame else (n,)
    best = np.full(shape, np.inf)
    bdy = np.zeros(shape, np.int64)
    bdx = np.zeros(shape, np.int64)
    for dy in DYS:
        for dx in DXS:
            Ps = np.roll(P, (dy, dx), axis=(2, 3))
            e = interior_err(Ps, Y, per_frame)
            m = e < best
            best[m], bdy[m], bdx[m] = e[m], dy, dx
    return best, bdy, bdx


def report(tag: str, P: np.ndarray, Y: np.ndarray) -> None:
    e0 = interior_err(P, Y, False)
    eA, dyA, dxA = best_shift(P, Y, False)
    eBf, dyB, dxB = best_shift(P, Y, True)
    # Window-level error with the per-frame optimal shifts applied: recompose from the
    # per-frame numerators/denominators of the SHIFTED fields.
    numB = np.zeros(len(P))
    denB = np.zeros(len(P))
    for t in range(P.shape[1]):
        Pt = np.stack([np.roll(P[i, t], (dyB[i, t], dxB[i, t]), axis=(0, 1))
                       for i in range(len(P))])
        d = Pt[:, RY, RX, :2] - Y[:, t, RY, RX, :2]
        numB += (d ** 2).sum((1, 2, 3))
        denB += (Y[:, t, RY, RX, :2] ** 2).sum((1, 2, 3))
    eB = np.sqrt(numB / denB)

    recA = 1.0 - eA.mean() / e0.mean()
    recB = 1.0 - eB.mean() / e0.mean()
    # Advection signature: slope of the mean best dx against output-frame index.
    tt = np.arange(P.shape[1])
    slope = np.polyfit(tt, dxB.mean(0), 1)[0]
    print(f"\n== {tag} ==")
    print(f"  interior rel-L2         e0 {e0.mean():.4f}")
    print(f"  A one shift/window      e  {eA.mean():.4f}   recovered {100*recA:5.1f}%   "
          f"median (dy,dx) ({int(np.median(dyA))},{int(np.median(dxA))})   "
          f"|shift|>0 in {100*np.mean((dyA != 0) | (dxA != 0)):.0f}% of windows")
    print(f"  B one shift/frame       e  {eB.mean():.4f}   recovered {100*recB:5.1f}%   "
          f"mean dx by frame: t0 {dxB[:, 0].mean():+.2f} -> t19 {dxB[:, -1].mean():+.2f}   "
          f"slope {slope:+.3f} cells/frame")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[])
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--aoa", type=int, nargs="+", default=None)
    args = ap.parse_args()

    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    X, Y = du["x"], du["y"]
    if args.aoa:
        aoa = np.array([int(str(i).split(":")[0].split("_")[1]) for i in du["ids"]])
        sel = np.isin(aoa, args.aoa)
        X, Y = X[sel], Y[sel]
        print(f"val_unseen restricted to AoA {args.aoa}: {len(X)} windows")

    # Positive control: persistence carries a REAL, growing phase lag by construction.
    P_pers = np.repeat(X[:, -1:], Y.shape[1], axis=1)
    report("persistence (positive control -- B must recover a lot here)", P_pers, Y)

    if args.ckpt:
        import torch
        import _fno
        from eval_v9 import infer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
                  for c in args.ckpt]
        P = infer(models, X, dev, alpha=args.alpha)
        report(f"model ({', '.join(os.path.basename(c) for c in args.ckpt)})", P, Y)
        print("\nReading: if the model's recovered% (A and B) is small next to the positive "
              "control's, the error is amplitude/shape, not phase -- WP-2 dies here. If B "
              "recovers a lot with a clear per-frame dx trend, a convection/phase fix is live "
              "and the slope says how big it is.")


if __name__ == "__main__":
    main()
