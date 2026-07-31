"""Does the data satisfy incompressibility, and how well?

A continuity penalty (div u = 0) is only useful if the TARGET satisfies it. Real PIV is a 2D
slice of a 3D flow, sampled on a coarse grid, with measurement noise -- all of which break the
2D divergence-free condition. Sim is a 2D CFD solve, so it should satisfy it nearly exactly.
The ratio between them says how strong a continuity penalty on real data can safely be.

    python divergence.py

Metric: rms(du/dx + dv/dy) / rms(|grad u|), dimensionless and grid-independent. 0 means
perfectly divergence-free; ~1 means the divergence is as large as a typical velocity gradient,
i.e. the constraint carries no information.
"""
from __future__ import annotations

import glob
import os

import h5py
import numpy as np

REAL = "/home/clockj/RealPDE/train_real/train_real"
SIM = "/home/clockj/RealPDE/train_sim/train_sim"
SUB = 2


def div_ratio(u, v, dx, dy):
    """u, v: (T, H, W). Central differences on the interior."""
    dudx = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dx)
    dvdy = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2 * dy)
    dudy = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dy)
    dvdx = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * dx)
    div = dudx + dvdy
    mag = np.sqrt(dudx ** 2 + dudy ** 2 + dvdx ** 2 + dvdy ** 2)
    return float(np.sqrt(np.mean(div ** 2)) / np.sqrt(np.mean(mag ** 2)))


def survey(root, label, sub, limit=6):
    files = sorted(glob.glob(os.path.join(root, "*.h5")))[:limit]
    if not files:
        print(f"{label}: no files under {root}")
        return
    print(f"\n{label}  ({root})")
    for path in files:
        with h5py.File(path, "r") as f:
            u = f["u"][:60, ::sub, ::sub].astype(np.float64)
            v = f["v"][:60, ::sub, ::sub].astype(np.float64)
            x = f["x"][::sub, ::sub].astype(np.float64)
            y = f["y"][::sub, ::sub].astype(np.float64)
        dx = float(np.mean(np.diff(x, axis=1)))
        dy = float(np.mean(np.diff(y, axis=0)))
        r_sub = div_ratio(u, v, dx, dy)

        # Same trajectory at NATIVE resolution: separates the physics from the discretization.
        with h5py.File(path, "r") as f:
            un = f["u"][:60].astype(np.float64)
            vn = f["v"][:60].astype(np.float64)
            xn = f["x"][:].astype(np.float64)
            yn = f["y"][:].astype(np.float64)
        r_nat = div_ratio(un, vn, float(np.mean(np.diff(xn, axis=1))),
                          float(np.mean(np.diff(yn, axis=0))))
        print(f"  {os.path.basename(path)[:-3]:14s} 32x64 {r_sub:6.3f}   native {r_nat:6.3f}")


def noise_fraction_of_divergence():
    """How much of the target's divergence field is temporally white, i.e. measurement noise?

    Central differencing amplifies white noise (the stencil divides by 2*dx = 0.0068), so even
    a modest velocity noise floor can dominate the divergence. If that fraction is high, a loss
    that MATCHES div(target) is a loss that fits sensor noise.
    """
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    for tag in ("unseen", "seen"):
        Y = np.load(_os.path.join(here, f"val_{tag}.npz"))["y"][..., :2].astype(np.float64)
        u, v = Y[..., 0], Y[..., 1]
        dudx = (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2]) / (2 * 0.003422)
        dvdy = (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2 * -0.003422)
        div = dudx + dvdy

        d = div - div.mean(1, keepdims=True)
        var = np.mean(d ** 2, axis=1)
        cov1 = np.mean(d[:, :-1] * d[:, 1:], axis=1)
        frac = 1.0 - cov1.sum() / var.sum()
        print(f"  {tag}: white-noise fraction of div(target) variance = {frac:.3f}")


if __name__ == "__main__":
    print("rms(div u) / rms(|grad u|)  -- 0 = divergence-free, ~1 = constraint is vacuous")
    survey(REAL, "REAL (PIV)", SUB)
    survey(SIM, "SIM (CFD)", SUB)
    print("\nIs the target's divergence signal or noise?")
    noise_fraction_of_divergence()
