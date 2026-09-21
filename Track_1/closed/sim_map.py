"""Gate 0 for the sim+real bet: measure the sim -> real domain gap, PAIRED.

Every real (Re, AoA) has an EXACT sim twin (sim is a strict superset: 100 conditions vs 82,
adding Re 15225 and 27975 plus the holes in real's grid). So the gap can be measured per
condition instead of per distribution, which is what makes a domain MAP fittable at all.

CLAUDE.md records the gap as "condition-independent rel-L2 0.38-0.47 after self-normalization"
and concludes sim is a weak prior. That number was measured WITHOUT the field-of-view mask.
Real PIV has ~9.9% of u,v elements exactly zero (outside the FOV, inside the airfoil body) and
sim has none, so those elements are pure numerator in a relative L2: sqrt(0.099) ~ 0.31 of the
norm on their own. This script separates the three candidate causes:

  1. SCALE      -- sim is non-dimensional, real is m/s. Fix: per-condition speed match.
  2. SUPPORT    -- sim has no FOV mask. Fix: paint real's zero set onto sim.
  3. RESOLUTION -- sim resolves eddies the PIV interrogation window cannot (STD_v/MEAN_u is
                   0.25 sim vs 0.103 real). Fix: a low-pass fitted from the radial spectra.

Anything left after all three is a genuine physics gap, and that is the number that decides
whether joint training can work.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

SIM = "/users/PAS1187/qwu/RealPDE/train_sim/train_sim"
REAL = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
SUB = 2


def rel(a: np.ndarray, b: np.ndarray, m: np.ndarray | None = None) -> float:
    """Relative L2 of a against b, optionally over a mask only."""
    if m is not None:
        a, b = a[m], b[m]
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def load(path: str, keys=("u", "v")) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return np.stack([f[k][:].astype(np.float32) for k in keys], -1)


def radial_spectrum(fluct: np.ndarray, nb: int = 16):
    """Energy per radial wavenumber bin of a (T, H, W, C) fluctuation field."""
    h, w = fluct.shape[1:3]
    ky = np.fft.fftfreq(h)[:, None] * h
    kx = np.fft.fftfreq(w)[None, :] * w
    kr = np.sqrt(ky ** 2 + kx ** 2)
    edges = np.linspace(0, kr.max() + 1e-6, nb + 1)
    ft = np.fft.fft2(fluct, axes=(1, 2))
    e = (np.abs(ft) ** 2).sum(-1).mean(0)  # (H, W), summed over channels, averaged over time
    out = np.zeros(nb)
    for i in range(nb):
        s = (kr >= edges[i]) & (kr < edges[i + 1])
        out[i] = e[s].mean() if s.any() else np.nan
    return out, 0.5 * (edges[:-1] + edges[1:])


def gauss_lowpass(x: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic Gaussian low-pass in the Fourier domain on (T, H, W, C). sigma in grid cells."""
    if sigma <= 0:
        return x
    h, w = x.shape[1:3]
    ky = np.fft.fftfreq(h)[:, None] * 2 * np.pi
    kx = np.fft.fftfreq(w)[None, :] * 2 * np.pi
    g = np.exp(-0.5 * (sigma ** 2) * (ky ** 2 + kx ** 2))[None, :, :, None]
    return np.real(np.fft.ifft2(np.fft.fft2(x, axes=(1, 2)) * g, axes=(1, 2))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="matched pairs to use (0 = all 82)")
    ap.add_argument("--frames", type=int, default=400, help="sim frames to read")
    ap.add_argument("--sigma", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0],
                    help="low-pass widths to try, in 64x128 grid cells")
    args = ap.parse_args()

    names = [os.path.basename(p)[:-3] for p in sorted(glob.glob(os.path.join(REAL, "*.h5")))]
    if args.n:
        names = names[:: max(len(names) // args.n, 1)][: args.n]
    print(f"{len(names)} matched (Re, AoA) pairs: {' '.join(names)}\n", flush=True)

    rows, spec_s, spec_r = [], [], []
    for nm in names:
        r = load(os.path.join(REAL, nm + ".h5"))[:, ::SUB, ::SUB]
        s = load(os.path.join(SIM, nm + ".h5"))[: args.frames, ::SUB, ::SUB]

        # --- 1. scale: put sim at real's measured freestream speed for this condition.
        s = s * (r[..., 0].mean() / s[..., 0].mean())

        # --- 2. support: real's zero set is the PIV field of view AND the airfoil body.
        # Exactly zero in EVERY frame of both channels is the mask; a live pixel is never
        # identically zero across 607 frames.
        mask = ~np.all(r == 0, axis=(0, 3))          # (H, W) True where real measures
        frac = 1.0 - mask.mean()

        mr, ms = r.mean(0), s.mean(0)                # time-mean fields (H, W, 2)
        sr, ss = r.std(0), s.std(0)                  # per-pixel temporal std

        row = dict(name=nm, zero=frac,
                   mean_raw=rel(ms, mr), mean_msk=rel(ms, mr, mask),
                   std_raw=rel(ss, sr), std_msk=rel(ss, sr, mask),
                   ti_r=float(r[..., 1][:, mask].std() / r[..., 0].mean()),
                   ti_s=float(s[..., 1][:, mask].std() / s[..., 0].mean()))
        # --- 3. resolution: how much low-pass makes sim's masked std field match real's.
        for sg in args.sigma:
            sf = gauss_lowpass(s, sg) if sg else s
            row[f"std_{sg}"] = rel(sf.std(0), sr, mask)
        rows.append(row)

        es, k = radial_spectrum(s[:, :, :, :] - ms, 16)
        er, _ = radial_spectrum(r - mr, 16)
        spec_s.append(es)
        spec_r.append(er)
        print(f"  {nm:>10}  zero {frac:5.1%}   mean {row['mean_raw']:.3f} -> "
              f"{row['mean_msk']:.3f} masked   std {row['std_raw']:.3f} -> "
              f"{row['std_msk']:.3f}   TI real {row['ti_r']:.3f} sim {row['ti_s']:.3f}",
              flush=True)

    def avg(key):
        return float(np.mean([r[key] for r in rows]))

    print(f"\n=== {len(rows)} pairs, means =========================================")
    print(f"zero fraction of real u,v          {avg('zero'):.1%}")
    print(f"mean field   rel-L2  raw           {avg('mean_raw'):.4f}")
    print(f"mean field   rel-L2  masked        {avg('mean_msk'):.4f}"
          f"   ({100*(1-avg('mean_msk')/avg('mean_raw')):.0f}% of the gap was the mask)")
    print(f"per-pixel std rel-L2 raw           {avg('std_raw'):.4f}")
    print(f"per-pixel std rel-L2 masked        {avg('std_msk'):.4f}")
    print(f"transverse TI  real {avg('ti_r'):.4f}   sim {avg('ti_s'):.4f} "
          f"({avg('ti_s')/avg('ti_r'):.2f}x)")

    print("\nlow-pass sweep (masked per-pixel std rel-L2, 32x64 cells):")
    best = min(args.sigma, key=lambda sg: avg(f"std_{sg}"))
    for sg in args.sigma:
        print(f"  sigma {sg:4.1f}   {avg(f'std_{sg}'):.4f}" + ("   <-- best" if sg == best else ""))

    es, er = np.mean(spec_s, 0), np.mean(spec_r, 0)
    _, k = radial_spectrum(np.zeros((1, 32, 64, 2), np.float32), 16)
    print("\nradial fluctuation energy, sim / real (32x64 grid):")
    for i in range(len(k)):
        if np.isfinite(es[i]) and er[i] > 0:
            print(f"  k {k[i]:5.1f}   sim/real {es[i]/er[i]:8.3f}   "
                  f"H(k)=sqrt(real/sim) {np.sqrt(er[i]/es[i]):.3f}")


if __name__ == "__main__":
    main()


def shape_probe(names, frames=400):
    """Can sim teach SHAPE, given that it cannot teach the mean or the amplitude?

    The gap measured above is in the two quantities a joint loss can be made blind to: the
    time-mean field (0.40 rel-L2 off) and the fluctuation amplitude (2.9-5.6x too energetic).
    What is left is WHERE the fluctuation sits and what its spectral shape is -- 'shape' in
    the round-10 oracle decomposition, and under the nine-round weights the most valuable
    reachable component. This measures those two, amplitude divided out.
    """
    out = []
    for nm in names:
        r = load(os.path.join(REAL, nm + ".h5"))[:, ::SUB, ::SUB]
        s = load(os.path.join(SIM, nm + ".h5"))[:frames, ::SUB, ::SUB]
        s = s * (r[..., 0].mean() / s[..., 0].mean())
        mask = ~np.all(r == 0, axis=(0, 3))
        sr, ss = r.std(0), s.std(0)
        # unit-norm the std fields: only their SHAPE is compared
        nr = sr / np.linalg.norm(sr[mask]);  ns = ss / np.linalg.norm(ss[mask])
        d_shape = rel(ns, nr, mask)
        # a shape-blind reference: how far is real's own std field from a flat one?
        flat = np.ones_like(nr);  flat /= np.linalg.norm(flat[mask])
        d_flat = rel(flat, nr, mask)
        # and the persistence-class reference: real's FIRST half vs its SECOND half
        h = len(r) // 2
        a, b = r[:h].std(0), r[h:].std(0)
        a = a / np.linalg.norm(a[mask]);  b = b / np.linalg.norm(b[mask])
        d_self = rel(a, b, mask)
        # spectral shape after energy normalization
        es, _ = radial_spectrum(s - s.mean(0), 16);  er, _ = radial_spectrum(r - r.mean(0), 16)
        es, er = es / np.nansum(es), er / np.nansum(er)
        d_spec = float(np.nanmean(np.abs(np.log(es / er))))
        out.append((nm, d_shape, d_flat, d_self, d_spec))
        print(f"  {nm:>10}  std-shape sim/real {d_shape:.3f}   (flat ref {d_flat:.3f}, "
              f"real self-consistency {d_self:.3f})   spec |log| {d_spec:.3f}", flush=True)
    a = np.array([o[1:] for o in out], float).mean(0)
    print(f"\n  MEAN   std-shape {a[0]:.4f}   flat reference {a[1]:.4f}   "
          f"real vs itself {a[2]:.4f}   normalized spectrum |log ratio| {a[3]:.4f}")
    print(f"  -> sim's fluctuation SHAPE is {'INFORMATIVE' if a[0] < a[1] else 'no better than flat'}"
          f": {a[0]:.3f} vs {a[1]:.3f} flat, against {a[2]:.3f} for real against itself")
