"""Can ensemble disagreement adapt the SPS band per window?

    python spread_band.py --ckpt unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth

The shipped band is one number per (t, y, x, channel), the same for every window.  The exact
per-element optimum of that form scores 49.5 on val_unseen; an oracle allowed to pick a
half-width per window AND element -- i.e. told each individual error -- scores 65.4.  So ~16 SPS
points sit in knowing which windows are hard, and that is worth 4.4x what it was before v9 made
the score linear.

Disagreement between independently-seeded ensemble members is the standard cheap proxy for
that, and here it is genuinely free: the three members already run, and their spread costs one
reduction on a tensor that is already on the GPU.

Candidates, all fitted on val_seen and reported on val_unseen:

  static            the incumbent: exact per-element optimum, no per-window information
  spread            h = optimal_half(c * spread), the static map discarded entirely
  static x ratio^g  h = static * (spread / spread_mean_over_windows)^g -- keeps the spatial and
                    temporal structure the static map captured and lets disagreement modulate
                    it.  g=0 is the incumbent, g=1 is fully proportional.

What would have to be true for this to ship: the gain has to clear the ~0.3 ms/window that
returning a per-window band costs, since `lower` and `upper` would then have to be built on the
GPU and transferred instead of broadcast from a 0.5 MB array on the host.  At v9 rates that is
about 0.1 of final, so roughly +0.5 SPS is the break-even.
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
import torch

import sps9
from sps9 import optimal_band, prep, sps_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "submit_fno"))


@torch.no_grad()
def members_and_spread(models, X, dev, alpha=0.75, batch=64):
    """The ensemble prediction and the per-element std across members, both in raw units.

    The spread is taken on the RAW member fields, before the fluctuation rescaling, because
    that is where the members actually differ; compose() is a deterministic function of the
    averaged field and would only add a common factor.
    """
    import _fno
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    std = torch.from_numpy(_fno.STD).to(dev)
    n = len(X)
    P = np.zeros((n, 20, 32, 64, 3), np.float32)
    S = np.zeros((n, 20, 32, 64, 3), np.float32)
    for i in range(0, n, batch):
        xb = torch.from_numpy(X[i:i + batch]).to(dev)[..., :2].float()
        prev = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), _fno.CH, 32, 64)
        ys, sigs = [], []
        for m in models:
            raw, ls = m(prev)
            ys.append(raw.reshape(len(xb), 20, 2, 32, 64).permute(0, 1, 3, 4, 2))
            sigs.append(ls.clamp(-8, 2).exp().permute(0, 2, 3, 1).unsqueeze(1))
        Ymem = torch.stack(ys)                      # (K, B, 20, 32, 64, 2), normalized units
        y = Ymem.mean(0)
        sig = torch.stack(sigs).mean(0)
        spread = Ymem.std(0, unbiased=False) * std  # back to raw units
        if alpha > 0:
            m_ = y.mean(1, keepdim=True)
            f_ = y - m_
            s_ = y.std(1, keepdim=True, unbiased=False)
            y = m_ + f_ * ((alpha * sig + (1 - alpha) * s_) / (s_ + 1e-6))
        P[i:i + batch, ..., :2] = (y * std + mean).float().cpu().numpy()
        S[i:i + batch, ..., :2] = spread.float().cpu().numpy()
    return P, S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--inflate", type=float, default=1.0)
    args = ap.parse_args()

    import _fno
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [_fno.load(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
              for c in args.ckpt]
    assert len(models) > 1, "spread needs at least two members"

    ds = np.load(os.path.join(HERE, "val_seen.npz"))
    du = np.load(os.path.join(HERE, "val_unseen.npz"))
    Ps, Sp_s, Ys = *members_and_spread(models, ds["x"], dev, args.alpha), ds["y"]
    Pu, Sp_u, Yu = *members_and_spread(models, du["x"], dev, args.alpha), du["y"]

    S = prep(Ps, Ys)
    U = prep(Pu, Yu)
    f = np.float32(args.inflate)
    sp_s, sp_u = Sp_s[..., :2], Sp_u[..., :2]

    # How well does spread track the actual error at all? A per-element correlation over
    # windows is the honest question: the band is already allowed to vary with (t, y, x), so
    # only the part of the variation that is per-WINDOW is new information.
    err_s = S[0]
    c_el = np.corrcoef(err_s.ravel(), sp_s.ravel())[0, 1]
    dev_e = err_s - err_s.mean(0, keepdims=True)
    dev_s = sp_s - sp_s.mean(0, keepdims=True)
    c_win = float((dev_e * dev_s).mean() / (dev_e.std() * dev_s.std()))
    print(f"corr(|err|, spread)          overall {c_el:.3f}   "
          f"after removing the per-element mean {c_win:.3f}")

    static = optimal_band(f * S[0], S[1], S[2])
    print(f"\n{'band':28s} {'median h':>9s} {'seen':>7s} {'unseen':>7s}")
    print(f"  {'static (incumbent)':26s} {np.median(static):9.5f} "
          f"{sps_score(*S, static):7.2f} {sps_score(*U, static):7.2f}")

    from band_v9 import optimal_half_gauss
    best = None
    for c in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        h = optimal_half_gauss(f * c * sp_s)
        v = sps_score(*S, h)
        if best is None or v > best[0]:
            best = (v, c)
    c = best[1]
    print(f"  {f'spread only c={c}':26s} {np.median(optimal_half_gauss(f*c*sp_s)):9.5f} "
          f"{best[0]:7.2f} {sps_score(*U, optimal_half_gauss(f*c*sp_u)):7.2f}")

    ratio_s = sp_s / (sp_s.mean(0, keepdims=True) + 1e-12)
    ratio_u = sp_u / (sp_u.mean(0, keepdims=True) + 1e-12)
    for g in (0.25, 0.5, 0.75, 1.0):
        hs = (static * ratio_s ** g).astype(np.float32)
        hu = (static * ratio_u ** g).astype(np.float32)
        print(f"  {f'static x ratio^{g}':26s} {np.median(hs):9.5f} "
              f"{sps_score(*S, hs):7.2f} {sps_score(*U, hu):7.2f}")

    # The ceiling for anything of this kind.
    orc = np.where(U[2], np.exp(-2.0 * U[0] / sps9.SIGMA), np.float32(0.0))
    print(f"  {'oracle (per-window+element)':26s} {'':>9s} {'':>7s} "
          f"{100*(U[1] @ orc.reshape(len(U[0]), -1).sum(1, dtype=np.float64))/U[3]:7.2f}")


if __name__ == "__main__":
    main()
