"""What frequency does the oscillatory head actually pick, and is it the shedding frequency?

    python osc_probe.py --ckpt unet/fno_ozdo.pth unet/fno_ozdo_b.pth        # CPU is fine

`--osc-global` gives the head J free angular frequencies per window, squashed by a sigmoid into
[2pi/200, 2pi/10] rad/frame. Nothing ties them to the flow. Two things are worth knowing before
spending another arm on this head, and both are free:

1. **Where do they land?** The shedding period for this airfoil is 60-75 frames, i.e.
   omega ~ 0.084-0.105. If the learned omegas cluster there, the band is already doing the
   constraining and narrowing it buys nothing. If they are scattered across the band, the
   constraint is loose and there is room.

2. **Do they follow Strouhal?** Vortex shedding obeys St = f*c/U_inf ~ const, so
   omega should be PROPORTIONAL to the window's own mean streamwise speed. That is testable
   here as a correlation, and it decides a real arm: an omega locked to St*U_inf extrapolates
   to unseen Reynolds numbers BY CONSTRUCTION, where a free per-window sigmoid has to learn
   the extrapolation from 74 trajectories. `report_20260831_round13.md` §5 measured that
   out-of-distribution behaviour is the thing in-distribution fit does not determine.

Reads the amplitudes too: a mode whose amplitude is ~0 is not a mode, and J=2 could be J=1.
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "submit_fno"))
sys.path.insert(0, os.path.join(HERE, "unet"))

import _fno  # noqa: E402
from models_alt import OSC_W_HI, OSC_W_LO  # noqa: E402


@torch.no_grad()
def probe(m, x_raw, dev, batch=64):
    """(omega (N,J), rms amplitude (N,J), window mean speed (N,))."""
    mean = torch.from_numpy(_fno.MEAN).to(dev)
    std = torch.from_numpy(_fno.STD).to(dev)
    om, am = [], []
    for i in range(0, len(x_raw), batch):
        xb = torch.from_numpy(x_raw[i:i + batch]).to(dev)[..., :2].float()
        prev = ((xb - mean) / std).permute(0, 1, 4, 2, 3).reshape(len(xb), _fno.CH,
                                                                 _fno.H, _fno.W)
        h = m.trunk(prev) if hasattr(m, "trunk") else _trunk(m, prev)
        k = 2 if m.osc_global else 3
        o = m.head_osc(h).reshape(len(xb), m.osc_modes, k, _fno.C, h.shape[-2], h.shape[-1])
        if m.osc_global:
            w = m.osc_w(h.mean((-2, -1)))
            om.append((OSC_W_LO + (OSC_W_HI - OSC_W_LO) * torch.sigmoid(w)).cpu().numpy())
        else:
            w = OSC_W_LO + (OSC_W_HI - OSC_W_LO) * torch.sigmoid(o[:, :, 2])
            om.append(w.mean((-1, -2, -3)).cpu().numpy())
        # sqrt(p^2 + q^2) is the oscillation's amplitude; rms over channels and pixels.
        a = (o[:, :, 0] ** 2 + o[:, :, 1] ** 2).mean((-1, -2, -3)).sqrt()
        am.append(a.cpu().numpy())
    return np.concatenate(om), np.concatenate(am)


def _trunk(m, prev):
    """`_fno.FNO2d.forward` up to `h`, transcribed line for line.

    NOTE this is `submit_fno/_fno.py`'s FNO2d, not `unet/models_alt.py`'s -- `load_any` builds
    the vendored one, which carries no `drop` module at all (dropout adds no parameters, so
    the shipped file deliberately omits it and eval is unaffected). Transcribing the training
    class instead raises on `m.drop`. If either forward() changes, change this.
    """
    h = m.lift(torch.cat([prev, m.grid.expand(len(prev), -1, -1, -1)], 1))
    for s, p, n in zip(m.spec, m.point, m.norm):
        h = h + torch.nn.functional.gelu(n(s(h) + p(h)))
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--set", default="val_unseen")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(os.path.join(HERE, f"{args.set}.npz"))
    X = d["x"].astype(np.float32)
    u = X[..., 0].mean(axis=(1, 2, 3))          # the window's own U_inf proxy
    print(f"{args.set}: {len(X)} windows, mean speed {u.min():.4f}-{u.max():.4f} m/s")
    print("shedding band: period 60-75 frames => omega 0.0838-0.1047\n")

    for c in args.ckpt:
        m = _fno.load_any(c if os.path.isabs(c) else os.path.join(HERE, c), dev)
        if getattr(m, "head_osc", None) is None:
            print(f"{os.path.basename(c)}: no oscillatory head, skipped"); continue
        om, am = probe(m, X, dev)
        print(f"--- {os.path.basename(c)}  (J={m.osc_modes}, global={m.osc_global}, "
              f"zeromean={m.osc_zeromean}) ---")
        for j in range(om.shape[1]):
            w, a = om[:, j], am[:, j]
            r = float(np.corrcoef(w, u)[0, 1])
            # Strouhal would make omega/u constant; report its scatter as a fraction.
            ratio = w / u
            frac_in = float(np.mean((w >= 0.0838) & (w <= 0.1047)))
            # Is the sigmoid pinned? omega = LO + (HI-LO)*sigmoid, so a window in the top 1%
            # of the band is one the head would push further if the band allowed it.
            sat = float(np.mean((w - OSC_W_LO) / (OSC_W_HI - OSC_W_LO) > 0.99))
            print(f"  mode {j}: omega {w.mean():.4f} +- {w.std():.4f} "
                  f"[{w.min():.4f}, {w.max():.4f}]  period {2*np.pi/w.mean():5.1f} fr | "
                  f"amp {a.mean():.4f} | corr(omega, U) {r:+.3f} | "
                  f"omega/U cv {ratio.std()/ratio.mean():.3f} | in shedding band {frac_in:.2f} "
                  f"| at band top {sat:.3f}")


if __name__ == "__main__":
    main()
