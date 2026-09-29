"""Synthetic re-blanking of a recording (arm I2): a mask, constant in time, set to the PHYSICAL ZERO in u and v of every
input frame and every target frame of the trajectory. The kit then leaves those elements unscored (target == 0), exactly as
it treats the recording's own blanked elements, and the package sees them as it sees real blanked pixels.

Grid (after the [::2, ::2] subsample): H = 32 rows, W = 64 columns; the flow runs toward increasing column (the time-mean
u grows with the column index and the airfoil's zero set sits at columns 0-20, lower rows), so the downstream edge is
column 63 and the near wake starts at column ~20.

    m1  border band 6 px wide on the downstream edge: columns 58-63, every row (192 px, 9.4 %)
    m2  the recording's own STATIC zero set (u = v = 0 in every frame of the trajectory) dilated by 3 px (7 x 7 box)
    m3  a 10 x 20 rectangle in the near wake: rows 12-21, columns 20-39 (200 px, 9.8 %); it covers mvpe probes
        (rows 8-24, columns 21/29/37), whose metric is not masked by the kit, so mvpe drops for every variant alike

The masks are applied to the windows, which is the same as applying them to the trajectory before windowing because every
window is a slice of the trajectory's frames and the mask is constant in time.
"""
from __future__ import annotations

import numpy as np

H, W = 32, 64
MASKS = ("m1", "m2", "m3")


def _dilate(m, r):
    o = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            s = np.zeros_like(m)
            ys, yd = (slice(dy, None), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, None))
            xs, xd = (slice(dx, None), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, None))
            s[yd, xd] = m[ys, xs]
            o |= s
    return o


def zero_sets(frames):
    """frames (T, H, W, >=2) physical -> (static, transient) boolean (H, W): u = v = 0 in all / in some but not all frames."""
    z = (frames[..., 0] == 0) & (frames[..., 1] == 0)
    st = z.all(0)
    return st, z.any(0) & ~st


def mask_for(name, frames):
    if name == "m1":
        m = np.zeros((H, W), bool); m[:, W - 6:] = True
    elif name == "m2":
        m = _dilate(zero_sets(frames)[0], 3)
    elif name == "m3":
        m = np.zeros((H, W), bool); m[12:22, 20:40] = True
    else:
        raise SystemExit(f"unknown mask {name!r} ({' | '.join(MASKS)})")
    return m


def apply(stream, name):
    """stream = t2_common.build_stream output. -> (new stream, {traj: stats})."""
    by = {}
    for nm, t, x, y, first in stream:
        by.setdefault(nm, []).append((t, x, y))
    masks, stats = {}, {}
    for nm, ws in by.items():
        fr = {}
        for t, x, y in ws:
            for i in range(len(x)):
                fr[t + i] = x[i]
            for i in range(len(y)):
                fr[t + len(x) + i] = y[i]
        frames = np.stack([fr[k] for k in sorted(fr)])
        st, tr = zero_sets(frames)
        m = mask_for(name, frames)
        masks[nm] = m
        stats[nm] = dict(mask_px=int(m.sum()), static_zero_px=int(st.sum()), transient_zero_px=int(tr.sum()),
                         mask_new_px=int((m & ~st).sum()), frames=int(len(frames)))
    out = []
    for nm, t, x, y, first in stream:
        m = masks[nm]
        x = x.copy(); y = y.copy()
        x[:, m, :2] = 0.0
        y[:, m, :2] = 0.0
        out.append((nm, t, x, y, first))
    return out, stats
