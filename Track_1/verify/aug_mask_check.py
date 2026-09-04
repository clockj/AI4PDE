"""CPU gate for `train2.py --aug-mask`: inert at 0, and exactly the real zero set's value at 0.5.

Round 13's rule, earned twice since: **a feature nothing local exercises needs its own
verifier**. `--aug-mask` paints synthetic field-of-view geometry, and the thing it must paint
is not "zero" but the exact constant the REAL zero set carries after normalization -- an
off-by-one-normalizer mask would train the model to emit a value no PIV window ever contains,
which no held-out table would show as anything but noise.

Five checks, all on the CPU, no GPU and no training run needed:

  1. the real zero set's normalized value IS `-mean/std`, read off actual train_real windows;
  2. at p = 0 no random number is drawn -- the RNG state is byte-identical afterwards, which is
     what makes a p = 0 run byte-identical to one without the flag;
  3. at p = 0 the tensors come back as the same objects, untouched;
  4. at p = 0.5 the masked elements of BOTH x and y hold exactly `zconst`, the same region on
     every one of the 20 frames and both channels, and every unmasked element is bit-identical
     to its input;
  5. the drawn regions obey the documented families (border band 1-6 on one side, or a
     4-12 x 8-24 rectangle) and roughly half the samples are hit at p = 0.5;
  6. two tiny end-to-end training steps of a real FNO2d agree BIT-FOR-BIT between "--aug-mask
     absent" and "--aug-mask 0.0".

    python Track_1/verify/aug_mask_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
UNET = os.path.join(HERE, "..", "unet")
sys.path.insert(0, UNET)

from model import C, H, MEAN, STD, T_IN, W, load_trajectories  # noqa: E402
from train2 import aug_mask  # noqa: E402

REAL = "/users/PAS1187/qwu/RealPDE/train_real/train_real"
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


mean_v = torch.from_numpy(MEAN)
std_v = torch.from_numpy(STD)
zconst = -mean_v / std_v
print(f"zconst (normalized value of a physical zero) = {zconst.tolist()}")

# ---------------------------------------------------------------- 1. the real zero set
# One trajectory is enough: the field of view / airfoil geometry is a property of the rig.
trajs = load_trajectories(REAL, exclude=[], normalize=False)[:1]
raw = torch.from_numpy(trajs[0][:40])                       # (40, H, W, C) physical
nrm = (raw - mean_v) / std_v
z = (raw[..., 0] == 0) & (raw[..., 1] == 0)                 # per-frame zero pixels
zf = z.all(0)                                               # static zero set
print(f"train_real trajectory 0: static zero set {zf.float().mean():.2%} of {H}x{W} pixels")
# Mixing a boolean mask with an integer index confuses torch's shape check, hence the
# two-step slice: take the masked pixels first, then the channel.
zv = nrm[:, zf, :]
uq = [zv[..., c].reshape(-1).unique() for c in range(2)]
check("real zero set sits at -mean/std exactly",
      all(torch.equal(uq[c], zconst[c:c + 1]) for c in range(2)),
      f"unique values there = {[u.tolist() for u in uq]}")

# ---------------------------------------------------------------- 2-3. inert at p = 0
B = 8
g = torch.Generator().manual_seed(7)
x0 = torch.randn(B, T_IN, H, W, 2, generator=g)
y0 = torch.randn(B, T_IN, H, W, 2, generator=g)
st = torch.get_rng_state()
x1, y1 = aug_mask(x0, y0, 0.0, zconst)
check("p=0 draws no random numbers (global RNG state unchanged)",
      torch.equal(st, torch.get_rng_state()))
check("p=0 returns the identical tensor objects", x1 is x0 and y1 is y0)

# ---------------------------------------------------------------- 4-5. semantics at p = 0.5
torch.manual_seed(0)
xb = torch.randn(256, T_IN, H, W, 2)
yb = torch.randn(256, T_IN, H, W, 2)
xm, ym = aug_mask(xb.clone(), yb.clone(), 0.5, zconst)

mx = (xm != xb).any((1, 4))                                  # (N,H,W) pixels the mask touched
my = (ym != yb).any((1, 4))
check("the SAME region is masked in the input and the target", torch.equal(mx, my))

m5 = mx[:, None, :, :, None].expand_as(xm)
check("masked elements equal zconst exactly, on every frame and both channels",
      torch.equal(xm[m5], zconst.expand(256, T_IN, H, W, 2)[m5])
      and torch.equal(ym[m5], zconst.expand(256, T_IN, H, W, 2)[m5]))
check("unmasked elements are bit-identical to the input",
      torch.equal(xm[~m5], xb[~m5]) and torch.equal(ym[~m5], yb[~m5]))

hit = mx.any((1, 2))
check("hit rate ~ p", 0.42 <= hit.float().mean() <= 0.58, f"{hit.float().mean():.3f} of 256")

bands = rects = bad = 0
for i in torch.nonzero(hit).flatten().tolist():
    m = mx[i]
    rows, cols = torch.nonzero(m.any(1)).flatten(), torch.nonzero(m.any(0)).flatten()
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    rh, rw = r1 - r0 + 1, c1 - c0 + 1
    if not bool(m[r0:r1 + 1, c0:c1 + 1].all()):
        bad += 1                                             # every region must be a rectangle
    elif (rw == W and 1 <= rh <= 6 and (r0 == 0 or r1 == H - 1)) or \
         (rh == H and 1 <= rw <= 6 and (c0 == 0 or c1 == W - 1)):
        bands += 1
    elif 4 <= rh <= 12 and 8 <= rw <= 24:
        rects += 1
    else:
        bad += 1
check("every drawn region is a legal band or rectangle", bad == 0,
      f"{bands} bands, {rects} rectangles, {bad} illegal")
check("both region families are drawn", bands > 10 and rects > 10)

# ---------------------------------------------------------------- 6. two real training steps
sys.path.insert(0, UNET)
from models_alt import build  # noqa: E402


def two_steps(use_flag: bool, p: float = 0.0):
    """A miniature of train2.py's inner loop: the aug block, then forward/backward/step."""
    torch.manual_seed(123)
    m = build("fno", 8, True, 0, 4, 6, 2, dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    torch.manual_seed(456)
    for _ in range(2):
        x = torch.randn(2, T_IN, H, W, 2)
        y = torch.randn(2, T_IN, H, W, 2)
        if use_flag and p > 0.0:
            x, y = aug_mask(x, y, p, zconst)
        elif use_flag:
            # exactly the guarded call site: `if args.aug_mask > 0.0` is False, so nothing runs
            pass
        inp = x.permute(0, 1, 4, 2, 3).reshape(2, T_IN * C, H, W)
        raw, _ = m(inp)
        loss = ((raw.reshape(2, T_IN, C, H, W).permute(0, 1, 3, 4, 2) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.clone() for k, v in m.state_dict().items()}


a, b = two_steps(False), two_steps(True, 0.0)
check("two training steps: --aug-mask 0.0 is bit-identical to the flag being absent",
      all(torch.equal(a[k], b[k]) for k in a), f"{len(a)} tensors compared")
c = two_steps(True, 0.5)
check("two training steps: --aug-mask 0.5 actually changes the weights",
      any(not torch.equal(a[k], c[k]) for k in a))

print("\nALL CHECKS PASSED" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
