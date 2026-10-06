"""Run the five twins over the UNSEEN validation windows and store each one's field (at alpha
0) and sigma head: the input of sigma_band.py.

    python dump_twins.py --twins <5 .pth> --val <dir> --out dump_unseen.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission"))
import _fno  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--twins", nargs="+", required=True)
ap.add_argument("--val", required=True, help="directory holding val_unseen.npz")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16)
d = np.load(os.path.join(a.val, "val_unseen.npz"))
X, Y, ids = d["x"], d["y"], d["ids"]
F, S = [], []
for n in a.twins:
    t0 = time.time()
    m = _fno.load_any(n, dev)
    s = np.empty((len(X), 1, 32, 64, 2), np.float32)
    f = _fno.predict_fno([m], X, dev, alpha=0.0, sigma_out=s, batch=16, amp=None)
    F.append(f[..., :2].astype(np.float32)); S.append(s.copy())
    print(os.path.basename(n), f.shape, "%.1fs" % (time.time() - t0), flush=True)
np.savez(a.out, F=np.stack(F), S=np.stack(S), x=X[..., :2], y=Y[..., :2], ids=ids,
         names=np.array([os.path.basename(n).replace(".pth", "") for n in a.twins]))
print("wrote", a.out)
