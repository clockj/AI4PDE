"""The vendored `_fno.UFNO2d` must be the same model as `unet/models_alt.UFNO2d`.

A family is only shippable once its class is INSIDE the archive: `models_alt.py` is not, and
`_fno.load_any`'s repo-import branch raises in the evaluation container on purpose. That leaves
two definitions of one model, which is exactly the situation that produced round 12's silent
member-DROPPING bug in `fuse()` -- a wrong answer that does not crash.

This loads one real checkpoint into BOTH classes and compares the fields and the sigmas.

    python Track_1/verify/diff_ufno.py --ckpt unet/arch_ufno_w96.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TRACK1, "submit_fno"))
sys.path.insert(0, os.path.join(TRACK1, "unet"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(TRACK1, "unet", "arch_ufno_w96.pth"))
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    import _fno
    import models_alt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(a.ckpt, map_location=dev)
    assert blob.get("arch") == "ufno", f"{a.ckpt}: arch is {blob.get('arch')}, not ufno"
    cfg = blob.get("config", {})

    vend = _fno.load_any(a.ckpt, dev)
    assert isinstance(vend, _fno.UFNO2d), f"load_any built {type(vend).__name__}"
    repo = models_alt.build("ufno", blob["width"], blob["var_head"], 0, blob["modes_y"],
                            blob["modes_x"], blob["n_layers"],
                            dropout=cfg.get("dropout", 0.0)).to(dev)
    repo.load_state_dict(blob["model_state_dict"])
    repo.eval()

    kv, kr = set(vend.state_dict()), set(repo.state_dict())
    if kv != kr:
        print(f"KEY MISMATCH  only-vendored {sorted(kv - kr)}  only-repo {sorted(kr - kv)}")
        return 1
    print(f"state_dict keys match ({len(kv)} tensors), "
          f"{sum(p.numel() for p in vend.parameters())/1e6:.2f}M params")

    x = torch.from_numpy(np.load(os.path.join(TRACK1, "val_unseen.npz"))["x"][:a.n]
                         .astype(np.float32)).to(dev)
    # (N,20,32,64,3) raw -> normalized (N,40,32,64), the layout both classes consume.
    xn = ((x[..., :2] - torch.as_tensor(_fno.MEAN, device=dev))
          / torch.as_tensor(_fno.STD, device=dev))
    xn = xn.permute(0, 1, 4, 2, 3).reshape(len(x), 40, 32, 64)
    with torch.no_grad():
        fv, sv = vend(xn)
        fr, sr = repo(xn)
    df = float((fv - fr).abs().max())
    ds = float((sv - sr).abs().max()) if sv is not None else 0.0
    print(f"max |field_vendored - field_repo|  {df:.3e}")
    print(f"max |logstd_vendored - logstd_repo| {ds:.3e}")
    ok = df < 1e-5 and ds < 1e-5
    print("OK -- the archive's class is the trained model" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
