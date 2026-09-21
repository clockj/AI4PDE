"""The three vendored non-FNO families must match their training definitions, and the FNO path
must not have moved.

    python verify/diff_arch.py [--old /path/to/previous/_fno.py]

`models_alt.py` is NOT in the submission archive, and `_fno.load_any` deliberately raises for
any family it does not carry itself -- so a family only becomes shippable once it is copied into
`_fno.py`. That copy is a second definition of the same object, and the failure mode is silent:
a renamed submodule, a default that drifted, an eval-only branch reproduced wrongly, and the
member loads and predicts something slightly different inside the container than it did in every
local measurement. `verify/diff_ufno.py` was written for exactly that and covered one family.

This covers all three, and adds the check that one was missing: that the vendored class is
reachable WITHOUT the repo. The subprocess below blocks `models_alt` at import time the way the
official smoke test blocks scipy/h5py, so a class that silently fell back to the repo branch
fails here instead of at scoring time.

The last section is the regression guard: the shipping FNO members must produce a bit-identical
field and sigma before and after the edit. Vendoring is additive by intent, and "by intent" is
not a measurement.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
sys.path.insert(0, os.path.join(ROOT, "submit_fno"))
sys.path.insert(0, os.path.join(ROOT, "unet"))

CASES = [("ufno", "arch_ufno_w96.pth"), ("afno", "arch_afno_w96.pth"),
         ("vit", "arch_vit_d384.pth")]

BLOCKED = r'''
import sys, os
class _Block:
    def find_module(self, name, path=None):
        return self if name == "models_alt" else None
    def load_module(self, name):
        raise ImportError("models_alt is not in the submission archive")
sys.meta_path.insert(0, _Block())
sys.path.insert(0, os.path.join(%r, "submit_fno"))
import torch, numpy as np, _fno
m = _fno.load_any(%r, "cpu")
x = torch.from_numpy(np.load(%r)["x"])
# no_grad matters: nn.MultiheadAttention takes a fused native kernel when grad is off and a
# python fallback when it is on, and the two agree only to float32 noise. predict_fno is
# @torch.no_grad, so the fused path is what ships and what this must reproduce.
with torch.no_grad():
    f, s = m(x)
np.save(%r, np.stack([f.detach().numpy().ravel()[:4096],
                      s.detach().numpy().ravel()[:4096]]))
print("OK")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None, help="previous _fno.py, for the FNO regression guard")
    args = ap.parse_args()

    import _fno
    import models_alt

    torch.manual_seed(0)
    x = torch.randn(2, _fno.CH, _fno.H, _fno.W)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "diff_arch_in.npz")
    np.savez(tmp, x=x.numpy())

    print("family   params      max|d field|   max|d sigma|   no-repo import")
    bad = 0
    for arch, fn in CASES:
        path = os.path.join(ROOT, "unet", fn)
        blob = torch.load(path, map_location="cpu")
        cfg = blob.get("config", {})
        ref = models_alt.build(arch, blob.get("width", 96), blob.get("var_head", True), 0,
                               blob.get("modes_y", 8), blob.get("modes_x", 12),
                               blob.get("n_layers", 4), dropout=cfg.get("dropout", 0.0))
        ref.load_state_dict(blob["model_state_dict"])
        ref.eval()
        van = _fno.load_any(path, "cpu")
        with torch.no_grad():
            f0, s0 = ref(x)
            f1, s1 = van(x)
        df = float((f0 - f1).abs().max())
        ds = float((s0 - s1).abs().max())

        out = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"diff_arch_{arch}.npy")
        src = BLOCKED % (os.path.abspath(ROOT), os.path.abspath(path), tmp, out)
        r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
        ok = r.returncode == 0 and "OK" in r.stdout
        if ok:
            got = np.load(out)
            ok = max(np.abs(got[0] - f1.detach().numpy().ravel()[:4096]).max(),
                     np.abs(got[1] - s1.detach().numpy().ravel()[:4096]).max()) < 1e-5
        n = sum(p.numel() for p in van.parameters())
        print(f"{arch:8s} {n/1e6:6.2f}M   {df:12.3e}   {ds:12.3e}   {'yes' if ok else 'NO'}")
        if not ok:
            print("   subprocess stderr:", r.stderr.strip().splitlines()[-3:])
        bad += int(df > 1e-4 or ds > 1e-4 or not ok)

    # ---- the FNO regression guard --------------------------------------------------------
    if args.old:
        spec = importlib.util.spec_from_file_location("_fno_old", args.old)
        old = importlib.util.module_from_spec(spec)
        sys.modules["_fno_old"] = old
        spec.loader.exec_module(old)
        ck = os.path.join(ROOT, "unet", "fno_r18_cmb3sim.pth")
        a, b = old.load_any(ck, "cpu"), _fno.load_any(ck, "cpu")
        with torch.no_grad():
            fa, sa = a(x)
            fb, sb = b(x)
        d = max(float((fa - fb).abs().max()), float((sa - sb).abs().max()))
        print(f"\nFNO path unchanged by the edit: max|delta| {d:.3e}"
              f"   {'BIT-IDENTICAL' if d == 0.0 else 'CHANGED -- STOP'}")
        bad += int(d != 0.0)

    print("\nARCH VENDORING OK" if bad == 0 else f"\n{bad} CHECK(S) FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
