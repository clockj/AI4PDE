"""Gate the auxiliary PRESSURE head (round 17 Q3j) before it trains anything.

    python Track_1/verify/phead_strip.py

`--sim-p-head` adds `head_p` to `models_alt.FNO2d` -- a training-only readout supervised on sim
windows, where a pressure field exists. Nothing downstream knows about it, and that is the
danger: `_fno._fno_from_blob` builds its VENDORED FNO2d from the blob's flags and calls
`load_state_dict` with strict=True, so a single unexpected `head_p.*` key would fail every
loader in the project -- the package builder, `fit_band`, `compare_v9`, `ens_v9` -- and only at
SCORING time, after the training was paid for. That is exactly the failure `mean_head` caused
once already (CLAUDE.md, `_fno_from_blob`'s docstring).

`train2.py` therefore strips `head_p.*` at save. Four checks, on CPU, in seconds:

  A  with `p_head=True` the head EXISTS and produces (B, T_OUT, H, W) from the same trunk
     features `head_field` reads -- i.e. the flag is not silently inert.
  B  the field output is BIT-IDENTICAL with and without the head at the same seed, so the
     auxiliary head cannot perturb the deployed path by construction (it is a leaf).
  C  the blob saved through train2.py's own save expression, with the strip applied, loads
     through `_fno.load_any` -- the loader every member list goes through.
  D  the loaded model reproduces the training model's field output bit for bit.

`RPDE_UNET` / `RPDE_SUBMIT` override the source directories so the gate can be run against a
development copy BEFORE it is installed, which is the only moment at which it is useful.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNET = os.environ.get("RPDE_UNET", os.path.join(ROOT, "unet"))
SUBMIT = os.environ.get("RPDE_SUBMIT", os.path.join(ROOT, "submit_fno"))
sys.path.insert(0, UNET)
sys.path.insert(0, SUBMIT)

# The shipped w96 L8 do45 recipe, which is what Q3j trains.
CFG = dict(width=96, modes_y=8, modes_x=12, n_layers=8, var_head=True,
           var_mlp=True, lift_mlp=True, dropout=0.45)


def build_model(p_head: bool):
    import models_alt
    torch.manual_seed(0)
    return models_alt.build("fno", CFG["width"], CFG["var_head"], 0,
                            CFG["modes_y"], CFG["modes_x"], CFG["n_layers"],
                            var_mlp=CFG["var_mlp"], lift_mlp=CFG["lift_mlp"],
                            dropout=CFG["dropout"], p_head=p_head).to("cpu")


def main() -> None:
    from model import C, CH, H, T_OUT, W
    ok = True
    x = torch.randn(2, CH, H, W)

    m1 = build_model(True).eval()
    m0 = build_model(False).eval()

    print("== A  the head exists and reads the trunk")
    keys = [k for k in m1.state_dict() if k.startswith("head_p.")]
    n_par = sum(m1.state_dict()[k].numel() for k in keys)
    with torch.no_grad():
        f1, s1, p1 = m1(x, return_p=True)
    good = keys and p1 is not None and tuple(p1.shape) == (2, T_OUT, H, W)
    ok &= bool(good)
    print(f"   head_p tensors {keys} ({n_par} parameters)")
    print(f"   p prediction {tuple(p1.shape)} expected {(2, T_OUT, H, W)}: {bool(good)}")
    with torch.no_grad():
        f0, s0 = m0(x)
    print(f"   without the flag: head_p is "
          f"{'None' if getattr(m0, 'head_p', 'MISSING') is None else 'PRESENT (BAD)'}, "
          f"state_dict has {sum(k.startswith('head_p.') for k in m0.state_dict())} head_p keys")
    ok &= getattr(m0, "head_p", "MISSING") is None
    ok &= not any(k.startswith("head_p.") for k in m0.state_dict())

    print("\n== B  the field path is unchanged by the head, at the same seed")
    d = (f1 - f0).abs().max().item()
    ok &= d == 0.0
    print(f"   max|field(p_head=True) - field(p_head=False)| = {d:.3e}  (0 required)")
    print(f"   the two models' non-head_p tensors are identical: "
          f"{all(torch.equal(m1.state_dict()[k], m0.state_dict()[k]) for k in m0.state_dict())}")

    print("\n== C/D  the saved blob loads through _fno.load_any")
    import _fno
    sd = m1.state_dict()
    n_before = len(sd)
    sd = {k: v for k, v in sd.items() if not k.startswith("head_p.")}   # train2.py's strip
    blob = {"model_state_dict": sd, "mode": "twohead",
            "width": CFG["width"], "var_head": True, "var_mlp": True, "lift_mlp": True,
            "point_mlp": False, "var_hidden": 128, "var_deep": False, "var_in_std": False,
            "var_pred_std": False, "fgn_dim": 0, "hist": False, "mean_head": False,
            "osc_modes": 0, "osc_global": False, "osc_zeromean": False, "tbasis": 0,
            "ctx": 20, "arch": "fno", "modes_y": CFG["modes_y"], "modes_x": CFG["modes_x"],
            "n_layers": CFG["n_layers"], "p": 256, "self_norm": False, "norm": "real",
            "mean": np.zeros(C, np.float32), "std": np.ones(C, np.float32),
            "config": {"sim_p_head": True, "lam_p_sim": 0.3, "dropout": CFG["dropout"]}}
    print(f"   stripped {n_before - len(sd)} head_p.* tensor(s) of {n_before}")
    with tempfile.TemporaryDirectory(prefix="phead_") as td:
        path = os.path.join(td, "fno_r17_simp_probe.pth")
        torch.save(blob, path)
        try:
            m2 = _fno.load_any(path, "cpu")
            print("   _fno.load_any: OK")
        except Exception as e:                                    # noqa: BLE001
            ok = False
            m2 = None
            print(f"   _fno.load_any: FAILED -- {type(e).__name__}: {e}")
        if m2 is not None:
            with torch.no_grad():
                f2 = m2(x)[0]
            d2 = (f2 - f1).abs().max().item()
            ok &= d2 == 0.0
            print(f"   max|field(loaded) - field(trained)| = {d2:.3e}  (0 required)")

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
