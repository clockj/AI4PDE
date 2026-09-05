"""End-to-end CPU smoke test of submit_fno/submission.zip, with the evaluation image's
missing packages blocked exactly the way the official kit's smoke_test_kit.py blocks them.

`validate_strict.py` is the authoritative pre-submission gate but needs a GPU.  This runs the
same load path on a login node: extract the zip somewhere clean, block scipy/h5py/matplotlib/
pandas, import submission.py the way the evaluator does (extract dir NOT on sys.path), and
call predict() on real val_unseen windows.  It checks the output contract, not the score.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The archive to gate. Overridable, because a round often has more than one candidate staged
# at once (round 13 builds an 8-member and a 10-member package side by side) and gating the
# wrong one is exactly the mistake this script exists to prevent.
ZIP = os.environ.get("RPDE_ZIP") or os.path.join(HERE, os.pardir, "submit_fno", "submission.zip")
MISSING_IN_EVAL_IMAGE = {"scipy", "h5py", "matplotlib", "pandas"}


class _BlockedTopLevel:
    """Reject the packages the Track-1 evaluation image does not ship."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ModuleNotFoundError(
                f"simulated evaluation image without {name.split('.')[0]}: blocked {name}")
        return None


def main() -> int:
    sys.meta_path.insert(0, _BlockedTopLevel(MISSING_IN_EVAL_IMAGE))
    tmp = tempfile.mkdtemp(prefix="rpde_smoke_")
    try:
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(tmp)
        print(f"extracted {os.path.getsize(ZIP)/1e6:.1f} MB zip -> {tmp}")
        total = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(tmp) for f in fs)
        print(f"extracted size {total/1e6:.2f} MB   (cap 256 MB)  "
              f"{'OK' if total < 256e6 else 'OVER CAP'}")

        # The evaluator imports submission.py by path; the extract dir is NOT on sys.path,
        # so submission.py must locate its own siblings.  Reproduce that.
        spec = importlib.util.spec_from_file_location(
            "rpde_submission", os.path.join(tmp, "submission.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rpde_submission"] = mod
        spec.loader.exec_module(mod)
        print("import submission.py: OK  (extract dir not on sys.path)")

        import torch
        print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}"
              f"  -> running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

        d = np.load(os.path.join(HERE, os.pardir, "val_unseen.npz"))
        X = d["x"].astype(np.float32)[:8]          # 8 windows is enough for the contract
        Y = d["y"].astype(np.float32)[:8]

        out = mod.predict(X, {"sample_id": np.arange(len(X))})
        if isinstance(out, dict):
            keys = sorted(out)
            print(f"predict() returned dict with keys {keys}")
            assert keys == ["lower", "prediction", "upper"], keys
            P, lo, up = out["prediction"], out["lower"], out["upper"]
        else:
            print("predict() returned a bare array (no SPS bounds)")
            P, lo, up = out, None, None

        ok = True

        def check(cond, msg):
            nonlocal ok
            ok &= bool(cond)
            print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")

        check(P.shape == X.shape, f"prediction shape {P.shape} == input {X.shape}")
        check(P.dtype == np.float32, f"prediction dtype {P.dtype} is float32")
        check(np.all(np.isfinite(P)), "prediction is finite everywhere")
        if lo is not None:
            check(lo.shape == P.shape and up.shape == P.shape,
                  f"bounds shapes {lo.shape}/{up.shape} match prediction")
            check(np.all(np.isfinite(lo)) and np.all(np.isfinite(up)), "bounds are finite")
            check(np.all(lo <= up), "lower <= upper for every element")
            cov = float(np.mean((Y[..., :2] >= lo[..., :2]) & (Y[..., :2] <= up[..., :2])))
            print(f"        band coverage on these 8 windows: {cov:.4f}")

        # The submission must be order-independent: the evaluator's window order is not ours.
        perm = np.array([5, 0, 7, 2, 1, 6, 3, 4])
        out2 = mod.predict(X[perm], {"sample_id": perm})
        P2 = out2["prediction"] if isinstance(out2, dict) else out2
        dperm = float(np.max(np.abs(P2 - P[perm])))
        check(dperm < 1e-4, f"permutation invariant (max|diff| {dperm:.2e} < 1e-4)")

        # Scored channels only: p must come back zero-filled.
        check(np.all(P[..., 2] == 0.0), "p channel returned as zeros")

        print("\nVERDICT: " + ("PACKAGE OK on CPU with the eval image's package set blocked."
                               if ok else "FAILED -- do not submit."))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
