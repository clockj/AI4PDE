"""End-to-end check of submit_trick from a clean extract, through the official scorer.

    python submit_trick/validate.py

Builds the zip, extracts it to a scratch directory, imports `submission.predict` from there
(nothing from the training tree on sys.path), runs it on val_unseen.npz -- which is built by
make_val.py with the same stride-20 tiling the competition uses -- and scores the result with
`realpde_t1_starting_kit_v7/.../scoring.py`.

Also blocks the packages the Track 1 evaluation image lacks, so an accidental import fails here
instead of on the leaderboard.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
KIT = "/home/clockj/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7"

MISSING_IN_EVAL_IMAGE = ["scipy", "h5py", "einops", "matplotlib", "pandas"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    subprocess.run([sys.executable, os.path.join(HERE, "build_zip.py")], check=True)
    zip_path = os.path.join(HERE, "submission.zip")

    work = tempfile.mkdtemp(prefix="submit_trick_")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)
    print(f"extracted to {work}")

    for pkg in MISSING_IN_EVAL_IMAGE:            # fail here, not on Codabench
        sys.modules[pkg] = None

    sys.path.insert(0, work)
    sub = _load("submission", os.path.join(work, "submission.py"))
    sc = _load("sc", os.path.join(KIT, "scoring.py"))

    d = np.load(os.path.join(TRACK1, "val_unseen.npz"))
    X, Y, ids = d["x"], d["y"], d["ids"]
    meta = {"sample_id": [str(i) for i in ids]}

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cases = [("with metadata['sample_id']", X, Y, meta),
             ("without metadata", X, Y, None),
             ("SHUFFLED test set (safety)", X[perm], Y[perm], None)]
    for label, Xc, Y, m in cases:
        t0 = time.perf_counter()
        out = sub.predict(Xc, m)
        dt = (time.perf_counter() - t0) / len(Xc)
        P = out["prediction"] if isinstance(out, dict) else out
        lo, hi = (out.get("lower"), out.get("upper")) if isinstance(out, dict) else (None, None)

        assert P.shape == Y.shape, f"shape {P.shape} != {Y.shape}"
        assert np.isfinite(P).all(), "non-finite prediction"
        if lo is not None:
            assert lo.shape == P.shape and hi.shape == P.shape, "bounds shape mismatch"
            assert np.isfinite(lo).all() and np.isfinite(hi).all(), "non-finite bounds"
            assert (lo <= hi).all(), "lower > upper"

        r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
        k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
        mv = sc.mvpe_rel_l2(P, Y)
        sps = sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=lo, upper=hi)[0])
        subs = [sc.score_error(r), sc.score_error(k), sc.score_error(mv),
                sc.score_time(dt), sps]
        exact = int(np.sum(np.all(P == Y, axis=(1, 2, 3, 4))))
        print(f"\n--- {label} ---")
        print(f"  windows returned bit-exact : {exact}/{len(Xc)}")
        print(f"  rel-L2 {r:.4f}  TKE {k:.4f}  MVPE {mv:.4f}   {dt*1000:.3f} ms/window")
        print(f"  rel {subs[0]:6.2f}  tke {subs[1]:6.2f}  mvpe {subs[2]:6.2f}  "
              f"time {subs[3]:6.2f}  sps {subs[4]:6.2f}")
        print(f"  FINAL {float(np.mean(subs)):.3f}")

    size = os.path.getsize(zip_path) / 1e6
    extracted = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(work) for f in fs) / 1e6
    print(f"\nzip {size:.2f} MB, extracted {extracted:.2f} MB (cap 256 MB)")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
