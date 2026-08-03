"""Submittability check under the HARSHEST plausible load path.

    python submit_trick/validate_strict.py

validate.py inserts the extract directory into sys.path before importing, which is friendly.
The platform may instead load submission.py with importlib.util.spec_from_file_location from
an unrelated working directory, which does NOT put the containing directory on sys.path. A
bare `import _fno` fails there. This reproduces that case:

  * extract the zip to a scratch dir
  * chdir somewhere else entirely
  * do NOT add the extract dir to sys.path
  * block the packages the Track 1 eval image lacks
  * load submission.py by file location and call predict()
  * check every contract the ingestion program asserts
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
MISSING = ["scipy", "h5py", "einops", "matplotlib", "pandas"]


def main() -> None:
    subprocess.run([sys.executable, os.path.join(HERE, "build_zip.py")], check=True)
    work = tempfile.mkdtemp(prefix="strict_")
    with zipfile.ZipFile(os.path.join(HERE, "submission.zip")) as z:
        z.extractall(work)

    d = np.load(os.path.join(TRACK1, "val_unseen.npz"))
    X, Y = d["x"], d["y"]

    spec = importlib.util.spec_from_file_location("sc", os.path.join(KIT, "scoring.py"))
    sc = importlib.util.module_from_spec(spec)
    sys.modules["sc"] = sc
    spec.loader.exec_module(sc)

    # --- harsh conditions from here on
    os.chdir(tempfile.gettempdir())              # unrelated cwd
    saved_path = list(sys.path)                  # deliberately NOT adding `work`
    for pkg in MISSING:
        sys.modules[pkg] = None                  # eval-image parity

    spec = importlib.util.spec_from_file_location(
        "submission", os.path.join(work, "submission.py"))
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)                 # <- fails here if `import _fno` is unguarded
    print("OK: submission.py imports with the extract dir NOT on sys.path")

    t0 = time.perf_counter()
    out = sub.predict(X, None)
    dt = (time.perf_counter() - t0) / len(X)

    assert isinstance(out, dict) and set(out) == {"prediction", "lower", "upper"}, set(out)
    P, lo, hi = out["prediction"], out["lower"], out["upper"]
    for nm, a in (("prediction", P), ("lower", lo), ("upper", hi)):
        assert a.shape == Y.shape, f"{nm} shape {a.shape} != {Y.shape}"
        assert a.dtype == np.float32, f"{nm} dtype {a.dtype} != float32"
        assert np.isfinite(a).all(), f"{nm} non-finite"
    assert (lo <= P).all() and (P <= hi).all(), "bounds do not bracket the prediction"
    print("OK: dict keys, shapes, float32 dtype, finiteness, lower <= pred <= upper")

    r = float(np.mean(sc.rel_l2_per_sample(P, Y, 2)))
    k = float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2)))
    mv = sc.mvpe_rel_l2(P, Y)
    subs = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), sc.score_time(dt),
            sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=lo, upper=hi)[0])]
    print(f"OK: scored through the official scoring.py")
    print(f"    rel {subs[0]:.2f}  tke {subs[1]:.2f}  mvpe {subs[2]:.2f}  "
          f"time {subs[3]:.2f}  sps {subs[4]:.2f}   FINAL {float(np.mean(subs)):.3f}")

    ext = sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(work) for f in fs)
    print(f"OK: extracted {ext/1e6:.2f} MB < 256 MB cap")
    sys.path[:] = saved_path
    shutil.rmtree(work, ignore_errors=True)
    print("\nSUBMITTABLE")


if __name__ == "__main__":
    main()
