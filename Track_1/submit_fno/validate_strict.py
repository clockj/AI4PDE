"""Submittability check under the HARSHEST plausible load path.

    python submit_fno/validate_strict.py

A friendly validator inserts the extract directory into sys.path before importing. The platform
may instead load submission.py with importlib.util.spec_from_file_location from an unrelated
working directory, which does NOT put the containing directory on sys.path -- a bare
`import _fno` fails there. This reproduces that case:

  * extract the zip to a scratch dir
  * chdir somewhere else entirely
  * do NOT add the extract dir to sys.path
  * block the packages the Track 1 eval image lacks
  * load submission.py by file location and call predict()
  * check every contract the ingestion program asserts
  * score through the official scoring.py

Note on the reported time_score: predict() here is timed end to end on val_unseen, which is
what the evaluator does, so it is slightly slower than the forward-pass-only 0.109 ms quoted in
the reports. Trust this number over that one for submission purposes.
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
KIT = "/home/clockj/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9"
MISSING = ["scipy", "h5py", "einops", "matplotlib", "pandas"]

# Imported up front and the path entry removed again: the point of this script is that the
# submission must load with NOTHING helpful on sys.path, so Track_1/ must not stay there.
sys.path.insert(0, TRACK1)
import final_v9  # noqa: E402
sys.path.pop(0)


def main() -> None:
    subprocess.run([sys.executable, os.path.join(HERE, "build_zip.py")] + sys.argv[1:],
                   check=True)
    work = tempfile.mkdtemp(prefix="strict_fno_")
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

    sub.predict(X[:32], None)                    # warm up CUDA before timing
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

    # This model reads only its own window, so shuffling the window order must not change what
    # it predicts. That is the property submit_trick/ deliberately violates, so it is worth
    # testing rather than asserting in a docstring.
    #
    # It is NOT testable bitwise. cuFFT and cuBLAS reductions are not invariant to batch
    # composition, so permuting perturbs the raw field by ~1e-4, and predict_fno's division by
    # (s_ + 1e-6) amplifies that to ~1e-3 wherever the temporal std is near zero. The same
    # ~2e-4 perturbation appears between batch-16 and batch-1 inference, and batch-1 is
    # independent by definition -- which is what proves the effect is numerical rather than a
    # real order dependence. So test the thing that actually matters: the SCORES must not move.
    perm = np.random.default_rng(0).permutation(len(X))
    Ps = sub.predict(X[perm], None)["prediction"]
    inv = P[perm]
    rs = float(np.mean(sc.rel_l2_per_sample(Ps, Y[perm], 2)))
    ks = float(np.mean(sc.tke_rel_l2_per_sample(Ps, Y[perm], 2)))
    mvs = sc.mvpe_rel_l2(Ps, Y[perm])
    drift = np.abs(Ps - inv).mean() / np.sqrt((P[..., :2] ** 2).mean())
    assert drift < 1e-4, f"permuting moved the field by {drift:.2e} of rms -- order dependence"
    for nm, a, b in (("rel_l2", r, rs), ("tke", k, ks), ("mvpe", mv, mvs)):
        assert abs(a - b) < 1e-5, f"{nm} changed under permutation: {a:.7f} -> {b:.7f}"
    print(f"OK: order-independent (uses no cross-window information; "
          f"mean field drift {drift:.1e} of rms, all three error metrics stable to 1e-5)")
    subs = [sc.score_error(r), sc.score_error(k), sc.score_error(mv), sc.score_time(dt),
            sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=lo, upper=hi)[0])]
    print("OK: scored through the v9 scoring.py")
    print(f"    rel_l2 {r:.4f}  tke {k:.4f}  mvpe {mv:.4f}   ({dt*1000:.3f} ms/window)")
    print(f"    rel {subs[0]:.2f}  tke {subs[1]:.2f}  mvpe {subs[2]:.2f}  "
          f"time {subs[3]:.2f}  sps {subs[4]:.2f}   mean5 {float(np.mean(subs)):.3f}")
    # v9 stopped publishing the combination rule, and it is no longer the plain mean: the
    # 2026-08-05 board reported 79.733 where the mean of its own five subscores is 78.999.
    # mean5 above is a tracking number; final_v9 projects the published-subscore weighting
    # fitted to three board rounds. Both are computed on LOCAL data, so neither is a board
    # prediction -- the board's errors run ~1.39x ours per element.
    print("    " + final_v9.report(subs) + "  [local data, not a board prediction]")

    ext = sum(os.path.getsize(os.path.join(r_, f)) for r_, _, fs in os.walk(work) for f in fs)
    print(f"OK: extracted {ext/1e6:.2f} MB < 256 MB cap")
    sys.path[:] = saved_path
    shutil.rmtree(work, ignore_errors=True)
    print("\nSUBMITTABLE")


if __name__ == "__main__":
    main()
