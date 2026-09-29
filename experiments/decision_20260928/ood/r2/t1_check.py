"""Gate R2, Track 1: an EXISTING archive under validate_strict.py's harsh load conditions, outputs saved.

    python t1_check.py ARCHIVE.zip OUT.npz [--fp32]

validate_strict.py always rebuilds its zip from checkpoints (it forwards its arguments to build_zip.py), so it cannot gate an
archive as shipped. This applies the same conditions and contracts to the archive itself: extract to a scratch dir, chdir
elsewhere, do NOT add the extract dir to sys.path, block validate_strict.MISSING, load by file location, predict() on
val_unseen (warm-up on 32 windows, then all 336 timed), check keys / shapes / float32 / finiteness / lower <= pred <= upper
and permutation stability of the three error metrics, score through the kit. --fp32 sets AMP = None before the first predict
(the package's own fallback when its load-time fp16 check fails; applied through _AMP_DT, fixed at import). Saves pred / lower / upper and the load line.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import zipfile

import numpy as np

WT = os.environ.get("OOD_WT", "/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/wt_ood")
sys.path.insert(0, os.path.join(WT, "Track_1", "submit_fno"))
import validate_strict as vs  # noqa: E402  (MISSING, KIT)


def main():
    zp, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    fp32 = "--fp32" in sys.argv
    work = tempfile.mkdtemp(prefix="r2t1_", dir=os.environ.get("R2_WORK", None))
    zipfile.ZipFile(zp).extractall(work)
    d = np.load(os.path.join(WT, "Track_1", "val_unseen.npz"))
    X, Y = d["x"], d["y"]
    kit = os.environ.get("RPDE_KIT", "/users/PAS1187/qwu/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9")
    spec = importlib.util.spec_from_file_location("sc", os.path.join(kit, "scoring.py"))
    sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    os.chdir(tempfile.gettempdir())
    for pkg in vs.MISSING:
        sys.modules[pkg] = None
    spec = importlib.util.spec_from_file_location("submission", os.path.join(work, "submission.py"))
    sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
    if fp32:
        # Track 1 fixes its autocast dtype at IMPORT (_AMP_DT from the baked AMP) and re-checks it at load; setting AMP after
        # the import is a no-op (found on 09-28: the first --fp32 run still reported amp float16). Force the fallback itself.
        sub.AMP = None
        sub._AMP_DT = None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sub.predict(X[:32], None)
    load_line = buf.getvalue().strip()
    t0 = time.perf_counter(); o = sub.predict(X, None); dt = (time.perf_counter() - t0) / len(X)
    P, L, U = o["prediction"], o["lower"], o["upper"]
    ok = (set(o) == {"prediction", "lower", "upper"} and all(a.shape == Y.shape and a.dtype == np.float32 and np.isfinite(a).all()
                                                               for a in (P, L, U)) and bool((L <= P).all() and (P <= U).all()))
    r, k, mv = float(np.mean(sc.rel_l2_per_sample(P, Y, 2))), float(np.mean(sc.tke_rel_l2_per_sample(P, Y, 2))), sc.mvpe_rel_l2(P, Y)
    perm = np.random.default_rng(0).permutation(len(X))
    Ps = sub.predict(X[perm], None)["prediction"]
    stable = all(abs(a - b) < 1e-5 for a, b in ((r, float(np.mean(sc.rel_l2_per_sample(Ps, Y[perm], 2)))),
                                                 (k, float(np.mean(sc.tke_rel_l2_per_sample(Ps, Y[perm], 2)))),
                                                 (mv, sc.mvpe_rel_l2(Ps, Y[perm]))))
    sps = sc.score_sps(sc.aggregate_sps(P, Y, 2, lower=L, upper=U)[0])
    np.savez(out, pred=P, lower=L, upper=U)
    res = dict(archive=zp, fp32=fp32, load_line=load_line, amp_active=getattr(sub, "AMP_ACTIVE", None), contracts=ok,
               permutation_stable=stable, rel=r, tke=k, mvpe=mv, sps=sps, ms_per_window=dt * 1e3,
               submittable=bool(ok and stable))
    json.dump(res, open(out.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
