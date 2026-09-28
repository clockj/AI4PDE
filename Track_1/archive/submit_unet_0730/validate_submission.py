"""End-to-end validation of a Track 1 submission directory against the OFFICIAL scorer.

Runs predict() on val_unseen.npz, writes predictions.npz + ref/targets.npz exactly as the
platform would, then invokes the kit's scoring.py and prints scores.json.

    python validate_submission.py                    # validate ./ (this dir)
    python validate_submission.py --dir /tmp/extract # validate an extracted zip
    python validate_submission.py --blocked          # + simulate the eval image package set

--blocked rejects scipy/h5py/matplotlib/pandas at import time, the set the Track 1 evaluation
image lacks (smoke_test_kit.py MISSING_IN_EVAL_IMAGE). A submission that passes with --blocked
imports nothing the container is missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
KIT = "/users/PAS1187/qwu/RealPDE/realpde_t1_starting_kit_v7/realpde_t1_starting_kit_v7"

MISSING_IN_EVAL_IMAGE = {"scipy", "h5py", "matplotlib", "pandas"}


class _BlockedTopLevel:
    """Same technique as the kit's smoke_test_kit.py: reject named top-level modules."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ModuleNotFoundError(
                "simulated evaluation image without %s: blocked %s"
                % (name.split(".")[0], name))
        return None


def load_submission(submission_dir: str):
    sys.path.insert(0, submission_dir)
    spec = importlib.util.spec_from_file_location(
        "participant_submission", os.path.join(submission_dir, "submission.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise SystemExit("submission.py must define predict(input_array, metadata=None)")
    return module


def normalize_output(out):
    """Mirror ingestion.normalize_prediction_output(): float32, bounds only if both given."""
    if isinstance(out, dict):
        res = {"prediction": np.asarray(out["prediction"], dtype=np.float32)}
        if "lower" in out and "upper" in out:
            res["lower"] = np.asarray(out["lower"], dtype=np.float32)
            res["upper"] = np.asarray(out["upper"], dtype=np.float32)
        elif ("lower" in out) != ("upper" in out):
            raise SystemExit("predict() returned only one of lower/upper; set both or neither")
        return res
    return {"prediction": np.asarray(out, dtype=np.float32)}


def check_contract(res, target):
    pred = res["prediction"]
    if pred.shape != target.shape:
        raise SystemExit(f"prediction shape {pred.shape} != target {target.shape}")
    if pred.dtype != np.float32:
        raise SystemExit(f"prediction dtype {pred.dtype} != float32")
    if not np.all(np.isfinite(pred)):
        raise SystemExit("prediction contains non-finite values -> all subscores would be 0")
    if "lower" in res:
        lo, up = res["lower"], res["upper"]
        if lo.shape != pred.shape or up.shape != pred.shape:
            raise SystemExit("bounds shape mismatch")
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(up))):
            raise SystemExit("bounds contain non-finite values")
        if not (np.all(lo <= pred) and np.all(pred <= up)):
            raise SystemExit("bounds violate lower <= prediction <= upper")
        print("  contract: prediction + bounds OK (finite, shapes match, lower<=pred<=upper)")
    else:
        print("  contract: bare prediction array OK (finite, shape and dtype match)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE, help="directory containing submission.py")
    ap.add_argument("--val", default=os.path.join(TRACK1, "val_unseen.npz"))
    ap.add_argument("--work", default=os.path.join(HERE, "_validate_run"))
    ap.add_argument("--blocked", action="store_true",
                    help="simulate the evaluation image's missing packages")
    args = ap.parse_args()

    if args.blocked:
        sys.meta_path.insert(0, _BlockedTopLevel(MISSING_IN_EVAL_IMAGE))
        for mod in sorted(MISSING_IN_EVAL_IMAGE):
            try:
                __import__(mod)
                raise SystemExit(f"blocker inactive: {mod} still importable")
            except ModuleNotFoundError:
                pass
        print("blocker active:", ", ".join(sorted(MISSING_IN_EVAL_IMAGE)), "rejected")

    d = np.load(args.val)
    X, Y = d["x"], d["y"]
    print(f"validating {args.dir} on {os.path.basename(args.val)}: {X.shape[0]} windows")

    module = load_submission(args.dir)
    t0 = time.perf_counter()
    out = module.predict(X, metadata=None)
    elapsed = time.perf_counter() - t0
    res = normalize_output(out)
    check_contract(res, Y)

    per_window = elapsed / X.shape[0]
    print(f"  predict(): {elapsed:.2f}s total, {per_window*1000:.2f} ms/window "
          "(includes one-time model load)")

    ref = os.path.join(args.work, "ref")
    os.makedirs(ref, exist_ok=True)
    np.savez_compressed(os.path.join(args.work, "predictions.npz"),
                        mean_t_neural_s=np.array([per_window], dtype=np.float32), **res)
    np.savez_compressed(os.path.join(ref, "targets.npz"), target=Y)

    out_dir = os.path.join(args.work, "scores")
    os.makedirs(out_dir, exist_ok=True)
    run = subprocess.run([sys.executable, os.path.join(KIT, "scoring.py"), args.work, out_dir],
                         capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f"official scoring.py failed:\n{run.stderr}")

    with open(os.path.join(out_dir, "scores.json")) as f:
        scores = json.load(f)
    print("\nofficial scoring.py:")
    for k, v in scores.items():
        print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")
    if "error" in scores:
        raise SystemExit("scorer reported an error -> submission would score zero")


if __name__ == "__main__":
    main()
