---
paths:
  - "Track_1/*.py"
  - "Track_1/verify/**"
  - "Track_1/qd_k/**"
  - "Track_2/stream_eval.py"
  - "Track_2/validate_t2.py"
  - "EVALUATION.md"
---

# Local instruments

* **Every local number is a sign before it is a size.** The corrections are in the `evaluate-change` skill and
  `EVALUATION.md` §5.
* **Defaults come from `Track_1/opoint.py`**, the operating point of the packages of record: ALPHA 0.80 with
  JOINT_GAIN, interval centred on y₀, static band `--band-p 0.45 --inflate 1.39 --fill-unscored`, SIGBLEND on top.
  `Track_1/submit_fno/build_final.sh` is the authority; if the two disagree, `opoint.py` is the bug.
  `Track_1/submit_fno/fit_band.py` keeps its own defaults because the build depends on them.
* **Read a band with `band_io.load_band`**, never with `np.load`. A SIGBLEND npz stores `base′` and `p′ = 0.225`; the
  σ-power factor is in its sidecar json. Without it the band is about 4× too wide, sps reads 27 instead of 54, and
  nothing fails. `band_io.interval` builds the interval as the package does; `sps9.sps_interval` scores it.
* **Score through `sps9`** (the kit's scorer, reconstruction verified to 2.4e-6 by `verify/diff_scoring_*.py`), and
  through the real code path (`_fno.predict_fno`, an extracted package), never a re-implementation.
* **Screen at the ensemble size that ships**: `ens_v9.py --aoa 15 20` for member sets, `compare_v9.py` for single
  checkpoints, each with its own band.
* **The AoA holdout** (`aoa_holdout_eval.py`) is the only out-of-distribution instrument. The twins do not exclude any
  Re cell, so they cannot measure extrapolation in Re.
* **Track-2 reads** use `stream_eval.py`; it must feed the revealed target normalised as the official runner does
  (target statistics). Results before 2026-09-28 used the input statistics.
* A verifier for a feature no local data exercises belongs in `verify/`; the strict gate passes such a package as if
  the feature were absent.
* Scripts find each other through `sys.path` inserts relative to `Track_1/`. A file that moves one level down needs the
  shim described in `Track_1/closed/README.md`.
