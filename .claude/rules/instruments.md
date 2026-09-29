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
* **Out of distribution** (the private set's regime): `aoa_holdout_eval.py` reads a single checkpoint at a held-out
  angle. Since 09-28 `experiments/decision_20260928/ood/` reads **ensembles** trained without AoA 20, AoA 10 or the six
  Re of `re_split.excluded()` (`relo` below, `rehi` above, `rein` inside the training range), on the real package path
  with a refitted band and the full stream stack (`EVALUATION.md` §3b). Decide at ensemble level on the stream stack;
  single members are a screen. The twins exclude no Re cell and cannot measure extrapolation in Re.
* **The reseeding spread is the yardstick**: four-member ensembles differ by up to 0.034 (relo), 0.011 (rehi), 0.012
  (in distribution) composite; a single F-FNO + sim member by 0.02 – 0.08 `wfinal` (`EVALUATION.md` §4).
* **Track-2 reads** use `stream_eval.py`; it must feed the revealed target normalised as the official runner does
  (target statistics). Results before 2026-09-28 used the input statistics.
* A verifier for a feature no local data exercises belongs in `verify/`; the strict gate passes such a package as if
  the feature were absent.
* Scripts find each other through `sys.path` inserts relative to `Track_1/`. A file that moves one level down needs the
  shim described in `Track_1/closed/README.md`.
