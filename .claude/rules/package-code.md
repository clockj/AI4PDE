---
paths:
  - "Track_1/submit_fno/**"
  - "Track_2/submit/**"
  - "Track_2/build_t2.py"
---

# The shipped code

`submission.py` (one per track) and `_fno.py` are what the evaluator runs. Every change here is judged by two questions:
is the output of the packages of record unchanged, and does the package still load on hardware we have never seen.

* **A knob turned off reproduces the previous output bit for bit.** A new feature enters as a module-level constant whose
  default is "off", a `sed` line in `build_round13.sh`, and an entry in `build_final.sh` if it ships.
* **Verify against the uploads, at tolerance 0**: `python Track_1/verify/final_identity.py` compares a rebuilt package
  with `../sub_candidates/final/` on Track 1 at the batch sizes the evaluator uses (14, 56, 99, 336 in one process) and
  on Track 2 over a full stream with revealed targets. Then `build_final.sh` with `RPDE_REF_T1` / `RPDE_REF_T2`.
* **The constants in the tree are defaults.** ALPHA 0.60, `JOINT_GAIN` False, `EBAND` / `SIGPOW` None, every stream
  knob off. `build_final.sh` is the list of what ships.
* **`_fno.py` is shared**: both archives carry it, about 25 repository tools import it, and inside the repository it
  builds F-FNO members from `Track_1/unet/models_alt.py` (`FFNO_IMPL = "auto"`), in an archive from its own vendored
  copy. Do not prune it; `Track_2/submit/_fno.py` is a symlink to it and must never be replaced by a copy.
* **The band formula exists twice**: `_fno.predict_fno` (Track 1, `EBAND`) and `Track_2/submit/submission.py::_step`
  (`SIGPOW`). A change to one is a change to both.
* **Anything decided at load time can fall back** (`AMP`, the Track-2 graph, the pinned pools). Each fallback must give
  the same numbers, only slower, and must not raise.
* **Containers** have no `scipy`, `h5py`, `matplotlib` or `pandas`; the platform may load `submission.py` with the
  extract directory off `sys.path`. Import only `torch`, `numpy` and the sibling `_fno.py`.
* **Bounds**: finite everywhere, `lower ≤ prediction ≤ upper`, `p` channel zero. A malformed bound zeroes every
  subscore, so the guards against non-finite values stay in both tracks.
* **Removed on 2026-09-28** and refused by `build_round13.sh`: `GRAPH*`, `SMALL_GRAPH*`, `BATCH_FP`, `GPU_FP`,
  `LN_MANUAL`, the `band.npy` fallback (Track 1); `SIGEMA*`, `STACKED_OUT`, `FALLBACK_FP`, `R1_SNAP` (Track 2). Commit
  `fdac00b` is the last tree that has them and the one that rebuilds the uploads byte for byte.
* **Track 1 never pools across the windows of a call.** `validate_strict.py` asserts permutation invariance, which a
  pooled statistic can still pass: read the code, not only the gate.
* **Latency**: the evaluator sends Track 1 14–99 windows per call; Track 2 is one window per step with a fixed cost of
  about 2.3 ms per step for moving data between processes. Measure only interleaved, on an idle node.
* Never edit these files, or a build script, while a build runs. Build in a scratch worktree.
* **One build per tree.** Two builds in one tree on different nodes overwrote each other's baked constants and the
  build still ended `BUILD_FINAL OK` (`flock` does not serialise across nodes on scratch). Since 09-28 `build_final.sh`
  takes a lock file (`Track_1/submit_fno/.build_final.lock`, `noclobber`, removed on exit) and, after the gates, reads
  every `RPDE_SHIP_*` knob, ALPHA, AMP, the path constants and the band's md5 back out of both archives
  (`Track_1/verify/baked_constants.py`, 41 checks); a mismatch fails the build. A lock left by a killed job names its
  host and pid.
