---
name: build-package
description: Build and gate the Track 1 and Track 2 submission archives (the final recipe, or a variant that changes one baked constant), check them against the uploaded packages, and stage them with registered predictions. Use before any upload or any package-against-package comparison.
---

# Build and gate a package

## Before you start

* The tree is committed (`git status --short` is empty). A build `sed`s constants into `Track_1/submit_fno/submission.py`,
  `Track_1/submit_fno/_fno.py` and `Track_2/submit/submission.py`.
* You are on a GPU node (an `rp_*` job, see the `osc-jobs` skill), not the login node.
* `Track_1/val_seen.npz` and `Track_1/val_unseen.npz` exist. If not: `( cd Track_1 && python make_val.py --data <train_real> )`.
* Build in a scratch worktree when anything else may touch the main tree:
  `git worktree add --detach /fs/scratch/PAS1187/qwu/tmp/realpde/<name> HEAD`, with
  `RPDE_U=<repo>/Track_1/unet` pointing at the checkpoints.

## The packages of record

```bash
RPDE_FORCE=1 \
RPDE_REF_T1=../sub_candidates/final/track1_k27fin_a080.zip \
RPDE_REF_T2=../sub_candidates/final/T2_MRGP2_track2_MRGP2.zip \
bash Track_1/submit_fno/build_final.sh /fs/scratch/PAS1187/qwu/tmp/realpde/<outdir>
```

It must end with `BUILD_FINAL OK`. That line means: Track 1 `SUBMITTABLE`, Track 2 `SUBMITTABLE` as an adaptive package,
`CAUSALITY OK`, and (with the two `RPDE_REF_*`) every file inside each archive byte-identical to the upload. The archive
bytes themselves differ because of zip timestamps. `RPDE_FORCE=1` skips the refusal to build while other `rp_*` jobs run;
drop it when the build's timing will be quoted.

## A variant

1. Change **one** `RPDE_SHIP_*` value, by exporting it in front of the command above with a different `RPDE_TAG`.
   `build_final.sh` lists every constant; `build_round13.sh` holds the `sed` line for each.
2. A new member set or a new ALPHA needs a new band: `RPDE_REPRO_FITBAND=1 bash reproduce.sh final` refits the static
   band and re-derives the SIGBLEND constants. Never reuse a band across member sets.
3. Build the same variant from the five held-out twins (`RPDE_A="<twins>"`) to read it locally; the all-data build
   supplies weights, never evidence.
4. Compare with the package of record on identical inputs: `python Track_1/verify/final_identity.py` for outputs,
   `Track_2/stream_eval.py --submission <dir>` for Track-2 scores.

## Staging

1. `sha256sum` both archives.
2. Write `../sub_candidates/<YYYYMMDD>/README.md`: file, sha256, what changed, the held-out evidence, **all five predicted
   subscores with brackets**, and the rule that decides between this package and the alternative. Add the PENDING row to
   `leaderboard/board.jsonl`.
3. Tell the owner. The owner uploads.

## What goes wrong

* `NOT SUBMITTABLE: stateless ...` on Track 2 means the package was gated as a frozen one; `RPDE_T2_ADAPTIVE=1` (set by
  `build_final.sh`) gates it as adaptive.
* An unset knob of `build_round13.sh` falls back to old behaviour (round-13 members, ALPHA 0.90, fp32). Pass them all.
* A band whose `p` is 0.225 is a SIGBLEND band: it must ship with `RPDE_SHIP_EBAND` and `RPDE_SHIP_SIGPOW` from its
  sidecar json, or it is about 4× too wide.
