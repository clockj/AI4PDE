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
* **One build per tree.** Two builds in one tree overwrite each other's baked constants, and on 09-28 such a build still
  ended `BUILD_FINAL OK`. `build_final.sh` refuses to start while `Track_1/submit_fno/.build_final.lock` exists; a lock
  left by a killed job names its host and pid: check the job is gone, then delete the file.

## The packages of record

```bash
RPDE_FORCE=1 \
RPDE_REF_T1=../sub_candidates/final/track1_k27fin_a080.zip \
RPDE_REF_T2=../sub_candidates/final/T2_MRGP2_track2_MRGP2.zip \
bash Track_1/submit_fno/build_final.sh /fs/scratch/PAS1187/qwu/tmp/realpde/<outdir>
```

It must end with `BUILD_FINAL OK`. That line means: Track 1 `SUBMITTABLE`, Track 2 `SUBMITTABLE` as an adaptive package,
`CAUSALITY OK`, `BAKED CONSTANTS OK` (since 09-28, `Track_1/verify/baked_constants.py` reads every `RPDE_SHIP_*` knob,
ALPHA, AMP, the path constants and the band's md5 back out of both archives: 41 checks), and (with the two `RPDE_REF_*`)
every data file inside each archive byte-identical to the upload. The code files differ from the uploads since the
refactor of 09-28 (guards, fewer knobs); outputs are identical at tolerance 0. For byte-identical code too, build from
`fdac00b`; the release is tag `final-20260927`. The archive
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

## The Decision-phase candidate

```bash
RPDE_REPRODUCE_GO=1 RPDE_REPRO_OUT=<checkpoints> RPDE_REPRO_LOG=<scratch> RPDE_REPRO_STAGE="dp_build" bash reproduce.sh decision
```

`decision` builds with the record's constants plus `RPDE_SHIP_PREV_SPACE=target` and the committed band of its recipe.
Two variables select the recipe: `RPDE_DP_RESIM_FFNO` and `RPDE_DP_RESIM_VIT` (default 0.25 each = "T4 all", the owner's
choice of 09-29, tag `dp_resim_all`; `RPDE_DP_RESIM_VIT=` empty = "T4 F-FNO only", tag `dp_resim_ffno`). Bands:
`Track_1/bands/dp_resim_{all,ffno}_sigblend_lam0.5.npz` with `Track_1/sigblend/band_dp_resim_{all,ffno}_sigblend_lam0.5.json`;
any other pair has no committed band and is refused unless `RPDE_REPRO_FITBAND=1`. Archives go to `RPDE_REPRO_PKG`
(default `$RPDE_REPRO_LOG/decision/pkg`), never to `../sub_candidates`. Rebuilt from the committed bands, both candidates'
archives equal the earlier builds file by file (jobs 7580924, 7580925). The candidate's checkpoints are in
`Track_1/unet/` (`*_resim*`); `reproduce.sh` refuses an `RPDE_REPRO_OUT` that resolves to `Track_1/unet/` while it holds
checkpoints, so point it at a scratch copy. Its built archives are kept in
`/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/candidate/` with `SHA256SUMS`. **Nothing is staged; no upload is decided.**

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
