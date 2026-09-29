# OOD stream instrument (Decision phase, wave 1)

Turns an ensemble trained without a whole flow regime into a Track-2 package with a band refitted to that ensemble,
streams the held-out regime through the **real package code** (`Track_2/stream_eval.run` on the extracted archive), and
reports the composite (`final_v9.W0` without `time`) per trajectory and pooled, for a control and knob variants.

## Commands

```bash
cd /users/PAS1187/qwu/workspace/realpde/main/experiments/decision_20260928/ood

# full-size regime ensembles (from dec0928/ckpt/<REGIME>/<regime>_{ffnosim_s<k>,vitp_s1,vitc_s3,vitc_s1}.pth)
bash ood_submit.sh a20_mirror A20 mirror          # 4 F-FNO+sim s1-4, VITP s1, VITC s3 (the package of record's shape)
bash ood_submit.sh a10_mirror A10 mirror
bash ood_submit.sh re_mirror  RE  mirror  01:45:00
bash ood_submit.sh a20_six    A20 six             # arm T1: F-FNO+sim s1-6 + the same two ViTs
bash ood_submit.sh a20_vitc2  A20 vitc2           # arm T2: VITC s1 + s3
OOD_MEMBERS="p1.pth p2.pth ..." bash ood_submit.sh NAME A20 list     # any explicit member list (F-FNO first)

# in distribution: the five twins of record with the band of record, on the 8 VAL_TRAJ trajectories
OOD_BAND=record bash ood_submit.sh id_twins ID twins

# chunk 2 (2026-09-28): readiness and the whole wave of a regime, in at most two afterany chains
bash ood_ready.sh A20                  # READY / NOT READY per layout (checkpoint + .done sentinel, or REUSED.txt)
bash ood_wave.sh A20 [A10 RE]          # mirror: all variants + c_* + re-blank m1-m3 (control, statsmask); six, vitc2: frozen base control
# options of ood_run.sh (env):
#   OOD_CSET=1                  add the 16 one-at-a-time stream-constant variants c_<knob>_<value> (arm I5)
#   OOD_REBLANK="m1 m2 m3"      stream OOD_REBLANK_VARIANTS (default control statsmask prev_input) on re-blanked
#                               recordings (ood_reblank.py; arm I2); json <variant>__<mask>.json, own report section
#   OOD_BANDSTYLE=ship          static band fitted the shipping way (val_seen AoA 15-20 minus the regime, in sample for the
#                               ensemble) instead of VAL_TRAJ minus the regime (held out); refuses at RE
#   OOD_SKIP_EXISTING=1         do not re-stream a variant whose json exists
#   OOD_DEPEND=<jobid>          (ood_submit.sh) start after that job
# the report also gives composite by revealed windows n and the arm-I4 hindsight bound (no GPU: python ood_report.py RUN)

# re-run parts of a run (e.g. new variants on an existing build)
OOD_STAGES="variants stream report" OOD_VARIANTS="control base famw" bash ood_submit.sh a20_mirror A20 mirror
```

Everything a run writes is under `/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ood/runs/<NAME>/`: `members.txt` (configs
checked), `band/` (fit split, static band, SIGBLEND npz + json), `build/` (both archives and the build log),
`pkg/<variant>/` (extracted package, `VARIANT.diff`), `json/<variant>.json` (every window), `report.md`,
`summary.json`, `log/`. Job logs: `ood/logs/rp_dp_ood_<NAME>_<id>.log`.

**Each run has its own source tree**, `runs/<NAME>/src` = `git archive` of the pinned commit (the HEAD of the scratch
worktree `ood/wt_ood`, 4e5ed37; `OOD_COMMIT` overrides), with `Track_1/val_{seen,unseen}.npz` symlinked from the main
tree. The build `sed`s `submission.py` / `_fno.py` and the band fit imports `_fno.py`, so two runs sharing one tree
corrupt each other: on 09-28 two jobs on different nodes shared `wt_ood` behind a `flock`, which does not serialise
across nodes on scratch, and one Track-2 archive was built from the unsubstituted defaults. The run checks that the three
`sed`ed files equal the commit before and after the build, and that the Track-2 gate's load line carries the shipped
stream constants.

## What each stage does

| stage | what | code driven |
|---|---|---|
| members | derive the member list from the layout; refuse a member whose training set contains an evaluation or fit-split trajectory, an all-data model, or sim without `--sim-exclude-held` | `ood_lib.py members` |
| band | fit split = `model.VAL_TRAJ` minus the regime. Static band: `fit_band.py --ckpt E --err-ckpt E --alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored --fit-on unseen --fit-aoa <AoA 15-20 left in the split>` (inflate 1.39, smoothing 3 = defaults). SIGBLEND: `dump_twins.py --val <fit split> --twins E`, `make_k0_band.py` (λ 0.5, inflate 1.29, 16 bins) | `fit_band.py`, `sigblend/*` unchanged |
| build | `build_final.sh` with `RPDE_A = RPDE_M = E`, `RPDE_BAND`/`RPDE_SIGBLEND` = the regime band, `RPDE_FORCE=1`; must end `BUILD_FINAL OK` (both gates, causality) | unchanged |
| variants | copies of the extracted archive, constants rewritten with `build_round13.sh`'s own patterns; diff checked | `ood_variants.py` |
| stream | one process per variant; `--causal-check` on `control`; revealed target TARGET-normalised | `ood_stream.py` → `stream_eval.run` |
| report | pooled per group, per trajectory, first 3 / first 14 windows, deltas to `control` and `base` with counts, first-window identity to `frozen` | `ood_report.py`, `ood_score.py` |

Regimes (`ood_lib.py regime`): `A20`/`A10` = every `train_real` trajectory at that AoA; `RE` = `re_split.SETS`
(groups `relo`, `rehi`, `rein`); `ID` = `VAL_TRAJ` (groups `all8`, `a15_20`).

Variants: `frozen` (all stream knobs off), `base` (asymmetric R1 only: D1c, VARCAL, TSM, BLK off), `control` (package of
record + `PREV_SPACE "target"`), `prev_input` (package of record as built), `statsmask` (control + `STATS_MASK`),
`famw` (control + `FW_N0 20.0`). All but `prev_input` assume target normalisation.

## Scoring

`ood_score.window_records` stores per window the kit's rel / tke / mvpe errors, the SPS accuracy factor a_i, the
containment sum and the scored / inside counts; any subset is then scored exactly as the kit scores that subset alone
(`selfcheck` compares the pooled result with the kit's `aggregate_sps`; the run fails above 1e-6; float32 rounding in the kit is ~1e-8).
Composite = W0 · (rel, tke, mvpe, sps). C = 100 · Σ con / Σ scored. Local ms/step is recorded, never in the composite.

## Known limits

* The static band is fitted on the AoA 15-20 part of the fit split, as the shipping construction does. At `A20` that
  part is one trajectory (21600_15, 42 windows); at `A10` and `RE` three.
* The fit split's errors are out of sample for a regime ensemble; the shipped band's static part is fitted on the
  twins' in-sample `val_seen` errors. The regime band is therefore a little wider than what the organisers' rebuild
  would ship for the same members.
* FAMW needs exactly two fused groups; the package fuses by architecture, so a layout with a third family (e.g. FNO w96
  next to F-FNO and ViT) runs FAMW off. `json/<variant>.json` records which corrections are live (`active`).
