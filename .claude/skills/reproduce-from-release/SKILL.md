---
name: reproduce-from-release
description: Re-create the shipped checkpoints, the band and both archives from the released data with reproduce.sh, and check the result against the packages of record. Use to verify Decision-phase readiness, after a recipe or training-code change, or on a new machine.
---

# Reproduce from the release

`REPRODUCE.md` is the specification; this is the procedure.

## The command

```bash
RPDE_REPRODUCE_GO=1 RPDE_REPRO_OUT=<scratch dir for checkpoints> RPDE_REPRO_LOG=<scratch dir for logs> \
  bash reproduce.sh final
```

It trains 5 held-out twins and 6 all-data members (about 37 A100-hours in sequence), then calls
`Track_1/submit_fno/build_final.sh`. By default the build reuses the committed SIGBLEND band, which is what reproducing
the scored packages requires. `RPDE_REPRO_FITBAND=1` refits the static band and re-derives the SIGBLEND constants from the
twins this run trained; use it for a new member set. The release of 09-27 is tag `final-20260927` (4e5ed37).

## The Decision-phase candidate: `decision`

```bash
RPDE_REPRODUCE_GO=1 RPDE_REPRO_OUT=<scratch dir for checkpoints> RPDE_REPRO_LOG=<scratch dir for logs> \
  bash reproduce.sh decision
```

The eleven runs of `final` with `--aug-resim` appended, under `_resim` names, then the band and the build with
`RPDE_SHIP_PREV_SPACE=target` (`REPRODUCE.md` §14). Two variables: `RPDE_DP_RESIM_FFNO` and `RPDE_DP_RESIM_VIT`, default
0.25 each = **"T4 all"**, the owner's choice of 09-29 (tag `dp_resim_all`); `RPDE_DP_RESIM_VIT=` (empty) = "T4 F-FNO
only" (tag `dp_resim_ffno`, seven runs; the ViTs are the record's). The committed band of the recipe is reused
(`Track_1/bands/dp_resim_{all,ffno}_sigblend_lam0.5.npz` + json); another pair is refused unless
`RPDE_REPRO_FITBAND=1`. Stages: `dp_twins_vit_p`, `dp_twins_vit_c`, `dp_twins_ffno`, `dp_members_vit_p`,
`dp_members_vit_c`, `dp_members_ffno`, `dp_band`, `dp_build`; one job per stage and seed (`RPDE_REPRO_SEEDS`) runs
the eleven trainings in parallel.

**Its acceptance read** (09-28, clean clone of `467769d`, jobs 7577300 – 7577311, 41.8 GPU-h): the from-scratch twins'
package composite 77.2817 against the campaign's 77.2821 (−0.0003, bar 0.005), every trajectory within ±0.0012;
`BUILD_FINAL OK` with 41 of 41 baked constants. A V100 and every forced fallback agree within 2.9e-3 relative L2 per
window (R2). The eleven checkpoints of that run are promoted into `Track_1/unet/` (`*_resim*`).

## Splitting it over jobs

One stage per job: `RPDE_REPRO_STAGE="twins_ffno"` runs only that function and no build. Stages: `twins_vit_p`,
`twins_vit_c`, `twins_ffno`, `members_vit_p`, `members_vit_c`, `members_ffno`. Narrow seeds with `RPDE_REPRO_SEEDS="1"`.
Every stage skips a checkpoint that already exists, so a rerun continues. A V100 job is capped at 6 h and an F-FNO
member takes about 4 h 50 min there.

## Rules

* **Run from a scratch copy of the repository**, never in the working tree: `reproduce.sh` refuses to overwrite
  checkpoints in `Track_1/unet/`, and a reduced-epoch smoke refuses to run where `../sub_candidates` exists.
* **Two software stacks are verified.** The evaluators' stack (torch 2.2.2, numpy 1.26.4) and the training stack on
  Pitzer (torch 2.7.1). The scripts export `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, which torch ≥ 2.6 needs.
* **Acceptance is by score, not by checksum.** Retraining with the same seed is not bit-identical (F-FNO weights differ
  by about 0.5 %, ViT by about 10 %). A retrained package must land within 0.005 of the original on the local composite;
  the 09-25 retrain of recipe A landed at +0.0006, the 09-28 retrain of the candidate at −0.0003 (its ViT twins came
  back bit-identical on the A100). On the organizers' hardware expect the reseeding spread, about 0.012 in distribution.
* **The build is bit-reproducible.** From the same checkpoints, `build_final.sh` with `RPDE_REF_T1` / `RPDE_REF_T2`
  must print `ALL 9 FILES BYTE-IDENTICAL` twice when built from `fdac00b`; from a later tree the data files are
  byte-identical and the outputs identical at tolerance 0 (`FINAL_IDENTITY OK`), the code files differ.
* **One build per tree.** `build_final.sh` takes a lock file and checks the constants it baked
  (`Track_1/verify/baked_constants.py`); two builds in one tree overwrote each other's constants on 09-28.

## A smoke

`RPDE_REPRO_EPOCHS=2 RPDE_REPRO_SEEDS=1 RPDE_REPRO_FITBAND=1` runs every stage in about half a GPU-hour and proves the
pipeline, not the method. Its archives are not packages.
