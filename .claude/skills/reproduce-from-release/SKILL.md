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
twins this run trained; use it for a new member set.

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
  the 09-25 retrain of recipe A landed at +0.0006.
* **The build is bit-reproducible.** From the same checkpoints, `build_final.sh` with `RPDE_REF_T1` / `RPDE_REF_T2`
  must print `ALL 9 FILES BYTE-IDENTICAL` twice.

## A smoke

`RPDE_REPRO_EPOCHS=2 RPDE_REPRO_SEEDS=1 RPDE_REPRO_FITBAND=1` runs every stage in about half a GPU-hour and proves the
pipeline, not the method. Its archives are not packages.
