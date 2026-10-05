# The final submission: Track 2's training package (2026-10-05)

What goes to the organizers for the final evaluation, how it is assembled, and what was checked. The owner sends it;
agents never do (`CLAUDE.md` rule 1).

## What the organizers ask for

Read on 2026-10-05 from the public competition metadata (`https://www.codabench.org/api/competitions/17385/`, no login):

* **FAQ, "How and when do finalists submit their training code?"** *"Invited teams receive an email from us. Each
  invited team sends one training package for each track it is invited to, by **9 October 2026, 24:00 UTC**. The
  templates and the full specification are in the starting kit of the Final Decision Phase, listed on the Files tab."*
* **FAQ, on fairness at retraining:** *"organizers hold only the released training data, so we accept training code
  only: no new simulation data, no code that runs new simulations from scratch, no model weights, and no pretrained
  models from other sources."*
* **The organizers' clarification of 2026-10-04:** the submitted package is evaluated on the development set too, and
  that result must reproduce the leaderboard row, or the team may be disqualified.
* **The Final Decision Phase on Codabench is the organizers' own**: "Organizer-controlled final verification phase on
  private test data", open 2026-09-28 00:00 – 09:00 UTC, now `Previous`, on both tracks. No phase accepts an upload,
  so no package can acquire a leaderboard row any more. `STATUS.md` and `CLAUDE.md` rule 1 said "one submission per
  track to 10-25"; that slot does not exist.
* **The template is not public.** `realpde_final_starting_kit_track2.zip` ("T2 Final Starting Kit v1", 5 649 bytes,
  uploaded 2026-09-28) needs a Codabench login. **Until the owner fetches it, the layout below is ours, not theirs.**

Track 1 finished #20 and is not invited; there is nothing to send for it.

## What is sent

The method of the leaderboard row: T2R24, `T2_MRGP2_track2_MRGP2.zip`, sha256 `1aa2896f…`, final 82.055553. Not the
Decision-phase candidate ("T4 all"): it is a different model, the leaderboard number is not its number, and it can no
longer get one of its own.

The package is **training code only**, taken from commit **`fdac00b`**, the commit both uploads rebuild from byte for
byte, code files included (`REPRODUCE.md` §13.5). A package cut from today's tree would build a `submission.py` that
differs from the uploaded one in 300 lines (the guards and the removed knobs of the 09-28 refactor) and would carry the
candidate's recipe and bands next to the record's.

| piece | source |
|---|---|
| the files of `allowlist.txt` | `fdac00b`, byte for byte: every file the pipeline opened in a traced run from the released data (`verify/job_trace.sh`), and nothing else |
| `patches/0001-…` | one line of `Track_2/t2_common.py`: the Track-2 gates read the data from `$RPDE` (`REPRODUCE.md` §9.1 item 7) |
| `overlay/run_track2.sh` | the entry point: three variables, one command; sets every variable of §9.1, creates the one symlink, refits the band |
| `overlay/README.md` | the manual the organizers read |

It holds no checkpoint, no band and no symlink. `reproduce.sh final` reuses the committed band by default; the package
has none, and its entry point always refits (`RPDE_REPRO_FITBAND=1`), because a band is fitted on our twins and the
organizers accept no fitted artefact.

```bash
TEAM="<team name on Codabench>" CONTACT="<e-mail>" bash final_package/make_package.sh /fs/scratch/PAS1187/qwu/tmp/realpde/final1005/dist
```

writes `realpde_track2_training_package/`, `.tar.gz` and `.zip` (the same bytes on every run) and prints their sha256.

## What was checked

All on Ascend (A100, torch 2.2.2), inside `verify/sandbox.sh`: bubblewrap, the author's home and scratch **empty**,
the environment cleared, the release / the kit / the Python environment bound at neutral paths. A literal path of ours
fails there as it would on the organizers' cluster; that is how item 7 of `REPRODUCE.md` §9.1 was found.

| job | tree | check | result |
|---|---|---|---|
| 7681561 | all of `fdac00b` + the patch | the eleven checkpoints of record + the committed band, against the uploads | gates 2/2, causality OK, **Track 1 and Track 2: all 9 files byte-identical**, `BUILD_FINAL OK` |
| 7681561 | the same, under `strace` | `run_track2.sh` from the released data, 2 epochs | `RUN_TRACK2 OK` in 31 min; 25 files of the commit opened (`allowlist.txt`); built `_fno.py` = the upload's, `submission.py` one line apart (`SIGPOW`) |
| 7681858 | the curated package, from its `.tar.gz` | `run_track2.sh` from the released data, 2 epochs | every stage, band refit, gates 2/2, causality OK, `RUN_TRACK2 OK` in 26 min |
| 7681858 | the curated package, from its `.zip` | the checkpoints and the band of record, against the uploads | **all 9 files byte-identical on both tracks**, `BUILD_FINAL OK` |
| 7681858 | the curated package, from its `.zip` | two training stages on the release without `train_real/7575_0.h5` | both trained (81 real trajectories) |
| 7682454 | the draft as it stands (`.tar.gz` `aed17c0e…`, `.zip` `0309a3dc…`; the entry point had gained its scipy check) | the three checks of 7681858 again | `RUN_TRACK2 OK`; **all 9 files byte-identical on both tracks**; both stages trained without `7575_0.h5` |

The two archives extract to the same tree and `MANIFEST.sha256` verifies. Peak host memory 21 GB; `$WORK` 4.2 GB; 0.5 GB
of intermediates inside the package tree. The archive of record itself: sha256 `1aa2896f…8f54bf` (= the leaderboard
row's), 9 files, 34.6 MB extracted, constants read back and equal to `REPRODUCE.md` §13.2 – 13.3, no `PREV_SPACE`.

## What is not checked, and what the owner decides

1. **The organizers' template.** Fetch `realpde_final_starting_kit_track2.zip` from the Files tab of the Track 2 page,
   and the invitation e-mail. The package is then renamed and rearranged to their layout and `verify/job_verify.sh` is
   run again on the result.
2. **`TEAM` and `CONTACT`** in the manual.
3. **`train_real/7575_0.h5`.** The data page says not to use it: it is a copy of `6300_0` (checked: `u` and `v`
   identical). Nothing in this repository ever excluded it, so the eleven networks of record were trained with it, and
   the manual says so in "Data". A re-training without it runs (checked) and is no longer the submitted method.
4. **A full 400-epoch run of this exact package.** The recipes and the training code are those of the 09-25
   re-training (+0.0006, `REPRODUCE.md` §12); the package itself was run at 2 epochs.
5. **Their hardware.** Expect the reseeding spread, about 0.012 on the composite (`REPRODUCE.md` §7), and a `time` of
   their own.

## Files

```
final_package/
  README.md           this file
  make_package.sh     assembles the package from fdac00b + allowlist + patches + overlay
  allowlist.txt       the files of fdac00b that go in
  patches/            applied on top of the commit
  overlay/            run_track2.sh, README.md (the organizers' manual)
  verify/             sandbox.sh (a third party's view of this machine), job_trace.sh, job_verify.sh
```
