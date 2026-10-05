# Status — 2026-10-05, the final submission

The only file that changes after a board result. Keep it under 40 lines; detail goes to `ROUNDS.md`.

| | board `final` | package (in `../sub_candidates/final/`) | registered before upload |
|---|---|---|---|
| Track 1 | **81.671149** (R37, 09-27) | `track1_k27fin_a080.zip` | 81.67 [81.58, 81.79] |
| Track 2 | **82.055553** (T2R24, 09-27) | `T2_MRGP2_track2_MRGP2.zip` | 82.000 [81.885, 82.219] |

* **Final ranks: Track 1 #20, Track 2 #8.** Track 2 is shortlisted; Track 1 is not, and nothing is sent for it.
* **What the organizers want is a training package, by 2026-10-09 24:00 UTC** (FAQ of both Codabench pages): code only,
  no weights, in the format of the "Final Decision Phase" starting kit (`realpde_final_starting_kit_track2.zip`, Files
  tab, login needed). They retrain it (8× A800), score the development set and a private set; **the development-set
  result must reproduce the leaderboard row** (clarification of 10-04) or the team may be disqualified.
* **No Codabench upload exists in this phase** (public API, read 10-05): the "Final Decision Phase" is
  organizer-controlled and closed on 09-28 09:00 UTC on both tracks. A package can no longer acquire a leaderboard row.
* **What is sent: the method of record, not the candidate.** `final_package/` assembles the Track 2 training package
  from `fdac00b` (the commit the uploads rebuild from byte for byte), with one entry point, `run_track2.sh`:
  `bash final_package/make_package.sh <scratch dir>`. Record and checks: `final_package/README.md`.
* **The candidate ("T4 all", `reproduce.sh decision`, tag `dp_resim_all`) stays unsent**: it is a different model
  measured against a number it never produced (+0.0125 expected on the board, +0.206 out of distribution, of which
  +0.709 at low Re). Record: `experiments/decision_20260928/`. Its checkpoints stay in `Track_1/unet/` (`*_resim*`).

## Open before the package is sent (the owner sends it)

1. **The organizers' template**: fetch the starting kit and the invitation e-mail; rearrange the package to it and run
   `final_package/verify/job_verify.sh` again.
2. **`TEAM` and `CONTACT`** for the manual (`make_package.sh` refuses to build a final package without them).
3. **`train_real/7575_0.h5`** (a copy of `6300_0`, which the data page says not to use) is in the training set of every
   network of record; the manual discloses it. The owner decides the wording.
4. **Not verified:** a 400-epoch run of the package itself (run at 2 epochs; the recipes are those of the 09-25
   retrain, +0.0006), and the organizers' hardware (expect the reseeding spread, about 0.012).

## Settled

* 10-05: `Track_2/t2_common.py` read the data from a literal path of this cluster, so both Track-2 gates failed on any
  other machine; it reads `$RPDE` now (`REPRODUCE.md` §9.1 item 7, found in a sandbox without the author's paths).
* 09-28: `PREV_SPACE = "target"` is a fix (+0.002 … +0.009) and is **not** in the package of record; under the 10-04
  rule the −0.0015 defect is cheaper than a change to the development-set numbers. `STATS_MASK` stays off.
* The newest handoff is `../handoffs/HANDOFF_20261005.md`.
