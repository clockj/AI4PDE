# Status — 2026-10-06, the final submission

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
* **What is sent: the method of record, not the candidate**, as training code only on the orphan branch
  **`final-submission`** (worktree `../code-finalsub`, one commit, 19 files): `fdac00b`'s training, band and build code
  reorganised and stripped of every closed experiment, the two shipped files byte-identical, one entry point
  `run_track2.sh`. Old and new code give bit-identical weights on the CPU and the archive of record from the
  checkpoints of record. Record and checks: `final_package/README.md`.
* **The candidate ("T4 all", `reproduce.sh decision`) stays unsent**: a different model measured against a number it
  never produced (+0.0125 expected on the board). Record: `experiments/decision_20260928/`; checkpoints `*_resim*`.

## Open before the package is sent (the owner sends it)

1. **The organizers' template**: fetch the starting kit (`realpde_final_starting_kit_track2.zip`, Files tab, login)
   and the invitation e-mail; rearrange the branch to it and run `final_package/verify/job_verify_branch.sh` again.
2. **`TEAM` and `CONTACT`** in the branch's `README.md` (placeholders `@@TEAM@@`, `@@CONTACT@@`).
3. **`train_real/7575_0.h5`** (a copy of `6300_0`, which the data page says not to use) is in the training set of every
   network of record; the manual discloses it. The owner decides the wording.
4. **Not verified:** a 400-epoch run of the branch itself (run at 2 epochs; the recipes are those of the 09-25
   retrain, +0.0006), and the organizers' hardware (expect the reseeding spread, about 0.012).
5. **Push** `final-submission` and `qwu/dev`; agents never push.

## Settled

* 10-05: the Track-2 gates read the data from a literal path of this cluster; they read `$RPDE` now (`REPRODUCE.md` §9.1).
* 09-28: `PREV_SPACE = "target"` is a fix and is **not** in the package of record: under the 10-04 rule it stays out.
* The newest handoff is `../handoffs/HANDOFF_20261006.md`.
