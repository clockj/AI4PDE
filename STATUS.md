# Status — 2026-10-07, the final submission in the organizers' shape

The only file that changes after a board result. Keep it under 40 lines; detail goes to `ROUNDS.md`.

| | board `final` | package (in `../sub_candidates/final/`) | registered before upload |
|---|---|---|---|
| Track 1 | **81.671149** (R37, 09-27) | `track1_k27fin_a080.zip` | 81.67 [81.58, 81.79] |
| Track 2 | **82.055553** (T2R24, 09-27; **Codabench submission 948792**) | `T2_MRGP2_track2_MRGP2.zip` | 82.000 [81.885, 82.219] |

* **Final ranks: Track 1 #20, Track 2 #8.** Track 2 is shortlisted; Track 1 is not, and nothing is sent for it.
* **What the organizers want, by 2026-10-09 24:00 UTC** (invitation e-mail of 09-29 + `FINAL_SUBMISSION.md` of the
  Final-phase kit, `../realpde_final_starting_kit_track2.zip`): a **Docker image** on their Track 2 base image in a
  private Docker Hub repo readable by `w3nhao`; a **zip on Google Drive** (Dockerfile, `solution.py`, code, README,
  METHOD, the training logs); an **e-mail** with team, username, track, image + digest, **submission id 948792**, the
  link. They run `solution.train('/data', '/output', 'cuda')` on one A800 80 GB, 16 cores, 128 GB, **within 24 h**,
  no network, `train_real` **without `7575_0.h5`**; then `solution.get_ttt_model('/output', 'cuda')` on the private
  set and on the development set, which must reproduce the row.
* **What is sent: branch `final-submission`** (worktree `../code-finalsub`, orphan): the method of record as training
  code in the kit's shape — `solution.py` (the eleven recipes, several trainings at a time on the one GPU, band,
  archive, gates), the kit's Dockerfile, `unet/ band/ build/ submission/`, `README.md` + `METHOD.md` from the
  templates, `logs/` of the runs of record. Record and checks: `final_package/README.md`.
* **The candidate ("T4 all", `reproduce.sh decision`) stays unsent** (`experiments/decision_20260928/`).

## Open before the package is sent (the owner sends it)

1. ~~Build and push the image~~ **done 10-08 05:51 UTC**: `williamqwu/realpde-t2@sha256:f5319731…` (public, not
   private; the manifest's layers verified identical to the image of 952d1e1 that the rehearsals run).
2. **The zip** (`git archive` of the branch) on Google Drive; **the e-mail** (text in the record, image and digest
   filled in; the Drive link remains).
3. **The 24 h limit on one GPU cannot be met** (measured: ~37 h; concurrency on one card gains nothing). The e-mail
   asks for two GPUs or 48 h; `solution.py` uses every GPU it sees. **The full rehearsal from the pushed image on
   2 × A100 (7712670) passed 10-08 03:21 EDT: `TRAIN OK` in 18.50 h, and the organizers' evaluation step on the rebuilt
   output gives rel 97.03 / tke 86.54 / mvpe 97.68 / sps 59.97 vs 97.03 / 86.57 / 97.68 / 59.99 for the archive of
   record on the same held-out trajectories.** The 1 × A100 (7712671) and 2 × V100 (55498053) copies finish 10-08.
4. **Push** `final-submission` and `qwu/dev`; agents never push.

## Settled

* 10-07: the training logs of the eleven runs of record were found and are in the branch (`logs/`); every checkpoint's
  `config` matches its log; torch 2.2.2+cu121 throughout; 43.6 GPU-hours (F-FNOs on V100S, ViTs on A100-40GB).
* 10-07: `train_real/7575_0.h5` is absent from the organizers' mount; the code skips it always; the README states that
  the networks of record were trained with it present.
* The newest handoff is `../handoffs/HANDOFF_20261007.md`.
