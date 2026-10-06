# The final submission: Track 2's training package (2026-10-06)

What goes to the organizers for the final evaluation, where it lives, and what was checked. The owner sends it;
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
  so no package can acquire a leaderboard row any more.
* **The template is not public.** `realpde_final_starting_kit_track2.zip` ("T2 Final Starting Kit v1", 5 649 bytes,
  uploaded 2026-09-28) needs a Codabench login; it is not on this machine. **Until the owner fetches it, the layout of
  the package is ours, not theirs.**

Track 1 finished #20 and is not invited; there is nothing to send for it.

## What is sent: branch `final-submission`

The method of the leaderboard row (T2R24, `T2_MRGP2_track2_MRGP2.zip`, sha256 `1aa2896f…`, final 82.055553), as
**training code only, reorganised**, on the orphan branch **`final-submission`** (no history in common with `main`),
checked out as the worktree **`../code-finalsub`**. One commit; 19 files; no checkpoint, no band, no symlink, no
record of the development.

| | |
|---|---|
| `run_track2.sh` | the entry point: `RPDE`, `RPDE_KIT`, `WORK`, one command (or one stage per job); the eleven recipes written out |
| `unet/` | `data.py`, `models_alt.py` (F-FNO, ViT, `build`), `simreal.py`, `train.py`: the training code of `fdac00b`'s `Track_1/unet/` with every closed experiment's flag and branch removed, the comments rewritten |
| `band/` | `make_val.py`, `fit_band.py`, `dump_twins.py`, `sigma_band.py`, `sps.py`: the band fit, with the shipped settings as constants |
| `submission/` | `submission.py`, `_fno.py`: **byte-identical to `fdac00b`'s tree files**; the build bakes the constants into copies |
| `build/` | `build.py` (the 19 constants, the FFNO_PATH block and SIGPOW, the int8 packing, the archive, every constant read back), `validate.py`, `causal_check.py`, `stream.py` |
| `README.md` | the organizers' manual; `TEAM` and `CONTACT` are placeholders |

`unet/` keeps that name and `models_alt.py` its name because the shipped `_fno.py` looks for `../unet/models_alt.py`
to rebuild an F-FNO through the training definition (the band fit runs that path, the evaluation container the
vendored copy). The entry point always refits the band (`fit_band.py` → `dump_twins.py` → `sigma_band.py`), because a
band is fitted on our twins and the organizers accept no fitted artefact; the `SIGPOW` literal of the built
`submission.py` is therefore the only line that differs from the upload's.

The earlier route (a curated cut of `fdac00b` assembled by `make_package.sh`, verified 10-05 by jobs 7681561,
7681858, 7682454) is superseded by the branch and was removed from this directory on 10-06.

## What was checked (2026-10-06)

| check | where | result |
|---|---|---|
| the baked `submission.py` and `_fno.py` against the uploaded archive's | login node, `build/build.py:bake` with the SIGPOW of the band of record | **byte-identical** (63 835 and 149 823 bytes) |
| old training code (`fdac00b`) vs new, deterministic arithmetic | `verify/cpu_equiv.sh`: CPU, 2 epochs, one window per trajectory, same seed, three recipes (F-FNO twin with the CFD term, ViT twin without, ViT member with) | **bit-identical weights** for all three, and the same checkpoint fields |
| the same on a GPU | job 55454190 (Pitzer V100), 2 epochs on the full data | F-FNO 4e-7, ViT 1e-2 apart — and the OLD code against itself on the same card: F-FNO 1.2e-7 in 75 of 79 tensors (job 55454404), ViT 3.1e-2 in 111 of 112 tensors (job 55454453). GPU kernels are not deterministic run to run; the CPU test is the equivalence test |
| the CFD map | job 55454190 | old and new caches **bit-identical** (all-data and twin tables) |
| the band from the eleven checkpoints of record, old vs new | job 55454190 | static band, twin dump, q = (0.48465, 0.49071) and the sigblend npz all **bit-identical** between the two codes (the V100 re-fit differs from the committed A100 band by up to 8e-4 in `base`, for both codes alike) |
| the archive from the checkpoints and the band of record | job 55454190 (V100) and the sandbox below | **all 9 files byte-identical to the upload** |
| the gates on that archive | job 55454190 | `SUBMITTABLE`, `CAUSALITY OK` |
| the branch as `git archive` would send it, in `verify/sandbox.sh` (none of our directories visible) | job 55454191 (Pitzer V100), commit `a716913` | `run_track2.sh` from the released data at 2 epochs to `RUN_TRACK2 OK` in 30 min; 9/9 byte-identical with the checkpoints and band of record; two stages trained without `7575_0.h5` |
| the final code, commit `a0a36b9` (= `9f3dd58`, the delivered commit, which differs only in `README.md`) | job 55454608 (Pitzer V100) | `RUN_TRACK2 OK` in 29 min, 9/9 byte-identical, two stages without `7575_0.h5` |

Peak host memory 21 GB; `$WORK` 4.2 GB.

## What is not checked, and what the owner decides

1. **The organizers' template.** Fetch `realpde_final_starting_kit_track2.zip` from the Files tab of the Track 2 page,
   and the invitation e-mail. The branch is then rearranged to their layout and `verify/job_verify_branch.sh` run again.
2. **`TEAM` and `CONTACT`** in `README.md`.
3. **`train_real/7575_0.h5`.** The data page says not to use it (a copy of `6300_0`; checked). Nothing in the repository
   ever excluded it, so the eleven networks of record were trained with it, and the manual says so under "Data".
4. **A full 400-epoch run of the branch.** The recipes and the training code are those of the 09-25 re-training
   (+0.0006); the branch was run at 2 epochs.
5. **Pushing the branch** (`git push origin final-submission`) and sending the package.

## Files

```
final_package/
  README.md                      this file
  verify/sandbox.sh              a third party's view of this machine (bubblewrap)
  verify/cpu_equiv.sh            old code vs new, bit for bit, on the CPU
  verify/job_equiv.sh            old vs new on a GPU: training, CFD map, band, archive, gates
  verify/job_verify_branch.sh    the branch in the sandbox: smoke, identity, without 7575_0
```
