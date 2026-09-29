# Status — 2026-09-29, Decision phase

The only file that changes after a board result. Keep it under 40 lines; detail goes to `ROUNDS.md`.

| | board `final` | package (in `../sub_candidates/final/`) | registered before upload |
|---|---|---|---|
| Track 1 | **81.671149** (R37, 09-27) | `track1_k27fin_a080.zip` | 81.67 [81.58, 81.79] |
| Track 2 | **82.055553** (T2R24, 09-27) | `T2_MRGP2_track2_MRGP2.zip` | 82.000 [81.885, 82.219] |

* **The packages of record** (method: `CLAUDE.md` "What ships") rebuild by `bash Track_1/submit_fno/build_final.sh
  OUTDIR`, data byte-identical and outputs identical at tolerance 0; from `fdac00b` the code files too. The release is
  the annotated tag **`final-20260927`** on `4e5ed37` (main after pull request #3).
* **Final ranks: Track 1 #20, Track 2 #8.** Track 2 is shortlisted: the organizers retrain from scratch (8× A800) and
  score a private set at novel AoA / Re. **One submission per track for the phase** (to 2026-10-25).
* **The candidate (owner, 09-29): "T4 all"**, the method of record with all six members and five twins trained with
  `--aug-resim 0.25`, plus `PREV_SPACE = "target"`. Built by `RPDE_REPRODUCE_GO=1 bash reproduce.sh decision`
  (defaults), tag `dp_resim_all`, band `Track_1/bands/dp_resim_all_sigblend_lam0.5.npz` with its sidecar json. Against
  the control: +0.206 out of distribution (mean of A20, A10, relo, rehi), +0.031 in distribution (twins, 7 of 8).
  Checkpoints: the eleven `*_resim*` in `Track_1/unet/`. Archives on scratch, `dec0928/candidate/track1_dp_resim_all.zip`
  (53291df1…) and `track2_dp_resim_all.zip` (bd0a3fb4…). "T4 F-FNO only" (`RPDE_DP_RESIM_VIT=`) stays reproducible
  and is not the choice. Zero propagation (arm I7) is not added. Record: `experiments/decision_20260928/`.
* **No upload is decided and nothing is staged** in `../sub_candidates/`. The board shows the packages of record.

## Open before a Decision-phase upload

1. **The owner's decision to upload**, per track. HOLD is valid.
2. **Registration and staging first** (rules 1, 2): five subscores with brackets, in distribution and held out, in
   `../sub_candidates/<date>/README.md` and a PENDING row of `leaderboard/board.jsonl`. Neither exists.
3. **The guards cost Track 2 latency**: +0.08 ms per step (+2.4 %), about −0.002 … −0.005 `final`; still on the critical path.
4. **Not verified:** the organizers' hardware and stack (only A100 and V100, torch 2.2.2; expect the reseeding spread,
   about 0.012 in distribution, not our retrain's 0.0003); the all-data members were trained once.

## Settled 09-28

* `PREV_SPACE = "target"`: +0.002 … +0.009 in every regime, never worse on a trajectory. `STATS_MASK` stays off.
* FAMW (+0.007 out of distribution) and the stream constants (a plateau) are closed; the shipped stack stays.
* The band: a refit from other twins moves the composite by about 1e-4; its static field is fitted in sample, as shipped.
* Reports: `../artifacts/20260927_realpde_phase_report.html` (its `STATS_MASK` advice is superseded) and
  `20260927_realpde_final_method.html`. The newest handoff is `../handoffs/HANDOFF_20260928.md`.
