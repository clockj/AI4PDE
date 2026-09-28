# Status — 2026-09-28, Main Development closed

The only file that changes after a board result. Keep it under 40 lines; detail goes to `ROUNDS.md`.

| | board `final` | package (in `../sub_candidates/final/`) | registered before upload |
|---|---|---|---|
| Track 1 | **81.671149** (R37, 09-27) | `track1_k27fin_a080.zip` | 81.67 [81.58, 81.79] |
| Track 2 | **82.055553** (T2R24, 09-27) | `T2_MRGP2_track2_MRGP2.zip` | 82.000 [81.885, 82.219] |

* **Both are built by one command**, `bash Track_1/submit_fno/build_final.sh OUTDIR`. At commit `fdac00b` the rebuild is
  byte-identical to the uploads, file by file (jobs 7569053, 7569179). Since the refactor of 09-28 the code files carry
  guards and fewer dead knobs, and the rebuild is **output-identical at tolerance 0** (`Track_1/verify/final_identity.py`,
  jobs 7569409, 7569482); `band.npz` and the six members are still byte-identical.
* **Final ranks (owner's read, 09-28): Track 1 #20, Track 2 #8.** The top ten are shortlisted, so **Track 2 is in the
  Decision phase and Track 1 is not.** The cut read at 21:39Z on 09-27 was T1 81.885 / T2 81.988.
* **Decision phase, to 2026-10-25.** Organizers retrain shortlisted methods from scratch (8× A800) and score a private
  set at novel AoA / Re. **One submission per track for the whole phase.** Nothing is staged. A candidate (the method
  of record + `--aug-resim 0.25` + `PREV_SPACE = "target"`) is measured in `experiments/decision_20260928/STATE.md`.
* **The method of record** (both tracks): recipe-A members (4 F-FNO w40 L8 + sim, 2 ViT d384 L8), ALPHA 0.80 with
  JOINT_GAIN, interval centred on the alpha-0 field, SIGBLEND band. Track 2 adds the causal stream stack: asymmetric R1
  with TSM and BLK, D1c with the empirical-Bayes shrink and the lead-time profile, VARCAL.

## Open before a Decision-phase upload

1. **The revealed target's normalisation.** The official Track-2 runner normalises `prev_target` with the TARGET
   statistics; the uploaded package assumes the input statistics, so its unscored-element test never fires on the
   board. `PREV_SPACE = "target"` fixes it: +0.0064 on the twins (8 of 8) and +0.002 … +0.009 in every held-out regime,
   no trajectory worse (09-28). **The default is still `"input"`, the uploaded behaviour; ship `"target"`.**
2. **Masked pixels in the online statistics: decided, `STATS_MASK` stays off** (09-28). Null on our data; on a
   re-blanked recording it loses on every trajectory (−0.02 … −4.8): the scorer takes rel, tke and mvpe over the whole
   field, and the unmasked statistics are what learns a recording's own blank.
3. **SIGBLEND constants** were fitted on the eight validation trajectories; re-derived from a fresh dump they move by
   0.4 % (job 7569179). Their evidence is the leave-one-trajectory-out read.
4. **The guards cost latency on Track 2**: +0.08 ms per step locally (+2.4 %), about −0.002 to −0.005 `final` by the
   measured slopes. They sit on the critical path of the graph; moving them to the side stream is not done.
5. **Which online corrections the private set wants: the ones that ship.** At the member count that ships the stream
   stack reads +0.54 … +1.15 on held-out regimes; FAMW reads +0.007 there and is closed; the constants sit on a plateau.

## Reports

`../artifacts/20260927_realpde_phase_report.html` (the four phases, with the recommendations for the finals package) and
`20260927_realpde_final_method.html` (the method). The newest handoff is `../handoffs/HANDOFF_20260927.md`.
