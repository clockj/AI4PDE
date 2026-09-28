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
  set at novel AoA / Re. **One submission per track for the whole phase.** Nothing is staged.
* **The method of record** (both tracks): recipe-A members (4 F-FNO w40 L8 + sim, 2 ViT d384 L8), ALPHA 0.80 with
  JOINT_GAIN, interval centred on the alpha-0 field, SIGBLEND band. Track 2 adds the causal stream stack: asymmetric R1
  with TSM and BLK, D1c with the empirical-Bayes shrink and the lead-time profile, VARCAL.

## Open before a Decision-phase upload

1. **The revealed target's normalisation.** The official Track-2 runner normalises `prev_target` with the TARGET
   statistics; the uploaded package assumes the input statistics, so its unscored-element test never fires on the
   board. `PREV_SPACE = "target"` fixes it: on the twins +0.0064 composite on all 8 trajectories (8 of 8 positive),
   +0.0027 on AoA 15–20, all of it in `sps`. **The default is still `"input"`, the uploaded behaviour; ship `"target"`.**
2. **Masked pixels in the online statistics.** D1c's pooled variance and VARCAL's sums include elements that are never
   scored. `STATS_MASK = True` excludes them; on the twins it is null (−0.0012, 3 of 8). The risk it addresses, a
   recording that blanks a different region, cannot be tested on our data. Undecided.
3. **SIGBLEND constants** were fitted on the eight validation trajectories; re-derived from a fresh dump they move by
   0.4 % (job 7569179). Their evidence is the leave-one-trajectory-out read.
4. **The guards cost latency on Track 2**: +0.08 ms per step locally (+2.4 %), about −0.002 to −0.005 `final` by the
   measured slopes. They sit on the critical path of the graph; moving them to the side stream is not done.
5. **Which online corrections the private set wants.** The stream stack reads +0.03 … +0.09 in distribution and
   +0.45 … +0.55 at a held-out AoA; FAMW (family weights) is off in the upload and reads +0.15 out of distribution.

## Reports

`../artifacts/20260927_realpde_phase_report.html` (the four phases, with the recommendations for the finals package) and
`20260927_realpde_final_method.html` (the method). The newest handoff is `../handoffs/HANDOFF_20260927.md`.
