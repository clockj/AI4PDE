# Documents brought in line with the campaign (scribe, 2026-09-29)

Documents only; no code, checkpoint, band or scratch file was touched, nothing committed. Sources: `PLAN.md`,
`STATE.md` (including its closing section of 09-29), `results/*.md` of this campaign, `../handoffs/HANDOFF_20260928.md`,
and the owner's decisions of 09-29 as given in the brief. Where the sources differed, `STATE.md` was followed (see the
end of this file).

## Per file

| file | what changed | why |
|---|---|---|
| `CLAUDE.md` (146 → 160 lines) | "Where things are": the release tag in the reproduce row; rows for the OOD instrument (`ood/README.md`) and the campaign. Rule 3: the static band's field is the twins' in-sample `val_seen` AoA 15–20 error ×1.39, SIGBLEND out of sample. "Data": zero targets are skipped only by the interval average and the coverage; rel, tke, mvpe and the accuracy factor count them; the zero set is not static (3–16 %). "What ships": tag `final-20260927`, `fdac00b` for byte-identical code; the candidate in two sentences, not uploaded. "Traps": one build per tree (lock, `baked_constants.py`), `--test-only`, `STATS_MASK` off. "Environment": Ascend first, Pitzer overflow. Repo map: `decision_20260928/` with `ood/`, `verify/baked_constants.py` | the "not scored" sentence was wrong (`scoring.py`); rule 3 read as if the field were held out (chunk 2, part A); the rest is new since 09-27 |
| `STATUS.md` (39 lines) | date 09-29; table kept; packages of record and tag; ranks; the candidate "T4 all" (recipe, command, band, OOD +0.206 / in distribution +0.031, checkpoints in `Track_1/unet/`, archives in `dec0928/candidate/` with sha256 prefixes); "T4 F-FNO only" not the choice; I7 not added; nothing staged, no upload decided. Open: the owner's decision, registration and staging, the guards' latency (unchanged), what is not verified. "Settled 09-28": PREV_SPACE, STATS_MASK, FAMW and the constants, the band. The uncommitted handoff pointer kept | the state changed |
| `ROADMAP.md` | Header: a dated paragraph, "How to read it" extended. Part 1: §1.42 (G0, five regimes), §1.43 (the augmentation: screen, ensemble, jackknife, the two recipes and where they differ, dose 0.5, board-like expectation, the owner's choice), §1.44 (ViT family, composition), §1.45 (split of the squared error after the stream), §1.46 (zero targets, STATS_MASK, zero propagation, natural blanking), §1.47 (in-sample static band, I6, refit 1e-4), §1.48 (R1, R1b, R2, not verified), with an italic preamble on regimes and composite. Dated notes: §1.12 (seed spread 0.04–0.08, the 12–58× was the unseeded FNO), §1.39 ("converged" qualified), §1.41 (ViT twins bit-identical on 09-28). Part 2: dated notes on "More ensemble members", "Band width for a novel angle", "The SPS band", the static-band row, the FAMW row (closed), the "one honest summary"; a new table "Closed, passed or qualified 2026-09-28" (G0, I1 … I7, T1 … T4, T4v, T4b, fixed compositions). Part 3: rewritten (the six priorities and what was done, the "what not to do" list with the override, what remains); the 09-27 Part 3 moved verbatim (checked by diff) before "Old Part 3" with a dated note. Part 4 standing rules 15 – 17 added (one build per tree; probe the queue; running scripts, second worktrees, clock times); dated note on rule 8 | the campaign's findings and the owner's decisions |
| `ROUNDS.md` | index row (no board round) and the entry "Decision-phase campaign (2026-09-28)" at the end: what was asked, the budget and waves (214.5 of 300), what passed and what closed with numbers, what else was established, the code, the mistakes, the decisions of 09-28 and 09-29 | append-only record |
| `EVALUATION.md` | pipeline diagram (the OOD instrument, the in-sample band, baked constants); the `rel_l2` row and a paragraph "Zero targets are scored, except by the interval"; §3: Re extrapolation back in use for the private set, AoA holdout no longer the only OOD instrument, a dated note on the in-sample static field; new §3b "The out-of-distribution stream instrument" (one run, regimes, layouts, cost, validation, limits); §4: the operating point states the in-sample field, a table of the reseeding spreads, the `aoa_holdout_eval.py` seed σ note, the `build_final.sh` row gains the lock and the constant check; §7: how the private set behaves | as asked |
| `.claude/rules/track2-stream.md` | PREV_SPACE: the regime reads, the candidate is built with `"target"`. Masked elements rewritten (what the scorer does, why STATS_MASK stays off, the re-blank evidence). "What ships": the G0 read and the plateau; open: `HIST_BETA`, the early-window prior (bound +0.034); closed: FAMW with numbers. Last bullet: the OOD instrument | as asked |
| `.claude/rules/bands.md` | the static fit is in sample (0.002 – 0.012); the two candidate bands and their reuse by `decision`; a refit from other twins ≈ 1e-4; a band from pooled held-out errors lost | as asked |
| `.claude/rules/instruments.md` | the AoA holdout is no longer the only OOD instrument; the Re regimes; decide at ensemble level on the stream stack; the reseeding spread as yardstick | as asked |
| `.claude/rules/training.md` | the candidate's eleven checkpoint names; the 09-28 retrain (ViT twins bit-identical, F-FNO 0.42 – 0.49 %) | as asked, and a contradiction with "ViT about 10 %" made visible |
| `.claude/rules/package-code.md` | one bullet: one build per tree, the lock file, `baked_constants.py` | the file speaks of builds |
| `.claude/skills/evaluate-change/SKILL.md` | out-of-distribution training arms: single-member screen with ≥ 4 matched seeds, decided at ensemble level on the stream stack; the regime table gains the instrument and the Re regimes; the reseeding spread as the yardstick, the jackknife | as asked |
| `.claude/skills/build-package/SKILL.md` | one build per tree and the lock; `BUILD_FINAL OK` now includes the constant check, and since the refactor the data files (not the code files) are byte-identical; `fdac00b` and the tag; a section on the candidate (`decision`, its two variables, the committed bands, where the archives are, nothing staged) | as asked; the "every file byte-identical" sentence was no longer true after the refactor |
| `.claude/skills/reproduce-from-release/SKILL.md` | the tag; a section on `decision` (variables, bands, stages, the acceptance read of 09-28, the promoted checkpoints); acceptance note on the 09-28 retrain and the organizers' scale; the byte-identity sentence qualified for trees after `fdac00b`; one build per tree | as asked |
| `experiments/README.md` | rows for `decision_20260928/` and for `audit_20260927/` (which was missing) | as asked |
| `README.md` | the tag, the Decision-phase position; the interval's base described as fitted on trajectories the twins trained on; the `decision` command and a paragraph on the candidate, not uploaded | "held-out twins" read as a held-out error field |
| `REPRODUCE.md` | after the top block: the tag, `fdac00b`, and a pointer to §14; §13.1 step 1: a dated note on the in-sample static field; §13.5: a dated note on PREV_SPACE and STATS_MASK; §14: the owner's choice, the promoted checkpoint names, the archives on scratch with sha256 prefixes | as asked |
| `Track_2/README.md` (not in the brief) | dated notes: zero targets are counted by rel, tke, mvpe; FAMW closed (I3) | the same wrong sentences as `CLAUDE.md`, in a document that is not a record |
| `Track_1/README.md` (not in the brief) | the `aoa_holdout_eval.py` row no longer says "the only OOD instrument" | the same |
| `Track_1/submit_fno/README.md` (not in the brief) | the band section states that the default fitting split is in sample | the same |

Not edited, as instructed: `PLAN.md`, `STATE.md`, `Track_1/closed/`, `Track_1/archive/`, `experiments/sprint_20260927/`,
`experiments/audit_20260927/` (other than being listed), `Track_1/bands/MANIFEST.md`, any code; `.claude/skills/osc-jobs/`
was read and is correct.

## Statements superseded by the campaign, left unchanged because they are records

| file:line | statement | superseded by |
|---|---|---|
| `experiments/decision_20260928/PLAN.md:34`, `:101` | "a single model's seed spread is 12 – 58× larger out of distribution" (premise of T1) | `STATE.md` "Single-member reads": 0.04 – 0.08 for the shipped recipe; `ROADMAP.md` §1.12 note |
| `experiments/decision_20260928/STATE.md:237` | "`build_final.sh` itself does not verify that the constants it baked are in the archive … to be fixed" | fixed in 467769d (`baked_constants.py`, lock); `STATE.md` "The candidate" |
| `experiments/decision_20260928/results/candidate_band.md:1`, `:28` | identity jobs "RUNNING at hand-back" | `STATE.md` "The candidate bands committed": both identical, 9 of 9 files |
| `experiments/decision_20260928/results/candidate_release.md:15` | a `MANIFEST.md` "entry stub for `dp_resim_sigblend_lam0.5.npz`" | two complete entries (`dp_resim_all`, `dp_resim_ffno`) |
| `experiments/decision_20260928/results/ood_chunk6.md:26` | "the tag becomes `dp_resim_v`" | repaired in 6ad14ea: tags `dp_resim_all` / `dp_resim_ffno` |
| `experiments/audit_20260927/README.md:37` | M2, `STATS_MASK` "Open in `STATUS.md`" | closed 09-28: stays off (`ROADMAP.md` §1.46) |
| `experiments/audit_20260927/audit_shipping.md:228 – 229` | "Suggest: measure `eb` with and without a `scored` mask …; if they differ, mask" | measured: masking loses on every re-blanked trajectory |
| `experiments/audit_20260927/refactor_pkg.md:115` | "The audit's concern was a *different* blanking pattern on the private set, which this split cannot test" | tested on synthetic re-blanks (chunk 2 part B) and on natural novel blanking at A20 (chunk 4 part 1) |
| `ROUNDS.md:383` (Appendix, verbatim) | the AoA holdout "is the only OOD instrument we have" | `EVALUATION.md` §3b |
| `ROUNDS.md:3343` (Round 37 entry) | "The online statistics include masked pixels" listed as a defect carried into the Decision phase | not a defect: `STATS_MASK` stays off |
| `Track_1/bands/MANIFEST.md:237` | the static band's construction "`--err-ckpt <the five held-out twins>`" without saying the fit windows are in sample for them | `ROADMAP.md` §1.47 (the MANIFEST is otherwise complete and was not to be edited) |
| `ROADMAP.md` Part 3 of 2026-09-27 (now verbatim near the end) | "No training campaign"; "Masked pixels … these sums must be masked"; FAMW to be re-read | the new Part 3 and §1.43, §1.46; Part 2 table |
| `ROADMAP.md` Part 2, older entries | FAMW "open for the private set"; "the models are converged for this data"; "One run is not a measurement" at 58× | each carries a dated note now; the text is unchanged |
| `../artifacts/20260927_realpde_phase_report.html` (outside the repository) | the recommendation to mask the online statistics (`STATS_MASK`) | §1.46; `STATE.md` decisions of 09-29 list the two reports as "to be updated" |
| `../artifacts/20260927_realpde_final_method.html` (outside the repository) | the sentence on the band's error field (as held out) | §1.47 |

I did not read the two reports in `../artifacts/`; the two rows above follow `HANDOFF_20260928.md` §4 item 5.

## Where the sources differed, and what was used

* **GPU-hours used.** `STATE.md`'s ledger read 214.4 at 23:03 on 09-28; its row of 09-29 00:45 (added by its
  maintainer during this work) reads **214.5**, as does the handoff. The documents say 214.5.
* **The split of the squared error after the stream.** `STATE.md` (chunk 5 summary) and the handoff say "72 – 76 %
  fluctuation"; the table in `results/ood_chunk5.md` Part 3 has relo at 64.7 % (A20 71.5, rein 73.9, rehi 76.2). The
  documents give the table's range, 65 – 76 %.
* **mvpe "over the whole field".** The brief (and `STATE.md`) say rel, tke, mvpe are norms over the whole field;
  `scoring.py` computes mvpe at its 36 probe pixels, unmasked. The documents say "computed without a mask".
