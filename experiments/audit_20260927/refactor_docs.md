# Documentation wrap-up, 2026-09-27: ROUNDS.md, ROADMAP.md, leaderboard/board.jsonl, leaderboard/README.md

Worktree `/fs/scratch/PAS1187/qwu/tmp/realpde/wrap0927/wt_docs` (HEAD 7b6ee8e). Patch:
`notes/patch_docs.diff` (1 826 lines; four files only). Build scripts and intermediate pieces:
`notes/docs_work/` (`make_board_rows.py`, `assemble_rounds.py`, `assemble_roadmap.py`, `rounds_*.md`, `roadmap_*.md`).

`Track_1/board_rounds.py` and `Track_1/final_v9.py` did not need copying: HEAD (fdac00b) already carries the new
rows and the 48-round fit, identical to `ref_board_rounds.py` and the main repo's `final_v9.py`. Neither is in the patch.

## What was added where

| file | numstat (+/−) | lines now | what |
|---|---|---|---|
| `leaderboard/board.jsonl` | +14 / 0 | 54 | R31–R37, T2R18–T2R24, the 20-record schema of R16 … T2R17 (same key order) |
| `ROUNDS.md` | +821 / −88 | 3 412 | dated note + index (~80 lines); Appendix A (the 09-15 section, moved up, 96 lines; the 88 deletions are its old place); Appendix B (old *Current state* 09-20, verbatim, + dated note); Appendix C (old *Scoring* incl. *Board latency*, *What ships*, *Instruments*, *Closed*, *Build and gate*, *Engineering traps*, verbatim, headings demoted one level, ~190 lines); a one-line pointer at the old place; seven day sections R31 … R37 and "Main Development closed: what the last week overturned" (384 lines). The file ends with the newest round. Every line of the old file is still present (checked as a multiset). |
| `ROADMAP.md` | +530 / −152 | 1 947 | new header (20 lines); dated notes under §1.7, §1.17; §1.31 – §1.41 (176 lines); Part 2: seven dated notes and a 30-row table "Closed, re-opened or qualified 2026-09-21 … 09-27"; Part 3 replaced by the Decision-phase plan (59 lines); Part 4: 2026-09-27 paths note, build/gate bullet, provenance note, 5 table rows updated, 12 rows added, machine note, method rules 12–14; old Part 3 and old header moved verbatim to the end. The only old lines not present any more are the `## Part 3` heading and the five Part-4 table rows that were updated. |
| `leaderboard/README.md` | +42 / −19 | 142 | "three" ledgers; conclusions now point to ROADMAP Part 1 / EVALUATION (and ROUNDS Appendix C); `predict_board.RATIO` named; ΔC rule; predictions go to `../../sub_candidates/<date>/README.md` + PENDING row; `FINAL_ONLY`; field notes (`pred` subsets, `*_note`); a paragraph on the 09-21 … 09-27 rows; weights row → `final_v9.W0`, the 42-round and 48-round fits named, max residual 0.0058, nonlinearity; `w_time`, ratio, `f`, latency rows updated |

ROADMAP §1.31 – §1.41: centre0; ALPHA 0.80; containment is the gap to the top ten; proxies over-read band-tightening
(QD) and widening loses; R1 works on the board; board `time` intercept and drift; information-limited residual; online
corrections pay out of distribution (three regimes + the 09-27 board); training-side levers exhausted; the `final` rule
not exactly linear; reproducibility (+0.0006, not bit for bit).

## Validation

* All 54 lines parse; the 14 new records have exactly the key list and order of the newest existing records (T2R17);
  every `pred` / `actual` value is a number or the object is `null`.
* **`lb.py --html` does NOT run on the repo's `lb.py` with the new rows** (KeyError `'time'` in `section_progress`, line
  463). Cause: the eight rounds of 09-21 … 09-24 have `actual = {"final": …}` only, and `lb.py` indexes `actual["time"]`
  (lines 462, 1709) and every subscore in the HTML per-subscore table (line 1853) unconditionally. I did not edit `lb.py`.
  A three-hunk guard is in `notes/docs_work/lb_finalonly.patch` (`git apply --check` passes on the worktree); with it
  `lb.py --html` runs (rc 0) and renders all 14 rows (`notes/docs_work/lbtest/lb_patched.{txt,html}`).
  The alternative would have been `actual: null` for final-only rounds, which lb.py would show as "pending"; I chose the
  truthful record.
* Further `lb.py` edits needed to *show* the new state (not to run): `RANK_READS` stops at 09-20 (reads exist for 09-21
  #15/#14; 09-24 #19/#14; 09-25 #20/#15; 09-27 01Z T1 #22 cut 81.885, T2 #15 cut 81.954; 21:39Z cut T1 81.885 / T2 81.988),
  `LEADER_READS` stops at 09-17, `PHASES` has no phase after 09-01, and the "read this first" prose is hand-authored. T2R19
  (`actual: null`) renders as "pending".
* `pred` convention: the task asked for unknown keys to be `null`. Existing records store only the known keys (T2R1 has no
  `final`/`time`; R10/R11 only `final`), and `lb.py` crashes on a `null` inside `pred`, so I followed the existing records.

## Facts I could not source (written as "not recorded" in the docs)

* Subscores, ids and `time` of R31 – R34 / T2R18 – T2R21; the T2R19 result; R32's archive name (only its band
  `r37_inflate155`, fcec17b2…); any registered prediction for R31, T2R18, R32, T2R20; point estimates for R35, T2R22, R36,
  T2R23 (only brackets / Devi's lower bound 81.61 / "Jonas's intervals, both missed", values unknown); R33's absolute
  registration (only the time component +0.715).
* Definitions of "r36" and "E1" (closed 09-22).
* Final ranks and the 23:59Z cut.
* The implied board-likeness values (0.63 / 0.78 / 0.80; per subscore 0.62 / 1.91 / 0.34 / 0.80) are a reconstruction
  from `result_c.md`'s registered line, not a number any agent wrote down; labelled as such.
* The "+0.72 / +0.73 sps" of R31/T2R18 are rounded (commit 70064ee), so R31's C (≈ 0.5996) and the T2R18 cross-track
  offset (≈ +0.0007) are approximate; R31's `time` cannot be recovered.

## Old text that contradicts the facts of record, and how it is annotated

* `CLAUDE.md` *Current state* 09-20 ("09-21: HOLD on both tracks", "no lever ≥ +0.05 remains", r34 of record) — now
  ROUNDS Appendix B, with a dated note; also item 4 of the closing section.
* `CLAUDE.md` "Ship frozen" / T2 adaptation closed; ROADMAP Part 2 "Residual EMA", "Online SPS band calibration" — dated
  notes in Part 2 (re-opened and shipped in per-pixel causal form); closing item 1. Gradient TTT noted as still closed.
* ALPHA 0.60 "plateau" (CLAUDE.md, ROADMAP §1.17) — dated note under §1.17, §1.32, closing item 2.
* Band "closed eight times" (CLAUDE.md, ROADMAP Part 2 "The SPS band") — dated note; §1.31/§1.33; closing item 3.
* "±0.07 single-draw", drift 0.03–0.30 / 0.4–0.7 ms (CLAUDE.md, ROADMAP §1.7) — dated note under §1.7, §1.36, item 5.
* "Halve the local deltas" — closing item 6 (does not hold for online corrections).
* "The public hidden set lies inside the training range" — closing item 7, qualified, not overturned.
* ROADMAP Part 2 "Closed on PHYSICS" (σ-amplitude) and "Post-hoc sigma calibration" — dated notes (VARCAL uses revealed
  targets; no model-side TKE arm was run). "The one honest summary: every lever that is not new information has been
  pulled" — dated note (five inference-side levers paid).
* ROADMAP old Part 3 and Part 4 "every uploaded archive byte for byte in `submitted/`" — `../sub_candidates/submitted/`
  was deleted on 09-27; noted in the moved Part 3's preface and in Part 4.
* The 42-round weight vector quoted in docs (0.4778 / …) is the rounded one; the exact 42-round fit reads the closing six
  rounds +0.005 … +0.013 high, the rounded vector within ±0.002 except T2R24 (+0.0065). Stated in ROUNDS and ROADMAP §1.40.
* leaderboard/README ratios 1.07/1.14/1.45 vs `predict_board.RATIO` 1.078/1.150/1.511 — both named.

## Pointers in my text to files not yet in this worktree

`STATUS.md`, `Track_1/opoint.py`, `Track_1/band_io.py`, `.claude/rules/` (named in ROADMAP Part 4 and the ROUNDS/ROADMAP
headers) exist only as untracked files in the main repo or in `notes/new_*`; they must land in the same commit.
`../artifacts/progress_report_20260927.md` is outside the repository (cited as "PR §n").
