---
name: record-board-result
description: Record a Codabench board result everywhere it belongs - ledgers, the fitted final rule, STATUS.md, ROUNDS.md, the staging directory and a dated handoff. Use when the user reports board scores for an upload.
---

# Record a board result

Work through the list in order; each step names the file and what to put there.

1. **`leaderboard/board.jsonl`** — fill `actual` (five subscores and `final`) on that round's record. The `pred` must
   already be there; if it is not, record the result with `pred: null` and say so. Never write a prediction afterwards.
2. **`Track_1/board_rounds.py`** — add the row to `ROUNDS` (Track 1) or `ROUNDS_T2` (Track 2) with a comment that says
   what the round measured: which lever, the subscore deltas against its anchor, prediction against result. A round
   whose five subscores are unknown goes to `FINAL_ONLY`.
3. **`Track_1/final_v9.py:BOARD`** — the same six numbers, in submission-date order. Then
   `( cd Track_1 && python final_v9.py )`: it refits the rule and checks both lists against each other. If the gate
   fails, a number was mistyped or the row lies outside the range the rule was fitted on; find out which before going on.
4. **`Track_1/predict_board.py`** — add the round to `ANCHORS` if later packages will be priced against it.
5. **`STATUS.md`** — the table, and any open item the result closed. Under 40 lines.
6. **`ROUNDS.md`** — append the write-up: what was uploaded, the prediction, the result per subscore, what it changes.
   If the result creates or overturns a rule, change the rule where it lives (`CLAUDE.md` Rules, `.claude/rules/`,
   `ROADMAP.md` Part 2) and say in `ROUNDS.md` that it did.
7. **`../sub_candidates/`** — the uploaded archive goes to `final/` if it is the new package of record; add its row to
   `MANIFEST.md`; append the staging README and the result to `HISTORY.md` verbatim. Its band goes to `Track_1/bands/`
   (`git add -f`) with a `MANIFEST.md` entry there.
8. **`python leaderboard/lb.py --html`** writes `../leaderboard.html`.
9. **A dated handoff** in `../handoffs/HANDOFF_YYYYMMDD.md`, self-contained: state, what is open, running jobs, traps.
10. **One commit** on `qwu/agent`. Do not push.

## Reading a result

* Compare subscore by subscore against the anchor. The four accuracy subscores are deterministic for an archive; `time`
  drifts by 0.04–0.06 `final` between uploads of the same archive.
* `sps = 100 · ā · C`. Recover ā from the three error subscores and C as the quotient: a change in `sps` with unchanged
  errors is containment.
* A board delta measured on one track transfers to the other at about 0.85×.
