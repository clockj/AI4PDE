---
paths:
  - "leaderboard/**"
  - "Track_1/board_rounds.py"
  - "Track_1/final_v9.py"
  - "Track_1/predict_board.py"
---

# The board ledgers

* A scored round goes into `board_rounds.py` (`ROUNDS` or `ROUNDS_T2`) **and** `final_v9.py:BOARD`; importing `final_v9`
  checks the two against each other as multisets and refits the rule. A round without its five subscores goes to
  `FINAL_ONLY` and takes no part in the fit.
* **`final_v9.W0` is the authority for the weights.** Never copy them into prose as constants; say which fit
  (42 rounds to 09-20, 48 rounds at close).
* **The rule is not exactly linear.** Rounds whose containment moves without the errors moving (the adaptive Track-2
  packages) miss the fit by 0.003–0.005, which is why the gate is 0.0060. Deltas between neighbouring packages are good
  to about 0.002.
* `board.jsonl` holds one record per round: `pred` is written before the upload and never edited afterwards; `actual`
  is filled when the score arrives. `thesis` says what the round was for.
* `python leaderboard/lb.py --html` renders `../leaderboard.html` and is idempotent.
* Round ids (`R37`, `T2R24`) count uploads; package tags (`r38`, `k27fin`) name builds. They do not match: `R30` is
  package `r34`.
