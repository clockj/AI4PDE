---
name: evaluate-change
description: Decide whether a change to the model, the band or the inference path is real, using held-out twins and the calibrated local instruments, and turn it into predicted board subscores with brackets. Use before proposing that anything ships.
---

# Evaluate a change

## 1. Write the criterion first

Before any number exists, write down: the instrument, the split, the threshold, and how many trajectories must agree.
An arm that misses is closed; the threshold does not move. Check `ROADMAP.md` Part 2 first: if the direction is
closed there, say what new evidence reopens it.

## 2. Measure on twins

* **Training change:** train held-out twins with `--seed` matched to a control, at least 3 seeds. One run at 2–3σ has
  regressed on replication four times. An arm that does not move its own training loss cannot move the held-out metric.
* **Inference change:** build the package from the five twins (`build-package` skill) and read it on `val_unseen`.
* **Read at the size that ships.** Error an arm removes per member can vanish in the six-member mean. Swap the arm's
  twin into the five-twin set and read the set.
* **Refit everything downstream** (band, SIGBLEND constants) before comparing a changed member set.

## 3. Read three regimes

| regime | instrument | what it stands for |
|---|---|---|
| in distribution, all 8 held-out trajectories | `ens_v9.py`, `Track_2/stream_eval.py` | the floor |
| in distribution, AoA 15–20 | the same with `--aoa 15 20` | the development board's error level |
| a whole AoA held out of training | `aoa_holdout_eval.py` (AoA 10 = interpolation, AoA 20 = extrapolation) | the private set |

The development board behaved like the held-out regimes on 09-27 (board-likeness 0.6–0.8 between the in-distribution
read and the held-out one). The private set is held-out by construction.

## 4. Correct the local number

* Instruments default to the operating point of the packages of record (`Track_1/opoint.py`). Results recorded before
  2026-09-27 used other defaults: the table is in `EVALUATION.md` §4.
* A local **error cut** is halved. A **member removal** and a **mean-field gain** are charged in full.
* Across architecture families a local proxy gives neither size nor sign. A board read of a family transfers across
  tracks at about 0.85×.
* Changes that fit the interval tighter to the error are over-read locally and can flip sign on the board. State them
  in ΔC.
* No local instrument reads containment as the board does. Never bank it in an estimate.

## 5. Predict

`python Track_1/predict_board.py --anchor <a recent round of the same shape>` gives the five subscores. Bracket `time`
with the measured slopes and do not count a latency gain in the expected value. Write estimate and bracket for each
subscore; the bracket of `final` follows from them.
