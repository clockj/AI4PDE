---
paths:
  - "Track_2/**"
---

# Track 2: the stream

* **Causality is the contract.** At step k the runner has revealed the targets of windows 0 … k−1 of the current
  trajectory and nothing else. State is updated from a target only after it arrived; the current window reads only
  what was accumulated before. `reset_ttt_state` clears every accumulator. Gate: `stream_eval.py --causal-check`.
* **The revealed target arrives normalised with the TARGET statistics** (`official/t2_v6/**/local_eval.py`); the input
  with the input statistics. The two differ in the fifth decimal. `PREV_SPACE` in `submission.py` says which one the
  package assumes (`"input"` is what was uploaded on 09-27); `--prev-space` in `stream_eval.py` and `validate_t2.py`
  says which one the harness feeds, and defaults to what the official runner does. `"target"` reads +0.002 … +0.009 in
  every held-out regime and +0.0064 in distribution, never worse on a trajectory; the Decision-phase candidate is built with it.
* **Zero-target elements** (about 10 % on `val_unseen`) are left out of the interval average and the coverage of `sps`
  only; rel, tke, mvpe and the accuracy factor of `sps` count them (`scoring.py`, `aggregate_sps`). Unmasked, D1c and
  VARCAL learn the current recording's blank from its revealed targets and pull the prediction to zero there, which is
  what the scorer rewards on a recording blanked differently. **`STATS_MASK` stays off**: null on our data (−0.0012),
  and on synthetic re-blanks it lost on every trajectory, in distribution by up to −4.77 (`ROADMAP.md` §1.46).
* **Guards.** A non-finite input zeroes that step's prediction only; a non-finite revealed target is treated as
  withheld. They run inside the graph on its critical path (+0.08 ms per step).
* **One model object per process.** Several `FrozenEnsemble` objects in one process crash with a CUDA illegal memory
  access; gates and reads use one.
* **What ships** (`../REPRODUCE.md` §13.3): asymmetric R1 with TSM and BLK, D1c with the empirical-Bayes shrink and the
  lead-time profile, VARCAL. The stack reads +0.54 … +1.15 at A20, A10 and relo over asymmetric R1 alone at the member
  count that ships (`ROADMAP.md` §1.42), and its constants sit on a plateau (leave one regime out: no gain).
  **Off, and open:** the gated history blend (`HIST_BETA`); an early-window interval prior, not built (hindsight bound
  +0.034 over A20, A10, relo, rehi). **Closed:** FAMW (`FW_N0`), +0.007 out of distribution and −0.001 in distribution
  on the full stack (its +0.15 was on symmetric R1 without D1c), and it breaks the first-window identity in floating
  point; gradient test-time training, closed-form readout refit, global residual correction, global band rescale
  (`closed/README.md`).
* **Every correction is shrunk by n/(n + N0)**, so the first window of a trajectory is bit-identical to the frozen
  package. Keep that property: it is what makes a knob-off build an identity.
* **Never port a stream statistic to Track 1.** Its batch holds future windows.
* **Return memory the caller owns.** The evaluator keeps every prediction until the run ends; a reused buffer makes
  all of them equal to the last one (rel-L2 0.67 instead of 0.074) and passes any harness that copies on the same line.
  `validate_t2.py` keeps the returned objects and compares them at the end.
* **The whole step is one CUDA graph**, corrections included, captured at load and accepted only if it replays
  bit-identically to eager. The accumulators are updated inside the graph, gated by a flag tensor: no host-side branch.
  Warm-up and capture replays run with the flag at 0.
* **The output arena covers 1 024 steps**; after that the step falls back to fresh copies, same values, about 0.2 ms
  slower. It holds 1.5 GB of page-locked host memory.
* **Reads are honest only on a package built from the held-out twins.** The out-of-distribution reads need ensembles
  trained without a whole angle of attack or Re band, read on the real package path with a refitted band:
  `experiments/decision_20260928/ood/README.md`.
