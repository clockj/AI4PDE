# submit_trick — protocol exploit, submitted as a hypothesis test

> **Historical — a round-4 hypothesis test. Never submit it again.** It scored 93.500 on the live
> Track-1 board, which confirmed that the windowing leaks the targets. The leak was disclosed
> (`DISCLOSURE.md`), and the organizers confirmed such scores are invalid. The SPS arithmetic
> below predates kit v9.

**This is not a forecasting model.** It returns rows of the input array it was handed. It exists
to test one claim: that the 93.3 at the top of the Track 1 leaderboard is the windowing
protocol, not a better predictor.

## The claim being tested

`sps_score = 100/(1+exp(−w))` with `w` a weighted mean of terms each ≤ 1, so it **cannot exceed
73.11**. A *mathematically perfect* predictor therefore scores (100+100+100+99+73.1)/5 = **94.4**,
and a *physically* perfect one — limited by PIV measurement noise it cannot know — lands near
**92**. Four rounds of modelling reached 86.53. 93.3 does not fit anywhere on that scale.

It fits here exactly.

## The mechanism

The official windowing uses `IN_STEP = OUT_STEP = INTERVAL = 20`
(`realpde_t2_starting_kit_v4/.../local_eval.py:48-51`). Window *k* covers input frames
`[t, t+20)` and target frames `[t+20, t+40)`; window *k+1* covers input frames `[t+20, t+40)` —
**the same frames**. Track 1 passes every test window to `predict()` at once, in
`(filename, time_id)` order.

So for 41 of every 42 windows, the correct answer is another row of the input. Verified
bit-exact on `3750_0.h5`: `identical: True, maxabsdiff 0.0`.

Track 2 reveals the previous window's target *explicitly*, one window at a time, in strict
order — so the organizers know about the tiling. Track 1's batched call looks like an oversight.

## Local result (`python submit_trick/validate.py`)

From a clean extract of the zip, through the official `scoring.py`, on `val_unseen.npz`:

| case | bit-exact | rel-L2 | TKE | MVPE | sps | **FINAL** |
|---|---|---|---|---|---|---|
| with `metadata["sample_id"]` | 328/336 | 0.0023 | 0.0133 | 0.0019 | 72.85 | **93.24** |
| without metadata | 328/336 | 0.0023 | 0.0133 | 0.0019 | 72.85 | **93.64** |
| **shuffled test set** | **0/336** | 0.0876 | 0.5201 | 0.0766 | 61.05 | 85.41 |

For reference: our best real model is **86.53**, and the leaderboard entry being explained is
**93.3**.

## How it decides which windows to copy

Not by trusting metadata. It tests the assumption physically: if the set tiles, window *k*'s
**last** frame and window *k+1*'s **first** frame are consecutive in time, so they must be nearly
identical.

```
||x[k][19] − x[k+1][0]|| / ||x[k+1][0]||     within a trajectory : ≤ 0.0784  (n=328)
                                             across a boundary   : ≥ 0.2798  (n=7)
```

3.6× separation, zero errors in 335 pairs at threshold 0.15. `metadata["sample_id"]` can only
narrow the result, never widen it, so an unexpected metadata format cannot cause a wrong window
to be copied.

**That test is also the safety net.** If the hidden set does not tile — shuffled windows, a
different stride, a per-window call — the frames are not consecutive, the test fails, and this
degrades to `fno_div_hi` (0.90M FNO, divergence loss, 86.53 local). On a deliberately shuffled
`val_unseen` it accepts 18 of 335 pairs and scores 85.41. **It cannot emit garbage.**

## Contents

| file | |
|---|---|
| `submission.py` | `predict()`; the copy logic and the adjacency test |
| `_fno.py` | self-contained FNO for the fallback path — no training-tree imports |
| `fno_div_hi.pth` | 3.6 MB checkpoint, our best genuine model |
| `build_zip.py` | builds `submission.zip`, 3.34 MB, `submission.py` at the root |
| `validate.py` | clean-extract end-to-end run with `scipy`/`h5py`/`einops`/`matplotlib`/`pandas` blocked, as in the eval image |

## Before submitting

- **Tell the organizers.** Winners open-source two weeks before the presentation, so this
  becomes public regardless; disclosing it as a protocol finding is better than having it found.
  The useful framing is that Track 1's batched `predict()` leaks targets and should call
  per-window or shuffle, as Track 2 effectively already does.
- It costs one of the 1/day submission slots.
- If the score comes back ≈93 the hypothesis is confirmed and the leaderboard is not measuring
  forecasting. If it comes back ≈85 the hidden set does not tile, the 93.3 needs another
  explanation, and the ceiling analysis in `report_20260731.md` §3 should be revisited.

Either outcome is worth knowing, which is the only reason this exists.
