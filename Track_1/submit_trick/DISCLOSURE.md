# Track 1 evaluation protocol leaks the targets

Reporting a flaw in the Track 1 evaluation protocol that lets a submission recover ~97.6% of the
ground truth exactly, without forecasting. We verified it on the live leaderboard
(**final_score 93.500**) and are disclosing rather than competing on it.

---

## 1. Summary

Track 1 builds test windows by tiling each trajectory at a stride equal to the output horizon,
and passes **all** windows to `predict()` in one call, in `(filename, time_id)` order.

Because the stride equals the horizon, **window *k*'s target is window *k+1*'s input** — the same
frames. So for every window except the last of each trajectory, the correct answer is already
present in the array handed to `predict()`. A submission that returns `input_array[k+1]` scores
near the maximum without modelling anything.

---

## 2. Mechanism

From `realpde_t2_starting_kit_v4/realpde_t2_starting_kit_v4/local_eval.py:48-51`:

```python
IN_STEP  = 20
OUT_STEP = 20
INTERVAL = 20          # window stride == output horizon
HORIZON  = IN_STEP + OUT_STEP
```

Window at `time_id = t` therefore spans:

```
input  = frames [t,      t + 20)
target = frames [t + 20, t + 40)
```

and the next window, at `time_id = t + 20`, has:

```
input  = frames [t + 20, t + 40)      <-- identical to the previous window's TARGET
```

Track 1's contract is `predict(input_array, metadata)` with `input_array` of shape
`(N, 20, 32, 64, 3)` containing **every** test window at once, sorted by `(filename, time_id)`.

## 3. Reproduction

```python
import numpy as np, h5py

f = h5py.File("3750_0.h5", "r")
u, v = f["u"][:, ::2, ::2], f["v"][:, ::2, ::2]
d = np.stack([u, v, np.zeros_like(u)], -1)
ids = list(range(0, len(d) - 40 + 1, 20))        # 868 frames -> 42 windows

k = 5
target_k    = d[ids[k] + 20 : ids[k] + 40]       # window k's target
input_k1    = d[ids[k + 1] : ids[k + 1] + 20]    # window k+1's input
print(np.array_equal(target_k, input_k1), np.abs(target_k - input_k1).max())
# -> True 0.0
```

Bit-exact, not approximate. On our 8-trajectory held-out split this covers **328 of 336
windows (97.6%)**; only the last window of each trajectory has no successor.

## 4. Evidence from the live leaderboard

We submitted a package that returns `input_array[k+1]` where the windows are contiguous, and a
small trained model for the remaining 1/42:

| subscore | our local prediction | **hidden test set** |
|---|---|---|
| rel_l2_score | 99.89 | 99.850 |
| tke_score | 99.34 | 99.345 |
| mvpe_score | 99.90 | 99.798 |
| time_score | 93.91 | 95.723 |
| sps_score | 72.85 | 72.785 |
| **final_score** | 93.18–93.64 | **93.500** |

For contrast, our best genuine forecasting model — after four rounds of architecture and loss
work — scores **86.53** on the same held-out split.

## 5. Why this is decisive rather than marginal

`sps_score = 100 / (1 + exp(-w))` where `w` is a weighted mean of terms each ≤ 1, so
**`sps_score` cannot exceed 73.11**. The best attainable `final_score` by *any* submission is
therefore:

```
(100 + 100 + 100 + ~99.6 + 73.11) / 5  =  94.54
```

Meanwhile a *physically* perfect forecaster — one that knows the true flow but cannot know the
PIV measurement noise realization — is bounded near **92**, because measurement noise floors
rel-L2 around 0.031 and TKE error near 0.13.

So the interval **92 → 94.54 is reachable only by reading the targets**, and scores above ~93 are
now clustered within ~1 point of a hard ceiling where the remaining ordering is decided by
inference latency on 1/42 of the windows rather than by prediction quality. We note at least one
entry above 94, which is consistent with the same approach plus a near-zero-cost fallback.

## 6. Suggested remedies

**A stride greater than the output horizon is the robust fix.** With `INTERVAL > OUT_STEP` no
window's target appears as another window's input, and the leak closes by construction.

**Per-window evaluation also works** — call `predict()` once per window with no cross-window
state, as Track 2 effectively already does by revealing the previous target explicitly.

**Shuffling the window order is *not* sufficient.** The adjacency is detectable from the data
itself: for tiled windows, window *k*'s last frame and window *k+1*'s first frame are consecutive
in time and therefore nearly identical. We measured the separation on held-out data:

```
||x[k][19] - x[k+1][0]|| / ||x[k+1][0]||
    same trajectory, consecutive : <= 0.0784   (n = 328)
    different trajectory         : >= 0.2798   (n = 7)
```

A 3.6x gap, zero misclassifications in 335 pairs. A submission can therefore re-sort shuffled
windows back into trajectory order and apply the same trick.

## 7. Disclosure

We found this while trying to reconcile the top of the leaderboard with the scorer in the
starting kit: our own measurements put a realistic ceiling for genuine forecasting near 87.5 and
a physically perfect one near 92, so entries above 93 did not fit. We built and submitted the
exploit once, deliberately, to test that explanation, and the returned score matched our local
prediction to within 0.1 on three of five subscores.

We are not pursuing this approach further and are happy to share the submission package, the
validation harness, or the full measurement notes if useful.
