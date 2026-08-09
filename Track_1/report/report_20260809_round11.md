# Round 11 — chasing 80.79 from 80.512

Board on 2026-08-08: **80.512025** (rel_l2 94.576394, tke 79.152828, mvpe 93.681162,
time 91.120851, sps 39.626333). Best other team **80.79**. Gap **+0.278**.

Plan of record: `plan/plan_20260809.md` (WP-A members, WP-B local compute, WP-C band, WP-D
sigma). This file records what round 11 actually measured. Nothing here has been submitted.

---

## 1. What board 80.512 fixes about the value model

Inverting `score_time(t) = 100/(1 + sqrt(t/0.72896))` at 91.120851 gives board t = **6.922 ms**.
Against the four-round board-time model

    board ms = 4.447 + 0.324 * invocations + 0.431 * local ms

and the shipped 3 invocations, the round-10 package ran at **local 3.487 ms/window**. That is
the number every WP-B decision below is priced against, and it is 2.5x the "1.383 ms" the
submission's own docstring still quotes: the per-window band added ~1.4 ms that was never
re-profiled after it shipped.

Also worth stating plainly, because it changes what WP-D is worth: **TKE is paid for twice.**
SPS's per-sample accuracy factor is

    a_i = 0.5*g(rel) + 0.3*g(tke) + 0.2*g(mvpe),      g(m) = 0.5/(0.5 + m)

so at the board's own metrics a = 0.710, and SPS's ceiling is 71.0 against the 39.6 we score.
Dropping board TKE from 0.527 to 0.45 moves `tke_score` +2.47 **and** lifts every SPS element
by 1.65% (+0.65 sps). At the mid subscore weights that is **+0.74 final from one metric**.

---

## 2. WP-B — local compute.  Measured, kept, worth ~+0.044

### 2a. The band stage was 1.45 ms/window of avoidable memory traffic

`predict()`'s per-window band did three things the obvious way and each cost a full-size pass
over a 165 MB array at N = 336:

| variant | ms/window |
|---|---|
| shipped | 1.4451 |
| 3-channel scale factor, no `np.concatenate` | 0.6344 |
| the same, with the three full-size ops through **torch** | **0.1299** |
| torch, broadcasting `band*r3` inside add/sub | 0.1746 |

numpy's ufuncs are single-threaded and this stage is bandwidth-bound, so torch's threaded
kernels are worth 5x on their own; dropping the zero-pressure `concatenate` is worth another
2.3x. `upper` doubles as the scratch buffer for `half`, so no fourth full-size array is live.

**−1.315 local ms → −0.567 board ms → +0.028 final.** The results differ from the shipped path
by 6e-8 (`x**p` vs `np.power(x, p, out=)` rounding) on a band of ~0.0065.

### 2b. The fused lift repeated an input that is identical for every member

`StackedFNO.forward` did `self.lift(x.repeat(1, K, 1, 1))` into a grouped conv. Every member
sees the *same* (window, grid) tensor at the lift, so `Conv2d(cin, K*width, 1)` with the
weights stacked is the same arithmetic without the K-times-wider input. Bit-exact (0.0 on all
five architecture variants).

### 2c. Mode-padded fusion — one fewer invocation, exactly

Members may now differ in retained modes: `StackedFNO` zero-pads each spectral weight to the
group maximum, and a zero weight on an extra mode contributes nothing to the einsum. The
modes-(12,20) member therefore joins its width-24 siblings instead of buying its own
invocation: the shipped package goes **3 groups → 2**, and with w48 restored it is 3 rather
than 4.

Measured back to back on the shipped members (interleaved reps, so contention drift hits both
equally), padding costs **+0.199 local ms/window** in extra spectral work — six members now run
at (12,20) modes instead of five at (8,12). That is 0.086 board ms against 0.324 saved:
**net −0.238 board ms → +0.012 final.** Positive, but a third less than a naive
invocations-only accounting would have claimed, and worth re-checking whenever the group
composition changes.

*Bug found and fixed while doing it:* `w1` multiplies `ft[:my]` and `w2` multiplies
`ft[-my:]`, so their row axes run in opposite directions in frequency. Padding both at the end
rotates the negative-frequency block and moved the field by 1.26 — an O(1) error that a
"looks fine" check would have shipped. `w2` pads at the front. After the fix the fused
ensemble matches the sequential one to 7e-7 (float32 reduction order over 20 modes instead
of 8), and a single m1220 member fused with itself is bit-exact.

---

## 3. WP-C — the band is closed.  +0.026 sps, i.e. +0.006 final

`band_cv_v9.py` runs the honest test the plan asked for: hold out one val_unseen AoA 15-20
trajectory, fit `base` on a chosen training set without it, score SPS on it, with the 1.39x
inflation applied to **both** sides so the CV cannot fake a level shift it can't observe.
84 configurations, 3 folds.

| fit set | best CV sps |
|---|---|
| val_seen AoA 15-20 (**shipped**), p = 0.45, k = 3 | 40.636 |
| pooled seen ∪ unseen-minus-fold, p = 0.45, k = 3 | 40.636 |
| val_seen, p = 0.50, k = 3 (**exactly what shipped**) | 40.610 |
| unseen-minus-fold only (2 trajectories) | 40.265 |

Three findings, all negative and all useful:

* **Pooling does not help.** Doubling the fitting set from 126 to 252 windows buys 0.000 sps.
  The base's 81,920 free parameters are not data-starved in the way the plan assumed —
  smoothing k = 3 is already doing that job.
* **The val_seen contamination costs nothing.** val_seen is in-sample for the `--err-ckpt`
  models, and the honest out-of-sample alternative is *worse*, because it has only two
  trajectories to fit on. The shipped choice was right for a reason nobody had checked.
* **p = 0.45 over p = 0.50 is 0.026 sps**, inside the fold-to-fold spread (folds run 42.7 /
  36.1 / 43.1). k = 3 is confirmed best at every p and every fit set.

### The band's inflation, re-inverted against an independent board point

`calib_board11.py` repeats round 9's inversion on the round-10 result, and this time reproduces
the band the package **actually** used, per-window sigma scaling included.

    board accuracy factor a = 0.7103  ->  SPS ceiling 71.03, scored 39.626 (55.8% of it)
    calibrated stretch f_hat = 1.360                     (round 9's inversion gave 1.39)

| build for | median half-width | sps at f_hat |
|---|---|---|
| 1.10 | 0.00910 | 38.91 |
| 1.25 | 0.00989 | 39.38 |
| **1.39 (shipped)** | 0.01058 | **39.58** |
| 1.55 | 0.01132 | 39.62 |
| 1.75 | 0.01215 | 39.46 |
| 2.30 | 0.01415 | 38.53 |

Two things come out of this and both are worth more than the CV above:

1. **The tail model is stable to 2% across two independent submissions** (1.39 → 1.360), and
   the build-scale row is flat over 1.25–1.75. Rebuilding at the optimum 1.55 is worth
   **+0.04 sps = +0.008 final**. `--inflate 1.39` stays.
2. **The proxy predicts board SPS to 0.05.** At f_hat the shipped band scores 39.58 against the
   board's actual 39.626. That is now a calibrated instrument, not just a ranking signal, and
   it is the right way to price any future SPS change.

### A second scaling axis: null (`band2_v9.py`)

`half = base * r_sigma**p * r_spread**q`. Sigma is the model's belief about the flow's
fluctuation amplitude; ensemble spread is its disagreement with itself — different quantities,
and a feature can be weak alone and still add. `base` refit for every (p, q). Same LOTO CV:

| | CV sps |
|---|---|
| p = 0.50, q = 0 (**shipped**) | 40.668 |
| p = 0.30, q = 0.15 (best) | 40.671 |
| p = 0, q = 0.30 (spread only) | 40.292 |

**+0.003 sps.** Spread alone is worth +0.15 over no scaling at all, sigma alone +0.53, and
together they are sigma alone. They are measuring the same thing.

WP-C is done. The band has no more in it, from three independent directions.

---

## 3b. Composition is flat too (`compose_v9.py`)

Every member set is now priceable end to end — accuracy through the board/proxy ratios, SPS
through the calibrated tail instrument with the band refit per composition, time through the
board-time model. Held-out members, so the accuracy columns are predictions:

| composition | inv | rel | tke | mvpe | sps |
|---|---|---|---|---|---|
| s1 s2 | 1 | 94.46 | 79.08 | 93.55 | 38.65 |
| s1 s2 s3 s4 | 1 | 94.54 | 79.12 | 93.65 | 39.28 |
| s1 s2 s3 s4 m1220 | 1 | 94.56 | 79.14 | 93.69 | 39.44 |
| s1 s2 s3 s4 m1220 w48 w64 | 3 | 94.58 | 79.14 | 93.68 | 39.64 |

Going from 4 members to 7 moves `rel_l2` by **0.04**, `tke` by 0.02, `mvpe` by 0.03 and `sps`
by 0.36 — worth about +0.11 final before time, ~+0.05 after. Most of it is SPS, because
ensembling shrinks the per-element error *tail*, which SPS charges for and rel-L2 barely
notices. Two members to four is worth four times as much as four to seven.

**The composition axis is exhausted.** Ship all eight; do not train more seeds.

---

## 3c. A board predictor, and what it says about the fallback package

`predict_board.py` combines the three calibrated instruments and reproduces round 10 to
**−0.009 final** (residual entirely the SPS instrument's 0.043). Run on the round-11 package —
8 members, 3 invocations, band refit, local 2.74 ms estimated:

| subscore | round 10 | predicted | Δ |
|---|---|---|---|
| rel_l2 | 94.576 | 94.585 | +0.008 |
| tke | 79.153 | 79.140 | −0.013 |
| mvpe | 93.681 | 93.683 | +0.002 |
| time | 91.121 | 91.312 | +0.191 |
| sps | 39.626 | 39.657 | +0.030 |
| **final** | **80.512** | **80.535** | **+0.023** |

Nearly all of it is the WP-B time saving. **Restoring w48 buys ~nothing on the proxy** — which
is the round-10 disagreement repeating, except that now the proxy's own error bar on rel and
mvpe is known (it over-predicted both by ~0.1 in round 10), so +0.008 is unresolvable either
way. w48 stays because it is free in invocations and the board hinted it matters, not because
this table supports it.

---

## 4. The recomposition gain is one-dimensional after all — but alpha is not 0.75

`gain_v9.py` sweeps the full two-parameter family `g = (a*sigma + b*s)/s`; five rounds of
`alpha` sweeps have only ever walked the line `a + b = 1`. On the 6-model held-out ensemble,
fit on val_seen AoA 15-20 and reported on val_unseen AoA 15-20 at 1.39x:

| a | b | rel-L2 | TKE | sps | final |
|---|---|---|---|---|---|
| 0.75 | 0.25 | 0.1059 | 0.4714 | 36.69 | 78.648  ← shipped |
| 0.90 | 0.10 | 0.1081 | **0.4525** | 36.39 | **78.742** |
| 1.00 | 0.00 | 0.1097 | 0.4456 | 36.12 | 78.735 |
| 0.60 | 0.55 | 0.1061 | 0.4829 | 36.55 | 78.510 |
| 0.45 | 0.70 | 0.1044 | 0.4981 | 36.82 | 78.444 |

**The ridge lies on `a + b ≈ 1`.** Nothing off the alpha line beats it, so five rounds of
parameterizing this as one number were not leaving anything on the table.

But the optimum on that line is **alpha ≈ 0.90, not 0.75**, and this is the third independent
measurement pointing the same way (round 10: +0.041…+0.070, called noise at the time). It is
worth +0.094 here and it costs nothing — a constant in `submission.py`. The grid is bumpy at
±0.15 between adjacent points because the SPS band is refit per row, so this gets re-measured
on the final member set with a fine grid before it ships, not adopted on this table alone.

---

## 5. WP-D — sigma.  Three gating arms, one of them new

`var 0.334` at convergence is a mean squared **log** error: sigma is typically off by a factor
of e^0.578 = 1.78. That reads as an information problem, not a capacity problem, which is why
the third arm below exists and why it is the one worth watching.

| arm | flag | what it adds | params |
|---|---|---|---|
| wider head | `--var-hidden 256` | boundary capacity | +3.2k |
| multi-depth head | `--var-deep` — head reads `concat(h1..h4)` | boundary capacity, multi-scale | +9.2k |
| **input std** | `--var-in-std` — head also reads the input window's own per-pixel temporal std | **information**: a quadratic of 20 frames the trunk would have to synthesize, and the log-std loss's target is the same statistic of the *output* window | +0.4k |

All three are readouts; the trunk is byte-identical, which is the side of the boundary/trunk
rule that has paid every time (lift +0.076, head_logstd +0.086, against `--point-mlp` −0.379
and every trunk-width change). All three round-trip through `_fno.load` and `fuse` bit-exactly,
and `models_alt.FNO2d` and `_fno.FNO2d` agree to 0.0 on identical weights for each.

Baselines on disk: `fno_vmlp_lift` 82.063, `_s2` 82.055, plus `fno_vl_s3/s4`. Seed sigma 0.055,
so nothing under ~0.11 is a result.

### Results so far

| arm | wfinal (AoA 15-20) | TKE | training `var` |
|---|---|---|---|
| baselines, 4 seeds | **82.046 ± 0.017** | 0.4823 ± 0.0028 | 0.36708 / 0.36572 / 0.36698 |
| `--var-hidden 256` | 82.038 | 0.4827 | **0.36562** |
| `--var-deep` | 82.047 | 0.4836 | **0.36801** |
| `--var-in-std` | 82.049 | 0.4823 | **0.36796** |
| `--lam-v 0.05 --lam-t 1.0` | **81.463** | **0.5059** | 0.4941 (worse by design) |
| `--lam-v 0.6` | 82.005 | 0.4868 | **0.36103** (better!) |

**All three null, and null in a way that closes the whole direction.** Not one of them fit the
training sigma any better: every `var` lands inside the baseline seed spread, and so does every
wfinal. That includes `--var-in-std`, which is not a capacity change at all — it hands the head
the input window's own per-pixel temporal std, the exact statistic the loss targets on the
*output* window, and a quadratic of 20 frames the trunk cannot cheaply synthesize.

`var` **is** the quantity sigma is judged on — a mean squared log error against `std_t(y)` — so
an arm that does not move `var` cannot move TKE, and none of them did. The head is at the
information floor of what a 20-frame window supports. Predicting a 20-frame amplitude envelope
20–40 frames ahead, in a flow whose shedding period is 60–75 frames, is simply hard, and the
head is already well past the naive estimators (TKE 0.483 against trajectory-mean 0.560 and
adjacent-window persistence 0.783).

### The objective arm is the round's best negative result

`--lam-v 0.05 --lam-t 1.0` moves sigma's gradient off the uniform log-MSE and onto the scored,
energy-weighted TKE rel-L2. It did exactly what it was designed to do to the **training** loss —
`tke` 0.3667 against the baseline's ~0.414 at epoch 85, an 11% improvement — and the held-out
TKE went the **wrong way**: 0.5059 against 0.4800. wfinal 81.463 against 82.046, a 34-sigma
loss, with rel-L2 0.1165 against 0.1111 and sps 40.73 against 42.34.

Training TKE down 11%, held-out TKE up 5%. That is textbook overfitting, and it means

> **`l_var` is not competing with the TKE objective. It is regularizing it.**

Strip the log-MSE and the head fits the training TKE field by memorizing it. This finally
explains round 10's "tripling `--lam-t` moves TKE by 0.0001": the objective was never the
problem, and adding more of the term that overfits was correctly resisted by the term that
regularizes.

Two follow-ups were **cancelled on this evidence rather than run**: `--lam-v 0.15 --lam-t 0.6`
(a milder version of a direction that is 34 sigma wrong) and `--lam-v 0.05 --lam-t 1.0
--detach-var` (detaching protects the *trunk*, but the failure is the *head* overfitting, and
round 10 measured detaching as costing TKE outright at 0.5625).

What it opens instead: if `l_var` regularizes, **is there more of that to have?** `--lam-v 0.6`
answers that, and the answer completes the picture:

| `lam_v` | training `var` | held-out TKE | wfinal |
|---|---|---|---|
| 0.05 (with `lam_t` 1.0) | 0.4941 | 0.5059 | 81.463 |
| **0.3 (shipped)** | 0.367 | **0.4823** | **82.046 ± 0.017** |
| 0.6 | **0.36103** | 0.4868 | 82.005 |

Training `var` improves **monotonically** with `lam_v`; held-out TKE is **U-shaped with its
minimum at the shipped 0.3**. `lam_v 0.6` is the only arm all round that fit log-sigma
measurably better than baseline — and its held-out TKE got *worse*.

That is the round's cleanest statement about sigma:

> **Fitting log-sigma better does not predict the amplitude envelope better.** The readout
> cannot move `var` at all, whatever its shape or inputs; the loss weights can move `var` but
> not held-out TKE; and 0.3 already sits at the optimum of the one family that moves it.

Sigma is at a **generalization** ceiling, not a capacity one and not a loss-shaping one.

A fourth information arm (`--var-pred-std`: feed the head the field head's own predicted std,
the denominator `compose()` divides its answer by) was **cancelled on this evidence rather than
run**. It is one more input to a readout already at its floor, and `sigma_blend_v9.py` had
separately found the global log-linear blend of exactly these estimators null out-of-sample.
The code is in the tree and bit-exact through `fuse`, so it costs nothing to revisit.
`var` is a mean squared log error, so 0.366 means sigma is typically wrong by a factor of 1.78,
and that is **not** because the head lacks parameters to express the answer. The head is at the
information floor of what its input can support.

That reprioritizes everything after it. The head predicts log-sigma about as well as it can
from this trunk, whatever shape the readout is given, so **the capacity arms are dead**. Two
kinds of arm survive that argument:

* **Information** — give the head something the trunk cannot cheaply synthesize.
  `--var-in-std` (the input window's own per-pixel temporal std) and `--var-pred-std`.
* **Objective** — change WHAT sigma optimizes rather than how well. `l_var` is a log-MSE,
  uniform over pixels; the scored `tke_rel_l2` is energy-weighted and only cares about the
  wake. `--lam-v 0.05 --lam-t 1.0` moves sigma's gradient from the first to the second.
  The information-floor argument says nothing against this one, so it was **moved ahead** of
  the information arms. Expect training `var` to get worse and TKE to get better; that is the
  whole trade.

`--var-pred-std` hands the head the temporal std of the **field head's own predicted window**.
`compose()` forms `g = (alpha*sigma + (1-alpha)*s)/s` with exactly that `s`, so as things stand
the head is asked for sigma in absolute terms and its answer is then divided by a quantity it
has never been shown; with it, the head can learn the ratio directly. Detached, so the variance
loss still cannot reshape the field trunk.

**Prior, honestly stated:** `sigma_blend_v9.py` already tested a *global log-linear* blend of
exactly these estimators (head, predicted-field std, input std, per-trajectory input std) and
it was null out-of-sample. These arms differ in that they learn a per-pixel nonlinear function
of them jointly with the trunk, which is a different hypothesis — but the prior is weaker than
it looked before that file was re-read.

### Ruled out before spending GPU on it

**Longer context for sigma only** (plan WP-D, third bullet) is **not deployable**. `predict()`
is handed 20-frame windows and nothing else; the only source of earlier frames is the previous
window in the same call, which makes the submission order-dependent — `validate_strict.py`
asserts permutation invariance to 1e-4 — and rests on the evaluator tiling its windows, which
is the same structural assumption as the disclosed leak. Dropped on those grounds, not on
round 4's negative result.

---

## 5b. Marginal costs, measured rather than assumed

| | local ms/window | board ms | final |
|---|---|---|---|
| one extra **invocation** | 0 | +0.324 | −0.016 |
| one extra fused w24 **member** | ~+0.27 (uncontended est.) | +0.116 | −0.006 |

**An extra fused member is not free.** It costs no invocation but it costs its forward pass, at
0.431 board ms per local ms. Round 8's member curve was worth about +0.012 per member at 8→9,
so past ~8 members the ensemble is roughly break-even. Above K = 10 in one group the 8 GB card
starts thrashing and the cost goes superlinear (K = 12 measured at +1.26 ms/member). This is
why the extra all-data seeds were demoted below a second seed of whichever WP-D arm wins.

---

## 6. Where that leaves the arithmetic

Everything below is measured, not projected, except the last row.

| | Δ final | status |
|---|---|---|
| band stage: 3-channel scale + torch's threaded ufuncs | **+0.028** | in the tree, SPS identical to 8e-8 |
| mode-padded fusion (m1220 joins the w24 group) | **+0.012** | in the tree, exact to 8e-8 relative |
| restore w48 — free in invocations, so the package is 8 members / 3 invocations | ~0 on the proxy | packaged |
| ALPHA 0.75 → 0.90 | **+0.058** (+0.098 by `final_v9`) | in the tree; band refit at 0.90 |
| band p 0.50 → 0.45 | +0.007 | same sign in two independent CVs |
| band: pooling / second axis / re-inverted inflate | +0.000 / +0.001 / +0.008 | **closed three ways** |
| composition (members, architectures) | flat | **closed**, priced end to end |
| TF32 / fp16 / longer sigma context | 0 / impossible / undeployable | closed |
| sigma: 3 head variants + 1 objective variant | 0, 0, 0, **−0.58** | see §5 |
| **total** | **≈ +0.10** | → predicted board **≈ 80.61** |

**Beating 80.79 needs +0.278 and this round delivers about +0.10.** That is worth stating
plainly rather than dressing up: round 11 closed more directions than it opened.

What it did produce, which is worth more than the +0.10 to the next round:

* **Three calibrated instruments.** `predict_board.py` reproduces round 10's board result to
  −0.009 final; `calib_board11.py` predicts board SPS to 0.05; `compose_v9.py` prices any member
  set in `final` with accuracy, invocations and local time together. Round 12 can price a change
  before building it, which round 11 could not.
* **A cheap screen for sigma arms.** `var` — the training log-MSE — is the quantity sigma is
  judged on. Three arms that did not move it did not move TKE either. Read the training curve
  at epoch 50 before scoring anything.
* **Two mechanisms, not just two numbers.** The variance head is at an *information* floor, not
  a capacity floor. And `l_var` regularizes the TKE objective rather than competing with it,
  which retroactively explains the one result from round 10 that never made sense.

---

## 7. What the round-11 evidence says about where to go next

Written after the round closed, from a literature review against the measurements above.
Full reasoning and citations: **`report/directions_20260809_architecture.md`**. Actionable
form: **`plan/plan_20260810.md`**.

The diagnosis in one line:

> We predict a conditional mean and bolt a hand-tuned variance onto it, but **~38% of the score
> is distributional** — `tke` (a second moment) and `sps` (an interval-coverage score) carry
> 0.47 of the 1.23 per-point weight, and they are our two weakest subscores.

Round 11's `lam_v` U-shape is the evidence that log-MSE is a *misaligned surrogate* rather than
a hard ceiling on sigma, which reopens the sigma question from a direction none of the six arms
tested: the **objective**, as a proper scoring rule.

| direction | verdict |
|---|---|
| **FGN-style probabilistic model** — parameter-space noise, CRPS loss, one forward pass per sample (arXiv 2506.10772, the engine behind WeatherNext 2) | **The bet.** Aligns training with the distributional 38%, and is the one generative family that survives a latency-scored task — this project already measured PDE-Refiner at 82.53 against its own MSE control's 83.83 at 4x compute. |
| **Phase as a co-trained second modality** (WN-C's "griddification" transposed: not a tracker, a sparse physically-meaningful modality co-trained with the dense field) | Maybe. A free diagnostic decides it: split the per-window error into phase/advection and amplitude parts. |
| Second model **family** for ensemble diversity | Live but budget-constrained: it must clear one invocation plus its forward time. |
| **Graph neural networks** | **No.** Wrong inductive bias for a dense uniform grid with fixed geometry; the mesh in GraphCast/WN-C exists because the Earth is a sphere. |
| **KANO** (ICLR 2026, `Papers/Track_1/2509.16825v6.pdf`) | **No.** Sells parameter efficiency; we are over-parameterized. Its core move is what `--point-mlp` did, at −0.379. Authors concede "orders heavier FLOPs", and FLOPs are scored. |

The pattern worth carrying forward: GNN and KANO are both proposals about **operator
expressivity**, and rounds 8–11 say expressivity stopped being the binding constraint some time
ago. What remains is that the fluctuation is partly unpredictable — `diag_spectral.py` puts only
~31% of fluctuation energy in the coherently predictable part — and we model that with a
log-MSE sigma head instead of a proper scoring rule.
