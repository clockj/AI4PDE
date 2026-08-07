# Round 9 planning — where the next point actually comes from

**Date:** 2026-08-07 · **Board:** 80.115065 (8-member all-data FNO, `submit_fno/submission.zip`)
**Status:** analysis and plan only. Nothing implemented, nothing submitted.

---

## 0. Summary

Three things came out of this session, in descending order of importance:

1. **The `final_score` weighting is now solved to within a one-parameter family.** Three board
   rounds pin it down. `time` is weighted **0.034–0.099**, not 0.2; `sps` **0.171–0.189**;
   `tke` **0.219–0.273**; `rel + mvpe` together **0.493–0.522**. Every plan written before this
   priced latency ~3× too high and priced `sps` and `tke` wrong.
2. **The exchange rate between accuracy and score is measured: +0.089 `final` per 1 % of
   error removed**, and **45 % of that comes through SPS** even though SPS carries the second
   *lowest* weight. SPS is the steepest function of the error by a factor of 2.3. The band is
   closed; the way to move SPS is to shrink the error tail.
3. **Board time is not a constant.** Round 8 concluded it was, from two points that happened to
   be close. Round 9 moved it 2.09 ms and cost ~0.084 `final`. Members are cheap but not free,
   and the cost is in *sequential forward passes*, which local timing understates 2.2×.

The DPOT question is answered in §5: **the benchmark paper does not say DPOT is best on the
airfoil.** On the Foil dataset U-Net wins every reported metric, and DPOT-L is best only when
averaged over all five datasets. Recommendation is not to build it — reasons in §5.

---

## 1. The board result

| | R6 (U-Net) | R7 (FNO ×3) | R8 (FNO ×8) | R8 − R7 |
|---|---|---|---|---|
| `rel_l2_score` | 94.4342 | 94.5754 | 94.6342 | +0.059 |
| `tke_score` | 78.1660 | 78.5278 | 78.8825 | +0.355 |
| `mvpe_score` | 93.7922 | 93.9100 | 93.8030 | −0.107 |
| `time_score` | 91.6973 | 91.7264 | 90.5066 | **−1.220** |
| `sps_score` | 36.9374 | 37.4680 | 38.2338 | +0.766 |
| mean5 | 79.005 | 79.241 | **79.212** | −0.030 |
| **`final_score`** | **79.733** | **79.984883** | **80.115065** | **+0.130** |

Note the row that matters: **mean5 went *down* while `final` went *up*.** The
`final ≈ mean5 + 0.73` rule from rounds 6–7 is dead — the offset was +0.728, +0.743, then
**+0.903**. It was never a rule, it was two coincidences.

---

## 2. The `final_score` weighting, solved

`scoring.py` states the leaderboard combines exactly these five subscores and nothing else, so
the space of hypotheses is small. Geometric and harmonic means are off by 5 points; a linear
weighted mean with weights summing to 1 and no intercept fits all three rounds **exactly**.

Three equations plus the sum constraint leave one free direction, so the solution is a line, not
a point. What matters is that most of the weights barely move along it:

| subscore | admissible range | midpoint |
|---|---|---|
| `rel_l2` | 0.000 – 0.493 | 0.266 |
| `tke` | **0.219 – 0.273** | 0.244 |
| `mvpe` | 0.000 – 0.522 | 0.241 |
| `time` | **0.034 – 0.099** | 0.069 |
| `sps` | **0.171 – 0.189** | 0.181 |
| `rel + mvpe` (sum) | **0.493 – 0.522** | 0.506 |

`rel` and `mvpe` trade against each other because they moved together across all three rounds;
only their **sum** is identified. Everything else is tight.

**Caveat, stated plainly:** three points, five unknowns, exact fit — this is *consistent with*
the data, not *validated by* it. Any five-weight linear model fits three points. What raises my
confidence is that the three tight conclusions hold across the **entire** admissible family,
including its extremes, so they do not depend on choosing a point on the line. The next
submission is a clean falsification test: predict `final` from the five subscores before looking.

### What changes because of this

* **Latency is ~3× cheaper than every plan assumed.** At the current 8.02 ms board:

  | board ms | `time_score` | Δ`final` |
  |---|---|---|
  | 6 | 91.68 | +0.08 |
  | 8.02 | 90.51 | 0 |
  | 20 | 85.79 | −0.33 |
  | 50 | 79.25 | −0.78 |
  | 100 | 72.97 | −1.21 |
  | 200 | 65.63 | −1.72 |

  A **25× increase in inference cost costs 1.2 points.** Test-time adaptation, deeper ensembles,
  sampling-based methods and multi-pass refinement are all now inside the budget. They were not
  when `time` was priced at 0.2.
* **`tke` is weighted highest of the accuracy metrics (0.244)** and also enters `sps` through
  `a_i` at weight 0.3. It looks like the obvious target — §4 shows it is not.
* **`sps`'s weight of 0.181 understates it badly.** See §4.

---

## 3. Board time is not constant — correcting round 8

Inverting `score_time`:

| | local ms/window | board ms/window |
|---|---|---|
| R6 U-Net | 2.648 | 5.976 |
| R7 FNO ×3 | 1.282 | 5.931 |
| R8 FNO ×8 | 2.211 | **8.020** |

Round 8 read the first two points as slope **0.033 ms board per ms local** and concluded board
time is harness overhead. Round 9 breaks that: the 8-member package is **faster locally** than
the U-Net (2.211 vs 2.648) and **2.04 ms slower on the board**. Local wall time does not predict
board wall time.

What does predict it is **the number of sequential model invocations**. `_fno.predict_fno` runs
`for m in models:` — eight separate forward passes. Locally those pipeline on the 4060 and cost
~0.11 ms each; on the board they cost ~0.42 ms each. Whatever the board runs on, per-invocation
overhead dominates there and does not here.

**Consequence:** ~0.42 board ms per member × 0.536 `time_score`/ms × 0.069 weight =
**−0.015 `final` per member**. Against +0.043 `final`/member measured for members 4–8, members
are still net positive, but the margin is ~3:1, not infinite. And it is recoverable: batching the
eight members into one forward pass (stack along the batch dimension, or fuse to a grouped
model) should return most of the 2.04 ms → **+0.08 `final` for no accuracy change.**

---

## 4. The exchange rate: what accuracy is worth

Measured, not modelled. Six **held-out** members on the AoA 15–20 proxy, errors scaled
synthetically (`P' = Y + k(P−Y)`), board tail inflation 1.39×, band refit at every `k`
(`Track_1/exchange_v9.py`):

| k | rel-L2 | TKE | MVPE | rel_s | tke_s | mvpe_s | **sps** | Δ`final` |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.1069 | 0.4796 | 0.0929 | 93.09 | 75.00 | 93.94 | **36.43** | — |
| 0.95 | 0.1015 | 0.4703 | 0.0882 | 93.41 | 75.36 | 94.22 | **37.51** | +0.438 |
| 0.90 | 0.0962 | 0.4607 | 0.0836 | 93.73 | 75.75 | 94.51 | **38.65** | +0.893 |
| 0.85 | 0.0908 | 0.4504 | 0.0789 | 94.06 | 76.16 | 94.80 | **39.86** | +1.369 |
| 0.80 | 0.0855 | 0.4391 | 0.0743 | 94.39 | 76.62 | 95.09 | **41.14** | +1.869 |
| 0.70 | 0.0748 | 0.4128 | 0.0650 | 95.06 | 77.70 | 95.68 | **43.98** | +2.965 |
| 0.50 | 0.0534 | 0.3401 | 0.0464 | 96.42 | 80.88 | 96.87 | **51.13** | +5.682 |

**≈ +0.089 `final` per 1 % of error removed**, essentially linear out to −20 %.

Decomposition at k = 0.90 (a 10 % error cut):

| channel | Δsubscore | × weight | Δ`final` | share |
|---|---|---|---|---|
| `sps` | **+2.22** | 0.181 | **+0.401** | **45 %** |
| `tke` | +0.75 | 0.244 | +0.183 | 21 % |
| `rel_l2` | +0.64 | 0.266 | +0.170 | 19 % |
| `mvpe` | +0.57 | 0.241 | +0.137 | 15 % |

### The two conclusions that should drive everything

**(a) SPS is where the return is, and you reach it through accuracy, not through the band.**
Per unit of relative error removed, SPS pays **2.3× more than any other channel**. It is weighted
0.181 and returns 45 % of the gain, because coverage-of-an-interval is a far steeper function of
the error than `100/(1+0.5e)` is. The band itself is closed — three independent attacks each
returned ~+0.5 SPS and the response is flat around every optimum. Do not reopen it. Shrink the
error and SPS follows on its own.

**(b) TKE is a trap.** It is the ugliest subscore (78.88) and carries the largest accuracy
weight, so it looks like the target. But it moves only **0.75 points per 10 % of error**, and
the report's earlier finding stands that the target's own TKE field carries ≈0.47 irreducible
PIV sampling noise against our 0.535 — so the *whole remaining budget* in TKE is about
**+0.6 `final` for anyone, ever**. Chasing it with a generative model is a large project for a
capped, small prize. §6 revisits this against the literature and reaches the same answer.

**(c) Corollary on method selection.** Because SPS scores per-element interval containment, it
rewards **tail** reduction disproportionately. A method that lowers mean error while leaving the
worst elements alone will underperform this table; a method that lowers the worst elements will
beat it. **Prefer robustness and regularization over average-case accuracy and capacity.** That
is also the direction round 8's capacity result already pointed.

### What this costs in absolute terms

| target `final` | error reduction needed |
|---|---|
| 80.6 | ~6 % |
| 81.0 | ~10 % |
| 82.0 | ~21 % |

For scale: **going from 3 to 8 ensemble members bought ~1.6 %.** Anything that gets us to 81
must be roughly six times larger than ensembling.

---

## 5. Does DPOT help? — reading the benchmark's own Foil rows

The premise needs correcting before the question can be answered.

**On the Foil dataset, DPOT is not the best model. U-Net is, on every metric reported.**

Real-world-finetuning column, Foil (Tables 1, 2 and 5 of `2270_RealPDEBench`):

| model | Rel L2 ↓ | MAE ↓ | R² ↑ | FE ↓ | 2-round | 3-round |
|---|---|---|---|---|---|---|
| **U-Net (23 M)** | **0.0145** | **0.00277** | **0.99186** | **14.64** | **0.0191** | **0.0227** |
| DPOT-L-FT (674 M) | 0.0161 | 0.00324 | 0.98898 | 18.24 | 0.0209 | 0.0245 |
| DPOT-S-FT (41 M) | 0.0174 | 0.00351 | 0.98890 | 26.62 | 0.0218 | 0.0253 |
| WDNO (92 M) | 0.0181 | 0.00344 | 0.98951 | 26.05 | 0.0232 | 0.0269 |
| FNO (269 M) | 0.0206 | 0.00404 | 0.98664 | 43.83 | 0.0271 | 0.0314 |

DPOT-L is 11 % worse than U-Net on Foil rel-L2 and 17 % worse on MAE, and DPOT-S — the size we
could actually afford — is 20 % worse. Where DPOT wins is **the average across all five
datasets** (§4.4, Figure 4: closest to the origin in the RMSE/frequency-error plane), driven by
Cylinder and Combustion. The paper's own sentence about Foil-like tasks points the other way:
*"convolution-based methods (U-Net, CNO) achieve lower RMSE, as the tasks resemble image
processing."*

**Second problem: those numbers are not comparable to our board at all.** Benchmark Foil is
99 trajectories × 3990 frames at 128×256, split at the parameter level. We get 82 trajectories ×
607 frames at 64×128, decimated to 32×64, tested on a set harder than any trajectory we hold.
Their best rel-L2 is 0.0145; our board is 0.1134 — an 8× gap that is about data volume,
resolution and test difficulty, not architecture. **No row in that paper can be used to predict
what any model would score here.**

**Third problem: DPOT's shape is wrong for this task.** DPOT is next-frame autoregressive
(T frames → 1 frame, rolled). Our task is one-shot 20 → 20. Rolling DPOT 20 steps means 20
forward passes of a 30 M+ transformer at their padded 128 resolution, and it reintroduces
rollout error accumulation that our one-shot model does not have. The one DPOT ingredient that
would transfer cheaply — **Gaussian noise injection on the input during training** — exists
precisely to damp that accumulation, and we have already measured it as a null in our
one-shot setting (0.05 → 83.163, 0.15 → 83.181 against a seed spread of 83.177–83.274). The null
and the mechanism are consistent: we do not have the disease the drug treats.

**Fourth problem: it fights round 8's clearest result.** Capacity is *monotonically harmful*
here — width 24 → 64 goes 81.67 / 81.54 / 81.33 / 81.17 on the proxy while training loss
improves 0.069 → 0.038. With 74 trajectories, 30 M parameters is the wrong direction unless the
pretraining is carrying the model, and our one measurement of sim-pretraining-carrying-the-model
is **+0.00**.

**Verdict.** Not recommended as a primary direction. The only version with positive expected
value is narrow: *fine-tune the released DPOT-S checkpoint as one more member of the ensemble*,
where round 8 measured architectural diversity at +0.05–0.07 per member. That is a multi-day
build for a ~+0.06 prize with real downside risk from 20 sequential transformer passes on the
board. It sits at the bottom of §7's list, and I would rather spend the time on §7 items 1–3.

---

## 6. Literature scan — 1842 abstracts, what maps and what does not

### Maps well

**Generative correction of spectral bias.** The single strongest theme, and it targets our exact
failure mode (L2 training → oversmoothed fluctuations → TKE 0.535).
[Learning Turbulent Flows with Generative Models](https://arxiv.org/abs/2509.08752) (Oommen et
al., 2025) is the closest match: an **adversarially trained neural operator** cuts energy-spectrum
error 15× *at neural-operator inference cost* — the discriminator is a training-time cost only,
so it does not touch `time_score`. Related: [Comparison of Generative Learning Methods for
Turbulence Surrogates](https://arxiv.org/abs/2411.16417) (DDPM and DCGAN both reproduce PIV wake
statistics; DCGAN much cheaper at inference),
[Stochastic Flow Matching](https://arxiv.org/abs/2410.19814),
[Correcting Neural Operator Spectral Bias via Diffusion Posterior Sampling](https://arxiv.org/abs/2606.03936).

*But* — §4(b). The prize is capped at ~+0.6 `final` and there is a second obstacle: our
`ALPHA = 0.75` fluctuation gain has already been swept with the band refit at every point, and
0.75 is optimal. That means the amplitude trade-off between rel-L2 and TKE is **already at its
optimum for the fluctuation field we produce**. A generative model only wins if its fluctuation
field is better *structured*, not merely larger — and the variance head already reaches
correlation 0.907 with the target. Worth one cheap diagnostic (§7.6) before any build.

**Robustness / OOD generalization by constraint rather than capacity.**
[Geometric and Physical Constraints Synergistically Enhance Neural PDE Surrogates](https://arxiv.org/abs/2506.05513)
finds symmetry constraints beat physical constraints, both beat data augmentation and the
pushforward trick, and — the relevant part — **doubly-constrained surrogates generalize better to
initial conditions beyond the training range**. That is our exact problem: the board is outside
our data. Our flow has no rotational or translational symmetry (fixed inflow, fixed airfoil), so
the direct transfer is limited, but the framing supports §7.2.

**Test-time adaptation.** Newly affordable per §2.
[IT³: Idempotent Test-Time Training](https://openreview.net/forum?id=MHiTGWDbIb) adapts using only
the current test instance with no auxiliary task. We are handed 20 frames of the actual test
trajectory at inference — genuinely in-distribution unlabelled data we currently throw away after
one forward pass. Unproven for this setting, but the budget now exists.

### Maps poorly — recorded so it is not re-derived

* **Sim-to-real pretraining** ([PhysGuard](https://arxiv.org/abs/2606.16602),
  [Emergent Transfer of a Physics Foundation Model](https://arxiv.org/abs/2606.01470),
  [Hypersonic data fusion](https://arxiv.org/abs/2511.20149), OmniFluids). The whole cluster
  addresses a stage we measured at **+0.00** (round 5: sim-norm → real-norm, 86.68 vs a
  from-scratch mean of 86.673). PhysGuard offers a *mechanism* for why (fine-tuning destroys the
  physics-critical Fisher directions), which is the only reason to revisit — and it would be one
  experiment, not a programme. Note also that the benchmark's own Foil rows show sim pretraining
  buying U-Net just 0.0159 → 0.0145, i.e. 9 %, the smallest gain of any of their five datasets.
* **Rollout stabilization** ([Controlling Transient Amplification](https://arxiv.org/abs/2605.08856),
  pushforward, commutativity regularization, DPOT's noise injection). All of it targets error
  accumulation across autoregressive steps. **We emit 20 frames in one shot and have no rollout.**
  Skip the entire category.
* **Conformal prediction for the band.** SPS is coverage × `exp(−2h/σ)`, not coverage. A
  conformal quantile optimizes the wrong objective; our exact solver already maximizes the real
  one. The only place conformal could contribute is choosing the 1.39× inflation factor
  principledly instead of by board back-fit — and that factor's response curve is flat.
* **Data assimilation / sparse reconstruction** (the largest cluster by count, ~290 abstracts).
  These reconstruct fields from sparse sensors. We are given the full field. Not our problem.

---

## 7. Plan, ranked by (expected `final`) / (effort × risk)

Nothing below has been implemented. Sizes are my estimates; the exchange rate in §4 converts them.

### 1. Batch the ensemble into one forward pass — **+0.08, hours, near-zero risk**
`_fno.predict_fno` loops `for m in models:`. Stack the eight members along the batch dimension
(or fuse to a grouped/blocked FNO) so the board sees one invocation instead of eight. §3 predicts
this recovers most of 2.04 board ms. Verify identical outputs bit-for-bit before and after.
No accuracy change, no retraining, no new evidence needed.

### 2. Decimation-phase augmentation — **est. 3–8 % error, days, low risk**
`model.py:load_trajectories` does `f["u"][:, ::SUB, ::SUB]` with `SUB = 2` on **64×128** raw
data. Phase (0,0) is the scored grid; phases (0,1), (1,0), (1,1) are **three more valid 32×64
views of the same physics that we currently discard**. That is a free 4× in training windows,
and unlike the Gaussian input noise we falsified, each one is a *real* flow sample rather than
synthetic perturbation. We are demonstrably in the data-limited regime — capacity is
monotonically harmful, training loss falls while validation worsens — which is exactly where this
kind of augmentation pays. Validate and score on phase (0,0) only.
*Risk:* the airfoil geometry shifts half a coarse cell between phases, so the model must tolerate
a slightly moved mask; if that costs more than the regularization buys, it shows up immediately
on the held-out proxy.

### 3. Train for the regime the board actually tests — **est. 3–10 % board error, days, low risk**
Error is governed by angle of attack: 0° ≈ 0.057, 5° ≈ 0.044, 10° ≈ 0.073, 15° ≈ 0.103,
20° ≈ 0.105, and the board (0.113) sits at or above our hardest trajectory. Training treats all
AoA equally, so ~60 % of the gradient comes from regimes the board barely tests. Options, cheapest
first: (a) reweight the sampler toward AoA 15–20; (b) per-window loss weighting by difficulty;
(c) a high-AoA specialist blended in. Evaluate **only** on the AoA 15–20 held-out proxy — the
"all" split will report this as a regression and be wrong to.
*This is the item most likely to move the board specifically*, because it is the only one aimed
at the distribution gap rather than at average accuracy.

### 4. Spend the newly-discovered latency budget — **est. −0.3 to −0.8 cost, needs 4–9 % to break even**
§2 says 20 ms costs 0.33 and 50 ms costs 0.78. Candidates in order of expected value:
   * **More members** — the curve is still rising, now correctly priced at −0.015/member.
   * **Multi-crop / phase-ensemble at inference** — predict on all four decimation phases and
     average back onto phase (0,0). Pairs naturally with item 2, is a genuine variance reduction,
     and costs 4× forward passes (~+1.3 board ms, −0.05 `final`).
   * **Test-time adaptation** — a few gradient steps on the input window's own internal dynamics
     (predict frames 11–20 from 1–10) before predicting 21–40. Highest ceiling, least evidence,
     and the only item here that would need a serious latency measurement first.

### 5. Re-fit the projection model and re-validate on the next board round — **free**
Replace `final ≈ mean5 + 0.73` (now falsified) with the §2 weights everywhere: `validate_strict.py`,
`ens_v9.py`, `compare_v9.py`, `alpha_board.py`. Record the predicted `final` **before** the next
submission so the weighting hypothesis gets a real test rather than a retrofit.

### 6. Diagnose whether TKE's residual is amplitude or structure — **hours, decides a large fork**
Before anyone builds a generative model, answer: is our fluctuation field the wrong *size* or the
wrong *shape*? Compare (i) current TKE, (ii) TKE after an oracle per-pixel amplitude rescale,
(iii) the 0.47 sampling-noise floor. If (ii) ≈ (i), amplitude is already right and the residual is
structural, so adversarial training is the only route and it is worth ≤+0.6. If (ii) ≪ (i), there
is a cheap calibration left. Cheap, and it either opens or closes §6's largest cluster.

### 7. DPOT-S as a diversity member — **est. +0.06, multi-day, high risk** — see §5
Only if 1–4 are done and the board has stopped moving.

### Explicitly not doing
Band form (closed, three attacks, flat response) · rollout stabilization (no rollout) ·
sim pretraining as a programme (+0.00 measured) · conformal intervals (wrong objective) ·
data assimilation (full field given) · more capacity (monotonically harmful) ·
input-noise augmentation, `--chan-pow`, `ALPHA ≠ 0.75`, spread-adaptive bands, fitted
fluctuation gains, input-mean blending for MVPE — all measured nulls, listed in `CLAUDE.md`.

---

## 8. Standing rules that still apply

* **Nothing gets decided on a number computed from an all-data checkpoint.** Three contamination
  artifacts in three rounds. All-data models supply weights, never evidence.
* **Evaluate on the AoA 15–20 held-out proxy**, not the full split. The full split is 46 % away
  from the board; the proxy is 5–6 %.
* **A local gain under ~0.4 mean5 is unproven** — two very different models landed 0.0004 apart
  on the board while differing 0.008 locally.
* Run-to-run σ is ~0.05 in v9 mean5.
* One submission per day, 100 per phase. Submitting is the user's call.

## 9. Open question for the organizers

The public RealPDEBench release (github.com/AI4Science-WestlakeU/RealPDEBench) contains the Foil
dataset at **99 trajectories × 3990 frames × 128×256** — 6.6× our frames and 4× our pixels, and
the hidden test set is plausibly drawn from held-out parameters of that same collection. Whether
using it is permitted is a rules question, not a modelling one, and it should be asked before any
work depends on the answer. Do not build on it in the meantime.
