# RealPDE — project roadmap

Written 2026-08-31 after the round-12 board result; **revised the same day** after the
Track-2 build. This is the handoff document: it records what the leaderboard actually says,
what is closed, and what to do next. Read it with `CLAUDE.md` (contracts, measurements,
per-round detail), `Track_1/board_rounds.py` (every number in Part 1 is reproducible from it)
and `Track_1/plan/plan_20260831.md` (the current round's plan and recorded predictions).

**One-line state:** Track 1 is saturated at ~80.55 and has not measurably improved since
2026-08-08. **Track 2 is now built, validated and submittable** — and it turned out to be
Track 1's model streamed, not a new lever: all three test-time-adaptation families were
measured and are negative. What Track 2 does buy is an equal prize pool for work already done,
plus a large `time_score` win from CUDA-graphing the batch-1 step.

**Revision, 2026-08-31 evening.** Two things changed the plan:

* **P0 is done.** `Track_2/` is complete: submission, streaming evaluator on real held-out
  trajectories, a strict submission gate, and the three adaptation families priced and closed
  (`Track_2/README.md`). Track 2 needs no more research, only a submission.
* **The board's error subscores are noise-free; only `time_score` is noisy** (Part 1.7 below).
  That changes how every future comparison should be read and reopens board A/B testing as an
  affordable instrument — there are ~27 submission slots left per track.

---

## Part 1 — What the leaderboard actually says

### 1.1 Every scored round

All eight submissions from the shared account, recovered from Codabench on 2026-08-31. Rounds
X1 and X2 are a collaborator's; no package for them is recorded in this repo.

| round | date | id | rel_l2 | tke | mvpe | time | sps | **final** |
|---|---|---|---|---|---|---|---|---|
| R6 U-Net, 1 inv | 08-04 | 878641 | 94.4342 | 78.1660 | 93.7922 | 91.6973 | 36.9374 | 79.73300 |
| R7 FNO ×3, 3 inv | 08-05 | 879768 | 94.5754 | 78.5278 | 93.9100 | 91.7264 | 37.4680 | 79.98488 |
| R9 FNO ×8, 8 inv | 08-06 | 881164 | 94.6342 | 78.8825 | 93.8030 | 90.5066 | 38.2338 | 80.11507 |
| R10 7 memb, 3 inv | 08-08 | 883017 | 94.5764 | 79.1528 | 93.6812 | 91.1209 | 39.6263 | 80.51202 |
| X1 collaborator | 08-09 | 883908 | 94.4619 | 79.5940 | 93.6552 | 92.1025 | 39.3281 | 80.51905 |
| R11 8 memb, 3 inv | 08-10 | 885003 | 94.5709 | 79.0805 | 93.3501 | 92.7771 | 39.5450 | **80.60907** |
| X2 collaborator | 08-11 | 886137 | 94.4616 | 78.8439 | 92.9265 | 93.2630 | 38.8785 | 80.37218 |
| R12 11 memb, 4 inv | 08-31 | 909571 | 94.4903 | 79.1677 | 93.4181 | 91.8036 | 39.6373 | 80.51584 |

Best other team, read 2026-08-25: **`np-user` 81.76136.** Our best raw score is R11's 80.60907.

### 1.2 Round 12 missed its prediction by −0.191, and the reason is latency

Predicted 80.707 (recorded pre-submission, `Track_1/plan/campaign_20260829.md`). Actual 80.516.
The round's whole thesis was "TKE carries the round": predicted `tke` **+0.716** over round 10,
delivered **+0.015**.

R12 against R11 — the only clean A/B, since they differ only by `+s6`, `+s7`, `+unet_all_g1`
and the band refit that follows:

| subscore | Δ | × weight | = final |
|---|---|---|---|
| rel_l2 | −0.081 | 0.545 | −0.044 |
| tke | +0.087 | 0.170 | +0.015 |
| mvpe | **+0.068** | 0.000 | +0.000 |
| **time** | **−0.973** | 0.081 | **−0.079** |
| sps | +0.092 | 0.204 | +0.019 |
| **total** | | | **−0.089** (actual −0.093) |

**The 11-member package is fractionally MORE accurate than the 8-member one. It lost on the
fourth invocation.** MVPE did not regress; an earlier same-day reading that blamed the U-Net
member compared against round 10, the wrong baseline, and is withdrawn.

### 1.3 At constant latency, rounds 10–12 are indistinguishable

`time_score` swings 2.8 points across the eight rounds and is only partly ours to control
(§1.5). Holding it at the median re-prices the rest:

| round | board final | normalised |
|---|---|---|
| R6 | 79.733 | 79.738 |
| R7 | 79.985 | 79.988 |
| R9 | 80.115 | 80.217 |
| **R10** | 80.512 | **80.564** |
| X1 | 80.519 | 80.492 |
| R11 | **80.609** | 80.527 |
| X2 | 80.372 | 80.251 |
| R12 | 80.516 | 80.513 |

**Real progress stopped on 2026-08-08.** R11's apparent +0.097 and R12's apparent −0.093 are
both mostly board-timing swing. Rounds 11 and 12 — more members, a second architecture family,
α 0.90, mode-padded fusion, two more seeds — bought **nothing measurable**.

### 1.4 The subscore weights, refitted — and they overturn the strategy

Eight rounds against five weights with Σw = 1 and no intercept is overdetermined (3 dof) and
fits to a max residual of **0.025**. Unlike rounds 9–10 this is a *tested* model rather than an
exact interpolation. (Round 10 rejected the no-intercept form on four points; it holds on eight.)

| subscore | fitted | admissible | round-9 claim |
|---|---|---|---|
| **rel_l2** | **0.545** | 0.327 – 0.637 | 0.266 |
| tke | 0.170 | 0.048 – 0.293 | 0.244 |
| **mvpe** | **0.000** | 0.000 – 0.160 | 0.241 |
| time | 0.081 | 0.054 – 0.130 | 0.069 |
| sps | 0.204 | 0.170 – 0.237 | 0.181 |

Consequences, all of which invalidate advice still sitting in `CLAUDE.md`'s older sections:

* **`rel_l2` is the most valuable subscore, at ~2× the assumed weight.** Round 10's
  "**stop optimizing rel-L2**, TKE and SPS carry the response" is **FALSIFIED**. It read the
  *curve* correctly (`100/(1+0.5e)` is flat near e ≈ 0.1) and paired it with a weight that was
  off by half.
* **`mvpe` is worth approximately nothing.** Its 96.75 ceiling, its regressions, and the effort
  round 2 spent chasing it were all noise. Stop scoring it as a decision variable.
* Re-pricing round 9's exchange rate: **a 10 % error cut is worth +0.93 `final`**, split
  **sps 49 % / rel_l2 38 % / tke 14 % / mvpe 0 %**. Closing the 1.25 to the leader needs a
  **~13 % error reduction** — more than rounds 8–12 delivered *combined*.

### 1.5 The board-time model is overfit; do not trust it

`board ms = 4.447 + 0.324·inv + 0.431·local ms` fits its own four fitting rounds to ±0.064 ms
and misses **R11 by −1.86 ms** and **R12 by −0.90 ms**. R7 and R11 both ran 3 invocations at
similar local cost and landed **1.5 ms apart**. Observed board latency spans 3.80–8.02 ms.

**Budget for invocation count; do not predict `time_score` to better than ±1 point.**

### 1.6 `predict_board.py` is 1-for-2

Round 11: predicted 80.603, actual 80.60907 (+0.006). Round 12: predicted 80.707, actual
80.51584 (−0.191). One good call and one bad. Keep predicting before submitting — it is the
only way any of this stays falsifiable — but do not treat its output as a number you can spend.

The whole round-12 miss was one subscore: `tke` predicted 79.869, actual 79.168. Its fixed
board/proxy ratio (1.117) was anchored on a single round. `Track_1/proxy_calib.py` reconstructs
the proxy for every recent round's twin member set so the ratio can be tested rather than
assumed — and, more usefully, so the proxy's **resolution limit** can be measured: the smallest
local delta that actually transfers.

### 1.7 The board's ERROR subscores are noise-free. Only `time_score` is noisy.

`rel_l2`, `tke`, `mvpe` and `sps` are deterministic functions of the submitted arrays and the
hidden targets; resubmitting the same zip reproduces them exactly. `time_score` is the only
stochastic term, and §1.5 shows it spans 2.5 ms across rounds with the same invocation count,
in an order that does not follow local time — **≈ ±0.07 `final` at 1σ, on its own.**

Two consequences, and they are the most useful things in this document:

* **Compare packages by subscore DELTA, never by `final`.** R11 → R12 measured, exactly:
  rel −0.081, tke +0.087, mvpe +0.068, sps +0.092 ⇒ **−0.010 `final`**. The 11-member package
  is a wash against the 8-member one and we know it to three decimals. The apparent −0.093 was
  the 4th invocation plus timing luck.
* **Never resubmit for a better timing draw.** It is gaming noise and it burns the slot.

### 1.8 The local proxy's resolution, measured: ~2 % of error, and it can invert

`Track_1/proxy_calib.py` reconstructs the held-out proxy for each recent round's twin member
set and compares it with that round's board result.

* **As a LEVEL instrument it works**: `board rel-L2 ≈ 1.078 × proxy rel-L2` on the AoA 15–20
  subset (1.081 / 1.063 / 1.090 over R10/R11/R12), stable to **±1.3 %** ≈ ±0.12 `final`.
  `tke` ×1.150 (±4.5 %), `mvpe` ×1.511 (±6.3 %).
* **As a DELTA instrument it fails, and inverts.** R11 → R12: proxy rel-L2 **−0.87 %**, board
  rel-L2 **+1.57 %**. The sign was wrong. R10 → R11: proxy +1.78 %, board +0.11 %.

> **A change must move the proxy's error by more than ~2 % to be worth shipping** — ≈ +0.19
> `final` at the measured exchange rate — and even then the board disagrees about the sign
> roughly a third of the time.

Every change made in rounds 9–12 was smaller than that. §1.3's "real progress stopped on
2026-08-08" is therefore a statement about the instrument, not about the ideas.

### 1.9 The `rel_l2` ceiling is withdrawn — the model already beats the "floor" it was built on

`CLAUDE.md` has quoted a `rel_l2` bound of **97.0–97.4** since round 3, built on a "time-mean
sampling floor" of **0.0531** from `diag_mean_floor.py`, and round 10 concluded from it that the
time-mean is closed ("already at its sampling floor: 0.0503 vs 0.0531 irreducible").
`Track_1/floor_aoa.py` and `Track_1/mean_floor2.py` measure both sides on the eight held-out
trajectories:

| | value |
|---|---|
| half-split "irreducible floor" | 0.0531 |
| sampling error against the 868-frame trajectory mean | 0.0735 |
| **our model's time-mean error** | **0.0438** |

**The model is below both, at every angle of attack — 0.0632 against 0.0830 at AoA 20.** It is
not a floor. Neither estimator is: both measure how much the target's 20-frame mean *varies*,
and the model is not guessing blind — it is handed the immediately preceding 20 frames of a
quasi-periodic flow, so much of that variation is phase information the input carries. The
irreducible error is the conditional variance of `mean_t(y)` given the input window, which
neither number computes.

**So there is no measured bound on `rel_l2` at all**, and it is the subscore worth 0.545. The
time-mean carries **29 % of rel-L2's error energy** (0.0438 against a total 0.0814, fluctuation
part 0.0685); driving it to zero would cut rel-L2 by **16 %** ≈ **+1.5 `final`**, well above the
proxy's ~2 % resolution.

This does not make it easy — `--w-mean 3`, `--w-mean 8` and blending toward the input window's
mean are all measured and null. It makes it **open**, where the map said closed.

---

### 1.11b AT THESE EFFECT SIZES, ONE RUN IS NOT A MEASUREMENT

Round 13 ran ~30 arms and produced **four** cases where a single run at 2–3σ regressed once
replicated or measured with a larger instrument:

| claim | first reading | after |
|---|---|---|
| the osc "constrainedness" ordering | monotone in constraint | the two extremes **swap** on the second split |
| out-of-distribution seed variance | ~1400× | **58×** (a range-of-2 is a poor σ estimator) |
| `--dropout` buys TKE | −3.1 % (336 unpaired windows) | **+0.95 % WORSE** (3 383 paired, z +2.74) |
| `--osc-zeromean` score | +3.14σ (1 run) | **+1.60σ** (4 runs) |

Every one of those was corrected only because the check was built. **This is a property of the
setup, not luck.** Two rules follow:

* **Replicates belong in the first job, not the follow-up.** A single arm at 2σ costs 18 min to
  replicate and is worth nothing without it.
* **The 126- and 336-window splits cannot resolve anything under ~2σ.** Use `kfold.py --paired`
  (3 383 windows, trajectory bootstrap) for any claim that will change a package.

Combined with §1.11's measurement — the proxy resolves σ = 0.028 `wfinal` while the board's
transfer resolution is ≈ 0.5 — **the proxy is precise but not accurate, ~18× finer than the
board can distinguish.** More seeds cannot fix that; the missing quantity is distribution
shift, not variance.

### 1.12 In-distribution fit does not determine out-of-distribution behaviour

`re_split.py`, six arms on the same 56 training trajectories. **Two runs of the identical
recipe are indistinguishable on the interior control and 22 % apart under Re extrapolation:**

| | `rein` rel-L2 | `rein` wfinal | `relo` rel-L2 | `relo` wfinal |
|---|---|---|---|---|
| control seed 1 | 0.0754 | 83.989 | 0.1147 | 82.349 |
| control seed 2 | 0.0754 | 83.990 | 0.1404 | 80.930 |

They are genuinely different models (max |w₁ − w₂| = 1.19; `train2.py` sets no seed).
~~Seed variance is ~1400× larger out of distribution~~ — **that came from a range-of-two and is
withdrawn.** With **four** controls (`re_var`): `rein` 83.999 ± **0.0122**, `relo` 81.647 ±
**0.7073**, `rehi` 78.936 ± **0.1408** ⇒ **σ is 58× larger on `relo` and 11.6× on `rehi`.**
Note `relo` is 5× noisier than `rehi`, so extrapolating **below** the training range is far
less stable than above it — and Re 2968, the known test condition, is below.

The hidden test set contains unseen Reynolds numbers, so board score carries a
which-draw-did-we-get term that **no in-distribution proxy can see**, and every proxy this
project has used is in-distribution in Re. That is a candidate explanation for §1.3 and §1.8 —
why local deltas stopped transferring and why rounds 10–12 are indistinguishable. It also says
the 8-member ensemble is averaging exactly this variance down, which fits the board's flatness
across member counts even though the accuracy gains were nil.

**Consequence for every future arm: one run is not a measurement out of distribution.**
`--dropout 0.15` is the first arm to beat both controls on all three sets (`relo` rel-L2
**0.1067** against 0.1147/0.1404) and it still needs replicates.

### 1.10 A submission's value is the ambiguity it RESOLVES, not the score it earns

`Track_1/sub_value.py`. A submission scores **and** adds one equation to the eight that pin the
weight family. For a candidate with predicted subscores `x`, every admissible `w` predicts a
different `w·x`; the **width of that interval is exactly what observing it resolves**.
`final_v9.predict` was already returning it as an error bar — it is now a design criterion.

The family's own spread directions say what a probe should move:

| axis | relative spread | direction |
|---|---|---|
| 1 | 1.00 | `rel_l2 +1.00  tke −0.79  mvpe −0.36  time −0.08  sps +0.23` |
| 2 | 0.64 | `rel_l2 +0.46  tke +0.99  mvpe −1.00  time −0.20  sps −0.25` |
| 3 | 0.17 | `rel_l2 −0.28  tke −0.21  mvpe −0.53  time +1.00  sps +0.02` |

**A uniformly better package moves along `1`, and Σw = 1 is the one thing already known.** That
is why twelve rounds of ensembling bought so little information as well as so little score, and
why the whole 1.172 of currently-open decision-ambiguity sits on trades. **Informative
submissions trade subscores against each other; they are not the best package.**

Already settled by the eight rounds, so stop spending slots testing them: **more members** is
−0.130 … −0.060 and **rel_l2 for tke one-for-one** is −0.586 … −0.111, both negative across the
entire family. Still open: **α** (−0.204 … +0.248) and **mvpe against tke** (−0.117 … +0.059).

### 1.11 The local proxy's variance is fixable, and `kfold.py` fixes it

§1.8's ~2 % resolution is not bias — it is variance from **eight** held-out trajectories.
`Track_1/kfold.py` partitions all 82 into five folds stratified over (AoA, Re), trains one
model per fold and scores each only on the fold it excluded: **3 383 out-of-fold windows over
the full grid** instead of 336 over eight arbitrary points. `--paired` compares two arms on
identical windows and bootstraps **trajectories** (windows tile a trajectory at stride 20 and
are not exchangeable), which replaces comparing a delta to a borrowed seed σ. It does **not**
fix the board's distribution shift; `re_split.py` is the instrument for that. Cost is K
trainings per arm — screen on the AoA proxy, confirm here.

### 1.13 AN ARM CAN FAIL BECAUSE THE SCORER DOES NOT IMPLEMENT IT

`--norm self` was closed on 2026-08-31 as *"not neutral — it is broken"*, on rel-L2 1.1436
against a control's 0.1147. It was the **instrument**. `_fno.predict_fno` — which every
evaluation path and the package itself run through — hard-coded the global `MEAN`/`STD`, so a
checkpoint that divides each window by its own mean speed before standardizing was fed an
input it had never seen and denormalized with the wrong affine. Only the retired `eval_wp.py`
ever read the flag.

**The tell was available before any debugging, and it is general.** Score a model whose
self-normalized output is *exactly the target* through the old path: rel-L2 **0.198 / 2.224 /
0.349** on `rein` / `relo` / `rehi`. The arm measured 0.147 / 1.144 / 0.299 — **it beat a
perfect model on all three sets, in the same order.** No model can do that. When an arm scores
worse than the theoretical worst case for its own hypothesis, suspect the harness, not the
idea.

> **Standing check.** Before believing that an arm which changes the *representation* — the
> normalizer, the output parameterization, the target space — is bad, confirm the scoring path
> implements the change. A tuning knob is safe (the same code runs either way); a
> representation change is not. `Track_1/verify/selfnorm_path.py` is the pattern: assert
> bit-identity against the committed reference for the default, assert the new affine
> round-trips, and refuse to mix the two conventions in one ensemble.

Fixed in `ecd1aa6`; nothing shipped moves (81 of 82 checkpoints are `--norm real`). See
`report_20260831_round13.md` §5.3b/§5.3c.

**The arm was then measured properly and is closed** (§26, three seeds on the Re split). It
does what Strategy item 5 promised *for the field* — `relo` rel-L2 **−11.1 %**, and it collapses
the low-Re seed variance **13×** (σ 0.0109 → 0.00083; `wfinal` 0.707 → 0.065), which is the
exact quantity §5.5 blamed for local numbers not transferring. It pays **+20.8 % TKE** on the
same set, reliably (seed σ 0.0068), because the variance head then predicts σ in
self-normalized units and so hard-codes a Reynolds-*independent* relative turbulence intensity.
Net: wash single-model, **−0.50 `wfinal` ensembled**, decisively worse in distribution.
**Do not ship it**, and do not run the field/σ hybrid either — priced at ~+0.05 after §7.3's
dilution against a certain −0.04.

## Part 2 — Closed. Do not re-propose without new evidence.

Everything here was closed by measurement, with the file that measured it. The list exists so a
new session does not spend a week re-deriving a negative.

**Closed with BOARD evidence (strongest):**

| direction | evidence |
|---|---|
| More ensemble members | R12 (11 memb) ≈ R11 (8 memb) ≈ R10 (7 memb) at constant latency, §1.3 |
| A second architecture family (U-Net) | gate M measured +0.091 proxy; board delivered ~+0.09 `tke_score` ≈ +0.015 final |
| Extra invocations | R12 paid 0.973 `time_score` for its 4th group and got it back nowhere |
| MVPE as an objective | weight ≈ 0.000, §1.4 |

**Closed with held-out local evidence** (each cost a real experiment; see `CLAUDE.md`):

* **Capacity, both directions.** w16 < w24 > w48 > w64; modes ladder flat. Architecture search
  finished (round 10b).
* **The SPS band.** Three independent attacks plus leave-one-trajectory-out CV over 84 configs;
  pooling the fitting split buys 0.000 sps (round 11, `band_cv_v9.py`).
* **The sigma head.** Readout shape/inputs (`--var-hidden`, `--var-deep`, `--var-in-std`) cannot
  move the training `var`; loss weights move `var` but not held-out TKE, and `--lam-v 0.3` is
  the optimum of the one family that moves it. Information floor *and* generalization ceiling,
  measured three ways including FGN (round 11–12).
* **FGN / CRPS probabilistic head.** Sample spread predicts σ at 0.451–0.469 against the
  log-MSE head's 0.329 (round 12, `eval_fgn.py`).
* **Self-normalization (`--norm self`), i.e. Strategy item 5.** Three seeds on the Re split
  through a scoring path that finally implements it (§1.13). It delivers the promised field
  gain — `relo` rel-L2 −11.1 %, seed σ down 13× — and pays +20.8 % TKE there because the σ head
  then predicts in self-normalized units. Wash single-model, **−0.50 `wfinal` ensembled**,
  worse in distribution. The field/σ hybrid is priced at ~+0.05 and not worth running
  (round 13, §26).
* **Phase / advection.** Fluctuation re-registration recovers 0.2 % on the real model where its
  synthetic control recovers 100 %. The error is decorrelation, not misregistration (round 12).
* **4× phase augmentation (`--phases 4`).** −0.009 at matched compute, **−0.369 at 4× steps
  (6.7σ)**. The three extra decimation phases are real but *correlated* views; they add far
  less independent information than their count suggests (round 10, report §5.6).
* **Sim pretraining.** +0.00 against a from-scratch control on a healthy model (round 5).
* **Diffusion (PDE-Refiner).** 82.53 vs its MSE control's 83.83 at 4× compute (round 1).
* **Physics losses except `--lam-grad`.** Divergence-to-zero fights the target; spectrum
  matching, Laplacian, momentum all null or negative.
* **Post-hoc sigma calibration.** Three separate nulls, worst −0.311 out-of-sample.
* Also null: long context, SWA, analog forecasting, input-noise augmentation, `--chan-pow`,
  `--w-mean`, `--lam-p`, `--lam-t` upward, `--point-mlp`, `--detach-var`, GNNs, KANO.

**Closed with held-out TRACK 2 evidence** (2026-08-31; detail and files in `Track_2/README.md`):

Windows tile at stride 20 with `T_in = T_out = 20`, so window k−1's revealed target IS window
k's input, bit for bit. **Anything computable from `prev_target_norm` alone was already
computable from `input_norm`, and Track 1 could compute it too.** Only two things are new: the
supervised PAIR `(x_{k-1}, y_{k-1})`, and the REALIZED ERROR of past predictions. All three
adaptation families that live in those two are measured and negative:

* **Residual EMA / output bias correction** — lag-1 residual correlation **+0.051**; the
  ORACLE per-trajectory additive correction is worth 1.75 % of rel-L2. Closes `CLAUDE.md`'s
  "strongest T2-specific play", which was in any case adapting toward the current input
  window's own std under another name.
* **Online SPS band calibration** — at the board's 1.39× error stretch the shipped band is
  already exactly optimal (best global rescale c = 1.00). Flat response, nothing to recover.
  A useful by-product for the Decision phase: over error stretches of 1.0×–1.7× the shipped
  band never gives up more than **0.667 sps (≈ 0.14 `final`)** against hindsight, so a harder
  or easier test set does not break it.
* **Gradient test-time training** — real but small: rel-L2 −1.6 % single member / −0.7 % on an
  ensemble, ≈ +0.02 `final`, against ~9.7 ms/step of backward pass ≈ **−0.5 `final`**. 40 arms,
  no sign change.

**"Transductive conditioning" is NOT clean, and should be struck from the plan.** `CLAUDE.md`
offers it as the legitimate cousin of the tiling leak — "per-trajectory statistics from all ~42
windows, reading no targets". That last clause is false. Window *k+1*'s input **is** window
*k*'s target, bit for bit. So any statistic pooled over more than one window is computed partly
from the answers, however aggregated; a trajectory-mean of the inputs contains 1/42 of each
window's own target. It also happens to survive `validate_strict.py`, because a pooled mean is
permutation-equivariant — so the existing guard would not catch it. **The only clean input for
window k is window k's own input window.** Nothing in the shipped package uses anything else,
and nothing should.

**Structurally excluded:** the window-tiling leak (`Track_1/submit_trick/DISCLOSURE.md`; the
organizers confirmed such scores are invalid), and any submission that is order-dependent
(`validate_strict.py` asserts permutation invariance).

**The one honest summary:** every lever that is not *new information* has been pulled. The
binding constraint remains what round 1 called it — **74 real trajectories** — and rounds 8–12
confirmed it from five directions.

---

## Part 3 — What to do next

Time remaining: **Main Development closes 2026-09-27**, Decision phase to 10-25, results 11-10.
**1 submission per day per track**, shared account — so **~27 slots per track**, against the
eight submissions the project has made in total. The binding constraint is no longer "don't
waste a submission"; it is that **we have very few measurements of the hidden test set and no
local proxy that predicts it.** Main Development is for learning; keep the best-known package
ready for the Decision phase.

**Standing rule for this phase: prefer the submission that most changes what we know, subject
to not obviously losing score.**

### P0 — DONE. Track 2 is built and submittable.

`Track_2/` holds the submission, a streaming evaluator over real held-out trajectories scored
by the official `scoring.py`, a strict pre-submission gate, and the three adaptation families
priced and closed. The package is 8 all-data members in **one fused invocation** with the
device-side step captured in a **CUDA graph** — bit-exact against eager (max abs difference
0.0e+00) and **3.61 ms → 0.989 ms/step**, worth ≈ +0.25 `final` for free. Full stream: 0.61 s
against a 10-minute limit.

Track 2 needs no further research. It needs submissions.

### P1 — Round 13: the SAME package on both tracks, at ALPHA 0.60 as a deliberate probe

Two questions in one slot.

**Whether the two tracks score the same hidden trajectories** is unknown and is the cheapest
high-value thing left to learn: if they do, every Track-1 instrument (the 1.39× error
inflation, the band, the AoA proxy, the weight fit) transfers to Track 2 for nothing. One
identical model on both boards answers it.

**And α is the best instrument the board offers.** Re-swept under the eight-round weights
(§1.4), **0.75 and 0.90 are tied** (+0.004 against a family width of ±0.10) — so round 11's
adoption of 0.90 was a coin-flip mispriced by the round-9 weights, and α is not a lever. But it
moves along the family's dominant undetermined axis (§1.10), so shipping **α = 0.60** costs
**−0.065 `final`** — *below the board's own `time_score` noise of ±0.07* — and resolves **0.637
of the 1.172** open ambiguity, against 0.026 for the staged package. The cross-track test is
unaffected: it needs both tracks to ship the *same* model, not a particular α.

Archives in `../sub_candidates/`, predictions recorded there before upload.

### P2 — Regularization: measured, small, and uniform. The family is done.

Weight decay is closed by mechanism (`GroupNorm` is scale-invariant in the weights feeding it,
so a weight-norm penalty reparameterizes rather than regularizes; at `1e-1` the spectral
weights lose 33 % of their norm while `head_field` *grows* to compensate). The
function-changing regularizers were then run, and so were two auxiliary losses and an
output-basis head, and they all landed in the same place:

| arm | mechanism | z vs 4 seeds |
|---|---|---|
| `oscg1` | oscillatory output basis, J = 1 | +2.08 |
| `--lam-tgrad 0.3` | temporal-derivative loss | +1.73 |
| `--lam-tgrad 1.0` | temporal-derivative loss | +1.67 |
| `--dropout 0.05` | channel dropout | +1.61 |
| `oscg2` | oscillatory output basis, J = 2 | +1.49 |
| `--dropout 0.15` | channel dropout | +1.28 |

**Six structurally unrelated changes span 0.8σ and all improve rel-L2 by 0.55–0.64 %.** That
reads as *the model is mildly under-regularized and any mild regularizer recovers ~0.6 %*, not
as six mechanisms each worth the same. Corroborated directly: the oscillatory head's gain was
measured to be **not** its stated phase mechanism
(`report/report_20260831_round13.md` §13.4).

→ **Expect combinations not to add**, expect none to clear the ~2 % transfer bar, and stop
screening new regularizers one at a time. `--aug-scale` is the exception still open, and only
on `re_split`: it is a Reynolds shift, so in-distribution numbers (−2.4σ, −3.5σ) measure the
wrong thing.

### P3 — The fluctuation's SHAPE — the ONLY open direction, and untouched by round 13

Two independent corrections landed on the same square this round.

* **The oracle split re-priced** (`report_20260831_round13.md` §3): σ +5.96, time-mean +1.61,
  **shape +2.96**. `shape` moves from last to second, above the time-mean, and it is the only
  piece that moves `rel_l2` (0.1086 → 0.0694, −36 %) — the subscore worth 0.545. Round 10's
  "the hardest part of the problem is worth the least" was a correct reading of a flat
  `100/(1+0.5e)` curve paired with a weight that was off by half.
* **The ceiling that closed it was a category error** (§4 of the same report).
  `diag_spectral.py`'s Γ² = 0.308 is computed from **our prediction**, so it is what a round-3
  U-Net achieved, not what is achievable — a perfect predictor scores 1 on the same instrument.
  Its legitimate conclusion (a post-hoc per-wavenumber filter is capped at +0.013) stands; the
  extrapolation to "the fluctuation is unreachable" does not. **Measured this round on the same
  336 windows: Γ² = 0.3288 (round-12 U-Net) → 0.3641 (current FNO) → 0.3865 (5-member
  ensemble), against the quoted 0.308.** It rises monotonically with model quality, which is
  what a model property looks like. **No measured bound on fluctuation predictability exists.**
  The fluctuation carries **70.6 %** of the squared error, and ensembling improves its PHASE,
  not its amplitude — so the residual is decorrelation, matching round 12's phase/advection
  null.

**Round 13 moved it, at the very end, and by less than the transfer bar.** For most of the
round every decomposition left the fluctuation part pinned at **0.0704–0.0707** — baseline,
`--w-mean 0.3` (3.3× more of the loss), `oscg2`, `osc2` alike — while **69 % of the squared
error lives there**. The loss-weight family cannot reach it (§12), the regularization family
does not touch it (§14.1), and the one arm built specifically for it was improving the
time-mean instead (§13.4).

**`ozdo` (`--osc-modes 2 --osc-global --osc-zeromean --dropout 0.05`) is the exception, over
four runs** (`report_20260831_round13.md` §22): fluctuation part **0.0695–0.0699**, Γ²
**0.37035 ± 0.00066 against 0.3641 (+1.66 %)**, wfinal +3.84σ on the proxy and +4.77σ on the
full split with a spread *tighter than the baseline seeds*. It is the first change in the
project to move either quantity.

**It still fails the §1.8 bar** — rel-L2 −1.13 % on the proxy against ~2 % — so it is not a
package change on its own, and four tight replicates do not fix that (they buy precision;
§1.8's failure is bias). **What it is: the first evidence that the fluctuation is reachable
at all, and a direction with two independent measured signatures to steer by.** Pull on the
thread — more zero-mean modes, the constraint without dropout at a larger J, the basis itself
— rather than shipping this point on it.

Round 12's phase/advection null still stands and constrains the *form* of any attack: the
fluctuation error is decorrelation, not misregistration, so no re-registration, no phase
co-training, no phase-aligned frames. What has never been tried is predicting the fluctuation
with anything other than the field head trained on per-element L2 — and §P2's diagnosis says
this is the *same problem* as generalization, because §1.12 shows out-of-distribution behaviour
is barely constrained by in-distribution fit and the fluctuation is where that error lives.

First arms, both cheap and both queued: **`--w-mean 0.3` and `--w-mean 0.1`** (the flag has
only ever been swept UP, on the strength of the ceiling that turned out not to exist), and
**`--lam-tgrad`** (match the target's temporal derivative — a fluctuation-shape loss, and the
temporal analogue of `--lam-grad`, the only auxiliary loss that has ever worked here).

### P3b — Out of distribution the TIME-MEAN is the failure, and it is reachable without training

`report_20260831_round13.md` §7. Decomposing §1.12's two identical-recipe controls: on `relo`
the **mean part** differs by 47 % between seeds (0.0764 vs 0.1126) while the **fluctuation
part** differs by 4 % — and the worse-mean seed is the *better* one there. In distribution both
agree to three decimals. **Out of distribution the fluctuation transfers and the mean field
does not.**

That reopens round 6's closed "blend the predicted time-mean toward the input window's mean":
the input window's mean is a **measurement**, so it cannot be wrong about the Reynolds number
the way a learned mean can. Round 6's null reproduces exactly on the interior control, and the
sign **flips** on both extrapolation sets. Gating β on how far the window's **own mean speed**
falls outside the training range makes it **free in distribution** — `rein` unchanged to four
decimals at every β — while gaining *more* out of it (ctrl2 `relo` −11.5 %). Safe by
construction: an in-range window gets exactly the current prediction, bit for bit.

**The dilution, stated up front** (§7.3): over all 82 trajectories the range is
0.0394–0.2921 m/s and Re 2968 sits only **2.7 %** outside it, against `relo`'s 12.0 %. The gate
fires at ≈ 0.18 where the experiment fired it at ≈ 0.8. Real, bounded below by zero, smaller
than §7.2 reads in isolation.

**Gates before shipping, in this order.** (a) Does it survive ensembling? An ensemble already
averages the mean-field variance down, so the single-model gain is an upper bound — `re_var`
runs the gated sweep on 4 members. (b) Only then, calibrate the ramp width against `relo`'s two
distances (Re 3750 at 12.0 % outside, Re 5025 at 5.1 %). Doing (b) first would fit a constant
to a gain that may not exist.

**A spin-off that does NOT work, recorded so it is not proposed again.** `kfold.py`'s five
models are a genuine bagged ensemble, so their disagreement would be real epistemic uncertainty
— unlike the shipped members, which all see the same data and whose spread round 10b measured
at only 0.347 correlation with |error| against σ's 0.551. But **with K folds each window is
excluded by exactly ONE model**, so there is no out-of-fold spread to measure: the quantity
needs several models that all held that window out. Bootstrap bagging gives ~37 % out-of-bag
per model, so a usable spread (≥ 4 estimates per window) needs **~10 models, ≈ 3 GPU-hours** —
and the band is measured closed three ways (`band_cv_v9.py`: pooling the fitting split buys
0.000). Not worth it unless something reopens the band.

### P4 — The Re-extrapolation split, built

`Track_1/re_split.py` builds three disjoint held-out sets — `val_relo` (Re ∈ {3750, 5025}, 378
windows), `val_rehi` (Re ∈ {25425, 26700}, 294) and `val_rein` (Re ∈ {12675, 19050}, 420, the
INTERIOR control) — leaving 56 trajectories to train on. Absolute scores on 56 trajectories
mean nothing; the **difference between an arm's `rein` and its `relo`/`rehi` result** is the
extrapolation effect with training size, recipe and seed held fixed.

The pre-registered pass condition is unchanged: **the split is only worth keeping if it
REORDERS something the AoA 15–20 proxy gets wrong.** If every arm ranks the same on the
interior control as on the extrapolation sets, it is one more local number that does not
predict the board, and it should be abandoned rather than trusted.

Arms: control ×2 seeds, `--norm self`, `--aug-scale 0.25`, `--wd 1e-2`, `--dropout 0.15`.
(The `--norm self` row is void — see §1.13. Both readings of it in the literature here are
now wrong: round 2's "−0.20 on the old split" used the U-Net-era `eval_wp.py`, which *did*
implement self-normalization but measured it on a split blind to what it is for; round 13's
"broken" used `predict_fno`, which did not implement it at all.)

### Explicitly NOT recommended

* **Another ensemble member, another seed, another architecture in the mix.** §1.3, board
  evidence.
* **Any Track-2 test-time adaptation.** Part 2, three measured families.
* **Re-tuning α.** Done this round and settled: 0.75 and 0.90 tie, so α is an
  instrument, not a lever. The band, capacity and the sigma head remain closed,
  several of them twice.
* **Resubmitting a package for a better `time_score` draw.** §1.7.

## Part 4 — Where things are

| what | where |
|---|---|
| Per-round detail, contracts, measured baselines | `CLAUDE.md` |
| All eight board rounds + weight fit + latency inversion | `Track_1/board_rounds.py` |
| Round-12 report and raw campaign rows | `Track_1/report/report_20260829_round12.md`, `Track_1/plan/campaign_20260829.md` |
| Shipped Track-1 package (11 memb / 4 inv) | `Track_1/submit_fno/` (`README.md` documents the build) |
| Scorer: official kits + reconstruction diff | `~/RealPDE/official/{t1_v9,t2_v6}/`, `Track_1/verify/diff_scoring_{official,real}.py` |
| GPU-free pre-submission gate | `Track_1/verify/cpu_smoke_submission.py` |
| **Track 2: everything** | `Track_2/README.md` — contract, the three closed adaptation families, the latency table |
| Track-2 submission + gate | `Track_2/submit/`, `Track_2/validate_t2.py`, `Track_2/build_t2.py` |
| Track-2 honest local eval | `Track_2/stream_eval.py` (real trajectories, official scoring), `Track_2/submit_ho/` |
| Ready-to-submit archives + recorded predictions | `../sub_candidates/README.md` |
| Round-13 plan and submission strategy | `Track_1/plan/plan_20260831.md` |
| Round-13 evidence, §1–§23 | `Track_1/report/report_20260831_round13.md` |
| Re-extrapolation split | `Track_1/re_split.py` |
| Out-of-fold local eval over all 82 trajectories | `Track_1/kfold.py` (`--split re` for Re bands) |
| Proxy-vs-board calibration | `Track_1/proxy_calib.py` |
| What a submission is WORTH as information | `Track_1/sub_value.py` |
| Normalizer-path gate | `Track_1/verify/selfnorm_path.py` |
| What the oscillatory head learned | `Track_1/osc_probe.py` |

**Machine.** OSC; data at `/users/PAS1187/qwu/RealPDE/`, env `realpde` (torch 2.2.2+cu121).
Training needs an A100 Slurm allocation — the login/OnDemand node has no GPU. The **12-core
cpuset binds before the GPU does**: pin training and evaluation to disjoint cores, never time
anything while training runs, and do not judge liveness by `nvidia-smi` (it reads 0 %).

**Standing method rules, each earned by a mistake:**

1. **Nothing is decided on a number computed from an all-data checkpoint.** They supply weights,
   never evidence. Three contamination artifacts, rounds 7/8/10.
2. **When testing a scaling parameter, refit everything downstream before comparing.** Fitting
   the band's `base` once at `p = 0` is what made round 6 declare the band closed.
3. **Record the predicted board score, all five subscores, before submitting.** It is the only
   reason any of §1.2 could be diagnosed. Predictions live in `../sub_candidates/README.md`
   next to the archive they belong to.
4. **Read the training curve before scoring an arm.** An arm that does not move the relevant
   training loss cannot move the held-out metric.
5. **Compare packages by board subscore DELTA, not by `final`.** §1.7: four of the five
   subscores are noise-free and `time_score` is worth ±0.07 `final` of eval-machine luck.
6. **A submission gate must keep what the model hands back and check it later.** Returning a
   reused staging buffer makes every stored prediction equal to the last one, scores rel-L2
   0.67 instead of 0.074, and passes every harness that copies out on the same line —
   `Track_2/validate_t2.py` caught exactly that on its first GPU run.
7. **Before believing an arm that changes the REPRESENTATION is bad, confirm the scoring path
   implements it.** §1.13: `--norm self` was closed on a number produced by a `predict_fno`
   that had no self-normalization branch at all. A tuning knob is safe — the same code runs
   either way — but a normalizer, an output parameterization or a target-space change is not.
   The alarm is free and general: **if an arm scores worse than the theoretical worst case for
   its own hypothesis, suspect the harness.**
8. **Replicates belong in the FIRST job, not the follow-up.** §1.11b: four single-run 2–3σ
   results regressed on replication in one round. Out of distribution the seed σ is 0.71
   `wfinal`, 58× the interior control's, so a one-run OOD number is unreadable by construction.
9. **A held-out list is DERIVED, never typed.** The (Re, AoA) grid is incomplete — there is no
   `3750_15`, and `5025` has all five angles — so a hand-written `--exclude` silently trains on
   a held-out trajectory. Call `re_split.excluded()` / `kfold.py --folds` and assert the count.
