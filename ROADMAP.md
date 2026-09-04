# RealPDE — project roadmap

Written 2026-08-31 after the round-12 board result; revised the same day after the Track-2
build; revised again 2026-09-01 after the round-13 board result, which changed the subscore
weights and turned the two boards into one instrument; revised 2026-09-02 after the round-14
PAIR, which answered where the hidden set lives and put dropout on the board; **revised
2026-09-04 after the round-16 pair, which took Track 1 to the account's best and closed the
first non-FNO family behind a k-fold gate.** This is the
handoff document: it records what the leaderboard actually says, what is closed, and what to do
next. Read it with `CLAUDE.md` (contracts, measurements, per-round detail),
`Track_1/board_rounds.py` (every number in Part 1 is reproducible from it) and
**`Track_1/plan/plan_20260904.md` (the CURRENT round's plan and recorded predictions)**.

**One-line state:** **Track 1 sits at 81.119780** (round 16 — the account's best on either
board, and the same 8 all-data **width-96** dropout FNOs that score **81.019213** as Track 2
round 3, the fourth cross-track identity check). Leader 81.76136 ⇒ **gap 0.641 ≈ 6.7 % of
error**, down from 0.742. Round 16 also spent a slot on the first non-FNO family, **U-FNO ×3,
and it MISSED by −0.45** (§1.20): the proxy that ranked it was wrong on `rel_l2` by 4.0 pp, so
every non-FNO family is now gated on the interior Re-band k-fold. **The plateau since
2026-08-08 is broken, and both breaks are regularization**: dropout (+0.200, T2 R2) and then
capacity at width 96 regularized harder (+0.166, T2 R3). Round 14 settled where the hidden set
lives — the gated mean-blend returned **all four error subscores bit-identical** to round
13's, so **it is INSIDE the training speed range** and every Re-*extrapolation* instrument
(`relo`, `rehi`, band 0) is retired. Round 15
settled how to attack it: **T2 R3's whole gain arrived through `sps` (+1.01) and `tke` while
`rel_l2` got WORSE** — capacity × dropout removes the per-element **tail**, not the rms, which
is what SPS's containment scoring pays for. Weights refit on **sixteen pooled rounds**: rel_l2
0.578 · tke 0.068 · mvpe 0.039 · time 0.083 · sps 0.231 (`w_time` *measured* at 0.0956 off the
R13/R14 pair).

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
| **R13 8 memb, 1 inv, α 0.60** | 08-31 | 910641 | 94.6798 | 77.9837 | 93.6133 | 93.1154 | 39.6893 | **80.64382** |

**Track 2**, first submission — the *identical* package, band and ALPHA as R13:

| round | date | id | rel_l2 | tke | mvpe | time | sps | **final** |
|---|---|---|---|---|---|---|---|---|
| T2 R1 8 memb, 1 inv, α 0.60 | 08-31 | 910642 | 94.6945 | 77.9422 | 93.3994 | 93.2878 | 39.7303 | **80.65294** |

Best other team on Track 1, read 2026-08-25: **`np-user` 81.76136.** Our best is R13's
80.64382.

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

### 1.4 The subscore weights, refitted twice — `tke` is now worth 0.097

**Nine** rounds against five weights with Σw = 1 and no intercept fits to a max residual of
**0.035**. This is a *tested* model, not an exact interpolation. (Round 10 rejected the
no-intercept form on four points; it holds on eight and on nine.)

| subscore | round-9 claim | eight-round | **nine-round** | admissible (9) |
|---|---|---|---|---|
| **rel_l2** | 0.266 | 0.545 | **0.582** | 0.319 – 0.672 |
| **tke** | 0.244 | 0.170 | **0.097** | 0.015 – 0.191 |
| **mvpe** | 0.241 | 0.000 | 0.011 | 0.000 – 0.218 |
| time | 0.069 | 0.081 | 0.087 | 0.040 – 0.160 |
| sps | 0.181 | 0.204 | **0.224** | 0.191 – 0.247 |

**Round 13 was submitted to measure this and it did.** ALPHA 0.90 → 0.60 is a near-pure
`rel_l2` ↔ `tke` trade — the family's dominant undetermined axis. The board returned
**rel +0.189 / tke −1.184**, worth **+0.019** at the eight-round weights and **+0.128** in
fact. The only fit is `tke` down and `rel_l2` up.

Consequences, all of which invalidate advice still sitting in `CLAUDE.md`'s older sections:

* **`rel_l2` is the most valuable subscore, at >2× the round-9 weight.** Round 10's
  "**stop optimizing rel-L2**, TKE and SPS carry the response" is **FALSIFIED**. It read the
  *curve* correctly (`100/(1+0.5e)` is flat near e ≈ 0.1) and paired it with a weight that was
  off by half.
* **`tke` is now the cheapest error subscore, below `time` and a sixth of `rel_l2`.** Rounds
  10–12 spent three campaigns on the variance branch against a price of 0.244 and then 0.170.
  Re-priced at 0.097, round 10's **+7.71 σ-oracle is worth +0.75 `final` for a model that
  knows the answer**, and its realistic first quarter +0.19. **The variance branch is retired.**
  (One correction to that: TKE is paid *twice*, since SPS's per-sample factor
  `a_i = 0.5g(rel)+0.3g(tke)+0.2g(mvpe)` carries a 0.3 TKE branch. Board TKE 0.565 → 0.50 is
  `tke_score` +0.19 **plus** +0.12 through SPS ⇒ ~1.6× the direct weight. Still small.)
* **`mvpe` is worth approximately nothing** (0.011; ≤0.218 at the family's extreme). Its 96.75
  ceiling and its regressions are not decision variables.
* Re-pricing round 9's measured exchange rate a third time: **a 10 % error cut is worth
  +0.948 `final`**, split **sps 52 % / rel_l2 39 % / tke 8 % / mvpe 1 %**. Closing the 1.118
  to the leader needs a **~11.8 % error reduction**.
* **More than half the return on accuracy now arrives through SPS**, which is per-element
  *containment*, not an rms. Attack the error TAIL, not the mean.

**Standing rule (earned three times now): when the weights move, re-decide every
weight-sensitive choice.** ALPHA was set to 0.90 in round 11 under weights that priced `tke` at
0.244; at 0.097 the optimum is **0.60**, and the two rounds shipped at 0.90 each gave away
≈0.075. See §1.16.

### 1.5 The board-time model drifts with the eval machine; an invocation costs ~0.55 ms

The old `4.447 + 0.324·inv + 0.431·local` fits its own four rounds to ±0.064 ms and misses
**R11 by −1.86**, **R12 by −0.90** and **R13 by −1.61 ms**. Refitting over all seven known
rounds gives `2.616 + 0.435·inv + 0.803·local` at an rms residual of **0.690 ms**, and the
residuals are ordered by DATE (Aug 4–6 all positive, Aug 10–31 all negative). The eval machine
gets faster; the intercept is not a constant.

**The one clean measurement** is R12 and R13, submitted the *same day*, which holds the machine
fixed: 4 → 1 invocation and 2.235 → 1.900 local ms moved board latency 5.811 → 3.985 ms, i.e.
**0.52 – 0.56 ms per invocation** — 60–70 % above the old slope.

At ~4 board ms the `100/(1+√(t/0.72896))` curve is steep: **1 board ms = −0.070 `final`**
(was −0.05), so **an invocation costs ≈0.036 `final`**. R13's 4 → 1 collapse was worth
**+0.114**, most of the round's measured gain.

**Budget for invocation count; do not predict `time_score` to better than ±1 point.**

### 1.6 `predict_board.py` predicts SUBSCORES well and `final` badly

| round | predicted | actual | miss |
|---|---|---|---|
| R11 | 80.603 | 80.60907 | +0.006 |
| R12 | 80.707 | 80.51584 | −0.191 |
| R13 | 80.491 | 80.64382 | +0.153 |

**Round 13 separates the two error sources for the first time, and they are not the same
thing.** Its *subscores* were predicted to rel_l2 **0.02**, tke 0.23, mvpe 0.20, sps 0.32 —
the error projection through `proxy_calib.py`'s inflation ratios worked. Priced at the weights
in force, those misses account for **+0.028** of a **+0.153** `final` miss. **The rest was the
weights.**

So: the *forward* model (local errors → board subscores) is in decent shape; the *pricing*
model is what has been wrong, twice, and each time in the direction of over-valuing `tke`.
Keep predicting before submitting — it is the only thing keeping this falsifiable — and quote
the predicted **subscores**, which are testable, alongside the `final`, which is a fit.

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
fix the board's distribution shift; **`--split re` does**, see §1.14. Cost is K trainings per
arm — screen on the AoA proxy, confirm here.

### 1.14 `kfold.py --split re` IS THE BOARD PROXY — the first local number that matches a LEVEL

Five contiguous Reynolds bands held out in turn, so every one of the 3383 out-of-fold windows
is a transfer test. The shipped recipe, single model:

| | Re-band k-fold | board R12 | |
|---|---|---|---|
| rel-L2 | 0.12613 | 0.116626 | 1.08× |
| TKE | 0.58788 | 0.526278 | 1.12× |
| **predicted `final`** | **80.562** | **80.516** | **+0.046** |

The AoA 15–20 proxy is *precise but not accurate* (§1.8, §19); the standard split sits 28 %
below the board on rel-L2. This lands within **0.05 `final`**. It is a **confirmation**
instrument at ~2.5 GPU-h per arm pair, not a screen — and the level agreement is one point,
partly two errors cancelling (single model vs an 11-member board ensemble, ~66 training
trajectories vs 82). Do not over-read it; do use it before shipping.

**Always read the per-band breakdown, never the pool alone.** Dropout's pooled −18.4 % rel-L2
is **entirely band 0** (Re 3750–7575, extrapolating below the training range, baseline 0.250);
bands 1–4 give −2.6 % and TKE +0.1 % (`report_20260831_round13.md` §28.2).

### 1.15 THE TEST FOR ANY ARM: does it remove SHARED error or INDEPENDENT error?

Round 13 measured both cases at matched ensemble size and they behave completely differently:

| arm | per model | at the ensemble | why |
|---|---|---|---|
| `ozdo` (osc + dropout 0.05) | +0.108 wfinal, **+3.84σ over 7 seeds**, replicated 4-for-4 | **−0.012**, TKE worse (§27) | removes error that is *independent* across members, which an 8-member average already removes |
| `--dropout 0.15` | `relo` −7 % single (§15) | ×3 vs ×3: `relo` **−10.2 %**, `rehi` −2.4 % (§17.2) | removes error the members **share** — they all train on the same in-range Re and extrapolate out of it the same way |

Three rounds of per-model deltas have now been diluted or reversed by ensembling (round 11's
members, round 12's U-Net, round 13's `ozdo`). **`ens_v9.py` is two minutes on a GPU. A
per-model screen that skips it is not evidence about the package.**

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

### 1.16 THE TWO TRACKS ARE ONE TEST SET AND ONE SCORING RULE — a slot is TWO experiments

Round 13 shipped the **identical** 8 members, band and ALPHA to both boards specifically to
find this out.

| subscore | T1 R13 | T2 R1 | diff | relative diff **in the error** |
|---|---|---|---|---|
| rel_l2 | 94.679785 | 94.694512 | +0.015 | **−0.29 %** |
| tke | 77.983720 | 77.942224 | −0.041 | **+0.24 %** |
| mvpe | 93.613336 | 93.399401 | −0.214 | **+3.59 %** |
| sps | 39.689258 | 39.730331 | +0.041 | **−0.10 %** |
| final | 80.643823 | 80.652944 | +0.009 | — |

1. **Same combination rule.** Track 1's nine-round weights applied to Track 2's subscores give
   **80.650** against a published **80.653** — a 0.003 residual, inside the fit's own 0.035.
   Track 2's `final_score` needs no separate calibration.
2. **Same (or near-identical) scored data.** Three of four error subscores agree to ≤0.3 % *of
   the error* — far too close for independently drawn test sets, and far too far for float32
   noise (`Track_2/verify_space.py` bounds the two code paths at 4e-6 physical, ~3e-5
   relative). Most likely one hidden set over slightly different window pools (a tail-padding
   or first-window difference of order one window per trajectory). `mvpe`'s **+3.59 %** is the
   exception and is what that looks like in the metric with the smallest effective sample size
   — 36 probe points per window, aggregated as a global ratio rather than a per-window mean.

**Every Track-1 instrument therefore transfers to Track 2** — the AoA proxy, the band, the
inflation ratios, the weights — carrying a small fixed per-track offset, of which only `mvpe`'s
is large, and `mvpe` is worth 0.011.

**The strategic consequence is the round's most useful output.** One submission per track per
day, two tracks scoring the same data with the same rule, and four error subscores that are
deterministic (§1.7) ⇒ **a day's pair is a controlled A/B with the test set, the scoring rule
and the eval machine all held fixed.** With ~26 days to 2026-09-27 the board is a **52-run
instrument, not a 26-run one.** Design slots in pairs.

*Caveat, and the next slot removes it:* the per-track offsets are measured from **one** model.
Whether they are a fixed property of the two window pools or vary with the model is untested —
so the round-14 pair also ships one model to both tracks, which turns the offsets into a
calibrated instrument at zero cost.

### 1.17 ALPHA is 0.60, and the SPS band is closed for the fourth time — against a MEASURED tail

Both settled by round 13's board result, both now anchored on data rather than assumption.

**ALPHA = 0.60.** `alpha_board.py` re-anchored on R13's *own* board errors at the alpha it
actually ran (rel 0.112383 / tke 0.564638 / mvpe 0.136448), scored with the nine-round weights:

| alpha | 0.00 | 0.30 | 0.45 | **0.60** | 0.75 | 0.90 |
|---|---|---|---|---|---|---|
| `final`, R13 twins | 80.106 | 80.308 | 80.367 | **80.394** | 80.384 | 80.319 |
| `final`, dropout twins | 80.045 | 80.268 | 80.337 | **80.380** | 80.388 | 80.345 |

Both families peak at 0.60 and the **minimax over the admissible weight family picks 0.60 on
both** — no other alpha has a positive worst case. (`mvpe` is constant to five decimals down
the column, the sanity check that composition only touches the fluctuation.) Rounds 11 and 12
shipped 0.90 and each gave away ≈0.075: it was chosen under weights pricing `tke` at 0.244.

**The band's inflation factor is 1.39 and that is exactly right.** The per-element error
stretch `f` — how much more often the board's elements miss their interval than the AoA 15-20
proxy's — was measured once, in round 7, on a 3-seed package at ALPHA 0.75 with a flat `.npy`
band, and used ever since. `Track_1/calib_band13.py` re-inverts it from round 13, where **two
boards returned two SPS values for one band**:

* **T1 ⇒ f = 1.360, T2 ⇒ f = 1.350.** They agree to 0.010. The tail is a property of the
  scoring, not of a draw.
* Board rel-L2 is only **1.075×** the proxy's while the element tail is **1.355×** ⇒ the tail
  is **1.26× heavier** than an rms-matched stretch. The round-7 diagnosis replicates exactly.
* Refitting the band at the calibrated stretch is worth **+0.000 sps**. Every fit between 1.39
  and 1.56 lands within 0.02. **The band is at its optimum; stop.**
* SPS's ceiling at board accuracy is `100·a = 70.63` and we score 39.69. The remaining 31
  points are **coverage**, i.e. accuracy — there is no band that captures them.

### 1.18 Round 14: the hidden set is inside the training range; dropout +0.200

Two submissions on 2026-09-02 14:57, both clean A/Bs against a package already on their own
board. Plan and full reasoning: `Track_1/plan/plan_20260902.md`.

| | id | rel_l2 | tke | mvpe | time | sps | **final** | vs baseline |
|---|---|---|---|---|---|---|---|---|
| **T1 R14** gated mean-blend β 0.15, R13's members and R13's bit-identical band | 912521 | 94.679785 | 77.983720 | 93.613336 | 92.978725 | 39.689258 | **80.630755** | R13: four error subscores **bit-identical**, `time` −0.137 |
| **T2 R2** 8 all-data dropout members (4×0.05, 4×0.15), α 0.60 | 912522 | 94.787655 | 77.667926 | 93.882222 | 93.633350 | 40.148056 | **80.852826** | T2 R1: **+0.200** |

**1. φ = 0. The hidden set is inside the training speed range.** The blend fires only for
windows whose mean streamwise speed leaves [0.0375, 0.2969] m/s, and all four error subscores
came back identical to six decimals — not one hidden window is outside. The "unseen Reynolds
numbers" are **interior**: interpolation, not extrapolation. `BLEND_BETA` returns to 0.0 in
every package (keep the code, drop the feature), and **`val_relo`, `val_rehi` and band 0 of the
Re-band k-fold are retired as decision instruments**, along with the `--sim-loss full` gain
that lived entirely inside them (§P4, round 14). The arm was designed so it could not lose;
what it bought was the answer, and **the null is the measurement**.

**2. Dropout transferred: +0.200, the largest real gain since round 10.** In error terms
against T2 R1: rel-L2 **−1.85 %**, TKE +1.60 %, MVPE −7.8 %, SPS **+1.05 %** (+0.418). Read
with arm 1 — *blend null, dropout gains* — the benefit is **not** Reynolds extrapolation but
in-range regularization. The **Re-band k-fold's interior bands 1–4** (rel −2.6 %) are the local
numbers closest to the board; the AoA 15-20 proxy had rel's **sign wrong** (+0.66 %). Every
instrument sits inside the **~4 % resolution limit**, so this is *"the closest instrument had
the sign right"*, not *"the instrument is confirmed"*.
→ **Track 1's best is still 80.644** only because dropout8 was never submitted there. It is the
next Track-1 slot, predicted 80.85 ± 0.07.

**3. `w_time` is measured, not fitted: 0.0956.** R13 and R14 differ in `time_score` alone
(−0.136707) and `final` moved −0.013068. First weight in this project read off a
single-subscore A/B, and it agrees with the fit (0.091, admissible 0.038–0.169).

**4. Twelve-round weights, POOLED across both tracks** (§1.16 licenses the pooling; the pooled
max residual 0.038 is no worse than the nine T1 rounds' 0.035, which is the evidence pooling
hid no second rule):

| | rel_l2 | tke | mvpe | time | sps |
|---|---|---|---|---|---|
| **twelve-round** | **0.568** | 0.085 | 0.029 | 0.091 | **0.226** |
| admissible | 0.294–0.679 | 0.010–0.187 | 0.000–0.237 | 0.038–0.169 | 0.192–0.248 |
| nine-round | 0.582 | 0.097 | 0.011 | 0.087 | 0.224 |

Nothing moved by more than its own interval; every round-13 conclusion stands. A 10 % error cut
is worth ~+0.95 `final`.

**5. Harness changes (plan §2), all committed.** `board_rounds.py` and `final_v9.py` carry all
twelve rounds and fit pooled across tracks; `leaderboard/lb.py` re-derives everything from them
and its calibration section now includes Track-2 rounds (level ratios over both tracks, delta
chains *within* a track, and a transition whose proxy did not move — R13→R14 — skipped). Pooled
ratios: rel **1.072**, tke **1.140**, mvpe **1.477**; **delta residual rms 4.01 %**, the
refitted resolution limit. **The decision instrument for a package candidate is now the Re-band
k-fold on INTERIOR bands 1–3 pooled** (Re 8850–22875, ≈2150 windows) — it matches the board's
situation and has 6× the AoA proxy's sample, but costs K trainings, so it *confirms* a
candidate; screening stays on the AoA 15-20 proxy + full split, always at ensemble size
(§1.15).

### 1.19 Round 15: capacity × dropout, and the gain is the TAIL

Two submissions on 2026-09-03. Predictions recorded before upload in
`../sub_candidates/README.md`; the round's work is `Track_1/report/report_20260902_round15.md`.

| | id | rel_l2 | tke | mvpe | time | sps | **final** | predicted |
|---|---|---|---|---|---|---|---|---|
| **T1 R15** T2R2's exact members, other board | 913176 | 94.780094 | 77.747643 | 94.122329 | 93.180177 | 40.140948 | **80.835124** | 80.853 |
| **T2 R3** 8 all-data w96 (4×do 0.30, 4×do 0.45), 400 ep, fp16, 1 inv | 913187 | 94.755824 | 78.174302 | 93.739707 | 92.508902 | 41.160630 | **81.019213** | 80.962 |

**1. The gain is the error TAIL, not the rms.** T2R3 against its T2R2 anchor, priced with the
fourteen-round weights:

| subscore | Δ | error Δ | × w | = final |
|---|---|---|---|---|
| rel_l2 | −0.032 | **+0.64 % (worse)** | 0.573 | −0.018 |
| tke | +0.506 | −2.90 % | 0.075 | +0.038 |
| mvpe | −0.143 | +2.49 % (worse) | 0.035 | −0.005 |
| time | −1.124 | 1.41 board ms | 0.088 | −0.099 |
| **sps** | **+1.013** | | 0.229 | **+0.232** |
| **total** | | | | **+0.148** (actual **+0.166**) |

The package is *worse* on rel-L2 and *better* on containment. That is round 9's corollary —
**tail reduction beats mean reduction** — demonstrated on the board for the first time rather
than derived from a re-pricing, and it says what to design for: robustness and regularization
over average-case accuracy. `rel_l2` alone would have called this a regression.

**2. The transfer beat the instrument.** The local 3-vs-3 delta was **−1.2 % of error**, below
the ~4 % transfer resolution, and it transferred anyway (+0.166); dropout's −2.6 % transferred
at +0.200. **The proxy is blunt, not conservative** — right sign, wrong size. Corroboration
that the residual is the *proxy's* and not a board draw: R14→R15 and T2R1→T2R2 are the **same
package change scored on the two boards**, and their residuals agree to 0.7 pp on all three
metrics (rel −2.66/−2.54, tke −1.29/−1.02, mvpe −9.71/−9.04).

**3. Latency: 1.41 board ms** (3.370 → 4.780) against **+0.93 predicted** by `lat_r15.py`. At
~4 ms one board ms is **−0.070 `final`**, so w96 handed back −0.099 of its +0.265 of accuracy.
fp16 packing was required by the 256 MB cap and is free (5.5e-4 rel-L2 of difference).

**4. Third cross-track identity check, and it passes.** R15 is T2R2's exact member set on the
other board: rel **+0.15 %**, tke **−0.46 %**, sps **−0.02 %** of the error, `mvpe` the
outlier again at −4.17 %. §1.16 now holds three times. **Track 1 gained +0.191 for nothing
new** — always submit a scored package on the other board.

**5. Weights, fourteen pooled rounds** (max residual 0.044): rel_l2 **0.573** · tke 0.075 ·
mvpe 0.035 · time 0.088 · sps **0.229**. Nothing moved beyond its own interval; `rel_l2 + sps`
have carried ~0.80 across the nine-, twelve- and fourteen-round fits. Calibration refit: level
ratios rel 1.068 / tke 1.136 / mvpe 1.454, delta residual rms **4.15 %**.

**6. What was skipped.** The interior Re-band k-fold confirmation (§1.18, `kfold.py --split re`,
bands 1–3) was not run for this package; it shipped on the 3-vs-3 alone. It worked. Run it
before the next capacity step anyway — the instrument was installed precisely so a package is
not chosen on the proxy's ±4 %.

### 1.20 Round 16: Track 1 to 81.120, and U-FNO missed by −0.45

Two submissions read 2026-09-04 01:45. Predictions recorded before upload in
`../sub_candidates/README.md`; the round's plan is `Track_1/plan/plan_20260904.md`.

| | id | rel_l2 | tke | mvpe | time | sps | **final** | predicted |
|---|---|---|---|---|---|---|---|---|
| **T1 R16** T2R3's exact members, other board | 914506 | 94.764049 | 78.433653 | 94.030389 | 92.699846 | 41.259088 | **81.119780** | 81.02 |
| **T2 R4** 3 all-data U-FNO (w96, do 0.30, 400 ep), fp16, 3 inv | 914508 | 94.661471 | 78.413683 | 93.727402 | 90.154359 | 41.600131 | **80.879200** | 81.33 |

**1. FOURTH cross-track identity check, and Track 1 is now the account's best.** R16 against
T2R3 in error terms: rel **−0.17 %**, tke **−1.52 %**, mvpe **−4.94 %**, sps +0.098 — the same
shape as the three earlier checks, `mvpe` the outlier. tke's 1.5 % is the largest tke offset of
the four; watch it, do not act on it. **+0.285 for nothing new**, which is §1.19 item 4 for the
second round running.

**2. U-FNO ×3 missed by −0.45, and most of the miss is ACCURACY.** Priced against T2R3 with the
fourteen-round weights: −0.143 predicted, −0.140 actual — **the weights held; the proxy did
not.** The 3-vs-3 said rel-L2 −4.4 % / sps +2.03; the board returned **rel +1.90 % (worse) /
sps +0.44**. That is a **4.0-pp residual on rel_l2**, the largest instrument miss on record,
and it landed on the first candidate ever to clear the ~4 % transfer resolution (§1.19 item 2).
Ensemble size (3 against 8) explains at most ~1 pp. Latency was the other half: **8.69 board
ms against 4.78** (−0.207), because the real cost is 5.457 ms/step, not the 1.28 the staging
README recorded — U-FNO is conv-heavy at batch 1, a CUDA graph removes launches and not FLOPs,
and `allow_tf32 = False` costs its 3×3 convs ~50 %. At T2R3's latency the package is +0.07.
→ **Working hypothesis: the local U-Net branch memorizes in distribution.** Held-out
trajectories share Reynolds numbers with training ones; the hidden set does not (φ = 0, but
*unseen interior* Re — §1.18). The **interior Re-band k-fold**, the only instrument that called
T2R3's shape correctly, was never run on U-FNO. **It is now the gate on every non-FNO family:
ViT, AFNO and the cross-family trio are all untrusted until they pass it**, since they were
ranked by the instrument that overrated U-FNO. `track1_r15_ufno3_a060.zip` is **WITHDRAWN** —
on Track 1 it costs 3 invocations ≈ −0.09 against +0.07 of accuracy.

**3. Weights, sixteen pooled rounds** (max residual **0.047**; R16 residual −0.041, T2R4
−0.007): rel_l2 **0.578** · tke 0.068 · mvpe 0.039 · time 0.083 · sps **0.231**. Nothing moved
beyond its own interval; `rel_l2 + sps` ≈ 0.81 for the fourth refit running.

### 1.21 Round 16 campaign: depth pays; the U-FNO miss is unexplained by every local instrument

`Track_1/plan/plan_20260904.md` §6 and `campaign_20260904.md`. Five-set 3-vs-3 (full-split
sps): w24 do05 42.79 → w96 do30 43.72 → **L8 43.98** at either dropout level — **depth is
the one FNO-family axis left that pays (+0.3 sps)**, and dropout 0.30 → 0.45 at depth 8 is
worth 0.00. Everything else tried at w96/do45 lost or tied: `--wd`, `--lam-tail` (CVaR),
800 epochs, input noise, **SAM** (monotone wrong in ρ); `--mode-drop` and `--aug-mask` sit at
the seed floor. **The U-FNO k-fold (interior bands) reproduced the proxy, not the board**, and
the boundary diagnostic refuted the FOV/zero-set explanation: U-FNO's gain is far-field wake,
monotone in Re, reversed at the top band (+1.3 %). Rule earned: **within the FNO family the
interior Re-band k-fold has called every board move; across families NO local instrument
predicts the board — a new family costs a slot, not a k-fold.** Next slot pair, staged with
recorded predictions: T2 `L8x4` do30 (81.035) / T1 `L8x4` do45 (81.188).

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

### P1 — DONE. Round 13 answered both of its questions. Round 14 is staged.

Round 13 shipped the same package to both tracks at ALPHA 0.60 to buy two answers, and got
them: **the tracks share a test set and a scoring rule** (§1.16) and **`tke` is worth 0.097,
not 0.170** (§1.4). It also became the best Track-1 score to date, 80.643823. The variance
branch is retired, ALPHA is settled at 0.60 (§1.17), and the band is closed against a measured
tail rather than an assumed one.

**Round 14, staged in `../sub_candidates/`: two mechanisms, one hypothesis, one slot.**

| | package | vs | what it isolates |
|---|---|---|---|
| **Track 1** | R13's members and **bit-identical band**, ALPHA 0.60, **+ the gated mean-blend** (β 0.15) | T1 R13 | inference-side robustness, zero training |
| **Track 2** | 8 all-data **dropout** members, 1 invocation, ALPHA 0.60 | T2 R1 | training-side robustness |

**The hypothesis is about the TEST SET, not about either arm: is the hidden set outside the
training Reynolds range?** That is the biggest open question about the board and it decides
where the remaining month goes. Both arms are aimed at it and they fail differently, so the
2×2 is readable:

| Track 1 blend | Track 2 dropout | conclusion |
|---|---|---|
| null | ≈0 or negative | **The board is in distribution.** Close the robustness line; spend the month on in-distribution accuracy. |
| gains | gains | **The board is Re-extrapolation.** Robustness is the main line; ship both together. |
| null | gains | Dropout's benefit is **not** Reynolds extrapolation. Surprising, and worth chasing. |
| gains | negative | The mean field is the reachable part; training-side regularization is the wrong tool. |

**The Track-1 arm cannot lose.** A window inside the training speed range gets the unblended
prediction *bit for bit* (`verify/blend_path.py` check A), so the package's floor is R13's four
error subscores exactly — **and that null is itself the measurement.** Predicted 80.67, bracket
**80.644 … 80.84**.

**The Track-2 arm can.** Dropout's sign flips between instruments: in distribution it is worse
(held-out AoA 15–20 twins at matched α: rel **+0.66 %**, tke +2.62 %, mvpe +1.25 %), out of it
better (`relo` −10.2 %, `rehi` −2.4 %, lowest Re k-fold band −36 %). Predicted **80.74
(+0.084)**, bracket **−0.096 … +0.807**; the downside is smaller than the board's own
`time_score` noise.

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

* **The oracle split, re-priced a THIRD time** — at the nine-round weights, ALPHA 0.60 and the
  measured element tail 1.355 (`oracle_split_v9.py`, R13 twins, AoA 15–20):

  | piece replaced by the target's | round-9 wts | eight-round | **nine-round** |
  |---|---|---|---|
  | σ (per-pixel fluctuation amplitude) | +7.71 | +5.96 | **+4.25** |
  | **shape** (phase-resolved fluctuation) | +1.95 | +2.96 | **+2.62** |
  | time-mean | +2.72 | +1.61 | **+1.86** |
  | everything | — | +21.04 | +20.35 |

  The **ordering σ > shape > mean is stable** across the last two weight vectors; only the
  round-9 vector ranked the time-mean above shape. What changed is the *mechanism*:

  | piece | rel_l2 | tke | mvpe | **sps** |
  |---|---|---|---|---|
  | σ | −2 % | 57 % | 0 | **45 %** |
  | shape | 44 % | 0 | 0 | **56 %** |
  | time-mean | 35 % | 0 | 3 % | **62 %** |

  **Between 45 % and 62 % of every oracle now arrives through SPS**, because SPS scores
  per-element containment and every component reduction shrinks per-element error. `tke` and
  `mvpe` together are worth almost nothing directly; they are worth something only through the
  error they carry into the interval test.

  So: σ is still the largest oracle, but **it is the one with five null arms behind it** —
  rounds 10 (five arms, +0.08 realized against +7.71), 11 (three readouts and two loss weights,
  all null, `lam_v` U-shaped with its minimum at the shipped 0.3) and 12 (FGN/CRPS failed its
  pre-registered screen). Its remaining value is also now half SPS, which is reachable only by
  reducing error, not by predicting amplitude better. **`shape` is the open one.**
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

**IT IS WORTH ZERO AT ENSEMBLE SCALE, and the thread is now closed** (§27, §30).

* **§27 — 4-vs-4 at matched member count: `ozdo` 78.318 against the baseline's 78.330, i.e.
  −0.012 `final`, with TKE worse.** The per-model +3.84σ, replicated four times with a tighter
  spread than the baseline's seven seeds, is *entirely gone* at four members. This is round
  10's dilution mechanism (`--var-mlp`'s 3.9 % fixed-field TKE cut → +0.08 ensembled).
* **§30 — `--tbasis 4` matches `ozdo` to −0.55 baseline σ.** Four fixed zero-mean temporal
  profiles shared by every window score identically to two window-adaptive sinusoids, so §23's
  striking unsupervised ω(U) correlation (+0.98…+0.995) is **decoration**. The operative
  constraint is *low-rank, zero-mean, smooth in time* — and that constraint's whole value is
  error the ensemble already removes.

**What the thread leaves behind is the sharper statement of the problem.** The fluctuation
carries 69 % of the squared error; a rank-4 zero-mean temporal branch removes ~1 % of rel-L2
per model; and the part it removes is exactly the part ensembling was already removing.
**Whatever remains in the fluctuation is error the members SHARE** — every model trained on
these 74 trajectories makes it — so by §1.15 it is the only part worth attacking, and neither
branch touches it. No arm in thirteen rounds has moved shared fluctuation error.

Do not spend more on this branch: not more modes, not a different band, not an input-derived
basis. The next arm on the fluctuation has to be aimed at *shared* error, and must be screened
with `ens_v9.py` at four members before anything else is measured.

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

### P3b — CLOSED-NULL on the board. The gated mean-blend measured where the hidden set is.

> **CLOSED 2026-09-02 (§1.18).** Round 14 shipped it and the board returned R13's four error
> subscores **bit-identical**: the gate never fired, φ = 0, the hidden set is inside the
> training speed range. The arm was built so that the null would itself be informative, and
> that is what it delivered. **`BLEND_BETA` → 0.0 in every package** — the code stays (with
> `verify/blend_path.py`), the feature does not. Everything below is the pre-board record; its
> `relo`/`rehi` numbers are correct and now measure a regime the board does not contain. Next
> steps: `Track_1/plan/plan_20260902.md`.

`report_20260831_round13.md` §7 and §33. Out of distribution the fluctuation transfers and the
mean field does not — decomposing two identical-recipe controls, the **mean part** of the error
differs by **47 %** between seeds on `relo` while the **fluctuation part** differs by 4 %. So
round 6's "blending the predicted time-mean toward the input window's mean makes MVPE
monotonically worse" was measured in the only regime where it is true. The input window's mean
is a **measurement**; it cannot be wrong about a Reynolds number it has never seen.

**Gate (a), at 4 members** — the gate that killed the oscillatory head:

| set | rel-L2 at β_eff ≈ 0.15 | mean part |
|---|---|---|
| `rein` | **0.0698 → 0.0698, unchanged at every β and every ramp** | 0.0366 |
| `relo` | 0.1080 → **0.1035** (−4.2 %) | 0.0785 → 0.0727 |
| `rehi` | 0.1137 → **0.1120** (−1.5 %) | 0.0638 → 0.0611 |

Single-model was −11.5 % on `relo`, so it dilutes and does not vanish: **the first arm in this
project to pass the §1.15 shared-versus-independent test.** **Gate (b) is a non-question** — the
optimum lies entirely in the *product* β × gate-strength, flat over β_eff 0.14–0.20, so the ramp
width is not an independent lever.

Shipped on Track 1 in round 14 at β 0.15, ramp 0.02, range (0.037523, 0.296857) measured over
the 3547 training **windows**. §33.1 records the calibration bug `verify/blend_path.py` caught
(the range had been measured over *trajectories* and applied per *window*, which fired the gate
at 0.56 strength on real validation data and falsified the whole bit-identity safety claim) —
and the general lesson: **a feature that nothing local exercises needs its own verifier, because
the strict gate will score the package as if the feature were absent and pass.**

**A spin-off that does NOT work, recorded so it is not proposed again.** `kfold.py`'s five
models are a genuine bagged ensemble, so their disagreement would be real epistemic uncertainty
— unlike the shipped members, which all see the same data and whose spread round 10b measured
at only 0.347 correlation with |error| against σ's 0.551. But **with K folds each window is
excluded by exactly ONE model**, so there is no out-of-fold spread to measure: the quantity
needs several models that all held that window out. Bootstrap bagging gives ~37 % out-of-bag
per model, so a usable spread (≥ 4 estimates per window) needs **~10 models, ≈ 3 GPU-hours** —
and the band is measured closed four ways now (§1.17). Not worth it unless something reopens it.

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
