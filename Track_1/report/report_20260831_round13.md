# Round 13 — measurements

Plan: `plan/plan_20260831.md`. Track-2 detail: `../Track_2/README.md`. This file carries what
was *measured*; the plan carries what was decided and why.

Every number below is `compare_v9.py --alpha 0.90 --aoa 15 20` — 126 windows of `val_unseen`,
the proxy that reads closest to the board (board rel-L2 is 1.078× it). The reference is **four
held-out baseline seeds**, not a borrowed sigma:

```
fno_vl_s1..s4    wfinal 81.748 +- 0.034     rel-L2 0.11303 +- 0.00040
```

Two bars, and they are different bars. **σ = 0.034 `wfinal` is whether an arm is distinguishable
from a re-seed.** **~2 % of error is whether it is distinguishable at the board**
(`proxy_calib.py`: R11 → R12 the proxy got the *sign* of the rel-L2 move wrong at 0.9 %).
Clearing the first is necessary; only the second is sufficient.

## 1. Weight decay — NULL, and the mechanism says the whole family is closed

`--wd` was added this round because `weight_decay=1e-5` had been hardcoded for twelve rounds
and never swept, and because every capacity result says the model memorizes 74 trajectories.

| arm | `wfinal` | Δ vs 4-seed mean | z | rel-L2 | Δ rel-L2 |
|---|---|---|---|---|---|
| baseline (4 seeds) | 81.748 ± 0.034 | — | — | 0.11303 | — |
| `--wd 1e-3` | 81.696 | −0.052 | −1.5 | 0.11340 | +0.33 % |
| `--wd 1e-2` | **81.784** | +0.037 | +1.1 | 0.11270 | **−0.29 %** |
| `--wd 3e-2` | 81.768 | +0.020 | +0.6 | 0.11290 | −0.11 % |
| `--wd 1e-1` | 81.609 | −0.139 | **−4.1** | 0.11430 | +1.13 % |

Nothing clears the seed spread except `1e-1`, which clears it **downward**. Every rel-L2 move
is ≤ 1.13 %, i.e. under the board's ~2 % transfer resolution even where it is real.

**Why, and this is the part worth keeping.** The trunk is
`GroupNorm(8, width)(spec(h) + point(h))` at every one of the four layers, and **GroupNorm is
scale-invariant in the weights that precede it** — multiply `spec` and `point` by c and the
layer's output is unchanged. So decaying them changes the parameterization, not the function.
Measured on the checkpoints (‖·‖₂ per module group):

| ckpt | spec | point | lift | head_field | head_logstd |
|---|---|---|---|---|---|
| baseline `1e-5` | 56.26 | 4.23 | 7.45 | 10.90 | 9.02 |
| `1e-2` | 52.94 | 3.97 | 6.98 | 10.71 | 8.79 |
| `1e-1` | **37.52** | **2.39** | **4.64** | **11.16** | 6.68 |

At `1e-1` the spectral weights lose **33 %** of their norm and the head *grows* to compensate —
the decay is being absorbed and re-emitted, which is exactly what a scale-invariant
parameterization does. What little functional effect survives arrives too late to help and
costs 4σ.

→ **Weight decay is closed, and so is any pure weight-norm penalty on this trunk.** The
regularizers that can still work are the ones that change the *function*, not its
parameterization: dropout, input/data augmentation, stochastic depth. Those are the remaining
arms (`--dropout`, `--aug-scale`).

**Replicated on the full 336-window split** (baselines `wfinal` 83.428 ± 0.051): `1e-3` −0.086,
`1e-2` +0.073, `3e-2` +0.017, `1e-1` +0.013 — every arm inside ±1.7σ, every rel-L2 move ≤ 0.2 %.

One detail worth keeping. **`1e-1` is neutral on the full split and −4.1σ on the AoA 15-20
proxy** (rel-L2 0.0850 vs the baseline's 0.0851 on the full set, against 0.1143 vs 0.1130 on
the hard subset). The one arm strong enough to escape the scale-invariance costs accuracy
*specifically in the hard regime* — which is the regime the board resembles. A regularizer that
averages out is not neutral; it is trading the easy trajectories against the hard ones, and the
score only pays for the hard ones.

## 2. ALPHA re-swept under the eight-round weights — 0.75 and 0.90 tie, and 0.60 is the probe

`alpha_board.py`, R13 held-out twin set, board-inflated errors (rel 1.074× / tke 1.163× /
mvpe 1.508×, element tail 1.39×), band refit at every row. `dmid/dlo/dhi` are
`final_v9.delta` against the shipped α = 0.90; `resolves` is `sub_value.py`'s ambiguity
reduction out of the 1.172 currently open.

| α | rel-L2 | TKE | rel | tke | sps | `final` | dmid | dlo | dhi | resolves |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.1097 | 0.6465 | 94.80 | 75.57 | 38.67 | 79.860 | −0.308 | −0.799 | +0.181 | 0.774 |
| 0.45 | 0.1108 | 0.6121 | 94.75 | 76.57 | 38.63 | 79.996 | −0.173 | −0.536 | +0.188 | 0.735 |
| **0.60** | 0.1123 | 0.5784 | 94.68 | 77.57 | 38.51 | 80.104 | **−0.065** | −0.295 | +0.164 | **0.637** |
| 0.75 | 0.1142 | 0.5485 | 94.60 | 78.48 | 38.30 | **80.170** | +0.004 | −0.101 | +0.107 | 0.320 |
| 0.90 (shipped) | 0.1166 | 0.5263 | 94.49 | 79.17 | 38.00 | 80.167 | 0 | 0 | 0 | 0.026 |
| 1.00 | 0.1184 | 0.5182 | 94.41 | 79.42 | 37.73 | 80.110 | −0.056 | −0.103 | −0.009 | 0.259 |

**0.75 and 0.90 are tied** (+0.004, against a family width of ±0.10). Round 11 adopted 0.90 on
three positive readings under the round-9 weights; the eight-round refit turns that into a
coin-flip. So ALPHA is not a lever — but it *is* the best instrument on the board.

**Shipping α = 0.60 as a deliberate probe.** It costs **−0.065 `final`** at the fitted weights,
which is **smaller than the board's own `time_score` noise** (±0.07 at 1σ, `ROADMAP.md` §1.7) —
i.e. below the resolution of a single round either way — and it resolves **0.637 of the 1.172**
open decision-ambiguity against **0.026** for shipping the staged package. The curve bends
there: 0.45 buys +0.098 more resolution for another −0.108, and 0.30 buys +0.137 more for
−0.243.

**Robustness check, because a probe that only resolves the decision it *is* would be
informative about itself and nothing else.** Drop the ALPHA entry from `PENDING`
(`sub_value.py --without ALPHA`) and re-rank against what remains — member count, `mvpe`
against `tke`, and `rel_l2` against `tke`: the 0.60 probe still resolves **0.321 of 0.721, 44 %**,
against 0.012 for the staged package. The information is about the weight family, which is what
decides where the remaining month goes, not about α. The cross-track test is unaffected — it needs both tracks to ship the *same* model, not
a particular α.

## 3. The oracle split re-priced — round 10's ordering is REVERSED

Same instrument (`oracle_split_v9.py`), same held-out twins, only the weights changed.

| piece replaced by the target's | round-9 weights | **eight-round weights** | change |
|---|---|---|---|
| σ (per-pixel fluctuation amplitude) | +7.71 | **+5.96** | −23 % |
| time-mean | +2.72 | **+1.61** | −41 % |
| **shape (phase-resolved fluctuation)** | +1.95 | **+2.96** | **+52 %** |
| everything | +21.04 | +20.87 | — |

**`shape` moves from last to second**, above the time-mean. Round 10's headline — *"the hardest
part of the problem is worth the least … stop optimizing rel-L2"* — was a correct reading of a
flat `100/(1+0.5e)` curve paired with a weight that was off by half, and it is now wrong on
both counts. `shape` is the only piece that moves rel-L2 (0.1086 → 0.0694, **−36 %**), and
rel-L2 carries 0.545 of the score.

Two consequences for the remaining month:

* **The time-mean's ceiling just dropped 41 %.** `--mean-head` is still worth running — it is
  in flight — but its oracle is +1.61, not +2.72.
* **σ is still the largest oracle and still the least reachable.** Rounds 11–12 measured the
  variance head at an information floor from three directions and turned a +7.71 oracle into
  **+0.08** in practice. Quote oracles to rank directions, never to forecast gains.

## 4. The "31 % of the fluctuation is predictable" ceiling is a CATEGORY ERROR

`ROADMAP.md` and `CLAUDE.md` both state that *"only ~31 % of fluctuation energy is coherently
predictable"*, citing `diag_spectral.py`'s energy-weighted Γ² = 0.308, and that sentence has
been used since round 3 to argue the fluctuation is out of reach. Read the definition:

```python
gamma(k) = |<F_P conj(F_Y)>| / sqrt(<|F_P|^2> <|F_Y|^2>)
```

`F_P` is **our prediction's** Fourier transform. Γ² is therefore **the fraction of the
fluctuation this model happens to get right in phase** — an achievement, not a bound. A perfect
predictor would score Γ² = 1 on the same instrument. Nothing in the project measures how much
of the fluctuation is *reachable*, and the 0.308 was measured on a **round-3 U-Net** that has
since been replaced twice over.

What the number does legitimately bound is what it was computed for: a post-hoc per-wavenumber
filter applied to *that* prediction, which is capped at +0.013. That conclusion stands. The
extrapolation to "the fluctuation is unreachable" does not.

Queued test, appended to the round-13 build job: re-measure Γ² on the current recipe, a single
member, a five-member ensemble and a different architecture family. **If Γ² moves at all, it is
a model property and no bound on the fluctuation exists** — which, with `shape` now the second
largest oracle at +2.96, reopens the largest genuinely-unexplored direction left.

## 5. The Re-extrapolation split — and the reason local validation has never transferred

`re_split.py`, six arms trained on the same 56 trajectories (26 held out), scored on the
interior control `rein` and the two extrapolation sets. Band fitted on `rein` only, so no arm
is judged by a band fitted to the set it is judged on.

| arm | `rein` rel-L2 | `rein` wfinal | `relo` rel-L2 | `relo` wfinal | `rehi` rel-L2 | `rehi` wfinal |
|---|---|---|---|---|---|---|
| control seed 1 | 0.0754 | 83.989 | 0.1147 | 82.349 | 0.1219 | 78.912 |
| control seed 2 | 0.0754 | **83.990** | **0.1404** | **80.930** | 0.1234 | 78.957 |
| `--norm self` | 0.1468 | 76.191 | **1.1436** | 48.589 | 0.2993 | 69.363 |
| `--aug-scale 0.25` | 0.0761 | 83.965 | 0.1247 | 81.206 | 0.1211 | 79.473 |
| `--wd 1e-2` | 0.0751 | 84.008 | 0.1301 | 80.887 | 0.1240 | 78.829 |
| **`--dropout 0.15`** | **0.0750** | **84.061** | **0.1067** | **82.857** | **0.1180** | **79.575** |

### 5.1 The headline is the two controls

**Two runs of the identical recipe are indistinguishable in-distribution and 22 % apart under
extrapolation.** `rein`: rel-L2 0.0754 vs 0.0754, `wfinal` 83.989 vs 83.990 — a gap of
**0.001**. `relo`: rel-L2 0.1147 vs 0.1404, `wfinal` 82.349 vs 80.930 — a gap of **1.42**.
They are genuinely different models (max |w₁ − w₂| = 1.19; `train2.py` sets no seed, so every
run is a fresh draw).

**Seed variance is ~1400× larger out of distribution than in it.** In-distribution fit does not
determine out-of-distribution behaviour, even for the same recipe on the same data.

That is, on its own, a candidate explanation for the central puzzle of §1.3 and §1.8 — why
local deltas stopped transferring after 2026-08-08 and why rounds 10–12 are indistinguishable
on the board. The hidden test set contains **unseen Reynolds numbers**, so board performance
carries a which-draw-did-we-get term that **no in-distribution local measurement can see**, and
every proxy this project has ever used is in-distribution in Re.

It also says something reassuring about the shipped package: an 8-member ensemble averages
exactly this variance down, which is consistent with the board's flatness across member counts
even though the accuracy gains were nil.

### 5.5 CORRECTION: the seed-variance ratio is ~58×, not ~1400×

§5.1 quoted "**seed variance is ~1400× larger out of distribution**" from **two** controls: a
0.001 `wfinal` gap on `rein` against 1.42 on `relo`. **A range-of-two is a poor σ estimator**,
and with the two extra controls `re_var` trained it collapses:

| set | `wfinal` (4 seeds) | rel-L2 |
|---|---|---|
| `rein` | 83.999 ± **0.0122** | 0.07537 ± 0.00005 |
| `relo` | 81.647 ± **0.7073** | 0.12498 ± 0.01092 |
| `rehi` | 78.936 ± **0.1408** | 0.12290 ± 0.00068 |

**σ ratio against the interior control: `relo` 58×, `rehi` 11.6×.** The 1400× figure is
withdrawn.

**The finding itself survives, and is what mattered:** identical-recipe seeds agree to
±0.012 `wfinal` in distribution and disagree by ±0.71 under low-Re extrapolation — a factor of
**58**, and 12× even for the milder high-Re shift. In-distribution fit still does not determine
out-of-distribution behaviour, board score still carries a which-draw term no in-distribution
proxy can see, and one run is still not a measurement out of distribution. Only the
multiplier changes.

Note also `relo` is much noisier than `rehi` (0.71 vs 0.14), i.e. **extrapolating BELOW the
training range is far less stable than above it** — and Re 2968, the known test condition,
is below.

### 5.2 `--dropout 0.15` is the first arm to clear anything

Best on **all three** sets: `rein` 84.061 (above both controls), `relo` 82.857 (above the better
control by 0.51, above the worse by 1.93), `rehi` 79.575 (above both by ~0.6). rel-L2 on `relo`
is **0.1067 against 0.1147/0.1404 — 7 % below the better control**, which is the first arm this
round to move error by more than the board's ~2 % transfer resolution.

Caveat, stated plainly: with a control spread of 1.42 `wfinal` on `relo`, a single dropout run
beating both controls is **suggestive, not established**. It needs replicate seeds, and it is
exactly what `kfold.py --paired` is for.

### 5.3 `--norm self` is not neutral — it is broken, and Strategy item 5 goes with it

`CLAUDE.md` Strategy §5 says *"Self-normalize before anything else … This is what makes Re
extrapolation work"*, and round 2 measured it as locally neutral (86.00 vs 86.07) and concluded
that neutrality was the pass condition because the local splits were blind to what it fixes.

The split built to see it says the opposite, and not by a little: **rel-L2 1.1436 on the low-Re
extrapolation set** against the control's 0.1147, and **−7.8 `wfinal` on the interior control
itself** against a same-split, same-data control. Whatever `--norm self` is doing, it is worse
everywhere and catastrophic exactly where it was supposed to help.

~~Two readings, and the measurement does not separate them: the *idea* is wrong, or the
*implementation* breaks when the window's own mean speed is small. Either way: do not ship
`--norm self`, and do not cite Strategy item 5 as untested — it is tested and negative.~~

### 5.3b RETRACTED — the instrument was broken, and §5.3 measured it rather than the arm

There was a third reading and it is the right one. **`_fno.predict_fno` hard-coded the global
`MEAN`/`STD`.** A `--norm self` checkpoint divides each window by its own mean streamwise speed
*before* standardizing, so it was fed an input it had never seen and its output was
denormalized with the wrong affine. Only the retired `eval_wp.py` ever read the `self_norm`
flag; `eval_v9.infer`, `compare_v9.py`, `re_split.py`, `kfold.py` and the package all go
through `predict_fno`, which did not.

The arithmetic identifies it beyond argument. Take a model whose self-normalized output is
**exactly the target** and score it through the old path:

| set | perfect self-norm model, scored through the old path | the arm actually measured |
|---|---|---|
| `rein` | 0.198 | 0.147 |
| `relo` | **2.224** | **1.144** |
| `rehi` | 0.349 | 0.299 |

**The arm beat a perfect model on all three sets, in the same order.** That is impossible for
a model and diagnostic of an instrument. The error scales with how far the window's own mean
speed sits from the global mean, which is precisely why the failure presented as a
Reynolds-extrapolation failure — `relo` is the low-Re set where the two affines diverge most.

**Nothing shipped is affected**: 81 of the 82 checkpoints on disk are `--norm real`, for which
the old constants are correct. `re_self.pth` is the only one, i.e. the bug touched exactly the
arm it invalidated.

Fixed in `_fno.py` (commit `ecd1aa6`), gated on the checkpoint's own flag and verified
bit-identical for `--norm real` by `Track_1/verify/selfnorm_path.py`. **`--norm self` has
never been measured with the current recipe.** Whether it helps is open; §5.3's verdict is
withdrawn, and Strategy item 5 returns to untested rather than closed. Round 2's "locally
neutral" stands as the only valid reading of it, and it was taken on an in-distribution split
that §1.12 says cannot see what self-normalization is for.

### 5.3c The same checkpoint, re-scored through the fixed path

No retraining — `re_self.pth` is the arm §5.3 condemned, run again after `ecd1aa6`:

| set | `re_self` rel-L2 | ctrl1 | `re_self` TKE | ctrl1 | `re_self` wfinal | ctrl1 |
|---|---|---|---|---|---|---|
| `rein` | 0.0766 | 0.0754 | 0.4696 | 0.4682 | 83.885 | 83.989 |
| `relo` | **0.1104** | 0.1147 | **0.7553** | 0.6031 | 81.596 | 82.345 |
| `rehi` | 0.1236 | 0.1219 | **0.5688** | 0.6035 | 79.170 | 78.911 |

It was never catastrophic — it is an ordinary model. rel-L2 goes 1.1436 → **0.1104** on `relo`
and 0.1468 → **0.0766** on `rein`, which confirms the diagnosis empirically rather than by
arithmetic alone.

What it now says, with the caveat that this is **one run against a `relo` seed σ of 0.71
wfinal** and therefore not yet readable: rel-L2 is **−3.7 % against the better control** exactly
on the low-Re extrapolation set (and −11.7 % against the 4-seed control mean of 0.12498, which
is inside the control spread), roughly neutral on the interior control, and **TKE is 25 %
worse on `relo`** — which is what sinks `wfinal` there despite the better field. Replicates are
queued (`jobs/selfnorm.sbatch`, two seeds per split from the start, per §1.11b).

### 5.4 The pre-registered pass condition, honestly

The condition was: *keep the split only if it REORDERS something the AoA 15–20 proxy gets
wrong.* Read strictly, **it does not** — `--dropout` wins on the interior control and on both
extrapolation sets, and no arm's ranking flips between them beyond the control spread.

Keeping the instrument anyway, for a reason that is measured rather than hoped for: it is the
only instrument in the project that can see §5.1, and §5.1 changes how every other local number
should be read. Recording this explicitly so it is not mistaken for the condition having passed
— it did not, and the reason for keeping it is a different one, found by running it.

## 6. Γ² is a MODEL property — measured, and the fluctuation ceiling is gone

`diag_spectral.py` (rewritten this round to take an ensemble and either split; the old `--ckpt`
took a single path, which is why nobody had ever run it on the shipped configuration), all on
the same 336 `val_unseen` windows at α = 0.90:

| prediction | energy-weighted Γ² | fluctuation energy captured | fluc rel-L2 | total rel-L2 |
|---|---|---|---|---|
| round-3 U-Net (the quoted 0.308) | 0.308 | — | — | — |
| round-12 U-Net | 0.3288 | 17.6 % | 0.0723 | 0.0893 |
| current recipe, 1 member | 0.3641 | 20.3 % | 0.0707 | 0.0851 |
| **current recipe, 5-member ensemble** | **0.3865** | **25.0 %** | 0.0684 | 0.0814 |

**Γ² rises monotonically with model quality — U-Net < single FNO < ensemble — and is 25 %
above the number that has been quoted as a ceiling since round 3.** A quantity that goes up
when you improve the model is not a bound on the problem. The claim is dead, measured, and the
only surviving reading of Γ² is the one it was computed for: a post-hoc per-wavenumber filter
on a *given* prediction is capped at +0.013.

Two more numbers worth keeping from the same table:

* **The fluctuation carries 70.6 % of the squared error** at the shipped ensemble
  ((0.0684/0.0814)²), the time-mean 29.4 % — consistent with `ROADMAP.md` §1.9's 29 %, arrived
  at independently.
* **Ensembling improves fluctuation PHASE, not amplitude.** The `gain` column is ~0.95–1.0
  across the energy-carrying wavenumbers (composition rescales amplitude by construction), so
  the 20.3 % → 25.0 % gain in captured energy is coherence. That is a hint about where the
  remaining fluctuation error lives: decorrelation, which matches round 12's phase/advection
  null — the error is *not* a registration offset, it is a loss of correlation.

Directional consequence: `shape` is now the second-largest oracle (+2.96, §3), the fluctuation
holds 70 % of the squared error, and nothing bounds how much of it is reachable. That is the
largest genuinely open direction in the project, and it is the same problem as generalization —
§5.1 shows out-of-distribution behaviour is barely constrained by in-distribution fit, and the
fluctuation is where that error lives.

## 7. Out of distribution the FLUCTUATION transfers and the TIME-MEAN does not

`diag_spectral.py --set` (extended this round to read `re_split`'s three sets), on the two
identical-recipe controls:

| set | ckpt | rel-L2 | **mean part** | fluc part |
|---|---|---|---|---|
| `rein` | ctrl1 | 0.0754 | 0.0382 | 0.0648 |
| `rein` | ctrl2 | 0.0754 | 0.0384 | 0.0647 |
| `relo` | ctrl1 | 0.1147 | **0.0764** | 0.0850 |
| `relo` | ctrl2 | 0.1404 | **0.1126** | 0.0815 |

**The whole of §5.1's 1400× seed variance is in the time-mean.** The mean part differs by
**47 %** between two runs of the same recipe; the fluctuation part differs by **4 %**, and the
worse-mean seed is the *better* one there. In distribution the two agree to three decimals on
both parts.

That is a mechanism, and it makes a round-6 null testable again.

### 7.1 Blending toward the input window's own mean — the sign flips out of distribution

Round 6 closed this: blending the predicted time-mean toward the input window's mean made MVPE
monotonically worse (0.0698 → 0.1240), because "the model's mean beats persistence by a lot".
True in distribution — the model's mean error on the interior control is 0.0382. But the input
window's time-mean is a **measurement**, not a prediction, so it cannot be wrong about the
Reynolds number in the way a learned mean can.

`mean_blend_re.py`, no training, post-hoc on a finished prediction:

| | β = 0 | 0.1 | 0.2 | **0.3** | 0.5 |
|---|---|---|---|---|---|
| ctrl1 `rein` rel-L2 | **0.0754** | 0.0754 | 0.0759 | 0.0770 | 0.0804 |
| ctrl1 `relo` rel-L2 | 0.1147 | **0.1131** | 0.1133 | 0.1153 | 0.1244 |
| ctrl2 `relo` rel-L2 | 0.1404 | 0.1337 | 0.1289 | **0.1262** | 0.1277 |
| ctrl2 `relo` mean part | 0.1126 | 0.1045 | 0.0987 | **0.0957** | 0.0981 |
| ctrl1 `rehi` rel-L2 | 0.1219 | 0.1208 | 0.1201 | **0.1198** | 0.1207 |
| ctrl2 `rehi` rel-L2 | 0.1234 | 0.1222 | 0.1214 | **0.1211** | 0.1218 |

* **Round 6's null reproduces exactly** on the interior control — β > 0 is monotonically worse
  there, and that is the control this experiment needs.
* **The sign flips on both extrapolation sets**, at both ends of the Re range.
* **It is a variance reducer, not an accuracy gain.** At β = 0.3 the bad-mean seed improves
  10.1 % and the good one 0.5 %; the two seeds' `relo` rel-L2 go from 0.1147/0.1404 (22 %
  apart) to 0.1153/0.1262 (**9 % apart**). It halves the out-of-distribution spread for ~1.5 %
  in distribution.

**The caveat that decides it.** The blend works by damping the variance of the predicted mean
field across draws, and **an 8-member ensemble already averages that variance down** — so the
single-model gain above is an *upper bound* on what the shipped package would see. The
measurement is queued in `jobs/re_var.sbatch`: the same sweep on a 4-member ensemble. If the
gain survives ensembling it is the largest no-training change available; if it does not, the
ensemble was already doing this and the finding is diagnostic rather than actionable.

### 7.2 Gating the blend on the window's own mean speed makes it free in distribution

The blend is a correction for a mean field the model had no data to learn, so applying it
inside the training range is pure cost and applying it outside is the whole gain. `predict()`
gets neither Re nor AoA, but the window's **own mean speed** is measurable and is proportional
to Re (`mean_u/Re = 1.02-1.23e-5`, constant to ±10 % across the range). So scale β per window by
how far its mean speed falls outside the training range — `mean_blend_re.py --gate`, training
range measured at **0.0659–0.2726 m/s** for `re_split`'s 56.

| | β = 0 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|
| ctrl1 `rein` | 0.0754 | **0.0754** | **0.0754** | **0.0754** |
| ctrl2 `rein` | 0.0754 | **0.0754** | **0.0754** | **0.0754** |
| ctrl1 `relo` | 0.1147 | **0.1122** | 0.1135 | 0.1262 |
| ctrl2 `relo` | 0.1404 | 0.1287 | **0.1242** | 0.1274 |
| ctrl1 `rehi` | 0.1219 | 0.1210 | 0.1206 | **0.1199** |
| ctrl2 `rehi` | 0.1234 | 0.1224 | 0.1219 | **0.1211** |

**Strictly better than the flat blend on both axes.** In distribution it is *unchanged to four
decimals at every β* — the gate is zero there, so round 6's cost disappears rather than being
traded away. Out of distribution it gains **more** than the flat version (ctrl2 `relo` −11.5 %
against −10.1 %) because β can be pushed higher where it is actually needed. Both seeds improve
at β = 0.3–0.5 and the `relo` seed spread falls from 22.4 % to 9.4 %.

**It is safe by construction**, which is the property that makes it shippable: an in-range
window gets exactly the current prediction, bit for bit. The bet is bounded below at zero and
only pays where the model is outside its training range — which is precisely where the hidden
test set is (the Track-2 example trajectory is Re 2968, mean speed ≈ 0.033 against a full-data
training minimum of ≈ 0.041, i.e. *further* outside than `relo`).

Two things must still hold before it ships, and one is queued: **(a)** it has to survive
ensembling (§7.1's caveat — `re_var` runs the gated sweep on a 4-member ensemble), and **(b)**
the shipped range must be recomputed over **all 82** trajectories, not `re_split`'s 56, because
the shipped members train on all of them.

### 7.3 How much of that reaches the board — the honest dilution

Recomputing the range over **all 82** trajectories, as the shipped members see it:
**0.039416 – 0.292070 m/s**. Placing the known test conditions on it:

| condition | mean speed | fraction outside the range | gate at `ref = 0.15` |
|---|---|---|---|
| Re 2968 (the T2 example trajectory) | 0.0326 | **2.7 %** | 0.18 |
| Re 3750 / 5025 | 0.041 / 0.055 | 0 % | 0 |
| Re 26700 | 0.294 | 0.7 % | 0.04 |

**The experiment fired the gate far harder than the real test set will.** `relo`'s Re 3750 and
5025 sit **12.0 %** outside `re_split`'s 56-trajectory range (gate ≈ 0.8); Re 2968 sits 2.7 %
outside the full-data range (gate ≈ 0.18). So the −11.5 % measured at an effective β ≈ 0.4
becomes an effective β ≈ 0.09 at the real condition, which on the same curve is worth roughly
−5 % of that trajectory's rel-L2 — and then only for whatever fraction of the hidden set sits
below Re 3750. The board's own rel-L2 being just 1.078× the in-distribution AoA proxy is
independent evidence that it is not heavily extrapolated.

`ref` is an unjustified constant: the ramp width was set at 0.15 with nothing measured at 2.7 %
outside to calibrate it. `relo` contains two distances (Re 3750 at 12.0 %, Re 5025 at 5.1 %),
so a per-trajectory breakdown would calibrate how fast the mean-field error actually grows with
distance — worth doing, but **only after §7.1's ensemble question is settled**, since if the
gain does not survive ensembling none of this matters.

Net: a bounded-below bet with a smaller upside than §7.2 suggests in isolation. Not a
substitute for the score the project actually needs.

## 8. `--mean-head` is null in distribution — which the re-pricing predicted

A decoupled time-mean readout (`head_mean`, its own 128-wide MLP on the shared trunk;
`head_field` then emits a forced zero-mean fluctuation so the two cannot trade). AoA 15-20
proxy, against the same four held-out baseline seeds:

| arm | `wfinal` | Δ | z | rel-L2 | Δ rel-L2 |
|---|---|---|---|---|---|
| baseline (4 seeds) | 81.748 ± 0.034 | — | — | 0.11303 | — |
| `--mean-head` seed 1 | 81.678 | −0.070 | −2.1 | 0.1133 | +0.24 % |
| `--mean-head` seed 2 | 81.746 | −0.002 | −0.0 | 0.1129 | −0.11 % |
| `--mean-head --w-mean 3` | 81.740 | −0.008 | −0.2 | 0.1132 | +0.15 % |

**Null.** The two seeds straddle the baseline mean and every rel-L2 move is under 0.25 %,
an order of magnitude below the board's ~2 % transfer resolution.

This is the outcome §3's re-pricing predicted: the time-mean's oracle fell from +2.72 to
**+1.61**, and the model's in-distribution mean error is already 0.0382 on the interior control
against a full-field 0.0754 — the component is not where the in-distribution error is.

**It does not close the arm, because the question it was built for is a different one.** §7
shows the time-mean is where out-of-distribution transfer *fails* — mean part 47 % apart
between identical-recipe seeds on `relo`, fluctuation part 4 % apart. A dedicated mean readout
might reduce that spread even though it does nothing in distribution, and the AoA proxy is
structurally blind to it (both val sets sit inside the training Re range). Two `--mean-head`
arms are queued on the Re split in `jobs/re_var.sbatch`, compared on the **mean/fluctuation
split across seeds** rather than on `wfinal` — a smaller spread in the mean part is the result
that would matter, not a better mean on any one seed.

Note also, for the oscillatory head (§P3): `compose()` rescales the predicted fluctuation's
per-pixel std to the variance head's σ, so **an output head's amplitude contribution is
renormalized away at inference and only its phase content survives.** That is the right
division of labour for `--osc-modes` — it is judged on exactly what it is for, and it cannot
damage TKE.

### 8.1 The mechanism table — the head works, and the component is at a floor

The arm targets the time-mean specifically, so the number to read is the mean/fluctuation
split, not `wfinal` (full `val_unseen`, 336 windows):

| ckpt | mean part | fluc part | rel-L2 |
|---|---|---|---|
| baseline s1 | 0.0470 | 0.0710 | 0.0851 |
| baseline s2 | 0.0478 | 0.0713 | 0.0858 |
| `--mean-head` s1 | 0.0464 | 0.0710 | 0.0848 |
| `--mean-head` s2 | **0.0461** | 0.0709 | 0.0846 |
| `--mean-head --w-mean 3` | **0.0461** | 0.0713 | 0.0849 |

**The head does exactly what it was built to do.** The mean part falls **2.5 %**
(0.04740 → 0.04620) consistently across all three arms, and the fluctuation part is
**untouched** (0.07115 → 0.07107, +0.0 %) — the decoupling is clean, which is the property
`--var-mlp` and `--lift-mlp` also had when they paid.

**And the gain is diluted by arithmetic, not by a trade.** The mean part carries
(0.0474/0.08545)² = **30.8 %** of the squared error, so a 2.5 % cut in it is **0.6 %** of
rel-L2 — which is what the table shows and what the ~2 % bar rejects.

**`--w-mean 3` reaching the identical 0.0461 is the informative part.** Tripling the loss
weight on the component buys nothing beyond giving it its own readout, so the in-distribution
time-mean is at an information floor around 0.046 per member (0.0438 for the shipped ensemble),
not at an optimization or capacity one. Together with §1.9's withdrawal of the bogus 0.0531
"sampling floor", the in-distribution time-mean is now closed on its own terms: **it is 30 % of
the squared error, it responds to a dedicated head by 2.5 %, and it does not respond to
anything else tried.**

The out-of-distribution question (§7) is untouched by this and remains open.

### 8.2 Pre-registered expectation for `--w-mean` DOWN (result not yet in)

Recorded before the arm reports, because §8.1 makes a falsifiable prediction and the project's
method is to write it down first.

`--w-mean` scales the time-mean half of the field loss; it has only ever been moved **up** (3
and 8 in round 3, both null), on the strength of a fluctuation ceiling that §4 shows was never
one. `tgrad.sbatch` now runs 0.3 and 0.1.

§8.1 measured the in-distribution time-mean as **information-limited**, not gradient-limited:
a dedicated readout moves it 2.5 % and tripling its loss weight on top moves it **0.0**. If
that reading is right, then:

* **down-weighting the mean should cost the mean part little** — it is not short of gradient;
* **the fluctuation part should improve**, since that is where the gradient goes;
* and because the fluctuation carries **69 %** of the squared error against the mean's 31 %,
  a small fluctuation gain outweighs a small mean loss.

The falsifier is clean: **if the mean part degrades sharply at `--w-mean 0.3`, the mean was
gradient-limited after all** and §8.1's "information floor" reading is wrong — which would
also mean `--w-mean 3`'s null had a different cause than stated. Read the mean/fluctuation
split at the end of the job, not `wfinal`.

## 9. The k-fold baseline — and AoA does NOT govern TKE

`kfold.py --score`, five stratified folds, every one of the 82 trajectories scored by a model
that never saw it. **3 383 out-of-fold windows** against the 336 every previous local number
came from.

```
rel-L2 0.08416 ± 0.00337   TKE 0.46481 ± 0.00819   MVPE 0.07156
scores rel 95.962  tke 81.142  mvpe 96.545  time 96.394  sps 47.854   (sps LOFO over folds)
```

| AoA | n | rel-L2 | TKE |
|---|---|---|---|
| 0 | 701 | 0.05195 | 0.44445 |
| 5 | 569 | 0.05324 | 0.46908 |
| 10 | 756 | 0.08547 | 0.46861 |
| 15 | 714 | 0.10612 | 0.45706 |
| 20 | 643 | 0.12073 | 0.48738 |

### 9.1 "Error is governed by AoA and by almost nothing else" is true of `rel_l2` only

Across the five angles **rel-L2 spans 2.32×** (0.0520 → 0.1207, monotone from 5° up) while
**TKE spans 1.10×** (0.4445 → 0.4874, and not monotone). `CLAUDE.md`'s round-6 statement was
made from per-trajectory numbers with one or two trajectories per cell; with **569–756 windows
per angle** it needs splitting in two: **AoA governs `rel_l2`. It does not govern TKE.**

Two corrections to the round-6 table fall out of the better statistics. AoA 0 and 5 are
**equal** (0.0520 vs 0.0532) where round 6 had 0° much worse than 5° (0.057 vs 0.044); and AoA
20 is now clearly the worst angle (0.1207) rather than tied with 15°.

### 9.2 What that does to the AoA 15-20 proxy, quantitatively

On the same out-of-fold population, restricting to 15° and 20° **inflates rel-L2 by 1.343× and
TKE by 1.014×.** The proxy is a `rel_l2` stress test and is essentially neutral on TKE.

Combine with `proxy_calib.py`'s measured board/proxy ratios (rel **1.078**, tke **1.150**):

* the board's `rel_l2` elevation over the full split is **mostly explained** by an AoA-hard
  mix — 1.343 of the total 1.448 is the mix, leaving 1.078 unexplained;
* the board's TKE elevation is **not explained by AoA at all** — the proxy contributes 1.014
  and the board still sits 1.150 above it.

So whatever makes the hidden set's TKE harder than ours is **not** its angle-of-attack
distribution. That is a genuinely open question and the AoA proxy cannot see it.

### 9.3 The two instruments agree to five decimals

k-fold on AoA 15-20 pools to rel-L2 **0.11304**; the standard split's four-seed baseline on its
126-window AoA subset is **0.11303**. Different training sets (65–66 trajectories against 74),
different windows (1 357 against 126), same answer. The 8-trajectory proxy is **unbiased for
the AoA 15-20 population** — it is noisy, not skewed, which is exactly the failure mode
`kfold.py` was built to fix and the one that `--paired` removes.

**Read the ±0.00337 correctly.** It is the trajectory-bootstrap SE of the *level* (4.0 % of
rel-L2), dominated by between-trajectory spread — 0.052 at AoA 0 against 0.121 at 20. It is
**not** the resolution for comparing two arms: on identical windows the trajectory effect
cancels, which is what `--paired` measures and what `kfold_do15` will report.

## 10. `--lam-tgrad` is a mild regularizer with a plateau at 0.3–1.0

Matching the target's temporal derivative — the time analogue of `--lam-grad`, the only
auxiliary loss that has ever worked here. AoA 15-20 proxy against the four held-out seeds
(81.748 ± 0.034, rel-L2 0.11303):

| λ_tgrad | training `field` | rel-L2 | Δ rel-L2 | `wfinal` | Δ | z |
|---|---|---|---|---|---|---|
| 0 (4 seeds) | — | 0.11303 | — | 81.748 ± 0.034 | — | — |
| **0.3** | 0.06600 | 0.1122 | −0.7 % | **81.806** | +0.058 | +1.7 |
| **1.0** | 0.06744 | **0.1118** | **−1.1 %** | **81.804** | +0.056 | +1.6 |
| 3.0 | 0.06994 | 0.1135 | +0.4 % | 81.618 | −0.130 | −3.8 |

**An inverted U with a plateau at 0.3–1.0 and a sharp fall at 3.0, while the training loss
degrades monotonically throughout.** Worse training fit with better held-out score is a
regularizer's signature, and the shape mirrors `--lam-grad`'s own shallow plateau (1–3, worse
at 10) with the optimum about 3× lower.

**Real, and not shippable on its own.** +1.7σ clears the seed spread, but −1.1 % of rel-L2 is
under the board's ~2 % transfer resolution (`proxy_calib.py`). It earns a place in a
combination and a `kfold --paired` confirmation, not a package by itself.

### 10.1 Correcting a mechanism I proposed before the result

Watching the training losses degrade, I argued that `--lam-tgrad` *should* fail: `--lam-grad`
works because it constrains what a mode-truncated spectral operator does **between** its
retained modes, and there is no analogous truncation in time — `head_field` emits all 20 frames
directly with no temporal basis. That predicted a null-to-negative arm and concluded the time
axis needs a basis before a temporal constraint can act on it.

**The prediction is wrong: it works, mildly, with no temporal basis at all.** The training-loss
degradation I read as "competing for capacity without adding signal" was regularization, which
is the same signature. The `--osc-modes` arm's motivation stands on its own — the shedding
period is 60–75 frames against a 20-frame window, and coherence at k = 2–6 is 0.55–0.79 — and
must not be propped up with the argument above, which is now falsified.

### 10.2 Replicated on the full split, compressed

Same three checkpoints, 336 windows instead of 126 (baselines 83.428 ± 0.050, rel-L2 0.08513):

| λ | `wfinal` | Δ | z | rel-L2 | Δ rel-L2 |
|---|---|---|---|---|---|
| 0.3 | 83.480 | +0.053 | +1.0 | 0.0843 | −1.0 % |
| 1.0 | 83.455 | +0.028 | +0.6 | 0.0849 | −0.3 % |
| 3.0 | 83.387 | −0.041 | −0.8 | 0.0850 | −0.1 % |

**The inverted U replicates in shape on both splits, compressed on the full one** — peak at
0.3, declining monotonically to 3.0. Two independent window sets agreeing on the shape is the
evidence that this is a real effect rather than the AoA subset's noise; the size, roughly
+0.05 `wfinal` and ~1 % of rel-L2, is the honest number and it is under the board's ~2 %
transfer resolution.

**Verdict: `--lam-tgrad 0.3` is a small real regularizer.** Combination candidate and a
`kfold --paired` subject; not a package on its own.

## 11. The oscillatory head is being USED — a zero-init diagnostic

Before any score: `head_osc`'s final conv is **zero-initialized**, so its weight norm after
training measures whether the optimizer found the oscillatory path worth using at all. On
`fno_osc2` (per-pixel ω, J = 2):

| tensor | ‖w‖ |
|---|---|
| `head_field.2.weight` (the 40-frame readout) | 7.300 |
| `head_osc.2.weight` (the (p, q, ω) readout) | **1.715** |

**Ratio 0.235 from a standing start.** The path is not inert — the head carries roughly a
quarter of the field readout's output weight norm, which is a lot for 4.7k parameters against
the field head's ~5k output weights, and it means gradient descent preferred spending there
rather than leaving it at zero.

Training `field` came out at 0.06581, mid-range among the baseline seeds (0.0617–0.0667), so
the head is used **without** improving the training fit — the same regularizer signature as
§10. Whether that converts to held-out score, and specifically whether it moves **coherence at
k = 2–6**, is what the rest of the job measures. Recording the diagnostic separately because it
is the one that distinguishes "the idea failed" from "the idea was never exercised", and those
need different follow-ups.

### 11.1 The global-ω head recovered the shedding frequency, unsupervised

`--osc-global` pools the trunk and emits J frequencies for the whole field. Nothing in the loss
mentions a frequency — the head is trained only through the 20-frame field error, from a
zero-initialized start — so what it converges to is a test of whether the parameterization
means what it was designed to mean. Replaying the trunk on 48 `val_unseen` windows and reading
ω off `fno_oscg1`:

```
band            omega 0.0314 - 0.6283 rad/frame   (periods 200 down to 10 frames)
learned omega   mean 0.0974   std 0.0001   min 0.0973   max 0.0976
=> period       64.5 frames    (range 64.4 - 64.6)
at the low rail 0%   at the high rail 0%   inside the 60-75 frame shedding band 100%
```

**The head placed itself at a 64.5-frame period — the middle of the documented 60–75 frame
vortex-shedding band — from a 20-frame input window, i.e. a third of a cycle, with no
supervision on frequency and no prior beyond the band edges.** It is nowhere near either rail,
so the band is not what is holding it there.

**And the σ of 0.0001 is the second finding: the frequency it learned is a CONSTANT.** It does
not vary window to window. That is physically consistent rather than a failure — the documented
shedding period spans only 60–75 frames (1.25×) while Re spans 7×, which means the PIV
acquisition rate is scaled with flow speed and the period *in frames* really is near-constant.
A per-window ω has nothing to adapt to.

Two consequences, both testable in this same job:

* the gain (if any) comes from **imposing an oscillatory basis at the right frequency**, not
  from adapting ω — so a **fixed, non-learned ω would do as well**, and the frequency MLP is
  removable weight;
* **harmonics should matter more than adaptivity**, which is exactly what `oscg1` vs `oscg2`
  vs `oscg4` measures.

It also predicts `osc2` (per-pixel ω) should not beat `oscg2`: per-pixel freedom buys nothing
when the true frequency is one number, and costs the memorization risk the global variant was
introduced to avoid.

## 12. `--w-mean` DOWN fails, and the reallocation buys the fluctuation *nothing*

The arm §8.2 pre-registered. AoA 15-20 proxy against the four held-out seeds
(81.748 ± 0.034, rel-L2 0.11303):

| arm | rel-L2 | Δ rel-L2 | `wfinal` | Δ | z |
|---|---|---|---|---|---|
| `--w-mean 0.3` | 0.1135 | +0.4 % | 81.688 | −0.060 | −1.8 |
| `--w-mean 0.1` | 0.1154 | +2.1 % | 81.489 | −0.259 | **−7.6** |

Monotonically worse. And the decomposition on the full split, which is what §8.2 said to read:

| | rel-L2 | mean part | fluc part | fluctuation energy captured |
|---|---|---|---|---|
| baseline `vl_s1` | 0.0851 | 0.0470 | **0.0707** | 20.3 % |
| `--w-mean 0.3` | 0.0859 | 0.0482 (+2.6 %) | **0.0707** | 20.0 % |

**Down-weighting the mean 3.3× made the mean 2.6 % worse and moved the fluctuation by
0.0000.** Not a small gain — none. Captured fluctuation energy went the wrong way, 20.3 % →
20.0 %.

### 12.1 What that does to §8.1 and §8.2, stated plainly

**§8.2's prediction was wrong**, on both halves: the mean is not nearly free to down-weight,
and the fluctuation does not absorb the gradient.

**§8.1's "information floor" was over-stated and needs narrowing.** Its evidence was that
*more* gradient (`--w-mean 3`) leaves the mean unchanged — consistent with an information
floor, but equally consistent with simply sitting at an optimum. The new arm distinguishes
them: less gradient makes the mean **worse**. A quantity that cannot be improved by more
gradient and degrades under less is **at its optimum**, which is weaker and more accurate than
"information-limited".

What survives from §8.1, and is the useful half: **the mean responds to architecture but not to
loss weight** — a dedicated readout moves it −2.5 %, tripling its loss weight moves it 0.0.

### 12.2 The loss-reweighting family is now closed in both directions

| change | mean part | fluc part |
|---|---|---|
| `--w-mean 3` (more gradient to the mean) | 0.0 | — |
| `--w-mean 0.3` (less) | **+2.6 %** | **0.0000** |
| `--mean-head` (architecture, not weight) | **−2.5 %** | 0.0 |

`w_mean = 1` sits at an optimum, and **neither component responds to how the loss is split
between them.** In particular the fluctuation — 69 % of the squared error — is **not
gradient-starved**: giving it 3.3× more of the loss changes its error in the fourth decimal.

→ **Both remaining levers, on both components, are architectural rather than
objective-weighting.** That is a measured argument for `--osc-modes` being the right *shape* of
attack on the fluctuation, replacing the one §10.1 retracted.

## 13. The oscillatory head: the value is in the CONSTRAINT, monotonically

AoA 15-20 proxy against the four held-out seeds (81.748 ± 0.034, rel-L2 0.11303). Ordered by
how tightly the head is constrained, tightest first:

| arm | ω | modes | `wfinal` | Δ | z | Δ rel-L2 |
|---|---|---|---|---|---|---|
| **`oscg1`** | global | 1 | **81.818** | +0.070 | **+2.08** | **−0.64 %** |
| `oscg2` | global | 2 | 81.798 | +0.050 | +1.49 | −0.55 % |
| `oscg4` | global | 4 | 81.768 | +0.020 | +0.60 | −0.11 % |
| `osc2` | per-pixel | 2 | 81.743 | −0.005 | −0.13 | −0.20 % |

**The ranking is monotone in constrainedness and nothing else.** One global oscillator beats
two, two beat four, and all three beat a per-pixel frequency. That is round 8's capacity result
appearing again at the output head: **the head's value is the constraint it imposes, not the
capacity it adds.**

### 13.1 Both halves of §11.1's prediction, scored

* **"Per-pixel ω should not beat global"** — correct, and strongly: `osc2` is null (−0.13σ)
  against `oscg1`'s +2.08σ. A per-pixel frequency is freedom to fit each pixel's 20-frame trace
  independently, which is the memorization mode the global variant was introduced to avoid.
* **"Harmonics should matter more than adaptivity"** — **wrong.** Adaptivity is worthless, but
  so are harmonics: J = 1 wins and each added mode erodes the gain monotonically. A single
  oscillator at the shedding frequency captures whatever is there.

The frequency MLP is also now removable weight, as §11.1 anticipated: with J = 1 and a ω that
came out **constant** at 64.5 frames across every window, a fixed non-learned ω should do the
same work with fewer parameters. Worth one arm if the direction is pursued.

### 13.2 A limitation of the zero-init usage diagnostic

§11 measured `head_osc`/`head_field` weight norms to check the path was exercised: `oscg2`
0.292 > `osc2` 0.235 > `oscg1` 0.165. **The best-scoring arm has the LOWEST usage.** So that
diagnostic answers "was the path exercised" and says nothing about whether it was exercised
well — it distinguishes "the idea failed" from "the idea was never tried", which is what it was
introduced for, and must not be read as a quality signal.

**Standing: +2.08σ, −0.64 % rel-L2.** Real against seed noise, under the board's ~2 % transfer
resolution — the same standing as `--lam-tgrad 0.3`, and a combination candidate rather than a
package.

### 13.3 RETRACTION: the constrainedness ordering does not survive the second split

§13's table was one split. On the full 336-window `val_unseen` (baselines 83.428 ± 0.050) the
two extremes **swap**:

| arm | Δ rel-L2 (full) | z (full) | z (AoA proxy) | ranking stable? |
|---|---|---|---|---|
| `osc2` per-pixel J=2 | **−1.20 %** | **+2.46** | −0.13 | **no** |
| `oscg1` global J=1 | −0.50 % | −0.79 | **+2.08** | **no** |
| `oscg2` global J=2 | −0.85 % | +1.62 | +1.49 | yes |
| `oscg4` global J=4 | −0.62 % | +0.91 | +0.60 | yes |

**"Monotone in constrainedness" is withdrawn.** It described one window set; the arms at both
ends of that ordering are the two that invert. `oscg1`'s collapse on the full split is entirely
TKE (0.5031 against a baseline mean of 0.4912, **+2.4 %**) while its rel-L2 is fine — so the
`wfinal` instability is TKE noise, not a real reordering.

**What IS robust: every one of the four arms improves rel-L2 on both splits**, by 0.5–1.2 %
(full) and 0.11–0.64 % (AoA). The head does something real and consistent; which variant is
"best" is not resolvable at this resolution and needs `kfold --paired`.

### 13.4 The mechanism check falsifies the premise — the gain is in the TIME-MEAN

This is what the pre-registered coherence read was for. Full split, decomposed:

| model | rel-L2 | **mean part** | **fluc part** | fluc energy captured | Γ² |
|---|---|---|---|---|---|
| baseline `vl_s1` | 0.0851 | 0.0470 | 0.0707 | 20.3 % | 0.3641 |
| `oscg2` | 0.0844 | **0.0458** (−2.6 %) | 0.0706 (−0.1 %) | 20.4 % | 0.3651 |
| `osc2` | 0.0841 | **0.0456** (−3.0 %) | 0.0704 (−0.4 %) | 21.3 % | 0.3693 |

**The head was built to fix fluctuation PHASE at the shedding scale. It did not.** Coherence
moved +0.3 % to +1.4 % in Γ², the fluctuation error moved −0.1 % to −0.4 %, and **the entire
rel-L2 gain is the time-mean, −2.6 % to −3.0 %.**

**Why, and it is arithmetic.** The learned period is 64.5 frames and the output window is 20,
so the oscillator spans **0.31 of a cycle** — and `mean_t cos(ωt)` over `t = 0…19` is
**+0.5110**, nowhere near zero. The oscillatory term therefore carries a large non-zero
time-mean by construction, and gradient descent used it as **a second mean-field pathway**, not
as an oscillation.

That explains everything else: why J = 1 / 2 / 4 and per-pixel barely differ (all four are
extra mean capacity), why the `wfinal` ranking is unstable (the real effect is a small
consistent mean gain plus TKE noise), and why the size matches `--mean-head`'s −2.5 % almost
exactly — **it is the same mechanism, arrived at accidentally.**

**The decisive follow-up is one line: force the osc term to be zero-mean over the 20 output
frames** (subtract its own time-mean before adding it). If the gain vanishes, the head is a
disguised mean-head and the phase idea is untested; if it survives, the phase claim earns a
second look. Until then §11.1's unsupervised-frequency result stands as a nice property of the
parameterization that **is not what produced the score**.

Prediction it also makes: `--osc-modes` and `--mean-head` should **not** be additive.

### 13.5 `oscz2` doubles as the capacity control both mean-arms needed

`--mean-head` and `--osc-modes` now both improve the time-mean by ~2.5–3 %, by two apparently
different routes. Without a control, "a dedicated mean readout helps" and "**any** extra
output-side capacity helps" are indistinguishable — and the second reading is live, because
round 10b's rule is that capacity at the boundaries pays.

`oscz2` is that control for free: it has **exactly the same parameters** as `oscg2` with only
the mean pathway removed. So

* `oscz2` still helps ⇒ the gain was output-side **capacity**, and `--mean-head`'s −2.5 % is
  probably the same thing rather than the decomposition working;
* `oscz2` is null ⇒ the gain was specifically the **mean pathway**, capacity per se buys
  nothing, and `--mean-head`'s clean mean/fluctuation separation stands.

Either way it also answers the phase question. One arm, three readings.

## 14. `--aug-scale` and `--dropout`, and then the pattern across all six arms

Completing the regularization family (AoA proxy, baselines 81.748 ± 0.034, rel-L2 0.11303):

| arm | `wfinal` | Δ | z | Δ rel-L2 |
|---|---|---|---|---|
| `--aug-scale 0.15` | 81.665 | −0.082 | −2.4 | +0.77 % |
| `--aug-scale 0.35` | 81.628 | −0.120 | −3.5 | +1.13 % |
| `--dropout 0.05` | 81.802 | +0.055 | +1.6 | −0.55 % |
| `--dropout 0.15` | 81.791 | +0.043 | +1.3 | −0.64 % |

`--aug-scale` degrades monotonically. That is the expected sign and is **not** evidence against
it: freestream rescaling is a first-order Reynolds shift, aimed at *out-of-range* Re, and in
distribution it can only add noise to conditions the model already sees. Its verdict belongs to
`re_split`, where it scored `relo` 81.206 against controls 82.349/80.930 — inside the control
spread, undecided rather than refuted.

### 14.1 THE PATTERN: six unrelated mechanisms, one indistinguishable gain

| arm | mechanism | `wfinal` | z |
|---|---|---|---|
| `oscg1` | oscillatory output basis, J = 1 | 81.818 | +2.08 |
| `--lam-tgrad 0.3` | temporal-derivative loss | 81.806 | +1.73 |
| `--lam-tgrad 1.0` | temporal-derivative loss | 81.804 | +1.67 |
| `--dropout 0.05` | channel dropout in the trunk | 81.802 | +1.61 |
| `oscg2` | oscillatory output basis, J = 2 | 81.798 | +1.49 |
| `--dropout 0.15` | channel dropout in the trunk | 81.791 | +1.28 |

**Six structurally unrelated changes — an auxiliary loss, an output-basis head at two mode
counts, and stochastic regularization at two rates — span 0.027 `wfinal`, i.e. 0.8σ. They are
indistinguishable from one another**, all sit at +1.3…+2.1σ, and all improve rel-L2 by
0.55–0.64 %.

That is far better explained by **"the model is mildly under-regularized, and any mild
regularizer recovers ~0.6 % of rel-L2"** than by six different mechanisms each happening to be
worth the same amount. And §13.4 already caught one of them red-handed: the oscillatory head's
gain was **not** the phase mechanism it was built for.

Three consequences, and they are the round's real output:

* **Expect combinations NOT to add.** If all six tap the same slack, stacking them recovers it
  once. The `mhosc` arm in `jobs/oscz.sbatch` tests exactly this for one pair.
* **None of them is shippable, and combining them probably will not change that.** 0.6 % of
  rel-L2 against a ~2 % transfer bar, six times over, is still 0.6 %.
* **The fluctuation is untouched by all of it.** Every decomposition this round puts the
  fluctuation part at 0.0704–0.0707 — baseline, `--w-mean 0.3`, `oscg2`, `osc2` alike — while
  69 % of the squared error lives there. **Nothing tried in round 13 moved the thing that
  carries most of the error.**

The honest read of the round: the regularization family is measured and small, the loss-weight
family is closed, and the fluctuation remains the open problem it was at the start — now with
one fewer wrong ceiling in front of it (§4, §6) and one arm whose stated mechanism was
disproved (§13.4).

### 14.2 CORRECTION to §14.1: dropout is not one of the six

§14.1 read "six unrelated mechanisms, one indistinguishable gain" **off the AoA proxy alone**.
That is the same single-split error §13.3 retracted an hour earlier, repeated. The full split
separates them:

| arm | z (full) | z (AoA) | ΔTKE (full) | ΔTKE (AoA) | sign stable? |
|---|---|---|---|---|---|
| `--aug-scale 0.15` | +1.15 | −2.43 | −1.39 % | +0.55 % | **no** |
| `--aug-scale 0.35` | +0.69 | −3.52 | −1.78 % | +0.96 % | **no** |
| **`--dropout 0.05`** | **+3.76** | +1.61 | **−3.18 %** | −0.30 % | yes |
| **`--dropout 0.15`** | **+3.84** | +1.28 | **−3.10 %** | +0.64 % | yes |
| `oscg2` | +1.62 | +1.49 | −0.27 % | +0.19 % | yes |
| `--lam-tgrad 0.3` | +1.05 | +1.73 | −0.15 % | −0.11 % | yes |

**What survives of §14.1:** the **rel-L2** gains really are uniform, 0.55–0.64 % across every
mechanism, which still reads as mild under-regularization recoverable by anything.

**What does not:** dropout is **qualitatively different**. On 336 windows it is **+3.8σ**, more
than twice any other arm, and it gets there through **TKE −3.1 %** where all four
non-dropout arms move TKE by under 0.3 %. Priced at the eight-round weights that TKE move alone
is worth ≈ **+0.08 `final`**.

**The caveat that keeps it from being a result.** Dropout's TKE gain is **absent on the AoA
proxy** (−0.30 %, +0.64 %), so it lives in the AoA 0/5/10 windows. The board looks like an
AoA-hard mix — and §9.2 measured that the proxy inflates TKE by only 1.014× while the board
sits 1.150× above it for reasons **not explained by AoA**. So nothing local tells us which
regime the board's TKE resembles, and this gain may or may not transfer.

`--aug-scale` flips sign between splits, which with §13.3's `osc2`/`oscg1` makes **four of
tonight's arms sign-unstable across nested splits**. At this effect size neither split is a
reliable instrument on its own; that is what `kfold --paired` is for, and its per-AoA cells
(n ≈ 700) are what would settle where dropout's TKE gain lives.

## 15. `--dropout 0.15` clears the bar out of distribution, with complete seed separation

`re_var`: three dropout seeds against four controls, same 56 training trajectories, band fitted
on `rein` only.

| set | controls (4) `wfinal` | dropout (3) `wfinal` | z | controls rel-L2 | dropout rel-L2 | Δ |
|---|---|---|---|---|---|---|
| `rein` (interior control) | 83.999 ± 0.012 | 84.061 ± 0.006 | **+5.06** | 0.07537 | 0.07490 | −0.6 % |
| `relo` (low-Re) | 81.647 ± 0.707 | 82.993 ± 0.278 | +1.90 | 0.12498 | **0.10720** | **−14.2 %** |
| `rehi` (high-Re) | 78.936 ± 0.141 | 79.545 ± 0.089 | **+4.33** | 0.12290 | **0.11783** | **−4.1 %** |

**Every dropout seed beats every control seed on rel-L2, on both extrapolation sets:**

```
relo: dropout worst 0.1141  <  control best 0.1147     SEPARATED
rehi: dropout worst 0.1180  <  control best 0.1219     SEPARATED
```

Complete separation of 3 against 4 is p = 1/35 = 0.029 one-sided **per set**, and the two sets
agree independently. This is not a seed draw.

**And the magnitudes clear the ~2 % transfer bar decisively** — −14.2 % on `relo` and −4.1 % on
`rehi`, against every other arm of the round at 0.5–1.2 %.

### 15.1 Why this changes the round's conclusion

§14.1 grouped dropout with five other arms as "one indistinguishable gain"; §14.2 already
separated it in distribution on TKE. This separates it much further, on the axis that matters:
**in distribution dropout looks like the others; out of distribution it is in a different
class.** The hidden test set has unseen Reynolds numbers, so out of distribution is the regime
the board occupies.

**Dropout also shrinks the seed spread it is measured against** — `relo` σ 0.707 → 0.278,
`rehi` 0.141 → 0.089. It attacks §5.1's instability directly, not just the mean level.

### 15.2 What it is NOT yet, and the two things standing between it and a package

* **The training set differs.** These models see **56** trajectories; the shipped members see
  all 82. Regularization generally matters more with less data, so the 56-trajectory effect is
  an **upper bound** on what the shipped recipe would get. The 74-trajectory evidence is
  weaker but same-signed: +3.8σ on the full split, TKE −3.1 % (§14.2).
* **The board is less extrapolated than `relo`.** Re 2968 sits ~2.7 % outside the full-data
  mean-speed range where `relo` sits ~12 % outside its 56-trajectory one, so expect dilution —
  the honest bracket is nearer `rehi`'s **−4 %** than `relo`'s −14 %. At the eight-round
  weights a 4 % rel-L2 cut is worth roughly **+0.2 `final`**, which would still be the largest
  single gain since round 10.

**Next, in order:** `kfold_do15` (queued) gives the paired confirmation on all 82 trajectories
with a trajectory bootstrap — the instrument built for exactly this question. If it holds,
rebuilding the package needs 8 dropout-trained all-data members, ~2.5 GPU-hours.

## 16. `--mean-head` does NOT fix out-of-distribution mean transfer — the pre-registered null

§8/§11a left this open: `--mean-head` is null in distribution, but §7 showed the time-mean is
where extrapolation fails, so a dedicated readout **might** reduce that spread even while doing
nothing in-distribution. `re_var` asked it directly, on `relo`:

| | mean part | fluc part |
|---|---|---|
| controls (4) | 0.0909 ± 0.0159 | 0.0837 ± 0.0036 |
| `--mean-head` (2) | **0.0958 ± 0.0261** | 0.0817 |

**No.** The mean part is **+0.31σ worse**, not better, and its seed spread is not reduced — the
worst `--mean-head` seed (0.1142) is as bad as the worst control (0.1125). The hypothesis is
falsified on the only split that could test it.

`--mean-head` is now **closed on both axes**: null in distribution (§8.1 — the mechanism works,
the component is at its optimum) and null out of it (here). It stays in the tree, defaulted
off, with this attached.

Which sharpens §15 by elimination: of everything tried this round, **only dropout improves
out-of-distribution transfer** — and it does so without going through the mean decomposition
that §7 made look like the obvious route.

## 17. Ensembling, dropout and the blend all attack the same thing — and dropout wins

`re_var` measured all three against each other on the Re split. The common target is §5.1's
out-of-distribution seed draw, which §7 localized to the time-mean.

### 17.1 Ensembling IS variance averaging out of distribution

Control ensembles against the single-model mean (rel-L2):

| members | `rein` | `relo` | `rehi` |
|---|---|---|---|
| ×2 | −2.0 % | −6.4 % | −2.4 % |
| ×3 | −2.7 % | −9.3 % | −3.3 % |
| ×4 | −3.1 % | **−10.2 %** | −3.7 % |

**Ensembling is worth ~3× more out of distribution than in it** (`wfinal` +1.16 on `relo`
against +0.38 on `rein`). That is a mechanism for the board's flatness across member counts
while local accuracy gains were nil: **the members were buying robustness, not accuracy**, and
the local proxies are in-distribution where robustness is worth least.

### 17.2 Dropout survives ensembling; the blend is dominated by it

At matched 3 members, dropout against control:

| set | dropout ×3 | control ×3 | Δ |
|---|---|---|---|
| `rein` | 0.0737 | 0.0733 | +0.5 % |
| `relo` | **0.1018** | 0.1134 | **−10.2 %** |
| `rehi` | **0.1160** | 0.1189 | −2.4 % |

**Dropout stacks on ensembling rather than duplicating it** — the caveat §7.1 raised against
the blend, which dropout passes.

The blend's own test (gated, 4-member control ensemble): `relo` 0.1122 → **0.1079** at β = 0.3
(**−3.8 %**), `rehi` 0.1184 → 0.1169 (−1.3 %), `rein` **unchanged at every β**. So the blend
**does** survive ensembling, and gating is what makes it worth having — ungated on the same
ensemble it manages only −1.4 % on `relo` and *costs* +2.7 % in distribution.

But it is dominated. On a single model the blend is worth −4.7 % on a mid-quality control and
only **−1.2 % on a dropout model** — it repairs bad draws, and dropout has already removed
them. **They are substitutes, not complements.**

### 17.3 Decision

**Ship dropout, not the blend.** Dropout is 2.7× larger at matched ensemble size, fixes the
fluctuation as well as the mean, needs no inference-time machinery, and subsumes most of what
the blend was for. The blend stays in the tree (`mean_blend_re.py`, `--gate`) as a
zero-training fallback if dropout fails its paired k-fold.

Expected board effect, stated conservatively: the board sits at a **milder** extrapolation than
either test set (Re 2968 is ~2.7 % outside the full-data mean-speed range against `relo`'s
~12 %), so the honest bracket for dropout is nearer `rehi`'s −2.4…−4 % than `relo`'s −10…−14 %.
At the eight-round weights that is roughly **+0.1 … +0.2 `final`**.

## 18. `--osc-zeromean`: §13.4's conclusion was premature, and the fluctuation finally moved

The decisive test. Full split, all one-member, α = 0.90:

| model | rel-L2 | mean part | fluc part | Γ² | Δmean | Δfluc | ΔΓ² |
|---|---|---|---|---|---|---|---|
| baseline `vl_s1` | 0.0851 | 0.0470 | 0.0707 | 0.3641 | — | — | — |
| `oscg2` (mean path **on**) | 0.0844 | 0.0458 | 0.0706 | 0.3651 | −2.6 % | −0.1 % | +0.3 % |
| **`oscz2`** (mean path **off**) | **0.0841** | 0.0460 | **0.0701** | **0.3723** | −2.1 % | **−0.8 %** | **+2.3 %** |
| `mh1` (`--mean-head`) | 0.0848 | 0.0464 | 0.0707 | 0.3658 | −1.3 % | +0.0 % | +0.5 % |
| `mhosc` (both) | 0.0839 | 0.0455 | 0.0703 | 0.3675 | −3.2 % | −0.6 % | +0.9 % |

Scores: `oscz2` **83.585 (+3.1σ)** on the full split against `oscg2`'s +1.6σ, and +1.55σ vs
+1.49σ on the AoA proxy. **Removing the mean pathway did not cost the gain — it improved it.**

### 18.1 Correcting §13.4

§13.4 concluded the head was "a mean-head in disguise" and the phase premise "falsified". **That
was premature and is withdrawn.** Two direct refutations:

* **The gain survives the mean pathway's removal** (81.798 → 81.800 on the proxy; 83.509 →
  83.585 on the full split). If the mean were the mechanism, zeroing it would have removed the
  gain.
* **With the shortcut gone, the head does what it was built to do.** `oscz2` produces the
  round's **largest coherence move, Γ² +2.3 %**, and the only real fluctuation move, **−0.8 %** —
  where `oscg2`, with the shortcut available, managed +0.3 % and −0.1 %.

The correct reading: **given a cheap mean pathway the optimizer takes it; deny it and the same
head improves fluctuation phase instead.** §13.4's decomposition was right about `oscg2` and
wrong to generalize from it.

### 18.2 What is capacity and what is not

`oscz2` **cannot** contribute a time-mean, yet its mean part still improves −2.1 %. So that
component is the head **shaping the trunk** — capacity, exactly the control §13.5 wanted — and
`--mean-head`'s −1.3 % is plausibly the same effect by another route. Consistent with §13.5's
first branch.

**Not additive**, as predicted: `mhosc` scores 83.567 against `oscz2`'s 83.585 — combining a
mean head with an osc head lands where the osc head already was.

### 18.3 The significance, kept in proportion

Every other arm this round left the fluctuation at 0.0704–0.0707 and Γ² within ±0.5 %.
**`--osc-zeromean` is the first change in the project to move fluctuation coherence at all** —
Γ² 0.3641 → 0.3723 from a **4 748-parameter** head — and the fluctuation carries 69 % of the
squared error.

It is still small: total rel-L2 −1.2 %, under the ~2 % transfer bar. But it is the first crack
in the one quantity §P3 identifies as the only open direction, and it says the right way to
attack the fluctuation is **a constrained output basis, with every cheaper pathway explicitly
denied**. The obvious next arms: `--osc-zeromean` at J = 1 and J = 4, with a fixed
(non-learned) ω, and combined with `--dropout 0.05`.

## 19. Seven baseline seeds: the noise floor holds, and the proxy is PRECISE but not ACCURATE

`twins` trained three more held-out w24 seeds, giving seven of the identical recipe on the AoA
proxy:

| | `wfinal` | rel-L2 |
|---|---|---|
| 4 seeds | 81.7475 ± **0.0339** | 0.11303 ± 0.00040 |
| **7 seeds** | 81.7506 ± **0.0282** | 0.11297 ± **0.00031** |

**Three more seeds tightened the floor rather than widening it** (σ 0.034 → 0.028), and the
mean moved 0.003. Every z-score quoted this round therefore grows by ~1.2×; no conclusion
changes sign or category. Restated against the seven-seed floor:

| arm | `wfinal` | z (4 seeds) | **z (7 seeds)** |
|---|---|---|---|
| `oscg1` | 81.818 | +2.08 | **+2.39** |
| `--lam-tgrad 0.3` | 81.806 | +1.73 | +1.96 |
| `--dropout 0.05` | 81.802 | +1.61 | +1.82 |
| `oscz2` | 81.800 | +1.55 | +1.75 |
| `oscg2` | 81.798 | +1.49 | +1.68 |
| `--dropout 0.15` | 81.791 | +1.28 | +1.43 |
| `osc2` (per-pixel) | 81.743 | −0.13 | −0.27 |
| `--w-mean 0.3` | 81.688 | −1.75 | −2.22 |
| `--mean-head` s1 | 81.678 | −2.05 | −2.57 |
| `--aug-scale 0.15` | 81.665 | −2.43 | −3.03 |

### 19.1 The gap that explains the whole project's transfer problem

The AoA proxy resolves **σ = 0.028 `wfinal`** across seven identical-recipe seeds — that is
0.034 % of its own value, a genuinely precise instrument. The board's transfer resolution is
~2 % of error, which is **≈ 0.5 `wfinal`**.

**So the proxy measures ~18× finer than the board can distinguish.** That is not a defect in
its precision; it is the definition of the problem. Every arm this round landed in the band
between those two numbers — real at 1.5–2.4σ locally, invisible at the board. **The proxy is
precise but not accurate**, and no amount of extra seeds fixes that, because the missing
quantity is distribution shift rather than variance.

This is why §15's dropout result matters and the other five do not: dropout is the only arm
measured on a **shifted** distribution (`re_split`), where its effect is −4…−14 % of error —
comfortably above the board's resolution rather than 18× below it.

## 20. The paired k-fold on dropout — and §14.2's TKE claim does not survive it

3 383 out-of-fold windows, identical windows for both arms, trajectory bootstrap on the
difference:

| metric | baseline | `--dropout 0.15` | Δ | SE | z | % |
|---|---|---|---|---|---|---|
| rel-L2 | 0.08416 | 0.08306 | −0.00110 | 0.00052 | **−2.10** | −1.31 % |
| **TKE** | 0.46481 | 0.46922 | +0.00441 | 0.00161 | **+2.74** | **+0.95 % WORSE** |
| MVPE | 0.07156 | 0.06854 | −0.00302 | — | — | −4.23 % |

Priced at the eight-round weights: rel_l2 **+0.028**, tke **−0.025**, mvpe +0.000, sps +0.066
→ **+0.069 `final` (admissible +0.036 … +0.104)**.

### 20.1 §14.2's "dropout also buys TKE" is withdrawn

§14.2 separated dropout from the other five arms on the strength of **TKE −3.1 %**, measured on
336 unpaired windows. On 3 383 **paired** windows with a trajectory bootstrap, TKE goes the
other way — **+0.95 % worse at z = +2.74**. The better instrument reverses the sign, and the
paired design removes exactly the between-trajectory variance that made the 336-window number
unreliable.

**That is the third time tonight a larger instrument has overturned a reading from a smaller
one** (§13.3 the osc ordering, §5.5 the 1400× seed ratio, and now this). The pattern is not
coincidence: at these effect sizes the 126- and 336-window splits are simply below the
resolution needed, and I have been over-reading them.

### 20.2 What the case for dropout now actually rests on

**In distribution dropout is worth ~+0.07 `final`, and its rel-L2 gain is largely cancelled by
a TKE loss.** That is not a reason to change the package.

**The case rests entirely on the out-of-distribution result** (§15): `relo` −14 %, `rehi` −4 %,
complete 3-vs-4 seed separation, stacking on ensembling. And **`kfold.py` structurally cannot
confirm that** — its folds are stratified over (AoA, Re), so every fold is in-distribution in
Re. I queued it as "the paired confirmation"; it is the paired confirmation of the *wrong
quantity*.

**The right instrument does not exist yet: a leave-Re-BAND-out k-fold**, folds defined by Re
bands rather than stratified across them, so every fold is an extrapolation test and the paired
bootstrap applies to the quantity that matters. That is the honest next step before spending
2.7 GPU-h on dropout-trained all-data members.

Recommendation, revised down from §15: **do not rebuild the package on this evidence.** Build
the Re-band k-fold first. `jobs/alldata_dropout.sbatch` stays written and unqueued.

## 21. The osc follow-up: the SCORE claim regresses, the COHERENCE claim replicates

Four zero-mean osc arms plus dropout and the combination, against the 7-seed floor
(AoA proxy 81.7506 ± 0.0282; full split 83.4275 ± 0.0502).

| arm | z (AoA) | z (full) | Δ rel-L2 (full) | **Γ²** | ΔΓ² | fluc part |
|---|---|---|---|---|---|---|
| baseline | — | — | — | 0.3641 | — | 0.0707 |
| `oscz2` (J2) | +1.75 | +3.14 | −1.20 % | 0.3723 | +2.3 % | 0.0701 |
| **`ozb`** (J2, replicate) | **−0.52** | **+1.31** | −0.62 % | 0.3684 | +1.2 % | 0.0704 |
| `oz1` (J1) | +0.76 | +0.95 | −0.62 % | 0.3676 | +1.0 % | 0.0703 |
| `oz4` (J4) | +0.05 | +1.03 | −0.73 % | 0.3696 | +1.5 % | 0.0702 |
| `do05` (dropout) | +1.82 | +3.76 | −0.73 % | **0.3591** | **−1.4 %** | 0.0703 |
| **`ozdo`** (both) | **+3.87** | **+4.75** | **−1.67 %** | 0.3701 | +1.6 % | **0.0699** |

### 21.1 §18's score claim does not survive its replicate

`oscz2` read +3.14σ on the full split; its replicate `ozb` reads **+1.31σ**, and on the AoA
proxy the pair is +1.75σ / **−0.52σ**. Two identical configs 0.092 apart on the full split
against a baseline σ of 0.050. **The single-run +3.1σ overstated the effect by about 2×** — the
four-run family mean is **+1.60σ** (full) and +0.51σ (proxy).

**Fourth instance tonight of a single run at 2–3σ regressing on replication.** The others:
§13.3 the osc ordering, §5.5 the 1400× ratio, §20.1 the dropout TKE claim. This is now a
measured property of the setup, not bad luck: **at these effect sizes one run is not a
measurement**, and replicates belong in the first job rather than the follow-up.

### 21.2 But the COHERENCE claim replicates, and it separates the two mechanisms

**Γ² across all four zero-mean arms: 0.3695 ± 0.0021 = +1.5 % over baseline**, consistent
across J = 1, 2 and 4 and across seeds. §18's central claim — that this is the first change to
move fluctuation coherence — **survives replication**, at +1.5 % rather than the single run's
+2.3 %.

**And dropout does the opposite: Γ² −1.4 %.** It improves rel-L2 while making coherence
*worse*, which is exactly consistent with §20's paired k-fold finding that it makes TKE worse
(+0.95 %, z = +2.74). Dropout is a regularizer that trades phase for robustness.

**That is why they add.** The two arms move different quantities in different directions:

| | coherence Γ² | fluctuation error | mechanism |
|---|---|---|---|
| `--osc-zeromean` | **+1.5 %** | −0.6 % | constrained temporal basis |
| `--dropout 0.05` | **−1.4 %** | −0.6 % | regularization |
| **combined** | **+1.6 %** | **−1.1 %** | both, and the coherence gain survives |

`ozdo` has the **lowest fluctuation error of any model in the project** (0.0699 against the
baseline's 0.0707) and the round's largest rel-L2 move, **−1.67 %**. Roughly additive: the
full-split additive prediction is 83.696 against 83.666 observed.

### 21.3 Standing

`ozdo` is **one run** and this section is a case study in why that is not enough. Three
replicates are queued (`jobs/ozdo_rep.sbatch`) with the pass condition fixed in advance: **the
four-run mean must clear +2σ against the 7-seed floor AND the rel-L2 move must clear ~2 %.**
Five arms this round cleared the first and none transferred.

If it holds at ~−1.5 % rel-L2 it is still under the transfer bar on its own — but combined with
whatever `kfold_re` says about dropout out of distribution, it is the first candidate of the
round with a mechanism, a replicated component, and two independent effects that stack.

## 22. `ozdo` replicated: the mechanism holds 4-for-4, the SCORE falls short of the bar

Three more runs of `--osc-modes 2 --osc-global --osc-zeromean --dropout 0.05`, judged against
the pre-registered condition from §21.3 — *the four-run mean must clear +2σ against the 7-seed
floor **AND** the rel-L2 move must clear the ~2 % board transfer bar.*

| | AoA 15–20 proxy (7 base vs 4 ozdo) | full val_unseen (4 vs 4) |
|---|---|---|
| rel-L2 | 0.11297 ± 0.00031 → 0.11170 | **−1.13 %** | 0.08513 ± 0.00053 → 0.08348 | **−1.94 %** |
| TKE | 0.46750 ± 0.00206 → 0.46780 | +0.06 % | 0.49123 ± 0.00434 → 0.48008 | **−2.27 %** |
| MVPE | 0.09783 → 0.09675 | −1.10 % | 0.07318 → 0.07107 | −2.87 % |
| sps | 42.129 ± 0.072 → 42.515 | +0.386 | 47.633 ± 0.138 → 48.298 | +0.665 |
| **wfinal** | 81.7506 ± 0.0283 → **81.8590** | **+3.84σ** | 83.4275 ± 0.0502 → **83.6668** | **+4.77σ** |

### 22.1 The replication test passes, and it is the first arm this round to do so

Four runs at 81.860 / 81.835 / 81.891 / 81.850 — a spread of 0.056, *tighter than the seven
baseline seeds* (σ 0.0237 against 0.0283). Contrast §21.1, where `oscz2`'s +3.14σ became
+1.31σ on its single replicate. This is not that.

### 22.2 The mechanism claim replicates exactly, and the fluctuation finally moved

`diag_spectral.py` on all four, against the baseline's Γ² 0.3641 / mean 0.0470 / fluc 0.0707:

| run | Γ² | time-mean part | fluctuation part |
|---|---|---|---|
| `ozdo` | 0.3701 | 0.0457 | 0.0699 |
| `ozdo_b` | 0.3699 | 0.0457 | 0.0697 |
| `ozdo_c` | 0.3713 | 0.0450 | 0.0695 |
| `ozdo_d` | 0.3701 | 0.0457 | 0.0697 |
| **mean** | **0.37035 ± 0.00066 (+1.66 %)** | **0.04553 (−3.1 %)** | **0.0697 (−1.4 %)** |

**This is the first change in the project to move the fluctuation part of the error.** `ROADMAP`
P3 records that every decomposition of this round — baseline, `--w-mean 0.3` with 3.3× the
gradient on it, `oscg2`, `osc2` — left it pinned at 0.0704–0.0707 while 69 % of the squared
error lives there. Four runs at 0.0695–0.0699 is outside that band, and Γ² moves with it at a
σ of 0.0007 across four seeds.

### 22.3 …and the score still fails the second half of the condition

**−1.13 % on the AoA 15–20 proxy against a bar of ~2 %.** The bar is not arbitrary and it is
not about noise: §1.8 calibrated it on three board rounds, where proxy deltas of −0.87 % and
+1.78 % predicted the board's *sign* wrongly. §19 measured the same thing from the other side —
the proxy resolves ~18× finer than the board can distinguish, so it is **precise but not
accurate**, and four tight replicates buy precision, which is not the failing quantity.

**Verdict, by the rule as written: `ozdo` does not qualify as a package change on its own.**
The full split reads −1.94 %, but the bar was calibrated on the proxy and moving to whichever
instrument gives the friendlier number is how five arms got shipped in rounds 9–12.

What it does qualify as is a **replicated mechanism with a measured, independent signature**
(Γ², the fluctuation part) that no previous arm has touched — which is a reason to keep
pulling on the osc/zero-mean thread, not a reason to rebuild the package around it.

## 23. What the oscillatory head actually learned — and it closes both physics arms

`Track_1/osc_probe.py` reads ω and the mode amplitudes straight out of the head over all 336
`val_unseen` windows. Four `ozdo` seeds plus `oscz2`, J = 2, `--osc-global`:

| | ω (slow mode) | ω (fast mode) | period | corr(ω, U) | in shedding band |
|---|---|---|---|---|---|
| `ozdo` | 0.204 ± 0.066 | 0.397 ± 0.118 | 31 / 16 fr | **+0.995 / +0.992** | **0.00** |
| `ozdo_b` | 0.248 ± 0.081 | 0.438 ± 0.119 | 25 / 14 fr | +0.991 / +0.978 | 0.00 |
| `ozdo_c` | 0.246 ± 0.081 | 0.433 ± 0.122 | 26 / 15 fr | +0.992 / +0.980 | 0.00 |
| `ozdo_d` | 0.240 ± 0.083 | 0.433 ± 0.122 | 26 / 15 fr | +0.994 / +0.982 | 0.00 |
| `oscz2` | 0.214 ± 0.071 | 0.412 ± 0.122 | 29 / 15 fr | +0.995 / +0.989 | 0.00 |

Both modes carry real amplitude (0.13–0.26 against a normalized field of order 1), the two
always separate by a factor ≈ 1.75, and every number above replicates across five independent
runs.

### 23.1 The head tracks the freestream speed with no supervision — and it is NOT Strouhal

`corr(ω, U) = +0.98…+0.995` on every mode of every seed. That looks like the shedding law
`St = f·c/U∞ = const`. It is not. Fitting the two forms:

| model | R² (slow mode) | R² (fast mode) |
|---|---|---|
| `ω = b·U` (Strouhal, zero intercept) | 0.69 / 0.70 | **0.52 / 0.35** |
| `ω = a + b·U` | **0.991 / 0.984** | **0.984 / 0.960** |
| `ω = b(AoA)·U`, own constant per angle | 0.82 / 0.78 | 0.73 / 0.71 |

The intercept is large and positive (a = +0.078…+0.205 against a slope term spanning
0.04–0.31 over the observed U range), and giving each angle of attack its own Strouhal constant
fits **worse** than one affine law — so AoA is not the missing variable either.

### 23.2 Both physics-motivated arms are therefore closed before being run

* **Narrow the band to the shedding band.** Wrong: **not one window of 336 lands in
  [0.0838, 0.1047]** on any seed. The head runs at 2.5–4.5× the shedding frequency, at periods
  of 14–31 frames. The reason is structural — a 20-frame output window spans only 0.27–0.33 of
  a 60–75 frame shedding cycle, so the fundamental is **not identifiable in-window**, and the
  head picks periods that fit about one cycle inside the data it has.
* **Lock ω = c·U (Strouhal).** Wrong: it is the zero-intercept model, and imposing it drops R²
  from 0.99 to 0.52–0.70.

The band ceiling binds only marginally — `2π/10 = 0.628`, and one mode of one seed has 12.5 %
of its windows within 1 % of it, the other seven mode-seeds 0.0 %. Widening the band is a real
but small arm, not the one the correlation suggested.

**What the head is doing, stated at the confidence the measurement supports:** it emits a
smooth, near-affine function of the window's own mean speed, in a frequency range the 20-frame
window can actually resolve, and the resulting term improves Γ² and the fluctuation error
(§22.2) across four seeds. Whether that is physics or a useful reparameterization is not
determined here — but the two things it is *not* are now measured, which is worth the 2 GPU-h
they would each have cost.

## 24. The osc head could not be shipped at all — `fuse()` refused it, at −0.11 `final`

§22 priced `ozdo` at roughly +0.10 `final` if it transfers, and recommended against shipping it
on the proxy delta. That recommendation was made without checking what it would have cost, and
the cost turns out to dominate the question.

`_fno.fuse` grouped members by their `state_dict` shape signature and then **passed any member
with a `head_osc` through unfused**, alongside `head_mean`, because `StackedFNO` enumerates its
heads explicitly and would silently drop one it did not know about. Measured:

```
4 ozdo members      -> 4 invocations
4 baseline members  -> 1 invocation
```

An eight-member `ozdo` package is therefore **8 sequential invocations**. At §1.5's board-time
model (`board ms = 4.447 + 0.324·inv + 0.431·local ms`, 1 board ms ≈ 0.05 `final`) that is
**+2.3 board ms ≈ −0.11 `final`** — larger than the arm's entire measured gain. Round 12 paid
0.973 `time_score` for one extra group and is the empirical anchor for that slope.

**So the only change this round that moved the fluctuation error was not shippable in any
configuration**, and no amount of further held-out evidence would have made it so.

### 24.1 Fusing it — everything the head does is linear in the member axis

`head_osc` is two 1×1 convs and groups exactly like `head_field`. `osc_w` is an MLP on the
**spatially pooled** trunk, i.e. a per-member `width → 64 → J` matmul, so stacking its weights
along a member axis and using `einsum("bki,koi->bko")` is the same block-diagonal map a grouped
conv would give without reshaping to 1×1 spatial. The cos/sin synthesis, the zero-meaning and
the sum over modes are all linear, so adding the per-member `osc` to `f_all` **before** the
member mean is identical to the sequential loop's add-then-average.

`Track_1/verify/fuse_osc.py`: 4 members → **1 invocation**, fused against sequential
`predict_fno` to **3e-7** at α = 0.0 / 0.60 / 0.90, sigma (which drives the SPS band) to
**1e-8**. TF32 must be off — it turns the spectral einsum's 1e-8 agreement into 1e-3 with
neither path being wrong.

**A repeat of the `self_norm` trap, caught this time.** `osc_zeromean` is a bool with no
parameters, so the shape signature cannot see it and two members disagreeing on it would have
fused into one module that can only be right for some of them. It is now part of the group key,
with `osc_modes` and `osc_global` (which *do* show up in the shapes, but are keyed explicitly
rather than left to that coincidence).

### 24.2 What this changes about §22's verdict

Nothing about the evidence; everything about the option set. Before: `ozdo` was +0.10 accuracy
against −0.11 latency, i.e. **not shippable at any confidence**. After: it is +0.10 against
**0.00**, judged against a proxy that §1.8 says gets the sign right about two thirds of the
time at this effect size. That is a different decision, and it is the one the round-14 package
faces. The proxy delta still does not clear the bar and the arm still should not be claimed as
a win — but "free upside with a replicated mechanism" and "upside that pays for itself twice
over in latency" are not the same proposition.

## 25. The oscillatory TERM is load-bearing — an inference-time ablation

`Track_1/osc_ablate.py` runs identical weights with `head_osc`'s output conv zeroed, which is
exactly the branch's own initialization. Two `ozdo` seeds, α = 0.90, all 336 `val_unseen`
windows:

| checkpoint | branch | rel-L2 | time-mean part | fluctuation part | TKE |
|---|---|---|---|---|---|
| `ozdo` | with osc | **0.08370** | 0.04568 | **0.06985** | 0.47839 |
| `ozdo` | osc OFF | 0.10091 | 0.04568 | 0.08975 | 0.48930 |
| `ozdo_c` | with osc | **0.08305** | 0.04501 | **0.06951** | 0.48122 |
| `ozdo_c` | osc OFF | 0.09352 | 0.04501 | 0.08167 | 0.49339 |
| *(4-seed baseline, §22)* | — | *0.08513* | *0.0470* | *0.0707* | *0.49123* |

**Two things fall out, and the second was not the question being asked.**

### 25.1 It is the TERM, not a regularizer — so the follow-up is about its form

Zeroing the branch does not return the model to baseline; it takes it **well past** baseline, to
rel-L2 0.101 / 0.094 against 0.085 and a fluctuation part of 0.0898 / 0.0817 against 0.0707.
The trunk has genuinely offloaded part of the fluctuation onto the oscillatory branch and no
longer predicts it itself. **The branch is load-bearing.**

That settles the §13.4 / §18 question that has been reopened twice in this round, and it
decides the next arm: the head belongs with the **fluctuation** work, not next to `--dropout`,
and the productive direction is the term's *form* — a free learned temporal basis, more modes,
a different band — rather than treating it as one more regularizer.

**The honest limit of this ablation.** It shows the branch is used; it does **not** show the
branch is uniquely good. The right control for "is this a good idea" is a model trained without
it, which is §22's baseline, and against that the fluctuation gain is only −1.4 %. What the
ablation adds is that the −1.4 % comes from a component doing real work, not from noise or from
a training-time side effect.

### 25.2 `--osc-zeromean` does exactly what it claims, to five decimals

The time-mean part is **identical with and without the branch** — 0.04568 vs 0.04568, 0.04501
vs 0.04501. The whole point of §18's zero-mean constraint was that the un-constrained head was
suspected of being a mean-head in disguise (`mean_t cos(ωt) = 0.511` over a 20-frame window at
the learned period). It now contributes **nothing** to the time-mean, by construction and by
measurement, while carrying 18–28 % of the fluctuation error. The "phase idea is untested"
worry from §13.4 is closed.

## 26. `--norm self`, measured correctly at last — and honestly closed

Three seeds on the Re split, scored through the fixed path (§5.3b), against the four controls.

| | control (4 seeds) | `--norm self` (3 seeds) | Δ | σ ratio |
|---|---|---|---|---|
| `rein` rel-L2 | 0.07537 ± 0.00005 | 0.07680 ± 0.00020 | **+1.89 %** | 0.2× |
| **`relo` rel-L2** | 0.12498 ± **0.01092** | **0.11107 ± 0.00083** | **−11.13 %** | **13.1×** |
| `rehi` rel-L2 | 0.12290 ± 0.00068 | 0.12377 ± 0.00076 | +0.71 % | 0.9× |
| **`relo` TKE** | 0.63155 ± 0.07251 | **0.76300 ± 0.00678** | **+20.81 %** | 10.7× |
| `rein` wfinal | 83.999 ± 0.012 | 83.859 ± 0.023 | −0.140 | |
| `relo` wfinal | 81.647 ± **0.707** | 81.521 ± **0.065** | −0.126 | **10.8×** |

And as 2-member ensembles, which is how it would actually ship:

| | `rein` rel | `relo` rel | `rehi` rel | `relo` TKE | `relo` wfinal |
|---|---|---|---|---|---|
| control ×2 | **0.0739** | 0.1170 | 0.1199 | **0.5650** | **82.495** |
| self-norm ×2 | 0.0752 | **0.1077** | 0.1213 | 0.7188 | 81.996 |

### 26.1 It does exactly what Strategy item 5 promised — for the FIELD

rel-L2 on the low-Re extrapolation set is **−11.1 %** single-model and **−7.9 %** at two
members, below even the best of the four controls. And the variance collapses: the controls
scatter σ = 0.0109 in rel-L2 and **0.707 in `wfinal`** across seeds on `relo`, where the
self-normalized models scatter **0.00083 and 0.065** — a **13× / 11× reduction** in exactly the
quantity §5.5 identified as the reason no local number has transferred since 2026-08-08, and
exactly what §17.1 showed ensembling was buying. The control's `relo` `wfinal` spans 1.419
across four seeds; self-normalization's spans **0.124** across three.

The mechanism is the one the strategy note gave: at Re 3750 the mean `u` is ~0.04 m/s against a
global normalizer centred on 0.155, so the absolute-units model is asked to extrapolate its
input distribution and every seed extrapolates differently. Dividing by the window's own mean
speed removes that axis, and the models stop disagreeing.

### 26.2 And it pays for it in TKE, reliably

`relo` TKE is **+20.8 %** single-model and **+27 %** ensembled, with a seed σ of 0.0068 — it is
*consistently* worse, not variably worse. The mechanism is structural, not a bug: under
self-normalization the variance head predicts σ in **self-normalized units**, so the deployed
physical σ is `σ̂ · s`. That hard-codes a **Reynolds-independent relative turbulence intensity**,
which the data does not have. The field pathway benefits from removing the Re scale; the
variance pathway is harmed by it.

### 26.3 Verdict: closed, and the obvious hybrid is not worth running either

`wfinal` is a wash single-model (−0.13) and a **loss ensembled (−0.50)**, and in distribution
the arm is decisively worse (rel-L2 +1.89 % at 4× the control spread). **Do not ship
`--norm self`. Strategy item 5 is now closed on a valid measurement.** §5.3 reached the same
verdict through a broken instrument, which does not make it retroactively right — the reasons
are entirely different, and "catastrophic" was wrong.

The hybrid §26.2 suggests — self-normalize the field, keep σ physical — was priced before
running it. Give it the field gain *and* the control's TKE: `relo` `rel_l2_score` 94.46 → 94.89
is **+0.23 wfinal on `relo`**, which §7.3's dilution (the board sits ~2.7 % outside the training
range against `relo`'s 12 %) cuts to roughly **+0.05**, against a *certain* in-distribution cost
of about −0.04. **Not worth a GPU-hour**, and it would need `compose()` to carry two unit
systems. The self-normalization family is done.

*(`osc_ablate.py` handles either temporal branch — `head_osc` and `head_tb` — since the
question and the ablation are identical for both.)*
