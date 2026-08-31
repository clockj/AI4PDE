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

Two readings, and the measurement does not separate them: the *idea* is wrong, or the
*implementation* breaks when the window's own mean speed is small (at Re 3750 the mean `u` is
~0.04 m/s, so dividing by it amplifies everything, including the zero-filled out-of-FOV
region). Either way: **do not ship `--norm self`, and do not cite Strategy item 5 as untested
— it is tested and negative.** Round 2's "neutral" was measured with 74 training trajectories
against this split's 56, so it is not a clean contradiction; the same-split control is.

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
