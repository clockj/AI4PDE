# Round 15 — capacity, regularization, and five new model families

Plan: `Track_1/plan/plan_20260902.md`. Raw rows and every table verbatim:
`Track_1/plan/campaign_20260902.md`. Instrument log: `leaderboard/`.
Machine: OSC Ascend, A100-PCIE-40G, at most 3 concurrent `rp_*` jobs.

**The round's bet, and it paid.** Round 14 measured that the shipped width-24 FNO *underfits*
— trained on eight trajectories and scored **in sample** it reaches ρ 0.588, barely above its
own held-out 0.609, while a width-96 model at 400 epochs reaches 0.945 — and the 2026-09-02
board paid **+0.200** for dropout. Round 15 tested the other half: **bigger models, regularized
harder, and other model families at that operating point.**

---

## 1. The headline

**U-FNO — an FNO block with a small U-Net branch beside it — is the first arm since round 8 to
clear the board's measured transfer resolution.**

3-vs-3 (ROADMAP 1.15), full split, against the *shipped* recipe:

| | rel-L2 | TKE | MVPE | sps | `wfinal` |
|---|---|---|---|---|---|
| w24 do05 ×3 (shipped) | 0.0787 | 0.5141 | 0.0697 | 42.79 | 80.865 |
| w96 FNO do30 ×3 | 0.0775 | 0.5112 | 0.0695 | 43.72 | 81.124 |
| **U-FNO ×3** | **0.0752** | **0.4911** | **0.0670** | **44.82** | **81.527** |

**−4.4 % rel-L2, −4.5 % TKE, −3.9 % MVPE, +2.03 sps, +0.662 `wfinal`.** `leaderboard/README.md`
measures the board's transfer resolution at **~4 % of error**; every candidate screened since
round 8 has been *below* it — dropout was −2.6 % on its best instrument, and the width-96
package staged tonight is **−1.2 %**. This is the first one to reach the limit. It also improves
**all four scored error metrics simultaneously**, which only `--lam-grad` has ever done, at the
tightest seed spread on record (full-split `wfinal` σ **0.008**).

---

## 2. The family ladder

Five new architectures, all at width 96 / dropout 0.30 / 400 epochs / held out on the standard
split, all exposing `FNO2d`'s exact interface so every comparison shares the normalizer,
windowing, variance composition, band and timing method.

| family | params | train `field` | seen/unseen gap | held-out ρ | proxy | **full split** |
|---|---|---|---|---|---|---|
| **U-FNO** | 15.57M | 0.0336 | 1.57× | **0.636** | **82.500** | **84.663** |
| **ViT** (d384, 8 layers, 128 tokens) | 14.66M | 0.0370 | 1.47× | 0.629 | 82.379 | 84.438 |
| **AFNO** | **0.24M** | 0.0785 | **1.05×** | 0.630 | 82.168 | 84.337 |
| FNO w96 do30 (round's baseline) | 14.24M | 0.0391 | 1.51× | 0.618 | 82.138 | 84.318 |
| Dilated ResNet | 2.38M | 0.0299 | 1.61× | 0.600 | 81.980 | 84.235 |
| U-Net w96 | 4.27M | 0.0300 | 1.64× | 0.586 | 81.897 | 84.029 |
| FNO w24 do05 (**shipped**) | 0.91M | — | — | 0.609 | 81.892 | 83.970 |

**Three of the five new families beat the FNO, and the two that lose are the two with no
spectral component at all.** Round 10b's "the architecture search is finished" was a statement
about *width*, at fixed regularization, inside *one* family.

**The two controls are what make U-FNO readable, and both are negative.**
* **Dilated ResNet** reaches the whole grid in one block with a purely local, translation-
  equivariant parameterization and no mode truncation. It **fits better in training** (0.0299
  against the FNO's 0.0391) and generalizes worse than every FNO here. So the Fourier
  parameterization is doing real work; it is not merely a cheap global receptive field.
* **U-Net at width 96** is barely above the shipped width-24 FNO and has the round's worst
  held-out ρ (0.586) despite the second-lowest training loss.

U-FNO's gain is therefore *not* "a conv net was better" and *not* "more parameters": it is the
spectral operator **plus** a local branch, and the local branch alone loses. That is the
mechanism the arm was proposed for — mode truncation is a low-pass and cannot represent the
sharp zero set of the PIV field of view and the airfoil body (3.3 % of elements, exactly zero),
which the 1×1 pointwise branch has no spatial extent to fix either.

**Diluting it hurts**: one U-FNO plus two FNOs scores 81.296, between the two pure sets and much
nearer the FNO. A U-FNO package wants to be U-FNO.

**AFNO is the most interesting negative-space result.** 0.24M parameters — a *quarter* of the
shipped model, 60× smaller than U-FNO — and it still beats the width-96 FNO. It is the only arm
that genuinely underfits (training `field` 0.0785, twice everyone else's) and the only one with
essentially no generalization gap (**1.05×** against 1.5–1.6×). Sharing spectral weights across
modes and soft-thresholding is a real regularizer. Not a package candidate as built; a
demonstration that the parameter budget is not what buys the score.

---

## 3. Capacity × regularization, inside the FNO

| arm | proxy | full split | note |
|---|---|---|---|
| w24 do05 (shipped) | 81.892 | 83.970 | |
| w96 do15 | 81.886 | 84.128 | round 14 |
| w96 do30 (3 seeds) | 82.098 ± 0.036 | 84.31 ± 0.01 | |
| w96 do45 (2 seeds) | 82.190 ± 0.007 | 84.33 ± 0.00 | |
| w96 do60 | 82.206 | 84.293 | |
| w128 do45 (3 seeds) | 82.19 ± 0.02 | 84.38 ± 0.02 | |
| **w96 do30, 8 layers** (3 seeds) | **82.19 ± 0.03** | **84.41 ± 0.01** | best FNO |
| w96 do30, modes (12,20) | 82.046 | 84.331 | null |
| w96 do30, `--phases 4` | 82.111 | 84.321 | null |

* **Round 8's capacity ladder is falsified on two of three axes.** Width, depth and mode count
  were all measured as monotonically harmful with the regularizer held *fixed*; **width and
  depth both pay once dropout moves with them**, and only mode count stays null.
* **The dropout optimum is broad and flat from 0.30 up**, not monotone: 0.45 and 0.60 gain
  +0.09 and +0.11 on the proxy over 0.30 and tie it on the full split. What does move is
  **stability** — the do45 seed spread is 0.007 against do30's 0.036, and the L8 and w128 trios
  are 0.014 and 0.021 on the full split. More capacity with more dropout buys seed variance
  reduction, which matters more for a package than the mean, because seed variance is exactly
  what ensembling pays to remove.
* **`--phases 4` is closed properly.** Round 14 put every data-side null "back in play" on the
  grounds that extra data cannot help a model that underfits. The model no longer underfits, the
  4× real augmentation was re-asked at matched optimizer steps, and the answer is the same.
* **Mixing dropout LEVELS does not help** (better on the proxy, worse on the full split);
  mixing **architectures** does, slightly. Round 8's "architecture beats seeds" survives; its
  regularization analogue does not.

---

## 4. Latency, and the shape of a package

`Track_1/lat_r15.py` (new) measures a member set end to end, replicating `predict()`'s timed
section rather than the forward pass, on an idle node, median of 5. `validate_strict` and
`lat_r15` agree to 2 %.

| set | memb | inv | A100 ms | 4060-eq | board ms | Δ`final` |
|---|---|---|---|---|---|---|
| shipped 8×w24 (fused) | 8 | 1 | 1.00 | 1.86 | 4.54 | — |
| 3× w96 FNO (fused) | 3 | 1 | 1.00 | 1.85 | 4.54 | −0.001 |
| 8× w96 FNO (fused) | 8 | 1 | 1.61 | 2.98 | 5.45 | **−0.063** |
| 8×w24 + 4×w96 | 12 | 2 | 1.50 | 2.77 | 5.71 | −0.083 |
| 3× U-FNO (unfused) | 3 | 3 | 1.23 | 2.28 | 5.75 | −0.085 |
| 4× U-FNO (unfused) | 4 | 4 | 1.45 | 2.68 | 6.51 | −0.138 |
| 6× U-FNO (unfused) | 6 | 6 | 1.90 | 3.52 | 8.05 | −0.246 |

* **The capacity bet is cheap.** Eight fused width-96 members cost **+0.93 board ms = −0.065
  `final`** against eight width-24 ones, a fifth of the accuracy they buy. Three width-96
  members cost the *same* board ms as eight width-24 ones. The package is launch-bound, not
  FLOP-bound: 16× the width costs 14 % more A100 time at one member.
* **What is NOT free is an invocation.** Mixing widths costs one (0.435 board ms), so
  `8×w24 + 4×w96` is worse than `8×w96` despite half the parameters. And **`_fno.fuse` folds
  `FNO2d` members only** — everything else is handed back one per invocation, so N U-FNO
  members are N invocations. That is the single biggest thing standing between U-FNO's +0.66
  local and a package: a `StackedUFNO` would save ~0.09–0.13 `final` at 3–4 members and is the
  first item for round 16.
* **8 × w96 does not fit in 256 MB** (456 MB fp32). `build_zip.py --fp16` /
  `build_t2.py --fp16` store the floating tensors as float16 → 228.6 MB, at a measured rel-L2
  cost of **5.5e-4** against a model rel-L2 of 0.078, i.e. 2e-6 of the error in quadrature.

---

## 5. `predict_board.py` is now anchor-relative, and its identity control found two bugs

The old path multiplied the proxy by fixed board/proxy ratios whose spread across rounds is
2.5–6.3 %, i.e. ±0.15 `final` — larger than anything since round 8. The new `--anchor` mode
scores **both** packages' held-out twins on one instrument and extrapolates only the *ratio of
two proxies*:

```
board_error(cand) = board_error(anchor) × proxy(cand)/proxy(anchor)
board_sps(cand)   = board_sps(anchor)   + [sps_proxy(cand) − sps_proxy(anchor)]
board_ms(cand)    = board_ms(anchor, MEASURED) + [model(cand) − model(anchor)]
```

The calibration ratio and the eval machine's latency drift both cancel into the anchor, whose
own recorded board result carries the level; `final` is priced from the delta with `final_v9`'s
twelve-round weights, so the intercept cancels too.

**Running it on a package against its own board round must return zero.** It did not, and the
two reasons were both real: SPS was being scored at the anchor's *proxy* metrics and the
candidate's *board* metrics, whose per-sample `a_i` factors differ (0.730 against 0.701) and
invented **−1.2 sps between a package and itself**; and board latency came from the seven-round
fit's *level*, which predicts 4.571 ms for a round that actually returned 3.370. Fixed, the
control returns **all five deltas exactly 0.000**. Every `--anchor` run should carry it.

---

## 5b. THE BOARD ANSWERED — 81.019, and it paid for the tail, not the mean

| | id | rel_l2 | tke | mvpe | time | sps | **final** |
|---|---|---|---|---|---|---|---|
| T1 `dropout8` | 913176 | 94.780094 | 77.747643 | 94.122329 | 93.180177 | 40.140948 | **80.835124** |
| **T2 `w96x8`** | 913187 | 94.755824 | 78.174302 | 93.739707 | 92.508902 | 41.160630 | **81.019213** |

**81.019 is the account's best on either board**, and the prediction recorded before upload was
**80.962** — a residual of **+0.057**, the best out-of-sample prediction this project has made
since round 11's +0.006. The `dropout8` identity check passes: rel / tke / sps reproduce T2 R2's
to 0.008 / 0.080 / 0.007.

**But the screen ranked the wrong quantity.** Against the T2 R2 anchor, in error terms:

| | the 3-vs-3 screen said | the board delivered |
|---|---|---|
| rel_l2 | **−1.14 %** | **+0.65 % (WORSE)** |
| tke | −2.11 % | −2.9 % |
| mvpe | −0.63 % | +2.5 % |
| **sps** | **+0.51** | **+1.013** |
| time | −0.75 | −1.124 (+1.4 board ms vs +0.93 predicted) |

**The rms got slightly worse and the score went up.** The entire gain is per-element interval
containment plus TKE. This is the *third* time the AoA proxy has had the sign of a rel-L2 move
wrong (R11→R12, dropout on T2 R2, now this), and the second consecutive round where the board
paid for the error **tail**.

**Adopted, for the rest of this round and the next**: rank arms by **`sps` and `tke`**, not by
`wfinal` and certainly not by rel-L2. `Track_1/plan/campaign_20260902.md` carries the `sps`
delta for every arm. The ranking is unchanged under that metric — U-FNO first, L8 the best FNO
— but the margins widen: U-FNO's edge over w96 do30 is +0.164 `wfinal` and **+1.10 sps**.

**The L8 (depth) arm is now the most interesting FNO**, because it has the round's best TKE
(0.4873 at 3-vs-3 against w96 do30's 0.5004) and the best FNO `sps` (43.98 full split) — the
exact shape the board just rewarded. Its replicates and ensemble test are already done (§3);
what it lacks is all-data members.

---

## 5c. The byte budget — why the archive is 212 MB, and what shrinks it

**An FNO is its spectral weights and nothing else.** For a width-96 member:
`w1`/`w2` **14.156M params = 99.4 %**; pointwise 1×1 0.3 %; the two heads 0.2 %; the lift MLP
0.1 %; GroupNorm 0.0 %. The formula is

```
spectral params = 4 · width² · modes_y · modes_x · n_layers
```

(the 4 is two blocks — positive and negative `ky` — times real and imaginary). Everything else
is 1×1 convolutions on a ≤128-channel latent and does not scale with grid or modes. Hence
228.6 MB extracted = 8 × 28.5 MB of fp16 spectral weights; deflate recovers only 7 % because
half-precision weights are near-incompressible.

| member | fp32 | fp16 | **int8** | max members / 256 MB (fp16 → int8) |
|---|---|---|---|---|
| w24 FNO L4 (old shipped) | 3.6 | 1.8 | 0.9 | 139 → 278 |
| **w96 FNO L4 (shipped, 81.019)** | 57.0 | **28.5** | **14.3** | **8 → 17** |
| w96 FNO **L8** | 113.8 | 56.9 | 28.6 | 4 → **8** |
| w128 FNO L4 | 101.2 | 50.6 | 25.4 | 5 → 10 |
| U-FNO w96 L4 | 62.3 | 31.2 | 17.0 | 8 → 15 |

Measured on the three held-out w96 twins (`Track_1/compress_spectral.py`):

* **int8 on the spectral weights is FREE.** Half the archive for **±0.05 % of error** — noise.
  One scale per output channel per tensor (output-channel granularity because the einsum
  contracts over the input axis, so a scale error is a per-channel gain error, not a mixing
  error). It takes 8 × w96 from 228.6 MB to 114.7 and makes **eight L8 members fit where four
  did**. Measured as a weight perturbation; storing genuine int8 + scales in the blob and
  de-quantizing in `_fno.load_any` is a small unwritten change and the first engineering task
  of round 16.
* **Low-rank factorization FAILS, decisively.** Truncating a layer's spectral tensor from
  (width², n_modes) to rank 48 keeps 69.6 % of its energy and multiplies rel-L2 by **2.9**
  (0.0776 → 0.2216); rank 24 by 3.4. The operator is genuinely mode-specific — there is no
  small set of shared channel-mixing matrices modulated across wavenumber. This does **not**
  contradict AFNO scoring well at 0.24M: AFNO *trains* under that constraint and finds a
  different solution; projecting a trained FNO onto it does not work. Rank 96 (no truncation)
  is exact, which confirms the harness.
* **Fewer members** stays the free lever, priced by the ensemble curve.

**Mixed-depth fusion.** `n_layers` changes the state_dict key set (`spec.4..7` etc. exist only
at L8), so L4 and L8 can never share a `StackedFNO` group. Measured: `4×L8` in one invocation
is **+0.009 `final`** against the shipped package (i.e. free), and `4×L4 + 4×L8` in two
invocations is **−0.073**. With int8 both fit the cap easily (114 MB and 172 MB).
**Latency has stopped being the constraint on this axis; the byte budget was, and int8 removes
it.**

---

## 6. What is staged

`../sub_candidates/README.md` holds the predictions, recorded before upload (rule 3).

| archive | track | predicted | **actual** |
|---|---|---|---|
| `track1_r15_dropout8_a060.zip` | 1 | 80.853 | **80.835** (913176) |
| `track{1,2}_r15_w96x8_a060.zip` | 1, 2 | 80.962 | **81.019** (913187) — account best |

Both passed `validate_strict.py`, `verify/cpu_smoke_submission.py` and (Track 2)
`validate_t2.py`. Track 1's `w96x8` archive is still unsubmitted and is the obvious next
Track-1 slot: the two boards share a test set and a rule, so it should reproduce 81.019 to
within the mvpe difference and the timing draw.

---

## 6b. The interior Re-band k-fold is the decision instrument, in fact

`kfold.py --split re --bands 1 2 3` (new `--bands`), w96 do30 against the round-13 w24 do15
folds, paired over **2152 windows / 52 trajectories** — 6× the AoA proxy's sample:

| | AoA 15-20 proxy | **interior Re k-fold** | **BOARD** |
|---|---|---|---|
| rel-L2 | −1.14 % | **−0.50 %** (z −2.59) | **+0.65 %** |
| TKE | −2.11 % | **−3.24 %** (z −7.82) | **−2.9 %** |
| MVPE | −0.63 % | +0.99 % | +2.5 % |
| sps | +0.51 | **+1.19** | **+1.013** |

**The k-fold called it and the proxy did not** — within 0.3 points on TKE, 0.18 on `sps`, right
sign on MVPE, and correctly non-committal on rel-L2 where the proxy was confidently wrong.

Its extrapolation bands, reported and never decided on, are a standing warning: at band 0 the
bigger model is **rel +4.75 %, TKE +16 %**. **Capacity is clearly harmful under Reynolds
extrapolation.** Round 14's φ = 0 — the hidden set is interior — had to come first; had it gone
the other way, this round's bet would have lost.

---

## 6c. The U-FNO package, and a correctness fix it forced

Three all-data U-FNO members staged for both tracks (`track{1,2}_r15_ufno3_a060.zip`, 94.1 MB,
3 invocations, −0.026 `final` of latency against the package already on the board). Three, not
four: three is what the 3-vs-3 measured, a fourth same-recipe member is worth <+0.1 `sps` and
here costs a whole invocation.

**Predicted, anchored on the 81.019 round** (identity control: all five deltas 0.000):
rel_l2 94.862 / tke 78.782 / mvpe 93.985 / time 92.31 / **sps 42.110** → **81.33** (81.35 on
the full-split proxy). `sps` +0.95 and `tke` +0.61 carry it.

**`cpu_smoke_submission.py` failed the first build** on permutation invariance (1.91e-04
against 1e-4) where `validate_strict.py` passed it. Isolated:

| | repeat | permute, TF32 on | permute, TF32 off |
|---|---|---|---|
| w96 FNO ×3 (fused) | 0.00e+00 | 9.10e-06 | 5.07e-07 |
| U-FNO ×3 | 0.00e+00 | **1.80e-04** | **8.64e-07** |

Repeats are bit-exact, so this is **not** nondeterminism: **TF32 tiles a GEMM/convolution across
the BATCH axis**, so a window's position in the batch decides its accumulation order and a
10-bit mantissa makes that visible. The U-FNO's 3×3 convolutions carry ~20× more of it than the
FNO's 1×1 + FFT path, which is why every previous package passed.
`torch.backends.{cuda.matmul,cudnn}.allow_tf32 = False` now ships in **both tracks**. Free
(round 11: TF32 is +0.05 ms and no subscore change past 1e-5), and it removes a real dependence
of our score on the evaluator's batching order.

---

## 7. Recommendation for the next two slots

1. **Track 2 → `track2_r15_ufno3_a060.zip`.** Predicted **81.33** (+0.32 over the account's
   best), it is a clean A/B against a package already on that board — the members differ in
   architecture and nothing else — and it is the round's strongest evidence.
2. **Track 1 → `track1_r15_w96x8_a060.zip`.** Already scored **81.019** on Track 2; Track 1 is
   still standing at 80.835. This costs nothing to run and is the third cross-track identity
   check. (If a bolder pair is wanted, send `ufno3` to both; the pair then measures the
   track-to-track mvpe offset at the new operating point instead.)
3. **Do not** submit anything to test a timing draw, and compare by subscore delta.

**Then, in order of measured value:**
* **Vendor `ViT2d` and `AFNO2d` and ship the cross-family trio.** U-FNO + ViT + AFNO at one
  seed each matches U-FNO ×3 (`sps` 44.85 vs 44.82) and costs the same 3 invocations. It is
  a *free* +0.03 today and, more importantly, the diversity axis is far from exhausted — the
  three families fail differently by construction.
* **int8 packaging.** Measured free (±0.05 % of error) and halves the archive. It is what makes
  8 L8 members, or 8 U-FNO members, or a mixed package fit at all.
* **A `StackedUFNO`.** Fusing 3–4 U-FNO members into one invocation is worth +0.09…0.13
  `final` and is pure engineering.
* **Run the interior Re-band k-fold on U-FNO.** It is now the only instrument that has
  predicted a board move correctly.
* **L8 all-data members** (training as this was written) as a fallback package: 4 in one
  invocation is +0.009 `final` against what is on the board, and its `sps`/`tke` shape is the
  one the board pays for.

**Closed this round, do not revisit**: `--phases 4` (null at matched steps on a model that no
longer underfits), more retained modes, mixing dropout levels, low-rank factorization of the
spectral tensor, plain U-Net and dilated-ResNet trunks.

---

## 8. Standing method rules this round earned or re-earned

* **A prediction instrument needs an identity control.** Predicting a package against its own
  board round must return zero on all five subscores. Two bugs, one of them a full −1.2 sps,
  were invisible without it.
* **A family is only shippable once its class is INSIDE the archive.** `models_alt.py` is not
  in the zip and `_fno.load_any`'s repo-import branch raises in the evaluation container by
  design. `UFNO2d` is vendored, and `verify/diff_ufno.py` checks the two definitions are the
  same model (bit-exact) — the same class of failure as round 12's silent member drop in
  `fuse()`, which was a wrong answer that did not crash.
* **Read the archive size before trusting a member count.** The latency table cleared eight
  width-96 members three hours before the build discovered they were 456 MB. The byte budget is
  one formula (`4·width²·my·mx·n_layers`) and should be evaluated in the plan, not the build.
* **Rank arms by the subscore the board pays for.** The screen said this package would improve
  rel-L2 by 1.14 %; the board made it 0.65 % *worse* and paid +1.013 `sps` anyway. Three of the
  last four board rounds have been decided by the error TAIL, and the AoA proxy has now got the
  sign of a rel-L2 move wrong three times.
* **Run BOTH submission gates.** `validate_strict.py` measures order-dependence as a *relative*
  drift of the mean field and passed the broken package at 1.1e-05;
  `cpu_smoke_submission.py` measures max|diff| absolutely and failed it at 1.91e-04. They are
  not redundant.
