# Retrain drift: how far the package's score moves when the recipe is trained again

2026-09-23, Ilse. Everything here is paired against a bit-matched control on the same instrument
(`screen_eval.eval_set`, ALPHA 0.60, `--band-p 0.45`, band refit per member set, fit on `val_seen`,
reported on `val_unseen` full 336 windows and on the AoA 15-20 proxy). No training in any of the
scoring runs; the checkpoints come from `checkpoints_r34` and from four F-FNO members retrained
under `REPRODUCE.md` recipes.

**Why this number exists.** The organisers retrain our recipe from scratch on a private set at a
novel angle of attack and Reynolds number. Everything we report from one set of weights carries an
unknown seed component, and until now nobody had measured it at package level.

## 1. What was measured

| perturbation | full 336 | AoA 15-20 | run |
|---|---|---|---|
| four F-FNO members reseeded, inside the shipped six | **+0.0102** | **+0.0260** | 20260922-172233 |
| the same four scored alone (no ViTs) | +0.0157 | +0.0414 | 20260922-172233 |
| one F-FNO member reseeded, mean of four draws | **+0.0033** | **+0.0073** | 20260922-203508 |
| the four single-member draws | -0.0004 .. +0.0083 | +0.0011 .. +0.0125 | 20260922-203508 |
| one ViT member reseeded, scored alone (K1) | +0.0463 | +0.0403 | 20260922-173942 |
| four F-FNO members reseeded, scored alone (K1), n=4 | +0.0203 +- 0.0081 | | 4 runs |

Instrument self-checks passed in every run: `M5_ship` 82.3600 / 79.7964 against the ledger, and
`A6_allnew` 84.3131 reproduced bit for bit between two independently submitted jobs.

## 2. Member perturbations are additive at package level

Pre-registered before the run (`#battleground-0921:a9e96e44`): additive if the residual is within
0.005, non-additive above 0.010.

```
                 sum of four single swaps    all four at once    residual
full 336               +0.0125                   +0.0102          -0.0023
AoA 15-20              +0.0293                   +0.0260          -0.0033
```

**Both venues pass.** This licenses something the team had been doing without warrant all week:
extrapolating a single-member floor to a member-set change. The price is a sub-additive term near
0.003 in the same direction.

## 3. The floor and the signal both depend on the venue, by similar factors

The same change reads 2.5-2.6x larger on the reported venue than on the full split (F4 +0.0414 vs
+0.0157; A6 +0.0260 vs +0.0102). Corin and Devi separately found the seed floor varies ~2x across
venues. So signal-to-noise is not wildly different between venues, but **any figure quoted without
its venue is unusable**, and that applies to the floor and to the effect alike.

A measured dilution factor falls out of the same pair: putting the four reseeded F members into the
six-member package moves the reading from +0.0157 to +0.0102, a factor of 0.65. Unchanged members
dilute by share.

## 4. Freezing the band versus refitting it: indistinguishable on F-member arms

`reseed-one` scored every set twice, once with the band refit per set and once with it frozen from
the shipped six. Both passes call the same `fit_band.py` on the same fitting set; the only
difference is whose errors fit the band.

```
sd(refit) / sd(frozen)     full 0.953     AoA 15-20 1.000
freeze cost (refit - frozen)   full +0.0009 [+0.0005, +0.0014]    AoA -0.0002
```

This replicates Corin's paired n=3 result (ratio 1.01) and Fen's bias measurement (+0.006) on a
third venue, at an order of magnitude below Fen's figure. **Choose refit or freeze on process
convenience; the line is closed.**

## 5. The reading is insensitive to the assumed board stretch

Recomputed from saved subscores at several values of the board difficulty factor `f` (error channels
only; the sps channel cannot be re-stretched from a stored scalar, so it is bracketed at its 1.39
value and at `sps_local`, which agree to 0.002):

```
                  f=1.00   f=1.09   f=1.11   f=1.39   f=1.55
A6 full          +0.0090  +0.0093  +0.0094  +0.0102  +0.0106
A6 AoA 15-20     +0.0229  +0.0237  +0.0239  +0.0260  +0.0272
```

Sweeping `f` across every value in circulation moves the reading 1.18x; changing venue moves it
2.5x. **`f` is not load-bearing for difference-type readings** whose delta sits in the three error
channels. It says nothing about an arm whose delta sits entirely in sps.

Reusable: `wfinal` rebuilds exactly from four saved scalars, so any reading can be moved to another
`f` after the fact with no GPU. `score_error(e) = 100/(1 + e/2)`, verified against the board's own
rel subscore; `final = final_v9.W0 @ [se(f*rel), se(f*tke), se(f*mvpe), T_FIXED, sps]`. Reconstruction
matched the stored `wfinal` to 1e-4.

## 6. Two instrument notes that affect anyone using `screen_eval`

**(a) It scores a symmetric interval; the package deploys an asymmetric one.**
`submission.py:583-586` builds `lo = min(c-h, pred)`, `hi = max(c+h, pred)` with the centre at
`y(alpha=0)`, so the deployed width is `h + max(h, |d|)` and coverage is
`-max(h,d) <= e0 <= max(h,-d)`. `screen_eval.score_block` uses `|pred - target| <= half` and a width
of `2h`; the file contains no reference to the centre. At shipped widths this binds on 0.6-0.7% of
elements across four venues and moves a row by -0.0005, so shipped-width readings need no
recomputation and difference-type readings are doubly insulated. It does bite ceiling calculations,
which push `h` toward `|E|` and bind 13-17% of the time — and `sps_local` feeds several of those
chains.

**(b) Bare coverage is not containment.** `CLAUDE.md:165` defines `sps = 100 * mean_i(a_i) *
containment` with `a_i` built from rel/tke/mvpe alone. `a_i` is therefore band-independent and
containment is the residual, which necessarily carries the `exp(-2h/sigma)` width discount. The
fraction of truth inside the band is a different quantity. `prep()` already returns the per-sample
`a_i` vector, so the true `a_bar` and containment are one line; `screen_eval.score_block` now emits
`cover` (bare, flagged as not containment), `a_bar` and `containment` under those names.

## 7. The gate this supports, written as an interval

The paired spread of a single-member swap is the honest divisor for "did this arm move relative to
its own control", and it is nineteen times smaller than the unpaired seed floor that was briefly
used for the same purpose:

```
paired sd (reported venue, n=4)   0.0049    95% chi-square interval [0.0028, 0.0183]
2.8 x sd                          0.0137    interval                [0.0078, 0.0512]
unpaired reported-venue floor     0.093  -> 2.8x = 0.26
```

The lower ends of the two readings of the paired figure agree; the upper ends differ 2.6x, and for a
veto gate the upper end decides. **Quote it as `[0.008, 0.051]`**: below 0.008 certainly fails, above
0.051 certainly passes, and in between is undecidable at n=4. The unpaired 0.093 remains correct for
a different question — whether a result survives a retrain — and the two must not be substituted for
one another.

A window bootstrap answers yet another question: it measures evaluation sampling noise, while the
paired spread measures training noise. A gate judging a retrained arm needs the larger of the two.

## 8. Still open

The ViT half. One ViT member reseeded and scored alone drifts +0.0463 against the F members'
+0.0203 +- 0.0081 — a ratio of 2.3, not the ~7 that "96% of package seed variance sits in the two
ViTs" would require. If that survives the second ViT, the package-level sigma ratio is near 1.9
rather than 5.0 and the both-halves drift lands near 0.019 full / 0.049 on the reported venue, which
is 8-20% of the +0.244 cut rather than 20-50%. Recorded before the measurement exists, with the
pre-registered rules unchanged. `drift-full` is written and waits only on a staging path: the two
reseeded ViT checkpoints are 58,714,383 bytes each against a 50 MB payload cap, and there is no
lossless way around it (no optimizer state to strip, deflate saves 5-10%, `/work` is not retained
between jobs, and fp16 would break the bit-match self-checks).

## 9. What I withdrew while producing this

Kept here because the withdrawn versions circulated and someone may still be holding one.

- **An ensemble-value pricing of the F-group size**, concluding the member axis was closed on
  Track 2. Every member in it passes `--all-data` and was scored on `val_unseen`, which it trained
  on. The bias inflates in-sample ensembling, so the accuracy cost of dropping a member was
  overstated and the net may be positive. What survives has no accuracy term: cutting the F group
  from four members to one frees at most 0.93 ms, short of the rivals at 3.4-4.8 ms.
- **"Bram's `58.4*(dC)^2` is inapplicable here."** That rested on reading bare coverage as
  containment. Recomputed properly, `dC = +0.000130` shares a sign with the score and the bound is
  structurally applicable; the quantitative mismatch survives but is now an order-of-magnitude
  estimate, not a measurement.
- **A caveat that the four single-member deltas share a control and are therefore correlated.** Each
  swap replaces a *different* shipped member, so they are approximately independent; Corin showed the
  shared control is the subtracted term and cancels to first order.
- The toolchain alternative for "all four deltas positive" is excluded by measurement: the 09-22
  reproduction was same-recipe, same-seed, toolchain-only, and its seven F-FNO stages agree to
  within 0.0002 — 35x below the effect at issue.
