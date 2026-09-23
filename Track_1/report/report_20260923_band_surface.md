# The shipped band/compose surface, audited out of sample (2026-09-23, Ilse)

Five shipped knobs, one instrument, one venue, one pre-registered rule, **zero GPU**. Four are at
their optimum or within noise of it; one is unresolved. The point of the report is not the four
negatives — it is that they are negatives *with a measured price for removing them*, and that the
venue they were measured on is a systematically loose proxy for the one that decides the prize.

## Instrument

@Emre's shared twin export (`pitzer-requests/emre/twin_export/results/export_unseen.npz`): the r34
held-out twins (`p2_ffno_l8_sim_s1..3`, `arch_vit_d384`, `arch_vit_cmb3sim`), 336 windows, with `y`,
the error field at ALPHA 0 and 1, and the sigma head. Compose is linear in alpha, so

    e(a) = (1-a)*e_a0 + a*e_a1 ,  pred = y + e(a) ,  centre = y + e(CENTRE_ALPHA)

is exact — every question below is numpy on a laptop. **No band/interval question in this family
needs a GPU job.** I had a cluster job built and queued for what turned out to be a 15-minute
local script.

* **venue** band fitted on AoA<15 (210 windows), scored on AoA>=15 (126) — the reporting slice
* **scoring** the DEPLOYED rule, `lo = min(c-h, pred)`, `hi = max(c+h, pred)`
  (`submit_fno/submission.py:586-592`), at the shipped `centre_alpha = 0.0` read out of
  `sub_candidates`' `band.npz`. Every other local instrument scores symmetric `pred +- h`.
* **rule, fixed before each sweep** a value replaces the shipped one only if it wins by
  `>= 0.03 final` on the full venue **and** in all three leave-one-trajectory-out views. If the
  argmax moves between views the knob is reported UNRESOLVED and no value is named.

## Result

| knob | shipped | out-of-sample argmax | d final | LOO | verdict |
|---|---|---|---|---|---|
| `band_p` | 0.45 | **0.45** | +0.000 | 2/3 | shipped is optimal — closed |
| `centre_alpha` | 0.00 | **0.00** | +0.000 | 3/3 | shipped is optimal — closed |
| `band_smooth` | 3 | 5 | +0.0055 | 3/3 | far under the 0.03 bar; 3 and 5 indistinguishable — closed |
| `inflate` | 1.39 | 1.26 | +0.002 sps ~ 0 | — | top flat over 1.15–1.55 — closed |
| `ALPHA` | 0.60 | 0.80 | +0.0425 | **1/3** | direction 3/3, **magnitude unresolved** |

**What removing each one would cost** — the number that says what these knobs are actually worth:

    band_p 0.45 -> 0          -0.527 final
    centre_alpha -> 0.60(=pred) -0.160 final
    band_smooth 3 -> 1        -0.109 final
    inflate 1.39 -> 1.00      -0.144 sps ~ -0.036 final
    all four                  ~ -0.8 final

## Two facts that changed how the rest should be read

**1. A regulariser cannot be priced in sample.** The 3-box smooth reads **-0.984 sps in sample**
and **+0.474 sps out of sample** on the same venue. I nearly published the in-sample number as a
finding; what caught it was running the 2x2 control (ratio x smooth), not caution. The sigma-ratio
term behaves the same way: +0.803 in sample, **+2.106 sps (+0.527 final)** out.

**2. Local instruments understate the shipped package.** They score symmetric `pred +- h`; the
package deploys the decoupled centre. On this venue that is **+0.463..+0.474 sps at fixed band**
(two independent routes: @Emre's and mine, 2.4 % apart) and **+0.641 sps = +0.160 final** once the
band is refitted around the centre, as `fit_band.py --centre-alpha` does. The gap is always
positive but venue-dependent, so it **largely cancels when comparing two packages and does not
cancel when predicting a board level**. This is a third independent confirmation of @Bram's
decoupled-centre arm (his +0.61 sps held out).

## Where the remaining `sps` actually is

`sps = 100 * a_bar * containment`; `a_bar` depends only on the three error metrics, so no interval
work can touch it. Decomposing the fit -> reported gap:

    gap 11.691 sps = 2.924 final
      a_bar   25.2 %   untouchable by any band
      contain 74.8 % , of which band TRANSPORT is 0.462 sps = 0.116 final = 4.0 % of the gap
                       (in-class oracle, in sample, so a generous upper bound)
      remainder ~70.9 % is out of reach OF THE SHIPPED FAMILY -- how much of it a richer band
                        could reach was NOT measured for this table; see the correction below

> **CORRECTION, appended the same night.** Where this report says the interval axis is capped at
> 0.116 `final`, that bounds only the shipped *family* -- element base x per-window
> `sigma^band_p` x 3-box smooth. A learned width head emits `h` per element per window and is a
> strictly richer family, so it is **not** bounded by that number, and reading my figure as a
> ceiling on ALL interval work is wrong by a factor of 33. The bound that holds for every family
> is pointwise: one element's scored quantity is `exp(-2h/SIGMA) * 1[|E| <= h]`, maximised at
> `h = |E|`, so `sps_max = 100 * sum_i a_i * sum_e exp(-2|E_ie|/SIGMA) * scored / n_scored`.
>
> | venue | shipped | shipped-family refit | **absolute** | shipped -> absolute |
> |---|---|---|---|---|
> | fit AoA<15 | 58.322 | 58.939 | **70.877** | +12.555 sps = **+3.136 final** |
> | REPORTED AoA>=15 | 46.631 | 47.604 | **62.218** | +15.587 sps = **+3.894 final** |
>
> So the reported venue has **+3.894 `final`** of interval headroom for a richer band, not 0.116.
> @Fen's "width-head corr 0.565 -> 0.70 is worth +0.44 `final`" sits at 11 % of that bound and was
> never in conflict with this report -- my scoping was the conflict. The absolute column is a
> POINTWISE ORACLE: it needs the target it is scored on, can never be shipped, and says "no head
> can beat this", not "these points are waiting to be collected".
>
> Lesson: **a reported upper bound must name the FAMILY it constrains in the same sentence.**
> I was the first person misled by my own -- and it is the second time tonight I mis-scoped a
> bound, after reading `sps9`'s "upper bound on what any band-fitting procedure can do" as
> covering the shipped band when it covers element-only bands. Script: `tools/abs_ceiling.py`.

`plan_20260917.md` §11.2 says the same thing in words — *"the 16-point sps loss at a novel angle is
not a width problem (the a-factor and where the error sits, not its scale)"* — and this is the
first time it has been given numbers.

## The limitation that matters more than any row above

Every number here is on the **reported venue**, which is **not** an AoA holdout: the twins trained
on other AoA 15/20 trajectories. On the one axis where the two venues can be compared they
**disagree in direction**: narrowing 1.39 -> 1.26 is worth **+0.000 final** here and
**-0.056..-0.087 final** on §11.2's AoA-20 holdout. The difference is angle-naivety, so on the
width axis the reporting venue is a **systematically loose proxy**.

> **RETRACTED, and the correction is the point.** An earlier version of this paragraph said run
> `20260922-230505` had established that the AoA-holdout checkpoints no longer existed and that
> §11.2 and §11.3 were therefore unreproducible, and priced a ~14.5 GPU-h rebuild against that.
> **That was wrong.** The checkpoints had always been at
> `/fs/scratch/PAS1187/qwu/tmp/realpde/screen/`; the gateway mounts only `~/envs`, the conda
> environments and `~/RealPDE`, so scratch is simply invisible from inside a job, and
> `~/RealPDE/screen` never existed. **I read a mount boundary as an absence** — and I had written
> the correct caveat ("I can only prove it is absent from the paths we can see") before stating
> the opposite as a headline. The rebuild was voided, never submitted, and @m4tr1x copied the
> files to `RealPDE/incoming/screen/` instead.

## What the OOD instrument actually said (run `20260923-023353`, 10 checkpoints)

Read against a pre-registration published before the job ran. `tools/read_aoa.py`.

**§11.2 and §11.3 reproduce exactly** on those checkpoints: held-out `sps` at inflate 1.39
**38.399** against §11.2's 38.40, fit split **54.444** against 54.44, `wfinal` **79.692** and
**80.943** against §11.3's 79.692 and 80.943, and the seed-2 pair **79.727 / 80.708** matching its
replicates. So the section this report leans on is reproducible, and was the whole time.

**The report's central claim is confirmed — and the conclusion it threatened survives for a better
reason.**

| arm | 3-pt vertex | argmax |
|---|---|---|
| `aoa20_ffno_s1` (no-sim — the checkpoint §11.2 actually used) | **1.554** | 1.55 |
| `aoa20_ffno_s2` (no-sim, seed 2) | **1.569** | 1.55 |
| `aoa20_ffnosim_s1` (the SHIPPED sim recipe) | **1.366** | 1.39 |
| `aoa20_ffnosim_s2` (shipped, seed 2) | **1.385** | 1.39 |

The two sim seeds' vertices agree to **0.019** (the pre-registered bar was 0.25), so the vertex is
resolved at **1.376**. The configuration moves the optimum by ~0.19: **§11.2's headline is wrong
about its own data — its vertex is 1.55 — and right about the shipped package, whose vertex is
1.376 ≈ the shipped 1.39.** The right answer from the wrong measurement. **The shipped value does
not move.**

**The pre-registered rule fired on all ten checkpoints.** In-class reachable — the whole shipped
construction refitted on the scored cell with inflate swept — came in at **0.567–1.184 sps
(0.14–0.30 `final`)**, under the 2-sps threshold at which I had committed to stop working on
intervals within this family. Because it bounds the *family*, `band_p`, `band_smooth` and
`centre_alpha` need no separate holdout sweeps either: none of them can exceed 0.30 `final`.

**But the axis is not closed, and the headroom transports.** The pointwise absolute bound at the
novel angle is **+15.65…+15.95 sps = +3.91…+3.98 `final`**, against +3.894 on the reported venue.
By output granularity (`tools/ood_granularity.py`, all oracle bounds, symmetric path):

| the head emits | `sps` | vs the rung below |
|---|---|---|
| one number (global) | 39.169 | — |
| one per lead-time index | 39.235 | +0.017 `final` |
| one per window | 39.600 | |
| one per (window, frame) | 39.774 | +0.151 `final` cumulative |
| a shared per-element field | 42.078 | +0.727 `final` cumulative |
| **one per element per window** | **57.083** | **+3.748 `final`** |

So every scalar-output head is capped at +0.15 `final`; a better *universal* field adds +0.16 more,
the shipped base already holding 98.5 % of that class; and **the whole remaining +3.75 requires the
head's spatial field to vary with the input window.** A per-frame profile is worth
**−0.0034 containment** — inside the gate, indistinguishable from zero — because the shipped base
already carries the lead-time profile, and carries it *sub*-proportionally, which is correct:
the scored quantity `exp(-2h/σ)·1[|E|≤h]` does not want a band proportional to `|E|`.

**Two things no band touches.** `a_bar` is 34–36 % of the fit→held-out gap, and at a novel angle
its channel priority is **rel 43 % > tke 39 % > mvpe 18 %** — a *different order* from
in-distribution cells, where tke leads. And sim's gain at the held-out angle is **64 % containment,
36 % accuracy** (both seeds agree to 0.4 pp), so "accuracy is the model's job, containment is the
band's" is false: a model change lands mostly on containment, where the band is already at its
family ceiling. sim does not consume the width head's room — floor and ceiling rise together and
the span stays 15.6–16.0 sps.

**The limit on all of the above, which the bounds cannot supply themselves:** @Devi measured a
*fitted* 81,920-parameter field — exactly the per-element-shared rung — going **+0.032 in-sample to
−0.555 held out**. Every rung here bounds *can it be done*, never *can it be got*; a head that
*predicts* the field from the input is not directly covered by that result, but the burden of
showing it generalises rests on the head, and nothing in this table helps it.

## Caveats carried by every number here

The AoA-20 holdout has **16** trajectories (per-trajectory pointwise ceilings 44.1–63.9); the
"three trajectories" limit below belongs to the REPORTED venue only. Everything from
`aoa_holdout_eval` and from the dumps is on the **symmetric** path (`absE = |P−Y|`, no clamping),
which is ~+0.47 sps below the deployed clamped path — differences survive that, absolute values
must not sit beside a board row. My interval scorer was checked against the repo's `sps_score`
(symmetric) and the kit's `aggregate_sps(lower=,upper=)` (deployed), agreeing to ~1e-10 on both.

Five twins, not the shipped four F + two ViT · AoA>=15 is three trajectories, so the replication is
leave-one-**trajectory**-out, not multi-seed · the in-class oracle is in sample and can never be
shipped · these `sps` compare across arms of this screen only, never against a board round.

Scripts: `tools/{bandp_sweep,centre_sweep,smooth_sweep,alpha_sweep,alpha_loo,abar_ceiling,
transport_2x2,ratio_control,deployed_interval,fit_absorption,board_abar}.py` in the agent workspace.
