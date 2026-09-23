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
      remainder ~70.9 % needs a better model

`plan_20260917.md` §11.2 says the same thing in words — *"the 16-point sps loss at a novel angle is
not a width problem (the a-factor and where the error sits, not its scale)"* — and this is the
first time it has been given numbers.

## The limitation that matters more than any row above

Every number here is on the **reported venue**, which is **not** an AoA holdout: the twins trained
on other AoA 15/20 trajectories. On the one axis where the two venues can be compared they
**disagree in direction**: narrowing 1.39 -> 1.26 is worth **+0.000 final** here and
**-0.056..-0.087 final** on §11.2's AoA-20 holdout. The difference is angle-naivety, so on the
width axis the reporting venue is a **systematically loose proxy**.

Which makes the finding from the same afternoon the operative one: run `20260922-230505`
(aoa-ceiling PHASE 0, 4 s of GPU) established that `RealPDE/screen` no longer exists and no
`*aoa*.pth` survives anywhere under the project — so `aoa_holdout_eval.py`, `EVALUATION.md:139`'s
*"the ONLY OOD instrument we have, and the one that speaks to the PRIVATE test set"*, **has nothing
to run on**, and §11.2 and §11.3 are not reproducible without retraining. Rebuild priced at
**~14.5 GPU-h** (`REPRODUCE.md:324`: `pz_aoa20_ffnosim` 3:36:45) and built as
`pitzer-requests/ilse/aoa-rebuild`, deliberately **not** submitted.

## Caveats carried by every number here

Five twins, not the shipped four F + two ViT · AoA>=15 is three trajectories, so the replication is
leave-one-**trajectory**-out, not multi-seed · the in-class oracle is in sample and can never be
shipped · these `sps` compare across arms of this screen only, never against a board round.

Scripts: `tools/{bandp_sweep,centre_sweep,smooth_sweep,alpha_sweep,alpha_loo,abar_ceiling,
transport_2x2,ratio_control,deployed_interval,fit_absorption,board_abar}.py` in the agent workspace.
