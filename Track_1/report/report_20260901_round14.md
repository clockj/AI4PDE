# Round 14 (2026-09-01) — sim closed, history closed, and the model turns out to underfit

One session, ~6 GPU-hours of a 200-hour budget. Nothing submitted. Two directions were
proposed, gated and closed; a third was found while closing them and is now the plan.

The session began from the user's reading of the record: rounds 11–13 bought nothing
measurable, every arm was an ~18-minute variation on the same 0.90M FNO, and the attempts were
not aggressive enough. That reading was right, and the reason turns out to be sharper than
"not aggressive enough".

---

## 1. Sim2real: gated in 4 GPU-hours instead of 172, and closed

### 1.1 Gate 0 — the domain gap, measured PAIRED for the first time

Sim is a **strict superset** of real's condition grid: every real `(Re, AoA)` has an exact CFD
twin, and sim adds Re 15225, Re 27975 and every hole real leaves. Nothing in the project had
used that, and it makes the gap measurable per condition instead of per distribution
(`Track_1/sim_map.py`, 12 matched pairs, 32×64, sim speed-matched to its twin):

| quantity | sim vs real | our model, for scale |
|---|---|---|
| time-mean field, rel-L2 | **0.400** | 0.045 |
| per-pixel std field, rel-L2 | 1.457 | — |
| … with amplitude divided out | 0.603 | σ head 0.329 |
| fluctuation energy ratio | 2.9× at k≈1, 5.6× at k≈10, 0.55× at k≈35 | — |
| transverse turbulence intensity | 0.263 vs 0.115 (2.28×) | — |

**Sim is wrong about the mean field and the amplitude by margins several times the model's own
error**, which explains round 5's `+0.00` mechanistically rather than as a null. Two candidate
repairs died here:

* **It is not the field of view.** Real has pixels that are identically zero (PIV FOV and the
  airfoil body) and sim has none, which in a *relative* L2 is pure numerator. But the zero
  fraction is **3.3%**, not the 9.9% CLAUDE.md records, and masking closes only **7%** of the
  mean gap.
* **It is not resolution.** The energy ratio is **non-monotone in k** — it peaks at 5.6 near
  k≈10 and falls below 1 above k≈32. A low-pass has a monotone transfer function, so no filter
  can produce this. It is 2D-CFD over-prediction of large-scale unsteadiness.

### 1.2 Gate 1 — the domain map works; the clever loss was the mistake

`Track_1/unet/simreal.py` maps sim into real's measurement space (speed match from the fitted
freestream law `mean_u = c(AoA)·Re`, amplitude match to the twin, then the twin's zero set
painted on) and `train2.py` gained `--sim/--lam-sim/--sim-loss`. `--sim-loss shape` normalizes
both prediction and target by their own time-mean and per-pixel std, so the term is blind to
exactly the two quantities sim is wrong about.

It fails in both regimes, monotonically in λ:

| arm | AoA 15-20 `wfinal` | full split | `rein` rel-L2 | `relo` | `rehi` |
|---|---|---|---|---|---|
| control | **81.850** | **83.816** | 0.0719 | 0.1160 | 0.1189 |
| shape, λ 0.3 | 81.651 | 83.660 | 0.0731 | **0.1451** | 0.1197 |
| shape, λ 1.0 | 81.139 | 83.163 | — | — | — |
| **full, λ 0.3** | — | — | 0.0718 | **0.1145** | **0.1167** |

The mechanism is visible in the numbers: dividing by a per-pixel temporal std that is near zero
in the freestream amplifies quiescent regions into noise, and `relo` MVPE goes 0.1110 → 0.1626
(+46%). **Being blind to what sim gets wrong is worse than matching it.**

The naive `--sim-loss full` — round 5's arm, but now with a domain map under it — looked
mildly positive out of distribution, and at three paired seeds and 3-vs-3 it looked strongly
positive (`relo` `wfinal` 82.618 → 84.240, rel-L2 −21.4%; `rehi` +0.45; `rein` flat).

### 1.3 That result was RETRACTED, re-measured, and stands

The map was leaking the held-out data.

`load_sim_mapped` sets each sim trajectory's mean speed, fluctuation amplitude and FOV mask
from its **real twin**, and it built the twin table from every file in the real directory
regardless of the split. Under `re_split` that is **26 held-out trajectories** handing all
three quantities to training — and **mean speed is exactly the quantity the model gets worst
at an unseen Reynolds number**, so the leak landed precisely where the gain was claimed.

`real_exclude` is now threaded from `train2.py --exclude`, the named trajectories are dropped
from the twin table, and orphaned conditions fall back to the freestream law and the nearest
surviving mask — the path sim's 18 twin-less conditions already used. Under `re_split` that is
**44 of 100** conditions placed by the law. The cache key carries a hash of the exclusion so
the two variants cannot be confused. An **all-data** run has no held-out set and leaks nothing,
so a package build is unaffected; the controls use no sim, so only the three `full` arms were
rebuilt (`simE`, which also reports the contaminated ensemble alongside for the record).

**The rule this earns, which is more valuable than the arm:** *any preprocessing that reads
statistics from the real dataset must be told the split.* A domain map, a per-condition
normalizer, a mask — each is a training-time read of data that may be held out, and none of
them is covered by `--exclude`, which only filters the trajectories the **loss** sees. This is
the same class of error as round 13's struck transductive conditioning, arriving through a
different door.

**And the re-run vindicates the arm.** Leak-free, 3-vs-3:

| set | control ×3 | +sim ×3 | Δ |
|---|---|---|---|
| `rein` (interior control) | 84.470 | **84.471** | +0.001 |
| `relo` (Re below the range) | 82.618 | **84.307** | **+1.689** |
| `rehi` (Re above it) | 79.180 | **79.652** | **+0.472** |

`relo` rel-L2 0.1216 → **0.0949, −22.0 %**, against −21.4 % with the leak in place — so the
leaked scalars contributed nothing and the gain is sim's Reynolds coverage, which is what
CLAUDE.md always said sim was for. Per model the low-Re seed spread tightens **3.2×** (σ 0.0132
→ 0.0041), the same variance collapse `--norm self` bought in round 13 but without its TKE
penalty. λ 0.3 and 1.0 tie, so the plateau is flat.

It also **deploys free**: a `--sim` checkpoint is architecturally identical to a shipped member
(the flag changes only the training loss), and `_fno.fuse` puts it in the same single
invocation — verified. So this direction costs no latency at all.

---

## 2. Trajectory history: legitimate in Track 2, and worth nothing

A per-window DMD fit on the protocol's 20 frames scores rel-L2 **0.1424** on `val_unseen` —
worse than the trivial input mean (0.1338), monotonically worse with rank. That is not an
implementation failure: **20 frames is under a third of a 60–75 frame shedding period**, so no
per-window fit can identify an oscillation it never completes. The same code given more history
of the same trajectory (`Track_1/ctx_probe.py`, reconstruction verified bit-identical to the
stored window at 0.0e+00):

| context (frames) | 20 | 40 | 60 | 80 | 120 | 160 | 200 |
|---|---|---|---|---|---|---|---|
| DMD | 0.1424 | 0.1216 | 0.1170 | 0.1153 | 0.1104 | 0.1075 | **0.1059** |
| mean of that context | 0.1338 | 0.1249 | 0.1201 | 0.1201 | 0.1179 | 0.1174 | 0.1171 |

and for the target window's time-mean alone: the input window's own 20 frames give **0.1022**,
every frame delivered so far **0.0783** (−23%), the whole trajectory including the future
0.0739. **A parameter-free method with history lands where the trained 8-member ensemble sits
with 20 frames.**

**This is a Track-2 question, not a Track-1 one, and the distinction is not a technicality.**
The official `local_eval.py` holds ONE model object, calls `ttt_step` in strict time order, and
calls `reset_ttt_state()` at trajectory boundaries — a hook that exists only because
within-trajectory state is expected to persist. Frames the stream has already delivered are
from *before* the forecast time, which is the ordinary situation of any deployed forecaster.
That is the opposite of the leak CLAUDE.md records, which reads window *k+1* — window *k*'s own
answer, bit for bit. Round 11 ruled long context out on Track-1 grounds (`predict()` gets one
window; reaching outside costs permutation invariance) and it was never revisited for the track
whose protocol is built for it.

**And it buys nothing** (`Track_1/hist_blend.py`, 3-member held-out ensemble, band refit for
every arm):

* **Blending toward the history mean is monotonically worse** — `wfinal` 84.110 → 84.088
  (β 0.15) → 83.952 → 83.632 → 82.436 (β 1.0). In distribution the model's own time-mean is
  **0.0449** against history's 0.0783, so there is nothing to gain.
* **DMD-200's per-window error correlates with the model's at +0.897.** Ten times the input,
  and it is nearly redundant. Every blend weight loses.
* A **global fluctuation gain** peaks at 1.05 for +0.040 — round 6 called this noise under
  weights pricing `tke` at 0.244, and it is still noise under 0.097.

The Track-2 submission carries the running-history mean behind `HIST_BETA = 0.0`, gated to
out-of-range windows, so the step is bit-identical until there is a reason to switch it on. The
gated case is untouched by the above: it rests entirely on the model's mean degrading at unseen
Re, which `re_split` can test.

---

## 3. Where the error actually is — and a correction to the binding-constraint claim

`Track_1/gap.py` splits the error exactly and orthogonally: `‖p−y‖² = T‖pm−ym‖² + ‖pf−yf‖²`,
and the fluctuation part again by what an oracle per-pixel amplitude would buy, with
`ρ = ⟨pf, yf⟩/(‖pf‖‖yf‖)`. On the shipped recipe's held-out twin:

| set | total | mean | fluct | ρ | decorrelation floor |
|---|---|---|---|---|---|
| `val_seen` (in-sample) | 0.0749 | 0.0382 | 0.0645 | 0.634 | 0.0629 |
| `val_unseen` | 0.0806 | 0.0452 | 0.0667 | 0.609 | 0.0645 |

* **The seen/unseen gap is 1.10×, not the 2.1× on record.** CLAUDE.md's "the binding constraint
  is 74 trajectories, not the model" comes from round 1's 0.049/0.084 on an unregularized
  U-Net. On the current recipe the model barely overfits — which is why every data-side arm has
  been null.
* **Amplitude is finished.** The fluctuation error 0.0667 is within 3% of the 0.0645 an optimal
  global rescaling would give, and the oracle amplitude correction is *negative* (−14.8%).
* What is left is **ρ**, and ρ is essentially the same in sample as out. That is neither an
  overfitting signature nor a data-starvation signature.

---

## 4. The finding: the shipped model UNDERFITS

If ρ is the same in sample as out, either the task is intrinsically stochastic at this horizon,
or the model cannot fit it at all. Those are distinguishable: train on **only** the 8
`val_unseen` trajectories and then score `val_unseen`, which makes it purely in-sample, and push
capacity and schedule.

| model | rel-L2 | mean | fluct | ρ |
|---|---|---|---|---|
| w24, 150 ep | 0.0774 | 0.0375 | 0.0679 | 0.588 |
| w64, 150 ep | 0.0539 | 0.0218 | 0.0496 | 0.798 |
| w64, 400 ep | 0.0403 | 0.0145 | 0.0351 | 0.903 |
| **w96, 400 ep** | **0.0316** | 0.0101 | 0.0268 | **0.945** |
| shipped w24, held-out | 0.0809 | 0.0452 | 0.0667 | 0.609 |

**1. There is no intrinsic 20-frame predictability barrier.** ρ reaches 0.945 and rel-L2 0.0316.
The input window contains the information; ρ = 0.609 is a property of our model. This is the
measured bound CLAUDE.md says does not exist, and it is not where anyone assumed it was.

**2. The shipped w24 reaches ρ 0.588 on eight MEMORIZED trajectories, against 0.609 held-out on
74. It cannot fit even eight.**

Round 8's ladder — w24 82.06 > w48 81.45 > w64 81.36 — added capacity with regularization and
schedule held fixed and scored generalization. That is a real result about *those* runs. The
corollary the project drew from it, that the model is expressive enough and capacity is the
wrong axis, does not follow and is now measured to be false. **The untested cell is capacity
WITH regularization.** It also puts every data-side null back in play — sim, `--phases 4`,
`--aug-scale`, weight decay were all measured on a model that could not fit the data it already
had, and extra data cannot help a model that underfits.

Latency does not block this: round 9 measured a 25× inference-cost increase at ≈1.2 `final`,
and an invocation at ≈0.036, so a w96 costs a few hundredths.

**Confound still open, and being controlled** (`capB`): the 8-trajectory runs have 912 windows
against the full split's 8436, so at equal *epochs* they took 9× fewer optimizer steps. w24 at
400 and 1350 epochs on the same 8 trajectories settles whether its low ρ is under-training or
under-parameterization.

---

## 5. What runs next

* `capA` — the diagonal: w48/w96 × dropout 0.15/0.30 at 400 epochs, held-out, against the
  shipped w24. Scored on the AoA 15-20 proxy, the full split, and `gap.py` so the mechanism is
  visible and not just the score.
* `capB` — the step control above, plus the matched no-dropout arms at w24 and w96.
* `simC` — three paired seeds and a 3-vs-3 ensemble of `--sim-loss full`, which is the only sim
  arm still alive and is inside the noise until then.

Nothing here changes what is staged in `sub_candidates/`. The round-14 pair (Track 1 gated
mean-blend, Track 2 dropout) still asks the best available question of the board and its
Track-1 arm still cannot lose.
