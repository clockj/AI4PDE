# Round 10 — two linear bottlenecks, and the SPS band reopened

**Date:** 2026-08-08 · **Our board:** 80.115065 · **Target:** 80.4 (another team) · **Nothing submitted.**
Plan this executes: `Track_1/plan/plan_20260807.md`. Constraint from the user: no extra
training data, so the RealPDEBench public Foil release stays out (§9 of the round-9 report).

---

## 0. Summary

Nineteen arms and three shipped changes, together **+0.311** against the 80.115 board:

| change | Δ | § |
|---|---|---|
| `--var-mlp --lift-mlp` + dropping w48 | **+0.112** | 5.4, 5.7 |
| `_fno.fuse()` — 3 invocations instead of 8 | **+0.067** | 4b |
| sigma-adaptive, smoothed SPS band | **+0.132** | 5.9 |

1. **Two linear bottlenecks, not a capacity shortage.** `head_logstd` was a linear probe on a
   24-channel trunk and `lift` a linear 42→24 compression of the whole input window. Fixing
   both is additive, for 10 976 parameters. **`point`, the same shape of layer but *inside*
   the trunk, is a clear loss (−0.379)** — so the rule is *capacity at the boundaries helps,
   capacity in the trunk hurts*, which is round 8's result seen from the other side (§5.8).
2. **The SPS band was never closed** — round 6 closed it against ensemble spread (corr 0.347)
   where the variance head's own sigma correlates 0.551. Two independent gains, +0.132 total,
   with no retraining (§5.9). Both required removing a selection bias that had produced a
   false negative the first time.
3. **Round 9's "TKE is a trap, ≤ +0.6 left" was wrong** — a perfect variance field is worth
   **+7.70** (§1), and the 0.47 "PIV sampling floor" it rested on was never a floor.
4. **But oracles rank directions; they do not forecast gains.** That +7.70 of headroom yielded
   **+0.08** across five arms. The ranking held — σ was the right target and TKE is where both
   architecture fixes paid — the magnitude was never on offer.

Everything else was negative or null: WP2 decimation phases, WP3 AoA reweighting in all three
forms, `--lam-t`, `--detach-var`, `--lam-f 0`, post-hoc sigma blending, ALPHA ≠ 0.75, and width
in both directions. Full table in §6.

---

## 1. The TKE branch, repriced

`Track_1/tke_branch_v9.py`. Held-out 3-seed ensemble, AoA 15-20 proxy, errors inflated 1.39×,
band refit per row, `final` from `final_v9.py` at `time_score` 90.507.

| sigma | rel-L2 | TKE | MVPE | rel | tke | mvpe | sps | **final** | Δ |
|---|---|---|---|---|---|---|---|---|---|
| `head` (shipped, α=0.75) | 0.1082 | 0.4831 | 0.0940 | 93.01 | 74.86 | 93.87 | 35.93 | **78.305** | — |
| `input` (input window's std) | 0.1237 | 0.7833 | 0.0940 | 92.08 | 64.75 | 93.87 | 31.38 | 74.777 | −3.53 |
| `traj-in` (per-traj mean of the above) | 0.1167 | 0.5609 | 0.0940 | 92.50 | 71.95 | 93.87 | 33.33 | 76.990 | −1.31 |
| `traj` (per-traj mean of the TRUE std) | 0.1167 | 0.5604 | 0.0940 | 92.50 | 71.97 | 93.87 | 33.33 | 76.994 | −1.31 |
| k = 0.25 toward oracle | 0.1069 | 0.3831 | 0.0940 | 93.08 | 78.97 | 93.87 | 37.01 | 79.518 | **+1.21** |
| k = 0.50 | 0.1065 | 0.2700 | 0.0940 | 93.11 | 84.20 | 93.87 | 38.40 | 81.046 | +2.74 |
| k = 0.75 | 0.1070 | 0.1426 | 0.0940 | 93.08 | 90.98 | 93.87 | 40.55 | 83.075 | +4.77 |
| `oracle` (the target's own std) | 0.1084 | 0.0003 | 0.0940 | 92.99 | 99.98 | 93.87 | 44.76 | **86.002** | **+7.70** |

Three things to read off it.

**The 0.47 floor was never a floor.** Round 9 argued TKE sits near an irreducible PIV-sampling
limit, so nothing was left. But the head is at **0.483**, and it already beats every
realization-noise baseline that can be constructed from the data: the per-trajectory mean of the
true std field scores **0.560**, and the immediately preceding 20-frame window — a second,
independent realization of the same flow — scores **0.783**. If 0.483 were sampling noise, a
predictor could not be *below* the disagreement between two realizations. It is below because
the target's 20-frame variance is not noise: it is a deterministic function of the trajectory,
and the model predicts the trajectory. The limit here is prediction horizon, not sampling.

**The gain is nearly free in the other channels.** rel-L2 moves 0.1082 → 0.1084 across the
entire sweep from shipped to oracle, and MVPE not at all (it reads the time-mean, which
fluctuation rescaling leaves invariant by construction). Unlike almost every other lever
measured this season, improving sigma costs nothing elsewhere.

**45% of the payout arrives through SPS**, exactly as the round-9 exchange rate predicted:
35.93 → 44.76. SPS carries the second-lowest weight (0.181) and still contributes ~1.6 of the
7.70, because interval containment is far steeper in the error than `100/(1+0.5e)` is.

### Why round 9 got this wrong

The round-9 exchange rate scaled *all four* error channels together by a common factor and
reported the decomposition of a 10% cut: `tke +0.75`. That arithmetic was right. The error was
in the sentence that followed it — reading "TKE moves only 0.75 per 10%" as "TKE is nearly
exhausted," when 10% of 0.483 is a small step down a curve that runs all the way to 0.483. The
floor claim did the rest of the damage. Nothing in the round-9 measurement supported it.

---

## 1b. All three pieces of a prediction, priced

`Track_1/oracle_split_v9.py`. Every prediction splits exactly into three parts, orthogonal
under the time inner product:

    y  =  mean_t(y)  +  sigma(y) · shape(y)          shape has unit temporal std per pixel

`mean_t` is what MVPE reads; `sigma` is what TKE reads; `shape` is the phase-resolved
fluctuation — the part that needs the model to know where in the shedding cycle each pixel
will be 20 to 40 frames out. Handing the model the *target's* value for one piece at a time
prices that piece's entire remaining headroom.

| replaced with the target's… | rel-L2 | TKE | MVPE | rel | tke | mvpe | sps | final | Δ |
|---|---|---|---|---|---|---|---|---|---|
| — (base) | 0.1082 | 0.4831 | 0.0940 | 93.01 | 74.86 | 93.87 | 35.93 | 78.305 | — |
| **σ** | 0.1085 | 0.0000 | 0.0940 | 92.99 | 100.00 | 93.87 | 44.77 | 86.009 | **+7.71** |
| mean | 0.0900 | 0.4831 | 0.0000 | 94.11 | 74.86 | 100.00 | 41.15 | 81.019 | +2.72 |
| shape | 0.0712 | 0.4831 | 0.0940 | 95.28 | 74.86 | 93.87 | 43.35 | 80.251 | +1.95 |
| mean + σ | 0.0902 | 0.0000 | 0.0000 | 94.10 | 100.00 | 100.00 | 51.21 | 88.948 | +10.64 |
| σ + shape | 0.0595 | 0.0000 | 0.0940 | 96.03 | 100.00 | 93.87 | 56.92 | 89.016 | +10.71 |
| everything | 0 | 0 | 0 | 100 | 100 | 100 | 100 | 99.343 | +21.04 |

**σ is worth 2.8× the time-mean and 4.0× the fluctuation phase.** And the two columns disagree
in the most useful possible way: replacing `shape` cuts rel-L2 by **34%** (0.1082 → 0.0712) and
buys only +1.95, while replacing σ leaves rel-L2 *unchanged* and buys +7.71. Chasing rel-L2 is
chasing the wrong number — near e ≈ 0.1 the `100/(1+0.5e)` curve is nearly flat, and TKE plus
SPS are where the response lives.

That is a direct answer to a question this project has circled for several rounds. The
phase-resolved fluctuation is the hardest part of the problem and the place all the
spectral-bias and generative literature points — **and it is worth the least of the three.**
The amplitude envelope is worth the most, and it is the easiest of the three to predict.

---

## 2. What is actually being asked of the variance head

`models_alt.py:FNO2d` and its submission mirror:

```python
self.head_field  = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, CH, 1))
self.head_logstd = nn.Conv2d(width, C, 1)          # <- one 1x1 conv, width=24
```

The field head is a two-layer MLP over 24 channels; the variance head is a **linear map** over
the same 24. Training weights it at `--lam-v 1.0` (log-std MSE, every pixel equal, including
the ~10% v9 does not score) and `--lam-t 0.5` (TKE relative L2, the quantity actually scored)
against `--lam-f 1.0` for the field.

Both of those are now suspect, and each is one flag:

* `--var-mlp` gives `head_logstd` the field head's shape, for 3 408 parameters (0.4% of 0.90M).
* `--lam-t 2.0` reweights toward the scored quantity.

---

## 3. Two post-hoc routes, both closed

Worth recording so neither gets retried.

**ALPHA under the round-9 weights** (`Track_1/alpha_board.py`, now reporting `final` next to
`mean4`; per-metric inflation rel 1.06× / tke 1.13× / mvpe 1.38×, anchored on the round-7
board). Held-out checkpoints, so this is a clean control rather than the all-data sweep that
produced round 7's false positive:

| α | 0.30 | 0.45 | 0.60 | **0.75** | 0.90 | 1.00 | 1.10 |
|---|---|---|---|---|---|---|---|
| `final` | 79.449 | 79.662 | 79.851 | **79.988** | 80.029 | 79.976 | 79.841 |

α = 0.90 wins by **+0.041**, against a run-to-run σ of ~0.05. **ALPHA stays 0.75.** Note the
anchor row reproduces the round-7 board (79.988 against 79.984883) — by construction, but it
confirms the plumbing.

**Blending sigma estimators** (`Track_1/sigma_blend_v9.py`). Log-space least squares over the
head's sigma, `std_t` of the predicted field, the input window's std and its per-trajectory
mean; fitted on all 336 val_seen windows, reported on the AoA 15-20 val_unseen subset:

| sigma | TKE | final | Δ |
|---|---|---|---|
| head, α = 0.75 | 0.4831 | 78.305 | — |
| head, α = 1 | 0.4632 | 78.300 | −0.005 |
| **blend** | 0.4975 | 77.993 | **−0.311** |

Worse out-of-sample, and worse on TKE itself. Same shape as the falsified fluctuation-gain
result: the fit leans on `traj-in` (+0.571 on u), which is smooth and helps on the split the
models trained on. This is now the **third** measured null for post-hoc sigma correction.
Sigma has to come from training.

---

## 4. Tooling changed this round

* **`Track_1/final_v9.py`** (new) — the round-9 weighting as code. `predict(subs)` returns
  (midpoint, min, max) over the whole admissible family, so the width of the interval travels
  with the estimate. Self-checks against all three board rounds to 1e-13. Wired into
  `compare_v9.py` (`wfinal`), `ens_v9.py` (`final`), `alpha_board.py` (`final`) and
  `validate_strict.py`. The falsified `final ≈ mean5 + 0.73` is gone from the tree.
* **`Track_1/tke_branch_v9.py`** (new) — §1.
* **`Track_1/sigma_blend_v9.py`** (new) — §3.
* **`model.py:load_trajectories(..., phases=)`** — reads the three discarded decimation phases.
  Phase (0,0) is asserted bit-identical to the old `[:, ::2, ::2]`.
* **`model.py:trajectory_names`** — names aligned with the loader's output order, so windows
  can be weighted by angle of attack.
* **`train2.py`** — `--phases`, `--aoa-boost`, `--aoa-hi`, `--aoa-only`, `--var-mlp`.
* **`_fno.py`** — `var_mlp` in the constructor and read from the checkpoint blob, so old and
  new checkpoints both load.

---

## 4b. WP1 fusion — shelved, then shipped after fixing a bad comparison

`submit_fno/_fno.py:StackedFNO` + `fuse()`. K same-shape members become one invocation: 1×1
convs become grouped convs, `GroupNorm(8, width)` per member becomes `GroupNorm(8K, K·width)`
(the group partition is identical because GroupNorm's groups are contiguous channel blocks),
and the spectral matmul `bixy,ioxy->boxy` gains a member axis. The 7-member held-out package
fuses 4 width-24 members and goes to **4 invocations**.

It is **exact**: max abs difference from the sequential loop is **1.5e-8** with TF32 disabled.
The 1.3e-3 seen with TF32 on is cuDNN picking a tensor-core kernel for the wider fused head,
not a logic error.

**Initially shelved, then shipped — the first decision was wrong.** The concern was that fusing
is slower locally (+0.475 ms/window end-to-end through `validate_strict.py`) while its payoff
rested on an unverified model of board time. That framing compared a *local* cost against a
*board* gain, which is not a comparison. Doing it properly, by regressing all three v9 board
rounds:

| model | slope | residuals |
|---|---|---|
| board ms ~ **invocations** | +0.316 ms | **±0.4 ms** |
| board ms ~ local ms | +0.380 | ±1.3 ms |

(U-Net ×1: 2.648 local → 5.976 board; FNO ×3: 1.282 → 5.931; FNO ×8: 2.211 → 8.020.)

The invocation model fits three times better. Under it, 8 → 4 invocations saves 1.26 board ms,
`time_score` 90.51 → 91.22, **+0.049 final**. Under the local-time model the +0.475 ms costs
**−0.007**. So the honest number is +0.05, not the +0.07 first claimed, and the downside is a
seventh of the upside.

Two further checks before shipping it: the fused path is **score-identical** (78.4925 against
78.4927 on the AoA 15-20 proxy, TF32 on), and `validate_strict.py` prints **SUBMITTABLE**
through the fused `_load()`. It is now the submission path.

The better reason to take it: fusing makes **additional same-architecture members cost zero
invocations**, which changes the member-count trade for every future round.

---

## 5. Results

Single checkpoints, AoA 15-20 held-out proxy, `wfinal` = the round-9 weighting on local errors
(a ranking number, not a board prediction). Baseline is three seeds of the shipped recipe:

| checkpoint | rel-L2 | TKE | MVPE | sps | mean5 | **wfinal** |
|---|---|---|---|---|---|---|
| `fno_grad` | 0.1120 | 0.4918 | 0.0967 | 42.06 | 81.666 | 81.916 |
| `fno_grad_s2` | 0.1114 | 0.4880 | 0.0976 | 42.19 | 81.713 | 81.966 |
| `fno_grad_s3` | 0.1118 | 0.4982 | 0.0975 | 42.00 | 81.607 | 81.849 |
| `fno_ctrl` (recipe guard, this round) | 0.1123 | 0.4938 | 0.0983 | 41.89 | 81.602 | 81.849 |
| *baseline mean, 4 seeds* | *0.1119* | *0.4930* | *0.0975* | *42.04* | *81.647* | ***81.895 ± 0.055*** |

### 5.1 Three arms discarded — wrong baseline recipe

The first three arms of this round (`--var-mlp`, `--lam-t 2.0`, and one more killed in flight)
all collapsed to rel-L2 ≈ 0.19 against the baseline's 0.112, with MVPE 2.4× worse. **The cause
was the harness, not the hypotheses.** The queue's base command was assembled from
`train2.py`'s argparse defaults instead of read off `fno_grad.pth`'s stored config, and three
flags differ:

| flag | argparse default | shipped recipe |
|---|---|---|
| `--lam-v` | 1.0 | **0.3** |
| `--lam-t` | 0.5 | **0.15** |
| `--alpha-train` | 1.0 | **0.0** |

`--alpha-train 1.0` is the damaging one: it computes the FIELD loss on the sigma-rescaled field,
so the field head is chasing a target that an untrained variance head is still distorting. Every
member of the shipped 8 uses 0.0.

**The "an MLP variance head damages the trunk" conclusion drawn from those runs is retracted.**
It was confounded three ways and shows nothing. Checkpoints deleted, arms requeued.

`fno_ctrl` — no flags changed at all — is now the first line of `queue.txt` and lands at
**81.849**, inside the three-seed spread. It is both the guard that the recipe is right and a
legitimate fourth baseline seed. Any arm below it in the queue is measured against 81.895.

The diagnostics in §1, §1b and §3 are unaffected: they ran on the pre-existing `fno_grad`
checkpoints, not on anything trained this round.

### 5.2 The sigma arms, on the correct recipe

| arm | change | rel-L2 | TKE | sps | **wfinal** | Δ vs 81.895 |
|---|---|---|---|---|---|---|
| `fno_ctrl` | none (guard) | 0.1123 | 0.4938 | 41.89 | 81.849 | −0.046 |
| `fno_vmlp` | `--var-mlp` | **0.1112** | **0.4871** | **42.26** | **81.981** | **+0.086** |
| `fno_t045` | `--lam-t 0.45` (3×) | 0.1139 | 0.4939 | 41.49 | 81.751 | −0.144 |
| `fno_vmlp_d` | `--var-mlp --detach-var` | **0.1099** | 0.5625 | 41.97 | 81.376 | −0.519 |

Three arms, one story, and they point the same way.

**Loss weight is not the constraint.** Tripling the TKE loss weight moves TKE by **0.0001**
(0.4938 → 0.4939) and costs the field (rel-L2 0.1123 → 0.1139). The variance head is not
under-trained; pushing harder on it just takes gradient from the field for nothing. `--lam-t`
stays at 0.15 and this direction is closed.

**Capacity might be.** The MLP head is the only thing so far that has moved TKE at all
(0.4930 → 0.4871, against a seed σ of 0.005) and it did so *without* costing rel-L2 — exactly
the signature §1b says to look for. At +0.086 on wfinal it is 1.6σ, which is suggestive and
not a result; a second seed is queued.

**And the trunk has to be shaped by the variance objective, not merely read by it.** Detaching
the same MLP head gives the **best rel-L2 of any arm** (0.1099 — the field head is free of the
variance gradient, as designed) and the **worst TKE** (0.5625 against 0.4871 attached). 0.5625
is essentially the `traj` row of §1 (0.560): a head reading a frozen field-only latent is no
better than a trajectory-constant predictor. This reproduces round 4's Run H on the U-Net
(detached 1×1 head at corr 0.679, below the trivial input-std estimator) and settles it for the
FNO. **`--detach-var` is closed for anything that cares about TKE.**

### 5.3 …but 100% variance shaping is worse, not better — prediction falsified

The obvious extrapolation from §5.2 was: detached = 0% of the trunk shaped by variance → TKE
0.5625; shared = partial → 0.487; so `--lam-f 0`, a network whose trunk exists *only* to
predict sigma, should be best. **It is not.**

`fno_varonly` has no trained field head, so `compare_v9.py` cannot score it. Evaluated the way
it would actually be used — as a sigma provider for a fixed 3-seed field ensemble, α = 1, so
the only thing varying down the column is where sigma came from:

| sigma from | trunk shaped by variance | TKE | rel-L2 | final |
|---|---|---|---|---|
| `fno_vmlp_d` (`--detach-var`) | 0% | 0.5625 | — | — |
| `fno_ctrl` (shared) | partial | 0.4851 | 0.1133 | 78.017 |
| **`fno_vmlp`** (shared + MLP head) | partial | **0.4660** | 0.1120 | 78.292 |
| `fno_varonly` (`--lam-f 0`) | 100% | 0.4892 | 0.1133 | 77.976 |

So the relationship is not monotone: 0% is much worse, 100% is *slightly* worse than partial.
**The field task is a useful auxiliary for the variance task** — a trunk that must also predict
the flow learns structure the variance head needs, and one trained on variance alone loses it.
The shipped shared-trunk arrangement is already at the optimum of that axis. `--lam-f 0` is
closed, and with it the "separate variance network" idea and its extra inference invocation.

**Only one thing has moved sigma: head capacity.** On this fixed-field comparison `--var-mlp`
takes TKE from 0.4851 to **0.4660**, a 3.9% relative cut and ~3.8× the 0.005 seed σ — a much
cleaner signal than the 1.6σ the whole-model `wfinal` comparison showed, because holding the
field fixed removes the field-side seed noise.

### 5.4 `--var-mlp` replicates, and it is worth about +0.09

| | rel-L2 | TKE | wfinal |
|---|---|---|---|
| `fno_vmlp` | 0.1112 | 0.4871 | 81.981 |
| `fno_vmlp_s2` | 0.1113 | 0.4872 | 81.989 |
| `fno_vmlp_s3` | 0.1108 | 0.4847 | 82.025 |
| *3 `--var-mlp` seeds* | *0.1111* | *0.4863 ± 0.0014* | ***81.998 ± 0.023*** |
| *4 baseline seeds* | *0.1119* | *0.4930 ± 0.0043* | *81.895 ± 0.055* |

Δ = **+0.103**, and the pooled standard error over 3-against-4 seeds is 0.031, so **3.4σ**.
The `--var-mlp` seeds are also three times tighter than the baseline's, on both TKE and wfinal.

At the ensemble level, which is what ships (3-against-3, band refit per set):

| set | rel-L2 | TKE | MVPE | sps | final |
|---|---|---|---|---|---|
| `fno_grad, _s2, _s3` | 0.1082 | 0.4831 | 0.0940 | 35.93 | 78.305 |
| **`fno_vmlp, _s2, _s3`** | **0.1076** | **0.4778** | 0.0945 | **36.10** | **78.387** |

**+0.082 at the ensemble level, for 3 408 parameters and no extra invocation.** rel-L2, TKE and
SPS all move the right way, so nothing is being traded away. **Adopted into the round-10
recipe.** Do not expect it to carry the round.

**Why so much smaller than the fixed-field gain?** Two dilutions, both structural. ALPHA = 0.75
means the composed field carries `0.75·σ_head + 0.25·std_t(field)`, so a quarter of any
improvement in σ_head is discarded by construction; and ensembling already averages sigma
across members, which reduces sigma error before the head's own quality matters. Re-sweeping
ALPHA on the `--var-mlp` pair confirms there is no unlock hiding there — α = 0.90 wins by
**+0.070** over 0.75, the same shape and the same inside-noise margin as the baseline's +0.041.
ALPHA stays 0.75.

### 5.5 WP3 — training for the board's regime makes things worse, monotonically

The premise: error is governed by angle of attack (0° 0.057 … 20° 0.105) and the board reads
0.113, at or above our hardest single trajectory, so equal weighting spends most of the
gradient where the board barely looks. Oversampling AoA 15/20 windows should therefore trade
full-split accuracy for board-like accuracy.

| arm | AoA 15-20 proxy | full split |
|---|---|---|
| `fno_ctrl` | **81.849** | **83.410** |
| `--aoa-boost 2.0` | 81.789 | 83.332 |
| `--aoa-boost 4.0` | 81.766 | 83.259 |
| `--aoa-only` (WP3c specialist) | 81.535 | — |

**Monotone down in the boost factor, on both splits**, and the specialist — the limit of the
same trend, trained on AoA 15/20 alone — is the worst of all. That is the cleanest available negative:
a real trade would win on the proxy and lose on the full split. This loses on the proxy too, so
there is no trade to tune — only a cost. Against the 4-seed baseline mean, −0.106 and −0.129.

**Why**: the wake physics is shared across angle of attack, so low-AoA windows teach the model
things that transfer to high-AoA ones. Oversampling does not add high-AoA information — there
is no more of it — it only removes effective data diversity. The model is **data-limited, not
regime-limited**, which is the same constraint round 8 found when capacity turned out to be
monotonically harmful.

WP3(c) proposed blending a specialist with the generalist rather than reweighting one model.
That fails too, and the way it fails is worth recording:

| set | rel-L2 | TKE | sps | final |
|---|---|---|---|---|
| generalist ×3 | 0.1082 | 0.4831 | 35.93 | **78.305** |
| 2 generalists + specialist | 0.1084 | 0.4850 | 35.85 | 78.268 |
| generalist ×3 + specialist (4 members) | 0.1079 | 0.4846 | 36.03 | 78.319 |

Substituting the specialist for a generalist seed **loses** (−0.037). Adding it as a fourth
member gains **+0.014** — which is simply what a fourth member is worth (round 6 measured the
4th at under +0.1 for any member), not anything the specialist contributes as a specialist.

**WP3 is closed in all three of its forms.**

That is an argument *for* WP2 rather than against it: decimation-phase augmentation adds real
samples instead of reweighting the ones we have.

### 5.6 WP2 — decimation phases

`load_trajectories` did `[:, ::2, ::2]` on 64×128 raw frames, keeping phase (0,0) — the scored
grid — and discarding (0,1), (1,0), (1,1). Those are the same PIV measurement sampled half a
coarse cell over: **real samples, 4× as many**, not the synthetic input-noise augmentation that
was falsified in round 6. Phase (0,0) is asserted bit-identical to the old loader; the other
phases differ from it by 4.8% in mean absolute terms, so they are genuinely distinct views.

`ph4_e38` is the cheap half of the question: 4 phases at **38 epochs**, which is 4× the windows
at the baseline's total gradient-step count. A win there is the *data*; a win only at 150
epochs is the extra optimization.

| arm | AoA 15-20 proxy | full split |
|---|---|---|
| `fno_ctrl` | 81.849 | 83.410 |
| *4-seed baseline mean* | *81.895* | — |
| `fno_ph4_e38` (compute-matched) | 81.886 | **83.471** |

Compute-matched, 4× real data is **a wash** on the proxy (−0.009 against the baseline mean,
well inside σ = 0.055) and marginally positive on the full split (+0.061). Worth noting only
because it is the first arm since `--var-mlp` that is not negative.

Training *longer* on the same 4× data is where it breaks:

| arm | AoA 15-20 proxy | full split |
|---|---|---|
| `fno_ctrl` | 81.849 | 83.410 |
| `fno_ph4_e38` (4 phases, baseline steps) | 81.886 | 83.471 |
| `fno_ph4_s1` (4 phases, 150 epochs = 4× steps) | **81.526** | **83.213** |

−0.369 against the baseline mean, 6.7σ — not seed noise, so the planned second seed was
dropped rather than spend 70 GPU-minutes confirming a rejection. (Accepting a +0.15 needs
three seeds; rejecting a −0.37 does not.)

**WP2 verdict: neutral at best.** The three extra decimation phases are real, distinct samples
— 4.8% mean absolute difference from phase (0,0), not a synthetic perturbation — and at matched
compute they buy nothing measurable. Given 4× the windows, the extra optimization then
overfits, exactly as round 8's capacity ladder predicted. The phases are *correlated* views of
the same PIV frames, so they add far less independent information than their count suggests.

`--phases` stays in the tree, defaulted off, with this result attached.

### 5.7 The one pattern that has worked, applied once more

Nine of ten arms this round were negative or neutral. The single winner, `--var-mlp`, was not
"more capacity" — round 8 established that trunk capacity is monotonically *harmful* — it was
capacity in a place that was **structurally thin**: a linear 1×1 probe reading a 24-channel
trunk, where the sibling head had two layers.

There is exactly one other layer of that shape. `lift` is `nn.Conv2d(42, 24, 1)` — a **linear**
map compressing the whole 20-frame input window (40 velocity channels plus 2 coordinate
channels) to 24, before the first spectral block sees anything. `--lift-mlp` gives it a 128-wide
hidden layer for 7 568 parameters. Queued as `fno_lift` (alone, to isolate it) and
`fno_vmlp_lift` ×2.

| arm | rel-L2 | TKE | sps | wfinal | Δ vs 81.895 |
|---|---|---|---|---|---|
| `fno_ctrl` | 0.1123 | 0.4938 | 41.89 | 81.849 | −0.046 |
| `fno_vmlp` (`--var-mlp`) | 0.1112 | 0.4871 | 42.26 | 81.981 | +0.086 |
| `fno_lift` (`--lift-mlp`) | 0.1117 | **0.4865** | 42.13 | 81.971 | **+0.076** |

Same size of gain, from the opposite end of the network, for 7 568 parameters. `--lift-mlp`
also improves TKE by as much as `--var-mlp` does (0.4865 against 0.4938) — which is not
obvious for a change at the *input*, and says the variance head was partly limited by what the
trunk was given rather than only by its own readout.

So the round's one reproducible finding generalizes: **this model's problem was never total
capacity, it was two linear bottlenecks.** Round 8 measured trunk width as monotonically
harmful and that still holds; these are 11k parameters combined, 1.2% of the model.

**And they are additive.**

| arm | rel-L2 | TKE | sps | wfinal | Δ vs 81.895 |
|---|---|---|---|---|---|
| `fno_ctrl` | 0.1123 | 0.4938 | 41.89 | 81.849 | −0.046 |
| `--var-mlp` | 0.1112 | 0.4871 | 42.26 | 81.981 | +0.086 |
| `--lift-mlp` | 0.1117 | 0.4865 | 42.13 | 81.971 | +0.076 |
| **`--var-mlp --lift-mlp`** | **0.1111** | **0.4800** | **42.34** | **82.063** | **+0.168** |

Predicted sum +0.162 against +0.168 observed — additive to within noise, and additive on TKE
as well (−0.0067, −0.0073, −0.0138). Two independent bottlenecks, two independent fixes.

Replicated: `fno_vmlp_lift` 82.063 and `fno_vmlp_lift_s2` 82.055, agreeing to 0.008 on wfinal
and 0.0004 on TKE. Mean **82.059 against 81.895, +0.164 at 5.9σ**.

And additive at the ensemble level too, 3-against-3 with the band refit per set:

| ensemble | rel-L2 | TKE | MVPE | sps | final |
|---|---|---|---|---|---|
| baseline ×3 | 0.1082 | 0.4831 | 0.0940 | 35.93 | 78.305 |
| `--var-mlp` ×3 | 0.1076 | 0.4778 | 0.0945 | 36.10 | 78.387 |
| **`--var-mlp --lift-mlp` ×3** | **0.1075** | **0.4726** | **0.0938** | **36.12** | **78.451** |

**Round-10 recipe: baseline + `--var-mlp --lift-mlp`.** 10 976 parameters, 1.2% of the model,
no extra inference invocation, **+0.146 at the ensemble level** over the shipped recipe.

---

## 5.9 The SPS band was not closed — it was closed against the wrong proxy

Round 6 declared the band finished after three attacks, one of which was a per-sample band
scaled by **ensemble spread**. That failed with the signature of noise: better out-of-sample,
worse in-sample. The conclusion drawn was "the band is closed". The conclusion that should have
been drawn was "ensemble spread is a poor error proxy" — it correlates with per-element
|error| at only **0.347** once the per-element mean is removed.

The variance head's own sigma correlates **0.551** (ensemble spread on the round-10 members is
0.482, itself well above round 6's 0.347). `Track_1/band_sigma_v9.py`:

    half(i, e) = base(e) * (sigma(i, e) / sigma_ref(e)) ** p

**The first version of this experiment reproduced round 6's result, and was wrong.** Fitting
`base` at p = 0 and then testing p ≠ 0 makes every p ≠ 0 worse in-sample *by construction*.
Refitting `base` for each p — deflating the errors by `r**p` first, since `|E| <= base·r**p`
is exactly `|E|/r**p <= base` — removes the handicap, and then p = 0.45 wins in **both**
directions:

| | in-sample | out-of-sample |
|---|---|---|
| shipped band (p = 0) | 44.538 | 39.886 |
| **p = 0.45** | **44.828** | **40.414** |

The in-sample curve is smooth and unimodal with its peak at the same p as out-of-sample. This
is not the round-6 pattern.

**Second gain: smoothing.** The base has 20·32·64·2 = **81 920 free parameters fitted from ~126
windows**, so `optimal_band`'s exact in-sample optimum is badly overfitted. A box blur over
(T, H, W) is pure variance reduction — and unlike p it *cannot* be selected in-sample, since
the unsmoothed band is the in-sample optimum by definition. Leave-one-**trajectory**-out CV
inside the fitting split (windows within a trajectory are far too correlated to fold on):

| k | 1 | 2 | **3** | 5 | 7 |
|---|---|---|---|---|---|
| CV sps | 42.808 | 43.065 | **43.337** | 43.277 | 43.175 |

CV picks k = 3, and k = 3 is independently the best on val_unseen (40.615 against 40.414).

**Total: 39.886 → 40.615 = +0.729 sps = +0.132 `final`, with no retraining.**

One false lead recorded: `fit_band.py`'s "board proxy 1.00x" column made an all-AoA fit look
worth +6.9 sps. It is not — that column scores against *uninflated* errors, which flatters a
band fitted to the easier full split. Repeating the comparison at a matched 1.39×, the AoA
15-20 fit still wins (40.414 against 40.323). The `--fit-aoa 15 20` default stands.

Shipped: `predict_fno(..., sigma_out=)` exposes sigma, `fit_band.py` writes `band.npz`
(`base`, `sigma_ref`, `p`), `submission.py` forms the per-window widths. Order-independence
still passes — `sigma_ref` is a stored constant, so a window's band depends only on that window.

---

### 5.8 The pattern does not extend to a third layer — and why not

`point` is `nn.Conv2d(width, width, 1)`, linear, one per FNO block. By the shape of the
argument it should have been the third win. It is a clear loss:

| | rel-L2 | TKE | wfinal |
|---|---|---|---|
| `--var-mlp --lift-mlp` | 0.1111 | 0.4800 | **82.059** (2 seeds) |
| `+ --point-mlp` | 0.1148 | 0.4999 | 81.680 (**−0.379**) |

Its training loss went *much* lower — field 0.052 against 0.067 — which is the overfitting
signature, not an optimization failure.

So the rule is not "fix linear 1×1 layers". It is **capacity at the boundaries helps, capacity
in the trunk hurts**. `lift` and `head_logstd` are the encoder and decoder — they change what
the trunk is *given* and what is *read out of it*. `point` is inside the trunk. +11 k
parameters at the boundaries gained +0.164; +25 k inside the trunk cost −0.379. That is round
8's capacity result seen from the other side, and the two now agree exactly.

### Width, extrapolated downward for the first time

If trunk capacity is harmful, the round-8 ladder (w24 81.67 > w48 81.33 > w64 81.17, monotone
*decreasing*) had never been tested below 24. On the round-10 recipe:

| width | wfinal |
|---|---|
| 16 | 81.865 |
| **24** | **82.059** |
| 48 | 81.451 |

**Width 24 is a genuine interior optimum**, not the end of a downward slope. (Width 20 is not
testable: `GroupNorm(8, 20)` requires channels divisible by groups.) The width axis is closed
in both directions.

### Capacity, re-tested with the bottlenecks gone

Round 8's "trunk width is monotonically harmful" was measured *before* the two linear
bottlenecks were fixed, so it was worth re-checking whether fixing them moved the ceiling.

| | wfinal |
|---|---|
| `fno_vmlp_lift` (width 24, round-10 recipe) | **82.063** |
| `fno_vmlp_lift_s2` (width 24, round-10 recipe) | 82.055 |
| `fno_vl_w48` (width 48, round-10 recipe) | 81.451 |
| `fno_w48` (width 48, old recipe) | 81.536 |

Width 48 is still ~0.6 behind width 24, and the round-10 recipe does not rescue it — it is
0.085 *worse* than the old-recipe w48. **The capacity verdict stands unchanged.** The wide
members earn their place in the package through ensemble *diversity* (round 8: each new
architecture +0.05…0.07 against +0.01…0.03 for an extra seed), not through individual quality.

## 6. Where the round landed

Thirteen arms. **Two adoptions, both the same idea; eleven negatives or nulls.**

| | Δ vs baseline 81.895 |
|---|---|
| `--var-mlp --lift-mlp` | **+0.164** (5.9σ) — **adopted** |
| `--var-mlp` | +0.086 |
| `--lift-mlp` | +0.076 |
| `--phases 4`, baseline steps | −0.009 |
| `--aoa-boost 2.0` | −0.106 |
| `--aoa-boost 4.0` | −0.129 |
| `--lam-t 0.45` | −0.144 |
| `--aoa-only` | −0.360 |
| `--phases 4`, 150 epochs | −0.369 |
| `--detach-var` | −0.519 |
| `--lam-f 0` | worse sigma than the shared trunk |
| post-hoc sigma blending | −0.311 out-of-sample |
| ALPHA 0.90 | +0.041, inside noise |

The pattern worth carrying forward: **every attempt to redistribute what the model already had
failed** — reweighting the data by regime, reweighting the losses, reallocating the trunk
between objectives, adding correlated samples, recombining sigma estimates after the fact. The
only thing that worked was removing two *linear* bottlenecks, and round 8 had already shown
that adding trunk width makes things worse. Capacity was never the constraint; two specific
layers that could not represent a nonlinearity were.

Second thing worth carrying: **oracles rank directions, they do not forecast gains.** §1b priced
perfect sigma at +7.71 and five arms extracted +0.08 of it. The ranking was still correct — σ
was the right thing to chase, and TKE is where both adopted fixes paid — but the magnitude was
never available.

### What is left, honestly

`oracle_split_v9.py` says +21.04 total headroom, of which σ is +7.71. Round 10 took ~0.15 of it.
The remaining σ gap is not a modelling oversight any more: it is the genuine difficulty of
predicting a 20-frame amplitude envelope 20 to 40 frames ahead, in a flow whose shedding period
is 60-75 frames. Nothing in the literature scan of round 9 addresses that, and the two cheap
post-hoc routes are both measured nulls.
