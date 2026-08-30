# Round 12 — 2026-08-29, OSC A100-40G: the port, two branches closed, one family added

One ~20 h autonomous session: migrate the whole project from the dead 4060/WSL2 box to an OSC
A100-40G Slurm node (job 7086504, node dies 2026-08-30 11:11), rebuild every binary artifact
(`.gitignore` excludes `*.pth/*.npz/*.zip` — nothing survived the clone), run the round-12
science plan (`plan_20260810.md`), and freeze a new package. Campaign log with all raw rows:
`Track_1/plan/campaign_20260829.md`. Board reference: round-10 build scored **80.512**
(best other team 80.79); the round-11 package predicted 80.603 — whether it was ever submitted
is not recorded here and needs checking against Codabench.

Everything below was measured this session. Git: branch `qwu/dev`, commits ba25e80 → f484524 →
c442474 → cfb8cba (+ the gate-P/Z commits after this file).

## 1. The port is verified end to end (gate S)

* `realpde` conda env recreated from `environment.yml` (torch 2.2.2+cu121 — the evaluators'
  stack); data pulled to `/users/PAS1187/qwu/RealPDE/`; `/home/clockj` paths rewritten in 40
  files.
* **The v9 `scoring.py` in the kit directory is a RECONSTRUCTION** (Codabench is auth-walled
  from this cluster; the real kit could not be downloaded). Verified against `sps9.py`'s
  independent SPS path to 6.5e-8 and against every documented score-curve point. **Retire this
  risk by fetching the official kit v9 with a Codabench login.**
* **The reconstruction had one real bug, caught by the anchor rows**: MVPE is rel-L2 per probe
  COLUMN (own denominator per x-station, mean over the 4 stations), not pooled over all 36
  probes. Pooled read −0.097 wfinal (5.7σ). Fixed at source; both anchors now hit the
  documented numbers to 4 decimals (persistence 0.1408 vs 0.141, time-mean 0.1240 vs 0.124).
  The one remaining un-discriminable assumption is TKE's `ddof` (anchor predictors are
  time-constant, so TKE ≡ 1 either way); ddof=0 assumed.
* **The recipe reproduces on the A100 at +1σ**: `fno_vl_s1` (BASE + `--var-mlp --lift-mlp`)
  proxy wfinal **82.063** against the 4060 reference 82.046 ± 0.017. One number confirms data,
  windowing, normalization, recipe and scorer simultaneously. 6.8 s/epoch, **18.4 min/run**.
* **The binding resource is the 12-core cpuset (cores 112–123), not the GPU.** `fetch()` is
  ~half the training step; one colliding numpy eval cost 1.7×. Rule used all session: training
  under `taskset -c 112-120`, evals under `-c 121-123`, timing only on an idle machine.
  `nvidia-smi` reads 0% during healthy training (tiny-kernel sampling artifact) — never judge
  liveness by it.
* Housekeeping: real trajectories are 607 frames (CLAUDE.md said 868; fixed); `train2.py
  --help` crashed on an unescaped `86%` (fixed); session interrupts kill background jobs, so
  the queue runs under `setsid nohup` (survived three later interrupts) and the endgame watch
  ran as a cron job, both now standard practice.

## 2. WP-1: the FGN probabilistic head FAILS its pre-registered screen — branch closed

The round's main bet (`directions_20260809_architecture.md`): replace the log-MSE sigma head
with an FGN-style head (arXiv 2506.10772) — FiLM-from-noise on the GroupNorms (z ∈ R^32 →
zero-init `Linear(32 → 2·width)` per block), K=4 sample passes, fair CRPS added to the
untouched BASE losses. Implementation verified to the bit before training: `forward(x)` ==
`forward(x, z=0)` == a `fgn_dim=0` model with the same weights (maxabsdiff 0.0), fair CRPS
matches the analytic Gaussian value at K=512 (0.23365 vs 0.23369) and is unbiased at K=4.

Screen (pre-registered in the campaign file, before results existed): *if no sample-spread σ
candidate beats the head's σ rel-L2, stop the branch.* Result on `fno_fgn1`
(`--fgn 4 --lam-crps 0.3`, K_eval=8), val_unseen:

| σ estimator | σ rel-L2 |
|---|---|
| log-MSE head (z=0) | **0.3293** |
| sample spread, mean-of-std | 0.4691 |
| sample spread, rms-of-var | 0.4507 |
| input-window std (zero-parameter anchor) | 0.5006 |

* Parameter-space noise does not manufacture amplitude information the trunk does not carry —
  round 11's information floor, measured a third way. Every spread-σ composition loses
  ~0.7 wfinal (TKE 0.58 vs 0.47). The sample mean also degrades the field (rel-L2 0.1146 vs
  z0's 0.1106): FiLM draws are not mean-preserving around the trained center.
* **The CRPS term is harmless but adds nothing**: fgn1's z=0 model ties the baseline seeds
  exactly (81.923 vs 81.921/81.874 at α 0.90), so the checkpoint was recycled as an ordinary
  extra seed in the gate-M control arm. Per the stop rule, fgn2 (λ 0.1) was never run.

**Verdict: closed by measurement.** "A proper scoring rule unlocks σ" was WP-1's premise; it is
now false at this scale. The σ head keeps its log-MSE objective (round 11: `lam_v` 0.3 is the
optimum of the only family that moves it).

## 3. WP-2: the phase/advection branch is dead — the fluctuation error is decorrelation

Diagnostic only, no training. First instrument (whole-field error-minimizing spatial shift)
rejected itself: its positive control — persistence, which carries a real ~0.44 cells/frame
advection lag — read 0.7% recovery, because the time-mean field dominates the norm and pins
the shift at zero. Second instrument re-registers the FLUCTUATION only, with a calibrated
synthetic control: a (1,3)-cell rolled fluctuation is recovered **100.0%** at exactly (−1,−3)
in 100% of windows.

On the real model: re-registration recovers **0.2%** of fluctuation error; zero shift is
preferred in 92% of windows (search ±2 y × ±6 x ≈ ±13 frames of shedding phase). Fluc-only
rel-L2 is ~0.85 — the error is **decorrelation/amplitude, not misregistration**, consistent
with `diag_spectral`'s ~31% coherently-predictable fluctuation energy. WN-C-style phase
co-training and any phase-aligned prediction frame are closed by this: there is no phase error
for them to fix.

## 4. WP-3: a second architecture family in the ensemble — the U-Net member WINS

`ens_mixed.py` (new): mixed-family ensemble scorer — average raw field and sigma in normalized
space across arbitrary `(field, logstd)` models, compose once, band refit per row; solo rows
validated against `compare_v9` exactly. Candidate: `unet_g1` (UNetTwoHead w64, round-4
grad-loss recipe `--lam-grad 0.3`), held-out. Battery at α 0.90, band refit per row:

| set | proxy wfinal | full wfinal |
|---|---|---|
| A: 7 FNO (s1–s4, m1220, w48, w64) | 82.623 | 84.099 |
| **B: A + unet_g1** | **82.714 (+0.091)** | **84.222 (+0.123)** |
| C: A + fgn1-z0 (extra-w24-seed control) | 82.642 (+0.019) | 84.119 (+0.020) |

* **The U-Net member is worth 4.5× an extra FNO seed, on both splits** — round 8's
  "architecture beats seeds" replicates across families, not just FNO variants. unet_g1 solo
  (81.66) also beats the w48/w64 solos. Cost: one invocation + ~0.4 local ms ≈ −0.03 final;
  net ≈ +0.07…+0.10.
* **Mode-diversity arms are NULL**: m0608/m1016 on top of B give proxy +0.011/+0.008/+0.015
  and full +0.005/−0.007/−0.003 — seed-level noise against a ~+0.006/member cost. The
  round-8 "+0.05–0.07 per architecture" does not extend to a 4th/5th mode variant once the set
  spans m1220 + w48 + w64 + U-Net. Composition closed.
* **α = 0.90 re-confirmed on the final member set** (proxy 82.549 / 82.598 / 82.623 / 82.626 /
  82.614 at 0.75 / 0.825 / 0.90 / 0.95 / 1.0; full peaks 0.75–0.90 and drops at 0.95).

## 5. The round-12 package

**11 members, 4 invocations**: 7×w24 all-data seeds (s1–s7) + m1220 mode-padded into one
StackedFNO group, + w48, + w64, + `unet_all_g1`. s6/s7 were added on the round-10 board lesson
(member count is the board's call, not the proxy's — dropping w48 on a +0.001 proxy margin
cost the board −0.06) at the measured marginal-seed price (+0.019 for member 8 in the C
control). Band: `fit_band --err-ckpt` = the 8 held-out twins (all-data errors are memorized —
the round-6 sps-36.94 trap), α 0.90, p 0.45, smooth k=3.

Pipeline changes that made a mixed package possible (all verified before the real build):

* `_fno.load_any` dispatches on each blob's own `arch` field; `_fno.py` vendors `block()` +
  `UNetTwoHead` with names identical to `unet/model.py` so `strict=True` loads. Routed through
  `submission.py`, `fit_band.py`, `predict_board.py`, `compose_v9.py`, `ens_v9.py`.
* **Two latent `fuse()` bugs found and fixed**: it crashed on a second non-FNO member
  (StackedFNO over a non-FNO group) and silently DROPPED non-fusable members (`append` where
  `extend` was needed). Member-count conservation is now asserted.
* Parity: packaged `predict_fno(unfused)` vs `ens_mixed.infer_mixed` is **bit-exact** (max
  |Δ| 0.0, 336 windows); through `fuse()` it is 2.7e-6 with cuDNN TF32 off and 1.4e-3 with it
  on — a pre-existing TF32 effect in StackedFNO's grouped 1×1 convs, score-neutral (≤4.3e-6 on
  every subscore).
* Mock build (held-out members standing in) passed end to end: SUBMITTABLE, 71.6 MB, scores
  equal to `ens_mixed`'s reference to 4 dp, import closure numpy+torch only.

### Gate P: real build + the A100→4060 timing conversion

The board-time model `board_ms = 4.447 + 0.324·inv + 0.431·local_ms` was fitted with 4060
timings, so A100 measurements must be converted. The round-11 SHAPE (5×w24 + m1220 + w48 +
w64, 3 invocations), rebuilt from all-data weights, ran `validate_strict` on the idle A100 at
**X = 1.082 ms/window** against the recorded 2.002 ms on the 4060 → **r = 1.850**.

Real build (`build_round12.sh`): band fitted on val_seen AoA 15–20 at inflate 1.39×, p 0.45,
smooth k 3 (half-width median 0.01037; sps 49.41 fitted, 54.84 on val_unseen against 25.46 for
the scorer's default band); `validate_strict` on the 11-member package → **SUBMITTABLE**,
82.56 MB extracted / 76.36 MB zipped, **Y = 1.208 ms/window** (A100) ≈ **2.235 ms
4060-equivalent** → board ms 6.706.

Board prediction (`predict_board --invocations 4 --local-ms 2.235`, errors from the held-out
8 — conservative, the package carries three more w24 seeds than the error ensemble), recorded
before any submission per the WP-4 discipline:

| subscore | round 10 (actual) | predicted | delta |
|---|---|---|---|
| rel_l2 | 94.576 | 94.521 | −0.055 |
| tke | 79.153 | 79.869 | **+0.716** |
| mvpe | 93.681 | 93.782 | +0.101 |
| time | 91.121 | 91.248 | +0.127 |
| sps | 39.626 | 39.628 | +0.002 |

**Predicted final = 80.707 (+0.195 on round 10's confirmed 80.512)**, delta-based — the level
fit of `final_v9` is falsified, but the intercept cancels in deltas. Best other team: 80.79.
TKE carries the round, which is where round 9's exchange-rate analysis said the open headroom
was; the U-Net family member and the extra seeds all buy TKE. Submitting is the user's call
(1/day); the zip is `Track_1/submit_fno/submission.zip`.

## 6. Standing state after round 12

* **Closed this round, by measurement**: FGN/CRPS sample-spread σ (WP-1), phase/advection
  error and any phase co-training (WP-2), 4th/5th FNO mode variants (composition), pooled-MVPE
  scorer ambiguity (per-column verified).
* **Open**: the official kit v9 download (user-side, Codabench login); whether round 11 was
  submitted; Track 2 (untouched, next session); the round-9 `final_score` weight family gains
  a data point when this package hits the board.
* All numbers this round were produced on the reconstructed scorer AFTER the MVPE fix; the
  pre-fix rows in early gate-S logs are 0.097 low and marked as such in the campaign file.
