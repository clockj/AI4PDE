# Design K2 — zero-training inference-side tricks on the point prediction (written 16:50 EDT, 09-27)

Lab: `$K/wt_2/Track_1/probe_k2/` (`lab2.py` = paired reads on the F-sprint dump `final0927/ckpt/b/dump_unseen.npz`, 5 held-out
twins, val_unseen 8 trajectories x 42 windows; metrics = kit `scoring.py` via `sps9.sample_metrics`; band = the shipped
construction refitted LOTO exactly as F-sprint `bandlab.py` does: centre0 = y0, p 0.45, inflate 1.39, contain-P; alpha 0.80).
Logs: `$K/logs/k2_p*.log`, `$K/logs/rp_k272_jit_14942616.log`. `final` = 0.4778 rel + 0.1010 tke + 0.0921 mvpe + 0.2498 sps (no time).
"hi" = AoA 15-20 (3 trajectories), "all" = 8. Stretch f multiplies the three errors and moves the target as in bandlab.

## 0. Verdict in one paragraph
One trick survives: **JOINT-CHANNEL GAIN** in the ALPHA compose. It is a mathematical identity-level improvement (same TKE map,
same time-mean, prediction strictly closer to the ensemble mean), sign 3/3, 8/8 (rel) and 16/16 on the AoA-20 hold-out. It is
SMALL: predicted board **+0.016 final [+0.005, +0.03]** per track. Everything else in the mandate measured zero or negative
(table in section 3). Direction 2 cannot close the +0.2 gap on its own; JOINT is a free add-on to whatever ships.

## 1. Trick #1 — JOINT-CHANNEL GAIN (ship)
**What.** Today compose rescales each channel separately: `P_c = m_c + (y0_c - m_c) * T_c / s_c`, `T_c = a*sigma_c + (1-a)*s_c`
(s_c = temporal std of the ensemble mean y0 over the 20 output frames, per pixel). TKE only scores
`k = 0.5 (var_t u + var_t v)` per pixel, i.e. ONE scalar per pixel. Among all 40-vectors (20 frames x 2 channels) with the
required energy, the one closest to the conditional mean is its radial projection = ONE gain for both channels:

    g = sqrt(T_u^2 + T_v^2) / sqrt(s_u^2 + s_v^2)      (PHYSICAL units)      P_c = m_c + g * (y0_c - m_c)

Same k map (tke identical up to rounding), same time-mean (mvpe identical), smaller ||P - y0|| at every pixel (Lagrangian:
min sum_c (g_c-1)^2 s_c^2 s.t. sum_c g_c^2 s_c^2 = const => g_u = g_v). No fitted constant, no new input: per-window legal on
T1 and T2 by construction. Cost: 2 sums + 2 sqrt on a (B,1,H,W,1) tensor.

**Measured (held-out twins, paired vs the shipped per-channel compose, alpha 0.80):**
| read | rel err | tke err | mvpe err | d rel score | d sps | d final | sign |
|---|---|---|---|---|---|---|---|
| hi, f 1.00 | -0.48 % | 0.00 % | 0.00 % | +0.022 | +0.014 | **+0.0140** | 3/3 |
| hi, f 1.29 | | | | +0.027 | +0.020 | **+0.0181** | 3/3 |
| all, f 1.00 | -0.46 % | 0.00 % | 0.00 % | +0.016 | +0.011 | **+0.0105** | 7/8 final, 8/8 rel |
| all, f 1.29 | | | | +0.020 | +0.018 | **+0.0141** | 7/8 |
| AoA-20 hold-out (F-sprint `ood_aoa20.npz`, 5 members never saw AoA 20, 16 trajectories, physical units) | **-0.87 %** | 0.000 | 0.000 | | | | 16/16 |

sps rises through abar AND because P sits closer to the centre y0 (less contain-P stretching).
With JOINT, alpha 0.85 reads +0.025/+0.028 (hi) and +0.019/+0.021 (all); alpha 0.90 +0.030/+0.032 (hi), +0.018 (all). **Do NOT
raise alpha**: the board paid alpha 0.8->0.9 with -0.032 (09-26 T2); JOINT removes only ~10 % of alpha's rel cost.

**Predicted board delta (each track, vs the same package without the knob):** rel +0.025 [+0.01, +0.04], tke 0, mvpe 0,
time 0 (bracket -0.003), sps +0.015 [0, +0.03] => **final +0.016 [+0.005, +0.03]**. Not halved: this is not a model error cut,
it is a smaller compose displacement, and the board charges alpha's rel cost at >= 1x local (09-26: -0.066 rel score board vs
-0.05 local for alpha 0.8->0.9).
**Kill criterion:** in the real package path on the twin package, rel (unstretched) must fall by >= 0.3 % on the full split with
tke and mvpe changed by < 0.01 %; otherwise NO-SHIP (it would mean the implementation is wrong, not the idea).
**Reproducibility:** inference-only constant; nothing to re-train. REPRODUCE.md needs one line (knob name + value).

## 2. Executor task specs

### E2a — implement the knob (bit-identical when off), verifier, build, gates
Work in a fresh clone `$K/wt_2x` (`git clone -q /users/PAS1187/qwu/workspace/realpde/workingtree.2/codebase $K/wt_2x`), symlink the
untracked inputs as in BUILD_RUNBOOK section 0.4. Three compose sites, all with `y` = (B,20,H,W,2) NORMALIZED, channel last:
1. `Track_1/submit_fno/_fno.py` ~line 2539-2543 (`predict_fno`; `std` is the (2,) tensor, (B,1,1,1,2) under self-norm).
2. `Track_1/submit_fno/submission.py` ~line 441-445 (graph instance; `self.std`).
3. `Track_2/submit/submission.py` ~line 505-511 (uses `sa`, which SIGEMA may replace; keep that).
Patch pattern (keep the ORIGINAL expression verbatim in the else branch so off = bit-identical):
```python
if JOINT_GAIN:
    w_ = (std * std).float()                      # channels weighted in PHYSICAL units: k = 0.5*sum_c (std_c * .)^2
    t_ = alpha * sig + (1 - alpha) * s_
    y = m_ + f_ * (torch.sqrt((t_ * t_ * w_).sum(-1, keepdim=True)) / (torch.sqrt((s_ * s_ * w_).sum(-1, keepdim=True)) + 1e-7))
else:
    y = m_ + f_ * ((alpha * sig + (1 - alpha) * s_) / (s_ + 1e-6))
```
**The std weighting is mandatory**: the compose runs in normalized space (std_u 0.0968, std_v 0.0160); an unweighted sum is a
different (wrong) projection. `JOINT_GAIN = False` as a module constant in `_fno.py` (next to `D2_DIRECT_PRED`) and in both
`submission.py` files; build knob `RPDE_SHIP_JOINT_GAIN` in `build_round13.sh` following the `RPDE_SHIP_D1C_N0` sed lines
(131-132) — it must sed `_fno.py` too (as DIRECT_PRED does) and be restored by the same `git checkout` list.
Check every other place that assumes "compose is linear in alpha" (`CENTRE_ALPHA/ALPHA` reconstruction, `_fno.py` ~2531,
`submission.py` ~250, ~842): with `CENTRE_ALPHA = 0` the centre is y0 itself and nothing changes; **assert CENTRE_ALPHA == 0
whenever JOINT_GAIN is on** (any other centre would use the broken linearity).
Band: reuse the shipped band (`RPDE_BAND=$M0/Track_1/bands/band_r38_a080_c0_fill.npz`) — one controlled lever; the centre0
error field does not depend on the compose.
Verifier `Track_1/verify/joint_gain.py` (GPU): (a) knob off: pred/lower/upper bit-identical to the baseline package
(`final0927/sub_candidates/track1_x27_a080.zip`) on val_unseen; (b) knob on: per-pixel `0.5*(var_t u + var_t v)` of pred equals
the knob-off map to 1e-5 relative, time-mean equal to 1e-7, `||P_on - y0|| <= ||P_off - y0||` for every window, `lower <= pred
<= upper`; (c) batch/permutation invariance (validate_strict asserts it); (d) T2: CUDA graph captured and bit-exact vs eager,
`stream_eval.py --causal-check` passes.
Build (T1 "27lz" + T2 "asnp" knob sets from BUILD_RUNBOOK section 1, alpha 0.80, `RPDE_FORCE=1`, `RPDE_SUB=$K/sub_candidates`,
`RPDE_TAG=k27jg_a080`, add `RPDE_SHIP_JOINT_GAIN=True`; second build with `RPDE_SHIP_D1C_N0=30.0`, tag `k27jg_d1c_a080`).
Gates: `validate_strict.py` and `validate_t2.py --adaptive` must print SUBMITTABLE. Job name `rp_k272_build`.

### E2b — real-code-path read on the twin package
Copy `final0927/jobs/rp_f27x_candref.sh` to `$K/jobs/rp_k272_candref.sh`, point W at `$K/wt_2x`, run it twice (knob off =
TAG base, knob on = TAG jg) with CAND_A/CAND_M = the shipped members / 5 twins. Report the static and asym-R1 stream rows
(rel/tke/mvpe/sps/final) for all 8 and AoA 15-20, paired. Expected: rel err -0.46 %, tke/mvpe unchanged, sps +0.01..0.02.
Write `$K/notes/result_2a.md` / `result_2b.md`.

## 3. Ruled out (all held-out, paired; numbers are relative error changes, negative = better)
| trick | hi (3) | all (8) | verdict |
|---|---|---|---|
| Scale-jitter ensemble, zero cost (F-FNO .95/1/1.05, ViT .975/1.025 and 3 other assignments) | rel +0.02..+0.10 %, tke 0..+0.11 % | rel 0..+0.05 % | dead; models are nearly scale-equivariant, jitter adds no diversity |
| Full scale TTA at 3x compute (upper bound) | rel -0.01..-0.02 % | -0.01..-0.02 % | dead |
| Additive u-shift jitter +-3 % U | rel 0.00 % | -0.01 % | dead |
| Family weights per frame x channel, LOTO (F-FNO weight 0.92 at frame 0 -> 0.6 after frame 3) | rel 0.00 %, mvpe +0.13 % | rel -0.03 %, tke -0.11 % | dead (< +0.003 final) |
| Family weights per frame x channel x x-column, LOTO | rel +0.09 %, mvpe +0.46 % | rel +0.12 % | dead |
| Family weights per frame x radial wavenumber band x channel, LOTO | rel 0.00 %, tke -0.07 % | rel -0.04 %, tke -0.13 % | dead (< +0.005) |
| Frame gain schedules (linear ramps +-0.15, +-0.30, same std and mean) | rel +0.08..+1.07 % | +0.08..+0.97 % | dead; uniform gain is the Lagrangian optimum |
| Slope-projected gain (g_t,c proportional to the held-out calibration slope b_t,c; b_u falls 1.02 -> 0.92 over the horizon) | rel -0.02 % | -0.01 % | dead |
| Input-window variance blended into the amplitude target (T1-legal "VC") beta 0.1-0.5 | tke -0.22 %..+12.5 %, rel +0.4..+3.1 % | same | dead |
| alpha rising with the pixel's relative sigma (continuous) | tke -2.0 %, rel +0.9 % | tke +2.7 %, rel +1.6 % | dead (flips across splits) |
| sigma from F-FNO only / ViT only / rms / geo-mean / 0.8-0.2 | tke +0.3..+2.0 % (rms: -0.29 %, rel +0.10 %) | same sign | dead; arithmetic mean of 5 is best |
| Mean field: LS blend of predicted and input time-mean | weights (0.99, 0.004) | (0.96, 0.04) | dead; mvpe cannot be bought from the input |
| Static bias map / U, LOTO (x1, x0.5; per pixel and per frame-pixel) | mvpe +0.05..+0.40 % | mvpe +0.05..+0.69 % | dead |
| Spatial smoothing of y0 (ramped in the horizon) or of the ViT mean | rel -0.04..+0.07 % | rel +0.06..+0.38 % | dead |

Diagnostics worth keeping: u carries 77-80 % of the normalised squared error, v 20-23 %; error grows with the horizon
(rel^2 share frame 0: 0.5 %, frame 19: 7.9 %); the compose at alpha 0.80 costs +10 % rel^2 (+4.6 % rel), most visibly in the
first three frames; F-FNO beats ViT at every frame (frame 0: 0.47 vs 0.87e-4, frame 19: 7.64 vs 7.87e-4).
Side note: my H100 forwards differ from the F-sprint A100 dump by up to 5.9e-4 (max abs, fp32 path) — cross-hardware noise.

## 4. Jobs
`rp_k272_jit` 14942616 (Cardinal gpu, COMPLETED, 1 GPU x ~2 min) -> `$K/dumps/k2_jitter_unseen.npz`. No job is running.
