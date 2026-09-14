# Track 2 — LTTTA (Long-Term Test-Time Adaptation)

Codabench: https://www.codabench.org/competitions/17385/ · Official kit:
`/users/PAS1187/qwu/RealPDE/official/t2_v6/realpde_t2_starting_kit_v6/`

Same 20-in/20-out forecasting problem as Track 1, same five subscores, same hidden real
targets — but streamed one window at a time in trajectory order at batch 1, in the evaluator's
normalized space, with the *previous* window's ground truth revealed each step. The contract,
the scoring rule and the current package are in `../CLAUDE.md`.

```python
# submission.py at the zip root
def get_ttt_model(submission_dir, device):
    # -> object with reset_ttt_state() and ttt_step(input_norm, prev_target_norm) -> (pred, info)
```

## The one fact that determines everything here

**Windows tile at stride 20 with `T_in = T_out = 20`, so window k−1's revealed target IS
window k's input, bit for bit.**

Therefore **anything computable from `prev_target_norm` alone is already computable from
`input_norm` — and Track 1 could compute it too.** Exactly two things are genuinely new:

1. the supervised **pair** `(x_{k-1}, y_{k-1})`: in-distribution training data from the very
   trajectory being forecast;
2. the **realized error** of predictions already made.

Everything an adaptation scheme can do lives in one of those two. Both were measured before
anything was built, and both are closed. (This also disposes of an early idea, EMA-adapting the
variance head toward the revealed window's per-pixel std: that std *is* the current input
window's std, a zero-parameter anchor at rel-L2 0.501 against the σ head's 0.329.)

## What is closed, with the file that closed it

| family | result | file |
|---|---|---|
| **Residual EMA / output bias correction** | lag-1 residual correlation **+0.051** (full field), +0.074 (time-mean). ORACLE per-trajectory additive correction: rel-L2 0.0800 → 0.0786, **1.75 %**. Every causal variant loses (−0.02 to −5.1 wfinal). | `ttt_offline.py` |
| **Online SPS band calibration** | at the board's 1.39× error stretch the shipped band is **already exactly optimal** (best global rescale c = 1.00, +0.000 sps); 1.0× leaves 0.667, 1.2× 0.114, 1.7× 0.276. Causal estimators recover none of it (−0.9 to −10). | `ttt_band.py` |
| **Gradient test-time training** | works, and does not pay. Best of 40 arms (all params, Adam 1e-5, rel-L2 loss, last pair): rel-L2 **−1.6 %** single member / −0.7 % on a 5-member ensemble, TKE and MVPE also down. Worth ≈ +0.02 `final`; costs ~9.7 ms/step of backward ⇒ **−0.5 `final`**. No lr, parameter subset, buffer or loss changes the sign. | `ttt_grad.py` |
| **Closed-form readout refit** (09-14) | an exact ridge solve of `head_field`'s and `head_logstd`'s last 1×1 conv, per trajectory, from every revealed window, with no backward pass. Best arm (ρ = 1000, the strongest prior): rel −0.18 % / tke +0.64 % / mvpe +0.06 % of error removed on one held-out twin, −0.01 / +0.48 / +0.14 on three; the largest rel cut (+1.02 %) costs −3.6 % tke. The trained readout is already near the per-trajectory optimum, and the update costs 1.07 ms per member at batch 1 ⇒ ≈ **−0.06 / −0.19 `final`**. | `readout_refit_check.py` |
| **Trajectory history** | legitimate here (the kit holds one model object and resets it at trajectory boundaries, so earlier windows are past measurements), and worth nothing: the model already carries what 200 frames of history add, and every blend loses. Shipped off as `HIST_BETA = 0.0`. | `../Track_1/dmd_probe.py`, `../Track_1/hist_blend.py` |

**The band is robust to error mis-specification**, which matters for the Decision phase:
stretching the per-element errors anywhere from 1.0× to 1.7×, the shipped band never gives up
more than **0.667 sps (≈ 0.14 `final`)** against the best rescaling chosen in hindsight. A
harder or easier test set does not break it.

**Conclusion: Track 2 is Track 1's model, streamed.** Ship the forecaster frozen. Track 2 is a
second prize pool for the same work, not a new algorithmic lever.

## The package

* **Built together with Track 1** by `../Track_1/submit_fno/build_round13.sh`, which runs
  `build_t2.py` and `validate_t2.py` with the same members, band and baked constants and stages
  `../../sub_candidates/track2_<tag>.zip`. The package of record and its board scores are in
  `../CLAUDE.md`, *Current state*.
* **`submit/submission.py`** runs the frozen ensemble in the evaluator's normalized space. The
  input needs no affine (`mean_in`/`std_in` equal `_fno.MEAN`/`_fno.STD`); the output goes
  through `mean_tgt`/`std_tgt`. It returns SPS bounds every step, and tensors the caller can keep.
  In `get_ttt_model`, which is untimed, it checks `AMP` against fp32 on a probe, then captures the
  whole device-side step in a CUDA graph, verified bit-exact against eager and dropped on any
  failure.
* **`submit/` in the working tree is a development copy.** Its `fno*.pth` are symlinks to the
  round-13 width-24 members, and `band8.npz` is whatever the last build wrote. It is not the
  package of record.

## Latency — the one thing worth engineering here

`ttt_step` is timed in its entirety at batch 1 (and `reset_ttt_state` is charged to the step
after a trajectory boundary), so Track 2 pays every kernel launch that Track 1 amortizes over a
batch.

**Measured on the board:**
* One extra invocation costs **+0.866 board ms** (Track 1: 1.151). Track 2's handicap over
  Track 1 is a fixed ~0.86 ms per step, not a per-invocation tax.
* FLOPs are charged: four more FNOs inside the fused group cost **+1.37 board ms**, about
  0.60 board ms per 4060-equivalent local ms (one reading).
* Heavy per-sample compute is expensive at batch 1. The ViT forward is costly, and a conv-heavy
  family is worse: U-FNO ×3 measured 5.457 ms/step locally and 8.69 board ms, because a CUDA
  graph removes launches, not FLOPs.

**Measured locally** (A100-PCIe; only comparisons interleaved in one process count):
* **fp16 autocast** (`AMP`, baked by the build): R24's package went 7.022 → 3.768 ms/step, with
  `ttt_step` outputs within 8.4e-4 of fp32 and the CUDA graph kept. R21's package read fp16
  *slower* in a separate-process run, so AMP is a per-package decision; gate it with
  `../Track_1/verify/amp_path.py --t2`. **On the board (T2R12)**, R24's archive with fp16 came
  in at 7.061 ms against xfam5's 6.255 (T2R9): fp16 recovered ~0.73 of a possible 3.61 board ms
  and did not pay for the second ViT's FLOPs. xfam5fp16 against T2R12 is the next read.
* **`SPEC_PATH = "bmm"`:** the complex einsum's strided weight copy was 28 % of the step. A
  batched real 2×2-block matmul removes it: −16.6 %, 2e-6 relative, replicated three times.
* **Where a graphed step goes** (`step_breakdown.py`, T2R12's archive, batch 1, 2026-09-14, on a
  shared node): `ttt_step` 3.9 ms = graph replay 3.7 + input copy 0.07 + outputs back 0.15 (CPU
  input; 0.02 / 0.04 with a CUDA input). Inside the replay: StackedFNO(4) 2.4 ms, StackedViT(2)
  1.5, average + compose + band 0.1. Eager, the same step is 12.2 ms over 653 kernels, so a launch
  costs ~15–22 µs and the graph removes all of it. What remains scales with FLOPs (graph b4/b1:
  FNO 1.73, ViT 1.32, post ~1.1). fp16 cuts the ViT stack 3.4× (5.0 → 1.5 ms) but makes the FNO
  stack slower at batch 1 (2.1 → 2.4 ms); FNO fp32 + ViT fp16 would be ~3.5 ms against 3.8.
* **Round 13's anatomy** (width-24 members) is why fusion and the graph exist:

| members | invocations | eager ms | **CUDA-graph ms** | graph `time_score` |
|---|---|---|---|---|
| 11 (Track 1's round-12 set) | 4 | 10.39 | 2.235 | 94.75 |
| 10 | 3 | 7.80 | 1.815 | 95.25 |
| **8 (w24 seeds + m1220)** | **1** | 3.61 | **0.989** | **96.45** |
| 1 | 1 | 3.01 | 0.468 | 97.53 |

* A single member costs **3.18 ms eagerly**, and almost all of it is kernel-launch overhead. An
  extra *invocation* costs ~2.4 ms; an extra member inside a fused group costs 0.09 ms.
* **Capturing the whole device-side step in a CUDA graph is bit-exact (max abs difference
  0.0e+00)** and cut 3.61 ms to 0.989.
* The 10-minute wall clock is not binding: 336 steps run in **0.61 s**.

## Honest local numbers

`stream_eval.py` streams real held-out trajectories through a submission and scores them with
the official `scoring.py`. Only held-out twins give honest numbers, because the all-data members
trained on `val_unseen`.

Round 13, `submit_ho5` (five held-out width-24 members, α 0.90, the 8 `val_unseen`
trajectories): rel-L2 0.0814 · TKE 0.4740 · MVPE 0.0694 · coverage 0.913 · sps 50.41.

```bash
python stream_eval.py --submission submit_ho5 --split unseen              # honest
python stream_eval.py --submission submit_ho5 --split unseen --aoa 15 20  # board-like subset
python validate_t2.py --zip <archive>                                     # the gate
```

## Layout

| file | what |
|---|---|
| `t2_common.py` | the evaluator's normalizer constants and a real-data stream builder in platform order |
| `submit/submission.py` | `get_ttt_model` / `reset_ttt_state` / `ttt_step`: frozen ensemble, AMP check, CUDA graph, SPS band |
| `submit/_fno.py` | a **symlink** to `Track_1/submit_fno/_fno.py`, so the two tracks cannot silently ship different models. `build_t2.py` resolves it before staging. |
| `submit_ho/`, `submit_ho5/` | the same code over round-13 HELD-OUT twin checkpoints (8 and 5 members) |
| `build_t2.py` | assembles the archive (`--ckpt`, `--band`, `--fp16` / `--int8`, `--linear`) |
| `validate_t2.py` | the gate: archive shape and size, import isolation, blocked packages, per-step contract, no aliasing, statelessness, whole-run bound consistency, timing |
| `stream_eval.py` | real trajectories through the streaming protocol, official scoring |
| `verify_space.py` | proves the normalized-space path reproduces Track 1's `predict()` (4e-6 physical) |
| `t1_vs_t2_score.py` | the two submitted archives side by side on the same windows (the Track-1-over-Track-2 `tke` gap) |
| `t2_latency.py`, `t2_profile.py`, `t2_graph_scan.py`, `t2_lat17.py` | the latency measurements |
| `ttt_offline.py`, `ttt_band.py`, `ttt_grad.py`, `readout_refit_check.py` | the four adaptation families, priced and closed |

## Two traps that cost real time here

* **`ttt_step` must return tensors the caller can KEEP.** Handing back a reused staging buffer
  makes every stored prediction equal to the last one — the run scores rel-L2 0.67 instead of
  0.074. It survives every local harness, because both the kit's `local_eval` and our
  `stream_eval` copy out of the returned tensor on the same line. `validate_t2.py` asserts it
  by keeping the returned objects and comparing them at the end; it caught exactly this on its
  first GPU run.
* **The input and target normalizer statistics are NOT the same.** `mean_in`/`std_in` are
  bit-identical to `_fno.MEAN`/`_fno.STD`, so the normalized input is what the trunk already
  wants; `mean_tgt`/`std_tgt` differ in the sixth decimal, and the output needs that affine
  correction to be exact. `verify_space.py` is the check.
