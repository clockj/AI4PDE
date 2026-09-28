# Track 2 — LTTTA (Long-Term Test-Time Adaptation)

Codabench: https://www.codabench.org/competitions/17385/ · Official kit:
`/users/PAS1187/qwu/RealPDE/official/t2_v6/realpde_t2_starting_kit_v6/`

Same 20-in/20-out forecasting problem as Track 1, same five subscores, same hidden real
targets — but streamed one window at a time in trajectory order at batch 1, in the evaluator's
normalized space, with the *previous* window's ground truth revealed each step. The contract and
the rules are in `../CLAUDE.md`, the state in `../STATUS.md`, the specification of the package in
`../REPRODUCE.md` §13.

```python
# submission.py at the zip root
def get_ttt_model(submission_dir, device):
    # -> object with reset_ttt_state() and ttt_step(input_norm, prev_target_norm) -> (pred, info)
```

## What the stream reveals

**Windows tile at stride 20 with `T_in = T_out = 20`, so window k−1's revealed target IS
window k's input, bit for bit.** Anything computable from `prev_target_norm` alone is therefore
already computable from `input_norm`. Exactly two things are new:

1. the supervised **pair** `(x_{k-1}, y_{k-1})`: training data from the very trajectory being
   forecast;
2. the **realized error** of predictions already made.

The first was measured in every form we could think of and does not pay (gradient test-time
training, closed-form readout refit, below). **The second is what the package of record uses**:
within one trajectory the flow condition does not change, so part of the error repeats, per pixel.

## The stream stack (shipped 2026-09-27, board 82.055553)

Three corrections, all per trajectory, per pixel and channel, strictly causal, training nothing.
With every knob off the package is bit-identical to the frozen one.

| correction | reads, from the revealed windows | changes, in the current window | knobs |
|---|---|---|---|
| **R1**, interval width | the scorer's own interval term for a 9 × 9 grid of lower / upper multipliers | the lower and the upper half-width, per lead block | `R1`, `R1_ASYM`, `R1_TSM 3`, `R1_BLK 4`, `R1_BLK_N0 20` |
| **D1c**, mean bias | `mean_t(y₀ − y)`, its scatter between windows, its per-frame profile | the prediction and the interval centre, by the same shift | `D1C_N0 30`, `D1C_EB`, `D1C_EB_N0 1`, `D1C_LEAD_N0 5` |
| **VARCAL**, fluctuation amplitude | the target's temporal variance against the composed prediction's | the compose gain | `VC_N0 30`, `VC_CLIP (0.25, 4)` |

Every correction is shrunk toward "no correction" by `n/(n + N0)`, n the number of revealed
windows, so the first window of a trajectory is exactly the frozen package. The band is the
SIGBLEND band (`SIGPOW`), the compose uses `JOINT_GAIN`, as on Track 1.

| read (twins, score without `time`, against asymmetric R1 alone) | gain | trajectories positive |
|---|---|---|
| in distribution, all 8 held-out trajectories | +0.085 | 8 / 8 |
| in distribution, AoA 15–20 | +0.025 | 3 / 3 |
| a held-out AoA 20 (extrapolation) | +0.446 | 16 / 16 |
| a held-out AoA 10 (interpolation) | +0.553 | 18 / 18 |
| **the board**, against the package of 09-25 (symmetric R1 only) | **+0.390** | rel +0.100, tke +1.150, mvpe +0.263, sps +0.826 |

Designs, criteria and job scripts: `../experiments/sprint_20260927/`.

### Rules of the stream stack

* **Causal.** State is updated from a target only after the runner has revealed it, and the current
  window reads only what was accumulated before. `stream_eval.py --causal-check` replaces every
  window after a cut with noise and requires every earlier step to stay bit-identical.
* **Per trajectory.** `reset_ttt_state` clears every accumulator.
* **Never on Track 1.** A Track-1 batch holds future windows; the same statistics there read the
  answers.
* **The revealed target is normalised with the TARGET statistics** by the official runner
  (`official/t2_v6/.../local_eval.py`), the input with the input statistics. They differ in the fifth
  decimal, enough to break an exact-zero test. `PREV_SPACE` selects which one the package assumes;
  `stream_eval.py --prev-space` and `validate_t2.py --prev-space` select which one the harness feeds.
* **Masked elements.** About 9 % of the targets are exactly 0 and are not scored. Which pixels are
  masked depends on the recording, so a statistic that includes them learns the mask of this recording.

## What stays closed, with the file that closed it

| family | result | file |
|---|---|---|
| **Gradient test-time training** | works, and does not pay. Best of 40 arms: rel-L2 −1.6 % on a single member, −0.7 % on a 5-member ensemble, for ~0.5 of latency | `closed/ttt_grad.py` |
| **Closed-form readout refit** (09-14) | an exact ridge solve of the heads' last 1×1 conv per trajectory: ≤ ~1 % of any error for 1.07 ms per member | `closed/readout_refit_check.py` |
| **Trajectory history as an input** | legitimate here, and worth nothing: the model already carries what a longer context would add | `closed/ttt_offline.py` |
| **A GLOBAL residual correction or band rescale** (09-01) | lag-1 residual correlation +0.051; the best global band rescale is 1.00. **The per-pixel, per-trajectory forms above are different estimators and ship** | `closed/ttt_offline.py`, `closed/ttt_band.py` |
| **Family weights online (FAMW)** | −0.004 on AoA 15–20, +0.15 at a held-out AoA 20, +0.12 ms per step: off in the package, open for the private set | `../experiments/sprint_20260927/f_sprint/notes/design_c.md` |

## The package

* **Built together with Track 1** by `../Track_1/submit_fno/build_final.sh`, which drives
  `build_round13.sh`, `build_t2.py` and the gates with the same members, band and baked constants.
* **`submit/submission.py`** runs the ensemble in the evaluator's normalized space. The input needs
  no affine (`mean_in`/`std_in` equal `_fno.MEAN`/`_fno.STD`); the output goes through
  `mean_tgt`/`std_tgt`. It returns SPS bounds every step, in memory the caller owns. In
  `get_ttt_model`, which is untimed, it checks `AMP` against fp32 on a probe, then captures the
  whole device-side step, corrections included, in a CUDA graph, verified bit-exact against eager
  and dropped on any failure.
* **The constants in the working tree are defaults** (ALPHA 0.60, every stream knob off). The shipped
  values are baked by the build.

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

`stream_eval.py` streams real held-out trajectories through an extracted package and scores them with
the official `scoring.py`. Only a package built from the held-out twins gives honest numbers, because
the all-data members trained on `val_unseen`.

```bash
python stream_eval.py --submission <extracted twin package> --split unseen              # honest
python stream_eval.py --submission <extracted twin package> --split unseen --aoa 15 20  # board-like subset
python stream_eval.py --submission <extracted package> --causal-check                   # causality
python validate_t2.py --adaptive --zip <archive>                                        # the gate
```

## Layout

| file | what |
|---|---|
| `t2_common.py` | the evaluator's normalizer constants and a real-data stream builder in platform order |
| `submit/submission.py` | `get_ttt_model` / `reset_ttt_state` / `ttt_step`: the ensemble, AMP check, CUDA graph, SPS band, the stream stack |
| `submit/_fno.py` | a **symlink** to `Track_1/submit_fno/_fno.py`, so the two tracks cannot silently ship different models. `build_t2.py` resolves it before staging. |
| `r1_ref/` | float64 reference implementations of symmetric R1 (numpy and torch); the shipped estimator adds the asymmetric table, TSM and BLK |
| `build_t2.py` | assembles the archive (`--ckpt`, `--band`, `--fp16` / `--int8`, `--linear`) |
| `validate_t2.py` | the gate: archive shape and size, import isolation, blocked packages, per-step contract, no aliasing, whole-run bound consistency, timing; statelessness for a frozen package, `--adaptive` for one that adapts |
| `stream_eval.py` | real trajectories through the streaming protocol, official scoring; `--causal-check` |
| `verify_space.py` | proves the normalized-space path reproduces Track 1's `predict()` (4e-6 physical) |
| `t2_lat17.py`, `step_breakdown.py` | the live latency instruments: ms/step for a member set under one fast-path variant, and where a graphed batch-1 step goes |
| `closed/` | the closed adaptation families and the three pre-graph latency probes (`t2_latency.py`, `t2_profile.py`, `t2_graph_scan.py`), each with its verdict in `closed/README.md` |
| `t1_vs_t2_score.py` | the two submitted archives side by side on the same windows |

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
