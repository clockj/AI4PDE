# Track 2 — LTTTA (Long-Term Test-Time Adaptation)

Codabench: https://www.codabench.org/competitions/17385/ · Official kit:
`~/RealPDE/official/t2_v6/realpde_t2_starting_kit_v6/`

Same 20-in/20-out forecasting problem as Track 1, same five subscores, same hidden real
targets — but streamed one window at a time in trajectory order at batch 1, in the evaluator's
normalized space, with the *previous* window's ground truth revealed each step.

```python
# submission.py at the zip root
def get_ttt_model(submission_dir, device):
    # -> object with reset_ttt_state() and ttt_step(input_norm, prev_target_norm)
```

---

## The one fact that determines everything here

**Windows tile at stride 20 with `T_in = T_out = 20`, so window k−1's revealed target IS
window k's input, bit for bit.** (Verified in Track 1 to `maxabsdiff 0.0`.)

Therefore **anything computable from `prev_target_norm` alone is already computable from
`input_norm` — and Track 1 could compute it too.** Exactly two things are genuinely new:

1. the supervised **pair** `(x_{k-1}, y_{k-1})`: in-distribution training data from the very
   trajectory being forecast, which is the only thing that attacks the project's binding
   constraint (74 real training trajectories);
2. the **realized error** of predictions already made.

Everything an adaptation scheme can do lives in one of those two. Both were measured before
anything was built, and both are closed — see below.

This also disposes of `CLAUDE.md`'s "strongest T2-specific play" (EMA-adapt the variance head
toward the revealed previous window's per-pixel std). That std *is* the current input window's
std, the zero-parameter anchor measured at rel-L2 0.501 against our σ head's 0.329. It was
never a Track-2 lever.

## What is closed, with the file that closed it

| family | result | file |
|---|---|---|
| **Residual EMA / output bias correction** | lag-1 residual correlation **+0.051** (full field), +0.074 (time-mean). ORACLE per-trajectory additive correction: rel-L2 0.0800 → 0.0786, **1.75 %**. Every causal variant loses (−0.02 to −5.1 wfinal). | `ttt_offline.py` |
| **Online SPS band calibration** | at the board's 1.39× error stretch the shipped band is **already exactly optimal** (best global rescale c = 1.00, +0.000 sps); 1.0× leaves 0.667, 1.2× 0.114, 1.7× 0.276. Causal estimators recover none of it (−0.9 to −10). | `ttt_band.py` |
| **Gradient test-time training** | works, and does not pay. Best of 40 arms (all params, Adam 1e-5, rel-L2 loss, last pair): rel-L2 **−1.6 %** single member / −0.7 % on a 5-member ensemble, TKE and MVPE also down. Worth ≈ +0.02 `final`; costs ~9.7 ms/step of backward ⇒ **−0.5 `final`**. No lr, parameter subset, buffer or loss changes the sign. | `ttt_grad.py` |

**A Track-1 result falls out of the same sweep, and it matters for the Decision phase.** The
band is **robust to error mis-specification**: stretching the per-element errors anywhere from
1.0× to 1.7×, the shipped band never gives up more than **0.667 sps (≈ 0.14 `final`)** against
the best rescaling chosen in hindsight. So a Decision-phase test set that is meaningfully
harder or easier than this one does not break it, and it does not need refitting for that
reason.

**Conclusion: Track 2 is Track 1's model, streamed.** Its value is a second, equal prize pool
for work already done — not a new algorithmic lever. Ship the forecaster frozen.

## What IS worth engineering here: latency

> **Rounds 23–25 update, measured on the board.**
> * **Invocations.** One extra invocation costs **0.866 board ms** on Track 2, against 1.151 on
>   Track 1. Track 2's handicap over Track 1 is a fixed per-step cost, not a per-invocation one.
> * **FLOPs are charged too.** Four more FNOs inside the fused group cost +1.37 board ms, about
>   0.60 board ms per 4060-equivalent local ms (a single reading).
> * **fp16 autocast** is baked as `AMP` in `submit/submission.py`. It is verified against fp32
>   in `get_ttt_model` before the CUDA graph is captured, and the graph is kept. On R24's
>   package it cut the step from 7.022 to 3.768 ms (interleaved, one process), with `ttt_step`
>   outputs within 8.4e-4 of fp32.
> * **AMP is a per-package choice measured on the board.** R21's package read fp16 *slower* in
>   a separate-process run. The gate is `Track_1/verify/amp_path.py --t2`.

Track 1 amortizes one batched call over every window. Track 2 pays the full launch cost every
step, and `ttt_step` is timed in its entirety. Measured on an A100-PCIe at batch 1:

| members | invocations | eager ms | **CUDA-graph ms** | graph `time_score` |
|---|---|---|---|---|
| 11 (Track 1's round-12 set) | 4 | 10.39 | 2.235 | 94.75 |
| 10 | 3 | 7.80 | 1.815 | 95.25 |
| **8 (w24 seeds + m1220)** | **1** | 3.61 | **0.989** | **96.45** |
| 1 | 1 | 3.01 | 0.468 | 97.53 |

* A single member costs **3.18 ms eagerly** and the arithmetic is a small fraction of it — the
  floor is kernel-launch overhead. An extra *invocation* costs ~2.4 ms; an extra member inside
  a fused `StackedFNO` group costs 0.09 ms.
* **Capturing the whole device-side step in a CUDA graph is bit-exact (max abs difference
  0.0e+00) and cuts 3.61 ms to 0.989.** At the fitted `time` weight that is ≈ **+0.25 `final`**
  for no change in any other subscore. Capture happens in `get_ttt_model`, which is not timed,
  and falls back to eager if it ever fails.
* The board has already priced the accuracy of the 3 members the 8-member set drops: **−0.010
  `final`** (R11 vs R12 subscore deltas). They pay latency for nothing.
* The 10-minute wall clock is not binding: 336 steps run in **0.61 s**.

## Measured

**Honest** (`submit_ho5` — the held-out twins of the shipped recipe, on the 8 `val_unseen`
trajectories, streamed):

| | rel-L2 | TKE | MVPE | coverage | sps |
|---|---|---|---|---|---|
| 5 held-out w24 members, α 0.90 | 0.0814 | 0.4740 | 0.0694 | 0.913 | 50.41 |

**Shipped** (all-data members; `val_unseen` is training data for them, so these are memorization
and are quoted only to show the package runs): rel-L2 0.0742, TKE 0.3841, MVPE 0.0599,
sps 52.70, **1.817 ms/step** on an A100-PCIE, whole 336-step stream in 0.61 s.

Reproduce with:

```bash
python stream_eval.py --submission submit_ho5 --split unseen            # honest
python stream_eval.py --submission submit_ho5 --split unseen --aoa 15 20  # board-like subset
python validate_t2.py --zip submission_t2.zip                            # the gate
```

## Layout

| file | what |
|---|---|
| `t2_common.py` | the evaluator's normalizer constants and a real-data stream builder in platform order |
| `submit/submission.py` | `get_ttt_model` / `reset_ttt_state` / `ttt_step`; frozen ensemble, CUDA graph, SPS band |
| `submit/_fno.py` | a **symlink** to `Track_1/submit_fno/_fno.py` — one source of truth, so the two tracks cannot silently ship different models. `build_t2.py` resolves it before staging. |
| `submit_ho/` | the same code over the HELD-OUT twin checkpoints — the only honest local numbers |
| `stream_eval.py` | streams real trajectories through a submission and scores with the official `scoring.py` |
| `verify_space.py` | proves the normalized-space path reproduces Track 1's `predict()` (4e-6 physical) |
| `validate_t2.py` | the submission gate: contract, import isolation, aliasing, statelessness, timing |
| `build_t2.py` | assembles the archive |
| `t2_latency.py`, `t2_profile.py`, `t2_graph_scan.py` | the latency measurements above |
| `ttt_offline.py`, `ttt_band.py`, `ttt_grad.py` | the three adaptation families, priced |

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
