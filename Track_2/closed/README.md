# Track 2 — closed experiments

The four adaptation families and the three pre-graph latency probes. Each asked a question that
is closed; the evidence tables are in [`../README.md`](../README.md) and
[`../../ROADMAP.md`](../../ROADMAP.md) Part 2. What is still worth running stays one level up:
`build_t2.py`, `validate_t2.py`, `stream_eval.py`, `verify_space.py`, `t2_common.py`,
`step_breakdown.py`, `t2_lat17.py`, `t1_vs_t2_score.py`.

**The path shim.** Each file's `HERE = Path(__file__).resolve().parent` gained one `.parent`, so
`HERE` still resolves to `Track_2/` and every `HERE / "submit"` and `HERE.parent / "Track_1"` is
unchanged; a five-line block after the docstring also puts the parent of `closed/` on `sys.path`
for the bare-name imports (`t2_common`, `sps9`, `final_v9`). All seven compile, load and pass
`--help` from here.

| file | the question | verdict |
|---|---|---|
| `ttt_offline.py` | price every BACKPROP-FREE adaptation rule offline, from a saved stream | lag-1 residual correlation +0.051; the ORACLE per-trajectory additive correction is worth 1.75 % of rel-L2, and every causal variant loses (−0.02 to −5.1 wfinal) |
| `ttt_band.py` | online SPS band calibration — the one thing Track 2 can do that Track 1 cannot | at the board's 1.39× error stretch the shipped band is already exactly optimal (best global rescale c = 1.00, +0.000 sps). Useful by-product: over stretches 1.0×–1.7× the band never gives up more than 0.667 sps |
| `ttt_grad.py` | gradient test-time training on the streamed trajectory | works and does not pay: best of 40 arms is rel-L2 −1.6 % single / −0.7 % ensembled ≈ +0.02 `final`, against ~9.7 ms/step of backward ⇒ −0.5 `final`. No lr, parameter subset, buffer or loss changes the sign |
| `readout_refit_check.py` | a closed-form ridge refit of the last 1×1 conv per trajectory, no backward pass (ALPaCA-style) | the trained readout is already near the per-trajectory optimum; the best arm removes ≤ 1 % of one error and costs 1.07 ms per member at batch 1 ⇒ ≈ −0.06 / −0.19 `final` (09-14) |
| `t2_latency.py` | what a step costs per member set at batch 1, eagerly | 3.18 ms for a single member, +2.4 ms per extra invocation, +0.09 ms per member inside a fused group — the measurement that motivated fusion |
| `t2_profile.py` | where the 3.2 ms eager floor goes | almost all of it is kernel-launch overhead; a CUDA graph removes it. Superseded by `../step_breakdown.py`, which profiles the **graphed** step that actually ships |
| `t2_graph_scan.py` | does graph capture make extra INVOCATIONS free at batch 1? | no — a graph removes launches, not FLOPs. The graph shipped; the question is closed (and on Track 1 the graph cost +0.565 board ms, R27) |
