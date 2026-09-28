**18:54 EDT ROUND 2 STATUS: GREEN — T2_MRGP2 (1aa2896f), T2_MRGP2nj (9134b0ff), K-sprint track2_k27mrg_cons_a080 (68a53737). No RED on any package. Round-1 verdicts (CONS, AGGR, C2, P2) unchanged.**

# qa_t2 — X3 adversarial QA of Track 2 candidates

**18:15 EDT FINAL VERDICT: T2_P2 (pkg/c_final/T2_P2_track2_p2_c2b2.zip, sha256 8ca1a1f2...) = GREEN — EC's recommended final. T2_C2 (8d64f487), T2_CONS (d88c5d21), T2_AGGR (bad3f25a) also GREEN. No RED on any package. Worst-case wall time 34 s (graph) / 163 s (eager fallback) vs the 600 s cap. INFO: P2 costs +0.35 local ms/step over CONS (B2 +0.23, C2 +0.14), same-node.**
No RED item found. Two INFO items below (both pre-existing in the fallback, neither reachable as a failure).

Harness: `$S/qa/qa_t2_harness.py <pkgdir> <graph|eager|eagerfp32> <outprefix>` (mimics t2_v6 local_eval: reset + prev_target=None on
each trajectory's first step, CPU float32 tensors in, device "cuda", prev_target normalized with TARGET stats as the kit does).
Loaded via `spec_from_file_location` with cwd=/tmp and the extract dir NOT on sys.path. Logs: `$S/qa/out/`, `$S/qa/out2/` (final order).
`eager` = torch.cuda.CUDAGraph made to raise -> the package's own fallback (load line shows `graph NONE graph_err RuntimeError('QA: capture disabled')`).

## Per package (graph + eager; CONS also eager-fp32)
| test | CONS | AGGR | P2PROXY | C2 twin | fallback |
|---|---|---|---|---|---|
| load line: amp float16, graph CAPTURED, graph_worst 0.0 | yes | yes | yes | yes | yes |
| A 8 val traj (336 steps): contract, shapes (1,20,32,64,3) f32, finite, lower<=upper, p==0, pred in band, fresh outputs | OK | OK | OK | OK | OK |
| B same stream twice in one process: bit-identical | OK | OK | OK | OK | OK |
| C lengths 1/2/3/14 back to back (prefix == A bitwise) | OK | OK | OK | OK | OK |
| C 99-window trajectories x3 | OK | OK | OK | OK | OK |
| C no reset between traj (not reachable: runner resets at every is_first) | finite | finite | finite | finite | finite |
| C prev_target given on first step after reset (ignored, == A) / CUDA input tensors (== A) | OK/OK | OK/OK | OK/OK | OK/OK | OK/OK |
| D masked block / half domain / all-zero field (physical 0 in x and y) | OK | OK | OK | OK | OK |
| D revealed target == our previous prediction (zero realised error), also + mask | OK | OK | OK | OK | OK |
| D constant-in-time target; constant input+target | OK | OK | OK | OK | OK |
| D speed x0.2 / x3 / x7 | OK | OK | OK | OK | OK |
| D outlier 50 (frame + pixel), whole windows = 50, v spike 1e3 in input | OK | OK | OK | OK | OK |
| D p channel garbage (randn*1e3) in input and prev_target | OK (p out = 0) | OK | OK | OK | OK |
| E full A stream after all stress: bit-identical to A | OK | OK | OK | OK | OK |
| F causality (targets >= j and inputs > j replaced by noise; steps <= j bit-identical) | OK | OK | OK | OK | OK |
| eager fallback == graph, bitwise, on A | yes | yes | yes | - | yes |
| no warnings recorded | yes | yes | yes | yes | yes |

## Wall time vs the 10-minute cap (worst case 100 traj x 99 windows = 9900 steps)
graph: 3.0-3.4 ms/step -> 30-34 s. Eager fallback: ~15 ms/step -> ~150 s. Load (untimed) 1.3-3.8 s. Far under 600 s.
P2PROXY read 3.41 vs CONS 3.04 ms/step (C99 median), different nodes, so only a hint that C2+B2 costs up to ~0.3 ms locally.

## Differential vs fallback `track2_x27asnp_d1c30_a080.zip` (val_unseen 336 windows, normalized units, |pred| mean 0.78)
* CONS (VC50): pred |d| mean 0.0038, q99 0.037, max 0.55 (win 288 = traj 21600_15 window 36, t0, v channel at the x=0 edge); half |d| mean 0.00026, median 0.
* AGGR (D1c10+VC50): pred |d| mean 0.0055, q99 0.043, max 0.54 (same element); half |d| mean 0.0023.
* P2PROXY: pred |d| mean 0.0054, q99 0.050, max 0.70 (same element); half |d| mean 0.0073, half mean 0.2018 vs 0.2035.
* Largest shifts sit in late windows of long trajectories (n/(n+N0) growth) on the v channel, where normalized v fluctuations are O(1);
  a VC std gain up to ~1.5 on a fluctuation of ~1 explains 0.5. Consistent with the specs; no isolated huge element.

## INFO (not RED)
1. **NaN input poisons the next trajectory** (all packages incl. fallback): a NaN in trajectory k's input makes trajectory k+1's
   first step all-NaN after reset, because the gated updates do `state += x * gflag` and NaN*0 = NaN (R1 / D1c read the previous
   window's buffers). Not reachable: 82 train_real files have 0 non-finite values (masked = exact 0, ~10 % of elements), and a NaN
   input already makes that window's own prediction NaN in any package.
2. **R1's unscored-element test never fires on the platform** (pre-existing since R1): `(yp - r1_z).abs() >= 1e-6` uses
   r1_z = -MEAN_in/STD_in, but the official runner normalizes prev_target with TARGET stats, so masked zeros sit 6.1e-5 (u) /
   2.3e-4 (v) away and count as scored. Our `stream_eval.py` normalizes prev_target with input stats, so local R1 reads assume
   the mask works. Measured (job 7567464, CONS on val_unseen, official t2_v6 scorer): tgt-stats prev_target (platform) vs input-stats (ours):
   rel/tke/mvpe equal to 1e-3 points, sps 59.662 vs 59.668 (-0.006 points, ~-0.0015 final). Negligible; closed.

## P2 builds (pkg/c_work2, 17:30-17:33) — harness run 17:44-17:47 (jobs 7567559/60)
* All four zips: members/band/_fno.py md5-identical to T2_CONS; submission.py == P2PROXY code apart from the load print line; `_fno.FFNO_PATH` baked.
* c2b2 (graph+eager), c2 (graph), consb2 (graph), aggrb2 (graph+eager): **every test OK** (same table as above), E bit-identical after stress,
  F causality max|diff| 0, graph CAPTURED graph_worst 0.0, amp float16, eager fallback also OK.
* ms/step on 3x99-window streams (graph; DIFFERENT nodes, so a hint only): CONS 3.04, c2 3.17, consb2 3.29, aggrb2 3.28, c2b2 3.39;
  eager fallback 16.0-16.4 -> worst case 9900 steps = 163 s < 600 s.

## Same-node interleaved latency (job 7567576, one A100 node, 3 rounds x 5 packages, 7x42 steps each, median ms/step; rounds agree to 0.01)
| FALLBACK | CONS (VC50) | p2_c2 | p2_consb2 | p2_c2b2 |
|---|---|---|---|---|
| 2.925 | 3.015 (+0.09) | 3.155 (+0.14 over CONS) | 3.242 (+0.23 over CONS) | 3.368 (+0.35 over CONS) |
INFO for the decision (not a failure): B2 (TSM+BLK) is NOT free locally, +0.23 ms; at T2 slope 0.60 [0.3, 0.9] ~ +0.07..0.2 board ms
~ -0.005..-0.015 final, against B2's claimed +0.02..+0.06. C2 costs +0.14 local ms. Sent to EC 17:50.
* 18:05 eager (c2, consb2) and eager-fp32 (c2b2) re-runs: ALL OK (job 7567852).

# Round 2 (18:35-18:40) — K-sprint merged package `trick0927/FINAL/track2_k27mrg_cons_a080.zip` (sha256 68a53737...) = "KMRG"
Jobs 7568175 (harness graph / pinfail / eager / eagerfp32), 7568176 + 7568191 (9 900-step runs + `ulimit -l 65536`), 7568177/78 (same on T2_P2, T2_CONS). Logs `$S/qa/out3/`.
* **Full harness: ALL OK in graph, eager fallback, eager-fp32, and arena-allocation-failure modes** (every row of the round-1 table: contract,
  lengths 1/2/3/14/99, all value stress incl. zero-variance targets, masked blocks, speed x0.2/x3/x7, outliers; E bit-identical after stress; F causality 0).
  JOINT_GAIN and SIGPOW: finite and `lower <= upper` on every step of every stress case.
* **ARENA, 9 900 steps in one process** (100 traj x 99 windows, every returned pred/lower/upper kept, crc32 at return, re-checked at the end):
  **0 of 9 900 changed**, 0 non-finite / bad-band. No wrap: `arena_i` stops at 1024 and later steps take the clone path (checked: steps 1022-1025 are
  distinct ndarrays, `np.shares_memory` False across the 1023/1024 boundary). reset_ttt_state does not touch `arena_i`.
  Official runner: even a wrap would be harmless there — `local_eval.py` L207 `torch.as_tensor(pred)`, L249-250 `postprocess_pred` (new tensor) +
  `.cpu().numpy().astype(np.float32)` appended immediately; L261-264 the same for lower/upper. Nothing keeps our returned object.
* **Arena speed-up exists only for the first 1 024 steps**: same process, median ms/step 3.02 (steps < 1024) vs 3.21 (1024-2047) vs 3.20 (>= 2048).
  On a ~9 900-step hidden set ~90 % of steps run at the clone-path speed, so the K-sprint's -0.40 ms (measured on 336 steps) overstates the board gain
  by up to ~10x if the board set is that long. INFO for the time prediction, not a failure.
* **Memory**: host RSS after load 3.56 GB (T2_P2 / T2_CONS: 1.68-2.0 GB); after 9 900 kept outputs 17.7 GB (P2 17.1 GB; the harness keeps 3 x 9 900 outputs,
  the runner keeps the same amount). VmLck/VmPin read 0 (CUDA-pinned memory shows only in VmRSS).
  Pinned-allocation failure (monkey-patched `pin_memory` to raise for the arena): arena None, clone path, **identical outputs, ALL OK**.
  `ulimit -l 65536`: no effect (cudaHostAlloc is not RLIMIT_MEMLOCK-bound), arena allocated, ALL OK. `ulimit -v` cap: NOT RUN — CUDA context
  creation needs a large virtual reservation, so any small -v cap kills CUDA init for every package alike; not a discriminating test.
  An OOM-kill by a container memory limit cannot be caught by the try/except; +1.9 GB RSS against P2 is the exposure.
* **Output format**: numpy float32 (1,20,32,64,3), C-contiguous, writeable — identical to T2_P2 and T2_CONS.
* **Band coupling**: KMRG band.npz md5 **e8309d9f** (SIGBLEND band), load line `sigpow (0.2423, 0.2454)`, p 0.225, centre_alpha 0.0 — matched pair.
  Shipped band (T2_P2, T2_CONS) md5 0cb37d82. Realised half-width on val_unseen: KMRG median 0.131 / mean 0.195 vs CONS 0.143 / 0.204 (same order, not 3x).
* **Differential vs T2_CONS** (val_unseen): pred |d| mean 0.017, q99 0.195, max 3.94 (win 310 = 26700_0, t0, y19 x23, **v channel**): KMRG predicts 5.21
  vs target 0.87 (CONS 1.27), band half 2.34 there. JOINT_GAIN applies one u/v gain, so v fluctuations are scaled by a u-dominated gain; in normalized
  v units this is large, in physical units ~0.07 m/s. Window rel-L2 there is BETTER (0.0517 vs 0.0519); mean rel 0.06095 vs 0.06135; coverage equal (0.926).
  Consistent with the spec (INFO), not a bug.

## Round 2b (18:49-18:53) — EC's `T2_MRGP2_track2_MRGP2.zip` (1aa2896f) and `T2_MRGP2nj_track2_MRGP2nj.zip` (9134b0ff); jobs 7568364-66
* Both: full harness ALL OK. MRGP2 was run in graph, pinfail (arena allocation refused), eager, eager-fp32 and ulimit -l 65536 modes; MRGP2nj in graph and eager.
  Every mode prints the same load facts: graph CAPTURED (graph_worst 0.0) or the forced eager fallback, amp float16 (None in fp32), arena True (False under pinfail).
  Output: ndarray float32 (1,20,32,64,3), C-contiguous, writeable — same as T2_P2.
* MRGP2, 9 900 steps in one process: **0 of 9 900 kept outputs changed**, 0 non-finite / bad-band. Steps 1023/1024 do not share memory.
  ms/step median 3.42 (< 1024) -> 3.57 (>= 1024); total 35.4 s. RSS 3.57 GB after load, 17.7 GB with all 9 900 outputs kept.
* Band coupling: both zips band.npz md5 e8309d9f with sigpow (0.2423, 0.2454) — matched. Half mean 0.194 vs P2 0.202 (same order).
* Differential: MRGP2nj prediction is **bit-identical to T2_P2** (pred |d| = 0; only the band differs, from SIGPOW). That confirms the merge left the prediction path untouched.
  MRGP2 vs the K-sprint KMRG: pred |d| mean 0.002, max 0.20 (C2 + B2), as expected. MRGP2 vs T2_P2: same JOINT_GAIN v-channel pattern as KMRG (max 3.95 at win 310 v).
  pinfail / ulimit outputs == graph outputs, bitwise. fp32 differs from fp16 by < 1e-3 mean.
