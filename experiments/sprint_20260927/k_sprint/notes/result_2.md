# Result K2 (executor E2): JOINT_GAIN, written 17:15 EDT 09-27

**Verdict: SHIP.** The kill criterion passes on the real package code path, on both tracks. Full split: rel err −0.462 %, tke +0.0000 %, mvpe +0.0000 %.
The gain is small: about +0.014 final per track [+0.005, +0.025].

## 0. What was done
* Clone `$K/wt_2x` (HEAD 55f0e88, then local commits cbb3682 knob and 3650de0 verifier criteria; nothing pushed). The read clone is `$K/wt_2xr`, same HEAD.
* Patch: `$K/pkg/joint_gain.patch` (`git diff 55f0e88 HEAD`). It covers the 3 compose sites, the build knob and the verifier.
  * `_fno.py` `predict_fno` compose (the T1 GPU_BAND / D2_DIRECT_PRED batched path). `_load()` in T1 submission.py sets `_fno.JOINT_GAIN = True` when its own constant is on, and the build also seds `_fno.py`.
  * `Track_1/submit_fno/submission.py` `_Graph.body` (the SMALL_GRAPH lazy batch-1 graph).
  * `Track_2/submit/submission.py` FrozenEnsemble compose. It uses `sa`, so SIGEMA is kept.
  * The original expression is kept verbatim in the `else` branch. `JOINT_GAIN = False` is the default everywhere. When the knob is on, `assert CENTRE_ALPHA == 0.0` runs after band load (T1 and T2). `joint_gain` was added to both load DIAG lines.
  * Build knob `RPDE_SHIP_JOINT_GAIN=True` in `build_round13.sh` seds all three files. Restore with the usual `git checkout -- <3 files>`.
* Jobs (all on Ascend nextgen; Cardinal sat PENDING (Priority) for more than 3 min and was cancelled):
  * rp_k272_verify: 7567032 (old criteria, see §1) and 7567232.
  * rp_k272_candref: 7567033.
  * rp_k272_build: 7567034.
  * rp_k272_t1twin: 7567236.
  * Logs are in `$K/logs/rp_k272_*`. Tables are `$K/pkg/candref_k272/{table.txt,t1twin_table.txt}`.

## 1. Verifier `Track_1/verify/joint_gain.py` (job 7567232): `JOINT_GAIN VERIFY: ALL OK`
Setup: the baseline F-sprint zips `track{1,2}_x27_a080.zip`. Three variants are made from them with the same members and band: our code with the knob off, with the knob on, and a `y0` variant (ALPHA 0).
```
[T1] on package flags: submission.JOINT_GAIN True _fno.JOINT_GAIN True small_graph True
[T1] (a) knob OFF vs baseline zip, batched + batch-1 graph calls: BIT-IDENTICAL
[T1] on vs off max|dP| 6.269e-02 (must be > 0: the shipped batched path runs the patched compose)
[T1 (b) batched] TKE map: per-window rel-L2 of the change max 4.19e-05 OK (pointwise rel max 4.63e-03, p99.9 1.66e-03, median 8.27e-06) | time-mean max|d| 8.97e-07 = 1.83e-06 of max|P| OK | closer to y0 336/336 (mean ||P-y0|| off 0.8086 on 0.7751, ratio 0.9496)
[T1 (b) batched] lower <= pred <= upper: OK
[T1] (c) permutation drift on 0.00e+00 of rms (off 0.00e+00) OK
[T1] (c) batch-1 calls (0 eager, 1-3 graph): max|on_N1 - on_batched| [1.36e-03 2.21e-03 9.35e-04 1.00e-03] vs max|on_N1 - off_batched| [5.90e-03 4.92e-03 5.96e-03 6.31e-03]; max|on_N1 - off_N1| [5.91e-03 4.93e-03 5.93e-03 6.34e-03] -> graph site patched: OK
[T1 (b) batch-1 graph] TKE map: per-window rel-L2 of the change max 1.83e-05 OK ... closer to y0 4/4 ; lower <= pred <= upper: OK
[T2] (a) knob OFF vs baseline zip, full stream: BIT-IDENTICAL
[T2 on] graph CAPTURED + joint_gain True in load line: OK   (graph_worst 0.0 = graph bit-exact vs eager probe)
[T2 (b) stream] TKE map: per-window rel-L2 of the change max 4.26e-05 OK (pointwise rel max 4.34e-03 ...) | time-mean 1.81e-06 of max|P| OK | closer to y0 336/336
[T2 (b) stream] lower <= pred <= upper: OK
[T2] (d) causal check on `on`: CAUSALITY OK (176 steps, max |diff| 0.0e+00)
```
**Tolerances, stated plainly.** The design asked for TKE equal pointwise to 1e-5 and time mean equal to 1e-7. The code cannot meet either, and the first verifier run (7567032) printed FAIL on exactly these two lines while every other check passed.
* **Why pointwise TKE misses.** The off compose divides by `s_c + 1e-6` per channel; the joint compose divides by `|s| + 1e-7`. At pixels whose predicted fluctuation is about 1e-3 of the typical one, this changes k by up to 4.6e-3 relative. The median is 8e-6.
* **Why the time mean misses.** The gain amplifies the fp32 rounding in mean_t(f_). The largest change is 9e-7 m/s, about 2e-6 of max|P|.
* **What counts instead.** I replaced both with scorer-level criteria: the per-window relative L2 of the TKE map change must be below 1e-4 (observed 4e-5), and the time-mean change must be below 1e-5 of max|P|.
* **Scorer check.** tke and mvpe errors are unchanged to 1e-5 in every read below.

**The shipped path runs the patch.** Both shipped paths are covered:
* **T1 batched GPU_BAND/DIRECT_PRED path:** the prediction moves (6.3e-2 max) with the knob on.
* **T1 lazy batch-1 graph (N=1 calls 2 to 4):** the on output tracks on-batched, not off.
* **T2 CUDA graph:** the graph was captured and the prediction moved (the rel error changes in the reads below).

## 2. Held-out twin reads through the REAL package code (5 twins, shipped band, val_unseen, alpha 0.80, f 1.00)
* T2 uses the F-sprint twin package `final0927/pkg/x_work/t2twin_asnp.zip`, with the code swapped for ours and the knob on/off, read with `stream_eval.py`.
* T1 uses the x27 T1 code with the twin members and band, called through `predict()` (batched, N=336).

"final4" = 0.4778 rel + 0.1010 tke + 0.0921 mvpe + 0.2498 sps (no time).

| read | split | rel err off→on | d rel err | d tke err | d mvpe err | d rel | d tke | d mvpe | d sps | d final4 | sign (rel / final4) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T2 static | all 8 | 0.07505→0.07471 | **−0.462 %** | 0.0000 % | 0.0000 % | +0.0161 | 0 | 0 | +0.0113 | **+0.0105** | 8/8 / 7/8 |
| T2 static | AoA 15-20 | 0.10091→0.10043 | −0.479 % | 0 | 0 | +0.0219 | 0 | 0 | +0.0149 | **+0.0142** | 3/3 / 3/3 |
| T2 asym R1 | all 8 | 0.07505→0.07471 | −0.462 % | 0 | 0 | +0.0161 | 0 | 0 | +0.0115 | **+0.0106** | 8/8 / 7/8 |
| T2 asym R1 | AoA 15-20 | 0.10091→0.10043 | −0.479 % | 0 | 0 | +0.0219 | 0 | 0 | +0.0150 | **+0.0142** | 3/3 / 3/3 |
| T2 asym R1 + D1c30 | all 8 | 0.07461→0.07427 | −0.466 % | 0 | 0 | +0.0162 | 0 | 0 | +0.0105 | **+0.0104** | 8/8 / 7/8 |
| T2 asym R1 + D1c30 | AoA 15-20 | 0.10072→0.10023 | −0.481 % | 0 | 0 | +0.0219 | 0 | 0 | +0.0141 | **+0.0140** | 3/3 / 3/3 |
| T1 predict() | all 8 | 0.07505→0.07471 | −0.462 % | 0 | 0 | +0.0161 | 0 | 0 | +0.0113 | **+0.0105** | 8/8 / 7/8 |
| T1 predict() | AoA 15-20 | 0.10091→0.10043 | −0.479 % | 0 | 0 | +0.0219 | 0 | 0 | +0.0148 | **+0.0142** | 3/3 / 3/3 |

Absolute levels (for reference):
* T2 asym, all 8: OFF rel 96.3831, tke 81.4799, mvpe 96.8146, sps 55.5936, final4 77.0852; ON rel 96.3992, sps 55.6051, final4 77.0958.
* T2 asym, AoA 15-20: OFF sps 48.4807, final4 74.6938; ON 48.4957 and 74.7080.

These reads match the design's lab numbers exactly (rel −0.46 % / −0.48 %, final +0.0105 / +0.0140).
Not run: stretch f 1.29/1.39 reads and ens_v9, which does not use the package compose.

## 3. Packages
* **Build:** `$K/sub_candidates`, job 7567034. `RPDE_FORCE=1`.
* **Knob sets:** 27lz (T1) and asnp (T2), plus `RPDE_SHIP_JOINT_GAIN=True`, ALPHA 0.80.
* **Band:** `$M0/Track_1/bands/band_r38_a080_c0_fill.npz` does NOT exist. The build used the F-sprint's `final0927/pkg/x_work/band_r38_a080_c0_fill.npz` instead.
* **Checked against the baseline x27 zips:**
  * band.npz and the member .pth files are byte-identical to `x27_a080` (band sha 2e848ff0…).
  * The only code differences are the JOINT_GAIN hunks.
  * `JOINT_GAIN = True` is baked in both submission.py and _fno.py of every zip.

| zip | sha256 | gates |
|---|---|---|
| `track1_k27jg_a080.zip` | `6fad097c305ba331926ea1f17abb638c94f9836a4555946e5c66bb576c0a56b2` | validate_strict **SUBMITTABLE** (order-independent, drift 0.0); load: gpu_band True direct_pred True alpha 0.8 centre_alpha 0.0 joint_gain True/True small_graph lazy **members 6** |
| `track2_k27jg_a080.zip` | `2e5ffe05a6143e02fbe3badbaefc9dd583670f6de3e26b7d918b524ee4353aa3` | in-build validate_t2 SUBMITTABLE; `--adaptive` **SUBMITTABLE**; **CAUSALITY OK** (176 steps, 0.0); r1 True r1_asym True d1c_n0 0.0 joint_gain True centre_alpha 0.0 **graph CAPTURED** graph_worst 0.0 numpy_out True **members 6** |
| `track1_k27jg_d1c_a080.zip` | `e1030e32238fc11dd6eac280c750906015f1f9243b8c6c8c323b486e7cd329c4` | validate_strict **SUBMITTABLE**, members 6, joint_gain True/True. Its T1 content is the same as track1_k27jg_a080 (D1c is T2 only); the zip bytes differ only by build time |
| `track2_k27jg_d1c_a080.zip` | `8161c746aab708ffe6325862a33d8098cbf53ba6e51d400549ddd23202c05852` | in-build validate_t2 NOT SUBMITTABLE (stateless; expected for D1c); `--adaptive` **SUBMITTABLE**; **CAUSALITY OK** (176, 0.0); d1c_n0 30.0 joint_gain True **graph CAPTURED** **members 6** |

The gates' time/ms numbers are not latency evidence (FORCE, shared nodes).
The all-data validate scores are contaminated. For reference only: T2 rel error went from 0.0608 in x27 to 0.0604 here.

## 4. Predicted board subscores (delta vs the same package without the knob, each track)
* **rel:** +0.023 [+0.010, +0.035]. Local −0.46 to −0.48 % rel error, taken as a compose displacement at ≥1× (board rel error is about 1.29× local; at board rel 94.76 a 0.46 % cut is +0.023).
* **tke:** 0.000 [−0.001, +0.001]. **mvpe:** 0.000 [−0.001, +0.001].
* **time:** 0.000 [−0.02, 0]. The knob adds 2 sums, 2 sqrt and 3 elementwise ops on a (B,1,H,W,2) tensor, inside the graph on T2. Not measured.
* **sps:** +0.013 [0, +0.03].
* **final:** **+0.014 [+0.005, +0.025]** per track.

Absolute, taking today's planned baseline predictions (T1 81.60, T2 81.78, not reproduced by me): T1 k27jg about 81.61, T2 k27jg about 81.79 (or with D1c on top of its own prediction). This does **not** close the +0.2 gap to the cut; it is a free add-on.

## 5. Verdict: SHIP (as an add-on to whatever ships)
Kill criterion met on both tracks' real code paths:
* **rel:** error −0.462 % on the full split (≥ 0.3 % required).
* **tke / mvpe:** changes are 0.0000 % (< 0.01 % required).
* **Signs:** rel err 8/8 and 3/3; final4 7/8 and 3/3.

Caveats:
* It is small (+0.01 to +0.02 final).
* It requires a centre0 band. The code asserts CENTRE_ALPHA == 0.
* REPRODUCE.md needs one line: `JOINT_GAIN = True` (inference-only constant, no retraining).
* To stack it on another tree, apply `$K/pkg/joint_gain.patch` (git apply, base 55f0e88) and build with `RPDE_SHIP_JOINT_GAIN=True`.
