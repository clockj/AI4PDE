# How we evaluate locally, and how the board corrects it

The competition publishes one number per submission and one submission per track per day. Every
other number in this repository is a local estimate of that number, and each one is wrong in a
measured, named way. This file says where the pipeline lives, what each instrument reads, and
which constant corrects it. It is the map; the evidence is in `ROUNDS.md`, the verdicts in
`ROADMAP.md` Part 2, and the standing rules in `CLAUDE.md`.

Every constant quoted below was read out of the code on 2026-09-19, not out of prose. Where a
document disagreed with the code, the code is what is printed here and the disagreement is noted.

---

## 1. The pipeline, end to end

```
 ~/RealPDE/train_real  (82 traj, u,v, 607 frames)      ~/RealPDE/train_sim  (100 traj, u,v,p)
        |                                                        |
        |  unet/model.py: [::2,::2] -> 32x64, T_in=T_out=20,     |  unet/simreal.py maps sim into
        |  stride 20, (filename,time_id) order, tail padded      |  real's measurement space
        v                                                        v
   SPLITS  -- model.VAL_TRAJ (8 traj) | re_split.excluded() (26) | kfold.py folds | AoA holdout
        |
        v
   TRAINING  unet/train2.py  (--all-data = shipping weights; --sim-exclude-held = held-out TWIN)
        |                     jobs + logs on /fs/scratch/PAS1187/qwu/tmp/realpde/
        |        all-data checkpoints supply WEIGHTS; twins supply EVIDENCE and the band's errors
        v
   INSTRUMENTS (local, every number a SIGN)
        compare_v9 -> ens_v9 -> kfold --split re --bands 1 2 3      (in-distribution)
        aoa_holdout_eval --holdout-aoa                              (out-of-distribution)
        screen_sets / screen_eval -> leaderboard/screen.jsonl       (screen arms)
        |
        v
   BAND   submit_fno/fit_band.py  (per-element SPS half-widths, error field from the TWINS)
        |
        v
   PACKAGE  submit_fno/build_round13.sh -> build_zip.py + Track_2/build_t2.py
        |     bakes ALPHA / AMP / SPEC_PATH / HOST_WRITE / FFNO_PATH / int8 into submission.py
        v
   GATES  submit_fno/validate_strict.py (T1, GPU, TIMES the package) | Track_2/validate_t2.py
          verify/*.py (scorer diff, fusion, packing, AMP, graph, blend, pkg identity)
        |
        v
   PREDICTION OF RECORD  predict_board.py --anchor <recent round of the same shape>
        |                five subscores + a bracket, written into ../sub_candidates/README.md
        |                BEFORE the user uploads.  One written afterwards is worthless.
        v
   UPLOAD (the user, never an agent) -> Codabench -> one board row, five subscores
        |
        v
   CALIBRATION  board_rounds.py (ROUNDS / ROUNDS_T2, latency slopes) -> final_v9.py:BOARD (weights)
                predict_board.py ANCHORS + RATIO + F_TAIL + LAT_*  -> leaderboard/board.jsonl
                lb.py --html (refits the ratios and the residual history)
        |
        +------> back into the instruments: the ratios, the tail, the slopes and the anchors
                 are what make the next local number readable.
```

---

## 2. The scorer

| | path |
|---|---|
| Official Track 1 kit (authority) | `~/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9/scoring.py` |
| Official Track 2 kit | `~/RealPDE/official/t2_v6/` |
| Our reconstruction (what everything imports) | `~/RealPDE/realpde_t1_starting_kit_v9/realpde_t1_starting_kit_v9/scoring.py` |
| Proof they agree — synthetic arrays | `Track_1/verify/diff_scoring_official.py` |
| Proof they agree — real `val_unseen` windows | `Track_1/verify/diff_scoring_real.py` |

The reconstruction was written before the kit was reachable from the cluster; it agrees with the
official program to **2.4e-6**. `Track_1/sps9.py` loads the reconstruction by path
(`importlib.util.spec_from_file_location`) and re-exports it as `sps9.sc`, so every instrument in
the repo scores through one module. Anything that reports a number it did not get from `sps9.sc`
is a re-implementation and is not trusted (`CLAUDE.md`, *Engineering traps*).

### The five subscores

| subscore | formula | what it reads |
|---|---|---|
| `rel_l2` | `100/(1 + 0.5·e)` | relative L2 per window, `u, v` only |
| `tke` | `100/(1 + 0.5·e)` | rel-L2 of `0.5·(var_t u + var_t v)` over the 20 output frames — phase-free |
| `mvpe` | `100/(1 + 0.5·e)` | rel-L2 of the time-mean at probes x ∈ {13,21,29,37}, y ∈ {8,10,…,24}, **per x-station**, averaged |
| `time` | `100/(1 + sqrt(t/0.72896))` | mean board **seconds** per step; 0 or non-finite ⇒ 0 |
| `sps` | `100·w` | per scored element (target inside `[lower, upper]`, non-zero): `(1−pm)·exp(−(upper−lower)/σ)`, `pm = e/(0.5+e)` per branch, branches DM 0.5 / TKE 0.3 / MVPE 0.2 |

`sps9.SIGMA = 0.0563870259` (it is `sc.SIGMA_GLOBAL`, read from the kit, not typed here).
Malformed bounds — wrong shape, non-finite, `lower > upper` — zero **every** subscore.

### SPS factorizes, so the band is a sort

`sps = 100 · mean_i(a_i) · containment`, with `a_i = 0.5·g(rel_i) + 0.3·g(tke_i) + 0.2·g(mvpe_i)`
and `g(m) = 0.5/(0.5+m)`. The per-sample factor is forced by the other three subscores;
containment is the residual. `Track_1/sps9.py:optimal_band` solves the exact per-element optimum
by sorting, which is why a band fit is seconds rather than a sweep, and why a round's `sps` can be
decomposed into "forced" and "containment" after the fact (that decomposition is how T2R14's
−0.310 and T2R15's −0.132 containment losses were read).

### The `final` rule is fitted, not published

`Track_1/final_v9.py` holds `BOARD`, every scored round's five subscores plus its `final`, pooled
across both tracks (`board_rounds.POOL_TRACKS = True`), and solves a free linear fit with no
Σw = 1 constraint. **As of 2026-09-19 it carries 40 rounds** (`board_rounds.ROUNDS` 24 on Track 1,
`ROUNDS_T2` 16) and fits

```
rel_l2 0.477825 · tke 0.101020 · mvpe 0.092077 · time 0.096548 · sps 0.249815
Σw = 1.017285      max |residual| = 0.003165      TOL = 0.006331
```

`final_v9` self-checks at import: it asserts that `BOARD` and `board_rounds` hold the same rounds
as a multiset and that every residual is inside `TOL`, so a mis-pasted row fails immediately.
`board_rounds.W_TIME_DIRECT = 0.095906` is `w_time` read independently, from pairs of rounds whose
four error subscores are identical — it agrees with the fit's 0.0965 and is the reason the fit is
believed at all.

> Prose drifts from this fit every time a round is pasted in. On 2026-09-19 `CLAUDE.md` and
> `leaderboard/README.md` both still said "32 rounds … Σw 1.018 … max residual 0.0025", which was
> true before the 09-18/09-19 rounds; `CLAUDE.md` and this file were corrected, `leaderboard/README.md`
> was not. **`final_v9.py` is the authority — print `W0` rather than quoting any document.**

**Exchange rate**, containment held fixed, per 10 % of error removed: `rel_l2` **+0.346**,
`tke` **+0.290**, `mvpe` **+0.100** of `final`. `Track_1/exchange_v9.py` is the script that
produces it (held-out 6-member ensemble on the AoA 15–20 proxy, errors synthetically scaled,
board tail applied, band refit at every scale).

---

## 3. Held-out splits — which one answers which question

| split | defined in | size | answers |
|---|---|---|---|
| `VAL_TRAJ` | `Track_1/unet/model.py` | 8 trajectories: `3750_0 6300_10 8850_20 11400_20 13950_10 16500_5 21600_15 26700_0` | the default honest split; `val_unseen.npz` is its cache, `val_seen.npz` the matched in-training control |
| AoA 15–20 subset of `val_unseen` | `--aoa 15 20` on `compare_v9` / `ens_v9` | 126 fitting windows / a 15–20 view for reporting | **the board-like proxy.** Error is governed by AoA and almost nothing else (out-of-fold rel-L2 0.052 / 0.053 / 0.086 / 0.106 / 0.121 at AoA 0…20), and this slice reads 0.1082 against the board's 0.1179 where the full split reads 0.0807 |
| full split | no `--aoa` | 336 windows | the level, not the board's regime |
| Re extrapolation | `Track_1/re_split.py` (`excluded()` derives **26** trajectories) | contiguous Re bands | **retired as a decision instrument**: R14's gate fired on zero hidden windows, so the public hidden set is interior in Re |
| interior Re k-fold | `Track_1/kfold.py --split re --bands 1 2 3 --paired` | all 82 trajectories, out-of-fold | **the confirmation step.** Overstates `rel_l2` by ~1 %; the signs of `tke` and `sps` are right |
| **AoA holdout** | `Track_1/aoa_holdout_eval.py --holdout-aoa {0,5,10,15,20}` | one angle removed from training *and* from the band's fit split | **the only OOD instrument we have**, and the one that speaks to the PRIVATE test set (novel AoA and Re). `--fit-set val_traj` (8 traj, the 09-14 screen protocol) or `re_excluded` (26) chooses the in-distribution list the arm also excluded; the band is fitted there |

`aoa_holdout_eval.py --inflate-sweep A B C` refits the band at each stretch and reads `sps` on
**both** splits, so the width that is right for a novel angle arrives with its in-distribution
price. Run on 09-19 it closed the band width for a novel angle: 1.39 is already the held-out
optimum.

**The rule that governs all of them: all-data checkpoints supply weights, never evidence.**
Rounds 7, 8 and 10 were each misled by a number computed from a model that had trained on the
split it was scored on. Evidence comes from the held-out twins, and so does the band's error
field — a band fitted on memorized errors scored `sps` 36.94 in round 6.

**A held-out list is derived, never typed** (`re_split.excluded()`, `kfold.py --folds`): the
(Re, AoA) grid is incomplete, so a hand-written `--exclude` silently trains on a held-out
trajectory.

---

## 4. Instrument catalogue

Every local number is a **sign, not a size**. The proxy over-reads error CUTS by about 2×; a
member REMOVAL is charged in full (T2R15); across model FAMILIES the proxy predicts neither size
nor sign (R16, R26) while a *board* read of a family transfers at 0.85× (R28).

**Instrument argparse defaults are NOT the operating point.** The package ships `ALPHA = 0.60`
and `band_p = 0.45`; the defaults below are older. Always pass `--alpha 0.60 --band-p 0.45`.

### Accuracy

| instrument | measures · invocation at the operating point | writes | known bias |
|---|---|---|---|
| `Track_1/compare_v9.py` | one checkpoint at a time, each with its **own** band. `--ckpt unet/x.pth --alpha 0.60 --aoa 15 20` | stdout | defaults `--alpha 0.75 --inflate 1.5`; a single model is not the shipped ensemble |
| `Track_1/ens_v9.py` | member **SETS** — the screen at the size that ships. `--sets "a.pth,b.pth" "a.pth,b.pth,c.pth" --alpha 0.60 --aoa 15 20` | stdout | defaults `--alpha 0.75`, `--time-score 90.5066` (round 8's) |
| `Track_1/screen_sets.py` | a screen arm's member sets by token (`FNO3`, `VIT2`, …). `--arm <arm> --set "S1=BASE" "S2=BASE,ckpt"` | delegates to `screen_eval.run_main("ens_v9", …)` | inherits `ens_v9`'s |
| `Track_1/screen_eval.py` | a screen arm end to end: accuracy, latency, projected `final`. `--arm <arm> --ckpt … --wave N --lat LAT.json` | upserts `leaderboard/screen.jsonl` under a lock; JSON in `Track_1/plan/` | projects with **gains halved, losses in full, board ms floored** (the 09-14 calibration) |
| `Track_1/kfold.py` | out-of-fold over all 82 trajectories, with error bars. `--score --split re --bands 1 2 3 --paired --alpha 0.60` | JSON in `--out-dir` (scratch) | defaults `--alpha 0.90`; overstates `rel_l2` ~1 % |
| `Track_1/aoa_holdout_eval.py` | a novel ANGLE, with the shipped band machinery. `--ckpt … --holdout-aoa 20 --fit-set val_traj` (defaults are already `--alpha 0.60 --band-p 0.45 --inflate 1.39`) | `--json` | one angle is one draw; the OOD seed σ is 0.71 `wfinal`, so replicate |
| `Track_1/proxy_calib.py` | board/proxy ratios per round, and the proxy's resolution (GPU, once per round) | stdout | it is the thing that measures the bias |
| `Track_1/predict_board.py` | the five board subscores **before** upload. `--ckpt <all-data> --err-ckpt <twins> --alpha 0.60 --invocations N --local-ms M --anchor <round> --anchor-err-ckpt …` | stdout → `../sub_candidates/README.md` | predicts a LEVEL to ±0.15; cannot resolve a change below ~2 % of error |
| `Track_1/make_val.py` · `Track_1/exchange_v9.py` · `Track_1/sub_value.py` · `Track_1/alpha_board.py` · `Track_1/calib_band13.py` · `Track_1/quant8.py --score BAND.npz` | the val caches every instrument reads · the exchange rate · a slot's information value · ALPHA at the board's error scale · the board's error tail · the **scored** cost of a packing scheme | stdout (`make_val` writes `Track_1/val_*.npz`, untracked, ~1 GB) | each is a single-purpose calibration script; see §5 for the constants three of them produce |

### Latency — quote a bracket, never a point

| instrument | measures | bias |
|---|---|---|
| `Track_1/screen_latency.py` | candidates and the shipped references interleaved in one process (**kernel level**). `--out LAT.json --cand <family:spec> …`, then `--pick LAT.json` | reads ~5× the package-level instrument (R28: +0.797 against +0.164; the board gave +0.404) |
| `Track_1/lat_r15.py` | end-to-end local ms/window for a member set (**package level**), priced into board ms | the other end of that bracket |
| `Track_1/lat_fast17.py` | what a fast path (fp16 / bmm / host-write) costs and changes | a precision cut is charged at `PREC_SLOPE_T1`, not `LOC_SLOPE_T1` |
| `Track_2/step_breakdown.py --dir EXTRACTED_T2_DIR` · `Track_2/t2_lat17.py` | where one graphed batch-1 `ttt_step` goes · Track-2 ms/step for a member set under one fast-path variant | batch 1 amortizes nothing; both need an idle node to count as evidence |

### Gates and verifiers

| gate | what it proves |
|---|---|
| `Track_1/submit_fno/validate_strict.py --band <band.npz> [--fp16 …]` | the Track-1 contract under the harshest load path, through the kit scorer — **and the package's local ms**. Run by `build_round13.sh`; GPU; must not run while an `rp_*`/`pz_*` job is alive |
| `Track_2/validate_t2.py --zip submission_t2.zip` | the Track-2 contract: isolation, no aliasing, statelessness, whole-run bound consistency (`--device cpu` on a login node) |
| `Track_2/stream_eval.py --submission submit_ho5 --split unseen [--aoa 15 20]` | honest Track-2 numbers: real trajectories through the streaming protocol, official scoring. Only held-out twins are honest here |
| `Track_2/verify_space.py` | the normalized-space path reproduces Track 1's `predict()` (4e-6 physical) |
| `Track_1/verify/pkg_identity.py --track t1 --a <old> --b <new>` | two BUILT archives return the same arrays. Hand it the archive **actually uploaded** (`../sub_candidates/submitted/`) |
| `Track_1/verify/blend_path.py --dir … --beta …` | the gated mean-blend, which **no local data exercises** — `validate_strict.py` passes such a package as if the feature were absent |
| `Track_1/verify/amp_path.py --t1 EXTRACTED_T1_DIR [--t2 …]` | a baked `AMP` runs fp16, scores fp32's numbers, and is faster (interleaved) |
| `leaderboard/lb.py [--html]` | the three ledgers, the fitted local→board correction, the published page. Idempotent |

Two rules sit above all of it. **Screen at the ensemble size that ships** — error an arm removes
per member vanishes in the ensemble average. **Refit everything downstream (band, ALPHA) before
comparing a changed member set**; fitting `base` once at `p = 0` is what made round 6 declare the
band closed.

Closed one-off diagnostics live in `Track_1/closed/` and `Track_2/closed/`, each with a README row
naming the question it asked and the verdict.

---

## 5. How board results correct the local numbers

Each constant below lives in code, was derived from named rounds, and has a re-derivation
procedure. Work in **deltas against an anchor**, never in levels: `predict_board.py --anchor`
cancels `LAT_C` and the eval machine's drift.

| constant | file · symbol | derived from · re-derived by |
|---|---|---|
| **subscore weights** — five subscores → `final` | `Track_1/final_v9.py:BOARD` → `W0` | all 40 scored rounds, pooled. Paste the new row into `board_rounds.ROUNDS`/`ROUNDS_T2` **and** `final_v9.BOARD`; the refit and the self-check run at import |
| `w_time`, read independently | `board_rounds.W_TIME_DIRECT = 0.095906` | five pairs of rounds whose four error subscores are identical (first R13→R14, largest R20→R23). Next such pair: divide the `final` delta by the `time` delta |
| **board/proxy error ratios** — proxy error → board error in non-anchor mode | `predict_board.RATIO = {rel 1.078, tke 1.150, mvpe 1.511}` | the mean of R10 / R11 / R12 reconstructed by `proxy_calib.py` (rel 1.081/1.063/1.090, tke 1.116/1.167/1.167, mvpe 1.451/1.537/1.543). Re-derive with `proxy_calib.py` on the new round (GPU) |
| **SPS element tail `f`** — how much wider the board's per-element errors are | `predict_board.F_TAIL = 1.360`; `fit_band.py --inflate` **default 1.39** | inverted from a board `sps` with a known band by `calib_band13.py` (1.360 T1 / 1.350 T2). The gap to 1.39 is deliberate: refitting at 1.36 buys 0.000, and the 09-19 `--inflate-sweep` found 1.39 is also the held-out optimum for a novel angle |
| **invocation cost** — one extra sequential forward | `board_rounds.INV_MS_T1 = 1.15111`, `INV_MS_T2 = 0.86558` (`predict_board.LAT_INV_T1/T2`) | R23 / T2R10: round 20's own checkpoints, packing and band with `FUSE = False` the only change; the error subscores came back identical to six decimals. Untested since |
| **FLOPs slope T1, within a family** | `board_rounds.LOC_SLOPE_T1 = 1.75` | R16→R17 (1.76), R21→R24 (1.74). Board ms per 4060-equivalent local ms |
| **FLOPs slope T1, cross-family** | `LOC_SLOPE_T1_XFAM = 0.89` | R28: +0.455 4060-eq bought +0.404 board ms. Bracket a cross-family candidate `[0, 1.75]` — Swin inverted outright |
| **FLOPs slope T2** | `LOC_SLOPE_T2 = 0.60` | four readings (0.60 / 0.937 / 0.529 / ≈0.28); ships the median. **Bracket every Track-2 latency lever with 0.3 and 0.9** |
| **precision slope** | `PREC_SLOPE_T1 = 0.65` | R24→R25: fp16 kept 37 % of its local cut. A precision change is charged less than a FLOPs change |
| **A100 → 4060-equivalent** | factor **1.850**, in `board_rounds.py`'s per-round notes | round 12. Makes a local ms comparable to the slopes above |
| latency intercept | `predict_board.LAT_C = 3.274` | the eleven one-invocation rounds. It cancels in `--anchor` mode, the only mode anyone should use |
| **anchors** | `predict_board.ANCHORS` — **30** rounds (`r10`…`r29`, `t1r14`, `t1r15drop`, `t2r2`…`t2r16L8`) | add the new round as its five actual subscores plus `final`; anchor on one of the **same shape** |
| **"halve a local delta"** | `screen_eval.py` `FORMULA`: gains halved, losses in full, board ms floored | R24 (the proxy over-reads error cuts ~2×) and **T2R15**, where the *un-halved* AoA 15–20 ratios predicted rel 94.811 / tke 78.457 / mvpe 93.886 against the actual 94.823 / 78.519 / 93.886. **Halve only an arm that REMOVES error; charge a member REMOVAL in full** |
| **cross-track transfer 0.85×** | `board_rounds.py`'s R28 note | R28 predicted from the T2R13−T2R12 **board** pair: rel +0.029 / tke +0.194 / mvpe −0.161 / sps +0.302 predicted, +0.029 / +0.159 / −0.142 / +0.272 delivered — every row the right sign, ~85 % of the size. A **proxy**-measured delta does not transfer at all |

> **Discrepancy worth knowing.** `CLAUDE.md` and `leaderboard/README.md` quote the board/proxy
> ratios as "rel ≈ 1.07, tke ≈ 1.14, mvpe ≈ 1.45" — that is `proxy_calib.py`'s latest measurement
> (1.068 / 1.136 / 1.454). The constants `predict_board.py` actually ships are the three-round
> means **1.078 / 1.150 / 1.511**. Both are defensible; only the second one runs.

### Track record — prediction of record against the board

From `leaderboard/board.jsonl` (`pred` written before upload, `actual` after). `Δ` is
actual − predicted for `final`.

| round | date | pred `final` | actual `final` | Δ | where the miss was |
|---|---|---|---|---|---|
| R25 (T1) | 09-13 | 81.356 | 81.344644 | −0.011 | `time` 92.416 → 92.298 |
| T2R12 | 09-13 | 81.223 | 81.164374 | −0.059 | `time` 91.691 → 91.040 |
| R26 (T1) | 09-16 | 81.453 | 81.218444 | **−0.235** | Swin: `time` 93.24 → 91.528 **and** rel/tke/mvpe all below — a cross-family proxy read |
| T2R13 | 09-16 | 81.237 | 81.314785 | **+0.078** | F-FNO under-predicted: `sps` 41.818 → 42.063 (containment) |
| R27 (T1) | 09-17 | 81.428 | 81.307815 | −0.120 | the CUDA graph: `time` 93.164 → 91.915. Errors bit-identical, as the fingerprint intended |
| T2R14 | 09-17 | 81.371 | 81.308359 | −0.063 | `sps` 42.015 → 41.791: containment −0.310, which no local instrument reads |
| R28 (T1) | 09-18 | 81.387 | 81.401196 | **+0.014** | first Track-1 slot both predicted and won; `time` beat the estimate because the cross-family slope is 0.89, not 1.75 |
| T2R15 | 09-18 | 81.431 | 81.265138 | **−0.166** | member removal: `sps` −0.321 of which containment −0.132, and the halving rule mis-applied |
| R29 (T1) | 09-19 | 81.400 | 81.400174 | +0.000 | a null fingerprint; four error subscores bit-identical to R28 |
| T2R16 | 09-19 | 81.327 | 81.323990 | −0.003 | `sps` 42.113 → 42.057: the unscored-element fill bought nothing on this hidden set |

The pattern: with an anchor of the **same shape**, the four error subscores land within a few
thousandths and `time` carries the miss. The three misses above 0.1 are all structural — a new
family (R26), a host-side latency claim (R27) and a member removal (T2R15).

---

## 6. Procedures

### Before an upload

1. Commit first. `build_round13.sh` restores the tree with `git checkout`, and it `sed`s constants
   into both `submission.py` files and leaves them there.
2. Refit the band if the member set or ALPHA changed — error field from the **held-out twins**.
3. Build both tracks from one member set, passing **every** knob (an unset one falls back to older
   behaviour):
   ```bash
   RPDE_A="<all-data members>" RPDE_M="<held-out twins>" RPDE_BAND=Track_1/bands/<band>.npz \
   RPDE_INT8=1 RPDE_LINEAR=tensor RPDE_SHIP_AMP='"fp16"' RPDE_SPEC_PATH=bmm RPDE_HOST_WRITE=torch \
   RPDE_TAG=<tag>_a060 bash Track_1/submit_fno/build_round13.sh 0.60
   ```
   It refuses while other `rp_*` jobs run (`validate_strict.py` times the package) and it does
   **not** propagate its gates' exit codes — read the `SUBMITTABLE` line per track.
4. Run the verifier for anything no local data exercises (`verify/blend_path.py`,
   `verify/amp_path.py`, `verify/graph_path.py`). The strict gate passes such a package as if the
   feature were absent.
5. `predict_board.py --anchor <a recent round of the same shape>` and write **all five** predicted
   subscores plus a bracket into `../sub_candidates/README.md` §NOW, and a PENDING row with that
   `pred` into `leaderboard/board.jsonl`. Bracket `time` with the measured slopes; for `sps`, the
   per-sample factor is the estimate and 1:1 the optimistic end — never bank containment.
6. The **user** uploads. One submission per track per UTC day. HOLD is a valid decision: never
   upload a package expected to score worse just to fill a slot.

### After a board result

1. `leaderboard/board.jsonl` — fill `actual` (five subscores + `final`) on that round's record.
2. `leaderboard/runs.jsonl` — set `shipped` on the arms it carried.
3. `Track_1/board_rounds.py` — paste the row into `ROUNDS` (T1) or `ROUNDS_T2` (T2) with the note
   that says what it measured, and update any latency slope it re-measures.
4. `Track_1/final_v9.py:BOARD` — the same five subscores plus `final`, in submission-date order.
   Import it once: it refits the weights and self-checks both lists.
5. `Track_1/predict_board.py` — add the round to `ANCHORS`; give the `board.jsonl` record a
   `backbone`; add a `LEADER_READS` row in `lb.py` when the leaders are read.
6. `python leaderboard/lb.py --html` and republish the page to the same link (it is in the latest
   handoff).
7. `ROUNDS.md` — append the write-up. `ROADMAP.md` Part 1 — new rules. `CLAUDE.md` — *Current
   state* plus every fact the round changed, staying under 30 000 characters.
8. `../sub_candidates/` — archive → `submitted/`, §NOW → `HISTORY.md`, the band → `Track_1/bands/`
   with a `MANIFEST.md` line.
9. A dated handoff in `../handoffs/`, then one commit (≤ 7/day on `qwu/dev`).

---

## 7. What no local instrument can read

| unreadable | why, and how we handle it |
|---|---|
| **containment** (the residual half of `sps`) | the band is refit per package and every refit scores well on its own twins, so nothing local reads the *board's* coverage. Never bank it; recover it afterwards by subtracting the forced `a`-factor part from the board's `sps`. It is where T2R14 lost 0.310 and T2R15 0.132, and where the second family earns its place |
| **the board's latency LEVEL** | the eval machine drifts by date and the two local timers disagree by 5× (R28: +0.797 kernel-level, +0.164 package-level, board +0.404). Anchor on a recent round of the **same shape** and quote both reads as the bracket. Never time while an `rp_*`/`pz_*` job is alive; only interleaved arms count |
| **a cross-FAMILY delta** | the proxy predicts neither size nor sign across families (R16's U-FNO, R26's Swin). Spend a slot: a *board* read of a family transfers to the other track at 0.85× (R28) |
| **anything decided at load time** (`AMP`, `GRAPH`, `FFNO_PATH`) | the constant is baked but the runtime path can still fall back. Ship a fingerprint round whose error subscores must come back bit-identical — R29 did exactly that for the tiled `_fno.py`. A fingerprint is a MAGNITUDE test, not a direction test (R27) |
| **a feature no local window exercises** (the gated mean-blend) | `validate_strict.py` passes such a package as if the feature were absent. It needs a dedicated verifier: `verify/blend_path.py`, `verify/aug_mask_check.py`, `verify/simmap_modes.py` |

The design consequence is the standing rule: **one controlled lever per slot**. A slot built so
that only one subscore can move buys a number no amount of local work can (R14's gate, R23's
unfused package, R29's tiling fingerprint). Compare packages by board-subscore delta, never by
`final`, and never resubmit for a timing draw.
