# How we evaluate locally, and how the board corrects it

The competition publishes one number per submission and one submission per track per day. Every
other number in this repository is a local estimate of that number, and each one is wrong in a
measured, named way. This file says where the pipeline lives, what each instrument reads, and
which constant corrects it. It is the map; the evidence is in `ROUNDS.md`, the verdicts in
`ROADMAP.md` Part 2, the rules in `CLAUDE.md`, and the procedures in `.claude/skills/`
(`evaluate-change`, `build-package`, `record-board-result`).

Every constant quoted below was read out of the code, not out of prose: on 2026-09-19 for §3–§5, and
again on 2026-09-27, when Main Development closed, for the `final` rule, the operating point and the
track record. Where a document disagreed with the code, the code is what is printed here.

---

## 1. The pipeline, end to end

```
 ~/RealPDE/train_real  (82 traj, u,v, 868 frames)      ~/RealPDE/train_sim  (100 traj, u,v,p)
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
        aoa_holdout_eval --holdout-aoa                              (out-of-distribution, single checkpoints)
        experiments/decision_20260928/ood/  (held-out AoA / Re regimes, ensembles, real package path, §3b)
        screen_sets / screen_eval -> leaderboard/screen.jsonl       (screen arms)
        |
        v
   BAND   submit_fno/fit_band.py  (per-element SPS half-widths, error field from the TWINS on val_seen AoA 15-20,
        |                          which they trained on: in sample, x1.39)
        |  sigblend/make_k0_band.py (the sigma power law, fitted on the twins' val_unseen errors)
        v
   PACKAGE  submit_fno/build_final.sh -> build_round13.sh -> build_zip.py + Track_2/build_t2.py
        |     bakes every RPDE_SHIP_* constant into submission.py (both tracks) and _fno.py
        v
   GATES  submit_fno/validate_strict.py (T1, GPU, TIMES the package)
          Track_2/validate_t2.py --adaptive | Track_2/stream_eval.py --causal-check
          verify/*.py (scorer diff, fusion, packing, AMP, blend, package identity, baked constants)
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
| `rel_l2` | `100/(1 + 0.5·e)` | relative L2 per window, `u, v` only, over the whole field (zero targets included) |
| `tke` | `100/(1 + 0.5·e)` | rel-L2 of `0.5·(var_t u + var_t v)` over the 20 output frames — phase-free |
| `mvpe` | `100/(1 + 0.5·e)` | rel-L2 of the time-mean at probes x ∈ {13,21,29,37}, y ∈ {8,10,…,24}, **per x-station**, averaged |
| `time` | `100/(1 + sqrt(t/0.72896))` | mean board **seconds** per step; 0 or non-finite ⇒ 0 |
| `sps` | `100·w` | per scored element (target inside `[lower, upper]`, non-zero): `(1−pm)·exp(−(upper−lower)/σ)`, `pm = e/(0.5+e)` per branch, branches DM 0.5 / TKE 0.3 / MVPE 0.2 |

`sps9.SIGMA = 0.0563870259` (it is `sc.SIGMA_GLOBAL`, read from the kit, not typed here).
Malformed bounds — wrong shape, non-finite, `lower > upper` — zero **every** subscore.

**Zero targets are scored, except by the interval** (corrected 2026-09-28; earlier text here and in `CLAUDE.md` said
they "are not scored"). About 10 % of `u, v` targets are exactly 0: outside the field of view or in the airfoil's
shadow. `rel_l2`, `tke`, `mvpe` and the accuracy factor `a_i` of `sps` are computed without a mask (over the whole field;
`mvpe` over its probes), zero-target elements included (`scoring.py`: `rel_l2_per_sample`, `tke_rel_l2_per_sample`, `mvpe_rel_l2_per_sample`, and `a` in
`aggregate_sps`). Only the interval average and the coverage of `sps` leave them out (`scored = Y != 0` in
`aggregate_sps`). The zero set is not static: 1.2 – 6 % of pixels are zero in every frame and 3.2 – 16 % in some frames
only. This is why `STATS_MASK` stays off (`ROADMAP.md` §1.46).

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
Σw = 1 constraint. Two fits matter:

```
to 2026-09-20, 42 rounds   rel_l2 0.4779 · tke 0.1009 · mvpe 0.0920 · time 0.0966 · sps 0.2498   max |residual| 0.0032
at the close,  48 rounds   rel_l2 0.4819 · tke 0.0981 · mvpe 0.0911 · time 0.0961 · sps 0.2491   max |residual| 0.0058
```

Every prediction registered during development used the first. **The rule is not exactly linear in the
five subscores.** The six closing rounds lie outside the range the first fit was made on (tke 79.5–80.7
against ≤ 78.95, sps 43.0–44.4 against ≤ 42.26); the exact 42-round fit reads all six high, by +0.005 to
+0.013. Under the refit the three Track-1 rows sit within 0.0007 and the three adaptive Track-2 rows
miss by 0.0026–0.0049: they are the first rows in which containment moves without the errors moving,
so they separate the weight of `sps` from the others for the first time. An intercept does not remove
the residual. Deltas between neighbouring packages are good to about 0.002.

`final_v9` self-checks at import: it asserts that `BOARD` and `board_rounds` hold the same rounds
as a multiset and that every residual is inside its gate (0.0060 since the close), so a mis-pasted
row fails immediately. `board_rounds.W_TIME_DIRECT = 0.095906` is `w_time` read independently, from
pairs of rounds whose four error subscores are identical. **`final_v9.py` is the authority — print
`W0` rather than quoting any document.**

**Exchange rate**, containment held fixed, per 10 % of error removed: `rel_l2` **+0.346**,
`tke` **+0.290**, `mvpe` **+0.100** of `final`. `Track_1/closed/exchange_v9.py` is the script that
produces it (held-out 6-member ensemble on the AoA 15–20 proxy, errors synthetically scaled,
board tail applied, band refit at every scale).

---

## 3. Held-out splits — which one answers which question

| split | defined in | size | answers |
|---|---|---|---|
| `VAL_TRAJ` | `Track_1/unet/model.py` | 8 trajectories: `3750_0 6300_10 8850_20 11400_20 13950_10 16500_5 21600_15 26700_0` | the default honest split; `val_unseen.npz` is its cache, `val_seen.npz` the matched in-training control |
| AoA 15–20 subset of `val_unseen` | `--aoa 15 20` on `compare_v9` / `ens_v9` | 126 fitting windows / a 15–20 view for reporting | **the board-like proxy.** Error is governed by AoA and almost nothing else (out-of-fold rel-L2 0.052 / 0.053 / 0.086 / 0.106 / 0.121 at AoA 0…20), and this slice reads 0.1082 against the board's 0.1179 where the full split reads 0.0807 |
| full split | no `--aoa` | 336 windows | the level, not the board's regime |
| Re extrapolation | `Track_1/re_split.py` (`excluded()` derives **26** trajectories) | contiguous Re bands | **retired as a decision instrument**: R14's gate fired on zero hidden windows, so the public hidden set is interior in Re. *2026-09-28: back in use for the PRIVATE set, whose Re is novel: its sets `relo` (9), `rehi` (7), `rein` (10) are the RE regime of §3b, read on ensembles* |
| interior Re k-fold | `Track_1/kfold.py --split re --bands 1 2 3 --paired` | all 82 trajectories, out-of-fold | **the confirmation step.** Overstates `rel_l2` by ~1 %; the signs of `tke` and `sps` are right |
| **AoA holdout** | `Track_1/aoa_holdout_eval.py --holdout-aoa {0,5,10,15,20}` | one angle removed from training *and* from the band's fit split | a single-checkpoint OOD instrument that speaks to the PRIVATE test set (novel AoA and Re); *2026-09-28: no longer the only one, see §3b for ensembles and the Re regimes*. `--fit-set val_traj` (8 traj, the 09-14 screen protocol) or `re_excluded` (26) chooses the in-distribution list the arm also excluded; the band is fitted there |

`aoa_holdout_eval.py --inflate-sweep A B C` refits the band at each stretch and reads `sps` on
**both** splits, so the width that is right for a novel angle arrives with its in-distribution
price. Run on 09-19 it closed the band width for a novel angle: 1.39 is already the held-out
optimum.

**The rule that governs all of them: all-data checkpoints supply weights, never evidence.**
Rounds 7, 8 and 10 were each misled by a number computed from a model that had trained on the
split it was scored on. Evidence comes from the held-out twins, and so does the band's error
field — a band fitted on memorized errors scored `sps` 36.94 in round 6.

*2026-09-28: "held-out" names the twins (held out of `VAL_TRAJ`), not the windows the band is fitted on. The static
band's error field is the twins' error on the three AoA 15 – 20 trajectories of `val_seen` (10125_20, 12675_20,
22875_15; 126 windows), which the twins trained on, times 1.39: `fit_band.py`'s default `--fit-on seen`, which
`build_round13.sh` uses. The SIGBLEND κ and q are fitted out of sample, on the eight `val_unseen` trajectories. Fitted
out of sample instead, the static band moves the composite by 0.002 – 0.012 (`ROADMAP.md` §1.47).*

**A held-out list is derived, never typed** (`re_split.excluded()`, `kfold.py --folds`): the
(Re, AoA) grid is incomplete, so a hand-written `--exclude` silently trains on a held-out
trajectory.

### 3b. The out-of-distribution stream instrument (2026-09-28)

`experiments/decision_20260928/ood/`; the commands, options and output layout are in its `README.md`. It turns an
ensemble trained without a whole regime into a package with its own band and streams the held-out regime through the
real package code. It is the instrument the Decision-phase campaign decided by (`ROADMAP.md` §1.42 – §1.48).

**One run** (one Ascend job, `bash ood_submit.sh NAME REGIME LAYOUT`):

1. **Members** from the layout; the run refuses a member whose training set holds an evaluation or fit-split
   trajectory, an all-data model, and sim co-training without `--sim-exclude-held`.
2. **Band**, refitted to the ensemble on `model.VAL_TRAJ` minus the regime: the static fit (`fit_band.py --fit-on
   unseen --fit-aoa <the AoA 15 – 20 trajectories left in the split>`), then SIGBLEND (`dump_twins.py`,
   `make_k0_band.py`), every script unchanged.
3. **Build** with `build_final.sh` in a source tree of its own (a `git archive` of the pinned commit); it must end
   `BUILD_FINAL OK`, and the run checks the `sed`-edited files before and after and the Track-2 gate's load line.
4. **Variants**: copies of the extracted archive with constants rewritten by the build's own patterns: `frozen` (every
   stream knob off, Track 1's case), `base` (asymmetric R1 only), `control` (the record + `PREV_SPACE "target"`),
   `prev_input` (the record as built), `statsmask`, `famw`, one-constant `c_*` variants, and synthetic re-blanks.
5. **Stream** each variant through `Track_2/stream_eval.run` with the revealed target normalised with the TARGET
   statistics; `--causal-check` on `control`.
6. **Report**: rel, tke, mvpe, `sps`, C and the composite (`W0` without `time`), pooled and per trajectory, over full
   streams and the first 3 and 14 windows, deltas to `control` and `base` with the number of trajectories positive.
   Subset scores are exact: the pooled result is checked against the kit's `aggregate_sps` (largest difference 1e-8).

**Regimes**, derived in code (`ood_lib.py regime`), never typed. Every regime model also excludes `VAL_TRAJ`, and sim
at a held-out condition:

| regime | held out of training | evaluated on | stands for |
|---|---|---|---|
| `A20` | every real and sim trajectory at AoA 20 | 16 real at AoA 20 | a novel angle, extrapolated |
| `A10` | every real and sim trajectory at AoA 10 | 18 real at AoA 10 | a novel angle, interpolated |
| `RE` | `re_split.excluded()` (26 real, six Re) and sim at those Re | `relo` 9, `rehi` 7, `rein` 10 | below, above and inside the training Re |
| `ID` | `VAL_TRAJ` (the twins) | the 8 trajectories and their AoA 15 – 20 slice | the floor |

"OOD mean" is the mean of A20, A10, relo and rehi.

**Layouts** (`ood_lib.py`): `mirror` (four F-FNO + sim s1 – 4, ViT-P s1, ViT-C + sim s3: the shape of the packages of
record), `six`, `vitc2`, `ffno4`, `ffno6`, `six_vitc2`, the arms' layouts (`t3`, `t4`, `t4v`, `t4b`), the twins' layouts
(`twins`, `twins_t4`, ...), or any list through `OOD_MEMBERS`. FAMW needs exactly two fused groups; a third family
switches it off, and the json records which corrections were live.

**Cost.** The regime ensembles are the expensive part: 24 trainings at the recipe of record, 60.3 A100 GPU-h (an F-FNO
+ sim or ViT-C member 2.3 – 2.8 h, a ViT-P 1.2 – 1.5 h). A run is 3 – 7 minutes for the base variants; a `mirror` run
with every variant, re-blanks and constant sweeps about 30 minutes at A20 and A10 and 45 at RE. The campaign's 99
instrument jobs used 10.2 GPU-h.

**Validation.** In distribution it reproduces `experiments/audit_20260927/refactor_pkg.md` to the printed digits
(stream stack against asymmetric R1 +0.0757, 8/8; `PREV_SPACE` target +0.0064, 8/8; `STATS_MASK` −0.0012). On the
sprint's stand-ins it reads the stream stack at +0.453 (AoA 20) and +0.567 (AoA 10) against the sprint's +0.446 and
+0.553. Knob-off variants are bit-identical to `frozen` on every trajectory's first window (except FAMW, whose mix
equals the mean only algebraically); the causal check reads 0 in every run.

**Limits.**
* One ensemble per arm per regime. The reseeding spread (§4) was measured at RE and in distribution, not at A20 or A10.
* A20's static band rests on one fit trajectory (21600_15); A10 and RE on three.
* The regime band is fitted out of sample for the regime ensemble, the shipped band in sample (§3): the regime band is
  a little wider than an organizer's rebuild would be (measured 0.002 – 0.012).
* The RE regime's in-distribution split (`VAL_TRAJ` minus RE) is in sample for its band; the twins' read is the clean
  in-distribution read.
* Re-blank masks are synthetic and overstate natural blanking (`ROADMAP.md` §1.46).
* Local ms per step comes from shared nodes and is not latency evidence.

---

## 4. Instrument catalogue

Every local number is a **sign, not a size**. The proxy over-reads error CUTS by about 2×; a
member REMOVAL is charged in full (T2R15); across model FAMILIES the proxy predicts neither size
nor sign (R16, R26) while a *board* read of a family transfers at 0.85× (R28).

**How large a local delta must be: the reseeding spreads** (2026-09-28, `experiments/decision_20260928/STATE.md`).
A difference inside these is not a difference.

| what varies | spread |
|---|---|
| one F-FNO + sim member, 6 seeds, sd of `wfinal` | 0.03 fit split · A20 0.08 · A10 0.08 · relo 0.07 · rehi 0.04 · rein 0.02 |
| four-member ensembles drawn from 6 control seeds (+ the same two ViTs), largest pairwise difference in composite, stream | relo 0.034 · rehi 0.011 · rein 0.010 · in distribution 0.012 |
| the same, frozen (Track 1's case) | relo 0.076 · rehi 0.030 · rein 0.013 · in distribution 0.010 |
| a from-scratch retrain of the five twins, same hardware and stack | 0.0003 |
| the same archive on a V100 instead of an A100, and every forced fallback | at most 0.0006 |

An organizer's retrain on other hardware is expected at the reseeding scale, not at the retrain's 0.0003. The
12 – 58× out-of-distribution seed ratio of `ROADMAP.md` §1.12 was the unseeded FNO of August.

**The operating point of the packages of record** is ALPHA 0.80 with JOINT_GAIN, the interval
centred on the alpha-0 field, and the SIGBLEND band on top of a static band fitted with
`--band-p 0.45 --inflate 1.39 --fill-unscored` on the twins' in-sample `val_seen` AoA 15 – 20 errors (§3). `Track_1/opoint.py` holds it, and
`Track_1/submit_fno/build_final.sh` is the authority for what is baked.
`Track_1/submit_fno/fit_band.py` keeps its own older defaults (`--alpha 0.75 --band-p 0.5`), because
the build passes what it needs explicitly.

**Instrument defaults moved to the packages of record on 2026-09-28** (`Track_1/opoint.py`; bands through
`Track_1/band_io.py`). Results recorded before 2026-09-27 used the OLD column:

| instrument | old default | new default |
|---|---|---|
| `eval_v9.py` `infer(..., alpha=)` · `--alphas` | 0.75 · `[0.75]` | `opoint.ALPHA` 0.80 · `[0.80]`; `--joint-gain` on |
| `compare_v9.py` `--alpha` · `--inflate` | 0.75 · 1.5 | 0.80 · 1.39; `--joint-gain` on |
| `ens_v9.py` `--alpha` · `--f` | 0.75 · 1.39 | 0.80 · 1.39 (`opoint.INFLATE`); `--joint-gain` on |
| `predict_board.py` `--alpha` · `--band` | 0.75 · `submit_fno/band.npz` (untracked build by-product) | 0.80 · `bands/k27_sigblend_lam0.5.npz` + sidecar; JOINT_GAIN for a y0-centred band; interval centred per the band; new anchor `r37` |
| `kfold.py` · `re_split.py` · `mean_blend_re.py` `--alpha` | 0.90 | 0.80; `--joint-gain` on |
| `aoa_holdout_eval.py` `--alpha` · centre · compose | 0.60 · the prediction · per-channel | 0.80 · `--centre-alpha 0` (y0) · JOINT_GAIN |
| `calib_band13.py` `--alpha` · board rows | 0.60 · R13 T1 + T2R1 | 0.80 · `--rounds "T1 R37"` (R13 selectable) |
| `quant8.py` · `lat_r15.py` · `lat_fast17.py` · `verify/quant_identity.py` `--alpha` | 0.60 | 0.80 |
| `lat_r15.py` · `verify/quant_identity.py` `--band` | untracked `Track_2/submit/band8.npz` · `submit_fno/band.npz` | `bands/k27_sigblend_lam0.5.npz` |
| `screen_eval.py` `ALPHA` (used by `screen_sets.py`) | 0.60 | 0.80 (`screen.jsonl` rows are at 0.60) |
| `alpha_board.py` `--alphas` | 0.30 0.45 0.60 0.75 0.90 | adds 0.80; `--anchor 0.60` and per-channel compose kept (the R13 row) |

### Accuracy

| instrument | measures · invocation at the operating point | writes | known bias |
|---|---|---|---|
| `Track_1/compare_v9.py` | one checkpoint at a time, each with its **own** band. `--ckpt unet/x.pth --aoa 15 20` | stdout | a single model is not the shipped ensemble |
| `Track_1/ens_v9.py` | member **SETS** — the screen at the size that ships. `--sets "a.pth,b.pth" "a.pth,b.pth,c.pth" --aoa 15 20` | stdout | `--time-score 90.5066` is round 8's |
| `Track_1/screen_sets.py` | a screen arm's member sets by token (`FNO3`, `VIT2`, …). `--arm <arm> --set "S1=BASE" "S2=BASE,ckpt"` | delegates to `screen_eval.run_main("ens_v9", …)` | inherits `ens_v9`'s |
| `Track_1/screen_eval.py` | a screen arm end to end: accuracy, latency, projected `final`. `--arm <arm> --ckpt … --wave N --lat LAT.json` | upserts `leaderboard/screen.jsonl` under a lock; JSON in `Track_1/plan/` | projects with **gains halved, losses in full, board ms floored** (the 09-14 calibration) |
| `Track_1/kfold.py` | out-of-fold over all 82 trajectories, with error bars. `--score --split re --bands 1 2 3 --paired` | JSON in `--out-dir` (scratch) | overstates `rel_l2` ~1 % |
| `Track_1/aoa_holdout_eval.py` | a novel ANGLE, with the shipped band machinery. `--ckpt … --holdout-aoa 20 --fit-set val_traj` | `--json` | one angle is one draw; the OOD seed σ was 0.71 `wfinal` on the unseeded FNO, 0.08 on seeded F-FNO + sim (09-28); replicate. For ensembles and the stream use §3b |
| `Track_1/proxy_calib.py` | board/proxy ratios per round, and the proxy's resolution (GPU, once per round) | stdout | it is the thing that measures the bias |
| `Track_1/predict_board.py` | the five board subscores **before** upload. `--ckpt <all-data> --err-ckpt <twins> --invocations N --local-ms M --anchor <round> --anchor-err-ckpt …` | stdout → the staging README | predicts a LEVEL to ±0.15; cannot resolve a change below ~2 % of error |
| `Track_1/make_val.py` · `Track_1/closed/exchange_v9.py` · `Track_1/sub_value.py` · `Track_1/alpha_board.py` · `Track_1/calib_band13.py` · `Track_1/quant8.py --score BAND.npz` | the val caches every instrument reads · the exchange rate · a slot's information value · ALPHA at the board's error scale · the board's error tail · the **scored** cost of a packing scheme | stdout (`make_val` writes `Track_1/val_*.npz`, untracked, ~1 GB) | each is a single-purpose calibration script; see §5 for the constants three of them produce |

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
| `Track_2/validate_t2.py [--adaptive] --zip submission_t2.zip` | the Track-2 contract: isolation, no aliasing, whole-run bound consistency, and for a frozen package statelessness. **A package that adapts is gated with `--adaptive`** |
| `Track_2/stream_eval.py --submission <extracted dir> --causal-check` | causality of an adaptive package: every window after a cut is replaced by noise and every step before it must stay bit-identical |
| `Track_1/submit_fno/build_final.sh OUTDIR` with `RPDE_REF_T1` / `RPDE_REF_T2` | the two packages of record rebuild, pass all three gates, and match the uploads file by file. Since 09-28 it also refuses a second build in the same tree (a lock file) and reads every baked constant and the band's md5 back out of both archives (`Track_1/verify/baked_constants.py`, 41 checks); a mismatch fails the build |
| `Track_2/stream_eval.py --submission <extracted twin package> --split unseen [--aoa 15 20]` | honest Track-2 numbers: real trajectories through the streaming protocol, official scoring. Only a package built from the held-out twins is honest here |
| `Track_2/verify_space.py --submission <T2 dir> --t1 <T1 dir>` | the normalized-space path of a FROZEN Track-2 package reproduces Track 1's `predict()` (4e-6 physical) |
| `Track_1/verify/pkg_identity.py --track t1 --a <old> --b <new>` | two BUILT archives return the same arrays. Hand it the archive **actually uploaded** (`../sub_candidates/final/`) |
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
| **subscore weights** — five subscores → `final` | `Track_1/final_v9.py:BOARD` → `W0` | all 48 scored rounds, pooled. Paste the new row into `board_rounds.ROUNDS`/`ROUNDS_T2` **and** `final_v9.BOARD`; the refit and the self-check run at import |
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
| R30 (T1) | 09-20 | 81.4179 | 81.418569 | +0.001 | right on `final`, wrong on the composition: the gain came through `mvpe` and containment, not `rel` and `tke` |
| T2R17 | 09-20 | 81.3422 | 81.333776 | −0.008 | `time`: a +0.704 ms draw |
| R34 (T1) | 09-24 | ≈ 81.75 | 81.5766 | **−0.17** | the QD error-head band: +0.11 locally, containment 0.5996 → 0.5966 on the board |
| T2R21 | 09-24 | ≈ 81.71 | 81.6177 | −0.09 | the same band on Track 2 |
| R37 (T1) | 09-27 | 81.67 | 81.671149 | +0.001 | every subscore inside its bracket |
| T2R24 | 09-27 | 82.000 | 82.055553 | +0.056 | two misses that cancel: `tke` +0.62 above its bracket, `time` −0.68 below |

The pattern: with an anchor of the **same shape**, the four error subscores land within a few
thousandths and `time` carries the miss. The misses above 0.1 are all structural — a new family
(R26), a host-side latency claim (R27), a member removal (T2R15), a band fitted tighter to the error
(R34). The uploads of 09-21 … 09-23 and 09-25 … 09-26 registered brackets, not point predictions, on
the agent team's server; they are in `ROUNDS.md`. **T2R24 is the first prediction made across
regimes**: its estimate is the mean of an in-distribution read and a held-out-AoA read, and the board
landed between the two, nearer the held-out one.

---

## 6. Procedures

The step-by-step procedures are skills, loaded when they are needed:

| when | skill |
|---|---|
| a change has to be judged before it can ship | `.claude/skills/evaluate-change/SKILL.md` |
| a package has to be built, gated and staged | `.claude/skills/build-package/SKILL.md` |
| the user reports a board result | `.claude/skills/record-board-result/SKILL.md` |
| the method has to be re-created from the release | `.claude/skills/reproduce-from-release/SKILL.md` |
| anything needs a GPU | `.claude/skills/osc-jobs/SKILL.md` |

Three rules frame all of them. The user uploads; the Decision phase has no upload at all (2026-10-05: its Codabench
phase is the organizers' own, and what they take is a training package, `final_package/README.md`). The prediction is
written before the upload. HOLD is a valid decision: never upload a package expected to score worse.

---

## 7. What no local instrument can read

| unreadable | why, and how we handle it |
|---|---|
| **containment** (the residual half of `sps`) | the band is refit per package and every refit scores well on its own twins, so nothing local reads the *board's* coverage. Never bank it; recover it afterwards by subtracting the forced `a`-factor part from the board's `sps`. It is where T2R14 lost 0.310 and T2R15 0.132, and where the second family earns its place |
| **the board's latency LEVEL** | the eval machine drifts by date and the two local timers disagree by 5× (R28: +0.797 kernel-level, +0.164 package-level, board +0.404). Anchor on a recent round of the **same shape** and quote both reads as the bracket. Never time while an `rp_*`/`pz_*` job is alive; only interleaved arms count |
| **a cross-FAMILY delta** | the proxy predicts neither size nor sign across families (R16's U-FNO, R26's Swin). Spend a slot: a *board* read of a family transfers to the other track at 0.85× (R28) |
| **anything decided at load time** (`AMP`, `GRAPH`, `FFNO_PATH`) | the constant is baked but the runtime path can still fall back. Ship a fingerprint round whose error subscores must come back bit-identical — R29 did exactly that for the tiled `_fno.py`. A fingerprint is a MAGNITUDE test, not a direction test (R27) |
| **how the private set behaves** | it is out of distribution in AoA and Re by construction. The development board turned out to sit between the in-distribution and the held-out reads (T2R24). Since 09-28 the instrument of §3b reads ensembles with a whole AoA or Re band held out (A20, A10, relo, rehi, rein) on the real package path; it cannot say which of those regimes the private set is. Read every change in these regimes (`evaluate-change` skill) |
| **a feature no local window exercises** (the gated mean-blend) | `validate_strict.py` passes such a package as if the feature were absent. It needs a dedicated verifier: `verify/blend_path.py`, `verify/aug_mask_check.py`, `verify/simmap_modes.py` |

The design consequence is the standing rule: **one controlled lever per slot**. A slot built so
that only one subscore can move buys a number no amount of local work can (R14's gate, R23's
unfused package, R29's tiling fingerprint). Compare packages by board-subscore delta, never by
`final`, and never resubmit for a timing draw.
