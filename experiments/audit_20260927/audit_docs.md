# Documentation audit — 2026-09-27 (read-only)

Repo `~/workspace/realpde/main` @ `qwu/agent` 55f0e88 plus an uncommitted working tree (4 modified, 10 untracked).
Line numbers are for the files as they stand now. "Facts of record" are the caller's (final boards, board
history 09-21..09-27, members unchanged since r34, ALPHA 0.80, centre0, SIGBLEND, JOINT_GAIN, latency knobs,
T2 causal stream stack, `build_final.sh`).

**Findings that are more than wording.** They go first because no doc rewrite fixes them:

* **F1. The final code is not committed.** `git diff --stat` shows +336/−16 in `_fno.py`, `build_round13.sh` and both
  `submission.py`. `build_final.sh`, `Track_1/sigblend/`, `Track_1/bands/k27_sigblend_lam0.5.npz` (md5 e8309d9f, which is
  the band inside MRGP2), `Track_1/bands/r38_a080_c0_fill.npz` and six `verify/` scripts are all untracked. The only
  other copies are `../sub_candidates/20260927/notes/*.patch` and scratch (`final0927/wt_mrgp2`). As things stand,
  `build_final.sh` cannot be run from a clean checkout.
* **F2. The board record stops at 09-20.** `board_rounds.py` has ROUNDS 25 + ROUNDS_T2 17 (last R30/T2R17), and
  `final_v9.BOARD` and `leaderboard/board.jsonl` (40 lines) end at the same place. `predict_board.ANCHORS` has no round
  after t2r16/R30. The 13 scored rounds from 09-21 to 09-27 (7 on T1, 6 on T2) appear in no ledger. Every doc that says
  "the live record is `board_rounds.py`" is currently pointing at a record that is a week out of date.
* **F3. `../sub_candidates/submitted/` holds 34 archives and none are from 09-21 onward**, so it is missing at least
  13 uploaded archives. Its README (l.17) calls it "Complete, because winners must open-source".
* **F4. `../handoffs/` ends at `HANDOFF_20260920.md`.** CLAUDE.md l.16-17 says the newest handoff overrides the repo. It
  is now the oldest source of truth. The real final-day state is in `../sub_candidates/20260927/README.md`.
* **F5. `REPRODUCE.md` l.10-11 still reads `FINAL_RECIPE_T1/T2 = <TO BE DECIDED…>`.** Its candidates (A / QD /
  QD_STATIC) do not include the two packages of record. It never mentions SIGBLEND/EBAND/SIGPOW, JOINT_GAIN, D1c,
  VARCAL, R1_TSM/R1_BLK or `build_final.sh`; SIGBLEND is covered only in the untracked `Track_1/sigblend/README.md`.
  `reproduce.sh` has no stage for the SIGBLEND band.
* **F6. Band provenance gaps.** `Track_1/bands/MANIFEST.md` has no entry for `r38_a080_c0_fill.npz` or
  `k27_sigblend_lam0.5.npz`, and it records `r37_inflate155` (09-22) as missing. It also has no entry for the 09-24 QD
  band, the 09-25/26 bands, or whatever band T2's 09-26 row shipped.

---

## 1. Stale or wrong statements

### CLAUDE.md

| file:line | statement | why stale | replacement / action |
|---|---|---|---|
| CLAUDE.md:6-7 | "working branch `qwu/dev`" | qwu/dev was merged to main (PR #2, 09-21); all 71 later commits are on `qwu/agent` | "Branches: `main` (release), `qwu/agent` (working). Merges to main go through a PR." |
| :9-12 | "Keep it under ~30 k characters … update *Current state*" | this rule is exactly what made the file a 30 k status log | delete; replace with "Stable facts only. Status lives in `STATUS.md`/handoff; rounds in `ROUNDS.md`." |
| :16-17 | "newest handoff … Where it disagrees with the repo docs, it wins" | the newest handoff (09-20) is older than REPRODUCE.md (09-26) and the 09-27 sprint README | "Read the newest dated handoff **if it is newer than the last commit touching the doc in question**." Write a 09-27 handoff (F4). |
| :26-63 | whole *Current state — 2026-09-20* (R30 81.418569, T2R17 81.333776, r34 of record, #23/#15, "HOLD 09-21", "no lever ≥ +0.05") | superseded: finals 81.671149 / 82.055553; r35 went up 09-21 (so the HOLD was not followed); T1 +0.253 and T2 +0.722 since | delete from CLAUDE.md. Put a dated status block in a handoff or `STATUS.md` (see §4a) |
| :57-60 | "Packages of record … band `r34_p2ffnoL8sim_fill.npz` · ALPHA 0.60" | band is now `k27_sigblend_lam0.5.npz` (+EBAND/SIGPOW), ALPHA 0.80, CENTRE_ALPHA 0 | "Packages of record: built by `Track_1/submit_fno/build_final.sh` (every constant is in that script)." |
| :67 | "One submission per track per day." | true for Main Development only; the Decision phase allows **one submission in total** per person (`Track_1/plan/lit_20260917.md:131`, Codabench API, 09-25) | "Main Development: 1/track/UTC day. Decision phase: 1 in total per track — confirm with the owner before any upload." |
| :81-83 | "Only baked constants ship … `AMP`, `GRAPH` need a fingerprint" | still right; the knob list is now much longer (§4b) | move to the package-code rule |
| :89-92 | Wrap-up checklist | procedure; it also names `qwu/dev` indirectly through EVALUATION §6 | move to the `record-board-result` skill |
| :112-115 | "every adaptation built on them is measured and closed … **Ship frozen.**" | wrong: T2 ships R1 (asymmetric per-pixel online band calibration, TSM, BLK), D1c (per-pixel bias removal, EB shrink, lead-time profile) and VARCAL. T2 beats T1 by +0.38 on the same members | "Track 2 may adapt **causally**: state from revealed targets only, reset in `reset_ttt_state`. The shipped stream stack is R1 + D1c + VARCAL + SIGPOW band (`Track_2/README.md`). Gradient TTT, readout refit and trajectory history stay closed." |
| :170-172 | "free linear fit … over **40** rounds … rel_l2 0.4778 …" | code: 42 rounds, W0 = 0.4779/0.1009/0.0920/0.0966/0.2498; will be 55 after backfill | delete from CLAUDE; "`final_v9.py` holds the fitted weights — print `W0`, never quote a doc" |
| :186-196 | board-latency constants | these are dev-board (A100-class) constants. The private set is timed on 8× A800, and the board GPU seen 09-25 was an A800-SXM4-80GB (REPRODUCE §0) | move to EVALUATION §5 (already there); delete here |
| :207-208 | "**FNO ×4** `fno_all_w96L8cmb3sim_s1..s4.pth` adds `--arch fno --width 96…`" as part of the shipped recipe | the w96 FNOs have not shipped since R28 | move to the training rule, labelled "historical FNO family (R19–R27)" |
| :222 | "compose … at **ALPHA 0.60** (0.55–0.70 is flat)" | ALPHA 0.80 (report_20260923_band_surface: +0.0425, direction 3/3); the tree default is still 0.60, the build bakes 0.80 | "ALPHA 0.80 baked by the build (tree default 0.60 is NOT the package)" |
| :223-225 | "Band: `half = base · clip(σ/σ_ref,0.3,3)^p`, p 0.45 …" | now centre = the α=0 field (`CENTRE_ALPHA 0`), `--fill-unscored`, SIGBLEND: `half = base'·clip(σ/σ_ref)^p'·σ^(λq_c)`, λ 0.5, p' = 0.225, applied via T1 `EBAND` (GPU-band path) / T2 `SIGPOW`; `JOINT_GAIN` | replace with a pointer to `Track_1/sigblend/README.md` and the package-code rule |
| :229-235 | latency settings list | missing WARM_TAILS, PIN_OUT 128, BATCH_CAP 56, DIRECT_PRED, NUMPY_OUT, ARENA_STEPS 1024, PAR_STREAMS, SMALL_GRAPH(_LAZY) = False, FFNO_PATH dense | move to the package-code rule, with `build_final.sh` as the authority |
| :250-252 | "Pass `--alpha 0.60 --band-p 0.45`" | operating point is now `--alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored --inflate 1.39` (sigblend README step 1) | move to the instruments rule with the new values |
| :274-276 | Closed: "Band width and shape … asymmetric, online calibration … ALPHA (a plateau)" | ALPHA moved to 0.80; the band shape changed (SIGBLEND); asymmetric online calibration ships on T2 (R1_ASYM) | "Closed for the **static** band family in distribution (width, p, smooth, inflate). ALPHA re-opened and moved to 0.80 (09-23). T2 online calibration: shipped as R1." |
| :292-293 | "Track 2 adaptation: residual EMA; online band calibration; … closed" | D1c is a causal per-pixel residual correction and R1 is online band calibration; both ship | keep only "gradient TTT, closed-form readout refit, trajectory history" as closed |
| :302-322 | build command `…build_round13.sh 0.60` | the packages of record come from `build_final.sh` (build_round13 at 0.80 plus ~25 RPDE_SHIP_* knobs); T2 also needs `validate_t2.py --adaptive` and `stream_eval.py --causal-check` | move to the `build-package` skill |
| :316 | "a build job restores the tree with `git checkout`" | `build_final.sh` snapshots and restores the three files itself; bare `build_round13.sh` still leaves the constants in place (it now also seds `_fno.py`) | state both in the skill |
| :363-374 | Environment: Ascend only; eval stack pinned | training moved to Pitzer V100 on 09-18 (memory); everything since 09-20 was built on torch 2.7.1 / numpy 2.2.6 (`~/envs/mural_osc`), which needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (REPRODUCE §0) | two-stack statement; move details to the `osc-jobs` skill |
| :378-380 | "1 submission per track per UTC day … Main Development **ends** 2026-09-27" | ended; Decision phase 09-28 → 10-25 with one submission in total; top-10 status unknown (cut at 21:39Z: T1 81.885, T2 81.988) | "Main Development closed 09-27 (T1 81.671149, T2 82.055553). Decision phase to 10-25: organizers retrain the top 10 from scratch; 1 submission total." |
| :386-396 | repo map | missing `REPRODUCE.md`, `reproduce.sh`, `Track_1/qd_k/`, `Track_1/sigblend/`, `Track_2/r1_ref/`, `submit_fno/build_final.sh` | update |

### README.md

| file:line | statement | why stale | replacement |
|---|---|---|---|
| README.md:9-12 | best board 81.401 / 81.324 "as of 2026-09-19"; :81-87 `--alpha 0.60`, `build_round13.sh 0.60` | finals 81.671149 / 82.055553; 0.80; `build_final.sh` | update |
| :19-20 | "Both tracks ship the same frozen model. Test-time adaptation … does not pay" | T2 ships a training-free causal stream stack that adds ≈ +0.38 over T1 | "Both tracks ship the same six trained members. Track 2 adds a training-free, strictly causal stream stack (online band calibration, per-pixel bias removal, amplitude calibration). Gradient TTT was measured and does not pay." |
| :22-24 | "Four width-96, 8-layer Fourier neural operators with heavy channel dropout" | shipped are four factorized FNOs (w40, L8, modes 16×32), co-trained on sim | fix |
| :28-31 | inference paragraph | missing ALPHA 0.80, α=0 band centre, SIGBLEND σ-power band, joint gain | fix |
| :46-49 | repo table | no `REPRODUCE.md` / `reproduce.sh`, which organizers will read first | add a row at the top |
| :71-77 | "train a held-out twin with the shipped FNO recipe" (w96 FNO command) | not the shipped recipe | replace with `bash reproduce.sh --recipe …` and a pointer to REPRODUCE §1.4-1.5 |

### EVALUATION.md

| file:line | statement | why stale | replacement |
|---|---|---|---|
| :38-42 | PACKAGE/GATES boxes (bakes ALPHA/AMP/…; gates) | missing CENTRE_ALPHA, JOINT_GAIN, EBAND/SIGPOW, R1*/D1C*/VC_N0, latency knobs, `build_final.sh`, and the T2 gates `validate_t2.py --adaptive` + `stream_eval.py --causal-check` | fix |
| :104-110 | "carries 40 rounds … 24 T1 / 16 T2" + weight vector | code: 25 + 17 = 42, W0 differs in the 4th decimal; 13 rounds still missing | "print `final_v9.W0`"; drop the literal vector |
| :118-121 | quoted note about CLAUDE.md and leaderboard README drift | historical; leaderboard README now says 42 | delete |
| :163-164, 170-177 | "The package ships `ALPHA = 0.60` … Always pass `--alpha 0.60 --band-p 0.45`" | 0.80, centre 0, fill-unscored | fix |
| :231 | ANCHORS "30 rounds (`r10`…`r29`, … `t2r16L8`)" | R30/T2R17 and 13 later rounds missing | fix after backfill |
| :240-256 | Track record table ends T2R16 | missing R30, T2R17 and every 09-21..27 round (the final picks recorded predictions 81.67 [81.58,81.79] T1 and 81.99 [81.87,82.21] T2, which came in at 81.671 and 82.056) | append |
| :264-306 | §6 Procedures ("Commit first … `build_round13.sh` 0.60 …", "≤ 7/day on `qwu/dev`") | build via `build_final.sh`; branch `qwu/agent` | move to the two skills; leave a pointer |

### Track_2/README.md

| file:line | statement | why stale | replacement |
|---|---|---|---|
| :29-32 | "Both were measured before anything was built, and both are closed." | the realized error is used by R1 and D1c, and the revealed variance by VARCAL (and SIGEMA, a knob that is off) | rewrite: "Both were closed on the 09-01 package with global/symmetric estimators; per-pixel causal versions (R1, D1c, VARCAL) shipped 09-25..27." |
| :38 | Residual EMA closed | D1c ships (per-pixel mean_t(C−y) over revealed windows / (n+N0), EB shrink, lead-time N0) | row → "superseded by D1c (shipped)" |
| :39 | Online SPS band calibration closed ("best global rescale c = 1.00") | R1 ships: per-pixel asymmetric k_lo/k_hi from past windows, TSM 3, BLK 4 | row → "global rescale null; per-pixel asymmetric R1 ships" |
| :49-50 | "**Track 2 is Track 1's model, streamed.** Ship the forecaster frozen … not a new algorithmic lever" | contradicted by the board (T2 82.056 vs T1 81.671 on the same members) | replace with a section "The stream stack" (R1, D1c, VARCAL, SIGPOW band, JOINT_GAIN, ARENA_STEPS, NUMPY_OUT, PAR_STREAMS), causality rules and the two extra gates |
| :54-57 | "Built together with Track 1 by `build_round13.sh` … package of record … in CLAUDE.md *Current state*" | `build_final.sh` | fix |
| :58-63 | submit/submission.py description (frozen ensemble, AMP, graph) | lacks the stream state and `reset_ttt_state` semantics, arena, numpy out | extend |
| :120-121 | honest numbers "Round 13, submit_ho5 … α 0.90" | historical | replace with a current held-out read of the final stack, or label as round-13 |
| :131-143 | Layout | missing `r1_ref/` (float64 references for R1); `eval/` is an empty directory | add `r1_ref/`; delete `eval/` or say what it is for |

### Track_1/README.md

| file:line | statement | why stale | replacement |
|---|---|---|---|
| :35-36 | "rescaled toward σ at `ALPHA = 0.60`, and per-element SPS half-widths come from `band.npz`" | 0.80, centre0, SIGBLEND factor, joint gain | fix |
| :46-55 | "The shipped FNOs:" + w96 command | shipped are F-FNO; command belongs to the historical family | "The shipped members: REPRODUCE.md §1.2 (ViTs), §1.4 (F-FNO); twins §1.3/§1.5." |
| :64-73 | Layout | missing `qd_k/` (QD error head + k derivation, closed?), `sigblend/` (the shipped band), `REPRODUCE`-related scripts | add |
| :127-131 | "`bash submit_fno/build_round13.sh ALPHA` builds…" | `build_final.sh` | fix |
| :133-138 | Status "As of 2026-09-19 the Track-1 best is 81.401196 (R28…)" | 81.671149 (09-27) | delete the Status section (status does not belong in a README), or one line with the final |

### Track_1/submit_fno/README.md

| file:line | statement | why stale | replacement |
|---|---|---|---|
| :11-21 | "As of R24/R25 (xfam6)": `fno.pth…fno_4.pth` = 4 w96 FNOs, 29 MB each; band `r24_xfam6.npz`; 146 MB | final: 4 F-FNO w40 L8 + 2 ViT, int8; band k27 SIGBLEND | regenerate from the final zip listing |
| :28, :41-49 | "xfam6 is two invocations"; constants table ALPHA 0.60, no CENTRE_ALPHA/JOINT_GAIN/EBAND/FFNO_PATH/WARM_TAILS/PIN_OUT/BATCH_CAP/DIRECT_PRED/SMALL_GRAPH*/SMALL_N/DIAG | | table = the `RPDE_SHIP_*` list in `build_final.sh`, with the tree default next to the shipped value |
| :57 | `fit_band.py … --alpha 0.60 --band-p 0.45` | add `--alpha 0.80 --centre-alpha 0 --fill-unscored --inflate 1.39`, then the SIGBLEND step | fix |
| :67-68 | "The band is closed" | SIGBLEND changed it on 09-27 | qualify |
| :76-87 | build command and "five stages" (seds into submission.py only) | `build_final.sh`; build_round13 now also seds `_fno.py` | fix |

### Other files in scope

| file:line | statement | why stale | replacement |
|---|---|---|---|
| Track_2/closed/README.md:17-18 | `ttt_offline.py` / `ttt_band.py` verdicts as closed | per-pixel causal successors ship | add "superseded in part by D1c / R1 (shipped 09-25..27)" |
| leaderboard/README.md:115 | "42 rounds (2026-09-20)" | true for the code, stale for the board (F2) | update after backfill |
| Track_1/bands/MANIFEST.md | no entries for `r38_a080_c0_fill`, `k27_sigblend_lam0.5`; `r37_inflate155` "still missing" | F6 | add entries (md5, round ids, knobs); archive r37 from its zip |
| ../sub_candidates/README.md:22-63 | "NOW — 2026-09-21: BOTH TRACKS HOLD … nothing staged" | r35 was uploaded 09-21; six more days of uploads; finals in `20260927/README.md` | rewrite NOW as a pointer to `20260927/README.md` + finals; move the 09-21 block to HISTORY.md |
| ../sub_candidates/README.md:17, :61 | "`submitted/` … Complete" (34 files) | F3 | copy the 13+ uploaded archives in; update count |
| ../sub_candidates/README.md:63 | "full state is in HANDOFF_20260920.md" | | → new 09-27 handoff |
| REPRODUCE.md:10-11 | `FINAL_RECIPE_T1/T2 = <TO BE DECIDED…>` | decided: T1 `track1_k27fin_a080`, T2 `T2_MRGP2` | fill in, command `bash Track_1/submit_fno/build_final.sh` with `RPDE_REF_T1/T2` |
| REPRODUCE.md:31-40 | three candidates table (A/QD/QD_STATIC) | the finals are A-members + SIGBLEND + stream stack | add a row "FINAL", mark A/QD/QD_STATIC as history |
| REPRODUCE.md §11 (797-813) | R1 only | D1c, VARCAL, R1_TSM/BLK, SIGPOW, JOINT_GAIN, ARENA unspecified | add §13 "The 09-27 finals: SIGBLEND band + T2 stream stack", folding in `Track_1/sigblend/README.md` |
| REPRODUCE.md §6.6 | "have the script restore the tree itself" | done by `build_final.sh` | mark closed for the final build |

---

## 2. Contradictions between documents

| # | statement A | statement B | which is right |
|---|---|---|---|
| 1 | CLAUDE.md:49 and sub_candidates/README.md:22 "09-21: HOLD on both tracks" | board history: r35 scored 09-21 (T1 81.596074 / T2 81.499485) | the board |
| 2 | CLAUDE.md:114 "Ship frozen"; Track_2/README.md:49-50; README.md:19 | REPRODUCE.md:797-813 (R1 approved for T2); `build_final.sh` (R1, D1c, VC knobs); 20260927/README | the latter |
| 3 | ALPHA 0.60 / "a plateau": CLAUDE.md:222, :276; EVALUATION.md:163; Track_1/README:36; submit_fno/README:43 | REPRODUCE.md:35 "ALPHA 0.80"; `build_final.sh`; report_20260923 | 0.80 (tree default 0.60 is not what ships) |
| 4 | CLAUDE.md:274-275 band closed incl. "asymmetric, online calibration"; Track_2/README:39; ROADMAP:1133-1141; submit_fno/README:67 | REPRODUCE §11 R1 (`R1_ASYM`), D1c, `Track_1/sigblend/README.md` | the latter |
| 5 | CLAUDE.md:316 / EVALUATION.md:268 "a build job restores the tree with git checkout" | `build_final.sh` header: snapshot + restore; REPRODUCE §6.6 says no script restores | `build_final.sh` restores the tree itself; bare `build_round13.sh` leaves it modified |
| 6 | CLAUDE.md:170 and EVALUATION.md:104 "40 rounds" | leaderboard/README.md:115 and ROADMAP Part 4 "42 rounds"; code 25+17 = 42 | the code (42), and all of them are missing 13 rounds |
| 7 | CLAUDE.md:248, leaderboard/README:29 ratios 1.07 / 1.14 / 1.45 | `predict_board.RATIO` 1.078 / 1.150 / 1.511 (EVALUATION:235-238 notes this) | the code is what runs; quote the constant name, not a number |
| 8 | CLAUDE.md:67, :378 "1 submission per track per day, 100 per phase" | lit_20260917.md:131 "Final Decision … `max_submissions_per_person = 1` for the WHOLE phase" | both, by phase; CLAUDE must say so |
| 9 | README.md:22 "four width-96 8-layer FNOs"; Track_1/README:46 "The shipped FNOs" (w96 cmd); submit_fno/README:17 | Track_1/README:24 and CLAUDE.md:212 (F-FNO w40 L8 sim); REPRODUCE §1.4 | F-FNO |
| 10 | CLAUDE.md:16-17 "newest handoff wins" | handoff 09-20 vs REPRODUCE 09-26, sprint README 09-27 | the newer docs, so the rule needs a date condition |
| 11 | CLAUDE.md:372-374 "environment.yml pins the eval stack … torch 2.2.2, numpy 1.26.4" (only stack named) | REPRODUCE.md:104-121: everything since 09-20 built on torch 2.7.1 / numpy 2.2.6 (`mural_osc`, Pitzer) + `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | both true; CLAUDE omits the build stack |
| 12 | ROADMAP.md Part 3 item 3 / sub_candidates:17 "every shipped band in `bands/`", "`submitted/` complete" | MANIFEST "r37_inflate155 still missing"; F3; F6 | neither holds today |
| 13 | ROADMAP.md:1-48 one-line state (R17, 81.169843) | ROADMAP.md:1168 (09-20) and the finals | finals |
| 14 | REPRODUCE.md:10-11 "TO BE DECIDED" / :13-20 rule "best row at 09-27 23:59Z snapshot" | 20260927/README FINAL PICKS + board: k27fin (T1), MRGP2 (T2) | the rule resolves to those two; fill it in |

---

## 3. CLAUDE.md triage (29,980 chars, 400 lines)

Test applied to each line: would removing it make an agent make a mistake? Char counts are `wc -m` on the line
ranges and are approximate where a range is split.

| lines | section | verdict | where | chars |
|---|---|---|---|---|
| 1-7 | title, links, remote | KEEP (fix branch) | CLAUDE | 400 |
| 9-12 | maintenance note (30 k, Current state) | DELETE | replaced by one "what goes where" line | 370 |
| 14-24 | Read first | KEEP, rewritten (add REPRODUCE.md, status doc; date condition on handoff) | CLAUDE | 730 |
| 26-63 | Current state 09-20 | DELETE (stale status; round detail already in ROUNDS.md) | status → handoff/STATUS.md | 3,220 |
| 65-80 | Standing rules: upload, qualifier, predict, evidence, deltas, one lever | KEEP (rewritten, §4a) | CLAUDE | 1,500 |
| 81-83 | only baked constants ship; fingerprint | MOVE | rule `package-code` | 300 |
| 84-85 | local timing | MOVE | skill `osc-jobs` + rule `instruments` | 150 |
| 86-88 | no trees under $HOME | KEEP | CLAUDE | 200 |
| 89-92 | wrap-up checklist | MOVE | skill `record-board-result` | 400 |
| 94-104 | the problem, both protocols | KEEP | CLAUDE | 700 |
| 106-111 | windows tile; only clean input is own window; no Re/AoA | KEEP (core leak rule) | CLAUDE | 650 |
| 112-115 | T2 "ship frozen" | DELETE (wrong); replaced by the causal rule | CLAUDE rules | 280 |
| 117-130 | data paths, HF mirror, file shapes, windowing | KEEP, shortened | CLAUDE | 700 |
| 131-136 | normalizer constants | MOVE (derivable: `_fno.MEAN/STD`, `t2_common.py`; the in/tgt trap goes in the rule) | rule `track2-stream` / `package-code` | 500 |
| 137-140 | units, sim normalizer | MOVE | rule `training` | 450 |
| 141-147 | sim coverage, public set in range, AoA error profile | MOVE | EVALUATION §3 (mostly already there) | 650 |
| 149-179 | Scoring (formulas, SPS factorization, weights, exchange, noise, cross-track) | MOVE (duplicate of EVALUATION §2/§5); keep a 3-line pointer in CLAUDE | EVALUATION | 2,080 (+400 kept) |
| 181-196 | Board latency | MOVE/DELETE (duplicate of EVALUATION §5) | EVALUATION | 1,370 |
| 200-218 | Members, recipes, argparse-defaults trap | MOVE | rule `training` + REPRODUCE §1 | 1,720 |
| 220-235 | Inference pipeline, fuse, latency settings | MOVE (rewritten for 0.80/centre0/SIGBLEND/knobs) | rule `package-code` | 1,510 |
| 237-241 | "Why this shape" pointer | DELETE | ROADMAP Part 1 | 430 |
| 243-268 | Instruments rules | MOVE | skill `evaluate-change` + rule `instruments` | 1,960 |
| 270-300 | Closed list | MOVE (ROADMAP Part 2 already has it; update T2/ALPHA/band items); CLAUDE keeps one line: "check ROADMAP Part 2 before proposing an arm" | ROADMAP | 2,660 |
| 302-322 | Build and gate | MOVE | skill `build-package` | 1,440 |
| 324-335 | Submission contracts | KEEP | CLAUDE | 820 |
| 337-342 | Eval containers | KEEP 2 lines; detail MOVE | CLAUDE + rule `package-code` | 250 + 200 |
| 344-350, 355-356 | traps: never train through compose; score the real code path; live jobs re-import the tree | KEEP | CLAUDE | 500 |
| 351-354 | split leak via preprocessing; unexercised feature needs a verifier | MOVE | rules `training`, `package-code` | 400 |
| 357-359 | HDF5 locking, Slurm snapshots, waiter sentinel | MOVE | skill `osc-jobs` | 300 |
| 361-374 | Environment | KEEP ~700 (account/partition, never touch others, paths, python for both stacks); MOVE cpuset / debug-nextgen / liveness detail | CLAUDE + skill `osc-jobs` | 700 + 450 |
| 376-382 | rules and dates | KEEP, updated (phase ended, Decision 1 submission) | CLAUDE | 380 |
| 384-400 | Repo map | KEEP, updated | CLAUDE | 1,190 |

**Bucket totals (≈ 29,980):**

| bucket | chars | share |
|---|---|---|
| KEEP in CLAUDE.md (before rewriting; the rewrite should land at ~9-11 k) | ≈ 9,850 | 33 % |
| MOVE → path-scoped rules | ≈ 5,500 | 18 % |
| MOVE → skills | ≈ 4,700 | 16 % |
| MOVE → reference docs (EVALUATION / ROADMAP / REPRODUCE; most are already duplicated there) | ≈ 6,750 | 23 % |
| DELETE (stale status, wrong T2 rule, maintenance note, pointer) | ≈ 4,300 | 14 % |

Roughly 45 % of the file duplicates text that already sits verbatim or near-verbatim in EVALUATION.md, ROADMAP.md Part 2/4,
or REPRODUCE.md. The three biggest sources of error in the current file are Current state, Closed and the ALPHA/band
lines, and all three are status that changed. None of them is a rule.

---

## 4. Proposed new structure

### 4a. New `CLAUDE.md` (target ≤ 200 lines, ~10 k chars)

```
# AI4PDE — RealPDE (NeurIPS 2026)   ~6   problem paragraph, links, branches main / qwu/agent
## Where things are                 ~12  REPRODUCE · EVALUATION · ROADMAP Part 2 (closed) · ROUNDS (indexed)
                                         · handoff (only if newer) · ../sub_candidates · board_rounds.py
## Rules                            ~40  wording below
## Problem and contracts            ~30  shapes/units, T1 vs T2 protocol, contracts 324-335, containers (2 lines)
## Data                             ~10  paths, HF mirror, windowing, u/v top-level datasets
## Scoring                          ~6   five subscores; final = fitted rule (final_v9.W0); deltas; bad bounds zero all
## What ships (shape only)          ~10  6 members · twin-fitted band · T2 causal stack · constants = build_final.sh
## Engineering traps                ~12  compose; tree defaults ≠ shipped; real code path; live jobs re-import; seds;
                                         TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
## Environment                      ~12  Ascend A100 (build/gate/time) · Pitzer V100 (train) · two stacks · scratch
## Dates                            ~5
## Repo map                         ~20
```

**Proposed wording for the Rules section:**

```markdown
## Rules

1. **The user uploads; agents never do.** Main Development allowed one submission per track per UTC
   day; the Decision phase (to 2026-10-25) allows **one submission in total** per track. Never
   stage an upload without the owner's explicit go.
2. **Predict before upload.** Write all five predicted subscores, each with a bracket, into
   `../sub_candidates/<date>/README.md` and a PENDING row in `leaderboard/board.jsonl` before the
   upload. A prediction written afterwards is worthless.
3. **Evidence comes only from held-out twins.** All-data checkpoints supply weights, never
   numbers, and never the band's error field. Derive held-out lists in code
   (`re_split.excluded()`, `kfold.py --folds`); never type an `--exclude`.
4. **Track 1: each window is forecast from its own input only.** Windows tile (window k's target
   is window k+1's input), so any statistic pooled across windows in one `predict()` call leaks
   targets, even when it passes the permutation check. The disclosed trick
   (`Track_1/submit_trick/`) is never shipped. Nothing that needs history belongs on Track 1.
5. **Track 2: strictly causal.** `ttt_step` may use only the current input and targets already
   revealed; all state is reset in `reset_ttt_state`; return tensors the caller can keep. Every
   T2 package passes `validate_t2.py --adaptive` and `stream_eval.py --causal-check`.
6. **Nothing external.** No external data, no new simulations, no outside pretrained weights.
   The released `train_real` / `train_sim` are the only inputs.
7. **Everything that ships must be re-creatable from the release.** Organizers retrain
   shortlisted methods from scratch, and an irreproducible method is disqualified. A change to
   members, band or a baked constant is not done until `REPRODUCE.md`, `reproduce.sh` and
   `build_final.sh` produce it, and the band is committed in `Track_1/bands/` with a `MANIFEST.md`
   entry.
8. **Only baked constants ship.** `submission.py` in the tree holds defaults; the shipped values
   are the `RPDE_SHIP_*` knobs in `build_final.sh`. `RPDE_*` environment switches steer
   measurement drivers only. Commit before building; never edit `_fno.py`, `submission.py` or a
   build script while a build runs.
9. **No generated trees, caches or bulk data under `$HOME`** (1 M-inode quota). Jobs, logs and
   scratch go to `/fs/scratch/PAS1187/qwu/tmp/realpde/` or `/tmp`; `Track_1/unet/*.pth` and
   `../sub_candidates/` are the exceptions.
10. **Never touch other projects' Slurm jobs** (`hd_*`, `p5_*`, `p6_*`, `tt26-*`, `m2_*`,
    `ondemand/*`). Ours are `rp_*` (Ascend) and `pz_*` (Pitzer). `debug-nextgen` needs the owner's OK.
    No local timing while any `rp_*`/`pz_*` job is alive.
11. **Commits:** at most 5-7 a day on `qwu/agent`; squash only what is not on `main` yet; merge
    to `main` through a GitHub PR. Never commit checkpoints, `val_*.npz` or archives.
12. **Compare by board-subscore deltas, never by `final`**; never resubmit for a timing draw;
    one controlled lever per slot.
```

(Items 1, 2, 3, 4, 5, 6, 7, 9, 10 and 11 are the survivors you listed. Item 12 is the old "deltas / one lever" rule,
which prevents a real mistake and costs two lines.)

### 4b. `.claude/rules/*.md` (path-scoped, loaded on demand)

| file | `paths:` | outline |
|---|---|---|
| `package-code.md` | `Track_1/submit_fno/**`, `Track_2/submit/**`, `Track_2/build_t2.py`, `Track_2/validate_t2.py` | inference pipeline: normalize → fused groups (`fuse()`, StackedFFNO dense / StackedViT) → average field+σ → compose at ALPHA 0.80 (JOINT_GAIN needs CENTRE_ALPHA 0) → band centred on the α=0 field, SIGBLEND factor (T1 `EBAND` via the GPU-band path, so SMALL_GRAPH off; T2 `SIGPOW`; a SIGBLEND npz without the factor is ~3× too wide) · the full knob table (tree default vs shipped, from `build_final.sh`) · AMP check and fallback, bf16 crash, TF32 off · T1 CUDA graph closed (R27) · containers: no scipy/h5py/matplotlib/pandas, vendor, extract dir off sys.path · `mean_tgt ≠ mean_in` (T2 output affine) · malformed bounds zero every subscore · a feature no local data exercises needs its own verifier · build_round13 seds `_fno.py` and both `submission.py` |
| `track2-stream.md` | `Track_2/**`, `Track_2/r1_ref/**` | causality contract (rule 5 expanded); what the stream stack is: R1 (+ASYM, TSM, BLK), D1c (N0, EB, lead-time), VARCAL (VC_N0), SIGEMA (off); NEVER port R1/D1c to T1 (the batch holds future windows); reused-buffer trap (rel-L2 0.67); ARENA_STEPS / host RSS caveat (3.6 GB); PAR_STREAMS must stay on for timing; `submit/` in the tree is a dev copy with w24 symlinks; still closed: gradient TTT, readout refit, trajectory history |
| `training.md` | `Track_1/unet/**`, `reproduce.sh`, `REPRODUCE.md` | `train2.py` argparse defaults are not the recipe (list); recipes live in REPRODUCE §1 and in `torch.load(ckpt)["config"]` (`git`, `train2_argv`); twins = drop `--all-data`, add `--sim-exclude-held`; `--exclude` only filters the loss, so preprocessing that reads real statistics must be told the split (R14 leak); units and the sim normalizer; `--seed` controls init and batch order, but a retrain is not bit-identical (F-FNO ~0.5 %, ViT ~10 %), so judge on package score; torch ≥ 2.6 needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; checkpoints are untracked and live in `Track_1/unet/`, with scratch as the only redundancy |
| `instruments.md` | `Track_1/*.py`, `Track_1/verify/**`, `Track_1/qd_k/**`, `Track_1/sigblend/**`, `Track_2/stream_eval.py` | every local number is a sign, not a size; the operating point is `--alpha 0.80 --band-p 0.45 --centre-alpha 0 --fill-unscored --inflate 1.39` and instrument defaults (0.75 / 0.90 / 0.5) are wrong; score through `sps9.sc` only; screen at the ensemble size that ships; the AoA-holdout is the only OOD instrument; halve error cuts and charge removals in full; cross-family proxy reads carry no sign; no local instrument reads containment |
| `board-ledgers.md` | `leaderboard/**`, `Track_1/board_rounds.py`, `Track_1/final_v9.py`, `Track_1/predict_board.py` | a round goes into `ROUNDS`/`ROUNDS_T2` **and** `final_v9.BOARD` (import self-check); `pred` before `actual`; `W0` is the authority, so never quote weights from prose; add an anchor per round; `lb.py --html` is idempotent, republish to the same artifact link; `_zh` fields optional |
| `bands.md` | `Track_1/bands/**` | a scored band is never refit to recover it (a refit on a different twin set is not reproducible); every band gets a MANIFEST line (md5, rounds, members, twins, flags); SIGBLEND npz travel with their EBAND/SIGPOW literals (`sigblend/*.json`) |

### 4c. Skills (`.claude/skills/<name>/SKILL.md`)

| name | description line | outline |
|---|---|---|
| `build-package` | "Build and gate the Track 1 and Track 2 archives (final recipe or a one-knob variant), verify identity, stage them in ../sub_candidates/. Use before any upload or package comparison." | preconditions (committed tree, no `rp_*` job when timing, Ascend GPU node) → `bash Track_1/submit_fno/build_final.sh [OUT]` with `RPDE_REF_T1/T2` for identity → for a variant: change ONE `RPDE_SHIP_*`, record it → read both `SUBMITTABLE` lines (exit codes are not propagated) → T2 `--adaptive` + `--causal-check` → `pkg_identity.py` against the uploaded archive → sha256 + staging README entry → predictions (hand off to `predict_board.py --anchor`) |
| `record-board-result` | "After a Codabench result: ledger it everywhere, refit, republish, write the handoff. Use when the user reports board scores." | board.jsonl actual · runs.jsonl shipped · board_rounds ROUNDS/_T2 · final_v9.BOARD (import) · predict_board ANCHORS · lb.py --html + republish · ROUNDS.md append · ROADMAP Part 2/Part 1 if a rule changed · sub_candidates: archive → submitted/, band → Track_1/bands + MANIFEST · dated handoff · one commit. **First use: backfill the 13 rounds of 09-21..09-27 (F2).** |
| `reproduce-from-release` | "Re-create the shipped checkpoints, band and archives from the released data with reproduce.sh; check acceptance. Use for Decision-phase readiness or when a recipe changes." | two stacks (§0) · `reproduce.sh --recipe …` stages and `RPDE_REPRO_*` knobs · run from a copy of the repo (§8.2) · parallel stage jobs on Pitzer, ≥ 330 min limits · acceptance ②③ (package wfinal ±0.005, band rel-L2) · SIGBLEND band step · `build_final.sh` identity |
| `osc-jobs` | "Submit, monitor and time Slurm jobs on OSC Ascend (A100) and Pitzer (V100) for this project." | accounts/partitions (`-A PAS1186 -M ascend -p nextgen`, Pitzer gpu-exp) · `--chdir`, `--output` on scratch, `--exclusive` only when timing · cpuset pinning; nvidia-smi ≈ 0 % is healthy · `HDF5_USE_FILE_LOCKING=FALSE` · Slurm snapshots the script at submit · a waiter's sentinel is the last line and exits when the job leaves the queue · inode quota check (`quota -s`) · never touch foreign jobs |
| `evaluate-change` | "Decide whether a model/band/inference change is real, using held-out twins and the calibrated local instruments, and predict its board subscores." | pre-register the rule → twins with `--seed` control, replicated → `ens_v9 --aoa 15 20` at ship size + full split → `kfold --split re --bands 1 2 3 --paired` → `aoa_holdout_eval` for OOD → refit band/ALPHA downstream → `predict_board --anchor <same shape>` with brackets; the calibration table (halving, cross-track 0.85×, latency slopes) is in EVALUATION §5 |

### 4d. `ROADMAP.md` (111 KB) and `ROUNDS.md` (199 KB)

* **ROUNDS.md: index it and keep it whole.** It is an append-only archive, and code, reports and plans resolve old
  pointers ("CLAUDE.md round 14") into it, so splitting it would break those links. Three changes:
  (1) a 30-line index table at the top (round → date → T1/T2 final → heading line);
  (2) move the "CLAUDE.md detail moved here on 2026-09-15" appendix (l.2584-2679) up with the other reference
  sections, so the file ends on the newest round;
  (3) append one section for 09-21..09-27 (r35 centre0, r37, f4v1, QD, A/R1, 09-26, finals K27/MRGP2), written from
  `20260927/README.md`, the sprint notes, and git messages 55f0e88..e474435.
  Nothing in it is loaded automatically, so its size costs nothing until someone reads it.
* **ROADMAP.md: split by lifetime.** Its header (l.1-48) is five weeks stale, and Part 3 (l.1166-1456) is mostly the
  round-13 plan.
  * `ROADMAP.md` (target ≤ 25 KB): a new 10-line header (finals, Decision phase); Part 1 reduced to a numbered list of
    board findings, one or two lines each, with a pointer to the §1.x prose; Part 2 (closed) kept and corrected (T2
    adaptation, ALPHA, band); Part 3 replaced by the Decision-phase plan.
  * Move §1.1-§1.30 prose and the old P0-P4 verbatim to the end of ROUNDS.md (or to `docs/archive/roadmap_20260920.md`)
    with an anchor note, so "ROADMAP §1.17" pointers still resolve.
  * Part 4 (where things are): fold into README.md's repo table and the rules; delete from ROADMAP.
* **EVALUATION.md** stays the reference that the skills and rules link to. Refresh §2 (weights pointer), §4 (operating
  point), §5 (anchors), the track-record table and §6 (→ skill pointers).

---

## 5. The three existing skills (all git-tracked under `.claude/skills/`, so they ship with the open-source release)

| skill | project-relevant? | verdict |
|---|---|---|
| `rockfish-workflow` | No. It covers Rockfish (JHU), `ml anaconda`, the conda env `OED` and `./command.rockfish`; this project runs on OSC Ascend/Pitzer. Its frontmatter `name: Rockfish HPC Workflow` is also not a valid skill slug. | **Remove from the repo.** Move it to `~/.claude/skills/` if the owner still uses Rockfish. Replace it with `osc-jobs` (§4c). |
| `debug-workflow` | Generic. Two parts conflict with this repo: "save ALL results to the same folder as the test codes / never a separate output folder" breaks the `$HOME` inode rule and the scratch convention, and "wait for approval before implementation" conflicts with autonomous job workflows. Its description ("Use when writing test code or debugging") matches almost any task. | **Remove from the repo** (move to user scope if wanted). If kept, rewrite it so outputs go to scratch. |
| `karpathy-guidelines` | Generic coding hygiene, MIT, harmless, and no conflict. It is not project knowledge, and its broad description makes it load often. | **Move to user scope** (`~/.claude/skills/`). Keep it in the repo only if the owner wants collaborators to inherit it. |

---

## 6. Open questions for the owner

1. **Committing the final code (F1).** Is the current working tree exactly the MRGP2/k27fin merged tree
   (`final0927/wt_mrgp2`)? `build_final.sh` with `RPDE_REF_T1/T2` pointed at the two uploaded zips would prove it.
   Once that passes, can the diff, `build_final.sh`, `sigblend/`, the two bands and the verify scripts be committed on
   `qwu/agent`, and PR'd to `main` before the Decision phase?
2. **The 13 post-09-20 board rows (F2).** Are the exact subscores, submission ids and predictions for 09-21..09-26 in a
   ledger on the other server (a `board.jsonl` or `realpde-packages/`)? Or should they be reconstructed from the
   sprint READMEs and Codabench?
3. **Uploaded archives since 09-21 (F3).** Where are the byte-exact uploads, for example `track1_r38_27lz_a080.zip`
   "on the other machine"? Should `submitted/` be completed from there?
4. **What the organizers retrain.** T1 and T2 finals differ: T2 adds a training-free stream stack, and both share the
   members and band recipe. Is `FINAL_RECIPE` one method with a T2 extension, and is `build_final.sh` the single
   command REPRODUCE.md should name? The SIGBLEND constants were fit in-sample on 8 validation trajectories; is that
   acceptable for the release, or should the recipe say so explicitly?
5. **The Decision-phase submission.** With one submission in total, is anything still planned for upload? If not,
   the `record-board-result` and `evaluate-change` skills are for the record and backfill only.
6. **Status location.** Should post-phase status live in a new dated handoff, a tracked `STATUS.md`, or only in
   `../sub_candidates/<date>/README.md`?
7. **The multi-agent team conventions** (named agents Ilse, Jonas, Devi and others, and `pitzer-requests/`). Does that
   team continue, and should its conventions (owner locks, audit ids) go into a rule or skill, or stay out of the repo?
8. **ROADMAP split.** Can Part 1/Part 3 prose move to an archive file, or must "ROADMAP §1.x" stay in ROADMAP.md?
9. **User-scope skills.** Should the three generic skills move to `~/.claude/skills/`, or be deleted?
10. **QD / `Track_1/qd_k/`.** QD never became the final recipe. Is it closed, so it moves to `Track_1/closed/` with a
    verdict, or kept as a documented alternative for the private (OOD) set?
