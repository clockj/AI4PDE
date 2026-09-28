# Track 1 — closed experiments

Every script here asked a question that is now **closed**, and is kept only so a future session
can read how it was closed rather than re-derive the negative. Nothing here is imported by a live
instrument, a gate, the package code or any job script; the verdicts are in
[`../../ROADMAP.md`](../../ROADMAP.md) Part 2 and the round detail in
[`../../ROUNDS.md`](../../ROUNDS.md). What is still worth running lives one level up — see
[`../README.md`](../README.md) and [`../../EVALUATION.md`](../../EVALUATION.md).

**The path shim.** These files used to sit at the top level of `Track_1/`, and they import
`sps9`, `eval_v9`, `final_v9` and each other by bare name and read `val_*.npz` and `unet/*.pth`
through `HERE`. Each one therefore carries a five-line block after its docstring that puts the
**parent of `closed/`** on `sys.path`, and every `os.path.dirname(os.path.abspath(__file__))` in
the file gained one `dirname`, so `HERE` still resolves to `Track_1/`. Nothing else changed.
All 51 files of the 09-19 move compile and load from here; the 45 with `argparse` were checked with `--help`, and
the six without (`tke_ceiling.py`, `verify_scoring_20260829.py`, `mvpe_fix_20260829.py`,
`diag_history.py`, `divergence.py`, `eval_all.py`) were checked by executing their module body.
`tke_ceiling.py` and `eval_all.py` still fail at run time for a reason that predates the move:
they load the **v7** starting kit by absolute path and `~/RealPDE/realpde_t1_starting_kit_v7/`
no longer exists.

## The SPS band — closed eight times

| file | the question | verdict |
|---|---|---|
| `band_v9.py` | the first v9 half-width exploration | superseded by `sps9.optimal_band` + `submit_fno/fit_band.py` |
| `band2_v9.py` | does a SECOND per-window scaling axis help? | no; σ correlates with \|error\| at 0.551, ensemble spread at 0.347 |
| `spread_band.py` | can ensemble disagreement adapt the band per window? | no (round 6) |
| `band_sigma_v9.py` | a per-SAMPLE band scaled by the model's own σ | no gain over the per-element band |
| `band_asym_v9.py` | must the interval be CENTRED on the prediction? | exact answer: centred is optimal |
| `band_shift_v9.py` | is an OFF-CENTRE interval worth anything? | no |
| `band_cv_v9.py` | choose the fitting set and (p, smooth) by leave-one-trajectory-out CV | 84 configs; pooling the fitting split buys 0.000 sps (round 11) |
| `band_kfold_v9.py` | fit the band's shape on k-fold out-of-fold errors | no better than the 126-window AoA 15–20 fit |
| `band_aoa.py` | fit the band on the regime the board resembles | the idea survived; the script did not — it is `fit_band.py --fit-aoa` |
| `inflate_v9.py` | how much to stretch the local error field before solving | superseded by `calib_band13.py` (the measured tail) and `fit_band.py --inflate` |
| `tail_map_v9.py` | WHERE the SPS misses are and what accuracy would flip them | containment is not reachable from a local instrument (`EVALUATION.md` §7) |
| `posthoc_v9.py` | post-hoc transforms of the shipped prediction array | every one null or negative |

## ALPHA, the σ head and TKE

| file | the question | verdict |
|---|---|---|
| `alpha_v9.py` | sweep ALPHA on local errors | superseded by `../alpha_board.py`, which sweeps against the board's error scale |
| `gain_v9.py` | the fluctuation gain as a 2-parameter family instead of 1 | the extra parameter buys nothing |
| `fluc_calib.py` | does a fitted per-pixel fluctuation gain beat the σ head? | no |
| `sigma_blend_v9.py` | can a blend of σ estimators beat the head alone? | no; post-hoc σ calibration is three separate nulls, worst −0.311 |
| `mcdrop_v9.py` | MC dropout as a zero-storage ensemble | member count is closed; MC members are not free members |
| `eval_fgn.py` | do an FGN model's SAMPLES beat the log-MSE σ head? | sample spread predicts σ at 0.451–0.469 against the head's 0.329 (round 12) |
| `tke_branch_v9.py` | how much `final` is left in the TKE branch, at most? | closed on PHYSICS: the 20-frame variance target is mostly sampling noise (ROADMAP Part 2) |
| `tke_ceiling.py` | how much of the TKE error is predictable? | same closure; we already score 0.538 against a hindsight long-run 0.570 |
| `oracle_split_v9.py` | where is the remaining score, by component of the prediction? | mean / σ / shape decomposition; answered, no arm survived |

## Calibration, superseded

| file | the question | verdict |
|---|---|---|
| `calib_board.py` | calibrate the band against a MEASURED board score (round 7) | superseded by `calib_board11.py`, then `../calib_band13.py` |
| `calib_board11.py` | re-invert the tail against the round-10 point | superseded by `../calib_band13.py`, which handles the `.npz` band |
| `verify_scoring_20260829.py` | anchor-verify the v9 scorer reconstruction against zero-parameter predictors | superseded by `../verify/diff_scoring_{official,real}.py` once the official kit arrived |
| `mvpe_fix_20260829.py` | runtime patch: make the reconstruction use the kit's per-column MVPE | the fix is in the reconstruction; the patch is history |

## Diagnostics — where the error lives

| file | the question | verdict |
|---|---|---|
| `diag_spectral.py` | split the error into time-mean and fluctuation parts | the fluctuation part is near its floor (coherent fraction Γ² = 0.308) |
| `diag_mean_floor.py` | is the time-mean error noise-limited? | its estimate is an OVERESTIMATE — see `mean_floor2.py` |
| `mean_floor2.py` | re-derive that floor honestly | the model already beats it; the rel-L2 "ceiling" is withdrawn (ROADMAP §1.9) |
| `floor_aoa.py` | is anything left in rel-L2 in the regime the board asks about? | no identified source |
| `gap.py` | what does meeting a NEW trajectory cost? | the binding constraint is the 74 trajectories, confirmed from five directions |
| `mvpe_probe.py` | is the time-mean beaten by carrying the input window's mean forward? | no |
| `diag_history.py` | does the trajectory's own past beat the 20-frame window? | trajectory history is worth nothing (also closed on Track 2) |
| `ctx_probe.py`, `dmd_probe.py`, `hist_blend.py` | long context and a parameter-free per-window linear model | every blend loses; shipped off as `HIST_BETA = 0.0` |
| `diag_phase.py`, `diag_phase_fluc.py` | how much of the error is phase/advection? | re-registration recovers 0.2 % on the real model where its synthetic control recovers 100 % |
| `diag_analog.py` | analog forecasting: is the wake periodic enough? | no |
| `diag_boundary.py` | where does U-FNO's local advantage live? | U-FNO is closed (R16: the board disagreed with every local instrument) |
| `divergence.py` | does the data satisfy incompressibility? | real PIV does not; ∇·u → 0 fights the target |
| `osc_probe.py`, `osc_ablate.py` | what does the oscillatory head learn, and is it a regularizer? | the temporal-basis heads are closed |
| `sim_map.py` | gate 0 for sim + real: the paired sim → real domain gap | the map shipped as `../unet/simreal.py`; this measurement is done |

## Ensembling, packaging and screen one-offs

| file | the question | verdict |
|---|---|---|
| `compose_v9.py` | price every candidate ensemble COMPOSITION in board `final` | member count is closed (R12 ≈ R11 ≈ R10 at constant latency) |
| `ens_mixed.py` | score a mixed FNO + U-Net ensemble | the U-Net family is closed |
| `ens_stack_v9.py` | post-hoc levers on a heterogeneous ensemble | uniform member weights are optimal |
| `eval_all.py` | score every model on both validation sets | superseded by `../compare_v9.py` / `../ens_v9.py` |
| `compress_spectral.py` | how big is an FNO package and what shrinks it? | superseded by int8 packing and `../quant8.py --score` |
| `prof_q1g.py` | round 17 Q1g: where the shipped path spends its time | its conclusion ("not FLOP-bound") was overturned by R24 — FLOPs ARE charged |
| `screen_oracle.py` | does knowing the TRUE (Re, AoA) remove held-out error? | no: the model already reads the condition from the window (09-14 screen §6o) |
| `screen_profile_swin.py` | where Swin's Track-2 step goes | Swin is closed on the board (R26) |

**A note on stale references.** `submit_fno/build_round11.sh` mentions `band_cv_v9.py` and
`gain_v9.py`, and `unet/run_ctx.sh` mentions `diag_analog.py`, but only in comments — no shell
script in the repo and no job script on scratch invokes anything in this directory.

## Moved here on 2026-09-27

| file | the question | verdict |
|---|---|---|
| `exchange_v9.py` | what is an across-the-board accuracy gain worth in each subscore? | frozen as the exchange rate in `../../EVALUATION.md` §2 (rel +0.346, tke +0.290, mvpe +0.100 per 10 % of error) |
| `eval_wp.py`, `eval_pair.py`, `calib_sigma.py` | the scorers of the U-Net era (kit v7) | superseded by `../eval_v9.py` and `../sps9.py`; they stop at import because the v7 kit is gone |
| `dinov2_init.py` | does a DINOv2-S initialisation help the ViT? | no gain (L8 +0.28 sps, L12 −0.12), and outside pretrained weights are now **prohibited** |
| `distill_labels.py`, `distill_pipeline.sh` | can one student replace the six-member ensemble? | no: the mean transfers, the variance does not (K1 81.779 against the teacher's 82.320). `train2.py --teacher` is the training side and stays in place |

The five Python files carry the same path shim as the rest of this directory.
