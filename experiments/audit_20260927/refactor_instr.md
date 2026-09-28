# Instrument refactor: one operating point, one band loader (2026-09-28)

Worktree `wrap0927/wt_instr` (detached at 7b6ee8e, nothing committed). Patch: `notes/patch_instr.diff`
(20 files, +715 / -152; new files `Track_1/opoint.py`, `Track_1/band_io.py`). GPU job **7569335** (5 min 51 s, A100 a0191).
Cross-check script (kept outside the repo): `wrap0927/jobs/instrI_xcheck.py`; job script `jobs/rp_w27i_xcheck.sh`;
log `logs/rp_w27i_xcheck_7569335.log`; fresh twin dump `instrI/dump_unseen_fresh.npz`.

## What changed

* **`Track_1/opoint.py`** (new, stdlib only). ALPHA 0.80, JOINT_GAIN True, CENTRE_ALPHA 0.0, BAND_P 0.45, INFLATE 1.39,
  F_TAIL 1.36, FILL_UNSCORED True, CLIP_LO/HI 0.3/3.0, SIGMA_EPS 1e-8, PARENT_BAND `bands/r38_a080_c0_fill.npz`,
  FINAL_BAND `bands/k27_sigblend_lam0.5.npz`, FINAL_SIDECAR `sigblend/band_k0_sigblend_lam0.5.json`, the
  SIDECARS map and `path()`. Its docstring says `submit_fno/build_final.sh` is the authority for what is BAKED.
  Nothing on the build path imports it.
* **`Track_1/band_io.py`** (new, numpy + opoint only, so importing it does not load the kit). I put it in a separate file
  instead of `sps9.py` because `fit_band.py` imports `sps9`. `sps9.py` gets ONE added function, `sps_interval`, and
  no existing line changed (the diff only adds lines).
* Instruments rewired: see the table. `EVALUATION.md` §4 gets the old/new table (nothing else in that file), and
  `closed/README.md` gets the one paragraph about `eval_v9.infer`'s default.

## Loader contract (`band_io`)

`load_band(path, sidecar=None) -> Band` (frozen dataclass): `base`, `sigma_ref` (float32, channels as stored), `p`,
`centre_alpha` (None when the npz has no key = centred on P, the pre-09-21 convention), `q` (2,) = lam·q_c (zeros
when static), `b` (2,) = the EBAND intercept (0), `sidecar`, `kind` ∈ {static, sigblend}, `describe()`.
* If |p − 0.45| > 1e-6, a sidecar is **required**. The loader looks for `<stem>.json` beside the npz, then in
  `opoint.SIDECARS`. If it finds none, it **raises** with the 3× explanation. `sidecar="static"` is an explicit escape
  hatch for an old static band fitted at another `--band-p`.
* The sidecar is parsed from the **EBAND literal Track 1 bakes** (`ast.literal_eval`). It must be `mode "mult"` with
  `clip None` and coefficients 1..4 all zero, else NotImplementedError. The loader cross-checks it against
  `T2_sigpow` (1e-6) and checks `p_new` against the npz's p, so a mismatched pair raises.
* It refuses an `--err-head` band (`err_mode`) and a `clip_lo/hi` that differs from the package's hard-coded 0.3/3.0.

`half_width(band, sigma_phys)` returns `base[..., :2] * np.power(np.clip(σ/σ_ref, 0.3, 3.0), float32(p))`, then for
SIGBLEND `* exp(log(σ + 1e-8)·q + b)`. Everything is float32, the same expressions as `submission.py` (host path) and
`_fno.predict_fno` (band_gpu + the SIGBLEND fast path). Output shape is (N, 20, H, W, 2).

`interval(band, y0, P, sigma_phys, alpha=None, joint_gain=False, half=None) -> (lower, upper)` on u and v:
* centre None → P ∓ half
* centre 0 → [min(y0 − h, P), max(y0 + h, P)]
* other c → y0 + (c/α)(P − y0), then the same min/max. This raises under JOINT_GAIN, as the package does.

Helpers:
* `centre()`
* `stretch_target(band, Y, y0, P, f)`: the board tail as a target move about the interval centre. For a P-centred band
  this is exactly the old `f·|P − Y|`; for a y0-centred band it is bandlab's convention.
* `joint_gain_for(band, flag)`: `--joint-gain/--no-joint-gain`, where neither flag means auto = `opoint.JOINT_GAIN` if
  the band is centred on y0, else off. An explicit `--joint-gain` with a non-y0 band raises, because the package asserts
  that pair.
* `sps9.sps_interval(a, Y, lower, upper, scored=None, n_scored=None) -> (sps, C, coverage%)` is the kit's per-element
  rule for an arbitrary interval, with `a` given.

## Defaults table (old → new; line numbers are the new file's)

| instrument | file:line | old | new |
|---|---|---|---|
| eval_v9 `infer(alpha=)` | eval_v9.py:114 | 0.75 | `opoint.ALPHA` 0.80 (signature kept; JOINT_GAIN left to the caller) |
| eval_v9 `--alphas`, `--joint-gain` | eval_v9.py:128,130 | [0.75], – | [0.80], on |
| compare_v9 `--alpha --inflate --joint-gain` | compare_v9.py:44,46,49 | 0.75, **1.5**, – | 0.80, 1.39, on |
| ens_v9 `F_BOARD`, `--alpha`, `--joint-gain` | ens_v9.py:40,49,51 | 1.39, 0.75, – | opoint.INFLATE, 0.80, on |
| predict_board `F_TAIL`, `--band`, `--alpha`, `--joint-gain`, `--f-tail` | predict_board.py:74,411,418,419 | 1.36, untracked `submit_fno/band.npz`, 0.75, – | opoint.F_TAIL, FINAL_BAND (+sidecar), 0.80, auto, 1.36 |
| kfold `--alpha --joint-gain --inflate` | kfold.py:301,303,306 | 0.90, –, 1.39 | 0.80, on, opoint.INFLATE |
| re_split `--alpha --joint-gain --inflate` | re_split.py:94,96,99 | 0.90, –, 1.39 | 0.80, on, opoint.INFLATE |
| mean_blend_re `--alpha --joint-gain` | mean_blend_re.py:92,94 | 0.90, – | 0.80, on |
| aoa_holdout_eval `--alpha --joint-gain --centre-alpha --band-p --inflate` | aoa_holdout_eval.py:145-156 | 0.60, –, (P), 0.45, 1.39 | 0.80, auto, 0 (y0), opoint.BAND_P, opoint.INFLATE |
| calib_band13 `--alpha`, `--rounds`, `--joint-gain` | calib_band13.py:74,~70,76 | 0.60, R13 T1+T2R1 fixed | 0.80, "T1 R37" (R13 selectable), auto |
| quant8 `--alpha --joint-gain` | quant8.py:64,66 | 0.60, – | 0.80, auto |
| lat_r15 `--band --alpha --joint-gain` | lat_r15.py:89,91,93 | untracked `Track_2/submit/band8.npz`, 0.60 | FINAL_BAND, 0.80, auto |
| lat_fast17 `--alpha --joint-gain` | lat_fast17.py:124,126 | 0.60 | 0.80, auto |
| alpha_board `F_ELEM`, `--alphas`, `--joint-gain` | alpha_board.py:70,90,~96 | 1.39, [.3 .45 .6 .75 .9] | opoint.INFLATE, adds 0.80, **off** (see below) |
| screen_eval `ALPHA, BAND_P, F_BOARD` | screen_eval.py:79 | 0.60, 0.45, 1.39 | opoint.ALPHA/BAND_P/INFLATE; `sweep["0.6"]` → `sweep[AKEY]` |
| verify/quant_identity `--alpha --joint-gain --band` | verify/quant_identity.py:38,40,43 | 0.60, –, untracked `submit_fno/band.npz` | 0.80, on, FINAL_BAND |

## How each band-scoring instrument now works (I3)

* **predict_board.** Band through `load_band`. P, σ and y0 come from one `predict_fno(..., centre_out=)` call. The
  interval and stretch go through `band_io`, and scoring through `sps_interval` at a constant board `a`. It sets
  `_fno.JOINT_GAIN` per call, for the candidate and the anchor separately.
  * New `--sidecar`, `--anchor-sidecar`, `--anchor-joint-gain` (auto = on only for anchors in `ANCHOR_JOINT_GAIN = {"r37"}`)
    and `--f-tail`.
  * New anchor **`r37`**: literal subscores copied from `board_rounds.py` R37. T2R24 is deliberately not added, because
    its adaptive streaming is not replayed locally.
* **calib_band13.** Band via `load_band`; the y0 centre from the twins; the stretch search on `sps_interval`. Refit rows
  fit about the band's centre. For SIGBLEND the refit is a static band at `opoint.BAND_P` (not p′), labelled as such.
  The two-track agreement line now works for any number of rows.
* **quant8 --score.** σ, P and y0 come from the same packed models; the package interval.
* **aoa_holdout_eval.** It fits its own band (kept), now about the chosen centre (|y0 − y| at centre 0, as in
  `fit_band --centre-alpha 0`). The package interval is used on every row, including the blend rows, where P moves and
  y0 does not. It does not replicate `--fill-unscored` or SIGBLEND; the docstring says so.
* **lat_fast17.** The timed section is unchanged: the pre-09-21 host-band replica. For a centred or SIGBLEND band, its sps
  column is rescored from one untimed pass through `band_io`.
* **lat_r15.** Loading only goes through `band_io`; the timed arithmetic is unchanged. The help text notes that the
  SIGBLEND exp is not timed.
* **screen_eval.score_block.** Goes through `band_io.half_width` and raises on a centred band, because it scores P ∓ half.
* **compare_v9, ens_v9, kfold, re_split, mean_blend_re, eval_v9** fit their own symmetric bands. That behaviour is kept;
  only the defaults changed, plus `--joint-gain/--no-joint-gain`, which sets `_fno.JOINT_GAIN`. None of them reads a band
  file.

## Cross-check (I4), job 7569335

Five held-out twins on val_unseen (8 trajectories, 336 windows). sps is the mean over trajectories, as in
make_k0_band. The fresh twin inference is **bit-identical** to the dump behind job 7569179 (max |dF| = max |dS| = 0).

| row (f = 1.00) | hi sps / C | lo sps / C | **all sps / C** |
|---|---|---|---|
| make_k0_band 7569179, static r38 | 47.794 / 64.005 | 57.805 / 73.330 | **54.051 / 69.833** |
| band_io, static r38, bandlab P | 47.794 / 64.005 | 57.805 / 73.330 | **54.051 / 69.833** |
| make_k0_band, SIGBLEND λ0.5 LOTO-7 | 47.902 / 64.151 | 58.538 / 74.244 | **54.549 / 70.459** |
| band_io, SIGBLEND k27 committed, in-sample | 47.895 / 64.142 | 58.605 / 74.326 | **54.589 / 70.507** |
| band_io, static r38, `_fno` α0.8 JOINT_GAIN (package P) | 47.808 / 63.995 | 57.814 / 73.321 | 54.062 / 69.824 |
| band_io, SIGBLEND k27, `_fno` α0.8 JOINT_GAIN (package P) | 47.910 / 64.132 | 58.613 / 74.314 | 54.599 / 70.496 |

* **Static row.** It matches 7569179 to every printed digit, at f = 1.00, 1.29 and 1.39, and in all three columns
  (sps, C, cov). The interval is therefore the package's.
* **SIGBLEND** comes out +0.040 all (+0.067 lo, −0.007 hi) against LOTO, for two reasons:
  1. In-sample: the committed κ and q were fitted on all eight trajectories, including the one being scored. LOTO refits
     per held-out trajectory, which costs the lo-AoA group most, where the gain is.
  2. The committed q = (0.4847, 0.4907) comes from the original 09-27 fit. 7569179's refit on this dump gives
     (0.4854, 0.4909), and its written EBAND differs in the 4th decimal (0.24270 against the committed 0.24233).
  Both differences are expected; neither is a formula difference.
* **Kit check.** A pooled kit `aggregate_sps(lower=, upper=)` against `sps9.sps_interval` agrees to 1e-6 on every
  variant (e.g. 54.519224 / 54.519224).
* **JOINT_GAIN** (package P): rel 0.07505 → 0.07470, tke and mvpe identical, sps +0.010 to +0.011. This matches R37's
  promise (tke and mvpe bit-identical, rel up).
* `quant8 --score` on the same twins with k27 gives fp32 sps **54.530**, consistent with the pooled 54.5298 above, so two
  instruments agree.

**The trap, quantified** (CPU, same dump). k27 read as static has median half 0.02902 against 0.00642 with its sidecar
(**4.5×**, not "about 3×"), and pooled sps drops **27.36 against 54.52**. Without a sidecar the loader now raises.

**predict_board** (AoA 15–20, 126 windows, `--local-ms 2.9`):
* **New defaults, absolute mode:** runs, k27 + sidecar, α 0.8, JG on. Predicted final 81.221; this is the R10-ratio
  absolute mode, a level only.
* **New defaults, `--anchor r37`**, same members and twins: returns r37 exactly (+0.000 on every subscore, final
  81.671). The identity control passes.
* **Old explicit arguments** (`--alpha 0.60 --band bands/r34_p2ffnoL8sim_fill.npz --anchor r30`): runs; JG auto-off
  (P-centred band). Returns r30 exactly (final 81.419). The identity control passes on the old path too.
* **Candidate r38 against anchor r37 (k27):** sps −0.023, final −0.006. On the AoA 15–20 proxy the SIGBLEND gain is
  small (LOTO hi +0.019 at f 1.29); the board paid +0.093 (R37 − R35), mostly from the low-AoA containment the hi proxy
  does not see.

**calib_band13** at the new defaults (R37, k27, JG) inverts R37's board sps 43.156 at **f = 1.290**. That is exactly
the `K0_INFL = 1.29` SIGBLEND was fitted at, against the 1.36 predict_board uses for older bands. The best static refit
is −0.042 below the shipped SIGBLEND at that f.

## Smoke (I5)

* `ast.parse`: all 18 touched .py pass.
* `--help` exits 0 for compare_v9, ens_v9, eval_v9, predict_board, kfold, re_split, mean_blend_re, aoa_holdout_eval,
  calib_band13, quant8, lat_r15, lat_fast17, alpha_board, verify/quant_identity and screen_sets.
  * mean_blend_re's `--help` crashed **at HEAD as well** (an unescaped `%` in the `--ref` help). I fixed it as `%%`.
* **Run on CPU (login node):**
  * `predict_board.py --sps-selftest`
  * `kfold.py --folds --split re`
  * `aoa_holdout_eval.py --ckpt unet/p2_ffno_l8_sim_s1.pth --holdout-aoa 20 --limit 1 --device cpu --inflate-sweep 1.39 1.8`
    at the new defaults, 56 s: fit sps 56.14, AoA-20 46.99.
  * **Parity:** the same with `--alpha 0.60 --no-joint-gain --centre-alpha none`, against the HEAD version of the
    script, gives **byte-identical output**.
* **Run on GPU (job 7569335):** predict_board ×4, calib_band13, ens_v9 (5 twins), compare_v9 (1 twin, AoA 15–20), and
  `quant8 --score` k27 (all schemes and the shipped packer).
* **Not run:**
  * lat_r15 and lat_fast17: they need an idle `--exclusive` node to mean anything, and other `rp_w27*` jobs were running.
  * eval_v9, re_split, mean_blend_re and alpha_board: they need val_re* caches or long runs; --help only.
  * screen_eval: it writes the shared ledger.

## Left alone, and why

* **The build path:** `submit_fno/*`, where `fit_band.py`'s defaults are load-bearing for `build_round13.sh`. Also
  `unet/`, `closed/` (except the README paragraph), `archive/`, `qd_k/`, `sigblend/`, `Track_2/`, `board_rounds.py`,
  `final_v9.py`, and the identity verifiers `verify/fuse_*` and `selfnorm_path`.
* **`proxy_calib.py` and `sub_value.py`:** their alphas are per-round records (R10 0.75, R11/12 0.90, R13 0.60), not
  defaults.
* **`alpha_board.py` `--anchor 0.60` and JOINT_GAIN off by default.** Its BOARD constants are R13's row, a per-channel
  compose at 0.60, and the inflation factors are defined only there. The help text says to pass
  `--board ... --anchor 0.80 --joint-gain` for R37.
* **Not operating-point constants, still dated:** `ens_v9 --time-score 90.5066` (round 8), `alpha_board --time-score`,
  `lat_*` BOARD_C/INV/LOC (round 13) and screen_eval `T_FIXED` / anchors r25/t2r12.
* **Prose kept at the stale 0.60:** EVALUATION §4 (per the "table only" instruction), and the docstrings of
  screen_eval and screen_sets.

## Surprising

1. The "about 3×" in the docs understates the trap: it is 4.5× in median half-width, and sps halves.
2. R37's board sps inverts to a tail f = 1.29 against the SIGBLEND band. That number was the SIGBLEND fit's own
   `K0_INFL`, while every other instrument still assumes 1.36 or 1.39.
3. screen_eval hard-coded `sweep["0.6"]` as "the single model at ALPHA". Changing ALPHA alone would have silently mixed a
   0.60 band with 0.80 predictions; it now uses `AKEY`.
4. With the per-channel compose, bandlab's compose (eps 1e-8) and `_fno`'s (eps 1e-6) agree to 4e-6 in pooled sps.
5. The new defaults move aoa_holdout_eval's sps by +0.6 (fit split) and +1.0 (AoA 20) against the old ones on the same
   checkpoint. That is the centre-0 gain, and it is why the old defaults under-read the OOD band.
