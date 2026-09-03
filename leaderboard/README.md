# Internal leaderboard

Two append-only JSONL ledgers plus one script. It answers the question thirteen rounds of prose
could not: **does a local gain transfer to the board, and by how much?**

```
leaderboard/
├── board.jsonl   one record per SUBMISSION  — package, thesis, PREDICTION, actual, local numbers
├── runs.jsonl    one record per LOCAL EXPERIMENT — arm, instrument, effect, verdict, did it ship
└── lb.py         loads both, fits the local→board correction, renders the page
```

```bash
python leaderboard/lb.py            # text
python leaderboard/lb.py --html     # also write ../../leaderboard.html — OUTSIDE the git tree (env RPDE_LB_HTML overrides)
```

## What it currently says

* **The plateau since 2026-08-08 is broken, and both breaks are regularization.** Dropout
  (+0.200, T2R2) and then capacity at width 96 regularized harder (+0.166, **T2R3 81.019** —
  the account's best on either board). Rounds R12–R14 still sit within 0.013 of each other at
  constant latency.
* **T2R3's gain arrived through `sps` (+1.01) and `tke`, while `rel_l2` got WORSE.** Capacity ×
  dropout removes the per-element **tail**, not the rms — which is what SPS's containment
  scoring pays for, and the first board demonstration of round 9's "tail reduction beats mean
  reduction". `rel_l2` alone would have called the package a regression.
* **R14's null is a measurement.** It shipped R13's package plus a mean-blend gated on the
  window's own speed, and the board returned all four error subscores *bit-identical* to R13's
  — the gate never fired, so the hidden set is inside the training speed range and the whole
  Re-*extrapolation* line is retired as a decision instrument. As a bonus the pair differs in
  `time_score` alone, which measures `w_time = 0.0956` directly.
* **Levels calibrate, deltas do not.** The proxy→board ratios are stable to ±2–8 % (`rel` 1.068,
  `tke` 1.136, `mvpe` 1.454), so the proxy predicts the board's *level* well. Round to round it
  explains almost none of the board's *movement*: **residual rms 4.15 % of the error**, range
  −9.7 to +5.9 %. `mvpe` inverts sign in both tracks.
* **⇒ The resolution limit is ~4 % of error — but it is BLUNT, not conservative.** Dropout
  (−2.6 % locally) and w96 (−1.2 % at 3-vs-3) both sat under it and both transferred. It gets
  the sign right and the size wrong; do not read a margin off it either way. Corroboration that
  the residual belongs to the proxy and not to a board draw: R14→R15 and T2R1→T2R2 are the
  **same package change scored on the two boards** and their residuals agree to 0.7 pp on all
  three metrics.
* **`ens` is the most predictive column in `runs.jsonl`.** `ozdo` was +3.84 σ per model and
  −0.012 at four members; `--var-mlp`'s 3.9 % fixed-field TKE cut became +0.08. An arm that has
  not been measured at the ensemble size that ships has not been measured.

## Appending

**After a board result** — this is the correction step:

1. fill `actual` on that round's record in `board.jsonl` (all five subscores + `final`);
2. set `shipped` on the arms it carried in `runs.jsonl`;
3. rerun `lb.py`. The ratios, the resolution limit and the prediction-residual history all refit.
4. paste the same subscores into `Track_1/board_rounds.py` (`ROUNDS` for Track 1,
   `ROUNDS_T2` for Track 2) **and** `Track_1/final_v9.py:BOARD`, keeping the row order
   ROUNDS-then-ROUNDS_T2 — that is what refits the subscore *weights*, which `lb.py` imports
   through `final_v9`. The fit pools both tracks; they share one test set and one rule.

**Before a submission**, write the record with `actual: null` and `pred` filled in. A prediction
recorded after the fact is worthless; the whole instrument depends on writing it down first.

**After a local experiment**, append to `runs.jsonl`. `ensemble_tested` is not optional — if the
arm was only measured on a single model, say so, and expect the effect to be smaller.

## Fields

`board.jsonl` — `round`, `track`, `id`, `date`, `package`, `thesis`, `invocations`,
`local_ms_a100` / `local_ms_4060`, `local_pkg` (the shipped package's own errors, val-contaminated
for all-data members), `proxy` (the **held-out twin** set on the AoA 15–20 slice of `val_unseen`,
scored at the ALPHA that package ships — this is the column the correction is fitted on),
`pred`, `actual`.

`runs.jsonl` — `date`, `round`, `arm`, `direction`, `instrument`, `effect`, `ensemble_tested`,
`verdict` (`SHIPPED` / `OPEN` / `MARGINAL` / `CLOSED` / `REJECTED`), `shipped`, `note`.

**Optional `_zh` fields.** The rendered page (`/users/PAS1187/qwu/workspace/realpde/leaderboard.html`, not committed) is bilingual (English / 中文), so the ledgers carry
optional translations alongside the English: `thesis_zh` in `board.jsonl`, `direction_zh` and
`effect_zh` in `runs.jsonl`. They are **optional** — the renderer falls back to the English
string when the field is absent, so a newly appended row needs nothing extra and the page stays
correct. Keep code names, flags, numbers and file names in Latin inside the Chinese
(`rel_l2`, `tke`, `sps`, `final`, `ALPHA`, `band`, `--dropout`, `wfinal`); translate only the
prose around them. Nothing else in the schema changed.

## The page

The rendered page is a two-column instrument page: a sticky ~220 px left rail carrying the page
name, an **EN / 中文** toggle, one nav anchor per section (read this first · submissions ·
progress at constant latency · local→board correction · experiment ledger) and the generated
stamp, next to a 1200 px content column. Under ~880 px the rail becomes a top bar with a
horizontally scrollable nav.

Both languages are emitted into the DOM and CSS hides one (`html[data-lang]`), so the page is
complete at rest with no JavaScript — English is what a JS-less reader sees. A small inline
script picks the language from `localStorage`, else `navigator.language`, and the toggle writes
both. Authored strings are declared once, side by side, through `lb.py`'s `t(en, zh)`; ledger
strings go through `tdata(en, zh)`, which is the fallback path above. SVG chart labels use
`tsvg()`, which emits two `<text>` nodes at one anchor.

## Where the numbers come from

| quantity | source | last measured |
|---|---|---|
| subscore weights | `Track_1/board_rounds.py` → `Track_1/final_v9.py`, nnls + LP over all board rounds, **pooled across both tracks** | 14 rounds, max residual 0.044 |
| proxy → board ratios | `predict_board.py --anchor` / `Track_1/proxy_calib.py` (needs a GPU; run once per round) | R10–R15 + T2R1…T2R3 |
| SPS element-tail `f` | `Track_1/calib_band13.py`, inverted from a board SPS for a known band | 1.360 (T1) / 1.350 (T2) |
| board latency | inverted from `time_score`; the clean number is the same-day R12/R13 pair | 0.52–0.56 ms per invocation |
| `w_time` | R13 vs R14 — identical error subscores, `time` alone moved | 0.0956 (fit: 0.091) |
