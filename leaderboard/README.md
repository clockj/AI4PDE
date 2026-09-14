# Internal leaderboard

Two append-only JSONL ledgers plus one script. Together they answer the question that prose
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

The HTML is published as the "RealPDE calibration log" artifact. Republish it to the same link
after every board result; the link is in the latest handoff.

## What the ledgers have shown

The durable conclusions live in `../CLAUDE.md` (*Scoring*, *Instruments*) and `../ROADMAP.md`
Part 1. The page's "read this first" box and its leader cell are authored inside `lb.py`, so edit
them by hand when the reading changes. Three conclusions matter when reading the ledgers:

* **Levels calibrate; deltas are a sign, not a size.** Board/proxy level ratios are stable
  (rel ≈ 1.07, tke ≈ 1.14, mvpe ≈ 1.45). Round to round, the proxy's residual is ~4 % of the
  error, it over-reads error cuts about 2×, and across model families it has been wrong in sign
  (U-FNO).
* **`ensemble_tested` is the most predictive field in `runs.jsonl`.** An arm that was not
  measured at the ensemble size that ships has not been measured.
* **Compare board rows by subscore delta.** The four error subscores are deterministic for a
  given archive; only `time` drifts with the eval machine.

## Appending

**Before a submission**, the five predicted subscores must be written down: in
`../../sub_candidates/README.md` at the latest, which is where they have lived since round 15,
or directly as a record here with `actual: null` and `pred` filled in. A prediction recorded
after the fact is worthless; the whole instrument depends on writing it down first. Copy the
prediction into this record no later than when the result is recorded.

**After a board result** — this is the correction step:

1. fill `actual` on that round's record in `board.jsonl` (all five subscores + `final`);
2. set `shipped` on the arms it carried in `runs.jsonl`;
3. paste the same subscores into `Track_1/board_rounds.py` (`ROUNDS` for Track 1, `ROUNDS_T2`
   for Track 2) **and** `Track_1/final_v9.py:BOARD`, which lists rounds in submission-date order.
   That refits the subscore *weights*, which `lb.py` imports through `final_v9`; the fit pools
   both tracks, which share one test set and one rule. At import `final_v9.py` checks that the
   two lists hold the same rounds (as a multiset — row order does not matter) and checks its own
   residuals;
4. add the round as an anchor in `Track_1/predict_board.py`;
5. rerun `lb.py --html` — the ratios, the resolution limit and the prediction-residual history
   all refit — and republish the page.

**After a local experiment**, append to `runs.jsonl`. `ensemble_tested` is not optional: if the
arm was only measured on a single model, say so, and expect the effect to be smaller.

## Fields

`board.jsonl` — `round`, `track`, `id`, `date`, `package`, `thesis`, `invocations`,
`local_ms_a100` / `local_ms_4060`, `local_pkg` (the shipped package's own errors, val-contaminated
for all-data members), `proxy` (the **held-out twin** set on the AoA 15–20 slice of `val_unseen`,
scored at the ALPHA that package ships — the column the correction is fitted on), `pred`,
`actual`.

`runs.jsonl` — `date`, `round`, `arm`, `direction`, `instrument`, `effect`, `ensemble_tested`,
`verdict` (`SHIPPED` / `OPEN` / `MARGINAL` / `CLOSED` / `REJECTED`), `shipped`, `note`.

**Optional `_zh` fields.** The rendered page is bilingual (English / 中文), so the ledgers carry
optional translations alongside the English: `thesis_zh` in `board.jsonl`, `arm_zh`,
`direction_zh` and `effect_zh` in `runs.jsonl`. The renderer falls back to the English string when
a field is absent, so a newly appended row needs nothing extra. Keep code names, flags, numbers
and file names in Latin inside the Chinese (`rel_l2`, `tke`, `sps`, `final`, `ALPHA`, `band`,
`--dropout`, `wfinal`); translate only the prose around them.

## The page

A two-column instrument page: a sticky ~220 px left rail with the page name, an **EN / 中文**
toggle, one nav anchor per section (read this first · submissions · progress at constant latency ·
local→board correction · experiment ledger) and the generated stamp, next to a 1200 px content
column. Under ~880 px the rail becomes a top bar with a horizontally scrollable nav.

Both languages are emitted into the DOM and CSS hides one (`html[data-lang]`), so the page is
complete without JavaScript, in English. A small inline script picks the language from
`localStorage`, else `navigator.language`, and the toggle writes both. Authored strings are
declared once, side by side, through `lb.py`'s `t(en, zh)`; ledger strings go through
`tdata(en, zh)`, which is the fallback path above. SVG chart labels use `tsvg()`, which emits two
`<text>` nodes at one anchor.

## Where the numbers come from

| quantity | source | last measured |
|---|---|---|
| subscore weights | `Track_1/board_rounds.py` → `Track_1/final_v9.py`: free linear fit (no Σw = 1) over every board round, **pooled across both tracks** | 30 rounds, max residual 0.0025 |
| `w_time`, read directly | pairs of rounds whose four error subscores are identical, so only `time` moved (first R13 → R14, largest R20 → R23) | 0.0956–0.0962 over five pairs (fit 0.0965) |
| proxy → board ratios | `Track_1/predict_board.py --anchor` / `Track_1/proxy_calib.py` (GPU; once per round) | rel 1.068 · tke 1.136 · mvpe 1.454 |
| SPS element tail `f` | `Track_1/calib_band13.py`, inverted from a board SPS for a known band | 1.360 (T1) / 1.350 (T2) |
| board latency | `board_rounds.py` `INV_MS_*` / `LOC_SLOPE_*`, `predict_board.py` `LAT_*` | invocation 1.151 / 0.866 ms (R23); FLOPs 1.75 / 0.60 board ms per 4060-eq local ms (R24) |
