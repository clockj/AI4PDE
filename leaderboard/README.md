# Internal leaderboard

Two append-only JSONL ledgers plus one script. It answers the question thirteen rounds of prose
could not: **does a local gain transfer to the board, and by how much?**

```
leaderboard/
├── board.jsonl   one record per SUBMISSION  — package, thesis, PREDICTION, actual, local numbers
├── runs.jsonl    one record per LOCAL EXPERIMENT — arm, instrument, effect, verdict, did it ship
├── lb.py         loads both, fits the local→board correction, renders index.html
└── index.html    generated; open it, or publish it
```

```bash
python leaderboard/lb.py            # text
python leaderboard/lb.py --html     # also regenerate index.html
```

## What it currently says

* **At constant board latency the last four rounds span 0.056 points.** The model has not
  measurably improved since 2026-08-08.
* **Levels calibrate, deltas do not.** The proxy→board ratios are stable to ±1–3 % (`rel` 1.077,
  `tke` 1.146, `mvpe` 1.498), so the proxy predicts the board's *level* well. Round to round it
  explains almost none of the board's *movement*: **residual rms 3.4 % of the error**, range
  −5.5 to +5.9 %. `mvpe` even inverts sign.
* **⇒ The resolution limit is ~3 % of error.** A local delta below that does not survive the
  trip, and every improvement measured since 2026-08-08 has been smaller than it.
* **`ens` is the most predictive column in `runs.jsonl`.** `ozdo` was +3.84 σ per model and
  −0.012 at four members; `--var-mlp`'s 3.9 % fixed-field TKE cut became +0.08. An arm that has
  not been measured at the ensemble size that ships has not been measured.

## Appending

**After a board result** — this is the correction step:

1. fill `actual` on that round's record in `board.jsonl` (all five subscores + `final`);
2. set `shipped` on the arms it carried in `runs.jsonl`;
3. rerun `lb.py`. The ratios, the resolution limit and the prediction-residual history all refit.
4. paste the same subscores into `Track_1/board_rounds.py:ROUNDS` — that is what refits the
   subscore *weights*, which `lb.py` imports through `Track_1/final_v9.py`.

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

## Where the numbers come from

| quantity | source | last measured |
|---|---|---|
| subscore weights | `Track_1/board_rounds.py` → `Track_1/final_v9.py`, nnls + LP over all board rounds | 9 rounds, max residual 0.035 |
| proxy → board ratios | `Track_1/proxy_calib.py` (needs a GPU; run once per round) | R10–R13 |
| SPS element-tail `f` | `Track_1/calib_band13.py`, inverted from a board SPS for a known band | 1.360 (T1) / 1.350 (T2) |
| board latency | inverted from `time_score`; the clean number is the same-day R12/R13 pair | 0.52–0.56 ms per invocation |
