# Internal leaderboard

Three append-only JSONL ledgers plus one script. Together they answer the question that prose
could not: **does a local gain transfer to the board, and by how much?**

```
leaderboard/
├── board.jsonl   one record per SUBMISSION  — package, thesis, PREDICTION, actual, local numbers
├── runs.jsonl    one record per LOCAL EXPERIMENT — arm, instrument, effect, verdict, did it ship
├── screen.jsonl  one record per BACKBONE-SCREEN ARM — written by forge (Track_1/plan/plan_20260914.md §7)
└── lb.py         loads all three, fits the local→board correction, renders the page
```

```bash
python leaderboard/lb.py            # text
python leaderboard/lb.py --html     # also write ../../leaderboard.html — OUTSIDE the git tree (env RPDE_LB_HTML overrides)
```

The HTML is published as the "RealPDE calibration log" artifact. Republish it to the same link
after every board result; the link is in the latest handoff.

## What the ledgers have shown

The durable conclusions live in `../ROADMAP.md` Part 1 and in `../EVALUATION.md` (how local
evaluation works and how the board corrects it). `../CLAUDE.md` no longer carries them; its old
*Scoring* and *Instruments* sections are kept verbatim in `../ROUNDS.md`, Appendix C. The page's
"read this first" box, its leader cell and the rank reads are authored inside `lb.py`, so edit them by
hand when the reading changes. Four conclusions matter when reading the ledgers:

* **Levels calibrate; deltas are a sign, not a size.** Board/proxy level ratios are stable; the
  constants in use are `Track_1/predict_board.py:RATIO` (1.078 / 1.150 / 1.511 for rel / tke / mvpe
  on 2026-09-27; older prose quotes 1.07 / 1.14 / 1.45). Round to round, the proxy's residual is ~4 %
  of the error, it over-reads error cuts about 2×, and across model families it has been wrong in sign
  (U-FNO).
* **Band changes are read in ΔC as well as `final`.** Local proxies over-read changes that tighten the
  band: QD's local +0.11 `final` was ΔC +0.006 and came back ΔC −0.003 (R34). Track 2's online
  corrections went the other way and beat their in-distribution read (T2R24). `../ROADMAP.md` §1.34,
  §1.38.
* **`ensemble_tested` is the most predictive field in `runs.jsonl`.** An arm that was not
  measured at the ensemble size that ships has not been measured.
* **Compare board rows by subscore delta.** The four error subscores are deterministic for a
  given archive; only `time` drifts with the eval machine.

## Appending

**Before a submission**, the five predicted subscores must be written down, each with a bracket:
in `../../sub_candidates/<date>/README.md` (the Decision-phase rule; until 09-20 they lived in
`../../sub_candidates/README.md`, and the final-day registrations are kept verbatim in
`../../sub_candidates/HISTORY.md`), and as a PENDING record here with `actual: null` and `pred`
filled in. A prediction recorded after the fact is worthless; the whole instrument depends on
writing it down first. Copy the prediction into this record no later than when the result is recorded.

**After a board result** — this is the correction step:

1. fill `actual` on that round's record in `board.jsonl` (all five subscores + `final`);
2. set `shipped` on the arms it carried in `runs.jsonl`;
3. paste the same subscores into `Track_1/board_rounds.py` (`ROUNDS` for Track 1, `ROUNDS_T2`
   for Track 2) **and** `Track_1/final_v9.py:BOARD`, which lists rounds in submission-date order.
   That refits the subscore *weights*, which `lb.py` imports through `final_v9`; the fit pools
   both tracks, which share one test set and one rule. At import `final_v9.py` checks that the
   two lists hold the same rounds (as a multiset — row order does not matter) and checks its own
   residuals. A round whose five subscores were not recorded goes to `board_rounds.FINAL_ONLY`
   instead and takes no part in the fit;
4. add the round as an anchor in `Track_1/predict_board.py`; give its `board.jsonl` record a
   `backbone`, and add a `LEADER_READS` row in `lb.py` whenever the board's leaders are read;
5. rerun `lb.py --html` — the ratios, the resolution limit and the prediction-residual history
   all refit — and republish the page.

**After a local experiment**, append to `runs.jsonl`. `ensemble_tested` is not optional: if the
arm was only measured on a single model, say so, and expect the effect to be smaller.

## Fields

`board.jsonl` — `round`, `track`, `id`, `date`, `package`, `backbone` (the package's composition,
e.g. `FNO ×4 + ViT ×2 fp16`; the first word of each ` + ` part is a family), `thesis`, `invocations`,
`local_ms_a100` / `local_ms_4060`, `local_pkg` (the shipped package's own errors, val-contaminated
for all-data members), `proxy` (the **held-out twin** set on the AoA 15–20 slice of `val_unseen`,
scored at the ALPHA that package ships — the column the correction is fitted on), `pred`,
`actual`. Later records also carry `proxy_note`, `pred_note`, `result_note` and `note`
(free text). `pred` holds only the subscores that were registered — a round predicted on `final`
alone has `{"final": …}` — and the brackets are written out in `pred_note`. `id` is the Codabench
submission id, or `null` where it was not reported.

**The rows of 2026-09-21 … 09-27** (R31–R37, T2R18–T2R24) were added on 2026-09-27 from
`Track_1/board_rounds.py`, `../../artifacts/progress_report_20260927.md` and the final-day notes in
`../experiments/sprint_20260927/`. The subscores of R35–R37 and T2R22–T2R24 are complete. The eight
rounds of 09-21 … 09-24 were uploaded from the agent team's server, and only their `final` was carried
back: their `actual` is `{"final": …}` and `result_note` begins "FINAL ONLY". T2R19, a byte-for-byte
resubmission of 09-22, has no recorded result: `actual: null` with a note. Registered predictions exist
only where a source keeps them: in full, with brackets, for R37 and T2R24; `final` alone for R34 and
T2R21; brackets or partial registrations in `pred_note` for R33, R35, T2R22, R36 and T2R23. **`lb.py`
as of 2026-09-27 cannot render a final-only `actual`**: the progress section and the per-subscore
table index `actual["time"]` and every subscore unconditionally. Three guarded lines fix it.

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
toggle, one nav anchor per section (read this first · optimization curve · backbone screen ·
submissions · progress at constant latency · local→board correction · experiment ledger) and the
generated stamp, next to a 1200 px content column. Under ~880 px the rail becomes a top bar with a
horizontally scrollable nav.

* **Optimization curve.** Board `final` of every scored round against its upload date, Track 1
  over Track 2 on one shared scale, with each track's running best as a step line and every
  dated leader read (`LEADER_READS`) as a diamond. Every point is labelled with its round and
  its `backbone`; a split marker is a package with two families.
* **Backbone screen.** Reads `screen.jsonl` and renders whatever is there, including nothing: a
  latency–accuracy scatter (x `ms_ratio_fno`, else `ms_b64_fp16`; y the better of
  `proj_dfinal_t1_standalone` and `proj_dfinal_t1_partner`; the marker's fill is the verdict)
  and a table of every arm. The numbers are local proxy projections, a sign and not a size.
* **Family colours** come from one fixed slot order (`lb.FAMILIES`): FNO, ViT and U-Net hold
  the three hues that stay distinct all-pairs in both themes; later families are also drawn as
  squares. Every point is direct-labelled, and every value is in a table.
* **Idempotent.** Nothing on the page depends on the clock, so `lb.py --html` rewrites the same
  bytes until a ledger changes; rerun it whenever `screen.jsonl` does.

Both languages are emitted into the DOM and CSS hides one (`html[data-lang]`), so the page is
complete without JavaScript, in English. A small inline script picks the language from
`localStorage`, else `navigator.language`, and the toggle writes both. Authored strings are
declared once, side by side, through `lb.py`'s `t(en, zh)`; ledger strings go through
`tdata(en, zh)`, which is the fallback path above. SVG chart labels use `tsvg()`, which emits two
`<text>` nodes at one anchor.

## Where the numbers come from

| quantity | source | last measured |
|---|---|---|
| subscore weights | `Track_1/board_rounds.py` → `Track_1/final_v9.py`: free linear fit (no Σw = 1) over every board round with five subscores, **pooled across both tracks**. **Print `final_v9.W0`; never copy weights from prose.** Two fits are quoted in the docs: **42 rounds to 2026-09-20** (0.4778 / 0.1010 / 0.0921 / 0.0965 / 0.2498 as rounded in `CLAUDE.md` then) and **48 rounds at the close, 2026-09-27** (0.4819 / 0.0981 / 0.0911 / 0.0961 / 0.2491, Σw 1.0164) | **48 rounds** (2026-09-27), max residual 0.0058 (R6, recorded to three decimals); the three adaptive Track-2 rows miss by 0.0026–0.0049, so the rule is not exactly linear and the gate is 0.0060. The 42-round fit reads the six closing rounds high by +0.005 … +0.013 |
| `w_time`, read directly | pairs of rounds whose four error subscores are identical, so only `time` moved (first R13 → R14, largest R20 → R23) | 0.0956–0.0962 over five pairs (fit 0.0965 at 42 rounds, 0.0961 at 48) |
| proxy → board ratios | `Track_1/predict_board.py --anchor` / `Track_1/proxy_calib.py` (GPU; once per round) | rel 1.068 · tke 1.136 · mvpe 1.454; the constant in use, `predict_board.RATIO`, is 1.078 / 1.150 / 1.511 |
| SPS element tail `f` | `Track_1/calib_band13.py`, inverted from a board SPS for a known band | 1.360 (T1) / 1.350 (T2) at round 13; re-inverted by the agent team on 09-24: f₀ ≈ 1.29 for the static band, f_QD ≈ 1.35 for the QD band (`../../artifacts/progress_report_20260927.md` §0.4) |
| board latency | `board_rounds.py` `INV_MS_*` / `LOC_SLOPE_*`, `predict_board.py` `LAT_*` | invocation 1.151 / 0.866 ms (R23); FLOPs 1.75 / 0.60 board ms per 4060-eq local ms (R24). Since 09-25: an uncontrolled intercept of ~2.6 ms, Track-1 calls of 14–99 windows, and same-path drift of −0.86 ms on Track 2 in one day (`../ROADMAP.md` §1.36) — price latency with a bracket, never a level |
