# qd_k — deriving the QD band's `k` (see REPRODUCE.md §10.3)
Vendored 2026-09-24 so the organisers can recreate `err_k` without anyone's workspace:
- `head_direct.py` (Fen; nested k per fold, K-permutation gate, `--fixed-k`, `--all-unseen`, `--csv`)
- `centre_band.py` (Bram; compose / fit_base / sample_weights), `sweep_guards.py` (Devi; grid guards)
- `export.py`, `load_export.py` (Emre; the twin export the scorer reads)
Files are verbatim except: absolute workspace paths -> `RPDE_QDK_EXPORT`, `RPDE_QDK_BAND` (defaults = the original paths) and
this directory for the helper imports. Outputs on 09-24: k* = 1.1399 in 18/18 folds, 48/48 leave-one-out runs.
