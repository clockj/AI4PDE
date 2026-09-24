# qd_k — deriving the QD band's `k` (see REPRODUCE.md §10.3)
Vendored 2026-09-24 so the organisers can recreate `err_k` without anyone's workspace:
- `head_direct.py` (Fen; nested k per fold, K-permutation gate, `--fixed-k`, `--all-unseen`, `--csv`)
- `centre_band.py` (Bram; compose / fit_base / sample_weights), `sweep_guards.py` (Devi; grid guards)
- `export.py`, `load_export.py` (Emre; the twin export the scorer reads)
Files are verbatim except: absolute workspace paths -> `RPDE_QDK_EXPORT`, `RPDE_QDK_BAND` (defaults = the original paths) and
this directory for the helper imports. Outputs on 09-24: k* = 1.1399 in 18/18 folds, 48/48 leave-one-out runs.

## Reynolds extrapolation check + finals `k` (added 2026-09-24, Fen's qdre; see REPRODUCE.md §10.4)
- `run_qdre_train.sh` — QD twin trained WITHOUT `re_split.excluded()` (26 traj: relo/rehi/rein), 56 traj kept. = Devi's qd_r_s1 + that
  one change (`train_run_vs_devi_qd_r_s1.diff`). `SEED=1` / `SEED=2`.
- `run_qdre_readout.sh` -> `re_readout.py` (Fen, md5 807fac9c = the file job 54611711 ran) — per-trajectory QD (k 1.1399) minus recipe
  band, official / oracle-control / widened-QD CSVs + `qdre_tables_<seed>.npz`.
- `re_gate.py` (Fen) — pre-registered gate: rehi one-sided 95% LB > 0 AND median rehi >= 0.5 x median rein.
- `finals_k.py` (Devi, sha256 3db526c8) — the finals-k rule (cross-select inside rehi, then rein/relo costs); `re_kcross.py` (Fen) same readout.
**For `pitzer submit`, write SEED as a literal in the job copy** (`SEED=2`, not `${SEED:-1}`): the gate passes no environment, so an
env default would silently run seed 1 (Fen). Paths: `RPDE_PY`, `RPDE_DATA` / `RPDE_REAL`, `RPDE_QDRE_WORK` (a directory holding `code/` = a copy of this repo, `wheels/` = the h5py
wheel, `ckpt/` for the readout), `RPDE_QDRE_OUT`; defaults are the original Pitzer job paths. Otherwise verbatim.
Result 09-24 (seeds 1+2, re-run locally by Emre 17:4xZ from Fen's tables, identical): gate PASS (rehi LB +0.185, 7/7);
oracle-control lower bound rehi +0.073 (LB +0.032); finals k: cross-select picks 1.2438 both ways but gains only +0.016 / +0.0003,
relo cost -0.073 vs bar -0.004 -> **finals k = board k = 1.1399**.
