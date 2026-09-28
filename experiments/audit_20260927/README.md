# Code audit at the close of Main Development, 2026-09-27

Four read-only audits of the repository as it stood after the final packages were committed (`fdac00b`), and the
refactors that followed. Each audit is kept as written; this file says what was done about each finding.

| audit | scope | file |
|---|---|---|
| shipping path | `Track_1/submit_fno/`, `Track_1/sigblend/`, `Track_2/`: every knob, the live path of both packages, defects | `audit_shipping.md` |
| training path | `Track_1/unet/`, `reproduce.sh`, `REPRODUCE.md`: the dependency closure of the final recipe, dead weight, reproducibility, the checkpoint inventory | `audit_training.md` |
| layout | every tracked file classified; the import graph; instrument defaults; broken references | `audit_layout.md` |
| documents | stale and contradictory statements; the triage of `CLAUDE.md` | `audit_docs.md` |

Refactor reports, each with its verification: `refactor_instr.md` (instruments), `refactor_pkg.md` (the shipped code),
`refactor_models.md` (the model module), `refactor_docs.md` (ledger and history).

## Findings and what was done

### Reproducibility

| finding | disposition |
|---|---|
| The final code, `build_final.sh`, `sigblend/` and the two final bands were uncommitted | committed in `fdac00b`; both packages rebuilt from a clean worktree of that commit, all 9 files of each archive byte-identical to the uploads (job 7569179) |
| `REPRODUCE.md` said `FINAL_RECIPE = TO BE DECIDED` | filled; §13 specifies the band, the constants and the stream stack |
| `reproduce.sh` had no target for the final packages | `final` is the default target; `RPDE_REPRO_FITBAND=1` re-derives the band. Both paths run end to end from the existing checkpoints (job 7569482); the refitted static band is bit-equal to the committed one |
| `sigblend/dump_twins.py` and `make_k0_band.py` hard-coded scratch and home paths | arguments and environment variables; no scratch defaults |
| `build_final.sh` ignored its gates' results | exits non-zero unless Track 1 and Track 2 are SUBMITTABLE and the causality check passes; checks that the validation files exist |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` set nowhere; `PY` not exported | both exported by `reproduce.sh` and `build_final.sh` |
| The SIGBLEND band had never been re-derived | re-derived from a fresh twin dump (job 7569179): q within 0.0008, `base′` within 0.4 %, the leave-one-out table reproduced to the printed digit |
| The sim-map cache key omits the data identity | documented (`.claude/rules/training.md`); use one cache directory per data release |
| Retraining is not bit-identical | known and accepted: acceptance is by score (`REPRODUCE.md` §7, §12) |

### The shipped code

| finding | disposition |
|---|---|
| M1. The official Track-2 runner normalises the revealed target with the TARGET statistics; the package assumed the input statistics, so its unscored-element test never fires on the board | knob `PREV_SPACE` (default `"input"` = the uploaded behaviour) and harness option `--prev-space` (default `target` = the official runner). Measured on the twins: the fix is worth +0.0064 composite on all 8 trajectories (8 of 8 positive) and +0.0027 on AoA 15–20, all in `sps`; it reproduces our earlier local reads exactly |
| M2. D1c's pooled statistics and VARCAL's sums include masked pixels | knob `STATS_MASK` (default off). Null on the twins (−0.0012, 3 of 8 positive); our data cannot test a different blanking pattern. Open in `STATUS.md` |
| M3. No guard against non-finite values; one NaN zeroes every subscore | guards on both tracks, bit-identical on finite inputs. A NaN window now zeroes only itself; a NaN revealed target behaves as a withheld one. The audit had missed that one NaN *input* poisoned the rest of a Track-2 trajectory; guarded too. Cost: Track 1 +0.4 %, Track 2 +0.08 ms per step (+2.4 %) |
| M4. Track 1 raised on every call when CUDA was unavailable | the host band path applies the σ-power factor; bounds within 7.5e-9 of the GPU path, a real CPU run within 1.8e-6 of an fp32 GPU run |
| m1, m2. Indexed device; pinned staging buffer without a fallback | fixed |
| m3, m5, m7, m8, m9. Graph tolerance, batch-size probe, environment reads, `FFNO_IMPL`, automatic reset | left as uploaded: each would change behaviour on the evaluator, and none is a defect in the scored configuration |
| Dead code in both `submission.py` | removed: the Track-1 CUDA graph and its small-batch variant, the batch and GPU fingerprints, `LN_MANUAL`, the `band.npy` fallback; on Track 2 `SIGEMA`, `STACKED_OUT`, `FALLBACK_FP`, `R1_SNAP`. Track 1 `submission.py` 920 → 691 lines. `build_round13.sh` refuses their `RPDE_SHIP_*` variables and names `fdac00b` as the last tree that has them. Kept because the private set may want them: `BLEND_BETA`, `HIST_BETA`, `FW_N0`; kept for `reproduce.sh` recipes: the error-head band, symmetric R1, the per-channel compose. **Outputs bit-identical to the uploads** at N = 14, 56, 99, 336 on Track 1 and over a full stream on Track 2 (job 7569409) |
| `_fno.py` carries about 1,450 lines of families that do not ship | left: about 25 repository tools and `reproduce.sh r25` import it |

### The training code

| finding | disposition |
|---|---|
| `models_alt.py` mixes the three live families with about 1,000 lines of closed ones | 31 names moved verbatim to `models_closed.py` and re-exported through `models_alt` (1955 → 1041 lines). Bitwise identical: seeded construction of 25 configurations, forward in eval and train mode, 19 checkpoints loaded both ways, a 2-epoch ViT training run. F-FNO training on the GPU is not deterministic even for the original code (two runs differ by 2e-5); under PyTorch's deterministic mode original and refactor are bitwise equal |
| About 700 lines of `train2.py` are closed arms | left: the organizers retrain from this file, and removing a guarded branch changes nothing they would run while risking the order of random draws |
| Tracked training logs, queue files, the first trainer | moved to `Track_1/archive/unet_r9_r12/` |
| 334 checkpoints (18.6 GB) in `Track_1/unet/`, of which 25 are read by a `reproduce.sh` target | inventory in `audit_training.md` §4. **On 09-28 the owner had classes E and F deleted: 263 files, 15.85 GB.** 71 remain (2.8 GB): the 11 final, the 14 of the superseded `reproduce.sh` recipes, 44 named by tracked code, 2 named by `board.jsonl`. Name, size, md5 and the full `train2.py` argument set of every deleted file: `Track_1/archive/checkpoints_deleted_20260928.json` |

### Layout and instruments

| finding | disposition |
|---|---|
| Historical packages and sweeps at the top level of `Track_1/` | `Track_1/archive/`, indexed |
| Dead top-level scripts | `Track_1/closed/`, with the path shim |
| No instrument could score the band that shipped; defaults at four different older operating points | `Track_1/opoint.py`, `Track_1/band_io.py`; every instrument defaults to the packages of record; the old defaults are tabled in `EVALUATION.md` §4 |
| Broken references in the documents | fixed where the target exists; files that stayed on the agent team's server are listed in `REPRODUCE.md` §13.6 |

### Documents

| finding | disposition |
|---|---|
| `CLAUDE.md` was 400 lines and a third of it was state that had changed | 146 lines of rules and stable facts; state in `STATUS.md`; six path-scoped rules in `.claude/rules/`; five procedures in `.claude/skills/`; the removed sections are kept verbatim in `ROUNDS.md` |
| "Ship frozen", "ALPHA 0.60", "the band is closed", "40 rounds" | corrected in `README.md`, `EVALUATION.md`, `Track_1/README.md`, `Track_2/README.md`, `ROADMAP.md` Part 2 |
| The ledger stopped at 09-20 | `board_rounds.py`, `final_v9.py`, `board.jsonl` carry every round to 09-27; the `final` rule refitted on 48 rounds |
| Two generic skills contradicted the project's rules | removed (`rockfish-workflow`, `debug-workflow`) |
