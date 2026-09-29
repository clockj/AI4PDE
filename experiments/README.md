# Experiment campaigns

Campaigns that span both tracks are kept here, one directory per campaign, each with an index of what was asked, how
it was run and what came back. Code that shipped lives in the main tree; code that did not is kept as a patch.

| directory | campaign | outcome |
|---|---|---|
| `sprint_20260927/` | the final four hours of Main Development: two parallel sessions, 11 designs | the two packages of record (Track 1 81.671149, Track 2 82.055553) |
| `audit_20260927/` | four read-only audits of the repository after the final packages (`fdac00b`), and the refactors that followed | guards and fewer knobs in the shipped code, outputs identical to the uploads at tolerance 0; the documents of the Decision phase |
| `decision_20260928/` | the Decision-phase campaign: 300 GPU-h granted (214.5 used), ensembles trained without AoA 20, AoA 10 or a band of Re, read on the real package path by the instrument in `ood/`; criteria in `PLAN.md`, tables and decisions in `STATE.md`, notes in `results/` | the Re-similarity augmentation (`--aug-resim 0.25`) passes in every regime; the candidate "T4 all" (owner, 09-29), built by `reproduce.sh decision`, not uploaded; `STATS_MASK`, FAMW, zero propagation and six other arms closed |

Single-track experiments stay with their track: `Track_1/closed/`, `Track_2/closed/` (closed questions, with verdicts),
`Track_1/archive/` (the first two months), `Track_1/plan/` and `Track_1/report/` (dated plans and write-ups).
