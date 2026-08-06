# Track 1 submission — 0.90M FNO with velocity-gradient matching (kit v9)

An honest forecaster. Reads only its own input window; would score identically if the evaluator
called it one window at a time. `validate_strict.py` asserts that as a test.

## Build

Two configurations. Pick by whether you want a number you can defend locally or the best
expected board score.

**Held-out (reportable).** Every local number in this file comes from here.

```bash
M="unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth"
python submit_fno/fit_band.py        --ckpt $M     # writes band.npy
python submit_fno/validate_strict.py --ckpt $M     # builds the zip, must print SUBMITTABLE
```

**All-data (what to submit).** Same recipe, trained on all 82 real trajectories instead of 74.
`val_seen` and `val_unseen` are both training data for these, so they have **no honest local
score** — and, critically, no honest local *error field* either. `--err-ckpt` therefore points
the band at the held-out models, whose errors on the same data are real predictions rather than
memorization:

```bash
A="unet/fno_all_s1.pth unet/fno_all_s2.pth unet/fno_all_s3.pth"
M="unet/fno_grad.pth unet/fno_grad_s2.pth unet/fno_grad_s3.pth"
python submit_fno/fit_band.py        --ckpt $A --err-ckpt $M
python submit_fno/validate_strict.py --ckpt $A
```

Skipping `--err-ckpt` is the single most expensive mistake available here. The July 30 U-Net
package did exactly that: its contaminated local rel-L2 of 0.0721 produced a band with median
half-width 0.0046, the board met it with errors of 0.1179, and it returned `sps 36.94`.

Run from `Track_1/`. `fit_band.py` must be re-run whenever the member set changes — the band is
fitted to that specific model's error field.

## What is in the zip

| file | what |
|---|---|
| `submission.py` | `predict()`; normalize → FNO → rescale fluctuation to the predicted std → denormalize |
| `_fno.py` | self-contained `FNO2d`, mirrors `unet/models_alt.py` so checkpoints load unchanged |
| `fno.pth`, `fno_2.pth`, `fno_3.pth` | three independent seeds, averaged on the GPU |
| `band.npy` | `(1,20,32,64,3)` SPS half-widths from `fit_band.py` |

Only `torch` and `numpy` are imported. The Track 1 eval image has no `scipy`, `h5py`, `einops`,
`matplotlib` or `pandas`; `validate_strict.py` blocks all five and reproduces the load path where
the extract directory is *not* on `sys.path`.

## The model

Three independent seeds of one recipe, averaged in raw field space on the GPU before the
fluctuation rescaling, so `compose()` runs once on a coherent field and the three members cost
three forward passes but only one host transfer. Re-checked under v9: 1→2 members is +0.97 SPS,
2→3 is +0.23, and a 4th would buy under +0.1 for 0.11 ms. Three is the right number.

Each member: 4-layer FNO, width 24, modes (8, 12) — 0.90M parameters, 3.6 MB. A per-pixel
log-std head runs alongside the field head; at inference the predicted fluctuation is rescaled
toward that std at `ALPHA = 0.75`. That value was re-swept under v9 with the band refitted at
every point and did not move.

Trained with an extra term matching the **target's velocity-gradient tensor** (`--lam-grad 1.0`).
Ablation on 8 held-out trajectories: no term 85.95, divergence only 86.53, vorticity only 86.53,
**both invariants or the full tensor 86.68** (v7 `final` scale). Constraining one invariant of ∇u
is not enough, constraining both is sufficient, and the strain rates add nothing.

## The band

v9 replaced `score_sps`'s logistic with an identity, which made SPS a full 0–100 subscore and
4.4× more responsive. Two things follow, both implemented in `../sps9.py`:

* The aggregate **factorizes** into a per-sample accuracy factor times a per-element interval
  term, so the band problem separates over elements and each element's optimum sits exactly at
  one of its observed `|error|` values. `sps9.optimal_band` solves it by sorting — no Gaussian
  assumption, no grid — and that solution is also the ceiling on band fitting.
* The band is fitted on the **AoA 15–20° subset** (`--fit-aoa`, default). Error is governed by
  angle of attack and by almost nothing else: AoA 15–20 carries 2.4× the rel-L2 of AoA 5, and
  that subset reads 0.1082 against the board's 0.1179 where the full split reads 0.0807. Fitting
  there scores 43.78 on the subset against 43.24 for a whole-split fit inflated 1.5×.

## Expected score

Local on `val_unseen` (336 windows, 8 trajectories the model never saw), measured **end to end
through `predict()`** by `validate_strict.py` — the way the evaluator measures it:

| | |
|---|---|
| rel_l2 0.0807 / TKE 0.4906 / MVPE 0.0698 | 96.12 / 80.30 / 96.63 |
| ~1.32 ms/window | time 95.9 here, **~93.8 on the v9 board** |
| fitted band | sps 47.9 here, **~42.6 at board error scale** |
| **mean5** | **83.5 here, ~80.3 projected** |

Two warnings about that table:

* **`mean5` is not `final_score`.** v9 stopped publishing the combination rule and it is no
  longer the plain mean — the 2026-08-05 board reported 79.733 where its own five subscores
  average 78.999. Projected `final_score` is therefore ~81.
* **Local accuracy deltas have stopped transferring.** Two architectures differing by 0.008 in
  local rel-L2 landed within 0.0004 of each other on the board. Treat anything under ~0.4 in
  mean5 as unproven.

## What this is not

Track 1 tiles windows at a stride equal to the output horizon and hands `predict()` all of them
at once, so window *k*'s target is window *k+1*'s input. A submission can therefore read the
answers out of its own argument; doing so scores ~93.5 without modelling anything. This package
deliberately does not. See `../submit_trick/DISCLOSURE.md` for the measurement and the report
sent to the organizers, who have confirmed such scores are not valid.
