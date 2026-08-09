# Track 1 submission — 7-member FNO ensemble, 3 fused invocations (kit v9, round 10)

An honest forecaster. Reads only its own input window; would score identically if the evaluator
called it one window at a time. `validate_strict.py` asserts that as a test.

Round 10 changed three things: `--var-mlp --lift-mlp` in the recipe (+0.112), `_fno.fuse()`
collapsing same-architecture members into one invocation each (+0.067), and a sigma-adaptive
smoothed SPS band (+0.132). See `../report/report_20260808_round10.md`.

## Build

Two configurations. Pick by whether you want a number you can defend locally or the best
expected board score.

**Held-out (reportable).** Every local number in this file comes from here.

```bash
M="unet/fno_vmlp_lift.pth unet/fno_vmlp_lift_s2.pth unet/fno_vl_s3.pth unet/fno_vl_s4.pth \
   unet/fno_vl_m1220.pth unet/fno_vl_w64.pth"
python submit_fno/fit_band.py        --ckpt $M     # writes band.npz
python submit_fno/validate_strict.py --ckpt $M     # builds the zip, must print SUBMITTABLE
```

**All-data, 7 members (what to submit).** Same recipe, trained on all 82 real trajectories
instead of 74. `val_seen` and `val_unseen` are both training data for these, so they have **no
honest local score** — and, critically, no honest local *error field* either. `--err-ckpt`
therefore points the band at the held-out models, whose errors on the same data are real
predictions rather than memorization:

```bash
A="unet/fno_all_vl_s1.pth unet/fno_all_vl_s2.pth unet/fno_all_vl_s3.pth \
   unet/fno_all_vl_s4.pth unet/fno_all_vl_s5.pth \
   unet/fno_all_vl_m1220.pth unet/fno_all_vl_w64.pth"
M="unet/fno_vmlp_lift.pth unet/fno_vmlp_lift_s2.pth unet/fno_vl_s3.pth unet/fno_vl_s4.pth \
   unet/fno_vl_m1220.pth unet/fno_vl_w64.pth"
python submit_fno/fit_band.py        --ckpt $A --err-ckpt $M
python submit_fno/validate_strict.py --ckpt $A
```

**`w48` is no longer a member.** Under fusion it earns back exactly its own invocation group
and nothing more (78.667 with it against 78.668 without), and on the round-10 recipe it is 0.6
behind w24 individually (81.451 against 82.059) where in round 8 it was only 0.3 behind.

`--ckpt` and `--err-ckpt` both matter to `fit_band.py` now, and for different reasons: the
error field comes from `--err-ckpt`, but `sigma_ref` — the normalizer for the per-window band —
comes from `--ckpt`, because it has to be the scale of the models that will emit sigma at test
time. Measured, the two agree to 0.1% (all-data mean sigma 0.00891 against held-out 0.00890 on
val_seen), so all-data training does not deflate the variance head; the check is cheap and
worth repeating if the recipe changes.

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
| `fno.pth` … `fno_8.pth` | eight members of three architectures, averaged on the GPU |
| `band.npy` | `(1,20,32,64,3)` SPS half-widths from `fit_band.py` |

Only `torch` and `numpy` are imported. The Track 1 eval image has no `scipy`, `h5py`, `einops`,
`matplotlib` or `pandas`; `validate_strict.py` blocks all five and reproduces the load path where
the extract directory is *not* on `sys.path`.

## The model

Eight members, averaged in raw field space on the GPU before the fluctuation rescaling, so
`compose()` runs once on a coherent field and eight members cost eight forward passes but only
one host transfer.

**Member count is no longer a trade.** Two board rounds returned `time_score` 91.697 and 91.726
for packages 2.07× apart in local latency, so board time is harness overhead and does not
respond to the model. The measured slope is 0.033 ms of board time per ms of local time: this
package runs 2.211 ms/window locally and is predicted at `time_score` 91.707, nineteen
thousandths below the 1.282 ms three-member build.

**The members are deliberately not all the same architecture.** Five are width-24 seeds; one is
width 48, one modes (12, 20), one width 64. Individually the bigger ones are *worse* — on the
AoA 15–20 proxy, width 24 scores 81.67 against 81.54 / 81.33 / 81.17 — because with 74
trajectories and a gradient loss, capacity buys memorization. But they fail differently, and in
an ensemble that is what counts. Held-out member curve: 3×w24 74.418 → +m1220 74.490 → +w48
74.563 → +w64 74.614 → 7 members 74.645 → 9 members 74.657. Each new *architecture* is worth
+0.05…0.07; each extra *same-width seed* only +0.01…0.03.

Each width-24 member: 4-layer FNO, modes (8, 12) — 0.90M parameters, 3.6 MB. A per-pixel
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
| 2.211 ms/window | time 94.8 here, **91.71 on the v9 board** (constant) |
| fitted band | sps 47.9 here, **~46 at board error scale** |
| **mean5** | **83.5 here, ~80.3 projected** |

Two warnings about that table:

* **`mean5` is not `final_score`.** v9 stopped publishing the combination rule and it is no
  longer the plain mean. Across two board rounds `final − mean5` was +0.727 and +0.743, so the
  working rule is `final ≈ mean5 + 0.73`. Projected `final_score` ~80.3.
* **The projection has been wrong before.** The same proxy predicted 81 last round and the
  board returned 79.985. Direction, not number.
* **Local accuracy deltas have stopped transferring.** Two architectures differing by 0.008 in
  local rel-L2 landed within 0.0004 of each other on the board. Treat anything under ~0.4 in
  mean5 as unproven.

## What this is not

Track 1 tiles windows at a stride equal to the output horizon and hands `predict()` all of them
at once, so window *k*'s target is window *k+1*'s input. A submission can therefore read the
answers out of its own argument; doing so scores ~93.5 without modelling anything. This package
deliberately does not. See `../submit_trick/DISCLOSURE.md` for the measurement and the report
sent to the organizers, who have confirmed such scores are not valid.
