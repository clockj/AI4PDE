# Track 1 submission — 11-member mixed-family ensemble, 4 fused invocations (kit v9, round 12)

> **STALE BELOW THIS BOX: everything under it documents the round-12 package.**
>
> **Current state (2026-09-13).** The last upload is **R25 = xfam6fp16**:
> * four all-data w96 L8 cmb3sim FNOs, fused into one `StackedFNO`;
> * R21's ViT plus one cmb3sim ViT, fused into one `StackedViT` — 2 invocations in all;
> * int8 packing (`--linear tensor`), band `Track_1/bands/r24_xfam6.npz`, ALPHA 0.60,
>   `AMP = "fp16"`.
>
> **Build:** `bash Track_1/submit_fno/build_round13.sh ALPHA`. `RPDE_A` names the shipped
> members and `RPDE_M` the held-out twins the band is fitted on. `RPDE_BAND` reuses a band byte
> for byte instead of refitting. The remaining switches are `RPDE_INT8=1`,
> `RPDE_LINEAR=tensor|row|fp16`, `RPDE_SHIP_AMP='"fp16"'`, `RPDE_SPEC_PATH=bmm`,
> `RPDE_HOST_WRITE=torch` and `RPDE_TAG=...`. Add `RPDE_FORCE=1` when other jobs hold GPUs;
> that is safe for any build whose timing is not being recorded.
>
> **Constants that ship are BAKED into `submission.py`:** `ALPHA`, `FUSE`, `AMP`, `SPEC_PATH`,
> `HOST_WRITE`, `BLEND_BETA`. Environment switches (`RPDE_AMP`, `RPDE_TF32`, ...) steer
> measurement drivers only; the board never sets them.
>
> **Gates:**
> * `validate_strict.py` runs inside the build;
> * `verify/pkg_identity.py` answers "is this the archive on the board?";
> * `verify/fuse_vit.py`, `verify/amp_path.py`, and `verify/cpu_smoke_submission.py` (GPU-free).
>
> History: `CLAUDE.md`, rounds 18–25.

An honest forecaster. Reads only its own input window; it would score identically if the
evaluator called it one window at a time, and `validate_strict.py` asserts that as a test.

Round 12 changed two things against the round-11 build: a **U-Net member joins the FNO
ensemble** (+0.091 proxy / +0.123 full wfinal — 4.5× the value of another FNO seed), and two
more width-24 seeds were added because fusion makes same-architecture members free in
invocation count. See `../report/report_20260829_round12.md`.

## Build

Two configurations. Pick by whether you want a number you can defend locally or the best
expected board score.

**Held-out (reportable).** Every honest local number in this file comes from here.

```bash
M="unet/fno_vl_s1.pth unet/fno_vl_s2.pth unet/fno_vl_s3.pth unet/fno_vl_s4.pth \
   unet/fno_vl_m1220.pth unet/fno_vl_w48.pth unet/fno_vl_w64.pth unet/unet_g1.pth"
python submit_fno/fit_band.py        --ckpt $M --alpha 0.90 --band-p 0.45
python submit_fno/validate_strict.py --ckpt $M
```

**All-data, 11 members (what ships).** Same recipes, trained on all 82 real trajectories
instead of 74. `val_seen` and `val_unseen` are both training data for these, so they have **no
honest local score** — and, critically, no honest local *error field* either. `--err-ckpt`
therefore points the band at the held-out twins, whose errors on the same data are real
predictions rather than memorization. One command does the whole build:

```bash
bash submit_fno/build_round12.sh      # refuses while train2.py runs or a member is missing
```

Skipping `--err-ckpt` is the single most expensive mistake available here. The July 30 U-Net
package did exactly that: its contaminated local rel-L2 of 0.0721 produced a band with median
half-width 0.0046, the board met it with errors of 0.1179, and it returned `sps 36.94`.

`fit_band.py` must be re-run whenever the member set **or `ALPHA`** changes — the band is
fitted to that specific ensemble's error field, at the alpha it will actually deploy.
`--ckpt` and `--err-ckpt` both matter, for different reasons: the error field comes from
`--err-ckpt`, but `sigma_ref` comes from `--ckpt`, because it has to be the scale of the models
that emit sigma at test time. Measured, the two agree to 0.1%.

## What is in the zip

| file | what |
|---|---|
| `submission.py` | `predict()`; normalize → ensemble → rescale fluctuation to the predicted std → denormalize → band |
| `_fno.py` | self-contained `FNO2d`, `UNetTwoHead`, `StackedFNO`, `fuse()`, `load_any` |
| `fno.pth` … `fno_11.pth` | 11 members of four architectures, averaged on the GPU |
| `band.npz` | `base` / `sigma_ref` / `p` — the sigma-adaptive SPS half-widths from `fit_band.py` |

82.56 MB extracted, 76.36 MB zipped, against a 256 MB cap. Only `torch` and `numpy` are
imported — despite the filenames, `load_any` dispatches on each blob's own `arch` field, so the
U-Net member loads through the same path.

## Verification

* **`validate_strict.py` is the authority.** It builds the zip, blocks `scipy`/`h5py`/`einops`/
  `matplotlib`/`pandas`, reproduces the load path where the extract directory is *not* on
  `sys.path`, asserts permutation invariance, and times `predict()` end to end. Needs a GPU.
* **`../verify/cpu_smoke_submission.py` is the login-node stand-in.** Same package blocking
  (matched to the official kit's `smoke_test_kit.py`, which confirms
  `MISSING_IN_EVAL_IMAGE = {scipy, h5py, matplotlib, pandas}`), same import path, real
  `val_unseen` windows, output-contract checks. No GPU, no timing.
* **`../verify/diff_scoring_{official,real}.py`** pin the local `scoring.py` against the
  official kit v9: worst disagreement 2.4e-6 on real data.

## The model

Eleven members, averaged in raw field space on the GPU before the fluctuation rescaling, so
`compose()` runs once on a coherent field and eleven members cost one host transfer.

**`fuse()` collapses same-architecture members into one invocation.** The seven width-24 seeds
and the modes-(12,20) member share one `StackedFNO` group (spectral weights zero-padded to the
group max — note `w2` pads at the **front**, since it multiplies `ft[-my:]`), leaving w48, w64
and the U-Net as their own. 11 members → **4 invocations**, exact to 1.5e-8 with TF32 off.

**Member count is a real trade, just a cheap one.** Board time is *not* the constant an earlier
version of this file claimed — four rounds fit
`board ms = 4.447 + 0.324·invocations + 0.431·local ms` to ±0.064 ms, and 1 board ms ≈ 0.05
`final`. An extra fused member costs ~0.27 local ms ≈ 0.006 `final`; a new invocation costs
0.324 board ms ≈ 0.016. Both are cheap against a +0.05…0.12 accuracy gain, which is why the
package grew rather than shrank.

**The members are deliberately not all the same architecture, and not all the same family.**
Individually the bigger FNOs are *worse* — on the AoA 15–20 proxy, width 24 scores 81.67
against 81.54 / 81.33 / 81.17 — because with 74 trajectories and a gradient loss, capacity buys
memorization. But they fail differently, and in an ensemble that is what counts. Each new
*architecture* is worth +0.05…0.07 where an extra same-width seed is worth +0.01…0.03, and
round 12 showed the effect is stronger still across model *families*: the U-Net member is worth
+0.091 where an extra w24 seed is worth +0.019. Mode-diversity variants beyond m1220 are null.

Each width-24 member: 4-layer FNO, modes (8, 12) — 0.90M parameters, 3.6 MB, with `--var-mlp`
and `--lift-mlp` (capacity at the boundaries helps; capacity in the trunk hurts). A per-pixel
log-std head runs alongside the field head; at inference the predicted fluctuation is rescaled
toward that std at **`ALPHA = 0.90`** (raised from 0.75 in round 11 after three independent
readings agreed, and re-confirmed on the round-12 member set).

Trained with an extra term matching the **target's velocity-gradient tensor** (`--lam-grad 1.0`)
and `--lam-v 0.3 --lam-t 0.15 --alpha-train 0.0`. **`train2.py`'s argparse defaults are not the
shipped recipe** — read it off a checkpoint's stored `config`.

## The band

v9 replaced `score_sps`'s logistic with an identity, which made SPS a full 0–100 subscore and
4.4× more responsive. Three things follow:

* The aggregate **factorizes** into a per-sample accuracy factor times a per-element interval
  term, so the band problem separates over elements and each element's optimum sits exactly at
  one of its observed `|error|` values. `../sps9.py:optimal_band` solves it by sorting.
* The half-width is **sigma-adaptive**: `half = base·(σ/σ_ref)**p` with p = 0.45, and `base` is
  box-smoothed over (T,H,W) at k = 3. Both were selected by leave-one-**trajectory**-out CV, not
  in-sample — the unsmoothed band at p = 0 *is* the in-sample optimum, which is how round 6
  wrongly declared the band closed.
* The band is fitted on the **AoA 15–20° subset** (`--fit-aoa`, default) at a 1.39× tail
  inflation. Error is governed by angle of attack and by almost nothing else, and that subset is
  the only local proxy in the board's neighbourhood.

Round 11 closed this for real: pooling the fitting split buys 0.000 sps across 84 CV configs.

## Expected score

The shipped package is all-data, so it has **no honest local score**. What it has is a board
*prediction*, from `../predict_board.py` — which predicted round 11 at **80.603** against an
actual **80.60907**, its first genuine out-of-sample test.

| subscore | round 10 (actual) | round 12 (predicted) |
|---|---|---|
| rel_l2 | 94.576 | 94.521 |
| tke | 79.153 | **79.869** |
| mvpe | 93.681 | 93.782 |
| time | 91.121 | 91.248 |
| sps | 39.626 | 39.628 |
| **final** | **80.512** | **80.707** |

Three warnings:

* **`mean5` is not `final_score`, and `final ≈ mean5 + 0.73` is FALSIFIED** (round 10's offset
  was +0.903, and mean5 moved down while `final` moved up). Use `../final_v9.py` and work in
  **deltas between rounds**, where the intercept cancels.
* **Local accuracy deltas have stopped transferring.** Two architectures differing by 0.008 in
  local rel-L2 landed within 0.0004 of each other on the board. Treat anything under ~0.4 in
  mean5 as unproven, and never decide a member set on a proxy margin of 0.001 — round 10 dropped
  w48 that way and the board charged −0.06 for it.
* **Timings must be converted.** The board-time model was fitted with RTX 4060 numbers; the A100
  runs this shape 1.850× faster. 1.208 ms/window measured → 2.235 ms 4060-equivalent → 6.706
  board ms.

## What this is not

Track 1 tiles windows at a stride equal to the output horizon and hands `predict()` all of them
at once, so window *k*'s target is window *k+1*'s input. A submission can therefore read the
answers out of its own argument; doing so scores ~93.5 without modelling anything. This package
deliberately does not. See `../submit_trick/DISCLOSURE.md` for the measurement and the report
sent to the organizers, who have confirmed such scores are not valid.
