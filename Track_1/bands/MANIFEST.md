# The fitted SPS bands of every scored round

**Why these are in the repo when `.gitignore` excludes `**/*.npz`.** A band is the one part of
a submission that is **not** reproducible from the checkpoints. Weights, `submission.py` and
`_fno.py` are one `build_round13.sh` away; `base` is 81 920 per-element half-widths fitted by
`sps9.optimal_band` on the held-out twins' error field, and refitting with a different twin set
or a different window order gives a different table. SPS scores per-element containment, so a
small difference in `base` flips whole populations of elements — the largest containment move
in this project's history (**+1.410 `sps`**, R9 → R10) *was* a band refit.

They are 0.52 MB each and there are **seven distinct bands across sixteen scored rounds**, so
version control is the right home. The blanket `**/*.npz` rule exists for the ~1 GB of derived
`val_*.npz` caches, which rebuild from the h5 in minutes.

## Contents

```
base       (1, 20, 32, 64, 3) float32   per-element half-widths; [..., 2] is the p channel,
                                        all zero and never scored, so 81 920 free numbers
sigma_ref  (1, 1, 32, 64, 3)  float32   reference sigma field
p          scalar                       0.45, the sigma-scaling exponent
```

Deployed by `submission.py` as

```
half  = base * clip(sigma_head / sigma_ref, 0.3, 3.0) ** p
lower, upper = pred - half, pred + half
```

`base` says how far this pixel, this frame, this channel is typically wrong; `(sigma/sigma_ref)**p`
rescales it per window by that window's own predicted fluctuation amplitude.

## Round → band

| band file | md5 (12) | rounds that shipped it | ids |
|---|---|---|---|
| `r13_8memb.npz` | `34bec6d33020` | R13, R14, **T2R1** | 910641, 912521, 910642 |
| `r15_dropout8.npz` | `879cb08cd8f5` | R15, **T2R2** | 913176, 912522 |
| `r16_w96x8.npz` | `22362b02d9ad` | R16, **T2R3** | 914506, 913187 |
| `t2r4_ufno3.npz` | `601bdd15262d` | **T2R4** | 914508 |
| `r17_L8do45x4.npz` | `34b9316b58b6` | R17, R18, **T2R5, T2R6** | 915380, 917177, 915381, 917178 |
| `r19_simx4.npz` | `2d444bab8cf1` | R19, **T2R7** | 918587, 918588 |
| `r20_cmb3simx4.npz` | `2eed9e6be104` | **R20, T2R8** | 920369, 920370 |

Recovered by extracting `band.npz` from each archive in `../../sub_candidates/submitted/` and
de-duplicating by md5; the six that also had a staged copy in `sub_candidates/` match it exactly.
**`r13_8memb.npz` existed only inside those three zips** — there was never a staged copy, so it
would have been lost the day anyone cleaned that directory.

Rounds R6–R12 predate the `submitted/` archive and their bands are gone.

## The rule

* A band that has **scored on a board** lives here and is committed, named
  `<first round that shipped it>_<package tag>.npz`.
* A band for a package that has **not** been submitted stays in `sub_candidates/` as
  `band_<tag>.npz`, where `build_round13.sh`'s callers stage it. Promote it here after upload.
* Every build prints its new band's md5 next to the previous one. Check they differ — round 18
  priced a whole package against a stale band because a failed build left the old file readable
  and nothing recorded which file that build had produced. `git log Track_1/bands/` is that
  record now.

| `r21_xfam5.npz` | `e32707e77ab6e484b9af5b369bf0a772` | R21 (921000), T2R9 (921001) |

`r21_xfam5.npz` is the first band fitted on a HETEROGENEOUS member set (three held-out cmb3sim
twins plus `arch_vit_d384`). Promoted 2026-09-11 after R21/T2R9 scored.

| `r24_xfam6.npz` | `94bbebc3f1fef6870bfab36bb35d8450` | R24 (id not reported) |
| `r26_p1swin4.npz` | `b5f0ebf4fda9fae12b1cf71789717fab` | R26 (id not reported) |
| `t2r13_p2ffnoL8x4.npz` | `b99d177d889126b0734bee0680bc69b0` | T2R13 (id not reported) |
| `t2r14_p2ffnoL4x4.npz` | `c7572ae00c07dccff52a39e17160fdda` | T2R14 (id not reported) |
| `t2r15_l8x4.npz` | `63cc4cb3e00704b4efaaaec8bc19c309` | T2R15 (id not reported) |

`r24_xfam6.npz` is fitted on R21's twins plus one cmb3sim-recipe ViT twin (`arch_vit_cmb3sim`),
for R21's members plus `arch_vit_all_cmb3sim_s3`. Promoted 2026-09-13 after R24 scored. T2R11
(x8v) shipped `r21_xfam5.npz` byte for byte and needs no row of its own.

`r24_xfam6.npz` shipped again, byte for byte, in R25 and T2R12 (2026-09-13): R24's archive with
fp16 autocast baked in.


`r26_p1swin4.npz` (promoted 2026-09-16) was fitted on the held-out Swin twins `w1_swin_unet_s1`,
`w2_swin_unet_w64_s2`, `w2_swin_unet_w64_s3` plus `arch_vit_d384` and `arch_vit_cmb3sim`, at ALPHA
0.60 and `--band-p 0.45`, for P1's four all-data Swin members plus R25's two ViTs. R26 scored
81.218444 with it and LOST to R25, so it is a record, not a recommendation.

`t2r13_p2ffnoL8x4.npz` (promoted 2026-09-16) was fitted on the held-out F-FNO L8 twins
`p2_ffno_l8_s1..s3` plus the same two ViT twins, at ALPHA 0.60 and `--band-p 0.45`, for P2-L8's four
all-data F-FNO w40 L8 members plus the two ViTs. **T2R13 is the Track-2 best (81.314785)** and its
`sps` +0.302 over T2R12 is far more than the per-sample factor alone, so this band together with the
F-FNO family moved containment. A future Track-2 band is refit with its member set, never this one
reused.

**The two pins T2R13 shipped with, recorded here because they lived only on scratch**
(`$SCRATCH/pkg_ffno/P2_ALPHA` and `P2_FNO_MD5`, REPRODUCE.md 6.5; scratch is purgeable, and a
package of record whose constants are only on scratch cannot be re-verified):

| | value | why it is a pin, not a note |
|---|---|---|
| `ALPHA` | **0.60** | it is a build ARGUMENT (`build_round13.sh 0.60`), not a default -- the script's own default is 0.90 -- and the band is only valid at the ALPHA it was fitted at |
| `_fno.py` | **`73c83f572c477b8aaa17566f168f72b7`** | the gated inference module the archive carries; the build jobs refuse unless the tree's `_fno.py` matches it |

Both were re-read from the uploaded archive on 09-18, not from the scratch files: unzipping
`track2_r27_p2ffno4_a060.zip` gives `_fno.py` md5 `73c83f57…` and a `submission.py` carrying
`ALPHA = 0.60`, `AMP = "fp16"`, `SPEC_PATH = "bmm"`, `FFNO_PATH = "dense"`. The repo's current
`Track_1/submit_fno/_fno.py` still has that md5.

**The Track-1 twin of this package, `track1_r31_p2ffnoL8_a060.zip`** (staged 09-18, zip md5
`467a7bad22330bbf56228b3c435bc24b`, 30.28 MB), carries **the same band, byte for byte**, and the
same two pins: `ALPHA = 0.60` and `_fno.py` `73c83f572c477b8aaa17566f168f72b7`, with `AMP = "fp16"`,
`SPEC_PATH = "bmm"`, `FFNO_PATH = "dense"`, `HOST_WRITE = "torch"`, `GRAPH = False`,
`BLEND_BETA = 0.0` and int8 `--linear tensor` packing. Its members are the same four all-data F-FNO
w40 L8 plus R25's two ViTs, so **it and T2R13 differ only in the track they are built for** -- which
is what makes the Track-2 board's accuracy and containment deltas transferable to Track 1.

`t2r14_p2ffnoL4x4.npz` (promoted 2026-09-17) was fitted inside build job 7332924 on the held-out
F-FNO **L4** twins `p2_ffno_l4_s1..s3` plus `arch_vit_d384` and `arch_vit_cmb3sim`, at ALPHA 0.60 and
`--band-p 0.45`, for P2-L4's four all-data F-FNO w40 L4 members plus the same two ViTs. It shipped in
`track2_r30_p2ffnoL4_a060.zip` (md5 `38c157835c2fc51efda865a9c5dcfe50`) and T2R14 scored 81.308359
with it, **−0.006 against T2R13**. It is the band that carries the campaign's one board measurement
of CONTAINMENT ACROSS DEPTH: the three error subscores moved by less than the forced +0.038 of sps,
and `sps` came back −0.273, so containment fell **−0.310** against T2R13's L8 band on the same recipe.
Every depth's band scores well on its own twins, so this is a record of something no local instrument
reads — not a defect in the fit. L8 stays the Track-2 package.

`t2r13_p2ffnoL8x4.npz` SHIPPED AGAIN, byte for byte, in **R28** (2026-09-18) — the Track-1 twin of
T2R13 — and **R28 is the Track-1 best at 81.401196**. So this one band now carries both boards' record
packages, and its `sps` moved containment on both: **+0.302 on Track 2, +0.253 on Track 1**. It is the
first band in this list to be scored on two boards with two different results, and both were wins.

`t2r15_l8x4.npz` (promoted 2026-09-18) was fitted on the held-out F-FNO L8 twins `p2_ffno_l8_s1..s3`
**alone** — no ViT twins — at ALPHA 0.60 and `--band-p 0.45`, for the four all-data F-FNO L8 members
with no ViT pair. It shipped in `track2_r28_l8x4_a060.zip` and T2R15 scored 81.265138 with it, −0.050
against T2R13. It is kept as the measurement of what the ViT pair's diversity is worth to a band: the
same members, the same recipe, the same ALPHA, and **containment −0.132** purely for the two ViTs the
error field no longer sees. A band fitted without the ViT twins is not wrong — it is fitted to a
genuinely narrower error field — which is precisely why the ViT pair stays in the package.
