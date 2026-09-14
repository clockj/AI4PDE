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

`r24_xfam6.npz` is fitted on R21's twins plus one cmb3sim-recipe ViT twin (`arch_vit_cmb3sim`),
for R21's members plus `arch_vit_all_cmb3sim_s3`. Promoted 2026-09-13 after R24 scored. T2R11
(x8v) shipped `r21_xfam5.npz` byte for byte and needs no row of its own.

`r24_xfam6.npz` shipped again, byte for byte, in R25 and T2R12 (2026-09-13): R24's archive with
fp16 autocast baked in.
