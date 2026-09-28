# SIGBLEND band — the band of the packages of record

Inference-only. It needs the recipe-A members and their five held-out twins, nothing else.

```
half  = base' · clip(σ/σ_ref, 0.3, 3)^p' · (σ + 1e-8)^(λ q_c)
base' = base^(1-λ) · κ_{t,c}^λ      p' = (1-λ) · 0.45      λ = 0.5      q = (0.4847, 0.4907)
```

`base`, `σ_ref` are the static band's; σ is the members' mean σ head in physical units; κ (per frame and channel) and q
(per channel) are a power law of the optimal half-width against σ, fitted on the twins' held-out errors.
The package evaluates the last factor itself: Track 1 through `EBAND` (mode `mult`, GPU-band path, so the batch-1 graph
is off), Track 2 through `SIGPOW`.

## Re-create it from the release

`RPDE_REPRODUCE_GO=1 RPDE_REPRO_FITBAND=1 bash reproduce.sh final` runs these three steps after training:

1. Static band: `python Track_1/submit_fno/fit_band.py --ckpt <6 members> --err-ckpt <5 twins> --alpha 0.80 --band-p 0.45
   --centre-alpha 0 --fill-unscored --out band_r38_a080_c0_fill.npz` (inflate 1.39 is the default).
2. Twin dump on the eight `val_unseen` trajectories: `python Track_1/sigblend/dump_twins.py dump.npz`.
3. `cd Track_1/sigblend && B_DUMP=dump.npz K0_BAND=band_r38_a080_c0_fill.npz K0_INFL=1.29 K0_NB=16 PYTHONPATH=..
   python make_k0_band.py OUTDIR`. It prints the leave-one-trajectory-out read and writes
   `band_k0_sigblend_lam{0.3,0.5,0.7}.npz` with a `.json` each (q, log κ, the `EBAND` literal, the `SIGPOW` exponents).

Without `RPDE_REPRO_FITBAND=1` the build reuses the committed pair, `../bands/k27_sigblend_lam0.5.npz` and
`band_k0_sigblend_lam0.5.json`, which is what reproducing the scored packages must do.

## Traps

* The npz and its two literals travel together. A SIGBLEND npz in a package that does not apply the σ-power factor is
  about 3× too wide. `submit_fno/build_final.sh` reads all three from one place.
* κ and q are fitted on all eight validation trajectories. The evidence is the leave-one-out table, never the in-sample one.
* `bandlab.py` is the numpy reference of the band's scoring; `../verify/sigblend.py` checks the package against it.
