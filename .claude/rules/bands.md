---
paths:
  - "Track_1/bands/**"
  - "Track_1/sigblend/**"
  - "Track_1/submit_fno/fit_band.py"
---

# Bands

* A band is the one part of a package that cannot be rebuilt from the checkpoints bit for bit. A band that has scored
  lives in `Track_1/bands/` (`git add -f`; `.gitignore` excludes `*.npz` elsewhere) with a `MANIFEST.md` entry: md5,
  the rounds that shipped it, members, twins, flags.
* **Reproducing a scored package reuses its committed band.** A refit is for a new member set or a new ALPHA.
* **The static band** is `fit_band.py --ckpt <members> --err-ckpt <twins> --alpha 0.80 --band-p 0.45 --centre-alpha 0
  --fill-unscored` (inflate 1.39 by default). The error field comes from the twins, never from the all-data members,
  and it is **in sample**: the default `--fit-on seen --fit-aoa 15 20` takes the twins' error on the three AoA 15 – 20
  trajectories of `val_seen` (126 windows), which the twins trained on. Fitted out of sample instead, the composite
  moves by 0.002 – 0.012. The candidate keeps the construction.
* **The SIGBLEND band** is its transform by `sigblend/make_k0_band.py` (λ 0.5, inflate 1.29, 16 σ-bins). The npz and the
  two literals in its json (`EBAND` for Track 1, `T2_sigpow` for Track 2) are one object: build with all three or none.
* κ and q are fitted on the eight `val_unseen` trajectories. The evidence for them is the leave-one-trajectory-out table
  the script prints. Re-derived from a fresh twin dump they moved by 0.4 % (job 7569179).
* **The Decision-phase candidate's bands** are committed: `dp_resim_all_sigblend_lam0.5.npz` ("T4 all", the owner's
  choice) and `dp_resim_ffno_sigblend_lam0.5.npz` ("T4 F-FNO only"), each with `sigblend/band_<name>.json` and a
  `MANIFEST.md` entry. `reproduce.sh decision` reuses the one of its recipe and refits only with `RPDE_REPRO_FITBAND=1`.
* A refit from another set of twins of the same recipe changes the composite by about 1e-4 (three refits, 09-28).
* A band refitted on pooled held-out-condition errors lost (ΔC×100 −0.018 out of distribution, `sps` −0.103 in
  distribution): the error field stays in-distribution.
* Never edit a committed band. A new band gets a new file name.
