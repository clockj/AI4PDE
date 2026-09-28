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
  --fill-unscored` (inflate 1.39 by default). The error field comes from the twins, never from the all-data members.
* **The SIGBLEND band** is its transform by `sigblend/make_k0_band.py` (λ 0.5, inflate 1.29, 16 σ-bins). The npz and the
  two literals in its json (`EBAND` for Track 1, `T2_sigpow` for Track 2) are one object: build with all three or none.
* κ and q are fitted on the eight `val_unseen` trajectories. The evidence for them is the leave-one-trajectory-out table
  the script prints. Re-derived from a fresh twin dump they moved by 0.4 % (job 7569179).
* Never edit a committed band. A new band gets a new file name.
