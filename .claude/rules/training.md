---
paths:
  - "Track_1/unet/**"
  - "reproduce.sh"
  - "REPRODUCE.md"
---

# Training code

* **`train2.py`'s argparse defaults are not the recipe.** Defaults that differ from it: `--epochs` 150 (400),
  `--alpha-train` 1.0 (0.0; 1.0 collapses the model), `--loss rel` (`mse`), `--lam-f` 1.0 (0.0 with the coverage loss),
  `--lam-v` / `--lam-t` 1.0 / 0.5 (0.15 / 0.15), `--sim-loss shape` (`full`), `--width` 64, `--n-layers` 4, modes (12, 16),
  `--arch unet`, `--seed` −1 (unseeded). The recipes are the fragments `FFNO`, `VITC`, `VITP`, `SIMF` in `reproduce.sh`,
  and every checkpoint records its own in `torch.load(p, weights_only=False)["config"]`.
* **The organizers retrain from this code.** Do not change `main()`'s order of random draws, the argparse defaults, or
  the bodies of `FFNO2d`, `ViT2d`, `FNO2d`, `_heads`, `_FFNOLayer`, `_GridIn`: parameter creation order fixes the seeded
  initialisation, and a flag that disappears breaks the config comparison in `REPRODUCE.md` §8.3.
* `models_alt.py` holds the families that ship or are still trained by a `reproduce.sh` target; `models_closed.py` holds
  the closed ones and is re-exported through `models_alt`. `Track_1/submit_fno/_fno.py` builds F-FNO members from
  `models_alt` when it runs inside the repository, so a change there changes what the band fit executes.
* **Twins** drop `--all-data` and add `--sim-exclude-held`. `--exclude` filters the loss only: preprocessing that reads
  real-data statistics (the sim map, masks) must be told the split through `real_exclude`, or it leaks.
* **Never train through `compose`** (`--alpha-train 0.0`).
* **`--aug-resim S`** (the Decision-phase candidate, REPRODUCE.md §14): a real window resampled in time by s and scaled by s,
  s log-uniform in [1/(1+S), 1+S]. At 0 (the default) nothing runs and nothing is drawn; above 0 its draws come from a
  generator of its own, so the default random stream keeps its order and only the data differ.
* It refuses `--hist`, `--ctx` ≠ 20, `--teacher`, `--subpix`, `--mixup`, `--norm self` and `--aug-scale`. Its verifier is
  `experiments/decision_20260928/patches/t4_aug_resim_check.py`.
* Real data is in m/s, sim is non-dimensional. `simreal.py` maps sim onto the real statistics; its cache
  (`RPDE_SIMCACHE`) is keyed without the data path, so use a cache directory per data release.
* A retrain with the same seed is not bit-identical (F-FNO about 0.5 % apart in weights, ViT about 10 %). Judge a
  retrain on the package score, within 0.005. (09-28: the candidate's from-scratch run retrained its two ViT twins
  bit-identically on an A100, its F-FNO twins 0.42 – 0.49 % apart; package −0.0003.)
* Checkpoints are untracked and live in `Track_1/unet/`. The eleven of the packages of record:
  `ffno_all_w40l8_sim_s1..s4`, `arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3` (members) and `p2_ffno_l8_sim_s1..s3`,
  `arch_vit_d384`, `arch_vit_cmb3sim` (twins). The eleven of the Decision-phase candidate ("T4 all", promoted from the
  from-scratch run on 09-29, md5 checked): `ffno_all_w40l8_sim_resim_s1..s4`, `arch_vit_all_resim_s1`,
  `arch_vit_all_cmb3sim_resim_s3` (members) and `p2_ffno_l8_sim_resim_s1..s3`, `arch_vit_d384_resim`,
  `arch_vit_cmb3sim_resim` (twins).
