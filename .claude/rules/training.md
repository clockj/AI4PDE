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
* Real data is in m/s, sim is non-dimensional. `simreal.py` maps sim onto the real statistics; its cache
  (`RPDE_SIMCACHE`) is keyed without the data path, so use a cache directory per data release.
* A retrain with the same seed is not bit-identical (F-FNO about 0.5 % apart in weights, ViT about 10 %). Judge a
  retrain on the package score, within 0.005.
* Checkpoints are untracked and live in `Track_1/unet/`. The eleven of the packages of record:
  `ffno_all_w40l8_sim_s1..s4`, `arch_vit_all_s1`, `arch_vit_all_cmb3sim_s3` (members) and `p2_ffno_l8_sim_s1..s3`,
  `arch_vit_d384`, `arch_vit_cmb3sim` (twins).
