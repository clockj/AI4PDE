"""Merge a trained F-FNO and a trained ViT into one `--init` file for `--arch ffvit` / `ffvit0`.
usage: make_ffvit_init.py <ffno.pth> <vit.pth> <out.pth>
Keys become `f.<name>` and `v.<name>`; everything else in FFViT2d (links, mix) keeps its own init."""
import sys
import torch

ff, vit, out = sys.argv[1:4]
a, b = torch.load(ff, map_location="cpu"), torch.load(vit, map_location="cpu")
assert a["arch"] == "ffno" and b["arch"] == "vit", (a["arch"], b["arch"])
sd = {f"f.{k}": v for k, v in a["model_state_dict"].items()}
sd.update({f"v.{k}": v for k, v in b["model_state_dict"].items()})
torch.save({"model_state_dict": sd, "parents": {"f": ff, "v": vit}}, out)
print(f"wrote {out}: {len(sd)} tensors from {ff} + {vit}")
