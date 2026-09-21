"""Transplant DINOv2 self-supervised ViT weights into `models_alt.ViT2d`, exactly.

The question the user asked -- does a DINOv3-style image foundation model help on 32x64 PIV
velocity fields -- is only answerable if the transplant is CORRECT. A silently wrong mapping
produces exactly what "pretraining does not transfer" looks like, so this file verifies itself
against a from-scratch reference implementation of the DINOv2 block before writing anything.

DINOv3 is gated on HuggingFace (HTTP 401 without an accepted licence) and ships under a custom
Meta licence, which sits badly with the competition's requirement that winners open-source the
complete solution. DINOv2 is Apache-2.0, freely redistributable, and structurally identical --
ViT-S/14 is dim 384 / depth 12 / 6 heads, which is our ViT2d's dim and head count exactly. So
the mechanism is tested on DINOv2; swapping in DINOv3-S/16 later is a URL change.

WHAT MAPS AND WHAT DOES NOT
  blocks       every LayerNorm, the attention and the MLP map 1:1. HF keeps q/k/v as three
               Linears and torch's MultiheadAttention wants them concatenated into
               `in_proj_weight` in [q; k; v] order, which is the layout it splits back out.
  LayerScale   DINOv2 blocks are `x + ls * f(norm(x))`; ViT2d is `x + f(n(x))`. Since `ls` is a
               per-channel scale applied AFTER the residual branch's last Linear, folding it
               into that Linear's weight rows and bias is algebraically exact -- verified below
               to float32 noise, not asserted.
  pos          DINOv2's is 1 cls + 37x37 patches at 518px; ours is an 8x16 grid of 4x4 patches.
               Bicubic-resampled. It is a learnable parameter either way.
  patch embed  NOT transplanted, and cannot be: DINOv2 reads 3 RGB channels through a 14x14
               kernel, we read 62 (20 frames x 3 fields + 2 coordinate channels) through 4x4.
  heads        NOT transplanted: `head_field` / `head_logstd` / `proj` are ours.

So this is a transplant of the TRUNK only, which is the part a foundation model is for.
"""
from __future__ import annotations

import argparse
import json
import struct

import numpy as np
import torch


def read_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Minimal reader -- the eval env has neither `safetensors` nor `transformers`."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    dt = {"F32": np.float32, "F16": np.float16, "BF16": None, "I64": np.int64}
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        npdt = dt[v["dtype"]]
        if npdt is None:
            raise ValueError(f"{k}: bf16 not handled")
        out[k] = torch.from_numpy(
            np.frombuffer(blob[a:b], dtype=npdt).reshape(v["shape"]).copy())
    return out


def convert(sd: dict[str, torch.Tensor], depth: int, gh: int, gw: int,
            take: str = "first") -> tuple[dict[str, torch.Tensor], list[int]]:
    """DINOv2 state dict -> ViT2d-named tensors for `depth` blocks."""
    nsrc = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layer."))
    if depth > nsrc:
        raise ValueError(f"asked for {depth} blocks, DINOv2 has {nsrc}")
    if take == "first":
        src_idx = list(range(depth))
    elif take == "even":                      # evenly spaced, endpoints included
        src_idx = [round(i * (nsrc - 1) / max(depth - 1, 1)) for i in range(depth)]
    else:
        raise ValueError(take)

    out: dict[str, torch.Tensor] = {}
    for dst_i, src_i in enumerate(src_idx):
        p = f"encoder.layer.{src_i}."
        q = f"blocks.{dst_i}."
        out[q + "n1.weight"] = sd[p + "norm1.weight"].clone()
        out[q + "n1.bias"] = sd[p + "norm1.bias"].clone()
        out[q + "n2.weight"] = sd[p + "norm2.weight"].clone()
        out[q + "n2.bias"] = sd[p + "norm2.bias"].clone()
        # torch splits in_proj into q, k, v by contiguous row blocks, in that order.
        out[q + "attn.in_proj_weight"] = torch.cat(
            [sd[p + "attention.attention.query.weight"],
             sd[p + "attention.attention.key.weight"],
             sd[p + "attention.attention.value.weight"]], 0).clone()
        out[q + "attn.in_proj_bias"] = torch.cat(
            [sd[p + "attention.attention.query.bias"],
             sd[p + "attention.attention.key.bias"],
             sd[p + "attention.attention.value.bias"]], 0).clone()
        # LayerScale folded into the row scaling of each branch's OUTPUT projection.
        ls1 = sd[p + "layer_scale1.lambda1"]
        ls2 = sd[p + "layer_scale2.lambda1"]
        out[q + "attn.out_proj.weight"] = (sd[p + "attention.output.dense.weight"]
                                           * ls1[:, None]).clone()
        out[q + "attn.out_proj.bias"] = (sd[p + "attention.output.dense.bias"] * ls1).clone()
        out[q + "mlp.0.weight"] = sd[p + "mlp.fc1.weight"].clone()
        out[q + "mlp.0.bias"] = sd[p + "mlp.fc1.bias"].clone()
        out[q + "mlp.3.weight"] = (sd[p + "mlp.fc2.weight"] * ls2[:, None]).clone()
        out[q + "mlp.3.bias"] = (sd[p + "mlp.fc2.bias"] * ls2).clone()

    out["norm_out.weight"] = sd["layernorm.weight"].clone()
    out["norm_out.bias"] = sd["layernorm.bias"].clone()

    # Position embedding: drop cls, resample the square patch grid onto ours.
    pe = sd["embeddings.position_embeddings"][:, 1:]            # (1, S*S, dim)
    s = int(round(pe.shape[1] ** 0.5))
    assert s * s == pe.shape[1], pe.shape
    dim = pe.shape[-1]
    pe = pe.reshape(1, s, s, dim).permute(0, 3, 1, 2)
    pe = torch.nn.functional.interpolate(pe, size=(gh, gw), mode="bicubic",
                                         align_corners=False)
    out["pos"] = pe.permute(0, 2, 3, 1).reshape(1, gh * gw, dim).contiguous()
    return out, src_idx


# ---------------------------------------------------------------------------------------------
def _ref_block(sd: dict[str, torch.Tensor], i: int, x: torch.Tensor, heads: int) -> torch.Tensor:
    """DINOv2's block, written straight from the published equations. The control."""
    p = f"encoder.layer.{i}."
    d = x.shape[-1]
    hd = d // heads

    def ln(t, w, b):
        return torch.nn.functional.layer_norm(t, (d,), sd[p + w], sd[p + b], 1e-6)

    h = ln(x, "norm1.weight", "norm1.bias")
    lin = torch.nn.functional.linear
    qq = lin(h, sd[p + "attention.attention.query.weight"], sd[p + "attention.attention.query.bias"])
    kk = lin(h, sd[p + "attention.attention.key.weight"], sd[p + "attention.attention.key.bias"])
    vv = lin(h, sd[p + "attention.attention.value.weight"], sd[p + "attention.attention.value.bias"])
    b, n, _ = h.shape
    sh = lambda t: t.reshape(b, n, heads, hd).transpose(1, 2)
    a = torch.softmax(sh(qq) @ sh(kk).transpose(-2, -1) / hd ** 0.5, -1) @ sh(vv)
    a = a.transpose(1, 2).reshape(b, n, d)
    a = lin(a, sd[p + "attention.output.dense.weight"], sd[p + "attention.output.dense.bias"])
    x = x + sd[p + "layer_scale1.lambda1"] * a
    h = ln(x, "norm2.weight", "norm2.bias")
    m = lin(torch.nn.functional.gelu(
        lin(h, sd[p + "mlp.fc1.weight"], sd[p + "mlp.fc1.bias"])),
        sd[p + "mlp.fc2.weight"], sd[p + "mlp.fc2.bias"])
    return x + sd[p + "layer_scale2.lambda1"] * m


def verify(sd, conv, depth, src_idx, heads=6, dim=384, n_tok=128) -> float:
    """Run random tokens through ViT2d's blocks and through the reference. Must agree."""
    import models_alt
    m = models_alt.ViT2d(dim=dim, depth=depth, var_head=True, heads=heads,
                         dropout=0.0, drop_path=0.0).eval()
    miss = m.load_state_dict(conv, strict=False)
    assert not miss.unexpected_keys, miss.unexpected_keys
    torch.manual_seed(0)
    x = torch.randn(2, n_tok, dim)
    worst = 0.0
    with torch.no_grad():
        a, b = x.clone(), x.clone()
        for dst_i, src_i in enumerate(src_idx):
            blk = m.blocks[dst_i]
            q = blk["n1"](a)
            a = a + blk["attn"](q, q, q, need_weights=False)[0]
            a = a + blk["mlp"](blk["n2"](a))
            b = _ref_block(sd, src_i, b, heads)
            e = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-9)
            worst = max(worst, e)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/fs/scratch/PAS1187/qwu/tmp/realpde/pretrained/"
                                     "dinov2_small.safetensors")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--take", default="first", choices=["first", "even"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = read_safetensors(args.src)
    gh, gw = 32 // args.patch, 64 // args.patch
    conv, src_idx = convert(sd, args.depth, gh, gw, args.take)
    e = verify(sd, conv, args.depth, src_idx, heads=args.heads, dim=args.dim, n_tok=gh * gw)
    print(f"transplant check: {args.depth} blocks {src_idx} -> max relative disagreement "
          f"with the reference DINOv2 block  {e:.2e}")
    if e > 1e-4:
        raise SystemExit("TRANSPLANT IS WRONG -- refusing to write")
    n = sum(v.numel() for v in conv.values())
    print(f"pos: std {conv['pos'].std():.4f}  (ViT2d's random init is trunc_normal std 0.02)")
    # `train2.py --init` reads exactly this key and copies by name+shape, so nothing else needs
    # to change to run the arm.
    torch.save({"model_state_dict": conv}, args.out)
    print(f"wrote {args.out}: {len(conv)} tensors, {n / 1e6:.2f}M values")


if __name__ == "__main__":
    main()
