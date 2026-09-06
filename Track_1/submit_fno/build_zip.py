"""Assemble the Track 1 FNO submission archive.

    python submit_fno/build_zip.py --ckpt ../unet/fno_grad.pth
    python submit_fno/validate_strict.py

Stages submission.py + _fno.py + fno.pth + band.npz at the archive ROOT (the layout the
platform expects) and writes submission.zip. The chosen checkpoint is always packaged under the
fixed name fno.pth, which is what submission.py loads; the source name is printed so the zip's
provenance stays traceable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile

# Must equal _fno._Q8_SUFFIX -- that is the only contract between the packer here and the
# unpacker there, and the assert below is what keeps the two from drifting apart silently.
Q8_SUFFIX = "._q8scale"

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK1 = os.path.dirname(HERE)
STAGE_DEFAULT = os.path.join(HERE, "_package")
ZIP_DEFAULT = os.path.join(HERE, "submission.zip")
SIZE_CAP_MB = 256

DEFAULT_CKPT = os.path.join(TRACK1, "unet", "fno_grad.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default=os.path.join(HERE, "band.npz"),
                    help="which fitted band to package. It MUST have been fitted for exactly "
                         "this member set at exactly this ALPHA (fit_band.py --ckpt/--alpha) "
                         "-- the band is solved against one ensemble's error field.")
    ap.add_argument("--stage", default=STAGE_DEFAULT)
    ap.add_argument("--out", default=ZIP_DEFAULT)
    ap.add_argument("--ckpt", nargs="+", default=[DEFAULT_CKPT],
                    help="checkpoint(s) to package. More than one is an ensemble, averaged in "
                         "raw field space on the GPU; the first is packaged as fno.pth and the "
                         "rest as fno_2.pth, fno_3.pth ...")
    ap.add_argument("--fp16", action="store_true",
                    help="store every floating tensor of the state_dict as float16. Halves the "
                         "archive; a width-96 member is 57.0 MB fp32 and 28.5 MB fp16, so eight "
                         "of them go from 456 MB (over the 256 MB cap) to 228 MB (under it). "
                         "torch's load_state_dict copies with a dtype cast, so the model is "
                         "still built and run in fp32 and _fno.py needs no change. MEASURED "
                         "cost on fno_all_w96do30_s1 over 16 val_unseen windows: max abs "
                         "4.3e-4, rel-L2 of the difference 5.5e-4 against a model rel-L2 of "
                         "0.078 -- the two add in quadrature, so it is 2e-6 of the error. "
                         "Safe here because this FNO stores its spectral weights as REAL "
                         "tensors with a trailing (real, imag) axis; a genuinely complex "
                         "parameter would need pack_ckpt_fp16.py's split instead.")
    ap.add_argument("--int8", action="store_true",
                    help="store the LARGE floating tensors of the state_dict as int8 with an "
                         "fp16 scale per (out, in) channel pair, and everything under 4096 "
                         "elements as fp16. Halves the archive AGAIN on top of --fp16: a "
                         "width-96 8-layer member goes 108.5 MB fp32 -> 54.3 fp16 -> 29.0 int8, "
                         "so EIGHT of them are 232 MB against the 256 MB cap where eight fp16 "
                         "would be 434. Measured cost over 96 val_unseen windows (quant8.py): "
                         "rel-L2 of the field difference 8.3e-4 against fp16's 2.7e-4 and a "
                         "model rel-L2 of 0.077, i.e. +6e-5 of the error in quadrature -- the "
                         "same argument that made --fp16 free. Coarser scale granularities are "
                         "20-40x worse and none is shippable. _fno.dequant8 undoes this at "
                         "load time, and it is a no-op on any blob that does not carry the "
                         "companion scale keys. Implies --fp16 for the tensors it leaves alone.")
    args = ap.parse_args()
    if args.int8:
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location("_fno_pack", os.path.join(HERE, "_fno.py"))
        _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
        assert _m._Q8_SUFFIX == Q8_SUFFIX, (
            f"packer/unpacker suffix drift: {Q8_SUFFIX!r} here, {_m._Q8_SUFFIX!r} in _fno.py")

    names = ["fno.pth"] + [f"fno_{i}.pth" for i in range(2, len(args.ckpt) + 1)]
    files = [
        (os.path.join(HERE, "submission.py"), "submission.py"),
        (os.path.join(HERE, "_fno.py"), "_fno.py"),
        (args.band, "band.npz"),
    ] + list(zip(args.ckpt, names))
    for src, name in zip(args.ckpt, names):
        print(f"packaging checkpoint: {src}  ->  {name}")

    STAGE, ZIP = args.stage, args.out
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    ckpt_names = set(names)
    for src, name in files:
        if not os.path.exists(src):
            raise SystemExit(f"missing required file: {src}\n"
                             f"(band.npz comes from: python submit_fno/fit_band.py --ckpt ...)")
        if (args.fp16 or args.int8) and name in ckpt_names:
            import torch
            blob = torch.load(src, map_location="cpu")
            sd = {}
            for k, v in blob["model_state_dict"].items():
                if not v.is_floating_point():
                    sd[k] = v
                elif args.int8 and v.numel() >= 4096 and v.dim() >= 2:
                    # Symmetric int8 with a scale per (out, in) channel pair. The spectral
                    # weights are (win, wout, my, mx, 2) real tensors and are ~97 % of the
                    # archive; the scale then costs 2 bytes per 2*my*mx values, which is the
                    # 0.6 MB/member difference between this and a per-tensor scale.
                    red = tuple(range(2, v.dim()))
                    sc_ = v.abs().amax(dim=red, keepdim=True).clamp(min=1e-12).div(127.0)
                    sc_ = sc_.half()
                    sd[k] = torch.clamp(torch.round(v / sc_.float()), -127, 127).to(torch.int8)
                    sd[k + Q8_SUFFIX] = sc_
                else:
                    sd[k] = v.half()
            blob["model_state_dict"] = sd
            torch.save(blob, os.path.join(STAGE, name))
        else:
            shutil.copy2(src, os.path.join(STAGE, name))

    extracted_mb = sum(os.path.getsize(os.path.join(STAGE, n)) for _, n in files) / 1e6
    if extracted_mb >= SIZE_CAP_MB:
        raise SystemExit(f"extracted size {extracted_mb:.1f} MB exceeds the {SIZE_CAP_MB} MB cap")

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for _, name in files:
            z.write(os.path.join(STAGE, name), arcname=name)

    print(f"staged  {STAGE}")
    for _, name in files:
        print(f"  {name:16s} {os.path.getsize(os.path.join(STAGE, name))/1e6:8.2f} MB")
    print(f"\nwrote   {ZIP}")
    print(f"  extracted {extracted_mb:.2f} MB / {SIZE_CAP_MB} MB cap")
    print(f"  zipped    {os.path.getsize(ZIP)/1e6:.2f} MB")
    print("\nnext: python submit_fno/validate_strict.py")


if __name__ == "__main__":
    main()
