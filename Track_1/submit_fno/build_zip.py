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
    args = ap.parse_args()

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
        if args.fp16 and name in ckpt_names:
            import torch
            blob = torch.load(src, map_location="cpu")
            blob["model_state_dict"] = {k: (v.half() if v.is_floating_point() else v)
                                        for k, v in blob["model_state_dict"].items()}
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
