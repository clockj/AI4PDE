"""Assemble the Track 1 FNO submission archive.

    python submit_fno/build_zip.py --ckpt ../unet/fno_grad.pth
    python submit_fno/validate_strict.py

Stages submission.py + _fno.py + fno.pth + band.npy at the archive ROOT (the layout the
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
STAGE = os.path.join(HERE, "_package")
ZIP = os.path.join(HERE, "submission.zip")
SIZE_CAP_MB = 256

DEFAULT_CKPT = os.path.join(TRACK1, "unet", "fno_grad.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[DEFAULT_CKPT],
                    help="checkpoint(s) to package. More than one is an ensemble, averaged in "
                         "raw field space on the GPU; the first is packaged as fno.pth and the "
                         "rest as fno_2.pth, fno_3.pth ...")
    args = ap.parse_args()

    names = ["fno.pth"] + [f"fno_{i}.pth" for i in range(2, len(args.ckpt) + 1)]
    files = [
        (os.path.join(HERE, "submission.py"), "submission.py"),
        (os.path.join(HERE, "_fno.py"), "_fno.py"),
        (os.path.join(HERE, "band.npy"), "band.npy"),
    ] + list(zip(args.ckpt, names))
    for src, name in zip(args.ckpt, names):
        print(f"packaging checkpoint: {src}  ->  {name}")

    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    for src, name in files:
        if not os.path.exists(src):
            raise SystemExit(f"missing required file: {src}\n"
                             f"(band.npy comes from: python submit_fno/fit_band.py --ckpt ...)")
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
