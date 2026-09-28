"""Assemble the Track 1 submission archive.

Stages submission.py + model.py + unet_mse.pth at the archive ROOT (the layout the platform
expects) and writes submission.zip.

    python build_zip.py
    python validate_submission.py --dir _package --blocked   # verify what was packaged
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

# Whichever checkpoint is chosen is packaged under the fixed name unet_mse.pth, which is what
# submission.py loads. Keep the source name in the build log so the zip's provenance is clear.
DEFAULT_CKPT = os.path.join(TRACK1, "unet", "wp2i_s5.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint to package")
    args = ap.parse_args()

    FILES = [
        (os.path.join(HERE, "submission.py"), "submission.py"),
        (os.path.join(TRACK1, "unet", "model.py"), "model.py"),
        (args.ckpt, "unet_mse.pth"),
        (os.path.join(HERE, "band.npy"), "band.npy"),
    ]
    print(f"packaging checkpoint: {args.ckpt}")

    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    for src, name in FILES:
        if not os.path.exists(src):
            raise SystemExit(f"missing required file: {src}")
        shutil.copy2(src, os.path.join(STAGE, name))

    extracted_mb = sum(os.path.getsize(os.path.join(STAGE, n)) for _, n in FILES) / 1e6
    if extracted_mb >= SIZE_CAP_MB:
        raise SystemExit(f"extracted size {extracted_mb:.1f} MB exceeds the {SIZE_CAP_MB} MB cap")

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for _, name in FILES:
            z.write(os.path.join(STAGE, name), arcname=name)

    print(f"staged  {STAGE}")
    for _, name in FILES:
        print(f"  {name:16s} {os.path.getsize(os.path.join(STAGE, name))/1e6:8.2f} MB")
    print(f"\nwrote   {ZIP}")
    print(f"  extracted {extracted_mb:.2f} MB / {SIZE_CAP_MB} MB cap")
    print(f"  zipped    {os.path.getsize(ZIP)/1e6:.2f} MB")
    print("\nnext: python validate_submission.py --dir _package --blocked")


if __name__ == "__main__":
    main()
