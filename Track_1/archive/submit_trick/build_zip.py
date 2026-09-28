"""Build submit_trick/submission.zip with submission.py at the zip root.

    python submit_trick/build_zip.py
"""
from __future__ import annotations

import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_CKPT = os.path.join(os.path.dirname(HERE), "unet", "fno_div_hi.pth")
FILES = ["submission.py", "_fno.py", "fno_div_hi.pth"]


def main() -> None:
    local_ckpt = os.path.join(HERE, "fno_div_hi.pth")
    if not os.path.exists(local_ckpt):
        shutil.copy2(SRC_CKPT, local_ckpt)

    out = os.path.join(HERE, "submission.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            p = os.path.join(HERE, f)
            if not os.path.exists(p):
                raise SystemExit(f"missing {f}")
            z.write(p, arcname=f)          # arcname without directories: zip ROOT
    print(f"built {out} ({os.path.getsize(out)/1e6:.2f} MB): {', '.join(FILES)}")


if __name__ == "__main__":
    main()
