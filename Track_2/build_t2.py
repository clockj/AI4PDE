"""Assemble the Track-2 LTTTA archive.

    python build_t2.py --ckpt <members...> --band <band.npz> --out submission_t2.zip

Layout at the archive ROOT, which is what the platform extracts and passes as
``submission_dir``:

    submission.py   get_ttt_model / reset_ttt_state / ttt_step
    _fno.py         self-contained FNO2d, UNetTwoHead, StackedFNO, fuse, load_any
    band.npz        base / sigma_ref / p -- the SPS half-widths
    fno.pth, fno_2.pth, ...   members, in the order given

Names are positional: ``_fno.load_any`` reads each blob's own ``arch`` field, so a mixed
ensemble packs under the same ``fno*.pth`` names.  The evaluation image
(``w3nhao/realpde-track2:v1``) ships torch and numpy, which is all this imports.
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_MB = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--band", default=os.path.join(HERE, "submit", "band8.npz"))
    ap.add_argument("--stage", default=os.path.join(HERE, "_pkg"))
    ap.add_argument("--out", default=os.path.join(HERE, "submission_t2.zip"))
    a = ap.parse_args()

    # `_fno.py` is a symlink to Track 1's, and is resolved below -- the two tracks must ship
    # the same model or the whole point of submitting one package to both boards is lost.
    t1_fno = os.path.join(HERE, os.pardir, "Track_1", "submit_fno", "_fno.py")
    if os.path.realpath(os.path.join(HERE, "submit", "_fno.py")) != os.path.realpath(t1_fno):
        raise SystemExit("submit/_fno.py is not Track 1's _fno.py; the tracks would diverge")

    names = ["fno.pth"] + [f"fno_{i}.pth" for i in range(2, len(a.ckpt) + 1)]
    files = [(os.path.join(HERE, "submit", "submission.py"), "submission.py"),
             (os.path.join(HERE, "submit", "_fno.py"), "_fno.py"),
             (a.band, "band.npz")] + list(zip(a.ckpt, names))
    if os.path.isdir(a.stage):
        shutil.rmtree(a.stage)
    os.makedirs(a.stage)
    for src, name in files:
        src = os.path.realpath(src)
        if not os.path.exists(src):
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, os.path.join(a.stage, name))
        if name.startswith("fno") and name.endswith(".pth"):
            print(f"  {os.path.basename(src):26s} -> {name}")
    mb = sum(os.path.getsize(os.path.join(a.stage, n)) for _, n in files) / 1e6
    if mb >= CAP_MB:
        raise SystemExit(f"extracted {mb:.1f} MB exceeds the {CAP_MB} MB cap")
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for _, name in files:
            z.write(os.path.join(a.stage, name), arcname=name)
    print(f"\nstaged {a.stage}\nwrote  {a.out}")
    print(f"  extracted {mb:.2f} MB / {CAP_MB} MB     zipped "
          f"{os.path.getsize(a.out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
