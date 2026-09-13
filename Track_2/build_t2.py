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
    ap.add_argument("--int8", action="store_true",
                    help="int8 the large tensors with an fp16 scale per (out, in) channel pair "
                         "and fp16 the rest, via _fno.pack_state_dict -- the same packer "
                         "Track_1/submit_fno/build_zip.py --int8 uses, and the inverse of "
                         "_fno.dequant8. A width-96 8-layer member is 108.5 MB fp32 / 54.3 fp16 "
                         "/ 29.0 int8, so EIGHT are 232 MB against the 256 MB cap where eight "
                         "fp16 would be 434. Verified end to end by "
                         "Track_1/verify/quant_identity.py.")
    ap.add_argument("--fp16", action="store_true",
                    help="store the members' floating tensors as float16, exactly as "
                         "Track_1/submit_fno/build_zip.py --fp16 does. Needed for width-96 "
                         "members: 8 x 57.0 MB is 456 MB against the 256 MB cap, 8 x 28.5 MB "
                         "is 228 MB. The two tracks MUST use the same setting or they stop "
                         "shipping the same model, which is the only reason to submit one "
                         "package to both boards.")
    ap.add_argument("--linear", choices=("tensor", "row", "fp16"), default="tensor",
                    help="what --int8 does to a 2-D weight -- see Track_1/submit_fno/"
                         "build_zip.py --linear and _fno.pack_state_dict. Must match Track 1's "
                         "setting, for the same reason --fp16 must.")
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
    # Same packer as Track 1's build_zip.py, loaded by path from the submission's own _fno.py
    # so the int8 format has exactly one implementation and exactly one reader.
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location(
        "_fno_pack", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "Track_1", "submit_fno", "_fno.py"))
    _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
    _pack = _m.pack_state_dict
    os.makedirs(a.stage)
    for src, name in files:
        src = os.path.realpath(src)
        if not os.path.exists(src):
            raise SystemExit(f"missing {src}")
        if (a.fp16 or a.int8) and name.startswith("fno") and name.endswith(".pth"):
            import torch
            blob = torch.load(src, map_location="cpu")
            blob["model_state_dict"] = _pack(blob["model_state_dict"],
                                             fp16=a.fp16 or a.int8, int8=a.int8,
                                             linear=a.linear)
            torch.save(blob, os.path.join(a.stage, name))
        else:
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
