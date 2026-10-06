"""Build the Track 2 archive: the shipped code with its constants baked in, the six members
packed to int8 / fp16, and the band.

    python build.py --members <6 .pth in order> --band <sigblend npz> --sigblend <its json> \
        --stage <dir> --out <archive.zip>

`submission/submission.py` and `submission/_fno.py` are kept in the tree with their default
constants; the archive gets copies with the constants of the submission written in (the table
`CONSTANTS` below, plus `SIGPOW`, the two sigma exponents of the band fitted by
band/sigma_band.py). The members are stored with `_fno.pack_state_dict` (int8 weights with an
fp16 scale per channel pair, fp16 for the rest), which `_fno.dequant8` undoes at load. The
archive holds exactly: submission.py, _fno.py, band.npz, fno.pth, fno_2.pth ... fno_6.pth.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = os.path.join(HERE, "..", "submission")
CAP_MB = 256

# (file, regex of the line's prefix, value). "line" replaces the whole line; "token" replaces the
# value up to the first space and keeps the line's comment.
CONSTANTS = [
    ("submission.py", "ALPHA", "line", "0.80"),            # the fluctuation adjustment
    ("submission.py", "FUSE", "line", "True"),             # members of a family fused into one invocation
    ("submission.py", "AMP", "line", '"fp16"'),            # half precision, checked against fp32 at load
    ("submission.py", "SPEC_PATH", "line", '"bmm"'),       # the spectral product as a batched matmul
    ("submission.py", "JOINT_GAIN", "line", "True"),       # one adjustment gain per pixel for u and v
    ("submission.py", "NUMPY_OUT", "line", "True"),        # host-side output path
    ("submission.py", "ARENA_STEPS", "digits", "1024"),    # page-locked output arena
    ("submission.py", "R1", "line", "True"),               # online per-pixel width calibration
    ("submission.py", "R1_ASYM", "line", "True"),          # separate lower and upper widths
    ("submission.py", "R1_TSM", "token", "3"),             # the calibration table box-smoothed 3 x 3
    ("submission.py", "R1_BLK", "token", "4"),             # a second table per block of 4 lead frames
    ("submission.py", "R1_BLK_N0", "token", "20.0"),
    ("submission.py", "D1C_N0", "token", "30.0"),          # online time-mean bias removal
    ("submission.py", "D1C_EB", "token", "True"),          # ... with the empirical-Bayes shrink
    ("submission.py", "D1C_EB_N0", "token", "1.0"),
    ("submission.py", "D1C_LEAD_N0", "token", "5.0"),      # ... and the lead-time profile
    ("submission.py", "VC_N0", "token", "30.0"),           # online fluctuation-amplitude calibration
    ("_fno.py", "D2_DIRECT_PRED", "line", "True"),         # prediction written straight into the output
    ("_fno.py", "JOINT_GAIN", "line", "True"),
]
FFNO_ANCHOR = "_fno._SPEC = SPEC_PATH\n"
FFNO_BLOCK = '''# FFNO_PATH is BAKED like SPEC_PATH (09-14, the P2-L8 line): the spectral path every F-FNO member runs --
# "dense" (the 1-D FFTs as matmuls on a precomputed basis, fp16-native) or "fft" (the as-trained FFTs); None keeps
# the as-built path. Read by _fno.load_any; inert for every other family (verify/fuse_ffno.py).
FFNO_PATH = "dense"
_fno.FFNO_PATH = FFNO_PATH
'''


def bake(name: str, sigpow: str) -> str:
    """The shipped file `name` with the submission's constants written in."""
    src = open(os.path.join(SUBMISSION, name)).read()
    for f, key, how, value in CONSTANTS:
        if f != name:
            continue
        pat = {"line": rf"(?m)^{key} = .*$", "token": rf"(?m)^{key} = [^ ]*",
               "digits": rf"(?m)^{key} = [0-9]*"}[how]
        src, n = re.subn(pat, lambda m: f"{key} = {value}", src)
        if n != 1:
            raise SystemExit(f"{name}: expected one line '{key} = ...', found {n}")
    if name == "submission.py":
        if src.count(FFNO_ANCHOR) != 1:
            raise SystemExit("submission.py: the FFNO_PATH anchor is not unique")
        src = src.replace(FFNO_ANCHOR, FFNO_ANCHOR + FFNO_BLOCK)
        src, n = re.subn(r"(?m)^SIGPOW = None ", lambda m: f"SIGPOW = {sigpow} ", src)
        if n != 1:
            raise SystemExit(f"submission.py: expected one line 'SIGPOW = None ...', found {n}")
    ast.parse(src)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs=6, required=True,
                    help="the six all-data checkpoints: the four F-FNO seeds, then the two ViTs")
    ap.add_argument("--band", required=True, help="band_sigblend_lam0.5.npz")
    ap.add_argument("--sigblend", required=True, help="its json (the SIGPOW exponents)")
    ap.add_argument("--stage", required=True, help="directory the archive's files are staged in")
    ap.add_argument("--out", required=True, help="the archive to write")
    a = ap.parse_args()

    q = json.load(open(a.sigblend))["T2_sigpow"]
    sigpow = "(%r, %r)" % (float(q[0]), float(q[1]))
    if os.path.isdir(a.stage):
        shutil.rmtree(a.stage)
    os.makedirs(a.stage)
    for name in ("submission.py", "_fno.py"):
        with open(os.path.join(a.stage, name), "w") as f:
            f.write(bake(name, sigpow))
    shutil.copy2(a.band, os.path.join(a.stage, "band.npz"))

    sp = importlib.util.spec_from_file_location("_fno_pack", os.path.join(SUBMISSION, "_fno.py"))
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    import torch
    names = ["fno.pth"] + [f"fno_{i}.pth" for i in range(2, 7)]
    for src, name in zip(a.members, names):
        blob = torch.load(src, map_location="cpu")
        blob["model_state_dict"] = mod.pack_state_dict(blob["model_state_dict"], fp16=True, int8=True,
                                                       linear="tensor")
        torch.save(blob, os.path.join(a.stage, name))
        print(f"  {os.path.basename(src):32s} -> {name}")
    files = ["submission.py", "_fno.py", "band.npz"] + names
    mb = sum(os.path.getsize(os.path.join(a.stage, n)) for n in files) / 1e6
    if mb >= CAP_MB:
        raise SystemExit(f"extracted {mb:.1f} MB exceeds the {CAP_MB} MB cap")
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(os.path.join(a.stage, name), arcname=name)

    # Read every constant back out of the archive.
    with zipfile.ZipFile(a.out) as z:
        texts = {n: z.read(n).decode() for n in ("submission.py", "_fno.py")}
        for n in files:
            print(f"  {hashlib.sha256(z.read(n)).hexdigest()[:16]}  {z.getinfo(n).file_size:>10}  {n}")
    for f, key, how, value in CONSTANTS:
        if not re.search(rf"(?m)^{re.escape(key)} = {re.escape(value)}( |$)", texts[f]):
            raise SystemExit(f"{f}: {key} = {value} is not in the archive")
    for line in (f"SIGPOW = {sigpow} ", 'FFNO_PATH = "dense"\n'):
        if line not in texts["submission.py"]:
            raise SystemExit(f"submission.py: {line.strip()} is not in the archive")
    print(f"wrote {a.out}  ({mb:.2f} MB extracted; every constant read back)")


if __name__ == "__main__":
    main()
