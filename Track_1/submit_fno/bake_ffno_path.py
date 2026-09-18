"""Bake FFNO_PATH, the F-FNO members' spectral path, into a track's `submission.py`.

    python Track_1/submit_fno/bake_ffno_path.py --t1 dense --t2 dense
    RPDE_FFNO_PATH=dense bash Track_1/submit_fno/build_round13.sh 0.60      # the usual caller

WHY THIS IS IN THE REPO.  It was written on 09-14 as a scratch-only helper
(`$SCRATCH/jobs/patch_ffno_path.py`) because nothing could be committed that day, and the P2-L8
build job called it by absolute path.  T2R13 -- the Track-2 record -- therefore shipped a constant
that `build_round13.sh` could not produce, so the round could not be rebuilt from the repository
alone.  The organizers re-train every shortlisted method from scratch on their own cluster before
scoring the private set, and a method they cannot reproduce is disqualified, so a build step that
lives only on scratch is a disqualification risk, not an inconvenience (REPRODUCE.md section 6).

WHAT IT DOES.  `_fno.load_any` calls `m.set_spectral_path(FFNO_PATH)` on every F-FNO member it
builds, and `StackedFFNO.set_spectral_path` does the same for the fused stack:

  "dense"  the 1-D FFTs as matmuls on a precomputed real basis -- fp16-native, and what wins at
           Track 1's batch 64 and at Track 2's batch-1 graph (both tracks bake dense today).
  "fft"    the as-trained `torch.fft` path.
  None     leave whatever the members were built with.

The constant is written next to `SPEC_PATH`, which it mirrors, because the evaluation container
sets no environment variables -- only baked constants ship.  It is inert for every other family.

The block is APPENDED after the `_fno._SPEC = SPEC_PATH` line rather than replacing a placeholder,
so a caller removes it the way it removes every other baked constant: `git checkout` on the two
`submission.py` files.  Running it twice REWRITES the value instead of refusing, so a build job
that bakes and restores repeatedly is safe.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FILES = {"t1": os.path.join(REPO, "Track_1", "submit_fno", "submission.py"),
         "t2": os.path.join(REPO, "Track_2", "submit", "submission.py")}
ANCHOR = "_fno._SPEC = SPEC_PATH\n"
BLOCK = '''# FFNO_PATH is BAKED like SPEC_PATH (09-14, the P2-L8 line): the spectral path every F-FNO member runs --
# "dense" (the 1-D FFTs as matmuls on a precomputed basis, fp16-native) or "fft" (the as-trained FFTs); None keeps
# the as-built path. Read by _fno.load_any; inert for every other family (verify/fuse_ffno.py).
FFNO_PATH = {value}
_fno.FFNO_PATH = FFNO_PATH
'''
EXISTING = re.compile(r"(?m)^FFNO_PATH = .*$")


def bake(track: str, value: str) -> str:
    path = FILES[track]
    src = open(path).read()
    lit = "None" if value == "none" else f'"{value}"'
    if EXISTING.search(src):
        # Already baked (a build that did not restore between runs): rewrite the value rather
        # than refuse, so the second build is not a different kind of build from the first.
        out = EXISTING.sub(f"FFNO_PATH = {lit}", src, count=1)
        how = "rewrote"
    else:
        if src.count(ANCHOR) != 1:
            sys.exit(f"{path}: expected exactly one {ANCHOR.strip()!r} line, found "
                     f"{src.count(ANCHOR)} -- refusing to guess where FFNO_PATH belongs")
        out = src.replace(ANCHOR, ANCHOR + BLOCK.format(value=lit))
        how = "added"
    ast.parse(out)
    tmp = path + ".tmp_ffno"
    with open(tmp, "w") as f:
        f.write(out)
    os.chmod(tmp, os.stat(path).st_mode)
    os.replace(tmp, path)          # atomic: a running job never reads a half-written file
    return f"{track}: {how} FFNO_PATH = {lit}  -> {os.path.relpath(path, REPO)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", choices=("dense", "fft", "none"))
    ap.add_argument("--t2", choices=("dense", "fft", "none"))
    a = ap.parse_args()
    if not (a.t1 or a.t2):
        sys.exit("nothing to do: pass --t1 and/or --t2")
    for track in ("t1", "t2"):
        v = getattr(a, track)
        if v:
            print(bake(track, v), flush=True)


if __name__ == "__main__":
    main()
