#!/bin/bash
# Assemble the Track 2 training package that goes to the organizers (final_package/README.md is the record).
#
#   TEAM="<team name>" CONTACT="<e-mail>" bash final_package/make_package.sh OUTDIR
#   DRAFT=1 bash final_package/make_package.sh OUTDIR          # placeholders left in README.md; for the checks only
#
# OUTDIR/realpde_track2_training_package/         the tree
# OUTDIR/realpde_track2_training_package.tar.gz   the same, and .zip: no symbolic link inside, so either survives any transport
#
# What goes in, and nothing else:
#   1. the files of allowlist.txt, byte for byte from commit REF -- the commit the uploaded archives rebuild from byte
#      for byte (REPRODUCE.md 13.5). The allowlist is every file the pipeline opened in a traced run (verify/job_trace.sh).
#   2. patches/*.patch, applied on top (harness files only; none enters a submission archive)
#   3. overlay/: run_track2.sh (the entry point) and README.md (the manual)
#   4. MANIFEST.sha256
# No checkpoint, no band, no data: the allowlist is refused if it names a binary artefact.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); MAIN=$(cd "$HERE/.." && pwd)
REF=fdac00b
NAME=realpde_track2_training_package
OUT=${1:?"usage: make_package.sh OUTDIR   (a scratch directory, never under \$HOME)"}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
case "$OUT/" in "$HOME"/*) echo "REFUSING: $OUT is under \$HOME (CLAUDE.md rule 10)"; exit 1 ;; esac
if [ "${DRAFT:-0}" != "1" ]; then
  : "${TEAM:?set TEAM=<team name as on Codabench>, or DRAFT=1}" "${CONTACT:?set CONTACT=<e-mail>, or DRAFT=1}"
fi
if grep -nE '\.(pth|pt|npz|npy|h5|zip|pkl|bin|ckpt|safetensors)$' "$HERE/allowlist.txt"; then
  echo "REFUSING: the allowlist names a binary artefact (above); the package is code only"; exit 1; fi

S=$OUT/$NAME
rm -rf "$S" "$S.tar.gz" "$S.zip"; mkdir -p "$S"
git -C "$MAIN" archive "$(git -C "$MAIN" rev-parse "$REF^{commit}")" -- $(grep -v '^#' "$HERE/allowlist.txt") | tar -x -C "$S"
( cd "$S" && for p in "$HERE"/patches/*.patch; do patch -s -p1 < "$p"; done )
cp "$HERE/overlay/run_track2.sh" "$HERE/overlay/README.md" "$S/"
if [ "${DRAFT:-0}" != "1" ]; then
  sed -i "s|@@TEAM@@|$TEAM|; s|@@CONTACT@@|$CONTACT|" "$S/README.md"
  if grep -n '@@[A-Z0-9]*@@' "$S/README.md"; then echo "REFUSING: README.md still has a placeholder (above)"; exit 1; fi
fi
L=$(find "$S" -type l); [ -z "$L" ] || { echo "REFUSING: symbolic link in the package: $L"; exit 1; }
B=$(find "$S" -type f \( -name '*.pth' -o -name '*.npz' -o -name '*.npy' -o -name '*.h5' -o -name '*.zip' \))
[ -z "$B" ] || { echo "REFUSING: binary artefact in the package: $B"; exit 1; }

# One timestamp (the commit's) and a sorted file list, so the two archives are the same bytes on every run.
T=$(git -C "$MAIN" log -1 --format=%ct "$REF")
( cd "$S" && find . -type f ! -name MANIFEST.sha256 | sed 's|^\./||' | LC_ALL=C sort | xargs -d '\n' sha256sum > MANIFEST.sha256 )
find "$S" -exec touch -h -d "@$T" {} +
( cd "$OUT" && tar --sort=name --mtime="@$T" --owner=0 --group=0 --numeric-owner -cf - "$NAME" | gzip -n > "$NAME.tar.gz" )
( cd "$OUT" && find "$NAME" | LC_ALL=C sort | TZ=UTC zip -q -X -@ "$NAME.zip" )

[ "${DRAFT:-0}" = "1" ] && D="   DRAFT (placeholders in README.md)" || D=""
echo "package $S: $(find "$S" -type f | wc -l) files, $(du -sh "$S" | cut -f1)   commit $(git -C "$MAIN" rev-parse --short "$REF")$D"
( cd "$OUT" && sha256sum "$NAME.tar.gz" "$NAME.zip" )
