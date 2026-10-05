#!/bin/bash
# sandbox.sh PKG WORK [NAME=HOSTPATH ...] -- COMMAND [ARG ...]
#
# Run COMMAND the way a third party would. The author's home and scratch directories are EMPTY inside, the environment
# is cleared, and only what the organizers would hold is visible, at neutral paths under $ORG:
#   $ORG/pkg   the extracted training package (read-write; the working directory)
#   $ORG/work  the output directory (read-write)
#   $ORG/data  the competition release            $ORG/kit  the Track 1 starting kit v9
#   $ORG/env   a Python 3.10 / torch 2.2.2 environment, first on PATH as python3
#   $ORG/NAME  every extra NAME=HOSTPATH, read-only (reference checkpoints and archives for the identity check)
# A hard-coded path of the author's therefore fails here as it would on the organizers' cluster.
set -uo pipefail
PKG=$(cd "$1" && pwd) || exit 2
WORK=$(cd "$2" && pwd) || exit 2
shift 2
ORG=/tmp/rp_org_${SLURM_JOB_ID:-$$}
mkdir -p "$ORG"/{pkg,work,data,kit,env,home/.local/lib/python3.10} || exit 2
BINDS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  n=${1%%=*}; s=${1#*=}; shift
  [ -e "$s" ] || { echo "sandbox.sh: no $s"; exit 2; }
  if [ -d "$s" ]; then mkdir -p "$ORG/$n"; else mkdir -p "$(dirname "$ORG/$n")" && : > "$ORG/$n"; fi
  BINDS+=(--ro-bind "$s" "$ORG/$n")
done
[ "${1:-}" = "--" ] && shift
QWU=/users/PAS1187/qwu
exec bwrap --bind / / --dev-bind /dev /dev --proc /proc \
  --bind "$PKG" "$ORG/pkg" --bind "$WORK" "$ORG/work" \
  --ro-bind $QWU/RealPDE "$ORG/data" \
  --ro-bind $QWU/RealPDE/official/t1_v9/realpde_t1_starting_kit_v9 "$ORG/kit" \
  --ro-bind $QWU/.conda/envs/realpde "$ORG/env" \
  --ro-bind $QWU/.local/lib/python3.10 "$ORG/home/.local/lib/python3.10" \
  "${BINDS[@]}" \
  --tmpfs $QWU --tmpfs /fs/scratch/PAS1187/qwu \
  --chdir "$ORG/pkg" \
  env -i ORG="$ORG" HOME="$ORG/home" USER="${USER:-org}" PATH="$ORG/env/bin:/usr/bin:/bin" LANG=C.UTF-8 \
      OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
      ${CUDA_VISIBLE_DEVICES:+CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} ${SLURM_JOB_ID:+SLURM_JOB_ID=$SLURM_JOB_ID} \
      "$@"
