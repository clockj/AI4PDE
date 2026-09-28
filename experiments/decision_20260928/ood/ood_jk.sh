#!/bin/bash
# Chunk 5, part 1: is T4's ensemble gain at RE a property of the arm or of one seed set?  (no training)
#   control leave-one-out: F-FNO + sim s1-4 minus one, + VITP s1 + VITC s3          re_jk_ctl_d<k>
#   T4 leave-one-out:      resim s1-4 minus one,       + the same two ViTs           re_jk_t4_d<k>
#   control reseeding:     four of the six control seeds (s3-6, s1,2,5,6, s1,3,4,6) + the two ViTs   re_draw_s<seeds>
# Each is a full run (its own band refit, build, frozen / control on RE, in-distribution read). NCHAIN afterany chains.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
N=${1:-4}
C=/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/ckpt/RE
T=/fs/scratch/PAS1187/qwu/tmp/realpde/dec0928/t4/ckpt/RE
VIT="$C/re_vitp_s1.pth $C/re_vitc_s3.pth"
declare -a TAIL; k=0
sub() {   # sub NAME "members"
  local c=$((k % N)); k=$((k + 1))
  out=$(env -u OOD_BAND -u OOD_BANDSTYLE -u OOD_SKIP_EXISTING -u OOD_VARIANT_SPECS OOD_MEMBERS="$2" \
        OOD_STAGES="band build variants stream idread report" OOD_VARIANTS="frozen control" OOD_IDREAD_VARIANTS="frozen control" \
        OOD_CSET=0 OOD_REBLANK= OOD_DEPEND="${TAIL[$c]:-}" bash "$HERE/ood_submit.sh" "$1" RE list 01:00:00) \
        || { echo "submit failed: $1"; exit 1; }
  echo "$out  (chain $c)"; TAIL[$c]=$(awk '/^SUBMITTED/ {print $2}' <<< "$out")
}
for d in 1 2 3 4; do
  cm=""; tm=""
  for s in 1 2 3 4; do [ "$s" = "$d" ] && continue; cm="$cm $C/re_ffnosim_s$s.pth"; tm="$tm $T/re_ffnosim_resim_s$s.pth"; done
  sub re_jk_ctl_d$d "${cm# } $VIT"
  sub re_jk_t4_d$d "${tm# } $VIT"
done
for set_ in "3 4 5 6" "1 2 5 6" "1 3 4 6"; do
  m=""; for s in $set_; do m="$m $C/re_ffnosim_s$s.pth"; done
  sub "re_draw_s${set_// /}" "${m# } $VIT"
done
echo "TAILS ${TAIL[*]}"
