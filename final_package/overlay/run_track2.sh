#!/bin/bash
# RealPDE Track 2 (LTTTA): from the released training data to the submission archive of our leaderboard row
# (T2_MRGP2_track2_MRGP2.zip, Main Development, final 82.055553). README.md next to this file is the manual.
#
#   RPDE=<data> RPDE_KIT=<kit> WORK=<dir> bash run_track2.sh                       # everything, in sequence
#   RPDE=<data> RPDE_KIT=<kit> WORK=<dir> RPDE_REPRO_STAGE=<stage> bash run_track2.sh   # one training stage (README, "In parallel")
#
#   RPDE      the competition release: holds train_real/train_real/*.h5 and train_sim/train_sim/*.h5
#   RPDE_KIT  the directory of the Track 1 starting kit v9 that holds the official scoring.py (band fit and gates use it)
#   WORK      a writable directory: checkpoints, caches, logs and the archive go here
#   PY        the Python interpreter (default: python3 on PATH); environment.yml lists what it needs
#
# Nothing fitted ships with this package: the eleven networks, the validation windows, the interval band and its
# constants are all produced here from the release. The directory of this script must be writable (the build writes
# its intermediate files into it), and one build runs per copy of it.
set -uo pipefail
cd "$(dirname "$0")"
die() { echo "run_track2.sh: $*" >&2; exit 1; }

[ -n "${RPDE:-}" ]     || die "set RPDE=<directory holding train_real/train_real and train_sim/train_sim>"
[ -n "${RPDE_KIT:-}" ] || die "set RPDE_KIT=<directory of the Track 1 starting kit v9, the one holding scoring.py>"
[ -n "${WORK:-}" ]     || die "set WORK=<writable directory for checkpoints, caches, logs and the archive>"
[ -d "$RPDE/train_real/train_real" ] || die "no $RPDE/train_real/train_real"
[ -d "$RPDE/train_sim/train_sim" ]   || die "no $RPDE/train_sim/train_sim"
[ -f "$RPDE_KIT/scoring.py" ]        || die "no scoring.py in $RPDE_KIT"
mkdir -p "$WORK" || die "cannot create $WORK"
RPDE=$(cd "$RPDE" && pwd); RPDE_KIT=$(cd "$RPDE_KIT" && pwd); WORK=$(cd "$WORK" && pwd)
export RPDE RPDE_KIT

PY=${PY:-python3}
PY=$(command -v "$PY") || die "no Python interpreter: set PY=<python with torch, numpy, scipy and h5py>"
export PY
"$PY" - <<'PE' || die "the interpreter $PY cannot run this package (see environment.yml)"
import sys
try:
    import torch, numpy, h5py, scipy.optimize          # scipy: the gates' score rule; checked now, not after the trainings
except Exception as e:
    sys.exit("missing module: %s" % e)
print("python %s  torch %s  numpy %s  h5py %s  scipy %s  cuda %s" % (
    sys.version.split()[0], torch.__version__, numpy.__version__, h5py.__version__, scipy.__version__,
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE"))
sys.exit(0 if torch.cuda.is_available() else "no CUDA device: training and the gates need one GPU")
PE
echo "data: $(ls "$RPDE"/train_real/train_real/*.h5 | wc -l) real and $(ls "$RPDE"/train_sim/train_sim/*.h5 | wc -l) sim trajectories (the release has 82 and 100)"

# Track 2 ships Track 1's _fno.py, and the build refuses a copy that is not that very file. The package carries no
# symbolic link, so that no transport can break one; it is made here.
L=Track_2/submit/_fno.py
if [ ! -L "$L" ] || [ "$(readlink "$L")" != "../../Track_1/submit_fno/_fno.py" ]; then
  rm -f "$L" && ln -s ../../Track_1/submit_fno/_fno.py "$L" || die "cannot create the symbolic link $L"
fi

export RPDE_SIMCACHE=${RPDE_SIMCACHE:-$WORK/simcache}     # the sim -> real map's cache; one directory per parallel job
export RPDE_REPRO_OUT=${RPDE_REPRO_OUT:-$WORK/ckpt}       # the eleven checkpoints and their training logs
export RPDE_REPRO_LOG=${RPDE_REPRO_LOG:-$WORK/log}        # band fit and build logs
export RPDE_REPRO_PKG=${RPDE_REPRO_PKG:-$WORK/pkg}        # the built archives
export RPDE_REPRO_FITBAND=1                               # fit the band from the twins trained here; none is shipped
mkdir -p "$RPDE_SIMCACHE" "$RPDE_REPRO_OUT" "$RPDE_REPRO_LOG" "$RPDE_REPRO_PKG" || die "cannot write under $WORK"

RPDE_REPRODUCE_GO=1 bash reproduce.sh final || die "reproduce.sh failed; the last lines above name the log to read"
[ -n "${RPDE_REPRO_STAGE:-}" ] && exit 0                  # a training stage only: nothing is built

Z=$RPDE_REPRO_PKG/track2_final_a080.zip
[ -f "$Z" ] || die "the build reported success but $Z is missing"
cp "$Z" "$WORK/submission_track2.zip" || die "cannot write $WORK/submission_track2.zip"
echo
echo "TRACK 2 SUBMISSION ARCHIVE: $WORK/submission_track2.zip"
sha256sum "$WORK/submission_track2.zip"
"$PY" -m zipfile -l "$WORK/submission_track2.zip"
echo "RUN_TRACK2 OK"
