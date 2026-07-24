#!/usr/bin/env bash
# One case, serial, detached.  usage: bash run_case.sh <m> [A_ikeda]
set -uo pipefail
cd "$(dirname "$0")"; ROOT="$PWD"
M=${1:?usage: run_case.sh <m> [A_ikeda]}
A=${2:-0.0}
if [ "$A" = "0.0" ] || [ "$A" = "0" ]; then CASE=A0_incised
elif [ "$A" = "2.89" ];                 then CASE=A2p89_alluvial
else CASE="A$(echo "$A" | tr '.' 'p')"; fi
WORK=/net/flood/data2/users/x_yan/tmp/thetis_run_${CASE}_m$M
rm -rf "$WORK"; mkdir -p "$WORK"; mkdir -p "$ROOT/experiments/$CASE"
cp -a geometry.py sw_note.py meander_thetis.py "$WORK/"
ln -sfn "$ROOT/experiments" "$WORK/experiments"
ln -sfn "$ROOT/../ikeda_1981" "$WORK/../ikeda_1981" 2>/dev/null || true
cd "$WORK"
micromamba run -n firedrake env OMP_NUM_THREADS=1 PYOP2_CACHE_DIR="$WORK/.cache" \
  THETIS_N_WAVE=$M THETIS_A_IKEDA=$A THETIS_CASE=$CASE \
  python meander_thetis.py \
  && touch "$ROOT/experiments/$CASE/RUN_m$M.done" \
  || touch "$ROOT/experiments/$CASE/RUN_m$M.failed"
