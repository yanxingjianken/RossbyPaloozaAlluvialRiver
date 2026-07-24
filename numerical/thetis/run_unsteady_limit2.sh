#!/usr/bin/env bash
# UNSTEADY (CrankNicolson, family A) limit-2 test: alpha~1 with the flow dt RETAINED,
# so a QGPV/Rossby wave can PROPAGATE (the steady solver drops dt -> cannot).  Tests
# whether the meander then migrates UPSTREAM.  usage: bash run_unsteady_limit2.sh <m>
set -uo pipefail
cd "$(dirname "$0")"; ROOT="$PWD"
M=${1:?}; CASE=rossby_limit2_unsteady
WORK=/net/flood/data2/users/x_yan/tmp/thetis_run_${CASE}_m$M
rm -rf "$WORK"; mkdir -p "$WORK"; mkdir -p "$ROOT/experiments/$CASE"
cp -a geometry.py sw_note.py meander_thetis.py "$WORK/"
ln -sfn "$ROOT/experiments" "$WORK/experiments"
ln -sfn "$ROOT/../ikeda_1981" "$WORK/../ikeda_1981" 2>/dev/null || true
cd "$WORK"
micromamba run -n firedrake env OMP_NUM_THREADS=1 PYOP2_CACHE_DIR="$WORK/.cache" \
  THETIS_N_WAVE=$M THETIS_A_IKEDA=0.0 THETIS_JET_RATIO=1.0 THETIS_PTS_PER_WL=48 \
  THETIS_FLOW_SOLVER=cranknicolson THETIS_CASE=$CASE \
  python meander_thetis.py \
  && touch "$ROOT/experiments/$CASE/RUN_m$M.done" || touch "$ROOT/experiments/$CASE/RUN_m$M.failed"
