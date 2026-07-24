#!/usr/bin/env bash
# SW-note LIMIT 2 (QGPV/shear-Rossby): alpha~1 (high m) + low Froude -> gravity
# sub-leading, PV leading order.  Tests UPSTREAM meander migration in the 2D SWE.
# usage: bash run_limit2.sh <m>
set -uo pipefail
cd "$(dirname "$0")"; ROOT="$PWD"
M=${1:?usage: run_limit2.sh <m>}
CASE=rossby_limit2
WORK=/net/flood/data2/users/x_yan/tmp/thetis_run_${CASE}_m$M
rm -rf "$WORK"; mkdir -p "$WORK"; mkdir -p "$ROOT/experiments/$CASE"
cp -a geometry.py sw_note.py meander_thetis.py "$WORK/"
ln -sfn "$ROOT/experiments" "$WORK/experiments"
ln -sfn "$ROOT/../ikeda_1981" "$WORK/../ikeda_1981" 2>/dev/null || true
cd "$WORK"
micromamba run -n firedrake env OMP_NUM_THREADS=1 PYOP2_CACHE_DIR="$WORK/.cache" \
  THETIS_N_WAVE=$M THETIS_A_IKEDA=0.0 THETIS_F_REF=0.15 THETIS_JET_RATIO=1.0 \
  THETIS_PTS_PER_WL=76 THETIS_CASE=$CASE \
  python meander_thetis.py \
  && touch "$ROOT/experiments/$CASE/RUN_m$M.done" \
  || touch "$ROOT/experiments/$CASE/RUN_m$M.failed"
