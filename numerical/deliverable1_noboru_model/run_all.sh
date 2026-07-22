#!/usr/bin/env bash
# Build everything: simulations, then the verification gate, then the figures.
#
#   bash run_all.sh
#
# OMP_NUM_THREADS=1 is required -- Dedalus degrades badly (and warns) without it.
# 03_verify.py runs BEFORE any figure script and aborts the whole run if it fails, so a
# broken model can never quietly produce a good-looking movie.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RUN="env OMP_NUM_THREADS=1 micromamba run -n dedalus python"

echo "=== 1/4  simulations ==========================================================="
$RUN noboru_model.py

cd postprocessing
echo
echo "=== 2/4  verification gate ====================================================="
$RUN 03_verify.py

echo
echo "=== 3/4  movies ================================================================"
$RUN 01_movie.py

echo
echo "=== 4/4  dispersion (river.pdf p.20) ==========================================="
$RUN 02_dispersion.py


cd "$HERE"
echo
echo "================================================================================"
echo "done.  outputs:"
ls -1 figures/
