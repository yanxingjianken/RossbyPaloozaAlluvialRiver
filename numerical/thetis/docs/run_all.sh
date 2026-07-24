#!/usr/bin/env bash
# Full pipeline: gates -> two runs -> IC figures -> movies.
# Assumes Phase 0 is green (build_env.sh; sentinel $OPT/FIREDRAKE_DONE).
set -uo pipefail
cd "$(dirname "$0")"
FD="micromamba run -n firedrake"
PP="micromamba run -n fourcastnetv2"

step(){ echo; echo "=============== $* ==============="; date; }
die(){ echo "FAILED: $*"; exit 1; }

step "1. gates (tier 1 + 2)"
$FD python tests/test_setup.py || die "test_setup"

step "2. IC figures (vision-check these!)"
$PP python postprocessing/01_ic.py || die "01_ic"

step "3. runs"
for m in 4 8; do
  echo "--- m=$m ---"
  sed -i "s/^    n_wave=[0-9]*,/    n_wave=$m,/" meander_thetis.py
  $FD env OMP_NUM_THREADS=1 python meander_thetis.py || die "run m=$m"
done
sed -i "s/^    n_wave=[0-9]*,/    n_wave=4,/" meander_thetis.py   # restore default

step "4. movies"
$PP python postprocessing/02_bank_evolution.py m4 m8 || die "02_bank_evolution"

step "DONE"
ls -la figures/
