#!/usr/bin/env bash
# Phase 0 — build the Firedrake + Thetis environment on dolma.
#
# Firedrake is NOT on conda-forge and PETSc has no binary wheel, so PETSc is
# compiled from source.  Toolchain: system gcc/gfortran 11.4 + system OpenMPI
# 4.1.2 (/usr/bin/mpicc); only python/ninja/cmake/pkg-config come from
# conda-forge, so the two toolchains are never mixed.
#
# Run detached:
#   nohup setsid bash build_env.sh > /net/flood/data2/users/x_yan/tmp/fd_build.log 2>&1 &
# Sentinels: $OPT/FIREDRAKE_DONE  or  $OPT/FIREDRAKE_FAILED
set -uo pipefail

OPT=/net/flood/data2/users/x_yan/opt
THETIS_DIR=/net/flood/data2/users/x_yan/rossby_palooza/numerical/thetis
FDCFG=$OPT/firedrake-configure
ENVNAME=firedrake
ENVPREFIX=/net/flood/home/x_yan/.conda/envs/$ENVNAME
NJOBS=32

# Put the env on PATH once instead of calling `micromamba run` repeatedly --
# concurrent micromamba invocations fight over ~/.cache/mamba/proc.
export PATH="$ENVPREFIX/bin:$PATH"

rm -f "$OPT/FIREDRAKE_DONE" "$OPT/FIREDRAKE_FAILED"
mkdir -p "$OPT"

step() { echo; echo "=============== $* ==============="; date; }
die()  { echo "FAILED: $*"; date; touch "$OPT/FIREDRAKE_FAILED"; exit 1; }

step "0. fetch firedrake-configure"
curl -fsSL -o "$FDCFG" \
  https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-configure \
  || die "could not fetch firedrake-configure"

PETSC_VER=$(python "$FDCFG" --os unknown --show-petsc-version) \
  || die "could not read supported PETSc version"
echo "supported PETSc version: $PETSC_VER"

step "1. clone PETSc $PETSC_VER"
if [ ! -d "$OPT/petsc/.git" ]; then
  git clone --depth 1 --branch "$PETSC_VER" https://gitlab.com/petsc/petsc.git "$OPT/petsc" \
    || die "petsc clone"
else
  echo "petsc already cloned, reusing"
fi

export PETSC_DIR=$OPT/petsc
export PETSC_ARCH=arch-firedrake-default

step "2. configure PETSc"
cd "$PETSC_DIR" || die "cd petsc"
if [ ! -f "$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables" ]; then
  # firedrake-configure prints "PETSC_ARCH=... --opt --opt ..." on one line
  CONF_OPTS=$(python "$FDCFG" --os unknown \
              --show-petsc-configure-options) || die "configure-options"
  echo "configure options: $CONF_OPTS"
  # The option string contains QUOTED flags, e.g.
  #   --COPTFLAGS='-O3 -march=native -mtune=native'
  # so plain $CONF_OPTS word-splitting hands PETSc "--COPTFLAGS='-O3" and
  # configure rejects it.  `eval set --` re-parses the quoting correctly.
  eval "set -- $CONF_OPTS"
  for a in "$@"; do echo "  arg: $a"; done
  ./configure "$@" || die "petsc configure"
else
  echo "petsc already configured, reusing"
fi

step "3. make PETSc (-j$NJOBS)"
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all -j$NJOBS \
  || die "petsc make"
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" check || echo "WARN: petsc check reported issues (non-fatal)"

step "4. export firedrake env"
# --show-env emits KEY=VAL pairs on one line; PETSC_DIR is relative to CWD
cd "$OPT" || die "cd opt"
FD_ENV=$(python "$FDCFG" --os unknown --show-env) \
  || die "show-env"
echo "firedrake env: $FD_ENV"
for kv in $FD_ENV; do export "${kv?}"; done
export PETSC_DIR=$OPT/petsc          # make sure it is absolute
export HDF5_DIR=$PETSC_DIR/$PETSC_ARCH
echo "PETSC_DIR=$PETSC_DIR  PETSC_ARCH=$PETSC_ARCH  HDF5_DIR=$HDF5_DIR  CC=${CC:-unset}"

step "5. pip install firedrake"
pip install --no-binary h5py 'firedrake[check]' \
  || die "pip install firedrake"

step "6. clone + install Thetis"
if [ ! -d "$THETIS_DIR/thetis-src/.git" ]; then
  git clone --depth 1 https://github.com/thetisproject/thetis.git "$THETIS_DIR/thetis-src" \
    || die "thetis clone"
fi
pip install -e "$THETIS_DIR/thetis-src" || die "pip install thetis"

step "7. smoke test: firedrake-check"
firedrake-check || die "firedrake-check"

step "8. smoke test: import thetis + trivial 2D solve"
python -c "
from thetis import *
mesh = RectangleMesh(8, 4, 100.0, 50.0)
P1 = get_functionspace(mesh, 'CG', 1)
bathy = Function(P1).assign(2.0)
solver = solver2d.FlowSolver2d(mesh, bathy)
solver.options.simulation_export_time = 10.0
solver.options.simulation_end_time = 20.0
solver.options.timestep = 5.0
solver.options.no_exports = True
solver.options.swe_timestepper_type = 'CrankNicolson'
solver.assign_initial_conditions()
solver.iterate()
print('THETIS_SMOKE_OK')
" || die "thetis smoke test"

step "DONE"
touch "$OPT/FIREDRAKE_DONE"
echo "PHASE0_COMPLETE"
