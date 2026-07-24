#!/usr/bin/env bash
# v2 end-to-end, unattended: IC plots -> B1 + B2 concurrently (MF=1, closure fields rebuilt from
# the evolving bed) -> gates -> all movies.  nohup-safe; sentinels V2_DONE / V2_FAILED.
#
#   nohup setsid bash v2_all.sh > /net/flood/data2/users/x_yan/tmp/v2.log 2>&1 &
#
# 64 ranks per case is the MEASURED efficiency knee (2.83 sim-s/wall-s; 128 ranks buys only 5%,
# 256 ranks is 2.7x SLOWER at 2413 cells/rank).  Both cases therefore run CONCURRENTLY at 64 ranks
# on 128 of the 384 cores -- ~10 h wall for both, versus ~20 h if run one after the other.
set -u
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/funwave_2d_sw
MM="micromamba run -n fourcastnetv2 python -u"
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
rm -f V2_DONE V2_FAILED

echo "[$(STAMP)] ===== 1/4  build geometry + IC plots ====="
# write_case for both cases so the IC plots have something to read
$MM - <<'PY'
import run_meander as rm
for i, r in enumerate(rm.RUNS):
    c = rm.cfg_for(r)
    tag, meta, t_spin, nr = rm.write_case(r["lam"], c)
    print(f"  {r['tag']}: {tag}  {meta['nx']}x{meta['ny']}  {nr} ranks  "
          f"spin-up {t_spin:.0f}s  morph {c['t_morph']:.0f}s (MF={c['Morph_factor']})", flush=True)
PY
$MM postprocessing/05_ic.py 2>&1 | grep -E "wrote|base flow" || echo "  (IC plots failed -- continuing)"

echo "[$(STAMP)] ===== 2/4  run all 4 cases concurrently: {A0,A2p89} x {k8,k4} ====="
# RUNS = [0]A0/k8(64) [1]A0/k4(128) [2]A2p89/k8(64) [3]A2p89/k4(128) = 384 ranks on 384 cores.
NCASES=$($MM -c "import run_meander as rm; print(len(rm.RUNS))")
PIDS=(); NAMES=()
for i in $(seq 0 $((NCASES-1))); do
  nm=$($MM -c "import run_meander as rm; r=rm.RUNS[$i]; print(r['group']+'_'+r['tag'])")
  $MM run_v2.py --case $i > /net/flood/data2/users/x_yan/tmp/v2_${nm}.log 2>&1 &
  PIDS+=($!); NAMES+=($nm)
  echo "  launched case $i ($nm) pid ${PIDS[-1]}"
done
NFAIL=0
for k in "${!PIDS[@]}"; do
  wait "${PIDS[$k]}"; rc=$?
  echo "[$(STAMP)] ${NAMES[$k]} exit=$rc"; tail -2 /net/flood/data2/users/x_yan/tmp/v2_${NAMES[$k]}.log
  [ "$rc" -ne 0 ] && NFAIL=$((NFAIL+1))
done
echo "[$(STAMP)] $NFAIL of $NCASES cases failed"
if [ "$NFAIL" -ge "$NCASES" ]; then echo "[$(STAMP)] ALL FAILED"; touch V2_FAILED; exit 1; fi

echo "[$(STAMP)] ===== 3/4  gates ====="
$MM postprocessing/01_validate.py > gates_v2.txt 2>&1 || true
tail -20 gates_v2.txt

echo "[$(STAMP)] ===== 4/4  movies ====="
# 07 = the 4-row per-case movie (momflux / velocity+vectors / FROUDE / yOz bank change)
$MM postprocessing/07_bank_evolution.py 2>&1 | grep -E "wrote|frame " | tail -4
$MM postprocessing/02_morphology.py     2>&1 | tail -1
$MM postprocessing/04_xsection.py       2>&1 | tail -1
$MM postprocessing/03_momflux.py        2>&1 | tail -1
$MM postprocessing/06_anomaly.py        2>&1 | grep wrote | tail -2

echo "[$(STAMP)] ===== DONE ====="
ls -la figures/*.png figures/*.mp4 2>/dev/null
touch V2_DONE
echo "[$(STAMP)] sentinel V2_DONE written."
