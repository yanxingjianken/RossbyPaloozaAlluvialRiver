#!/usr/bin/env bash
# Detached bank-migration pipeline: spin-up (from rest/base flow) -> erosion -> gates -> movies.
# nohup-safe; sentinel BANK_DONE / BANK_FAILED; log tmp/bank.log.
set -u
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/funwave_2d_sw
MM="micromamba run -n fourcastnetv2 python -u"
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
rm -f BANK_DONE BANK_FAILED

echo "[$(STAMP)] ===== 1/3 run: spin-up (base flow) -> morph (bank erosion) ====="
$MM bank_run.py || { echo "RUN FAILED"; touch BANK_FAILED; exit 1; }

echo "[$(STAMP)] ===== 2/3 gates ====="
$MM postprocessing/01_validate.py > gates_bank.txt 2>&1 || true
tail -25 gates_bank.txt

echo "[$(STAMP)] ===== 3/3 movies: bed (xOy) + cross-section (yOz) + momflux + total/anomaly flow ====="
$MM postprocessing/02_morphology.py 2>&1 | tail -1
$MM postprocessing/04_xsection.py   2>&1 | tail -1
$MM postprocessing/03_momflux.py    2>&1 | tail -1
$MM postprocessing/06_anomaly.py    2>&1 | tail -1 || echo "  (06_anomaly not ready / failed -- non-fatal)"

echo "[$(STAMP)] ===== DONE ====="
ls -la figures/*.mp4 2>/dev/null
touch BANK_DONE
echo "[$(STAMP)] sentinel BANK_DONE written."
