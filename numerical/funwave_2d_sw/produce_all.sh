#!/usr/bin/env bash
# ============================================================================
# Detached end-to-end production for the two-wavelength meander experiment.
#   B1 (lam=1040) and B2 (lam=1560), identical apex curvature C0=A k^2, full
#   gap-1 secondary-flow physics (friction modulation + bedload deflection +
#   Talmon stabiliser).  Produces the three movies:
#     figures/morph_AB_*.mp4     xOy plan view, bed change, S1/S2 section marks
#     figures/xsection_AB_*.mp4  yOz transverse sections at S1/S2
#     figures/momflux_AB_*.mp4   cross-channel momentum flux
# nohup-safe: survives an HPC disconnect.  Writes PRODUCTION_DONE (or _FAILED)
# and logs to tmp/production.log.  Watch with:  tail -f tmp/production.log
# ============================================================================
set -u
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/funwave_2d_sw
MM="micromamba run -n fourcastnetv2 python -u"
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
rm -f PRODUCTION_DONE PRODUCTION_FAILED
fail() { echo "[$(STAMP)] *** FAILED: $1"; touch PRODUCTION_FAILED; exit 1; }

echo "[$(STAMP)] ===== waiting for the decisive ON/OFF validation test to finish ====="
while pgrep -f "g1ful[l].py" >/dev/null; do sleep 20; done
echo "[$(STAMP)] decisive test done.  Its verdict (validation of the gap-1 bedload deflection):"
echo "----- ON (full secondary-flow model) -----";  tail -6 /net/flood/data2/users/x_yan/tmp/g1full_logs/ON.log  2>/dev/null
echo "----- OFF (stock FUNWAVE) -----";              tail -6 /net/flood/data2/users/x_yan/tmp/g1full_logs/OFF.log 2>/dev/null

echo "[$(STAMP)] ===== 1/6 geometry checks ====="
$MM tests/test_bathy.py            || fail "geometry checks"

echo "[$(STAMP)] ===== 2/6 build B1 + B2 (full gap-1 physics) ====="
$MM run_meander.py                 || fail "build"

echo "[$(STAMP)] ===== 3/6 calibrate head_factor on this geometry (2 rounds) ====="
$MM run_meander.py --calibrate 2500 --apply || fail "calibrate round 1"
$MM run_meander.py --calibrate 2500 --apply || fail "calibrate round 2"

echo "[$(STAMP)] ===== 4/6 run B1 + B2: spin-up (rigid) then morph (mobile), concurrently ====="
$MM run_meander.py --launch        || fail "launch"

echo "[$(STAMP)] ===== 5/6 post-run gates ====="
$MM postprocessing/01_validate.py | tee gates_production.txt
GATE=${PIPESTATUS[0]}

echo "[$(STAMP)] ===== 6/6 figures: morphology (xOy) + cross-section (yOz) + momentum flux ====="
$MM postprocessing/02_morphology.py || fail "morphology movie"
$MM postprocessing/04_xsection.py   || fail "cross-section movie"
$MM postprocessing/03_momflux.py    || fail "momflux movie"

echo "[$(STAMP)] ===== derivation figures + PDF ====="
$MM derivations/make_figs.py 2>/dev/null || echo "  (make_figs skipped)"
( cd derivations && pdflatex -interaction=nonstopmode funwave_meander_model.tex >/dev/null 2>&1 \
                 && pdflatex -interaction=nonstopmode funwave_meander_model.tex >/dev/null 2>&1 \
                 && rm -f *.aux *.log *.out *.toc ) || echo "  (pdflatex skipped)"

echo "[$(STAMP)] ===== DONE ====="
echo "gate exit = $GATE  ($( [ $GATE -eq 0 ] && echo 'ALL GATES PASSED -- movies are readable as results' || echo 'SOME GATES FAILED -- see gates_production.txt' ))"
ls -la figures/*.mp4 2>/dev/null
touch PRODUCTION_DONE
echo "[$(STAMP)] sentinel PRODUCTION_DONE written."
