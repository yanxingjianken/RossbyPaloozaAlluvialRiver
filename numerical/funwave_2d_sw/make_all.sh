#!/usr/bin/env bash
# End-to-end: geometry checks -> build -> run both cases concurrently -> gates ->
# figures -> movies.  Everything derived; nothing here is hand-made.
#
#     bash make_all.sh              # full pipeline (~90 min of FUNWAVE)
#     bash make_all.sh --smoke      # 1-frame movies, skips the FUNWAVE runs
#
# The gate verdict is captured and re-printed at the end: movies are produced either way,
# but a failed gate means they must not be read as results.
set -u
cd "$(dirname "$0")"
MM="micromamba run -n fourcastnetv2 python"
SMOKE=""
[ "${1:-}" = "--smoke" ] && SMOKE=1

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

say "1/6  geometry and base-state checks"
$MM tests/test_bathy.py || { echo "GEOMETRY CHECKS FAILED -- stopping"; exit 1; }

say "2/6  build cases"
$MM run_meander.py

if [ -z "$SMOKE" ]; then
  say "2b   calibrate head_factor ON THIS GEOMETRY (2 rounds), written back into RUNS"
  $MM -u run_meander.py --calibrate 2500 --apply || { echo 'CALIBRATION FAILED -- stopping'; exit 1; }
  $MM -u run_meander.py --calibrate 2500 --apply || { echo 'CALIBRATION FAILED -- stopping'; exit 1; }
  grep -E 'head_factor=[0-9.]+' run_meander.py | grep 'dict(tag=' | sed 's/^/  calibrated: /' 
fi

if [ -z "$SMOKE" ]; then
  say "3/6  run: spin-up (rigid bed) then morph (mobile bed), both cases concurrently"
  $MM -u run_meander.py --launch
else
  say "3/6  SKIPPED (smoke)"
fi

say "4/6  post-run gates"
$MM postprocessing/01_validate.py > gates.txt 2>&1
GATE=$?
tail -40 gates.txt

say "5/6  derivation figures + PDF"
$MM derivations/make_figs.py
( cd derivations && pdflatex -interaction=nonstopmode funwave_meander_model.tex >/dev/null 2>&1 \
                && pdflatex -interaction=nonstopmode funwave_meander_model.tex >/dev/null 2>&1 \
                && rm -f *.aux *.log *.out *.toc )
echo "  derivations/funwave_meander_model.pdf"

say "6/6  movies"
FR=""; [ -n "$SMOKE" ] && FR="--max-frames 1"
$MM postprocessing/02_morphology.py $FR
$MM postprocessing/03_momflux.py   $FR

say "SUMMARY"
if [ $GATE -eq 0 ]; then
  echo "  GATES PASSED -- the movies are readable as results."
else
  echo "  *** GATES FAILED *** -- see gates.txt.  The movies exist but must NOT be"
  echo "      read as results until the failing gate is understood."
  grep -E "^\s*\[FAIL\]" gates.txt | sed 's/^/      /'
fi
ls -la figures/*.mp4 2>/dev/null | sed 's/^/  /'
exit $GATE
