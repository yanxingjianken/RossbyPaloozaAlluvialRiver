#!/usr/bin/env bash
# Regenerate every product from outputs/run_*.h5, in dependency order.
#
#   bash make_all_products.sh
#
# Run this AFTER experiments.py.  It never touches the simulations -- if a number
# in the PDF or a movie disagrees with the data, re-run this, not the sims.
set -euo pipefail
cd "$(dirname "$0")"
MM="micromamba run -n dedalus env OMP_NUM_THREADS=1"

echo "=== 1/4  results table (generated from the HDF5, never typed) ==="
$MM python derivations/make_results_table.py

echo "=== 2/4  dispersion figure ==="
(cd postprocessing && $MM python 01_dispersion.py)

echo "=== 3/4  absolute-Eulerian momentum-flux movies (one per configuration) ==="
(cd postprocessing && $MM python 02_eulerian_momflux.py)

echo "=== 4/4  derivation PDF (twice: \\ref/\\label need a second pass) ==="
cd derivations
lualatex -interaction=nonstopmode -halt-on-error sw_sn_meander.tex > /tmp/tex_pass1.log 2>&1
lualatex -interaction=nonstopmode -halt-on-error sw_sn_meander.tex > /tmp/tex_pass2.log 2>&1
grep -c "??" sw_sn_meander.log > /dev/null 2>&1 || true
if grep -q "LaTeX Warning: There were undefined references" sw_sn_meander.log; then
    echo "  WARNING: undefined references remain -- check sw_sn_meander.log"
fi
cd ..

echo
echo "=== done ==="
ls -la figures/ | tail -n +2
echo "PDF: $(ls -la derivations/sw_sn_meander.pdf | awk '{print $5" bytes  "$6" "$7" "$8}')"
