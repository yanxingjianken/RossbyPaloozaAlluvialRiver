#!/usr/bin/env bash
# Re-run ONLY the morph phase for B1 + B2 with the sink-fixed binary (buffer no longer accretes
# and blows up), reusing the existing full spin-ups.  Then regenerate gates + all 3 movies.
# nohup-safe; sentinels MORPH_RERUN_DONE / _FAILED; log tmp/rerun_morph.log.
set -u
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/funwave_2d_sw
MM="micromamba run -n fourcastnetv2 python -u"
EXE=/net/flood/data2/users/x_yan/rossby_palooza/numerical/funwave_2d_sw/work_river/funwave-SEDIMENT-CHECK_MASS_CONSERVATION-MIXING--mpif90-parallel-double
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
rm -f MORPH_RERUN_DONE MORPH_RERUN_FAILED
B1=runs/lam1040_C3p00e-3_U0p85_hf1p856_D50500um_MF5
B2=runs/lam1560_C3p00e-3_U0p85_hf1p960_D50500um_MF5

run_one() {  # $1 = run dir.  Rank count is PER-CASE: read PX*PY from the input (B1=64, B2=128);
             # hardcoding 64 makes FUNWAVE STOP with "processors /= Px*Py" on the larger grid.
  local base px py nr; base=$(readlink -f "$1")
  px=$(grep -E "^PX" "$base/morph/input.txt" | grep -oE "[0-9]+")
  py=$(grep -E "^PY" "$base/morph/input.txt" | grep -oE "[0-9]+")
  nr=$((px * py))
  rm -rf "$base/morph_test" "$base/morph/output" "$base/morph/input_test.txt"
  mkdir -p "$base/morph/output"
  ( cd "$base/morph" && mpirun --oversubscribe -np "$nr" --mca btl_vader_single_copy_mechanism none \
       "$EXE" input.txt > run.log 2>&1 )
}

echo "[$(STAMP)] ===== re-run morph B1 + B2 concurrently (sink-fixed binary, reuse spin-ups) ====="
run_one "$B1" & P1=$!
run_one "$B2" & P2=$!
wait $P1 $P2
for b in "$B1" "$B2"; do
  echo "  $(basename $b): $(grep -c 'Normal Termination' $b/morph/run.log 2>/dev/null | tr -d '\n') normal-term, frames=$(ls $b/morph/output/eta_* 2>/dev/null | grep -v 99999 | wc -l), blowup=$(ls $b/morph/output/*99999* 2>/dev/null | wc -l)"
done

echo "[$(STAMP)] ===== gates ====="
$MM postprocessing/01_validate.py | tee gates_production.txt

echo "[$(STAMP)] ===== movies: morphology (xOy) + cross-section (yOz) + momentum flux ====="
$MM postprocessing/02_morphology.py || { echo FAIL morph movie; touch MORPH_RERUN_FAILED; }
$MM postprocessing/04_xsection.py   || { echo FAIL xsection movie; touch MORPH_RERUN_FAILED; }
$MM postprocessing/03_momflux.py    || { echo FAIL momflux movie; touch MORPH_RERUN_FAILED; }

echo "[$(STAMP)] ===== DONE ====="
ls -la figures/*.mp4 2>/dev/null
[ -f MORPH_RERUN_FAILED ] || touch MORPH_RERUN_DONE
echo "[$(STAMP)] sentinel written."
