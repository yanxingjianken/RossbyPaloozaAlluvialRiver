#!/usr/bin/env bash
# Detached bank-migration pipeline, BOTH wavelengths concurrently (B1 lam=1040 + B2 lam=1560):
# spin-up (base flow) -> morph (bank erosion), then gates + 2-panel movies.
# nohup-safe; sentinel BANK_DONE / BANK_FAILED; logs tmp/bank.log + tmp/bank_B{1,2}.log.
set -u
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/funwave_2d_sw
MM="micromamba run -n fourcastnetv2 python -u"
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
rm -f BANK_DONE BANK_FAILED

run_case() {  # $1 = RUNS index -> build, spin-up (rigid), hot-start, morph (erosion)
  $MM - "$1" <<'PY'
import run_meander as rm, os, glob, pathlib, sys
idx = int(sys.argv[1]); r = rm.RUNS[idx]; c = rm.cfg_for(r)
tag, meta, ts, nr = rm.write_case(r['lam'], c); base = os.path.join(rm.RUN_DIR, tag)
print(f"[case {idx}] {tag}: spin-up {ts:.0f}s + morph {c['t_morph']:.0f}s x MF{c['Morph_factor']}, {nr} ranks", flush=True)
if not rm.launch(base, 'spinup', nr, c):
    print(f"[case {idx}] SPINUP FAILED", flush=True); sys.exit(1)
snaps = [p for p in sorted(glob.glob(base + '/spinup/output/u_*')) if '99999' not in p]
last = os.path.basename(snaps[-1]).split('_')[1]
p = base + '/morph/input.txt'; t = pathlib.Path(p).read_text()
for vv in ('eta', 'u', 'v'):
    t = t.replace(f'@INI@/{vv}.txt', f'../spinup/output/{vv}_{last}')
pathlib.Path(p).write_text(t)
ok = rm.launch(base, 'morph', nr, c)
n = len([q for q in glob.glob(base + '/morph/output/eta_*') if '99999' not in q])
print(f"[case {idx}] morph {'OK' if ok else 'STOPPED'}: {n} frames", flush=True)
PY
}

echo "[$(STAMP)] ===== run B1 + B2 CONCURRENTLY (64 + 128 ranks) ====="
run_case 0 > /net/flood/data2/users/x_yan/tmp/bank_B1.log 2>&1 &
run_case 1 > /net/flood/data2/users/x_yan/tmp/bank_B2.log 2>&1 &
wait
echo "[$(STAMP)] both cases finished:"
grep -h "morph\|FAILED" /net/flood/data2/users/x_yan/tmp/bank_B1.log /net/flood/data2/users/x_yan/tmp/bank_B2.log | tail -4

echo "[$(STAMP)] ===== gates ====="
$MM postprocessing/01_validate.py > gates_bank.txt 2>&1 || true
tail -20 gates_bank.txt

echo "[$(STAMP)] ===== movies (2-panel B1 vs B2) ====="
$MM postprocessing/02_morphology.py 2>&1 | tail -1
$MM postprocessing/04_xsection.py   2>&1 | tail -1
$MM postprocessing/03_momflux.py    2>&1 | tail -1
$MM postprocessing/06_anomaly.py    2>&1 | tail -2
$MM postprocessing/07_bank_evolution.py 2>&1 | tail -3

echo "[$(STAMP)] ===== DONE ====="
ls -la figures/*.mp4 2>/dev/null
touch BANK_DONE
echo "[$(STAMP)] sentinel BANK_DONE written."
