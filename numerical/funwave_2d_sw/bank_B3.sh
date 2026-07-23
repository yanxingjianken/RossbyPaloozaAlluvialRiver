#!/usr/bin/env bash
# Detached pipeline for the ADDED case B3 (lam=1560, FIX A=92 m == B1's amplitude; gentle reach,
# R/W=7.05).  Runs independently of the B1+B2 pipeline (bank_all2.sh) so it does NOT touch their
# dirs.  Self-calibrates head_factor in-place (starts from B1's 1.856, a near-identical gentle
# reach), then full spin-up -> hot-start morph.  Movies are made by the unified final pass once
# BOTH pipelines finish, so this script makes none.  Sentinel B3_DONE / B3_FAILED.
set -u
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/funwave_2d_sw
MM="micromamba run -n fourcastnetv2 python -u"
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }
rm -f B3_DONE B3_FAILED

echo "[$(STAMP)] ===== B3 (lam=1560, fix A=92) : calibrate -> spin-up -> morph ====="
$MM - <<'PY'
import run_meander as rm, os, glob, shutil, pathlib, sys
import numpy as np
run = dict(rm.RUNS[2])                       # B3
assert run["tag"] == "B3", run
TOL = 0.03                                   # accept |U_mean - U_target|/U_target < 3% (U ~ sqrt(hf))
hf = run["head_factor"]
base = None
for it in range(4):
    r = dict(run); r["head_factor"] = hf
    c = rm.cfg_for(r)
    tag, meta, t_spin, nr = rm.write_case(1560.0, c)
    b = os.path.join(rm.RUN_DIR, tag)
    print(f"[B3] calib it{it}: hf={hf:.4f} -> spin-up {t_spin:.0f}s, {nr} ranks ({tag})", flush=True)
    if not rm.launch(b, "spinup", nr, c):
        print("[B3] SPINUP FAILED", flush=True); sys.exit(1)
    m = rm.measure_speed(b, c)
    if not m:
        print("[B3] measure_speed FAILED", flush=True); sys.exit(1)
    err = abs(m["U_meas"] - c["U"]) / c["U"]
    print(f"[B3]   U_mean={m['U_meas']:.3f} (target {c['U']:.2f}), err={err*100:.1f}%, "
          f"hf_new={m['hf_new']:.4f}", flush=True)
    if err < TOL:
        base = b; hf_final = hf
        print(f"[B3] CONVERGED at hf={hf:.4f} (U within {TOL*100:.0f}%)", flush=True); break
    hf = m["hf_new"]
else:
    # did not converge in 4 tries: use the last (closest) spin-up rather than fail outright
    base = b; hf_final = hf
    print(f"[B3] not within {TOL*100:.0f}% after 4 tries; proceeding with hf={hf_final:.4f}", flush=True)

# discard the throwaway (non-converged) B3 spin-up dirs so the movie glob sees only the real one
for d in glob.glob(os.path.join(rm.RUN_DIR, "lam1560_C1p42*")):
    if os.path.abspath(d) != os.path.abspath(base):
        shutil.rmtree(d, ignore_errors=True)
print(f"[B3] kept {os.path.basename(base)}; final head_factor {hf_final:.4f}", flush=True)

# hot-start the morph phase from the last spin-up snapshot (mirrors bank_all2.sh)
c = rm.cfg_for({**run, "head_factor": hf_final})
snaps = [p for p in sorted(glob.glob(base + '/spinup/output/u_*')) if '99999' not in p]
last = os.path.basename(snaps[-1]).split('_')[1]
p = base + '/morph/input.txt'; t = pathlib.Path(p).read_text()
for vv in ('eta', 'u', 'v'):
    t = t.replace(f'@INI@/{vv}.txt', f'../spinup/output/{vv}_{last}')
pathlib.Path(p).write_text(t)
nr = int(np.load(base + '/bathy/grid.npz')['nranks'])
ok = rm.launch(base, 'morph', nr, c)
n = len([q for q in glob.glob(base + '/morph/output/eta_*') if '99999' not in q])
print(f"[B3] morph {'OK' if ok else 'STOPPED'}: {n} frames  (final hf={hf_final:.4f})", flush=True)
# record the converged head_factor so RUNS[2] can be updated for the record
pathlib.Path('B3_HF.txt').write_text(f"{hf_final:.4f}\n")
PY
rc=$?
if [ $rc -ne 0 ]; then echo "[$(STAMP)] B3 FAILED (rc=$rc)"; touch B3_FAILED; exit 1; fi
echo "[$(STAMP)] ===== B3 DONE ====="
touch B3_DONE