#!/usr/bin/env python3
"""Bank-migration run: spin the full nonlinear SW up from the analytic BASE flow (u'=v'=0), then
turn on erosion and let the banks evolve.  Single case (lam=1040), gap-1 A=2.89 ON, interior fully
erodible, smooth-Zs buffers, short avalanching.  Reuses run_meander's build/launch machinery."""
import os, sys, glob, pathlib
sys.path.insert(0, "/net/flood/data2/users/x_yan/rossby_palooza/numerical/funwave_2d_sw")
import run_meander as rm

r = rm.RUNS[0]; c = rm.cfg_for(r)                     # B1, lam=1040, calibrated hf
tag, meta, ts, nr = rm.write_case(r["lam"], c)
base = os.path.join(rm.RUN_DIR, tag)
print(f"bank-migration {tag}: spin-up {ts:.0f}s + morph {c['t_morph']:.0f}s x MF{c['Morph_factor']} "
      f"(= {c['t_morph']*c['Morph_factor']/86400:.1f} morph-days), {nr} ranks", flush=True)

# Phase 1 — spin-up (rigid bed): the meander perturbation u',v' develops from the base flow
if not rm.launch(base, "spinup", nr, c):
    print("SPINUP FAILED", flush=True); sys.exit(1)

# hot-start the morph phase from the last steady spin-up snapshot
snaps = [p for p in sorted(glob.glob(os.path.join(base, "spinup", "output", "u_*"))) if "99999" not in p]
last = os.path.basename(snaps[-1]).split("_")[1]
p = os.path.join(base, "morph", "input.txt"); t = pathlib.Path(p).read_text()
for vv in ("eta", "u", "v"):
    t = t.replace(f"@INI@/{vv}.txt", f"../spinup/output/{vv}_{last}")
pathlib.Path(p).write_text(t)

# Phase 2 — erosion ON: banks retreat + deposit
ok = rm.launch(base, "morph", nr, c)
nfr = len([q for q in glob.glob(os.path.join(base, "morph", "output", "eta_*")) if "99999" not in q])
print(f"morph {'OK' if ok else 'STOPPED'}: {nfr} frames", flush=True)
