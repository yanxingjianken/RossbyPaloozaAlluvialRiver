#!/usr/bin/env python3
"""Dispersion experiment: run the driver once per (wavenumber, Froude) pair.

Kept OUT of sw_sn_driver.py -- the driver does exactly ONE run of its own CONFIG.
This script imports it and calls that same single-run function in a loop, so the
physics lives in exactly one place.

    micromamba run -n dedalus env OMP_NUM_THREADS=1 python sweep_dispersion.py

Each run writes outputs/run_<tag>.h5 carrying its measured sigma and c; then
postprocessing/01_dispersion.py collects them into the dispersion figure.

Why sweep FROUDE and not just one value: F is an input parameter (the river's
U/sqrt(gH)), not the answer.  Both wave families exist at any F -- gravity waves
at speed 1/F, the vortical/Rossby branch at O(U).  Only by CHANGING F (here the
gravity speed moves 3.3 -> 1.1) and seeing whether sigma,c move can we tell which
branch the meander rides.  They do not move => the meander is the vortical wave.
"""
import numpy as np

import sw_sn_driver as M

# --- the experiment grid (edit here) --------------------------------------- #
KSTARS = np.linspace(0.15, 1.20, 12)      # meander wavenumbers to test
FROUDES = (0.3, 0.6, 0.9)                 # gravity speeds 1/F = 3.3, 1.7, 1.1
OVERRIDES = dict(Cbar_amp=0.15,           # finite base meander for the experiment
                 Cf=0.05, Ns=48, Nn=96, t_end=50.0)


def main():
    n_tot = len(KSTARS) * len(FROUDES)
    print(f"# dispersion sweep: {len(KSTARS)} wavenumbers x {len(FROUDES)} Froude "
          f"= {n_tot} runs")
    for F in FROUDES:
        print(f"\n## Froude = {F}  (gravity speed 1/F = {1/F:.2f})")
        for k in KSTARS:
            cfg = dict(M.CONFIG, kstar=float(k), Froude=float(F), **OVERRIDES)
            M.run_ivp_SW(cfg)
    print(f"\ndone: {n_tot} runs in outputs/ -- now run "
          f"postprocessing/01_dispersion.py")


if __name__ == "__main__":
    main()
