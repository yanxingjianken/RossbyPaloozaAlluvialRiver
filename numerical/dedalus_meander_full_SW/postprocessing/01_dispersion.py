#!/usr/bin/env python3
"""01: dispersion relation sigma(k), c(k) of the meander, measured from the IVP.

Reads the sweep runs in ../outputs/ (each stores sigma_meas, c_meas) and plots
the growth rate and phase speed vs meander wavenumber, grouped by Froude number.
The gravity-wave speed 1/F is marked -- if the meander branch tracks it, the
meander is gravity-coupled; if it is F-insensitive, it is the vortical/Rossby
branch.  Run the sweeps first, e.g.:

    python ../sw_sn_driver.py --mode sweep --Froude 0.3
    python ../sw_sn_driver.py --mode sweep --Froude 0.6
    python 01_dispersion.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pp_lib as PP

rows = PP.sweep_dispersion()
if not rows:
    raise SystemExit("no ../outputs/run_*.h5 -- run `sw_sn_driver.py --mode sweep` first")

Fs = sorted(set(r["F"] for r in rows))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Fs)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=120)
for F, col in zip(Fs, colors):
    sub = sorted([r for r in rows if r["F"] == F], key=lambda r: r["k"])
    k = np.array([r["k"] for r in sub])
    sig = np.array([r["sigma"] for r in sub])
    c = np.array([r["c"] for r in sub])
    Cb = sub[0]["Cbar"]
    lab = rf"$F={F:g}$ ($\bar C={Cb:.2g}$)"
    ax1.plot(k, sig, "o-", color=col, lw=1.8, ms=4, label=lab)
    ax2.plot(k, c, "o-", color=col, lw=1.8, ms=4, label=lab)
    ax2.axhline(1.0 / F, color=col, lw=1.0, ls=":", alpha=0.7)   # gravity speed 1/F

ax1.axhline(0, color="k", lw=0.6)
ax1.set_xlabel(r"meander wavenumber $k$"); ax1.set_ylabel(r"growth rate $\sigma$")
ax1.set_title("meander growth (>0 = unstable)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax2.set_xlabel(r"meander wavenumber $k$"); ax2.set_ylabel(r"phase speed $c$")
ax2.set_title(r"migration speed (dotted = gravity $1/F$)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig.suptitle("dedalus_meander_full_SW: meander dispersion relation (from IVP)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(PP.FIG_DIR, "fig01_dispersion.png")
fig.savefig(out)
print(f"wrote {os.path.relpath(out, PP.PKG)}  ({len(rows)} runs, {len(Fs)} Froude values)")
