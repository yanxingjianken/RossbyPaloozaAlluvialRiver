#!/usr/bin/env python3
"""01: dispersion relation sigma(k), c(k) — from ONE broadband run per configuration.

The perturbation is broadband (all wavelengths excited at once), and with a straight
base channel the s-Fourier modes decouple exactly, so a single run already contains
sigma(k) and c(k) for every resolvable k.  The driver stores them as disp_* datasets.

Runs are identified by their PHYSICS — bed H, initial bank sinuosity, bottom friction
Cf, base speed U0 — never by "which wavelength was perturbed".

Modes that did not achieve >=3 e-foldings are drawn hollow: their fitted growth rate is
transient contamination, not an eigenvalue, and must not be read as physics.

    python 01_dispersion.py            # every outputs/run_*.h5
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pp_lib as PP

# measured in the parameter study: the short-wave cutoff is set by the SHEAR, not by nu
KD_LAW = 1.10

runs = sorted(glob.glob(os.path.join(PP.OUT_DIR, "run_*.h5")))
if not runs:
    raise SystemExit("no ../outputs/run_*.h5 — run ../sw_sn_driver.py first")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=120)

# Colour by Delta, not by filename order.  Delta is what controls the short-wave cutoff
# (k_peak*Delta ~ 1.10), so an arbitrary per-file colour hides the one systematic the
# figure exists to show.
_D = [float(PP.load_run(p)["attrs"]["Delta"]) for p in runs]
_norm = plt.Normalize(min(_D + [-0.6]), max(_D + [0.6]))
colors = [plt.cm.coolwarm_r(_norm(d)) for d in _D]

for path, col in zip(runs, colors):
    r = PP.load_run(path)
    a = r["attrs"]
    if "disp_k" not in r:
        continue
    k, sg, c = r["disp_k"], r["disp_sigma"], r["disp_c"]
    ok = r["disp_converged"] > 0
    lab = (f"H={'flat' if float(a['cross_amp'])==0 else float(a['cross_amp']):}"
           f", bank={float(a['bank_sinuosity']):.3g}"
           f", $C_f$={float(a['Cf']):.3g}"
           f", $U_0$={float(a['U0']):.2g}, $\\Delta$={float(a['Delta']):+.2g}")
    ax1.plot(k[ok], sg[ok], "o-", color=col, lw=1.8, ms=4, label=lab)
    ax1.plot(k[~ok], sg[~ok], "o", mfc="none", color=col, ms=4, alpha=0.5)
    # the measured law k_peak * Delta ~ 1.10: mark where it PREDICTS this run's peak,
    # so the figure tests the claim instead of merely being consistent with it
    D = float(a["Delta"])
    if D > 0.05 and np.any(ok):
        ax1.axvline(KD_LAW / D, color=col, ls=":", lw=1.2, alpha=0.85)
    ax2.plot(k[ok], c[ok], "o-", color=col, lw=1.8, ms=4, label=lab)
    ax2.plot(k[~ok], c[~ok], "o", mfc="none", color=col, ms=4, alpha=0.5)
    # Doppler-shifted gravity band: the correct reference is Ubar +/- 1/F, not 1/F
    F = float(a["Froude"]); U0 = float(a["U0"]); Uc = U0 + float(a["Delta"])
    for sgn in (+1, -1):
        ax2.axhspan(min(U0, Uc) + sgn / F, max(U0, Uc) + sgn / F,
                    color=col, alpha=0.07)

ax1.axhline(0, color="k", lw=0.6)
ax1.set_xlabel(r"wavenumber $k$"); ax1.set_ylabel(r"growth rate $\sigma$")
ax1.set_title(r"meander growth — colour = $\Delta$ (the shear);  dotted = "
              rf"$k_{{\rm peak}}={KD_LAW:g}/\Delta$;  hollow = <3 e-foldings",
              fontsize=9.5)
ax1.legend(fontsize=6.5, loc="upper left"); ax1.grid(alpha=0.3)
ax2.set_xlabel(r"wavenumber $k$"); ax2.set_ylabel(r"phase speed $c$")
ax2.set_title(r"migration speed (shaded = Doppler gravity bands $\bar U\pm1/F$)", fontsize=9.5)
ax2.legend(fontsize=6.5); ax2.grid(alpha=0.3)
fig.suptitle("dedalus_meander_full_SW — dispersion from a broadband perturbation, one run "
             "per configuration.\nStrongly sheared runs ($\\Delta{=}0.6$) peak at "
             "$k\\simeq1.8$ and turn over; weak/zero/reversed shear rises to the grid edge "
             "(those $\\sigma_{\\max}$ are resolution-limited, not physical).",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(PP.FIG_DIR, "fig01_dispersion.png")
fig.savefig(out)
print(f"wrote {os.path.relpath(out, PP.PKG)}  ({len(runs)} configuration(s))")
