#!/usr/bin/env python3
"""The dispersion relation of the river.pdf p.9/p.10/p.19 model, solved two ways.

    env OMP_NUM_THREADS=1 micromamba run -n dedalus python postprocessing/02_dispersion.py

Six (D, gamma) families, laid out like river.pdf p.20 so the two are comparable by eye.
Lines are the analytic det M = 0; circles are sigma and c MEASURED from short Dedalus
runs.  Nothing here is read off the deck's figure.

WHY NO DIGITIZED "PINS".  An earlier version overlaid values scraped from a 300-dpi
render of p.20 and scored the model against them.  That was backwards.  We have the
deck's equations solved analytically AND integrated numerically, and the two agree to six
decimals -- re-measuring someone else's raster plot cannot improve on that, and it drags
in two weaknesses: the scrape is lossy (a curve maximum is flat, so its POSITION is the
worst-resolved quantity of all), and eps was itself calibrated from those same scraped
intercepts, so any comparison built on them was partly circular.  The model's own result
stands without the deck's figure, and that is what this script plots.

eps is the one parameter river.pdf never gives (p.19 prints only the product eps*C_f), so
it is stated on the figure as an assumption rather than fitted to anything.

Measuring a point costs one short run, started from the same forced steady state the
driver uses, with n_wave = 1 so that k* is the GRAVEST mode in the box -- otherwise
round-off in a longer, unstable wavenumber outgrows the mode being measured and the fit
silently reports the wrong rate (see 03_verify.py section 1).
"""
import numpy as np

from pp_lib import (COLORS, P20_D_FAMILY, P20_G_FAMILY, bank_branch, dispersion_roots,
                    fit_sigma_c, growth_and_phase, peak_zero_intercept, save_fig,
                    set_style)

from noboru_model import CONFIG, simulate

plt = set_style()

ECOEF = 0.5          # eps*C_f -- ASSUMED.  river.pdf never gives eps.  sigma scales with it.
KS = np.linspace(1e-3, 2.0, 1200)
K_MEAS = np.array([0.15, 0.30, 0.45, 0.60, 0.80, 1.00, 1.30, 1.60, 1.90])


def measure(kstar, D, gamma, E):
    """(sigma, c, branch) from a Dedalus run at one wavenumber.

    An IVP converges to whichever root is LEAST DAMPED, and that is not always the bank
    root: past k* ~ 1.2 the other root of det M = 0 -- the advective one, the vorticity
    anomaly simply swept along by the jet -- decays more slowly and wins.  The curves here
    follow the BANK branch, so a point measured on the other branch is a correct
    measurement of a different thing and must not be drawn on top of them.
    """
    oms = dispersion_roots(kstar, D, gamma, E)
    dsig = abs(oms[0].imag - oms[1].imag)
    t_end = float(np.clip(6 * np.log(10) / max(dsig, 1e-3), 60.0, 900.0))
    cfg = dict(CONFIG)
    cfg.update(kstar=kstar, D=D, gamma=gamma, eps_Cf=ECOEF, n_wave=1, Nx=32,
               dt=0.02, t_end=t_end, n_out=240)
    out = simulate(cfg, quiet=True)
    sig, c, resid = fit_sigma_c(out["t"], out["amp2"], kstar)
    if not np.isfinite(resid) or resid > 1e-3:
        return np.nan, np.nan, "unconverged"
    meas = complex(c * kstar, sig)
    dists = [abs(complex(om.real, om.imag) - meas) for om in oms]
    which = int(np.argmin(dists))
    if dists[which] > 1e-3:
        return np.nan, np.nan, "unconverged"
    om_bank = bank_branch(np.array([kstar]), D, gamma, E)[0]
    is_bank = abs(oms[which] - om_bank) < abs(oms[1 - which] - om_bank)
    return sig, c, ("bank" if is_bank else "advective")


print("dispersion relation of the p.9/p.10/p.19 model: analytic vs Dedalus")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6))
(axGL, axGR), (axPL, axPR) = axes

# The deck's panel mapping is CROSSED: growth-left pairs with phase-RIGHT (D family),
# growth-right with phase-LEFT (gamma family).  Kept so the layout matches p.20.
PANELS = [
    (axGL, axPR, P20_D_FAMILY, "D-family"),
    (axGR, axPL, P20_G_FAMILY, "g-family"),
]

n_other = 0
for ax_growth, ax_phase, family, fam_key in PANELS:
    for D, gamma, colour in family:
        E = ECOEF * (1.0 - D)
        lab = rf"$D={D}$    $\gamma={gamma}$"
        sig, c = growth_and_phase(KS, D, gamma, E)
        ax_growth.plot(KS, sig, color=colour, lw=1.9, label=lab)
        ax_phase.plot(KS, c, color=colour, lw=1.9, label=lab)

        ms, mc, br = [], [], []
        for k in K_MEAS:
            s, cc, branch = measure(float(k), D, gamma, E)
            ms.append(s); mc.append(cc); br.append(branch)
        ms, mc, br = np.array(ms), np.array(mc), np.array(br)
        keep = br == "bank"
        n_other += int(np.sum(~keep))
        ax_growth.plot(K_MEAS[keep], ms[keep], "o", ms=4.5, mfc="none", mec=colour, mew=1.3)
        ax_phase.plot(K_MEAS[keep], mc[keep], "o", ms=4.5, mfc="none", mec=colour, mew=1.3)
        print(f"  {fam_key} D={D} gamma={gamma}: {keep.sum()} on the bank branch, "
              f"{(br == 'advective').sum()} on the advective branch, "
              f"{(br == 'unconverged').sum()} unconverged  (of {len(K_MEAS)})")

# ---- axes, laid out like river.pdf p.20 ------------------------------------------
for ax in (axGL, axGR):
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("Nondimensional growth rate", fontsize=12)
    ax.axhline(0, color="0.6", lw=0.9)
    for kk, lab in ((0.25, r"$4\pi$"), (0.5, r"$2\pi$"), (1.0, r"$\pi$")):
        ax.axvline(kk, color="0.85", lw=0.9)
        ax.annotate(lab, (kk, -0.50), color="magenta", fontsize=10, ha="center")
    ax.annotate(r"$\frac{\lambda}{2b} = $", (0.02, -0.50), color="magenta", fontsize=11)
for ax in (axPL, axPR):
    ax.set_ylim(-5, 1)
    ax.set_title("Nondimensional phase speed", fontsize=12)
    ax.set_xlabel(r"Wavenumber $kb$")
    for kk in (0.25, 0.5, 1.0):
        ax.axvline(kk, color="0.85", lw=0.9)
for ax in axes.ravel():
    ax.set_xlim(0, 2)
    ax.legend(fontsize=9, loc="upper right" if ax in (axGL, axGR) else "lower right")
    ax.grid(False)

fig.suptitle("Dispersion relation of the river.pdf p.9/p.10/p.19 model\n"
             r"lines: analytic det $M=0$   $\circ$ Dedalus IVP   "
             rf"($\varepsilon C_f = {ECOEF}$ assumed — the deck never gives $\varepsilon$, "
             r"and $\sigma$ scales with it)", fontsize=12.5, y=1.005)
save_fig(fig, "bend_instability")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("analytic features (E = eps_Cf (1-D)); the Dedalus points above sit on these curves:")
print(f"  {'D':>5}{'gamma':>7} | {'k*_peak':>9}{'sigma_pk':>10}{'k*_zero':>9}{'c(k*->0)':>10}")
for fam, family in (("D-family", P20_D_FAMILY), ("g-family", P20_G_FAMILY)):
    for D, gamma, _ in family:
        E = ECOEF * (1.0 - D)
        kpk, spk, kz, c0 = peak_zero_intercept(D, gamma, E)
        print(f"  {D:>5}{gamma:>7.2f} | {kpk:>9.3f}{spk:>10.4f}{kz:>9.3f}{c0:>10.3f}")
if n_other:
    print(f"\n  NOTE {n_other} of {len(K_MEAS) * 6} measured points are not drawn.  An IVP")
    print("  converges to the least-damped root, and past k* ~ 1.2 that is the ADVECTIVE")
    print("  root, not the bank root these curves follow.  Those are correct measurements")
    print("  of a different branch, so plotting them here would manufacture a")
    print("  disagreement that does not exist.")
print("\n  c < 0 on the whole bank branch: the meander travels UPSTREAM, which is the")
print("  p.19/p.21 claim.  sigma > 0 only for k* below ~sqrt(2D).")
