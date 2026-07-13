#!/usr/bin/env python3
"""fig10-12: the N-point channel eigenproblem (continuum generalisation).

fig10: N = 3 eigenvalues coincide with the 2x2 closure (both closures) plus
       the varicose bank mode om = -iE.
fig11: N-convergence of the peak growth rate vs the deck pin -- resolution
       does NOT close the peak gap (either closure).
fig12: most-unstable eigenfunction at N = 101.
"""
import numpy as np

from vorticity_lib import (COLORS, FRICTIONS, Params, ECOEF,
                           dispersion_roots, channel_modes, kstar_peak,
                           load_deck_pins, set_style, save_fig)

plt = set_style()

pR = Params(D=0.6, gamma=0.05, friction="rayleigh")
pin = [r for r in load_deck_pins()
       if r["family"] == "D-family" and r["D"] == 0.6][0]

# --------------------------------------------------------------- fig10 ---- #
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, fr in zip(axes, FRICTIONS):
    E = ECOEF[fr] * (1 - 0.6)
    k = 0.7
    om2 = dispersion_roots(k, 0.6, 0.05, E, fr)
    om3, _ = channel_modes(3, k, 0.6, 0.05, E, fr)
    ax.plot(om3.real, om3.imag, "o", ms=11, mfc="none", mec=COLORS["growth"],
            mew=2, label="N=3 GEP")
    ax.plot(om2.real, om2.imag, "x", ms=9, color="k", mew=2, label="2x2 closure")
    ax.plot([0], [-E], "+", ms=12, color=COLORS["erosion"], mew=2,
            label=r"varicose $\omega=-iE$")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.set_xlabel(r"Re$\,\omega^*$")
    ax.set_ylabel(r"Im$\,\omega^*$")
    ax.set_title(f"{fr}:  N=3 $\\equiv$ closure ($k^*={k}$)")
    ax.legend(fontsize=9)
save_fig(fig, "fig10_N3_equals_closure")

# --------------------------------------------------------------- fig11 ---- #
Ns = [3, 5, 7, 9, 11, 15, 21, 31, 51, 101]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for fr, col in (("rayleigh", COLORS["growth"]), ("momentum", COLORS["momentum"])):
    E = ECOEF[fr] * (1 - 0.6)
    p = Params(D=0.6, gamma=0.05, friction=fr)
    kpk, _, _ = kstar_peak(p)
    sig = []
    for N in Ns:
        ks = np.linspace(max(kpk - 0.25, 0.02), kpk + 0.25, 21)
        s = max(max(channel_modes(N, float(k), 0.6, 0.05, E, fr)[0].imag)
                for k in ks)
        sig.append(float(s))
    ax.plot(Ns, sig, "o-", color=col, lw=1.8, label=f"{fr} (N-point peak)")
ax.axhline(pin["sigma_peak"], color=COLORS["deckpin"], lw=2, ls="--",
           label="deck p.8 peak (pin)")
ax.set_xscale("log")
ax.set_xticks(Ns, [str(n) for n in Ns])
ax.set_xlabel("N (cross-channel points)")
ax.set_ylabel(r"$\sigma_{\rm pk}$")
ax.set_title(r"resolution does not close the peak gap ($D=0.6$, $\gamma=0.05$)")
ax.legend(fontsize=9)
save_fig(fig, "fig11_N_convergence")

# --------------------------------------------------------------- fig12 ---- #
N = 101
kpk, _, _ = kstar_peak(pR)
w, V = channel_modes(N, kpk, 0.6, 0.05, pR.E, "rayleigh")
i = int(np.argmax(w.imag))
phi = V[:, i]
ic = int(np.argmax(np.abs(phi)))
phi = phi / phi[N // 2]
y = np.linspace(-1, 1, N)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
axes[0].plot(np.abs(phi), y, color=COLORS["jet"], lw=2)
axes[0].set_xlabel(r"$|\hat\psi(y)|$ (centre-normalised)")
axes[0].set_ylabel(r"$y/b$")
axes[0].set_title("most-unstable eigenfunction, amplitude")
axes[1].plot(np.degrees(np.angle(phi)), y, color=COLORS["upstream"], lw=2)
axes[1].set_xlabel(r"phase$(\hat\psi)$ (deg)")
axes[1].set_title("phase tilt (friction-enabled momentum flux)")
fig.suptitle(rf"N=101 bank mode at $k^*={kpk:.2f}$ "
             rf"($D=0.6$, $\gamma=0.05$, rayleigh)", y=1.02)
save_fig(fig, "fig12_eigenfunction")

print("06_continuum_solver: done.")
