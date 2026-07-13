#!/usr/bin/env python3
"""fig01-02: base state of the vorticity-meander theory (deck p. 4).

fig01: parabolic jet, the three streamfunction levels, and the constant
       positive vorticity gradient (the channel beta-analogue).
fig02: why rigid banks are neutral (no inflection point) and the erodible
       bank is the engine (deck pp. 4, 7).
"""
import numpy as np

from vorticity_lib import (COLORS, u_profile, zeta_gradient, set_style,
                           save_fig)

plt = set_style()

D = 0.5
y = np.linspace(-1, 1, 401)
u = u_profile(y, D)

# --------------------------------------------------------------- fig01 ---- #
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8),
                         gridspec_kw={"width_ratios": [1.2, 1, 1]})

ax = axes[0]
ax.fill_betweenx(y, 0, u, color=COLORS["water_fill"], alpha=0.8)
ax.plot(u, y, color=COLORS["jet"], lw=2.5)
for yy in np.linspace(-0.9, 0.9, 10):
    ax.annotate("", xy=(float(u_profile(yy, D)), yy), xytext=(0, yy),
                arrowprops=dict(arrowstyle="->", color=COLORS["jet"], lw=1.0))
for yy, lab in ((1.0, r"$\psi_1\;(y=+b)$"), (0.0, r"$\psi_2\;(y=0)$"),
                (-1.0, r"$\psi_3\;(y=-b)$")):
    ax.axhline(yy, color="k", lw=0.8, ls=":" if yy else "--")
    ax.text(1.02, yy + 0.04, lab, fontsize=11)
ax.set_xlim(0, 1.35)
ax.set_xlabel(r"$\bar u(y)\,/\,(U_0+\Delta)$")
ax.set_ylabel(r"$y/b$")
ax.set_title(r"parabolic jet  $\bar u = U_0 + \frac{\Delta}{b^2}(b^2-y^2)$")

ax = axes[1]
ax.plot(-np.gradient(u, y), y, color=COLORS["upstream"], lw=2.5)
ax.set_xlabel(r"$\bar\zeta = -\bar u_y$")
ax.set_title(r"base vorticity  $\bar\zeta = \frac{2\Delta}{b^2}y$")
ax.axvline(0, color="k", lw=0.8)

ax = axes[2]
ax.plot(np.full_like(y, zeta_gradient(D)), y, color=COLORS["erosion"], lw=2.5)
ax.set_xlim(0, 2.4)
ax.set_xlabel(r"$\partial\bar\zeta/\partial y$  (nondim)")
ax.set_title(r"constant $\bar\zeta_y = 2D$:  the channel $\beta$")
for a in axes[1:]:
    a.set_ylabel(r"$y/b$")
fig.suptitle("Base state (deck p. 4): a channel jet with a "
             r"constant positive vorticity gradient  ($D=%.1f$)" % D, y=1.04)
save_fig(fig, "fig01_jet_levels_beta")

# --------------------------------------------------------------- fig02 ---- #
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

ax = axes[0]
ax.plot(-np.gradient(np.gradient(u, y), y), y, color=COLORS["decay"], lw=2.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(-0.5, 2.4)
ax.set_xlabel(r"$-\bar u_{yy}$")
ax.set_ylabel(r"$y/b$")
ax.set_title("no inflection point "
             r"($-\bar u_{yy}=2D>0$ everywhere)")
ax.text(0.15, -0.75, "Rayleigh criterion:\nrigid banks are NEUTRAL\n"
        r"(self-test: Im$\,\omega\leq 0$ for $\varepsilon=0$)",
        fontsize=10, color=COLORS["decay"])

ax = axes[1]
x = np.linspace(0, 4 * np.pi, 400)
for s, col, lab in ((1.0, COLORS["psi1"], r"bank $\psi_1$"),
                    (0.0, COLORS["psi2"], r"interior $\psi_2$")):
    ax.plot(x, s + 0.35 * np.sin(x - (0 if s else 0.9)), color=col, lw=2,
            label=lab)
ax.annotate("", xy=(2.4, 0.62), xytext=(2.4, 0.95),
            arrowprops=dict(arrowstyle="->", color=COLORS["erosion"], lw=2))
ax.text(2.55, 0.75, r"$\partial_t\psi_1'=\frac{\varepsilon C_f U_0}{b}"
        r"(\psi_2'-\psi_1')$" + "\n(bank relaxes toward interior, p. 7)",
        fontsize=10, color=COLORS["erosion"])
ax.legend(loc="lower left", fontsize=10)
ax.set_xlabel(r"downstream $x$")
ax.set_yticks([])
ax.set_title("the erodible bank is the engine")
save_fig(fig, "fig02_neutral_vs_bank_engine")

print("01_setup_schematic: done.")
