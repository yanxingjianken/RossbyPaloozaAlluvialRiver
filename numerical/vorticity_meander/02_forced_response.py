#!/usr/bin/env python3
"""fig03-05: forced steady state (deck pp. 5-6).

fig03: planforms of the forced response at k* = 0.3 and 1.5 (D = 0.5,
       gamma = 0) -- the deck p. 5 pair, resonant vs stiff interior.
fig04: |psihat_2/psihat_1| over (k*, D) with the k*^2 = 2D boundary.
fig05: friction phase lag and the lateral momentum flux (deck p. 6):
       mean(u'v')_y = (gamma/2D) mean(zeta'^2) > 0 at the centre.
"""
import numpy as np

from vorticity_lib import COLORS, forced_response, set_style, save_fig

plt = set_style()

D = 0.5

# --------------------------------------------------------------- fig03 ---- #
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
x = np.linspace(0, 25, 1200)          # in units of 2b (deck axis)
for ax, ks in zip(axes, (0.3, 1.5)):
    r = complex(forced_response(ks, D, 0.0))
    kx = ks * (2.0 * x)               # x is in 2b units; k* multiplies x/b
    amp = 0.55
    y1 = 1.0 + amp * np.cos(kx)
    y2 = 0.0 + amp * np.real(r * np.exp(1j * kx))
    y3 = -1.0 + amp * np.cos(kx)
    ax.plot(x, y1, color=COLORS["psi1"], lw=1.8, label=r"$\psi_1,\psi_3$ (banks)")
    ax.plot(x, y3, color=COLORS["psi1"], lw=1.8)
    ax.plot(x, y2, color=COLORS["psi2"], lw=1.8, label=r"$\psi_2$ (centre)")
    ax.annotate("", xy=(2.2, -3.6), xytext=(0.2, -3.6),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["jet"], lw=3))
    ax.set_ylim(-5, 5)
    ax.set_title(rf"$k^*={ks}$,  $D={D}$,  $\gamma=0$:   "
                 rf"$\hat\psi_2/\hat\psi_1 = {r.real:.2f}$"
                 + ("  (resonant band $k^{*2}<2D$)" if ks**2 < 2 * D
                    else "  (stiff interior)"))
    ax.set_ylabel(r"$y/b$")
axes[0].legend(loc="upper right", fontsize=10)
axes[1].set_xlabel(r"Downstream distance ($\times 2b$)")
save_fig(fig, "fig03_forced_planforms")

# --------------------------------------------------------------- fig04 ---- #
ks = np.linspace(0.01, 2.0, 400)
Ds = np.linspace(0.05, 0.95, 300)
K, DD = np.meshgrid(ks, Ds)
amp = np.abs(forced_response(K, DD, 0.0))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
pc = ax.pcolormesh(K, DD, amp, cmap="RdBu_r", vmin=0, vmax=2, shading="auto")
fig.colorbar(pc, ax=ax, label=r"$|\hat\psi_2/\hat\psi_1|$")
ax.plot(np.sqrt(2 * Ds), Ds, color="k", lw=2.2, ls="--",
        label=r"$k^{*2}=2D$ (equality, exact)")
ax.set_xlim(0, 2)
ax.set_xlabel(r"$k^*=kb$")
ax.set_ylabel(r"$D=\Delta/(U_0+\Delta)$")
ax.set_title(r"interior amplification (deck p. 5 box):  $|\hat\psi_2|>|\hat\psi_1|$"
             r"  iff  $k^{*2}<2D$   ($\gamma=0$)")
ax.legend(loc="upper right", fontsize=10)
save_fig(fig, "fig04_resonance_band")

# --------------------------------------------------------------- fig05 ---- #
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                         gridspec_kw={"width_ratios": [1, 1.35]})

ax = axes[0]
ks = np.linspace(0.01, 2.0, 500)
for g, ls in ((0.0, ":"), (0.05, "-"), (0.1, "--"), (0.2, "-.")):
    ph = np.degrees(np.angle(forced_response(ks, D, g)))
    ax.plot(ks, ph, ls=ls, color=COLORS["upstream"], lw=1.8,
            label=rf"$\gamma={g}$")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"$k^*$")
ax.set_ylabel(r"phase of $\hat\psi_2/\hat\psi_1$  (deg)")
ax.set_title("friction tilts the interior wave")
ax.legend(fontsize=9)

ax = axes[1]
g = 0.1
ksho = 0.3
r = complex(forced_response(ksho, D, g))
x = np.linspace(0, 25, 1200)
kx = ksho * 2.0 * x
amp = 0.55
ax.plot(x, 1.0 + amp * np.cos(kx), color=COLORS["psi1"], lw=1.8)
ax.plot(x, -1.0 + amp * np.cos(kx), color=COLORS["psi1"], lw=1.8)
ax.plot(x, amp * np.real(r * np.exp(1j * kx)), color=COLORS["psi2"], lw=1.8)
for xc, up in ((3.2, True), (8.4, False), (13.7, True), (18.9, False)):
    yy = 1.55 if up else -1.55
    ax.annotate("", xy=(xc + 1.3, yy + (0.5 if up else -0.5)),
                xytext=(xc, yy),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["upstream"], lw=2.4))
ax.text(5.2, 3.4, r"$\overline{u'v'}>0$", color=COLORS["upstream"], fontsize=12)
ax.text(5.2, -4.2, r"$\overline{u'v'}<0$", color=COLORS["upstream"], fontsize=12)
ax.text(14.5, 3.3, r"$\partial_y\overline{u'v'}=\frac{\gamma}{2D}"
        r"\overline{\zeta_2'^2}>0$", fontsize=12, color="k")
ax.annotate("", xy=(2.2, -2.9), xytext=(0.2, -2.9),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["jet"], lw=3))
ax.set_ylim(-5, 5)
ax.set_xlabel(r"Downstream distance ($\times 2b$)")
ax.set_title(rf"momentum leaves the centre for the banks"
             rf"  ($k^*={ksho}$, $\gamma={g}$, deck p. 6)")
save_fig(fig, "fig05_friction_lag_momentum_flux")

print("02_forced_response: done.")
