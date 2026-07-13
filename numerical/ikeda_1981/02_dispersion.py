#!/usr/bin/env python3
"""Quantitative dispersion-relation figures for the Ikeda et al. (1981) explainer.

Produces figures/fig06..fig10:

    fig06  Growth rate alpha0(k): the unstable band and the selected wavenumber
    fig07  Celerity c0(k): bends ALWAYS migrate downstream
    fig08  Combined dispersion diagram (growth rate + migration frequency)
    fig09  Wavelength selection & scaling (theory-only "validation")
    fig10  Parameter-sensitivity map  alpha0(k, A)

All curves are drawn in the rescaled wavenumber  kappa = k / C_f, in which the
dispersion relations are C_f-independent (friction only sets the physical
scale).  Concretely, evaluating the ikeda_lib relations at C_f = 1 with argument
kappa returns exactly alpha0/C_f^2, omega0/C_f^2, c0/C_f -- so we reuse the same
verified functions.

Usage
-----
    micromamba run -n fourcastnetv2 python 02_dispersion.py
"""
from __future__ import annotations

import numpy as np

import ikeda_lib as L

plt = L.set_style()


# Scaled evaluators (C_f = 1  =>  x-axis is kappa = k/C_f, y is alpha/C_f^2 etc.)
def a_scaled(kappa, A, F):
    return L.growth_rate(kappa, 1.0, A, F)


def w_scaled(kappa, A, F):
    return L.frequency(kappa, 1.0, A, F)


def c_scaled(kappa, A, F):
    return L.celerity(kappa, 1.0, A, F)


def kappa_OM(A, F):
    return L.beta_param(A, F)                 # = k_OM / C_f


def kappa_c(A, F):
    return np.sqrt(2.0 * (A + F**2))          # = k_c / C_f


# --------------------------------------------------------------------------- #
#  fig06 -- growth rate: unstable band and selected wavenumber
# --------------------------------------------------------------------------- #
def fig06_growth_rate():
    F = L.PARAMS.F
    kap = np.linspace(0, 3.2, 1000)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    cases = [(2.89, L.COLORS["growth"], "alluvial,  $A=2.89$"),
             (1.50, "#66a61e", "$A=1.5$"),
             (0.50, "#a6761d", "$A=0.5$")]
    for A, col, lab in cases:
        ax.plot(kap, a_scaled(kap, A, F), color=col, lw=2.4, label=lab)

    # canonical alluvial case annotations
    A0 = 2.89
    kc = kappa_c(A0, F)
    kom = kappa_OM(A0, F)
    aom = a_scaled(kom, A0, F)
    band = kap <= kc
    ax.fill_between(kap[band], 0, a_scaled(kap[band], A0, F),
                    color=L.COLORS["growth"], alpha=0.12)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(kom, color=L.COLORS["apex"], ls="--", lw=1.2)
    ax.plot(kom, aom, "o", color=L.COLORS["apex"], ms=6, zorder=5)
    ax.annotate(rf"$k_{{OM}}=\beta\,C_f\approx{kom:.2f}\,C_f$"
                "\n(fastest-growing)",
                (kom, aom), (kom + 0.25, aom * 1.02),
                fontsize=10.5, color=L.COLORS["apex"],
                arrowprops=dict(arrowstyle="->", color=L.COLORS["apex"]))
    ax.annotate(rf"$k_c=\sqrt{{2(A+F^2)}}\,C_f\approx{kc:.2f}\,C_f$"
                "\n(neutral: growth $\\to$ 0)",
                (kc, 0), (kc - 0.05, aom * 0.42), ha="right",
                fontsize=10.5, color=L.COLORS["decay"],
                arrowprops=dict(arrowstyle="->", color=L.COLORS["decay"]))
    ax.annotate("UNSTABLE\n(bends grow)", (0.55 * kom, aom * 0.28),
                color=L.COLORS["growth"], fontsize=10, ha="center")
    ax.annotate("stable\n(bends decay)", (kc + 0.45, -0.6 * aom),
                color=L.COLORS["decay"], fontsize=10, ha="center")

    ax.set_title("Growth rate: a band of wavelengths is unstable, one grows fastest")
    ax.set_xlabel(r"rescaled wavenumber  $\kappa = k/C_f$")
    ax.set_ylabel(r"growth rate  $\alpha_0 / C_f^{2}$")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-1.1 * aom, 1.7 * aom)
    ax.legend(loc="upper right", title="secondary-flow strength")
    L.save_fig(fig, "fig06_growth_rate")


# --------------------------------------------------------------------------- #
#  fig07 -- celerity: always downstream
# --------------------------------------------------------------------------- #
def fig07_celerity():
    A, F = L.PARAMS.A, L.PARAMS.F
    kap = np.linspace(0, 3.2, 1000)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(kap, c_scaled(kap, A, F), color=L.COLORS["velocity"], lw=2.6,
            label=r"celerity  $c_0=\omega_0/k$")
    ax.axhline(0, color="k", lw=0.7)
    kom = kappa_OM(A, F)
    kc = kappa_c(A, F)
    ax.axvspan(0, kc, color=L.COLORS["growth"], alpha=0.10, label="unstable band")
    ax.axvline(kom, color=L.COLORS["apex"], ls="--", lw=1.2)
    ax.plot(kom, c_scaled(kom, A, F), "o", color=L.COLORS["apex"], ms=6, zorder=5)
    ax.annotate(rf"$c_{{OM}}\approx 1.17\,k_{{OM}}$",
                (kom, c_scaled(kom, A, F)), (kom + 0.2, c_scaled(kom, A, F) * 0.6),
                fontsize=11, color=L.COLORS["apex"],
                arrowprops=dict(arrowstyle="->", color=L.COLORS["apex"]))
    ax.annotate(r"$c_0>0$ for every $k$  $\Rightarrow$  bends always migrate DOWNSTREAM"
                "\n(a correction of Ikeda et al. 1976: no stable upstream-migrating bends)",
                (0.15, c_scaled(3.0, A, F) * 0.82), fontsize=10.5,
                color=L.COLORS["velocity"])

    ax.set_title("Migration speed is positive at every wavelength")
    ax.set_xlabel(r"rescaled wavenumber  $\kappa = k/C_f$")
    ax.set_ylabel(r"celerity  $c_0 / C_f$")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, c_scaled(3.2, A, F) * 1.12)
    ax.legend(loc="lower right")
    L.save_fig(fig, "fig07_celerity")


# --------------------------------------------------------------------------- #
#  fig08 -- combined dispersion diagram
# --------------------------------------------------------------------------- #
def fig08_combined():
    A, F = L.PARAMS.A, L.PARAMS.F
    kap = np.linspace(0, 3.2, 1000)
    kom, kc = kappa_OM(A, F), kappa_c(A, F)

    fig, ax1 = plt.subplots(figsize=(9.4, 5.4))
    ax2 = ax1.twinx()
    ax2.grid(False)

    l1, = ax1.plot(kap, a_scaled(kap, A, F), color=L.COLORS["growth"], lw=2.6,
                   label=r"growth rate  $\alpha_0/C_f^2$")
    ax1.fill_between(kap[kap <= kc], 0, a_scaled(kap[kap <= kc], A, F),
                     color=L.COLORS["growth"], alpha=0.12)
    ax1.axhline(0, color="k", lw=0.7)
    l2, = ax2.plot(kap, w_scaled(kap, A, F), color=L.COLORS["curvature"], lw=2.6,
                   ls="--", label=r"migration freq.  $\omega_0/C_f^2$")

    ax1.axvline(kom, color=L.COLORS["apex"], ls=":", lw=1.2)
    ax1.annotate(r"$k_{OM}$", (kom, 0), (kom, ax1.get_ylim()[1] * 0.06),
                 ha="center", fontsize=11, color=L.COLORS["apex"])

    ax1.set_title("Dispersion relation: growth selects the wavelength, "
                  "frequency sets the speed")
    ax1.set_xlabel(r"rescaled wavenumber  $\kappa = k/C_f$")
    ax1.set_ylabel(r"growth rate  $\alpha_0 / C_f^{2}$", color=L.COLORS["growth"])
    ax2.set_ylabel(r"migration frequency  $\omega_0 / C_f^{2}$",
                   color=L.COLORS["curvature"])
    ax1.tick_params(axis="y", colors=L.COLORS["growth"])
    ax2.tick_params(axis="y", colors=L.COLORS["curvature"])
    ax1.set_xlim(0, 3.2)
    ax1.legend(handles=[l1, l2], loc="upper center")
    L.save_fig(fig, "fig08_dispersion_combined")


# --------------------------------------------------------------------------- #
#  fig09 -- wavelength selection & scaling (theory-only validation)
# --------------------------------------------------------------------------- #
def fig09_wavelength_scaling():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # (a) predicted meander wavelength lambda_OM/H0 = 2 pi / (beta C_f) vs C_f
    Cf = np.logspace(-3, -1.2, 200)
    cases = [(2.89, 0.30, L.COLORS["growth"], "alluvial  $A=2.89,\\ F=0.3$"),
             (0.00, 0.30, L.COLORS["velocity"], "incised  $A=0,\\ F=0.3$"),
             (0.00, 0.70, L.COLORS["curvature"], "incised  $A=0,\\ F=0.7$")]
    for A, F, col, lab in cases:
        beta = L.beta_param(A, F)
        lam = 2.0 * np.pi / (beta * Cf)          # lambda_OM / H0
        axL.loglog(Cf, lam, color=col, lw=2.6, label=lab)
    axL.axvspan(0.005, 0.03, color="0.85", alpha=0.6, zorder=0)
    axL.annotate("typical alluvial\n$C_f$ range", (0.012, 3e2), ha="center",
                 fontsize=9.5, color="0.35")
    axL.set_title("(a) Predicted meander wavelength (no fitted constant)")
    axL.set_xlabel(r"friction coefficient  $C_f$")
    axL.set_ylabel(r"$\lambda_{OM}/H_0 = 2\pi/(\beta\,C_f)$")
    axL.legend(loc="upper right", fontsize=10)
    axL.grid(True, which="both", alpha=0.25)

    # (b) selection sharpness: beta = k_OM/C_f as a function of (A + F^2)
    s = np.linspace(0, 8, 300)               # s = A + F^2
    beta = np.sqrt(4.0 * np.sqrt(1.0 + 0.5 * s) - 4.0)
    axR.plot(s, beta, color=L.COLORS["channel"], lw=2.8)
    for A, F, col, lab in cases:
        sv = A + F**2
        axR.plot(sv, L.beta_param(A, F), "o", color=col, ms=9)
        axR.annotate(lab.split("  ")[0], (sv, L.beta_param(A, F)),
                     (sv + 0.15, L.beta_param(A, F) - 0.12), fontsize=9.5, color=col)
    axR.axhline(1.50, color=L.COLORS["growth"], ls=":", lw=1.2)
    axR.annotate(r"$\beta\simeq1.50$ (alluvial)", (5.4, 1.53),
                 fontsize=9.5, color=L.COLORS["growth"])
    axR.set_title(r"(b) Secondary flow controls selection:  $\beta=k_{OM}/C_f$")
    axR.set_xlabel(r"$A + F^{2}$   (secondary-flow + Froude)")
    axR.set_ylabel(r"$\beta = k_{OM}/C_f$")
    axR.set_xlim(0, 8)
    axR.set_ylim(0, beta.max() * 1.08)

    fig.suptitle("Wavelength selection: friction sets the scale, secondary flow "
                 "sharpens it  (theory curves, Eqs. 20–21, 24a)", y=1.02, fontsize=13)
    L.save_fig(fig, "fig09_wavelength_scaling")


# --------------------------------------------------------------------------- #
#  fig10 -- parameter-sensitivity map alpha0(k, A)
# --------------------------------------------------------------------------- #
def fig10_sensitivity():
    F = L.PARAMS.F
    kap = np.linspace(0.01, 3.4, 400)
    Avals = np.linspace(0.0, 6.0, 400)
    KK, AA = np.meshgrid(kap, Avals)
    Z = L.growth_rate(KK, 1.0, AA, F)          # alpha/C_f^2

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    vmax = np.nanmax(Z)
    pcm = ax.pcolormesh(KK, AA, np.where(Z > 0, Z, np.nan), cmap="viridis",
                        vmin=0, vmax=vmax, shading="auto")
    # neutral curve (alpha = 0)  kappa_c(A) = sqrt(2(A+F^2))
    ax.plot(np.sqrt(2.0 * (Avals + F**2)), Avals, color="#00e5ee", lw=2.2,
            label=r"neutral  $k_c=\sqrt{2(A+F^2)}\,C_f$")
    # ridge of maximum growth  kappa_OM(A) = beta(A)
    ax.plot(np.sqrt(4.0 * np.sqrt(1.0 + 0.5 * (Avals + F**2)) - 4.0), Avals,
            color=L.COLORS["erosion"], lw=2.4, ls="--",
            label=r"fastest-growing  $k_{OM}=\beta\,C_f$")
    ax.axhline(2.89, color="white", lw=1.0, ls=":")
    ax.annotate("alluvial $A=2.89$", (0.1, 2.98), color="white", fontsize=10)

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"growth rate  $\alpha_0/C_f^2$")
    ax.set_title("Stronger secondary flow $\\to$ shorter selected wavelength "
                 "(larger $k_{OM}$)")
    ax.set_xlabel(r"rescaled wavenumber  $\kappa = k/C_f$")
    ax.set_ylabel(r"secondary-flow parameter  $A$")
    ax.set_xlim(0, 3.4)
    ax.set_ylim(0, 6)
    ax.legend(loc="lower right", framealpha=0.85)
    ax.grid(False)
    L.save_fig(fig, "fig10_sensitivity")


def main():
    print("02_dispersion.py -> figures/fig06..fig10")
    fig06_growth_rate()
    fig07_celerity()
    fig08_combined()
    fig09_wavelength_scaling()
    fig10_sensitivity()


if __name__ == "__main__":
    main()
