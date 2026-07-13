#!/usr/bin/env python3
"""Residual / collapse figures for the Schumm (1967) explainer.

Produces figures/fig03..fig05:

    fig03  measured vs calculated wavelength (paper Fig. 3 replica):
           Eq. (1) as dots, Eq. (2) as open circles, 1:1 line, +-2 SE band
    fig04  the discovery figure: residuals from the Q-ONLY fit vs M --
           the trend that sediment type explains (and Eq. 1 removes)
    fig05  the collapse: lambda * M^0.74 vs Qm -- the tenfold class spread
           folds onto a single power law

Usage
-----
    micromamba run -n fourcastnetv2 python 02_residuals.py
"""
from __future__ import annotations

import numpy as np

import schumm_lib as L

plt = L.set_style()

D = L.load_sections()


def fig03_measured_vs_calculated():
    calc1 = L.wavelength_qm(D["Qm"], D["M"])
    calc2 = L.wavelength_qma(D["Qma"], D["M"])

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    g = np.logspace(2.3, 4.6, 50)
    se = L.PUBLISHED["eq1"]["see_log"]
    ax.fill_between(g, g / 10**(2 * se), g * 10**(2 * se),
                    color=L.COLORS["band"], alpha=0.55, zorder=1,
                    label=r"$\pm 2$ SE (0.16 log units)")
    ax.loglog(g, g, "-", color=L.COLORS["fit"], lw=1.5, zorder=2)
    ax.loglog(calc1, D["lam"], "o", color=L.COLORS["fit"], ms=6.5,
              mec="white", mew=0.5, zorder=3, label="Eq. (1), $Q_m$")
    ax.loglog(calc2, D["lam"], "o", mfc="none", mec=L.COLORS["suspended"],
              ms=7.5, mew=1.6, zorder=3, label="Eq. (2), $Q_{ma}$")

    ax.set_xlabel("calculated meander wavelength (ft)")
    ax.set_ylabel("measured meander wavelength (ft)")
    ax.set_title("Paper Fig. 3: measured vs calculated wavelength\n"
                 "(dots Eq. 1, open circles Eq. 2)")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(200, 4e4)
    ax.set_ylim(200, 4e4)
    L.save_fig(fig, "fig03_measured_vs_calculated")


def fig04_residual_vs_M():
    # Q-only fit (Carlston-style), then residuals against M.
    y = np.log10(D["lam"])
    X = np.column_stack([np.ones_like(y), np.log10(D["Qm"])])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for c in ("bedload", "mixed", "suspended"):
        m = D["cls"] == c
        ax.semilogx(D["M"][m], resid[m], "o", color=L.COLORS[c], ms=7.5,
                    mec="white", mew=0.6, label=c)
    Mg = np.logspace(np.log10(1.0), np.log10(50.0), 50)
    # Slope in log-log residual space is exactly -exp_m about the M-mean.
    Mbar = 10 ** np.mean(np.log10(D["M"]))
    ax.semilogx(Mg, -L.PARAMS.exp_m * (np.log10(Mg) - np.log10(Mbar)), "--",
                color=L.COLORS["fit"], lw=1.8,
                label=r"slope $-0.74$ (Eq. 1's $M$ exponent)")
    ax.axhline(0, color=L.COLORS["reference"], lw=1.0)
    ax.set_xlabel("percent silt-clay $M$")
    ax.set_ylabel(r"residual of $Q$-only fit  ($\log_{10}$ units)")
    ax.set_title("What discharge alone leaves behind is an $M$ trend --\n"
                 "the sediment-load control Schumm quantified")
    ax.legend(fontsize=10)
    L.save_fig(fig, "fig04_residual_vs_M")


def fig05_collapse():
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=False)

    for c in ("bedload", "mixed", "suspended"):
        m = D["cls"] == c
        axs[0].loglog(D["Qm"][m], D["lam"][m], "o", color=L.COLORS[c],
                      ms=7, mec="white", mew=0.5, label=c)
        axs[1].loglog(D["Qm"][m], D["lam"][m] * D["M"][m] ** L.PARAMS.exp_m,
                      "o", color=L.COLORS[c], ms=7, mec="white", mew=0.5)
    Qg = np.logspace(np.log10(15.0), np.log10(8000.0), 60)
    axs[1].loglog(Qg, L.PARAMS.coef_qm * Qg ** L.PARAMS.exp_qm, "-",
                  color=L.COLORS["fit"], lw=2.0,
                  label=r"$1890\,Q_m^{0.34}$")
    axs[0].set_title("raw: tenfold spread at fixed $Q_m$")
    axs[1].set_title(r"rescaled: $\lambda M^{0.74}$ collapses onto one law")
    axs[0].set_xlabel("$Q_m$ (cfs)")
    axs[1].set_xlabel("$Q_m$ (cfs)")
    axs[0].set_ylabel(r"$\lambda$ (ft)")
    axs[1].set_ylabel(r"$\lambda\,M^{0.74}$")
    axs[0].legend(fontsize=10)
    axs[1].legend(fontsize=11)
    fig.suptitle("Sediment type is not noise -- it is the second axis", y=1.02)
    L.save_fig(fig, "fig05_collapse")


def main():
    print("02_residuals.py -> figures/fig03..fig05")
    fig03_measured_vs_calculated()
    fig04_residual_vs_M()
    fig05_collapse()


if __name__ == "__main__":
    main()
