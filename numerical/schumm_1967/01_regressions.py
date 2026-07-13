#!/usr/bin/env python3
"""Regression figures for the Schumm (1967) explainer -- paper Figs. 1-2 replicas.

Produces figures/fig01..fig02:

    fig01  lambda vs mean annual discharge Qm (paper Fig. 1): class-coded
           sections, Carlston's single-variable line, and Eq. (1) drawn at
           the class-median M values -- the "family of lines" sediment adds
    fig02  lambda vs mean annual flood Qma (paper Fig. 2): class-coded
           sections, Dury's bankfull line, Eq. (2) at class-median M

All points are the 36 transcribed sections (data/schumm_1967_sections.csv,
PP 598 Tables 6 and 1) -- real data, no synthesis.

Usage
-----
    micromamba run -n fourcastnetv2 python 01_regressions.py
"""
from __future__ import annotations

import numpy as np

import schumm_lib as L

plt = L.set_style()

D = L.load_sections()
MARKERS = {"bedload": "s", "mixed": "o", "suspended": "^"}
LABELS = {"bedload": "bedload ($M<5$)", "mixed": "mixed ($5{-}20$)",
          "suspended": "suspended ($M>20$)"}


def scatter_by_class(ax, Q, lam):
    for c in ("bedload", "mixed", "suspended"):
        m = D["cls"] == c
        ax.loglog(Q[m], lam[m], MARKERS[c], color=L.COLORS[c], ms=7.5,
                  mec="white", mew=0.6, label=LABELS[c], zorder=3)


def class_median_M():
    return {c: float(np.median(D["M"][D["cls"] == c]))
            for c in ("bedload", "mixed", "suspended")}


def fig01_qm():
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    scatter_by_class(ax, D["Qm"], D["lam"])

    Qg = np.logspace(np.log10(15.0), np.log10(8000.0), 100)
    ax.loglog(Qg, L.carlston_line(Qg), "--", color=L.COLORS["reference"],
              lw=2.0, label=r"Carlston '65:  $\lambda=106\,Q_m^{0.46}$")
    for c, Mc in class_median_M().items():
        ax.loglog(Qg, L.wavelength_qm(Qg, Mc), "-", color=L.COLORS[c],
                  lw=1.6, alpha=0.75)
        ax.annotate(rf"$M={Mc:.0f}$", (Qg[-1], L.wavelength_qm(Qg[-1], Mc)),
                    color=L.COLORS[c], fontsize=10,
                    xytext=(4, 0), textcoords="offset points")

    ax.set_xlabel("mean annual discharge $Q_m$ (cfs)")
    ax.set_ylabel(r"meander wavelength $\lambda$ (ft)")
    ax.set_title("Paper Fig. 1: one Carlston line cannot hold all channel types --\n"
                 r"Eq. (1) $\lambda = 1890\,Q_m^{0.34}/M^{0.74}$ splits into an $M$-family")
    ax.legend(loc="upper left", fontsize=10)
    L.save_fig(fig, "fig01_lambda_vs_qm")


def fig02_qma():
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    scatter_by_class(ax, D["Qma"], D["lam"])

    Qg = np.logspace(np.log10(500.0), np.log10(60000.0), 100)
    ax.loglog(Qg, L.dury_line(Qg), "--", color=L.COLORS["reference"],
              lw=2.0, label=r"Dury '65 (bankfull):  $\lambda=30\,Q_b^{0.5}$")
    for c, Mc in class_median_M().items():
        ax.loglog(Qg, L.wavelength_qma(Qg, Mc), "-", color=L.COLORS[c],
                  lw=1.6, alpha=0.75)
        ax.annotate(rf"$M={Mc:.0f}$", (Qg[-1], L.wavelength_qma(Qg[-1], Mc)),
                    color=L.COLORS[c], fontsize=10,
                    xytext=(4, 0), textcoords="offset points")

    ax.set_xlabel("mean annual flood $Q_{ma}$ (cfs)")
    ax.set_ylabel(r"meander wavelength $\lambda$ (ft)")
    ax.set_title("Paper Fig. 2: low silt-clay plots above Dury's line, high below --\n"
                 r"Eq. (2) $\lambda = 234\,Q_{ma}^{0.48}/M^{0.74}$")
    ax.legend(loc="upper left", fontsize=10)
    L.save_fig(fig, "fig02_lambda_vs_qma")


def main():
    print("01_regressions.py -> figures/fig01..fig02")
    fig01_qm()
    fig02_qma()


if __name__ == "__main__":
    main()
