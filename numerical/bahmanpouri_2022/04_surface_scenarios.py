#!/usr/bin/env python3
"""Surface-velocity scenarios (paper Figs. 4 and 6 style).

Produces figures/fig07..fig08:

    fig07  Sajó CS1: parabolic vs elliptic transverse surface-velocity
           distributions anchored at the printed UAV maximum
           (1.113 m/s at x = 17 m) and the digitized water edges
    fig08  Freiberger Mulde CS3: same, anchored at the Fig.-6 peak

The observed UAV/ADCP traces shown in the paper's figures live only as
plotted curves; the *scenario* curves and anchors here are fully
reconstructed from printed values + digitized bank positions.

Usage
-----
    micromamba run -n fourcastnetv2 python 04_surface_scenarios.py
"""
from __future__ import annotations

import numpy as np

import bahmanpouri_lib as L

plt = L.set_style()


def draw(which, Usurf_max, x_peak, figname, title):
    x, D = L.load_bathymetry(which)
    wet = D > 0.005
    xl, xr = x[wet].min(), x[wet].max()
    xg = np.linspace(xl, xr, 400)
    par = L.usurf_parabolic(xg, x_peak, xl, xr, Usurf_max)
    ell = L.usurf_elliptic(xg, x_peak, xl, xr, Usurf_max)

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(10.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0]})
    ax.plot(xg, par, "-", color=L.COLORS["parabolic"], lw=2.4,
            label="entropy– parabolic")
    ax.plot(xg, ell, "--", color=L.COLORS["elliptic"], lw=2.4,
            label="entropy– elliptic")
    ax.plot(x_peak, Usurf_max, "D", color=L.COLORS["uav"], ms=10, mec="white",
            label=rf"observed max: {Usurf_max:.3g} m/s at x = {x_peak:g} m")
    ax.set_ylabel(r"$U_{surf}$ (m/s)")
    ax.set_ylim(0, Usurf_max * 1.18)
    ax.legend(fontsize=10, loc="lower center")
    ax.set_title(title)

    axb.fill_between(x, 0, -D, color=L.COLORS["water_fill"])
    axb.plot(x, -D, color=L.COLORS["bed"], lw=1.6)
    axb.set_ylabel("bed (m)")
    axb.set_xlabel("distance (m)")
    axb.invert_xaxis()
    L.save_fig(fig, figname)


def main():
    print("04_surface_scenarios.py -> figures/fig07..fig08")
    draw("sajo_cs1", L.SAJO_CS1["Usurf_max"], L.SAJO_CS1["x_peak"],
         "fig07_sajo_surface", "Sajó CS1 surface-velocity scenarios "
         "(paper Fig. 4 style)")
    draw("mulde_cs3", 0.88, 7.5,
         "fig08_mulde_surface", "Freiberger Mulde CS3 surface-velocity "
         "scenarios (paper Fig. 6 style)")


if __name__ == "__main__":
    main()
