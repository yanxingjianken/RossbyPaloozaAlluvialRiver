#!/usr/bin/env python3
"""Vertical entropy profiles (Eq. 1): shape family, the dip, and Fig. 8a.

Produces figures/fig03..fig04:

    fig03  Eq.-1 profile family across the observed M range, with and
           without dip -- how M sets fullness and h submerges the maximum
    fig04  paper Fig. 8a analogue: the profile at Sajo CS1's deepest
           vertical, dip solved so the entropy maximum matches Table 2's
           Umax_entropy from the printed surface anchor

Usage
-----
    micromamba run -n fourcastnetv2 python 02_vertical_profiles.py
"""
from __future__ import annotations

import numpy as np

import bahmanpouri_lib as L

plt = L.set_style()


def fig03_family():
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.6), sharey=True)
    D = 1.0
    yy = np.linspace(0, D, 300)

    for M, col in ((1.16, "#9ecae1"), (2.17, "#4292c6"), (3.24, "#084594")):
        axs[0].plot(L.u_vertical(yy, 1.0, D, 0.0, M), (D - yy),
                    color=col, lw=2.2, label=rf"$M={M}$")
    axs[0].invert_yaxis()
    axs[0].set_title("no dip ($h=0$): maximum at the surface")
    axs[0].set_xlabel(r"$U/U_{maxv}$")
    axs[0].set_ylabel("depth below surface (m)")
    axs[0].legend(fontsize=10)

    M = 3.24
    for h, col in ((0.0, "#c7e0f0"), (0.1, "#7fb8d8"), (0.2, "#2c7fb8"),
                   (0.3, "#08519c")):
        axs[1].plot(L.u_vertical(yy, 1.0, D, h, M), (D - yy),
                    color=col, lw=2.2, label=rf"$h={h:.1f}$ m")
        axs[1].plot(float(L.u_vertical(D - h, 1.0, D, h, M)), h, "o",
                    color=col, ms=6)
    axs[1].invert_yaxis()
    axs[1].set_title(rf"dip submerges the maximum ($M={M}$)")
    axs[1].set_xlabel(r"$U/U_{maxv}$")
    axs[1].legend(fontsize=10, title="dip depth")
    fig.suptitle("Eq. (1): the entropy parameter sets fullness; "
                 "the dip sets where the core sits", y=1.02)
    L.save_fig(fig, "fig03_profile_family")


def fig04_sajo_vertical():
    x, D = L.load_bathymetry("sajo_cs1")
    P = L.SAJO_CS1
    i = int(np.argmax(D))
    Di = D[i]
    # Surface velocity at that vertical from the parabolic scenario, then dip
    # solved so the vertical's maximum reaches Table 2's entropy maximum.
    wet = D > 0.005
    us = L.usurf_parabolic(x, P["x_peak"], x[wet].min(), x[wet].max(),
                           P["Usurf_max"])[i]
    t2 = {c.name: c for c in L.load_table2()}["Sajo CS1"]
    h = L.dip_solve(us, Di, P["M"], t2.Umax_entropy)
    Umaxv = L.umaxv_from_surface(us, Di, h, P["M"])

    yy = np.linspace(0, Di, 400)
    prof = L.u_vertical(yy, Umaxv, Di, h, P["M"])

    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    ax.plot(prof, Di - yy, "-", color=L.COLORS["entropy"], lw=2.6,
            label="entropy profile (Eq. 1)")
    ax.plot(us, 0.0, "v", color=L.COLORS["uav"], ms=11,
            label=rf"surface obs. $U_{{surf}}={us:.2f}$ m/s")
    ax.plot(Umaxv, h, "o", color=L.COLORS["adcp"], ms=9,
            label=rf"$U_{{maxv}}={Umaxv:.2f}$ at dip $h={h:.2f}$ m")
    ax.axhline(h, color=L.COLORS["adcp"], lw=0.8, ls=":")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.35)
    ax.set_xlabel(r"$U$ (m/s)")
    ax.set_ylabel("depth below surface (m)")
    ax.set_title(f"Sajó CS1, deepest vertical (x = {x[i]:.1f} m):\n"
                 "one surface value + Eq. 4 dip → the full profile "
                 "(paper Fig. 8a analogue)")
    ax.legend(fontsize=10, loc="lower left")
    L.save_fig(fig, "fig04_sajo_vertical")
    print(f"  dip solved: h = {h:.3f} m ({100*h/Di:.0f}% of D) at x = {x[i]:.1f} m")


def main():
    print("02_vertical_profiles.py -> figures/fig03..fig04")
    fig03_family()
    fig04_sajo_vertical()


if __name__ == "__main__":
    main()
