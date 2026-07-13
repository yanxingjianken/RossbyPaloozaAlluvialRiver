#!/usr/bin/env python3
"""2D iso-velocity cross-section maps (the paper's Figs. 5 and 7 style).

Produces figures/fig05..fig06:

    fig05  Sajó CS1: entropy-reconstructed U(x, y) over the digitized
           bathymetry, parabolic scenario; top panel without dip, bottom
           with an ILLUSTRATIVE dip taper (flagged below) that submerges
           the core as in the paper's parabolic panel
    fig06  Freiberger Mulde CS3: same construction

Orientation follows the paper: depth increases downward, distance axis as
digitized (0 at the right bank in the source figures), colorbar in m/s.

DIP_TAPER flag: the paper identifies the dip per vertical via Moramarco
et al. (2017); those profile data are in the SI.  For the *visual* only, the
dip here is the Table-2-calibrated value at the deepest vertical, tapered
linearly to zero at the banks -- labelled on the figure as illustrative.
The no-dip panel carries no such assumption.

Usage
-----
    micromamba run -n fourcastnetv2 python 03_cross_section_maps.py
"""
from __future__ import annotations

import numpy as np

import bahmanpouri_lib as L

plt = L.set_style()

DIP_TAPER = "illustrative: dip at deepest vertical (Table-2 calibrated), linear taper to banks"


def build_map(which, P, t2name=None, dip_frac=None, ny=120):
    x, D = L.load_bathymetry(which)
    wet = D > 0.005
    xl, xr = x[wet].min(), x[wet].max()
    us = L.usurf_parabolic(x, P["x_peak"], xl, xr, P["Usurf_max"])
    i0 = int(np.argmax(D))
    if t2name is not None:
        # dip calibrated so the deepest vertical reaches Table 2's entropy max
        t2 = {c.name: c for c in L.load_table2()}[t2name]
        h0 = L.dip_solve(us[i0], D[i0], P["M"], t2.Umax_entropy)
    else:
        # fixed illustrative dip fraction (used where the Table-2 target is
        # unreachable from the UAV anchor -- the paper notes a 15% UAV/ADCP
        # discrepancy on the Freiberger Mulde)
        h0 = dip_frac * D[i0]
    # taper dip with distance from the deepest vertical (illustrative),
    # capped so h never exceeds 35% of the local depth
    w = np.maximum(0.0, 1.0 - np.abs(x - x[i0]) / max(x[i0] - xl, xr - x[i0]))
    h_tap = np.minimum(h0 * w, 0.35 * np.maximum(D, 1e-6))
    grids = {}
    for tag, hh in (("nodip", np.zeros_like(x)), ("dip", h_tap)):
        U = np.full((ny, x.size), np.nan)
        yy = np.linspace(0.0, 1.0, ny)          # fraction of local depth
        for i in np.where(wet)[0]:
            Umaxv = L.umaxv_from_surface(us[i], D[i], hh[i], P["M"])
            yphys = yy * D[i]
            U[:, i] = L.u_vertical(yphys, Umaxv, D[i], hh[i], P["M"])
        grids[tag] = (yy, U)
    return x, D, grids, h0, x[i0]


def draw(which, P, figname, title, t2name=None, dip_frac=None):
    x, D, grids, h0, x0 = build_map(which, P, t2name=t2name, dip_frac=dip_frac)
    fig, axs = plt.subplots(2, 1, figsize=(11.0, 7.6), sharex=True)
    for ax, tag, lab in ((axs[0], "nodip", "no dip ($h=0$)"),
                         (axs[1], "dip", f"with dip (h₀={h0:.2f} m at x={x0:.1f} m; {DIP_TAPER})")):
        yy, U = grids[tag]
        # depth-below-surface coordinates per column
        Xg = np.tile(x, (yy.size, 1))
        Zg = (1.0 - yy)[:, None] * D[None, :]   # depth below surface
        pc = ax.contourf(Xg, Zg, U, levels=np.linspace(0, 1.3, 14),
                         cmap="turbo")
        ax.plot(x, D, "-", color=L.COLORS["bed"], lw=1.6)
        ax.fill_between(x, D, D.max() * 1.12, color="white", zorder=2)
        ax.plot(x, D, "-", color=L.COLORS["bed"], lw=1.6, zorder=3)
        ax.set_ylim(D.max() * 1.1, 0)
        ax.set_ylabel("depth (m)")
        ax.set_title(lab, fontsize=11)
        cb = fig.colorbar(pc, ax=ax, pad=0.01)
        cb.set_label("U (m/s)")
    axs[1].set_xlabel("distance (m)")
    axs[1].invert_xaxis()      # paper orientation: 0 at the right
    fig.suptitle(title, y=0.99)
    L.save_fig(fig, figname)


def main():
    print("03_cross_section_maps.py -> figures/fig05..fig06")
    draw("sajo_cs1", L.SAJO_CS1, "fig05_sajo_cs1_map",
         "Sajó CS1: entropy velocity field from ONE surface value "
         "(digitized bathymetry, parabolic scenario; paper Fig. 5 style)",
         t2name="Sajo CS1")
    # FM CS3 pipeline anchors are SI-only; use the digitized section with the
    # Table-2 M and the UAV-observed peak from Fig. 6 axes.  The Table-2
    # entropy max (0.98) is unreachable from the UAV anchor at M = 1.16
    # (the paper notes a 15% UAV/ADCP gap on this river), so the dip is a
    # fixed illustrative 12% of the deepest vertical.
    P3 = dict(Usurf_max=0.88, x_peak=7.5, M=1.16)
    draw("mulde_cs3", P3, "fig06_mulde_cs3_map",
         "Freiberger Mulde CS3: entropy velocity field "
         "(digitized bathymetry, parabolic scenario; paper Fig. 7 style)",
         dip_frac=0.12)


if __name__ == "__main__":
    main()
