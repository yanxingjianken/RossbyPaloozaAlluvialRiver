#!/usr/bin/env python3
"""Static schematic / definition figures for the Ikeda et al. (1981) explainer.

Produces figures/fig01..fig05:

    fig01  Planform definition sketch      (recreates paper Fig. 2a)
    fig02  Channel cross-section           (recreates paper Fig. 2b)
    fig03  Secondary-flow schematic        (the meaning of parameter A, Eq. 6)
    fig04  Bank-erosion geometry           (recreates paper Fig. 3, Eq. 11)
    fig05  The phase-lag mechanism         (the "money figure": curvature vs u_b)

These are annotated schematics; where a curve is quantitative (fig05) it is
computed from the verified relations in ikeda_lib.

Usage
-----
    micromamba run -n fourcastnetv2 python 01_schematics.py
"""
from __future__ import annotations

import numpy as np

import ikeda_lib as L

plt = L.set_style()
from matplotlib.patches import FancyArrowPatch, Arc, Polygon  # noqa: E402


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def meander(x, amp, wavelength):
    """Sinusoidal centreline and its slope for a plan-view meander."""
    k = 2.0 * np.pi / wavelength
    yc = amp * np.sin(k * x)
    dy = amp * k * np.cos(k * x)
    return yc, dy


def banks(x, yc, dy, b):
    """Left/right bank coordinates offset by +/- b normal to the centreline."""
    norm = np.hypot(1.0, dy)
    nx, ny = -dy / norm, 1.0 / norm          # unit normal (points to +y side)
    return (x + b * nx, yc + b * ny), (x - b * nx, yc - b * ny)


def channel_polygon(left, right):
    """Closed polygon of the wetted channel from the two bank lines."""
    xs = np.concatenate([left[0], right[0][::-1]])
    ys = np.concatenate([left[1], right[1][::-1]])
    return np.column_stack([xs, ys])


# --------------------------------------------------------------------------- #
#  fig01 -- planform definition sketch (paper Fig. 2a)
# --------------------------------------------------------------------------- #
def fig01_planform():
    wavelength, amp, b = 10.0, 2.2, 0.9
    x = np.linspace(0, 2.5 * wavelength, 900)
    yc, dy = meander(x, amp, wavelength)
    left, right = banks(x, yc, dy, b)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.add_patch(Polygon(channel_polygon(left, right), closed=True,
                         facecolor=L.COLORS["water_fill"], edgecolor="none", zorder=1))
    ax.plot(left[0], left[1], color=L.COLORS["bank"], lw=2, zorder=3)
    ax.plot(right[0], right[1], color=L.COLORS["bank"], lw=2, zorder=3)
    ax.plot(x, yc, "--", color=L.COLORS["channel"], lw=1.6, zorder=4)

    # point bars on the inner bank of each bend (schematic tan blobs)
    k = 2.0 * np.pi / wavelength
    for xc in np.arange(0.25, 2.5, 0.5) * wavelength:
        yb = amp * np.sin(k * xc)
        inner = -np.sign(np.sin(k * xc))            # inner bank side at the apex
        xb = np.linspace(xc - 0.16 * wavelength, xc + 0.16 * wavelength, 40)
        ycb, dyb = meander(xb, amp, wavelength)
        (lx, ly), (rx, ry) = banks(xb, ycb, dyb, b)
        if inner > 0:
            bx, by = lx, ly
        else:
            bx, by = rx, ry
        ax.fill(np.r_[bx, bx[::-1]],
                np.r_[by, by - inner * 0.55],
                color=L.COLORS["deposition"], alpha=0.9, zorder=2, lw=0)

    # radius of curvature at the first apex (x = wavelength/4)
    xa = wavelength * 0.25
    ya = amp
    r0 = 1.0 / (amp * k**2)                          # |1/curvature| at the crest
    cx, cy = xa, ya - r0
    ax.plot([cx, xa], [cy, ya], color=L.COLORS["apex"], lw=1.1, ls=":")
    ax.add_patch(Arc((cx, cy), 2 * r0, 2 * r0, theta1=70, theta2=110,
                     color=L.COLORS["apex"], lw=1.0, ls=":"))
    ax.annotate(r"$r_0$", ((cx + xa) / 2 + 0.2, (cy + ya) / 2), fontsize=13)
    ax.plot(xa, ya, "o", color=L.COLORS["apex"], ms=4)

    # local s-n intrinsic axes at a mid-limb point
    xs = wavelength * 0.5
    ys, dys = meander(np.array([xs]), amp, wavelength)
    ys, dys = ys[0], dys[0]
    tnorm = np.hypot(1, dys)
    tx, ty = 1 / tnorm, dys / tnorm
    nx, ny = -ty, tx
    ax.add_patch(FancyArrowPatch((xs, ys), (xs + 2.0 * tx, ys + 2.0 * ty),
                 arrowstyle="-|>", mutation_scale=14, color="k", lw=1.6))
    ax.add_patch(FancyArrowPatch((xs, ys), (xs + 1.4 * nx, ys + 1.4 * ny),
                 arrowstyle="-|>", mutation_scale=14, color="k", lw=1.6))
    ax.annotate(r"$\tilde s$", (xs + 2.0 * tx + 0.2, ys + 2.0 * ty), fontsize=13)
    ax.annotate(r"$\tilde n$", (xs + 1.4 * nx, ys + 1.4 * ny + 0.25), fontsize=13)

    # width bracket 2b on a straight-ish crossing
    xw = wavelength * 1.0
    yw, dyw = meander(np.array([xw]), amp, wavelength)
    yw, dyw = yw[0], dyw[0]
    (lx, ly), (rx, ry) = banks(np.array([xw]), np.array([yw]), np.array([dyw]), b)
    ax.annotate("", (rx[0], ry[0]), (lx[0], ly[0]),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax.annotate(r"$2b$", ((lx[0] + rx[0]) / 2, (ly[0] + ry[0]) / 2 - 0.55),
                ha="center", fontsize=12)

    # centreline label + flow direction
    ax.annotate(r"centreline  $\tilde y(\tilde x,\,\tilde t)$",
                (wavelength * 1.75, amp * np.sin(k * wavelength * 1.75) + 1.15),
                color=L.COLORS["channel"], fontsize=12)
    ax.add_patch(FancyArrowPatch((0.3, -amp - 2.1), (3.3, -amp - 2.1),
                 arrowstyle="-|>", mutation_scale=18, color=L.COLORS["water"], lw=2.4))
    ax.annotate("flow", (1.8, -amp - 1.8), color=L.COLORS["water"], fontsize=12, ha="center")

    # inner/outer bank labels at the first apex
    ax.annotate("outer bank\n(erosion)", (xa, ya + b + 0.35), ha="center",
                color=L.COLORS["erosion"], fontsize=10.5)
    ax.annotate("inner bank\n(point bar)", (xa, ya - b - 0.75), ha="center",
                color=L.COLORS["bank"], fontsize=10.5)

    ax.set_title("Meandering channel with erodible banks (definition sketch)")
    ax.set_xlabel(r"downstream  $\tilde x$")
    ax.set_ylabel(r"$\tilde y$")
    ax.set_aspect("equal")
    ax.set_ylim(-amp - 2.6, amp + b + 1.6)
    ax.grid(False)
    L.save_fig(fig, "fig01_planform_definition")


# --------------------------------------------------------------------------- #
#  fig02 -- channel cross-section (paper Fig. 2b)
# --------------------------------------------------------------------------- #
def fig02_cross_section():
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    b = 1.0
    n = np.linspace(-b, b, 200)

    # bed: deep scour at outer (right) bank, shallow point bar at inner (left) bank
    bed = -0.35 - 0.9 * (0.5 * (n / b + 1.0)) ** 1.6      # rises toward inner bank
    surf = 0.02 * n / b                                    # superelevation tilt
    ax.fill_between(n, bed, surf, color=L.COLORS["water_fill"], zorder=1)
    ax.fill_between(n, -1.5, bed, color=L.COLORS["deposition"], alpha=0.55, zorder=0)
    ax.plot(n, bed, color=L.COLORS["bank"], lw=2, zorder=3, label="bed  " + r"$\tilde\eta$")
    ax.plot(n, surf, color=L.COLORS["water"], lw=2, zorder=3,
            label=r"water surface  $\tilde\zeta$")

    # banks (vertical-ish walls)
    ax.plot([-b, -b], [bed[0], surf[0]], color=L.COLORS["bank"], lw=2)
    ax.plot([b, b], [bed[-1], surf[-1]], color=L.COLORS["bank"], lw=2)

    # depth arrow h and width 2b
    ax.annotate("", (0.55, surf[150]), (0.55, bed[150]),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax.annotate(r"$\tilde h$", (0.62, (surf[150] + bed[150]) / 2), fontsize=13)
    ax.annotate("", (b, 0.28), (-b, 0.28),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax.annotate(r"$2b$", (0, 0.34), ha="center", fontsize=12)

    ax.annotate("point bar\n(inner bank)", (-b + 0.05, -1.15), fontsize=10.5,
                color=L.COLORS["bank"], ha="left")
    ax.annotate("scour pool\n(outer bank)", (b - 0.05, -1.15), fontsize=10.5,
                color=L.COLORS["erosion"], ha="right")

    ax.set_title("Cross-section A–A' through a bend")
    ax.set_xlabel(r"transverse  $\tilde n$")
    ax.set_ylabel("elevation")
    ax.set_ylim(-1.5, 0.6)
    ax.legend(loc="upper center", ncol=2)
    ax.grid(False)
    L.save_fig(fig, "fig02_cross_section")


# --------------------------------------------------------------------------- #
#  fig03 -- secondary-flow schematic (parameter A, Eq. 6)
# --------------------------------------------------------------------------- #
def fig03_secondary_flow():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    b = 1.0
    n = np.linspace(-b, b, 200)
    bed = -0.4 - 0.95 * (0.5 * (n / b + 1.0)) ** 1.6
    surf = 0.06 * n / b
    ax.fill_between(n, bed, surf, color=L.COLORS["water_fill"], zorder=0)
    ax.plot(n, bed, color=L.COLORS["bank"], lw=2, zorder=3)
    ax.plot(n, surf, color=L.COLORS["water"], lw=2, zorder=3)

    # helical secondary circulation: surface -> outer, bed -> inner (one cell)
    yy, zz = np.meshgrid(np.linspace(-0.8 * b, 0.8 * b, 11),
                         np.linspace(-0.9, -0.05, 7))
    u = 0.5 * (zz + 0.5)          # transverse velocity: +toward outer near surface
    w = -0.35 * yy / b
    ax.quiver(yy, zz, u, w, color=L.COLORS["channel"], alpha=0.8,
              scale=9, width=0.004, zorder=2)

    # centrifugal force near the surface (outward, toward outer bank)
    ax.add_patch(FancyArrowPatch((-0.1, 0.16), (0.75, 0.16), arrowstyle="-|>",
                 mutation_scale=16, color=L.COLORS["erosion"], lw=2.2))
    ax.annotate("centrifugal drift\n(high-momentum surface water)",
                (0.32, 0.30), ha="center", color=L.COLORS["erosion"], fontsize=10)

    # high-velocity core marker near outer bank
    ax.scatter([0.72], [-0.3], s=420, color=L.COLORS["velocity"], alpha=0.35, zorder=1)
    ax.annotate("high-velocity\ncore", (0.72, -0.3), ha="center", va="center",
                fontsize=9.5, color="#8a4b00")

    ax.annotate("inner bank\n(shoaling, deposition)", (-b, -1.2), ha="left",
                fontsize=10, color=L.COLORS["bank"])
    ax.annotate("outer bank\n(deep, erosion)", (b, -1.2), ha="right",
                fontsize=10, color=L.COLORS["erosion"])

    ax.set_title(r"Secondary (helical) flow in a bend  —  closure  $\eta'/H=-A\,\mathcal{C}'\tilde n$")
    ax.set_xlabel(r"transverse  $\tilde n$   (inner $\to$ outer)")
    ax.set_ylabel("depth")
    ax.set_xlim(-b - 0.15, b + 0.15)
    ax.set_ylim(-1.35, 0.5)
    ax.grid(False)
    L.save_fig(fig, "fig03_secondary_flow")


# --------------------------------------------------------------------------- #
#  fig04 -- bank-erosion geometry (paper Fig. 3, Eq. 11)
# --------------------------------------------------------------------------- #
def fig04_bank_erosion():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.linspace(0, 6, 400)
    theta = np.deg2rad(28)
    y0 = 0.55 * x                                   # a locally straight bank segment
    dn = 0.9                                          # normal erosion distance xi*dt
    # shift outward along the bank normal
    nx, ny = -np.sin(np.arctan(0.55)), np.cos(np.arctan(0.55))
    y1x, y1y = x + dn * nx, y0 + dn * ny

    ax.plot(x, y0, color=L.COLORS["bank"], lw=2.5, label=r"bank at $\tilde t$")
    ax.plot(y1x, y1y, "--", color=L.COLORS["erosion"], lw=2.5,
            label=r"bank at $\tilde t+\Delta\tilde t$")

    # erosion normal vector xi*dt and lateral shift (dy/dt)*dt
    x0 = 3.0
    p0 = np.array([x0, 0.55 * x0])
    p_norm = p0 + dn * np.array([nx, ny])
    p_lat = p0 + np.array([0, dn / np.cos(theta)])   # vertical (lateral y) shift
    ax.add_patch(FancyArrowPatch(p0, p_norm, arrowstyle="-|>", mutation_scale=15,
                 color=L.COLORS["erosion"], lw=2))
    ax.add_patch(FancyArrowPatch(p0, p_lat, arrowstyle="-|>", mutation_scale=15,
                 color=L.COLORS["channel"], lw=2))
    ax.annotate(r"$\tilde\zeta\,\Delta\tilde t$", p0 + 0.5 * (p_norm - p0) + [0.18, -0.15],
                color=L.COLORS["erosion"], fontsize=13)
    ax.annotate(r"$\dfrac{\partial\tilde y}{\partial\tilde t}\,\Delta\tilde t$",
                (p_lat[0] + 0.12, p_lat[1] - 0.05), color=L.COLORS["channel"], fontsize=13)
    ax.add_patch(Arc(p0, 1.3, 1.3, angle=0,
                     theta1=90, theta2=np.degrees(np.arctan2(ny, nx)),
                     color="k", lw=1.2))
    ax.annotate(r"$\theta$", p0 + [0.02, 0.72], fontsize=13)

    ax.annotate(r"$\gamma\,\dfrac{\partial\tilde y}{\partial\tilde t}=\tilde\zeta,"
                r"\quad \gamma=\cos\theta$",
                (0.5, 3.4), fontsize=15,
                bbox=dict(boxstyle="round", fc="#f3f0fa", ec=L.COLORS["curvature"]))

    ax.set_title("Bank erosion  →  centreline migration (constant width)")
    ax.set_xlabel(r"$\tilde x$")
    ax.set_ylabel(r"$\tilde y$")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(False)
    L.save_fig(fig, "fig04_bank_erosion")


# --------------------------------------------------------------------------- #
#  fig05 -- the phase-lag mechanism (THE money figure)
# --------------------------------------------------------------------------- #
def fig05_phase_lag():
    p = L.PARAMS
    kk = L.k_OM(p.Cf, p.A, p.F)           # display at the most-unstable wavenumber
    wavelength = 2 * np.pi / kk
    x = np.linspace(0, 2.4 * wavelength, 1400)

    # Cosine centreline so curvature C = -y'' is in phase with curvature_of_sine.
    curv = L.curvature_of_sine(x, kk, eps=1.0)          # proportional to cos(k x)
    ub = L.near_bank_velocity(x, kk, p.Cf, p.A, p.F, bstar=10.0, eps=1.0)
    curv_n = curv / np.max(np.abs(curv))
    ub_n = ub / np.max(np.abs(ub))
    lag = float(L.phase_lag_deg(kk, p.Cf, p.A, p.F))
    lag_frac = lag / 360.0

    amp_pf, b_pf = 0.14 * wavelength, 0.045 * wavelength   # visible planform scale
    yc = amp_pf * np.cos(kk * x)
    dy = -amp_pf * kk * np.sin(kk * x)
    left, right = banks(x, yc, dy, b_pf)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.1, 1.2], hspace=0.14))

    # -- top: planform with erosion arrows a phase-lag downstream of each apex --
    ax = axes[0]
    ax.add_patch(Polygon(channel_polygon(left, right), closed=True,
                         facecolor=L.COLORS["water_fill"], edgecolor="none"))
    ax.plot(left[0], left[1], color=L.COLORS["bank"], lw=1.6)
    ax.plot(right[0], right[1], color=L.COLORS["bank"], lw=1.6)
    ax.plot(x, yc, "--", color=L.COLORS["channel"], lw=1.2)

    xmax = x[-1]
    labelled = False
    # apices: crests at x = m*lambda (outer bank up), troughs at lambda/2+m*lambda (down)
    for m in range(0, int(xmax / wavelength) + 1):
        for xa, sgn in [(m * wavelength, +1.0),
                        (m * wavelength + 0.5 * wavelength, -1.0)]:
            if xa > xmax:
                continue
            ax.axvline(xa, color=L.COLORS["apex"], lw=0.8, ls=":", alpha=0.45)
            ax.plot(xa, sgn * amp_pf, "o", color=L.COLORS["apex"], ms=4, zorder=5)
            xe = xa + lag_frac * wavelength            # erosion maximum location
            if xe > xmax:
                continue
            yb = sgn * amp_pf + sgn * b_pf
            ax.add_patch(FancyArrowPatch((xe, yb), (xe, yb + sgn * 0.6 * b_pf),
                         arrowstyle="-|>", mutation_scale=13,
                         color=L.COLORS["erosion"], lw=2.0, zorder=6))
            if not labelled:
                ax.annotate("apex", (xa, sgn * (amp_pf + b_pf) + 0.35 * b_pf),
                            ha="center", fontsize=9.5, color=L.COLORS["apex"])
                ax.annotate("erosion max\n(downstream)",
                            (xe + 0.02 * wavelength, yb + sgn * 1.1 * b_pf),
                            ha="left", va="center", fontsize=9,
                            color=L.COLORS["erosion"])
                labelled = True
    ax.set_ylabel("planform")
    ax.set_ylim(-amp_pf - 2.4 * b_pf, amp_pf + 2.4 * b_pf)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("The phase-lag mechanism: erosion peaks downstream of the apex → growth + downstream migration")

    # -- bottom: curvature vs near-bank velocity, with the lag annotated --
    ax = axes[1]
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(x, curv_n, color=L.COLORS["curvature"], lw=2.2,
            label=r"centreline curvature  $\mathcal{C}(s)$")
    ax.plot(x, ub_n, color=L.COLORS["velocity"], lw=2.2,
            label=r"near-bank velocity  $u_b(s)$")
    # mark the lag between the first curvature peak and the u_b peak
    xc_peak = 0.25 * wavelength
    xu_peak = xc_peak + lag_frac * wavelength
    ax.plot([xc_peak, xc_peak], [0, 1], color=L.COLORS["curvature"], lw=0.8, ls=":")
    ax.plot([xu_peak, xu_peak], [0, 1], color=L.COLORS["velocity"], lw=0.8, ls=":")
    ax.annotate("", (xu_peak, 1.08), (xc_peak, 1.08),
                arrowprops=dict(arrowstyle="<->", color=L.COLORS["erosion"], lw=1.6))
    ax.annotate(rf"phase lag $\approx {lag:.0f}^\circ \approx {lag_frac:.2f}\,\lambda$",
                ((xc_peak + xu_peak) / 2, 1.18), ha="center",
                color=L.COLORS["erosion"], fontsize=11)
    ax.set_ylabel("normalised")
    ax.set_xlabel(r"downstream distance  $s$")
    ax.set_ylim(-1.25, 1.45)
    ax.legend(loc="lower right", ncol=2)

    L.save_fig(fig, "fig05_phase_lag")


def main():
    print("01_schematics.py -> figures/fig01..fig05")
    fig01_planform()
    fig02_cross_section()
    fig03_secondary_flow()
    fig04_bank_erosion()
    fig05_phase_lag()


if __name__ == "__main__":
    main()
