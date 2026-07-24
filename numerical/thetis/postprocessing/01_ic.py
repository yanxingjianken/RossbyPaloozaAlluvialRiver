#!/usr/bin/env python3
"""Initial-condition figures: xOy plan view and yOz cross-section, both runs.

The IC is fully analytic (geometry.py + the exact base state), so this runs
without Firedrake and is the FIRST thing to look at.  Prior work in this repo
lost an entire build to an initial condition that was posed backwards while
every numerical check passed -- the thing that caught it was a human looking at
frame 0.  Vision-check these before any production run.

    python postprocessing/01_ic.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_lib as pp  # noqa: E402
import geometry as geo  # noqa: E402

plt = pp.set_style()
from matplotlib.patches import Rectangle  # noqa: E402


def _plan_field(d: geo.Design, nx: int = 900, nn: int = 61):
    """(X, Y, depth, u, v) of the base state on the initial planform."""
    x = np.linspace(0.0, d.L, nx)
    ntil = np.linspace(-1.0, 1.0, nn)
    yN, yS = geo.initial_banks(x, d)
    c = geo.centreline(yN, yS)
    b = geo.half_width(yN, yS)

    X = np.repeat(x[None, :], nn, axis=0)
    Y = c[None, :] + ntil[:, None] * b[None, :]
    H = geo.base_depth(ntil, d)[:, None] * np.ones_like(X)
    ub = geo.base_velocity(ntil, d)[:, None] * np.ones_like(X)

    (tx, ty), _ = pp.channel_frame(x, c)
    u = ub * tx[None, :]
    v = ub * ty[None, :]
    return x, ntil, X, Y, H, u, v, yN, yS, c


def make_ic_figure(m: int):
    cfg = geo.Config(n_wave=m)
    d = geo.build_design(cfg)
    x, ntil, X, Y, H, u, v, yN, yS, c = _plan_field(d)

    fig = plt.figure(figsize=(14.5, 10.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.30, 1.20], hspace=0.60)

    # ---------------- (a) full domain, aspect exaggerated ----------------
    ax = fig.add_subplot(gs[0])
    exag = 8.0
    ax.plot(x, yN, "k-", lw=1.4)
    ax.plot(x, yS, "k-", lw=1.4)
    ax.plot(x, c, color="0.55", lw=0.9, ls="--", label="centreline")
    ax.fill_between(x, yS, yN, color="#cfe3f7", zorder=0)
    span = 1.35 * (np.max(yN) - np.min(yS)) / 2.0
    for x0, x1, lab, col in [(0.0, d.x_m0, "straight entry reach\n(flow conditioning)", "#f7e2b0"),
                             (d.x_m1, d.L, "straight exit", "#f7e2b0")]:
        ax.add_patch(Rectangle((x0, -span), x1 - x0, 2 * span,
                               color=col, alpha=0.55, zorder=-1))
        ax.text(0.5 * (x0 + x1), -0.60 * span, lab, ha="center", va="center",
                fontsize=9.0, color="#7a5c00")
    ax.axvline(d.x_m0, color="0.4", lw=0.8, ls=":")
    ax.axvline(d.x_m1, color="0.4", lw=0.8, ls=":")
    x_sec = _apex_x(d, m)
    ax.axvline(x_sec, color="crimson", lw=1.6, ls="-", alpha=0.85)
    ax.text(x_sec, 0.80 * span, "yOz section", color="crimson", fontsize=9.5,
            ha="center", va="bottom")
    ax.set_xlim(0, d.L)
    ax.set_ylim(-span, span)
    ax.set_aspect(exag)
    ax.set_xlabel("x  [m]  (down-valley)")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"(a) plan view, whole domain   L = {d.L:.0f} m, "
                 f"W = {d.W:.1f} m   [y stretched x{exag:g}]")
    ax.grid(alpha=0.2)

    # ---------------- (b) zoom, TRUE aspect ------------------------------
    ax = fig.add_subplot(gs[1])
    lam_m = d.L_m / m
    a0_m = d.cfg.amp0_over_b * d.b
    xz0 = x_sec - 0.25 * lam_m
    xz1 = x_sec + 0.25 * lam_m
    sel = (x >= xz0) & (x <= xz1)
    pcm = ax.pcolormesh(X[:, sel], Y[:, sel], H[:, sel],
                        cmap="Blues", shading="gouraud")
    step_x = max(1, int(sel.sum() // 16))
    step_n = max(1, len(ntil) // 6)
    ax.quiver(X[::step_n, sel][:, ::step_x], Y[::step_n, sel][:, ::step_x],
              u[::step_n, sel][:, ::step_x], v[::step_n, sel][:, ::step_x],
              color="0.15", scale=11.0, width=0.0045, alpha=0.85)
    ax.plot(x[sel], yN[sel], "k-", lw=1.8)
    ax.plot(x[sel], yS[sel], "k-", lw=1.8)
    ax.axvline(x_sec, color="crimson", lw=1.6)
    ax.set_aspect("equal")          # TRUE aspect; see the title
    ax.set_xlim(xz0, xz1)
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"(b) TRUE aspect, {(xz1 - xz0) / lam_m:.2f} wavelength:  "
                 f"colour = frozen depth H(n),  arrows = base jet\n"
                 fr"the bend itself is NOT visible here and cannot be -- $a_0$={a0_m:.2f} m "
                 fr"is {100 * d.cfg.amp0_over_b / 2:.1f}% of the {d.W:.1f} m width; "
                 fr"see (a), stretched x{exag:g}, for the planform", fontsize=11)
    cb = fig.colorbar(pcm, ax=ax, pad=0.012, fraction=0.028)
    cb.set_label("H  [m]")
    ax.grid(False)

    # ---------------- (c) yOz cross-section ------------------------------
    ax = fig.add_subplot(gs[2])
    nfine = np.linspace(-1, 1, 401)
    n_m = nfine * d.b
    Hn = geo.base_depth(nfine, d)
    # reference the section to its own free surface: absolute elevations here
    # are dominated by -I*x (5.05 m at x=1123 m) and hide the shape.
    eta_ref = float(d.I * x_sec)
    zb = geo.bed_elevation(x_sec, nfine, d, eta_ref=eta_ref)
    eta0 = float(geo.base_elevation(x_sec, d, eta_ref=eta_ref))

    ax.fill_between(n_m, zb.min() - 0.35, zb, color="#c8a980", zorder=1)
    ax.fill_between(n_m, zb, eta0, color="#cfe3f7", zorder=0)
    ax.plot(n_m, zb, color="#6b4c1e", lw=2.4, zorder=3, label=r"bed $z_b(n)$  (frozen)")
    ax.axhline(eta0, color="#1f6fb4", lw=2.4, zorder=3,
               label=r"free surface $\eta_0$  (flat in $n$)")
    for sgn in (-1, 1):
        ax.axvline(sgn * d.b, color="k", lw=2.2, zorder=4)
    ax.set_xlim(-1.22 * d.b, 1.22 * d.b)
    ax.set_ylim(zb.min() - 0.35, eta0 + 0.34)
    ax.set_xlabel("n  [m]   (transverse, 0 = centreline)")
    ax.set_ylabel("elevation rel. to local\nfree surface  [m]")
    ax.grid(alpha=0.25)

    axu = ax.twinx()
    axu.plot(n_m, geo.base_velocity(nfine, d), color="crimson", lw=2.2,
             label=r"base jet $\bar u(n)$")
    axu.set_ylabel(r"$\bar u$  [m/s]", color="crimson")
    axu.tick_params(axis="y", colors="crimson")
    axu.set_ylim(0, 1.28 * geo.base_velocity(0.0, d))
    axu.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axu.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=10)
    ax.set_title(f"(c) yOz cross-section at x = {x_sec:.0f} m  (local surface = 0) "
                 f"(bend apex)   H: {Hn.min():.3f}-{Hn.max():.3f} m,  "
                 fr"$\bar u$: {geo.base_velocity(1.0, d):.3f}-"
                 fr"{geo.base_velocity(0.0, d):.3f} m/s")

    a0 = cfg.amp0_over_b * d.b
    k_m = geo.wavenumber_of(m, d)
    fig.suptitle(
        fr"IC, wavenumber $m={m}$:  $\lambda$={lam_m:.1f} m = {lam_m / d.W:.2f} W,  "
        fr"$k$={k_m:.4e} ($k/k_{{OM}}$={k_m / d.k_OM:.2f}, $k/k_c$={k_m / d.k_c:.2f}),  "
        fr"$a_0$={a0:.2f} m = {cfg.amp0_over_b:.2f} b,  "
        fr"$C_f$={cfg.Cf}, $F$={cfg.F_ref}, $A$={cfg.A_ikeda} (incised)",
        fontsize=12.5, y=0.985)

    pp.save_fig(fig, f"IC_m{m}")
    return d, x_sec


def _apex_x(d: geo.Design, m: int) -> float:
    """x of a bend apex near the middle of the meander reach.

    c = a0*sin(2*pi*m*(x-x_m0)/L_m); apexes are the quarter-wave points.
    """
    lam_m = d.L_m / m
    j = m // 2                       # middle-ish bend
    return d.x_m0 + (j + 0.25) * lam_m


def main():
    import sys as _sys
    if len(_sys.argv)>1 and _sys.argv[1].startswith("A"): pp.set_case(_sys.argv[1])
    print("=" * 74)
    print("01_ic.py -- initial-condition figures")
    print("=" * 74)
    for m in (4, 8):
        d, xs = make_ic_figure(m)
        yN, yS = geo.initial_banks(np.linspace(0, d.L, d.nx + 1), d)
        print(f"  m={m}: apex section at x = {xs:.1f} m, "
              f"bank amplitude {geo.centreline(yN, yS).max():.3f} m")
    print("\nVISION-CHECK BOTH FIGURES BEFORE ANY PRODUCTION RUN.")


if __name__ == "__main__":
    main()
