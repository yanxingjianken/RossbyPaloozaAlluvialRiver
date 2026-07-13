#!/usr/bin/env python3
"""Fattening & skewing geometry -- paper Figs. 1, 2, 4 recreated.

Produces figures/fig01..fig03:

    fig01  the sine-generated curve (theta_max = 115 deg, paper Fig. 1a)
           and the fattening operation cos kx - 0.1 cos 3kx (Fig. 1b idea)
    fig02  what J_F and J_S do to a bend: third-mode family sweep of
           Eq. (5) -- fattening rounds the apex, skewing leans it
    fig03  Eq. (5) with the published Beaver River constants
           (d0 = 0.98, J_F = 0.073, J_S = 0.103; paper Fig. 4's solid line)
           + machine check: `harmonics` recovers the constants exactly

Usage
-----
    micromamba run -n fourcastnetv2 python 01_fattening_skewing_geometry.py
"""
from __future__ import annotations

import numpy as np

import parker_lib as L

plt = L.set_style()


def fig01_sine_generated():
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.6))
    x, y = L.sine_generated_curve(115.0, n_wave=2.0)
    axs[0].plot(x, y, color=L.COLORS["channel"], lw=2.4)
    axs[0].set_aspect("equal")
    axs[0].set_title(r"sine-generated curve, $\theta_{max}=115^\circ$ "
                     "(paper Fig. 1a)")
    axs[0].set_xlabel("x")

    xx = np.linspace(0, 2, 600)
    axs[1].plot(xx, np.cos(2 * np.pi * xx), "--", color=L.COLORS["linear"],
                lw=1.8, label=r"$\cos kx$")
    axs[1].plot(xx, np.cos(2 * np.pi * xx) - 0.1 * np.cos(6 * np.pi * xx),
                color=L.COLORS["fatten"], lw=2.4,
                label=r"$\cos kx - 0.1\cos 3kx$")
    axs[1].legend(fontsize=10)
    axs[1].set_title("subtracting a third mode fattens the bend (Fig. 1b)")
    axs[1].set_xlabel(r"$x/\lambda$")
    L.save_fig(fig, "fig01_sine_generated")


def fig02_jf_js_family():
    fig, axs = plt.subplots(2, 1, figsize=(10.2, 6.6), sharex=True)
    xx = np.linspace(0, 2, 800)
    k = 2 * np.pi
    for jf, col in ((0.0, "#cccccc"), (0.05, "#b09cd0"), (0.10, "#8867b8"),
                    (0.15, L.COLORS["fatten"])):
        axs[0].plot(xx, L.planform_eq5(xx, 1.0, k, 1.0, jf, 0.0), color=col,
                    lw=2.0, label=rf"$\delta_0^2 J_F={jf}$")
    for js, col in ((0.0, "#cccccc"), (0.05, "#f3c08a"), (0.10, "#eda14f"),
                    (0.15, L.COLORS["skew"])):
        axs[1].plot(xx, L.planform_eq5(xx, 1.0, k, 1.0, 0.0, js), color=col,
                    lw=2.0, label=rf"$\delta_0^2 J_S={js}$")
    axs[0].set_title("fattening $J_F$: apex rounds and broadens")
    axs[1].set_title("skewing $J_S$: bend leans upstream/downstream")
    axs[1].set_xlabel(r"$x/\lambda$")
    for ax in axs:
        ax.legend(fontsize=9, ncol=4)
    L.save_fig(fig, "fig02_jf_js_family")


def fig03_beaver():
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    xx = np.linspace(0, 3, 1200)
    k = 2 * np.pi
    yb = L.planform_eq5(xx, 1.0, k, **{k_: L.BEAVER[k_] for k_ in ()},
                        delta0=L.BEAVER["delta0"], JF=L.BEAVER["JF"],
                        JS=L.BEAVER["JS"])
    y1 = np.cos(k * xx)
    ax.plot(xx, y1, "--", color=L.COLORS["linear"], lw=1.6,
            label="pure first mode")
    ax.plot(xx, yb, color=L.COLORS["channel"], lw=2.6,
            label=(r"Eq. (5), Beaver River fit: $\delta_0=0.98$, "
                   r"$J_F=0.073$, $J_S=0.103$"))
    ax.set_xlabel(r"$x/\lambda$")
    ax.set_title("The published Beaver River planform approximation "
                 "(paper Fig. 4, solid line)")
    ax.legend(fontsize=10, loc="lower right")
    L.save_fig(fig, "fig03_beaver")

    # machine check: harmonic extraction recovers the constants exactly
    c = L.harmonics(yb, xx, k, 3)
    c = c * np.exp(-1j * np.angle(c[0])) ** np.arange(1, 4)
    # y contains -d0^2 (J_F cos + J_S sin): c3 = -d0^2 J_F + i d0^2 J_S
    jf = -c[2].real / L.BEAVER["delta0"] ** 2
    js = c[2].imag / L.BEAVER["delta0"] ** 2
    print(f"  harmonic recovery: J_F = {jf:.4f} (0.073), J_S = {js:.4f} (0.103)")
    assert abs(jf - L.BEAVER["JF"]) < 1e-10 and abs(js - L.BEAVER["JS"]) < 1e-10


def main():
    print("01_fattening_skewing_geometry.py -> figures/fig01..fig03")
    fig01_sine_generated()
    fig02_jf_js_family()
    fig03_beaver()


if __name__ == "__main__":
    main()
