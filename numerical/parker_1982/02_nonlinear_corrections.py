#!/usr/bin/env python3
"""Amplitude corrections and the wavelength-shift regime map.

Produces figures/fig04..fig06:

    fig04  alpha(0)/alpha0M and c(0)/c0M vs delta0M (Eqs. 26a, 32):
           finite amplitude slows growth and migration
    fig05  k_M/k0M vs delta0M for alluvial (e below/above e*) and incised:
           the sign of the wavelength shift (Eq. 30)
    fig06  the (F, e) regime map with the e*(F) boundary and the printed
           anchors 5.1 / 2.7 (paper Fig. 5 replica)

Usage
-----
    micromamba run -n fourcastnetv2 python 02_nonlinear_corrections.py
"""
from __future__ import annotations

import numpy as np

import parker_lib as L

plt = L.set_style()

D0 = np.linspace(0.0, 0.5, 200)


def fig04_corrections():
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True)
    for e, col in ((0.0, "#a6cee3"), (2.0, "#1f78b4"), (5.0, "#08306b")):
        p = L.Params(A=2.89, F=0.3, e=e)
        axs[0].plot(D0, [L.alpha_kOM(d, p) / L.alpha0M(p) for d in D0],
                    color=col, lw=2.2, label=rf"$e={e}$")
        axs[1].plot(D0, [L.c_kM(d, p) / L.c0M(p) for d in D0],
                    color=col, lw=2.2)
    inc = L.Params(A=0.0, F=0.3, e=1.0)
    axs[0].plot(D0, [L.alpha_kOM(d, inc) / L.alpha0M(inc) for d in D0], "--",
                color=L.COLORS["skew"], lw=2.0, label="incised, $e=1$")
    axs[1].plot(D0, [L.c_kM(d, inc) / L.c0M(inc) for d in D0], "--",
                color=L.COLORS["skew"], lw=2.0)
    axs[0].set_ylabel(r"$\alpha(0)/\alpha_{0M}$  (Eq. 26a)")
    axs[1].set_ylabel(r"$c(0)|_{k_M}/c_{0M}$  (Eq. 32)")
    for ax in axs:
        ax.set_xlabel(r"$\delta_{0M} = k_{0M}\,\epsilon$")
        ax.axhline(1.0, color="#cccccc", lw=0.8)
    axs[0].legend(fontsize=10)
    fig.suptitle("Finite amplitude slows both growth and downstream migration "
                 "(Sec. 6(1))", y=1.02)
    L.save_fig(fig, "fig04_corrections")


def fig05_kM_shift():
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    cases = [(L.Params(A=2.89, F=0.05, e=0.0), L.COLORS["erosion"],
              "alluvial, $e=0<e^*$: wavelength decreases"),
             (L.Params(A=2.89, F=0.05, e=6.0), L.COLORS["nonlinear"],
              "alluvial, $e=6>e^*{=}5.1$: wavelength increases"),
             (L.Params(A=0.0, F=0.3, e=2.0), L.COLORS["skew"],
              "incised: $k_M<k_{0M}$ always")]
    for p, col, lab in cases:
        ax.plot(D0, [L.kM_over_k0M(d, p) for d in D0], color=col, lw=2.4,
                label=lab)
    ax.axhline(1.0, color="#999999", lw=1.0)
    ax.set_xlabel(r"$\delta_{0M}$")
    ax.set_ylabel(r"$k_M/k_{0M}$  (Eq. 30)")
    ax.set_title("Which way does the selected wavelength move with amplitude?")
    ax.legend(fontsize=10)
    L.save_fig(fig, "fig05_kM_shift")


def fig06_regime_map():
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    Fg = np.linspace(1e-3, 2.2, 300)
    estar = np.array([L.e_threshold(F) for F in Fg])
    ax.plot(Fg, estar, color=L.COLORS["channel"], lw=2.6)
    ax.fill_between(Fg, estar, 6.0, color="#c7e0f0", alpha=0.6)
    ax.fill_between(Fg, 0.0, np.clip(estar, 0, None), color="#fde2cf",
                    alpha=0.6)
    ax.text(0.9, 4.6, "increased wavelength\n($k_M<k_{0M}$)", fontsize=11,
            color=L.COLORS["channel"], ha="center")
    ax.text(0.55, 0.9, "decreased wavelength\n($k_M>k_{0M}$)", fontsize=11,
            color="#b3591c", ha="center")
    ax.plot(0.0, 5.117, "o", ms=8, color=L.COLORS["pde"])
    ax.annotate("  $e^*=5.1$ ($F\\ll1$, printed)", (0.0, 5.117), fontsize=10)
    ax.plot(1.0, L.e_threshold(1.0), "o", ms=8, color=L.COLORS["pde"])
    ax.annotate("  $e^*=2.7$ ($F=1$, printed)", (1.0, L.e_threshold(1.0)),
                fontsize=10)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 6)
    ax.set_xlabel("$F$")
    ax.set_ylabel("$e$")
    ax.set_title("Paper Fig. 5 replica: the $e^*(F)$ boundary from Eq. (30)\n"
                 "(alluvial, $A=2.89$)")
    L.save_fig(fig, "fig06_regime_map")


def main():
    print("02_nonlinear_corrections.py -> figures/fig04..fig06")
    fig04_corrections()
    fig05_kM_shift()
    fig06_regime_map()


if __name__ == "__main__":
    main()
