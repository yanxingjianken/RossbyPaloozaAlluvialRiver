#!/usr/bin/env python3
"""Phi(M) -- the entropy backbone curve and where the seven transects sit.

Produces figures/fig01..fig02:

    fig01  Phi(M) with the seven Table-2 transects, literature ranges from
           the paper's Sec. 4.1 shaded, and the FreibergerMulde CS3 printed
           inconsistency called out explicitly
    fig02  Phi = Um/Umax identity check: printed Phi vs ADCP ratio vs
           Phi(M) row by row (the anomaly is visible, not hidden)

Usage
-----
    micromamba run -n fourcastnetv2 python 01_phi_curve.py
"""
from __future__ import annotations

import numpy as np

import bahmanpouri_lib as L

plt = L.set_style()

T2 = L.load_table2()

# Literature Phi(M) ranges quoted in the paper (Sec. 4.1).
LIT = [
    ("US natural channels (Chiu 2000)", 0.66, 0.80),
    ("Tiber River (Moramarco 2019)", 0.60, 0.68),
    ("Godavari/Ulhas (Vyas 2021)", 0.63, 0.69),
    ("Amazon (Bahmanpouri 2022)", 0.40, 0.61),
]


def fig01_phi_curve():
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    Mg = np.linspace(0.01, 8.0, 400)
    ax.plot(Mg, L.phi_of_M(Mg), "-", color=L.COLORS["entropy"], lw=2.4,
            label=r"$\Phi(M) = \frac{e^M}{e^M-1} - \frac{1}{M}$  (Eq. 2)")

    for i, (lab, lo, hi) in enumerate(LIT):
        ax.axhspan(lo, hi, color=f"C{i+2}", alpha=0.10)
        ax.text(8.05, (lo + hi) / 2, lab, fontsize=8.5, va="center",
                color=f"C{i+2}")

    for c in T2:
        anom = c.name == "FreibergerMulde CS3"
        col = L.COLORS["anomaly"] if anom else L.COLORS["adcp"]
        ax.plot(c.M, c.phi, "o", color=col, ms=8, mec="white", mew=0.7, zorder=5)
    ax.annotate("FM CS3 as printed:\n$(M{=}1.16,\\ \\Phi{=}0.678)$ is off the "
                "curve\n$\\Phi(1.16){=}0.595{=}0.68/1.14$ (its own ADCP ratio)",
                xy=(1.16, 0.678), xytext=(2.6, 0.545), fontsize=10,
                color=L.COLORS["anomaly"],
                arrowprops=dict(arrowstyle="->", color=L.COLORS["anomaly"]))
    ax.plot(1.16, float(L.phi_of_M(1.16)), "s", color=L.COLORS["anomaly"],
            ms=8, mfc="none", mew=1.8, zorder=5)

    ax.set_xlim(0, 8)
    ax.set_ylim(0.38, 0.95)
    ax.set_xlabel("entropy parameter $M$")
    ax.set_ylabel(r"$\Phi(M) = U_m / U_{max}$")
    ax.set_title("The entropy backbone: seven ADCP-calibrated transects on "
                 r"$\Phi(M)$ (Table 2)")
    ax.legend(loc="lower right", fontsize=10)
    L.save_fig(fig, "fig01_phi_curve")


def fig02_identity_check():
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    names = [c.name.replace("FreibergerMulde", "FM ") for c in T2]
    xpos = np.arange(len(T2))
    w = 0.27
    ax.bar(xpos - w, [c.phi for c in T2], w, color=L.COLORS["adcp"],
           label=r"printed $\Phi(M)$")
    ax.bar(xpos, [c.Um / c.Umax for c in T2], w, color=L.COLORS["water"],
           label=r"ADCP ratio $U_m/U_{max}$")
    ax.bar(xpos + w, [float(L.phi_of_M(c.M)) for c in T2], w,
           color=L.COLORS["entropy"], label=r"$\Phi$(printed $M$)")
    ax.axvspan(len(T2) - 1.5 + 0.08, len(T2) - 0.5 - 0.08, color=L.COLORS["anomaly"],
               alpha=0.10)
    ax.text(len(T2) - 1, 0.755, "printed $\\Phi$ inconsistent\n(codified typo)",
            ha="center", fontsize=9.5, color=L.COLORS["anomaly"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0.5, 0.8)
    ax.set_ylabel(r"$\Phi$")
    ax.set_title("Row-by-row identity: three routes to $\\Phi$ agree on 6/7 transects")
    ax.legend(fontsize=10, ncol=3, loc="upper left")
    L.save_fig(fig, "fig02_identity_check")


def main():
    print("01_phi_curve.py -> figures/fig01..fig02")
    fig01_phi_curve()
    fig02_identity_check()


if __name__ == "__main__":
    main()
