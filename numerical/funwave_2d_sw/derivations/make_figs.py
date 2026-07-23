#!/usr/bin/env python3
"""Figures for funwave_meander_model.tex -- built from run_meander.py itself, so the
document can never drift from the code.

    micromamba run -n fourcastnetv2 python derivations/make_figs.py
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run_meander as rm  # noqa: E402

CFG = rm.CONFIG
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def fig_plan(path):
    """xOy: the two reaches, drawn to true down-valley scale with the lateral axis
    exaggerated so the bends are legible."""
    L = rm.reach_length(CFG)
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.4), sharex=True)
    x = np.linspace(0, L, 4001)
    for ax, r, col in zip(axes, rm.RUNS, ("#1D9E75", "#7F77DD")):
        lam = r["lam"]
        xc_, yc_ = rm.centreline(lam, CFG)[:2]
        x = xc_; yc = yc_; A = rm.amplitude(lam, CFG)
        ax.fill_between(x, yc - CFG["b"], yc + CFG["b"], color=col, alpha=0.35, lw=0)
        ax.plot(x, yc, color=col, lw=0.8, ls="--")
        for xb in (CFG["buffer_len"], L - CFG["buffer_len"]):
            ax.axvline(xb, color="0.35", lw=0.7, ls=":")
        for xs in (CFG["straight_len"], L - CFG["straight_len"]):
            ax.axvline(xs, color="0.6", lw=0.6, ls="-")
        half = A + CFG["b"] + CFG["m_bank"] * (CFG["H_b"] - CFG["h_plain"]) + CFG["plain"]
        ax.set_ylim(-half, half)
        ax.set_ylabel("y [m]")
        ax.text(0.012, 0.88, rf"$\lambda$ = {lam:.0f} m, {L/lam:.0f} bends, "
                             rf"$A$ = {A:.0f} m = {A/(2*CFG['b']):.2f} $W$, "
                             rf"$\sigma$ = {rm.sinuosity(lam, CFG):.3f}",
                transform=ax.transAxes, fontsize=7.5, va="top")
    axes[0].text(CFG["straight_len"] / 2, axes[0].get_ylim()[1] * 0.55, "straight",
                 fontsize=6.5, ha="center", color="0.35")
    axes[0].text(L / 2, axes[0].get_ylim()[1] * 0.55, "interior (analysed)",
                 fontsize=6.5, ha="center", color="0.35")
    axes[-1].set_xlabel("down-valley $x$ [m]")
    axes[-1].set_xlim(0, L)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_section(path):
    """yOz: the cross-section, the still-water surface, and the freeboard."""
    b, mb = CFG["b"], CFG["m_bank"]
    n_out = b + mb * (CFG["H_b"] - CFG["h_plain"]) + CFG["plain"]
    n = np.linspace(-n_out, n_out, 4001)
    h = rm.section_depth(n, CFG)
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    ax.fill_between(n, -h, -4.0, color="#D3D1C7", alpha=0.7, lw=0)
    ax.plot(n, -h, color="0.25", lw=1.2)
    ax.fill_between(n, -h, 0.0, color="#5DCAA5", alpha=0.45, lw=0)
    ax.axhline(0.0, color="#0F6E56", lw=1.0)
    ax.annotate("", xy=(n_out * 0.88, 0), xytext=(n_out * 0.88, -CFG["h_plain"]),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#993C1D"))
    ax.text(n_out * 0.86, -CFG["h_plain"] - 0.30,
            f"always-wet shelf, $h$ = {CFG['h_plain']:.2f} m", ha="right", fontsize=7,
            color="#993C1D")
    ax.text(n_out * 0.99, 0.12, "still-water datum $\\eta = 0$", ha="right",
            fontsize=7, color="#0F6E56")
    ax.annotate("", xy=(0, 0), xytext=(0, -CFG["H_c"]),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.2"))
    ax.text(2.0, -CFG["H_c"] / 2, f"$H_c$ = {CFG['H_c']:.1f} m", fontsize=7)
    ax.annotate("", xy=(b, 0), xytext=(b, -CFG["H_b"]),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.2"))
    ax.text(b + 2.0, -CFG["H_b"] / 2, f"$H_b$ = {CFG['H_b']:.1f} m", fontsize=7)
    ax.annotate("", xy=(-b, -3.6), xytext=(b, -3.6),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.2"))
    ax.text(0, -3.45, f"$W = 2b$ = {2*b:.0f} m", ha="center", fontsize=7)
    ax.text(b + mb * CFG["H_b"] * 0.5, -0.55, f"1:{mb:.0f}", fontsize=7, color="0.3")
    ax.set_xlim(-n_out, n_out)
    ax.set_ylim(-4.0, 0.9)
    ax.set_xlabel("cross-channel $n$ [m]")
    ax.set_ylabel("$z$ [m]")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_plan(os.path.join(HERE, "fig_plan.pdf"))
    fig_section(os.path.join(HERE, "fig_section.pdf"))
    print("wrote fig_plan.pdf, fig_section.pdf")
