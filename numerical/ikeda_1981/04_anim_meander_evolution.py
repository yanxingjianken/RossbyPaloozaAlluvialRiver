#!/usr/bin/env python3
"""Animation 2/3 -- meander growth & wavelength selection.

A straight-ish channel is seeded with a deterministic superposition of several
wavenumbers (an analytic initial condition for the linear PDE, NOT fabricated
data).  Each mode evolves by its own linear normal mode y = e^{alpha0(k) t}
cos(k x - omega0(k) t):

    * modes with k < k_c grow, modes with k > k_c decay;
    * the fastest-growing mode near k_OM overtakes the others.

So an initially irregular set of wiggles organises itself into a regular train
of bends at the selected wavelength lambda_OM, marching downstream.

    top    : the evolving planform (amplitude rescaled each frame to stay in
             view; the true growth factor is printed and shown below).
    bottom : the growth-rate curve alpha0(k) with each seeded mode as a dot
             whose size tracks its current amplitude -- selection made visible.

Output: figures/meander_evolution.mp4 (+ _preview.png)

Usage
-----
    micromamba run -n fourcastnetv2 python 04_anim_meander_evolution.py
    micromamba run -n fourcastnetv2 python 04_anim_meander_evolution.py --max-frames 1
"""
from __future__ import annotations

import argparse

import numpy as np

import ikeda_lib as L
from matplotlib.patches import Polygon, FancyArrowPatch

plt = L.set_style()

P = L.PARAMS
KOM = L.k_OM(P.Cf, P.A, P.F)
LAM = 2.0 * np.pi / KOM
A_OM = L.alpha_OM(P.Cf, P.A, P.F)

# Seeded modes (in rescaled kappa = k/C_f): straddle the peak and the cutoff.
KAPPAS = np.array([0.6, 1.0, 1.5, 2.0, 2.44, 2.9])
KS = KAPPAS * P.Cf
KLAB = [rf"$\kappa={kk:.1f}$" for kk in KAPPAS]
# Deterministic low-discrepancy phase offsets so the seed is spatially uniform
# (a modelling choice for the analytic initial condition -- not fabricated data).
PHASES = (2.0 * np.pi * ((np.arange(len(KS)) + 1) * 0.6180339887 % 1.0))
OMS = np.array([L.frequency(k, P.Cf, P.A, P.F) for k in KS])
ALPHAS = np.array([L.growth_rate(k, P.Cf, P.A, P.F) for k in KS])

X = np.linspace(0.0, 3.0 * LAM, 1600)
AMP_VIS = 0.13 * LAM
B_VIS = 0.040 * LAM
T_MAX = np.log(9.0) / A_OM                 # run until the dominant mode grows ~9x


def _banks(yc, dy):
    norm = np.hypot(1.0, dy)
    nx, ny = -dy / norm, 1.0 / norm
    return (X + B_VIS * nx, yc + B_VIS * ny), (X - B_VIS * nx, yc - B_VIS * ny)


# scaled growth curve for the bottom panel
_kap = np.linspace(0, 3.3, 600)
_acurve = L.growth_rate(_kap, 1.0, P.A, P.F)          # alpha/C_f^2
_amax = _acurve.max()


def render(t):
    # centreline from the multimode superposition (phased, deterministic)
    amps = np.exp(ALPHAS * t)
    yc_raw = np.zeros_like(X)
    for k, w, ph, a in zip(KS, OMS, PHASES, amps):
        yc_raw = yc_raw + a * np.cos(k * X - w * t + ph)
    dx = X[1] - X[0]
    dy_raw = np.gradient(yc_raw, dx)
    scale = AMP_VIS / np.max(np.abs(yc_raw))
    yc, dy = yc_raw * scale, dy_raw * scale
    left, right = _banks(yc, dy)

    # modal amplitudes now (relative), and the true growth factor of the peak mode
    rel = amps / amps.max()
    growth_factor = np.exp(A_OM * t)

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.4), dpi=100,
                             gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.30))

    # -- top: evolving planform --
    ax = axes[0]
    ax.add_patch(Polygon(np.column_stack([np.r_[left[0], right[0][::-1]],
                                          np.r_[left[1], right[1][::-1]]]),
                         closed=True, facecolor=L.COLORS["water_fill"], edgecolor="none"))
    ax.plot(left[0], left[1], color=L.COLORS["bank"], lw=1.5)
    ax.plot(right[0], right[1], color=L.COLORS["bank"], lw=1.5)
    ax.plot(X, yc, "--", color=L.COLORS["channel"], lw=1.0)
    ax.add_patch(FancyArrowPatch((0.06 * X[-1], -1.9 * AMP_VIS),
                                 (0.24 * X[-1], -1.9 * AMP_VIS), arrowstyle="-|>",
                                 mutation_scale=15, color=L.COLORS["water"], lw=2.2))
    ax.annotate("downstream migration", (0.15 * X[-1], -2.5 * AMP_VIS), ha="center",
                fontsize=9, color=L.COLORS["water"])
    ax.annotate(rf"true amplitude $\times{growth_factor:.0f}$"
                "   (rescaled to fit)", (0.98 * X[-1], 1.9 * AMP_VIS), ha="right",
                fontsize=10, color=L.COLORS["erosion"])
    ax.set_ylim(-2.9 * AMP_VIS, 2.4 * AMP_VIS)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Wavelength selection: an irregular seed organises into a regular "
                 "train at $\\lambda_{OM}$")

    # -- bottom: growth curve + seeded modes sized by current amplitude --
    ax = axes[1]
    ax.plot(_kap, _acurve, color=L.COLORS["growth"], lw=2.2)
    ax.fill_between(_kap, 0, _acurve, where=_acurve > 0,
                    color=L.COLORS["growth"], alpha=0.10)
    ax.axhline(0, color="k", lw=0.6)
    for kap, r, lab in zip(KAPPAS, rel, KLAB):
        a = L.growth_rate(kap, 1.0, P.A, P.F)
        col = L.COLORS["erosion"] if abs(kap - 1.5) < 0.01 else L.COLORS["apex"]
        ax.scatter(kap, a, s=40 + 620 * r**1.3, color=col,
                   alpha=0.85, zorder=5, edgecolor="white", linewidth=0.8)
    ax.axvline(1.5, color=L.COLORS["erosion"], ls=":", lw=1.0)
    ax.annotate(r"$k_{OM}$ dominates", (1.5, _amax * 1.02), ha="center",
                color=L.COLORS["erosion"], fontsize=10)
    ax.set_xlim(0, 3.3)
    ax.set_ylim(-0.4 * _amax, 1.25 * _amax)
    ax.set_xlabel(r"rescaled wavenumber  $\kappa = k/C_f$   (dot size $\propto$ amplitude)")
    ax.set_ylabel(r"$\alpha_0/C_f^2$")

    rgb = L.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    n_full = 150
    ts = np.linspace(0.0, T_MAX, n_full)
    if args.max_frames:
        ts = ts[: args.max_frames]

    print(f"04_anim_meander_evolution.py  (k_OM={KOM:.4f}, T_max={T_MAX:.0f}, "
          f"{len(ts)} frames)")
    frames = [render(t) for t in ts]
    L.write_mp4(frames, "meander_evolution", fps=args.fps)


if __name__ == "__main__":
    main()
