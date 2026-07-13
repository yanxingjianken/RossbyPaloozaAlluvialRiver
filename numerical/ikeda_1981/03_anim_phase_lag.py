#!/usr/bin/env python3
"""Animation 1/3 -- the phase-lag mechanism (why meanders migrate downstream).

A travelling meander at the most-unstable wavenumber k_OM.  Amplitude is held
essentially fixed here so the eye can focus on the mechanism (growth is the
subject of animation 04):

    top    : the channel planform sliding downstream past a fixed reference,
             with the outer-bank erosion maximum a phase-lag downstream of the
             tracked apex.
    bottom : centreline curvature C(s) and near-bank velocity u_b(s) travelling
             together, separated by the fixed ~64 deg (~0.18 lambda) phase lag.

Because the erosion maximum sits on the downstream flank of every bend, the
whole train marches downstream -- that is the celerity c0 = omega0/k.

Output: figures/phase_lag.mp4 (+ figures/phase_lag_preview.png)

Usage
-----
    micromamba run -n fourcastnetv2 python 03_anim_phase_lag.py
    micromamba run -n fourcastnetv2 python 03_anim_phase_lag.py --max-frames 1
"""
from __future__ import annotations

import argparse

import numpy as np

import ikeda_lib as L
from matplotlib.patches import FancyArrowPatch, Polygon

plt = L.set_style()

P = L.PARAMS
K = L.k_OM(P.Cf, P.A, P.F)
LAM = 2.0 * np.pi / K
LAG = float(L.phase_lag_deg(K, P.Cf, P.A, P.F))
LAG_FRAC = LAG / 360.0

NWAVE = 2.6                       # window width in wavelengths
X = np.linspace(0.0, NWAVE * LAM, 1400)
AMP_PF = 0.13 * LAM               # planform amplitude (visual)
B_PF = 0.042 * LAM                # planform half-width (visual)
XREF = 0.55 * LAM                 # a fixed reference station


def _banks(yc, dy):
    norm = np.hypot(1.0, dy)
    nx, ny = -dy / norm, 1.0 / norm
    return (X + B_PF * nx, yc + B_PF * ny), (X - B_PF * nx, yc - B_PF * ny)


def render(phase):
    """Draw one frame at travelling-wave phase `phase` (radians); return RGB."""
    shift = phase / K                                    # = c t (spatial offset)
    yc = AMP_PF * np.cos(K * X - phase)
    dy = -AMP_PF * K * np.sin(K * X - phase)
    left, right = _banks(yc, dy)

    curv = L.curvature_of_sine(X - shift, K, eps=1.0)
    ub = L.near_bank_velocity(X - shift, K, P.Cf, P.A, P.F, bstar=10.0, eps=1.0)
    curv_n = curv / (K**2)                               # -> cos(k x - phase), in [-1, 1]
    ub_n = ub / np.max(np.abs(
        L.near_bank_velocity(X, K, P.Cf, P.A, P.F, bstar=10.0, eps=1.0)))

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), dpi=100, sharex=True,
                             gridspec_kw=dict(height_ratios=[1.05, 1.15], hspace=0.14))

    # -- top: planform --
    ax = axes[0]
    ax.add_patch(Polygon(np.column_stack([np.r_[left[0], right[0][::-1]],
                                          np.r_[left[1], right[1][::-1]]]),
                         closed=True, facecolor=L.COLORS["water_fill"], edgecolor="none"))
    ax.plot(left[0], left[1], color=L.COLORS["bank"], lw=1.6)
    ax.plot(right[0], right[1], color=L.COLORS["bank"], lw=1.6)
    ax.plot(X, yc, "--", color=L.COLORS["channel"], lw=1.1)

    # fixed reference station (pattern slides past it)
    ax.axvline(XREF, color="0.5", lw=1.0, ls=":")
    ax.annotate("fixed station", (XREF, AMP_PF + 1.9 * B_PF), ha="center",
                fontsize=9, color="0.4")

    # tracked apex nearest window centre + its downstream erosion maximum
    xc = 0.5 * NWAVE * LAM
    m = round((xc - shift) / LAM)
    x_apex = shift + m * LAM
    if 0 < x_apex < X[-1]:
        ax.plot(x_apex, AMP_PF, "o", color=L.COLORS["apex"], ms=7, zorder=6)
        xe = x_apex + LAG_FRAC * LAM
        if xe < X[-1]:
            yb = AMP_PF + B_PF
            ax.add_patch(FancyArrowPatch((xe, yb), (xe, yb + 0.7 * B_PF),
                         arrowstyle="-|>", mutation_scale=14,
                         color=L.COLORS["erosion"], lw=2.2, zorder=7))
            ax.annotate("erosion max", (xe, yb + 1.5 * B_PF), ha="center",
                        fontsize=8.5, color=L.COLORS["erosion"])

    ax.add_patch(FancyArrowPatch((0.08 * X[-1], -AMP_PF - 2.0 * B_PF),
                                 (0.30 * X[-1], -AMP_PF - 2.0 * B_PF),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=L.COLORS["water"], lw=2.4))
    ax.annotate("downstream migration", (0.19 * X[-1], -AMP_PF - 3.1 * B_PF),
                ha="center", fontsize=9.5, color=L.COLORS["water"])
    ax.set_ylim(-AMP_PF - 3.7 * B_PF, AMP_PF + 3.0 * B_PF)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Phase-lag mechanism at $k_{OM}$: erosion peaks downstream of the apex "
                 "$\\Rightarrow$ the bend train migrates")

    # -- bottom: curvature vs near-bank velocity --
    ax = axes[1]
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(X, curv_n, color=L.COLORS["curvature"], lw=2.2,
            label=r"curvature  $\mathcal{C}(s)$")
    ax.plot(X, ub_n, color=L.COLORS["velocity"], lw=2.2,
            label=r"near-bank velocity  $u_b(s)$")
    if 0 < x_apex < X[-1]:
        xe = x_apex + LAG_FRAC * LAM
        ax.plot([x_apex, x_apex], [0, 1], color=L.COLORS["curvature"], lw=0.8, ls=":")
        ax.plot([xe, xe], [0, 1], color=L.COLORS["velocity"], lw=0.8, ls=":")
        ax.annotate("", (xe, 1.12), (x_apex, 1.12),
                    arrowprops=dict(arrowstyle="<->", color=L.COLORS["erosion"], lw=1.5))
        ax.annotate(rf"lag $\approx{LAG:.0f}^\circ$", ((x_apex + xe) / 2, 1.22),
                    ha="center", color=L.COLORS["erosion"], fontsize=10)
    ax.set_ylim(-1.3, 1.45)
    ax.set_ylabel("normalised")
    ax.set_xlabel(r"downstream distance  $s$")
    ax.legend(loc="lower right", ncol=2, fontsize=10)

    rgb = L.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None,
                    help="render only this many frames (smoke test)")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    n_full = 150                                  # migrate ~1.5 wavelengths
    phases = np.linspace(0, 1.5 * 2 * np.pi, n_full)
    if args.max_frames:
        phases = phases[: args.max_frames]

    print(f"03_anim_phase_lag.py  (k_OM={K:.4f}, lag={LAG:.1f} deg, {len(phases)} frames)")
    frames = [render(p) for p in phases]
    L.write_mp4(frames, "phase_lag", fps=args.fps)


if __name__ == "__main__":
    main()
