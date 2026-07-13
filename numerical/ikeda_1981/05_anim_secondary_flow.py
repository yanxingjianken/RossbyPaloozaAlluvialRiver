#!/usr/bin/env python3
"""Animation 3/3 -- secondary (helical) flow and the closure parameter A.

The St Venant equations cannot themselves produce the cross-stream circulation
in a bend; Ikeda et al. borrow the three-dimensional closure (their Eq. 6)

    eta'/H = -A C' n,          A ~ 2.89 (alluvial),

i.e. the transverse bed slope (and the shifted high-velocity core) is
proportional to the local centreline curvature C, with strength set by A.

This animation sweeps a marker along a meander (top).  In the cross-section
(bottom) the secondary circulation, the super-elevated water surface, the tilted
bed, and the outer-bank high-velocity core all respond to the local curvature --
and REVERSE as the channel passes through an inflection into the next bend.

Output: figures/secondary_flow.mp4 (+ _preview.png)

Usage
-----
    micromamba run -n fourcastnetv2 python 05_anim_secondary_flow.py
    micromamba run -n fourcastnetv2 python 05_anim_secondary_flow.py --max-frames 1
"""
from __future__ import annotations

import argparse

import numpy as np

import ikeda_lib as L
from matplotlib.patches import FancyArrowPatch

plt = L.set_style()

P = L.PARAMS
NWAVE = 1.6
SPLAN = np.linspace(0, NWAVE * 2 * np.pi, 600)      # planform phase coordinate
NN = np.linspace(-1, 1, 220)                         # transverse coordinate


def cross_section(ax, C):
    """Draw the bend cross-section for local (normalised) curvature C in [-1,1]."""
    # bed deeper toward the outer bank (sign of C); super-elevated surface there
    bed = -1.0 - 0.28 * C * NN
    surf = 0.07 * C * NN
    depth = surf - bed

    # downstream velocity (into page): core shifted toward the outer bank
    n_core = 0.6 * np.tanh(1.6 * C)
    Z = np.linspace(-1.35, 0.25, 200)
    NG, ZG = np.meshgrid(NN, Z)
    bedG = -1.0 - 0.28 * C * NG
    surfG = 0.07 * C * NG
    frac = np.clip((ZG - bedG) / (surfG - bedG + 1e-9), 0, 1)
    speed = (0.35 + 0.65 * frac) * np.exp(-((NG - n_core) / 0.55) ** 2)
    speed = np.where((ZG >= bedG) & (ZG <= surfG), speed, np.nan)
    pcm = ax.pcolormesh(NG, ZG, speed, cmap="YlOrRd", vmin=0, vmax=1.05,
                        shading="auto", zorder=0)

    # secondary circulation (one rotating cell), strength & sense follow C
    yq = np.linspace(-0.82, 0.82, 11)
    zq = np.linspace(-1.05, -0.08, 7)
    YQ, ZQ = np.meshgrid(yq, zq)
    bq = -1.0 - 0.28 * C * YQ
    sq = 0.07 * C * YQ
    inside = (ZQ >= bq) & (ZQ <= sq)
    zeta = np.clip((ZQ - bq) / (sq - bq + 1e-9) * 2 - 1, -1, 1)   # -1 bed .. +1 surf
    U = 0.5 * C * zeta                       # transverse: toward outer near surface
    W = -0.32 * C * YQ                       # vertical: down at outer, up at inner
    ax.quiver(YQ[inside], ZQ[inside], U[inside], W[inside],
              color=L.COLORS["channel"], scale=7.5, width=0.005, zorder=3)

    ax.plot(NN, bed, color=L.COLORS["bank"], lw=2.4, zorder=4)
    ax.plot(NN, surf, color=L.COLORS["water"], lw=2.4, zorder=4)
    ax.plot([-1, -1], [bed[0], surf[0]], color=L.COLORS["bank"], lw=2)
    ax.plot([1, 1], [bed[-1], surf[-1]], color=L.COLORS["bank"], lw=2)

    # mark the outer bank (whichever side, per sign of C)
    if abs(C) > 0.08:
        side = np.sign(C)
        ax.add_patch(FancyArrowPatch((side * 1.02, 0.16), (side * 1.02, -0.15),
                     arrowstyle="-|>", mutation_scale=13,
                     color=L.COLORS["erosion"], lw=2.2, zorder=6))
        ax.annotate("outer bank\n(erosion)", (side * 0.9, 0.34), ha="center",
                    fontsize=9, color=L.COLORS["erosion"])
        ax.annotate("inner bank\n(point bar)", (-side * 0.9, -1.28), ha="center",
                    fontsize=9, color=L.COLORS["bank"])
    else:
        ax.annotate("inflection: nearly straight\n(secondary flow vanishes)",
                    (0, 0.30), ha="center", fontsize=9.5, color="0.4")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.45, 0.55)
    ax.set_xlabel(r"transverse  $\tilde n$")
    ax.set_ylabel("depth")
    ax.grid(False)
    return pcm


def render(s, add_cbar=False):
    C = np.cos(s)                                    # local (normalised) curvature
    fig = plt.figure(figsize=(8.8, 6.4), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.55, 1.0], hspace=0.42)

    # -- top: planform with a moving station marker --
    axp = fig.add_subplot(gs[0])
    amp = 1.0
    yplan = amp * np.cos(SPLAN)
    axp.plot(SPLAN, yplan, color=L.COLORS["channel"], lw=2.5)
    axp.fill_between(SPLAN, yplan - 0.28, yplan + 0.28,
                     color=L.COLORS["water_fill"], zorder=0)
    axp.plot(s, np.cos(s), "o", color=L.COLORS["erosion"], ms=11, zorder=5)
    axp.axvline(s, color=L.COLORS["erosion"], lw=1.0, ls=":")
    axp.annotate("cross-section here", (s, 1.5), ha="center", fontsize=9,
                 color=L.COLORS["erosion"])
    axp.set_xlim(SPLAN[0], SPLAN[-1])
    axp.set_ylim(-1.9, 2.0)
    axp.set_yticks([])
    axp.set_xlabel(r"downstream distance  $s$  (plan view)")
    axp.grid(False)
    axp.set_title(r"Secondary flow tracks curvature:  $\eta'/H=-A\,\mathcal{C}'\tilde n$"
                  f"   (A = {P.A})")

    # -- bottom: cross-section --
    axc = fig.add_subplot(gs[1])
    pcm = cross_section(axc, C)
    if add_cbar:
        cb = fig.colorbar(pcm, ax=axc, pad=0.02, fraction=0.045)
        cb.set_label("downstream velocity (into page)")

    rgb = L.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    n_full = 150
    ss = np.linspace(0, NWAVE * 2 * np.pi, n_full)
    if args.max_frames:
        ss = ss[: args.max_frames]

    print(f"05_anim_secondary_flow.py  (A={P.A}, {len(ss)} frames)")
    frames = [render(s, add_cbar=True) for s in ss]
    L.write_mp4(frames, "secondary_flow", fps=args.fps)


if __name__ == "__main__":
    main()
