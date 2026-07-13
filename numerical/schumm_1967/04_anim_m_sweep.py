#!/usr/bin/env python3
"""Animation -- the paper's thesis in motion: sediment moves the whole line.

The lambda-Q_m log-log plane holds the 36 real sections (class-coded).  A
single line, Eq. (1) at the current silt-clay value M, sweeps M through the
observed range 1.3 -> 44.6 percent.  Sections whose own M is close to the
current value light up: the line visits each channel-type cluster in turn.
One number -- sediment -- slides the wavelength law across a tenfold range.

Output: figures/m_sweep.mp4 (+ figures/m_sweep_preview.png)

Usage
-----
    micromamba run -n fourcastnetv2 python 04_anim_m_sweep.py
    micromamba run -n fourcastnetv2 python 04_anim_m_sweep.py --max-frames 1
"""
from __future__ import annotations

import argparse

import numpy as np

import schumm_lib as L

plt = L.set_style()

D = L.load_sections()
MARKERS = {"bedload": "s", "mixed": "o", "suspended": "^"}
QG = np.logspace(np.log10(15.0), np.log10(8000.0), 120)


def render(M_now):
    fig, ax = plt.subplots(figsize=(8.8, 6.4))

    # proximity in log-M: highlight sections near the sweeping value
    w = np.exp(-0.5 * ((np.log10(D["M"]) - np.log10(M_now)) / 0.12) ** 2)
    for c in ("bedload", "mixed", "suspended"):
        m = D["cls"] == c
        ax.loglog(D["Qm"][m], D["lam"][m], MARKERS[c], color=L.COLORS[c],
                  ms=7, mec="white", mew=0.5, alpha=0.35, zorder=3)
        for q, lam, wi in zip(D["Qm"][m], D["lam"][m], w[m]):
            if wi > 0.15:
                ax.loglog([q], [lam], MARKERS[c], color=L.COLORS[c],
                          ms=7 + 7 * wi, mec=L.COLORS["fit"],
                          mew=1.6 * wi, zorder=4)

    cls_now = L.classify(M_now)
    ax.loglog(QG, L.wavelength_qm(QG, M_now), "-", color=L.COLORS[str(cls_now)],
              lw=3.0, zorder=5)
    ax.text(0.03, 0.94,
            rf"$\lambda = 1890\,Q_m^{{0.34}}/M^{{0.74}}$" "\n"
            rf"$M = {M_now:.1f}\%$  ({cls_now})",
            transform=ax.transAxes, fontsize=13, va="top",
            color=L.COLORS[str(cls_now)])

    ax.set_xlim(15, 8000)
    ax.set_ylim(250, 4e4)
    ax.set_xlabel("mean annual discharge $Q_m$ (cfs)")
    ax.set_ylabel(r"meander wavelength $\lambda$ (ft)")
    ax.set_title("One equation, one sliding sediment index")
    rgb = L.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    n_full = 150
    # sweep M log-uniformly up and back: 1.3 -> 44.6 -> 1.3
    up = np.logspace(np.log10(1.3), np.log10(44.6), n_full // 2)
    Ms = np.concatenate([up, up[::-1]])
    if args.max_frames:
        Ms = Ms[: args.max_frames]

    print(f"04_anim_m_sweep.py  ({len(Ms)} frames)")
    frames = [render(M) for M in Ms]
    L.write_mp4(frames, "m_sweep", fps=args.fps)


if __name__ == "__main__":
    main()
