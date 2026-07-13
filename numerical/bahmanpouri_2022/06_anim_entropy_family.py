#!/usr/bin/env python3
"""Animation -- why calibrating M matters.

M sweeps through 0.6 -> 6.0 (bracketing the observed 1.16-3.24).  Left: the
Eq.-1 vertical profile morphs from starved to full while the (M, Phi) point
slides along the entropy backbone.  Right: the Sajó CS1 discharge estimated
from the SAME single surface velocity as a function of M, against the
measured 11.21 m3/s -- the sensitivity that ADCP calibration removes.

Output: figures/entropy_family.mp4 (+ _preview.png)

Usage
-----
    micromamba run -n fourcastnetv2 python 06_anim_entropy_family.py
    micromamba run -n fourcastnetv2 python 06_anim_entropy_family.py --max-frames 1
"""
from __future__ import annotations

import argparse

import numpy as np

import bahmanpouri_lib as L

plt = L.set_style()

X, DEPTH = L.load_bathymetry("sajo_cs1")
P = L.SAJO_CS1
MS = np.linspace(0.6, 6.0, 90)
QS = np.array([L.discharge(X, DEPTH, m, P["Usurf_max"], P["x_peak"],
                           "parabolic")["Q"] for m in MS])


def render(M_now, Q_now):
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.8),
                            gridspec_kw={"width_ratios": [1.0, 1.1, 1.25]})

    # profile
    yy = np.linspace(0, 1, 250)
    axs[0].plot(L.u_vertical(yy, 1.0, 1.0, 0.0, M_now), 1 - yy,
                color=L.COLORS["entropy"], lw=2.6)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0, 1.05)
    axs[0].set_xlabel(r"$U/U_{maxv}$")
    axs[0].set_ylabel("depth fraction")
    axs[0].set_title(rf"Eq. 1 profile,  $M={M_now:.2f}$")

    # backbone
    Mg = np.linspace(0.05, 7, 300)
    axs[1].plot(Mg, L.phi_of_M(Mg), color=L.COLORS["entropy"], lw=2.2)
    axs[1].plot(M_now, float(L.phi_of_M(M_now)), "o", color=L.COLORS["adcp"],
                ms=11, mec="white")
    for c in L.load_table2():
        axs[1].plot(c.M, float(L.phi_of_M(c.M)), ".", color="#999999", ms=7)
    axs[1].set_xlabel("$M$")
    axs[1].set_ylabel(r"$\Phi(M)$")
    axs[1].set_ylim(0.5, 0.85)
    axs[1].set_title("entropy backbone (dots: the 7 transects)")

    # discharge sensitivity
    axs[2].plot(MS, QS, color=L.COLORS["water"], lw=2.2)
    axs[2].axhline(P["Q"], color="#555555", lw=1.4, ls="--")
    axs[2].text(0.7, P["Q"] * 1.01, "measured 11.21 m³/s", fontsize=9,
                color="#555555")
    axs[2].axvspan(1.16, 3.24, color=L.COLORS["band"], alpha=0.5)
    axs[2].text(2.2, QS.min() * 1.01, "observed\n$M$ range", ha="center",
                fontsize=9, color=L.COLORS["water"])
    axs[2].plot(M_now, Q_now, "o", color=L.COLORS["adcp"], ms=11, mec="white")
    axs[2].set_xlabel("$M$")
    axs[2].set_ylabel("Q (m³/s)")
    axs[2].set_title(rf"Sajó CS1 from one $U_{{surf}}$:  $Q={Q_now:.2f}$ m³/s")

    fig.tight_layout()
    rgb = L.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    idx = np.arange(len(MS))
    idx = np.concatenate([idx, idx[::-1]])
    if args.max_frames:
        idx = idx[: args.max_frames]
    print(f"06_anim_entropy_family.py  ({len(idx)} frames)")
    frames = [render(MS[i], QS[i]) for i in idx]
    L.write_mp4(frames, "entropy_family", fps=args.fps)


if __name__ == "__main__":
    main()
