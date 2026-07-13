#!/usr/bin/env python3
"""Shape evolution of the composite solution (Eq. 35) -- paper Fig. 6.

Produces figures/fig07 and shape_evolution.mp4:

    fig07  the paper's Fig. 6 replica: y = e^t cos(k(x-ct))
           - e^{3t}{0.05 cos 3k(x-ct) + 0.05 sin 3k(x-ct)}, k = pi/3,
           at t = 0 (solid) and t = 0.45 (dashed)
    shape_evolution.mp4  the same relation swept continuously in t --
           fattening and skewing intensify as the bend grows and migrates

Usage
-----
    micromamba run -n fourcastnetv2 python 03_shape_evolution.py
    micromamba run -n fourcastnetv2 python 03_shape_evolution.py --max-frames 1
"""
from __future__ import annotations

import argparse

import numpy as np

import parker_lib as L

plt = L.set_style()

K1 = L.FIG6["k"]              # pi/3 (third mode = pi)
J = L.FIG6["J"]               # 0.05
C = 1.0 / 3.0                 # migration speed used for the visual sweep
X = np.linspace(0.0, 4 * 2 * np.pi / K1, 1200)


def fig6_curve(t):
    ph = K1 * (X - C * t)
    return np.exp(t) * np.cos(ph) - np.exp(3 * t) * (
        J * np.cos(3 * ph) + J * np.sin(3 * ph))


def fig07_fig6_replica():
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    ax.plot(X / (2 * np.pi / K1), fig6_curve(0.0), "-",
            color=L.COLORS["channel"], lw=2.4, label="$t=0$")
    ax.plot(X / (2 * np.pi / K1), fig6_curve(0.45), "--",
            color=L.COLORS["erosion"], lw=2.4, label="$t=0.45$")
    ax.set_xlabel(r"$x/\lambda$")
    ax.legend(fontsize=11)
    ax.set_title("Paper Fig. 6 replica (Eq. 35 with $J_F=J_S=0.05$): the bend "
                 "grows, migrates,\nfattens and skews -- with a point of least "
                 "apparent migration upstream of each apex")
    L.save_fig(fig, "fig07_fig6_replica")


def render(t):
    fig, ax = plt.subplots(figsize=(11.4, 4.2))
    ax.plot(X / (2 * np.pi / K1), fig6_curve(0.0), "-",
            color="#cccccc", lw=1.4, label="$t=0$")
    ax.plot(X / (2 * np.pi / K1), fig6_curve(t), "-",
            color=L.COLORS["erosion"], lw=2.6, label=f"$t={t:.2f}$")
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel(r"$x/\lambda$")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title("Eq. (35): fattening + skewing intensify as $e^{2t}$")
    rgb = L.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    fig07_fig6_replica()
    ts = np.linspace(0.0, 0.45, 120)
    if args.max_frames:
        ts = ts[: args.max_frames]
    print(f"03_shape_evolution.py ({len(ts)} frames)")
    frames = [render(t) for t in ts]
    L.write_mp4(frames, "shape_evolution", fps=args.fps)


if __name__ == "__main__":
    main()
