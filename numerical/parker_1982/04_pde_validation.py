#!/usr/bin/env python3
"""The full nonlinear bend PDE (Eq. 7) vs the weakly nonlinear theory.

Produces figures/fig08..fig10 and pde_bend.mp4:

    fig08  harmonic amplitudes |c1|, |c3| from the PDE (markers) on the
           Eq.-35 predictions (lines); the even harmonic sits at the
           numerical floor (mu2 = 0 by symmetry)
    fig09  measured growth rate and third-harmonic |J| vs delta0M,
           against alpha(0) (Eq. 26a) and |J_M| (Eqs. 34a/b)
    fig10  planform snapshots: PDE (solid) vs Eq. 35 (dashed) at
           t = 0 and one e-folding -- fattening/skewing emerge in the
           direct integration
    pde_bend.mp4  the PDE planform evolving with the Eq.-35 overlay

Run at C_f = 0.1 (the asserted quantities are C_f-free; a larger C_f
compresses the domain and the run to seconds -- see README).

Usage
-----
    micromamba run -n fourcastnetv2 python 04_pde_validation.py
    micromamba run -n fourcastnetv2 python 04_pde_validation.py --max-frames 1
"""
from __future__ import annotations

import argparse

import numpy as np

import parker_lib as L

plt = L.set_style()

P = L.Params(Cf=0.1, A=2.89, F=0.3, e=1.0)
D0 = 0.10
EPS = D0 / L.k0M(P)
KM = L.kM_over_k0M(D0, P) * L.k0M(P)
LDOM = 2 * np.pi / KM
N = 128
X = LDOM * np.arange(N) / N
AM = L.alpha_kOM(D0, P)
WM = L.omega_kM(D0, P)
JF, JS = L.JF_M(P), L.JS_M(P)


def run(t_out):
    y0 = L.planform_eq35(X, 0.0, EPS, P)
    return L.evolve_bend_pde(y0, LDOM, t_out, P)[1]


def fig08_harmonics():
    ts = np.linspace(0.0, 1.0 / AM, 9)
    Y = run(ts)
    a1 = np.array([abs(L.harmonics(Y[i], X, KM, 4)[0]) for i in range(len(ts))])
    a2 = np.array([abs(L.harmonics(Y[i], X, KM, 4)[1]) for i in range(len(ts))])
    a3 = np.array([abs(L.harmonics(Y[i], X, KM, 4)[2]) for i in range(len(ts))])

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    tt = ts * AM
    ax.semilogy(tt, a1, "o", color=L.COLORS["channel"], ms=7, label="PDE $|c_1|$")
    ax.semilogy(tt, EPS * np.exp(AM * ts), "-", color=L.COLORS["channel"],
                lw=1.6, label=r"Eq. 35: $\epsilon e^{\alpha_M t}$")
    ax.semilogy(tt, a3, "s", color=L.COLORS["fatten"], ms=7, label="PDE $|c_3|$")
    ax.semilogy(tt, EPS * D0**2 * np.hypot(JF, JS) * np.exp(3 * AM * ts), "-",
                color=L.COLORS["fatten"], lw=1.6,
                label=r"Eq. 35: $\epsilon\,\delta_{0M}^2|J|\,e^{3\alpha_M t}$")
    ax.semilogy(tt, a2, "x", color=L.COLORS["linear"], ms=7,
                label="PDE $|c_2|$ (must vanish: $\\mu_2=0$)")
    ax.set_xlabel(r"$\alpha_M t$")
    ax.set_ylabel("harmonic amplitude")
    ax.set_title("Direct Eq.-7 integration tracks the composite solution "
                 "mode by mode")
    ax.legend(fontsize=9)
    L.save_fig(fig, "fig08_harmonics")


def fig09_delta_sweep():
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8))
    d0s = np.array([0.05, 0.10, 0.15, 0.20])
    gr, jj = [], []
    for d0 in d0s:
        eps = d0 / L.k0M(P)
        km = L.kM_over_k0M(d0, P) * L.k0M(P)
        Ld = 2 * np.pi / km
        x = Ld * np.arange(N) / N
        y0 = L.planform_eq35(x, 0.0, eps, P)
        aM = L.alpha_kOM(d0, P)
        T = 0.5 / aM
        Y = L.evolve_bend_pde(y0, Ld, [0.0, T], P)[1]
        c0 = L.harmonics(Y[0], x, km, 4)
        c1 = L.harmonics(Y[1], x, km, 4)
        gr.append(np.log(abs(c1[0]) / abs(c0[0])) / T)
        d_t = d0 * abs(c1[0]) / abs(c0[0])
        jj.append(abs(c1[2]) / (abs(c1[0]) * d_t**2))
    axs[0].plot(d0s, gr, "o", ms=8, color=L.COLORS["pde"], label="PDE")
    dd = np.linspace(0, 0.22, 100)
    axs[0].plot(dd, [L.alpha_kOM(d, P) for d in dd], "-",
                color=L.COLORS["nonlinear"], lw=2, label="Eq. (26a)")
    axs[0].axhline(L.alpha0M(P), color="#cccccc", lw=1.0)
    axs[0].text(0.005, L.alpha0M(P) * 1.001, "linear $\\alpha_{0M}$",
                fontsize=9, color="#888888")
    axs[0].set_xlabel(r"$\delta_{0M}$")
    axs[0].set_ylabel(r"growth rate at $t=0$")
    axs[0].legend(fontsize=10)
    axs[1].plot(d0s, jj, "s", ms=8, color=L.COLORS["pde"], label="PDE $|c_3|/(|c_1|\\delta^2)$")
    axs[1].axhline(np.hypot(JF, JS), color=L.COLORS["fatten"], lw=2,
                   label=r"$|J_M|$ from Eqs. (34a,b)")
    axs[1].set_xlabel(r"$\delta_{0M}$")
    axs[1].set_ylabel(r"third-harmonic coefficient")
    axs[1].legend(fontsize=10)
    fig.suptitle("Weakly nonlinear predictions hold in the direct integration",
                 y=1.02)
    L.save_fig(fig, "fig09_delta_sweep")


def fig10_planforms(Y=None, ts=None):
    ts = [0.0, 1.0 / AM]
    Y = run(ts)
    fig, axs = plt.subplots(2, 1, figsize=(10.6, 6.2), sharex=True)
    for ax, i, t in zip(axs, (0, 1), ts):
        ax.plot(X / LDOM, Y[i], "-", color=L.COLORS["pde"], lw=2.4,
                label="PDE (Eq. 7)")
        ax.plot(X / LDOM, L.planform_eq35(X, t, EPS, P), "--",
                color=L.COLORS["erosion"], lw=1.8, label="Eq. (35)")
        ax.set_title(rf"$\alpha_M t = {t*AM:.1f}$", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
    axs[1].set_xlabel(r"$x/\lambda_M$")
    fig.suptitle("Planform: direct integration vs composite solution", y=0.99)
    L.save_fig(fig, "fig10_planforms")


def render(t, y):
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    ax.plot(X / LDOM, y, "-", color=L.COLORS["pde"], lw=2.6, label="PDE (Eq. 7)")
    ax.plot(X / LDOM, L.planform_eq35(X, t, EPS, P), "--",
            color=L.COLORS["erosion"], lw=1.8, label="Eq. (35)")
    ax.set_ylim(-3.5 * EPS * np.exp(1.0), 3.5 * EPS * np.exp(1.0))
    ax.set_xlabel(r"$x/\lambda_M$")
    ax.set_title(rf"nonlinear bend PDE,  $\alpha_M t = {t*AM:.2f}$")
    ax.legend(fontsize=9, loc="upper right")
    rgb = L.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    fig08_harmonics()
    fig09_delta_sweep()
    fig10_planforms()

    n_full = 120
    ts = np.linspace(0.0, 1.0 / AM, n_full)
    if args.max_frames:
        ts = ts[: args.max_frames]
    print(f"04_pde_validation.py  ({len(ts)} frames)")
    Y = run(ts)
    frames = [render(t, y) for t, y in zip(ts, Y)]
    L.write_mp4(frames, "pde_bend", fps=args.fps)


if __name__ == "__main__":
    main()
