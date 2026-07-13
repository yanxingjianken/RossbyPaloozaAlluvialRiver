#!/usr/bin/env python3
"""Numerical PDE demo -- wavelength selection computed, not asserted.

Integrates the linearized bend equation (Eq. 16) itself,

    y_xt + 2 C_f y_t = y_xxx - C_f (A + F^2) y_xx,

pseudo-spectrally (periodic x, per-mode classical RK4; see the PDE block of
ikeda_lib.py) from a LOCALIZED disturbance: 12 equal-amplitude modes
cophased at one point -- i.e. a single smooth bump, the analytic stand-in
for a bank slump on a straight channel.  The band brackets the
most-unstable wavenumber k_OM (mode 8) and extends beyond the neutral
wavenumber k_c.  Eq. (16) grows the bump into a wavetrain that spreads
downstream while its spectrum collapses onto k_OM -- Sec. 5's wavelength
selection *produced by a numerical solution of the PDE* and checked
mode-by-mode against the closed-form normal modes (Eq. 17-18).

The initial condition is an analytic mode superposition (theory, not data);
no synthetic/random inputs are used anywhere.

Produces figures/fig11..fig13 and figures/pde_selection.mp4:

    fig11  Waterfall y(x, t): broad-band tangle -> periodic k_OM train
    fig12  Amplitude spectrum vs time: PDE (markers) on top of the
           closed-form prediction a0 exp(alpha0(k) t) (lines)
    fig13  RK4 convergence: max error vs dt with the dt^4 reference
    pde_selection.mp4  planform + live spectrum, side by side

Usage
-----
    micromamba run -n fourcastnetv2 python 06_pde_selection_demo.py
    micromamba run -n fourcastnetv2 python 06_pde_selection_demo.py --max-frames 1
"""
from __future__ import annotations

import argparse

import numpy as np

import ikeda_lib as L

plt = L.set_style()

P = L.PARAMS
KOM = L.k_OM(P.Cf, P.A, P.F)
KC = L.k_cutoff(P.Cf, P.A, P.F)
AOM = L.alpha_OM(P.Cf, P.A, P.F)

# Domain: mode j = 8 lands exactly on k_OM (8 selected wavelengths per box).
LDOM = 16.0 * np.pi / KOM
N = 256
MODES = list(range(3, 15))          # k_j/k_OM = j/8: 0.375 .. 1.75 (past k_c)
A0 = 1.0                            # equal seed amplitude per mode
T_END = 6.0 / AOM                   # six e-foldings of the winner

X = LDOM * np.arange(N) / N


def k_of(j):
    return 2.0 * np.pi * j / LDOM


XC = 0.35 * LDOM      # centre of the initial disturbance (room to spread right)


def seed():
    """Localized disturbance: equal-amplitude modes cophased at x = XC.

    Summing the 12 modes in phase at one point is exactly a truncated
    impulse -- a single smooth bump of width ~lambda_OM at XC, the analytic
    stand-in for a localized bank irregularity (slump, log jam) on an
    otherwise straight channel.  Eq. (16) then does the rest: the bump seeds
    a wavetrain whose spectrum collapses onto k_OM while the packet grows
    and spreads DOWNSTREAM -- the convective-instability picture behind
    Sec. 5's wavelength selection.  (Analytic IC; no RNG anywhere.)
    """
    return sum(A0 * np.cos(k_of(j) * (X - XC)) for j in MODES)


def run(t_out):
    """One RK4 integration of Eq. (16) through all requested times."""
    return L.evolve_linear_pde(seed(), LDOM, t_out, P.Cf, P.A, P.F)[1]


# --------------------------------------------------------------------------- #
#  fig11 -- waterfall: the selection story in physical space
# --------------------------------------------------------------------------- #
def fig11_waterfall():
    n_slices = 9
    times = np.linspace(0.0, T_END, n_slices)
    Y = run(times)

    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    cmap = plt.get_cmap("viridis")
    for i, (t, y) in enumerate(zip(times, Y)):
        ynorm = y / np.max(np.abs(y))
        off = n_slices - 1 - i
        col = cmap(i / (n_slices - 1))
        ax.plot(X / (2 * np.pi / KOM), 0.85 * ynorm + off, color=col, lw=1.7)
        ax.text(8.12, off, rf"$\alpha_{{OM}}t={t*AOM:.1f}$" "\n"
                rf"$\times{np.max(np.abs(y))/np.max(np.abs(Y[0])):.0f}$",
                va="center", fontsize=9, color=col)

    ax.axvspan(0, 1, color=L.COLORS["growth"], alpha=0.08)
    ax.text(0.5, n_slices - 0.25, r"one $\lambda_{OM}$", ha="center",
            fontsize=10, color=L.COLORS["growth"])
    ax.set_xlim(0, 8)
    ax.set_yticks([])
    ax.set_xlabel(r"downstream distance  $x/\lambda_{OM}$")
    ax.set_title("Eq. (16) integrated: a localized disturbance grows into a "
                 r"$k_{OM}$ wavetrain spreading downstream"
                 "\n(each trace renormalised; growth factor at right)")
    L.save_fig(fig, "fig11_pde_waterfall")


# --------------------------------------------------------------------------- #
#  fig12 -- spectrum vs time: PDE markers on closed-form prediction lines
# --------------------------------------------------------------------------- #
def fig12_spectrum_evolution():
    times = np.array([0.0, T_END / 3.0, 2.0 * T_END / 3.0, T_END])
    Y = run(times)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    cmap = plt.get_cmap("viridis")
    jj = np.array(MODES)
    kap = np.array([k_of(j) for j in jj]) / P.Cf     # kappa = k/C_f axis

    for i, (t, y) in enumerate(zip(times, Y)):
        col = cmap(i / (len(times) - 1))
        _, amp = L.spectrum(y, LDOM)
        pred = A0 * np.exp(L.growth_rate(np.array([k_of(j) for j in jj]),
                                         P.Cf, P.A, P.F) * t)
        ax.plot(kap, pred, "-", color=col, lw=1.6,
                label=rf"$\alpha_{{OM}}t={t*AOM:.0f}$")
        ax.plot(kap, amp[jj], "x", color=col, ms=7.5, mew=2.0)

    ax.axvline(KOM / P.Cf, color=L.COLORS["growth"], lw=1.2, ls="--")
    ax.axvline(KC / P.Cf, color=L.COLORS["decay"], lw=1.2, ls="--")
    ax.text(KOM / P.Cf, ax.get_ylim()[1], r" $\kappa_{OM}$",
            color=L.COLORS["growth"], va="top", fontsize=11)
    ax.text(KC / P.Cf, A0 * 3, r" $\kappa_c$", color=L.COLORS["decay"], fontsize=11)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa = k/C_f$")
    ax.set_ylabel("mode amplitude")
    ax.set_title("PDE spectrum (crosses) lands on the normal-mode prediction "
                 r"$a_0 e^{\alpha_0(k)t}$ (lines)")
    ax.legend(loc="center left", fontsize=10)
    L.save_fig(fig, "fig12_pde_spectrum")


# --------------------------------------------------------------------------- #
#  fig13 -- convergence: the integrator really is 4th order
# --------------------------------------------------------------------------- #
def fig13_convergence():
    # Small stable grid (see ikeda_lib._pde_self_test note): single k_OM mode.
    N32 = 32
    x32 = LDOM * np.arange(N32) / N32
    y1 = np.cos(k_of(8) * x32)
    smag = abs(complex(L.pde_symbol(KOM, P.Cf, P.A, P.F)))
    dt0 = 0.2 / smag
    T = 8.0 * dt0
    _, Ye = L.evolve_linear_pde_exact(y1, LDOM, [T], P.Cf, P.A, P.F)
    dts = dt0 / 2.0 ** np.arange(4)
    errs = np.array([np.max(np.abs(
        L.evolve_linear_pde(y1, LDOM, [T], P.Cf, P.A, P.F, dt=d)[1] - Ye))
        for d in dts])
    order = np.polyfit(np.log(dts), np.log(errs), 1)[0]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.loglog(dts * smag, errs, "o-", color=L.COLORS["channel"], lw=2,
              label=rf"measured (slope $\approx$ {order:.2f})")
    ref = errs[0] * (dts / dts[0]) ** 4
    ax.loglog(dts * smag, ref, "--", color=L.COLORS["decay"], lw=1.5,
              label=r"$\propto \Delta t^4$")
    ax.set_xlabel(r"$|s(k_{OM})|\,\Delta t$")
    ax.set_ylabel(r"max $|y_{RK4} - y_{exact}|$")
    ax.set_title("Classical RK4 on the diagonal Eq. (16): 4th-order convergence")
    ax.legend()
    L.save_fig(fig, "fig13_pde_convergence")
    print(f"  measured order = {order:.3f}")


# --------------------------------------------------------------------------- #
#  animation -- planform + live spectrum
# --------------------------------------------------------------------------- #
def render(t, y):
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.8, 4.6), gridspec_kw={"width_ratios": [1.55, 1.0]})

    ynorm = y / np.max(np.abs(y))
    axL.plot(X / (2 * np.pi / KOM), ynorm, color=L.COLORS["channel"], lw=2.0)
    axL.set_ylim(-1.25, 1.25)
    axL.set_xlim(0, 8)
    axL.set_xlabel(r"$x/\lambda_{OM}$")
    axL.set_yticks([])
    axL.set_title(rf"centreline (renormalised),  $\alpha_{{OM}}t = {t*AOM:.2f}$,"
                  rf"  max amplitude $\times{np.max(np.abs(y))/A0:.1f}\,a_0$")

    jj = np.array(MODES)
    kk = np.array([k_of(j) for j in jj])
    _, amp = L.spectrum(y, LDOM)
    pred = A0 * np.exp(L.growth_rate(kk, P.Cf, P.A, P.F) * t)
    kap = kk / P.Cf
    width = 0.85 * (kap[1] - kap[0])
    cols = [L.COLORS["growth"] if kv < KC else L.COLORS["decay"] for kv in kk]
    axR.bar(kap, amp[jj], width=width, color=cols, alpha=0.75)
    axR.plot(kap, pred, "o", color=L.COLORS["apex"], ms=4.5,
             label=r"normal-mode $a_0e^{\alpha_0 t}$")
    axR.axvline(KOM / P.Cf, color=L.COLORS["growth"], lw=1.2, ls="--")
    axR.set_yscale("log")
    axR.set_ylim(A0 * 1e-3, A0 * 1.5e3)
    axR.set_xlabel(r"$\kappa = k/C_f$")
    axR.set_title("mode amplitudes (bars: PDE)")
    axR.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    rgb = L.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    fig11_waterfall()
    fig12_spectrum_evolution()
    fig13_convergence()

    n_full = 180
    times = np.linspace(0.0, T_END, n_full)
    if args.max_frames:
        times = times[: args.max_frames]
    Y = run(times)
    print(f"06_pde_selection_demo.py  (k_OM={KOM:.4f}, {len(times)} frames)")
    frames = [render(t, y) for t, y in zip(times, Y)]
    L.write_mp4(frames, "pde_selection", fps=args.fps)


if __name__ == "__main__":
    main()
