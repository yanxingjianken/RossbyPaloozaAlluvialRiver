#!/usr/bin/env python3
"""fig04-05: THE headline experiment -- initial bank planforms of many
wavelengths evolving together.

One channel of length Lx = 20*pi seeded with 15 sinusoidal bank waves
(k* = 0.1 ... 1.5, equal amplitudes, cophased). The linear dynamics evolves
every wavelength independently: the growth band k*^2 < 2D amplifies, the
short waves decay, and every surviving crest marches UPSTREAM. Per-mode
demodulation turns the single run into a measured dispersion relation,
checked against the d3 EVP.

fig04: bank-planform waterfall (growth + upstream drift + selection).
fig05: measured sigma(k), c(k) dots vs EVP line vs 2x2 closure, both closures.

Run: micromamba run -n dedalus env OMP_NUM_THREADS=1 python 03_multi_k_selection.py
"""
import numpy as np

from channel_lib import (COLORS, Params, VL, build_ivp, demodulate,
                         evp_bank_mode, fit_sigma_c, run_ivp, seed_banks,
                         t_filament, set_style, save_fig)

plt = set_style()

D, G = 0.6, 0.05
LX = 20 * np.pi                    # dk = 0.1
MODES = list(range(1, 16))         # k* = 0.1 ... 1.5
NX, NY, DT, T_END, A0 = 64, 384, 0.02, 150.0, 1e-4

measured = {}
runs = {}
for friction in ("rayleigh", "momentum"):
    print(f"=== {friction} ===")
    p = Params(D=D, gamma=G, friction=friction)
    built = build_ivp(p, Lx=LX, Nx=NX, Ny=NY)
    seed_banks(built, [(0.1 * m, A0, 0.0) for m in MODES])
    res = run_ivp(built, dt=DT, t_end=T_END, rec_dt=0.5)
    runs[friction] = res
    sin_ch = 0.5 * (res['top'] + res['bot'])
    rows = []
    for m in MODES:
        k = 0.1 * m
        om = evp_bank_mode(k, p, Ny=96)
        w1 = min(T_END, 0.9 * t_filament(k, D, NY))
        sig, c, r2 = fit_sigma_c(res['t'], demodulate(sin_ch, m), k,
                                 (50.0, w1))
        rows.append((k, sig, c, om))
        tol_s = max(0.02 * abs(om.imag), 5e-4)
        tol_c = max(0.02 * abs(om.real / k), 5e-3)
        flag = ""
        assert abs(sig - om.imag) < tol_s, \
            f"{friction} m={m}: sigma {sig:.5f} vs EVP {om.imag:.5f}"
        assert abs(c - om.real / k) < tol_c, \
            f"{friction} m={m}: c {c:.4f} vs EVP {om.real/k:.4f}"
        if om.imag < 0:
            assert sig < 0, f"m={m} should decay"
            flag = " (decays ok)"
        print(f"  k*={k:.1f}: sigma {sig:+.5f} (EVP {om.imag:+.5f})  "
              f"c {c:+.4f} (EVP {om.real/k:+.4f})  R2={r2:.5f}{flag}")
    measured[friction] = rows

# ------------------------------------------------------------- fig04 ---- #
res = runs["rayleigh"]
fig, ax = plt.subplots(figsize=(10.5, 6.2))
times = (0.0, 50.0, 100.0, 150.0)
x2b = res['x'] / 2.0
for jt, tt in enumerate(times):
    i = int(np.argmin(np.abs(res['t'] - tt)))
    prof = res['top'][i]
    scale = 0.85 / max(np.max(np.abs(res['top'][int(np.argmin(np.abs(res['t'] - times[-1])))])), 1e-300)
    ax.plot(x2b, jt + scale * prof, color=COLORS['psi1'], lw=1.6)
    ax.text(-1.2, jt, rf"$t={tt:.0f}$", ha="right", va="center", fontsize=11)
ax.annotate("", xy=(3.2, -0.75), xytext=(0.4, -0.75),
            arrowprops=dict(arrowstyle="-|>", color=COLORS['jet'], lw=3))
ax.text(0.4, -0.95, "flow", color=COLORS['jet'], fontsize=11)
ax.set_xlabel(r"Downstream distance ($\times 2b$)")
ax.set_yticks([])
ax.set_title("bank planform: 15 seeded wavelengths -> growth-band selection "
             "+ upstream march (rayleigh; common scale)")
save_fig(fig, "fig04_selection_waterfall")

# ------------------------------------------------------------- fig05 ---- #
kf = np.linspace(0.02, 1.6, 200)
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
for jf, friction in enumerate(("rayleigh", "momentum")):
    p = Params(D=D, gamma=G, friction=friction)
    rows = measured[friction]
    ks = np.array([r[0] for r in rows])
    sm = np.array([r[1] for r in rows])
    cm = np.array([r[2] for r in rows])
    oe = np.array([r[3] for r in rows])
    axg, axc = axes[jf]
    # EVP line (dense k) -- use GEP-continued targets for robustness
    tline = VL.bank_branch(kf, p.D, p.gamma, p.E, p.friction)
    axg.plot(kf, tline.imag, color=COLORS['decay'], lw=1.0, ls=":",
             label="2x2 closure")
    oL = np.array([evp_bank_mode(float(k), p, Ny=96,
                                 target=complex(tline[i]))
                   for i, k in enumerate(kf[::8])])
    axg.plot(kf[::8], oL.imag, color='k', lw=1.5, label="d3 EVP")
    axc.plot(kf[::8], oL.real / kf[::8], color='k', lw=1.5)
    axc.plot(kf, tline.real / kf, color=COLORS['decay'], lw=1.0, ls=":")
    axg.plot(ks, sm, 'o', ms=7, color=COLORS['dedalus'], mec='k', mew=0.6,
             label="IVP (multi-k run)")
    axc.plot(ks, cm, 'o', ms=7, color=COLORS['dedalus'], mec='k', mew=0.6)
    axg.plot(ks, oe.imag, '+', ms=8, color=COLORS['erosion'], mew=1.5)
    axg.axhline(0, color='k', lw=0.8)
    axc.axhline(0, color='k', lw=0.8)
    axg.axvline(np.sqrt(2 * D), color=COLORS['upstream'], lw=1.0, ls='--')
    axg.text(np.sqrt(2 * D) + 0.02, 0.12, r"$k^{*2}=2D$", fontsize=9,
             color=COLORS['upstream'])
    axg.set_title(f"growth -- {friction}")
    axc.set_title(f"phase speed -- {friction}")
    axg.set_ylabel(r"$\sigma^*$")
    axc.set_ylabel(r"$c^*$")
    axc.set_ylim(-1.2, 0.3)
    axg.legend(fontsize=9)
for ax in axes[1]:
    ax.set_xlabel(r"$k^*=kb$")
axes[1, 0].set_xlabel(r"$k^*=kb$")
fig.suptitle("dispersion measured from ONE multi-wavelength IVP: dots = "
             "per-mode demodulation; line = d3 EVP; + = EVP at seeds",
             y=0.995, fontsize=12)
save_fig(fig, "fig05_measured_dispersion")

print("03_multi_k_selection: done.")
