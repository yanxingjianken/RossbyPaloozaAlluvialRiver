#!/usr/bin/env python3
"""fig03: single-wavelength IVP runs vs the d3 EVP (the sharpest closed loop).

For (D, gamma) = (0.6, 0.05), both closures, k* in {0.3, 0.44, 0.9, 1.3}:
seed the erodible banks with one sinusoid (harmonic-extension IC), evolve,
demodulate, and fit (sigma, c). Controls: rigid banks (must not grow) and a
dt-halving accuracy check.

Run: micromamba run -n dedalus env OMP_NUM_THREADS=1 python 02_single_k_ivp.py
"""
import numpy as np

from channel_lib import (COLORS, Params, build_ivp, demodulate, evp_bank_mode,
                         fit_sigma_c, run_ivp, seed_banks, t_filament,
                         set_style, save_fig)

plt = set_style()

D, G = 0.6, 0.05
KS = (0.3, 0.44, 0.9, 1.3)
NY, DT, A0 = 192, 0.02, 1e-4

fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.0), sharex='col')
results = {}
for jf, friction in enumerate(("rayleigh", "momentum")):
    print(f"=== {friction} ===")
    p = Params(D=D, gamma=G, friction=friction)
    for jk, k in enumerate(KS):
        om = evp_bank_mode(k, p, Ny=96)
        sig_p, c_p = om.imag, om.real / k
        tres = 0.9 * t_filament(k, D, NY)
        t_end = min(max(6.0 / sig_p, 40.0) if sig_p > 0 else 120.0, tres)
        w0 = t_end / 3.0 if sig_p > 0 else 10.0
        built = build_ivp(p, Lx=2 * np.pi / k, Nx=8, Ny=NY)
        seed_banks(built, [(k, A0, 0.0)])
        res = run_ivp(built, dt=DT, t_end=t_end, rec_dt=0.25)
        a = demodulate(0.5 * (res['top'] + res['bot']), 1)
        sig, c, r2 = fit_sigma_c(res['t'], a, k, (w0, t_end))
        ds, dc = abs(sig - sig_p), abs(c - c_p)
        print(f"  k*={k}: sigma {sig:+.5f} (EVP {sig_p:+.5f}, d={ds:.1e}) "
              f" c {c:+.4f} (EVP {c_p:+.4f}, d={dc:.1e})  R2={r2:.6f}")
        assert ds < max(0.02 * abs(sig_p), 5e-4), \
            f"{friction} k*={k}: sigma off ({ds:.2e})"
        assert dc < max(0.02 * abs(c_p), 5e-3), \
            f"{friction} k*={k}: c off ({dc:.2e})"
        results[(friction, k)] = (res['t'], a, om)

        ax = axes[jf, jk]
        ax.semilogy(res['t'], np.abs(a), color=COLORS['dedalus'], lw=1.8,
                    label="IVP $|a_1|(t)$")
        tt = np.linspace(w0, t_end, 50)
        aref = np.abs(a[np.argmin(np.abs(res['t'] - w0))])
        ax.semilogy(tt, aref * np.exp(sig_p * (tt - w0)), 'k--', lw=1.2,
                    label=r"EVP $e^{\sigma t}$")
        ax.set_title(rf"{friction}, $k^*={k}$: $c={c:+.3f}$", fontsize=10)
        if jk == 0:
            ax.set_ylabel(r"$|a_1|$")
        if jf == 1:
            ax.set_xlabel(r"$t$")
        ax.legend(fontsize=7, loc="lower right")

# rigid-bank control: interior seed, no growth allowed
p = Params(D=D, gamma=G)
k = 0.44
built = build_ivp(p, Lx=2 * np.pi / k, Nx=8, Ny=NY, banks="rigid")
x, y = built['x'], built['y']
built['psi']['g'] = A0 * np.cos(k * x) * (1 - y**2)
res = run_ivp(built, dt=DT, t_end=60.0)
a = demodulate(res['top'], 1)          # rigid: tracked centre row
sig_r, _, _ = fit_sigma_c(res['t'], a, k, (20.0, 60.0))
print(f"rigid control k*={k}: sigma {sig_r:+.5f} (must be < 0)")
assert sig_r < 0, "rigid banks grew!"

# dt-halving accuracy check (rayleigh, k*=0.44)
k = 0.44
p = Params(D=D, gamma=G)
om = evp_bank_mode(k, p, Ny=96)
fits = {}
for dt in (0.02, 0.01):
    built = build_ivp(p, Lx=2 * np.pi / k, Nx=8, Ny=NY)
    seed_banks(built, [(k, A0, 0.0)])
    res = run_ivp(built, dt=dt, t_end=50.0)
    a = demodulate(0.5 * (res['top'] + res['bot']), 1)
    fits[dt], _, _ = fit_sigma_c(res['t'], a, k, (17.0, 50.0))
drift = abs(fits[0.01] - fits[0.02]) / abs(fits[0.01])
print(f"dt-halving: sigma(0.02)={fits[0.02]:.6f} sigma(0.01)={fits[0.01]:.6f} "
      f"drift {100*drift:.4f}%")
assert drift < 0.001, "sigma not converged in dt"

fig.suptitle("single-k erodible-bank IVP vs d3 EVP  "
             r"($D=0.6$, $\gamma=0.05$; growth panels log-scale)",
             y=1.00, fontsize=12)
save_fig(fig, "fig03_single_k_ivp")
print("02_single_k_ivp: done.")
