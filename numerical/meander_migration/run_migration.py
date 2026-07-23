#!/usr/bin/env python3
"""Evolve the Ikeda linear bend equation (16) from a small broadband perturbation and save the
migrating centreline y(x,t).  Rigid bed, MIGRATING bank -- the meander waveform grows and travels
down-valley, and the preferred (max-growth) wavelength emerges from the noise.

    micromamba run -n fourcastnetv2 python run_migration.py

Writes migration.npz for figures.py.  Everything is derived from bend_model.BendParams; nothing
here bakes in a wavelength -- the selection is an OUTPUT of the dispersion relation.
"""
import os
import numpy as np
import bend_model as bm

HERE = os.path.dirname(os.path.abspath(__file__))

# --- parameters: Cf=0.01 puts the selected wavelength at ~12.5 channel widths (Leopold-Wolman) -
P = bm.BendParams(H0=3.0, U0=0.85, Cf=0.01, A=2.89, b=50.0)
km = P.k_max()
lam_sel = P.wavelength(km)                       # selected meander wavelength [m]
print(f"F={P.F:.3f}  A={P.A}  A+F^2={P.B:.3f}  (F^2 = {100*P.F2/P.B:.1f}% -> secondary-flow driven)")
print(f"selected: k_max={km:.4f}, lambda={lam_sel:.0f} m = {lam_sel/(2*P.b):.1f} W, "
      f"c0={P.migration_speed(km):.4f}")

# --- domain: ~6 selected wavelengths, dimensionless x (scaled by H0), periodic ----------------
n_lam = 6
Lx = n_lam * lam_sel / P.H0                       # dimensionless domain length
N = 2048
x = np.linspace(0.0, Lx, N, endpoint=False)
dx = x[1] - x[0]

# --- initial condition: small broadband noise (dimensionless amplitude ~1e-3 of the depth) ----
rng = np.random.default_rng(0)
y0 = 1e-3 * rng.standard_normal(N)
y0 -= y0.mean()
# gently band-limit so the very shortest grid modes (which are damped anyway) don't dominate
Y0 = np.fft.fft(y0); kk = 2 * np.pi * np.fft.fftfreq(N, d=dx)
Y0[np.abs(kk) > 5 * P.k_cut()] = 0.0
y0 = np.real(np.fft.ifft(Y0))

# --- evolve to several e-folding times of the fastest mode -----------------------------------
alpha_max = P.growth(km)
Tend = 9.0 / alpha_max                             # ~9 e-foldings
times = np.linspace(0.0, Tend, 240)
Y = bm.evolve(y0, dx, P, times)                    # (nt, N), dimensionless
ub = bm.near_bank_velocity(Y, times)               # near-bank velocity perturbation u'_b(x,t)

np.savez(os.path.join(HERE, "migration.npz"),
         x=x * P.H0, y=Y * P.H0, ub=ub, times=times, dx=dx,
         H0=P.H0, U0=P.U0, Cf=P.Cf, A=P.A, F=P.F, b=P.b,
         k_max=km, lam_sel=lam_sel, alpha_max=alpha_max,
         c0=P.migration_speed(km))
print(f"wrote migration.npz: {len(times)} frames, domain {n_lam} wavelengths, "
      f"final amplitude {np.abs(Y[-1]).max()*P.H0:.1f} m")
