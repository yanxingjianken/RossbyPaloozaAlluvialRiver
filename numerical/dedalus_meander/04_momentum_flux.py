#!/usr/bin/env python3
"""fig06-07: the wave's lateral momentum transport, measured in the IVP.

Deck p. 6 says the meander wave carries downstream momentum from the jet
core to the banks. For the FREE GROWING mode the rayleigh centre balance
generalizes the steady-forced factor gamma/(2D) to

    -<v' zeta'> / <zeta'^2> |_{y=0} = (gamma + sigma) / (2D),

i.e. growth itself tilts the wave: the flux survives even at gamma = 0.
This script measures the <u'v'>(y) profile in the IVP (k* = 0.44, D = 0.6),
compares it with the EVP eigenfunction's 0.5*Re(u v*) profile, and checks
the centre ratio for gamma = 0.05 and the gamma = 0 twin.

Run: micromamba run -n dedalus env OMP_NUM_THREADS=1 python 04_momentum_flux.py
"""
import numpy as np

from channel_lib import (COLORS, Params, build_ivp, evp_bank_mode, run_ivp,
                         seed_banks, set_style, save_fig)

plt = set_style()

K, D = 0.44, 0.6
NY, DT, A0 = 192, 0.02, 1e-4

fig6, axes6 = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
fig7, ax7 = plt.subplots(figsize=(7.2, 4.2))

cases = (("rayleigh", 0.05, COLORS['growth']),
         ("momentum", 0.05, COLORS['momentum']),
         ("rayleigh", 0.0, COLORS['upstream']))
for jc, (friction, G, col) in enumerate(cases):
    p = Params(D=D, gamma=G, friction=friction)
    om, mode = evp_bank_mode(K, p, Ny=96, return_mode=True)
    sig = om.imag
    t_end = min(6.0 / sig, 60.0)
    built = build_ivp(p, Lx=2 * np.pi / K, Nx=8, Ny=NY)
    seed_banks(built, [(K, A0, 0.0)])
    res = run_ivp(built, dt=DT, t_end=t_end, rec_dt=0.5, prof_dt=1.0)

    y = res['y']
    # late-time measured profile, normalized by its max
    uv = res['uv'][-1]
    uvn = uv / np.max(np.abs(uv))
    # EVP eigenfunction profile 0.5*Re(u v*) = 0.5*Re(-dpsi * (ik psi)*)
    uv_mode = 0.5 * np.real(-mode['dpsi'] * np.conj(1j * K * mode['psi']))
    uv_mode_n = uv_mode / np.max(np.abs(uv_mode))
    uv_mode_i = np.interp(y, mode['y'], uv_mode_n)
    linf = np.max(np.abs(uvn - uv_mode_i))
    print(f"{friction} gamma={G}: sigma={sig:.4f}  "
          f"|uv_IVP - uv_EVP|_inf = {linf:.2e}")
    assert linf < 0.02, "IVP flux profile deviates from eigenmode > 2%"
    # flux carries momentum out of the core: increasing through the centre
    up = np.interp(0.25, y, uvn) - np.interp(-0.25, y, uvn)
    assert up > 0, "flux not out of the channel core"

    if jc < 2:
        ax = axes6[jc]
        ax.plot(uv_mode_i, y, color='k', lw=1.6, label="EVP eigenmode")
        ax.plot(uvn[::4], y[::4], 'o', ms=4, color=col, label="IVP (late)")
        ax.axvline(0, color='k', lw=0.7)
        ax.set_title(rf"{friction}, $\gamma={G}$")
        ax.set_xlabel(r"$\langle u'v'\rangle$ (normalized)")
        ax.legend(fontsize=9)

    # centre ratio vs (gamma + sigma)/(2D)   [rayleigh balance]
    tsel = res['tprof'] >= res['tprof'][-1] / 2
    ratio = np.array([
        -np.interp(0.0, y, res['vzeta'][i]) / np.interp(0.0, y, res['zeta2'][i])
        for i in np.where(tsel)[0]])
    pred = (G + sig) / (2 * D)
    lab = rf"{friction}, $\gamma={G}$"
    ax7.plot(res['tprof'][tsel], ratio, 'o-', ms=3.5, lw=1.0, color=col,
             label=lab)
    ax7.axhline(pred, color=col, lw=1.2, ls='--')
    err = abs(np.median(ratio) - pred) / pred
    print(f"  centre ratio -<v'z'>/<z'^2> = {np.median(ratio):.4f}  "
          f"(gamma+sigma)/2D = {pred:.4f}  err {100*err:.2f}%")
    if friction == "rayleigh":
        assert err < 0.03, "centre flux ratio off > 3% (rayleigh balance)"

axes6[0].set_ylabel(r"$y/b$")
fig6.suptitle(rf"lateral momentum flux of the growing mode ($k^*={K}$): "
              "IVP vs EVP eigenfunction", y=1.02)
save_fig(fig6, "fig06_uv_profiles")

ax7.set_xlabel(r"$t$")
ax7.set_ylabel(r"$-\langle v'\zeta'\rangle/\langle\zeta'^2\rangle$ at $y=0$")
ax7.set_title("centre flux ratio: measured vs $(\\gamma+\\sigma)/2D$ "
              "(dashed) -- growth keeps the flux alive at $\\gamma=0$")
ax7.legend(fontsize=9)
save_fig(fig7, "fig07_flux_ratio")

print("04_momentum_flux: done.")
