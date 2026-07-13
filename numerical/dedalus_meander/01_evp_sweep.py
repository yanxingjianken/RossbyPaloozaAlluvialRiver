#!/usr/bin/env python3
"""fig01-02: Dedalus EVP sweep vs the vorticity_lib theory curves.

For the six deck parameter sets and both friction closures: sigma(k*) and
c(k*) of the bank mode from the d3 eigenvalue problem (dots), overlaid on
the N=201 FD GEP curves (lines, the continuum target) and the 2x2 closure
(dotted). Hard asserts at k* in {0.2,...,1.4} against the Richardson-
extrapolated (201/401) GEP.

Run: micromamba run -n dedalus env OMP_NUM_THREADS=1 python 01_evp_sweep.py
"""
import numpy as np

from channel_lib import (COLORS, Params, VL, evp_bank_mode, gep_bank_mode,
                         gep_richardson, set_style, save_fig)

plt = set_style()

SETS = (((0.3, 0.05), (0.6, 0.05), (0.9, 0.05)),
        ((0.6, 0.03), (0.6, 0.06), (0.6, 0.09)))
FAMNAMES = ("D-family", "g-family")
CURVE_COLS = ("#2040d0", "#7b2d8b", "#e03020")

K_DOTS = np.arange(0.05, 1.56, 0.05)
K_LINE = np.linspace(0.02, 1.6, 80)
K_ASSERT = np.arange(0.2, 1.41, 0.2)

for friction in ("rayleigh", "momentum"):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
    print(f"=== {friction} closure ===")
    for jc, (fam, famname) in enumerate(zip(SETS, FAMNAMES)):
        axg, axc = axes[0, jc], axes[1, jc]
        for (D, g), col in zip(fam, CURVE_COLS):
            p = Params(D=D, gamma=g, friction=friction)
            # continuation targets along k (avoids branch mix-ups)
            tgrid = np.linspace(1e-3, 1.6, 120)
            tline = VL.bank_branch(tgrid, p.D, p.gamma, p.E, p.friction)

            # GEP N=201 lines
            oL = np.array([gep_bank_mode(k, p, 201) for k in K_LINE])
            axg.plot(K_LINE, oL.imag, color=col, lw=1.7,
                     label=rf"$D={D}$, $\gamma={g}$")
            axc.plot(K_LINE, oL.real / K_LINE, color=col, lw=1.7)
            # 2x2 closure dotted
            sig2, c2 = VL.growth_curve(K_LINE, p)
            axg.plot(K_LINE, sig2, color=col, lw=1.0, ls=":")
            axc.plot(K_LINE, c2, color=col, lw=1.0, ls=":")
            # Dedalus dots
            oD = np.array([
                evp_bank_mode(k, p, Ny=96,
                              target=complex(tline[np.argmin(np.abs(tgrid - k))]))
                for k in K_DOTS])
            axg.plot(K_DOTS, oD.imag, 'o', ms=4, color=col, mec='k', mew=0.4)
            axc.plot(K_DOTS, oD.real / K_DOTS, 'o', ms=4, color=col,
                     mec='k', mew=0.4)

            # hard asserts vs Richardson GEP
            sig_pk = max(oL.imag.max(), 1e-6)
            worst = 0.0
            for k in K_ASSERT:
                tgt = complex(tline[np.argmin(np.abs(tgrid - k))])
                od = evp_bank_mode(k, p, Ny=96, target=tgt)
                orr = gep_richardson(k, p)
                ds = abs(od.imag - orr.imag)
                dc = abs(od.real / k - orr.real / k)
                assert ds <= max(0.01 * sig_pk, 5e-4), \
                    f"{friction} D={D} g={g} k={k}: dsigma {ds:.2e}"
                assert dc <= max(0.01 * abs(orr.real / k), 5e-3), \
                    f"{friction} D={D} g={g} k={k}: dc {dc:.2e}"
                worst = max(worst, ds)
            print(f"  D={D} g={g}: {len(K_ASSERT)} assert points, "
                  f"worst |dsigma| = {worst:.2e}. OK")
        axg.axhline(0, color='k', lw=0.8)
        axc.axhline(0, color='k', lw=0.8)
        axg.set_title(f"growth rate -- {famname}")
        axc.set_title(f"phase speed -- {famname}")
        axg.set_ylim(-0.35, 0.25)
        axc.set_ylim(-5, 1)
        axc.set_xlabel(r"wavenumber $k^*=kb$")
        axg.legend(fontsize=9, loc="upper right")
    axes[0, 0].set_ylabel(r"$\sigma^*$")
    axes[1, 0].set_ylabel(r"$c^*$")
    fig.suptitle(f"Dedalus EVP (dots) vs FD GEP N=201 (lines) vs 2x2 closure "
                 f"(dotted) -- {friction} closure", y=0.995, fontsize=12)
    save_fig(fig, f"fig0{1 if friction == 'rayleigh' else 2}_evp_sweep_{friction}")

print("01_evp_sweep: done.")
