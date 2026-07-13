#!/usr/bin/env python3
"""meander_mode.mp4: the growing bank mode marching UPSTREAM.

Animates the bank mode at (D = 0.6, gamma = 0.05, rayleigh) on the long-wave
flank of the growth band (k* = 0.15, where the upstream march is fastest
relative to the domain).  A REAL property of this mode is that growth beats
propagation (e^{sigma 2 pi/|om_r|} ~ e^6 per phase period), so the planform
is drawn at FIXED (normalised) amplitude -- the linear mode shape -- and the
true exponential growth is reported by the e^{sigma t} counter.  Bank lines
y = +-b + Re[psihat_1 e^{i(k*x - om* t)}], centre streamline with
psihat_2 = (W + E)/E psihat_1 from the bank equation.

Usage: python 05_anim_meander_mode.py [--max-frames N]
"""
import argparse

import numpy as np

from vorticity_lib import (COLORS, Params, bank_branch, set_style,
                           fig_to_rgb, write_mp4)

plt = set_style()

ap = argparse.ArgumentParser()
ap.add_argument("--max-frames", type=int, default=96)
args = ap.parse_args()

p = Params(D=0.6, gamma=0.05)
kan = 0.15                              # long-wave flank: visible march
om = complex(bank_branch([kan], p.D, p.gamma, p.E)[0])
W = -1j * om
psi1 = 1.0
psi2 = (W + p.E) / p.E * psi1           # bank equation
c = om.real / kan                       # < 0: upstream (b per time unit)

x = np.linspace(0, 25, 1200)            # in 2b units
kx = kan * 2.0 * x
x0, x1 = 21.0, 2.5                      # crest transit (2b units)
T = (x0 - x1) / (abs(c) / 2.0)          # time for the transit
ts = np.linspace(0.0, T, args.max_frames)
amp = 0.55                              # fixed display amplitude (mode shape)

frames = []
for t in ts:
    fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=110)
    ph = np.exp(1j * (kx - om.real * t))
    ax.plot(x, 1.0 + amp * np.real(psi1 * ph), color=COLORS["psi1"], lw=1.9)
    ax.plot(x, -1.0 + amp * np.real(psi1 * ph), color=COLORS["psi1"], lw=1.9)
    ax.plot(x, amp * np.real(psi2 * ph), color=COLORS["psi2"], lw=1.9)
    xc = x0 + (c / 2.0) * t             # crest tracker, marches upstream
    ax.axvline(xc, color=COLORS["upstream"], lw=2.2, alpha=0.85)
    ax.axvline(x0, color=COLORS["upstream"], lw=1.0, ls=":", alpha=0.7)
    ax.annotate("", xy=(xc, 3.1), xytext=(x0, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["upstream"],
                                lw=1.8))
    ax.annotate("", xy=(2.6, -2.6), xytext=(0.4, -2.6),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["jet"], lw=3))
    ax.text(0.4, -3.6, "flow", color=COLORS["jet"], fontsize=11)
    ax.text(0.4, 2.5,
            rf"$k^*={kan}$   $c^*={c:.3f}$ (upstream)   "
            rf"$\sigma^*={om.imag:.3f}$:  amplitude gain "
            rf"$e^{{\sigma t}} = \times{np.exp(om.imag * t):,.0f}$",
            fontsize=11)
    ax.set_ylim(-4, 4)
    ax.set_xlim(0, 25)
    ax.set_xlabel(r"Downstream distance ($\times 2b$)")
    ax.set_title(rf"bank mode, $D={p.D}$, $\gamma={p.gamma}$: normalised "
                 "planform marching upstream while growing")
    fig.tight_layout()
    frames.append(fig_to_rgb(fig))
    plt.close(fig)

write_mp4(frames, "meander_mode", fps=16)
print("05_anim_meander_mode: done.")
