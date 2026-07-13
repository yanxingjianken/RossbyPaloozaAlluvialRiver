#!/usr/bin/env python3
"""planform_upstream.mp4: the 2-D field marching upstream while growing.

Long-wave flank run (k* = 0.15, where |c*| is large relative to sigma*):
psi'(x,y,t) contours + bank lines, normalized per frame by the measured
sinuous amplitude (linear mode shape -- honest convention of
../vorticity_meander/05_anim_meander_mode.py), with the true e^{sigma t}
gain reported by a counter and the crest tracker read from the measured
phase (not from theory).

Usage: python 05_anim_planform.py [--max-frames N]
"""
import argparse

import numpy as np

from channel_lib import (COLORS, Params, build_ivp, demodulate, evp_bank_mode,
                         fit_sigma_c, run_ivp, seed_banks, set_style,
                         warp_fill, fig_to_rgb, write_mp4)

plt = set_style()

ap = argparse.ArgumentParser()
ap.add_argument("--max-frames", type=int, default=92)
args = ap.parse_args()

K = 0.15
p = Params(D=0.6, gamma=0.05)
LX = 2 * np.pi / K
om = evp_bank_mode(K, p, Ny=96)

built = build_ivp(p, Lx=LX, Nx=32, Ny=128)
seed_banks(built, [(K, 1e-4, 0.0)])
T0, T1 = 15.0, 107.0
res = run_ivp(built, dt=0.02, t_end=T1, rec_dt=0.5,
              snap_dt=(T1 - T0) / (args.max_frames - 1) if args.max_frames > 1 else 1.0)

a1 = demodulate(0.5 * (res['top'] + res['bot']), 1)
sig, c, _ = fit_sigma_c(res['t'], a1, K, (T0, T1))
print(f"measured: sigma={sig:.4f} c={c:.4f} (EVP {om.imag:.4f}, "
      f"{om.real/K:.4f})")

x2b = res['x'] / 2.0
y = res['y']
frames = []
sel = [i for i, t in enumerate(res['tsnap']) if t >= T0][:args.max_frames]
i0 = sel[0]
gain0 = None
for i in sel:
    t = res['tsnap'][i]
    it = int(np.argmin(np.abs(res['t'] - t)))
    amp = np.abs(a1[it])
    if gain0 is None:
        gain0 = amp
    # crest position from the measured phase: Re[a e^{ikx}] max at kx=-arg a
    xc = (-np.angle(a1[it]) / K) % LX / 2.0

    fig, ax = plt.subplots(figsize=(9.6, 3.8), dpi=110)
    scale = 0.5 / amp
    warp_fill(ax, x2b, y, scale * res['psis'][i],
              scale * res['top'][it], scale * res['bot'][it])
    ax.axvline(xc, color=COLORS['upstream'], lw=2.2, alpha=0.9)
    ax.annotate("", xy=(2.4, -2.15), xytext=(0.3, -2.15),
                arrowprops=dict(arrowstyle="-|>", color=COLORS['jet'], lw=3))
    ax.text(0.3, -2.55, "flow", color=COLORS['jet'], fontsize=11)
    ax.text(0.3, 1.75,
            rf"$k^*={K}$  $c^*={c:+.3f}$ (upstream)  $\sigma^*={sig:.3f}$:"
            rf"  gain $\times{amp/gain0:,.0f}$",
            fontsize=11)
    ax.set_ylim(-2.8, 2.3)
    ax.set_xlim(0, LX / 2)
    ax.set_xlabel(r"Downstream distance ($\times 2b$)")
    ax.set_title(r"Dedalus channel: $\psi'$ + erodible banks, normalized "
                 "planform marching upstream")
    fig.tight_layout()
    frames.append(fig_to_rgb(fig))
    plt.close(fig)

write_mp4(frames, "planform_upstream", fps=16)
print("05_anim_planform: done.")
