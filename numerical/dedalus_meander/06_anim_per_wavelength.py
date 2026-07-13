#!/usr/bin/env python3
"""per-wavelength 4-panel diagnostic movies: each seeded wavelength -> its OWN mp4.

The fig04 waterfall seeds 15 bank sinusoids in one channel; here each of
those wavelengths is run ALONE on the same Lx = 20*pi window and rendered
as its own 2x2 movie:
  (0,0) psi_total = psibar + psi'  (streamlines of the meandering jet)
  (0,1) psi'                        (perturbation)
  (1,0) u'v'                        (momentum flux / Reynolds stress)
  (1,1) stats + log|a(t)| erosion-growth curve
Because the window is common, the movies show wavelengths at true relative
size (k*=0.1 = 1 meander across the reach, k*=1.5 = 15). The subtitle carries
lambda/2b, measured sigma* (vs EVP), phase speed c* AND group speed c_g
(upstream / downstream), + an honest e^{sigma t} gain. The growth curve makes
the erosion explicit (the field panels are amplitude-normalized so the mode
stays visible; the pattern's translation is the phase speed c*, NOT mean-flow
advection -- the bank eq has no d/dx term).

Outputs: figures/per_wavelength/planform_k<..>.mp4 (+ previews) and
figures/fig08_per_wavelength_grid.png (final-frame contact sheet).

Usage:
  micromamba run -n dedalus env OMP_NUM_THREADS=1 python 06_anim_per_wavelength.py
  ... python 06_anim_per_wavelength.py --kstars 0.3,0.5,1.2 --friction momentum
"""
import argparse
import os

import numpy as np

from channel_lib import (COLORS, FIG_DIR, Params, VL, build_ivp, demodulate,
                         evp_bank_mode, fit_sigma_c, four_panel_frames, run_ivp,
                         seed_banks, t_filament, set_style, save_fig,
                         warp_fill, write_mp4)

plt = set_style()

ap = argparse.ArgumentParser()
ap.add_argument("--kstars", type=str, default=None,
                help="comma list of k* (default: 0.1..1.5 step 0.1)")
ap.add_argument("--friction", choices=("rayleigh", "momentum"),
                default="rayleigh")
ap.add_argument("--frames", type=int, default=48)
args = ap.parse_args()

D, G = 0.6, 0.05
LX = 20 * np.pi                       # dk = 0.1 (same reach as fig04/05)
NY, DT, A0 = 128, 0.02, 1e-4
KSTARS = ([float(s) for s in args.kstars.split(",")] if args.kstars
          else [round(0.1 * m, 1) for m in range(1, 16)])

SUBDIR = os.path.join(FIG_DIR, "per_wavelength")
os.makedirs(SUBDIR, exist_ok=True)


def nx_for(m):
    """Resolve m periods with >=12 pts/period, power of two, capped at 192."""
    return int(min(192, max(64, 1 << int(np.ceil(np.log2(12 * m))))))


def group_speed(k, p, dk=0.02):
    """c_group = d omega_r / dk of the bank mode (branch-continued)."""
    ks = np.array([max(k - dk, 1e-3), k, k + dk])
    om = VL.bank_branch(ks, p.D, p.gamma, p.E, p.friction)
    return float((om[2].real - om[0].real) / (ks[2] - ks[0]))


def classify(sig, c):
    g = ("grows" if sig > 5e-3 else "decays" if sig < -5e-3 else "~neutral")
    d = ("upstream" if c < -2e-2 else "downstream" if c > 2e-2
         else "~stationary")
    return g, d


p = Params(D=D, gamma=G, friction=args.friction)
sheet = []
print(f"=== per-wavelength movies ({args.friction}, D={D}, gamma={G}) ===")
for k in KSTARS:
    m = int(round(k * LX / (2 * np.pi)))
    Nx = nx_for(m)
    om = evp_bank_mode(k, p, Ny=96)
    sig_e, c_e = om.imag, om.real / k
    tf = 0.9 * t_filament(k, D, NY)
    if sig_e > 1e-4:                    # grower
        t_end = float(min(5.0 / sig_e, tf, 160.0))
        win = (t_end / 3.0, t_end)
    else:                              # decayer / neutral
        t_end = float(min(3.0 / max(abs(sig_e), 1e-3), tf, 120.0))
        win = (max(4.0, 0.25 * t_end), t_end)
    snap_dt = t_end / args.frames

    built = build_ivp(p, Lx=LX, Nx=Nx, Ny=NY)
    seed_banks(built, [(k, A0, 0.0)])
    res = run_ivp(built, dt=DT, t_end=t_end, rec_dt=snap_dt, snap_dt=snap_dt)

    a = demodulate(0.5 * (res['top'] + res['bot']), m)
    sig, c, r2 = fit_sigma_c(res['t'], a, k, win)
    c_g = group_speed(k, p)
    g, d = classify(sig, c)
    tag = f"{k:.2f}".replace('.', 'p')
    title = (rf"$k^*={k:g}$   $\lambda/2b={np.pi/k:.1f}$   "
             rf"$\sigma^*={sig:+.3f}$ ({g})   crests $c^*={c:+.3f}$ ({d})   "
             rf"momentum $c_g={c_g:+.3f}$")
    print(f"  k*={k:>3} m={m:>2} Nx={Nx:>3}: sigma {sig:+.4f} (EVP {sig_e:+.4f})"
          f"  c {c:+.4f} (EVP {c_e:+.4f})  c_g {c_g:+.4f}  {g},{d}  t_end={t_end:.0f}")

    stats = dict(sigma=sig, sigma_evp=sig_e, c_phase=c, c_group=c_g,
                 gamma=G, E=p.E, friction=args.friction)
    frames = four_panel_frames(res, m, k, D, plt, stats, title, t0=0.0)
    write_mp4(frames, os.path.join("per_wavelength", f"planform_k{tag}"),
              fps=14)

    # keep final normalized planform for the contact sheet
    pf = res['psis'][-1]
    sc = 0.9 / max(np.max(np.abs(pf)), 1e-300)
    sheet.append((k, m, sc * pf, sc * res['top'][-1], sc * res['bot'][-1],
                  sig, c, g, d))

# ---- fig08: final-frame contact sheet (the wavelength ladder) ----------- #
ncol = 5
nrow = int(np.ceil(len(sheet) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.0 * nrow + 0.6),
                         squeeze=False)
for ax in axes.ravel():
    ax.axis('off')
for idx, (k, m, pf, top, bot, sig, c, g, d) in enumerate(sheet):
    ax = axes[idx // ncol][idx % ncol]
    ax.axis('on')
    xg = np.linspace(0, LX / 2.0, pf.shape[0])
    yg = np.linspace(-1, 1, pf.shape[1])
    warp_fill(ax, xg, yg, pf, top, bot, vlim=0.9)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlim(0, LX / 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(rf"$k^*={k:g}$  $\sigma^*={sig:+.2f}$  $c^*={c:+.2f}$",
                 fontsize=8.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.suptitle(f"per-wavelength final planforms ({args.friction}): growth-band "
             r"selection + upstream drift (each normalized; common reach $20\pi$)",
             y=0.985, fontsize=12)
save_fig(fig, "fig08_per_wavelength_grid")

print(f"06_anim_per_wavelength: done ({len(sheet)} movies + contact sheet).")
