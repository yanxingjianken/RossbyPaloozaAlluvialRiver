#!/usr/bin/env python3
"""Figures for the Ikeda migrating-meander model:
  figures/dispersion.png        -- growth rate + migration speed vs wavelength (gravity vs vortical)
  figures/meander_migration.mp4 -- the centreline waveform migrating DOWN-VALLEY + growing from noise

    micromamba run -n fourcastnetv2 python figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import bend_model as bm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")


def make_dispersion():
    P = bm.BendParams(Cf=0.01)
    kc = P.k_cut(); km = P.k_max()
    k = np.linspace(1e-4 * kc, 1.4 * kc, 2000)
    lam = P.wavelength(k) / (2 * P.b)                     # wavelength in channel widths
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(lam, P.growth(k), "b")
    ax[0].axhline(0, color="0.6", lw=0.7)
    ax[0].axvline(P.wavelength(km) / (2 * P.b), color="r", ls="--",
                  label=f"selected $\\lambda$ = {P.wavelength(km)/(2*P.b):.1f} W")
    ax[0].set_xlabel("meander wavelength $\\lambda$ / W"); ax[0].set_ylabel("growth rate $\\alpha_0$")
    ax[0].set_title("bend AMPLITUDE growth (eq 16/17)"); ax[0].legend()
    ax[1].plot(lam, P.migr_freq(k) / k, "g")
    ax[1].axvline(P.wavelength(km) / (2 * P.b), color="r", ls="--")
    ax[1].set_xlabel("meander wavelength $\\lambda$ / W")
    ax[1].set_ylabel("downstream migration speed $c_0=\\omega_0/k$")
    ax[1].set_title("bends always migrate DOWNSTREAM ($\\omega_0>0$)")
    fig.suptitle(f"Ikeda-Parker-Sawai 1981 dispersion:  driver $A+F^2$ = {P.B:.2f}  "
                 f"($A$=2.89 secondary-flow/vortical vs $F^2$={P.F2:.3f} gravity → "
                 f"{100*P.F2/P.B:.1f}% gravity: NOT a gravity wave, $c_0\\ll\\sqrt{{gH}}$)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "dispersion.png"), dpi=120)
    plt.close(fig)
    print("wrote figures/dispersion.png")


def make_movie(fps=12):
    d = np.load(os.path.join(HERE, "migration.npz"))
    x, Y, ub, t = d["x"], d["y"], d["ub"], d["times"]
    b = float(d["b"]); lam = float(d["lam_sel"]); c0 = float(d["c0"]); H0 = float(d["H0"])
    amp = np.abs(Y).max(axis=1)                            # growth curve (dimensional, m)
    disp_amp = 1.3 * b                                     # fixed DISPLAY amplitude (linear theory: amplitude is arbitrary)
    # track a crest to show migration: follow the argmax of the (normalised) final-pattern phase
    mp4 = os.path.join(OUT, "meander_migration.mp4")
    os.makedirs(OUT, exist_ok=True)
    w = imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    # migration reference marker: a fixed material phase point advected at c0 (dimensionless speed
    # -> physical via the H0 length and the E0*b* time already folded into t); show relative shift
    x0_marker = x[len(x) // 2]
    for f in range(len(t)):
        yy = Y[f] / max(np.abs(Y[f]).max(), 1e-12) * disp_amp   # shape, display-normalised
        fig, ax = plt.subplots(2, 1, figsize=(12, 4.6), height_ratios=[2, 1], sharex=True)
        # channel planform: centreline +/- half width
        ax[0].fill_between(x, yy - b, yy + b, color="#9ec9e8", alpha=0.7, lw=0)
        ax[0].plot(x, yy, color="#1f4e79", lw=1.4)
        ax[0].plot(x, yy - b, color="0.4", lw=0.6); ax[0].plot(x, yy + b, color="0.4", lw=0.6)
        # downstream-migrating phase marker (crest of the selected mode)
        xm = (x0_marker + c0 * t[f] * H0 * 60) % (x[-1] - x[0])   # scaled for visibility
        ax[0].axvline(xm, color="r", ls=":", lw=1.0)
        ax[0].set_ylim(-3.2 * b, 3.2 * b); ax[0].set_ylabel("y [m]")
        ax[0].set_title(f"MIGRATING meander (rigid bed, mobile bank).  growth factor "
                        f"×{amp[f]/amp[0]:.0f}   |   red dotted = downstream phase marker (c$_0$>0)",
                        fontsize=9, loc="left")
        # near-bank velocity perturbation u'_b that drives the migration (Ikeda eq 12)
        ax[1].plot(x, ub[f] / max(np.abs(ub[f]).max(), 1e-12), color="#c0504d", lw=1.0)
        ax[1].axhline(0, color="0.6", lw=0.6)
        ax[1].set_ylim(-1.3, 1.3); ax[1].set_ylabel("$u'_b$ (norm.)")
        ax[1].set_xlabel("down-valley x [m]   (outer bank erodes where $u'_b>0$; bed is RIGID)")
        fig.suptitle(f"Ikeda 1981 linear bend theory   |   $\\lambda_{{sel}}$ = {lam/(2*b):.1f} W, "
                     f"secondary-flow driven   |   t = {t[f]:.0f} (E$_0$b* units)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, wd = buf.shape[:2]
        w.append_data(buf[:h - (h % 2), :wd - (wd % 2)].copy())
        plt.close(fig)
    w.close()
    print(f"wrote {mp4} ({len(t)} frames)")


if __name__ == "__main__":
    make_dispersion()
    make_movie()
