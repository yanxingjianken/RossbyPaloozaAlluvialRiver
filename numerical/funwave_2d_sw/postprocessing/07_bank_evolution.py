#!/usr/bin/env python3
"""Three-row bank-evolution movie -- the animated counterpart of the IC plot (05_ic.py).
Covers BOTH phases so the meander perturbation is seen spinning up on the rigid bed, then the
banks eroding once the sediment module switches on.  All three rows show the bank moving.

  Row 1 (plan xOy) : cross-channel MOMENTUM FLUX  u_s' u_n'  in the channel frame (the project's
                     T_shear diagnostic), + the migrating bank line + the t=0 bank line.
  Row 2 (plan xOy) : total VELOCITY |U| with the total-velocity vectors, + the same bank lines.
  Row 3 (two yOz)  : transverse sections at the S1/S2 apexes -- current bed (brown) vs the initial
                     bed (grey dashed) = bank retreat, the FREE-SURFACE elevation (water), and the
                     total down-channel velocity u_s(n) (red, twin axis).

The bank has no wet/dry line to trace (the shelf is always wet, h_plain=0.2 m), so the plan-view
bank is the current-bed iso-contour at mid bank-face depth (H_b+h_plain)/2; it slides sideways as
the bank erodes.  One movie per wavelength: bank_evolution_lam{lam:.0f}.mp4.

    micromamba run -n fourcastnetv2 python postprocessing/07_bank_evolution.py [--max-frames N]
"""
import argparse
import glob
import os
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import run_meander as rm  # noqa: E402

CFG = rm.CONFIG
OUT = os.path.join(ROOT, "figures")


def load(p):
    return np.loadtxt(p).T


def phase_u(base, ph):
    return [p for p in sorted(glob.glob(os.path.join(base, ph, "output", "u_*"))) if "99999" not in p]


def sibling(u_path, var):
    """Path of another field (v/eta/dep) at the same time-stamp as this u_ snapshot."""
    return u_path.replace(os.sep + "u_", os.sep + var + "_")


def sect_mean(a, m):
    """Cross-sectional (per-x-column) mean over channel cells only, broadcast back."""
    num = np.where(m, a, 0.0).sum(axis=1)
    den = np.maximum(m.sum(axis=1), 1)
    return (num / den)[:, None]


def momflux(u, v, tx, ty, chan):
    """Channel-frame Reynolds flux u_s' u_n' (deviation from the cross-sectional mean)."""
    u_s = u * tx + v * ty
    u_n = -u * ty + v * tx
    usp = u_s - sect_mean(u_s, chan)
    unp = u_n - sect_mean(u_n, chan)
    return np.where(chan, usp * unp, np.nan)


def build_frames(base):
    """[(phase, u_path)] for spin-up then morph; the bed is rigid in spin-up, mobile in morph."""
    return ([("spin-up", p) for p in phase_u(base, "spinup")]
            + [("morph", p) for p in phase_u(base, "morph")])


def _plot(base, args):
    tag = os.path.basename(base)
    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    lam = float(g["lam"]); L = float(g["L"]); half = float(g["half"])
    buf = float(g["buffer_len"]); b = CFG["b"]
    x, y, n, kap = g["x"], g["y"], g["n"], g["kappa"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    cg = rm.cfg_from_grid(g)
    n_cc, s_cc, tx, ty, kap_cc = rm.channel_coords(X, Y, lam, cg)
    chan = (np.abs(g["n"]) <= b) & (g["Zs"] > 0)
    bed0 = load(os.path.join(base, "bathy", "depth.txt"))          # rigid bed (spin-up)
    bank_lvl = 0.5 * (CFG["H_b"] + CFG["h_plain"])                 # mid bank-face iso-depth

    frames = build_frames(base)
    if not frames:
        print(f"no flow frames for {tag}"); return
    if args.max_frames:
        frames = frames[:args.max_frames]
    isp = sum(1 for f in frames if f[0] == "spin-up")

    # fixed scales from a late frame
    ul = load(frames[-1][1]); vl = load(sibling(frames[-1][1], "v"))
    smax = float(np.nanpercentile(np.hypot(ul, vl), 99.5))
    interior = (X > buf + 250) & (X < L - buf - 250)      # drop the buffer->erodible transition spike
    flim = float(np.nanpercentile(np.abs(momflux(ul, vl, tx, ty, chan))[interior], 99.0))
    flim = max(flim, 1e-6)
    zmin = -float(np.nanmax(bed0)) - 0.3; zmax = 0.6
    toe = b + CFG["m_bank"] * (CFG["H_b"] - CFG["h_plain"]) + 8.0
    xlo, xhi = buf - 100, L - buf + 100                   # crop plan rows to the meandering reach

    # sections (same x that 02/04 mark)
    secs = []
    for k, xs in enumerate(rm.section_x(lam, cg)):
        i = int(np.argmin(np.abs(x - xs)))
        ks = float(np.sign(kap_cc[i, np.argmax(np.abs(kap_cc[i]))]))
        secs.append(dict(label=f"S{k+1}", x=float(xs), i=i, ksign=ks))

    os.makedirs(OUT, exist_ok=True)
    # lam+curvature label: B2 and B3 share lam=1560 (fix-C0 vs fix-A), so lam alone would collide
    label = "_".join(tag.split("_")[:2])
    mp4 = os.path.join(OUT, f"bank_evolution_{label}.mp4")
    w = imageio.get_writer(mp4, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
    st = max(1, int(round(len(x) / 80)))          # quiver stride down-valley
    sty = max(1, int(round(len(y) / 26)))         # quiver stride cross-valley
    dt = float(CFG["plot_intv"]); MF = CFG["Morph_factor"]

    for kf, (ph, up) in enumerate(frames):
        u = load(up); v = load(sibling(up, "v")); eta = load(sibling(up, "eta"))
        spd = np.hypot(u, v)
        bed = load(sibling(up, "dep")) if (ph == "morph" and os.path.exists(sibling(up, "dep"))) else bed0
        fx = momflux(u, v, tx, ty, chan)

        fig = plt.figure(figsize=(14, 11.0), constrained_layout=True)
        gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.0, 1.0, 1.25])

        # -- Row 1: momentum flux (plan) --------------------------------------
        ax0 = fig.add_subplot(gs[0, :])
        pc0 = ax0.pcolormesh(X, Y, fx, cmap="RdBu_r", vmin=-flim, vmax=flim, shading="auto")
        _bank_lines(ax0, X, Y, bed, np.abs(n), bank_lvl, b)
        _decor(ax0, buf, L, half, xlo, xhi, y_lab=True)
        fig.colorbar(pc0, ax=ax0, label="$u_s' u_n'$  [m$^2$ s$^{-2}$]  (channel frame)")
        ax0.set_title("cross-channel MOMENTUM FLUX  (yellow = migrating bank, grey = $t{=}0$ bank)",
                      fontsize=9, loc="left")

        # -- Row 2: total speed + vectors (plan) ------------------------------
        ax1 = fig.add_subplot(gs[1, :])
        pc1 = ax1.pcolormesh(X, Y, np.where(spd > 1e-6, spd, np.nan), cmap="viridis",
                             vmin=0, vmax=smax, shading="auto")
        ax1.quiver(X[::st, ::sty], Y[::st, ::sty], u[::st, ::sty], v[::st, ::sty],
                   scale=smax * 22, width=0.0013, color="w")
        _bank_lines(ax1, X, Y, bed, np.abs(n), bank_lvl, b)
        _decor(ax1, buf, L, half, xlo, xhi, y_lab=True)
        fig.colorbar(pc1, ax=ax1, label="total $|U|$  [m/s]")
        ax1.set_title("total VELOCITY  $|U|$  + velocity vectors", fontsize=9, loc="left")
        ax1.set_xlabel("down-valley x [m]")

        # -- Row 3: FROUDE number (plan) --------------------------------------
        # FroudeCap was REMOVED in v2, so the toe is free to go supercritical.  In v1 the cap
        # pinned 41 bank-toe cells at Fr = 1.000 exactly; this panel makes that behaviour visible
        # instead of hidden.  Diverging map centred on Fr = 1 so critical flow reads at a glance.
        axF = fig.add_subplot(gs[2, :])
        Htot = np.maximum(bed + eta, 1e-6)
        Fr = np.where(spd > 1e-6, spd / np.sqrt(9.81 * Htot), np.nan)
        pcF = axF.pcolormesh(X, Y, Fr, cmap="coolwarm", vmin=0.0, vmax=2.0, shading="auto")
        axF.contour(X, Y, np.nan_to_num(Fr), levels=[1.0], colors="k", linewidths=1.2)  # critical
        _bank_lines(axF, X, Y, bed, np.abs(n), bank_lvl, b)
        _decor(axF, buf, L, half, xlo, xhi, y_lab=True)
        fig.colorbar(pcF, ax=axF, label="Froude  $|U|/\\sqrt{gH}$")
        n_sup = int(np.nansum(Fr > 1.0))
        axF.set_title(f"FROUDE number  (black = critical Fr=1; no FroudeCap in v2)   "
                      f"max Fr = {np.nanmax(Fr):.2f},  supercritical cells = {n_sup}",
                      fontsize=9, loc="left")

        # -- Row 4: yOz sections ----------------------------------------------
        u_s = u * tx + v * ty
        for ci, sc in enumerate(secs[:2]):
            ax = fig.add_subplot(gs[3, ci])
            i = sc["i"]; ks = sc["ksign"]
            sel = np.abs(n[i]) <= toe
            nn = (n[i] * ks)[sel]; o = np.argsort(nn); nn = nn[o]
            zb = (-bed[i])[sel][o]; zb0 = (-bed0[i])[sel][o]
            sf = np.maximum(eta[i][sel][o], zb)               # water surface (dry -> bed)
            us = u_s[i][sel][o]
            ax.fill_between(nn, zb, sf, color="#8fbfe0", zorder=1)          # water
            ax.fill_between(nn, zmin, zb, color="#c8a06a", zorder=2)        # bed
            # SHADE the bank change so it is unmistakable: red = eroded away (bed dropped),
            # green = deposited (bed rose) -- the band between the initial and current bed.
            ax.fill_between(nn, zb, zb0, where=(zb < zb0), color="#e23b3b", alpha=0.8,
                            zorder=3, label="erosion")
            ax.fill_between(nn, zb0, zb, where=(zb > zb0), color="#2ca02c", alpha=0.8,
                            zorder=3, label="deposition")
            ax.plot(nn, zb0, color="0.25", lw=1.1, ls="--", zorder=4)       # initial bed
            ax.plot(nn, zb, color="#5a3c1e", lw=1.4, zorder=5)             # current bed
            ax.plot(nn, sf, color="#1f6fb2", lw=0.9, zorder=6)            # free surface
            ax.axvline(0, color="0.6", lw=0.6, ls=":")
            ax.set_xlim(nn.min(), nn.max()); ax.set_ylim(zmin, zmax)
            ax2 = ax.twinx(); ax2.plot(nn, us, "r-", lw=1.2, zorder=6)
            ax2.set_ylim(0, max(1.2, smax * 1.05)); ax2.tick_params(axis="y", colors="r")
            if ci == 1:
                ax2.set_ylabel("total $u_s(n)$ [m/s]", color="r")
            ax.set_title(f"yOz {sc['label']} @ x={sc['x']:.0f} m  "
                         "(red=eroded, green=deposited vs $t{=}0$; blue=surface, red line=$u_s$)",
                         fontsize=8)
            ax.set_xlabel("n·sgn($\\kappa$) [m]   (<0 outer · >0 inner)")
            if ci == 0:
                ax.set_ylabel("elevation [m]")

        th = (kf if ph == "spin-up" else kf - isp) * dt
        tm = th * MF / 86400.0 if ph == "morph" else 0.0
        fig.suptitle(f"BANK MIGRATION — {tag[:34]}   |   $\\lambda$={lam:.0f} m   "
                     f"phase: {ph}   $t_{{hydro}}$={th:6.0f} s"
                     + (f"   $t_{{morph}}$={tm:5.2f} d (MF={MF})" if ph == "morph"
                        else "   (rigid bed: flow spins up)")
                     + "   |   gap-1 $A$=2.89 ON", fontsize=10)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, wd = rgb.shape[:2]
        w.append_data(rgb[:h - (h % 2), :wd - (wd % 2)].copy())
        plt.close(fig)
        if (kf + 1) % 10 == 0 or kf == len(frames) - 1:
            print(f"  {tag}: frame {kf+1}/{len(frames)}")
    w.close()
    print(f"wrote {mp4}  ({len(frames)} frames: {isp} spin-up + {len(frames)-isp} morph)")


def _bank_lines(ax, X, Y, bed, absn, lvl, b):
    """Migrating bank (current bed iso-depth) + static t=0 bank edge |n|=b."""
    ax.contour(X, Y, bed, levels=[lvl], colors="#ffd21e", linewidths=1.3)     # moves with erosion
    ax.contour(X, Y, absn, levels=[b], colors="0.45", linewidths=0.6)          # t=0 reference


def _decor(ax, buf, L, half, xlo, xhi, y_lab=False):
    for xb in (buf, L - buf):
        ax.axvline(xb, color="r", ls=":", lw=0.7)
    ax.set_xlim(xlo, xhi); ax.set_ylim(-half, half)
    if y_lab:
        ax.set_ylabel("y [m]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--match", default="", help="only plot run tags containing this substring")
    args = ap.parse_args()
    bases = sorted(glob.glob(os.path.join(ROOT, "runs", "*")))
    bases = [b for b in bases if args.match in os.path.basename(b)]
    if not bases:
        print("no matching runs/"); return 1
    for base in bases:
        _plot(base, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
