#!/usr/bin/env python3
"""4-row bank-evolution movie, one mp4 per run.

Rows, per the specification:
  1  total speed |u| + velocity vector arrows (xOy plan, migrating banks drawn)
  2  anomaly speed |u'| + arrows,  u' = u - ubar mapped to the current geometry
  3  cross-channel momentum flux u'_s u'_n in the channel frame
  4  yOz section at a fixed x (bend apex): bed/banks/free surface, t=0 banks grey
  5  ZOOMED xOy of ONE bank over one bend, y-axis stretched hard so the bank
     MIGRATION is actually visible (rows 4 and 5 are the two migration views:
     yOz cross-section, and a zoomed-in plan of a single bank)

All plot axes are kept the SAME WIDTH (colorbars live in reserved slots) so the
rows line up.  Covers spin-up AND the morphological phase.

Two gotchas carried over from the FUNWAVE work in this repo:
  * colour limits come from the CLEAN INTERIOR only -- a spike at the
    erodible/rigid interface otherwise saturates every scale and washes the
    signal out;
  * frames are cropped to EVEN pixel dimensions (libx264 yuv420p requires it).

    python postprocessing/02_bank_evolution.py [m4|m8]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_lib as pp  # noqa: E402
import geometry as geo  # noqa: E402

plt = pp.set_style()


def load_frames(tag: str):
    patt = os.path.join(pp.OUT_DIR, f"run_{tag}_f*.npz")
    files = sorted(glob.glob(patt))
    if not files:
        raise SystemExit(f"no frames matching {patt} -- run meander_thetis.py first")
    return files


def grid_geometry(F, d):
    """Physical (X, Y) of the fixed reference output grid, plus the frame."""
    xo, no = F["xo"], F["no"]
    x_ref, yN, yS = F["x_ref"], F["yN"], F["yS"]
    cx = np.interp(xo, x_ref, geo.centreline(yN, yS))
    bx = np.interp(xo, x_ref, geo.half_width(yN, yS))
    X = np.repeat(xo[None, :], no.size, axis=0)
    Y = cx[None, :] + no[:, None] * bx[None, :]
    dcdx = np.gradient(geo.centreline(yN, yS), x_ref, edge_order=2)
    slope = np.interp(xo, x_ref, dcdx)[None, :] * np.ones_like(X)
    tn = np.sqrt(1.0 + slope**2)
    return X, Y, (1.0 / tn, slope / tn), (-slope / tn, 1.0 / tn)


def channel_decompose(F, d):
    """u' and its (s, n) components on the fixed output grid."""
    X, Y, (tx, ty), (nx_, ny_) = grid_geometry(F, d)
    u, v = F["u"], F["v"]
    ub = geo.base_velocity(F["no"], d)[:, None] * np.ones_like(u)
    up = u - ub * tx
    vp = v - ub * ty
    return X, Y, up, vp, up * tx + vp * ty, up * nx_ + vp * ny_


def make_movie(tag: str, m: int, fps: int = 12):
    files = load_frames(tag)
    d = geo.build_design(geo.Config(n_wave=m))
    f0 = np.load(files[0], allow_pickle=True)
    x_ref = f0["x_ref"]
    yN0, yS0 = f0["yN"], f0["yS"]
    x_sec = _apex_x(d, m)

    # ---- fixed colour limits from the CLEAN INTERIOR of a late frame -----
    probe = np.load(files[min(len(files) - 1, int(0.8 * len(files)))], allow_pickle=True)
    Xp, Yp, up_p, vp_p, usp, unp = channel_decompose(probe, d)
    keep = pp.interior_mask(Xp, d)                       # clean interior ONLY
    spd_max = float(np.nanpercentile(np.hypot(probe["u"], probe["v"])[keep], 99.5))
    ano_max = max(float(np.nanpercentile(np.hypot(up_p, vp_p)[keep], 99.5)), 1e-9)
    flx_lo, flx_hi = pp.sym_limits((usp * unp)[keep], 99.0)

    frames = []
    for i, fn in enumerate(files):
        F = np.load(fn, allow_pickle=True)
        yN, yS = F["yN"], F["yS"]
        X, Y, up, vp, us_p, un_p = channel_decompose(F, d)
        u, v = F["u"], F["v"]

        fig = plt.figure(figsize=(15.0, 13.0))
        # 5 plot rows + a narrow shared colorbar column, so EVERY plot axis has
        # exactly the same width and the rows line up.  No set_aspect (which
        # would fight the fixed column width); the y exaggeration is emergent
        # from the wide-and-short cell shape and reported per panel.
        gs = fig.add_gridspec(5, 2, width_ratios=[1.0, 0.018],
                              height_ratios=[1, 1, 1, 1.05, 1.05],
                              hspace=0.62, wspace=0.02)
        pax = [fig.add_subplot(gs[r, 0]) for r in range(5)]
        cax = [fig.add_subplot(gs[r, 1]) for r in range(5)]
        xz0, xz1 = d.x_m0, d.x_m1              # show the meander reach

        def _ystretch(ax):
            ax.figure.canvas.draw()
            bb = ax.get_window_extent()
            dx = (ax.get_xlim()[1] - ax.get_xlim()[0])
            dy = (ax.get_ylim()[1] - ax.get_ylim()[0])
            return (bb.height / dy) / (bb.width / dx)

        def plan(ax, cx, field, cmap, lo, hi, label, arrows=None):
            tc = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=lo, vmax=hi,
                               shading="gouraud")
            if arrows is not None:
                sx, sn = max(1, X.shape[1] // 55), max(1, X.shape[0] // 7)
                ax.quiver(X[::sn, ::sx], Y[::sn, ::sx],
                          arrows[0][::sn, ::sx], arrows[1][::sn, ::sx],
                          color="0.12", width=0.0016, alpha=0.8)
            ax.plot(x_ref, yN, "k-", lw=1.5)
            ax.plot(x_ref, yS, "k-", lw=1.5)
            ax.plot(x_ref, yN0, color="0.55", lw=0.9, ls="--")
            ax.plot(x_ref, yS0, color="0.55", lw=0.9, ls="--")
            ax.axvline(x_sec, color="crimson", lw=1.2, alpha=0.8)
            ax.set_xlim(xz0, xz1)
            ax.set_ylabel("y [m]")
            ax.grid(False)
            cb = fig.colorbar(tc, cax=cx)
            cb.set_label(label)

        ax1 = pax[0]
        plan(ax1, cax[0], np.hypot(u, v), "viridis", 0.0, spd_max,
             r"$|\mathbf{u}|$ [m/s]", arrows=(u, v))
        ax1.set_title(r"(1) total speed $|\mathbf{u}|$ + velocity vectors "
                      f"[y stretched x{_ystretch(ax1):.0f}]")

        ax2 = pax[1]
        plan(ax2, cax[1], np.hypot(up, vp), "magma", 0.0, ano_max,
             r"$|\mathbf{u}'|$ [m/s]", arrows=(up, vp))
        ax2.set_title(r"(2) anomaly speed $|\mathbf{u}'|$ + anomaly vectors "
                      r"($\mathbf{u}' = \mathbf{u} - \bar u\,\hat{s}$)")

        ax3 = pax[2]
        plan(ax3, cax[2], us_p * un_p, "RdBu_r", flx_lo, flx_hi,
             r"$u'_s u'_n$ [m$^2$/s$^2$]")
        ax3.set_title(r"(3) cross-channel momentum flux $u'_s u'_n$ "
                      "(channel frame)")
        ax3.set_xlabel("x [m]")

        # ---------------- row 4: yOz at the apex ---------------------------
        ax4 = pax[3]
        nfine = np.linspace(-1, 1, 401)
        eta_ref = float(d.I * x_sec)
        zb = geo.bed_elevation(x_sec, nfine, d, eta_ref=eta_ref)
        cN = float(np.interp(x_sec, x_ref, yN))
        cS = float(np.interp(x_sec, x_ref, yS))
        cN0 = float(np.interp(x_sec, x_ref, yN0))
        cS0 = float(np.interp(x_sec, x_ref, yS0))
        n_now = np.linspace(cS, cN, nfine.size)

        # local free surface from the solution
        j = int(np.argmin(np.abs(F["xo"] - x_sec)))
        eta_plot = float(np.nanmean(F["elev"][:, j])
                         - geo.base_elevation(x_sec, d))

        ax4.fill_between(n_now, zb.min() - 0.3, zb, color="#c8a980")
        ax4.fill_between(n_now, zb, eta_plot, color="#cfe3f7")
        ax4.plot(n_now, zb, color="#6b4c1e", lw=2.2, label="bed (frozen)")
        ax4.axhline(eta_plot, color="#1f6fb4", lw=2.0, label="free surface")
        for yv, st, lab in ((cN, "-", "bank now"), (cS, "-", None),
                           (cN0, "--", "bank t=0"), (cS0, "--", None)):
            ax4.axvline(yv, color="k" if st == "-" else "0.55",
                        lw=2.2 if st == "-" else 1.4, ls=st, label=lab)
        ax4.set_xlim(min(cS, cS0) - 3, max(cN, cN0) + 3)
        ax4.set_ylim(zb.min() - 0.3, 0.4)
        ax4.set_xlabel("y [m]  (transverse)")
        ax4.set_ylabel("elev. rel. local\nsurface [m]")
        ax4.legend(loc="lower center", ncol=4, fontsize=9)
        ax4.set_title(f"(4) yOz section at x = {x_sec:.0f} m:  bank migration  "
                      f"(N {cN - cN0:+.3f} m,  S {cS - cS0:+.3f} m,  "
                      f"width {cN - cS:.2f} m vs {cN0 - cS0:.2f} m at t=0)")
        cax[3].set_visible(False)

        # ---------------- row 5: ZOOMED plan of ONE bank -------------------
        # A single bend of the NORTH bank, y-axis stretched hard so the
        # migration (order 0.1-0.4 m on a 35 m channel) is actually visible.
        ax5 = pax[4]
        lam_m = d.L_m / m
        wx0, wx1 = x_sec - 0.5 * lam_m, x_sec + 0.5 * lam_m
        sel = (F["xo"] >= wx0) & (F["xo"] <= wx1)
        xw = F["xo"][sel]
        # near-bank anomaly-speed field, north half only, as background
        half = F["no"] >= 0.0
        ano = np.hypot(up, vp)[np.ix_(half, sel)]
        Yw = Y[np.ix_(half, sel)]
        Xw = X[np.ix_(half, sel)]
        yNw = np.interp(xw, x_ref, yN)
        # The output grid stops at n = 0.97 b (deliberate: the DG field's wall
        # value is a trace).  On this hard-stretched y-axis that 0.5 m leaves a
        # visible white strip between the colour and the bank line, so extend
        # the near-bank |u'| the last 3% up to the bank (near-bank value holds
        # to the wall).
        Xw = np.vstack([Xw, xw[None, :]])
        Yw = np.vstack([Yw, yNw[None, :]])
        ano = np.vstack([ano, ano[-1:, :]])
        pc = ax5.pcolormesh(Xw, Yw, ano, cmap="magma", vmin=0.0, vmax=ano_max,
                            shading="gouraud")
        yNw0 = np.interp(xw, x_ref, yN0)
        ax5.fill_between(xw, yNw0, yNw, color="#d62728", alpha=0.35,
                         label="migrated strip")
        ax5.plot(xw, yNw0, color="0.35", lw=1.6, ls="--", label="bank t=0")
        ax5.plot(xw, yNw, color="k", lw=2.4, label="bank now")
        ax5.axvline(x_sec, color="crimson", lw=1.2, alpha=0.8)
        # y-limits: tight band around the two bank curves, auto-padded
        ylo = min(yNw.min(), yNw0.min())
        yhi = max(yNw.max(), yNw0.max())
        pad = max(0.15, 0.6 * (yhi - ylo))
        ax5.set_ylim(ylo - pad, yhi + pad)
        ax5.set_xlim(wx0, wx1)
        yzoom = (wx1 - wx0) / (yhi - ylo + 2 * pad)
        ax5.set_xlabel("x [m]")
        ax5.set_ylabel("y [m]  (NORTH bank)")
        ax5.legend(loc="lower right", ncol=3, fontsize=8.5)
        mig_apex = float(np.interp(x_sec, xw, yNw) - np.interp(x_sec, xw, yNw0))
        ax5.set_title(f"(5) zoomed plan, N bank, one bend [y stretched "
                      f"x{yzoom:.0f}]:  apex migration {mig_apex:+.3f} m  "
                      f"(red = eroded/deposited strip, colour |u'|)", fontsize=11)
        cb5 = fig.colorbar(pc, cax=cax[4])
        cb5.set_label(r"$|\mathbf{u}'|$ [m/s]")

        phase = str(F["phase"])
        fig.suptitle(f"m={m}   t = {float(F['t']):.0f} s   phase: {phase}   "
                     f"frame {i + 1}/{len(files)}", fontsize=13, y=0.995)

        frames.append(pp.even_crop(pp.fig_to_rgb(fig)))
        plt.close(fig)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(files)} frames")

    pp.write_mp4(frames, f"bank_evolution_{tag}", fps=fps)


def _apex_x(d: geo.Design, m: int) -> float:
    return d.x_m0 + (m // 2 + 0.25) * (d.L_m / m)


def main():
    args = sys.argv[1:]
    if args and args[0].startswith("A"):        # first arg may be the case
        pp.set_case(args.pop(0))
    tags = args or ["m4", "m8"]
    for tag in tags:
        m = int(tag.lstrip("m"))
        print(f"  building movie for {tag} ...")
        make_movie(tag, m)


if __name__ == "__main__":
    main()
