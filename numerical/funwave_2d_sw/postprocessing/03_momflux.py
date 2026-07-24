#!/usr/bin/env python3
"""Cross-channel momentum-flux movie, in channel coordinates -- the link to the linear
(s,n) package's T_shear diagnostic.

    micromamba run -n fourcastnetv2 python postprocessing/03_momflux.py [--max-frames 1]

Projects the Cartesian velocity onto the local channel frame,

    u_s =  u t_x + v t_y ,      u_n = -u t_y + v t_x ,

removes the cross-sectional mean at each arc length s, and plots the Reynolds-type flux
u_s' u_n' together with the shear production

    T_shear = - <u_s' u_n'> d<u_s>/dn ,

integrated over the channel.  T_shear > 0 means the disturbance is EXTRACTING energy from
the mean-flow shear; T_shear <= 0 means it is losing energy to the mean flow.  In the
linear package every growing run had T_shear <= 0, which is what retracted the
"meander = vortical/Rossby wave" reading; this script measures the same sign in the
nonlinear, mobile-bed run so the two can be compared on the flow (never on the migration
rate -- the bank closures differ).

One fixed colour scale for the whole movie and both panels; every frame captions the
morphological-factor inflation.
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


def case(base):
    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    us = [p for p in sorted(glob.glob(os.path.join(base, "morph", "output", "u_*")))
          if "99999" not in p]
    if not us:
        return None
    X, Y = np.meshgrid(g["x"], g["y"], indexing="ij")
    _, s, tx, ty, _ = rm.channel_coords(X, Y, float(g["lam"]), rm.cfg_from_grid(g))
    chan = (np.abs(g["n"]) <= CFG["b"]) & (g["Zs"] > 0)
    return dict(tag=os.path.basename(base), g=g, us=us, X=X, Y=Y, s=s,
                tx=tx, ty=ty, chan=chan, n=g["n"])


def flux(c, f):
    """u_s' u_n' and the channel-integrated shear production for one snapshot."""
    u = load(c["us"][f])
    v = load(c["us"][f].replace("/u_", "/v_"))
    u_s = u * c["tx"] + v * c["ty"]
    u_n = -u * c["ty"] + v * c["tx"]
    # deviations from the cross-sectional mean at each x-column, over channel cells only
    m = c["chan"]
    def sect_mean(a):
        num = np.where(m, a, 0.0).sum(axis=1)
        den = np.maximum(m.sum(axis=1), 1)
        return (num / den)[:, None]
    usp, unp = u_s - sect_mean(u_s), u_n - sect_mean(u_n)
    fx = np.where(m, usp * unp, np.nan)
    # T_shear = -<u_s' u_n'> d<u_s>/dn, using the as-built n
    n1 = c["n"]
    with np.errstate(invalid="ignore"):
        dusdn = np.gradient(np.where(m, u_s, np.nan), axis=1) / np.gradient(n1, axis=1)
    T = -np.nansum(np.where(m, usp * unp * dusdn, 0.0)) / max(m.sum(), 1)
    return fx, float(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    cs = [c for c in (case(b) for b in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "*")))) if c]
    if not cs:
        print("no morph output yet"); return 1
    nfr = min(len(c["us"]) for c in cs)
    if args.max_frames:
        nfr = min(nfr, args.max_frames)

    # scale from the CLEAN interior: exclude the buffer->erodible transition spike near x=buffer_len
    def interior_flux(c, f):
        fx = flux(c, f)[0]
        buf = float(c["g"]["buffer_len"]); L = float(c["g"]["L"])
        return np.where((c["X"] > buf + 250) & (c["X"] < L - buf - 250), fx, np.nan)
    lim = max(np.nanpercentile(np.abs(interior_flux(c, nfr - 1)), 99.0) for c in cs)
    lim = max(lim, 1e-6)
    print(f"{len(cs)} cases, {nfr} frames, fixed scale +/-{lim:.4f} m^2/s^2 (interior; buffer spike excluded)")

    Lmax = max(float(c["g"]["L"]) for c in cs)
    bufmin = min(float(c["g"]["buffer_len"]) for c in cs)   # crop off the straight buffers (no bends,
    x0, x1 = bufmin - 100, Lmax - bufmin + 100              # so u_s'u_n' is identically 0 there)
    halves = [float(c["g"]["half"]) for c in cs]
    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, f"momflux_all_MF{CFG['Morph_factor']}.mp4")
    writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=None)
    hist = [[] for _ in cs]
    for f in range(nfr):
        fig = plt.figure(figsize=(13, 3.6 + 5.0 * sum(halves) / Lmax))
        gs = GridSpec(len(cs) + 1, 1, figure=fig,
                      height_ratios=[h / sum(halves) * 6.0 for h in halves] + [0.5],
                      hspace=0.32)
        for i, c in enumerate(cs):
            ax = fig.add_subplot(gs[i])
            fx, T = flux(c, f)
            hist[i].append(T)
            pc = ax.pcolormesh(c["X"], c["Y"], fx, cmap="RdBu_r", vmin=-lim, vmax=lim,
                               shading="auto", rasterized=True)
            ax.contour(c["X"], c["Y"], np.abs(c["n"]), levels=[CFG["b"]],
                       colors="0.4", linewidths=0.6)
            for xb in (CFG["buffer_len"], float(c["g"]["L"]) - CFG["buffer_len"]):
                ax.axvline(xb, color="0.3", ls=":", lw=0.8)
            ax.set_xlim(x0, x1); ax.set_ylim(-halves[i], halves[i])
            ax.set_aspect("equal"); ax.set_ylabel("y [m]")
            ax.set_title(f"$\\lambda$ = {float(c['g']['lam']):.0f} m   "
                         f"$T_{{shear}}$ = {T:+.3e} m$^2$s$^{{-3}}$   "
                         f"({'EXTRACTING from' if T > 0 else 'LOSING to'} the mean shear)",
                         fontsize=10, loc="left")
            if i == len(cs) - 1:
                ax.set_xlabel("down-valley x [m]")
        cax = fig.add_subplot(gs[-1])
        fig.colorbar(pc, cax=cax, orientation="horizontal",
                     label="$u_s' u_n'$ [m$^2$ s$^{-2}$]  (channel frame, deviation from "
                           "the cross-sectional mean)")
        th = f * CFG["plot_intv"]
        fig.suptitle(f"cross-channel momentum flux   |   $t_{{hydro}}$ = {th:6.0f} s   "
                     f"$t_{{morph}}$ = {th*CFG['Morph_factor']/86400:6.1f} d   "
                     f"(Morph_factor = {CFG['Morph_factor']})", fontsize=10)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, w = buf.shape[:2]                       # libx264 + yuv420p require EVEN dims
        buf = buf[:h - (h % 2), :w - (w % 2)].copy()
        writer.append_data(buf)
        plt.close(fig)
    writer.close()
    print(f"wrote {mp4}")
    for c, h in zip(cs, hist):
        sg = "T_shear > 0 throughout" if min(h) > 0 else (
             "T_shear <= 0 throughout" if max(h) <= 0 else "T_shear CHANGES SIGN")
        print(f"  {c['tag']}: {sg}   (min {min(h):+.3e}, max {max(h):+.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
