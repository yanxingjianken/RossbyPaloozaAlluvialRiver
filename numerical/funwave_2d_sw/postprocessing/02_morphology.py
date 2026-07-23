#!/usr/bin/env python3
"""Side-by-side morphology movie for the two bank-wavenumber cases.

    micromamba run -n fourcastnetv2 python postprocessing/02_morphology.py
    micromamba run -n fourcastnetv2 python postprocessing/02_morphology.py --max-frames 1

ONE movie, not two: the comparison IS the result, and the two runs have different
morphological clocks, so leaving the reader to align two separate files invites error.

House rules: a single fixed colour scale for the whole movie and BOTH panels, true units,
no per-frame normalisation, and every frame captions the morphological-factor inflation so
it is never invisible.  Both panels are drawn at the SAME metres-per-inch, so B2's shorter
reach really does look shorter.
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


def case(base):
    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    dep = sorted(glob.glob(os.path.join(base, "morph", "output", "dep_*")))
    msk = sorted(glob.glob(os.path.join(base, "morph", "output", "mask_*")))
    if not dep:
        return None
    X, Y = np.meshgrid(g["x"], g["y"], indexing="ij")
    n, _, _, _, kap = rm.channel_coords(X, Y, float(g["lam"]), CFG)
    return dict(tag=os.path.basename(base), g=g, dep=dep, msk=msk, X=X, Y=Y,
                n=n, kap=kap, lam=float(g["lam"]), A=float(g["A"]),
                sinu=float(g["sinuosity"]), L=float(g["L"]))


def load(p):
    return np.loadtxt(p).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0, help="smoke-test with N frames")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    cs = [c for c in (case(b) for b in sorted(glob.glob(os.path.join(ROOT, "runs", "*")))) if c]
    if not cs:
        print("no morph output yet"); return 1
    nfr = min(len(c["dep"]) for c in cs)
    if args.max_frames:
        nfr = min(nfr, args.max_frames)
    print(f"{len(cs)} cases, {nfr} common frames")

    # bed elevation change z_b = -Depth, so dz_b = -(Depth_t - Depth_0):
    #   dz_b < 0 erosion,  > 0 deposition
    base0 = [load(c["dep"][0]) for c in cs]
    last = [-(load(c["dep"][nfr - 1]) - b0) for c, b0 in zip(cs, base0)]
    lim = max(np.percentile(np.abs(d), 99.5) for d in last)
    lim = max(lim, 1e-3)
    print(f"fixed colour limit +/-{lim:.3f} m (99.5th percentile of the final frames)")

    Lmax = max(c["L"] for c in cs)
    halves = [float(c["g"]["half"]) for c in cs]
    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, f"morph_AB_C{CFG['C0']*1e3:.2f}e-3_MF{CFG['Morph_factor']}.mp4")
    writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=None)

    for f in range(nfr):
        fig = plt.figure(figsize=(13, 3.2 + 5.0 * sum(halves) / Lmax))
        gs = GridSpec(len(cs) + 1, 1, figure=fig,
                      height_ratios=[h / sum(halves) * 6.0 for h in halves] + [0.35],
                      hspace=0.28)
        for i, c in enumerate(cs):
            ax = fig.add_subplot(gs[i])
            dz = -(load(c["dep"][f]) - base0[i])
            pc = ax.pcolormesh(c["X"], c["Y"], dz, cmap="RdBu_r", vmin=-lim, vmax=lim,
                               shading="auto", rasterized=True)
            # initial banks (grey) and the current waterline (black)
            ax.contour(c["X"], c["Y"], np.abs(c["n"]), levels=[CFG["b"]],
                       colors="0.45", linewidths=0.6)
            if c["msk"] and f < len(c["msk"]):
                ax.contour(c["X"], c["Y"], load(c["msk"][f]), levels=[0.5],
                           colors="k", linewidths=0.8)
            # non-erodible buffer
            for xb in (c["lam"], c["L"] - c["lam"]):
                ax.axvline(xb, color="0.3", ls=":", lw=0.8)
            ax.set_xlim(0, Lmax); ax.set_ylim(-halves[i], halves[i])
            ax.set_aspect("equal")
            ax.set_ylabel("y [m]")
            ax.set_title(f"$\\lambda$ = {c['lam']:.0f} m,  A = {c['A']:.0f} m "
                         f"= {c['A']/(2*CFG['b']):.2f} W,  sinuosity {c['sinu']:.2f}"
                         f"   (A$k^2$ = {CFG['C0']:.3e} m$^{{-1}}$, same for both)",
                         fontsize=10, loc="left")
            if i == len(cs) - 1:
                ax.set_xlabel("down-valley x [m]")
        cax = fig.add_subplot(gs[-1])
        fig.colorbar(pc, cax=cax, orientation="horizontal",
                     label="bed elevation change $\\Delta z_b$ [m]   "
                           "(+ deposition, $-$ erosion)")
        th = f * CFG["plot_intv"]
        tm = th * CFG["Morph_factor"]
        closure = (f"Ikeda-1981 secondary-flow closure ON ($A$ = {CFG.get('A_ikeda', 2.89)}, "
                   f"$C_d^{{eff}} = C_d(1+A\\kappa n)$)" if CFG.get("SecondaryFlow")
                   else "stock FUNWAVE: no secondary-flow closure")
        fig.suptitle(f"FUNWAVE-TVD  NSWE + Exner + avalanching   |   "
                     f"$t_{{hydro}}$ = {th:6.0f} s   "
                     f"$t_{{morph}}$ = {tm/86400:6.1f} d  "
                     f"(Morph_factor = {CFG['Morph_factor']}, {closure})", fontsize=10)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, w = buf.shape[:2]                       # libx264 + yuv420p require EVEN dims;
        buf = buf[:h - (h % 2), :w - (w % 2)].copy()   # a matplotlib canvas is often odd
        writer.append_data(buf)
        plt.close(fig)
        if (f + 1) % 10 == 0 or f == nfr - 1:
            print(f"  frame {f+1}/{nfr}")
    writer.close()
    print(f"wrote {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
