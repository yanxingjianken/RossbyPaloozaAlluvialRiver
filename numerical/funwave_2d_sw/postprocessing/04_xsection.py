#!/usr/bin/env python3
"""yOz cross-section movie at the bend apexes marked (grey dashed, S1/S2) on the xOy morphology
movie.  One panel per (case, section): the transverse bed profile z_b(y) = -Depth and the water
column, evolving over morphological time.  A point bar shows as the bed rising on the INNER bank
(shallow) and scouring on the OUTER bank (deep); stock FUNWAVE does the opposite (inner scour).

    micromamba run -n fourcastnetv2 python postprocessing/04_xsection.py
    micromamba run -n fourcastnetv2 python postprocessing/04_xsection.py --max-frames 1

Sections come from run_meander.section_x -- the SAME x the morphology movie marks, so the two
figures always refer to the same place.
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
    dep = sorted(glob.glob(os.path.join(base, "morph", "output", "dep_*")))
    dep = [p for p in dep if "99999" not in p]
    eta = sorted(glob.glob(os.path.join(base, "morph", "output", "eta_*")))
    eta = [p for p in eta if "99999" not in p]
    if not dep:
        return None
    x = g["x"]; y = g["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    cg = rm.cfg_from_grid(g)
    n, _, _, _, kap = rm.channel_coords(X, Y, float(g["lam"]), cg)
    secs = []
    for k, xs in enumerate(rm.section_x(float(g["lam"]), cg)):
        i = int(np.argmin(np.abs(x - xs)))
        # inner bank is n*sign(kappa) > 0 (test_bathy 7b); use the apex column's kappa sign
        ks = float(np.sign(kap[i, np.argmax(np.abs(kap[i]))]))
        secs.append(dict(label=f"S{k+1}", x=float(xs), i=i, ksign=ks,
                         n=n[i], y=y, nn=n[i] * ks))
    return dict(tag=os.path.basename(base), g=g, dep=dep, eta=eta, secs=secs,
                lam=float(g["lam"]), A=float(g["A"]), sinu=float(g["sinuosity"]),
                Depth0=load(dep[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    cs = [c for c in (case(b) for b in sorted(glob.glob(os.path.join(ROOT, "runs", "*")))) if c]
    if not cs:
        print("no morph output yet"); return 1
    nfr = min(len(c["dep"]) for c in cs)
    if args.max_frames:
        nfr = min(nfr, args.max_frames)
    nsec = max(len(c["secs"]) for c in cs)
    print(f"{len(cs)} cases x {nsec} sections, {nfr} common frames")

    # fixed transverse window (|n| out to the bank toe + a little shelf) and a fixed elevation
    # scale from the initial bed, shared across the whole movie
    b = CFG["b"]; toe = b + CFG["m_bank"] * (CFG["H_b"] - CFG["h_plain"]) + 8.0
    zmin = min(-float(np.nanmax(c["Depth0"])) for c in cs) - 0.3
    zmax = 0.6

    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, f"xsection_all_MF{CFG['Morph_factor']}.mp4")
    writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=None)

    for f in range(nfr):
        fig, axes = plt.subplots(len(cs), nsec, figsize=(5.2 * nsec, 3.0 * len(cs)),
                                 squeeze=False)
        for ri, c in enumerate(cs):
            Depth = load(c["dep"][min(f, len(c["dep"]) - 1)])
            Eta = load(c["eta"][min(f, len(c["eta"]) - 1)]) if c["eta"] else np.zeros_like(Depth)
            for ci in range(nsec):
                ax = axes[ri][ci]
                if ci >= len(c["secs"]):
                    ax.axis("off"); continue
                s = c["secs"][ci]; i = s["i"]
                sel = np.abs(s["n"]) <= toe
                nn = s["nn"][sel]                       # transverse coord, >0 inner
                order = np.argsort(nn)
                nn = nn[order]
                zb = -Depth[i][sel][order]              # bed elevation
                zb0 = -c["Depth0"][i][sel][order]       # initial bed (grey)
                surf = Eta[i][sel][order]
                surf = np.maximum(surf, zb)             # dry cells: no water
                ax.fill_between(nn, zb, surf, color="#8fbfe0", zorder=1)      # water
                ax.fill_between(nn, zmin, zb, color="#c8a06a", zorder=2)      # bed
                # bank change vs t=0: red = eroded, green = deposited
                ax.fill_between(nn, zb, zb0, where=(zb < zb0), color="#e23b3b", alpha=0.8, zorder=3)
                ax.fill_between(nn, zb0, zb, where=(zb > zb0), color="#2ca02c", alpha=0.8, zorder=3)
                ax.plot(nn, zb0, color="0.25", lw=1.0, ls="--", zorder=4)     # initial bed
                ax.plot(nn, zb, color="#5a3c1e", lw=1.3, zorder=5)
                ax.axvline(0, color="0.6", lw=0.6, ls=":")
                ax.set_xlim(nn.min(), nn.max()); ax.set_ylim(zmin, zmax)
                ax.set_title(f"{c['tag'][:16]}  {s['label']} @ x={s['x']:.0f} m", fontsize=9)
                if ci == 0:
                    ax.set_ylabel("elevation [m]")
                if ri == len(cs) - 1:
                    ax.set_xlabel("transverse  n·sgn($\\kappa$) [m]   (<0 outer  ·  >0 inner)")
                ax.text(0.02, 0.06, "OUTER", transform=ax.transAxes, fontsize=7, color="0.3")
                ax.text(0.98, 0.06, "INNER", transform=ax.transAxes, fontsize=7, color="0.3",
                        ha="right")
        th = f * CFG["plot_intv"]; tm = th * CFG["Morph_factor"]
        fig.suptitle(f"transverse (yOz) cross-sections at the apexes marked on the plan view   |   "
                     f"$t_{{morph}}$ = {tm/86400:5.2f} d   (MF = {CFG['Morph_factor']})",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, w = buf.shape[:2]
        writer.append_data(buf[:h - (h % 2), :w - (w % 2)].copy())
        plt.close(fig)
        if (f + 1) % 10 == 0 or f == nfr - 1:
            print(f"  frame {f+1}/{nfr}")
    writer.close()
    print(f"wrote {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
