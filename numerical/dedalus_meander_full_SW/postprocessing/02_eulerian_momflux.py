#!/usr/bin/env python3
"""02: the single fully-Eulerian momentum-flux movie.

Maps the (s,n) channel-fitted fields back to the LAB-frame meandering channel
(so the two banks n=+/-b are the exact edges of the coloured flow -- banks bound
the flow, no gap), with ONE fixed amplitude scale (true e^{sigma t} growth, not
per-frame normalised).  Panels: u_s', u_n', free surface eta', and the momentum
flux u_s'u_n' (the meander's lateral momentum transport), each with a colorbar.

    python 02_eulerian_momflux.py [run_tag]        # default: first outputs/run_*.h5
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim

import pp_lib as PP


def render(path):
    res = PP.load_run(path)
    a = res["attrs"]
    tag = os.path.splitext(os.path.basename(path))[0].replace("run_", "")
    frame_xy = PP.labframe_mesh(res)                       # base + perturbation centerline
    ts = res["t"]
    nfr = len(ts)
    zc_amp = np.array([np.max(np.abs(res["zc"][i])) for i in range(nfr)])

    # ONE fixed scale from the final frame (true growth; NOT per-frame norm)
    def gscale(arr):
        return 1.0 / max(np.percentile(np.abs(arr), 99.5), 1e-30)
    Gus = gscale(res["us"][-1]); Gun = gscale(res["un"][-1])
    Get = gscale(res["eta"][-1]); Guv = gscale(PP.momflux(res, -1))
    panels = [("us", r"$u_s'$", Gus, res["us"], "RdBu_r"),
              ("un", r"$u_n'$", Gun, res["un"], "RdBu_r"),
              ("eta", r"$\eta'$ (free surface)", Get, res["eta"], "PuOr_r"),
              ("uv", r"momentum flux $u_s'u_n'$", Guv, None, "RdBu_r")]

    Xf, Yf, _ = frame_xy(nfr - 1)
    xlim = (Xf.min() - 1.5, Xf.max() + 1.5)
    ylim = (Yf.min() - 1.5, Yf.max() + 1.5)

    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5), dpi=110)
    axs = axs.ravel()
    F = float(a["Froude"]); Cb = (float(a["A_bank"]) * float(a["kmeander"]) ** 2
                                  if a["Cbar_amp"] in ("None", b"None") else float(a["Cbar_amp"]))
    # colorbars ONCE (fixed -1..1 ScalarMappables; independent of the per-frame mesh)
    for ax, (key, title, G, fld, cmap) in zip(axs, panels):
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label=title + " (fixed scale)")

    def draw(i):
        X, Y, _ = frame_xy(i)
        for ax, (key, title, G, fld, cmap) in zip(axs, panels):
            ax.clear()                                    # clears the main ax only (colorbar axes persist)
            F2d = PP.momflux(res, i) if key == "uv" else fld[i]
            ax.pcolormesh(X, Y, G * F2d, cmap=cmap, vmin=-1, vmax=1,
                          shading="gouraud", rasterized=True)
            ax.plot(X[:, 0], Y[:, 0], "k", lw=1.6)        # the two banks = channel walls
            ax.plot(X[:, -1], Y[:, -1], "k", lw=1.6)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)
        fig.suptitle(
            rf"full shallow-water meander (s,n) $\to$ lab frame  |  "
            rf"$F={F:g}$, $\bar C={Cb:.2g}$, $k={float(a['kstar']):g}$  |  "
            rf"$t={ts[i]:.1f}$, bank $\times{zc_amp[i]/max(zc_amp[0],1e-30):.2g}$  "
            rf"(fully-Eulerian, banks bound the flow)", fontsize=11)
        return []

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    anim = manim.FuncAnimation(fig, draw, frames=nfr, blit=False)
    out = os.path.join(PP.FIG_DIR, f"momflux_eulerian_{tag}.mp4")
    anim.save(out, fps=14, dpi=110)
    # save the final frame as a preview PNG
    draw(nfr - 1)
    fig.savefig(os.path.join(PP.FIG_DIR, f"momflux_eulerian_{tag}_preview.png"))
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, PP.PKG)}  ({nfr} frames)")


def main():
    args = sys.argv[1:]
    if args:
        p = os.path.join(PP.OUT_DIR, args[0] if args[0].endswith(".h5") else args[0] + ".h5")
    else:
        cands = sorted(glob.glob(os.path.join(PP.OUT_DIR, "run_*.h5")))
        if not cands:
            raise SystemExit("no ../outputs/run_*.h5 -- run the driver first")
        p = cands[0]
    render(p)


if __name__ == "__main__":
    main()
