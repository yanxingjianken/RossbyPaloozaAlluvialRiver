#!/usr/bin/env python3
"""Initial-condition plots for the bank-migration run: the carved geometry + base flow BEFORE any
evolution.  Top = xOy bird's-eye (bed elevation, meander banks, buffers, erodible zone, base-flow
speed); bottom = yOz transverse sections at two apexes (bed h(n) + base jet U(n) + water surface).

    micromamba run -n fourcastnetv2 python postprocessing/05_ic.py
"""
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import run_meander as rm
CFG = rm.CONFIG; OUT = os.path.join(ROOT, "figures")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="", help="substring of the run tag to plot (default: all)")
    args = ap.parse_args()
    bases = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "*")))
    bases = [b for b in bases if args.match in os.path.basename(b)]
    if not bases:
        print("no matching runs/ -- build with run_meander.py first"); return 1
    for base in bases:
        _plot(base)
    return 0


def _plot(base):
    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    x, y = g["x"], g["y"]; n, kap = g["n"], g["kappa"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    Depth = np.loadtxt(os.path.join(base, "bathy", "depth.txt")).T      # (nx,ny)
    Zs = np.loadtxt(os.path.join(base, "bathy", "hard.txt")).T
    u = np.loadtxt(os.path.join(base, "ini", "u.txt")).T
    v = np.loadtxt(os.path.join(base, "ini", "v.txt")).T
    eta = np.loadtxt(os.path.join(base, "ini", "eta.txt")).T
    spd = np.hypot(u, v)
    b = CFG["b"]; L = float(g["L"]); buf = float(g["buffer_len"]); lam = float(g["lam"])
    half = float(g["half"])

    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.3])

    # --- xOy panel 1: bed elevation z_b = -Depth --------------------------------
    ax = fig.add_subplot(gs[0, :])
    pc = ax.pcolormesh(X, Y, -Depth, cmap="terrain", shading="auto")
    ax.contour(X, Y, np.abs(n), levels=[b], colors="k", linewidths=0.6)         # bank edge |n|=b
    for xb in (buf, L - buf):
        ax.axvline(xb, color="r", ls=":", lw=1.0)                               # buffer edges
    ax.set_ylim(-half, half); ax.set_ylabel("y [m]"); fig.colorbar(pc, ax=ax, label="bed elev. $z_b$ [m]")
    C0g = 1.0 / float(g["R_min"])                    # THIS run's curvature (B3 differs from CONFIG)
    ax.set_title(f"IC — carved meander bed  |  $\\lambda$={lam:.0f} m, C0={C0g:.1e} 1/m, "
                 f"R/W={1/C0g/(2*b):.1f}, sinuosity {float(g['sinuosity']):.2f}   "
                 f"(red = non-erodible buffers)", fontsize=9, loc="left")

    # --- xOy panel 2: base-flow speed + erodible zone ---------------------------
    ax = fig.add_subplot(gs[1, :])
    pc = ax.pcolormesh(X, Y, np.where(spd > 0, spd, np.nan), cmap="viridis", shading="auto")
    ax.contour(X, Y, Zs, levels=[1e5], colors="w", linewidths=0.7)              # erodible-zone edge
    ax.contour(X, Y, np.abs(n), levels=[b], colors="k", linewidths=0.4)
    st = 24
    ax.quiver(X[::st, ::st], Y[::st, ::st], u[::st, ::st], v[::st, ::st], scale=25, width=0.0015, color="w")
    ax.set_ylim(-half, half); ax.set_ylabel("y [m]"); ax.set_xlabel("down-valley x [m]")
    fig.colorbar(pc, ax=ax, label="base-flow $|U|$ [m/s]")
    ax.set_title("IC — analytic BASE flow ($u'=v'=0$: normal flow along the tangent; the meander "
                 "perturbation spins up from here).  white = erodible-interior edge", fontsize=9, loc="left")

    # --- yOz: transverse sections at two apexes ---------------------------------
    secs = rm.section_x(lam, rm.cfg_from_grid(g))
    S = float(g["S"]); toe = b + CFG["m_bank"] * (CFG["H_b"] - CFG["h_plain"]) + 10
    for k, xs in enumerate(secs[:2]):
        i = int(np.argmin(np.abs(x - xs))); ks = float(np.sign(kap[i, np.argmax(np.abs(kap[i]))]))
        sel = np.abs(n[i]) <= toe; nn = n[i][sel] * ks; o = np.argsort(nn)
        zb = -Depth[i][sel][o]; sf = eta[i][sel][o]; U = spd[i][sel][o]
        ax = fig.add_subplot(gs[2, k])
        ax.fill_between(nn[o], zb, np.maximum(sf, zb), color="#8fbfe0")          # water
        ax.fill_between(nn[o], -4, zb, color="#c8a06a")                          # bed
        ax.plot(nn[o], zb, color="#5a3c1e", lw=1.5)
        ax2 = ax.twinx(); ax2.plot(nn[o], U, "r-", lw=1.3); ax2.set_ylabel("$U(n)$ [m/s]", color="r")
        ax2.tick_params(axis="y", colors="r"); ax2.set_ylim(0, 1.2)
        ax.set_xlabel("n·sgn($\\kappa$) [m]  (<0 outer, >0 inner)"); ax.set_ylim(-3.6, 0.6)
        if k == 0: ax.set_ylabel("elevation [m]")
        ax.set_title(f"yOz section S{k+1} @ x={xs:.0f} m  (bed h(n) + base jet U(n))", fontsize=9)
    fig.suptitle("BANK-MIGRATION run — initial condition (rigid non-flat bed shape, erodible banks, "
                 "gap-1 secondary flow A=2.89 ON)", fontsize=11)
    # key the filename on lam AND curvature: B2 and B3 share lam=1560 (fix-C0 vs fix-A) and would
    # otherwise overwrite each other's per-case figure.
    label = os.path.basename(os.path.dirname(base)) + "_" + "_".join(os.path.basename(base).split("_")[:2])          # e.g. lam1560_C1p42e-3
    fp = os.path.join(OUT, f"IC_bank_migration_{label}.png")
    fig.savefig(fp, dpi=115); plt.close(fig)
    print(f"wrote {fp}")
    print(f"  base flow: |U| interior mean {spd[(np.abs(n)<=b)&(Zs>1e5)].mean():.2f} m/s, "
          f"max {spd[(np.abs(n)<=b)&(Zs>1e5)].max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
