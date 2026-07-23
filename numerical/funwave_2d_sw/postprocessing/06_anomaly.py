#!/usr/bin/env python3
"""Total AND anomaly flow movie for the bank-migration run.  Covers BOTH phases so the meander
perturbation is seen SPINNING UP from ~0 (spin-up, rigid bed) then evolving with the banks (morph).

  top    = TOTAL speed |U| with the eroding bank outline
  bottom = ANOMALY = current flow minus the initial BASE flow (u−u_ic, v−v_ic): the meander-induced
           perturbation u',v' (0 at t=0, grows as the bends force it, then tracks the bank change)

    micromamba run -n fourcastnetv2 python postprocessing/06_anomaly.py [--max-frames N]
"""
import argparse, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import run_meander as rm
CFG = rm.CONFIG; OUT = os.path.join(ROOT, "figures")


def load(p):
    return np.loadtxt(p).T


def phase_frames(base, ph):
    return [p for p in sorted(glob.glob(os.path.join(base, ph, "output", "u_*"))) if "99999" not in p]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10); args = ap.parse_args()
    bases = sorted(glob.glob(os.path.join(ROOT, "runs", "*")))
    if not bases:
        print("no runs/"); return 1
    for base in bases:
        _plot(base, args)
    return 0


def _plot(base, args):
    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    lam = float(g["lam"])
    x, y, n = g["x"], g["y"], g["n"]; X, Y = np.meshgrid(x, y, indexing="ij")
    b = CFG["b"]; L = float(g["L"]); buf = float(g["buffer_len"]); half = float(g["half"])
    u_ic = load(os.path.join(base, "ini", "u.txt")); v_ic = load(os.path.join(base, "ini", "v.txt"))
    # frames: spin-up then morph; tag each with its phase + hydro time
    frames = [("spin-up", p) for p in phase_frames(base, "spinup")] + \
             [("morph", p) for p in phase_frames(base, "morph")]
    if not frames:
        print(f"no flow frames yet for {os.path.basename(base)}"); return
    if args.max_frames:
        frames = frames[:args.max_frames]
    # fixed colour scales from a late frame
    ulate = load(frames[-1][1]); vlate = load(frames[-1][1].replace("/u_", "/v_"))
    smax = np.nanpercentile(np.hypot(ulate, vlate), 99.5)
    amax = np.nanpercentile(np.hypot(ulate - u_ic, vlate - v_ic), 99.0)
    amax = max(amax, 1e-3)
    dep0 = load(os.path.join(base, "morph", "output", "dep_00001")) if \
        glob.glob(os.path.join(base, "morph", "output", "dep_00001")) else g["Depth"]
    os.makedirs(OUT, exist_ok=True)
    # lam+curvature label: B2 and B3 share lam=1560 (fix-C0 vs fix-A), so lam alone would collide
    label = "_".join(os.path.basename(base).split("_")[:2])
    mp4 = os.path.join(OUT, f"anomaly_bank_migration_{label}.mp4")
    w = imageio.get_writer(mp4, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
    dt_sp = float(CFG["plot_intv"])
    isp = sum(1 for f in frames if f[0] == "spin-up")
    for k, (ph, p) in enumerate(frames):
        u = load(p); v = load(p.replace("/u_", "/v_"))
        spd = np.hypot(u, v); anom = np.hypot(u - u_ic, v - v_ic)
        wet = spd > 1e-6
        fig, ax = plt.subplots(2, 1, figsize=(13, 5.2), constrained_layout=True, sharex=True)
        pc0 = ax[0].pcolormesh(X, Y, np.where(wet, spd, np.nan), cmap="viridis", vmin=0, vmax=smax, shading="auto")
        pc1 = ax[1].pcolormesh(X, Y, np.where(wet, anom, np.nan), cmap="magma", vmin=0, vmax=amax, shading="auto")
        for a in ax:
            a.contour(X, Y, np.abs(n), levels=[b], colors="w", linewidths=0.4)
            for xb in (buf, L - buf):
                a.axvline(xb, color="r", ls=":", lw=0.7)
            a.set_ylim(-half, half); a.set_ylabel("y [m]")
        st = 26
        ax[1].quiver(X[::st, ::st], Y[::st, ::st], (u - u_ic)[::st, ::st], (v - v_ic)[::st, ::st],
                     scale=6, width=0.0015, color="c")
        fig.colorbar(pc0, ax=ax[0], label="total $|U|$ [m/s]")
        fig.colorbar(pc1, ax=ax[1], label="anomaly $|u'|$ [m/s]")
        ax[0].set_title("TOTAL flow speed", fontsize=9, loc="left")
        ax[1].set_title("ANOMALY $u',v'$ = flow $-$ base (meander-induced; 0 at t=0)", fontsize=9, loc="left")
        ax[1].set_xlabel("down-valley x [m]")
        th = (k if ph == "spin-up" else k - isp) * dt_sp
        tm = th * CFG["Morph_factor"] / 86400 if ph == "morph" else 0.0
        fig.suptitle(f"bank migration — {ph}   |   $t_{{hydro}}$={th:6.0f} s"
                     + (f"   $t_{{morph}}$={tm:5.2f} d" if ph == "morph" else "   (rigid bed: flow spins up)")
                     + f"   |   gap-1 $A$=2.89 ON", fontsize=10)
        fig.canvas.draw()
        buf_rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, wd = buf_rgb.shape[:2]
        w.append_data(buf_rgb[:h - (h % 2), :wd - (wd % 2)].copy())
        plt.close(fig)
    w.close()
    print(f"wrote {mp4} ({len(frames)} frames: {isp} spin-up + {len(frames)-isp} morph)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
