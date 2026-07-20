#!/usr/bin/env python3
"""02: per-run 2-D planform movies from the driver's HDF5 outputs.

Reads every ../outputs/run_*.h5 (written by `meander_driver.py --mode ivp/sweep`)
and renders a planform movie (psi' warped into the meandering channel + gain
counter) via the verified channel_lib renderer.  No solver re-run.

Run:  micromamba run -n dedalus env OMP_NUM_THREADS=1 python 02_planform_movies.py
      ... python 02_planform_movies.py run_k0p30_amp0p30_rayleigh   # one file
"""
import glob
import os
import sys

import numpy as np

import pp_lib as PP

plt = PP.CL.set_style()


def render(path):
    res = PP.load_run(path)
    a = res["attrs"]
    k = float(a["kstar"])
    m = int(a["mode_index"])
    amp = float(a["cross_amp"])
    fr = a["friction"] if isinstance(a["friction"], str) else a["friction"].decode()
    # measured sigma/c from the bank series (demodulate mode m)
    aser = PP.CL.demodulate(0.5 * (res["top"] + res["bot"]), m)
    t = res["t"]
    win = (t[-1] / 3, t[-1])
    sig, c, r2 = PP.CL.fit_sigma_c(t, aser, k, win)
    Hmin, Hmax = res["Hbed"].min(), res["Hbed"].max()
    title = (rf"$k^*={k:g}$  $\lambda/2b={np.pi/k:.1f}$  bed $a_H={amp:g}$ "
             rf"($H\in[{Hmin:.2f},{Hmax:.2f}]$)  $\sigma^*={sig:+.3f}$  "
             rf"$c^*={c:+.3f}$  ({fr})")
    frames = PP.CL.planform_frames(res, m, k, plt, title, t0=0.0)
    tag = os.path.splitext(os.path.basename(path))[0].replace("run_", "")
    # write into dedalus_meander2/figures via channel_lib.write_mp4 (uses CL.FIG_DIR);
    # redirect by temporarily pointing CL.FIG_DIR at our figures dir.
    old = PP.CL.FIG_DIR
    PP.CL.FIG_DIR = PP.FIG_DIR
    try:
        PP.CL.write_mp4(frames, f"planform_{tag}", fps=16)
    finally:
        PP.CL.FIG_DIR = old


def main():
    if len(sys.argv) > 1:
        paths = [os.path.join(PP.OUT_DIR, a if a.endswith(".h5") else a + ".h5")
                 for a in sys.argv[1:]]
    else:
        paths = sorted(glob.glob(os.path.join(PP.OUT_DIR, "run_*.h5")))
    if not paths:
        raise SystemExit("no outputs/run_*.h5 -- run the driver --mode ivp/sweep first")
    for p in paths:
        print(f"rendering {os.path.basename(p)} ...")
        render(p)
    print(f"02_planform_movies: done ({len(paths)} movie(s)).")


if __name__ == "__main__":
    main()
