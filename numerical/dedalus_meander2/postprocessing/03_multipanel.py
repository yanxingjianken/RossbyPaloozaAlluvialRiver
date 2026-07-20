#!/usr/bin/env python3
"""03: rich multi-panel movie from a driver HDF5.

5 panels: psi_total, psi', momentum flux u'v', a y-z cross-section (Ikeda Fig-2b
view; depth-averaged bed+jet+banks), and the dispersion (sigma, c*, c_g).  Shows
far more than the streamfunction alone -- momentum transfer + the bed cross-section
+ the full dispersion, all in one movie.

Run:  micromamba run -n dedalus env OMP_NUM_THREADS=1 python 03_multipanel.py [run_tag]
"""
import glob
import os
import sys

import pp_lib as PP

plt = PP.CL.set_style()


def render(path):
    res = PP.load_run(path)
    frames = PP.multipanel_frames(res, plt)
    tag = os.path.splitext(os.path.basename(path))[0].replace("run_", "")
    old = PP.CL.FIG_DIR
    PP.CL.FIG_DIR = PP.FIG_DIR
    try:
        PP.CL.write_mp4(frames, f"multipanel_{tag}", fps=14)
    finally:
        PP.CL.FIG_DIR = old


def main():
    if len(sys.argv) > 1:
        paths = [os.path.join(PP.OUT_DIR, a if a.endswith(".h5") else a + ".h5")
                 for a in sys.argv[1:]]
    else:
        paths = sorted(glob.glob(os.path.join(PP.OUT_DIR, "run_*.h5")))
    if not paths:
        raise SystemExit("no outputs/run_*.h5 -- run the driver first")
    for p in paths:
        print(f"rendering multipanel {os.path.basename(p)} ...")
        render(p)
    print(f"03_multipanel: done ({len(paths)}).")


if __name__ == "__main__":
    main()
