#!/usr/bin/env python3
"""postprocessing/pp_lib.py -- shared helpers for dedalus_meander_full_SW.

The core solver (../sw_sn_driver.py) writes raw HDF5 to ../outputs/.  Here we
read them and render: the dispersion relation, and the fully-Eulerian
momentum-flux movie (the (s,n) fields mapped back to the lab-frame meandering
channel so the two banks bound the flow).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT_DIR = os.path.join(PKG, "outputs")
FIG_DIR = os.path.join(PKG, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_run(path):
    """Read one run HDF5 into a dict (fields + grids + attrs)."""
    with h5py.File(path, "r") as h:
        res = {k: h[k][:] for k in h.keys()}
        res["attrs"] = {k: h.attrs[k] for k in h.attrs}
    return res


def sweep_dispersion(pattern="run_*.h5"):
    """Collect (k, sigma, c, Froude, Cbar) from a directory of sweep runs."""
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, pattern))):
        with h5py.File(p, "r") as h:
            a = h.attrs
            cb = (a["A_bank"] * a["kmeander"] ** 2 if a["Cbar_amp"] in ("None", b"None")
                  else float(a["Cbar_amp"]))
            rows.append(dict(k=float(a["kstar"]), sigma=float(a["sigma_meas"]),
                             c=float(a["c_meas"]), F=float(a["Froude"]),
                             Cbar=float(cb)))
    return rows


# --------------------------------------------------------------------------- #
#  Lab-frame map: (s,n) channel-fitted grid -> Cartesian (x,y) meandering strip
# --------------------------------------------------------------------------- #
def centerline(s, cbar_of_s, zc=None):
    """Cartesian centerline (xc,yc) and unit normal (nx,ny) from curvature.

    theta(s)=INT cbar ds ;  (xc,yc)=INT (cos,sin)theta ds ;  normal=(-sin,cos)theta.
    If zc (perturbation lateral offset) is given, shift the centerline by zc*normal.
    """
    ds = s[1] - s[0]
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (cbar_of_s[1:] + cbar_of_s[:-1]) * ds)])
    xc = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * ds)])
    yc = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * ds)])
    nx, ny = -np.sin(theta), np.cos(theta)
    if zc is not None:
        xc = xc + zc * nx
        yc = yc + zc * ny
    return xc, yc, nx, ny


def labframe_mesh(res, gain=1.0):
    """(X,Y) lab coords (Ns,Nn) for the (s,n) grid using the BASE curvature.

    gain scales the display of the perturbation centerline zc (visual).
    Returns a function frame(i) -> (X, Y, zc_i) for time index i.
    """
    s, n = res["s"], res["n"]
    a = res["attrs"]
    km = float(a["kmeander"])
    cb_amp = (float(a["A_bank"]) * km ** 2 if a["Cbar_amp"] in ("None", b"None")
              else float(a["Cbar_amp"]))
    cbar_s = cb_amp * np.cos(km * s)

    def frame(i):
        zc_i = gain * res["zc"][i]
        xc, yc, nx, ny = centerline(s, cbar_s, zc=zc_i)
        X = xc[:, None] + n[None, :] * nx[:, None]
        Y = yc[:, None] + n[None, :] * ny[:, None]
        return X, Y, res["zc"][i]

    return frame


def momflux(res, i):
    """Perturbation momentum flux  u_s' * u_n'  (Reynolds stress) at frame i."""
    return res["us"][i] * res["un"][i]
