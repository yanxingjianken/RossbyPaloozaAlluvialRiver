#!/usr/bin/env python3
"""postprocessing/pp_lib.py -- shared helpers for dedalus_meander_full_SW.

The core solver (../sw_meander.py) writes raw HDF5 to ../outputs/.  Here we
read them and render: the dispersion relation, and the fully-Eulerian
momentum-flux movie (the (s,n) fields mapped back to the lab-frame meandering
channel so the two banks bound the flow).
"""
from __future__ import annotations

import os

import numpy as np
import h5py

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT_DIR = os.path.join(PKG, "outputs")
FIG_DIR = os.path.join(PKG, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, PKG)
import sw_meander as MD          # noqa: E402  (base profiles for the y-z cross-section)


def cfg_from_attrs(a):
    """Reconstruct a driver CONFIG dict from HDF5 attrs (None strings -> None)."""
    cfg = dict(MD.CONFIG)
    for k in cfg:
        if k in a:
            v = a[k]
            if isinstance(v, bytes):
                v = v.decode()
            cfg[k] = None if v == "None" else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
    return cfg


def load_run(path):
    """Read one run HDF5 into a dict (fields + grids + attrs)."""
    with h5py.File(path, "r") as h:
        res = {k: h[k][:] for k in h.keys()}
        res["attrs"] = {k: h.attrs[k] for k in h.attrs}
    return res


# NOTE: a sweep_dispersion() used to live here, collecting one (k, sigma, c) per run
# from the attr `kstar` -- i.e. from "which wavelength this run perturbed".  With
# broadband seeding that attr no longer identifies anything: every run excites every
# k, and the dispersion relation lives in the per-mode disp_* datasets written by the
# driver (see 01_dispersion.py).  Keeping the old helper around would let a caller
# silently rebuild a one-point-per-run "dispersion relation" that is not one, so it is
# deleted rather than deprecated.


# --------------------------------------------------------------------------- #
#  Lab-frame map: (s,n) channel-fitted grid -> Cartesian (x,y) meandering strip
# --------------------------------------------------------------------------- #
def centerline(s, cbar_of_s, zc=None):
    """Cartesian centerline (xc,yc) and unit normal (nx,ny) from curvature.

    Sign convention MUST match the solver's metric sigma = 1 + n*C (derivations
    sec.1): with the left normal (-sin,cos)theta that requires
        theta(s) = -INT cbar ds
    (using +INT would render the mirror image, i.e. the metric of 1 - n*C).
    (xc,yc)=INT (cos,sin)theta ds.  If zc is given, shift the centerline by zc*normal.
    """
    ds = s[1] - s[0]
    theta = -np.concatenate([[0.0], np.cumsum(0.5 * (cbar_of_s[1:] + cbar_of_s[:-1]) * ds)])
    xc = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * ds)])
    yc = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * ds)])
    nx, ny = -np.sin(theta), np.cos(theta)
    if zc is not None:
        xc = xc + zc * nx
        yc = yc + zc * ny
    return xc, yc, nx, ny


# NOTE: a labframe_mesh(res, gain) used to live here.  Its `gain` argument scaled the
# bank displacement for display only -- exactly the amplification that makes two movies
# of the SAME evolution look like different physics.  The absolute-Eulerian convention
# (one gain, chosen once from linear validity, applied to every field and frame alike)
# is implemented in 02_eulerian_momflux.py; a helper whose signature invites a
# per-movie display gain works against it, so it is deleted.


def momflux(res, i):
    """Perturbation momentum flux  u_s' * u_n'  (Reynolds stress) at frame i."""
    return res["us"][i] * res["un"][i]
