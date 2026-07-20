#!/usr/bin/env python3
"""postprocessing/pp_lib.py -- shared helpers for dedalus_meander2 figures/movies.

Design: the CORE solver (../meander_driver.py) writes RAW HDF5 to ../outputs/.
Everything here READS those HDF5 files (never re-runs the solver) and renders.

Reuse (DRY): the verified renderers + shared helper block live in the sibling
constant-depth package `dedalus_meander/channel_lib.py`.  We import them and feed
them a `res` dict adapted from HDF5, so plots stay byte-identical in style.

Env/run:  micromamba run -n dedalus env OMP_NUM_THREADS=1 python <script>.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                        # dedalus_meander2/
OUT_DIR = os.path.join(PKG, "outputs")
FIG_DIR = os.path.join(PKG, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- import the verified renderers + physics (single source of truth) ------ #
sys.path.insert(0, os.path.join(PKG, "..", "dedalus_meander"))
import channel_lib as CL             # noqa: E402  (set_style/warp_fill/... + helper block)
sys.path.insert(0, PKG)
import meander_driver as MD          # noqa: E402  (channel_modes_H, evp, profiles)

COLORS = CL.COLORS


def load_run(path):
    """Read a driver HDF5 into a `res`-style dict usable by channel_lib renderers.

    Keys: Lx, x, y, top[Nt,Nx], bot[Nt,Nx], psis[Nt,Nx,Ny], tsnap, t, Hbed[Nx,Ny],
    plus `attrs` (all CONFIG values + mode_index, sigma_evp, t_end).
    """
    with h5py.File(path, "r") as h:
        res = dict(
            Lx=float(h.attrs["Lx"]),
            x=h["x"][:], y=h["y"][:],
            top=h["top"][:], bot=h["bot"][:],
            psis=h["psi"][:], t=h["t"][:],
            tsnap=h["t"][:],                       # bank + snapshot cadence aligned
            attrs={k: h.attrs[k] for k in h.attrs},
        )
        Hbed = h["Hbed"][:]
        Nx = res["x"].size
        res["Hbed"] = np.broadcast_to(Hbed, (Nx, res["y"].size)).copy()
    return res


def group_velocity(kstar, cfg, dk=0.02, N=201):
    """c_g = d Re(omega)/dk of the bank branch, from the variable-H GEP."""
    ks = np.array([kstar - dk, kstar + dk])
    wr = []
    for kk in ks:
        w, _ = MD.channel_modes_H(N, float(kk), cfg)
        tgt = complex(MD.VL.bank_branch([kk], cfg["D"], cfg["gamma"],
                                        MD.bank_E(cfg), cfg["friction"])[0])
        wr.append(w[np.argmin(np.abs(w - tgt))].real)
    return (wr[1] - wr[0]) / (2 * dk)


def dispersion(cfg, ks=None, N=201):
    """(ks, sigma, c_phase, c_group) of the bank branch over k*, variable-H GEP."""
    if ks is None:
        ks = np.linspace(0.05, 1.55, 60)
    sig, cph, cg = [], [], []
    for kk in ks:
        o = MD.gep_bank_mode_H(N, float(kk), cfg)
        sig.append(o.imag)
        cph.append(o.real / kk)
        cg.append(group_velocity(float(kk), cfg, N=N))
    return ks, np.array(sig), np.array(cph), np.array(cg)
