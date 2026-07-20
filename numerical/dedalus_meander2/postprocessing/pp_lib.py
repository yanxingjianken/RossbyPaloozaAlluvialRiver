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


def multipanel_eulerian_frames(res, plt):
    """FULLY-EULERIAN multipanel: fixed rectangular domain [0,Lx]x[-1,1],
    NO per-frame amplitude normalization (ONE fixed scale for the whole movie,
    set by the final frame => the field brightens and the banks grow out of a
    near-straight channel = the true exponential growth), NO warp_fill.  The y-z
    cross-section is taken at a FIXED x (fully Eulerian, not crest-tracking).
    """
    a = res["attrs"]
    cfg = _cfg_from_attrs(a)
    k = float(a["kstar"]); D = cfg["D"]
    x, y, Lx = res["x"], res["y"], res["Lx"]
    x2b = x / 2.0
    Hbed = res["Hbed"]
    ub_y = MD.ubar(y, cfg)
    psis = res["psis"]; tops = res["top"]; bots = res["bot"]
    # ONE fixed scale from the final frame (constant across frames -> shows growth)
    Gc = 0.95 / max(np.max(np.abs(psis[-1])), 1e-30)          # psi' colour scale
    bser = 0.5 * (tops + bots)
    Gb = 0.55 / max(np.max(np.abs(bser[-1])), 1e-30)          # bank displacement
    dyPf = np.gradient(psis[-1], y, axis=1)
    dxPf = np.gradient(psis[-1], x, axis=0)
    Guv = 0.95 / max(np.max(np.abs(-(dyPf * dxPf) / Hbed**2)), 1e-30)
    a0 = np.max(np.abs(bser[0]))
    ixf = psis.shape[1] // 2                                   # FIXED x-slice (mid-domain)
    Hy = Hbed[ixf] if Hbed.shape[0] > 1 else Hbed[0]
    ks, sig, cph, cg = dispersion(cfg, ks=np.linspace(0.05, 1.5, 30))   # once

    def rect(ax, F, vlim, cmap="RdBu_r"):
        ax.pcolormesh(x2b, y, F.T, cmap=cmap, vmin=-vlim, vmax=vlim,
                      shading="gouraud", rasterized=True)
        ax.axhline(1, color="0.6", lw=0.6, ls=":"); ax.axhline(-1, color="0.6", lw=0.6, ls=":")

    frames = []
    for i in range(len(psis)):
        amp = np.max(np.abs(bser[i]))
        dtop, dbot = Gb * tops[i], Gb * bots[i]
        uv = -(np.gradient(psis[i], y, axis=1) * np.gradient(psis[i], x, axis=0)) / Hbed**2
        fig, axs = plt.subplots(2, 3, figsize=(15.0, 6.0), dpi=100)
        # psi' (the growing perturbation, TRUE scale) + banks as displaced lines
        rect(axs[0, 1], Gc * psis[i], 1.0)
        axs[0, 1].set_title(r"$\psi'$ (fixed scale; grows)", fontsize=10)
        # psi_total = base + growing perturbation
        base = (-y + D * y**3 / 3.0)[None, :]
        rect(axs[0, 0], base + Gc * psis[i], 1.5)
        axs[0, 0].set_title(r"$\psi_{\rm total}=\bar\psi+\psi'$", fontsize=10)
        rect(axs[0, 2], Guv * uv, 1.0, cmap="PuOr_r")
        axs[0, 2].set_title(r"momentum flux $\overline{u'v'}$", fontsize=10)
        for ax in axs[0]:
            ax.plot(x2b, 1 + dtop, color=COLORS["psi1"], lw=1.8)      # true bank lines
            ax.plot(x2b, -1 + dbot, color=COLORS["psi1"], lw=1.8)
            ax.set_xlim(0, Lx / 2); ax.set_ylim(-1.75, 1.75)
            ax.set_xlabel(r"downstream $x/2b$ (FIXED domain)", fontsize=8)

        # y-z cross-section at a FIXED x (Eulerian): bed + jet + growing bank wiggle
        axc = axs[1, 0]
        u_tot = ub_y / Hy
        zg = np.linspace(-Hy.max() * 1.1, 0.25, 60)
        U2 = np.where(zg[:, None] > -Hy[None, :], u_tot[None, :], np.nan)
        pc = axc.pcolormesh(y, zg, U2, cmap="viridis", shading="auto")
        axc.plot(y, -Hy, color=COLORS["bank"], lw=2.5)
        axc.fill_between(y, -Hy, zg.min(), color=COLORS["bank"], alpha=0.25)
        axc.axhline(0, color="0.3", lw=1.5)
        axc.plot([-1 + Gb * bots[i][ixf]] * 2, [-Hy[0], 0.15], color=COLORS["psi1"], lw=3)
        axc.plot([1 + Gb * tops[i][ixf]] * 2, [-Hy[-1], 0.15], color=COLORS["psi1"], lw=3)
        fig.colorbar(pc, ax=axc, fraction=0.045, pad=0.02, label=r"$\bar u/H$")
        axc.set_xlim(-1.6, 1.6); axc.set_ylim(zg.min(), 0.4)
        axc.set_xlabel(r"cross-channel $y/b$"); axc.set_ylabel(r"depth $z$")
        axc.set_title(rf"$y$-$z$ cross-section at FIXED $x/2b={x2b[ixf]:.1f}$", fontsize=9)

        axd = axs[1, 1]
        axd.plot(ks, sig, color=COLORS["growth"], lw=1.8, label=r"$\sigma^*$")
        axd.plot(ks, cph, color=COLORS["upstream"], lw=1.8, label=r"$c^*$")
        axd.plot(ks, cg, color=COLORS["momentum"], lw=1.8, label=r"$c_g$")
        axd.axhline(0, color="k", lw=0.6); axd.axvline(k, color="0.5", lw=1, ls=":")
        axd.set_xlabel(r"$k^*$"); axd.set_title("dispersion", fontsize=10)
        axd.legend(fontsize=8, ncol=3, loc="upper right"); axd.set_ylim(-0.5, 0.6)

        axs[1, 2].axis("off")
        stat = ("FULLY EULERIAN\n(fixed domain, no warp,\nno per-frame norm)\n\n"
                rf"$k^*={k:g}$   bed $H\in[{Hbed.min():.2f},{Hbed.max():.2f}]$" "\n"
                rf"true bank ampl $={amp:.1e}$" "\n"
                rf"growth $=\times{amp/max(a0,1e-30):.2g}$ (real)")
        axs[1, 2].text(0.03, 0.92, stat, va="top", ha="left", fontsize=11,
                       transform=axs[1, 2].transAxes)
        fig.tight_layout()
        frames.append(CL.fig_to_rgb(fig)); plt.close(fig)
    return frames


def _cfg_from_attrs(a):
    """Reconstruct a driver CONFIG dict from HDF5 attrs (for profiles/dispersion)."""
    cfg = dict(MD.CONFIG)
    for k in cfg:
        if k in a:
            v = a[k]
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, str) and v == "None":
                v = None
            cfg[k] = v
    return cfg


def multipanel_frames(res, plt, max_frames=72):
    """5-panel variable-H movie: psi_total, psi', momentum flux u'v', a y-z
    cross-section (Ikeda Fig-2b view; DEPTH-AVERAGED -- bed H(y), banks, jet, no
    resolved vertical flow), and a dispersion/stats panel.

    Reuses channel_lib.warp_fill for the three planform panels.
    """
    a = res["attrs"]
    cfg = _cfg_from_attrs(a)
    k = float(a["kstar"]); m = int(a["mode_index"]); D = cfg["D"]
    x, y, Lx = res["x"], res["y"], res["Lx"]
    x2b = x / 2.0
    Hbed = res["Hbed"]                                   # (Nx, Ny)
    ub_y = MD.ubar(y, cfg)                               # base jet ubar(y)
    # dispersion once (cheap GEP)
    ks, sig, cph, cg = dispersion(cfg, ks=np.linspace(0.05, 1.5, 45))
    a1 = CL.demodulate(0.5 * (res["top"] + res["bot"]), m)
    gain0 = np.abs(a1[0])
    # base transport streamfunction (flat-bed form is fine for the planform base)
    pb = CL.psibar(y, D)

    frames = []
    for i in range(len(res["psis"])):
        amp = np.abs(a1[i]); scale = 0.5 / max(amp, 1e-300)
        pp = res["psis"][i]                              # Psi' (Nx, Ny)
        dtop, dbot = scale * res["top"][i], scale * res["bot"][i]
        xc = (-np.angle(a1[i]) / k) % Lx / 2.0
        ixc = int(np.argmin(np.abs(x2b - xc)))
        # momentum flux u'v' = -(1/H^2) dyPsi' dxPsi'
        dyP = np.gradient(pp, y, axis=1)
        dxP = np.gradient(pp, x, axis=0)
        uv = -(dyP * dxP) / Hbed**2
        uv = uv / max(np.max(np.abs(uv)), 1e-30)

        fig, axs = plt.subplots(2, 3, figsize=(15.0, 6.0), dpi=100)
        CL.warp_fill(axs[0, 0], x2b, y, pb[None, :] + scale * pp, dtop, dbot, vlim=1.3)
        axs[0, 0].set_title(r"$\psi_{\rm total}$ (streamlines)", fontsize=10)
        CL.warp_fill(axs[0, 1], x2b, y, scale * pp, dtop, dbot, vlim=0.55)
        axs[0, 1].set_title(r"$\psi'$ (perturbation)", fontsize=10)
        CL.warp_fill(axs[0, 2], x2b, y, uv, dtop, dbot, vlim=1.0)
        axs[0, 2].set_title(r"momentum flux $\overline{u'v'}$ "
                            r"($-\frac{1}{H^2}\partial_y\Psi'\partial_x\Psi'$)",
                            fontsize=10)
        for ax in axs[0]:
            ax.axvline(xc, color=COLORS["upstream"], lw=1.5, alpha=0.8)
            ax.set_xlim(0, Lx / 2); ax.set_ylim(-2.6, 2.6)
            ax.set_xlabel(r"downstream $x/2b$", fontsize=8)

        # ---- y-z cross-section at the tracked slice x=xc (Ikeda Fig 2b view) ---
        axc = axs[1, 0]
        Hy = Hbed[ixc]                                   # bed depth H(y) at xc
        u_tot = ub_y / Hy                                # depth-averaged jet u=ubar/H
        zg = np.linspace(-Hy.max() * 1.1, 0.25, 60)
        U2 = np.where(zg[:, None] > -Hy[None, :], u_tot[None, :], np.nan)
        pc = axc.pcolormesh(y, zg, U2, cmap="viridis", shading="auto")
        axc.plot(y, -Hy, color=COLORS["bank"], lw=2.5)        # bed profile H(y)
        axc.fill_between(y, -Hy, zg.min(), color=COLORS["bank"], alpha=0.25)
        axc.axhline(0, color="0.3", lw=1.5)                   # rigid lid
        axc.plot([-1, -1], [-Hy[0], 0.15], color=COLORS["psi1"], lw=3)   # walls
        axc.plot([1, 1], [-Hy[-1], 0.15], color=COLORS["psi1"], lw=3)
        fig.colorbar(pc, ax=axc, fraction=0.045, pad=0.02,
                     label=r"$\bar u/H$")
        axc.set_xlim(-1.4, 1.4); axc.set_ylim(zg.min(), 0.4)
        axc.set_xlabel(r"cross-channel $y/b$"); axc.set_ylabel(r"depth $z$")
        axc.set_title(rf"$y$-$z$ cross-section at $x/2b={xc:.1f}$ "
                      r"(depth-averaged; no vertical flow)", fontsize=9)

        # ---- dispersion + stats -------------------------------------------- #
        axd = axs[1, 1]
        axd.plot(ks, sig, color=COLORS["growth"], lw=1.8, label=r"$\sigma^*$")
        axd.plot(ks, cph, color=COLORS["upstream"], lw=1.8, label=r"$c^*$")
        axd.plot(ks, cg, color=COLORS["momentum"], lw=1.8, label=r"$c_g$")
        axd.axhline(0, color="k", lw=0.6); axd.axvline(k, color="0.5", lw=1, ls=":")
        axd.set_xlabel(r"$k^*$"); axd.set_title("dispersion", fontsize=10)
        axd.legend(fontsize=8, ncol=3, loc="upper right")
        axd.set_ylim(-0.5, 0.6)

        axs[1, 2].axis("off")
        Hmin, Hmax = Hbed.min(), Hbed.max()
        oi = ks[np.argmin(np.abs(ks - k))]
        stat = (rf"$k^*={k:g}$   $\lambda/2b={np.pi/k:.1f}$" "\n"
                rf"bed $H\in[{Hmin:.2f},{Hmax:.2f}]$" "\n"
                rf"$\sigma^*_{{\rm EVP}}={a['sigma_evp']:.3f}$" "\n"
                rf"$D={D}$  $\gamma={cfg['gamma']}$" "\n"
                rf"gain $e^{{\sigma t}}=\times{amp/gain0:.2g}$")
        axs[1, 2].text(0.05, 0.9, stat, va="top", ha="left", fontsize=12,
                       transform=axs[1, 2].transAxes)
        fig.tight_layout()
        frames.append(CL.fig_to_rgb(fig)); plt.close(fig)
    return frames
