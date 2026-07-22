#!/usr/bin/env python3
"""Shared helpers for the deliverable1_noboru_model postprocessing scripts.

Contains (a) the byte-identical rossby_palooza shared plotting/mp4 helper block,
(b) the reconstructed det M = 0 of river.pdf p.19, (c) the p.16 momentum-flux
diagnostic, and (d) the RealFourier mode-amplitude guard.

Nothing here reads river.pdf; every deck-sourced number arrives via a page-cited
constant defined below.  Nothing here reads a value off a figure.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(HERE, "figures")
OUT_DIR = os.path.join(HERE, "outputs")
sys.path.insert(0, HERE)

os.makedirs(FIG_DIR, exist_ok=True)

# Deck colours, matching the river.pdf "Streamfunctions" panels (pp.12-19):
# psi1 and psi3 blue, psi2 red.
COLORS = {
    "psi1": "#08519c",        # bank streamfunctions (deck blue)
    "psi2": "#d7301f",        # centre streamfunction (deck red)
    "growth": "#238b45",
    "upstream": "#6a51a3",
    "deckpin": "#252525",
    "flux_pos": "#6a51a3",
    "flux_neg": "#d7301f",
}

# The six (D, gamma) families of the river.pdf p.20 figure, with the deck's own
# line colours.  growth-left / phase-right = D family; growth-right / phase-left
# = gamma family (the deck's panel mapping really is crossed).
P20_D_FAMILY = [(0.3, 0.05, "#2b3fbf"), (0.6, 0.05, "#7b3f9e"), (0.9, 0.05, "#e0201b")]
P20_G_FAMILY = [(0.6, 0.03, "#2b3fbf"), (0.6, 0.06, "#7b3f9e"), (0.6, 0.09, "#e0201b")]

# === shared helper block v1 (keep byte-identical across rossby_palooza packages) ===
def set_style():
    """Apply a consistent matplotlib style (Agg backend, readable fonts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "cm",
    })
    return plt


def save_fig(fig, name, subdir=None, close=True):
    """Save a figure into figures/ (or a subdir) as PNG; return the path."""
    out_dir = FIG_DIR if subdir is None else os.path.join(FIG_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    if not name.lower().endswith(".png"):
        name += ".png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")
    return path


def fig_to_rgb(fig):
    """Rasterise a drawn matplotlib figure to an (H, W, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def write_mp4(frames, name, fps=20):
    """Write a list of RGB frames to figures/<name>.mp4 via imageio+libx264.

    Also drops a representative preview PNG next to the mp4 (middle frame),
    following the chedan_talk/ convention.  Returns the mp4 path.
    """
    import imageio.v2 as imageio
    from PIL import Image

    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    mp4_path = os.path.join(FIG_DIR, name)
    imageio.mimsave(mp4_path, frames, fps=fps, codec="libx264",
                    quality=8, macro_block_size=1)
    prev = frames[len(frames) // 2]
    prev_path = mp4_path[:-4] + "_preview.png"
    Image.fromarray(prev).save(prev_path)
    print(f"  wrote {os.path.relpath(mp4_path, HERE)}  ({len(frames)} frames)")
    print(f"  wrote {os.path.relpath(prev_path, HERE)}")
    return mp4_path
# === end shared helper block ===


# --------------------------------------------------------------------------- #
#  river.pdf p.19 dispersion relation  [RECONSTRUCTION -- see noboru_model.py]
# --------------------------------------------------------------------------- #
from noboru_model import (bank_E, forced_ratio,  # noqa: E402,F401
                          initial_condition)


# --------------------------------------------------------------------------- #
#  The dispersion relation.  DIAGNOSTIC ONLY -- the time integration never touches
#  it; it lives here rather than in the driver so that the driver contains nothing
#  that computes an answer the simulation is meant to produce.
# --------------------------------------------------------------------------- #
def dispersion_roots(kstar, D, gamma, E):
    """Both roots omega* of det M = 0 for the 3-level closure.

    [NOT IN DECK -- RECONSTRUCTION]  river.pdf p.19 prints only
    "M(omega)[psihat1'; psihat2'] = 0  =>  det M = 0"; the entries of M and the
    resulting quadratic are never written out.  What follows is derived here from
    p.9 + p.10 + p.19 by eliminating psihat1 between the bank equation and the centre
    vorticity equation, with W = -i omega*:

        centre (p.9 + p.10, using psihat1 = psihat3):
            (W + i k* + gamma) [2 psihat1 - (2 + k*^2) psihat2] + 2 i D k* psihat2 = 0
        bank (p.19):
            (W + E) psihat1 = E psihat2

        =>  (2 + k*^2) W^2 + A1 W + A0 = 0
            A1 = (2 + k*^2)(i k* + gamma + E) - 2 i D k* - 2 E
            A0 = E [k*^2 (i k* + gamma) - 2 i D k*]

    Asserted against the Dedalus IVP in 03_verify.py, which agrees to seven decimals.
    Note sigma and c both scale with E, i.e. with the assumed eps_Cf; the growth band,
    the sign of c and the psihat2/psihat1 ratios do not.
    """
    k = float(kstar)
    A2 = 2.0 + k**2
    A1 = (2.0 + k**2) * (1j * k + gamma + E) - 2j * D * k - 2.0 * E
    A0 = E * (k**2 * (1j * k + gamma) - 2j * D * k)
    return 1j * np.roots([A2, A1, A0])          # omega* = i W




def bank_mode(kstar, D, gamma, E):
    """(omega*, psihat1, psihat2) of the bank-erosion branch, psihat2 normalised to 1.

    The branch is selected as the root with the larger Im omega* (the one the IVP
    converges to).  psihat1 then follows from the p.19 bank equation.
    """
    om = dispersion_roots(kstar, D, gamma, E)
    om = om[np.argmax(om.imag)]
    W = -1j * om
    psi2_hat = 1.0 + 0j
    psi1_hat = E * psi2_hat / (W + E)            # (W + E) psihat1 = E psihat2   [p.19]
    return om, psi1_hat, psi2_hat



def bank_branch(ks, D, gamma, E):
    """omega*(k*) along the bank-erosion branch, continued from its k*->0 limit.

    Continuation matters: near k* ~ gamma the two roots of the quadratic approach each
    other and a naive "take the larger Im" pick jumps between branches.  The analytic
    small-k* limit is W ~ i E D k*/gamma, so start there and track by proximity.
    """
    ks = np.asarray(ks, dtype=float)
    out = np.empty(ks.size, dtype=complex)
    prev = 1j * E * D * ks[0] / max(gamma, 1e-12)
    for i, k in enumerate(ks):
        W = -1j * dispersion_roots(float(k), D, gamma, E)
        prev = W[np.argmin(np.abs(W - prev))]
        out[i] = 1j * prev
    return out


def growth_and_phase(ks, D, gamma, E):
    """(sigma, c) along the bank branch:  sigma = Im omega*,  c = Re omega*/k*."""
    om = bank_branch(ks, D, gamma, E)
    return om.imag, om.real / np.asarray(ks, dtype=float)


def peak_zero_intercept(D, gamma, E, ks=None):
    """(k*_peak, sigma_peak, k*_zero, c(k*->0)) -- the four p.20-readable features."""
    if ks is None:
        ks = np.linspace(1e-4, 2.0, 4000)
    sig, c = growth_and_phase(ks, D, gamma, E)
    i = int(np.argmax(sig))
    kzero = np.nan
    for j in range(i, len(ks) - 1):
        if sig[j] > 0 >= sig[j + 1]:
            kzero = ks[j] + (ks[j + 1] - ks[j]) * sig[j] / (sig[j] - sig[j + 1])
            break
    return float(ks[i]), float(sig[i]), float(kzero), float(c[0])




# --------------------------------------------------------------------------- #
#  Diagnostics computed from the raw run output
# --------------------------------------------------------------------------- #
def load_run(tag):
    """Load outputs/run_<tag>.npz as a plain dict."""
    path = os.path.join(OUT_DIR, f"run_{tag}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run noboru_model.py first")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def zeta2_of(run):
    """zeta2'(x,t) = (psi1' + psi3' - 2 psi2')/b^2 + d_xx psi2'   [p.9]

    d_xx by spectral differentiation on the periodic x-grid, which is exact for the
    single harmonic the run contains.
    """
    b = float(run["b"])
    x, L = run["x"], float(run["L"])
    kx = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=(x[1] - x[0]))
    d_xx = np.fft.irfft(-(kx**2) * np.fft.rfft(run["psi2"], axis=-1), n=x.size, axis=-1)
    return (run["psi1"] + run["psi3"] - 2 * run["psi2"]) / b**2 + d_xx


def v2_of(run):
    """v2' = d(psi2')/dx -- the cross-channel velocity at the centre level.

    [NOT DEFINED IN DECK]  river.pdf p.16 uses v2' in the momentum-flux formula but
    never defines it; this is the standard reading v' = d(psi')/dx.
    """
    x = run["x"]
    kx = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=(x[1] - x[0]))
    return np.fft.irfft(1j * kx * np.fft.rfft(run["psi2"], axis=-1), n=x.size, axis=-1)


def momentum_flux(run):
    """The p.16 momentum-flux statement, computed exactly as the deck prints it:

        d/dy <u'v'>  ~=  -<v2' zeta2'>  =  (D gamma / 2b) <zeta2'^2>  >  0

    Returns (lhs, rhs, zeta2_sq) as functions of time, where lhs = -<v2' zeta2'>_x is
    the quantity the deck's middle expression names and rhs is the deck's own
    right-hand side.  Both are plotted; ONLY the sign is asserted, because that is all
    the deck asserts -- note the deck prints the coefficient as (D gamma/2b) whereas
    the steady-state algebra of the p.10 balance gives (gamma/2D), and the two are not
    the same number.  No u' at the banks is ever needed or invented: the deck gives no
    expression for it, and the arrows on pp.16/18/19 (u'v' > 0 top, < 0 bottom) follow
    from u'v' being odd in y for the sinuous mode.
    """
    b, D, gamma = float(run["b"]), float(run["D"]), float(run["gamma"])
    z2, v2 = zeta2_of(run), v2_of(run)
    lhs = -np.mean(v2 * z2, axis=-1)
    z2sq = np.mean(z2**2, axis=-1)
    rhs = (D * gamma / (2.0 * b)) * z2sq
    return lhs, rhs, z2sq


def fit_sigma_c(t, amp, kstar, frac=0.6):
    """(sigma, c, residual) from the late-time slope of log|amp| and unwrapped phase.

    frac sets the fit window start as a fraction of the record.  The residual is the
    max abs deviation of log|amp| from the straight-line fit over that window -- if the
    subdominant root has not yet died the fit is curved and the residual shows it,
    rather than the caller silently trusting a blended rate.
    """
    t = np.asarray(t)
    m = t >= frac * t[-1]
    a = np.asarray(amp)[m]
    tt = t[m]
    good = np.abs(a) > 0
    if good.sum() < 4:
        return np.nan, np.nan, np.inf
    tt, a = tt[good], a[good]
    la = np.log(np.abs(a))
    p = np.polyfit(tt, la, 1)
    resid = float(np.max(np.abs(la - np.polyval(p, tt))))
    om_re = -np.polyfit(tt, np.unwrap(np.angle(a)), 1)[0]
    return float(p[0]), float(om_re / kstar), resid


