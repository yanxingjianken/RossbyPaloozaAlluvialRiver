#!/usr/bin/env python3
"""Shared core for the Ikeda, Parker & Sawai (1981) bend-theory explainer.

Reference
---------
Ikeda, S., Parker, G. & Sawai, K. (1981)
"Bend theory of river meanders. Part 1. Linear development."
J. Fluid Mech. 112, 363-377.

This module is the single source of truth for the *verified* linear
bend-theory equations.  Every figure/animation script imports from here so the
mathematics is defined exactly once.  The relations below were transcribed from
the original PDF (pp. 366-370) and independently re-derived (see the module
self-test) -- an automated text extraction of the paper had garbled the
dispersion relation (it dropped the square on C_f), which the derivation caught.

The linearized bend equation (their Eq. 16), for the lateral centreline
position y(x, t) of a sinuous, erodible-bank channel, is

    y_xt + 2 C_f y_t = y_xxx - C_f (A + F^2) y_xx.                     (16)

Substituting the travelling normal mode (their Eq. 17)

    y = eps * exp(alpha0 * t) * cos(k x - omega0 t)                   (17)

and separating real/imaginary parts gives the dispersion relations (Eq. 18):

    omega0(k) = C_f k^3 (2 + A + F^2) / (k^2 + 4 C_f^2)
    c0(k)     = omega0 / k                       (downstream migration speed)
    alpha0(k) = (2 C_f^2 (A + F^2) k^2 - k^4) / (k^2 + 4 C_f^2)   (growth rate)

Instability (alpha0 > 0) for      k < k_c = sqrt(2) C_f sqrt(A + F^2)   (19)
Fastest-growing wavenumber        k_OM = beta C_f                       (20)
    with   beta^2 = 4 sqrt(1 + 0.5 (A + F^2)) - 4.                      (21)

Symbols
-------
    C_f  friction coefficient (dimensionless, ~0.01-0.03; held constant)
    A    secondary-flow / transverse-bed-slope parameter (O(1); 2.89 alluvial,
         0 for a laterally flat / incised bed)
    F    Froude number  F = U0 / sqrt(g H0)  (subcritical, F^2 << A for alluvial)
    k    dimensionless wavenumber, k = 2 pi H0 / lambda  (H0 = reach-mean depth)
    alpha0 growth rate, omega0 migration frequency, c0 = omega0/k celerity

A convenient rescaling kappa = k / C_f makes the dispersion curves
C_f-independent (friction only sets the physical scale):

    alpha0 / C_f^2 = (2 (A+F^2) kappa^2 - kappa^4) / (kappa^2 + 4)
    omega0 / C_f^2 = kappa^3 (2 + A + F^2)          / (kappa^2 + 4)
    c0     / C_f   = kappa^2 (2 + A + F^2)           / (kappa^2 + 4)
    kappa_OM = beta,   kappa_c = sqrt(2 (A + F^2)).

Usage
-----
    from ikeda_lib import PARAMS, growth_rate, k_OM, save_fig, ...
    micromamba run -n fourcastnetv2 python ikeda_lib.py    # runs self-test
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Canonical parameter sets (from the paper)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """A channel parameter set for the linear theory."""
    Cf: float = 0.01     # friction coefficient
    A: float = 2.89      # secondary-flow parameter (alluvial average, Suga 1963)
    F: float = 0.30      # Froude number (subcritical)
    name: str = "alluvial"


# The two cases singled out in the paper (Sec. 5).
PARAMS = Params(Cf=0.01, A=2.89, F=0.30, name="alluvial")          # A ~ 2.89
PARAMS_INCISED = Params(Cf=0.01, A=0.0, F=0.30, name="incised")    # A = 0 (flat bed)


# --------------------------------------------------------------------------- #
#  Dispersion relations  (Eq. 18-21) -- the verified core
# --------------------------------------------------------------------------- #
def growth_rate(k, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Amplitude growth rate alpha0(k)  (Eq. 18, third relation)."""
    k = np.asarray(k, dtype=float)
    return (2.0 * Cf**2 * (A + F**2) * k**2 - k**4) / (k**2 + 4.0 * Cf**2)


def frequency(k, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Migration frequency omega0(k)  (Eq. 18, first relation)."""
    k = np.asarray(k, dtype=float)
    return Cf * k**3 * (2.0 + A + F**2) / (k**2 + 4.0 * Cf**2)


def celerity(k, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Downstream bend-migration speed c0 = omega0 / k  (Eq. 18, second)."""
    k = np.asarray(k, dtype=float)
    # written directly to stay finite as k -> 0
    return Cf * k**2 * (2.0 + A + F**2) / (k**2 + 4.0 * Cf**2)


def k_cutoff(Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Neutral (marginal) wavenumber k_c; alpha0 > 0 for 0 < k < k_c  (Eq. 19)."""
    return np.sqrt(2.0) * Cf * np.sqrt(A + F**2)


def beta_param(A=PARAMS.A, F=PARAMS.F):
    """beta in k_OM = beta C_f  (Eq. 21)."""
    return np.sqrt(4.0 * np.sqrt(1.0 + 0.5 * (A + F**2)) - 4.0)


def k_OM(Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Wavenumber of maximum growth k_OM = beta C_f  (Eq. 20)."""
    return beta_param(A, F) * Cf


def alpha_OM(Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Peak growth rate at k_OM  (Eq. 22): alpha_OM = 0.25 beta^2 k_OM^2."""
    b = beta_param(A, F)
    return 0.25 * b**2 * k_OM(Cf, A, F) ** 2


def omega_OM(Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Migration frequency at k_OM  (Eq. 23): 0.5 k_OM^2 beta (1 + beta^2/4)."""
    b = beta_param(A, F)
    return 0.5 * k_OM(Cf, A, F) ** 2 * b * (1.0 + 0.25 * b**2)


def wavelength_over_H0(k):
    """Dimensionless meander wavelength lambda / H0 = 2 pi / k  (k = 2 pi H0/lambda)."""
    k = np.asarray(k, dtype=float)
    return 2.0 * np.pi / k


# --------------------------------------------------------------------------- #
#  Near-bank velocity & the phase lag  (Eq. 7 / 10, linearized)
# --------------------------------------------------------------------------- #
def curvature_of_sine(x, k, eps=1.0):
    """Centreline curvature C(x) = -y_xx for y = eps cos(k x)  (linear, gamma=1).

    For y = eps cos(k x):  y_xx = -eps k^2 cos(k x),  so C = eps k^2 cos(k x)
    -- curvature is in phase with the centreline displacement (max at the apex).
    """
    return eps * k**2 * np.cos(k * x)


def near_bank_velocity(x, k, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F, bstar=10.0, eps=1.0):
    """Perturbation velocity u_b(x) at the outer bank for a frozen sinusoidal bend.

    Linearised near-bank momentum balance (their Eq. 10 with chi = gamma = 1):

        u_x + 2 C_f u = b* [ -C_x + C_f (A + F^2) C ],    C = curvature.

    The steady periodic response to C = eps k^2 exp(i k x) has complex amplitude

        u_hat = b* eps k^2 [ C_f (A + F^2) - i k ] / (2 C_f + i k),

    whose phase is *not* aligned with the curvature: u_b peaks a finite distance
    DOWNSTREAM of the curvature (bend-apex) maximum.  That lag is the engine of
    the whole theory -- it puts the fastest bank erosion just past the apex, so
    the bend both grows and marches downstream.

    Returns the real near-bank velocity u_b(x).
    """
    u_hat = bstar * eps * k**2 * (Cf * (A + F**2) - 1j * k) / (2.0 * Cf + 1j * k)
    return np.real(u_hat * np.exp(1j * k * x))


def phase_lag_deg(k, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Downstream phase lag (degrees) of near-bank velocity behind curvature.

    Positive value => u_b peaks a fraction (lag/360) of a wavelength downstream
    of the bend apex.
    """
    u_hat = (Cf * (A + F**2) - 1j * k) / (2.0 * Cf + 1j * k)
    # curvature has phase 0; u_b phase is arg(u_hat); lag downstream is -arg.
    return -np.degrees(np.angle(u_hat))


# --------------------------------------------------------------------------- #
#  Centreline generator (linear normal mode)  (Eq. 17)
# --------------------------------------------------------------------------- #
def centerline(x, t, k, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F, amp0=1.0):
    """Single linear normal mode y = amp0 exp(alpha0 t) cos(k x - omega0 t)."""
    a = growth_rate(k, Cf, A, F)
    w = frequency(k, Cf, A, F)
    return amp0 * np.exp(a * t) * np.cos(k * x - w * t)


def centerline_multimode(x, t, ks, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F, amps=None):
    """Superposition of several linear modes (deterministic initial condition).

    Each wavenumber grows/migrates at its own alpha0(k), omega0(k); the
    fastest-growing mode (near k_OM) emerges to dominate -- wavelength selection.
    This is an analytic initial condition for the linear PDE, not fabricated data.
    """
    ks = np.atleast_1d(ks)
    if amps is None:
        amps = np.ones_like(ks, dtype=float)
    y = np.zeros_like(x, dtype=float)
    for kj, aj in zip(ks, amps):
        y = y + centerline(x, t, kj, Cf, A, F, amp0=aj)
    return y


# --------------------------------------------------------------------------- #
#  Numerical evolution of the linear bend PDE (Eq. 16)  [added 2026-07-06]
# --------------------------------------------------------------------------- #
# Substituting y = yhat(k, t) exp(i k x) into Eq. (16),
#
#     y_xt + 2 C_f y_t = y_xxx - C_f (A + F^2) y_xx,
#
# gives   (i k + 2 C_f) yhat_t = [-i k^3 + C_f (A + F^2) k^2] yhat,   i.e.
#
#     yhat_t = s(k) yhat,   s(k) = (C_f (A + F^2) k^2 - i k^3) / (2 C_f + i k).
#
# Rationalising with (2 C_f - i k) / (k^2 + 4 C_f^2) yields
#
#     Re s(k) = (2 C_f^2 (A + F^2) k^2 - k^4) / (k^2 + 4 C_f^2) = alpha0(k)
#     Im s(k) = -C_f k^3 (2 + A + F^2)       / (k^2 + 4 C_f^2) = -omega0(k),
#
# so the symbol is a THIRD independent derivation of the dispersion relation
# (after the paper's Eq. 18 and this module's re-derivation); the self-test
# pins all three against each other.  Note s -> -k^2 as k -> inf (after
# dividing by the (i k + 2 C_f) operator the PDE is diffusion-like, NOT
# stiffly dispersive), which is why fixed-step RK4 below is adequate.


def pde_symbol(k, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Exact Fourier symbol s(k) of Eq. (16): yhat_t = s(k) yhat.

    Re s = alpha0 (growth), Im s = -omega0 (migration); see block comment.
    """
    k = np.asarray(k, dtype=complex)
    return (Cf * (A + F**2) * k**2 - 1j * k**3) / (2.0 * Cf + 1j * k)


def _rk4_gain(z):
    """Gain of one classical-RK4 step for yhat' = s yhat, with z = s dt."""
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def evolve_linear_pde(y0, L, t_out, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F,
                      dt=None):
    """Integrate Eq. (16) pseudo-spectrally on a periodic domain with RK4.

    Parameters
    ----------
    y0    : (N,) initial centreline y(x, 0) sampled on x = L*arange(N)/N.
    L     : domain length (units of H0, consistent with k = 2 pi H0/lambda).
    t_out : non-decreasing array of output times.
    dt    : fixed RK4 step; default 0.2 * 2.8 / max|s(k)| (RK4 stability
            bound with a 5x accuracy margin).  Each output interval is
            subdivided into an integer number of equal steps <= dt, so the
            integrator lands on every t_out exactly.  If you pass dt
            explicitly, keep max|s(k)| * dt inside the RK4 stability region
            (~2.8): beyond it, even *empty* high-k bins (holding only
            ~1e-16 FFT rounding noise) amplify explosively -- a genuine
            RK4 instability, not an artefact of the diagonal formulation.

    Returns
    -------
    (x, Y) with x (N,) and Y (len(t_out), N) real.

    The PDE is linear-diagonal in Fourier space, so one RK4 step multiplies
    each mode by the gain R(s dt); this is bit-for-bit classical RK4 while
    staying fully vectorised.  Asserted against `evolve_linear_pde_exact`
    (order-4 convergence) in the self-test.
    """
    y0 = np.asarray(y0, dtype=float)
    N = y0.size
    x = L * np.arange(N) / N
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    s = pde_symbol(k, Cf, A, F)
    if dt is None:
        dt = 0.2 * 2.8 / float(np.max(np.abs(s)))
    yhat = np.fft.rfft(y0).astype(complex)
    Y = np.empty((len(t_out), N), dtype=float)
    t_prev = 0.0
    for i, t in enumerate(np.asarray(t_out, dtype=float)):
        span = t - t_prev
        if span < 0:
            raise ValueError("t_out must be non-decreasing")
        if span > 0:
            n = max(1, int(np.ceil(span / dt - 1e-12)))
            yhat = yhat * _rk4_gain(s * (span / n)) ** n
        Y[i] = np.fft.irfft(yhat, n=N)
        t_prev = t
    return x, Y


def evolve_linear_pde_exact(y0, L, t_out, Cf=PARAMS.Cf, A=PARAMS.A, F=PARAMS.F):
    """Closed-form evolution yhat(k, t) = yhat(k, 0) exp(s(k) t)  (Eq. 17-18).

    Same interface as `evolve_linear_pde`; the exact oracle for validation.
    """
    y0 = np.asarray(y0, dtype=float)
    N = y0.size
    x = L * np.arange(N) / N
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    s = pde_symbol(k, Cf, A, F)
    yhat0 = np.fft.rfft(y0).astype(complex)
    Y = np.empty((len(t_out), N), dtype=float)
    for i, t in enumerate(np.asarray(t_out, dtype=float)):
        Y[i] = np.fft.irfft(yhat0 * np.exp(s * t), n=N)
    return x, Y


def spectrum(y, L):
    """One-sided amplitude spectrum (k_j, amp_j) of a real periodic sample.

    amp_j is the physical cosine amplitude of mode k_j = 2 pi j / L (DC
    halved), so y = a cos(k_j x + phi) returns amp == a at index j.
    """
    y = np.asarray(y, dtype=float)
    N = y.size
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    amp = np.abs(np.fft.rfft(y)) * 2.0 / N
    amp[0] /= 2.0
    return k, amp


# --------------------------------------------------------------------------- #
#  Plot styling & saving
# --------------------------------------------------------------------------- #
# A small, consistent, colourblind-friendly palette used across all figures.
COLORS = {
    "water": "#2c7fb8",
    "water_fill": "#c7e0f0",
    "channel": "#08519c",
    "erosion": "#d7301f",     # outer-bank erosion (red)
    "deposition": "#d9b38c",  # point bar / inner-bank deposition (tan)
    "curvature": "#6a51a3",   # purple
    "velocity": "#e6820a",    # orange
    "growth": "#238b45",      # green (unstable band)
    "decay": "#969696",       # grey (stable)
    "apex": "#252525",
    "bank": "#7f5539",
}


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


# --------------------------------------------------------------------------- #
#  Animation helper: render matplotlib frames to mp4 (imageio / libx264)
# --------------------------------------------------------------------------- #
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
    # preview still (middle frame)
    prev = frames[len(frames) // 2]
    prev_path = mp4_path[:-4] + "_preview.png"
    Image.fromarray(prev).save(prev_path)
    print(f"  wrote {os.path.relpath(mp4_path, HERE)}  ({len(frames)} frames)")
    print(f"  wrote {os.path.relpath(prev_path, HERE)}")
    return mp4_path


# --------------------------------------------------------------------------- #
#  Self-test: re-derive & assert the paper's headline numbers
# --------------------------------------------------------------------------- #
def _self_test():
    print("Ikeda, Parker & Sawai (1981) -- linear bend theory, verified core")
    print("-" * 68)

    # Alluvial case: A = 2.89, F^2 << A  =>  beta ~ 1.50, and the Eq.(24) numbers.
    al = Params(Cf=0.01, A=2.89, F=0.0)
    beta = beta_param(al.A, al.F)
    print(f"Alluvial (A={al.A}, F=0):  beta = {beta:.4f}   (paper Eq.24a: 1.50)")
    assert abs(beta - 1.50) < 0.01, "beta should be ~1.50 for the alluvial case"

    kom = k_OM(al.Cf, al.A, al.F)
    print(f"  k_OM       = {kom:.5f}   (= {kom/al.Cf:.3f} C_f; paper: 1.50 C_f)")
    assert abs(kom / al.Cf - 1.50) < 0.01

    # alpha_OM = 0.564 k_OM^2 ; omega_OM = 1.17 k_OM^2 ; c_OM = 1.17 k_OM
    aom = alpha_OM(al.Cf, al.A, al.F)
    wom = omega_OM(al.Cf, al.A, al.F)
    com = wom / kom
    print(f"  alpha_OM   = {aom/kom**2:.4f} k_OM^2   (paper Eq.24b: 0.564)")
    print(f"  omega_OM   = {wom/kom**2:.4f} k_OM^2   (paper Eq.24c: 1.17)")
    print(f"  c_OM       = {com/kom:.4f} k_OM       (paper Eq.25 : 1.17)")
    assert abs(aom / kom**2 - 0.564) < 0.01
    assert abs(wom / kom**2 - 1.17) < 0.02
    assert abs(com / kom - 1.17) < 0.02

    # Direct dispersion formula must agree with the k_OM expansion at k = k_OM.
    assert abs(float(growth_rate(kom, al.Cf, al.A, al.F)) - aom) < 1e-12
    assert abs(float(frequency(kom, al.Cf, al.A, al.F)) - wom) < 1e-12

    # Growth rate must vanish at the neutral wavenumber and be the maximum at k_OM.
    kc = k_cutoff(al.Cf, al.A, al.F)
    print(f"  k_c (neutral) = {kc:.5f}  (= {kc/al.Cf:.3f} C_f)")
    assert abs(float(growth_rate(kc, al.Cf, al.A, al.F))) < 1e-12, "alpha(k_c) must be 0"
    ktest = np.linspace(1e-4, kc, 2001)
    assert np.argmax(growth_rate(ktest, al.Cf, al.A, al.F)) > 0
    assert abs(ktest[np.argmax(growth_rate(ktest, al.Cf, al.A, al.F))] - kom) < 2e-4

    # Celerity is positive across the whole band => bends ALWAYS migrate downstream.
    assert np.all(celerity(ktest, al.Cf, al.A, al.F) > 0), "c0 must be > 0 (downstream)"
    print("  c0(k) > 0 for all k  ->  bends always migrate downstream. OK")

    # Incised limit (A=0, small F): k_OM ~ C_f F  (Eq. 28).
    F = 0.1
    inc = Params(Cf=0.01, A=0.0, F=F)
    print(f"Incised (A=0, F={F}):  k_OM = {k_OM(inc.Cf, inc.A, inc.F)/inc.Cf:.4f} C_f "
          f"(paper Eq.28: ~C_f F = {F} C_f)")
    assert abs(k_OM(inc.Cf, inc.A, inc.F) / inc.Cf - F) < 0.02

    # Phase lag is a genuine downstream lag (0 < lag < 90 deg) at k_OM.
    lag = float(phase_lag_deg(kom, al.Cf, al.A, al.F))
    print(f"  near-bank velocity lags curvature by {lag:.1f} deg at k_OM "
          f"(~{lag/360:.2f} wavelength downstream)")
    assert 0.0 < lag < 90.0

    _pde_self_test()

    print("-" * 68)
    print("All self-tests passed.")


def _pde_self_test():
    """Self-test of the Eq.-16 PDE block: symbol, integrator order, selection."""
    print("PDE block (Eq. 16, pseudo-spectral) -- symbol / integrator / selection")

    # (1) The Fourier symbol is a third derivation of the dispersion relation.
    for p, tol in ((PARAMS, 1e-14), (Params(Cf=1.0, A=2.89, F=0.30, name="rescaled"), 1e-12)):
        kk = np.linspace(1e-4, 8.0 * p.Cf, 400)
        s = pde_symbol(kk, p.Cf, p.A, p.F)
        d_re = np.max(np.abs(s.real - growth_rate(kk, p.Cf, p.A, p.F)))
        d_im = np.max(np.abs(s.imag + frequency(kk, p.Cf, p.A, p.F)))
        print(f"  symbol vs Eq.18 ({p.name}):  max|Re s - alpha0| = {d_re:.2e}, "
              f"max|Im s + omega0| = {d_im:.2e}")
        assert d_re < tol and d_im < tol, "pde_symbol must reproduce Eq. 18"
    assert abs(complex(pde_symbol(0.0))) < 1e-30, "s(0) = 0 (mean is conserved)"

    # Shared selection grid: mode j = 8 lands exactly on k_OM.
    kom = k_OM()
    L = 16.0 * np.pi / kom          # k_j = 2 pi j / L  =>  k_8 = k_OM
    N = 256
    x = L * np.arange(N) / N
    kj = lambda j: 2.0 * np.pi * j / L

    # (2) RK4 (auto dt) vs the exact oracle, three modes, one e-folding.
    y0 = sum(np.cos(kj(j) * x) for j in (5, 8, 11))
    T1 = 1.0 / alpha_OM()
    _, Yn = evolve_linear_pde(y0, L, [T1])
    _, Ye = evolve_linear_pde_exact(y0, L, [T1])
    rel = np.max(np.abs(Yn - Ye)) / np.max(np.abs(Ye))
    print(f"  RK4 vs exact (auto dt, T = 1/alpha_OM):  rel err = {rel:.2e}")
    assert rel < 1e-8, "RK4 with the default step must track the exact solution"

    # (3) Order check: halving dt cuts the single-mode error ~16x (order 4).
    # Run on a SMALL grid (N=32) so the coarse explicit dt keeps even the
    # highest grid mode inside the RK4 stability region (see evolve docstring:
    # outside it, empty high-k bins amplify FFT rounding noise explosively --
    # the first run of this test demonstrated exactly that failure mode).
    N32 = 32
    x32 = L * np.arange(N32) / N32
    y1 = np.cos(kj(8) * x32)
    smag = abs(complex(pde_symbol(kom)))
    dt0 = 0.2 / smag
    kmax32 = 2.0 * np.pi * (N32 // 2) / L
    assert abs(complex(pde_symbol(kmax32))) * dt0 < 2.8, "order-test grid must be RK4-stable"
    T2 = 8.0 * dt0
    _, Ye2 = evolve_linear_pde_exact(y1, L, [T2])
    errs = [np.max(np.abs(evolve_linear_pde(y1, L, [T2], dt=d)[1] - Ye2))
            for d in (dt0, dt0 / 2.0)]
    ratio = errs[0] / errs[1]
    print(f"  dt-halving error ratio = {ratio:.2f}   (classical RK4: ~16)")
    assert 14.0 <= ratio <= 18.0, "integrator must converge at 4th order"

    # (4) Wavelength selection from a broad-band analytic IC (modes 3..14,
    # equal amplitudes, brackets k_OM and extends past k_c): the k_OM mode
    # must win, and every mode's final/initial amplitude ratio must match the
    # closed-form exp(alpha0 T) factor -- machine-checkable selection.
    modes = list(range(3, 15))
    y2 = sum(1e-3 * np.cos(kj(j) * x) for j in modes)
    T3 = 6.0 / alpha_OM()
    _, Ys = evolve_linear_pde(y2, L, [T3])
    ks, amp = spectrum(Ys[-1], L)
    jwin = int(np.argmax(amp[1:])) + 1
    print(f"  selection: argmax mode j = {jwin} (k = {ks[jwin]:.5f}; k_OM = {kom:.5f})")
    assert jwin == 8, "fastest-growing mode k_OM must dominate the spectrum"
    a8 = growth_rate(kj(8))
    worst = 0.0
    for j in modes:
        pred = np.exp((float(growth_rate(kj(j))) - float(a8)) * T3)
        got = amp[j] / amp[8]
        worst = max(worst, abs(got - pred) / pred)
    print(f"  selection amplitude ratios vs exp((alpha_j - alpha_8) T): "
          f"worst rel err = {worst:.2e}")
    assert worst < 1e-6, "spectral ratios must match the normal-mode prediction"


if __name__ == "__main__":
    _self_test()
