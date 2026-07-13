#!/usr/bin/env python3
"""Shared core for the Rossby-Palooza vorticity-meander theory (6/30 deck).

REBUILD v2 (2026-07-12) -- the v1 package was torn down and rebuilt from
scratch: every equation re-derived by hand from the deck, the deck-p.8 pins
re-digitized independently at 300 dpi (they agree with v1's reads within the
stated reading error), and the friction closure -- v1's `FLAG_FRICTION`, the
flagged prime suspect for the growth-peak discrepancy -- is now implemented
in TWO variants so their consequences can be compared quantitatively:

    friction='rayleigh' : Rayleigh damping of perturbation vorticity at
                          nondimensional rate gamma (the deck-literal
                          reading; what v1 shipped).
    friction='momentum' : curl of the linearised bottom drag
                          -(C_f/H)|u|u  -- the closure the St Venant
                          tradition (Ikeda et al. 1981, their eq. 3b) uses;
                          its streamwise component is damped twice as fast
                          as the cross-stream one (the "factor 2" of the
                          |u|u linearisation).

Reference
---------
"Alluvial Rivers & Jet streams", Rossby Palooza group meeting deck,
2026-06-30 (literature/Rossby_Palooza_meet_0630.pdf).  All equations cite
the deck's printed page footer ("Rossby Palooza Page N").  This package
REPRODUCES IN-PROGRESS GROUP WORK: where the deck is silent the assumption
is a named FLAG_* constant, and quantified disagreements are codified in the
self-test rather than hidden (see "Status" below).

The model (deck pp. 4-7)
------------------------
Slow alluvial river (0.1 < Fr < 0.3, p. 3) => free surface is slaved
(rigid-lid) and the depth-averaged flow obeys 2-D vorticity dynamics.
Base state: parabolic jet in a channel of half-width b,

    ubar(y) = U0 + (Delta/b^2)(b^2 - y^2)                        (p. 4)

whose vorticity  zetabar = -ubar_y = (2 Delta/b^2) y  has CONSTANT positive
cross-channel gradient  zetabar_y = -ubar_yy = 2 Delta/b^2 -- the channel
analogue of the planetary beta (hence "Rossby" Palooza).  Linearised
perturbation vorticity equation (continuum form; psi' streamfunction,
zeta' = lap psi'):

    (d/dt + ubar d/dx) zeta' + (2 Delta/b^2) psi'_x = F[psi'],

with F the friction closure (above).  [FLAG_BASE: the parabolic jet is
taken as externally maintained; friction acts on the perturbation only.]

Nondimensionalisation (deck p. 5 box): lengths by b, speeds by U0+Delta
(channel-centre speed), time by b/(U0+Delta)  [FLAG_TSCALE: the deck does
not print its time unit; this choice makes the centre advection speed
exactly 1 and reproduces the printed gamma = C_f b/H];
k* = k b,  D = Delta/(U0+Delta),  gamma = C_f b / H.

Friction closures, nondimensional (normal mode psihat(y) e^{i(k*x - om*t)}):

    rayleigh :  Fhat = -gamma (psihat'' - k*^2 psihat)
    momentum :  Fhat = -gamma [ 2 ubar psihat'' + 2 ubar_y psihat'
                                - ubar k*^2 psihat ]
    (curl of  F' = -(C_f/H)(2 ubar u', ubar v'), u' = -psi_y, v' = psi_x)

Three-level closure (deck p. 4): psi_1(y=+b), psi_2(0), psi_3(-b), sinuous
symmetry psihat_1 = psihat_3; centre vorticity by the 3-point Laplacian

    zetahat_2 = 2 psihat_1 - (2 + k*^2) psihat_2        (b = 1 units).

Centre equation (W = -i om*):

  rayleigh: (W + i k* + gamma) zetahat_2 + 2 i D k* psihat_2 = 0
  momentum: (W + i k*) zetahat_2 + 2 i D k* psihat_2
            + gamma [2(2 psihat_1 - 2 psihat_2) - k*^2 psihat_2] = 0

Bank erosion (p. 7, taken literally -- no advection term at the bank):

    d psi_1'/dt = (eps C_f U0 / b)(psi_2' - psi_1')
    =>  (W + E) psihat_1 = E psihat_2,
    E = eps C_f U0/(U0+Delta) = ECOEF[friction] (1 - D)
    [FLAG_EPS: ECOEF = eps C_f, calibrated from the deck's own p.8
    phase-speed intercepts -- see below].

det M(om*) = 0 is a QUADRATIC in W (function `dispersion_coeffs`):

    A2 W^2 + A1 W + A0 = 0,   A2 = 2 + k*^2,
    A0 = E [k*^2 (i k* + gamma) - 2 i D k*]          (BOTH closures),
    A1(rayleigh) = (2+k*^2)(i k* + gamma + E) - 2 i D k* - 2E,
    A1(momentum) = A1(rayleigh) + 2 gamma.

The two closures differ by ONE term: +2 gamma in A1.  Consequences:

    k* -> 0 bank-mode phase speed   c0 = -E D / gamma      (rayleigh)
                                    c0 = -E D / (2 gamma)   (momentum)

so the p.8 phase-speed intercepts are fit EQUALLY WELL by
(rayleigh, ECOEF = 0.5) and (momentum, ECOEF = 1.0): the intercepts alone
cannot distinguish the closures (an exact E <-> 2E degeneracy at k* -> 0).
The growth-rate PEAKS break the degeneracy -- see `07_friction_closures.py`
and the README.

Verified consequences (all asserted in the self-test)
-----------------------------------------------------
1. Forced steady response (om = 0), gamma = 0, both closures:
       psihat_2/psihat_1 = 2/(2 + k*^2 - 2D)
   reproducing the deck p. 5 box EXACTLY: |psihat_2| > |psihat_1| iff
   k*^2 < 2D (equality at k*^2 = 2D to 1e-12), and no pole for k* > 0 when
   D < 1 ("no resonance ... even in the inviscid limit").
2. Rigid banks (eps = 0) are NEUTRAL: the parabolic jet has no inflection
   point; all modes satisfy Im om* <= 1e-12 (both closures).
3. Erodible banks destabilise: small-E bank mode at gamma = 0 has
       sigma/E -> (2D - k*^2)/(2 + k*^2 - 2D)   (growth for k*^2 < 2D),
   and with friction the k* -> 0 phase speed is c* -> -E D/gamma_eff
   (UPSTREAM -- the deck's headline, p. 7), gamma_eff = gamma resp. 2 gamma.
4. The closure quadratics obey A0(ray) = A0(mom) and
   A1(mom) - A1(ray) = 2 gamma exactly (independent re-derivation check).
5. The deck's own p. 8 phase intercepts obey c0 gamma_eff/(D(1-D)) = -ECOEF
   for a single constant per closure: 0.5 (rayleigh) / 1.0 (momentum),
   per-curve spread < 3% (function `calibrate_ecoef`).
6. p. 4 closure = discretisation: the N = 3 channel eigenproblem contains
   the 2x2 closure exactly (1e-10), plus the varicose bank mode om = -iE
   that the psihat_1 = psihat_3 symmetry removes -- BOTH closures.
7. N-point growth converged by N ~ 51 (asserted 51 vs 101), both closures.

Status -- codified discrepancy (deck p. 8 growth PEAKS)
-------------------------------------------------------
rayleigh (deck-literal, calibrated ECOEF = 0.5): growth-peak heights are a
factor 3.18-4.42 LOWER than the deck's plotted peaks, peak k* shifted
+0.09-0.19 right (v1's finding, reproduced by this rebuild from scratch).
momentum (calibrated ECOEF = 1.0): factor 2.28-3.22 LOWER, shift
+0.13-0.22 -- switching to the Ikeda-consistent drag closes roughly one
third of the gap while fitting the phase intercepts exactly as well, so
the friction closure alone does NOT resolve the peak discrepancy.
Remaining suspects: the p. 6 "pressure drag due to wavy banks" absent from
the p. 7 bank equation, and the growth-axis normalisation (FLAG_TSCALE).
Resolution is NOT the cause either way (N-point solver converges to the
closure peaks by N ~ 11).

Usage
-----
    from vorticity_lib import Params, growth_curve, dispersion_roots, ...
    micromamba run -n fourcastnetv2 python vorticity_lib.py   # self-test
"""
from __future__ import annotations

import csv
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

# Calibrated erosion coefficient eps*C_f per friction closure [FLAG_EPS],
# from the six deck-p.8 phase-speed intercepts via c0 = -E D/gamma_eff with
# E = ECOEF (1 - D) and gamma_eff = gamma (rayleigh) / 2 gamma (momentum).
ECOEF = {"rayleigh": 0.5, "momentum": 1.0}

FRICTIONS = ("rayleigh", "momentum")


def _gamma_eff_factor(friction):
    """k*->0 effective-friction factor: c0 = -E D/(factor * gamma)."""
    if friction == "rayleigh":
        return 1.0
    if friction == "momentum":
        return 2.0
    raise ValueError(f"unknown friction closure {friction!r}")


# --------------------------------------------------------------------------- #
#  Parameters and base state (deck pp. 4-5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """Deck parameter set; velocities in units of U0+Delta, lengths of b."""
    D: float = 0.5           # Delta / (U0 + Delta)
    gamma: float = 0.05      # C_f b / H
    name: str = "deck"
    friction: str = "rayleigh"

    @property
    def E(self):
        """Nondimensional bank-erosion rate E = ECOEF[friction] (1 - D)."""
        return ECOEF[self.friction] * (1.0 - self.D)


# The six parameter sets of the deck p. 8 figure (deck-literal rayleigh).
DECK_D_FAMILY = tuple(Params(D=d, gamma=0.05, name=f"D={d}") for d in (0.3, 0.6, 0.9))
DECK_G_FAMILY = tuple(Params(D=0.6, gamma=g, name=f"g={g}") for g in (0.03, 0.06, 0.09))


def u_profile(y, D):
    """Base jet u(y) = (1-D) + D (1 - y^2) in U0+Delta units (deck p. 4)."""
    y = np.asarray(y, dtype=float)
    return (1.0 - D) + D * (1.0 - y**2)


def u_profile_y(y, D):
    """Cross-channel shear ubar_y = -2 D y of the base jet."""
    y = np.asarray(y, dtype=float)
    return -2.0 * D * y


def zeta_gradient(D):
    """Background vorticity gradient -u_yy = 2D (the channel 'beta', p. 4)."""
    return 2.0 * D


# --------------------------------------------------------------------------- #
#  Forced steady response (deck pp. 5-6)
# --------------------------------------------------------------------------- #
def forced_response(kstar, D, gamma=0.0, friction="rayleigh"):
    """psihat_2 / psihat_1 for the steady (om = 0) forced problem.

    rayleigh: 2 (i k* + gamma) / [(2 + k*^2)(i k* + gamma) - 2 i D k*]
    momentum: 2 (i k* + 2 gamma) / [i k*(2 + k*^2 - 2D) + gamma (4 + k*^2)]
    Both reduce at gamma = 0 to 2/(2 + k*^2 - 2D) -- the p. 5 box identity.
    """
    k = np.asarray(kstar, dtype=complex)
    if friction == "rayleigh":
        return 2.0 * (1j * k + gamma) / ((2.0 + k**2) * (1j * k + gamma) - 2j * D * k)
    if friction == "momentum":
        return 2.0 * (1j * k + 2.0 * gamma) / (
            1j * k * (2.0 + k**2 - 2.0 * D) + gamma * (4.0 + k**2))
    raise ValueError(f"unknown friction closure {friction!r}")


def reynolds_stress_factor(kstar, D, gamma):
    """In-phase part of -v2' zeta2' per zeta2'^2 (deck p. 6 statement).

    From the steady rayleigh centre balance (i k* + gamma) zetahat_2 =
    -2 i D k* psihat_2:  -mean(v2' zeta2')/mean(zeta2'^2) = gamma/(2D) > 0
    -- momentum leaves the centre for the banks whenever friction acts.
    (The deck prints the factor as D gamma/2b; the sign/positivity statement
    is what is asserted.  Deck-literal rayleigh closure.)
    """
    return gamma / (2.0 * D)


# --------------------------------------------------------------------------- #
#  Bank-erosion dispersion relation (deck p. 7): quadratic in W = -i om*
# --------------------------------------------------------------------------- #
def dispersion_coeffs(kstar, D, gamma, E, friction="rayleigh"):
    """Coefficients (A2, A1, A0) of A2 W^2 + A1 W + A0 = 0, W = -i om*.

    Derived by eliminating psihat_1 between the bank equation
    (W + E) psihat_1 = E psihat_2 and the centre vorticity equation (see
    module docstring for both closures).  The closures share A2 and A0 and
    differ by exactly +2 gamma in A1 (asserted in the self-test).
    """
    k = kstar
    A2 = 2.0 + k**2
    A1 = (2.0 + k**2) * (1j * k + gamma + E) - 2j * D * k - 2.0 * E
    if friction == "momentum":
        A1 = A1 + 2.0 * gamma
    elif friction != "rayleigh":
        raise ValueError(f"unknown friction closure {friction!r}")
    A0 = E * (k**2 * (1j * k + gamma) - 2j * D * k)
    return A2, A1, A0


def dispersion_roots(kstar, D, gamma, E, friction="rayleigh"):
    """Both roots om* = i W of det M = 0 at one wavenumber."""
    W = np.roots(dispersion_coeffs(kstar, D, gamma, E, friction))
    return 1j * W


def bank_branch(ks, D, gamma, E, friction="rayleigh"):
    """The bank-erosion mode, continued in k* from its analytic k* -> 0 limit
    W ~ i E D k*/gamma_eff (proximity continuation avoids branch mix-ups
    where the bank and centre roots interact near k* ~ gamma)."""
    ks = np.asarray(ks, dtype=float)
    W = np.empty(ks.size, dtype=complex)
    geff = _gamma_eff_factor(friction) * gamma
    prev = 1j * E * D * ks[0] / max(geff, 1e-12)
    for i, k in enumerate(ks):
        rr = np.roots(dispersion_coeffs(k, D, gamma, E, friction))
        prev = rr[np.argmin(np.abs(rr - prev))]
        W[i] = prev
    return 1j * W          # om* along the branch


def growth_curve(ks, p: Params):
    """(sigma, c) of the bank mode along ks: sigma = Im om*, c = Re om*/k*."""
    om = bank_branch(ks, p.D, p.gamma, p.E, p.friction)
    return om.imag, om.real / np.asarray(ks, dtype=float)


def kstar_peak(p: Params, ks=None):
    """(k*_peak, sigma_peak, k*_zero) of the bank mode."""
    if ks is None:
        ks = np.linspace(1e-3, 2.0, 2000)
    sig, _ = growth_curve(ks, p)
    i = int(np.argmax(sig))
    kz = np.nan
    for j in range(i, len(ks) - 1):
        if sig[j] > 0 >= sig[j + 1]:
            kz = ks[j] + (ks[j + 1] - ks[j]) * sig[j] / (sig[j] - sig[j + 1])
            break
    return float(ks[i]), float(sig[i]), float(kz)


# --------------------------------------------------------------------------- #
#  N-point channel eigenproblem (continuum generalisation; deck p. 8 goals)
# --------------------------------------------------------------------------- #
def channel_matrices(N, kstar, D, gamma, E, friction="rayleigh"):
    """Generalised eigenproblem A phi = om* B phi on y in [-1, 1], N odd.

    Interior rows (rayleigh):
        (-i om* + i k* u(y) + gamma)(phi'' - k*^2 phi) + 2 i D k* phi = 0
    Interior rows (momentum):
        (-i om* + i k* u(y))(phi'' - k*^2 phi) + 2 i D k* phi
        + gamma [2 u phi'' + 2 u_y phi' - u k*^2 phi] = 0
    (centred finite differences).  Bank rows (literal deck p. 7, nonlocal to
    the centre node):  -i om* phi_b = E (phi_center - phi_b).
    At N = 3 (h = 1 = b) this IS the 2x2 closure -- asserted in the
    self-test for BOTH closures to 1e-10.
    """
    y = np.linspace(-1.0, 1.0, N)
    h = y[1] - y[0]
    u = u_profile(y, D)
    uy = u_profile_y(y, D)
    ic = N // 2
    A = np.zeros((N, N), dtype=complex)
    B = np.zeros((N, N), dtype=complex)
    for j in range(1, N - 1):
        D2 = np.zeros(N, dtype=complex)
        D2[j - 1] += 1.0 / h**2
        D2[j] += -2.0 / h**2
        D2[j + 1] += 1.0 / h**2
        L = D2.copy()
        L[j] += -kstar**2
        if friction == "rayleigh":
            A[j] = (1j * kstar * u[j] + gamma) * L
        elif friction == "momentum":
            D1 = np.zeros(N, dtype=complex)
            D1[j - 1] += -1.0 / (2.0 * h)
            D1[j + 1] += 1.0 / (2.0 * h)
            fric = 2.0 * u[j] * D2 + 2.0 * uy[j] * D1
            fric[j] += -u[j] * kstar**2
            A[j] = 1j * kstar * u[j] * L + gamma * fric
        else:
            raise ValueError(f"unknown friction closure {friction!r}")
        A[j, j] += 2j * D * kstar
        B[j] = 1j * L
    for b_ in (0, N - 1):
        A[b_, b_] = -E
        A[b_, ic] = E
        B[b_, b_] = -1j
    return A, B


def channel_modes(N, kstar, D, gamma, E, friction="rayleigh"):
    """All finite eigenvalues om* (and eigenvectors) of the N-point GEP."""
    from scipy.linalg import eig
    A, B = channel_matrices(N, kstar, D, gamma, E, friction)
    w, V = eig(A, B)
    ok = np.isfinite(w) & (np.abs(w) < 1e6)
    return w[ok], V[:, ok]


# --------------------------------------------------------------------------- #
#  Digitised deck p. 8 pins & the ECOEF calibration
# --------------------------------------------------------------------------- #
def load_deck_pins():
    """The six curves' (k*_peak, sigma_peak, k*_zero, c0) read off deck p. 8."""
    path = os.path.join(DATA_DIR, "deck_p8_pins.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(x for x in f if not x.startswith("#")):
            rows.append({k: (v if k == "family" else float(v))
                         for k, v in r.items()})
    return rows


def calibrate_ecoef(friction, pins=None):
    """Per-curve eps*C_f implied by the p.8 intercepts, and their mean.

    c0 = -E D/(gamma_eff) with E = eps C_f (1-D) and gamma_eff =
    _gamma_eff_factor(friction) * gamma  =>
    eps C_f = -c0 * gamma_eff / (D (1-D)).  A single constant fitting all
    six curves is the FLAG_EPS calibration; the self-test asserts the
    per-curve spread is < 3% for both closures.
    """
    if pins is None:
        pins = load_deck_pins()
    fac = _gamma_eff_factor(friction)
    vals = [-r["c0"] * fac * r["gamma"] / (r["D"] * (1.0 - r["D"])) for r in pins]
    return np.array(vals), float(np.mean(vals))


# --------------------------------------------------------------------------- #
#  Plot styling & saving
# --------------------------------------------------------------------------- #
COLORS = {
    "jet": "#2c7fb8",
    "water_fill": "#c7e0f0",
    "bank": "#7f5539",
    "erosion": "#d7301f",
    "psi1": "#08519c",        # bank streamlines (deck blue)
    "psi2": "#d7301f",        # centre streamline (deck red)
    "growth": "#238b45",
    "decay": "#969696",
    "upstream": "#6a51a3",
    "deckpin": "#252525",
    "momentum": "#e6550d",    # momentum-closure overlay
}

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
#  Self-test
# --------------------------------------------------------------------------- #
def _self_test():
    print("Rossby-Palooza vorticity-meander theory (6/30 deck) -- rebuild v2")
    print("-" * 78)

    # Base state (deck p. 4).
    for D in (0.3, 0.6, 0.9):
        assert abs(float(u_profile(1.0, D)) - (1.0 - D)) < 1e-14
        assert abs(float(u_profile(-1.0, D)) - (1.0 - D)) < 1e-14
        assert abs(float(u_profile(0.0, D)) - 1.0) < 1e-14
        y = 0.37
        fd = -(u_profile(y + 1e-5, D) - 2 * u_profile(y, D)
               + u_profile(y - 1e-5, D)) / 1e-10
        assert abs(fd - zeta_gradient(D)) < 1e-3, "zetabar_y = -u_yy = 2D"
        fd1 = (u_profile(y + 1e-6, D) - u_profile(y - 1e-6, D)) / 2e-6
        assert abs(fd1 - float(u_profile_y(y, D))) < 1e-6, "ubar_y = -2Dy"
    print("base state: parabolic jet, constant vorticity gradient 2D. OK")

    # Closure quadratics: A2, A0 shared; A1 differs by exactly +2 gamma.
    for (k, D, g, E) in ((0.3, 0.5, 0.05, 0.2), (0.9, 0.7, 0.08, 0.15),
                         (1.7, 0.2, 0.02, 0.4), (0.05, 0.9, 0.09, 0.1)):
        r2, r1, r0 = dispersion_coeffs(k, D, g, E, "rayleigh")
        m2, m1, m0 = dispersion_coeffs(k, D, g, E, "momentum")
        assert r2 == m2 and r0 == m0, "A2, A0 must be closure-independent"
        assert abs((m1 - r1) - 2.0 * g) < 1e-15, "A1(mom) - A1(ray) = 2 gamma"
    print("closure quadratics: A1(momentum) = A1(rayleigh) + 2 gamma exact. OK")

    # Deck p. 5 box: |psi2/psi1| > 1 iff k*^2 < 2D (gamma = 0), equality at
    # k*^2 = 2D; finite for all k* > 0, D < 1 (no resonance).  Both closures
    # coincide at gamma = 0.
    for D in (0.2, 0.5, 0.8):
        ks = np.linspace(1e-3, 3.0, 1500)
        for fr in FRICTIONS:
            amp = np.abs(forced_response(ks, D, 0.0, fr))
            assert np.all((amp > 1.0) == (ks**2 < 2 * D)), "p.5 box identity"
            assert abs(float(np.abs(forced_response(np.sqrt(2 * D), D, 0.0, fr))) - 1.0) < 1e-12
            assert np.all(np.isfinite(amp)), "no resonance for k* > 0"
        dif = np.max(np.abs(forced_response(ks, D, 0.0, "rayleigh")
                            - forced_response(ks, D, 0.0, "momentum")))
        assert dif < 1e-14, "closures must coincide at gamma = 0"
    print("deck p.5 boxed identity |psi2|>|psi1| iff k*^2<2D: exact, both closures. OK")

    # Rigid banks are neutral (no inflection point): eps = 0, both closures.
    for p in DECK_D_FAMILY + DECK_G_FAMILY:
        for k in (0.05, 0.3, 1.0, 1.8):
            for fr in FRICTIONS:
                om = dispersion_roots(k, p.D, p.gamma, 0.0, fr)
                assert np.all(om.imag <= 1e-12), "rigid banks must be neutral/damped"
                om0 = dispersion_roots(k, p.D, 0.0, 0.0, fr)
                assert np.all(om0.imag <= 1e-12)
    print("rigid banks (eps=0): all modes neutral or damped, both closures. OK")

    # Small-E analytics: growth band and upstream propagation per closure.
    D, g, Etiny = 0.6, 0.05, 1e-8
    for fr in FRICTIONS:
        for k in (0.3, 0.6, 0.9):
            om = bank_branch([k], D, 0.0, Etiny, fr)[0]
            pred = Etiny * (2 * D - k**2) / (2.0 + k**2 - 2 * D)
            assert abs(om.imag - pred) < 1e-5 * abs(pred) + 1e-16, \
                "sigma/E -> (2D - k*^2)/(2 + k*^2 - 2D) at gamma = 0"
        om = bank_branch([1e-4], D, g, Etiny, fr)[0]
        c = om.real / 1e-4
        c_pred = -Etiny * D / (_gamma_eff_factor(fr) * g)
        assert abs(c - c_pred) < 1e-3 * abs(c_pred), \
            f"c -> -E D/({_gamma_eff_factor(fr):.0f} gamma) as k* -> 0 ({fr})"
    print("small-E analytics: growth band k*^2<2D; c0 = -E D/gamma (ray), "
          "-E D/(2 gamma) (mom). OK")

    # N = 3 GEP contains EXACTLY the 2x2 closure (its sinuous pair) plus the
    # varicose bank mode om = -iE that the psihat_1 = psihat_3 symmetry
    # removes from the deck's closure -- both closures.
    for fr in FRICTIONS:
        for (Dv, gv) in ((0.4, 0.04), (0.7, 0.08)):
            E = ECOEF[fr] * (1.0 - Dv)
            for k in (0.2, 0.7, 1.4):
                om2 = dispersion_roots(k, Dv, gv, E, fr)
                om3, _ = channel_modes(3, k, Dv, gv, E, fr)
                assert om3.size == 3, "N=3 has 2 sinuous + 1 varicose mode"
                for o in om2:
                    assert np.min(np.abs(om3 - o)) < 1e-10, \
                        f"N=3 GEP must contain the closure ({fr})"
                assert np.min(np.abs(om3 - (-1j * E))) < 1e-10, \
                    "varicose bank mode om = -iE"
    print("N=3 GEP = 2x2 sinuous closure + varicose om=-iE (1e-10), both closures. OK")

    # Resolution convergence of the most-unstable growth (N-point solver).
    for fr in FRICTIONS:
        p = Params(0.6, 0.05, friction=fr)
        sig = {}
        for N in (51, 101):
            s = max(channel_modes(N, 0.4, p.D, p.gamma, p.E, fr)[0].imag)
            sig[N] = float(s)
        assert abs(sig[101] - sig[51]) <= 0.01 * abs(sig[101]) + 1e-12, \
            f"N-point growth converged ({fr})"
        print(f"N-point convergence ({fr:>8}): sigma(51) = {sig[51]:.5f}, "
              f"sigma(101) = {sig[101]:.5f}. OK")

    # lambda/2b axis-marker arithmetic (deck p. 8 magenta annotations).
    for kstar, lam_over_2b in ((0.25, 4 * np.pi), (0.5, 2 * np.pi), (1.0, np.pi)):
        assert abs(np.pi / kstar - lam_over_2b) < 1e-12
    # gamma -> 0 continuity of the bank branch.
    for fr in FRICTIONS:
        s1 = bank_branch([0.4], 0.6, 1e-8, 0.2, fr)[0]
        s0 = bank_branch([0.4], 0.6, 0.0 + 1e-12, 0.2, fr)[0]
        assert abs(s1 - s0) < 1e-6
    print("lambda/2b markers and gamma->0 continuity. OK")

    # ---- FLAG_EPS calibration from the p.8 intercepts, per closure ------- #
    pins = load_deck_pins()
    assert len(pins) == 6
    for fr in FRICTIONS:
        vals, mean = calibrate_ecoef(fr, pins)
        assert np.max(np.abs(vals - mean)) < 0.03 * mean, \
            f"single-ECOEF fit must hold to 3% ({fr}); got {vals}"
        assert abs(mean - ECOEF[fr]) < 0.03 * ECOEF[fr], \
            f"calibrated ECOEF ({fr}) drifted from {ECOEF[fr]}: {mean:.4f}"
        print(f"FLAG_EPS ({fr:>8}): per-curve eps*C_f = "
              + ", ".join(f"{v:.3f}" for v in vals)
              + f"  -> single constant {mean:.3f}. OK")

    # ---- Calibrated deck-p.8 pins, per closure --------------------------- #
    # The intercepts CANNOT distinguish the closures (E <-> 2E degeneracy);
    # the growth peaks CAN.  Documented bands (observed +- margin):
    #   rayleigh: deck/model peak ratio 3.18-4.42 -> [2.5, 5.5] (v1, rebuilt)
    #   momentum: deck/model peak ratio 2.28-3.22 -> [1.9, 3.7] (v2 finding:
    #   the Ikeda-consistent drag closes ~1/3 of the gap, does NOT close it)
    RATIO_BAND = {"rayleigh": (2.5, 5.5), "momentum": (1.9, 3.7)}
    SHIFT_BAND = {"rayleigh": (0.05, 0.25), "momentum": (0.08, 0.30)}
    for fr in FRICTIONS:
        print(f"--- {fr} closure, ECOEF = {ECOEF[fr]} ---")
        print(f"{'family':>10} {'D':>4} {'gam':>5} | {'c0':>6} {'deck':>6} | "
              f"{'kzero':>6} {'deck':>5} | {'spk':>6} {'deck':>5} {'ratio':>6}")
        for r in pins:
            p = Params(D=r["D"], gamma=r["gamma"], friction=fr)
            kpk, spk, kz = kstar_peak(p)
            om = bank_branch([1e-3], p.D, p.gamma, p.E, fr)[0]
            c0 = om.real / 1e-3
            ratio = r["sigma_peak"] / spk
            print(f"{r['family']:>10} {r['D']:>4} {r['gamma']:>5} | {c0:6.2f} "
                  f"{r['c0']:6.2f} | {kz:6.2f} {r['kzero']:5.2f} | "
                  f"{spk:6.3f} {r['sigma_peak']:5.2f} {ratio:6.2f}")
            # phases: all six intercepts within 15%
            assert abs(c0 - r["c0"]) <= 0.15 * abs(r["c0"]), \
                f"{r['family']} ({fr}): phase intercept off"
            # zero crossings: within 0.06
            assert abs(kz - r["kzero"]) <= 0.06, f"{r['family']} ({fr}): kzero off"
            # peak-height discrepancy codified per closure
            lo, hi = RATIO_BAND[fr]
            assert lo <= ratio <= hi, \
                f"{r['family']} ({fr}): growth-peak ratio {ratio:.2f} left [{lo},{hi}]"
            slo, shi = SHIFT_BAND[fr]
            assert slo <= kpk - r["kstar_peak"] <= shi, \
                f"{r['family']} ({fr}): peak shift {kpk - r['kstar_peak']:.2f}"
        # Ordering facts that hold on both sides:
        pk = {d: kstar_peak(Params(D=d, gamma=0.05, friction=fr))[1]
              for d in (0.3, 0.6, 0.9)}
        assert pk[0.6] > pk[0.3] and pk[0.6] > pk[0.9], \
            "sigma_peak maximised near D=0.6 at gamma=0.05 (as in the deck)"
        pg = {g: kstar_peak(Params(D=0.6, gamma=g, friction=fr))[1]
              for g in (0.03, 0.06, 0.09)}
        assert pg[0.03] > pg[0.06] > pg[0.09], "sigma_peak decreases with gamma"
        # Upstream propagation on all six; |c| small by k* = 1.5.
        for r in pins:
            p = Params(D=r["D"], gamma=r["gamma"], friction=fr)
            sig, c = growth_curve(np.array([0.05, 1.5]), p)
            assert c[0] < -0.3, "upstream at small k*"
            assert abs(c[1]) < 0.06, "phase speed ~ 0 by k* = 1.5"
    print("calibrated pins: phases +-15%, crossings +-0.06, per-closure peak "
          "bands codified, orderings + upstream propagation. OK")

    print("-" * 78)
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
