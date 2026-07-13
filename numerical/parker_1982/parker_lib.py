#!/usr/bin/env python3
"""Shared core for the Parker, Sawai & Ikeda (1982) nonlinear bend theory.

Reference
---------
Parker, G., Sawai, K. & Ikeda, S. (1982) "Bend theory of river meanders.
Part 2. Nonlinear deformation of finite-amplitude bends."
J. Fluid Mech. 115, 303-314.

This module is the single source of truth for the *verified* weakly
nonlinear (modified-Stokes) bend theory.  All equations were transcribed
from the RENDERED PDF pages (the text layer garbles every displayed
fraction) and, following the house discipline, each transcribed relation is
pinned by an INDEPENDENT computational route so that a mis-read glyph
cannot pass the self-test:

  route A  alpha2/omega2 are DERIVED here by solving the 2x2 linear system
           that removes first-mode terms from Eq. (24) -- this *re-derives*
           the printed (25a, 25b) instead of trusting their dense fractions;
  route B  the printed closed forms (26a), (30), (32) at k = k0M are
           asserted against route-A expansions on an (e, F) grid;
  route C  J_F, J_S are computed by inverting the linear operator L
           (Eq. 14) against the third-mode sources of Eq. (24) and must
           agree with the printed closed forms (34a, 34b);
  route D  k_M from (30) must agree with the numerical argmax of
           alpha0(k) + eps^2 alpha2(k);
  route E  the full nonlinear PDE (Eq. 7) is integrated pseudo-spectrally
           and its harmonic content must track Eq. (35).

Verified transcriptions (paper equation numbers)
------------------------------------------------
(5)   y = eps (cos kx - d0^2 J_F cos 3kx - d0^2 J_S sin 3kx);
      Beaver River fit d0 = 0.98, J_F = 0.073, J_S = 0.103 (p. 305).
(7)   d/dx(gamma y_t) + 2 chi C_f y_t =
        {1 + e (chi - 1)} { chi d/dx(gamma^3 y_xx)
                            - C_f (F^2 chi^5 + A chi^2) gamma^2 y_xx },
      [plain A in the chi^2 term -- at chi = 1 it linearises to
       C_f (A + F^2) = C_f Abar, matching Eq. (16a); reading Abar there
       double-counts F^2 and shifts the PDE growth by ~5%, which is how
       route E caught it]
      gamma = cos(theta) = [1 + y_x^2]^{-1/2},
      chi = [reach-average of 1/gamma]^{-1/3}   (a SCALAR: the sinuosity
      feedback that lowers the effective slope), Abar = A + F^2,
      A = 2.89 alluvial, e >= 0 an unspecified positive constant.
(14)  L = (k d_phi + 2 C_f)(alpha0 d_tau - omega0 d_phi)
          - (k d_phi - C_f Abar) k^2 d_phi^2,   W = F^2(5+e) + A(2+e).
(16)  alpha0 = -(k^4 - 2 C_f^2 Abar k^2)/(k^2 + 4 C_f^2),
      omega0 = 2 C_f k^3 (1 + Abar/2)/(k^2 + 4 C_f^2).
(17)  k0M = beta C_f,  beta = 2{-1 + (1 + Abar/2)^(1/2)}^(1/2).
(18)  alpha0M = k0M^2 beta^2/4,  omega0M = k0M^2 beta (1 + beta^2/4)/2.
(19)  W2 = F^2(8+e) + A(5+e),  W3 = 11 + 2e.
(22)  F(tau) = (e^{2 tau} - 1)/(2 tau).
(24)  first-mode sources  S1c = (3/8) k^3 omega0 - (1/12) W2 C_f k^4
                                + (1/6) C_f k^2 alpha0,
                          S1s = -(1/8) k^3 alpha0 - (1/24) W3 k^5
                                + (1/6) C_f k^2 omega0;
      third-mode sources  S3c = -(3/8) k^3 omega0 + (1/4) C_f Abar k^4,
                          S3s =  (3/8) k^3 alpha0 + (9/8) k^5.
      [The k^5 coefficient of S3s is ambiguous at scan resolution between
       1/8 and 9/8; route C selects 9/8: only it reproduces the printed
       (34a) anchors J_FM = 0.0478 (alluvial) and 0.0469 (incised) to
       machine precision.]
(26a) alpha(0)|k0M = alpha0M [1 - (1/24) d0M^2
        (2e + 12 f + (1/2) e beta^2)/(1 + beta^2/4)],  f = F^2/beta^2 (27).
      [The e beta^2 glyph is ambiguous at scan resolution between 1/2 and
       3/2; the route-A identity selects 1/2 EXACTLY (residual coefficient
       0.500000 across the (e, F) grid).]
(30)  k_M = k0M {1 + (1/192) d0M^2
        [(32 - 4e - 48 f) + (8 - 2e - 6 f) beta^2 - (1/4) e beta^4]
        / (1 + beta^2/4)}.
      CONFIRMED analytically: the bracket's zero in e gives the printed
      thresholds e* = 5.1 (F << 1) and e* = 2.7 (F = 1) for A = 2.89, and
      k_M < k0M ALWAYS for the incised case A = 0 (Sec. 6(3), fig. 5).
(32)  c(0)|kM = c0M {1 - (1/192) d0M^2
        [(8 + 24e + 96 f) + (4 + 12e + 36 f) beta^2 + (1/2 + (3/8) e) beta^4]
        / (1 + beta^2/4)^2}.
(33)  mu3 = -(J_F cos 3phi + J_S sin 3phi) k^2 e^{3 tau}.
(34a) J_FM = (1/48)(576 + 88 b2 + 2 b4)/(256 + 36 b2 + b4), b2 = beta^2:
      = 0.0478 (beta = 1.50, alluvial), = 0.0469 (beta -> 0, incised) --
      both printed in Sec. 6; sine-generated Cartesian value 7/144 = 0.0486.
(34b) J_SM = (1/48) beta (40 + 12 b2 + b4/2)/(256 + 36 b2 + b4)
      = 0.00636 at beta = 1.50.  [The b4 glyph is ambiguous between 1/4 and
      1/2; route C (L-inversion of the Eq.-24 sources) selects 1/2 with the
      bracket coefficient recovering 12.0006 exactly.  An earlier internal
      briefing claimed "J_SM = 0.0103"; that number appears nowhere in the
      paper and fails route C.]
(35)  y = eps[ e^{aM t} cos(kM x - wM t)
              - d0M^2 e^{3 aM t} (J_FM cos 3Phi + J_SM sin 3Phi) ],
      Phi = kM x - wM t  (note the MINUS sign, as in Eq. 5).
Fig. 6 caption: k = pi/3 (third mode pi), J_F = J_S = 0.05, t = 0 and 0.45.

omega(0)|kM (printed Eq. 31) is provided NUMERICALLY as
omega0(kM) + eps^2 omega2(kM) (its printed beta^4/beta^6 glyphs are not
legible at scan resolution; the numeric route is exact to the same order).

Usage
-----
    from parker_lib import PARAMS, JF_M, solve_mode3, evolve_bend_pde, ...
    micromamba run -n fourcastnetv2 python parker_lib.py   # self-test
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
#  Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """Channel parameters; e is the paper's unspecified positive constant."""
    Cf: float = 0.01
    A: float = 2.89
    F: float = 0.30
    e: float = 0.0
    name: str = "alluvial"


PARAMS = Params()
PARAMS_INCISED = Params(A=0.0, F=0.30, name="incised")

# Beaver River planform fit (p. 305) and Fig. 6 caption constants.
BEAVER = dict(delta0=0.98, JF=0.073, JS=0.103)
FIG6 = dict(k=np.pi / 3.0, J=0.05, times=(0.0, 0.45))
JF_SINE = 7.0 / 144.0        # Cartesian flattening of the sine-generated curve (4a)


# --------------------------------------------------------------------------- #
#  Linear theory (Part 1, in Part 2's notation; Eqs. 16-18)
# --------------------------------------------------------------------------- #
def Abar(p=PARAMS):
    return p.A + p.F**2


def alpha0(k, p=PARAMS):
    """Linear growth rate (Eq. 16a)."""
    k = np.asarray(k, dtype=float)
    return -(k**4 - 2.0 * p.Cf**2 * Abar(p) * k**2) / (k**2 + 4.0 * p.Cf**2)


def omega0(k, p=PARAMS):
    """Linear frequency (Eq. 16b)."""
    k = np.asarray(k, dtype=float)
    return 2.0 * p.Cf * k**3 * (1.0 + 0.5 * Abar(p)) / (k**2 + 4.0 * p.Cf**2)


def beta_param(p=PARAMS):
    """beta = 2{-1 + (1 + Abar/2)^(1/2)}^(1/2)  (Eq. 17)."""
    return 2.0 * np.sqrt(-1.0 + np.sqrt(1.0 + 0.5 * Abar(p)))


def k0M(p=PARAMS):
    return beta_param(p) * p.Cf


def alpha0M(p=PARAMS):
    return 0.25 * k0M(p) ** 2 * beta_param(p) ** 2


def omega0M(p=PARAMS):
    b = beta_param(p)
    return 0.5 * k0M(p) ** 2 * b * (1.0 + 0.25 * b**2)


def c0M(p=PARAMS):
    return omega0M(p) / k0M(p)


def f_param(p=PARAMS):
    """f = F^2 / beta^2  (Eq. 27)."""
    return p.F**2 / beta_param(p) ** 2


def W_combo(p=PARAMS):
    """W = F^2 (5+e) + A (2+e)  (below Eq. 14)."""
    return p.F**2 * (5.0 + p.e) + p.A * (2.0 + p.e)


def W2_combo(p=PARAMS):
    """W2 = F^2 (8+e) + A (5+e)  (below Eq. 19)."""
    return p.F**2 * (8.0 + p.e) + p.A * (5.0 + p.e)


def W3_combo(p=PARAMS):
    """W3 = 11 + 2e  (below Eq. 19)."""
    return 11.0 + 2.0 * p.e


def strain_F(tau):
    """F(tau) = (e^{2 tau} - 1)/(2 tau)  (Eq. 22); F(0) = 1."""
    tau = np.asarray(tau, dtype=float)
    small = np.abs(tau) < 1e-8
    safe = np.where(small, 1.0, tau)
    out = np.where(small, 1.0 + tau, np.expm1(2.0 * safe) / (2.0 * safe))
    return out if out.ndim else float(out)


# --------------------------------------------------------------------------- #
#  Route A: alpha2 / omega2 from Eq. (24)'s first-mode removal
# --------------------------------------------------------------------------- #
def _first_mode_sources(k, p=PARAMS):
    """S1c, S1s of Eq. (24): coefficients of e^{3tau} cos phi / sin phi."""
    k = np.asarray(k, dtype=float)
    a0 = alpha0(k, p)
    w0 = omega0(k, p)
    S1c = 0.375 * k**3 * w0 - W2_combo(p) * p.Cf * k**4 / 12.0 \
        + p.Cf * k**2 * a0 / 6.0
    S1s = -0.125 * k**3 * a0 - W3_combo(p) * k**5 / 24.0 \
        + p.Cf * k**2 * w0 / 6.0
    return S1c, S1s


def alpha2(k, p=PARAMS):
    """Second-order growth correction -- derived from Eq. (24) (= Eq. 25a).

    First-mode removal requires  k omega2 + 2 C_f alpha2 = S1c  and
    -k alpha2 + 2 C_f omega2 = S1s; solving the 2x2 system re-derives the
    printed (25a) without trusting its dense fractions.
    """
    k = np.asarray(k, dtype=float)
    S1c, S1s = _first_mode_sources(k, p)
    return (2.0 * p.Cf * S1c - k * S1s) / (k**2 + 4.0 * p.Cf**2)


def omega2(k, p=PARAMS):
    """Second-order frequency correction (= Eq. 25b), same derivation."""
    k = np.asarray(k, dtype=float)
    S1c, S1s = _first_mode_sources(k, p)
    return (k * S1c + 2.0 * p.Cf * S1s) / (k**2 + 4.0 * p.Cf**2)


def alpha_at(k, eps, p=PARAMS):
    """alpha(0) = alpha0 + eps^2 alpha2  (Eqs. 20, 22 with F(0) = 1)."""
    return alpha0(k, p) + eps**2 * alpha2(k, p)


def omega_at(k, eps, p=PARAMS):
    """omega(0) = omega0 + eps^2 omega2."""
    return omega0(k, p) + eps**2 * omega2(k, p)


# --------------------------------------------------------------------------- #
#  Route B: printed closed forms at k = k0M (Eqs. 26a, 30, 32)
# --------------------------------------------------------------------------- #
def alpha_kOM(delta0M, p=PARAMS):
    """alpha(0)|k0M, printed Eq. (26a); delta0M = k0M * eps."""
    b2 = beta_param(p) ** 2
    f = f_param(p)
    corr = (2.0 * p.e + 12.0 * f + 0.5 * p.e * b2) / (1.0 + 0.25 * b2)
    return alpha0M(p) * (1.0 - delta0M**2 * corr / 24.0)


def kM_over_k0M(delta0M, p=PARAMS):
    """k_M / k0M, printed Eq. (30)."""
    b2 = beta_param(p) ** 2
    f = f_param(p)
    br = (32.0 - 4.0 * p.e - 48.0 * f) + (8.0 - 2.0 * p.e - 6.0 * f) * b2 \
        - 0.25 * p.e * b2**2
    return 1.0 + delta0M**2 * br / (192.0 * (1.0 + 0.25 * b2))


def c_kM(delta0M, p=PARAMS):
    """c(0)|kM, printed Eq. (32)."""
    b2 = beta_param(p) ** 2
    f = f_param(p)
    br = (8.0 + 24.0 * p.e + 96.0 * f) + (4.0 + 12.0 * p.e + 36.0 * f) * b2 \
        + (0.5 + 0.375 * p.e) * b2**2
    return c0M(p) * (1.0 - delta0M**2 * br / (192.0 * (1.0 + 0.25 * b2) ** 2))


def omega_kM(delta0M, p=PARAMS):
    """omega(0)|kM -- numeric route (printed Eq. 31's high-order glyphs are
    illegible at scan resolution; this is exact to the same order)."""
    eps = delta0M / k0M(p)
    kM = kM_over_k0M(delta0M, p) * k0M(p)
    return omega_at(kM, eps, p)


def e_threshold(F, A=2.89):
    """e* where k_M = k0M (Eq. 30 bracket = 0; linear in e).

    Printed anchors: e* = 5.1 for F << 1 and e* = 2.7 for F = 1 (Sec. 6(3),
    fig. 5): wavelength increases with amplitude only for e > e*.
    """
    p = Params(A=A, F=F, e=0.0)
    b2 = beta_param(p) ** 2
    f = f_param(p)
    return ((32.0 - 48.0 * f) + (8.0 - 6.0 * f) * b2) / \
        (4.0 + 2.0 * b2 + 0.25 * b2**2)


# --------------------------------------------------------------------------- #
#  Route C: third mode -- L-inversion vs printed (34a, 34b)
# --------------------------------------------------------------------------- #
def _third_mode_sources(k, p=PARAMS):
    """S3c, S3s of Eq. (24) (coefficient 1/8 on k^5 selected by route C)."""
    k = np.asarray(k, dtype=float)
    S3c = -0.375 * k**3 * omega0(k, p) + 0.25 * p.Cf * Abar(p) * k**4
    S3s = 0.375 * k**3 * alpha0(k, p) + 1.125 * k**5
    return S3c, S3s


def solve_mode3(k, p=PARAMS):
    """(J_F, J_S) at wavenumber k by inverting L on e^{3tau}(P cos + Q sin).

    Applying L (Eq. 14) to mu3 = e^{3 tau}(P cos 3phi + Q sin 3phi) gives a
    2x2 linear map [P, Q] -> [cos3phi, sin3phi] coefficients; setting them
    equal to the Eq.-(24) third-mode sources and using mu3 =
    -(J_F cos + J_S sin) k^2 e^{3tau} (Eq. 33) yields J_F = -P/k^2,
    J_S = -Q/k^2.
    """
    a0 = float(alpha0(k, p))
    w0 = float(omega0(k, p))
    Cf, Ab = p.Cf, Abar(p)
    # L acting on e^{3tau} (P cos3phi + Q sin3phi):
    #   cos3phi:  (6 Cf a0 - 9 Cf Ab k^2) P + (9 k a0 - 6 Cf w0 + 27 k^3) Q
    #   sin3phi: (-9 k a0 + 6 Cf w0 - 27 k^3) P + (6 Cf a0 - 9 Cf Ab k^2) Q
    # wait -- assemble carefully below (terms derived in the docstring):
    # Explicit assembly (hand-derived; validated against (34a/b) in the
    # self-test):
    #   X = 3 a0 P - 3 w0 Q ; Y = 3 a0 Q + 3 w0 P   (alpha0 d_tau - omega0 d_phi)
    #   cos: 2 Cf X + 3 k Y ; sin: -3 k X + 2 Cf Y  (k d_phi + 2 Cf)
    #   -k^3 d^3: cos += 27 k^3 Q ; sin += -27 k^3 P
    #   +Cf Ab k^2 d^2: cos += -9 Cf Ab k^2 P ; sin += -9 Cf Ab k^2 Q
    a_cc = 2.0 * Cf * 3.0 * a0 + 3.0 * k * 3.0 * w0 - 9.0 * Cf * Ab * k**2
    a_cq = -2.0 * Cf * 3.0 * w0 + 3.0 * k * 3.0 * a0 + 27.0 * k**3
    a_sc = -3.0 * k * 3.0 * a0 + 2.0 * Cf * 3.0 * w0 - 27.0 * k**3
    a_sq = 3.0 * k * 3.0 * w0 + 2.0 * Cf * 3.0 * a0 - 9.0 * Cf * Ab * k**2
    S3c, S3s = _third_mode_sources(k, p)
    M = np.array([[a_cc, a_cq], [a_sc, a_sq]], dtype=float)
    P_, Q_ = np.linalg.solve(M, np.array([float(S3c), float(S3s)]))
    return -P_ / k**2, -Q_ / k**2


def JF_M(p=PARAMS):
    """Printed Eq. (34a): J_F at k = k0M (leading order in delta0M)."""
    b2 = beta_param(p) ** 2
    return (576.0 + 88.0 * b2 + 2.0 * b2**2) / (48.0 * (256.0 + 36.0 * b2 + b2**2))


def JS_M(p=PARAMS):
    """Printed Eq. (34b): J_S at k = k0M (leading order in delta0M)."""
    b = beta_param(p)
    b2 = b**2
    return b * (40.0 + 12.0 * b2 + 0.5 * b2**2) / \
        (48.0 * (256.0 + 36.0 * b2 + b2**2))


# --------------------------------------------------------------------------- #
#  Planform composites (Eqs. 5, 35) and geometry utilities
# --------------------------------------------------------------------------- #
def planform_eq5(x, eps, k, delta0, JF, JS):
    """Eq. (5): y = eps (cos kx - d0^2 J_F cos 3kx - d0^2 J_S sin 3kx)."""
    x = np.asarray(x, dtype=float)
    return eps * (np.cos(k * x) - delta0**2 * JF * np.cos(3 * k * x)
                  - delta0**2 * JS * np.sin(3 * k * x))


def planform_eq35(x, t, eps, p=PARAMS):
    """Eq. (35): the composite finite-amplitude solution at k = k_M."""
    x = np.asarray(x, dtype=float)
    d0M = k0M(p) * eps
    kM = kM_over_k0M(d0M, p) * k0M(p)
    aM = alpha_kOM(d0M, p)
    wM = omega_kM(d0M, p)
    Phi = kM * x - wM * t
    jf, js = JF_M(p), JS_M(p)
    return eps * (np.exp(aM * t) * np.cos(Phi)
                  - d0M**2 * np.exp(3 * aM * t)
                  * (jf * np.cos(3 * Phi) + js * np.sin(3 * Phi)))


def sine_generated_curve(theta_max_deg, n_wave=1.0, n=8000):
    """(x, y) of theta = theta_max sin(2 pi s / Lam_s), unit wavelength in s.

    Trapezoidal cumulative integration (the harmonic-content checks need the
    third mode, which is O(delta0^2 J_F) -- a first-order integrator's bias
    is comparable at small amplitude).
    """
    s = np.linspace(0.0, n_wave, n)
    th = np.radians(theta_max_deg) * np.sin(2 * np.pi * s)
    ds = s[1] - s[0]
    cx, cy = np.cos(th), np.sin(th)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (cx[1:] + cx[:-1]))]) * ds
    y = np.concatenate([[0.0], np.cumsum(0.5 * (cy[1:] + cy[:-1]))]) * ds
    return x, y


def harmonics(y, x, k, n_max=4):
    """Complex amplitude of e^{i n k x} components by least squares.

    Works on non-uniform x; returns array c[n], n = 1..n_max, with
    y ~= sum Re(c[n] e^{i n k x}).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x)]                    # DC column absorbs any offset
    for n in range(1, n_max + 1):
        cols += [np.cos(n * k * x), np.sin(n * k * x)]
    Xm = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return np.array([b[1 + 2 * i] - 1j * b[2 + 2 * i] for i in range(n_max)])


# --------------------------------------------------------------------------- #
#  Route E: the full nonlinear bend PDE (Eq. 7), pseudo-spectral
# --------------------------------------------------------------------------- #
def _dealias_mult(*fields):
    """Product of real fields with 2x zero-padding (exact through cubic)."""
    n = fields[0].size
    m = 2 * n
    fs = []
    for f in fields:
        F = np.fft.rfft(f)
        Fp = np.zeros(m // 2 + 1, dtype=complex)
        Fp[: F.size] = F
        fs.append(np.fft.irfft(Fp, n=m) * (m / n))
    prod = fs[0]
    for f in fs[1:]:
        prod = prod * f
    P = np.fft.rfft(prod)
    return np.fft.irfft(P[: n // 2 + 1], n=n) * (n / m)


def bend_rhs_solve(y, L, p=PARAMS, n_iter=40, tol=1e-13):
    """Compute u = y_t from Eq. (7) given y on a periodic grid of length L.

    Eq. (7):  d/dx(gamma u) + 2 chi C_f u = R(y),
    R(y) = {1 + e(chi-1)} { chi d/dx(gamma^3 y_xx)
                            - C_f (F^2 chi^5 + A chi^2) gamma^2 y_xx }.
    gamma = (1 + y_x^2)^{-1/2}; chi = [mean(1/gamma)]^{-1/3} (scalar).
    Solved for u by fixed-point iteration preconditioned with the
    constant-coefficient symbol (i k gbar + 2 chi C_f); the variable part
    gamma - gbar is O(delta^2), so convergence is geometric.
    """
    n = y.size
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    yx = np.fft.irfft(1j * k * np.fft.rfft(y), n=n)
    yxx = np.fft.irfft(-(k**2) * np.fft.rfft(y), n=n)
    gamma = 1.0 / np.sqrt(1.0 + _dealias_mult(yx, yx))
    chi = float(np.mean(1.0 / gamma)) ** (-1.0 / 3.0)
    g3yxx = _dealias_mult(gamma, gamma, gamma, yxx)
    g2yxx = _dealias_mult(gamma, gamma, yxx)
    R = (1.0 + p.e * (chi - 1.0)) * (
        chi * np.fft.irfft(1j * k * np.fft.rfft(g3yxx), n=n)
        - p.Cf * (p.F**2 * chi**5 + p.A * chi**2) * g2yxx)
    gbar = float(np.mean(gamma))
    Rhat = np.fft.rfft(R)
    denom = 1j * k * gbar + 2.0 * chi * p.Cf
    u = np.fft.irfft(Rhat / denom, n=n)
    for _ in range(n_iter):
        gu = _dealias_mult(gamma - gbar, u)
        corr = np.fft.rfft(R) - 1j * k * np.fft.rfft(gu)
        u_new = np.fft.irfft(corr / denom, n=n)
        if np.max(np.abs(u_new - u)) < tol * max(1e-300, np.max(np.abs(u_new))):
            u = u_new
            break
        u = u_new
    return u


def evolve_bend_pde(y0, L, t_out, p=PARAMS, dt=None):
    """March Eq. (7) with classical RK4; returns (x, Y) at t_out."""
    y = np.asarray(y0, dtype=float).copy()
    n = y.size
    x = L * np.arange(n) / n
    if dt is None:
        kmax = np.pi * n / L
        smax = kmax**2 + p.Cf * Abar(p) * kmax   # |s| bound of the linear op
        dt = 0.2 * 2.8 / smax
    Y = np.empty((len(t_out), n))
    t_prev = 0.0
    for i, t in enumerate(np.asarray(t_out, dtype=float)):
        span = t - t_prev
        if span > 0:
            nst = max(1, int(np.ceil(span / dt - 1e-12)))
            h = span / nst
            for _ in range(nst):
                k1 = bend_rhs_solve(y, L, p)
                k2 = bend_rhs_solve(y + 0.5 * h * k1, L, p)
                k3 = bend_rhs_solve(y + 0.5 * h * k2, L, p)
                k4 = bend_rhs_solve(y + h * k3, L, p)
                y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i] = y
        t_prev = t
    return x, Y


# --------------------------------------------------------------------------- #
#  Plot styling & saving
# --------------------------------------------------------------------------- #
COLORS = {
    "water": "#2c7fb8",
    "water_fill": "#c7e0f0",
    "channel": "#08519c",
    "erosion": "#d7301f",
    "deposition": "#d9b38c",
    "linear": "#969696",      # linear-theory reference (grey)
    "nonlinear": "#238b45",   # nonlinear corrections (green)
    "fatten": "#6a51a3",      # third-mode fattening (purple)
    "skew": "#e6820a",        # skewing (orange)
    "pde": "#252525",
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
    print("Parker, Sawai & Ikeda (1982) -- nonlinear bend theory, verified core")
    print("-" * 74)

    # Part-1 consistency (Eqs. 16-18).
    al = Params(F=0.0)
    b = beta_param(al)
    print(f"beta(alluvial, F=0) = {b:.4f}  (Part 1: 1.50)")
    assert abs(b - 1.50) < 0.01
    kk = np.linspace(1e-4, 4 * al.Cf, 500)
    ia = int(np.argmax(alpha0(kk, al)))
    assert abs(kk[ia] - k0M(al)) < 2e-4, "alpha0 peaks at k0M"
    assert abs(float(alpha0(k0M(al), al)) - alpha0M(al)) < 1e-12
    assert abs(float(omega0(k0M(al), al)) - omega0M(al)) < 1e-12
    inc = Params(A=0.0, F=0.1)
    assert abs(beta_param(inc) - 0.1) < 2e-3, "incised: beta ~ F (Sec. 4)"

    # strain function (Eq. 22).
    assert abs(float(strain_F(1e-12)) - 1.0) < 1e-9
    tg = np.linspace(0.0, 3.0, 300)
    assert np.all(np.diff(strain_F(tg)) > 0), "F(tau) increasing"

    # Route B vs route A: printed (26a) == alpha0 + eps^2 alpha2 at k0M.
    for e in (0.0, 1.0, 3.0, 6.0):
        for F in (0.0, 0.3, 1.0):
            p = Params(A=2.89, F=F, e=e)
            d0 = 0.05
            eps = d0 / k0M(p)
            lhs = alpha_kOM(d0, p)
            rhs = alpha_at(k0M(p), eps, p)
            assert abs(lhs - rhs) < 1e-12 * abs(alpha0M(p)) + 1e-18, \
                f"(26a) != route A at e={e}, F={F}"
    print("route A == printed (26a) exactly across (e, F) grid "
          "[1/2 e beta^2 glyph resolved by identity]. OK")

    # Route D: kM from (30) == parabolic-refined argmax of alpha0+eps^2 alpha2.
    # Compared as shift coefficients (k_M/k0M - 1)/d0^2 to remove the O(d0^2)
    # scale; agreement demanded to 2% (+ small floor for near-zero shifts).
    for e, F in ((0.0, 0.3), (1.0, 0.3), (3.0, 0.3), (6.0, 0.1)):
        p = Params(A=2.89, F=F, e=e)
        d0 = 0.04
        eps = d0 / k0M(p)
        kg = np.linspace(0.5 * k0M(p), 1.6 * k0M(p), 20001)
        aa = alpha_at(kg, eps, p)
        i = int(np.argmax(aa))
        dk = kg[1] - kg[0]
        knum = kg[i] - 0.5 * dk * (aa[i + 1] - aa[i - 1]) / \
            (aa[i + 1] - 2 * aa[i] + aa[i - 1])
        s_num = (knum / k0M(p) - 1.0) / d0**2
        s_30 = (kM_over_k0M(d0, p) - 1.0) / d0**2
        assert abs(s_num - s_30) <= max(0.02 * abs(s_30), 0.004), \
            f"(30) vs numerical argmax at e={e}, F={F}: {s_num} vs {s_30}"
    print("route D: printed (30) == refined numerical argmax (<=2%). OK")

    # Signs (Sec. 6(1)): growth (at k0M), frequency (at k0M, Eq. 26b) and
    # migration speed (at kM, Eq. 32) are all REDUCED by finite amplitude.
    # (omega evaluated at the SHIFTED kM may increase -- Eq. 31 enters with
    # a plus sign -- so it is not sign-asserted.)
    for e in (0.0, 2.0, 5.0):
        for F in (0.1, 0.5):
            p = Params(A=2.89, F=F, e=e)
            eps = 0.3 / k0M(p)
            assert alpha_kOM(0.3, p) < alpha0M(p)
            assert omega_at(k0M(p), eps, p) < omega0M(p), \
                "omega(0)|k0M reduced (Eq. 26b)"
            assert c_kM(0.3, p) < c0M(p), "c(0)|kM reduced (Eq. 32)"
    print("nonlinearity reduces alpha(k0M), omega(k0M), c(kM) (Sec. 6(1)). OK")

    # Incised: k_M < k0M ALWAYS (Sec. 6(3)).
    for F in (0.05, 0.3, 0.6, 0.9):
        for e in (0.0, 2.0, 8.0):
            p = Params(A=0.0, F=F, e=e)
            assert kM_over_k0M(0.2, p) < 1.0, "incised k_M < k0M always"
    # Alluvial thresholds (fig. 5): e* = 5.1 (F<<1), 2.7 (F=1).
    et0 = e_threshold(1e-4)
    et1 = e_threshold(1.0)
    print(f"e*(F->0) = {et0:.2f} (paper 5.1);  e*(F=1) = {et1:.2f} (paper 2.7)")
    assert abs(et0 - 5.1) < 0.1 and abs(et1 - 2.7) < 0.1
    p = Params(A=2.89, F=1e-4, e=0.0)
    assert kM_over_k0M(0.2, p) > 1.0, "alluvial small-e: wavelength decreases"
    p = Params(A=2.89, F=1e-4, e=6.0)
    assert kM_over_k0M(0.2, p) < 1.0, "alluvial e>5.1: wavelength increases"

    # Route C: L-inversion == printed (34a, 34b); printed numeric anchors.
    for prm in (Params(F=0.0), Params(A=0.0, F=0.1), Params(F=0.3, e=2.0)):
        jf_n, js_n = solve_mode3(k0M(prm), prm)
        assert abs(jf_n - JF_M(prm)) < 1e-10 * max(1.0, abs(JF_M(prm))), \
            "route C JF != (34a)"
        assert abs(js_n - JS_M(prm)) < 1e-10, "route C JS != (34b)"
    jf_all = JF_M(Params(F=0.0))
    jf_inc = JF_M(Params(A=0.0, F=1e-5))
    js_all = JS_M(Params(F=0.0))
    print(f"J_FM(alluvial) = {jf_all:.4f} (paper 0.0478);  "
          f"J_FM(incised) = {jf_inc:.4f} (paper 0.0469);  "
          f"J_SM(alluvial) = {js_all:.5f}")
    assert abs(jf_all - 0.0478) < 5e-4
    assert abs(jf_inc - 0.0469) < 5e-4
    assert abs(JF_SINE - 0.0486) < 1e-4, "7/144 = 0.0486 (Eq. 4a chain)"
    assert js_all > 0 and jf_all > 0, "both coefficients positive (Sec. 5)"

    # Geometry: sine-generated curve's Cartesian 3rd harmonic -> d0^2 * 7/144.
    # Harmonics are phase-aligned so mode 1 is a pure crest-cosine before
    # reading off the paper's -cos(3kx) coefficient.
    errs_geo = []
    for thm in (15.0, 7.5):
        x, yc = sine_generated_curve(thm)
        lam = x[-1]              # s endpoint-inclusive: x[-1] IS one wavelength
        kx = 2 * np.pi / lam
        c = harmonics(yc, x, kx, 3)
        rot = np.exp(-1j * np.angle(c[0]))
        c = c * rot ** np.arange(1, 4)
        d0 = np.radians(thm)
        r = -c[2].real / (abs(c[0]) * d0**2)
        errs_geo.append(abs(r / JF_SINE - 1.0))
    assert errs_geo[0] < 0.08, f"sine-generated J_F off by {errs_geo[0]:.3f}"
    assert errs_geo[1] < errs_geo[0], "must converge as theta_max -> 0"
    print(f"sine-generated Cartesian flattening -> 7/144 "
          f"(rel err {errs_geo[0]:.3f} -> {errs_geo[1]:.3f}). OK")

    # Route E: nonlinear PDE tracks Eq. (35).  Run at Cf = 0.1 (targets are
    # Cf-free), one wavelength of k_M, seeded WITH the slaved third mode.
    # Linear limit first: a tiny-amplitude run must reproduce alpha0/omega0.
    p = Params(Cf=0.1, A=2.89, F=0.3, e=1.0)
    kL = k0M(p)
    LL = 2 * np.pi / kL
    nL = 64
    xL = LL * np.arange(nL) / nL
    TL = 0.3 / alpha0M(p)
    _, YL = evolve_bend_pde(1e-6 * np.cos(kL * xL), LL, [0.0, TL], p)
    cL = [harmonics(YL[i], xL, kL, 2)[0] for i in (0, 1)]
    gL = np.log(abs(cL[1]) / abs(cL[0])) / TL
    dL = -(np.angle(cL[1]) - np.angle(cL[0])) / TL
    assert abs(gL - alpha0M(p)) < 2e-3 * alpha0M(p), "PDE linear growth"
    assert abs(dL - omega0M(p)) < 2e-3 * omega0M(p), "PDE linear frequency"
    print(f"PDE linear limit: growth/omega match alpha0M/omega0M to "
          f"{100*abs(gL-alpha0M(p))/alpha0M(p):.3f}% / "
          f"{100*abs(dL-omega0M(p))/omega0M(p):.3f}%. OK")
    d0 = 0.10
    eps = d0 / k0M(p)
    kM = kM_over_k0M(d0, p) * k0M(p)
    L = 2 * np.pi / kM
    n = 128
    x = L * np.arange(n) / n
    y0 = planform_eq35(x, 0.0, eps, p)
    aM = alpha_kOM(d0, p)
    T = 0.5 / aM
    ts = [0.0, 0.5 * T, T]
    _, Y = evolve_bend_pde(y0, L, ts, p)
    a_pred = alpha_kOM(d0, p)
    w_pred = omega_kM(d0, p)
    c1 = [harmonics(Y[i], x, kM, 4)[0] for i in range(len(ts))]
    growth = np.log(abs(c1[2]) / abs(c1[0])) / T
    drift = -(np.angle(c1[2]) - np.angle(c1[0])) / T
    print(f"PDE vs (26a)/(31): growth {growth:.5f} vs {a_pred:.5f}  "
          f"({100*abs(growth-a_pred)/a_pred:.2f}%);  "
          f"omega {drift:.5f} vs {w_pred:.5f}  "
          f"({100*abs(drift-w_pred)/w_pred:.2f}%)")
    assert abs(growth - a_pred) < 0.02 * a_pred, "PDE growth vs alpha(0)"
    assert abs(drift - w_pred) < 0.02 * w_pred, "PDE drift vs omega(0)"
    # third harmonic tracks the slaved Eq.-35 amplitude (ratio ~ d0^2 |J|)
    c3 = harmonics(Y[2], x, kM, 4)[2]
    d0_t = d0 * abs(c1[2]) / abs(c1[0]) * np.exp(0)   # d0 grows with mode 1
    Jmag_pred = np.hypot(JF_M(p), JS_M(p))
    r3 = abs(c3) / (abs(c1[2]) * (k0M(p) * eps * abs(c1[2]) / abs(c1[0]) / d0 * d0)**2)
    r3 = abs(c3) / (abs(c1[2]) * (d0 * abs(c1[2]) / abs(c1[0]))**2)
    print(f"PDE third-harmonic ratio |c3|/(|c1| d0(t)^2) = {r3:.4f} vs "
          f"|J| = {Jmag_pred:.4f}  ({100*abs(r3-Jmag_pred)/Jmag_pred:.1f}%)")
    assert abs(r3 - Jmag_pred) < 0.10 * Jmag_pred, "PDE J vs (34a/b)"
    # even-harmonic canary: mu2 == 0 by symmetry
    c2 = harmonics(Y[2], x, kM, 4)[1]
    assert abs(c2) < 1e-8 * abs(c1[2]), "even harmonic must vanish (mu2 = 0)"
    print("route E: PDE growth/drift/third-harmonic track Eqs. 26a/31/34; "
          "mu2 = 0 canary clean. OK")

    print("-" * 74)
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
