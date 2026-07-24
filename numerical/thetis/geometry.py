#!/usr/bin/env python3
"""Channel geometry, frozen bed and base state for the Thetis meander model.

Pure numpy -- no Firedrake -- so it can be developed and tested while the
Firedrake environment builds, and so the design numbers can be checked against
``../ikeda_1981/ikeda_lib.py`` independently of the solver.

Coordinates
-----------
``x``     down-valley coordinate (Ikeda's Cartesian ``x``; the meander is
          ``y = c(x)``).  Used as the streamwise reference coordinate.
``ntil``  transverse reference coordinate, ``ntil in [-1, 1]``; ``n = ntil*b``.

The reference domain is the rectangle ``[0, L] x [-1, 1]``.  The physical
channel is obtained by

    X = x
    Y = c(x) + ntil * b(x)          c = (y_N + y_S)/2 ,  b = (y_N - y_S)/2

i.e. the banks are located at their y-position **at each valley station x**
(vertical cuts), not by a normal offset from the centreline.  The two
constructions differ only at O(theta^2) where ``theta = arctan(dc/dx)``;
:func:`shear_report` measures it and :func:`check_map` asserts it is small.
Vertical cuts can never produce an inverted cell, which a normal-offset map
does as soon as ``|ntil * b * kappa| -> 1``.

Base state
----------
The jet is prescribed quadratic (the vorticity-gradient provider),

    ubar(ntil) = U0 + Delta * (1 - ntil**2) ,

and the **bed follows from it** so that the base state is an exact steady
solution rather than something that decays during spin-up:

    C_f * ubar**2 / H = g*I + nu * d2ubar/dn2         (streamwise balance)

    =>  H(ntil) = C_f * ubar(ntil)**2 / (g*I + nu*ubar_nn) ,  ubar_nn = -2*Delta/b**2

With ``nu -> 0`` this is ``ubar = sqrt(g*I*H/C_f)`` -- Ikeda eq. (2), i.e.
``ubar ~ sqrt(H)``.  Prescribing a quadratic jet therefore *forces* a quartic
depth profile; the transverse depth contrast is ``(ubar_centre/ubar_bank)**2``.

The bed elevation is ``z_b(x, ntil) = eta_ref - I*x - H(ntil)``: it falls
downstream at the valley slope (that is what drives the flow) while its shape
and the depth are the same at every ``x``.  "The bed does not change" means
``d z_b / d t = 0``, not ``d z_b / d x = 0``.

Run ``python geometry.py`` for the self-test.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

G = 9.81

# np.trapezoid is numpy>=2.0; np.trapz is the <2.0 spelling.  The Firedrake env
# and fourcastnetv2 sit on opposite sides of that split.
_trapz = getattr(np, "trapezoid", None) or np.trapz

HERE = os.path.dirname(os.path.abspath(__file__))
IKEDA_DIR = os.path.normpath(os.path.join(HERE, "..", "ikeda_1981"))


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """Every physical knob.  Derived quantities live in :class:`Design`."""

    # --- flow / bed -------------------------------------------------------
    Cf: float = 0.05        # quadratic drag coefficient (Ikeda's C_f)
    F_ref: float = 0.30     # reference Froude number  =>  I = Cf * F_ref**2
    H_ref: float = 1.0      # target width-mean depth [m]
    jet_ratio: float = 0.30 # Delta / U0 -- cross-channel jet excess
    nu: float = 0.05        # horizontal viscosity [m^2/s] (DG stabilisation)

    # --- secondary flow ---------------------------------------------------
    A_ikeda: float = 0.0    # 0 = Ikeda's INCISED case (user decision 2026-07-23).
                            # 2.89 = alluvial (Suga 1963).  See docs/model.md.

    # --- planform ---------------------------------------------------------
    lam_over_W: float = 12.0   # meander wavelength in channel widths
    n_wave: int = 4            # wavenumber over the meander reach (4 or 8)
    n_wave_ref: int = 4        # the wavenumber that is tuned to k_OM
    amp0_over_b: float = 0.05  # initial bank amplitude a0 / b  (small)
    n_lam_reach: int = 4       # meander reach length in reference wavelengths
    L_in_over_W: float = 5.0   # straight entry reach (flow conditioning)
    L_out_over_W: float = 3.0  # straight exit reach

    # --- bank erosion / deposition ---------------------------------------
    # Ikeda (11)-(13): gamma*dy/dt = E*u'_b.  river.pdf p.19 is the same law
    # with E = eps*C_f (see docs/model.md).  E_e != E_d is the user's
    # extension; it is NOT in Ikeda and has no calibration on disk.
    E_erode: float = 1.0e-6
    E_deposit: float = 1.0e-6
    morph_factor: float = 1.0

    # --- numerics ---------------------------------------------------------
    n_cells_across: int = 28
    pts_per_wavelength: int = 48


# --------------------------------------------------------------------------- #
#  Derived design
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Design:
    cfg: Config
    I: float          # valley slope
    U0: float         # jet speed at the banks [m/s]
    Delta: float      # jet excess at the centreline [m/s]
    b: float          # half width [m]
    lam: float        # reference meander wavelength [m]
    L_m: float        # meander reach length [m]
    L_in: float
    L_out: float
    L: float          # total domain length [m]
    k_OM: float       # Ikeda fastest-growing wavenumber (units 1/H_ref)
    k_c: float        # Ikeda cutoff wavenumber
    nx: int
    ny: int

    @property
    def W(self) -> float:
        return 2.0 * self.b

    @property
    def x_m0(self) -> float:
        return self.L_in

    @property
    def x_m1(self) -> float:
        return self.L_in + self.L_m


def _ikeda_wavenumbers(cfg: Config):
    """k_OM and k_c from the *verified* ikeda_lib, not from local arithmetic."""
    if IKEDA_DIR not in sys.path:
        sys.path.insert(0, IKEDA_DIR)
    from ikeda_lib import k_OM as _k_OM  # noqa: E402

    k_om = float(_k_OM(Cf=cfg.Cf, A=cfg.A_ikeda, F=cfg.F_ref))
    k_c = cfg.Cf * np.sqrt(2.0 * (cfg.A_ikeda + cfg.F_ref**2))
    return k_om, k_c


def _ubar_mean_square(U0: float, Delta: float) -> float:
    """Width mean of ubar**2 over ntil in [-1, 1], ubar = U0 + Delta(1-ntil^2)."""
    a, d = U0 + Delta, Delta
    # mean of (a - d*t^2)^2 over t in [-1,1] = a^2 - 2*a*d/3 + d^2/5
    return a * a - 2.0 * a * d / 3.0 + d * d / 5.0


def build_design(cfg: Config) -> Design:
    """Derive every geometric and base-flow quantity from :class:`Config`."""
    I = cfg.Cf * cfg.F_ref**2                       # Ikeda (2): F^2 = I/Cf
    k_om, k_c = _ikeda_wavenumbers(cfg)

    # Ikeda's k is non-dimensionalised by depth: k = 2*pi*H/lambda
    lam = 2.0 * np.pi * cfg.H_ref / k_om
    b = 0.5 * lam / cfg.lam_over_W
    L_m = cfg.n_lam_reach * lam
    L_in = cfg.L_in_over_W * 2.0 * b
    L_out = cfg.L_out_over_W * 2.0 * b
    L = L_in + L_m + L_out

    # Solve for U0 so that the width-mean depth equals H_ref.
    # H = Cf*ubar^2 / (g*I + nu*ubar_nn),  ubar_nn = -2*Delta/b^2 (constant)
    r = cfg.jet_ratio
    U0 = np.sqrt(G * I * cfg.H_ref / cfg.Cf)        # nu = 0, flat-jet seed
    for _ in range(200):
        Delta = r * U0
        denom = G * I - cfg.nu * 2.0 * Delta / b**2
        if denom <= 0:
            raise ValueError("viscous term exceeds the slope term; reduce nu")
        H_mean = cfg.Cf * _ubar_mean_square(U0, Delta) / denom
        U0 *= np.sqrt(cfg.H_ref / H_mean)
        if abs(H_mean - cfg.H_ref) < 1e-14 * cfg.H_ref:
            break
    Delta = r * U0

    nx = max(8, int(round(L / (lam / cfg.pts_per_wavelength))))
    ny = cfg.n_cells_across

    return Design(cfg=cfg, I=I, U0=U0, Delta=Delta, b=b, lam=lam, L_m=L_m,
                  L_in=L_in, L_out=L_out, L=L, k_OM=k_om, k_c=k_c, nx=nx, ny=ny)


# --------------------------------------------------------------------------- #
#  Base state (frozen in time; function of ntil only)
# --------------------------------------------------------------------------- #
def base_velocity(ntil, d: Design):
    """Quadratic jet ubar(ntil) = U0 + Delta*(1 - ntil^2) [m/s]."""
    ntil = np.asarray(ntil, dtype=float)
    return d.U0 + d.Delta * (1.0 - ntil**2)


def base_depth(ntil, d: Design):
    """Frozen depth profile H(ntil) [m] -- the exact steady-balance bed."""
    ntil = np.asarray(ntil, dtype=float)
    ubar_nn = -2.0 * d.Delta / d.b**2                # d2/dn2, constant
    denom = G * d.I + d.cfg.nu * ubar_nn
    return d.cfg.Cf * base_velocity(ntil, d) ** 2 / denom


def width_mean(f, d: Design, order: int = 8) -> float:
    """Width mean over ntil in [-1, 1] by Gauss-Legendre quadrature.

    ``H(ntil)`` is a *quartic* in ntil, so the trapezoid rule carries an O(h^2)
    error (~1e-6 at 401 points) that would mask a genuine design error.  An
    8-node Gauss-Legendre rule is exact through degree 15.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return float(np.dot(weights, f(nodes, d)) / 2.0)


def base_residual(ntil, d: Design):
    """Streamwise momentum residual of the base state -- must be ~0."""
    ntil = np.asarray(ntil, dtype=float)
    ub = base_velocity(ntil, d)
    H = base_depth(ntil, d)
    ubar_nn = -2.0 * d.Delta / d.b**2
    return G * d.I + d.cfg.nu * ubar_nn - d.cfg.Cf * ub**2 / H


def bed_elevation(x, ntil, d: Design, eta_ref: float = 0.0):
    """z_b(x, ntil) = eta_ref - I*x - H(ntil).

    Falls downstream at the valley slope (this is what drives the flow) while
    its cross-sectional shape is the same at every x and never changes in time.
    """
    x = np.asarray(x, dtype=float)
    return eta_ref - d.I * x - base_depth(ntil, d)


def base_elevation(x, d: Design, eta_ref: float = 0.0):
    """Base free-surface elevation eta_0(x) = eta_ref - I*x (flat across n)."""
    return eta_ref - d.I * np.asarray(x, dtype=float)


# --------------------------------------------------------------------------- #
#  Planform
# --------------------------------------------------------------------------- #
def _taper(x, d: Design, n_ramp_lam: float = 0.5):
    """Cosine ramp: 0 in the straight reaches, 1 in the meander core."""
    x = np.asarray(x, dtype=float)
    ramp = n_ramp_lam * d.lam
    t = np.zeros_like(x)
    inside = (x >= d.x_m0) & (x <= d.x_m1)
    xi = np.clip((x - d.x_m0) / ramp, 0.0, 1.0)
    xo = np.clip((d.x_m1 - x) / ramp, 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * xi)) * 0.5 * (1.0 - np.cos(np.pi * xo))
    t[inside] = w[inside]
    return t


def initial_banks(x, d: Design):
    """Initial (y_N, y_S) for the configured wavenumber.

    Both banks carry the same sinusoid, so the channel is a constant-width
    meander at t=0 and the centreline is c(x) = a0*taper*sin(2*pi*m*(x-x_m0)/L_m).
    """
    x = np.asarray(x, dtype=float)
    a0 = d.cfg.amp0_over_b * d.b
    phase = 2.0 * np.pi * d.cfg.n_wave * (x - d.x_m0) / d.L_m
    c = a0 * _taper(x, d) * np.sin(phase)
    return c + d.b, c - d.b


def centreline(yN, yS):
    return 0.5 * (np.asarray(yN) + np.asarray(yS))


def half_width(yN, yS):
    return 0.5 * (np.asarray(yN) - np.asarray(yS))


def curvature(x, c):
    """kappa = c'' / (1 + c'^2)^{3/2} on a (possibly non-uniform) x grid."""
    x, c = np.asarray(x, float), np.asarray(c, float)
    c1 = np.gradient(c, x, edge_order=2)
    c2 = np.gradient(c1, x, edge_order=2)
    return c2 / (1.0 + c1**2) ** 1.5


def wavenumber_of(m: int, d: Design) -> float:
    """Ikeda-normalised wavenumber k = 2*pi*H_ref/lambda_m for m waves."""
    return 2.0 * np.pi * d.cfg.H_ref / (d.L_m / m)


# --------------------------------------------------------------------------- #
#  Reference -> physical map
# --------------------------------------------------------------------------- #
def channel_map(x, ntil, yN, yS):
    """(x, ntil) -> (X, Y) with the banks at their y-position at each x."""
    c = centreline(yN, yS)
    b = half_width(yN, yS)
    return np.asarray(x, float), c + np.asarray(ntil, float) * b


def shear_report(x, yN, yS):
    """Max |theta| of the centreline and the implied normal-width error."""
    c = centreline(yN, yS)
    theta = np.arctan(np.gradient(c, np.asarray(x, float), edge_order=2))
    tmax = float(np.max(np.abs(theta)))
    return tmax, float(1.0 - np.cos(tmax))


def check_map(x, yN, yS, theta_max_deg: float = 20.0):
    """Assert the vertical-cut map is well behaved."""
    b = half_width(yN, yS)
    assert np.all(b > 0), "channel width collapsed"
    tmax, werr = shear_report(x, yN, yS)
    assert np.degrees(tmax) < theta_max_deg, (
        f"centreline slope {np.degrees(tmax):.1f} deg exceeds "
        f"{theta_max_deg} deg; the vertical-cut map is no longer accurate")
    return tmax, werr


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    print("=" * 74)
    print("geometry.py self-test")
    print("=" * 74)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    cfg = Config()
    d = build_design(cfg)
    ntil = np.linspace(-1, 1, 401)

    print("\n-- design --")
    print(f"  Cf={cfg.Cf}  F_ref={cfg.F_ref}  A={cfg.A_ikeda}  nu={cfg.nu}")
    print(f"  I  = {d.I:.6e}          (= Cf*F^2, Ikeda eq. 2)")
    print(f"  U0 = {d.U0:.4f} m/s     Delta = {d.Delta:.4f} m/s")
    print(f"  b  = {d.b:.3f} m        W = {d.W:.3f} m")
    print(f"  lambda_OM = {d.lam:.2f} m   = {d.lam / d.W:.3f} W")
    print(f"  L_in/L_m/L_out = {d.L_in:.1f} / {d.L_m:.1f} / {d.L_out:.1f} m")
    print(f"  L = {d.L:.1f} m         mesh {d.nx} x {d.ny}")
    print(f"  k_OM = {d.k_OM:.6e}     k_c = {d.k_c:.6e}")

    # 1. base state is an exact steady solution
    res = base_residual(ntil, d)
    scale = G * d.I
    check("base state satisfies the streamwise balance",
          np.max(np.abs(res)) / scale < 1e-13,
          f"max|res|/gI = {np.max(np.abs(res)) / scale:.2e}")

    # 2. depth positive, contrast as predicted
    H = base_depth(ntil, d)
    ub = base_velocity(ntil, d)
    ratio_pred = (ub.max() / ub.min()) ** 2
    check("depth positive everywhere", H.min() > 0, f"H_min = {H.min():.4f} m")
    check("depth contrast = (velocity contrast)^2",
          abs(H.max() / H.min() - ratio_pred) < 1e-12,
          f"H_max/H_min = {H.max() / H.min():.5f}")
    H_mean = width_mean(base_depth, d)
    check("width-mean depth hits H_ref (Gauss-Legendre, exact for a quartic)",
          abs(H_mean - cfg.H_ref) < 1e-13,
          f"mean H = {H_mean:.14f} m")
    check("trapezoid agrees with GL to its own O(h^2)",
          abs(_trapz(H, ntil) / 2.0 - H_mean) < 1e-5,
          f"trapz-GL = {_trapz(H, ntil) / 2.0 - H_mean:.2e}")

    # 3. Froude / aspect ratio actually realised
    Fr = ub / np.sqrt(G * H)
    print(f"\n  realised Froude: min {Fr.min():.4f}  max {Fr.max():.4f}"
          f"   (uniform by construction: F^2 = I/Cf)")
    check("Froude uniform across the section", np.ptp(Fr) < 1e-12,
          f"ptp = {np.ptp(Fr):.2e}")

    # 4. Ikeda wavenumbers: m=4 at k_OM, m=8 damped
    k4 = wavenumber_of(4, d)
    k8 = wavenumber_of(8, d)
    check("m=4 sits at k_OM", abs(k4 - d.k_OM) / d.k_OM < 1e-12,
          f"k4/k_OM = {k4 / d.k_OM:.12f}")
    check("m=8 is beyond the cutoff", k8 > d.k_c,
          f"k8/k_c = {k8 / d.k_c:.4f}")

    sys.path.insert(0, IKEDA_DIR)
    from ikeda_lib import growth_rate  # noqa: E402
    a4 = growth_rate(k4, Cf=cfg.Cf, A=cfg.A_ikeda, F=cfg.F_ref) / cfg.Cf**2
    a8 = growth_rate(k8, Cf=cfg.Cf, A=cfg.A_ikeda, F=cfg.F_ref) / cfg.Cf**2
    print(f"  alpha0/Cf^2 : m=4 {a4:+.6e}   m=8 {a8:+.6e}")
    check("linear theory: m=4 grows", a4 > 0)
    check("linear theory: m=8 decays", a8 < 0)

    # 5. planform / map
    x = np.linspace(0.0, d.L, d.nx + 1)
    for m in (4, 8):
        dm = build_design(Config(**{**cfg.__dict__, "n_wave": m}))
        yN, yS = initial_banks(x, dm)
        c = centreline(yN, yS)
        tmax, werr = check_map(x, yN, yS)
        straight = (x < dm.x_m0 - 1e-9) | (x > dm.x_m1 + 1e-9)
        check(f"m={m}: entry/exit reaches straight",
              np.max(np.abs(c[straight])) < 1e-12,
              f"max|c| = {np.max(np.abs(c[straight])):.2e} m")
        check(f"m={m}: width constant at t=0",
              np.ptp(half_width(yN, yS)) < 1e-12)
        check(f"m={m}: no centreline drift",
              abs(0.5 * (c.max() + c.min())) < 1e-9,
              f"mid-range = {0.5 * (c.max() + c.min()):.2e} m")
        print(f"  m={m}: max|theta| = {np.degrees(tmax):.2f} deg, "
              f"normal-width error = {werr:.2e}")

    # 6. bed frozen in time / falls downstream
    z0 = bed_elevation(0.0, ntil, d)
    zL = bed_elevation(d.L, ntil, d)
    check("bed falls downstream by I*L",
          abs((z0 - zL).mean() - d.I * d.L) < 1e-10,
          f"drop = {(z0 - zL).mean():.4f} m")
    check("bed cross-section identical at every x",
          np.max(np.abs((z0 - z0.mean()) - (zL - zL.mean()))) < 1e-12)

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
