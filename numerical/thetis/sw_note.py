#!/usr/bin/env python3
"""Direct solution of the CORRECTED River_Meandering_SW note, eqs (28)-(30).

Source: ``literature/River_Meandering_SW.pdf`` ("Shallow water analysis",
Yaoxuan Zeng, 2026-07-23), as corrected in
``docs/River_Meandering_SW_corrected.pdf``.  The corrections that matter here:

  * the bed is restored (``h = eta - z_b``), so a base state exists at all and
    the depth may vary across the channel;
  * the base balance forces ``H_0(ntil) = H_0 * xi(ntil)**2`` with
    ``U_0(ntil) = U_0 * xi(ntil)``, hence the friction terms carry ``1/xi`` and
    ``1/xi**2`` rather than ``xi`` and ``xi**2``;
  * continuity carries the depth inside the derivatives,
    ``d_s(xi**2 u) + d_n(xi**2 v) + (eps2/eps1)(d_t h + xi d_s h) = 0``.

For a steady single streamwise Fourier mode ``C(s) = exp(i k s)`` and
``(u, v, h) = (uh, vh, hh)(ntil) * exp(i k s)`` the system reduces to

    (28)  i k xi uh + vh xi' + (e2/(e1 Fr^2)) i k hh
                             + 2 Fc uh/xi - Fc (e2/e1) hh/xi^2 = 0
    (29)  i k xi vh - (Ci/(e1 a^2)) xi^2
                             + (e2/(e1 Fr^2 a^2)) hh' + Fc vh/xi = 0
    (30)  i k xi^2 uh + (xi^2 vh)' + (e2/e1) i k xi hh = 0

with ``vh(+-1) = 0``.  (28) is algebraic in ``uh``, so this is a linear
two-point BVP for ``(hh, q)`` with ``q = xi^2 vh`` -- two first-order ODEs and
two boundary conditions.

This module is the reference the Thetis run must reduce to in the note's own
regime (``tests/test_sw_note.py``).  Run ``python sw_note.py`` for the
self-test, which includes a closed-form solution for the flat-jet limit.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import geometry as geo  # noqa: E402


# --------------------------------------------------------------------------- #
#  Non-dimensional parameters (the note's eq. 16)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoteParams:
    """The note's dimensionless groups.

    ``alpha = b/Lambda``, ``gamma = V/U``, ``eps1 = U/U_0``,
    ``eps2 = H/H_0``, ``Fr = U_0/sqrt(g H_0)``, ``Fc = C_f Lambda/H_0``,
    ``Ci = C b``.  The note sets ``alpha = gamma``.  ``Lambda = lambda/2pi``
    (the note never defines it -- see errata E7).
    """

    alpha: float
    Fr: float
    Fc: float
    Ci: float
    eps1: float = 1.0    # amplitude scales only set the overall normalisation
    eps2: float = 1.0    # of the linear response; ratios are what matter
    k: float = 1.0       # streamwise wavenumber in units of 1/Lambda
    jet_ratio: float = 0.30

    @property
    def gamma(self) -> float:
        return self.alpha

    def xi(self, ntil):
        """Base velocity shape, ubar = U_0 * xi.  Normalised to width-mean 1."""
        ntil = np.asarray(ntil, dtype=float)
        r = self.jet_ratio
        raw = 1.0 + r * (1.0 - ntil**2)
        return raw / (1.0 + 2.0 * r / 3.0)          # <raw> over [-1,1] = 1 + 2r/3

    def dxi(self, ntil):
        ntil = np.asarray(ntil, dtype=float)
        r = self.jet_ratio
        return -2.0 * r * ntil / (1.0 + 2.0 * r / 3.0)


def params_from_design(d: geo.Design, m: int, Lambda_over_lam: float = 1.0 / (2 * np.pi)) -> NoteParams:
    """Build :class:`NoteParams` at the geometry design point.

    ``Lambda`` is the note's streamwise length scale; the note never defines it
    (errata E7) and the choice shifts ``alpha`` by 2*pi and ``eps ~ alpha^2`` by
    ~40x, which is enough to move the problem between the two distinguished
    limits.  We state the convention: ``Lambda = lambda/(2 pi)``.
    """
    lam_m = d.L_m / m
    Lambda = Lambda_over_lam * lam_m
    a0 = d.cfg.amp0_over_b * d.b
    kappa_amp = a0 * (2.0 * np.pi / lam_m) ** 2      # curvature amplitude of a sine
    return NoteParams(
        alpha=d.b / Lambda,
        Fr=d.cfg.F_ref,
        Fc=d.cfg.Cf * Lambda / d.cfg.H_ref,
        Ci=kappa_amp * d.b,
        k=2.0 * np.pi * Lambda / lam_m,              # = 1 when Lambda = lam/2pi
        jet_ratio=d.cfg.jet_ratio,
    )


def epsilon_table(d: geo.Design, m: int) -> dict:
    """The plan's section 6.4 table, computed rather than assumed."""
    p = params_from_design(d, m)
    return {
        "m": m,
        "lambda_m [m]": d.L_m / m,
        "Lambda [m]": (d.L_m / m) / (2 * np.pi),
        "alpha = b/Lambda": p.alpha,
        "Fr": p.Fr,
        "Fr^2": p.Fr**2,
        "Fc = Cf*Lambda/H0": p.Fc,
        "Ci = kappa*b": p.Ci,
        "k*Lambda": p.k,
    }


# --------------------------------------------------------------------------- #
#  The BVP
# --------------------------------------------------------------------------- #
def _uh_of(hh, vh, ntil, p: NoteParams, lid: float = 1.0, pgrad: float = 1.0):
    """Solve (28) algebraically for uh.

    ``lid`` scales the ONE term of (28) that carries the bare ratio
    ``eps2/eps1`` -- the depth-drag term ``Fc (eps2/eps1) hh/xi**2``.  It is the
    s-momentum half of the rigid-lid knob (see :func:`solve_mode`): ``lid=1`` is
    the full system (28); ``lid=0`` is the limit-2 s-momentum (note eq. 23),
    which drops exactly this term while KEEPING the pressure/superelevation term.

    ``pgrad`` scales the streamwise pressure-gradient / superelevation term
    ``(eps2/(eps1 Fr**2)) i k hh`` -- Ikeda eq. (7)'s ``-U**2 d_s C'`` forcing of
    the near-bank velocity.  Its coefficient ``eps2/(eps1 Fr**2)`` is O(1) in
    BOTH the note's limits (the QGPV limit 2 keeps it), so it is NOT a
    limit-selecting term; ``pgrad`` exists only to TEST that it is the term that
    sets the downstream near-bank phase (``postprocessing/06_gravity_term.py``
    shows pgrad: 1->0 collapses the downstream lag to ~0).  ``pgrad=1`` is
    physical.
    """
    xi = p.xi(ntil)
    dxi = p.dxi(ntil)
    num = -(vh * dxi
            + pgrad * (p.eps2 / (p.eps1 * p.Fr**2)) * 1j * p.k * hh
            - lid * p.Fc * (p.eps2 / p.eps1) * hh / xi**2)
    den = 1j * p.k * xi + 2.0 * p.Fc / xi
    return num / den


def solve_mode(p: NoteParams, n: int = 801, tol: float = 1e-10, lid: float = 1.0,
               pgrad: float = 1.0):
    """Solve (28)-(30) for one streamwise mode.

    Returns ``(ntil, uh, vh, hh)`` as complex arrays.
    State vector for :func:`solve_bvp` is the real/imag split of ``(hh, q)``
    with ``q = xi**2 * vh``; ``solve_bvp`` is real-only.

    ``lid`` is the **rigid-lid / free-surface-DIVERGENCE knob**.  It multiplies
    the exactly two terms that carry the bare ratio ``eps2/eps1`` (the
    free-surface divergence in continuity (30) and the depth-drag in
    s-momentum (28)) -- the two terms the note's limit 2 orders out
    (``eps2/eps1 ~ eps``).  ``lid=1`` is the FULL shallow-water system (28)-(30)
    = limit 1 / Thetis; ``lid=0`` is the non-divergent limit-2 system (notes
    eqs 23-25) whose curl is the QGPV / shear-Rossby equation (26).  Sweeping
    ``lid`` 1->0 does NOT flip the near-bank phase (still downstream): the
    divergence is not the migration-direction term.

    ``pgrad`` scales the cross-channel-superelevation streamwise pressure
    gradient (see :func:`_uh_of`) -- the term that DOES set the downstream phase
    (``pgrad`` 1->0 collapses the lag).  Its coefficient ~ ``1/(Fr**2 alpha**2)``
    is >= 1 for every subcritical (Fr<1) long-wave (alpha<1) meander and O(1/eps)
    in Ikeda's limit 1, so it can never be small -- it is the shallow-water term
    that forces downstream migration and is retained even by the QGPV limit 2.
    The deck's rigid-lid vorticity model (no free surface at all) is the only
    reduction that drops it, which is why it -- alone -- migrates upstream.
    ``pgrad=1`` is physical.
    """
    ntil0 = np.linspace(-1.0, 1.0, n)

    def rhs(t, y):
        hh = y[0] + 1j * y[1]
        q = y[2] + 1j * y[3]
        xi = p.xi(t)
        vh = q / xi**2
        uh = _uh_of(hh, vh, t, p, lid=lid, pgrad=pgrad)
        # (29) -> hh'  (the cross-channel superelevation balance; the alpha**2
        # factors are geometric, not a limit choice, so lid never touches it)
        dhh = ((p.Ci / (p.eps1 * p.alpha**2)) * xi**2
               - 1j * p.k * xi * vh
               - p.Fc * vh / xi) * (p.eps1 * p.Fr**2 * p.alpha**2 / p.eps2)
        # (30) -> q'   free-surface divergence term scaled by lid (eps2/eps1)
        dq = -(1j * p.k * xi**2 * uh
               + lid * (p.eps2 / p.eps1) * 1j * p.k * xi * hh)
        return np.vstack([dhh.real, dhh.imag, dq.real, dq.imag])

    def bc(ya, yb):
        # q = xi^2 vh = 0 at both banks (no flow through the walls)
        return np.array([ya[2], ya[3], yb[2], yb[3]])

    y0 = np.zeros((4, ntil0.size))
    sol = solve_bvp(rhs, bc, ntil0, y0, tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    ntil = np.linspace(-1.0, 1.0, n)
    y = sol.sol(ntil)
    hh = y[0] + 1j * y[1]
    q = y[2] + 1j * y[3]
    vh = q / p.xi(ntil) ** 2
    uh = _uh_of(hh, vh, ntil, p, lid=lid, pgrad=pgrad)
    return ntil, uh, vh, hh


# --------------------------------------------------------------------------- #
#  Closed form for the flat-jet limit (xi = 1)  -- the self-test anchor
# --------------------------------------------------------------------------- #
def solve_mode_flat_analytic(p: NoteParams, ntil):
    """Exact solution of (28)-(30) for ``xi == 1``.

    With ``xi = 1`` the coefficients are constant:

        uh = -P hh ,  P = (e2/e1)(i k/Fr^2 - Fc)/(i k + 2 Fc)
        vh' = A hh ,  A = i k (P - e2/e1)
        hh' = B - C vh ,  B = Ci Fr^2/e2 ,  C = (i k + Fc) e1 Fr^2 a^2/e2

    hence ``vh'' + A C vh = A B`` with ``vh(+-1) = 0``, giving

        vh = (B/C) [1 - cos(mu*ntil)/cos(mu)] ,  mu = sqrt(A C).
    """
    ntil = np.asarray(ntil, dtype=float)
    e21 = p.eps2 / p.eps1
    P = e21 * (1j * p.k / p.Fr**2 - p.Fc) / (1j * p.k + 2.0 * p.Fc)
    A = 1j * p.k * (P - e21)
    B = p.Ci * p.Fr**2 / p.eps2
    C = (1j * p.k + p.Fc) * p.eps1 * p.Fr**2 * p.alpha**2 / p.eps2
    mu = np.sqrt(A * C)
    vh = (B / C) * (1.0 - np.cos(mu * ntil) / np.cos(mu))
    hh = (vh * 0 + 1) * 0j
    # hh from hh' = B - C vh, integrated from the centre with hh(0) chosen so
    # that the mode has zero cross-channel mean of hh (gauge; only hh' enters).
    # hh(ntil) = B*ntil - C * int_0^ntil vh
    intv = (B / C) * (ntil - np.sin(mu * ntil) / (mu * np.cos(mu)))
    hh = B * ntil - C * intv
    uh = -P * hh
    return uh, vh, hh


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    print("=" * 74)
    print("sw_note.py self-test  (corrected River_Meandering_SW eqs 28-30)")
    print("=" * 74)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    # ---- 1. flat jet: numerical BVP vs the closed form -------------------
    p_flat = NoteParams(alpha=0.26, Fr=0.30, Fc=3.35, Ci=2.0e-3,
                        k=1.0, jet_ratio=0.0)
    ntil, uh, vh, hh = solve_mode(p_flat, n=401)
    ua, va, ha = solve_mode_flat_analytic(p_flat, ntil)
    # hh carries an additive gauge constant; compare its derivative-bearing part
    dv = np.max(np.abs(vh - va)) / max(np.max(np.abs(va)), 1e-300)
    hh_c = hh - hh[hh.size // 2]
    ha_c = ha - ha[ha.size // 2]
    dh = np.max(np.abs(hh_c - ha_c)) / max(np.max(np.abs(ha_c)), 1e-300)
    print(f"\n-- flat jet (xi=1), alpha={p_flat.alpha} Fr={p_flat.Fr} "
          f"Fc={p_flat.Fc} Ci={p_flat.Ci:g}")
    print(f"   max|vh| = {np.max(np.abs(va)):.6e}   (analytic)")
    check("BVP reproduces the closed-form vh", dv < 1e-6, f"rel err {dv:.2e}")
    check("BVP reproduces the closed-form hh", dh < 1e-6, f"rel err {dh:.2e}")

    # ---- 2. boundary conditions ------------------------------------------
    check("vh vanishes at both banks",
          max(abs(vh[0]), abs(vh[-1])) < 1e-10 * max(np.max(np.abs(vh)), 1e-30),
          f"|vh(-1)|={abs(vh[0]):.2e} |vh(+1)|={abs(vh[-1]):.2e}")

    # ---- 3. linearity in the forcing -------------------------------------
    p2 = NoteParams(**{**p_flat.__dict__, "Ci": 2 * p_flat.Ci})
    _, uh2, vh2, _ = solve_mode(p2, n=401)
    check("response is linear in Ci",
          np.max(np.abs(vh2 - 2 * vh)) / np.max(np.abs(vh)) < 1e-8,
          f"rel dev {np.max(np.abs(vh2 - 2 * vh)) / np.max(np.abs(vh)):.2e}")

    # ---- 4. sheared jet: grid convergence --------------------------------
    p_jet = NoteParams(alpha=0.26, Fr=0.30, Fc=3.35, Ci=2.0e-3,
                       k=1.0, jet_ratio=0.30)
    _, _, v_a, _ = solve_mode(p_jet, n=401)
    _, _, v_b, _ = solve_mode(p_jet, n=1601)
    rel = np.max(np.abs(v_a - v_b[::4])) / np.max(np.abs(v_b))
    print(f"\n-- sheared jet (jet_ratio=0.30), max|vh| = {np.max(np.abs(v_a)):.6e}")
    check("grid converged 401 -> 1601", rel < 1e-6, f"rel diff {rel:.2e}")

    # ---- 5. shear changes the answer (i.e. xi actually enters) -----------
    d_shear = np.max(np.abs(v_a - vh)) / np.max(np.abs(vh))
    check("the sheared jet differs from the flat one", d_shear > 1e-3,
          f"rel diff {d_shear:.3f}")

    # ---- 6. epsilon table at the design point ----------------------------
    d = geo.build_design(geo.Config())
    print("\n-- epsilon table at the design point (plan section 6.4) --")
    for m in (4, 8):
        tab = epsilon_table(d, m)
        print(f"   m={m}: " + "  ".join(
            f"{k}={v:.4g}" for k, v in tab.items() if k != "m"))
    t4 = epsilon_table(d, 4)
    check("k*Lambda = 1 by the Lambda = lambda/2pi convention",
          abs(t4["k*Lambda"] - 1.0) < 1e-12)
    check("alpha < 1 (channel narrower than the meander scale)",
          t4["alpha = b/Lambda"] < 1.0, f"alpha = {t4['alpha = b/Lambda']:.4f}")

    print("\n" + "=" * 74)
    if fails:
        print(f"FAILED ({len(fails)}): " + ", ".join(fails))
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
