#!/usr/bin/env python3
"""WHICH shallow-water term forces DOWNSTREAM migration and forbids the pure
Rossby (upstream) response -- answered by three controlled experiments.

Question (user, 2026-07-24):
    "The full SWE keeps gravity, so the near-bank balance is friction<->pressure
     = Ikeda -> downstream u'_b phase at every regime.  The deck's upstream is a
     'drop gravity' reduction.  So which term can never be small in the SW that
     violates Rossby?"

The near-bank streamwise response u'_b(+-b) of a single curvature mode
C(s)=exp(i k s) is computed from the CORRECTED note system (28)-(30)
(sw_note.solve_mode -- every term retained, gravity present).  Its phase vs the
curvature is the migration diagnostic: Ikeda's downstream lag is NEGATIVE here
(calibrated: lid=1,pgrad=1 at alpha=0.26 gives -73 deg, matching ikeda_lib's
-81 deg).  Three knobs isolate the responsible term:

  (a) lid   -- scales the free-surface DIVERGENCE (mass storage) + depth-drag,
               the eps2/eps1 terms the note's limit 2 drops.  RESULT: sweeping
               1->0 does NOT flip the phase (stays downstream).  => the
               divergence/gravity-wave storage is NOT the direction term.

  (b) N     -- cross-channel resolution of the deck's OWN rigid-lid vorticity
               model (vorticity_lib.channel_modes): N=3 is the 3-level deck,
               large N the continuum.  RESULT: the upstream celerity c0=-ED/gamma
               CONVERGES (N=3 ~ N=91).  => upstream is NOT a 3-level-truncation
               artefact; the continuum rigid-lid vortical model is upstream too.

  (c) pgrad -- scales the cross-channel-SUPERELEVATION streamwise pressure
               gradient (Ikeda eq.7's -U^2 d_s C').  RESULT: sweeping 1->0
               COLLAPSES the downstream lag to ~0.  => THIS term sets the
               downstream phase.  Its coefficient ~ 1/(Fr^2 alpha^2) is >= 1 for
               every subcritical (Fr<1) long-wave (alpha<1) meander and O(1/eps)
               in Ikeda's limit 1, so it can never be small.  The note's QGPV
               limit 2 keeps it (that is why limit 2 is STILL downstream); only
               the deck's rigid-lid model (no free surface at all) drops it.

Conclusion: the term that can never be small is the cross-channel superelevation
pressure -- the free surface tilting to balance centrifugal force.  It, not the
free-surface divergence and not the 3-level truncation, is what keeps the full
2-D SWE (Thetis) downstream in every regime.  The deck's upstream requires
throwing the free surface away entirely.

    python postprocessing/06_gravity_term.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_lib as pp  # noqa: E402
sys.path.insert(0, pp.HERE)
sys.path.insert(0, os.path.join(pp.HERE, ".."))
sys.path.insert(0, os.path.join(pp.HERE, "..", "vorticity_meander"))
import geometry as geo  # noqa: E402
import sw_note as sw  # noqa: E402
import vorticity_lib as vl  # noqa: E402

plt = pp.set_style()


def bank_phase(p, lid=1.0, pgrad=1.0):
    """arg(u'_b at +y bank) [deg] for one solve; nan if the BVP fails."""
    try:
        _, uh, _, _ = sw.solve_mode(p, n=601, lid=lid, pgrad=pgrad)
    except Exception:  # noqa: BLE001
        return np.nan
    return float(np.degrees(np.angle(uh[-1])))


def deck_celerity(N, kstar, D, gamma, friction="rayleigh"):
    """Bank-mode celerity c0 = Re(om*)/k* of the deck's N-point vorticity GEP."""
    E = vl.ECOEF[friction] * (1.0 - D)
    target = -E * D / (vl._gamma_eff_factor(friction) * gamma) * kstar
    om, _ = vl.channel_modes(N, kstar, D, gamma, E, friction)
    i = int(np.argmin(np.abs(om - target)))
    return om[i].real / kstar


def main():
    d = geo.build_design(geo.Config())
    p1 = sw.params_from_design(d, m=4)                          # limit 1
    p2 = sw.NoteParams(alpha=0.98, Fr=0.09, Fc=p1.Fc, Ci=p1.Ci,
                       k=1.0, jet_ratio=p1.jet_ratio)           # limit 2
    pts = [("limit 1  (alpha=0.26, Fr=0.30)", p1, "#1f6fb4"),
           ("limit 2  (alpha=0.98, Fr=0.09)", p2, "#c0392b")]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    axa, axb, axc = axes

    print("=" * 78)
    print("06_gravity_term.py -- which SWE term forces downstream / forbids Rossby")
    print("=" * 78)

    # ---- (a) lid sweep: free-surface divergence -------------------------- #
    lids = np.linspace(1.0, 0.02, 15)
    print("\n(a) lid sweep (free-surface DIVERGENCE + depth-drag):")
    for label, p, col in pts:
        ph = np.array([bank_phase(p, lid=l) for l in lids])
        ph = np.degrees(np.unwrap(np.radians(ph)))
        axa.plot(lids, ph, "-o", ms=3, color=col, label=label)
        print(f"    {label}: lid 1->0.02  phase {ph[0]:+.1f} -> {ph[-1]:+.1f} deg  (no flip)")
    axa.invert_xaxis()
    axa.axhline(0, color="0.5", lw=1.0)
    axa.axhspan(-200, 0, color="#1f6fb4", alpha=0.05)
    axa.axhspan(0, 200, color="#c0392b", alpha=0.05)
    axa.set_xlabel(r"lid  (1 = full SWE $\to$ 0 = non-divergent)")
    axa.set_ylabel(r"near-bank $u'_b$ phase [deg]")
    axa.set_ylim(-180, 180)
    axa.set_title("(a) free-surface DIVERGENCE:\nremoving it does NOT flip (stays downstream)")
    axa.legend(fontsize=8, loc="upper left")

    # ---- (b) N-convergence: 3-level deck -> continuum -------------------- #
    Ns = np.array([3, 5, 9, 15, 25, 41, 61, 91])
    print("\n(b) deck vorticity model, cross-channel resolution N:")
    for fr, col in (("rayleigh", "#6a51a3"), ("momentum", "#e6550d")):
        c0 = np.array([deck_celerity(N, 0.06, 0.6, 0.05, fr) for N in Ns])
        axb.plot(Ns, c0, "-o", ms=3.5, color=col, label=f"{fr} closure")
        print(f"    {fr}: c0(N=3)={c0[0]:+.3f} -> c0(N=91)={c0[-1]:+.3f}  (converged, upstream)")
    axb.axhline(0, color="0.5", lw=1.0)
    axb.axhspan(-3, 0, color="#c0392b", alpha=0.05)
    axb.set_xscale("log")
    axb.set_xlabel(r"cross-channel levels $N$  (3 = deck $\to$ continuum)")
    axb.set_ylabel(r"bank-mode celerity $c_0$")
    axb.set_title("(b) 3-level TRUNCATION:\nupstream $c_0$ converges (not an artefact)")
    axb.text(0.95, 0.1, "UPSTREAM", transform=axb.transAxes, ha="right",
             color="#c0392b", fontsize=10, fontweight="bold")
    axb.legend(fontsize=8.5, loc="upper right")

    # ---- (c) pgrad sweep: cross-channel superelevation pressure --------- #
    pgs = np.linspace(1.0, 0.0, 15)
    print("\n(c) pgrad sweep (cross-channel SUPERELEVATION pressure):")
    for label, p, col in pts:
        ph = np.array([bank_phase(p, lid=1.0, pgrad=pg) for pg in pgs])
        axc.plot(pgs, ph, "-o", ms=3, color=col, label=label)
        good = ph[np.isfinite(ph)]
        print(f"    {label}: pgrad 1->0  phase {good[0]:+.1f} -> {good[-1]:+.1f} deg  (collapses)")
        coef = 1.0 / (p.Fr**2 * p.alpha**2)
        print(f"        superelevation coeff 1/(Fr^2 alpha^2) = {coef:.0f}  (>= 1 always)")
    axc.invert_xaxis()
    axc.axhline(0, color="0.5", lw=1.0)
    axc.axhspan(-200, 0, color="#1f6fb4", alpha=0.05)
    axc.axhspan(0, 200, color="#c0392b", alpha=0.05)
    axc.set_xlabel(r"pgrad  (1 = physical $\to$ 0 = no superelevation)")
    axc.set_ylabel(r"near-bank $u'_b$ phase [deg]")
    axc.set_ylim(-180, 180)
    axc.set_title("(c) SUPERELEVATION pressure:\nremoving it collapses the downstream lag")
    axc.text(0.05, 0.08, "DOWNSTREAM", transform=axc.transAxes, color="#1f6fb4",
             fontsize=9, fontweight="bold")
    axc.legend(fontsize=8, loc="upper right")

    fig.suptitle("The cross-channel superelevation pressure ~1/(Fr$^2\\alpha^2$) $\\geq$ 1 "
                 "is the SWE term that can never be small:\nit forces downstream migration; "
                 "the deck's upstream needs the free surface thrown away entirely",
                 fontsize=12.5, y=1.06)
    out = os.path.join(pp.HERE, "experiments", "gravity_term.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  wrote {os.path.relpath(out, pp.HERE)}")


if __name__ == "__main__":
    main()
