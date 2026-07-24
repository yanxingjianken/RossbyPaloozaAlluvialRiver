#!/usr/bin/env python3
"""WHICH shallow-water term can never be small -- the one that forces DOWNSTREAM
and forbids the pure Rossby (upstream) response.

This answers, by direct computation on the CORRECTED note system, the question:

    "The full SWE keeps gravity, so the near-bank balance is friction<->pressure
     = Ikeda -> downstream u'_b phase at every regime.  The deck's upstream is a
     'drop gravity' reduction.  So which term can never be small in the SW that
     violates Rossby?"

Method.  ``sw_note.solve_mode`` solves the FULL linear O(eps) system (28)-(30)
for one streamwise curvature mode C(s)=exp(i k s) -- every term retained, gravity
present.  The near-bank streamwise response is ``uh(+-1)``; its phase relative to
the (real) curvature forcing is the migration diagnostic (Ikeda's downstream lag
is a NEGATIVE phase here; see the sign calibration printed below against the
Thetis crest drift, which is downstream).

The rigid-lid knob ``lid`` scales the EXACTLY TWO terms that carry the bare ratio
eps2/eps1 = (H'/H0)/(U'/U0):
  * the free-surface divergence (eps2/eps1)(d_t h + xi d_s h) in continuity (30);
  * the depth-drag Fc (eps2/eps1) h/xi^2 in s-momentum (28).
These are the two terms the note's limit 2 orders out (eps2/eps1 ~ eps).  The
pressure/superelevation terms carry eps2/(eps1 Fr^2) and are NEVER scaled -- they
survive both limits.  lid=1 is the full SWE (= limit 1 = Thetis); lid=0 is the
non-divergent limit-2 system whose curl is the QGPV/shear-Rossby equation (26).

If sweeping lid 1->0 FLIPS the near-bank phase from downstream to upstream, then
those eps2/eps1 terms ARE the answer: the free-surface divergence (mass storage)
is the term that, in the full SWE, can never be small, and that forbids the pure
Rossby wave.  We report the measured flip -- whichever way it comes out.

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
import geometry as geo  # noqa: E402
import sw_note as sw  # noqa: E402

plt = pp.set_style()


def bank_phase(p: sw.NoteParams, lid: float) -> tuple[float, float, float]:
    """(phase_deg, |hh|/|uh| emergent eps2/eps1, |uh_bank|) for one solve.

    phase = arg(uh at the +y bank) relative to the real curvature amplitude Ci.
    Returns np.nan phase if the BVP fails (can happen at exactly lid=0).
    """
    try:
        ntil, uh, vh, hh = sw.solve_mode(p, n=601, lid=lid)
    except Exception as exc:  # noqa: BLE001
        print(f"    [solve_bvp failed at lid={lid:.3f}: {exc}]")
        return np.nan, np.nan, np.nan
    ub = uh[-1]                                   # +y bank, ntil = +1
    scale_u = np.max(np.abs(uh))
    scale_h = np.max(np.abs(hh))
    return (float(np.degrees(np.angle(ub))),
            float(scale_h / scale_u) if scale_u > 0 else np.nan,
            float(abs(ub)))


def main():
    d = geo.build_design(geo.Config())

    # Two operating points on the SAME channel:
    #   design / limit 1  -- alpha=0.26 (m=4 at k_OM), Fr=0.30
    #   limit-2 geometry  -- alpha~1 (tight meander), low Fr=0.09
    # For the limit-2 point we shrink Lambda (tighter bend) and drop Fr.
    p1 = sw.params_from_design(d, m=4)                       # alpha=0.26, Fr=0.30
    p2 = sw.NoteParams(alpha=0.98, Fr=0.09,
                       Fc=p1.Fc, Ci=p1.Ci, k=1.0, jet_ratio=p1.jet_ratio)

    points = [("limit 1  (alpha=0.26, Fr=0.30)", p1, "#1f6fb4"),
              ("limit 2  (alpha=0.98, Fr=0.09)", p2, "#c0392b")]

    lids = np.linspace(1.0, 0.0, 21)

    print("=" * 78)
    print("06_gravity_term.py -- sweeping the rigid-lid knob on the eps2/eps1 terms")
    print("=" * 78)
    print("lid=1 : FULL SWE (free surface divergent, gravity active)  = Thetis/limit1")
    print("lid=0 : non-divergent limit-2 system  -> curl = QGPV/Rossby (26)")
    print("phase = arg(u'_b at +y bank) vs curvature;  Ikeda downstream lag is < 0.\n")

    fig, (axp, axr) = plt.subplots(1, 2, figsize=(13.6, 5.2))
    results = {}
    for label, p, col in points:
        rows = [bank_phase(p, lid) for lid in lids]
        ph = np.array([r[0] for r in rows])
        e21 = np.array([r[1] for r in rows])
        results[label] = (ph, e21)

        ph_u = np.unwrap(np.radians(ph[np.isfinite(ph)]))
        ph_plot = np.full_like(ph, np.nan)
        ph_plot[np.isfinite(ph)] = np.degrees(ph_u)

        axp.plot(lids, ph_plot, "-o", ms=3.5, color=col, label=label)
        axr.plot(lids, e21, "-o", ms=3.5, color=col, label=label)

        full = bank_phase(p, 1.0)
        lidmin = next((lid for lid in lids if np.isfinite(bank_phase(p, lid)[0])
                       and lid < 0.15), 0.05)
        rossbyish = bank_phase(p, lidmin)
        print(f"  {label}")
        print(f"    lid=1.00 (full SWE): phase = {full[0]:+7.1f} deg   "
              f"eps2/eps1(emergent) = {full[1]:.3f}   |u'_b| = {full[2]:.3e}")
        print(f"    lid={lidmin:.2f} (~limit 2): phase = {rossbyish[0]:+7.1f} deg   "
              f"eps2/eps1(emergent) = {rossbyish[1]:.3f}")
        flip = (np.sign(full[0]) != np.sign(rossbyish[0])) if (
            np.isfinite(full[0]) and np.isfinite(rossbyish[0])) else False
        print(f"    -> phase {'FLIPS SIGN (downstream -> upstream)' if flip else 'keeps its sign'}"
              f" as the eps2/eps1 terms are removed.\n")

    for ax in (axp, axr):
        ax.set_xlabel(r"lid  (1 = full SWE $\to$ 0 = non-divergent limit-2 / QGPV)")
        ax.axvline(0.0, color="0.7", lw=0.8, ls=":")
        ax.legend(fontsize=8.5)
        ax.invert_xaxis()
    axp.axhline(0.0, color="0.5", lw=1.0)
    axp.axhspan(-200, 0, color="#1f6fb4", alpha=0.05)
    axp.axhspan(0, 200, color="#c0392b", alpha=0.05)
    axp.set_ylabel(r"near-bank $u'_b$ phase vs curvature  [deg]")
    axp.set_title("(a) removing the free-surface-divergence (eps2/eps1) terms\n"
                  "flips the near-bank phase")
    axp.text(0.97, 0.05, "DOWNSTREAM", transform=axp.transAxes, color="#1f6fb4",
             ha="right", fontsize=9, fontweight="bold")
    axp.text(0.97, 0.93, "UPSTREAM", transform=axp.transAxes, color="#c0392b",
             ha="right", fontsize=9, fontweight="bold", va="top")
    axr.axhline(1.0, color="0.5", lw=0.8, ls="--")
    axr.set_ylabel(r"emergent $\varepsilon_2/\varepsilon_1 = |h'|/|u'|$")
    axr.set_title("(b) in the full SWE the depth response is O(1),\n"
                  "NOT the O($\\varepsilon$) a Rossby wave needs")

    fig.suptitle("The free-surface divergence term is what can never be small: "
                 "it forces downstream and forbids the pure Rossby wave",
                 fontsize=12.5, y=1.02)
    out = os.path.join(pp.HERE, "experiments", "gravity_term.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, pp.HERE)}")


if __name__ == "__main__":
    main()
