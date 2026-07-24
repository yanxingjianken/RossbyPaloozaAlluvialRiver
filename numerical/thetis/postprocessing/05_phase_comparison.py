#!/usr/bin/env python3
"""The decisive phase comparison: WHY Thetis is downstream and the deck is upstream.

Same erosion law (river.pdf p19 == Ikeda, E=eps*Cf, corrected PDF). The migration
direction is the sign of the near-bank velocity u'_b phase relative to the bend
curvature.  Two FLOW models compute u'_b with OPPOSITE-sign phase:

  * Ikeda gravity/friction flow (limit 1): u'_b LAGS the apex (downstream) --
    from the near-bank friction<->pressure balance; c0 = omega0/k > 0.
  * 3-level QGPV flow (limit 2, deck): u'_b = (psi2-psi1)/b LEADS (upstream) --
    from the PV/vorticity streamfunction; c0 = -E D/gamma < 0, at ALL aspect
    ratios (so upstream is a QGPV-model property, not a limit-2-geometry one).

Thetis full 2-D SWE keeps gravity -> Ikeda downstream phase in every regime,
even alpha~1 / low Fr.  Real rivers migrate downstream (Ikeda validated), so the
full SWE is the physical one; the deck's upstream is a gravity-dropping artefact.

    python postprocessing/05_phase_comparison.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_lib as pp  # noqa: E402
sys.path.insert(0, os.path.join(pp.HERE, "..", "ikeda_1981"))
sys.path.insert(0, os.path.join(pp.HERE, "..", "vorticity_meander"))
import ikeda_lib as ik  # noqa: E402
import vorticity_lib as vl  # noqa: E402

plt = pp.set_style()
Cf, F, D = 0.05, 0.30, 0.5
BSTAR = 26.0                       # representative b/H0 (17.5 lim1 .. 35 lim2)


def ikeda_phase(alpha):
    k = alpha / BSTAR              # Ikeda k normalised by H0
    uh = (Cf * (0 + F**2) - 1j * k) / (2 * Cf + 1j * k)
    return np.degrees(np.angle(uh))


def qgpv_phase(alpha, gamma=0.1):
    r = vl.forced_response(alpha, D, gamma, friction="rayleigh")
    return np.degrees(np.angle(r - 1.0))


def main():
    al = np.linspace(0.08, 1.5, 60)
    ph_ik = np.array([ikeda_phase(a) for a in al])
    ph_qg = np.array([qgpv_phase(a) for a in al])

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.axhspan(-180, 0, color="#1f6fb4", alpha=0.06)
    ax.axhspan(0, 180, color="#c0392b", alpha=0.06)
    ax.axhline(0, color="0.5", lw=1.0)
    ax.plot(al, ph_ik, color="#1f6fb4", lw=2.5,
            label=r"Ikeda gravity flow (limit 1) $\to$ DOWNSTREAM")
    ax.plot(al, ph_qg, color="#c0392b", lw=2.5,
            label=r"3-level QGPV flow (limit 2, deck) $\to$ UPSTREAM")
    # Thetis: what we ACTUALLY measured is the crest-drift DIRECTION (downstream)
    # at alpha=0.26 (A0/A2.89) and alpha~0.98 (limit 2) -- not an exact phase.
    # Show that as arrows in the downstream (blue) half, NOT fake phase points.
    for a in (0.26, 0.98):
        ax.annotate("", xy=(a, -150), xytext=(a, -120),
                    arrowprops=dict(arrowstyle="-|>", color="#12406e", lw=2.2))
    ax.scatter([], [], marker=r"$\downarrow$", s=90, color="#12406e",
               label="Thetis full 2-D SWE: measured crest drift DOWNSTREAM\n"
                     r"(at $\alpha$=0.26 and $\alpha\approx$0.98)")
    ax.axvline(0.26, color="0.6", lw=0.8, ls=":")
    ax.axvline(0.98, color="0.6", lw=0.8, ls=":")
    ax.text(0.26, 165, "limit 1\n(A0, A2.89)", ha="center", fontsize=8.5, color="0.4")
    ax.text(0.98, 165, "limit 2\n(α≈1)", ha="center", fontsize=8.5, color="0.4")
    ax.text(1.45, -90, "DOWNSTREAM\n(u$'_b$ lags apex)", ha="right", color="#1f6fb4",
            fontsize=10, fontweight="bold", va="center")
    ax.text(1.45, 90, "UPSTREAM\n(u$'_b$ leads apex)", ha="right", color="#c0392b",
            fontsize=10, fontweight="bold", va="center")
    ax.set_xlabel(r"aspect ratio  $\alpha = k^* = k b$   (limit 1 small $\to$ limit 2 $\approx$1)")
    ax.set_ylabel(r"near-bank $u'_b$ phase vs curvature  [deg]")
    ax.set_ylim(-180, 180)
    ax.set_xlim(0.08, 1.5)
    ax.set_title("Same erosion law, opposite FLOW-RESPONSE phase:\n"
                 "the deck's upstream is a QGPV (gravity-dropping) feature, at every aspect ratio",
                 fontsize=11.5)
    ax.legend(loc="lower right", fontsize=9)
    out = os.path.join(pp.HERE, "experiments", "phase_comparison.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, pp.HERE)}")


if __name__ == "__main__":
    main()
