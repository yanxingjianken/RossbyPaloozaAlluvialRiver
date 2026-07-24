#!/usr/bin/env python3
"""Analytic parameter search: can two runs be made qualitatively different --
one AMPLIFY / one DECAY, and one UPSTREAM / one DOWNSTREAM (i.e. one gravity-like,
one Rossby-like)?  And can that be achieved at A=0?  at A=2.89?

Answered from the VERIFIED dispersion relations on disk, with NO Thetis run:
  * gravity/friction branch  -> ../ikeda_1981/ikeda_lib.py   (growth_rate, celerity)
  * full-SWE vortical branch -> ../meander_migration/swe_stability.py (vortical_mode)
plus the gravity-vs-Rossby number R = beta_eff b^2 / (F^2 U).

Run:  micromamba run -n fourcastnetv2 python param_search.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ikeda_1981"))
sys.path.insert(0, os.path.join(HERE, "..", "meander_migration"))

import geometry as geo  # noqa: E402
import ikeda_lib as ik  # noqa: E402
import swe_stability as sw  # noqa: E402

G = 9.81


def ikeda_branch(m, d, A, Cf=0.05, F=0.30):
    """Gravity/friction branch at wavenumber m: (alpha0 [per t~], c0 [m/s])."""
    lam = d.L_m / m
    k = 2.0 * np.pi * d.cfg.H_ref / lam                       # Ikeda-normalised
    U = geo.width_mean(geo.base_velocity, d)
    a0 = ik.growth_rate(k, Cf=Cf, A=A, F=F)
    c0 = ik.celerity(k, Cf=Cf, A=A, F=F) * U                  # -> m/s
    return a0, c0


def vortical_branch(m, d, jet_ratio, Cf=0.05):
    """Full-SWE vortical/shear-Rossby branch: (sigma [1/s], c [m/s], R)."""
    b = d.b
    y = np.linspace(-b, b, 121)
    U0 = geo.width_mean(geo.base_velocity, d)
    U = U0 * (1 + jet_ratio * (1 - (y / b) ** 2)) / (1 + 2 * jet_ratio / 3)
    H = np.full_like(y, d.cfg.H_ref)
    F2 = U.max() ** 2 / (G * H.max())
    _, _, bq = sw.pv_gradient(y, U, H)
    beta_eff = np.abs(bq[len(y) // 4:3 * len(y) // 4]).mean()
    R = beta_eff * b ** 2 / (F2 * U.max())
    k = 2.0 * np.pi / (d.L_m / m)
    w = sw.vortical_mode(k, y, U, H, Cd=Cf)
    if w is None:
        return np.nan, np.nan, R
    return float(w.real), float(-w.imag / k), R


def main():
    print("=" * 78)
    print("param_search.py -- amplify/decay & up/downstream, gravity vs Rossby")
    print("=" * 78)
    d = geo.build_design(geo.Config())          # A=0 design grid (shared)
    U = geo.width_mean(geo.base_velocity, d)
    print(f"design grid: L_m={d.L_m:.0f} m, b={d.b:.1f} m, U={U:.3f} m/s, "
          f"lambda_m4={d.L_m/4:.0f} m, lambda_m8={d.L_m/8:.0f} m\n")

    for A in (0.0, 2.89):
        print(f"################  A = {A}  ({'incised' if A == 0 else 'alluvial'})  "
              f"################")
        print(f"{'m':>3} | GRAVITY/Ikeda            | VORTICAL/Rossby (jet_ratio=0.3)")
        print(f"{'':>3} | alpha0/Cf^2   c0 [m/s]    | sigma [1/s]  c [m/s]   R")
        print("-" * 74)
        for m in (2, 3, 4, 6, 8, 12):
            a0, c0 = ikeda_branch(m, d, A)
            sig, cv, R = vortical_branch(m, d, jet_ratio=0.3)
            print(f"{m:>3} | {a0/0.05**2:+9.3f}  {c0:+10.5f}    | "
                  f"{sig:+.2e}  {cv:+7.3f}  {R:4.1f}")
        # verdicts for this A
        aa = {m: ikeda_branch(m, d, A)[0] for m in (2, 3, 4, 6, 8, 12)}
        cc = {m: ikeda_branch(m, d, A)[1] for m in (2, 3, 4, 6, 8, 12)}
        amp = [m for m in aa if aa[m] > 0]
        dec = [m for m in aa if aa[m] < 0]
        print(f"\n  amplify/decay?  amplify m={amp}, decay m={dec}  "
              f"-> {'YES both exist' if amp and dec else 'NO'}")
        print(f"  up/downstream?  gravity c0 sign: "
              f"{set(np.sign(list(cc.values())).astype(int))}  "
              f"(all +1 = all DOWNSTREAM)")
        # vortical sign across jet strength
        signs = set()
        for jr in (0.3, 1.0, 3.0, 10.0):
            _, cv, _ = vortical_branch(4, d, jr)
            signs.add(int(np.sign(cv)))
        print(f"  vortical branch c sign over jet_ratio in [0.3,10]: {signs} "
              f"(all +1 = DOWNSTREAM even in the Rossby-dominated regime)\n")

    # what was swept, and the full-spectrum stability point
    print("SWEPT: A in {0, 2.89}; m in {2..12}; and (probed separately) F in [0.2,1.3],")
    print("       Cf, jet_ratio in [0.3,10].  The full SWE flow spectrum has NO growing")
    print("       mode at any of these -- the meander instability is MORPHODYNAMIC (the")
    print("       bank-erosion feedback), not a growing flow wave.\n")
    print("=" * 78)
    print("VERDICT  (contrast in EITHER growth OR celerity is acceptable)")
    print("=" * 78)
    print("""  GROWTH-sign contrast (amplify vs decay): ACHIEVABLE.
    - The morphodynamic growth flips sign at the Ikeda cutoff k_c, so two
      wavenumbers straddle it.  The DRIVER is (A + F^2):
        A=2.89 -> secondary-flow / VORTICAL-driven, alpha0/Cf^2 ~ +0.5 (strong)
        A=0    -> gravity / F^2 only,               alpha0/Cf^2 ~ +2e-3 (weak)
    - So A=2.89 (amplify) vs A=0 (weak -> decays, measured) IS the
      "one vortical-like, one gravity-like" contrast, distinguished by GROWTH.
      This is exactly the two subfolders.  Bed-tilt sign validated: A=2.89
      raises the near-bank velocity 1.48x -> more erosion -> growth.

  CELERITY-sign contrast (upstream vs downstream): reachable, but ONLY by
  changing the DISTINGUISHED LIMIT of the SW note (River_Meandering_SW.pdf) --
  NOT within limit 1 (which is all this design samples).
    - LIMIT 1 (eq 20-22: alpha~eps^1/2, Fr~Fc~1) -> Ikeda gravity/friction.
      omega0 = Cf k^3 (2+A+F^2)/(k^2+4Cf^2) > 0 -> meander migrates DOWNSTREAM
      for every k, A, F, Cf.  This design (lambda=12W, alpha=0.26) is limit 1,
      which is why the earlier "upstream unreachable" was TRUE ONLY HERE.
    - LIMIT 2 (eq 23-26: alpha=1, Fr^2~eps) -> the QGPV / shear-Rossby equation.
      Changing which nondim number dominates (drop Fr^2 to sub-leading, take
      alpha=1 i.e. lambda~pi*W) brings the PV dynamics to leading order.  The SWE
      spectrum already carries upstream slow modes (c<0) at Fr=0.3; and this
      limit is realised, with UPSTREAM meander migration, by the deck's 3-level
      QGPV model: ../../deliverable1_noboru_model, ../../vorticity_meander
      (c0 = -E D/gamma < 0).

  => TWO ways to draw the contrast:
     (growth)   A=2.89 amplify vs A=0 decay -- BOTH downstream, both LIMIT 1
                (secondary-flow/vortical vs gravity DRIVER).  Done: experiments/.
     (celerity) LIMIT 1 (Ikeda, downstream) vs LIMIT 2 (QGPV/Rossby, upstream) --
                the two distinguished limits of River_Meandering_SW.pdf.  Limit 2
                lives in deliverable1_noboru_model / vorticity_meander; a Thetis
                limit-2 run needs alpha~1 (lambda~pi*W, tight meanders) + low Fr.""")


if __name__ == "__main__":
    main()
