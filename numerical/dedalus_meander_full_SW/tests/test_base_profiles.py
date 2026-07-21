#!/usr/bin/env python3
"""Base-state sanity checks for sw_meander (kept OUT of the shared driver).

    micromamba run -n dedalus env OMP_NUM_THREADS=1 python tests/test_base_profiles.py

Checks the analytic base state before any Dedalus assembly:
  1. parabolic jet   Ubar_s(0)=U0+Delta, Ubar_s(+/-b)=U0
  2. constant channel-beta   d2 Ubar_s/dn2 = -2 Delta/b^2  (the Rossby restoring)
  3. metric positivity   sigma = 1 + n Cbar > 0  (no channel folding)
  4. superelevation   outer bank higher, sign follows Cbar
  5. straight limit   Cbar=0  =>  etabar == 0
  6. run_tag() separates every knob it claims to key on (anti-overwrite)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sw_meander as M


def test_base_profiles():
    cfg = dict(M.CONFIG)
    n = np.linspace(-cfg["b"], cfg["b"], 201)

    # 1. parabolic jet
    U = M.ubar_s(n, cfg)
    assert abs(U[len(U) // 2] - M.center_speed(cfg)) < 1e-9, "centre speed != U0+Delta"
    assert abs(U[0] - cfg["U0"]) < 1e-9, "bank-edge speed != U0"
    print(f"1. parabolic jet: centre={U[len(U)//2]:.3f}, edge={U[0]:.3f}  OK")

    # 2. constant channel-beta
    d2 = np.gradient(np.gradient(U, n), n)
    assert abs(np.median(d2) - M.ubar_s_nn(cfg)) < 1e-2, "d2Ubar/dn2 not constant"
    print(f"2. channel-beta: d2Ubar/dn2={np.median(d2):.4f} == {M.ubar_s_nn(cfg):.4f} "
          f"(=> d(zetabar)/dn = {2*cfg['Delta']/cfg['b']**2:.4f} = const)  OK")

    # 3. metric positivity (no folding)
    s = np.linspace(0, 2 * np.pi / cfg["kmeander"], 101)
    S, N = np.meshgrid(s, n, indexing="ij")
    sig = M.sigma_metric(S, N, cfg)
    assert sig.min() > 0, "meander too tight: sigma<=0 (folding); reduce Cbar_amp/A_bank"
    print(f"3. metric: min(1+n*Cbar)={sig.min():.3f} > 0  OK")

    # 4/5. superelevation, and its straight-channel limit
    cfg_bend = dict(cfg, Cbar_amp=0.15 / cfg["b"])   # Cbar*b = 0.15
    eb = M.etabar(s, n, cfg_bend)
    ic = int(np.argmax(np.abs(M.cbar(s, cfg_bend))))
    tilt = eb[ic, -1] - eb[ic, 0]
    assert abs(tilt) > 0, "no superelevation at finite curvature"
    print(f"4. superelevation at max-curvature s: (n=+b)-(n=-b) = {tilt:+.4e}  OK")
    eb0 = M.etabar(s, n, dict(cfg, Cbar_amp=0.0, A_bank=0.0))
    assert np.max(np.abs(eb0)) < 1e-12, "straight channel must have etabar == 0"
    print(f"5. straight limit: max|etabar|={np.max(np.abs(eb0)):.1e}  OK")

    print("\nALL BASE-PROFILE CHECKS PASSED")


def test_run_tag_separates_physics():
    """Every physical knob must change the filename, or runs overwrite each other.

    This is a regression test: Delta was once absent from run_tag(), so the plug-flow
    (Delta=0) and reversed-shear (Delta<0) experiments -- the two sharpest runs in the
    study -- both landed on the SAME file and the second silently destroyed the first.
    A collision is invisible in the output (you still get N-1 plausible files), so it
    has to be caught here.
    """
    base = dict(M.CONFIG)
    knobs = dict(cross_amp=base["cross_amp"] + 0.3, Cbar_amp=0.15 / base["b"], Cf=base["Cf"] * 2,
                 U0=base["U0"] + 0.4, Delta=-base["Delta"])
    tag0 = M.run_tag(base)
    for knob, val in knobs.items():
        tag = M.run_tag(dict(base, **{knob: val}))
        assert tag != tag0, (f"run_tag() ignores '{knob}': changing it leaves the tag "
                             f"at {tag0!r}, so two runs would overwrite each other")
    print(f"6. run_tag separates {', '.join(knobs)}  ({tag0})  OK")


if __name__ == "__main__":
    test_base_profiles()
    test_run_tag_separates_physics()
