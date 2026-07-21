#!/usr/bin/env python3
"""Three checks raised against Ikeda-Parker-Sawai (1981) and the (s,n) operators.

    micromamba run -n dedalus env OMP_NUM_THREADS=1 python tests/test_operators_and_closure.py

Q1  IPS81 n-momentum for the steady base flow reduces to -C u^2 = -g d_n xi - tau_n/(rho h).
    We drop tau_n.  Is that legitimate, and is the base state then OVER-determined?
Q2  IPS81 continuity carries an explicit  C*v*h  term.  Where is ours?
Q3  Are the curvilinear operators (grad, div, curl) actually right?

Q3 is answered by construction, not by algebra: build a Cartesian field whose divergence
and curl are known in closed form, resolve it onto the local (t,n) basis, apply OUR
formulas by finite differences in (s,n), and compare against the Cartesian truth.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "postprocessing"))
import sw_meander as M
import pp_lib as PP


def main():
    cfg = dict(M.CONFIG, Cbar_amp=0.15 / M.CONFIG["b"])   # dimensionless Cbar*b = 0.15
    b = cfg["b"]

    # ================================================================= Q3 ====
    print("Q3. Are the (s,n) operators right?  (checked against Cartesian truth)")
    Ns, Nn = 1200, 121
    s = np.linspace(0.0, M.domain_length(cfg), Ns)
    n = np.linspace(-0.9 * b, 0.9 * b, Nn)
    ds, dn = s[1] - s[0], n[1] - n[0]
    Cb = M.cbar(s, cfg)
    xc, yc, nx, ny = PP.centerline(s, Cb)
    # n = (-sin,cos)th  =>  t = (cos,sin)th = (ny, -nx).   (Writing (-ny, nx) instead
    # gives -t, which silently flips F_s and destroys the div/curl cancellation -- that
    # was a bug in an earlier version of THIS TEST, not in the operators.)
    tx, ty = ny, -nx
    X = xc[:, None] + n[None, :] * nx[:, None]
    Y = yc[:, None] + n[None, :] * ny[:, None]
    sig = 1.0 + n[None, :] * Cb[:, None]

    # A Cartesian field with div and curl known in closed form.  Keep |F| comparable to
    # |div F| and |curl F|: a field like (x^2, xy) has |F| ~ 6e3 while curl ~ 2, and that
    # 3000:1 cancellation multiplies the finite-difference error by 3000 -- which looks
    # like a broken operator but converges at a clean 2nd order all the same.
    #   F = (-y + sin(0.4 x),  x + cos(0.3 y))
    #   div F = 0.4 cos(0.4 x) - 0.3 sin(0.3 y)      curl F = 2
    Fx = -Y + np.sin(0.4 * X)
    Fy = X + np.cos(0.3 * Y)
    div_true = 0.4 * np.cos(0.4 * X) - 0.3 * np.sin(0.3 * Y)
    curl_true = 2.0 + 0.0 * X

    # resolve onto the LOCAL basis (t and n depend on s only)
    Fs = Fx * tx[:, None] + Fy * ty[:, None]
    Fn = Fx * nx[:, None] + Fy * ny[:, None]

    d_s = lambda A: np.gradient(A, ds, axis=0, edge_order=2)
    d_n = lambda A: np.gradient(A, dn, axis=1, edge_order=2)
    div_ours = (d_s(Fs) + d_n(sig * Fn)) / sig                 # our eq. for div
    curl_ours = (d_s(Fn) - d_n(sig * Fs)) / sig                # our eq. for zeta

    core = (slice(3, -3), slice(3, -3))
    for name, got, want in (("div  (1/s)[d_s F_s + d_n(s F_n)]", div_ours, div_true),
                            ("curl (1/s)[d_s F_n - d_n(s F_s)]", curl_ours, curl_true)):
        rel = np.max(np.abs(got[core] - want[core])) / np.max(np.abs(want[core]))
        print(f"   {name:38s} max rel err = {rel:.2e}  "
              f"{'OK (see convergence below)' if rel < 5e-3 else '*** WRONG ***'}")
        assert rel < 5e-3
    # the decisive evidence is not the size of the error but its ORDER: halving the
    # spacing must quarter it.  A wrong operator plateaus instead.
    print("   convergence (2nd order => the formulas are right, the residual is the FD):")
    prev = None
    for NS, NN in ((600, 61), (1200, 121), (2400, 241)):
        s2 = np.linspace(0.0, M.domain_length(cfg), NS)
        n2 = np.linspace(-0.9 * b, 0.9 * b, NN)
        C2 = M.cbar(s2, cfg)
        xc2, yc2, nx2, ny2 = PP.centerline(s2, C2)
        X2 = xc2[:, None] + n2[None, :] * nx2[:, None]
        Y2 = yc2[:, None] + n2[None, :] * ny2[:, None]
        sg2 = 1.0 + n2[None, :] * C2[:, None]
        F2s = (-Y2 + np.sin(0.4 * X2)) * ny2[:, None] + (X2 + np.cos(0.3 * Y2)) * (-nx2[:, None])
        F2n = (-Y2 + np.sin(0.4 * X2)) * nx2[:, None] + (X2 + np.cos(0.3 * Y2)) * ny2[:, None]
        c2 = ((np.gradient(F2n, s2[1] - s2[0], axis=0, edge_order=2)
               - np.gradient(sg2 * F2s, n2[1] - n2[0], axis=1, edge_order=2)) / sg2)
        e = np.max(np.abs(c2[core] - 2.0)) / 2.0
        print(f"      Ns={NS:5d} Nn={NN:4d}  curl rel err {e:.3e}"
              + (f"   ratio {prev/e:.1f}x" if prev else ""))
        prev = e
    assert prev < 3e-4

    # a scalar gradient, same idea
    phi = np.sin(0.7 * X) * np.cos(0.5 * Y)
    gx_true = 0.7 * np.cos(0.7 * X) * np.cos(0.5 * Y)
    gy_true = -0.5 * np.sin(0.7 * X) * np.sin(0.5 * Y)
    gs_ours, gn_ours = d_s(phi) / sig, d_n(phi)               # our eq. for grad
    gs_true = gx_true * tx[:, None] + gy_true * ty[:, None]
    gn_true = gx_true * nx[:, None] + gy_true * ny[:, None]
    rel = max(np.max(np.abs(gs_ours[core] - gs_true[core])),
              np.max(np.abs(gn_ours[core] - gn_true[core]))) / np.max(np.abs(gs_true[core]))
    print(f"   grad (1/s)d_s phi t + d_n phi n        max rel err = {rel:.2e}  "
          f"{'OK' if rel < 2e-3 else '*** WRONG ***'}")
    assert rel < 2e-3
    print("   => all three operators reproduce Cartesian truth; they are NOT wrong.")

    # ================================================================= Q2 ====
    print("\nQ2. IPS81 continuity has an explicit C*v*h.  Where is ours?")
    # ours (x sigma):   d_s(h u_s) + d_n(sigma h u_n) = 0
    # product rule:     d_n(sigma h u_n) = sigma d_n(h u_n) + (d_n sigma) h u_n
    #                                    = sigma d_n(h u_n) +      C      h u_n
    # so the IPS81 term is INSIDE our conservative d_n(sigma h u_n).  Verify numerically.
    # the identity rests on d_n(sigma) = C, which is EXACT (sigma = 1 + n*C):
    err_exact = np.max(np.abs(np.gradient(sig, n, axis=1, edge_order=2) - Cb[:, None]))
    print(f"   d_n(sigma) == C exactly              : max err {err_exact:.2e}")
    assert err_exact < 1e-12
    # and the product rule then holds to finite-difference accuracy on a sample field
    hu = (1.0 + 0.3 * np.cos(0.2 * X)) * (0.5 + 0.4 * np.sin(0.3 * Y))    # stand-in for h*u_n
    rel = (np.max(np.abs(d_n(sig * hu)[core] - (sig * d_n(hu) + Cb[:, None] * hu)[core]))
           / np.max(np.abs(d_n(sig * hu)[core])))
    print(f"   d_n(s h u_n) == s d_n(h u_n) + C h u_n : rel err {rel:.2e} (FD)")
    assert rel < 1e-4
    print("   => the C*v*h term is not missing; it is the (d_n sigma)=C piece of our")
    print("      conservative dn(sighb*un).  IPS81 writes the weak-curvature form")
    print("      (sigma->1 on the d_n term); ours keeps sigma exactly.")

    # ================================================================= Q1 ====
    print("\nQ1. Base-state n-momentum: is tau_n droppable, and is eta_bar over-determined?")
    print("   (a) tau_n for the BASE state.  Chezy drag tau_n = rho C_f |u| v, and the")
    print("       base flow has v_bar = 0 identically  =>  tau_n = 0.")
    print("       So IPS81's -C u^2 = -g d_n xi - tau_n/(rho h) reduces EXACTLY to our")
    print("       g d_n etabar = Cbar Ubar^2 / sigma.  Nothing is dropped.")

    print("   (b) but that FIXES eta_bar, and the s-momentum is then left over:")
    # steady base s-momentum with d_s Ubar = 0:   0 = -(g/sigma) d_s etabar - C_f Ubar^2/h
    nn = np.linspace(-b, b, 81)
    eb = M.etabar(s, nn, cfg)
    Ub = M.ubar_s(nn, cfg)[None, :]
    Hn = M.bed_depth(nn, cfg)[None, :]
    sg = 1.0 + nn[None, :] * Cb[:, None]
    pressure = -(M.g_eff(cfg) / sg) * np.gradient(eb, s, axis=0, edge_order=2)
    friction = -cfg["Cf"] * Ub ** 2 / Hn
    resid = pressure + friction
    print(f"       |pressure term|  max = {np.max(np.abs(pressure)):.4f}")
    print(f"       |friction term|  max = {np.max(np.abs(friction)):.4f}")
    print(f"       |residual|       max = {np.max(np.abs(resid)):.4f}"
          f"   ({100*np.max(np.abs(resid))/np.max(np.abs(friction)):.0f}% of friction)")
    assert np.max(np.abs(resid)) > 1e-6
    print("   => the base state IS over-determined: prescribing Ubar(n), H(n) and Cbar(s)")
    print("      fixes eta_bar through n-momentum, after which s-momentum cannot also")
    print("      hold.  A real river closes it with a down-valley SLOPE, g*S = C_f U^2/h,")
    print("      which this model does not carry explicitly.  The mismatch is exactly the")
    print("      'implicit mean forcing' the derivation invokes, and it cancels from the")
    print("      PERTURBATION equations -- which is why the linear results are unaffected.")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
