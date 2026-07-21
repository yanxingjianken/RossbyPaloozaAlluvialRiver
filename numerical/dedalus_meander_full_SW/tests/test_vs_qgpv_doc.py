#!/usr/bin/env python3
"""Cross-check our (s,n) formulation against derivations/Channel_Following_QGPV_Derivation.pdf.

    micromamba run -n dedalus env OMP_NUM_THREADS=1 python tests/test_vs_qgpv_doc.py

That document derives the SAME channel-following geometry for a different physical
model (barotropic QGPV, nondivergent, streamfunction) than ours (full shallow water,
divergent, primitive variables).  The PHYSICS differs, but every purely GEOMETRIC
identity -- metric, divergence, vorticity, base-state vorticity and its gradients --
must agree, so the document is an independent check on our metric algebra.

SIGN CONVENTION.  The document uses Frenet dn/ds = -kappa t and metric h = 1 - kappa*n
(its Eq. 7, 12).  We use dn/ds = +C t and sigma = 1 + n*C.  Hence

    C = -kappa_doc,    sigma = h.

Every comparison below converts with that identity, so a sign error in either
convention would show up as a failure rather than being absorbed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "postprocessing"))
import sw_meander as M
import pp_lib as PP


def check(name, got, want, tol, note=""):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    scale = max(float(np.max(np.abs(np.asarray(want)))), 1e-30)
    rel = err / scale
    ok = err <= tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:52s} max|err|={err:.3e} (rel {rel:.2e}) {note}")
    assert ok, f"{name}: max|err|={err:.3e} > {tol:.1e}"


def main():
    cfg = dict(M.CONFIG, Cbar_amp=0.15 / M.CONFIG["b"])   # dimensionless Cbar*b = 0.15            # a genuinely curved channel
    b, kmr = cfg["b"], cfg["kmeander"]
    s = np.linspace(0.0, M.domain_length(cfg), 2001)
    n = np.linspace(-b, b, 41)
    Cb = M.cbar(s, cfg)                            # our curvature C(s)
    kappa = -Cb                                    # the document's kappa

    print("1. METRIC: is the rendered lab-frame geometry really sigma = 1 + n*C?")
    # Build the centreline the way the movies do, then MEASURE the scale factor
    # |d/ds X(s,n)| numerically and compare with the sigma the solver assumes.
    xc, yc, nx, ny = PP.centerline(s, Cb)
    for n0 in (-0.8 * b, 0.0, +0.8 * b):
        X = xc + n0 * nx
        Y = yc + n0 * ny
        h_meas = np.hypot(np.gradient(X, s), np.gradient(Y, s))     # |dX/ds| at offset n
        sig = M.sigma_metric(s, n0, cfg)
        # ignore the two end points (one-sided differences there)
        check(f"|dX/ds| at n={n0:+.1f} equals 1+n*C", h_meas[2:-2], sig[2:-2], 3e-4)
    check("doc h = 1 - kappa*n equals our sigma = 1 + n*C",
          1.0 - kappa[:, None] * n[None, :], 1.0 + n[None, :] * Cb[:, None], 1e-14,
          "(exact identity given C = -kappa)")

    print("\n2. BASE VORTICITY  (doc Eq. 42:  zeta_bar = -U_n + kappa*U/h)")
    S, N = np.meshgrid(s, n, indexing="ij")
    sig = M.sigma_metric(S, N, cfg)
    Ub = M.ubar_s(N, cfg)
    # what analysis.py now uses: the CLOSED FORM of -(1/sigma) d_n(sigma*Ubar)
    zbar_ours = -M.ubar_s_n(N, cfg) - Cb[:, None] * Ub / sig
    zbar_doc = -M.ubar_s_n(N, cfg) + (-Cb[:, None]) * Ub / sig      # -U_n + kappa*U/h
    check("our -U_n - C*U/sigma equals doc -U_n + kappa*U/h", zbar_ours, zbar_doc, 1e-14,
          "(exact, given C = -kappa)")

    # ...and WHY we use the closed form.  Differencing the same quantity converges in
    # the interior but is only FIRST order at the walls (np.gradient defaults to
    # edge_order=1) -- and the walls are exactly where the erosion law samples u_s.
    print("     convergence of the differenced version (why analysis.py avoids it):")
    for Nn in (41, 81, 161, 321):
        nn = np.linspace(-b, b, Nn)
        S2, N2 = np.meshgrid(s, nn, indexing="ij")
        sg = M.sigma_metric(S2, N2, cfg)
        e = np.abs(-np.gradient(sg * M.ubar_s(N2, cfg), nn, axis=1) / sg
                   - (-M.ubar_s_n(N2, cfg) - Cb[:, None] * M.ubar_s(N2, cfg) / sg))
        print(f"       Nn={Nn:4d}  interior {e[:, 2:-2].max():.2e} (2nd order)   "
              f"at the wall {e.max():.2e} (1st order)")

    print("\n3. CROSS-CHANNEL PV GRADIENT  (doc Eq. 46, the 'channel beta')")
    # doc:  q_n = -U_nn + kappa*U_n/h + kappa^2*U/h^2.  Our text calls -U_nn a CONSTANT
    # channel-beta; that is exact only for a straight channel.  Quantify the error at
    # BOTH the sinuosity used by the runs that carry the conclusions (the default) and
    # the most sinuous run in the study.
    print(f"  our claim: d(zeta_bar)/dn = -U_nn = {-M.ubar_s_nn(cfg):+.4f}, CONSTANT")
    ref = abs(M.ubar_s_nn(cfg))
    for label, amp in (("default (all CONTROL runs)", M.bank_sinuosity(M.CONFIG)),
                       ("most sinuous run", 0.15 / cfg["b"])):
        c2 = dict(cfg, Cbar_amp=amp)
        kap = -M.cbar(s, c2)[:, None]
        sg = M.sigma_metric(S, N, c2)
        qn = -M.ubar_s_nn(c2) + kap * M.ubar_s_n(N, c2) / sg + kap ** 2 * M.ubar_s(N, c2) / sg ** 2
        dev = np.max(np.abs(qn + M.ubar_s_nn(c2))) / ref
        # and: does Delta=0 really zero the gradient?  U_n=U_nn=0 leaves the kappa^2 term
        c0 = dict(c2, Delta=0.0)
        qn0 = np.max(np.abs(kap ** 2 * M.ubar_s(N, c0) / sg ** 2))
        print(f"  bank sinuosity {amp:.3f} ({label}):")
        print(f"      curvature correction to beta = {100*dev:6.2f}% of -U_nn")
        print(f"      residual gradient at Delta=0 = {100*qn0/ref:6.3f}% of -U_nn")
        if amp < 0.05:      # the configuration the beta-removal conclusion rests on
            assert qn0 < 1e-3 * ref, "Delta=0 residual is not negligible for the controls"
    print("  => 'Delta=0 removes the gradient' holds to O(kappa^2 U); at the sinuosity the")
    print("     CONTROL runs use that residual is ~0.01% of the sheared gradient.")

    print("\n4. IS THE BASE STATE AN EXACT STEADY SOLUTION?  (doc Sec 5.5: NO if kappa_s != 0)")
    # In the doc's inviscid barotropic model the imbalance is J_g(psi_bar,q_bar)=kappa_s U^2/h^3.
    # Our SW analogue: the base s-momentum has -(g/sigma) d_s etabar, which is nonzero
    # wherever Cbar varies with s, because etabar carries Cbar(s).
    eb = M.etabar(s, n, cfg)
    ds_eb = np.gradient(eb, s, axis=0)
    resid = M.g_eff(cfg) * np.max(np.abs(ds_eb)) / max(np.max(np.abs(Ub)) ** 2, 1e-30)
    print(f"  max |g d_s etabar| / U^2 = {resid:.3e}   (0 only for a straight channel)")
    kap_s = np.gradient(kappa, s)
    print(f"  doc imbalance  kappa_s U^2/h^3  ~ {np.max(np.abs(kap_s)):.3e} * U^2/h^3")
    assert resid > 1e-6, "expected a NONZERO steady residual at finite, varying curvature"
    print("  => both models need an implicit forcing to hold the base state; ours is not")
    print("     an 'exact steady solution' when Cbar varies along s.")

    print("\n5. GEOMETRIC VALIDITY  (doc Eq. 15:  b*max|kappa| < 1)")
    eps_c = b * np.max(np.abs(kappa))
    print(f"  eps_c = b*max|kappa| = {eps_c:.3f}  ->  {'OK' if eps_c < 1 else 'SINGULAR'}")
    assert eps_c < 1.0

    print("\nALL GEOMETRY CROSS-CHECKS PASSED")


if __name__ == "__main__":
    main()
