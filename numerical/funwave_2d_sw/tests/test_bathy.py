#!/usr/bin/env python3
"""Geometry and base-state checks for run_meander.py -- run BEFORE any FUNWAVE run.

    micromamba run -n fourcastnetv2 python tests/test_bathy.py

Every check is pinned by a SECOND, independent route: the analytic quantity in
run_meander.py is compared against a numerical one recomputed from the built arrays.
A test that only re-evaluates the same formula proves nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_meander as rm  # noqa: E402

CFG = rm.CONFIG
LAMS = [r["lam"] for r in rm.RUNS]
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# --------------------------------------------------------------------------- #
print("\n=== 1. the drive: |kappa|max must equal C0 EXACTLY for both runs ===")
# This is the whole premise.  The previous tapered-sine construction overshot C0 by 12-97%
# in the ramp, by an amount that depended on ramp/lam -- so the two cases did NOT share the
# drive.  A curvature-defined (sine-generated) curve fixes it identically.
kmax = {}
for lam in LAMS:
    x, y, s, _, _, kap = rm.centreline(lam, CFG)
    L = rm.reach_length(CFG); m = (x >= 0) & (x <= L)
    kmax[lam] = float(np.abs(kap[m]).max())
    check(f"lam={lam:.0f}: |kappa|max = C0", abs(kmax[lam] / CFG["C0"] - 1) < 5e-3,
          f"{kmax[lam]:.5e} vs C0 {CFG['C0']:.5e}  ({100*(kmax[lam]/CFG['C0']-1):+.2f}%)")
check("both runs share |kappa|max", abs(kmax[LAMS[0]] / kmax[LAMS[1]] - 1) < 5e-3,
      f"ratio = {kmax[LAMS[0]]/kmax[LAMS[1]]:.5f}")
W = 2 * CFG["b"]
for lam in LAMS:
    check(f"lam={lam:.0f}: R/W >= 2 (measured stability threshold)",
          1 / kmax[lam] / W >= 2.0, f"R/W = {1/kmax[lam]/W:.2f}")

print("\n=== 2. geometry of the built curve ===")
L = rm.reach_length(CFG)
check(f"lam requested <= lam_max = 5.073/C0", max(LAMS) <= rm.lam_max(CFG),
      f"lam_max = {rm.lam_max(CFG):.0f} m, longest requested {max(LAMS):.0f} m")
for lam in LAMS:
    x, y, s, _, _, _ = rm.centreline(lam, CFG)
    m = (x >= 0) & (x <= L)
    span = x[m][-1] - x[m][0]
    check(f"lam={lam:.0f}: the curve spans the reach without drifting",
          abs(span - L) / L < 1e-3, f"x span = {span:.1f} m vs L = {L:.0f} m")
    nb = span / lam
    check(f"lam={lam:.0f}: whole number of bends", abs(nb - round(nb)) < 5e-3,
          f"{nb:.4f} bends, sinuosity {rm.sinuosity(lam, CFG):.4f}, "
          f"amplitude {rm.amplitude(lam, CFG)/W:.2f} W")
interior = L - 2 * CFG["buffer_len"]
for lam in LAMS:
    nb = interior / lam
    check(f"lam={lam:.1f}: whole bends in the INTERIOR", abs(nb - round(nb)) < 1e-9,
          f"interior {interior:.0f} m = {nb:.4f} bends")
ks = [rm.wavenumber(l) for l in LAMS]
r = max(ks[1] / ks[0], ks[0] / ks[1])
check("wavenumbers differ by at least 1.4x", r >= 1.4, f"k ratio = {r:.2f}")

print("\n=== 3. no folding: the inner bank must not cross the centreline ===")
for lam in LAMS:
    f = rm.fold_margin(lam, CFG)
    check(f"lam={lam:.0f}: A k^2 b < 1", f < 1.0,
          f"A k^2 b = {f:.3f}   (R_min/b = {1/CFG['C0']/CFG['b']:.2f})")

print("\n=== 4. sinuosity: exact integral, and how wrong the small-Ak expansion is ===")
for lam in LAMS:
    A, k = rm.amplitude(lam, CFG), rm.wavenumber(lam)
    from scipy.special import j0
    s_meas = rm.sinuosity(lam, CFG)
    s_theory = 1.0 / j0(rm.theta_max(lam, CFG))
    check(f"lam={lam:.0f}: sinuosity within 20% of the untapered 1/J0(theta_m)",
          abs(s_meas / s_theory - 1) < 0.20,
          f"measured {s_meas:.4f}, 1/J0(theta_m) {s_theory:.4f} "
          f"(the taper straightens the buffers, so measured is lower)")

print("\n=== 5. cross-section endpoints ===")
for tgt, n in (("H_c", 0.0), ("H_b", CFG["b"])):
    got = float(rm.section_depth(np.array([n]), CFG)[0])
    check(f"h(n={n:.0f}) = {tgt}", abs(got - CFG[tgt]) < 1e-9, f"{got:.6f} m")
n_plain = CFG["b"] + CFG["m_bank"] * (CFG["H_b"] - CFG["h_plain"]) + 1.0
got = float(rm.section_depth(np.array([n_plain]), CFG)[0])
check("shelf clamps at h_plain (ALWAYS WET, no wet/dry line on the bank)",
      abs(got - CFG["h_plain"]) < 1e-9, f"h = {got:.3f} m > 0")
# the shelf must sit below the erosion threshold or it washes away and the design collapses
S = rm.slope(CFG)
U_p = np.sqrt(rm.G_ACCEL * CFG["h_plain"] * S / CFG["Cd"])
tau_p = CFG["Cd"] * U_p ** 2
tau_cr = (CFG["Sdensity"] - 1) * rm.G_ACCEL * CFG["D50"] * CFG["Shields_cr_bedload"]
check("shelf is below the bedload threshold", tau_p < tau_cr,
      f"tau/tau_cr = {tau_p/tau_cr:.3f} at U_shelf = {U_p:.3f} m/s")
# ... but the bank FACE must cross it, or there is no bank erosion at all
U_f = np.sqrt(rm.G_ACCEL * CFG["H_b"] * S / CFG["Cd"])
check("bank face DOES cross the threshold (bank erosion is preserved)",
      CFG["Cd"] * U_f ** 2 > tau_cr,
      f"tau/tau_cr = {CFG['Cd']*U_f**2/tau_cr:.3f} at the bank edge (h = {CFG['H_b']} m)")

print("\n=== 6. constant PV gradient: numeric from section_depth vs analytic pv_gradient ===")
# independent route: build U(n) from the normal-flow balance on the ACTUAL section array,
# form q = zeta/h with zeta = -dU/dn, and differentiate.
n = np.linspace(-CFG["b"] * 0.98, CFG["b"] * 0.98, 20001)
h = rm.section_depth(n, CFG)
U = np.sqrt(rm.G_ACCEL * h * rm.slope(CFG) / CFG["Cd"])
q = (-np.gradient(U, n)) / h
# Test the DEFINING property (q linear in n <=> dq/dn constant) by a least-squares fit
# rather than by differentiating twice: np.gradient is one-sided at the two endpoints, so
# max-min of gradient(gradient(.)) measures the FD stencil, not the physics.  (Measured:
# 6.9% spread over the full range, 7.4e-8 after trimming 2 points from each end.)
slope_fit, _ = np.polyfit(n, q, 1)
resid = np.abs(q - np.polyval(np.polyfit(n, q, 1), n)).max() / np.abs(q).max()
check("q = zeta/h is linear in n (=> dq/dn constant)", resid < 1e-4,
      f"max residual / max|q| = {resid:.2e}")
check("fitted dq/dn = analytic pv_gradient()",
      abs(slope_fit - rm.pv_gradient(CFG)) / abs(rm.pv_gradient(CFG)) < 1e-6,
      f"fit={slope_fit:.9e}, analytic={rm.pv_gradient(CFG):.9e} 1/(m^2 s)")

print("\n=== 7. channel_coords: |n| is the true minimum distance to the centreline ===")
lam = LAMS[0]
A, k = rm.amplitude(lam, CFG), rm.wavenumber(lam)
rng = np.random.default_rng(0)
Xp = rng.uniform(lam * 0.3, lam * 0.7, 300)
Yp = rng.uniform(-A - CFG["b"], A + CFG["b"], 300)
n_kd, s_kd = rm.channel_coords(Xp[:, None], Yp[:, None], lam, CFG)[:2]
# Reference: the built curve, REFINED 20x.  n is the PERPENDICULAR distance, whereas a
# brute-force min over discrete samples returns sqrt(d^2 + delta^2) with delta the along-curve
# offset of the nearest sample -- so a coarse reference is biased high by O(delta^2/2d).
xc0, yc0 = rm.centreline(lam, CFG)[:2]
tt = np.linspace(0, len(xc0) - 1, (len(xc0) - 1) * 20 + 1)
xf = np.interp(tt, np.arange(len(xc0)), xc0)
yf = np.interp(tt, np.arange(len(yc0 := yc0)), yc0)
d_brute = np.array([np.min(np.hypot(Xp[i] - xf, Yp[i] - yf)) for i in range(len(Xp))])
rel = np.abs(np.abs(n_kd.ravel()) - d_brute) / np.maximum(d_brute, 1e-6)
check("|n| matches brute-force nearest distance", rel.max() < 5e-3,
      f"max relative error = {rel.max()*100:.3f}% over {len(Xp)} random points")
# sample the centreline INSIDE the reach: channel_coords only samples the curve over
# [-lam/2, L+lam/2], so points beyond that project onto the sampled end and n is meaningless.
inside = (xf >= 0) & (xf <= rm.reach_length(CFG))
xi, yi = xf[inside][::2000], yf[inside][::2000]
on_line = rm.channel_coords(xi[:, None], yi[:, None], lam, CFG)[0]
check("n = 0 on the centreline", np.abs(on_line).max() < 0.02,
      f"max |n| = {np.abs(on_line).max():.2e} m over {len(xi)} on-curve points")
check("n changes sign across the centreline",
      n_kd.ravel().min() < 0 < n_kd.ravel().max())

print("\n=== 7b. sign convention: n*sign(kappa) > 0 must be the INNER bank ===")
# Geometric ground truth: the centre of curvature sits at P + (1/kappa) * N_left.  The
# inner bank is the one NEARER to it, at distance R - b; the outer is at R + b.
# Getting this backwards silently inverts every inner/outer conclusion, so check it
# against distances, not against the algebra that produced it.
for lam in LAMS:
    A, k = rm.amplitude(lam, CFG), rm.wavenumber(lam)
    # probe at an apex INSIDE the reach: the taper makes x=0 a straight section, where the
    # curvature is ~0 and inner/outer is undefined.  (This test silently broke when the
    # straight lead-in was added -- the third time in this project that a code change
    # outran its test.)
    xc_, yc_, _, _, _, kp_ = rm.centreline(lam, CFG)
    inr = (xc_ > CFG["buffer_len"]) & (xc_ < rm.reach_length(CFG) - CFG["buffer_len"])
    ipos = np.where(inr)[0][np.argmax(kp_[inr])]
    ineg = np.where(inr)[0][np.argmin(kp_[inr])]
    for ii, what in ((ipos, "kappa>0 apex"), (ineg, "kappa<0 apex")):
        P = np.array([xc_[ii], yc_[ii]])
        n_p, _, tx, ty, kap = rm.channel_coords(np.array([[P[0]]]), np.array([[P[1]]]), lam, CFG)
        kap = float(kap)
        N = np.array([-float(ty), float(tx)])            # left normal
        Cc = P + N / kap                                 # centre of curvature
        probes = {sgn: P + sgn * CFG["b"] * N for sgn in (+1, -1)}
        d = {sgn: np.hypot(*(probes[sgn] - Cc)) for sgn in (+1, -1)}
        inner_sgn = min(d, key=d.get)                    # nearer to the centre = inner
        check(f"lam={lam:.0f} {what}: inner bank is n*sign(kappa) > 0",
              inner_sgn * np.sign(kap) > 0,
              f"kappa={kap:+.4e}, |P-C| at n=+b: {d[+1]:.1f} m, at n=-b: {d[-1]:.1f} m "
              f"-> inner is n{'>' if inner_sgn > 0 else '<'}0")

print("\n=== 8. built arrays ===")
for lam in LAMS:
    Depth, Zs, _ini, x, y, m = rm.build_case(lam, CFG)
    check(f"lam={lam:.0f}: shape = (nx, ny)", Depth.shape == (m["nx"], m["ny"]),
          f"{Depth.shape}")
    check(f"lam={lam:.0f}: >=40 cells across the channel width",
          2 * CFG["b"] / CFG["dx"] >= 40, f"{2*CFG['b']/CFG['dx']:.0f} cells")
    nb = CFG["m_bank"] * (CFG["H_b"] + CFG["freeboard"]) / CFG["dx"]
    check(f"lam={lam:.0f}: >=5 cells across the bank face", nb >= 5, f"{nb:.1f} cells")
    # the deepest point of every cross-section must be within the channel
    j_deep = np.argmax(Depth, axis=1)
    n_deep = rm.channel_coords(x[:, None], y[j_deep][:, None], lam, CFG)[0]
    check(f"lam={lam:.0f}: thalweg lies inside |n|<=b",
          np.abs(np.diag(n_deep)).max() <= CFG["b"] + CFG["dx"],
          f"max |n_thalweg| = {np.abs(np.diag(n_deep)).max():.1f} m (b={CFG['b']:.0f})")
    check(f"lam={lam:.0f}: bed drops downstream by S*L_channel",
          Depth.mean(axis=1)[-1] > Depth.mean(axis=1)[0],
          f"drop = {(Depth.mean(axis=1)[-1]-Depth.mean(axis=1)[0]):.3f} m over "
          f"{m['L']*m['sinuosity']:.0f} m of channel")
    check(f"lam={lam:.0f}: Zs = 0 in the buffer, large inside",
          Zs[0, 0] == 0.0 and Zs[m["nx"] // 2, m["ny"] // 2] > 1e5)

print("\n=== 9. normal-flow consistency of the design ===")
Sd, S = rm.slope_design(CFG), rm.slope(CFG)
check("g H_c S_design = Cd U^2 (straight-channel normal flow)",
      abs(rm.G_ACCEL * CFG["H_c"] * Sd - CFG["Cd"] * CFG["U"] ** 2) < 1e-12,
      f"S_design = {Sd:.4e}")
check("S_bed = head_factor * S_design", abs(S - CFG["head_factor"] * Sd) < 1e-15,
      f"S_bed = {S:.4e} (head_factor = {CFG['head_factor']:.3f}), head drop over the reach = "
      f"{S*rm.reach_length(CFG)*rm.sinuosity(LAMS[0], CFG):.3f} m")
ws = rm.settling_velocity(CFG)
ustar = np.sqrt(CFG["Cd"]) * CFG["U"]
check("w_s in the medium-sand range", 0.05 < ws < 0.10,
      f"w_s = {ws:.4f} m/s, u*/w_s = {ustar/ws:.2f} "
      f"({'bedload' if ustar/ws < 1 else 'suspension'}-dominated)")
check("MinDepthPickup clears the log-law singularity",
      CFG["MinDepthPickup"] > 20 * np.e * 2.5 * CFG["D50"] / 30,
      f"MinDepthPickup = {CFG['MinDepthPickup']} m vs singularity at "
      f"{np.e*2.5*CFG['D50']/30:.2e} m")
Tc = CFG["H_c"] / (2.0 * ws)
# Morph_interval averages the suspended-load Exner forcing (P-D) before it moves the bed, so it
# should span the suspension relaxation time T_c.  HOW MANY T_c is "enough" is regime-dependent:
# when the load is suspension-dominated the smoothing must be aggressive (>>T_c) or unequilibrated
# transients drive the bed; when it is BEDLOAD-dominated (u*/w_s<1, this case) the suspended part
# is a minor forcing and the instantaneous bedload deflection carries the point bar, so spanning
# ~1 T_c suffices -- and a SMALLER MI is actually safer (the P_ave/D_ave step-change at each
# refresh is smaller).  Stability is bounded separately by the morphodynamic-Courant gate G5 in
# 01_validate.py.  (The old blanket >5 T_c rule failed a bedload case the decisive run validated.)
bedload_dom = ustar / ws < 1.0
need = 1.0 if bedload_dom else 5.0
check(f"Morph_interval spans the suspension relaxation ({'bedload' if bedload_dom else 'suspension'}"
      f"-dominated -> need {need:.0f} T_c)", CFG["Morph_interval"] >= need * Tc,
      f"Morph_interval = {CFG['Morph_interval']:.0f} s = {CFG['Morph_interval']/Tc:.1f} T_c "
      f"(T_c = {Tc:.1f} s)")

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
