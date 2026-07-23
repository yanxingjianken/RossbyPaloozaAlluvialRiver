#!/usr/bin/env python3
"""Linear shallow-water stability of a straight channel with mean jet U(y) and bed H(y) --
the EXTENSION of Ikeda that carries BOTH a gravity branch and a vortical/Rossby branch.

Solves the cross-channel eigenvalue problem (swe_meander_stability.md eq 3.1) for sigma(k):

    sigma eta = -ikU eta - ik H u - (H v)_y
    sigma u   = (rU/H - ikg) eta + (-ikU - 2r) u - U' v
    sigma v   = -g eta_y + (-ikU - r) v                       r = Cd U / H,  banks: v=0

Eigenvalues split into (a) two fast GRAVITY branches |Im sigma| ~ sqrt(gH) k and (b) one slow
VORTICAL / shear-Rossby branch |Im sigma| ~ Uk, whose restoring is the background PV gradient
    q_y = -U''/H + U'H'/H'^2  =  beta_shear + beta_topo         (eq 2.2)
Ikeda keeps only the gravity response (F^2); this code keeps the PV branch too, and reports the
dimensionless gravity-vs-Rossby number R = beta_eff b^2 / (F^2 U)  (eq 6.1).
"""
import numpy as np

G = 9.81


def base_state(y, b, H0=3.0, Hbank=1.5, U0=0.85, jet="quadratic", flat_bed=False, uniform_jet=False):
    """Prescribe the mean jet U(y) and bed H(y).  quadratic jet U0(1-(y/b)^2) -> U''=const (the
    reviewer's -U_yy = beta_shear); parabolic bed H (the cross-channel depth -> beta_topo)."""
    Y = y / b
    U = np.full_like(y, U0) if uniform_jet else U0 * (1.0 - Y ** 2)          # quadratic jet
    H = np.full_like(y, H0) if flat_bed else H0 - (H0 - Hbank) * Y ** 2      # parabolic bed
    return U, H


def pv_gradient(y, U, H):
    """q_y = -U''/H + U'H'/H^2 = beta_shear + beta_topo (eq 2.2), split into its two sources."""
    U1 = np.gradient(U, y); U2 = np.gradient(U1, y); H1 = np.gradient(H, y)
    beta_shear = -U2 / H
    beta_topo = U1 * H1 / H ** 2
    return beta_shear, beta_topo, beta_shear + beta_topo


def _Dy(y):
    """2nd-order central first-derivative matrix on the (uniform) grid y (one-sided at ends)."""
    N = len(y); h = y[1] - y[0]; D = np.zeros((N, N))
    for i in range(1, N - 1):
        D[i, i - 1] = -1 / (2 * h); D[i, i + 1] = 1 / (2 * h)
    D[0, 0] = -1 / h; D[0, 1] = 1 / h; D[-1, -2] = -1 / h; D[-1, -1] = 1 / h
    return D


def dispersion(k, y, U, H, Cd=0.00154, g=G):
    """Eigenvalues sigma of eq (3.1) at wavenumber k.  State w=(eta,u,v) stacked; v=0 at banks."""
    N = len(y); D = _Dy(y); I = np.eye(N)
    U1 = np.gradient(U, y); H1 = np.gradient(H, y); r = Cd * U / H
    Z = np.zeros((N, N))
    # row blocks [eta; u; v]
    A_ee = np.diag(-1j * k * U);            A_eu = np.diag(-1j * k * H);  A_ev = -np.diag(H1) - np.diag(H) @ D
    A_ue = np.diag(r * U / H - 1j * k * g); A_uu = np.diag(-1j * k * U - 2 * r); A_uv = -np.diag(U1)
    A_ve = -g * D;                          A_vu = Z;                     A_vv = np.diag(-1j * k * U - r)
    A = np.block([[A_ee, A_eu, A_ev], [A_ue, A_uu, A_uv], [A_ve, A_vu, A_vv]])
    # rigid banks: v=0 at the two end rows -> replace those v-rows with sigma*v=0
    for j in (0, N - 1):
        row = 2 * N + j
        A[row, :] = 0.0; A[row, row] = 0.0     # sigma v_bank = 0
    w, V = np.linalg.eig(A)
    return w, V, N


def vortical_mode(k, y, U, H, Cd=0.00154, g=G):
    """Extract the SLOW vortical / shear-Rossby eigenvalue: advective phase speed (0<c<2U0),
    weakly damped, and SMOOTH cross-channel structure (few sign changes) -- this rejects the
    fast gravity branch (|Im s|~sqrt(gH)k_y, grid-scale) and the v=0 boundary artefacts (s~0)."""
    w, V, N = dispersion(k, y, U, H, Cd, g)
    U0 = U.max(); r = (Cd * U / H).mean()
    c = -w.imag / k                                       # phase speed
    cand = [(i, w[i]) for i in range(len(w))
            if 0.02 * U0 < c[i] < 2.5 * U0 and w[i].real < 3 * r and w[i].real > -50 * r]
    if not cand:
        return None
    # prefer the smoothest v-eigenvector (fewest interior sign changes = lowest cross-channel mode)
    def roughness(i):
        vv = np.real(V[2 * N:3 * N, i]); vv = vv[2:-2]
        return int(np.sum(np.abs(np.diff(np.sign(vv))) > 0))
    i = min(cand, key=lambda t: (roughness(t[0]), abs(c[t[0]] - U0)))[0]
    return w[i]


if __name__ == "__main__":
    b = 50.0
    y = np.linspace(-b, b, 121)
    U, H = base_state(y, b)
    U0 = U.max(); H0 = H.max()
    F2 = U0 ** 2 / (G * H0)
    bs, bt, bq = pv_gradient(y, U, H)
    # dimensionless gravity-vs-Rossby number R (eq 6.1), evaluated near the jet core
    beta_eff = np.abs(bq[len(y) // 4:3 * len(y) // 4]).mean()
    R = beta_eff * b ** 2 / (F2 * U0)
    print("=== base state (quadratic jet + parabolic bed) ===")
    print(f"  U0={U0:.2f} m/s  H0={H0:.1f} m  F^2={F2:.4f}")
    print(f"  beta_shear(core)={bs[len(y)//2]:+.2e}  beta_topo(core)={bt[len(y)//2]:+.2e}  1/m/s")
    print(f"  q_y changes sign (Rayleigh-Kuo)? {np.any(bq > 0) and np.any(bq < 0)}")
    print(f"\n=== gravity vs Rossby (eq 6.1) ===")
    print(f"  R = beta_eff b^2/(F^2 U) = {R:.1f}   ->  {'ROSSBY/vortical-dominated' if R > 1 else 'gravity-dominated'}")
    print(f"  (R >> 1 confirms river meandering is a shear/topographic-Rossby instability, not a gravity wave)")
    # a couple of wavenumbers: show the gravity vs vortical eigenvalue split
    print(f"\n=== SWE eigen-branches (Im sigma = frequency) at a few k ===")
    for lam_W in (10, 20, 40):
        k = 2 * np.pi / (lam_W * 2 * b)
        w, V, N = dispersion(k, y, U, H)
        fast = np.sort(np.abs(w.imag))[-2:]        # gravity
        slow = np.sort(np.abs(w.imag))[:3]         # vortical
        cg = np.sqrt(G * H0)
        print(f"  lambda={lam_W}W (k={k:.4f}): gravity |Im s|~{fast.mean():.3f} (sqrt(gH)k={cg*k:.3f}), "
              f"vortical |Im s|~{slow[slow>1e-9].min() if np.any(slow>1e-9) else 0:.4f} (Uk={U0*k:.4f})")
