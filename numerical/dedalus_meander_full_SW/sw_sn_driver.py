#!/usr/bin/env python3
# =============================================================================
#  sw_sn_driver.py  --  THE single core Dedalus file for dedalus_meander_full_SW
# =============================================================================
#  FULL shallow-water primitive equations (u_s, u_n, eta) in channel-fitted
#  CURVILINEAR (s,n) coordinates, to identify WHAT WAVE the river meander is.
#
#  Unlike the rigid-lid / streamfunction packages (dedalus_meander, _meander2,
#  rigid_lid_to_share) this keeps the FREE SURFACE eta (mass eqn has eta_t) so
#  BOTH gravity waves AND the vortical/Rossby wave live in the same system, and
#  the flow is DIVERGENT (no streamfunction).  The channel meanders with FINITE
#  amplitude (no small-meander assumption): the metric sigma = 1 + n*Cbar(s)
#  carries the curvature Cbar(s), and the base jet follows the channel.
#
#  Linear PERTURBATION dynamics on a prescribed finite-meander base; erodible
#  banks (dt(xi)=E*u_s'(+/-b)) feed back through the curvature perturbation C'.
#  IVP ONLY (no EVP / no validation oracle) -- outputs a dispersion relation and
#  one fully-Eulerian momentum-flux movie (see postprocessing/).
#
# -----------------------------------------------------------------------------
#  ENVIRONMENT & RUN (micromamba env `dedalus`, Dedalus v3.0.5):
#
#    micromamba run -n dedalus env OMP_NUM_THREADS=1 python sw_sn_driver.py
#
#  That is the ONLY run.  Everything is configured in the CONFIG dict at the top
#  of this file -- edit it and re-run; there are no modes and no CLI options.
#  Writes RAW HDF5 to outputs/ only; all figures/movies live in postprocessing/,
#  base-state checks in tests/, the k/Froude experiment in sweep_dispersion.py,
#  and the derivation of every equation below in derivations/.
# =============================================================================
"""dedalus_meander_full_SW core driver: (s,n) shallow-water meander IVP."""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_CMD = ("micromamba run -n dedalus env OMP_NUM_THREADS=1 "
           "python sw_sn_driver.py")


# =========================================================================== #
#  CONFIG  --  the single-file knobs (edit here; CLI can override, see main)
# =========================================================================== #
CONFIG = dict(
    # --- base jet (parabolic; center max, quadratic decay to banks) -------- #
    #   Ubar_s(n) = U0 + Delta*(1 - (n/b)^2)  ->  center = U0+Delta, edge = U0
    #   d2/dn2 Ubar_s = -2 Delta/b^2 = const  ->  cross-channel vorticity
    #   gradient  d(zetabar)/dn = 2 Delta/b^2 = CONST = the channel-beta.
    U0=0.4,              # bank-edge speed
    Delta=0.6,           # parabolic excess (center speed = U0 + Delta)
    b=1.0,               # channel half-width (n in [-b, b])
    # --- free surface / gravity -------------------------------------------- #
    Froude=0.5,          # F = Uc/sqrt(g H0);  g_eff = 1/F^2 (gravity speed 1/F)
    H0=1.0,              # mean still-water depth (nondim)
    # --- finite-meander geometry  Cbar(s) = A_bank*kmeander^2 cos(kmeander s) #
    A_bank=0.20,         # centerline lateral amplitude (init bank shape ampl.)
    kmeander=0.30,       # meander (bend) wavenumber along s
    Cbar_amp=None,       # DIRECT base-curvature amplitude override (None -> A_bank*kmeander^2);
                         #   set e.g. 0.3 for a genuinely tight/finite meander (needs Cbar_amp*b<1)
    # --- bed  H(s,n)  (adjustable expression; see bed_depth() below) -------- #
    #   H(s,n) = H0 * [1 + cross_amp*(1 - (n/b)^2)] * [1 + along_amp*cos(k_bed s)]
    #            \______ PARABOLIC across the channel ______/  \__ COSINE along it __/
    #   i.e. a parabolic thalweg in n (deepest mid-channel, like the jet profile)
    #   times a sinusoidal bedform train in s (alternate bars).  Examples:
    #     cross_amp=0.3, along_amp=0    -> H(n) only : parabolic thalweg, uniform along s
    #     cross_amp=0,   along_amp=0.2  -> H(s) only : flat across, cosine bars along s
    #     cross_amp=0.3, along_amp=0.2  -> H(s,n)    : both
    #     cross_amp=0,   along_amp=0    -> FLAT bed H = H0   <-- the default below
    cross_amp=0.0,       # parabolic cross-channel thalweg amplitude (0 = flat in n)
    along_amp=0.0,       # cosine along-channel bedform amplitude   (0 = uniform in s)
    along_kbed=0.30,     # along-channel bedform wavenumber k_bed
    # --- friction (C_f drag on BOTH s and n momentum) ---------------------- #
    Cf=0.05,             # bottom drag coefficient  (linearized r = Cf*Ubar_s/hbar)
    # --- bank erodibility -------------------------------------------------- #
    ECOEF=0.50,          # bank-erosion rate coefficient  (E = ECOEF*(U0))
    # --- lateral eddy viscosity (wall closure + regularization) ------------ #
    nu=3e-3,             # nu * curvilinear-Laplacian on (u_s, u_n)
    # --- perturbation / numerics ------------------------------------------- #
    kstar=0.30,          # seeded bank-perturbation wavenumber (dispersion sweep point)
    Ls=None,             # s-domain length; None -> 2*pi / kmeander * n_bends
    n_bends=4,           # integer meander wavelengths in the s-domain (periodicity)
    Ns=64,               # streamwise Fourier
    Nn=192,              # cross-channel Chebyshev
    dt=0.01,
    t_end=None,          # None -> auto
    A0=1e-4,             # bank seed amplitude
)


# =========================================================================== #
#  Base-state profiles  (pure numpy; smoke-testable before any Dedalus build)
# =========================================================================== #
def center_speed(cfg):
    """Uc = U0 + Delta (jet center-line speed; the velocity scale)."""
    return cfg["U0"] + cfg["Delta"]


def g_eff(cfg):
    """Nondim gravity  g = 1/Froude^2  (gravity-wave speed sqrt(g H0) = 1/F)."""
    return 1.0 / cfg["Froude"] ** 2


def bank_E(cfg):
    """Bank-erosion rate  E = ECOEF * U0  (near-bank velocity closure)."""
    return cfg["ECOEF"] * cfg["U0"]


def ubar_s(n, cfg):
    """Parabolic base jet  Ubar_s(n) = U0 + Delta (1 - (n/b)^2).

    Center-line (n=0) max = U0+Delta, bank edge (n=+/-b) = U0.
    """
    n = np.asarray(n, dtype=float)
    return cfg["U0"] + cfg["Delta"] * (1.0 - (n / cfg["b"]) ** 2)


def ubar_s_n(n, cfg):
    """d Ubar_s / dn = -2 Delta n / b^2   (the base shear)."""
    n = np.asarray(n, dtype=float)
    return -2.0 * cfg["Delta"] * n / cfg["b"] ** 2


def ubar_s_nn(cfg):
    """d^2 Ubar_s / dn^2 = -2 Delta / b^2 = CONST  ==>  the channel-beta.

    The cross-channel vorticity gradient d(zetabar)/dn = -d2(Ubar_s)/dn2
    = 2 Delta/b^2 is CONSTANT -- the Rossby restoring mechanism.
    """
    return -2.0 * cfg["Delta"] / cfg["b"] ** 2


def cbar(s, cfg):
    """Base-channel curvature  Cbar(s) = A_bank*kmeander^2 cos(kmeander s).

    A finite-amplitude meander (curvature of a cosine centerline of lateral
    amplitude A_bank).  Cbar=0 everywhere -> a straight channel (sanity limit).
    """
    s = np.asarray(s, dtype=float)
    amp = (cfg["Cbar_amp"] if cfg.get("Cbar_amp") is not None
           else cfg["A_bank"] * cfg["kmeander"] ** 2)
    return amp * np.cos(cfg["kmeander"] * s)


def sigma_metric(s, n, cfg):
    """Curvilinear metric  sigma = 1 + n*Cbar(s)  (arc-length stretch at offset n)."""
    return 1.0 + n * cbar(s, cfg)


def bed_depth(s, n, cfg):
    """Still-water depth H(s,n) (EDIT this expression for a different bed).

    Default: H0 * [1 + cross_amp (1 - (n/b)^2)] * [1 + along_amp cos(kbed s)].
    cross_amp>0 -> deeper thalweg (cross-channel); along_amp>0 -> along-s bars.
    """
    Hn = cfg["H0"] * (1.0 + cfg["cross_amp"] * (1.0 - (n / cfg["b"]) ** 2))
    if cfg["along_amp"]:
        Hn = Hn * (1.0 + cfg["along_amp"] * np.cos(cfg["along_kbed"] * s))
    return Hn


def etabar(s, n, cfg):
    """Base superelevation eta_bar(s,n) from centrifugal balance.

    n-momentum steady balance (Ubar_n=0):  g d_n eta_bar = Cbar Ubar_s^2 / sigma.
    Integrate from the center-line (eta_bar(n=0)=0):
        eta_bar(s,n) = (Cbar(s)/g) * INT_0^n  Ubar_s(n')^2 / sigma(s,n')  dn'.
    Outer bank (n same sign as Cbar) is higher.  Cbar=0 -> eta_bar=0 (flat).
    """
    g = g_eff(cfg)
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    ng = np.linspace(-cfg["b"], cfg["b"], 4001)          # fine n'-quadrature grid
    i0 = int(np.argmin(np.abs(ng)))                      # index nearest n'=0
    usq = ubar_s(ng, cfg) ** 2
    out = np.zeros((s_arr.size, n_arr.size))
    for i, sv in enumerate(s_arr):
        Cb = float(cbar(sv, cfg))
        f = usq / (1.0 + ng * Cb)                        # integrand Ubar_s^2/sigma
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(ng))])
        cum = cum - cum[i0]                              # zero the integral at n'=0
        out[i, :] = (Cb / g) * np.interp(n_arr, ng, cum)
    return out[0, 0] if np.isscalar(s) and np.isscalar(n) else np.squeeze(out)


# =========================================================================== #
#  Dedalus IVP:  linear (s,n) shallow water on the finite-meander base
# =========================================================================== #
#  We multiply every equation THROUGH by sigma=1+nCbar to clear the rational
#  1/sigma (keeps NCCs polynomial-type / banded, per the Dedalus perf docs).
#  Prognostic:  u_s, u_n, eta (2-D) ; bank offsets xi_p, xi_m (1-D in s).
#  Curvature feedback: C' = -d_ss(0.5(xi_p+xi_m)); it enters the n-momentum as
#  the centrifugal forcing +C'*Ubar_s^2 (moved to LHS => fully implicit, linear).
#  Walls (fixed n=+/-b): u_n=0 (no-penetration, NO non-divergence assumed) +
#  free-slip d_n u_s=0, closed with 2 tau per viscous velocity component.
#  Bank erosion: dt(xi_+/-) = E*u_s(+/-b)  (Ikeda near-bank velocity).
# --------------------------------------------------------------------------- #
def _sdomain(cfg):
    """s-domain length (periodic).  Sized by the base meander wavenumber if there
    is a finite base curvature, else by the seeded perturbation wavenumber kstar
    (so a straight-base movie run holds an integer number of clean meander bends).
    """
    if cfg["Ls"] is not None:
        return cfg["Ls"]
    cb = (cfg["Cbar_amp"] if cfg.get("Cbar_amp") is not None
          else cfg["A_bank"] * cfg["kmeander"] ** 2)
    kdom = cfg["kmeander"] if cb != 0 else cfg["kstar"]
    return cfg["n_bends"] * 2.0 * np.pi / kdom


def build_ivp_SW(cfg, Ns=None, Nn=None):
    """Assemble the linear (s,n) shallow-water IVP. Returns dict(solver, fields)."""
    import dedalus.public as d3
    Ns = Ns or cfg["Ns"]
    Nn = Nn or cfg["Nn"]
    b = cfg["b"]
    g = g_eff(cfg)
    E = bank_E(cfg)
    Ls = _sdomain(cfg)

    coords = d3.CartesianCoordinates("s", "n")
    dist = d3.Distributor(coords, dtype=np.float64)
    sbasis = d3.RealFourier(coords["s"], size=Ns, bounds=(0.0, Ls))
    nbasis = d3.Chebyshev(coords["n"], size=Nn, bounds=(-b, b))
    s, n = dist.local_grids(sbasis, nbasis)           # s:(Ns,1)  n:(1,Nn)
    ds = lambda A: d3.Differentiate(A, coords["s"])
    dn = lambda A: d3.Differentiate(A, coords["n"])

    # --- prognostic fields --------------------------------------------------
    us = dist.Field(name="us", bases=(sbasis, nbasis))
    un = dist.Field(name="un", bases=(sbasis, nbasis))
    eta = dist.Field(name="eta", bases=(sbasis, nbasis))
    zc = dist.Field(name="zc", bases=(sbasis,))        # centerline lateral offset (the meander)
    t_us1 = dist.Field(name="t_us1", bases=(sbasis,))
    t_us2 = dist.Field(name="t_us2", bases=(sbasis,))
    t_un1 = dist.Field(name="t_un1", bases=(sbasis,))
    t_un2 = dist.Field(name="t_un2", bases=(sbasis,))

    # --- NCC coefficient fields (on the grid) -------------------------------
    def f2d(name, arr):
        f = dist.Field(name=name, bases=(sbasis, nbasis)); f["g"] = arr; return f

    def f1n(name, arr):
        f = dist.Field(name=name, bases=(nbasis,)); f["g"] = arr; return f

    Cb = cbar(s, cfg)                                  # (Ns,1)
    sig_a = 1.0 + n * Cb                                # (Ns,Nn)  sigma
    hb_a = bed_depth(0.0, n, cfg) + 0.0 * s            # (Ns,Nn) still-water depth H(n)
    Ub_a = ubar_s(n, cfg) + 0.0 * s                    # base jet (broadcast)
    Ubn_a = ubar_s_n(n, cfg) + 0.0 * s
    Ubsq_a = Ub_a ** 2
    rs_a = 2.0 * cfg["Cf"] * ubar_s(n, cfg) / bed_depth(0.0, n, cfg)   # s-drag (factor 2)
    rn_a = 1.0 * cfg["Cf"] * ubar_s(n, cfg) / bed_depth(0.0, n, cfg)   # n-drag

    sig = f2d("sig", sig_a)
    hb = f1n("hb", ubar_s(n, cfg) * 0 + bed_depth(0.0, n, cfg))        # H(n), 1-D in n
    Ub = f1n("Ub", ubar_s(n, cfg))
    Ubsq = f1n("Ubsq", ubar_s(n, cfg) ** 2)
    sighb = f2d("sighb", sig_a * (bed_depth(0.0, n, cfg) + 0.0 * s))   # sigma*H
    coefUn = f2d("coefUn", sig_a * Ubn_a + Cb * Ub_a)                  # coeff of un in s-mom
    twoCbUb = f2d("twoCbUb", 2.0 * Cb * Ub_a)                          # coeff of us in n-mom
    sig_rs = f2d("sig_rs", sig_a * rs_a)
    sig_rn = f2d("sig_rn", sig_a * rn_a)

    lift_basis = nbasis.derivative_basis(2)
    lift = lambda A, k: d3.Lift(A, lift_basis, k)
    nu = cfg["nu"]
    ns = dict(us=us, un=un, eta=eta, zc=zc,
              t_us1=t_us1, t_us2=t_us2, t_un1=t_un1, t_un2=t_un2,
              sig=sig, hb=hb, Ub=Ub, Ubsq=Ubsq, sighb=sighb, coefUn=coefUn,
              twoCbUb=twoCbUb, sig_rs=sig_rs, sig_rn=sig_rn, g=g, nu=nu, E=E, b=b,
              ds=ds, dn=dn, lap=d3.Laplacian, lift=lift, dt=d3.TimeDerivative)

    problem = d3.IVP([us, un, eta, zc, t_us1, t_us2, t_un1, t_un2],
                     namespace=ns)
    # continuity (x sigma): sig*dt(eta) + hb*ds(us) + Ub*ds(eta) + d_n(sigma H un)=0
    problem.add_equation("sig*dt(eta) + hb*ds(us) + Ub*ds(eta) + dn(sighb*un) = 0")
    # s-momentum (x sigma)
    problem.add_equation("sig*dt(us) + Ub*ds(us) + coefUn*un + g*ds(eta)"
                         " + sig_rs*us - sig*nu*lap(us)"
                         " + lift(t_us1,-1) + lift(t_us2,-2) = 0")
    # n-momentum (x sigma): centrifugal curvature feedback +Ubsq*d_ss(zeta_c)
    # n-momentum: centrifugal curvature feedback  C'=-d_ss(zc)  =>  +Ubsq*d_ss(zc)
    problem.add_equation("sig*dt(un) + Ub*ds(un) - twoCbUb*us + sig*g*dn(eta)"
                         " + sig_rn*un - sig*nu*lap(un)"
                         " + Ubsq*ds(ds(zc))"
                         " + lift(t_un1,-1) + lift(t_un2,-2) = 0")
    # walls (fixed n=+/-b): no-penetration + free-slip
    problem.add_equation(f"un(n={b}) = 0")
    problem.add_equation(f"un(n={-b}) = 0")
    problem.add_equation(f"dn(us)(n={b}) = 0")
    problem.add_equation(f"dn(us)(n={-b}) = 0")
    # meander (centerline) erosion: dt(zc) = E * ANTISYMMETRIC near-bank velocity
    #   0.5*[u_s(+b) - u_s(-b)]  = fast-outer minus slow-inner  (Ikeda bend growth)
    problem.add_equation(f"dt(zc) - E*0.5*(us(n={b}) - us(n={-b})) = 0")

    solver = problem.build_solver(d3.RK222)
    return dict(solver=solver, us=us, un=un, eta=eta, zc=zc,
                s=s.ravel(), n=n.ravel(), Ls=Ls, dist=dist, cfg=cfg,
                Hbed=bed_depth(0.0, n, cfg).ravel(), sigma=sig_a)


def seed_ivp_SW(built, k, amp):
    """Seed a pure bend  zc = amp*cos(k s); flow = 0.

    The curvature C'=-d_ss(zc)=amp k^2 cos(k s) forces the n-momentum; the flow
    then erodes the centerline -> the meander grows/propagates.  Zero init flow.
    """
    us, un, eta, zc = built["us"], built["un"], built["eta"], built["zc"]
    s = built["s"]
    Ls = built["Ls"]
    # snap to the nearest EXACT domain Fourier mode so the seed is periodic
    # (a non-commensurate cos(k s) has an edge jump -> spectral spread -> a
    #  localized packet instead of a clean single-wavelength meander).
    m = max(1, int(round(k * Ls / (2 * np.pi))))
    k_eff = m * 2 * np.pi / Ls
    built["k_eff"] = k_eff
    for f in (us, un, eta):
        f["g"][:] = 0.0
    tb = amp * np.cos(k_eff * s.ravel())
    zc.change_scales(1)
    zc["g"][:] = tb[:, None] if zc["g"].ndim == 2 else tb


def measure_sigma_c(ts, series, k, Ls):
    """Fit growth sigma and phase speed c of the demodulated centerline mode."""
    ts = np.asarray(ts)
    m = int(round(k * Ls / (2 * np.pi)))
    coeff = np.array([np.fft.rfft(z)[m] for z in series])
    amp = np.abs(coeff)
    j = len(ts) // 3                                   # drop the transient
    sig = np.polyfit(ts[j:], np.log(amp[j:] + 1e-30), 1)[0]
    cph = -np.polyfit(ts[j:], np.unwrap(np.angle(coeff))[j:], 1)[0] / k
    return sig, cph


def run_ivp_SW(cfg, tag=None):
    """Run one (s,n) SW IVP (seed centerline at kstar), write raw HDF5. -> path."""
    import h5py
    k = cfg["kstar"]
    built = build_ivp_SW(cfg)
    seed_ivp_SW(built, k, cfg["A0"])
    solver = built["solver"]
    us, un, eta, zc = built["us"], built["un"], built["eta"], built["zc"]

    t_end = cfg["t_end"] if cfg["t_end"] is not None else 60.0
    n_steps = int(round(t_end / cfg["dt"]))
    rec_every = max(1, n_steps // 80)
    solver.stop_iteration = n_steps + 1
    ts, uss, uns, etas, zcs = [], [], [], [], []

    def record():
        for f in (us, un, eta, zc):
            f.change_scales(1)
        ts.append(solver.sim_time)
        uss.append(np.array(us["g"]))
        uns.append(np.array(un["g"]))
        etas.append(np.array(eta["g"]))
        zcs.append(np.array(zc["g"]).ravel())

    record()
    for it in range(n_steps):
        solver.step(cfg["dt"])
        if (it + 1) % rec_every == 0:
            record()

    ts = np.array(ts)
    sig, cph = measure_sigma_c(ts, zcs, k, built["Ls"])
    if tag is None:
        cb = cfg["Cbar_amp"] if cfg.get("Cbar_amp") is not None else cfg["A_bank"] * cfg["kmeander"] ** 2
        tag = (f"k{k:.2f}_F{cfg['Froude']:.2f}_Cb{cb:.3f}"
               f"_Cf{cfg['Cf']:.3f}").replace(".", "p")
    path = os.path.join(OUT_DIR, f"run_{tag}.h5")
    with h5py.File(path, "w") as h:
        h.create_dataset("t", data=ts)
        h.create_dataset("us", data=np.array(uss))
        h.create_dataset("un", data=np.array(uns))
        h.create_dataset("eta", data=np.array(etas))
        h.create_dataset("zc", data=np.array(zcs))
        h.create_dataset("s", data=built["s"])
        h.create_dataset("n", data=built["n"])
        h.create_dataset("Hbed", data=built["Hbed"])
        h.create_dataset("sigma_metric", data=built["sigma"])
        for kk, vv in cfg.items():
            h.attrs[kk] = "None" if vv is None else vv
        h.attrs["Ls"] = built["Ls"]
        h.attrs["mode_index"] = int(round(k * built["Ls"] / (2 * np.pi)))
        h.attrs["sigma_meas"] = sig
        h.attrs["c_meas"] = cph
        h.attrs["g_eff"] = g_eff(cfg)
        h.attrs["RUN_CMD"] = RUN_CMD
    print(f"  wrote {os.path.relpath(path, HERE)}  "
          f"(k={k}, {len(ts)} frames, sigma={sig:+.4f}, c={cph:+.4f})")
    return path, built


if __name__ == "__main__":
    # ONE run of the case configured in CONFIG at the top of this file.
    # (Edit CONFIG and re-run; there are no modes and no command-line options.)
    run_ivp_SW(CONFIG)
