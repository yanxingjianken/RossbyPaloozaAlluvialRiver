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
    # --- how to perturb the (steady) base state ---------------------------- #
    #   The base state is an exact steady solution, so SOMETHING must perturb it;
    #   for a LINEAR system any perturbation is admissible.
    #   "single"    : seed one mode cos(k_eff s) -> clean sigma,c for that k only.
    #   "broadband" : seed ALL resolvable modes at once (white centreline noise).
    #                 With Cbar=0 the base is s-translation-invariant, so the
    #                 s-Fourier modes DECOUPLE EXACTLY -- each then evolves on its
    #                 own and ONE run yields the WHOLE dispersion relation by
    #                 demodulating every mode separately.  (With Cbar!=0 the modes
    #                 couple (Floquet) and per-mode fitting is NOT a dispersion
    #                 relation.)
    #   The perturbation is ALWAYS broadband, so a run is characterised by its
    #   PHYSICS (bed H, bank sinuosity, friction, base speed) and never by "which
    #   wavelength was poked" -- the run contains all of them at once.
    seed_type="broadband",
    kstar=0.30,          # only used if seed_type="single" (diagnostic single-mode runs)
    Ls=None,             # s-domain length; None -> 2*pi / kmeander * n_bends
    n_bends=4,           # integer meander wavelengths in the s-domain (periodicity)
    Ns=64,               # streamwise Fourier
    Nn=192,              # cross-channel Chebyshev
    dt=0.01,
    t_end=None,          # None -> auto
    # Hard cap on how far the FASTEST mode may grow before the run stops.  With a
    # broadband seed, a mode more than ~log(1/eps)~36 e-foldings below the leader is
    # buried in the leader's FFT round-off and reads back the LEADER's growth rate --
    # so this cap, not t_end, is what keeps sigma(k) meaningful.  See run_ivp_SW.
    max_efold=25.0,
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
    # Always sized by the meander wavenumber: with a broadband perturbation the
    # domain is not tied to any single seeded wavelength, and a fixed length keeps
    # the mode grid (dk = 2*pi/Ls) identical across runs so they are comparable.
    return cfg["n_bends"] * 2.0 * np.pi / cfg["kmeander"]


def build_ivp_SW(cfg, Ns=None, Nn=None):
    """Assemble the linear (s,n) shallow-water IVP. Returns dict(solver, fields)."""
    import dedalus.public as d3
    Ns = Ns or cfg["Ns"]
    Nn = Nn or cfg["Nn"]
    b = cfg["b"]
    g = g_eff(cfg)
    E = bank_E(cfg)
    Ls = _sdomain(cfg)
    # The solver evaluates the bed ONLY at s=0, i.e. it uses H(n).  An along-channel
    # bed would make the base state s-dependent (discharge conservation d_s(hbar*Ubar)=0
    # then forces Ubar=Ubar(s,n)), which is NOT implemented.  Fail loudly rather than
    # silently ignoring the knob.
    if cfg["along_amp"]:
        raise NotImplementedError(
            "along_amp>0 (along-channel bed) is not wired into the solver: the bed is "
            "evaluated at s=0 only, so H=H(n). Supporting it requires an s-dependent "
            "base flow from discharge conservation. Set along_amp=0.")

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
    # free-surface drag feedback: linearising -C_f|u|u_s/h in the DEPTH as well as
    # the velocity gives  -2C_f Ubar/hbar * u_s'  +  C_f Ubar^2/hbar^2 * eta'.
    # The second term is Ikeda-Parker-Sawai's superelevation-drag term (their eq. 7,
    # term 2) -- a deeper outer column feels less drag, so the outer flow speeds up.
    # It is the ONLY route by which the free surface drives bend flow, i.e. the only
    # place F^2 enters classical bend theory.  (Omitting it was a real error.)
    reta_a = cfg["Cf"] * ubar_s(n, cfg) ** 2 / bed_depth(0.0, n, cfg) ** 2

    sig = f2d("sig", sig_a)
    hb = f1n("hb", ubar_s(n, cfg) * 0 + bed_depth(0.0, n, cfg))        # H(n), 1-D in n
    Ub = f1n("Ub", ubar_s(n, cfg))
    Ubsq = f1n("Ubsq", ubar_s(n, cfg) ** 2)
    sighb = f2d("sighb", sig_a * (bed_depth(0.0, n, cfg) + 0.0 * s))   # sigma*H
    coefUn = f2d("coefUn", sig_a * Ubn_a + Cb * Ub_a)                  # coeff of un in s-mom
    twoCbUb = f2d("twoCbUb", 2.0 * Cb * Ub_a)                          # coeff of us in n-mom
    sig_rs = f2d("sig_rs", sig_a * rs_a)
    sig_rn = f2d("sig_rn", sig_a * rn_a)
    sig_reta = f2d("sig_reta", sig_a * reta_a)         # superelevation-drag (IPS81 term 2)

    lift_basis = nbasis.derivative_basis(2)
    lift = lambda A, k: d3.Lift(A, lift_basis, k)
    nu = cfg["nu"]
    ns = dict(us=us, un=un, eta=eta, zc=zc,
              t_us1=t_us1, t_us2=t_us2, t_un1=t_un1, t_un2=t_un2,
              sig=sig, hb=hb, Ub=Ub, Ubsq=Ubsq, sighb=sighb, coefUn=coefUn,
              twoCbUb=twoCbUb, sig_rs=sig_rs, sig_rn=sig_rn, sig_reta=sig_reta,
              g=g, nu=nu, E=E, b=b,
              ds=ds, dn=dn, lap=d3.Laplacian, lift=lift, dt=d3.TimeDerivative)

    problem = d3.IVP([us, un, eta, zc, t_us1, t_us2, t_un1, t_un2],
                     namespace=ns)
    # continuity (x sigma): sig*dt(eta) + hb*ds(us) + Ub*ds(eta) + d_n(sigma H un)=0
    problem.add_equation("sig*dt(eta) + hb*ds(us) + Ub*ds(eta) + dn(sighb*un) = 0")
    # s-momentum (x sigma)
    problem.add_equation("sig*dt(us) + Ub*ds(us) + coefUn*un + g*ds(eta)"
                         " + sig_rs*us - sig_reta*eta - sig*nu*lap(us)"
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
                Hbed=bed_depth(0.0, n, cfg).ravel(), sigma=sig_a,
                # coords + the sigma NCC are exported so classify_mode() can take its
                # derivatives SPECTRALLY (see the note there -- the divergence is a
                # near-cancellation and finite differences destroy it)
                coords=coords, sigf=sig)


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
    sv = s.ravel()
    if built["cfg"].get("seed_type", "single") == "broadband":
        # excite EVERY resolvable mode with equal amplitude and fixed pseudo-random
        # phases (deterministic => reproducible).  Because the modes decouple at
        # Cbar=0, one such run contains the whole dispersion relation.
        Ns_loc = sv.size
        rng = np.random.default_rng(1234)
        tb = np.zeros_like(sv)
        for m in range(1, Ns_loc // 2):                # up to just below Nyquist
            km = m * 2 * np.pi / Ls
            tb += np.cos(km * sv + rng.uniform(0, 2 * np.pi))
        tb *= amp / np.max(np.abs(tb))                 # normalise peak to amp
        built["k_eff"] = None                          # not a single-mode run
    else:
        tb = amp * np.cos(k_eff * sv)
    zc.change_scales(1)
    zc["g"][:] = tb[:, None] if zc["g"].ndim == 2 else tb


def measure_sigma_c(ts, series, k, Ls):
    """Fit growth sigma and phase speed c of the demodulated centerline mode.

    Returns (sigma, c, k_eff, n_efold).  NOTE: the phase rate must be divided by
    the wavenumber ACTUALLY seeded, k_eff = 2*pi*m/Ls (the seed is snapped to an
    exact domain mode), not by the requested k -- on a coarse k-grid those differ
    by up to ~10% and that error alone puts a spurious sawtooth into c(k).
    `n_efold` lets the caller reject modes that never showed clean exponential
    growth (the fit is meaningless below ~3 e-foldings).
    """
    ts = np.asarray(ts)
    m = max(1, int(round(k * Ls / (2 * np.pi))))
    k_eff = m * 2 * np.pi / Ls
    coeff = np.array([np.fft.rfft(z)[m] for z in series])
    amp = np.abs(coeff)
    j = len(ts) // 3                                   # drop the transient
    sig = np.polyfit(ts[j:], np.log(amp[j:] + 1e-30), 1)[0]
    cph = -np.polyfit(ts[j:], np.unwrap(np.angle(coeff))[j:], 1)[0] / k_eff
    n_efold = float(np.log(max(amp[-1], 1e-300) / max(amp[j], 1e-300)))
    return sig, cph, k_eff, n_efold


def per_mode_dispersion(ts, zcs, Ls, n_min_efold=3.0):
    """Demodulate EVERY centreline Fourier mode -> the whole dispersion relation.

    Legitimate because with a straight base channel (bank sinuosity 0) the base
    state is s-translation-invariant, so the s-Fourier modes decouple exactly and
    each evolves independently.  One broadband run therefore contains sigma(k) and
    c(k) for all k.  (With a finite base sinuosity the modes couple -- Floquet --
    and these per-mode numbers are NOT a dispersion relation; `converged` and the
    caller must say so.)

    Returns arrays (k, sigma, c, n_efold, converged); modes are flagged NOT converged
    (and must not be trusted) if either gate fails:

      (a) fewer than n_min_efold e-foldings   -> the fit is to transient adjustment,
          and its very SIGN depends on the fit window;
      (b) ROUND-OFF CONTAMINATION.  The modes span an enormous dynamic range: once the
          fastest mode reaches amplitude A_max(t), every other mode in the FFT is
          polluted at the level eps*A_max(t), so a mode whose amplitude has dropped
          below that floor is measuring the LEADER's round-off, not itself.  Gate (a)
          cannot catch this -- round-off noise inherits the leader's growth rate, so a
          contaminated mode looks perfectly converged.  It shows up instead as a
          sawtooth in sigma(k): a run whose leader grows ~59 e-foldings reaches a
          cross-mode dynamic range of 1e17, past 1/eps, and a third of its modes go
          bad.  Because the floor RISES with the leader, contamination is a time
          WINDOW, not a verdict: each mode is valid until the floor overtakes it.  We
          therefore fit each mode only over its own valid window rather than discarding
          it, and only call it not-converged if too little valid data survives.
    """
    ts = np.asarray(ts)
    Z = np.array(zcs)
    C = np.fft.rfft(Z, axis=1)
    j = len(ts) // 3                                    # drop the spin-up
    # time-dependent round-off floor set by the LEADING mode at each instant
    floor_t = (1e3 * np.finfo(float).eps) * np.max(np.abs(C[:, 1:]), axis=1)
    ks, sg, cs, ne, ok = [], [], [], [], []
    for m in range(1, C.shape[1] - 1):
        a = C[:, m]
        amp = np.abs(a)
        if not np.all(np.isfinite(amp)) or amp[j] <= 0:
            continue
        k = m * 2 * np.pi / Ls
        # this mode's valid window: from the end of spin-up until the leader's
        # round-off floor overtakes it (clean modes keep the full window)
        clean = amp > floor_t
        i1 = len(ts)
        if not clean[-1]:
            bad = np.nonzero(~clean[j:])[0]
            i1 = j + int(bad[0]) if len(bad) else len(ts)
        w = slice(j, max(i1, j + 2))
        nfit = len(ts[w])
        ks.append(k)
        sg.append(np.polyfit(ts[w], np.log(amp[w] + 1e-300), 1)[0])
        cs.append(-np.polyfit(ts[w], np.unwrap(np.angle(a))[w], 1)[0] / k)
        n = float(np.log(max(amp[w][-1], 1e-300) / max(amp[w][0], 1e-300)))
        ne.append(n)
        ok.append(float(n >= n_min_efold and nfit >= 8))
    return (np.array(ks), np.array(sg), np.array(cs), np.array(ne), np.array(ok))


def bank_sinuosity(cfg):
    """Base-channel sinuosity = curvature amplitude of the initial meandering bank.

    Cbar_amp (or A_bank*kmeander^2).  0 = straight channel.  Cbar_amp*b < 1 is
    required for the mapping not to fold.
    """
    return (cfg["Cbar_amp"] if cfg.get("Cbar_amp") is not None
            else cfg["A_bank"] * cfg["kmeander"] ** 2)


def run_tag(cfg):
    """Name a run by its PHYSICS, not by which wavelength was perturbed.

    The perturbation is broadband (all wavelengths at once), so the meaningful
    identifiers are the bed, the initial bank sinuosity, the bottom friction and
    the base speed:

        H<bed>_bank<sinuosity>_Cf<friction>_U<base speed>dU<jet excess>

    bed: 'flat' if the bed is uniform, else 'cross<amp>' (parabolic thalweg).

    The jet excess Delta has to appear too: it sets the cross-channel shear, hence
    the vorticity gradient d_n zetabar = 2*Delta/b^2, so two runs differing only in
    Delta are physically different and must not collide (plug flow Delta=0 vs a
    reversed-shear wake Delta<0 are the sharpest test in the study).
    """
    bed = "flat" if cfg["cross_amp"] == 0 else f"cross{cfg['cross_amp']:.2f}"
    return (f"H{bed}_bank{bank_sinuosity(cfg):.3f}_Cf{cfg['Cf']:.3f}"
            f"_U{cfg['U0']:.2f}dU{cfg['Delta']:+.2f}").replace(".", "p")


def classify_mode(built):
    """Diagnose WHICH kind of mode the current perturbation is.

    This is the classification the model is for.  Froude-insensitivity alone is a
    NECESSARY but not sufficient test: every slow (balanced) mode passes it.  What
    distinguishes a free vortical/Rossby wave from a forced boundary mode is where
    the energy comes from and whether the perturbation carries PV:

      div_ratio  = ||delta'|| / ||zeta'||   -> ~0 means balanced (not a gravity wave)
      pv_ratio   = ||q'||     / ||zeta'||   -> ~1 means vortex stretching is minor
      eta_over_u = ||eta'||   / ||u'||      -> should scale as F^2 for a balanced mode
      T_shear    = -INT u_s'u_n' d_n(Ubar)  -> the ONLY channel by which the mean-flow
                   PV gradient (the "channel-beta") can power a growing vortical wave.
                   If T_shear <= 0 the mean flow is a SINK and the mode is NOT a
                   free vortical wave, whatever its Froude behaviour.
      T_bend     = -INT u_n' Ubar^2 d_ss(zeta_c) -> work done on the fluid by the
                   moving bank (the erosion closure).

    NOTE ON THE DERIVATIVES.  div and zeta MUST be taken spectrally, not with
    np.gradient.  The divergence of a nearly-balanced flow is a small residual left
    over from cancelling two O(k|u'|) terms, d_s(u_s) against d_n(sigma u_n).
    Finite-differencing a Chebyshev grid commits an O(1) relative error on each of
    those terms, so the "residual" you measure is the differencing error, not the
    divergence.  Doing exactly that reported div/zeta ~ 0.7 for a mode whose actual
    ratio (from continuity, delta ~ sigma_growth*eta/H) is ~ 0.006 -- a hundredfold
    overstatement that would have been written into the paper as "not balanced".
    Dedalus differentiates the represented field exactly, so we use its operators.
    """
    cfg = built["cfg"]
    s, n = built["s"], built["n"]
    for f in (built["us"], built["un"], built["eta"], built["zc"]):
        f.change_scales(1)
    us = np.array(built["us"]["g"]); un = np.array(built["un"]["g"])
    eta = np.array(built["eta"]["g"]); zc = np.array(built["zc"]["g"]).ravel()
    sig = built["sigma"] if np.ndim(built["sigma"]) == 2 else np.ones_like(us)
    hb = bed_depth(0.0, n, cfg)[None, :] * np.ones_like(us)
    Ub = ubar_s(n, cfg)[None, :] * np.ones_like(us)
    Ub_n = ubar_s_n(n, cfg)[None, :] * np.ones_like(us)

    import dedalus.public as d3          # local, as in build_ivp_SW (slow import)

    coords, sigf = built["coords"], built["sigf"]
    Ds = lambda A: d3.Differentiate(A, coords["s"])
    Dn = lambda A: d3.Differentiate(A, coords["n"])

    def gval(expr):
        """Evaluate a Dedalus expression onto the grid (spectrally exact)."""
        f = expr.evaluate()
        f.change_scales(1)
        return np.array(f["g"])

    usf, unf = built["us"], built["un"]
    zeta = gval((Ds(unf) - Dn(sigf * usf)) / sigf)       # relative vorticity
    div = gval((Ds(usf) + Dn(sigf * unf)) / sigf)        # divergence (small residual!)
    zbar = -np.gradient(sig * Ub, n, axis=1) / sig       # base vorticity (analytic, smooth)
    q = zeta / hb - zbar * eta / hb ** 2                 # perturbation PV

    L2 = lambda A: float(np.sqrt(np.sum(A ** 2 * sig)))
    unorm = float(np.sqrt(np.sum((us ** 2 + un ** 2) * sig)))
    zc_ss = np.gradient(np.gradient(zc, s), s)[:, None] * np.ones_like(us)
    T_bend = float(-np.sum(un * Ub ** 2 * zc_ss * sig))
    T_shear = float(-np.sum(us * un * Ub_n * sig))

    d = dict(div_ratio=L2(div) / max(L2(zeta), 1e-300),
             pv_ratio=L2(q) / max(L2(zeta), 1e-300),
             eta_over_u=L2(eta) / max(unorm, 1e-300),
             T_bend=T_bend, T_shear=T_shear,
             shear_share=T_shear / max(abs(T_bend), 1e-300))
    d["summary"] = (f"div/zeta={d['div_ratio']:.3f} (0=balanced)  "
                    f"|q|/|zeta|={d['pv_ratio']:.3f}  "
                    f"|eta|/|u|={d['eta_over_u']:.3e}  "
                    f"T_shear/|T_bend|={d['shear_share']:+.3f} "
                    f"({'mean flow is a SINK' if T_shear <= 0 else 'mean flow feeds the mode'})")
    return d


def run_ivp_SW(cfg, tag=None):
    """Run one (s,n) SW IVP (broadband seed), write raw HDF5. -> path.

    The run stops at t_end OR when the leading centreline mode has grown
    cfg['max_efold'] e-foldings, whichever comes FIRST.  That second limit is not a
    convenience, it is what makes the broadband trick work at all:

    with all modes seeded together, two modes separated by dsigma reach an amplitude
    ratio exp(dsigma*T).  Once that ratio passes 1/eps ~ 4.5e15 (about 36 e-foldings)
    the slower mode is buried in the FFT round-off of the faster one and is
    unrecoverable FROM THAT RUN by any post-processing -- it simply reads back the
    leader's growth rate.  A fixed t_end cannot serve every configuration: t_end=120
    gave the reference run a harmless 13 e-foldings but the plug-flow run 58, at which
    point a third of its spectrum was noise masquerading as a converged mode.
    Tying the stop to the leader's growth makes the usable dynamic range a property of
    the diagnostic rather than of how fast the configuration happens to grow.
    """
    import h5py
    k = cfg["kstar"]
    built = build_ivp_SW(cfg)
    seed_ivp_SW(built, k, cfg["A0"])
    solver = built["solver"]
    us, un, eta, zc = built["us"], built["un"], built["eta"], built["zc"]

    t_end = cfg["t_end"] if cfg["t_end"] is not None else 60.0
    max_efold = cfg.get("max_efold") or 25.0
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

    def leader_efold():
        """e-foldings of the largest centreline Fourier mode since the seed."""
        c0 = np.max(np.abs(np.fft.rfft(zcs[0])[1:]))
        cn = np.max(np.abs(np.fft.rfft(zcs[-1])[1:]))
        return float(np.log(max(cn, 1e-300) / max(c0, 1e-300)))

    record()
    stopped_early = False
    for it in range(n_steps):
        solver.step(cfg["dt"])
        if (it + 1) % rec_every == 0:
            record()
            if leader_efold() >= max_efold:
                stopped_early = True
                break

    ts = np.array(ts)
    n_ef = leader_efold()
    if stopped_early:
        print(f"  [stopped at t={ts[-1]:.1f} of {t_end:g}: leading mode reached "
              f"{n_ef:.1f} e-foldings (max_efold={max_efold:g}); running longer would "
              f"bury the slower modes in its round-off]")
    sig, cph, k_eff, n_efold = measure_sigma_c(ts, zcs, k, built["Ls"])
    if tag is None:
        tag = run_tag(cfg)
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
        h.attrs["k_eff"] = k_eff            # the wavenumber ACTUALLY seeded (use this, not kstar)
        h.attrs["n_efold"] = n_efold        # e-foldings over the fit window; <3 => do not trust
        h.attrs["leader_efold"] = n_ef      # total growth of the FASTEST mode
        h.attrs["stopped_early"] = int(stopped_early)   # 1 => max_efold cap hit, not t_end
        h.attrs["t_actual"] = float(ts[-1])
        h.attrs["bank_sinuosity"] = bank_sinuosity(cfg)
        h.attrs["tag"] = tag
        # the whole dispersion relation, from this one broadband run
        dk, dsig, dc, dne, dok = per_mode_dispersion(ts, zcs, built["Ls"])
        for nm, arr in (("disp_k", dk), ("disp_sigma", dsig), ("disp_c", dc),
                        ("disp_nefold", dne), ("disp_converged", dok)):
            h.create_dataset(nm, data=arr)
        # mode-classification diagnostics (PV / divergence / energy budget)
        for q, v in classify_mode(built).items():
            if q != "summary":
                h.attrs[f"diag_{q}"] = v
        h.attrs["g_eff"] = g_eff(cfg)
        h.attrs["RUN_CMD"] = RUN_CMD
    dk, dsig, dc, dne, dok = per_mode_dispersion(ts, zcs, built["Ls"])
    nconv = int(dok.sum())
    msg = ""
    if nconv:
        i = int(np.argmax(np.where(dok > 0, dsig, -np.inf)))
        msg = (f", fastest converged mode k={dk[i]:.3f} sigma={dsig[i]:+.4f} "
               f"c={dc[i]:+.4f}")
    print(f"  wrote {os.path.relpath(path, HERE)}  ({len(ts)} frames, "
          f"{nconv}/{len(dk)} modes converged{msg})")
    print(f"    {classify_mode(built)['summary']}")
    return path, built


if __name__ == "__main__":
    # ONE run of the case configured in CONFIG at the top of this file.
    # (Edit CONFIG and re-run; there are no modes and no command-line options.)
    run_ivp_SW(CONFIG)
