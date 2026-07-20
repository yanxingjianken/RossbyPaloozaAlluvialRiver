#!/usr/bin/env python3
# =============================================================================
#  meander_driver.py  --  THE single core Dedalus file for dedalus_meander2
# =============================================================================
#  Variable-depth H(x,y) potential-vorticity (PV) generalization of the
#  constant-depth vorticity-meander model in ../dedalus_meander/.
#
#  Dropping the flat-bed assumption turns absolute-vorticity conservation into
#  PV conservation  D/Dt(zeta/H) = (1/H) curl F, and the Rossby restoring force
#  gains a TOPOGRAPHIC-shear beta.  Linearized about a jet ubar(y) over a bed
#  Hbar(y) (x-homogeneous), the model has the SAME skeleton as the flat-bed one:
#
#      dt(zeta') + ubar(y) dx(zeta') + beta_top(y) dx(Psi') = curl F'
#      beta_top(y) = d/dy( zetabar / Hbar ),   zeta' = div( (1/Hbar) grad Psi' )
#
#  with the mass-transport streamfunction  u=-(1/H)dy(Psi), v=(1/H)dx(Psi).
#  Flat bed Hbar=1  ->  beta_top=2D, zeta'=lap(Psi')  ->  EXACTLY ../dedalus_meander.
#
#  Implementation (well-banded, per Dedalus perf docs + the user's own QG
#  scripts): carry zeta' as an AUXILIARY field (first-order reduction) and write
#  the elliptic constraint MULTIPLIED THROUGH by Hbar^2 to clear the 1/H:
#      Hbar*lap(Psi') - Hbar_y*dy(Psi') - Hbar2*zeta' = 0     (all-polynomial NCC)
#
# -----------------------------------------------------------------------------
#  ENVIRONMENT & RUN (micromamba env `dedalus`, Dedalus v3.0.5):
#
#    micromamba run -n dedalus env OMP_NUM_THREADS=1 python meander_driver.py --mode selftest
#    micromamba run -n dedalus env OMP_NUM_THREADS=1 python meander_driver.py --mode evp
#    micromamba run -n dedalus env OMP_NUM_THREADS=1 python meander_driver.py --mode ivp
#    micromamba run -n dedalus env OMP_NUM_THREADS=1 python meander_driver.py --mode sweep
#
#  All figure/video generation lives in ./postprocessing/ (reads outputs/*.h5).
#  This file writes RAW HDF5 only -- it never plots.
# =============================================================================
"""dedalus_meander2 core driver: variable-depth PV meander model + self-test."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# --------------------------------------------------------------------------- #
#  Paths + theory bridge to the constant-depth package (single source of truth)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "vorticity_meander"))
import vorticity_lib as VL          # noqa: E402
from vorticity_lib import ECOEF     # noqa: E402,F401

RUN_CMD = ("micromamba run -n dedalus env OMP_NUM_THREADS=1 "
           "python meander_driver.py")


# =========================================================================== #
#  CONFIG  --  the single-file knobs (edit here; CLI can override, see main)
# =========================================================================== #
CONFIG = dict(
    # --- base jet / channel ---------------------------------------------- #
    D=0.6,               # Delta/(U0+Delta); implies U0 = 1 - D (edge speed)
    Lx=20 * np.pi,       # domain length in b-units (half-width b = 1)
    # --- bed topography  H(x,y) = Hbar(y) * (1 + along) ------------------ #
    #   cross-channel (y): Hbar(y) = 1 + cross_amp*(1 - y^2)  [deeper thalweg]
    #   along-channel (x): factor (1 + along_amp*cos(k_bed*x + phase))
    #   cross_amp/along_amp = 0  ->  flat bed (recovers ../dedalus_meander)
    cross_amp=0.0,       # cross-channel depth bump (thalweg); 0 = flat
    along_amp=0.0,       # along-channel bedform amplitude (x); 0 = x-homogeneous
    along_kbed=0.30,     # along-channel bedform wavenumber (only if along_amp>0)
    along_phase=0.0,
    # --- rotation (topographic-beta knob) -------------------------------- #
    #   f0 = 0  ->  physical NON-ROTATING river: the bed only MODULATES the
    #              shear-beta d/dy(zetabar/Hbar); a bed bump alone (D=0) gives
    #              NO beta (no planetary vorticity to squash).
    #   f0 != 0 ->  formal planetary vorticity: beta_top = d/dy((zetabar+f0)/Hbar)
    #              gains -f0*Hbar_y/Hbar^2  =>  TRUE topographic Rossby waves
    #              (waves even at D=0).  Beyond the real river; a study knob.
    f0=0.0,
    # --- friction -------------------------------------------------------- #
    gamma=0.05,          # bottom-friction number  C_f b / H0
    friction="rayleigh", # "rayleigh" | "momentum"
    ECOEF=None,          # bank erodibility eps*C_f; None -> ECOEF[friction]
    # --- perturbation / numerics ----------------------------------------- #
    kstar=0.30,          # single-mode wavenumber (ivp) or EVP point
    Ny=192,              # cross-channel Chebyshev / GEP points
    Nx=64,               # streamwise Fourier (ivp)
    dt=0.02,
    t_end=None,          # None -> auto (~5/sigma)
    A0=1e-4,             # bank seed amplitude
)


# =========================================================================== #
#  Base-state profiles  (all analytic; Hbar polynomial => well-banded NCCs)
# =========================================================================== #
def hbar(y, cfg):
    """Cross-channel base depth Hbar(y) = 1 + cross_amp*(1-y^2) (nondim /H0)."""
    y = np.asarray(y, dtype=float)
    return 1.0 + cfg["cross_amp"] * (1.0 - y**2)


def hbar_y(y, cfg):
    """d Hbar/dy = -2 cross_amp y."""
    y = np.asarray(y, dtype=float)
    return -2.0 * cfg["cross_amp"] * y


def bed_depth(x, y, cfg):
    """Full 2-D bed depth H(x,y) = Hbar(y) * (1 + along_amp cos(k_bed x + ph)).

    x-dependence (along_amp>0) couples Fourier modes -> IVP-only (no EVP).
    Returns an array broadcast over (x, y) meshes.
    """
    Hb = hbar(y, cfg)
    if cfg["along_amp"] == 0.0:
        return Hb
    along = 1.0 + cfg["along_amp"] * np.cos(cfg["along_kbed"] * x
                                            + cfg["along_phase"])
    return Hb * along


def ubar(y, cfg):
    """Base jet ubar(y) = (1-D) + D(1-y^2) -- reuse the constant-depth profile."""
    return VL.u_profile(y, cfg["D"])


def ubar_y(y, cfg):
    """Cross-channel shear ubar_y = -2 D y."""
    return VL.u_profile_y(y, cfg["D"])


def zeta_bar(y, cfg):
    """Base relative vorticity zetabar = -ubar_y = 2 D y."""
    return -ubar_y(y, cfg)


def zeta_bar_y(y, cfg):
    """d zetabar/dy = 2 D (constant)."""
    y = np.asarray(y, dtype=float)
    return np.full_like(y, 2.0 * cfg["D"])


def beta_top(y, cfg):
    """Joint beta = d/dy((zetabar + f0)/Hbar), the base-PV gradient.

    = zetabar_y/Hbar - (zetabar + f0)*Hbar_y/Hbar^2.
    Non-rotating river (f0=0): the bed MODULATES the shear-beta; flat bed -> 2D,
    and D=0 gives beta=0 (no relative vorticity to redistribute -- there is NO
    standalone topographic beta without rotation).
    f0!=0: adds -f0*Hbar_y/Hbar^2 = true topographic Rossby beta (waves at D=0).
    """
    Hb = hbar(y, cfg)
    f0 = cfg.get("f0", 0.0)
    return (zeta_bar_y(y, cfg) / Hb
            - (zeta_bar(y, cfg) + f0) * hbar_y(y, cfg) / Hb**2)


def ecoef(cfg):
    return ECOEF[cfg["friction"]] if cfg["ECOEF"] is None else cfg["ECOEF"]


def bank_E(cfg):
    """Nondim bank-erosion rate E = eps*C_f*(1-D)."""
    return ecoef(cfg) * (1.0 - cfg["D"])


# =========================================================================== #
#  Variable-depth FD generalized eigenproblem (the validation oracle)
#  Mirrors VL.channel_matrices but with Hbar(y) coefficients.  A phi = w* B phi
# =========================================================================== #
def _fd_matrices(N):
    """Centred FD first/second-derivative matrices on y in [-1,1], N points."""
    y = np.linspace(-1.0, 1.0, N)
    h = y[1] - y[0]
    D1 = np.zeros((N, N))
    D2 = np.zeros((N, N))
    for j in range(1, N - 1):
        D1[j, j - 1] = -1.0 / (2 * h)
        D1[j, j + 1] = 1.0 / (2 * h)
        D2[j, j - 1] = 1.0 / h**2
        D2[j, j] = -2.0 / h**2
        D2[j, j + 1] = 1.0 / h**2
    return y, D1, D2


def channel_matrices_H(N, kstar, cfg):
    """(A, B) for the variable-H bank eigenproblem A phi = omega* B phi.

    This is the validation ORACLE (dense solve; rational coeffs are fine here).
    It encodes the SAME physics the d3 solver builds (well-banded x Hbar^2 form),
    written in eliminated form  phi = Psi'(y),  zeta' = M_H Psi'  with
        M_H = diag(1/Hbar)(D2 - k^2) - diag(Hbar_y/Hbar^2) D1.
    PV tendency (eliminated):  i w* M_H Psi'
        = [ i k* diag(ubar) M_H + i k* diag(beta_top) + FRIC ] Psi'
      rayleigh:  FRIC = diag(gamma/Hbar) M_H            (leading -(gamma/H) zeta';
                 the O(grad H) curl correction is dropped -- flat-bed exact)
      momentum:  FRIC = diag(gamma/Hbar) [2 ubar D2 + 2 ubar_y D1 - ubar k^2]
    Bank rows (nonlocal, Psi'-relaxation, deck p.7):  -i w* Psi_b = E(Psi_c - Psi_b).
    Flat bed Hbar=1 -> M_H=(D2-k^2), beta_top=2D  -> equals VL.channel_matrices
    exactly (worst |Delta omega| = 0 in the self-test).
    """
    gamma, E, fr = cfg["gamma"], bank_E(cfg), cfg["friction"]
    y, D1, D2 = _fd_matrices(N)
    Hb, Hby = hbar(y, cfg), hbar_y(y, cfg)
    ub, uby = ubar(y, cfg), ubar_y(y, cfg)
    bt = beta_top(y, cfg)
    ic = N // 2
    I = np.eye(N)
    L = D2 - kstar**2 * I
    M_H = np.diag(1.0 / Hb) @ L - np.diag(Hby / Hb**2) @ D1

    A = np.zeros((N, N), dtype=complex)
    B = np.zeros((N, N), dtype=complex)
    for j in range(1, N - 1):
        rowM = M_H[j]
        A[j] = 1j * kstar * ub[j] * rowM + 1j * kstar * bt[j] * I[j]
        if fr == "rayleigh":
            A[j] += (gamma / Hb[j]) * rowM
        elif fr == "momentum":
            A[j] += (gamma / Hb[j]) * (2.0 * ub[j] * D2[j] + 2.0 * uby[j] * D1[j]
                                       - ub[j] * kstar**2 * I[j])
        else:
            raise ValueError(f"unknown friction closure {fr!r}")
        B[j] = 1j * rowM
    for b_ in (0, N - 1):
        A[b_, b_] = -E
        A[b_, ic] = E
        B[b_, b_] = -1j
    return A, B


def channel_modes_H(N, kstar, cfg):
    """Finite eigenvalues/vectors of the variable-H GEP."""
    from scipy.linalg import eig
    A, B = channel_matrices_H(N, kstar, cfg)
    w, V = eig(A, B)
    ok = np.isfinite(w) & (np.abs(w) < 1e6)
    return w[ok], V[:, ok]


def gep_bank_mode_H(N, kstar, cfg):
    """Bank eigenvalue from the variable-H GEP, nearest the analytic branch."""
    target = complex(VL.bank_branch([kstar], cfg["D"], cfg["gamma"],
                                    bank_E(cfg), cfg["friction"])[0])
    w, _ = channel_modes_H(N, kstar, cfg)
    return complex(w[np.argmin(np.abs(w - target))])


def gep_richardson_H(kstar, cfg, Ns=(201, 401)):
    """O(h^2)->O(h^4) Richardson extrapolation of the FD bank eigenvalue."""
    o1 = gep_bank_mode_H(Ns[0], kstar, cfg)
    o2 = gep_bank_mode_H(Ns[1], kstar, cfg)
    return (4.0 * o2 - o1) / 3.0


# =========================================================================== #
#  Dedalus variable-H EVP  (zeta auxiliary + x Hbar^2 well-banded form)
#  Shares the SAME equations with the IVP (built in P1c).  1-D in y at fixed k*.
# =========================================================================== #
def _pv_eq(friction):
    """PV-tendency equation string (multiplied through by Hbar^2 => polynomial
    NCCs; eigenvalue-preserving row scaling).  Coefficient NCCs are prebuilt in
    the namespace: H2=Hbar^2, H2u=Hbar^2*ubar, beta2=Hbar^2*beta_top, gamH=gamma*Hbar,
    gHu=gamma*Hbar*ubar, gHuy=gamma*Hbar*ubar_y."""
    if friction == "rayleigh":
        return ("H2*dt(zeta) + H2u*dx(zeta) + beta2*dx(psi) + gamH*zeta = 0")
    if friction == "momentum":
        return ("H2*dt(zeta) + H2u*dx(zeta) + beta2*dx(psi)"
                " + 2*gHu*dy(dy(psi)) + 2*gHuy*dy(psi) + gHu*dx(dx(psi)) = 0")
    raise ValueError(f"unknown friction closure {friction!r}")

# Elliptic constraint (Hbar^2-weighted, well-banded, all-polynomial NCCs):
#   Hbar*lap(psi) - Hbar_y*dy(psi) - H2*zeta = 0     (zeta = div((1/Hbar) grad psi))
_ELLIPTIC_EQ = ("Hbar*lap(psi) - Hbar_y*dy(psi) - H2*zeta"
                " + lift(tau1,-1) + lift(tau2,-2) = 0")
_BANK_EQS = ("psi(y=1) - psib_top = 0",
             "psi(y=-1) - psib_bot = 0",
             "dt(psib_top) + E*psib_top - E*psi(y=0) = 0",
             "dt(psib_bot) + E*psib_bot - E*psi(y=0) = 0")


def _nccs_1d(dist, ybasis, y, cfg):
    """Build the polynomial NCC coefficient fields on a 1-D y-basis."""
    import dedalus.public as d3      # noqa: F401  (namespace parity)
    Hb, Hby = hbar(y, cfg), hbar_y(y, cfg)
    ub, uby, bt = ubar(y, cfg), ubar_y(y, cfg), beta_top(y, cfg)
    vals = dict(Hbar=Hb, Hbar_y=Hby, H2=Hb**2, H2u=Hb**2 * ub,
                beta2=Hb**2 * bt, gamH=cfg["gamma"] * Hb,
                gHu=cfg["gamma"] * Hb * ub, gHuy=cfg["gamma"] * Hb * uby)
    fields = {}
    for name, arr in vals.items():
        f = dist.Field(name=name, bases=(ybasis,))
        f["g"] = arr
        fields[name] = f
    return fields


def build_evp_H(kstar, cfg, Ny=None):
    """Variable-H EVP at fixed k*; returns dict(solver, fields, y)."""
    import dedalus.public as d3
    Ny = Ny or cfg["Ny"]
    E = bank_E(cfg)
    ycoord = d3.Coordinate("y")
    dist = d3.Distributor(ycoord, dtype=np.complex128)
    ybasis = d3.Chebyshev(ycoord, size=Ny, bounds=(-1.0, 1.0))
    y = dist.local_grid(ybasis)

    psi = dist.Field(name="psi", bases=(ybasis,))
    zeta = dist.Field(name="zeta", bases=(ybasis,))
    psib_top = dist.Field(name="psib_top")
    psib_bot = dist.Field(name="psib_bot")
    tau1 = dist.Field(name="tau1")
    tau2 = dist.Field(name="tau2")
    omega = dist.Field(name="omega")

    dy = lambda A: d3.Differentiate(A, ycoord)
    dtev = lambda A: -1j * omega * A
    dxev = lambda A: 1j * kstar * A
    lapev = lambda A: dy(dy(A)) - kstar**2 * A
    lift_basis = ybasis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    ns = dict(psi=psi, zeta=zeta, psib_top=psib_top, psib_bot=psib_bot,
              tau1=tau1, tau2=tau2, E=E, dx=dxev, dy=dy, lap=lapev, lift=lift,
              dt=dtev)
    ns.update(_nccs_1d(dist, ybasis, y, cfg))

    problem = d3.EVP([psi, zeta, psib_top, psib_bot, tau1, tau2],
                     eigenvalue=omega, namespace=ns)
    problem.add_equation(_ELLIPTIC_EQ)
    problem.add_equation(_pv_eq(cfg["friction"]))
    for eq in _BANK_EQS:
        problem.add_equation(eq)
    solver = problem.build_solver()
    return dict(solver=solver, psi=psi, zeta=zeta,
                psib_top=psib_top, psib_bot=psib_bot, y=y, kstar=kstar)


def evp_bank_mode_H(kstar, cfg, Ny=None, nev=10, target=None):
    """Targeted sparse bank eigenvalue omega* of the variable-H EVP."""
    built = build_evp_H(kstar, cfg, Ny)
    solver = built["solver"]
    if target is None:
        # flat-bed analytic branch is a good shift-invert target
        target = complex(VL.bank_branch([kstar], cfg["D"], cfg["gamma"],
                                        bank_E(cfg), cfg["friction"])[0])
    sp = solver.subproblems[0]
    solver.solve_sparse(sp, N=nev, target=target)
    w = solver.eigenvalues
    ok = np.isfinite(w) & (np.abs(w) < 50.0)
    w = w[ok]
    if w.size == 0:
        return complex("nan")
    return complex(w[np.argmin(np.abs(w - target))])


# =========================================================================== #
#  Dedalus variable-H IVP  (2-D, same equations; writes RAW HDF5 to outputs/)
# =========================================================================== #
def build_ivp_H(cfg, Nx=None, Ny=None):
    """2-D variable-H IVP (RealFourier x Chebyshev), zeta-auxiliary form.

    NCCs are y-only (validated H=Hbar(y) case).  x-varying bed (along_amp>0)
    is a Phase-3 extension (base flow over a wavy bed needs its own treatment);
    here along_amp is carried in the config/HDF5 but the coefficient fields use
    Hbar(y).  Returns dict(solver, fields, grids, diagnostic ops).
    """
    import dedalus.public as d3
    Nx = Nx or cfg["Nx"]
    Ny = Ny or cfg["Ny"]
    E = bank_E(cfg)
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0.0, cfg["Lx"]))
    ybasis = d3.Chebyshev(coords["y"], size=Ny, bounds=(-1.0, 1.0))
    x, y = dist.local_grids(xbasis, ybasis)

    psi = dist.Field(name="psi", bases=(xbasis, ybasis))
    zeta = dist.Field(name="zeta", bases=(xbasis, ybasis))
    psib_top = dist.Field(name="psib_top", bases=(xbasis,))
    psib_bot = dist.Field(name="psib_bot", bases=(xbasis,))
    tau1 = dist.Field(name="tau1", bases=(xbasis,))
    tau2 = dist.Field(name="tau2", bases=(xbasis,))

    # y-only NCC coefficient fields (polynomial => well-banded)
    yg = y.ravel()
    Hb, Hby = hbar(yg, cfg), hbar_y(yg, cfg)
    ub, uby, bt = ubar(yg, cfg), ubar_y(yg, cfg), beta_top(yg, cfg)
    ncc_vals = dict(Hbar=Hb, Hbar_y=Hby, H2=Hb**2, H2u=Hb**2 * ub,
                    beta2=Hb**2 * bt, gamH=cfg["gamma"] * Hb,
                    gHu=cfg["gamma"] * Hb * ub, gHuy=cfg["gamma"] * Hb * uby)
    nccs = {}
    for name, arr in ncc_vals.items():
        f = dist.Field(name=name, bases=(ybasis,))
        f["g"] = arr[None, :]
        nccs[name] = f

    dx = lambda A: d3.Differentiate(A, coords["x"])
    dy = lambda A: d3.Differentiate(A, coords["y"])
    lift_basis = ybasis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    ns = dict(psi=psi, zeta=zeta, psib_top=psib_top, psib_bot=psib_bot,
              tau1=tau1, tau2=tau2, E=E, dx=dx, dy=dy, lap=d3.Laplacian,
              lift=lift, dt=d3.TimeDerivative)
    ns.update(nccs)

    problem = d3.IVP([psi, zeta, psib_top, psib_bot, tau1, tau2], namespace=ns)
    problem.add_equation(_ELLIPTIC_EQ)
    problem.add_equation(_pv_eq(cfg["friction"]))
    for eq in _BANK_EQS:
        problem.add_equation(eq)
    solver = problem.build_solver(d3.RK222)

    uv_op = (-dy(psi)) * dx(psi)           # u'v'  (up to 1/H factors; diagnostic)
    return dict(solver=solver, psi=psi, zeta=zeta, psib_top=psib_top,
                psib_bot=psib_bot, x=x.ravel(), y=y.ravel(), Nx=Nx, Ny=Ny,
                Lx=cfg["Lx"], dist=dist, uv_op=uv_op)


def seed_ivp_H(built, modes):
    """Analytic DAE-consistent IC: sinusoidal banks + harmonic Psi (zeta~0).

    modes = list of (kstar, amplitude, x0).  Banks seeded same-sign (sinuous
    psihat_1=psihat_3); Psi given the flat-bed harmonic extension cosh(k y)/cosh(k)
    so relative vorticity starts ~0 (the small variable-H mismatch projects out).
    """
    psi, zeta = built["psi"], built["zeta"]
    top, bot = built["psib_top"], built["psib_bot"]
    x, y = built["x"], built["y"]
    psi["g"][:] = 0.0
    zeta["g"][:] = 0.0
    tb = np.zeros_like(x)
    for (k, a, x0) in modes:
        tb = tb + a * np.cos(k * (x - x0))
        psi["g"] += (a * np.cos(k * (x[:, None] - x0))
                     * np.cosh(k * y[None, :]) / np.cosh(k))
    top["g"][:] = tb[:, None]        # bank field grid is (Nx, 1) in a 2-D dist
    bot["g"][:] = tb[:, None]


def run_ivp_H(cfg, tag=None):
    """Run one variable-H IVP and write raw HDF5 to outputs/.  Returns the path."""
    import h5py
    k = cfg["kstar"]
    built = build_ivp_H(cfg)
    seed_ivp_H(built, [(k, cfg["A0"], 0.0)])
    # auto duration ~ 5 e-foldings from the EVP growth rate
    om = evp_bank_mode_H(k, cfg, Ny=min(cfg["Ny"], 128))
    sig = float(om.imag)
    if cfg["t_end"] is not None:
        t_end = cfg["t_end"]
    elif sig > 1e-3:
        t_end = min(5.0 / sig, 160.0)
    else:
        t_end = min(3.0 / max(abs(sig), 1e-3), 120.0)
    dt = cfg["dt"]
    nstep = int(round(t_end / dt))
    nframes = 72
    rec_every = max(1, nstep // nframes)

    solver = built["solver"]
    solver.stop_iteration = nstep + 1
    psi, top, bot = built["psi"], built["psib_top"], built["psib_bot"]
    dist = built["dist"]
    ts, tops, bots, psis = [], [], [], []

    def record():
        ts.append(solver.sim_time)
        top.change_scales(1); bot.change_scales(1); psi.change_scales(1)
        tops.append(np.array(top["g"]).ravel())      # (Nx,1) -> (Nx,)
        bots.append(np.array(bot["g"]).ravel())
        psis.append(np.array(psi["g"]))              # (Nx, Ny)

    record()
    for it in range(1, nstep + 1):
        solver.step(dt)
        if it % rec_every == 0:
            record()

    m = int(round(k * cfg["Lx"] / (2 * np.pi)))
    tag = tag or f"k{k:.2f}_amp{cfg['cross_amp']:.2f}_{cfg['friction']}".replace(".", "p")
    path = os.path.join(OUT_DIR, f"run_{tag}.h5")
    with h5py.File(path, "w") as h:
        h.create_dataset("t", data=np.array(ts))
        h.create_dataset("top", data=np.array(tops))
        h.create_dataset("bot", data=np.array(bots))
        h.create_dataset("psi", data=np.array(psis))
        h.create_dataset("x", data=built["x"])
        h.create_dataset("y", data=built["y"])
        h.create_dataset("Hbed", data=bed_depth(built["x"][:, None],
                                                built["y"][None, :], cfg))
        for kk, vv in cfg.items():
            h.attrs[kk] = ("None" if vv is None else vv)
        h.attrs["mode_index"] = m
        h.attrs["sigma_evp"] = sig
        h.attrs["t_end"] = t_end
        h.attrs["RUN_CMD"] = RUN_CMD
    print(f"  wrote {os.path.relpath(path, HERE)}  "
          f"(k*={k}, {len(ts)} frames, t_end={t_end:.0f}, sigma_evp={sig:.4f})")
    return path, built


# --- IVP measurement (demodulate + fit) mirrors channel_lib -------------- #
def demodulate(series, m):
    """Complex amplitude a_m(t) of cos-mode m via rFFT (series [Nt, Nx])."""
    Nx = series.shape[1]
    return (2.0 / Nx) * np.fft.rfft(series, axis=1)[:, m]


def fit_sigma_c(t, a, kstar, window):
    """(sigma, c, r2) from a ~ e^{sigma t} e^{-i omega_r t} over a fit window."""
    t = np.asarray(t)
    sel = (t >= window[0]) & (t <= window[1])
    assert sel.sum() >= 5, "fit window too short"
    tt, aa = t[sel], a[sel]
    la = np.log(np.abs(aa))
    A = np.vstack([tt, np.ones_like(tt)]).T
    (sigma, _), res, *_ = np.linalg.lstsq(A, la, rcond=None)
    ss = 1.0 - (res[0] if len(res) else 0.0) / max(np.var(la) * len(la), 1e-30)
    ph = np.unwrap(np.angle(aa))
    (dphi, _), *_ = np.linalg.lstsq(A, ph, rcond=None)
    return float(sigma), float(-dphi / kstar), float(ss)


# =========================================================================== #
#  Self-test  --  Stage 1: flat-bed reduction (this milestone)
# =========================================================================== #
def _selftest_reduction():
    """H=1: the variable-H GEP must reproduce VL.channel_modes to ~1e-6."""
    print("Stage 1 -- flat-bed reduction (variable-H GEP  ==  VL.channel_modes)")
    print("-" * 74)
    flat = dict(CONFIG, cross_amp=0.0, along_amp=0.0)
    worst = 0.0
    for fr in ("rayleigh", "momentum"):
        cfg = dict(flat, friction=fr)
        E = bank_E(cfg)
        for kstar in (0.15, 0.3, 0.5, 0.9, 1.3):
            for N in (81, 151):
                wH, _ = channel_modes_H(N, kstar, cfg)
                wV, _ = VL.channel_modes(N, kstar, cfg["D"], cfg["gamma"], E, fr)
                # match each VL eigenvalue to the nearest variable-H eigenvalue
                for o in wV:
                    d = np.min(np.abs(wH - o))
                    worst = max(worst, d)
                    assert d < 1e-6, (f"{fr} k*={kstar} N={N}: eig mismatch "
                                      f"{d:.2e} (o={o:.5f})")
        print(f"  {fr:>8}: all eigenvalues match across k*,N.  worst so far "
              f"= {worst:.2e}")
    print(f"Stage 1 PASSED  (worst |Delta omega| = {worst:.2e} < 1e-6)")
    return worst


def _selftest_evp():
    """Stage 2: d3 EVP == variable-H GEP (Richardson) for flat AND bumped bed.
       Stage 3: topographic Rossby (f0!=0, D=0) supports a propagating wave."""
    print("\nStage 2 -- d3 EVP  ==  variable-H GEP (flat + bumped bed)")
    print("-" * 74)
    worst = 0.0
    for amp in (0.0, 0.3):
        for fr in ("rayleigh", "momentum"):
            cfg = dict(CONFIG, cross_amp=amp, friction=fr, Ny=192)
            for kstar in (0.3, 0.5, 0.9):
                od3 = evp_bank_mode_H(kstar, cfg)
                ogep = gep_richardson_H(kstar, cfg)
                d = abs(od3 - ogep)
                worst = max(worst, d)
                assert d < 2e-3, (f"amp={amp} {fr} k*={kstar}: d3 {od3:.5f} vs "
                                  f"GEP {ogep:.5f} |d|={d:.2e}")
            print(f"  amp={amp} {fr:>8}: d3==GEP over k*  (worst {worst:.2e})")
    print(f"Stage 2 PASSED  (worst |d3 - GEP| = {worst:.2e} < 2e-3)")

    print("\nStage 3 -- topographic Rossby (f0=1, D=0, bumped bed): waves exist")
    print("-" * 74)
    cfg = dict(CONFIG, D=0.0, f0=1.0, cross_amp=0.4, gamma=0.0, friction="rayleigh")
    # bed alone (f0=0) gives NO beta; f0!=0 gives topographic Rossby.
    assert np.allclose(beta_top(np.linspace(-1, 1, 9),
                                dict(cfg, f0=0.0)), 0.0), \
        "non-rotating bed bump must give ZERO beta (no standalone topo beta)"
    w, _ = channel_modes_H(301, 0.3, cfg)
    prop = w[np.argmax(np.abs(w.real))]
    assert abs(prop.real) > 1e-2, "f0!=0 must give a propagating topographic wave"
    print(f"  f0=0 bed bump: beta_top == 0 (correct, non-rotating).")
    print(f"  f0=1 bed bump: propagating wave omega* = {prop:.4f} "
          f"(Re != 0 => topographic Rossby).  PASSED")
    return worst


def _selftest_ivp():
    """Stage 4: variable-H IVP == EVP (measured sigma, c) for flat + bumped bed."""
    print("\nStage 4 -- variable-H IVP  ==  EVP (measured sigma, c)")
    print("-" * 74)
    k = 0.3
    for amp in (0.0, 0.3):
        cfg = dict(CONFIG, cross_amp=amp, Ny=96, dt=0.05,
                   Lx=2 * np.pi / k)             # single wavelength -> mode m=1
        built = build_ivp_H(cfg, Nx=16, Ny=96)
        seed_ivp_H(built, [(k, 1e-3, 0.0)])
        solver = built["solver"]
        top, bot = built["psib_top"], built["psib_bot"]
        om = evp_bank_mode_H(k, cfg, Ny=96)
        t_end = min(5.0 / max(om.imag, 1e-3), 120.0)
        dt = cfg["dt"]
        nstep = int(round(t_end / dt))
        solver.stop_iteration = nstep + 1
        ts, series = [], []
        for it in range(nstep + 1):
            if it:
                solver.step(dt)
            top.change_scales(1); bot.change_scales(1)
            ts.append(solver.sim_time)
            series.append(0.5 * (np.array(top["g"]).ravel()
                                 + np.array(bot["g"]).ravel()))
        a = demodulate(np.array(series), 1)
        sig, c, r2 = fit_sigma_c(np.array(ts), a, k, (t_end / 3, t_end))
        sig_e, c_e = om.imag, om.real / k
        ds = abs(sig - sig_e) / max(abs(sig_e), 1e-3)
        dc = abs(c - c_e) / max(abs(c_e), 1e-3)
        print(f"  amp={amp}: IVP sigma={sig:.4f} (EVP {sig_e:.4f}, {ds*100:.1f}%)"
              f"  c={c:.4f} (EVP {c_e:.4f}, {dc*100:.1f}%)  R2={r2:.4f}")
        assert ds < 0.03 and dc < 0.03, f"amp={amp}: IVP-vs-EVP off (>3%)"
    print("Stage 4 PASSED  (IVP == EVP within 3%, flat + bumped bed)")


def main():
    ap = argparse.ArgumentParser(description="dedalus_meander2 variable-H PV driver")
    ap.add_argument("--mode", default="selftest",
                    choices=("selftest", "evp", "ivp", "sweep"))
    ap.add_argument("--kstar", type=float, default=None)
    ap.add_argument("--cross-amp", type=float, default=None)
    ap.add_argument("--along-amp", type=float, default=None)
    ap.add_argument("--friction", default=None, choices=("rayleigh", "momentum"))
    args = ap.parse_args()
    cfg = dict(CONFIG)
    for k, a in (("kstar", args.kstar), ("cross_amp", args.cross_amp),
                 ("along_amp", args.along_amp), ("friction", args.friction)):
        if a is not None:
            cfg[k] = a

    if args.mode == "selftest":
        _selftest_reduction()
        _selftest_evp()
        _selftest_ivp()
        print("\nAll self-tests passed.")
    elif args.mode == "evp":
        om = evp_bank_mode_H(cfg["kstar"], cfg)
        og = gep_richardson_H(cfg["kstar"], cfg)
        print(f"k*={cfg['kstar']} cross_amp={cfg['cross_amp']} {cfg['friction']}: "
              f"d3 omega* = {om:.5f}  (sigma={om.imag:.4f}, c={om.real/cfg['kstar']:.4f})"
              f"  | GEP {og:.5f}  |d|={abs(om-og):.2e}")
    elif args.mode == "ivp":
        run_ivp_H(cfg)
    elif args.mode == "sweep":
        ks = [round(0.1 * m, 1) for m in range(1, 16)]  # k* = 0.1 .. 1.5
        print(f"sweep over k* = {ks}  (cross_amp={cfg['cross_amp']}, "
              f"{cfg['friction']})")
        for kv in ks:
            run_ivp_H(dict(cfg, kstar=kv))
    else:
        raise SystemExit(f"unknown mode '{args.mode}'")


if __name__ == "__main__":
    main()
