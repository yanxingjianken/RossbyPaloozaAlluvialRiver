#!/usr/bin/env python3
"""Linear shallow water on a meandering, erodible channel -- (s,n) curvilinear.

    micromamba run -n dedalus env OMP_NUM_THREADS=1 python sw_meander.py

Edit CONFIG below and run.  This file ONLY generates simulation output: it writes raw
fields to outputs/run_<tag>.h5 and stops.  Every diagnostic (growth rates, dispersion,
mode classification) lives in postprocessing/analysis.py, so that nothing here can
quietly bake an interpretation into the data.

Physics.  Free surface eta is prognostic, so gravity waves and the vortical branch
coexist and the flow is genuinely divergent (there is no streamfunction).  With the
channel-fitted metric sigma = 1 + n*C(s) and still-water depth H(n):

    eta_t   + (1/s)[d_s(H u_s) + d_n(s H u_n)]                     = 0
    d_t u_s + (Ub/s) d_s u_s + (d_n Ub + C Ub/s) u_n + (g/s) d_s eta
              + r_s u_s - r_e eta - nu lap(u_s)                    = 0
    d_t u_n + (Ub/s) d_s u_n - 2 C Ub u_s / s     + g d_n eta
              + r_n u_n - nu lap(u_n) + Ub^2 d_ss(zc)              = 0

Every equation is multiplied through by sigma to clear the rational 1/sigma, which
keeps the Dedalus NCC matrices banded.  Drag is anisotropic (r_s = 2 C_f Ub/H versus
r_n = C_f Ub/H: a streamwise perturbation changes |u| as well as its direction), and
r_e = C_f Ub^2/H^2 is the superelevation-drag term of Ikeda-Parker-Sawai (1981) -- the
only route by which the free surface drives bend flow, hence the only place F enters.

Walls at n = +/-b: no penetration u_n = 0 (this assumes nothing about interior
divergence) plus free slip d_n u_s = 0, closed with two tau terms per viscous velocity.
The centreline erodes by the antisymmetric near-bank velocity (Ikeda bend growth) and
feeds back through the curvature C' = -d_ss(zc).

Full derivation of every term: derivations/sw_sn_meander.pdf
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

G_ACCEL = 9.81            # m/s^2 -- everything in this file is SI

CONFIG = dict(
    # =================== CHANNEL GEOMETRY [m] ==============================
    # Typical sand-bed meandering river.  The aspect ratio b/H ~ 17 is what puts
    # this in the regime Ikeda-Parker-Sawai and the bar-theory literature describe.
    b=50.0,              # half-width [m]      -> river width W = 2b = 100 m
    H0=3.0,              # mean still-water depth [m]   -> b/H = 17
    # =================== BASE JET [m/s] ====================================
    # Ub(n) = U0 + Delta*(1 - (n/b)^2): parabolic on purpose, so d2Ub/dn2 is CONSTANT
    # and the cross-channel vorticity gradient d(zeta_bar)/dn = 2*Delta/b^2 is uniform
    # -- the channel analogue of a planetary beta.  Delta IS that gradient; Delta=0
    # removes it.  NOTE the two are not independent in effect: the erosion law reads
    # u_s AT THE BANK, where Ub = U0, so compare runs at MATCHED U0 or the comparison
    # is confounded (this inverted a conclusion once -- see the derivation note).
    U0=0.8,              # speed at the bank edge [m/s]  (this drives the erosion)
    Delta=0.6,           # excess at the centre [m/s]    -> centreline 1.4 m/s
    # =================== BED [m] ===========================================
    # FLAT bed by default.  H(n) = H0*[1 + cross_amp*(1 - (n/b)^2)] also supports a
    # parabolic thalweg (deepest mid-channel, like the jet): set e.g. cross_amp=0.30
    # for a 30% deeper centreline.  An ALONG-channel bed is deliberately NOT wired in:
    # it would make the base flow s-dependent through discharge conservation.
    cross_amp=0.0,       # 0 = flat bed;  >0 = parabolic thalweg [-]
    # =================== BACKGROUND MEANDER ================================
    # Cbar(s) = Cbar_amp * cos(kmeander * s), curvature in [1/m]
    kmeander=6.0e-3,     # bend wavenumber [1/m] -> wavelength 1047 m ~ 10.5 W,
                         #   which is the observed meander scale (Leopold-Wolman)
    A_bank=10.0,         # centreline amplitude of the initial bend train [m]
    Cbar_amp=None,       # override the curvature amplitude [1/m] (None -> A_bank*k^2)
    n_bends=4,           # whole bends in the periodic domain -> reach 4189 m
    Ls=None,             # domain length [m] (None -> n_bends * 2pi/kmeander)
    # =================== FRICTION / VISCOSITY ==============================
    Cf=0.005,            # bottom drag coefficient [-]; sand-bed rivers are 0.002-0.01
    nu=0.15,             # lateral eddy viscosity [m^2/s].  Physical estimate is
                         #   alpha*u_*H with u_*=sqrt(Cf)*U ~ 0.1 m/s, H=3 m,
                         #   alpha~0.1-0.6  ->  0.03-0.18 m^2/s.
    # =================== BANK EROSION ======================================
    # dt(zeta_c) = E * 0.5*[u_s(+b) - u_s(-b)]
    # FIELD erodibility is ~1e-8 m/s (about 0.3 m/yr of bank retreat).  At that value
    # the instability e-folds in ~2 centuries, which cannot be integrated.  E below is
    # inflated ~1e7x so the mode develops in an hour of simulated time.  sigma is
    # LINEAR in E, so a field growth rate is sigma * (1e-8 / E) -- the movies print the
    # implied bank-migration rate so the inflation is never invisible.
    E=0.1,               # bank erodibility [m/s]  (field value ~1e-8)
    # =================== THE PERTURBATION: one drop of ink =================
    # A single localised bump on the centreline, flow at rest.  Being localised it is
    # automatically broadband -- a narrow bump contains every wavenumber -- so one run
    # still yields the whole dispersion relation.  A true delta would also excite the
    # grid scale; the finite width is what keeps the seeded spectrum smooth.
    A0=0.05,             # seed amplitude [m] (irrelevant: the system is linear)
    seed_s0=0.25,        # release point, as a fraction of the reach
    seed_width=50.0,     # bump half-width [m] (= one channel half-width)
    # =================== NUMERICS ==========================================
    Ns=128,              # streamwise Fourier modes
    Nn=96,               # cross-channel Chebyshev modes
    dt=0.4,              # timestep [s]  (all linear terms are implicit, so this is
                         #   an accuracy choice, not a CFL limit)
    t_end=4300.0,        # [s] ~ 72 min
    n_out=81,            # snapshots written
    # Stop early once the FASTEST mode has grown this many e-foldings.  Modes further
    # than ~log(1/eps) ~ 36 e-foldings below the leader drown in its FFT round-off and
    # read back the LEADER's growth rate, which looks perfectly converged.  A single
    # fixed t_end cannot serve every configuration.
    max_efold=25.0,
)


# --------------------------------------------------------------------------- #
#  Base state (pure numpy -- inspectable without touching Dedalus)
# --------------------------------------------------------------------------- #
def g_eff(cfg):         return G_ACCEL                  # SI: no 1/F^2 rescaling
def bank_E(cfg):        return cfg["E"]                 # [m/s], set directly
def froude(cfg):        return center_speed(cfg) / np.sqrt(G_ACCEL * cfg["H0"])
def center_speed(cfg):  return cfg["U0"] + cfg["Delta"]
def ubar_s(n, cfg):     return cfg["U0"] + cfg["Delta"] * (1.0 - (n / cfg["b"]) ** 2)
def ubar_s_n(n, cfg):   return -2.0 * cfg["Delta"] * n / cfg["b"] ** 2
def ubar_s_nn(cfg):     return -2.0 * cfg["Delta"] / cfg["b"] ** 2   # = -d(zeta_bar)/dn
def bed_depth(n, cfg):  return cfg["H0"] * (1.0 + cfg["cross_amp"] * (1.0 - (n / cfg["b"]) ** 2))
def domain_length(cfg): return cfg["Ls"] or cfg["n_bends"] * 2 * np.pi / cfg["kmeander"]
def sigma_metric(s, n, cfg): return 1.0 + n * cbar(s, cfg)   # >0 required (no folding)


def bank_sinuosity(cfg):
    """Curvature amplitude of the initial bend train (0 = straight)."""
    return cfg["Cbar_amp"] if cfg["Cbar_amp"] is not None else cfg["A_bank"] * cfg["kmeander"] ** 2


def cbar(s, cfg):
    return bank_sinuosity(cfg) * np.cos(cfg["kmeander"] * s)


def etabar(s, n, cfg):
    """Base superelevation from centrifugal balance  g d_n etabar = Cbar Ub^2/sigma.

    Not used by the solver (it is a steady residual that cancels from the perturbation
    equations) but it IS the base state, so it lives here rather than in postprocessing.

    Integrating outwards from etabar(n=0)=0:
        etabar(s,n) = (Cbar(s)/g) * INT_0^n Ub(n')^2 / (1 + n' Cbar(s)) dn'
    The 1/sigma inside the integral depends on s, so the quadrature cannot be done once
    and reused -- it is carried out per s (vectorised).  Returns (len(s), len(n)).
    """
    s_a = np.atleast_1d(np.asarray(s, dtype=float))
    n_a = np.atleast_1d(np.asarray(n, dtype=float)).ravel()
    ng = np.linspace(-cfg["b"], cfg["b"], 2001)                 # quadrature grid in n'
    Cb = cbar(s_a, cfg).reshape(-1, 1)                          # (S,1)
    f = ubar_s(ng, cfg)[None, :] ** 2 / (1.0 + ng[None, :] * Cb)
    I = np.concatenate([np.zeros((len(s_a), 1)),                # trapezoid, not a
                        np.cumsum(0.5 * (f[:, 1:] + f[:, :-1])  # left-rectangle sum
                                  * np.diff(ng)[None, :], axis=1)], axis=1)
    I -= I[:, [int(np.argmin(np.abs(ng)))]]                     # anchor at n=0
    prof = np.array([np.interp(n_a, ng, row) for row in I])     # (S,N)
    # squeeze so a scalar s yields a plain n-profile, which is how callers use it
    return np.squeeze(Cb * prof / g_eff(cfg))


def run_tag(cfg):
    """Name a run by its PHYSICS -- never by "which wavelength was poked", because the
    seed is a localised bump containing all of them.  Delta must appear: it IS the
    vorticity gradient, so runs differing only in Delta are physically different."""
    bed = "flat" if cfg["cross_amp"] == 0 else f"cross{cfg['cross_amp']:.2f}"
    # tag the bank by the DIMENSIONLESS sinuosity Cbar*b (the QGPV note's eps_c, and the
    # quantity the metric validity condition is stated in), not by the raw curvature in
    # 1/m -- in SI that is O(1e-4) and would print as 0.000 for every run.
    return (f"H{bed}_bank{bank_sinuosity(cfg)*cfg['b']:.3f}_Cf{cfg['Cf']:.4f}"
            f"_U{cfg['U0']:.2f}dU{cfg['Delta']:+.2f}").replace(".", "p")


# --------------------------------------------------------------------------- #
#  Build, seed, run
# --------------------------------------------------------------------------- #
def build(cfg):
    """Assemble the linear (s,n) shallow-water IVP.  Returns a dict of handles."""
    import dedalus.public as d3
    b, g, E, Ls = cfg["b"], g_eff(cfg), bank_E(cfg), domain_length(cfg)

    coords = d3.CartesianCoordinates("s", "n")
    dist = d3.Distributor(coords, dtype=np.float64)
    sbasis = d3.RealFourier(coords["s"], size=cfg["Ns"], bounds=(0.0, Ls))
    nbasis = d3.Chebyshev(coords["n"], size=cfg["Nn"], bounds=(-b, b))
    s, n = dist.local_grids(sbasis, nbasis)             # s:(Ns,1)  n:(1,Nn)
    ds = lambda A: d3.Differentiate(A, coords["s"])
    dn = lambda A: d3.Differentiate(A, coords["n"])

    us, un, eta = (dist.Field(name=x, bases=(sbasis, nbasis)) for x in ("us", "un", "eta"))
    zc = dist.Field(name="zc", bases=(sbasis,))         # centreline offset = the meander
    taus = [dist.Field(bases=(sbasis,)) for _ in range(4)]   # 2 per viscous velocity

    # NCC coefficient fields.  Everything is pre-multiplied by sigma so no 1/sigma
    # survives; a rational NCC would make the matrices dense.
    Cb, Ub, Ubn, Hn = cbar(s, cfg), ubar_s(n, cfg), ubar_s_n(n, cfg), bed_depth(n, cfg)
    sig_a = 1.0 + n * Cb
    def f2d(arr):                       # coefficient field on (s,n)
        f = dist.Field(bases=(sbasis, nbasis)); f["g"] = arr; return f

    def f1n(arr):                       # coefficient field on n only
        f = dist.Field(bases=(nbasis,)); f["g"] = arr; return f

    ns = dict(
        us=us, un=un, eta=eta, zc=zc,
        t1=taus[0], t2=taus[1], t3=taus[2], t4=taus[3],
        sig=f2d(sig_a), hb=f1n(Hn + 0 * n), Ub=f1n(Ub + 0 * n), Ubsq=f1n(Ub ** 2 + 0 * n),
        sighb=f2d(sig_a * (Hn + 0 * s)),
        coefUn=f2d(sig_a * (Ubn + 0 * s) + Cb * (Ub + 0 * s)),   # u_n coeff in s-mom
        twoCbUb=f2d(2.0 * Cb * (Ub + 0 * s)),                    # u_s coeff in n-mom
        sig_rs=f2d(sig_a * (2.0 * cfg["Cf"] * Ub / Hn + 0 * s)),         # streamwise drag
        sig_rn=f2d(sig_a * (1.0 * cfg["Cf"] * Ub / Hn + 0 * s)),         # cross-stream drag
        sig_re=f2d(sig_a * (cfg["Cf"] * Ub ** 2 / Hn ** 2 + 0 * s)),     # superelevation drag
        g=g, nu=cfg["nu"], E=E, b=b, ds=ds, dn=dn,
        lap=d3.Laplacian, lift=lambda A, k: d3.Lift(A, nbasis.derivative_basis(2), k),
        dt=d3.TimeDerivative)

    problem = d3.IVP([us, un, eta, zc] + taus, namespace=ns)
    problem.add_equation("sig*dt(eta) + hb*ds(us) + Ub*ds(eta) + dn(sighb*un) = 0")
    problem.add_equation("sig*dt(us) + Ub*ds(us) + coefUn*un + g*ds(eta) + sig_rs*us"
                         " - sig_re*eta - sig*nu*lap(us) + lift(t1,-1) + lift(t2,-2) = 0")
    problem.add_equation("sig*dt(un) + Ub*ds(un) - twoCbUb*us + sig*g*dn(eta) + sig_rn*un"
                         " - sig*nu*lap(un) + Ubsq*ds(ds(zc))"
                         " + lift(t3,-1) + lift(t4,-2) = 0")
    problem.add_equation(f"un(n={b}) = 0")                  # no penetration
    problem.add_equation(f"un(n={-b}) = 0")
    problem.add_equation(f"dn(us)(n={b}) = 0")              # free slip
    problem.add_equation(f"dn(us)(n={-b}) = 0")
    # the meander erodes by the ANTISYMMETRIC near-bank velocity: outer bank fast,
    # inner bank slow.  (Independent banks instead give the channel-WIDENING mode.)
    problem.add_equation(f"dt(zc) - E*0.5*(us(n={b}) - us(n={-b})) = 0")

    return dict(solver=problem.build_solver(d3.RK222), us=us, un=un, eta=eta, zc=zc,
                s=s.ravel(), n=n.ravel(), Ls=Ls, sigma=sig_a, cfg=cfg,
                coords=coords, sigf=ns["sig"], Hbed=bed_depth(n, cfg).ravel())


def run(cfg=CONFIG):
    """Seed one drop of ink upstream, integrate, write outputs/run_<tag>.h5."""
    import h5py
    st = build(cfg)
    s, solver = st["s"], st["solver"]

    # the initial condition IS the perturbation: a single localised bump on the
    # centreline, flow at rest.  Its curvature C' = -d_ss(zc) forces the n-momentum.
    w = cfg["seed_width"] * cfg["b"]
    st["zc"].change_scales(1)
    bump = cfg["A0"] * np.exp(-(s - cfg["seed_s0"] * st["Ls"]) ** 2 / (2 * w ** 2))
    st["zc"]["g"] = bump.reshape(st["zc"]["g"].shape)   # zc lives on (s,) -> (Ns,1)
    for f in (st["us"], st["un"], st["eta"]):
        f.change_scales(1)
        f["g"] = 0.0

    n_steps = int(round(cfg["t_end"] / cfg["dt"]))
    every = max(1, n_steps // (cfg["n_out"] - 1))
    solver.stop_iteration = n_steps + 1
    rec = {k: [] for k in ("t", "us", "un", "eta", "zc")}

    def snap():
        for f in (st["us"], st["un"], st["eta"], st["zc"]):
            f.change_scales(1)
        rec["t"].append(solver.sim_time)
        for k in ("us", "un", "eta"):
            rec[k].append(np.array(st[k]["g"]))
        rec["zc"].append(np.array(st["zc"]["g"]).ravel())

    def leader_efold():
        c = [np.max(np.abs(np.fft.rfft(z)[1:])) for z in (rec["zc"][0], rec["zc"][-1])]
        return float(np.log(max(c[1], 1e-300) / max(c[0], 1e-300)))

    snap()
    for it in range(n_steps):
        solver.step(cfg["dt"])
        if (it + 1) % every == 0:
            snap()
            if leader_efold() >= cfg["max_efold"]:
                print(f"  stopped at t={rec['t'][-1]:.1f}: leading mode reached "
                      f"{leader_efold():.1f} e-foldings (max_efold)")
                break

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"run_{run_tag(cfg)}.h5")
    with h5py.File(path, "w") as h:
        for k, v in rec.items():
            h.create_dataset(k, data=np.array(v))
        for k, v in (("s", st["s"]), ("n", st["n"]), ("Hbed", st["Hbed"]),
                     ("sigma_metric", st["sigma"])):
            h.create_dataset(k, data=v)
        for k, v in cfg.items():
            h.attrs[k] = "None" if v is None else v
        h.attrs["Ls"] = st["Ls"]
        h.attrs["bank_sinuosity"] = bank_sinuosity(cfg) * cfg["b"]   # dimensionless Cbar*b
        h.attrs["Cbar_1_per_m"] = bank_sinuosity(cfg)                # the raw curvature
        h.attrs["Froude"] = froude(cfg)          # DERIVED in SI, not an input any more
        h.attrs["units"] = "SI: m, s, m/s; g=9.81 m/s^2"
        h.attrs["tag"] = run_tag(cfg)
    print(f"wrote outputs/run_{run_tag(cfg)}.h5  ({len(rec['t'])} snapshots)")
    return path


if __name__ == "__main__":
    run(CONFIG)
