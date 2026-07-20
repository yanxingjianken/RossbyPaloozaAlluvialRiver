#!/usr/bin/env python3
# =============================================================================
#  rigid_lid_meander.py
#  -------------------------------------------------------------------------
#  Linear rigid-lid channel flow over a VARIABLE BED  H(x,y),  with two
#  erodible banks -- a single self-contained Dedalus v3 IVP.
#
#  Edit CONFIG at the top and run:
#     micromamba run -n dedalus env OMP_NUM_THREADS=1 python rigid_lid_meander.py
#
#  Nothing else is imported from this project -- just  dedalus, numpy, matplotlib.
#  Output (written next to this script):
#     rigid_lid_meander_snapshots.npz   raw t / psi' / banks / bed
#     rigid_lid_meander_demo.png        3 growing snapshots + growth curve + bed
#
#  ------------------------------  the physics  ------------------------------
#  Depth-averaged, low-Froude, rigid-lid flow over a bed of depth H(x,y):
#    * mass-transport streamfunction Psi:  u = -(1/H) d_y Psi,  v = (1/H) d_x Psi
#      (the depth-integrated transport H*u is non-divergent under the rigid lid).
#    * potential vorticity  q = zeta / H,   zeta = div( (1/H) grad Psi ).
#    * linearize about a jet ubar(y) over the bed; the base flow follows from
#      DISCHARGE CONSERVATION   u0(x,y) = Hbar(y) ubar(y) / H(x,y)
#      (fast over shallow bars, slow over deep pools; transport Hbar*ubar fixed).
#    * two erodible banks at y = +/-1 obey a relaxation kinematics
#      dt(psib) = E ( Psi_centreline - psib );  a growing bank wiggle = a meander.
#
#  Bed:  H(x,y) = [1 + cross_amp (1 - y^2)] * [1 + along_amp cos(kbed x + phase)]
#      cross_amp>0 -> deeper thalweg  (y-dependence, "H(y)")
#      along_amp>0 -> along-channel bars (x-dependence, "H(x)")
#      both>0 -> full H(x,y);  both=0 -> flat bed.
#
#  Numerics -- why q (not zeta) is the prognostic:  carrying q = zeta/H keeps the
#  elliptic operator polynomial-type,
#        H lap(Psi) - H_x d_x Psi - H_y d_y Psi - H^3 q = 0,
#  instead of a rational 1/H coefficient that Dedalus would turn into a dense
#  matrix.  Because H depends on x the Fourier modes couple => this is genuinely
#  a 2-D IVP (no normal-mode / eigenvalue shortcut).
# =============================================================================
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: write a PNG, no display
import matplotlib.pyplot as plt
import dedalus.public as d3


# ============================= CONFIG (edit me) ============================= #
CONFIG = dict(
    # --- base jet / channel geometry (channel half-width b = 1) ----------- #
    D=0.60,             # jet shape; centre-line speed 1, bank edge speed U0 = 1 - D
    Lx=20 * np.pi,      # channel length in half-widths

    # --- bed  H(x,y) = [1 + cross_amp(1-y^2)] [1 + along_amp cos(kbed x)] -- #
    cross_amp=0.30,     # cross-channel thalweg (deeper mid-channel); 0 = flat in y
    along_amp=0.15,     # along-channel bars; 0 = uniform along x
    along_kbed=0.30,    # along-channel bed wavenumber (= kstar -> bar/bend resonance)
    along_phase=0.0,

    # --- friction / bank erodibility ------------------------------------- #
    gamma=0.05,         # linear bottom friction
    ECOEF=0.50,         # bank-erosion coefficient   (E = ECOEF * (1 - D))

    # --- perturbation + numerics ----------------------------------------- #
    kstar=0.30,         # seeded meander wavenumber
    A0=1e-4,            # initial bank amplitude
    Nx=64, Ny=192,      # streamwise Fourier / cross-channel Chebyshev resolution
    dt=0.02,
    t_end=40.0,         # integration time  (the bank grows ~ e^{sigma t})
    n_snap=80,          # number of snapshots to record
)
# =========================================================================== #


# ----- base-state profiles (all inlined; nothing else imported) ------------ #
def ubar(y, D):
    """Parabolic base jet: 1 at the centre-line, U0 = 1 - D at the banks."""
    return (1.0 - D) + D * (1.0 - y**2)


def hbar(y, cfg):
    """Cross-channel base depth  Hbar(y) = 1 + cross_amp (1 - y^2)."""
    return 1.0 + cfg["cross_amp"] * (1.0 - y**2)


def bed_depth(x, y, cfg):
    """Full bed depth  H(x,y) = Hbar(y) [1 + along_amp cos(kbed x + phase)]."""
    H = hbar(y, cfg)
    if cfg["along_amp"]:
        H = H * (1.0 + cfg["along_amp"]
                 * np.cos(cfg["along_kbed"] * x + cfg["along_phase"]))
    return H


def build(cfg):
    """Assemble the linear H(x,y) PV IVP.  Returns dict(solver, fields, grids)."""
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    E, gamma = cfg["ECOEF"] * (1.0 - cfg["D"]), cfg["gamma"]

    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0.0, cfg["Lx"]))
    ybasis = d3.Chebyshev(coords["y"], size=Ny, bounds=(-1.0, 1.0))
    x, y = dist.local_grids(xbasis, ybasis)          # x:(Nx,1)   y:(1,Ny)
    dx = lambda A: d3.Differentiate(A, coords["x"])
    dy = lambda A: d3.Differentiate(A, coords["y"])

    # unknowns: 2-D perturbation streamfunction Psi' and PV q'; the two bank
    # lines ptop/pbot (y=+/-1); two Chebyshev tau terms for the elliptic BCs.
    psi = dist.Field(name="psi", bases=(xbasis, ybasis))
    q = dist.Field(name="q", bases=(xbasis, ybasis))
    ptop = dist.Field(name="ptop", bases=(xbasis,))
    pbot = dist.Field(name="pbot", bases=(xbasis,))
    tau1 = dist.Field(name="tau1", bases=(xbasis,))
    tau2 = dist.Field(name="tau2", bases=(xbasis,))

    def field2d(name, arr):
        f = dist.Field(name=name, bases=(xbasis, ybasis))
        f["g"] = arr
        return f

    # --- coefficient fields (2-D because the bed varies in x) -------------- #
    Hb, ub = hbar(y, cfg), ubar(y, cfg["D"])         # (1,Ny)
    Harr = bed_depth(x, y, cfg)                       # (Nx,Ny)
    H = field2d("H", Harr)
    Hx = dx(H).evaluate(); Hx.name = "Hx"             # spectral bed gradients
    Hy = dy(H).evaluate(); Hy.name = "Hy"
    H3 = field2d("H3", Harr**3)
    Hbub = field2d("Hbub", np.broadcast_to(Hb * ub, Harr.shape).copy())  # transport
    u0 = field2d("u0", (Hb * ub) / Harr)              # discharge-conserving base flow
    invH = field2d("invH", 1.0 / Harr)
    qbar0 = (-dy(u0) * invH).evaluate(); qbar0.name = "qbar0"   # base PV = -d_y(u0)/H
    qbx = dx(qbar0).evaluate(); qbx.name = "qbx"      # its gradients (spectral NCCs)
    qby = dy(qbar0).evaluate(); qby.name = "qby"

    lift_basis = ybasis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    ns = dict(psi=psi, q=q, ptop=ptop, pbot=pbot, tau1=tau1, tau2=tau2,
              E=E, gamma=gamma, H=H, Hx=Hx, Hy=Hy, H3=H3, Hbub=Hbub,
              qbx=qbx, qby=qby, dx=dx, dy=dy, lap=d3.Laplacian, lift=lift,
              dt=d3.TimeDerivative)

    problem = d3.IVP([psi, q, ptop, pbot, tau1, tau2], namespace=ns)
    # (1) diagnostic elliptic constraint  q = zeta'/H,  multiplied through by H:
    problem.add_equation("H*lap(psi) - Hx*dx(psi) - Hy*dy(psi) - H3*q"
                         " + lift(tau1,-1) + lift(tau2,-2) = 0")
    # (2) linear PV tendency: transport advection + advection of base PV + drag
    problem.add_equation("H*dt(q) + Hbub*dx(q) - qbx*dy(psi) + qby*dx(psi)"
                         " + gamma*q = 0")
    # (3,4) the banks ARE streamlines Psi'(y=+/-1) = ptop/pbot
    problem.add_equation("psi(y=1) - ptop = 0")
    problem.add_equation("psi(y=-1) - pbot = 0")
    # (5,6) erodible-bank relaxation toward the centre-line streamline Psi'(y=0)
    problem.add_equation("dt(ptop) + E*ptop - E*psi(y=0) = 0")
    problem.add_equation("dt(pbot) + E*pbot - E*psi(y=0) = 0")

    solver = problem.build_solver(d3.RK222)
    return dict(solver=solver, psi=psi, q=q, ptop=ptop, pbot=pbot,
                x=x.ravel(), y=y.ravel(), Hbed=Harr, Lx=cfg["Lx"])


def seed(built, cfg):
    """Initial condition: a small sinuous bank wiggle + its harmonic Psi' (q'~=0)."""
    psi, q = built["psi"], built["q"]
    ptop, pbot = built["ptop"], built["pbot"]
    x, y = built["x"], built["y"]
    k, a = cfg["kstar"], cfg["A0"]
    psi["g"][:] = 0.0
    q["g"][:] = 0.0
    tb = a * np.cos(k * x)                                        # bank displacement
    # harmonic extension cosh(k y)/cosh(k) so relative vorticity starts near zero
    psi["g"] += a * np.cos(k * x[:, None]) * np.cosh(k * y[None, :]) / np.cosh(k)
    ptop["g"][:] = tb[:, None]
    pbot["g"][:] = tb[:, None]


def run(cfg):
    """Solve the IVP, record snapshots, and write the .npz + summary figure."""
    built = build(cfg)
    seed(built, cfg)
    solver = built["solver"]
    psi, ptop, pbot = built["psi"], built["ptop"], built["pbot"]

    n_steps = int(round(cfg["t_end"] / cfg["dt"]))
    rec_every = max(1, n_steps // cfg["n_snap"])
    solver.stop_iteration = n_steps + 1
    ts, psis, tops, bots = [], [], [], []

    def record():
        for f in (psi, ptop, pbot):
            f.change_scales(1)
        ts.append(solver.sim_time)
        psis.append(np.array(psi["g"]))
        tops.append(np.array(ptop["g"]).ravel())
        bots.append(np.array(pbot["g"]).ravel())

    record()
    for it in range(n_steps):
        solver.step(cfg["dt"])
        if (it + 1) % rec_every == 0:
            record()
        if (it + 1) % (10 * rec_every) == 0:
            amp = np.max(np.abs(0.5 * (tops[-1] + bots[-1])))
            print(f"  t={solver.sim_time:6.2f}   max|bank|={amp:.3e}")

    ts = np.array(ts)
    psis, tops, bots = np.array(psis), np.array(tops), np.array(bots)
    here = os.path.dirname(os.path.abspath(__file__))
    np.savez(os.path.join(here, "rigid_lid_meander_snapshots.npz"),
             t=ts, psi=psis, top=tops, bot=bots,
             x=built["x"], y=built["y"], Hbed=built["Hbed"])
    _summary_figure(built, cfg, ts, psis, tops, bots,
                    os.path.join(here, "rigid_lid_meander_demo.png"))
    grew = np.max(np.abs(0.5 * (tops[-1] + bots[-1]))) / cfg["A0"]
    print(f"done: {len(ts)} snapshots, bank grew x{grew:.1f}.  "
          f"Wrote rigid_lid_meander_demo.png + _snapshots.npz")


# --------------------------------------------------------------------------- #
#  Minimal visualization: render Psi' INSIDE the meandering channel so the two
#  moving banks are the exact boundary of the flow field (no gap / no overflow).
# --------------------------------------------------------------------------- #
def _warp_fill(ax, x2b, y, field, dtop, dbot, vlim):
    Y = (y[:, None] + (1 + y)[:, None] / 2 * dtop[None, :]
         + (1 - y)[:, None] / 2 * dbot[None, :])
    X = np.broadcast_to(x2b, (len(y), len(x2b)))
    ax.pcolormesh(X, Y, field.T, shading="gouraud", cmap="RdBu_r",
                  vmin=-vlim, vmax=vlim)
    ax.plot(x2b, 1 + dtop, "k", lw=1.4)
    ax.plot(x2b, -1 + dbot, "k", lw=1.4)
    ax.set_xlim(0, x2b[-1]); ax.set_ylim(-2.2, 2.2)


def _summary_figure(built, cfg, ts, psis, tops, bots, path):
    x2b, y = built["x"] / 2.0, built["y"]
    bser = 0.5 * (tops + bots)
    G = 0.5 / max(np.max(np.abs(bser[-1])), 1e-30)   # ONE fixed scale -> true growth
    idx = [0, len(ts) // 2, len(ts) - 1]
    fig, axs = plt.subplots(2, 3, figsize=(14, 6), dpi=110)
    for j, i in enumerate(idx):
        _warp_fill(axs[0, j], x2b, y, G * psis[i], G * tops[i], G * bots[i], 0.6)
        axs[0, j].set_title(f"$\\psi'$ at t={ts[i]:.1f}", fontsize=10)
        axs[0, j].set_xlabel("downstream x/2b")
    # bed H(x,y)
    axb = axs[1, 0]
    pcm = axb.pcolormesh(x2b, y, built["Hbed"].T, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=axb, label="H(x,y)")
    axb.set_title("bed depth H(x,y)", fontsize=10)
    axb.set_xlabel("x/2b"); axb.set_ylabel("y/b")
    # growth curve
    axg = axs[1, 1]
    axg.semilogy(ts, np.max(np.abs(bser), axis=1) / cfg["A0"], lw=2)
    axg.set_xlabel("t"); axg.set_ylabel("bank amplitude / A0")
    axg.set_title(r"meander growth ($\sim e^{\sigma t}$)", fontsize=10)
    axg.grid(True, which="both", alpha=0.3)
    # config readout
    axs[1, 2].axis("off")
    axs[1, 2].text(
        0.02, 0.92,
        f"H(x,y):  cross_amp={cfg['cross_amp']}   along_amp={cfg['along_amp']}\n"
        f"k*={cfg['kstar']}   D={cfg['D']}   gamma={cfg['gamma']}\n"
        f"grid {cfg['Nx']}x{cfg['Ny']}    dt={cfg['dt']}   t_end={cfg['t_end']}",
        va="top", fontsize=11, transform=axs[1, 2].transAxes)
    fig.suptitle("rigid-lid meander over a variable bed H(x,y) "
                 "-- the two banks bound the flow", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    run(CONFIG)
