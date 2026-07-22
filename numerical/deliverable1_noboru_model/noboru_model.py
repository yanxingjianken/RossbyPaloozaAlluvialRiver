#!/usr/bin/env python3
"""The 3-level meander model of river.pdf, integrated forward in time with Dedalus.

    env OMP_NUM_THREADS=1 micromamba run -n dedalus python noboru_model.py

Edit CONFIG/RUNS below and run.  This file ONLY generates simulation output: it writes
raw fields to outputs/run_<tag>.npz and stops.  Every diagnostic (growth rates, momentum
flux, dispersion curves) lives in postprocessing/, so that nothing here can quietly bake
an interpretation into the data.

SOURCE
------
literature/river.pdf -- 21-page deck "Meanders of alluvial rivers as forced Rossby waves"
(Keynote, /Title "NoonBalloon2026", 2026-07-20).  It supersedes the 8-page
Rossby_Palooza_meet_0630.pdf, which is a OneNote capture of the same talk.
Every equation below carries the river.pdf page that prints it.

THE MODEL (deck-printed, dimensional)
-------------------------------------
Base jet, p.9:      ubar(y) = -d(psibar)/dy = U0 + (Delta/b^2)(b^2 - y^2)
Three levels, p.9:  psi_j(x,t) = psibar(y_j) + psihat_j exp[i(kx - omega t)],  y_j = +b, 0, -b

    zeta2' = grad^2 psi' ~= (psihat1 + psihat3 - 2 psihat2)/b^2 - k^2 psihat2      [p.9]

    [d/dt + (U0+Delta) d/dx] zeta2' + (2 Delta/b^2) d(psi2')/dx
                                              = -C_f (U0+Delta)/H * zeta2'        [p.10]

    d(psi1')/dt = (eps C_f U0 / b) (psi2' - psi1')                                [p.19]

NONDIMENSIONALISATION
---------------------
Lengths in b, speeds in U0+Delta, time in b/(U0+Delta).  The time unit is NOT stated in
the deck, but it is not free either: p.10's friction coefficient C_f(U0+Delta)/H becomes
C_f(U0+Delta)T/H, which equals the sidebar's gamma = C_f b/H (pp.12-20) ONLY if
T = b/(U0+Delta).  The deck's own two statements pin it exactly.  Then

    k* = k b,   D = Delta/(U0+Delta),   gamma = C_f b/H          [pp.12-20 sidebar]
    E  = eps C_f U0/(U0+Delta) = eps_Cf (1 - D)                  [p.19, nondimensionalised]

    zeta2'   = (psi1' + psi3' - 2 psi2')/b^2 + d^2(psi2')/dx^2
    d_t zeta2' + d_x zeta2' + 2 D d_x psi2' + gamma zeta2' = 0
    d_t psi1' = E (psi2' - psi1'),   d_t psi3' = E (psi2' - psi3')

These three constant-coefficient linear PDEs in x are what Dedalus integrates below,
written in exactly that form -- zeta2' is a linear combination of the state, so dt(zeta2)
needs no hand-elimination.  Substituting a single mode reproduces the deck's det M = 0
(asserted in postprocessing/03_verify.py, which agrees to seven decimals).

WHAT IS *NOT* FROM THE DECK  [NOT IN DECK]
------------------------------------------
river.pdf contains no time integration at all -- all 21 pages are normal-mode or
steady-state.  So the periodic domain, n_wave, dt, t_end and Dedalus itself are this
package's contribution, not the deck's.  The INITIAL CONDITION is not: the deck's own
forced steady state (pp.12-18, p.11) is the state the flume (pp.17-18) is physically in
before the banks are allowed to erode, and that is where the integration starts.  See
initial_condition().  Also not in the deck:
a numerical value for eps (see eps_Cf below); the operator form d^2/dx^2 in place of
p.9's modal -k^2 (equivalent for one mode, more general here); and the explicit entries
of M(omega) -- p.19 prints only "det M = 0".

Notation table and full provenance: docs/lit_review.md
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

# Length unit.  Kept as a named constant, and never silently dropped, so that the
# 3-point Laplacian below reads exactly as river.pdf p.9 prints it: .../b^2.
# In these units the channel half-width IS the unit, hence b = 1 (see docstring).
b = 1.0


CONFIG = dict(
    # ===================== PHYSICS ========================================
    # The COMPLETE nondimensional parameter set.  b, H, U0, Delta and C_f do NOT
    # appear individually anywhere below -- they enter only through these groups:
    #   b enters ONLY through k* = kb  and  gamma = C_f b/H
    #   H and C_f enter ONLY through gamma = C_f b/H
    #   U0 and Delta enter ONLY through D = Delta/(U0+Delta)
    kstar=0.30,      # k* = k b                                  [pp.12-20 sidebar]
    D=0.50,          # D  = Delta/(U0 + Delta)                   [pp.12-20 sidebar]
    gamma=0.10,      # gamma = C_f b / H                         [pp.12-20 sidebar]
    # eps is the bank erodibility of p.19.  river.pdf NEVER defines it and NEVER gives
    # it a value -- it appears only inside the product eps*C_f*U0/b.  [NOT IN DECK]
    # 0.5 is an ASSUMPTION, not a citation and not a fit to anything.  Both sigma and
    # c(k*->0) = -E D/gamma scale with it, so it sets the RATE of everything while
    # changing none of the structure -- the growth band, the sign of c, the
    # psihat2/psihat1 ratios and the k* of the peak are all independent of it.
    eps_Cf=0.50,     # eps * C_f   ->   E = eps_Cf * (1 - D)      [p.19; value assumed]
    # ===================== NUMERICS =======================================
    # Domain is n_wave whole wavelengths: L = n_wave * 2pi/k*.  n_wave is chosen per
    # run so the reach matches the slides' 25 x 2b x-axis (pp.12-19).
    n_wave=2,
    Nx=256,          # Fourier modes.  The IC excites ONE wavenumber and the system is
                     # linear + constant-coefficient in x, so no other mode is ever
                     # generated; Nx only controls how smooth the plotted curves are.
    dt=0.02,         # every term is linear and treated implicitly, so dt is an
                     # accuracy choice, not a CFL limit (asserted in 03_verify.py)
    t_end=40.0,
    n_out=200,       # snapshots written
    # ===================== INITIAL CONDITION ================================
    # There is exactly ONE physical setup, so there is no choice of KIND here -- see
    # initial_condition().  A0 is the sinuosity the channel is carved with at t=0, in the
    # same units as psi (b(U0+Delta)).  It is a physical input, not a display knob: the
    # movies plot psi in its own units with no gain, so A0 is what sets how wavy the flume
    # starts.  |psibar(b)| = 1 - D/3 = 0.833 at D=0.5 is the natural yardstick.
    A0=0.02,
)

# The two headline runs: identical but for k*.  n_wave differs ONLY so that both reaches
# are ~25 x 2b like the slides (L = 20.9 x 2b and 25.1 x 2b respectively).
#   k* = 0.3 : k*^2 = 0.09 < 2D = 1.0  -> resonant, grows, marches upstream   [p.12 top]
#   k* = 1.5 : k*^2 = 2.25 > 2D = 1.0  -> non-resonant, decays, ~stationary   [p.12 bottom]
# A0 differs per run only because one grows and the other decays: each starts at a
# sinuosity that leaves the whole evolution on-screen at fixed axis limits.  Growth is
# NOT rescaled -- k*=0.3 really does grow ~25x, and you watch it happen.
# Both runs start from the physical initial condition, so both are meanders in the deck's
# sense.  No downstream-travelling case appears, and that is a result rather than an
# omission: c <= 0 on the whole bank branch (README, "Can the meander ever travel
# downstream?").
RUNS = [
    dict(tag="k0.30", kstar=0.30, n_wave=2, A0=0.02),    # grows, UPSTREAM  c=-0.243
    dict(tag="k1.50", kstar=1.50, n_wave=12, A0=0.40),   # decays, stationary c=-0.0001
]


# --------------------------------------------------------------------------- #
#  Derived quantities (pure numpy -- inspectable without touching Dedalus)
# --------------------------------------------------------------------------- #
def bank_E(cfg):
    """E = eps C_f U0/(U0+Delta) = eps_Cf (1-D), the p.19 erosion rate, nondimensional."""
    return cfg["eps_Cf"] * (1.0 - cfg["D"])


def ubar(y, cfg):
    """Base jet in units of U0+Delta:  ubar = (1-D) + D(1 - y^2) = 1 - D y^2.  [p.9]

    Check: ubar(0) = 1 = (U0+Delta)/(U0+Delta); ubar(+-b) = 1-D = U0/(U0+Delta).
    """
    return 1.0 - cfg["D"] * (np.asarray(y, dtype=float) / b) ** 2


def zeta_gradient(cfg):
    """Background vorticity gradient d(zeta_bar)/dy = 2 Delta/b^2 -> 2D nondimensional.

    This is the channel's beta-analogue and the whole reason the deck calls these
    forced Rossby waves.  It is the coefficient of d(psi2')/dx in p.10.
    """
    return 2.0 * cfg["D"]


def domain_length(cfg):
    """L = n_wave * 2pi/k*, so the domain holds a whole number of wavelengths."""
    return cfg["n_wave"] * 2.0 * np.pi / cfg["kstar"]


def fourier_mode_number(cfg):
    """Which Fourier mode index the wavenumber k* occupies in the periodic box.

    Pure bookkeeping, NOT physics: n = k* L/(2pi) = n_wave, and it exists only so the
    postprocessing can read the right slot out of Dedalus's RealFourier coefficient
    array (mode n lives at [2n], [2n+1]).  Contrast initial_condition(), which returns
    the actual starting FIELDS -- the two are unrelated despite both mentioning modes.
    """
    return int(round(cfg["kstar"] * domain_length(cfg) / (2.0 * np.pi)))




# ---------------------------------------------------------------------------- #
#  WHAT IS PHYSICS HERE AND WHAT IS DIAGNOSTIC
#
#  The two functions below ARE needed by the time integration: simulate() calls
#  initial_condition(), which calls forced_ratio() to build psi2' at t=0.
#
#  The dispersion relation (dispersion_roots, bank_mode, bank_branch) is NOT -- the IVP
#  never evaluates it.  It lives in postprocessing/pp_lib.py with the other diagnostics,
#  so that this file contains nothing that computes an answer the simulation is supposed
#  to produce on its own.
# ---------------------------------------------------------------------------- #
def forced_ratio(kstar, D, gamma):
    """psihat2/psihat1 of the FORCED steady state -- the deck's pp.12-18 problem.

    river.pdf pp.12-16 and 18 are titled "Forced steady state" and "Forced-dissipative
    steady state", and p.11 states the closure as psihat2 = f(psihat1): the banks are
    GIVEN and the interior is slaved to them.  p.11 never writes f, so this is a
    reconstruction -- set W = 0 (steady) in the p.10 centre balance, with psihat1 = psihat3:

        (i k* + gamma)[2 psihat1 - (2 + k*^2) psihat2] + 2 i D k* psihat2 = 0
        =>  psihat2/psihat1 = 2(i k* + gamma) / [(2 + k*^2)(i k* + gamma) - 2 i D k*]

    At gamma = 0 this collapses to 2/(2 + k*^2 - 2D), which is exactly the p.14 box:
    |psihat2| > |psihat1| if k*^2 < 2D.  Asserted in postprocessing/03_verify.py.
    """
    k = complex(kstar)
    return 2.0 * (1j * k + gamma) / ((2.0 + k**2) * (1j * k + gamma) - 2j * D * k)


def initial_condition(x, cfg):
    """The one physical initial state: a carved wavy channel, flow in equilibrium with it.

    WHY THIS AND NOTHING ELSE.  river.pdf poses a FORCED problem before it poses an
    unstable one.  pp.12-16 and 18 prescribe a wavy channel and solve for the interior
    response; p.11 makes the slaving explicit as psihat2 = f(psihat1).  The experiment on
    pp.17-18 is the same statement in foam and dye: a RIGID CARVED WAVY CHANNEL with water
    running through it.  The banks are the imposed meander; the interior flow is what
    answers.  Only on p.19 does the deck release the banks and let them erode.

    So the initial-value problem the deck actually sets up is: take the forced-dissipative
    steady state on a given wavy channel, then switch the banks from rigid to erodible and
    watch.  That is

        psi1' = psi3' = A cos(k* x)                     the carved meander (p.9 sinuous)
        psi2' = Re[ f(k*, D, gamma) * A exp(i k* x) ]   the interior response (p.11)

    It is uniquely determined by the deck -- no free choice, hence no option in CONFIG.

    The state is NOT an eigenmode of the p.19 bank-erosion problem, so it evolves from the
    first step: the dynamics still has to FIND the growing mode rather than be handed it.
    Verification check 7 asserts that it satisfies the p.16 steady balance exactly at t=0
    and has left it by the end -- the cheapest test that the problem is posed the way
    river.pdf poses it.
    """
    kstar, A0 = cfg["kstar"], cfg["A0"]
    f = forced_ratio(kstar, cfg["D"], cfg["gamma"])
    cs, sn = np.cos(kstar * x), np.sin(kstar * x)
    bank = A0 * cs                                    # psi1' = psi3', wavy: the channel
    centre = A0 * (f.real * cs - f.imag * sn)         # Re[f A e^{i k x}]: the response
    return bank, centre, bank


# --------------------------------------------------------------------------- #
#  The simulation
# --------------------------------------------------------------------------- #
def simulate(cfg, quiet=False, _test_ic=None):
    """Integrate the 3-level model and return the raw output as a dict.

    Importable: postprocessing/02_dispersion.py calls this directly rather than
    shelling out, so the driver stays a pure simulator.

    _test_ic is a TEST-ONLY hook, used by postprocessing/03_verify.py and nowhere else.
    Two of its checks need deliberately UNPHYSICAL states -- a pure varicose perturbation
    and a lopsided channel -- to measure a symmetry the physical initial condition makes
    identically zero.  Those are properties of the test, not of the model, so they live in
    the test file; CONFIG stays single-valued and there is no menu of physics to pick from.
    """
    import logging

    import dedalus.public as d3

    if quiet:
        logging.getLogger("subsystems").setLevel(logging.ERROR)
        logging.getLogger("solvers").setLevel(logging.ERROR)

    kstar, D, gamma = cfg["kstar"], cfg["D"], cfg["gamma"]
    E = bank_E(cfg)
    L = domain_length(cfg)

    # ---- basis: periodic in x only.  The three levels ARE the y-direction; the deck's
    # ---- model has no y-grid, its cross-channel derivative is the 3-point stencil.
    xcoord = d3.Coordinate("x")
    dist = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.RealFourier(xcoord, size=cfg["Nx"], bounds=(0, L))

    # psi1, psi2, psi3 hold the PERTURBATION psi'_j.  The mean part psibar(y_j) is
    # never evolved: p.9's base state is externally maintained, and both the p.10
    # friction and the p.19 erosion act on the primed part only.
    psi1 = dist.Field(name="psi1", bases=xbasis)
    psi2 = dist.Field(name="psi2", bases=xbasis)
    psi3 = dist.Field(name="psi3", bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)

    # zeta2' = (psi1' + psi3' - 2 psi2')/b^2 + d^2(psi2')/dx^2                    [p.9]
    # p.9 prints the second term modally as -k^2 psihat2; the operator form is used
    # here because we integrate in x rather than assume a single mode.  [NOT IN DECK]
    zeta2 = (psi1 + psi3 - 2 * psi2) / b**2 + dx(dx(psi2))

    # b is a module-level constant, so it must be added explicitly -- locals() alone
    # would leave the /b**2 in the equation strings undefined.
    problem = d3.IVP([psi1, psi2, psi3], namespace=dict(locals(), b=b))

    # The centre vorticity equation, written exactly as river.pdf p.10 prints it.
    #
    #     [d/dt + (U0+Delta) d/dx] zeta2' + (2 Delta/b^2) d(psi2')/dx = -C_f(U0+Delta)/H zeta2'
    #
    # nondimensionalised (advection speed 1, beta coefficient 2D, friction gamma).  No
    # hand-elimination is needed: zeta2' is a LINEAR combination of the state, so Dedalus
    # accepts dt(zeta2) directly and builds the mass matrix itself.  (An earlier version
    # expanded this by hand into six terms, having substituted the p.19 bank law to remove
    # dt(psi1') and dt(psi3').  That was unnecessary -- the two forms agree to 1.4e-15 --
    # and it buried the one equation a reader most wants to recognise.)
    problem.add_equation("dt(zeta2) + dx(zeta2) + 2*D*dx(psi2) + gamma*zeta2 = 0")

    # Erodible banks [p.19].  The deck prints only the psi1' equation; psi3' is the same
    # law at the other bank, which the y -> -y symmetry of the base state requires.
    problem.add_equation("dt(psi1) + E*psi1 - E*psi2 = 0")
    problem.add_equation("dt(psi3) + E*psi3 - E*psi2 = 0")

    solver = problem.build_solver(d3.SBDF2)
    solver.stop_sim_time = cfg["t_end"]

    # ---- initial condition: the carved wavy channel, in equilibrium -----------------
    x = dist.local_grid(xbasis)
    if _test_ic is None:
        g1, g2, g3 = initial_condition(x, cfg)
    else:
        g1, g2, g3 = _test_ic(x, cfg)
    psi1["g"], psi2["g"], psi3["g"] = g1, g2, g3

    # ---- march ---------------------------------------------------------------------
    n = fourier_mode_number(cfg)
    stride = max(1, int(round(cfg["t_end"] / cfg["dt"] / cfg["n_out"])))
    t, g1, g2, g3, a1, a2, a3 = [], [], [], [], [], [], []

    def amp(field):
        """Complex amplitude of Fourier mode n.

        Dedalus RealFourier stores mode n at coefficient indices [2n], [2n+1] with
        f = c[2n] cos(kx) - c[2n+1] sin(kx), i.e. amplitude c[2n] + i c[2n+1].
        Reading [2],[3] instead returns a round-off mode that still fits a perfectly
        clean growth rate -- the WRONG one.  Everything goes through this function.
        """
        c = field["c"]
        return c[2 * n] + 1j * c[2 * n + 1]

    def record():
        t.append(solver.sim_time)
        g1.append(psi1["g"].copy()); g2.append(psi2["g"].copy()); g3.append(psi3["g"].copy())
        a1.append(amp(psi1)); a2.append(amp(psi2)); a3.append(amp(psi3))

    record()
    while solver.proceed:
        solver.step(cfg["dt"])
        if solver.iteration % stride == 0:
            record()

    out = dict(
        x=np.asarray(x), t=np.asarray(t),
        psi1=np.asarray(g1), psi2=np.asarray(g2), psi3=np.asarray(g3),
        amp1=np.asarray(a1), amp2=np.asarray(a2), amp3=np.asarray(a3),
        kstar=kstar, D=D, gamma=gamma, eps_Cf=cfg["eps_Cf"], E=E,
        b=b, L=L, n_wave=cfg["n_wave"], mode_n=n, dt=cfg["dt"],
    )
    if not quiet:
        # configuration only -- no analytic growth rate here.  Printing the answer the
        # simulation is supposed to produce would put a diagnostic in the driver, and it
        # is the postprocessing's job to say whether the run reproduced it.
        print(f"    k*={kstar:<5} D={D} gamma={gamma} E={E:.3f}  L={L:.2f} "
              f"({L / 2:.1f} x 2b, {cfg['n_wave']} waves)  "
              f"A0={cfg['A0']}  t_end={cfg['t_end']}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("noboru_model: the river.pdf 3-level model, integrated (Dedalus)")
    print("-" * 78)
    for run in RUNS:
        cfg = dict(CONFIG); cfg.update(run)
        tag = cfg.pop("tag")
        print(f"  run {tag}:")
        out = simulate(cfg)
        path = os.path.join(OUT_DIR, f"run_{tag}.npz")
        np.savez_compressed(path, **out)
        print(f"    wrote {os.path.relpath(path, HERE)}  "
              f"({out['t'].size} snapshots, {out['x'].size} x-points)")
    print("-" * 78)
    print("done.  Diagnostics live in postprocessing/ -- this file writes data only.")


if __name__ == "__main__":
    main()
