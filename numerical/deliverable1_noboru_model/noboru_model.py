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

Eliminating d_t psi1' and d_t psi3' from d_t zeta2' gives three constant-coefficient
linear PDEs in x, which is what Dedalus integrates below.  Substituting a single mode
reproduces the deck's det M = 0 exactly (asserted in postprocessing/03_verify.py).

WHAT IS *NOT* FROM THE DECK  [NOT IN DECK]
------------------------------------------
river.pdf contains no time integration at all -- all 21 pages are normal-mode or
steady-state.  So the initial condition, the periodic domain, n_wave, dt, t_end and
Dedalus itself are this package's contribution, not the deck's.  Also not in the deck:
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
    # 0.5 is CALIBRATED, not cited: it is the value for which c(k*->0) = -E D/gamma
    # reproduces all six phase-speed intercepts of the p.20 figure (see README).
    eps_Cf=0.50,     # eps * C_f   ->   E = eps_Cf * (1 - D)      [p.19 + calibration]
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
    # ===================== INITIAL CONDITION  [NOT IN DECK] ================
    # Kick the centreline, leave both banks straight:  psi2' = cos(k* x), psi1' = psi3' = 0.
    # Nothing is seeded into the banks -- the instability has to build them.  This is
    # also the stronger test: the dynamics must DISCOVER the deck's normal mode rather
    # than be handed it.  "eigen" and "varicose" exist only for 03_verify.py.
    ic="kick",       # "kick" | "eigen" | "varicose" | "asym"
    A0=1.0,          # IC amplitude (the system is linear, so this only sets units)
)

# The two headline runs: identical but for k*.  n_wave differs ONLY so that both reaches
# are ~25 x 2b like the slides (L = 20.9 x 2b and 25.1 x 2b respectively).
#   k* = 0.3 : k*^2 = 0.09 < 2D = 1.0  -> resonant, grows, marches upstream   [p.12 top]
#   k* = 1.5 : k*^2 = 2.25 > 2D = 1.0  -> non-resonant, decays, ~stationary   [p.12 bottom]
RUNS = [
    dict(tag="k0.30", kstar=0.30, n_wave=2),
    dict(tag="k1.50", kstar=1.50, n_wave=12),
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


def mode_index(cfg):
    """Which Fourier mode number the IC excites:  n = k* L/(2pi) = n_wave."""
    return int(round(cfg["kstar"] * domain_length(cfg) / (2.0 * np.pi)))


def dispersion_roots(kstar, D, gamma, E):
    """Both roots omega* of det M = 0 for the 3-level closure.

    [NOT IN DECK -- RECONSTRUCTION]  river.pdf p.19 prints only
    "M(omega)[psihat1'; psihat2'] = 0  =>  det M = 0"; the entries of M and the
    resulting quadratic are never written out.  What follows is derived here from
    p.9 + p.10 + p.19 by eliminating psihat1 between the bank equation and the centre
    vorticity equation, with W = -i omega*:

        centre (p.9 + p.10, using psihat1 = psihat3):
            (W + i k* + gamma) [2 psihat1 - (2 + k*^2) psihat2] + 2 i D k* psihat2 = 0
        bank (p.19):
            (W + E) psihat1 = E psihat2

        =>  (2 + k*^2) W^2 + A1 W + A0 = 0
            A1 = (2 + k*^2)(i k* + gamma + E) - 2 i D k* - 2 E
            A0 = E [k*^2 (i k* + gamma) - 2 i D k*]

    It reproduces the p.20 phase-speed intercepts and growth-rate zero crossings; it
    does NOT reproduce the p.20 peak heights (see postprocessing/04_missing_term.py).
    """
    k = float(kstar)
    A2 = 2.0 + k**2
    A1 = (2.0 + k**2) * (1j * k + gamma + E) - 2j * D * k - 2.0 * E
    A0 = E * (k**2 * (1j * k + gamma) - 2j * D * k)
    return 1j * np.roots([A2, A1, A0])          # omega* = i W


def bank_mode(kstar, D, gamma, E):
    """(omega*, psihat1, psihat2) of the bank-erosion branch, psihat2 normalised to 1.

    The branch is selected as the root with the larger Im omega* (the one the IVP
    converges to).  psihat1 then follows from the p.19 bank equation.
    """
    om = dispersion_roots(kstar, D, gamma, E)
    om = om[np.argmax(om.imag)]
    W = -1j * om
    psi2_hat = 1.0 + 0j
    psi1_hat = E * psi2_hat / (W + E)            # (W + E) psihat1 = E psihat2   [p.19]
    return om, psi1_hat, psi2_hat


# --------------------------------------------------------------------------- #
#  The simulation
# --------------------------------------------------------------------------- #
def simulate(cfg, quiet=False):
    """Integrate the 3-level model and return the raw output as a dict.

    Importable: postprocessing/02_dispersion.py and 04_missing_term.py call this
    directly rather than shelling out, so the driver stays a pure simulator.
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

    # Centre vorticity equation [p.10], with d_t psi1' and d_t psi3' eliminated via the
    # p.19 bank law so that the time-derivative operator acts only on the state:
    #     d_t zeta2' = E(psi2'-psi1') + E(psi2'-psi3') + d_t[d_xx psi2' - 2 psi2'/b^2]
    # then  d_t zeta2' + d_x zeta2' + 2D d_x psi2' + gamma zeta2' = 0.
    problem.add_equation(
        "dt(dx(dx(psi2)) - 2*psi2/b**2)"                      # d_t of the psi2' share
        " + dx(dx(dx(psi2))) + (dx(psi1) + dx(psi3) - 2*dx(psi2))/b**2"   # d_x zeta2'
        " + 2*D*dx(psi2)"                                     # beta term, 2 Delta/b^2
        " + gamma*((psi1 + psi3 - 2*psi2)/b**2 + dx(dx(psi2)))"           # friction
        " + E*(2*psi2 - psi1 - psi3)/b**2"                    # from d_t psi1', d_t psi3'
        " = 0"
    )
    # Erodible banks [p.19].  The deck prints only the psi1' equation; psi3' is the same
    # law at the other bank, which the y -> -y symmetry of the base state requires.
    problem.add_equation("dt(psi1) + E*psi1 - E*psi2 = 0")
    problem.add_equation("dt(psi3) + E*psi3 - E*psi2 = 0")

    solver = problem.build_solver(d3.SBDF2)
    solver.stop_sim_time = cfg["t_end"]

    # ---- initial condition  [NOT IN DECK: river.pdf has no time integration] -------
    x = dist.local_grid(xbasis)
    A0 = cfg["A0"]
    ic = cfg["ic"]
    if ic == "kick":
        # Wiggle the centreline; both banks start perfectly straight.
        psi2["g"] = A0 * np.cos(kstar * x)
        psi1["g"] = np.zeros_like(x)
        psi3["g"] = np.zeros_like(x)
    elif ic == "eigen":
        # The deck's own normal mode at t=0: psihat_j exp(i k* x), sinuous by construction.
        _, p1h, p2h = bank_mode(kstar, D, gamma, E)
        cs, sn = np.cos(kstar * x), np.sin(kstar * x)
        psi1["g"] = A0 * (p1h.real * cs - p1h.imag * sn)
        psi3["g"] = psi1["g"].copy()
        psi2["g"] = A0 * (p2h.real * cs - p2h.imag * sn)
    elif ic == "varicose":
        # psi1' = -psi3', psi2' = 0.  zeta2' vanishes identically, so this excites ONLY
        # the antisymmetric subspace, which must decay at exactly -E for every k*,D,gamma.
        psi1["g"] = A0 * np.cos(kstar * x)
        psi3["g"] = -A0 * np.cos(kstar * x)
        psi2["g"] = np.zeros_like(x)
    elif ic == "asym":
        # Both subspaces at once: the sinuous part grows at sigma, the varicose part
        # decays at -E.  Used to show psihat1 = psihat3 is reached, not assumed.
        psi2["g"] = A0 * np.cos(kstar * x)
        psi1["g"] = 0.5 * A0 * np.cos(kstar * x)
        psi3["g"] = np.zeros_like(x)
    else:
        raise ValueError(f"unknown ic {ic!r}")

    # ---- march ---------------------------------------------------------------------
    n = mode_index(cfg)
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
        b=b, L=L, n_wave=cfg["n_wave"], mode_n=n, dt=cfg["dt"], ic=ic,
    )
    if not quiet:
        om, _, _ = bank_mode(kstar, D, gamma, E)
        print(f"    k*={kstar:<5} D={D} gamma={gamma} E={E:.3f}  L={L:.2f} "
              f"({L / 2:.1f} x 2b, {cfg['n_wave']} waves)  "
              f"analytic sigma={om.imag:+.5f} c={om.real / kstar:+.5f}")
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
