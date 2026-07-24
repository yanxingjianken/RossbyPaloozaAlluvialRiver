#!/usr/bin/env python3
"""Thetis driver: depth-averaged 2D SW on a meandering channel with migrating banks.

ONE run of the CONFIG at the head of this file.  No CLI.  **Writes data only** --
every figure and movie lives in postprocessing/ (rossby_palooza convention).

Physics and provenance: docs/model.md.  Geometry and the exact base state:
geometry.py.  The corrected channel-following reduction this must reduce to:
sw_note.py.

Structure
---------
The bed is frozen, so there is no Exner equation.  The only prognostic
morphology is the pair of bank curves (y_N, y_S), advanced by the Ikeda law

    d_t y_N = +E u'_N ,   d_t y_S = -E u'_S ,   E = E_erode (u'>0) | E_deposit (u'<0)

Because the banks move, the mesh moves.  The reference topology never changes,
so a bank update is a *coordinate* update: rebuild the mesh from new coordinates,
rebuild the solver, and copy (uv, elev) across by direct DOF copy -- exact,
because DOFs correspond 1:1 between two meshes that share a topology.  The
null-rebuild test in tests/test_setup.py is what protects this.

    micromamba run -n firedrake python meander_thetis.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import geometry as geo  # noqa: E402

from thetis import *  # noqa: E402,F403
from firedrake import COMM_WORLD  # noqa: E402


# =========================================================================== #
#  CONFIG -- every knob for this run
# =========================================================================== #
CONFIG = dict(
    # --- which run -------------------------------------------------------
    n_wave=4,                 # 4 or 8: bank wavenumber over the meander reach

    # --- flow solver -----------------------------------------------------
    # 'steady'        : solve the steady SWE as a Newton BVP each morphological
    #                   step (docs/timescale_review.md, family B).  NO spin-up,
    #                   ~2 Newton iters/step -> reach long morphological time and
    #                   see amplify-vs-decay and up/downstream phase migration.
    # 'cranknicolson' : the original time-marched + morfac path (family A).
    flow_solver="steady",

    # --- physics (see geometry.Config for the rest) ----------------------
    A_ikeda=0.0,              # 0 = Ikeda incised case (user decision)
    E_erode=1.0e-6,           # Ikeda E for u' > 0  (E = eps*C_f)
    E_deposit=1.0e-6,         # Ikeda E for u' < 0

    # --- steady-path morphology -----------------------------------------
    n_morph_iter=200,        # upper bound (early-stops on decay-to-noise or growth)
    target_step_frac=0.004,   # adaptive morphological dt: fastest bank point
                              # moves this fraction of b0 per step
    export_every=2,           # snapshot every N iterations

    # --- cranknicolson-path time (only used if flow_solver='cranknicolson') --
    morph_factor=6000.0,
    dt=2.0,
    t_spinup=4200.0,
    t_morph=21000.0,
    export_dt=273.0,
    morph_every=100,

    # --- guards ----------------------------------------------------------
    min_width_frac=0.35,      # abort if b(s) drops below this fraction of b0
    max_step_frac=0.02,       # max bank displacement per morph step, in units of b0
    theta_max_deg=20.0,       # vertical-cut map validity
)


# =========================================================================== #
#  Mesh construction
# =========================================================================== #
class Reference:
    """The pristine reference rectangle [0, L] x [-1, 1] and its DOF coordinates.

    EVERY warp starts from here.  Warping from the *current* mesh is wrong:
    after one warp, ``coordinates[:, 1]`` is the physical y = c + ntil*b, not
    ntil, so a second warp would treat a physical position as a reference one
    and destroy the mesh -- while the DOF copy of (uv, elev) still "succeeds".
    The null-rebuild gate in tests/test_setup.py exists to catch exactly this.
    """

    def __init__(self, d: geo.Design):
        mesh = RectangleMesh(d.nx, d.ny, d.L, 2.0, quadrilateral=False)
        mesh.coordinates.dat.data[:, 1] -= 1.0        # y: [0,2] -> [-1,1]
        self.mesh = mesh
        self.x = mesh.coordinates.dat.data_ro[:, 0].copy()
        self.ntil = mesh.coordinates.dat.data_ro[:, 1].copy()

    def warp(self, yN: np.ndarray, yS: np.ndarray, x_ref: np.ndarray):
        """A NEW mesh on the same topology with the banks at (yN, yS)."""
        new = self.mesh.coordinates.copy(deepcopy=True)
        c = np.interp(self.x, x_ref, geo.centreline(yN, yS))
        b = np.interp(self.x, x_ref, geo.half_width(yN, yS))
        new.dat.data[:, 0] = self.x
        new.dat.data[:, 1] = c + self.ntil * b
        return Mesh(new)


# =========================================================================== #
#  Base state as Firedrake fields
# =========================================================================== #
def base_fields(mesh, d: geo.Design, yN, yS, x_ref, ref: "Reference"):
    """(bathymetry, u_base_vector, elev_base) on the current mesh.

    bathymetry is Thetis's POSITIVE-DOWNWARD depth below the datum, so
    bathymetry = -z_b  and  h = elev + bathymetry.  Getting this sign wrong
    turns the thalweg into a ridge; tests/test_setup.py asserts it.
    """
    P1 = get_functionspace(mesh, "CG", 1)
    P1v = VectorFunctionSpace(mesh, "CG", 1)

    x = ref.x
    ntil = ref.ntil          # exact, from the reference rectangle

    # secondary-flow bed tilt (Ikeda eq. 6): zero for A=0 (frozen bed), else a
    # curvature-slaved point-bar/pool that follows the current planform.
    A = getattr(d.cfg, "A_ikeda", 0.0)
    if abs(A) > 1e-15:
        cl = geo.centreline(yN, yS)
        kappa = geo.curvature(x_ref, cl)                 # signed curvature [1/m]
        kap_x = np.interp(x, x_ref, kappa)
        dH = geo.secondary_bed_tilt(ntil, kap_x, d, A)   # depth increment [m]
    else:
        dH = 0.0

    bathy = Function(P1, name="bathymetry_2d")
    bathy.dat.data[:] = geo.base_depth(ntil, d) + d.I * x + dH   # = -z_b (+ tilt)

    elev = Function(P1, name="elev_base")
    elev.dat.data[:] = geo.base_elevation(x, d)

    # base flow along the local channel tangent
    cl = geo.centreline(yN, yS)
    dcdx = np.gradient(cl, x_ref, edge_order=2)
    slope = np.interp(x, x_ref, dcdx)
    tnorm = np.sqrt(1.0 + slope**2)
    ub = geo.base_velocity(ntil, d)
    uvec = Function(P1v, name="uv_base")
    uvec.dat.data[:, 0] = ub / tnorm
    uvec.dat.data[:, 1] = ub * slope / tnorm
    return bathy, uvec, elev, ntil


# =========================================================================== #
#  Solver
# =========================================================================== #
def make_solver(mesh, d: geo.Design, yN, yS, x_ref, cfg, ref: "Reference"):
    bathy, uv_base, elev_base, ntil = base_fields(mesh, d, yN, yS, x_ref, ref)
    steady = cfg.get("flow_solver", "steady") == "steady"

    solver = solver2d.FlowSolver2d(mesh, bathy)
    o = solver.options
    o.element_family = "dg-dg"
    o.polynomial_degree = 1
    o.quadratic_drag_coefficient = Constant(d.cfg.Cf)
    o.horizontal_viscosity = Constant(d.cfg.nu)
    o.use_grad_div_viscosity_term = False
    o.use_lax_friedrichs_velocity = True
    o.check_volume_conservation_2d = not steady
    o.simulation_end_time = 1e30              # driven manually
    o.no_exports = True                       # we harvest fields ourselves
    o.timestep = cfg["dt"]
    o.simulation_export_time = cfg.get("export_dt", cfg["dt"])
    if steady:
        # Family B: solve the steady SWE directly (Newton), no time-marching,
        # no spin-up.  Seeded from the previous solution each step, Newton
        # converges in ~2 iterations (docs/timescale_review.md).
        o.swe_timestepper_type = "SteadyState"
        o.swe_timestepper_options.solver_parameters = {
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_max_it": 40,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    else:
        o.swe_timestepper_type = "CrankNicolson"

    # Boundary ids of RectangleMesh: 1 = x-min, 2 = x-max, 3 = y-min, 4 = y-max
    P1 = get_functionspace(mesh, "CG", 1)
    un_in = Function(P1, name="un_inflow")
    un_in.dat.data[:] = -geo.base_velocity(ntil, d)        # inward normal
    solver.bnd_functions["shallow_water"] = {
        1: {"un": un_in},                                  # profile, ONE characteristic
        2: {"elev": Constant(float(geo.base_elevation(d.L, d)))},
        3: {"un": Constant(0.0)},                          # free-slip bank
        4: {"un": Constant(0.0)},                          # free-slip bank
    }
    solver.assign_initial_conditions(uv=uv_base, elev=elev_base)
    return solver, ntil


def carry_state(old_solver, new_solver):
    """Copy (uv, elev) between two solvers whose meshes share a topology.

    Exact by DOF correspondence -- NOT an interpolation.  Only legitimate
    because warp_mesh never changes the topology.
    """
    uo, eo = old_solver.fields.solution_2d.subfunctions
    un, en = new_solver.fields.solution_2d.subfunctions
    un.dat.data[:] = uo.dat.data_ro[:]
    en.dat.data[:] = eo.dat.data_ro[:]


# =========================================================================== #
#  Bank diagnostics and the erosion law
# =========================================================================== #
def near_bank_excess(solver, d: geo.Design, yN, yS, x_ref, frac: float = 0.90):
    """u'_N(s), u'_S(s): near-bank streamwise excess over the reach mean.

    Sampled at |ntil| = frac rather than exactly at the wall: the wall value of
    a DG field is the trace, and the erosion law wants the near-bank flow, which
    is what Ikeda's u'_b = (u')_{n=b} means in a depth-averaged model.
    """
    uv = solver.fields.solution_2d.subfunctions[0]
    cl = geo.centreline(yN, yS)
    b = geo.half_width(yN, yS)
    dcdx = np.gradient(cl, x_ref, edge_order=2)
    tnorm = np.sqrt(1.0 + dcdx**2)
    tx, ty = 1.0 / tnorm, dcdx / tnorm

    out = []
    for sgn in (+1.0, -1.0):
        pts = np.column_stack([x_ref, cl + sgn * frac * b])
        vals = np.array(uv.at(pts, dont_raise=True, tolerance=1e-6))
        vals = np.array([[np.nan, np.nan] if v is None else v for v in vals],
                        dtype=float)
        us = vals[:, 0] * tx + vals[:, 1] * ty          # streamwise component
        out.append(us)
    uN, uS = out
    ref = np.nanmean(np.concatenate([uN, uS]))
    return uN - ref, uS - ref


def advance_banks(yN, yS, uN, uS, dt_morph, d: geo.Design, cfg, x_ref):
    """Ikeda (11)-(13) with independent erosion/deposition coefficients."""
    def rate(up):
        E = np.where(up > 0.0, cfg["E_erode"], cfg["E_deposit"])
        return E * np.nan_to_num(up, nan=0.0)

    cl = geo.centreline(yN, yS)
    gamma = 1.0 / np.sqrt(1.0 + np.gradient(cl, x_ref, edge_order=2) ** 2)  # cos(theta)

    erodible = (x_ref > d.x_m0) & (x_ref < d.x_m1)     # straight reaches are rigid
    taper = geo._taper(x_ref, d)                       # smooth to zero at the ends

    dN = dt_morph * rate(uN) / gamma * erodible * taper
    dS = -dt_morph * rate(uS) / gamma * erodible * taper

    cap = cfg["max_step_frac"] * d.b
    n_clip = int(np.sum(np.abs(dN) > cap) + np.sum(np.abs(dS) > cap))
    dN = np.clip(dN, -cap, cap)
    dS = np.clip(dS, -cap, cap)
    return yN + dN, yS + dS, n_clip


def bank_mode(yN, yS, x_ref, d: geo.Design, m: int):
    """Complex amplitude of the centreline at the fundamental wavenumber.

    Demodulated over the CLEAN interior of the meander reach, so
    |A| = amplitude  (growing -> amplify, shrinking -> decay) and
    arg(A) = phase    (its drift in time IS the up/downstream migration:
                       downstream if the crest x increases with t).
    Returns (A_complex, crest_x) where crest_x is the x of the first bend crest.
    """
    c = geo.centreline(yN, yS)
    pad = 0.2 * d.lam
    sel = (x_ref > d.x_m0 + pad) & (x_ref < d.x_m1 - pad)
    xs, cs = x_ref[sel], c[sel]
    k = 2.0 * np.pi * m / d.L_m
    w = np.exp(-1j * k * (xs - d.x_m0))
    A = 2.0 * np.mean(cs * w)                      # complex Fourier amplitude
    # crest x of the fundamental: c ~ Re[A e^{ik(x-x0)}] = |A|cos(k(x-x0)+phi)
    # first maximum at k(x-x0)+phi = 0 -> x = x0 - phi/k
    phi = np.angle(A)
    crest = d.x_m0 - phi / k
    while crest < d.x_m0:
        crest += 2.0 * np.pi / k
    return A, float(crest)


def check_guards(yN, yS, x_ref, d: geo.Design, cfg):
    b = geo.half_width(yN, yS)
    if np.min(b) < cfg["min_width_frac"] * d.b:
        raise RuntimeError(f"width collapsed: min b = {np.min(b):.3f} m "
                           f"({np.min(b) / d.b:.2f} b0)")
    geo.check_map(x_ref, yN, yS, theta_max_deg=cfg["theta_max_deg"])


# =========================================================================== #
#  Driver
# =========================================================================== #
def _env_overrides(cfg):
    """Override CONFIG from THETIS_* environment variables (robust vs sed-injection).

    Used by run_case.sh / run_limit2.sh so a launch never edits the source.
    """
    spec = {"THETIS_N_WAVE": ("n_wave", int), "THETIS_A_IKEDA": ("A_ikeda", float),
            "THETIS_F_REF": ("F_ref", float), "THETIS_JET_RATIO": ("jet_ratio", float),
            "THETIS_PTS_PER_WL": ("pts_per_wavelength", int),
            "THETIS_CASE": ("case_override", str),
            "THETIS_FLOW_SOLVER": ("flow_solver", str)}
    for env, (key, typ) in spec.items():
        v = os.environ.get(env)
        if v is not None:
            cfg[key] = typ(v)
    return cfg


def main():
    cfg = _env_overrides(dict(CONFIG))
    gcfg = geo.Config(n_wave=cfg["n_wave"], A_ikeda=cfg["A_ikeda"],
                      E_erode=cfg["E_erode"], E_deposit=cfg["E_deposit"],
                      morph_factor=cfg["morph_factor"],
                      F_ref=cfg.get("F_ref", 0.30),
                      jet_ratio=cfg.get("jet_ratio", 0.30),
                      pts_per_wavelength=cfg.get("pts_per_wavelength", 30))
    d = geo.build_design(gcfg)
    tag = f"m{cfg['n_wave']}"
    case = cfg.get("case_override") or geo.case_name(cfg["A_ikeda"])
    out_npz = os.path.join(HERE, "experiments", case, "outputs", f"run_{tag}.npz")
    if COMM_WORLD.rank == 0:
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    x_ref = np.linspace(0.0, d.L, d.nx + 1)
    yN, yS = geo.initial_banks(x_ref, d)
    yN0, yS0 = yN.copy(), yS.copy()

    print_output(f"=== meander_thetis  {tag}  case={case} (A={cfg['A_ikeda']}) ===")
    print_output(f"L={d.L:.1f} m  W={d.W:.2f} m  lambda={d.L_m / cfg['n_wave']:.1f} m "
                 f"({d.L_m / cfg['n_wave'] / d.W:.2f} W)  mesh {d.nx}x{d.ny}")
    print_output(f"transit L/U = {d.L / geo.width_mean(geo.base_velocity, d):.1f} s;"
                 f" spinup {cfg['t_spinup']} s, morph {cfg['t_morph']} s")

    ref = Reference(d)
    mesh = ref.warp(yN, yS, x_ref)
    solver, _ = make_solver(mesh, d, yN, yS, x_ref, cfg, ref)
    m = cfg["n_wave"]
    steady = cfg.get("flow_solver", "steady") == "steady"
    print_output(f"flow solver: {'SteadyState (family B)' if steady else 'CrankNicolson+morfac (A)'}")

    frames = dict(t=[], yN=[], yS=[], phase=[], A=[], crest=[])
    clipped_total = 0
    wall0 = time.time()

    # Output grid, fixed in REFERENCE coordinates for the whole run (MPI-safe:
    # Function.at() is collective and identical on every rank).
    xo = np.linspace(0.0, d.L, 420)
    no = np.linspace(-0.97, 0.97, 41)

    state = {"t": 0.0, "solver": solver, "mesh": mesh}

    def snapshot(phase):
        uvf, elf = state["solver"].fields.solution_2d.subfunctions
        cl, bb = geo.centreline(yN, yS), geo.half_width(yN, yS)
        cx = np.interp(xo, x_ref, cl)
        bx = np.interp(xo, x_ref, bb)
        XX = np.repeat(xo[None, :], no.size, axis=0)
        YY = cx[None, :] + no[:, None] * bx[None, :]
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        uvv = np.array(uvf.at(pts, dont_raise=True, tolerance=1e-6), dtype=object)
        elv = np.array(elf.at(pts, dont_raise=True, tolerance=1e-6), dtype=object)
        uvv = np.array([[np.nan, np.nan] if v is None else v for v in uvv], float)
        elv = np.array([np.nan if v is None else v for v in elv], float)
        A, crest = bank_mode(yN, yS, x_ref, d, m)

        frames["t"].append(state["t"])
        frames["yN"].append(yN.copy())
        frames["yS"].append(yS.copy())
        frames["phase"].append(phase)
        frames["A"].append(A)
        frames["crest"].append(crest)
        if COMM_WORLD.rank == 0:
            np.savez_compressed(
                out_npz.replace(".npz", f"_f{len(frames['t']) - 1:04d}.npz"),
                t=state["t"], phase=phase, x_ref=x_ref, yN=yN, yS=yS,
                xo=xo, no=no, A_re=A.real, A_im=A.imag, crest=crest,
                u=uvv[:, 0].reshape(XX.shape), v=uvv[:, 1].reshape(XX.shape),
                elev=elv.reshape(XX.shape),
            )

    def rebuild():
        check_guards(yN, yS, x_ref, d, cfg)
        new_mesh = ref.warp(yN, yS, x_ref)
        new_solver, _ = make_solver(new_mesh, d, yN, yS, x_ref, cfg, ref)
        carry_state(state["solver"], new_solver)        # seed Newton / carry state
        state["mesh"], state["solver"] = new_mesh, new_solver

    # ----------------------------------------------------------------------- #
    if steady:
        # Family B: each iteration is ONE steady Newton solve + one bank move.
        # No spin-up.  The morphological dt adapts so the fastest bank point
        # moves target_step_frac*b per step (a CFL-like bed limiter).
        A0 = None
        for it in range(cfg["n_morph_iter"] + 1):
            state["solver"].timestepper.advance(state["t"], update_forcings=None)
            uN, uS = near_bank_excess(state["solver"], d, yN, yS, x_ref)
            if it % cfg["export_every"] == 0:
                snapshot("morph")
                A = frames["A"][-1]
                if A0 is None:
                    A0 = A
                gr = np.log(abs(A) / abs(A0)) if abs(A0) > 0 else 0.0
                print_output(f"  it {it:4d}  t={state['t']:.3e}s  |A|={abs(A):.4f} m "
                             f"log(|A|/|A0|)={gr:+.3f}  crest_x={frames['crest'][-1]:.1f} m "
                             f"({time.time() - wall0:.0f}s)")
            if it == cfg["n_morph_iter"]:
                break
            # natural stop: decayed to noise, or grown past the linear range
            A = frames["A"][-1]
            if abs(A) < 0.02 * abs(A0):
                print_output(f"  stop: decayed to <2% of initial amplitude")
                break
            if abs(A) > 4.0 * abs(A0):
                print_output(f"  stop: grown past 4x initial (linear range exceeded)")
                break
            # adaptive morphological timestep
            def rate(up):
                E = np.where(up > 0.0, cfg["E_erode"], cfg["E_deposit"])
                return E * np.nan_to_num(up, nan=0.0)
            cl = geo.centreline(yN, yS)
            gamma = 1.0 / np.sqrt(1.0 + np.gradient(cl, x_ref, edge_order=2) ** 2)
            taper = geo._taper(x_ref, d)
            mag = np.max(np.abs(np.concatenate([rate(uN), rate(uS)]) )) + 1e-30
            dt_morph = cfg["target_step_frac"] * d.b / mag
            yN, yS, nclip = advance_banks(yN, yS, uN, uS, dt_morph, d, cfg, x_ref)
            clipped_total += nclip
            state["t"] += dt_morph
            rebuild()
    else:
        # Family A: the original CrankNicolson spin-up + morfac march.
        dt = cfg["dt"]
        dt_morph_eff = cfg["morph_every"] * dt * cfg["morph_factor"]
        t_end = cfg["t_spinup"] + cfg["t_morph"]
        t_next_export, n_step = 0.0, 0
        snapshot("spinup")
        while state["t"] < t_end - 1e-9:
            state["solver"].timestepper.advance(state["t"], update_forcings=None)
            state["t"] += dt
            n_step += 1
            morphing = state["t"] > cfg["t_spinup"]
            if morphing and n_step % cfg["morph_every"] == 0:
                uN, uS = near_bank_excess(state["solver"], d, yN, yS, x_ref)
                yN, yS, nclip = advance_banks(yN, yS, uN, uS, dt_morph_eff,
                                              d, cfg, x_ref)
                clipped_total += nclip
                rebuild()
            if state["t"] >= t_next_export - 1e-9:
                snapshot("morph" if morphing else "spinup")
                t_next_export += cfg["export_dt"]

    if COMM_WORLD.rank == 0:
        np.savez_compressed(
            out_npz, t=np.array(frames["t"]), yN=np.array(frames["yN"]),
            yS=np.array(frames["yS"]), phase=np.array(frames["phase"]),
            A=np.array(frames["A"]), crest=np.array(frames["crest"]),
            x_ref=x_ref, yN0=yN0, yS0=yS0, k_fundamental=2 * np.pi * m / d.L_m,
            config=np.array([repr(cfg)], dtype=object),
            design=np.array([repr(d)], dtype=object),
            clipped=clipped_total, flow_solver=cfg.get("flow_solver", "steady"),
        )
    print_output(f"wrote {os.path.relpath(out_npz, HERE)}  "
                 f"({len(frames['t'])} frames, {clipped_total} clipped bank steps)")
    if frames["A"]:
        A = frames["A"]
        print_output(f"AMPLITUDE: |A| {abs(A[0]):.4f} -> {abs(A[-1]):.4f} m  "
                     f"({'GROWTH' if abs(A[-1]) > abs(A[0]) else 'DECAY'});  "
                     f"crest {frames['crest'][0]:.1f} -> {frames['crest'][-1]:.1f} m "
                     f"({'DOWNSTREAM' if frames['crest'][-1] > frames['crest'][0] else 'UPSTREAM'})")


if __name__ == "__main__":
    main()
