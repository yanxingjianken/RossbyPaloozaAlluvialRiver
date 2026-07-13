#!/usr/bin/env python3
"""Dedalus v3 channel model of the vorticity-meander equations (deck p. 8 goal #1).

This package is the first rung of the group's planned numerical model: the
EXACT equations of ../vorticity_meander/THEORY.md solved as a genuine 2-D
(x, y) initial-value / eigenvalue problem in Dedalus v3.0.5, validated
against the N-point generalized eigenproblem (`channel_modes`) and the 2x2
closure (`bank_branch`) of ../vorticity_meander/vorticity_lib.py -- the
single source of truth for parameters and theory curves (imported here, not
duplicated).

FIDELITY STATEMENT (user requirement: "faithful to equations")
--------------------------------------------------------------
The discretized system is exactly the THEORY.md system and nothing else:
linear dynamics only, the two friction closures (rayleigh / momentum) as the
only friction, NO artificial hyperviscosity (deliberately not implemented --
Orr filamentation of the vorticity continuum is handled by fit-window rules
and resolution, see `t_filament`), no extra boundary terms.

The model
---------
Nondimensional (lengths by b, speeds by U0+Delta, time by b/(U0+Delta)):

  interior   dt(lap(psi)) + ubar(y) dx(lap(psi)) + 2D dx(psi) = F[psi]
             ubar(y) = (1-D) + D (1-y^2),  y in [-1, 1]
             F rayleigh  = -gamma lap(psi)
             F momentum  = -gamma [2 ubar dy(dy(psi)) + 2 ubar_y dy(psi)
                                   + ubar dx(dx(psi))]
  banks      psi(y=+-1) = psib_+-(x, t)          (Dirichlet through bank fields)
             dt(psib_+-) = E (psi(y=0) - psib_+-) (deck p.7, literal: relax
                                                   toward the CENTERLINE value)
  rigid      psi(y=+-1) = 0  (control: must be neutral -- no inflection point)

E = ECOEF[friction] (1-D) with ECOEF = {rayleigh: 0.5, momentum: 1.0}
(vorticity_lib calibration).

Dedalus formulation (all verified by the 2026-07-13 smoke test)
---------------------------------------------------------------
RealFourier(x) x Chebyshev(y in [-1,1]), float64 (IVP) / 1-D Chebyshev,
complex128 (EVP, dt -> -i omega, dx -> i k*). Bank fields psib_+- live on
(xbasis,) and are PROGNOSTIC problem variables -- their evolution equations
carry dt() and the interior interpolation psi(y=0) on the LHS (linear
operators; this idiom builds and steps cleanly in d3.0.5). Two tau fields
lifted to ybasis.derivative_basis(2) at n = -1, -2 close each kx pencil.
Everything sits on the LHS => the IMEX steppers run fully implicit; dt is
accuracy-, not CFL-, limited. NCCs ubar, ubar_y are y-only degree<=2
polynomials (exact NCC expansion). The IVP and the EVP share the SAME
equation strings (namespace rebinding of dt/dx/lap).

kx = 0: closed and regular (BC rows pin the a+by Poisson nullspace given
psib; the constant-offset mode (psi,psib,psib)=(c,c,c) is invariant and is
never excited by zero-mean-in-x seeds; asserted in the self-test).

Initial conditions are analytic and DAE-consistent (house rule: no RNG):
bank sinusoids + their harmonic extension into the interior (zeta'(0) = 0),
   sinuous  : psi = sum_m a_m cos(k_m x + ph_m) cosh(k_m y)/cosh(k_m)
   varicose : psi = sum_m a_m cos(k_m x + ph_m) sinh(k_m y)/sinh(k_m)

Measurement: demodulate the sinuous bank signal (psib_top + psib_bot)/2 by
rFFT in x; sigma = d ln|a_m|/dt, c = -d arg(a_m)/dt / k on a fit window
t <= 0.9 t_filament(k) (Orr rule; t_filament = Ny/(4 D k)).

Run everything with:
    micromamba run -n dedalus env OMP_NUM_THREADS=1 python <script>.py
    python channel_lib.py   # self-test (< ~2 min)
"""
from __future__ import annotations

import os
import sys

import numpy as np

# --------------------------------------------------------------------------- #
#  Paths & theory bridge (single source of truth: vorticity_meander)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(HERE, "..", "vorticity_meander"))
import vorticity_lib as VL                                   # noqa: E402
from vorticity_lib import ECOEF, FRICTIONS, Params           # noqa: E402,F401

RUN_CMD = "micromamba run -n dedalus env OMP_NUM_THREADS=1 python"

# Shared interior equation strings (IVP and EVP rebind dt/dx/lap in namespace)
INTERIOR_EQ = {
    "rayleigh":
        "dt(lap(psi)) + ubar*dx(lap(psi)) + twoD*dx(psi) + gamma*lap(psi)"
        " + lift(tau1,-1) + lift(tau2,-2) = 0",
    "momentum":
        "dt(lap(psi)) + ubar*dx(lap(psi)) + twoD*dx(psi)"
        " + gamma*(2*ubar*dy(dy(psi)) + 2*ubary*dy(psi) + ubar*dx(dx(psi)))"
        " + lift(tau1,-1) + lift(tau2,-2) = 0",
}
BANK_EQS = ("psi(y=1) - psib_top = 0",
            "psi(y=-1) - psib_bot = 0",
            "dt(psib_top) + E*psib_top - E*psi(y=0) = 0",
            "dt(psib_bot) + E*psib_bot - E*psi(y=0) = 0")
RIGID_EQS = ("psi(y=1) = 0", "psi(y=-1) = 0")


def t_filament(kstar, D, Ny):
    """Orr under-resolution time: cross-channel filament scale reaches the
    Chebyshev spacing at t ~ Ny/(4 D k*); fit only for t <= 0.9 * this."""
    return Ny / (4.0 * D * max(kstar, 1e-12))


# --------------------------------------------------------------------------- #
#  IVP builder
# --------------------------------------------------------------------------- #
def build_ivp(p: Params, Lx, Nx, Ny, banks="erodible", timestepper=None):
    """Build the d3 IVP. Returns a dict with solver, fields, grids, ops."""
    import dedalus.public as d3

    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
    ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-1, 1))
    x, y = dist.local_grids(xbasis, ybasis)

    psi = dist.Field(name='psi', bases=(xbasis, ybasis))
    tau1 = dist.Field(name='tau1', bases=(xbasis,))
    tau2 = dist.Field(name='tau2', bases=(xbasis,))
    ubar = dist.Field(name='ubar', bases=(ybasis,))
    ubar['g'] = VL.u_profile(y, p.D)
    ubary = dist.Field(name='ubary', bases=(ybasis,))
    ubary['g'] = VL.u_profile_y(y, p.D)

    dx = lambda A: d3.Differentiate(A, coords['x'])           # noqa: E731
    dy = lambda A: d3.Differentiate(A, coords['y'])           # noqa: E731
    lift_basis = ybasis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)             # noqa: E731

    ns = dict(psi=psi, tau1=tau1, tau2=tau2, ubar=ubar, ubary=ubary,
              twoD=2.0 * p.D, gamma=p.gamma, E=p.E,
              dx=dx, dy=dy, lap=d3.Laplacian, lift=lift,
              dt=d3.TimeDerivative)

    if banks == "erodible":
        psib_top = dist.Field(name='psib_top', bases=(xbasis,))
        psib_bot = dist.Field(name='psib_bot', bases=(xbasis,))
        ns.update(psib_top=psib_top, psib_bot=psib_bot)
        variables = [psi, psib_top, psib_bot, tau1, tau2]
        eqs = (INTERIOR_EQ[p.friction],) + BANK_EQS
    elif banks == "rigid":
        psib_top = psib_bot = None
        variables = [psi, tau1, tau2]
        eqs = (INTERIOR_EQ[p.friction],) + RIGID_EQS
    else:
        raise ValueError(f"unknown banks mode {banks!r}")

    problem = d3.IVP(variables, namespace=ns)
    for eq in eqs:
        problem.add_equation(eq)
    solver = problem.build_solver(timestepper or d3.RK222)

    uv_op = (-dy(psi) * dx(psi))          # u'v' (diagnostic product)
    zeta_op = d3.Laplacian(psi)
    vzeta_op = (dx(psi) * zeta_op)        # v' zeta'
    return dict(solver=solver, psi=psi, psib_top=psib_top, psib_bot=psib_bot,
                x=x, y=y, Lx=Lx, Nx=Nx, Ny=Ny, p=p, banks=banks,
                uv_op=uv_op, zeta_op=zeta_op, vzeta_op=vzeta_op)


def seed_banks(built, modes, kind="sinuous"):
    """Analytic DAE-consistent IC: bank sinusoids + harmonic extension.

    modes: iterable of (kstar, amplitude, phase). kstar must be a multiple of
    2*pi/Lx (asserted). kind: 'sinuous' (psib_top = psib_bot, cosh extension)
    or 'varicose' (psib_top = -psib_bot, sinh extension).
    """
    x, y, Lx = built['x'], built['y'], built['Lx']
    psi, top, bot = built['psi'], built['psib_top'], built['psib_bot']
    dk = 2 * np.pi / Lx
    ptop = np.zeros_like(x)
    pint = np.zeros(np.broadcast_shapes(x.shape, y.shape))
    for k, a, ph in modes:
        m = k / dk
        assert abs(m - round(m)) < 1e-9, f"k*={k} not commensurate with Lx"
        wave = a * np.cos(k * x + ph)
        ptop = ptop + wave
        if kind == "sinuous":
            pint = pint + wave * np.cosh(k * y) / np.cosh(k)
        elif kind == "varicose":
            pint = pint + wave * np.sinh(k * y) / np.sinh(k)
        else:
            raise ValueError(kind)
    if built['banks'] == "erodible":
        top['g'] = ptop
        bot['g'] = ptop if kind == "sinuous" else -ptop
    psi['g'] = pint


def run_ivp(built, dt, t_end, rec_dt=0.25, prof_dt=None, snap_dt=None):
    """Step the IVP, recording bank series (and optional diagnostics).

    Returns dict: t[Nt], top[Nt,Nx], bot[Nt,Nx]; if prof_dt: tprof, uv[.,Ny]
    (x-mean u'v'), vzeta[.,Ny], zeta2[.,Ny] (x-mean zeta'^2); if snap_dt:
    tsnap, psis[.,Nx,Ny].
    """
    solver, psi = built['solver'], built['psi']
    top, bot = built['psib_top'], built['psib_bot']
    Nx = built['Nx']

    def bank_row(f):
        if f is None:
            # rigid: use the psi row nearest y=0 as the tracked signal
            psi.change_scales(1)
            j0 = int(np.argmin(np.abs(built['y'].ravel())))
            return psi['g'][:, j0].copy()
        f.change_scales(1)
        return f['g'][:, 0].copy()

    nstep = int(round(t_end / dt))
    rec_every = max(1, int(round(rec_dt / dt)))
    prof_every = int(round(prof_dt / dt)) if prof_dt else 0
    snap_every = int(round(snap_dt / dt)) if snap_dt else 0

    t, tops, bots = [], [], []
    tprof, uvs, vzs, z2s = [], [], [], []
    tsnap, psis = [], []

    def record():
        t.append(solver.sim_time)
        tops.append(bank_row(top))
        bots.append(bank_row(bot))

    def record_prof():
        uv = built['uv_op'].evaluate(); uv.change_scales(1)
        vz = built['vzeta_op'].evaluate(); vz.change_scales(1)
        ze = built['zeta_op'].evaluate(); ze.change_scales(1)
        tprof.append(solver.sim_time)
        uvs.append(uv['g'].mean(axis=0))
        vzs.append(vz['g'].mean(axis=0))
        z2s.append((ze['g'] ** 2).mean(axis=0))

    def record_snap():
        psi.change_scales(1)
        tsnap.append(solver.sim_time)
        psis.append(psi['g'].copy())

    record()
    if prof_every:
        record_prof()
    if snap_every:
        record_snap()
    for i in range(1, nstep + 1):
        solver.step(dt)
        if i % rec_every == 0:
            record()
        if prof_every and i % prof_every == 0:
            record_prof()
        if snap_every and i % snap_every == 0:
            record_snap()

    out = dict(t=np.array(t), top=np.array(tops), bot=np.array(bots),
               x=built['x'].ravel().copy(), y=built['y'].ravel().copy(),
               Nx=Nx, Lx=built['Lx'])
    if prof_every:
        out.update(tprof=np.array(tprof), uv=np.array(uvs),
                   vzeta=np.array(vzs), zeta2=np.array(z2s))
    if snap_every:
        out.update(tsnap=np.array(tsnap), psis=np.array(psis))
    return out


# --------------------------------------------------------------------------- #
#  EVP builder (1-D in y at prescribed k*; shares the equation strings)
# --------------------------------------------------------------------------- #
def build_evp(kstar, p: Params, Ny, banks="erodible"):
    import dedalus.public as d3

    ycoord = d3.Coordinate('y')
    dist = d3.Distributor(ycoord, dtype=np.complex128)
    ybasis = d3.Chebyshev(ycoord, size=Ny, bounds=(-1, 1))
    y = dist.local_grid(ybasis)

    psi = dist.Field(name='psi', bases=(ybasis,))
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    omega = dist.Field(name='omega')
    ubar = dist.Field(name='ubar', bases=(ybasis,))
    ubar['g'] = VL.u_profile(y, p.D)
    ubary = dist.Field(name='ubary', bases=(ybasis,))
    ubary['g'] = VL.u_profile_y(y, p.D)

    dy = lambda A: d3.Differentiate(A, ycoord)                # noqa: E731
    dtev = lambda A: -1j * omega * A                          # noqa: E731
    dxev = lambda A: 1j * kstar * A                           # noqa: E731
    lapev = lambda A: dy(dy(A)) - kstar**2 * A                # noqa: E731
    lift_basis = ybasis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)             # noqa: E731

    ns = dict(psi=psi, tau1=tau1, tau2=tau2, ubar=ubar, ubary=ubary,
              twoD=2.0 * p.D, gamma=p.gamma, E=p.E,
              dx=dxev, dy=dy, lap=lapev, lift=lift, dt=dtev)

    if banks == "erodible":
        psib_top = dist.Field(name='psib_top')
        psib_bot = dist.Field(name='psib_bot')
        ns.update(psib_top=psib_top, psib_bot=psib_bot)
        variables = [psi, psib_top, psib_bot, tau1, tau2]
        eqs = (INTERIOR_EQ[p.friction],) + BANK_EQS
    else:
        psib_top = psib_bot = None
        variables = [psi, tau1, tau2]
        eqs = (INTERIOR_EQ[p.friction],) + RIGID_EQS

    problem = d3.EVP(variables, eigenvalue=omega, namespace=ns)
    for eq in eqs:
        problem.add_equation(eq)
    solver = problem.build_solver()
    return dict(solver=solver, psi=psi, psib_top=psib_top,
                psib_bot=psib_bot, y=y, Ny=Ny, kstar=kstar, p=p, dy=dy)


def _finite(vals, cap=50.0):
    ok = np.isfinite(vals) & (np.abs(vals) < cap)
    return vals[ok], np.where(ok)[0]


def evp_bank_mode(kstar, p: Params, Ny=128, nev=8, target=None,
                  return_mode=False, check_resolution=False):
    """Eigenvalue omega* of the bank mode from the d3 EVP (sparse, targeted).

    target defaults to the 2x2-closure prediction (bank_branch). The chosen
    eigenvalue is the finite one nearest the target; for erodible banks the
    eigenvector's sinuous symmetry (psib_top ~ psib_bot) is asserted.
    """
    if target is None:
        target = complex(VL.bank_branch([kstar], p.D, p.gamma, p.E,
                                        p.friction)[0])
    built = build_evp(kstar, p, Ny)
    solver = built['solver']
    sp = solver.subproblems[0]
    solver.solve_sparse(sp, N=nev, target=target)
    vals, idx = _finite(solver.eigenvalues)
    assert vals.size, "no finite eigenvalues near target"
    j = idx[int(np.argmin(np.abs(vals - target)))]
    om = complex(solver.eigenvalues[j])

    solver.set_state(j, sp.subsystems[0])
    tb = complex(built['psib_top']['g'].ravel()[0])
    bb = complex(built['psib_bot']['g'].ravel()[0])
    scale = max(abs(tb), abs(bb))
    assert scale > 0 and abs(tb - bb) / scale < 1e-6, \
        f"bank mode not sinuous at k*={kstar} (top={tb}, bot={bb})"

    if check_resolution:
        om2 = evp_bank_mode(kstar, p, Ny=3 * Ny // 2, nev=nev, target=om)
        assert abs(om2 - om) < 1e-6 * max(1.0, abs(om)), \
            f"eigenvalue not converged at Ny={Ny}: {om} vs {om2}"

    if return_mode:
        built['psi'].change_scales(1)
        dpsi = built['dy'](built['psi']).evaluate()
        dpsi.change_scales(1)
        mode = dict(y=built['y'].ravel().copy(),
                    psi=built['psi']['g'].copy(),
                    dpsi=dpsi['g'].copy(),
                    psib=(tb, bb), omega=om, kstar=kstar)
        return om, mode
    return om


def gep_bank_mode(kstar, p: Params, N):
    """Bank-mode eigenvalue from vorticity_lib's N-point FD GEP."""
    target = complex(VL.bank_branch([kstar], p.D, p.gamma, p.E, p.friction)[0])
    w, _ = VL.channel_modes(N, kstar, p.D, p.gamma, p.E, p.friction)
    return complex(w[int(np.argmin(np.abs(w - target)))])


def gep_richardson(kstar, p: Params, Ns=(201, 401)):
    """Richardson-extrapolated (O(h^2) -> O(h^4)) GEP bank eigenvalue."""
    o1 = gep_bank_mode(kstar, p, Ns[0])
    o2 = gep_bank_mode(kstar, p, Ns[1])
    return (4.0 * o2 - o1) / 3.0


# --------------------------------------------------------------------------- #
#  Measurement: demodulation + fits
# --------------------------------------------------------------------------- #
def demodulate(series, m):
    """Complex amplitude a_m(t) of cos-mode m from real series[Nt, Nx]."""
    Nx = series.shape[1]
    return 2.0 / Nx * np.fft.rfft(series, axis=1)[:, m]


def fit_sigma_c(t, a, kstar, window):
    """(sigma, c, r2) from complex amplitude a(t) over window=(t0, t1).

    a ~ e^{sigma t} e^{-i omega_r t}; c = omega_r / k = -(d arg a/dt)/k.
    """
    t = np.asarray(t)
    sel = (t >= window[0]) & (t <= window[1]) & (np.abs(a) > 0)
    assert sel.sum() >= 5, "fit window too short"
    ts, As = t[sel], a[sel]
    la = np.log(np.abs(As))
    sig, b0 = np.polyfit(ts, la, 1)
    resid = la - (sig * ts + b0)
    r2 = 1.0 - resid.var() / max(la.var(), 1e-300)
    ph = np.unwrap(np.angle(As))
    dph, _ = np.polyfit(ts, ph, 1)
    c = -dph / kstar
    return float(sig), float(c), float(r2)


# --------------------------------------------------------------------------- #
#  Plot styling & saving
# --------------------------------------------------------------------------- #
COLORS = {
    "jet": "#2c7fb8",
    "water_fill": "#c7e0f0",
    "bank": "#7f5539",
    "erosion": "#d7301f",
    "psi1": "#08519c",        # bank streamlines (deck blue)
    "psi2": "#d7301f",        # centre streamline (deck red)
    "growth": "#238b45",
    "decay": "#969696",
    "upstream": "#6a51a3",
    "deckpin": "#252525",
    "momentum": "#e6550d",    # momentum-closure overlay
    "dedalus": "#0868ac",     # dedalus measurement dots
}


def warp_fill(ax, x2b, y, psi2d, dtop, dbot, vlim=0.55):
    """Draw a psi' field INSIDE the meandering channel it belongs to.

    Display-only linear warp (clearly a cartoon): the field is solved on the
    fixed domain y in [-1, 1], but is rendered on the deformed mesh
    Y(x, y) = y + (1+y)/2 * dtop(x) + (1-y)/2 * dbot(x) whose top/bottom edges
    are the bank lines y = +1 + dtop, y = -1 + dbot. This makes the coloured
    field fill the wavy channel exactly (no sticking out / no gaps), instead
    of a fixed rectangle overlaid by displaced bank curves.

    psi2d: (Nx, Ny) already amplitude-scaled; dtop/dbot: (Nx,) scaled bank
    displacements (same scale as psi2d).
    """
    Nx, Ny = len(x2b), len(y)
    X2d = np.broadcast_to(x2b, (Ny, Nx))
    wt = (1.0 + y) / 2.0
    wb = (1.0 - y) / 2.0
    Y2d = y[:, None] + wt[:, None] * dtop[None, :] + wb[:, None] * dbot[None, :]
    ax.pcolormesh(X2d, Y2d, psi2d.T, shading="gouraud", cmap="RdBu_r",
                  vmin=-vlim, vmax=vlim, rasterized=True)
    ax.plot(x2b, 1.0 + dtop, color=COLORS["psi1"], lw=2.0)
    ax.plot(x2b, -1.0 + dbot, color=COLORS["psi1"], lw=2.0)


def planform_frames(res, mode_index, kstar, plt, title, t0=0.0):
    """RGB frames of the 2-D planform (psi' contours + erodible banks).

    Each frame is normalized to the instantaneous mode amplitude (fixed
    display size = the linear mode shape; the true e^{sigma t} growth is the
    honest 'gain' counter), and the upstream crest tracker is read from the
    MEASURED demodulated phase, not from theory. `res` must have psis, top,
    bot recorded at the SAME cadence as the snapshots (call run_ivp with
    rec_dt == snap_dt). Animates frames with tsnap >= t0.
    """
    assert len(res['top']) == len(res['psis']), \
        "planform_frames needs rec_dt == snap_dt (aligned bank/psi cadence)"
    a_all = demodulate(0.5 * (res['top'] + res['bot']), mode_index)
    ts = res['tsnap']
    Lx, x2b, y = res['Lx'], res['x'] / 2.0, res['y']
    sel = np.where(ts >= t0)[0]
    gain0 = np.abs(a_all[sel[0]])
    frames = []
    for i in sel:
        amp = np.abs(a_all[i])
        scale = 0.5 / max(amp, 1e-300)
        xc = (-np.angle(a_all[i]) / kstar) % Lx / 2.0
        fig, ax = plt.subplots(figsize=(9.8, 3.8), dpi=110)
        warp_fill(ax, x2b, y, scale * res['psis'][i],
                  scale * res['top'][i], scale * res['bot'][i])
        ax.axvline(xc, color=COLORS['upstream'], lw=2.0, alpha=0.9)
        xa, aw = 0.02 * Lx, 0.09 * (Lx / 2.0)
        ax.annotate("", xy=(xa + aw, -2.2), xytext=(xa, -2.2),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS['jet'], lw=3))
        ax.text(xa, -2.6, "flow", color=COLORS['jet'], fontsize=11)
        ax.text(xa, 1.72, title + rf"   gain $\times{amp / gain0:.2g}$",
                fontsize=10)
        ax.set_ylim(-2.85, 2.35)
        ax.set_xlim(0, Lx / 2.0)
        ax.set_xlabel(r"Downstream distance ($\times 2b$)")
        fig.tight_layout()
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
    return frames


def psibar(y, D):
    """Base-jet streamfunction: -dpsibar/dy = ubar = 1 - D y^2 (nondim)."""
    y = np.asarray(y, dtype=float)
    return -y + D * y**3 / 3.0


def four_panel_frames(res, mode_index, kstar, D, plt, stats, title, t0=0.0):
    """2x2 diagnostic movie frames for one seeded wavelength.

    Panels: (0,0) total streamfunction psi_total = psibar(y) + psi' with
    streamline overlay; (0,1) perturbation psi'; (1,0) momentum flux u'v'
    (u'=-psi'_y, v'=psi'_x); (1,1) stats + a log|a(t)| erosion-growth curve
    that makes the growth explicit (the field panels are amplitude-normalized
    so the mode stays visible; the growth lives in this curve and the gain).

    All field panels are drawn warped into the meandering channel (edges =
    the two bank lines). `stats` = dict(sigma, sigma_evp, c_phase, c_group).
    Requires rec_dt == snap_dt (aligned cadence).
    """
    assert len(res['top']) == len(res['psis'])
    a = demodulate(0.5 * (res['top'] + res['bot']), mode_index)
    ts, Lx, x, y = res['tsnap'], res['Lx'], res['x'], res['y']
    x2b, Nx, Ny = x / 2.0, len(x), len(y)
    sel = np.where(ts >= t0)[0]
    a0 = np.abs(a[sel[0]])
    pb = psibar(y, D)
    logamp = np.log10(np.maximum(np.abs(a), 1e-300) / max(a0, 1e-300))
    wt, wb = (1.0 + y) / 2.0, (1.0 - y) / 2.0
    X2d = np.broadcast_to(x2b, (Ny, Nx))

    def _warp(ax, field2d, dtop, dbot, cmap, vlim, wall_c):
        Y2d = y[:, None] + wt[:, None] * dtop[None, :] + wb[:, None] * dbot[None, :]
        pc = ax.pcolormesh(X2d, Y2d, field2d.T, shading="gouraud", cmap=cmap,
                           vmin=-vlim, vmax=vlim, rasterized=True)
        ax.plot(x2b, 1.0 + dtop, color=wall_c, lw=1.4)
        ax.plot(x2b, -1.0 + dbot, color=wall_c, lw=1.4)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlim(0, Lx / 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        return Y2d, pc

    frames = []
    for i in sel:
        amp = np.abs(a[i])
        scale = 0.5 / max(amp, 1e-300)
        dtop, dbot = scale * res['top'][i], scale * res['bot'][i]
        psi_p = scale * res['psis'][i]
        psi_tot = pb[None, :] + psi_p
        raw = res['psis'][i]
        u = -np.gradient(raw, y, axis=1)
        v = np.gradient(raw, x, axis=0)
        uv = u * v
        mflux = np.max(np.abs(uv)) or 1.0

        # figsize*dpi must be EVEN in both dims (libx264 yuv420p): 1188 x 594
        # figsize*dpi must be EVEN in both dims (libx264 yuv420p): 1296 x 594
        fig, axs = plt.subplots(2, 2, figsize=(12.0, 5.5), dpi=108)

        def _cbar(pc, ax, label):
            cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.02, shrink=0.92)
            cb.set_label(label, fontsize=8)
            cb.ax.tick_params(labelsize=7)

        Y2d, pc0 = _warp(axs[0, 0], psi_tot, dtop, dbot, "RdBu_r", 1.3,
                         COLORS['psi1'])
        axs[0, 0].contour(X2d, Y2d, psi_tot.T, levels=12, colors="k",
                          linewidths=0.35, alpha=0.45)
        axs[0, 0].set_title(r"$\psi_{\rm total}=\bar\psi+\psi'$  (streamlines)",
                            fontsize=10)
        _cbar(pc0, axs[0, 0], r"$\psi_{\rm total}$ (nondim, units $b\,U_c$)")
        _, pc1 = _warp(axs[0, 1], psi_p, dtop, dbot, "RdBu_r", 0.55,
                       COLORS['psi1'])
        axs[0, 1].set_title(r"$\psi'$  (perturbation)", fontsize=10)
        _cbar(pc1, axs[0, 1], r"$\psi'/|\psi'_{\rm bank}|$ (per-frame norm.)")
        _, pc2 = _warp(axs[1, 0], uv / mflux, dtop, dbot, "PuOr_r", 1.0, "k")
        axs[1, 0].set_title(r"momentum flux $\overline{u'v'}$", fontsize=10)
        _cbar(pc2, axs[1, 0], r"$u'v'/\max|u'v'|$ (per-frame norm.)")

        ax = axs[1, 1]
        ax.axis("off")
        cpu = "upstream" if stats['c_phase'] < 0 else "downstream"
        cgu = ("upstream" if stats['c_group'] < -1e-3 else
               "downstream" if stats['c_group'] > 1e-3 else "~0")
        txt = "\n".join([
            rf"$k^*={kstar:g}$    $\lambda/2b={np.pi/kstar:.1f}$",
            rf"$\sigma^*={stats['sigma']:+.3f}$  (EVP {stats['sigma_evp']:+.3f})",
            rf"crests  $c^*={stats['c_phase']:+.3f}$  ({cpu})",
            rf"momentum $c_g={stats['c_group']:+.3f}$  ({cgu})",
        ])
        ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=10,
                linespacing=1.4, transform=ax.transAxes)
        ax.text(0.02, 0.40, rf"gain $e^{{\sigma t}}=\times"
                rf"{amp / max(a0,1e-300):.2g}$", va="top", ha="left",
                fontsize=12, color=COLORS['erosion'], transform=ax.transAxes)
        iax = ax.inset_axes([0.15, 0.04, 0.78, 0.26])
        iax.plot(ts, logamp, color=COLORS['growth'], lw=1.6)
        iax.plot(ts[i], logamp[i], "o", color=COLORS['erosion'], ms=6)
        iax.axhline(0, color="0.7", lw=0.7)
        iax.set_xlabel("t (erosion-growth curve)", fontsize=8)
        iax.set_ylabel(r"$\log_{10}\frac{|a|}{|a_0|}$", fontsize=8)
        iax.tick_params(labelsize=7)
        iax.grid(alpha=0.3)

        fr = stats.get('friction', 'rayleigh')
        gm = stats.get('gamma', float('nan'))
        Ev = stats.get('E', float('nan'))
        footer = (
            rf"chosen (nondim): $D=\Delta/(U_0+\Delta)={D}$,  "
            rf"$\gamma=C_f b/H={gm}$,  $E=\varepsilon C_f(1-D)={Ev:.2g}$ ({fr})"
            "\n"
            rf"units $b=1$, $U_c\equiv U_0+\Delta=1$  $\Rightarrow$  "
            rf"$U_0=1-D={1 - D:.1f}$, $\Delta=D={D}$, "
            rf"$\bar u(y)=1-Dy^2$, $\bar u(\pm b)=U_0={1 - D:.1f}$, "
            rf"$\beta=\bar\zeta_y=2D={2 * D:.1f}$   "
            r"($C_f,H,\varepsilon$ enter only via $\gamma,E$)")
        fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8.2)

        fig.suptitle(title, fontsize=11.5, y=0.995)
        fig.tight_layout(rect=[0, 0.075, 1, 0.94])
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
    return frames


# === shared helper block v1 (keep byte-identical across rossby_palooza packages) ===
def set_style():
    """Apply a consistent matplotlib style (Agg backend, readable fonts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "cm",
    })
    return plt


def save_fig(fig, name, subdir=None, close=True):
    """Save a figure into figures/ (or a subdir) as PNG; return the path."""
    out_dir = FIG_DIR if subdir is None else os.path.join(FIG_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    if not name.lower().endswith(".png"):
        name += ".png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")
    return path


def fig_to_rgb(fig):
    """Rasterise a drawn matplotlib figure to an (H, W, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def write_mp4(frames, name, fps=20):
    """Write a list of RGB frames to figures/<name>.mp4 via imageio+libx264.

    Also drops a representative preview PNG next to the mp4 (middle frame),
    following the chedan_talk/ convention.  Returns the mp4 path.
    """
    import imageio.v2 as imageio
    from PIL import Image

    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    mp4_path = os.path.join(FIG_DIR, name)
    imageio.mimsave(mp4_path, frames, fps=fps, codec="libx264",
                    quality=8, macro_block_size=1)
    prev = frames[len(frames) // 2]
    prev_path = mp4_path[:-4] + "_preview.png"
    Image.fromarray(prev).save(prev_path)
    print(f"  wrote {os.path.relpath(mp4_path, HERE)}  ({len(frames)} frames)")
    print(f"  wrote {os.path.relpath(prev_path, HERE)}")
    return mp4_path
# === end shared helper block ===


# --------------------------------------------------------------------------- #
#  Self-test  (< ~2 min; run: micromamba run -n dedalus env OMP_NUM_THREADS=1
#  python channel_lib.py)
# --------------------------------------------------------------------------- #
def _self_test():
    print("Dedalus channel model of the vorticity-meander equations -- self-test")
    print("-" * 74)

    # 1. Theory bridge.
    assert ECOEF == {"rayleigh": 0.5, "momentum": 1.0}
    assert abs(Params(D=0.6, gamma=0.05).E - 0.2) < 1e-15
    p6 = Params(D=0.6, gamma=0.05, friction="momentum")
    assert abs(p6.E - 0.4) < 1e-15
    print("theory bridge to ../vorticity_meander/vorticity_lib.py. OK")

    # 2. IVP build-and-step smoke, both closures (the bank-ODE idiom).
    for fr in FRICTIONS:
        p = Params(D=0.6, gamma=0.05, friction=fr)
        built = build_ivp(p, Lx=2 * np.pi / 0.5, Nx=8, Ny=32)
        seed_banks(built, [(0.5, 1e-3, 0.0)])
        for _ in range(10):
            built['solver'].step(0.02)
        for f in (built['psi'], built['psib_top'], built['psib_bot']):
            f.change_scales(1)
            assert np.all(np.isfinite(f['g'])), f"{fr}: non-finite fields"
    print("IVP smoke (bank-ODE idiom, both closures): builds + steps finite. OK")

    # 3. EVP vs Richardson-extrapolated FD GEP at one point, both closures.
    for fr in FRICTIONS:
        p = Params(D=0.6, gamma=0.05, friction=fr)
        om_d3 = evp_bank_mode(0.5, p, Ny=96)
        om_fd = gep_richardson(0.5, p)
        err = abs(om_d3 - om_fd)
        print(f"  EVP({fr:>8}) k*=0.5: d3 {om_d3:.6f}  GEP_R {om_fd:.6f}  "
              f"|diff| {err:.2e}")
        assert err < 5e-4, f"{fr}: EVP vs GEP disagree ({err:.2e})"
    print("d3 EVP == vorticity_lib GEP (Richardson 201/401) to 5e-4. OK")

    # 4. Rigid banks neutral (dense spectrum), both closures.
    for fr in FRICTIONS:
        p = Params(D=0.6, gamma=0.05, friction=fr)
        built = build_evp(0.5, p, Ny=48, banks="rigid")
        solver = built['solver']
        solver.solve_dense(solver.subproblems[0])
        vals, _ = _finite(solver.eigenvalues, cap=1e6)
        assert vals.size > 10
        assert np.max(vals.imag) <= 1e-8, \
            f"{fr}: rigid banks must be neutral (max Im = {np.max(vals.imag)})"
    print("rigid banks: whole finite spectrum has Im omega <= 1e-8. OK")

    # 5. Short single-k IVP vs the d3 EVP (growth AND upstream phase speed).
    p = Params(D=0.6, gamma=0.05)
    kst = 0.5
    om_ref = evp_bank_mode(kst, p, Ny=96)
    built = build_ivp(p, Lx=2 * np.pi / kst, Nx=8, Ny=96)
    seed_banks(built, [(kst, 1e-3, 0.0)])
    res = run_ivp(built, dt=0.05, t_end=60.0, rec_dt=0.25)
    sin_ch = 0.5 * (res['top'] + res['bot'])
    a1 = demodulate(sin_ch, 1)
    sig, c, r2 = fit_sigma_c(res['t'], a1, kst, (20.0, 60.0))
    ds = abs(sig - om_ref.imag) / abs(om_ref.imag)
    dc = abs(c - om_ref.real / kst) / abs(om_ref.real / kst)
    print(f"  IVP k*={kst}: sigma {sig:.5f} (EVP {om_ref.imag:.5f}, "
          f"{100*ds:.2f}%)  c {c:.4f} (EVP {om_ref.real/kst:.4f}, "
          f"{100*dc:.2f}%)  R2={r2:.6f}")
    assert ds < 0.03 and dc < 0.03, "IVP fit disagrees with EVP > 3%"
    assert c < 0, "phase speed must be upstream"
    assert r2 > 0.999, "poor exponential fit"

    # 6. Purity: kx=0, other-mode leakage, varicose channel at round-off.
    seed0 = np.max(np.abs(a1[0]))
    var_ch = 0.5 * (res['top'] - res['bot'])
    a_var = np.max(np.abs(demodulate(var_ch, 1)))
    a_dc = np.max(np.abs(np.fft.rfft(sin_ch, axis=1)[:, 0])) / res['Nx']
    a_m2 = np.max(np.abs(demodulate(sin_ch, 2)))
    amax = np.max(np.abs(a1))
    for name, v in (("varicose", a_var), ("kx=0", a_dc), ("m=2", a_m2)):
        assert v < 1e-8 * amax, f"{name} channel not at round-off: {v:.2e}"
    print(f"  purity vs max amp {amax:.2e}: varicose {a_var:.2e}, "
          f"kx=0 {a_dc:.2e}, m=2 {a_m2:.2e}. OK  (seed {seed0:.1e})")

    # 7. Varicose control: decays at exactly omega = -iE, no propagation.
    built = build_ivp(p, Lx=2 * np.pi / kst, Nx=8, Ny=96)
    seed_banks(built, [(kst, 1e-3, 0.0)], kind="varicose")
    resv = run_ivp(built, dt=0.05, t_end=40.0)
    av = demodulate(0.5 * (resv['top'] - resv['bot']), 1)
    sv, cv, _ = fit_sigma_c(resv['t'], av, kst, (10.0, 40.0))
    print(f"  varicose: sigma {sv:.5f} (theory {-p.E:.5f}), c {cv:.2e}")
    assert abs(sv + p.E) < 0.03 * p.E, "varicose decay != -E"
    assert abs(cv) < 5e-3, "varicose must not propagate"
    print("varicose bank mode decays at -E with c=0. OK")

    print("-" * 74)
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
