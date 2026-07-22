#!/usr/bin/env python3
"""Two meandering-channel FUNWAVE-TVD runs that differ ONLY in bank wavenumber.

    micromamba run -n fourcastnetv2 python run_meander.py          # build both cases
    micromamba run -n fourcastnetv2 python run_meander.py --launch # build and mpirun them

Edit CONFIG below and run.  This file ONLY builds cases and (optionally) launches them:
it writes runs/<tag>/{bathy,work} and stops.  Every diagnostic lives in postprocessing/,
so that nothing here can quietly bake an interpretation into the data.

Physics.  Nonlinear depth-averaged shallow water (DISPERSION=F, so Gamma1=Gamma2=0 and
FUNWAVE integrates the NSWE), with a mobile bed (van Rijn pickup + Cao deposition +
Meyer-Peter-Mueller bedload + Exner) and mobile banks (toe erosion + avalanching at
tan phi).  The channel is carved into the bathymetry; banks are topographic, so the
waterline is free and the width is NOT imposed.

THE CONTROLLED VARIABLE IS THE BANK WAVENUMBER, AND ONLY IT.
For a centreline y_c = A cos(k x) the apex curvature is C = A k^2, so holding A fixed
while changing k would change the bend FORCING by k^2 -- the two runs would differ in
drive strength, not just wavelength, and no growth-rate comparison could be attributed.
We therefore hold C = A k^2 FIXED and set A = C0/k^2.  Both cases then share the apex
radius R_min = 1/C0, the metric margin A k^2 b, and the drive; only k differs.

Cross-section.  h(n) = (w0 + beta n^2 / 2)^-2 with w0 = H_c^-1/2, beta = 2(H_b^-1/2 - w0)/b^2.
Under the local normal-flow balance U = sqrt(g h S / Cd) this bed is the one that gives an
EXACTLY constant cross-channel potential-vorticity gradient d(zeta/h)/dn, which is the
mobile-bed analogue of the constant channel-beta of the (s,n) linear model in
../dedalus_meander_full_SW.  See README.md for the derivation.

NOT included: the transverse bed-slope deflection of bedload and the secondary-flow
(helical) correction.  Stock FUNWAVE directs bedload along atan2(v,u) and nothing else,
i.e. this is the A=0 limit of Ikeda et al. (1981).  The standing PREDICTION for these two
runs is therefore outer-bank scour WITHOUT a point bar.  Stating it here, before the runs,
is what makes them informative.
"""
import argparse
import os
import subprocess

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "runs")
EXE = os.path.join(HERE, "work_river",
                   "funwave-SEDIMENT-CHECK_MASS_CONSERVATION-MIXING--mpif90-parallel-double")

G_ACCEL = 9.81            # m/s^2 -- everything in this file is SI

CONFIG = dict(
    # =================== CHANNEL CROSS-SECTION [m] =========================
    b=50.0,              # half-width  -> W = 2b = 100 m
    H_c=3.0,             # depth on the centreline
    H_b=1.5,             # depth at the bank edge n = +/-b (rivers ARE shallow at the margin;
                         #   this is also what sets the PV gradient -- see README)
    m_bank=5.0,          # bank slope 1:m_bank.  At dx=2.5 m this is 6 cells across the
                         #   bank face; 1:3 would be 3.6 cells and the waterline chatters.
    freeboard=1.5,       # floodplain height above the still-water datum [m]
    plain=25.0,          # flat floodplain beyond the bank toe [m]
    # =================== MEANDER ===========================================
    # C0 = A k^2 is held FIXED across runs; A is derived.
    # C0 = 5.0e-3 1/m  <=>  R_min = 200 m = R/W = 2.0, the lower edge of the Leopold-Wolman
    # natural band (R_c/W ~ 2-3).  The first attempt used C0 = 8.496e-3 (R/W = 1.18, tighter
    # than any stable natural meander): the inner bank sits at R-b = 67.7 m, free-vortex
    # scaling accelerates the flow (R+b)/(R-b) = 2.47x, the friction slope rises 6.1x, and
    # the reach needed ~2.8 m of head against the 0.46 m supplied -- it drew down and blew up.
    C0=5.0e-3,           # apex curvature [1/m]  ->  R_min = 200 m = 2.0 W = 4.0 b
    # The reach is a COMMON MULTIPLE of both wavelengths, so both cases share the same
    # down-valley domain, the same nx, and the same inlet/outlet geometry -- only the bend
    # DENSITY differs.  L = 6 * 1047 = 12 * 523.5 = 6282 m.
    lam_ref=1040.0,      # reference wavelength [m].  Chosen so L/dx = 6*1040/2.5 = 2496 is
                         #   a multiple of PAD_X (32) -- see decompose() on why divisibility
                         #   is load-bearing.  Still ~10 W, the Leopold-Wolman scale.
    n_bends_ref=6,       # bends of lam_ref in the reach  ->  L_valley = 6282 m
    # Buffer at each end, made non-erodible because FUNWAVE hard-zeros the sediment flux
    # at the open boundaries.  It is a FIXED PHYSICAL LENGTH, not a bend count: the
    # artefact's scale is the sediment adaptation length (L_a = U H/(gamma w_s) ~ 20 m)
    # plus the flow adjustment, neither of which scales with the bend wavelength.  Counting
    # bends instead would discard 33% of B1's reach but only 17% of B2's.
    buffer_len=1040.0,   # [m] -> interior 4160 m = 4 B1 bends = 8 B2 bends, identical
    # =================== FLOW ==============================================
    U=1.0,               # target reach-mean velocity [m/s]
    # The straight-channel normal slope does NOT include bend losses, so imposing it
    # together with the inlet velocity over-specifies the problem and the reach drains.
    # head_factor scales the BED SLOPE and the boundary head together (so the state stays
    # self-consistent: H = h(n) everywhere at uniform flow) and is CALIBRATED:
    #     micromamba run -n fourcastnetv2 python run_meander.py --calibrate
    # measures the achieved mean speed U_meas and reports  hf_new = hf * (U/U_meas)^2.
    head_factor=1.558,   # [-] multiplier on the straight-channel normal slope.
                         #   CALIBRATED 2026-07-22: iteration 1 at hf=1.0 gave U_mean =
                         #   0.801 (B1) / 0.808 (B2) m/s against a design 1.00, i.e. the
                         #   straight-channel head delivers only 80% of the target speed.
    Cd=0.00154,          # drag coefficient.  NOT free: it is the log-law value that
                         #   FUNWAVE's own sediment module uses internally,
                         #   Cd = kappa^2/[ln(30 H/k_s) - 1]^2 with k_s = 2.5 D50.
                         #   Leaving the example's 0.002 makes flow and transport
                         #   see a 30% different bed.
    # =================== SEDIMENT ==========================================
    D50=5.0e-4,          # median grain size [m] -- medium sand
    Sdensity=2.68,       # rho_s/rho_w
    n_porosity=0.47,
    Shields_cr=0.055,    # suspension threshold
    Shields_cr_bedload=0.047,
    tan_phi=0.7,         # angle of repose -> the bank-collapse (avalanching) threshold
    # =================== TIMESCALES ========================================
    # T_flow ~ 0.2 s, T_c = H/(gamma w_s) ~ 20 s, T_bed ~ 1e7 s.  Morph_factor bridges the
    # last gap; Morph_interval must be >> T_c or the Exner forcing is an unequilibrated
    # suspension transient.  Both are printed by report() so the inflation is never invisible.
    Morph_factor=1000,   # [-] integer
    Morph_interval=200.0,  # [s]  ~10 T_c
    Aval_interval=200.0,   # [s]
    MinDepthPickup=0.01,   # [m]  0.1 (the shipped example) switches OFF bank-toe pickup,
                         #   which is the entire bank-retreat mechanism.  It cannot be 0
                         #   either: the log-law drag is singular at H = e k_s/30 = 1.1e-4 m.
    # =================== NUMERICS ==========================================
    dx=2.5,              # [m] -> 40 cells across the channel width
    spin_transits=1.0,   # phase-1 length, in channel transit times L_channel/U.  Per-run,
                         #   so both cases enter phase 2 equally converged; starting both
                         #   from rest instead would give B1 a 2.5x longer spin-up.
    t_morph=8000.0,      # [s] phase-2 hydrodynamic time, IDENTICAL for both runs.
                         #   t_morph * Morph_factor = 8e6 s = 93 d ~ ONE bar-formation
                         #   timescale T_bed = (1-n_p) H W / q_b, so the run spans roughly
                         #   one bar and no more.  Longer just accumulates MF error.
    CFL=0.5,             # [-]  (0.3 tested: does NOT fix the B2 blow-up)
    MinDepth=0.01,       # [m] wet/dry threshold, also used as MinDepthFrc
    plot_intv=250.0,     # [s] snapshot interval -> 80 frames
    max_ranks=64,
)

# The two cases.  ONLY lambda differs; A is derived from C0 so the drive is identical.
RUNS = [dict(tag="B1", lam=1040.0),
        dict(tag="B2", lam=520.0)]        # exactly lam_ref/2 so the reach is a common multiple


# --------------------------------------------------------------------------- #
#  Derived quantities (pure numpy -- inspectable and testable without FUNWAVE)
# --------------------------------------------------------------------------- #
def amplitude(lam, cfg):
    """A = C0/k^2 -- the whole point of the design."""
    return cfg["C0"] / wavenumber(lam) ** 2


def wavenumber(lam):
    return 2.0 * np.pi / lam


def slope_design(cfg):
    """Straight-channel normal slope: g H S = Cd U^2.  Excludes bend losses."""
    return cfg["Cd"] * cfg["U"] ** 2 / (G_ACCEL * cfg["H_c"])


def slope(cfg):
    """The slope actually built into the bed and the boundary head.  Calibrated."""
    return cfg["head_factor"] * slope_design(cfg)


def taper(x, cfg):
    """Amplitude ramp: 0 at the domain ends, 1 across the analysed interior.

    x = 0 and x = L are CRESTS of y_c = A cos(kx), i.e. points of MAXIMUM curvature, and the
    tide sponge is 30 cells (75 m) wide.  Clamping U = U_design, V = 0 inside the tightest
    part of a bend throttles the inlet: measured Q(x=0) collapsed 279 -> 162 m3/s and the
    reach drained until it blew up.  The sponge occupies 14% of a B2 bend but only 7% of a
    B1 bend, which is why B2 failed at a head B1 tolerated.
    Ramping the amplitude to zero over the (non-analysed) buffer puts the sponge in a
    STRAIGHT channel, where U = U_design, V = 0 is exactly the right state.  Raised cosine,
    so both position and slope are continuous at the join.
    """
    L, w = reach_length(cfg), cfg["buffer_len"]
    d = np.minimum(np.clip(x, 0.0, L), L - np.clip(x, 0.0, L))     # distance to nearer end
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(d / w, 0.0, 1.0)))


def reach_length(cfg):
    """Down-valley domain length -- the SAME for every run, by construction."""
    return cfg["lam_ref"] * cfg["n_bends_ref"]


def fold_margin(lam, cfg):
    """A k^2 b.  Must be < 1 or the inner bank crosses the centreline."""
    return amplitude(lam, cfg) * wavenumber(lam) ** 2 * cfg["b"]


def sinuosity(lam, cfg):
    """Exact arc length per wavelength.  The small-Ak expansion 1 + (Ak)^2/4 is 8% high
    at Ak = 1.4, which is where these runs sit -- so integrate."""
    A, k = amplitude(lam, cfg), wavenumber(lam)
    x = np.linspace(0.0, lam, 20001)
    integrand = np.sqrt(1.0 + (A * k * np.sin(k * x)) ** 2)
    # np.trapz, not np.trapezoid: the fourcastnetv2 env is on numpy 1.x
    return float(np.trapz(integrand, x)) / lam


def settling_velocity(cfg):
    """Soulsby (1997).  The shipped example's 0.0125 m/s is for ~0.1 mm sand, not 0.5 mm."""
    nu = 1.0e-6
    Dstar = cfg["D50"] * ((cfg["Sdensity"] - 1.0) * G_ACCEL / nu ** 2) ** (1.0 / 3.0)
    return (nu / cfg["D50"]) * (np.sqrt(10.36 ** 2 + 1.049 * Dstar ** 3) - 10.36)


def pv_gradient(cfg):
    """d(zeta/h)/dn for the h(n) = (w0 + beta n^2/2)^-2 section under U = sqrt(g h S/Cd).
    Constant by construction -- this returns that constant."""
    w0, wb = cfg["H_c"] ** -0.5, cfg["H_b"] ** -0.5
    beta = 2.0 * (wb - w0) / cfg["b"] ** 2
    return np.sqrt(G_ACCEL * slope(cfg) / cfg["Cd"]) * beta


def section_depth(n, cfg):
    """Still-water depth of the channel section at cross-channel coordinate n [m].

    |n| <= b : the constant-PV-gradient profile.
    |n| >  b : bank rising at 1:m_bank, clipped at the floodplain height.
    """
    b, w0 = cfg["b"], cfg["H_c"] ** -0.5
    beta = 2.0 * (cfg["H_b"] ** -0.5 - w0) / b ** 2
    inside = (w0 + beta * np.minimum(np.abs(n), b) ** 2 / 2.0) ** -2
    bank = cfg["H_b"] - (np.abs(n) - b) / cfg["m_bank"]
    return np.where(np.abs(n) <= b, inside, np.maximum(bank, -cfg["freeboard"]))


def channel_coords(X, Y, lam, cfg, ds_frac=0.25):
    """Signed perpendicular distance n, arc length s, unit tangent (tx, ty) and SIGNED
    curvature kappa of the nearest point on the centreline y_c = A cos(kx).

    Sign conventions, which the whole inner/outer-bank analysis rests on:
      n > 0      is to the LEFT of the tangent.
      kappa > 0  means the curve turns left, so the centre of curvature is to the left.
    Hence  n * sign(kappa) > 0  is the INNER bank and  < 0  the OUTER bank.  Using n
    alone would be wrong: the outer bank swaps sides every half wavelength.

    Nearest-point projection onto a densely sampled centreline.  The first-order
    approximation n ~ (y - y_c) cos(theta) is wrong by O(10%) at these sinuosities,
    and n is what the entire cross-section depends on -- so project properly.
    """
    A, k = amplitude(lam, cfg), wavenumber(lam)
    L = reach_length(cfg)
    # sample beyond both ends so points near x=0 and x=L still find a true nearest point
    xc = np.arange(-lam / 2.0, L + lam / 2.0, cfg["dx"] * ds_frac)
    yc = A * taper(xc, cfg) * np.cos(k * xc)
    seg = np.hypot(np.diff(xc), np.diff(yc))
    sc = np.concatenate([[0.0], np.cumsum(seg)])
    xp, yp = np.gradient(xc), np.gradient(yc)
    tn = np.hypot(xp, yp)
    tx, ty = xp / tn, yp / tn
    # Signed curvature by finite difference on the sampled curve.  The analytic form for
    # A cos(kx) no longer applies once the amplitude is tapered, and FD on a 0.625 m
    # sampling is exact to ~1e-4 of C0 at R = 200 m (checked in tests/test_bathy.py).
    xpp, ypp = np.gradient(xp), np.gradient(yp)
    kap = (xp * ypp - yp * xpp) / (xp ** 2 + yp ** 2) ** 1.5

    idx = cKDTree(np.column_stack([xc, yc])).query(np.column_stack([X.ravel(), Y.ravel()]))[1]
    dx_, dy_ = X.ravel() - xc[idx], Y.ravel() - yc[idx]
    n = (tx[idx] * dy_ - ty[idx] * dx_)          # signed: + is to the left of the tangent
    r = lambda a: a.reshape(X.shape)
    return r(n), r(sc[idx]), r(tx[idx]), r(ty[idx]), r(kap[idx])


def build_case(lam, cfg):
    """Return (Depth, Zs, ini, x, y, meta) for one wavelength.

    `ini` is the analytic normal-flow state (eta, u, v) used as the initial condition:
    starting from rest would spend a whole transit time spinning up, and B1's channel is
    2.5x longer than B2's, so a from-rest start would give the two cases DIFFERENT
    spin-up durations -- a confound in exactly the variable being compared.
    """
    A, k, L = amplitude(lam, cfg), wavenumber(lam), reach_length(cfg)
    half = A + cfg["b"] + cfg["m_bank"] * (cfg["H_b"] + cfg["freeboard"]) + cfg["plain"]
    nx = int(np.ceil(L / cfg["dx"] / PAD_X)) * PAD_X
    ny = int(np.ceil(2.0 * half / cfg["dx"] / PAD_Y)) * PAD_Y
    half = ny * cfg["dx"] / 2.0                    # widen the (dry) floodplain to suit
    x = (np.arange(nx) + 0.5) * cfg["dx"]
    y = -half + (np.arange(ny) + 0.5) * cfg["dx"]
    X, Y = np.meshgrid(x, y, indexing="ij")

    n, s, tx, ty, _ = channel_coords(X, Y, lam, cfg)
    S = slope(cfg)
    h_sec = section_depth(n, cfg)
    # bed drops downstream ALONG THE CHANNEL: using arc length s (not x) makes the
    # sinuosity correction S_valley = sinuosity * S_channel automatic instead of a
    # separate constant that can be forgotten.
    s0 = s.mean()
    Depth = h_sec + S * (s - s0)

    # analytic normal flow: eta = -S(s - s0) so that H = eta + Depth = h_sec exactly,
    # and |u| = sqrt(g h S / Cd) directed along the local centreline tangent.
    eta = -S * (s - s0)
    # IC speed = the STRAIGHT-channel normal flow for the calibrated bed slope,
    # |u| = sqrt(g h S_bed / Cd) = sqrt(head_factor) * U * sqrt(h/H_c).
    # The reach settles BELOW this because bend losses eat the difference, so the transient
    # is a deceleration.  Starting instead at the design U makes it an ACCELERATION, and
    # that transient blew B2 up at t = 963 s (hf = 1.558; also fails at CFL = 0.3).
    speed = np.where(h_sec > 0.0,
                     np.sqrt(G_ACCEL * np.maximum(h_sec, 0.0) * S / cfg["Cd"]), 0.0)
    ini = dict(eta=eta, u=speed * tx, v=speed * ty)

    # Buffer over the first and last bend, needed because FUNWAVE's FLUX_SCALAR_BC
    # hard-zeros the sediment flux on all four boundaries, which puts an artificial
    # scour/deposition zone at the inlet and outlet.
    # Zs is the MAX ERODIBLE THICKNESS and is applied as `IF(Zb>Zs) Zb=Zs` with Zb>0 =
    # erosion -- so Zs=0 blocks EROSION ONLY.  Deposition (Zb<0) is NOT capped and the
    # buffer does accrete (measured: 0 eroding cells, 120 depositing cells, -0.27 m).
    # That is acceptable -- the point is to stop the boundary artefact propagating into
    # the interior -- but the buffer is NOT a frozen bed and must not be analysed.
    Zs = np.where((X < cfg["buffer_len"]) | (X > L - cfg["buffer_len"]), 0.0, 1.0e6)

    sinu = sinuosity(lam, cfg)
    meta = dict(lam=lam, k=k, A=A, L=L, nx=nx, ny=ny, half=half, S=S,
                sinuosity=sinu, fold=fold_margin(lam, cfg), n_bends=L / lam,
                R_min=1.0 / cfg["C0"], ws=settling_velocity(cfg),
                dqdn=pv_gradient(cfg), buffer_len=cfg["buffer_len"],
                L_channel=L * sinu, transit=L * sinu / cfg["U"])
    return Depth, Zs, ini, x, y, meta


# --------------------------------------------------------------------------- #
#  Write
# --------------------------------------------------------------------------- #
def run_tag(lam, cfg):
    """Name a run by its PHYSICS.  A is derived from C0, so lambda and C0 identify it."""
    return (f"lam{lam:.1f}_C{cfg['C0']*1e3:.2f}e-3_D50{cfg['D50']*1e6:.0f}um"
            f"_Cf{cfg['Cd']:.5f}_MF{cfg['Morph_factor']}").replace(".", "p")


INPUT_TEMPLATE = """\
!INPUT FILE FOR FUNWAVE_TVD -- generated by run_meander.py, do not hand-edit
!  phase = {phase}
TITLE = meander_{tag}_{phase}
PX = {px}
PY = {py}
DEPTH_TYPE = DATA
DEPTH_FILE = ../bathy/depth.txt
RESULT_FOLDER = output/
Mglob = {nx}
Nglob = {ny}
TOTAL_TIME = {total_time}
PLOT_INTV = {plot_intv}
SCREEN_INTV = {plot_intv}
DX = {dx}
DY = {dx}
  ! ---------------- INITIAL CONDITION ----------------
INI_UVZ = T
ETA_FILE = {ini_dir}/eta.txt
U_FILE = {ini_dir}/u.txt
V_FILE = {ini_dir}/v.txt
  ! ---------------- STEADY DISCHARGE IN / STAGE OUT ----------------
  ! TIDE_BC nudges ETA, U *and* V toward these values inside a 30-cell sponge at each end
  ! (mod_tide.F:323-330).  Every component MUST be given at BOTH ends: TideEast_U defaults
  ! to ZERO (mod_tide.F:454), which walls the outflow while the ETA nudge keeps deleting the
  ! water piling up behind it -- a sink that drains the reach, steepens the head gradient from
  ! the pinned inlet, accelerates the flow, and blows up.  Measured on the first attempt:
  ! reach-mean eta -0.2 -> -1.16 m in 1250 s, MaxTotalU 1.0 -> 2.4 m/s, BlowUp at t = 3420 s.
  ! The channel is aligned with x at both ends (y_c' = 0 at x = 0 and x = L = 6 lam), so the
  ! self-consistent normal-flow state is U = U_design, V = 0 at both.
WAVEMAKER = NONE
TIDAL_BC_ABS = T
TideBcType = CONSTANT
TideWest_ETA = {eta_west}
TideWest_U = {U}
TideWest_V = 0.0
TideEast_ETA = {eta_east}
TideEast_U = {U}
TideEast_V = 0.0
PERIODIC = F
  ! ---------------- PHYSICS ----------------
DISPERSION = F
Cd = {Cd}
C_smg = 0.25
nu_bkg = 0.0
  ! ---------------- NUMERICS ----------------
CFL = {CFL}
FroudeCap = 1.0
MinDepth = {MinDepth}
MinDepthFrc = {MinDepth}
VISCOSITY_BREAKING = T
  ! ---------------- OUTPUT ----------------
NumberStations = 0
DEPTH_OUT = T
U = T
V = T
ETA = T
MASK = T
  ! ---------------- SEDIMENT ----------------
Bed_Change = {bed_change}
BedLoad = {bed_change}
Avalanche = {bed_change}
D50 = {D50}
Sdensity = {Sdensity}
n_porosity = {n_porosity}
WS = {ws}
Shields_cr = {Shields_cr}
Shields_cr_bedload = {Shields_cr_bedload}
Tan_phi = {tan_phi}
Kappa1 = 0.3333
Kappa2 = 1.0
MinDepthPickup = {MinDepthPickup}
Morph_factor = {Morph_factor}
Morph_interval = {Morph_interval}
Aval_interval = {Aval_interval}
Hard_bottom = T
Hard_bottom_file = ../bathy/hard.txt
PLOT_INTV_SEDIMENT = {plot_intv}
"""


PAD_X, PAD_Y = 32, 8      # nx, ny are padded to multiples of these


def decompose(meta, cfg):
    """MPI decomposition: ~1e4 cells per rank, powers of two, and px | nx, py | ny.

    Divisibility is NOT cosmetic.  A case with Nglob = 93 on py = 2 ran to "Normal
    Termination" with the water in place (PhaseS = 5.0) but the INI_UVZ velocity field
    silently dropped to ZERO everywhere -- no error, no warning, MaxTotalU = 0.  nx and ny
    are padded in build_case so the powers of two below always divide them."""
    px, py = 1, 1
    while (meta["nx"] * meta["ny"] / (px * py) > 1.0e4 and px * py < cfg["max_ranks"]
           and (2 * px <= PAD_X or 2 * py <= PAD_Y)):
        if meta["nx"] / px >= 2 * meta["ny"] / py and 2 * px <= PAD_X:
            px *= 2
        elif 2 * py <= PAD_Y:
            py *= 2
        else:
            break
    assert meta["nx"] % px == 0 and meta["ny"] % py == 0, \
        f"decomposition {px}x{py} does not divide grid {meta['nx']}x{meta['ny']}"
    return px, py


def phase_input(tag, phase, meta, cfg, ini_dir, total_time, bed_change):
    px, py = decompose(meta, cfg)
    drop = meta["S"] * meta["L_channel"]
    kw = dict(cfg)
    kw.update(tag=tag, phase=phase, nx=meta["nx"], ny=meta["ny"], ws=meta["ws"],
              px=px, py=py, ini_dir=ini_dir, total_time=total_time,
              bed_change="T" if bed_change else "F",
              eta_west=+drop / 2.0, eta_east=-drop / 2.0)
    return INPUT_TEMPLATE.format(**kw), px * py


def write_case(lam, cfg):
    Depth, Zs, ini, x, y, meta = build_case(lam, cfg)
    tag = run_tag(lam, cfg)
    base = os.path.join(RUN_DIR, tag)
    for sub in ("bathy", "ini", "spinup/output", "morph/output"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    # FUNWAVE reads  DO J=Jbeg,Jend / READ(1,*)(Var(I,J),I=Ibeg,Iend)
    # i.e. Nglob rows of Mglob values -> transpose our (nx, ny) arrays.
    np.savetxt(os.path.join(base, "bathy", "depth.txt"), Depth.T, fmt="%14.7e")
    np.savetxt(os.path.join(base, "bathy", "hard.txt"), Zs.T, fmt="%14.7e")
    for k, v in ini.items():
        np.savetxt(os.path.join(base, "ini", f"{k}.txt"), v.T, fmt="%14.7e")

    # phase 1: rigid bed, from the analytic state, one transit time -> true steady flow
    t_spin = cfg["spin_transits"] * meta["transit"]
    txt, nr = phase_input(tag, "spinup", meta, cfg, "../ini", t_spin, False)
    open(os.path.join(base, "spinup", "input.txt"), "w").write(txt)
    # phase 2: mobile bed, hot-started from phase 1.  ini_dir is patched at launch time
    # to the LAST spinup snapshot -- the index depends on TOTAL_TIME/PLOT_INTV rounding,
    # so it is globbed rather than computed.
    txt, _ = phase_input(tag, "morph", meta, cfg, "@INI@", cfg["t_morph"], True)
    open(os.path.join(base, "morph", "input.txt"), "w").write(txt)

    np.savez(os.path.join(base, "bathy", "grid.npz"), x=x, y=y, Depth=Depth, Zs=Zs,
             t_spin=t_spin, nranks=nr, **{k: v for k, v in meta.items()})
    return tag, meta, t_spin, nr


def report(runs, cfg):
    print(f"  Cd = {cfg['Cd']:.5f}   w_s = {settling_velocity(cfg):.4f} m/s")
    print(f"  S_design (straight) = {slope_design(cfg):.3e} ; head_factor = "
          f"{cfg['head_factor']:.3f}  ->  S_bed = {slope(cfg):.3e}")
    Rmin = 1.0 / cfg["C0"]
    print(f"  R_min = {Rmin:.1f} m = {Rmin/(2*cfg['b']):.2f} W  (natural band R/W ~ 2-3); "
          f"free-vortex U_max/U = {(Rmin+cfg['b'])/(Rmin-cfg['b']):.2f}")
    print(f"  C0 = A k^2 = {cfg['C0']:.3e} 1/m   (FIXED across runs -- same drive)")
    print(f"  d(zeta/h)/dn = {pv_gradient(cfg):.3e} 1/(m^2 s)  (constant by construction)")
    print(f"  t_morph = {cfg['t_morph']:.0f} s hydrodynamic = "
          f"{cfg['t_morph']*cfg['Morph_factor']/86400:.0f} days morphological "
          f"(Morph_factor = {cfg['Morph_factor']}), SAME for both runs")
    print(f"  reach L = {reach_length(cfg):.0f} m down-valley for BOTH runs "
          f"(= {cfg['n_bends_ref']} x {cfg['lam_ref']:.1f} m); buffer {cfg['buffer_len']:.0f} m "
          f"each end -> interior {reach_length(cfg)-2*cfg['buffer_len']:.0f} m")
    print(f"\n{'tag':<40} {'lam':>7} {'bends':>6} {'A':>7} {'A/W':>5} {'sinu':>6} {'Ak2b':>6} "
          f"{'nx':>5} {'ny':>4} {'cells':>9} {'ranks':>5} {'L_chan':>7} {'t_spin':>7}")
    for tag, m, t_spin, nr in runs:
        flag = "  <-- FOLDS" if m["fold"] >= 1.0 else ""
        bad = "" if abs(m["n_bends"] - round(m["n_bends"])) < 1e-9 else "  <-- NOT A WHOLE NUMBER OF BENDS"
        print(f"{tag:<40} {m['lam']:7.1f} {m['n_bends']:6.2f} {m['A']:7.1f} "
              f"{m['A']/(2*cfg['b']):5.2f} {m['sinuosity']:6.3f} {m['fold']:6.3f} "
              f"{m['nx']:5d} {m['ny']:4d} {m['nx']*m['ny']:9d} {nr:5d} "
              f"{m['L_channel']:7.0f} {t_spin:7.0f}{flag}{bad}")


def health(base, phase, cfg):
    """Is the run physically alive?  'Normal Termination' is NOT a success criterion.

    FUNWAVE only flags a blow-up above EtaBlowVal = 100 x max_depth (~300 m here), so a reach
    that drains completely -- eta -> -73 m, every cell dry, u identically 0 -- still prints
    'Normal Termination'.  That false positive was believed twice in this project before it
    was caught.  Check the fields instead.
    """
    import glob
    out = os.path.join(base, phase, "output")
    if glob.glob(os.path.join(out, "*_99999")):
        return False, "blow-up dump (*_99999) present"
    su = [p for p in sorted(glob.glob(os.path.join(out, "eta_*"))) if "99999" not in p]
    if not su:
        return False, "no eta snapshots"
    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    eta = np.loadtxt(su[-1]).T
    H = eta + g["Depth"]
    X, Y = np.meshgrid(g["x"], g["y"], indexing="ij")
    n = channel_coords(X, Y, float(g["lam"]), cfg)[0]
    chan = np.abs(n) <= cfg["b"]
    wet = float((H[chan] > cfg["MinDepth"]).mean())
    if wet < 0.90:
        return False, f"channel only {100*wet:.0f}% wet (drained); eta min {eta.min():.1f} m"
    if abs(eta[chan]).max() > 5.0:
        return False, f"|eta| in channel reached {abs(eta[chan]).max():.1f} m"
    return True, f"channel {100*wet:.0f}% wet, |eta|max {abs(eta[chan]).max():.2f} m"


def launch(base, phase, nranks, cfg):
    work = os.path.join(base, phase)
    with open(os.path.join(work, "run.log"), "w") as log:
        subprocess.run(["mpirun", "--oversubscribe", "-np", str(nranks),
                        "--mca", "btl_vader_single_copy_mechanism", "none", EXE],
                       cwd=work, stdout=log, stderr=subprocess.STDOUT, check=False)
    terminated = "Normal Termination" in open(os.path.join(work, "run.log")).read()
    ok, why = health(base, phase, cfg)
    ok = ok and terminated
    print(f"    {phase}: {'OK' if ok else 'FAILED'} -- {why}"
          + ("" if terminated else " (no Normal Termination)"))
    return ok


def measure_speed(base, cfg):
    """Mean speed in the erodible channel from the last spin-up snapshot, and the implied
    head_factor.  U ~ sqrt(S), so a reach running slow needs its slope raised by (U/U_meas)^2."""
    import glob
    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    su = [p for p in sorted(glob.glob(os.path.join(base, "spinup", "output", "u_*")))
          if "99999" not in p]
    if not su:
        return None
    ld = lambda p: np.loadtxt(p).T
    u, v = ld(su[-1]), ld(su[-1].replace("/u_", "/v_"))
    X, Y = np.meshgrid(g["x"], g["y"], indexing="ij")
    n = channel_coords(X, Y, float(g["lam"]), cfg)[0]
    chan = (np.abs(n) <= cfg["b"]) & (g["Zs"] > 0)
    sp = np.hypot(u, v)
    U_meas = float(sp[chan].mean())
    return dict(snapshot=os.path.basename(su[-1]), U_meas=U_meas, U_max=float(sp[chan].max()),
                hf_new=cfg["head_factor"] * (cfg["U"] / max(U_meas, 1e-6)) ** 2)


def calibrate(cfg, seconds):
    """Short rigid-bed spin-ups to find the head the SINUOUS reach actually needs.

    The straight-channel normal slope omits bend losses; imposing it together with the inlet
    velocity over-specifies the problem and the reach drains until it blows up.  U ~ sqrt(S),
    so one measurement gives a good next guess and two iterations converge."""
    import copy
    print(f"\n=== CALIBRATION: ~{seconds:.0f} s rigid-bed spin-up per case, "
          f"head_factor = {cfg['head_factor']:.3f} ===")
    out = []
    for r in RUNS:
        c = copy.deepcopy(cfg)
        _, meta, _, _ = write_case(r["lam"], c)              # get transit time
        c["spin_transits"] = seconds / meta["transit"]
        tag, meta, t_spin, nr = write_case(r["lam"], c)
        base = os.path.join(RUN_DIR, tag)
        print(f"  {tag}: {t_spin:.0f} s on {nr} ranks ...", flush=True)
        if not launch(base, "spinup", nr, c):
            print("    FAILED -- see run.log")
            continue
        m = measure_speed(base, c)
        if m:
            print(f"    U_mean = {m['U_meas']:.3f} m/s (design {cfg['U']:.2f}), "
                  f"U_max = {m['U_max']:.3f} m/s  ->  head_factor = {m['hf_new']:.3f}")
            out.append(m["hf_new"])
    if out:
        print(f"\n  suggested head_factor = {max(out):.3f}   (max over cases -- both share one"
              f" bed slope by design)")
        print(f"  set CONFIG['head_factor'], re-run --calibrate until it stops moving, "
              f"then --launch")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="run spinup then morph for each case")
    ap.add_argument("--calibrate", type=float, metavar="SECONDS", default=0,
                    help="short rigid-bed spin-up to measure the required head_factor")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(CONFIG, args.calibrate)
        return

    built = [write_case(r["lam"], CONFIG) for r in RUNS]
    report(built, CONFIG)

    if not args.launch:
        return
    import glob
    for tag, _, _, nr in built:
        base = os.path.join(RUN_DIR, tag)
        print(f"\n=== {tag} ===")
        if not launch(base, "spinup", nr, CONFIG):
            continue
        snaps = sorted(glob.glob(os.path.join(base, "spinup", "output", "u_*")))
        if not snaps:
            print("    no spinup snapshot written -- cannot hot start"); continue
        last = os.path.basename(snaps[-1]).split("_")[1]
        print(f"    hot start from spinup snapshot {last}")
        p = os.path.join(base, "morph", "input.txt")
        txt = open(p).read().replace("@INI@/eta.txt", f"../spinup/output/eta_{last}") \
                            .replace("@INI@/u.txt", f"../spinup/output/u_{last}") \
                            .replace("@INI@/v.txt", f"../spinup/output/v_{last}")
        open(p, "w").write(txt)
        launch(base, "morph", nr, CONFIG)


if __name__ == "__main__":
    main()
