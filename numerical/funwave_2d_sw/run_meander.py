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
import pathlib
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
    h_plain=0.20,        # [m] depth of the ALWAYS-WET shelf beyond the bank.  Not a dry
                         #   floodplain: an oblique wet/dry bank is what breaks the run
                         #   (see section_depth).  0.20 m keeps the shelf below the erosion
                         #   threshold (tau/tau_cr = 0.19) and leaks ~1.7%/unit width.
    freeboard=1.5,       # [m] retained only for the figure scripts
    plain=75.0,          # flat floodplain beyond the bank toe [m].  v2: 25 -> 75, i.e. +50 m of
                         #   MIGRATION HEADROOM.  v1 measured only 36-44 m of clearance from the
                         #   bank edge to the closed y-wall (~0.4 W); bed change AT the wall was
                         #   exactly 0.000 m so no sponge is needed, but a migrating bank must not
                         #   be allowed to reach it.  Gate on clearance every run.
    Cd_floodplain_mult=30.0,  # [-] drag multiplier on the always-wet floodplain (rough/vegetated).
                         #   Confines the flow to the channel (shelf otherwise carries ~5% at 0.3 m/s
                         #   along the meander tangent) WITHOUT a wet/dry boundary.  Channel unchanged.
    # =================== MEANDER ===========================================
    # C0 = A k^2 is held FIXED across runs; A is derived.
    # C0 = 5.0e-3 1/m  <=>  R_min = 200 m = R/W = 2.0, the lower edge of the Leopold-Wolman
    # natural band (R_c/W ~ 2-3).  The first attempt used C0 = 8.496e-3 (R/W = 1.18, tighter
    # than any stable natural meander): the inner bank sits at R-b = 67.7 m, free-vortex
    # scaling accelerates the flow (R+b)/(R-b) = 2.47x, the friction slope rises 6.1x, and
    # the reach needed ~2.8 m of head against the 0.46 m supplied -- it drew down and blew up.
    C0=3.0e-3,           # apex curvature [1/m] -> R_min = 333 m = 3.33 W.  NOT free: a
                         #   sine-generated curve has lam <= 5.073/C0, so C0 = 5.0e-3 caps
                         #   lam at 1015 m and could not carry EITHER 1040 or 1560 m.  See
                         #   theta_max().  R/W = 3.33 also clears the R/W >= 2 stability
                         #   threshold measured earlier by a wide margin.
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
    interior_len=6240.0, # [m] the ANALYSED reach: 8 x 780 = 4 x 1560, whole bends in both cases.
                         #   The domain is interior + 2*buffer (see reach_length).
    buffer_len=1950.0,   # [m] = straight_len + 1560.  The AMPLITUDE RAMP is buffer-straight =
                         #   1560 m = 2 x lam(B1) = 1 x lam(B2): an INTEGER number of wavelengths
                         #   in BOTH cases.  This is what puts the straight lead-in/lead-out ON
                         #   THE AXIS: measured straight offset -3.3 m (B1) and +0.1 m (B2) from
                         #   the bend mid-range.  A ramp spanning a fractional wavelength makes
                         #   the first bend weaker than the rest, which biases the excursion and
                         #   throws the straights off-axis (measured +40 m at the old 1040 m ramp).
    straight_len=390.0,  # [m] of the buffer that is DEAD STRAIGHT before the amplitude ramp
                         #   starts.  The tide sponge is 30 cells (75 m); putting it in a
                         #   genuinely uniform straight channel is the only place where the
                         #   BC state (U = U_design, V = 0, eta = const) is exactly right.
                         #   Measured: every B2 blow-up localises at x < 31 m or x > 6160 m,
                         #   i.e. INSIDE the sponge, never in the bends.
    # =================== FLOW ==============================================
    U=0.85,              # target reach-mean velocity [m/s].  NOT 1.0: measured bisection
                         #   showed B2 (12 bends) drains at head_factor >= ~1.25, i.e. its
                         #   ceiling is U ~ 0.88-0.92, while U = 1.0 needs hf ~ 1.53.  Its
                         #   curvature reverses every lam/2 = 260 m against a bend-flow
                         #   adjustment length H/(2 Cf) ~ 974 m, so transverse momentum
                         #   accumulates instead of resetting (U_max/U_mean = 1.66, exactly
                         #   the free-vortex bound).  Both cases must share the discharge, so
                         #   the target drops to what BOTH can carry.
    # The straight-channel normal slope does NOT include bend losses, so imposing it
    # together with the inlet velocity over-specifies the problem and the reach drains.
    # head_factor scales the BED SLOPE and the boundary head together (so the state stays
    # self-consistent: H = h(n) everywhere at uniform flow) and is CALIBRATED:
    #     micromamba run -n fourcastnetv2 python run_meander.py --calibrate
    # measures the achieved mean speed U_meas and reports  hf_new = hf * (U/U_meas)^2.
    head_factor=1.0,     # [-] multiplier on the straight-channel normal slope.  PER-RUN:
                         #   each case is calibrated separately (see RUNS below) so that both
                         #   reach the SAME U.  The differing bed slope is then each geometry's
                         #   own resistance -- a consequence of the wavenumber, not a free
                         #   variable, so it is not a confound.
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
    Morph_factor=8,      # [-] v2 production MF, to SEE the bed+bank surface evolve in reasonable
                         #   wall time.  Validity limit MF << T_morph/T_adjust: use the FASTEST
                         #   morphological process (the bar), measured T_bar ~ 1.5e5 s, and
                         #   T_adjust = H/(Cd U) = 2292 s, so T_bar/T_adjust ~ 65 and MF <= ~10 is
                         #   safe (MF=1000 in the old doc was invalid).  MF=8 sits under that, and at
                         #   A=0 the forcing (F^2=0.09) is far weaker than the A=2.89 case, so dz/dt
                         #   is smaller and the true ceiling is HIGHER -- MF=8 is comfortably safe.
                         #   The archived MF=1 run is the convergence anchor this is checked against.
                         #   The hot-start (run_v2.py carries depth_cur.txt across chunks) makes each
                         #   re-adjustment a cheap PERTURBATION relaxation, not a cold spin-up.
                         # (old note, retained) NOT 50: the gap-2 transverse relaxation has a stability
                         #   window tau in [|Zb_target| dt MF/(1e-3 H), H/(Zb_rate MF)] that is
                         #   EMPTY at MF=50 (4318 > 3750) -- which is why every ON run either blew
                         #   up (tau too small) or was ON==OFF (tau too large).  MF<=10 opens the
                         #   window; MF=5 gives [432, 37500] s, tau=2000 sits comfortably inside.
                         #   Cost: 10x more hydro time per morphological day, but a point bar
                         #   forms in ~1000 s morphological time, so it is affordable.

    Morph_interval=30.0,   # [s] = 1.5 x T_c (T_c=20.1 s).  P_ave/D_ave refresh every MI and JUMP;
                         #   at MI=200 that step-change x MF blew the bed up at t=150 s (measured:
                         #   MI={20,5} healthy past 400 s, MI=200 dead at 150).  This flow is
                         #   BEDLOAD-dominated (u*/w_s=0.45), so the bar is carried by the
                         #   instantaneous bedload deflection, not the suspended load -- MI need
                         #   only span ~1 T_c (light smoothing), and SMALLER MI is in fact safer
                         #   (smaller refresh jump).  30 s sits between the measured-healthy MI=20
                         #   and the config's old MI=40, giving margin for the longer production
                         #   morph.  The ON/OFF decisive test ran clean at MI=10.
    Aval_interval=50.0,    # [s] SHORT (was 200): bank-migration needs frequent small slump kicks,
                         #   not rare large ones -- avalanching relaxes only the single steepest
                         #   neighbour per interval, so a multi-directional bank collapse needs
                         #   several intervals; 50 s keeps each kick within the MF stability window.
    MinDepthPickup=0.01,   # [m]  0.1 (the shipped example) switches OFF bank-toe pickup,
                         #   which is the entire bank-retreat mechanism.  It cannot be 0
                         #   either: the log-law drag is singular at H = e k_s/30 = 1.1e-4 m.
    # =================== GAP 1: secondary-flow closure (Ikeda 1981) ========
    # The one mechanism a depth-averaged 2D model structurally lacks: the helical secondary
    # flow of a bend, which advects fast near-surface fluid to the OUTER bank.  Ikeda-Parker-
    # Sawai 1981 close it WITHOUT resolving the helix -- their tangential-momentum eq (3b),
    #   U du'/ds = -g dxi'/ds - (Cf U^2/H)(2u'/U - xi'/H + eta'/H),
    # carries the secondary flow only through the eta'/H term, with eta'/H = -A C n the
    # equilibrium transverse bed tilt it produces (eq 6; A=2.89 alluvial, Suga 1963).  FUNWAVE
    # ALREADY solves the other two friction terms exactly and nonlinearly (its own -Cd u|u|
    # is the 2u'/U piece, its own -g d(eta)/ds is the -xi'/H piece), so the ONE missing term
    # is (Cf U^2/H)(eta'/H).  Adding only that term turns the drag into a spatial modulation
    #   Cd_eff = Cd (1 + A kappa n)       [this code's (kappa,n) sign convention]
    # written to cd.txt and read via FRICTION_MATRIX.  Outer bank (kappa*n < 0, verified in
    # tests/test_bathy.py 7b) gets Cd_eff < Cd -> the flow accelerates outward, reversing the
    # depth-averaged free-vortex tendency that otherwise scours the INNER bank.  The sediment
    # bed shear (mod_sediment.F:1328) uses its OWN log-law drag on the local speed |U|, NOT
    # this Cd field, so the faster outer flow raises Tau_xy there with no steady-state
    # Cd*U^2 = gHS cancellation -- gap-2 could never do this because it never touched the flow.
    SecondaryFlow=True,        # gap-1 MOMENTUM half (friction modulation).  OFF = stock Cd.
    # gap 1 BEDLOAD half -- the bar-CREATING mechanism.  The friction modulation above only
    # redistributes the depth-averaged flow (weak at R/W~3: the k_f/sqrt(k_f^2+k_lam^2)~0.17
    # adjustment/wavelength factor caps it).  The point BAR is a bedload feature: the helical
    # secondary flow deflects the near-bed velocity (hence bedload) toward the INNER bank by
    # delta = A kappa H / f_slope.  Balanced against the Talmon down-slope term (BedSlopeDeflection),
    # the transverse bedload balance drives the bed to Ikeda eq (6) dz_b/dn = A kappa H -- deep
    # outer, shallow inner.  mod_sediment.F reads the curvature field kappa from Curv_file.
    SecondaryBedload=True,     # gap-1 A/B toggle for the bedload deflection.  OFF = flow-only.
    BedRelaxation=False,       # the Zb->Zb_target relaxation (superseded by SecondaryBedload;
                               #   had a Morph_factor over-amplification).  Kept OFF behind its
                               #   own flag so the Talmon stabiliser can run without it.
    # =================== GAP 2: transverse bed-slope deflection ============
    # Added to mod_sediment.F (BedSlopeDeflection / A_bedslope).  Stock FUNWAVE = the Ikeda
    # A=0 limit: bedload goes along atan2(V,U) only, so a bend scour hole cannot shed
    # sediment sideways and deepens without bound (MEASURED: inner-bank hole reached H=0 at
    # t=940 s and blew up -- the failure that looked numerical for a whole day was physics).
    # The Talmon 1995 deflection tilts bedload downhill; A_bedslope is calibrated so the
    # equilibrium transverse tilt reproduces Ikeda's alluvial A=2.89 (Suga 1963), capping the
    # relief at A*C0*H = 1.3 m -- just below the 1.5 m runaway, so it ARRESTS the hole.
    BedSlopeDeflection=True,   # gap-2 TALMON down-slope only now (the Zb_target relaxation moved
                               #   to BedRelaxation).  ON as the STABILISER that balances the gap-1
                               #   bedload deflection: deflection builds the bar, down-slope sheds
                               #   sediment out of the deepening hole, and their balance IS Ikeda
                               #   eq (6).  Without it the outer scour hole would run away.
    A_bedslope=9.0,            # Talmon coefficient (bedload-direction form, kept for the flag)
    A_ikeda=0.0,               # v2 A=0: the Ikeda INCISED case (I81 PARAMS_INCISED), user decision
                               #   2026-07-24.  A_secondary=A_ikeda (input :750), so A_ikeda=0 zeros
                               #   ALL THREE secondary-flow closures at once: the friction modulation
                               #   in cd.txt (:825), the bedload deflection delta=A*kappa*H (:838),
                               #   and the equilibrium tilt bedslope.txt (:851).  The 3D helical
                               #   circulation that eq(6) parameterises cannot be produced by a
                               #   depth-averaged model at all, so A=0 is the HONEST 2D limit -- only
                               #   the free-surface superelevation F^2=0.09 drives the bend, against a
                               #   free vortex (u~1/r) that is faster on the INSIDE.  Which bank erodes
                               #   is therefore a MEASURED OUTPUT, not an assumption (inner-bank
                               #   erosion is a valid result).  KEPT active: the floodplain drag
                               #   confinement in cd.txt (numerical, not secondary flow) and the
                               #   Talmon down-slope diffusion A_bedslope (gravitational, not helical).
                               #   (was 2.89, Ikeda alluvial from Suga 1963.)
    Kslope=2000.0,             # tau_relax [s], inside the MF=5 window [432, 37500]: first-order relaxation of Zb toward the Ikeda
                               #   tilt Zb_target.  NOT a diffusivity -- the Laplacian form
                               #   injected a spurious source and blew up at t=20s (correct
                               #   point-bar sign, uncontrolled amplitude).  >=5000 keeps the
                               #   bed pre-tilted (build_case) so departure starts at ZERO -> no initial shock,
                               #   tau can be short; 200s damps scour departures fast enough to compete.
    # =================== NUMERICS ==========================================
    dx=2.5,              # [m] -> 40 cells across the channel width
    spin_transits=1.0,   # phase-1 length, in channel transit times L_channel/U.  Per-run,
                         #   so both cases enter phase 2 equally converged; starting both
                         #   from rest instead would give B1 a 2.5x longer spin-up.
    t_morph=60000.0,     # [s] HYDRODYNAMIC integration time.  Morphological span = t_morph * MF =
                         #   6e4 * 8 = 4.8e5 s ~ 5.6 days -- about 3x the A=2.89 bar-saturation time,
                         #   deliberately generous because the A=0 forcing (F^2=0.09) is weaker and
                         #   evolves the bed more slowly per morphological second.  Wall ~6 h/case at
                         #   64-128 ranks (measured 2.83 sim-s/wall-s), both cases concurrently.
                         # (superseded note) 3e4 -> 1.5e5 because MF went 5 -> 1.  The MORPHOLOGICAL span is
                         #   t_morph*MF, so 1.5e5 s x 1 reproduces the same bed evolution the v1
                         #   run reached as 3e4 s x 5 -- but fully coupled, on one clock.
                         # (old note, retained) At MF=5 the
                         #   morphological span is t_morph*MF = 1.5e5 s ~ 1.74 d, several point-
                         #   bar development times for the gap-1 secondary-flow deflection to
                         #   build the bar.  plot_intv=250 -> 120 frames.  Wall ~40 min/case;
                         #   both cases run concurrently, so ~40 min + ~15 min spin-up.
    TideEast_U=0.0,      # [m/s] outflow velocity nudge.  0 makes the outlet a SELF-ADJUSTING
                         #   sink: momentum is damped and the eta nudge removes exactly the
                         #   water that arrives, so it cannot over-extract.  Setting it to the
                         #   design U over-extracts by the calibration error (the reach runs
                         #   ~18% slow before hf converges) and drains the reach -- which is
                         #   why calibration could never converge for B2.
    CFL=0.5,             # [-]  (0.3 tested: does NOT fix the B2 blow-up)
    MinDepth=0.01,       # [m] wet/dry threshold, also used as MinDepthFrc
    plot_intv=1000.0,    # [s] snapshot interval -> 60 morph frames over t_morph=6e4 s (each frame
                         #   is 8000 s of MORPHOLOGICAL time apart at MF=8, so the surface visibly
                         #   changes frame to frame -- the point of this run).
                         #   v2: 250 -> 1000.  At 250 s the morph phase wrote 600 frames x ~5 ASCII
                         #   fields x 618k values = 27.8 GB, formatted value by value.  120 frames
                         #   is already more than a movie needs (v1 movies used 120) and cuts that
                         #   to 5.6 GB.  Measured strong scaling saturates at 64 ranks (2.83
                         #   sim-s/wall-s; 128 ranks buys 5%, 256 ranks is 2.7x SLOWER at 2413
                         #   cells/rank), so wall time has to be bought by doing less work, not by
                         #   adding cores.
    max_ranks=256,       # a case is rank-limited, not core-limited: at 64 ranks the larger
                         #   case runs 14.7k cells/rank against the smaller one's 6.9k and
                         #   takes twice the wall clock.  PAD_X/PAD_Y allow up to 32x8.
)

# The two cases.  ONLY lambda differs; A is derived from C0 so the drive is identical.
# head_factor is PER-RUN and calibrated (run_meander.py --calibrate) so both reach the same U.
# lam = 520 was abandoned: it blew up at EVERY head_factor tested (0.80-1.558) and with
# every BC variant, while a straight channel on the same grid ran clean.  lam = 1560 and
# 1040 are both healthy.  head_factor from the hf=1.0 probes (U_meas 0.721 / 0.695).
# v2 CASE MATRIX -- TWO runs, EQUAL AMPLITUDE, different wavelength.
#
#   B1: 8 bends over the 6240 m interior, lam =  780 m
#   B2: 4 bends over the same interior,   lam = 1560 m
#
# Both are built to the SAME amplitude A = 50 m, so C0 is an OUTPUT of (lam, A), solved by
# C0_for_amplitude() below and pinned here.  Equal A with a 2x wavelength ratio necessarily gives a
# 4x curvature contrast (kappa ~ A k^2): B1 R/W = 3.28, B2 R/W = 12.52.  A = 50 m is the largest
# common amplitude that keeps B1 off the tight end -- at A = 80 m B1 falls to R/W = 2.24, tighter
# than the v1 case that blew up (R/W = 3.33).  Two things make R/W = 3.28 acceptable here: (1) the
# v1 blow-up was at A=2.89 STRONG forcing, whereas this run is A=0 (F^2=0.09) -- a far weaker drive,
# so the tight bend is much less stressed; (2) the hot-start decoupled scheme (run_v2.py) keeps the
# flow quasi-steady between bed updates.  MF=8 is well under the T_bar/T_adjust~65 ceiling.
#
# head_factor is a SLOPE multiplier (length-independent) and is re-calibrated per case on U; the
# seeds below are v1's gentle-reach value, since both v2 cases are far less sinuous (1.040, 1.010)
# than the v1 pair and therefore need less head.
#
# 2x2 MATRIX (user 2026-07-24): {A=0 incised, A=2.89 alluvial} x {k=8 lam780, k=4 lam1560}.
# Each `group` is its own subfolder (runs/A0/, runs/A2p89/).  `A_ikeda` per run overrides the CONFIG
# default; A=2.89 turns the three secondary-flow closures back on (friction, deflection, tilt), A=0
# leaves only F^2=0.09.  Same head_factor for both A values: the secondary friction modulation is
# antisymmetric in n, so it does not change the channel-mean drag or U.
_WV = [("B1", 780.0, 3.052121e-03), ("B2", 1560.0, 7.985212e-04)]   # (tag, lam, C0); R/W 3.28 / 12.52
RUNS = ([dict(tag=t, lam=lam, C0=c0, head_factor=1.6915, A_ikeda=0.0,  group="A0")    for t, lam, c0 in _WV]
      + [dict(tag=t, lam=lam, C0=c0, head_factor=1.6915, A_ikeda=2.89, group="A2p89") for t, lam, c0 in _WV])
# -> RUNS[0]=A0/k8  RUNS[1]=A0/k4  RUNS[2]=A2p89/k8  RUNS[3]=A2p89/k4


def C0_for_amplitude(lam, A_target, cfg):
    """Apex curvature C0 that makes the BUILT centreline reach amplitude A_target at wavelength lam.

    The construction prescribes C0 and MEASURES A, so pinning A instead requires this inversion.
    Solves for the max deflection angle theta_m on the gentle branch (amplitude is monotone in
    theta_m) and returns C0 = theta_m * k_s = 2 pi J0(theta_m) theta_m / lam.  Used to generate the
    C0 values pinned in RUNS; kept so they are reproducible rather than magic numbers."""
    import copy
    from scipy.special import j0
    from scipy.optimize import brentq

    def built_A(th_m):
        c = copy.deepcopy(cfg); c["C0"] = 2.0 * np.pi * j0(th_m) * th_m / lam
        return amplitude(lam, c)

    th = brentq(lambda t: built_A(t) - A_target, 1e-4, THETA_PEAK * 0.99)
    return float(2.0 * np.pi * j0(th) * th / lam)


def cfg_for(run, cfg=None):
    """CONFIG with this run's per-case overrides applied."""
    import copy
    c = copy.deepcopy(cfg if cfg is not None else CONFIG)
    c.update({k: v for k, v in run.items() if k not in ("tag", "lam")})
    return c        # exactly lam_ref/2 so the reach is a common multiple


def cfg_from_grid(g, cfg=None):
    """CONFIG with C0 (and buffer_len) restored from an AS-BUILT grid.npz, so postprocessing
    recomputes the SAME centreline that was carved.  channel_coords()/section_x() depend on C0,
    and a run whose C0 differs from the global CONFIG (the fix-amplitude B3: C0=1.42e-3 vs 3e-3)
    would otherwise get its (n,s,tx,ty) channel frame placed on the wrong curve.  R_min = 1/C0
    is stored in every grid, so this recovers it without a rebuild."""
    import copy
    c = copy.deepcopy(cfg if cfg is not None else CONFIG)
    c["C0"] = 1.0 / float(g["R_min"])
    if "buffer_len" in g:
        c["buffer_len"] = float(g["buffer_len"])
    return c


# --------------------------------------------------------------------------- #
#  Derived quantities (pure numpy -- inspectable and testable without FUNWAVE)
# --------------------------------------------------------------------------- #
def amplitude(lam, cfg):
    """Peak lateral excursion of the built centreline [m].  MEASURED, not prescribed: with a
    curvature-defined curve the amplitude is an output of (C0, lam), not an input."""
    x, y, _, _, _, _ = centreline(lam, cfg)
    L = reach_length(cfg)
    m = (x >= 0) & (x <= L)
    return float(np.abs(y[m]).max())


def wavenumber(lam):
    return 2.0 * np.pi / lam


def slope_design(cfg):
    """Straight-channel normal slope: g H S = Cd U^2.  Excludes bend losses."""
    return cfg["Cd"] * cfg["U"] ** 2 / (G_ACCEL * cfg["H_c"])


def slope(cfg):
    """The slope actually built into the bed and the boundary head.  Calibrated."""
    return cfg["head_factor"] * slope_design(cfg)


def loglaw_cd(H, cfg):
    """Depth-dependent log-law drag  Cd(H) = kappa^2 / [ln(30 H / k_s) - 1]^2,
    k_s = 2.5 D50, kappa^2 = 0.16.

    This is FUNWAVE's OWN sediment-module bed-shear coefficient (mod_sediment.F:1372), so using it
    for the MOMENTUM drag makes flow and transport see the SAME bed.  The scalar cfg['Cd'] is exactly
    this evaluated once at H = H_c, and the momentum equation then held it FROZEN while the sediment
    module recomputed the shear from the live depth every step -- a 1.5-2.2x mismatch on the shallow
    bank face, and the whole point at A=0 where F^2 is the only bend driver.  Writing cd.txt from this
    (and refreshing it from the evolving bed each morph chunk, run_v2.py) restores the consistency.
    Equals cfg['Cd'] at H = H_c by construction; the argument is floored at 5 to stay off the log-law
    singularity (30H/k_s = e) for ultra-shallow cells."""
    ks = 2.5 * cfg["D50"]
    arg = np.maximum(30.0 * np.maximum(H, cfg.get("MinDepth", 0.01)) / ks, 5.0)
    return 0.16 / (np.log(arg) - 1.0) ** 2


THETA_PEAK = 1.25578          # argmax of J0(theta)*theta ; J0(th)*th = 0.80736 there


def lam_max(cfg):
    """Longest down-valley wavelength a sine-generated curve of curvature C0 can have."""
    return 2.0 * np.pi * 0.80736 / cfg["C0"]


def theta_max(lam, cfg):
    """Solve  lam = 2 pi J0(theta) theta / C0  on the gentle (low-theta) branch.

    For a sine-generated curve theta = theta_m sin(k_s s) the sinuosity is 1/J0(theta_m) and
    the curvature amplitude is theta_m k_s, so (C0, lam) are a COUPLED constraint, not two
    free knobs.  J0(theta) theta peaks at 0.807, hence lam <= 5.073/C0.  The original design
    asked for C0 = 5.0e-3 with lam = 1040 and 1560 m, i.e. beyond lam_max = 1015 m -- no such
    meander exists.  The tapered-sine construction "achieved" it only by letting the local
    curvature overshoot C0 by 12-97%, putting R/W down to 1.33 where the flow is unstable.
    """
    from scipy.optimize import brentq
    from scipy.special import j0
    if lam > lam_max(cfg):
        raise ValueError(f"lam={lam:.0f} m exceeds lam_max={lam_max(cfg):.0f} m for "
                         f"C0={cfg['C0']:.3e}; lower C0 or shorten lam")
    return brentq(lambda th: 2.0 * np.pi * j0(th) * th / cfg["C0"] - lam, 1e-9, THETA_PEAK)


def centreline(lam, cfg, ds_frac=0.25):
    """Sine-generated centreline with a tapered DEFLECTION ANGLE.

        theta(s) = theta_m T(s) sin(k_s s),   kappa = d theta / d s

    Tapering theta (not kappa) keeps int kappa ds = 0 over every period, so the curve has no
    net rotation and does not drift -- tapering kappa directly made the reach wander by tens
    of channel widths.  The price is a curvature overshoot in the ramp,
    |kappa|max = C0 (1 + |T'|/k_s); with C0 = 3.0e-3 that still leaves R/W >= 2, the measured
    stability threshold, which build_case() asserts.

    Returns (x, y, s, t_x, t_y, kappa) sampled in arc length.
    """
    from scipy.special import j0
    th_m = theta_max(lam, cfg)
    ks = cfg["C0"] / th_m
    sinu = 1.0 / j0(th_m)
    L, ds = reach_length(cfg), cfg["dx"] * ds_frac
    s = np.arange(-lam * sinu, (L + lam) * sinu * 1.05, ds)
    th = th_m * taper_arc(s, cfg, sinu) * np.sin(ks * s)
    kap = np.gradient(th, ds)
    x = np.cumsum(np.cos(th)) * ds
    y = np.cumsum(np.sin(th)) * ds
    i0 = int(np.argmin(np.abs(s)))
    x -= x[i0]; y -= y[i0]
    keep = (x >= -lam / 2.0) & (x <= L + lam / 2.0)
    xk, yk = x[keep], y[keep]
    # Re-centre the reach in y.  Tapering theta zeroes the NET ROTATION per period, but the
    # POSITION y = int sin(theta) ds still drifts: the reach holds a non-integer bend count and
    # the end bends are amplitude-tapered, so int sin(theta) ds over [0,L] != 0.  Un-corrected the
    # channel hugs one closed (PERIODIC=F) wall -- measured mean y=-103 m for lam=1560, only 99 m
    # of floodplain to the lower wall vs 304 m to the top.  Shift by the excursion MID-RANGE (not
    # the mean) so the closest approach to each wall is equal and amplitude() sizes a symmetric
    # domain half.
    inr = (xk >= 0.0) & (xk <= L)
    if inr.any():
        yk = yk - 0.5 * (yk[inr].max() + yk[inr].min())
    return xk, yk, s[keep], np.cos(th[keep]), np.sin(th[keep]), kap[keep]


def taper_arc(s, cfg, sinu):
    """Amplitude ramp expressed in ARC length: the buffer is a DOWN-VALLEY length, so it
    corresponds to sinu times as much arc."""
    L = reach_length(cfg) * sinu
    b0, s0 = cfg["buffer_len"] * sinu, cfg["straight_len"] * sinu
    d = np.minimum(np.clip(s, 0.0, L), L - np.clip(s, 0.0, L))
    return 0.5 * (1.0 - np.cos(np.pi * np.clip((d - s0) / max(b0 - s0, 1e-9), 0.0, 1.0)))


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
    L, s, w = reach_length(cfg), cfg["straight_len"], cfg["buffer_len"]
    d = np.minimum(np.clip(x, 0.0, L), L - np.clip(x, 0.0, L))     # distance to nearer end
    ramp = max(w - s, 1e-9)
    return 0.5 * (1.0 - np.cos(np.pi * np.clip((d - s) / ramp, 0.0, 1.0)))


def section_x(lam, cfg, n_sec=2):
    """Down-valley x of interior bend apexes.  For a sine-generated curve the apex (peak |kappa|,
    where theta=0 so the tangent points down-valley) is exactly where a FIXED-x grid slice cuts
    the channel transversely -- the natural yOz cross-section.  Returns up to n_sec apex x-values
    of ALTERNATING curvature sign, spread over the interior, so the same locations mark the xOy
    movie (grey dashed) and cut the yOz movie -- both always refer to the same place."""
    xc, _, _, _, _, kap = centreline(lam, cfg)
    buf, L = cfg["buffer_len"], reach_length(cfg)
    ak = np.abs(kap)
    inr = (xc > buf) & (xc < L - buf)
    # local maxima of |kappa| inside the reach = apexes
    ext = np.zeros_like(ak, bool)
    ext[1:-1] = (ak[1:-1] >= ak[:-2]) & (ak[1:-1] > ak[2:]) & (ak[1:-1] > 0.5 * ak[inr].max())
    idx = np.where(ext & inr)[0]
    apex = [(float(xc[i]), float(np.sign(kap[i]))) for i in idx]
    # walk from the interior centre outward, taking alternating signs
    if not apex:
        return []
    apex.sort(key=lambda t: abs(t[0] - 0.5 * L))
    out, signs = [], set()
    for x, sg in apex:
        if sg not in signs:
            out.append(x); signs.add(sg)
        if len(out) >= n_sec:
            break
    return sorted(out)


def reach_length(cfg):
    """Down-valley domain length -- the SAME for every run, by construction.

    v2: built as ANALYSED interior + two buffers, rather than lam_ref * n_bends_ref, because the
    interior must hold a whole number of bends of BOTH wavelengths (8 x 780 = 4 x 1560 = 6240 m)
    while the buffers are sized by the amplitude ramp, not by the wavelength."""
    return cfg["interior_len"] + 2.0 * cfg["buffer_len"]


def fold_margin(lam, cfg):
    """kappa_max * b.  Must be < 1 or the inner bank crosses the centreline."""
    return cfg["C0"] * cfg["b"]


def sinuosity(lam, cfg):
    """Arc length per unit down-valley length, measured on the built centreline."""
    x, _, s, _, _, _ = centreline(lam, cfg)
    L = reach_length(cfg)
    m = (x >= 0) & (x <= L)
    return float((s[m][-1] - s[m][0]) / (x[m][-1] - x[m][0]))


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
    |n| >  b : bank falling at 1:m_bank to a SHALLOW, ALWAYS-WET shelf of depth h_plain.

    The shelf is not cosmetic.  A dry floodplain puts a wet/dry boundary along the bank, and
    on an oblique (staircased) bank that boundary is what destroys the run: measured, same
    lambda/buffer/head, topographic bank -> blow-up, MASK_STRUC vertical wall -> healthy, and a
    STRAIGHT channel (grid-aligned wet/dry line) is stable in every configuration ever tried.
    FUNWAVE's own beach cases work because their shoreline is grid-aligned (sediment_rip:
    depth varies 2.88 m std across-shore, 0.11 m along-shore).

    Keeping the whole domain wet removes the wetting-drying algorithm from the problem while
    LEAVING THE BANK ERODIBLE -- which a vertical wall does not, since MASK_STRUC is never
    updated after initialisation.  The shelf itself does not wash away: at h_plain = 0.2 m the
    normal-flow shear is tau/tau_cr = 0.19, below threshold, while the bank face spans
    0.2-3 m depth so its toe does cross threshold.  It leaks ~1.7% of the channel discharge
    per unit width (~4% in total here), which is measured and reported by the gates.
    """
    b, w0 = cfg["b"], cfg["H_c"] ** -0.5
    beta = 2.0 * (cfg["H_b"] ** -0.5 - w0) / b ** 2
    inside = (w0 + beta * np.minimum(np.abs(n), b) ** 2 / 2.0) ** -2
    bank = cfg["H_b"] - (np.abs(n) - b) / cfg["m_bank"]
    return np.where(np.abs(n) <= b, inside, np.maximum(bank, cfg["h_plain"]))


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
    k, L = wavenumber(lam), reach_length(cfg)
    xc, yc, sc, tx, ty, kap = centreline(lam, cfg)

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

    n, s, tx, ty, kap_g = channel_coords(X, Y, lam, cfg)
    # the ramp can overshoot C0; R/W < 2 is the measured instability threshold, so assert it
    kmax = float(np.abs(kap_g).max())
    assert 1.0 / kmax >= 2.0 * 2 * cfg["b"], (
        f"lam={lam:.0f}: |kappa|max={kmax:.3e} -> R/W={1/kmax/(2*cfg['b']):.2f} < 2, unstable")
    S = slope(cfg)
    h_sec = section_depth(n, cfg)
    # gap 2: NO pre-tilt of the bed.  (Baking the Ikeda tilt into Depth_ini and zeroing the
    # relaxation target made the term cancel bar formation -- sign flip, v3b, worse than OFF.)
    # Keep a symmetric IC and let the Fortran relax Zb toward the NON-ZERO Ikeda target field
    # Zb_target = A kappa H n (written to bedslope.txt below), with tau_relax just above the
    # initial-shock floor of ~4300 s.  tilt stays zero.
    tilt = np.zeros_like(h_sec)
    # bed drops downstream ALONG THE CHANNEL: using arc length s (not x) makes the
    # sinuosity correction S_valley = sinuosity * S_channel automatic instead of a
    # separate constant that can be forgotten.
    s0 = s.mean()
    Depth = h_sec + tilt + S * (s - s0)

    # analytic normal flow: eta = -S(s - s0) so that H = eta + Depth = h_sec exactly,
    # and |u| = sqrt(g h S / Cd) directed along the local centreline tangent.
    eta = -S * (s - s0)
    # IC speed = the STRAIGHT-channel normal flow for the calibrated bed slope,
    # |u| = sqrt(g h S_bed / Cd) = sqrt(head_factor) * U * sqrt(h/H_c).
    # The reach settles BELOW this because bend losses eat the difference, so the transient
    # is a deceleration.  Starting instead at the design U makes it an ACCELERATION, and
    # that transient blew B2 up at t = 963 s (hf = 1.558; also fails at CFL = 0.3).
    # SPATIAL drag: the floodplain beyond the bank toe is HIGH-friction (rough/vegetated), so its
    # normal-flow speed is sqrt(g h S/(mult*Cd)) -- nearly quiescent.  This confines the flow to the
    # channel (the always-wet shelf otherwise carries ~5% at 0.30 m/s, following the meander tangent
    # unphysically) WITHOUT a wet/dry boundary.  The channel's own normal-flow balance is unchanged,
    # so U and the head calibration are unaffected.
    toe = cfg["b"] + cfg["m_bank"] * (cfg["H_b"] - cfg["h_plain"])
    # IC drag uses the SAME depth-dependent log law as cd.txt, so the analytic initial state is
    # already close to the solver's steady state (shorter spin-up) instead of being built on a
    # scalar Cd the solver no longer uses.
    cd_base_ic = loglaw_cd(h_sec, cfg)
    Cd_ic = np.where(np.abs(n) > toe, cd_base_ic * cfg.get("Cd_floodplain_mult", 30.0), cd_base_ic)
    speed = np.where(h_sec > 0.0,
                     np.sqrt(G_ACCEL * np.maximum(h_sec, 0.0) * S / Cd_ic), 0.0)
    ini = dict(eta=eta, u=speed * tx, v=speed * ty)

    # Buffer over the first and last bend, needed because FUNWAVE's FLUX_SCALAR_BC
    # hard-zeros the sediment flux on all four boundaries, which puts an artificial
    # scour/deposition zone at the inlet and outlet.
    # Zs is the MAX ERODIBLE THICKNESS and is applied as `IF(Zb>Zs) Zb=Zs` with Zb>0 =
    # erosion -- so Zs=0 blocks EROSION ONLY.  Deposition (Zb<0) is NOT capped and the
    # buffer does accrete (measured: 0 eroding cells, 120 depositing cells, -0.27 m).
    # That is acceptable -- the point is to stop the boundary artefact propagating into
    # the interior -- but the buffer is NOT a frozen bed and must not be analysed.
    # SMOOTH ramp instead of a hard 0/LARGE step: with the interior fully erodible, the buffer
    # edge is the ONLY erodibility jump, and a hard oblique step there seeds a knickpoint (the
    # design-review risk).  Ramp Zs from 0 (buffer) to LARGE (interior) over ~8 dx at each edge;
    # the sink-freeze (Zb=0 where Zs<1) then stays scoped to the buffer core, keeping the
    # erodible interior Exner-conservative.
    edge = 8.0 * cfg["dx"]
    d_edge = np.minimum(X - cfg["buffer_len"], (L - cfg["buffer_len"]) - X)   # dist past nearest buffer edge
    Zs = np.clip(d_edge / edge, 0.0, 1.0) * 1.0e6

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
    return (f"lam{lam:.0f}_C{cfg['C0']*1e3:.2f}e-3_U{cfg['U']:.2f}_hf{cfg['head_factor']:.3f}"
            f"_D50{cfg['D50']*1e6:.0f}um_MF{cfg['Morph_factor']}").replace(".", "p")


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
TideEast_U = {TideEast_U}
TideEast_V = 0.0
PERIODIC = F
  ! ---------------- PHYSICS ----------------
DISPERSION = F
Cd = {Cd}
  ! gap 1: FRICTION_MATRIX reads a spatially-varying Cd (init.F:854, IN_Cd) that carries the
  ! Ikeda-1981 secondary-flow closure Cd_eff = Cd(1 + A kappa n).  When SecondaryFlow is OFF
  ! this is F and the scalar Cd above is used, reproducing stock FUNWAVE exactly.
FRICTION_MATRIX = {friction_matrix}
FRICTION_FILE = ../bathy/cd.txt
C_smg = 0.25
nu_bkg = 0.0
  ! ---------------- NUMERICS ----------------
CFL = {CFL}
! FroudeCap deliberately NOT set (v2).  It is a non-conservative velocity limiter; with it at 1.0
! the v1 runs had 41 cells pinned at Fr = 1.000 EXACTLY, all at |n| = 56-57 m (the bank toe) in
! 0.11-0.14 m of water.  The flow wanted to go supercritical there and was being clamped -- exactly
! where bank erosion is the quantity of interest.  Left unset the toe is free; the capped-cell
! count is replaced by a max-Froude gate, and a Froude panel in the plan-view movie.
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
BedSlopeDeflection = {bed_slope_deflection}
A_bedslope = {A_bedslope}
Kslope = {Kslope}
BedSlope_file = ../bathy/bedslope.txt
  ! gap 1 bedload half: secondary-flow deflection of bedload toward the inner bank
SecondaryBedload = {secondary_bedload}
A_secondary = {A_ikeda}
Curv_file = ../bathy/kappa.txt
BedRelaxation = {bed_relaxation}
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
              friction_matrix="T" if cfg.get("SecondaryFlow") else "F",
              secondary_bedload="T" if cfg.get("SecondaryBedload") else "F",
              bed_relaxation="T" if cfg.get("BedRelaxation") else "F",
              bed_slope_deflection="T" if cfg.get("BedSlopeDeflection") else "F",
              A_bedslope=cfg.get("A_bedslope", 9.0), Kslope=cfg.get("Kslope", 0.5),
              A_ikeda=cfg.get("A_ikeda", 2.89),
              eta_west=+drop / 2.0, eta_east=-drop / 2.0)
    return INPUT_TEMPLATE.format(**kw), px * py


def case_base(cfg, tag):
    """Output directory for a case: RUN_DIR / group / tag.  `group` (e.g. 'A0', 'A2p89') puts a
    run in its own subfolder so the same (lam, C0) geometry can be run at several A values without
    collision.  Defaults to RUN_DIR/tag when no group is set."""
    return os.path.join(RUN_DIR, cfg.get("group", ""), tag)


def write_case(lam, cfg):
    Depth, Zs, ini, x, y, meta = build_case(lam, cfg)
    tag = run_tag(lam, cfg)
    base = case_base(cfg, tag)
    for sub in ("bathy", "ini", "spinup/output", "morph/output"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    # FUNWAVE reads  DO J=Jbeg,Jend / READ(1,*)(Var(I,J),I=Ibeg,Iend)
    # i.e. Nglob rows of Mglob values -> transpose our (nx, ny) arrays.
    np.savetxt(os.path.join(base, "bathy", "depth.txt"), Depth.T, fmt="%14.7e")
    np.savetxt(os.path.join(base, "bathy", "hard.txt"), Zs.T, fmt="%14.7e")
    # gap 1: the Ikeda-1981 secondary-flow closure as a spatially-varying drag.  eq (3b)+(6)
    # give Cd_eff = Cd(1 + A kappa n); FUNWAVE reads it via FRICTION_MATRIX.  Sign is the
    # GEOMETRICALLY VERIFIED one (test_bathy.py 7b: outer bank is kappa*n < 0 -> Cd_eff < Cd
    # -> the flow accelerates outward).  Tapered linearly to 0 across the bank face (b -> toe)
    # because the secondary flow is a channel-bend phenomenon, not a floodplain one, and
    # floored/capped at [0.2, 1.8] Cd as a safety net (the taper keeps it inside [0.57,1.43]
    # in the channel, so the clip never binds there -- it only guards the tight-apex ramp).
    if cfg.get("SecondaryFlow"):
        Xg1, Yg1 = np.meshgrid(x, y, indexing="ij")
        nf1, _, _, _, kf1 = channel_coords(Xg1, Yg1, lam, cfg)
        A_ik = cfg.get("A_ikeda", 2.89)
        toe = cfg["b"] + cfg["m_bank"] * (cfg["H_b"] - cfg["h_plain"])
        taper = np.clip((toe - np.abs(nf1)) / (toe - cfg["b"]), 0.0, 1.0)
        # BASE drag is now the DEPTH-DEPENDENT log law (see loglaw_cd), not the scalar Cd, so the
        # momentum drag matches the sediment module's own bed shear cell by cell.  The secondary-flow
        # modulation (A) and the floodplain multiplier ride on top of it.
        cd_field = loglaw_cd(Depth, cfg) * np.clip(1.0 + A_ik * kf1 * nf1 * taper, 0.2, 1.8)
        # HIGH-friction floodplain beyond the bank toe (rough/vegetated): confines the flow to the
        # channel so it follows the bank, instead of the always-wet shelf carrying ~5% everywhere.
        # Ramped over ~4 dx from the toe so the drag is not a hard oblique step.
        fp = np.clip((np.abs(nf1) - toe) / (4.0 * cfg["dx"]), 0.0, 1.0)          # 0 at toe -> 1 on floodplain
        cd_field = cd_field * (1.0 + (cfg.get("Cd_floodplain_mult", 30.0) - 1.0) * fp)
        np.savetxt(os.path.join(base, "bathy", "cd.txt"), cd_field.T, fmt="%14.7e")
    # gap 1 BEDLOAD half: the channel curvature kappa, read by mod_sediment.F to deflect the
    # bedload toward the inner bank by delta = A kappa H / f_slope (Ikeda 1981).  Tapered to the
    # channel (the bedload magnitude already vanishes on the immobile shelf); sign is kappa's own.
    if cfg.get("SecondaryBedload"):
        Xg4, Yg4 = np.meshgrid(x, y, indexing="ij")
        nf4, _, _, _, kf4 = channel_coords(Xg4, Yg4, lam, cfg)
        toe = cfg["b"] + cfg["m_bank"] * (cfg["H_b"] - cfg["h_plain"])
        taper4 = np.clip((toe - np.abs(nf4)) / (toe - cfg["b"]), 0.0, 1.0)
        np.savetxt(os.path.join(base, "bathy", "kappa.txt"), (kf4 * taper4).T, fmt="%14.7e")
    # gap 2: the Ikeda equilibrium tilt is now BAKED INTO Depth_ini (the pre-tilt in
    # build_case), so the relaxation target for Zb (the CHANGE from Depth_ini) is ZERO -- the
    # term simply damps scour departures back to the pre-tilted equilibrium.  Writing an
    # explicit zero field keeps the Fortran read path uniform and self-documenting.
    if cfg.get("BedSlopeDeflection"):
        Xg2, Yg2 = np.meshgrid(x, y, indexing="ij")
        nf2, _, _, _, kf2 = channel_coords(Xg2, Yg2, lam, cfg)
        A_ik = cfg.get("A_ikeda", 2.89)
        zbt = np.where(np.abs(nf2) <= cfg["b"], -A_ik * kf2 * cfg["H_c"] * nf2, 0.0)
        np.savetxt(os.path.join(base, "bathy", "bedslope.txt"), zbt.T, fmt="%14.7e")
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

    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    n_ab, _, _, _, kap_ab = channel_coords(Xg, Yg, lam, cfg)
    np.savez(os.path.join(base, "bathy", "grid.npz"), x=x, y=y, Depth=Depth, Zs=Zs,
             n=n_ab, kappa=kap_ab, t_spin=t_spin, nranks=nr,
             **{k: v for k, v in meta.items()})
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
    # as-built mask: never recompute from CONFIG.  A config edit between build and
    # analysis silently shifts the centreline and makes these sample floodplain cells
    # as channel (observed: a bogus 'eta = +1.625 m, freeboard negative').
    chan = np.abs(g["n"]) <= cfg["b"]
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
    # as-built mask: never recompute from CONFIG.  A config edit between build and
    # analysis silently shifts the centreline and makes these sample floodplain cells
    # as channel (observed: a bogus 'eta = +1.625 m, freeboard negative').
    chan = (np.abs(g["n"]) <= cfg["b"]) & (g["Zs"] > 0)
    sp = np.hypot(u, v)
    U_meas = float(sp[chan].mean())
    return dict(snapshot=os.path.basename(su[-1]), U_meas=U_meas, U_max=float(sp[chan].max()),
                hf_new=cfg["head_factor"] * (cfg["U"] / max(U_meas, 1e-6)) ** 2)


def calibrate(cfg, seconds, apply=False):
    """Short rigid-bed spin-ups to find the head the SINUOUS reach actually needs.

    The straight-channel normal slope omits bend losses; imposing it together with the inlet
    velocity over-specifies the problem and the reach drains until it blows up.  U ~ sqrt(S),
    so one measurement gives a good next guess and two iterations converge."""
    import copy
    print(f"\n=== CALIBRATION: ~{seconds:.0f} s rigid-bed spin-up per case, "
          f"head_factor = {cfg['head_factor']:.3f} ===")
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        c = cfg_for(r, cfg)
        _, meta, _, _ = write_case(r["lam"], c)              # get transit time
        c["spin_transits"] = seconds / meta["transit"]
        tag, meta, t_spin, nr = write_case(r["lam"], c)
        base = os.path.join(RUN_DIR, tag)
        if not launch(base, "spinup", nr, c):
            return None
        m = measure_speed(base, c)
        if not m:
            return None
        print(f"    {r['tag']}: U_mean = {m['U_meas']:.3f} m/s (target {cfg['U']:.2f}), "
              f"U_max = {m['U_max']:.3f}  ->  head_factor {c['head_factor']:.3f} "
              f"-> {m['hf_new']:.3f}", flush=True)
        return (r["tag"], m["hf_new"])

    with ThreadPoolExecutor(max_workers=len(RUNS)) as ex:
        out = [r for r in ex.map(one, RUNS) if r]
    if out and apply:
        # Write the result back into this file.  The calibration is only valid for the
        # geometry it was measured on; decoupling the two silently invalidated it twice in
        # this project (buffer_len and C0 were both changed after a calibration).
        src = pathlib.Path(__file__).read_text()
        import re
        for tag, hf in out:
            # Match on the tag and rewrite whatever float follows head_factor.  An exact-string
            # replace keyed on repr(float) fails silently when the source spells the same value
            # differently ("1.00" vs "1.0") -- which it did, so two calibration rounds both ran
            # from hf=1.0 and the write-back was a no-op.  Verify the substitution happened.
            pat = re.compile(rf'(dict\(tag="{re.escape(tag)}",\s*lam=[0-9.]+,\s*head_factor=)'
                             r'([0-9.]+)(\))')
            src, n_sub = pat.subn(rf'\g<1>{hf:.4f}\g<3>', src)
            if n_sub != 1:
                raise RuntimeError(f"head_factor write-back for {tag} matched {n_sub} sites, "
                                   f"expected 1 -- refusing to continue with a stale calibration")
            print(f"    {tag}: head_factor -> {hf:.4f}")
        pathlib.Path(__file__).write_text(src)
        print("\n  head_factor written back into RUNS.  Re-run --calibrate until it stops "
              "moving, then --launch.")
    elif out:
        print("\n  set head_factor PER RUN in RUNS above, then --launch")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="run spinup then morph for each case")
    ap.add_argument("--calibrate", type=float, metavar="SECONDS", default=0,
                    help="short rigid-bed spin-up to measure the required head_factor")
    ap.add_argument("--apply", action="store_true",
                    help="write the calibrated head_factor back into RUNS")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(CONFIG, args.calibrate, apply=args.apply)
        return

    built = [write_case(r["lam"], cfg_for(r)) for r in RUNS]
    report(built, CONFIG)

    if not args.launch:
        return
    # The cases are INDEPENDENT, so run them concurrently: 64 + 64 ranks on a 384-core
    # node halves the wall clock and still leaves two thirds of the machine idle.
    import glob
    from concurrent.futures import ThreadPoolExecutor

    def one_case(i):
        tag, _, _, nr = built[i]
        base = os.path.join(RUN_DIR, tag)
        rc = cfg_for(RUNS[i])
        if not launch(base, "spinup", nr, rc):
            return f"{tag}: spinup FAILED"
        snaps = [q for q in sorted(glob.glob(os.path.join(base, "spinup", "output", "u_*")))
                 if "99999" not in q]
        if not snaps:
            return f"{tag}: no spinup snapshot -- cannot hot start"
        last = os.path.basename(snaps[-1]).split("_")[1]
        fp = os.path.join(base, "morph", "input.txt")
        txt = open(fp).read().replace("@INI@/eta.txt", f"../spinup/output/eta_{last}") \
                             .replace("@INI@/u.txt", f"../spinup/output/u_{last}") \
                             .replace("@INI@/v.txt", f"../spinup/output/v_{last}")
        open(fp, "w").write(txt)
        ok = launch(base, "morph", nr, rc)
        return f"{tag}: {'DONE' if ok else 'morph FAILED'} (hot start {last})"

    total = sum(b[3] for b in built)
    print(f"\nlaunching {len(built)} cases concurrently on {total} ranks")
    with ThreadPoolExecutor(max_workers=len(built)) as ex:
        for msg in ex.map(one_case, range(len(built))):
            print("  " + msg)


if __name__ == "__main__":
    main()
