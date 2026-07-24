# The fast-flow / slow-morphology timescale problem, and how Thetis handles it

*Literature review + method note for `numerical/thetis/`.*
*Question addressed: the hydrodynamics equilibrates fast (one transit), bank
erosion is slow (decades); after each bank move must the flow be re-spun-up on
the fast timescale? What is the good way?*

**Citation discipline.** Every DOI below was read **verbatim** from a file on
disk. Methods whose primary source is *not* on disk are described and their
authors named, but marked **[unverified — not on disk]** and given **no DOI**.
Claims about what Thetis does are grounded in its source tree (`thetis-src/`,
line numbers given), which is on disk.

---

## 1. The problem, quantified for this package

Two timescales, measured by `geometry.py`:

| timescale | value | set by |
|---|---|---|
| `T_hydro` — flow adjustment (one transit `L/Ū`) | **2 099.5 s** | gravity + advection + friction |
| `T_bank` — real bank migration | **decades** | field erodibility `E/U ~ 10⁻⁸` |

Ratio `T_bank / T_hydro ~ 10⁸`. A naive loop that (i) moves the bank, then
(ii) **re-runs the flow from rest to steady state** before the next move would
spend ~one transit of compute per infinitesimal bank increment — astronomically
wasteful, and the reason morphodynamic modelling exists as a subfield.

The physical fact that makes this tractable is **slaving**: because
`T_hydro ≪ T_bank`, the flow is always in quasi-equilibrium with the *current*
planform. Move the bank by a small `δ`, and the flow is perturbed only by
`O(δ/b)` and re-equilibrates in `O(T_hydro)` — you never return to rest. Every
method below is a different way of exploiting this one fact.

---

## 2. Three families of solution

### A. Morphological acceleration factor (morfac) — *online coupling*

Keep marching the flow through the bank/bed changes without ever restarting;
multiply the morphological change per hydrodynamic step by a factor `MF` so a
short hydrodynamic run represents a long morphological time. The bed/bank sees
`MF·δ` while the flow sees `δ`. Provided `MF` is small enough that the geometry
change per flow-relaxation time stays in the slaved regime, the flow tracks its
moving equilibrium and is **never re-spun-up** — the state is simply carried
across each update.

This is the workhorse of coastal/estuarine morphodynamics (Delft3D, XBeach,
Telemac, Thetis). The idea is usually attributed to **Roelvink (2006)**, "Coastal
morphodynamic evolution techniques" (the morfac/continuity and RAM approaches),
building on **Lesser et al. (2004)** (the Delft3D-FLOW morphodynamic
formulation), with the input-reduction / representative-forcing side developed by
**Latteux (1995)** and **de Vriend et al. (1993)**. **[unverified — not on
disk; no DOI cited.]**

**In Thetis this is a first-class field.** `exner_eq.py:72–75`:

```python
morfac = fields.get('morfac')
porosity = fields.get('porosity')
fac = Constant(morfac/(1.0-porosity))     # bed-continuity acceleration
```

i.e. the Exner (bed-continuity) update is scaled by `MF/(1−porosity)`. The
`sediment_meander_2d` example (Yen & Lee 180° bend) runs `morfac = 50`. The DG
morphodynamic scheme Thetis implements is documented in **Clare et al. (2020)**,
"Hydro-morphodynamics 2D modelling using a discontinuous Galerkin
discretisation", *Computers & Geosciences* 140, 104658,
**doi:10.1016/j.cageo.2020.104658** *(verified verbatim from
`thetis-src/examples/sediment_meander_2d/meander_example.py:10–11`)*.

**This package already uses (A).** `meander_thetis.py`: the flow state `(uv,
elev)` is copied across every rebuild (`carry_state`, exact DOF copy — never a
re-spin-up), and `CONFIG['morph_factor'] = 6000`. Because the bed is frozen there
is no Exner equation; the morfac multiplies the **bank** law instead
(`advance_banks`, `dt_morph_eff = morph_every·dt·MF`). So the answer to *"must I
re-spin-up after each bank move?"* is **already no** — the flow rides its moving
equilibrium.

*Limitation of (A).* `MF` is bounded above by morphodynamic stability: if the
geometry moves too far per flow-relaxation time the flow stops tracking and the
scheme can go unstable or bias the bed. There is no free lunch — larger `MF`
trades accuracy for speed, and the safe value is problem-dependent.

### B. Steady-state flow solver — *replace spin-up with a Newton solve* ★ recommended

The deeper point: for this problem the flow is not merely quasi-steady, it is
**genuinely steady** between bank moves (a river reach at constant discharge).
So do not *time-march* to equilibrium at all — solve the **steady** shallow-water
equations directly as a nonlinear boundary-value problem, one Newton solve per
bank update. There is then **no fast timescale to spin up**: the concept of
"spin-up" only exists for an initial-value time integrator.

**Thetis ships exactly this.** `timeintegrator.py:255`:

```python
class SteadyState(TimeIntegrator):
    """Time integrator that solves the steady state equations,
       leaving out the mass terms"""
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE
    ...
    self.solver_parameters.setdefault('snes_type', 'newtonls')   # Newton line-search
    self.F = self.equation.residual('all', solution, solution, fields, fields, bnd_conditions)
```

It drops the `∂ₜ` mass terms and does a Newton solve of the steady residual
(`options.py:62` `SteadyStateTimeStepperOptions2d` defaults to a direct LU
solve). Selectable simply by `options.swe_timestepper_type = 'SteadyState'`.

The morphological loop becomes:

```
repeat:
    solve steady SWE on the current geometry     # ONE Newton solve, no spin-up
    read near-bank u', move the banks (× MF)     # the slow-timescale step
    rebuild the mesh
```

Cost per bank update: one Newton solve (a handful of linear solves) instead of
`~T_hydro/dt ≈ 1000` explicit-ish steps. This is the numerical embodiment of
Ikeda's own **quasi-steady** assumption (§C) and is the principled fix for the
exact concern raised.

*Caveats, stated honestly.* (i) The steady DG-SWE with nonlinear advection is a
harder algebraic problem than one time step — Newton needs a good initial guess
(the previous solve provides it, since the bank moved little) and may need
continuation for strong bends. (ii) It gives only the equilibrium; any genuinely
unsteady flow physics (which this quasi-steady problem does not have) would be
lost — that is the whole point.

**Verified on this package's exact setup (2026-07-24).** Switching
`options.swe_timestepper_type = 'SteadyState'` and seeding Newton with the
analytic base flow, the SNES converges **quadratically in 2 iterations**:

```
0 SNES Function norm 2.04e-01
1 SNES Function norm 8.74e-05
2 SNES Function norm 1.88e-10        -> max|u| = 1.014 m/s, all finite
```

One steady solve (≈2 linear solves after JIT) replaces the ~2 834 s of
Crank–Nicolson spin-up (2 transits × 1.35 s/step). The excellent initial guess —
the base flow is already within `O(δ/b)` of the bend equilibrium — is exactly why
Newton needs only two steps, and is the same slaving that justifies the whole
approach. **Status: verified feasible; the production runs still use
Crank–Nicolson + morfac (family A), and migrating the morphological loop to
`SteadyState` is now a validated one-line change.**

### C. Analytic slaving — *eliminate the flow entirely (the bend equation)*

The limit of (B): if the flow can be solved steadily, it can in principle be
solved *symbolically* and eliminated, collapsing the whole system to a single
evolution equation for the planform `y(x, t)` on the slow timescale. This is
precisely **Ikeda, Parker & Sawai (1981)**, "Bend theory of river meanders.
Part 1. Linear development", *J. Fluid Mech.* 112, 363–377,
**doi:10.1017/S0022112081000451** *(verified verbatim from the page render)*.
Their eq. (16),

```
y_xt + 2 C_f y_t = y_xxx − C_f (A + F²) y_xx ,
```

*is* the bank law with the flow already slaved out (their §2–3 quasi-steady
expansion). Integrating it is the ultimate acceleration — **no hydrodynamic
solve at all**, just march the planform. The finite-amplitude extension is
**Parker, Sawai & Ikeda (1982)**, "Bend theory of river meanders. Part 2.
Nonlinear deformation of finite-amplitude bends", *J. Fluid Mech.* 115, 303–314,
**doi:10.1017/S0022112082000767** *(verified verbatim from the PDF text layer)*.

The trade-off is the mirror image of Thetis's: (C) is essentially free but valid
only within its linear/weakly-nonlinear closure and its prescribed cross-stream
structure; Thetis (B) buys the **full nonlinear 2-D flow** on an arbitrary
planform at the cost of one Newton solve per step. The repo already has (C):
`../ikeda_1981/` (linear) and `../meander_migration/` (the bend model + an SWE
extension).

---

## 3. Where the three sit — a map

```
  cost per bank update        physics retained            in this repo
  ------------------------    ------------------------    ------------------------
C  ~0 (integrate an ODE/PDE)  linear closure, slaved       ikeda_1981/, meander_migration/
B  1 Newton solve             FULL nonlinear steady flow   thetis/  (SteadyState — recommended)
A  ~1000 time steps           full unsteady flow (unused)  thetis/  (current: CrankNicolson+morfac)
```

The flow here has **no unsteady physics to resolve** (steady discharge, no tides,
no waves), so paying for (A)'s time integration buys nothing that (B) does not
give more cheaply. **Recommendation: switch the morphological loop from
Crank–Nicolson to Thetis's `SteadyState` solver** — it removes the fast-timescale
spin-up the question is about, matches Ikeda's quasi-steady assumption, and is
already implemented upstream.

---

## 4. Annotated bibliography

**Verified on disk (DOI read verbatim):**

- **Ikeda, Parker & Sawai (1981)**, *JFM* 112, 363–377,
  doi:10.1017/S0022112081000451. The quasi-steady slaving that turns the coupled
  flow+bank system into the single planform equation (16); family (C). Read from
  the 200-dpi page render (the text layer drops the display math).
- **Parker, Sawai & Ikeda (1982)**, *JFM* 115, 303–314,
  doi:10.1017/S0022112082000767. Finite-amplitude (weakly nonlinear) extension of
  the slaved bend equation.
- **Clare et al. (2020)**, *Computers & Geosciences* 140, 104658,
  doi:10.1016/j.cageo.2020.104658. The discontinuous-Galerkin hydro-morphodynamic
  scheme in Thetis, including the morfac/Exner coupling of family (A); the
  reference cited by Thetis's own `sediment_meander_2d` example.

**Thetis source (on disk, not a paper):**

- `thetis-src/thetis/timeintegrator.py:255` — `SteadyState` integrator (family B).
- `thetis-src/thetis/exner_eq.py:72–75` — `morfac/(1−porosity)` bed-continuity
  acceleration (family A).
- `thetis-src/thetis/options.py:62` — `SteadyStateTimeStepperOptions2d` (direct LU).
- `thetis-src/examples/sediment_meander_2d/` — morfac = 50 worked example.

**Methods referenced, primary source NOT on disk — [unverified], no DOI:**

- **Roelvink (2006)**, "Coastal morphodynamic evolution techniques" — the morfac,
  continuity-correction and RAM online-acceleration techniques (family A).
- **Lesser et al. (2004)** — the Delft3D-FLOW morphodynamic formulation that
  popularised the online morfac.
- **Latteux (1995)**; **de Vriend et al. (1993)** — input reduction /
  representative-forcing schematisation (relevant to tidal/wave morphodynamics;
  **not** needed here, since the river forcing is already a single steady state).

*These four are named for completeness and to point the reader at the standard
literature; their bibliographic details must be checked against the originals
before being cited in any paper — they were not verified from a file on disk.*
