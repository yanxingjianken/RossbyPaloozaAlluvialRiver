# Governing equations, boundary conditions, drag, and bank erosion

Specification for the Thetis meandering-channel model in `numerical/thetis/`.
Written **before** the production runs: the code is written *from* this document.

Every number quoted here is computed by `geometry.py` / `sw_note.py`, not asserted.
Reproduce with `python geometry.py`, `python sw_note.py`.

Sources, all on disk:

| ref | file | role |
|---|---|---|
| **I81** | `literature/Ikeda et al. - 1981 - Bend theory of river meanders. Part 1…pdf` | the bend theory; eqs (1)-(13) read verbatim from 200-dpi page renders (the PDF text layer drops all display math) |
| **SW** | `literature/River_Meandering_SW.pdf` + `docs/River_Meandering_SW_corrected.pdf` | the channel-following SW reduction, as corrected |
| **RP** | `literature/river.pdf` p.19 | the group's 3-level bank-erosion law |

---

## 1. Domain and coordinates

Down-valley coordinate `x`; the meander is `y = c(x)` (I81's Cartesian frame).
Transverse reference coordinate `n~ ∈ [-1, 1]`, so `n = n~·b`.

The reference domain is the rectangle `[0, L] × [-1, 1]`, mapped to the physical
channel by

```
X = x ,   Y = c(x) + n~·b(x) ,   c = (y_N + y_S)/2 ,  b = (y_N - y_S)/2
```

i.e. the banks sit at their y-position **at each valley station** (vertical
cuts), not at a normal offset from the centreline. The two constructions differ
at O(theta^2), `theta = arctan(c')`; measured at t=0 this is **8.5e-5 (m=4)** and
**3.4e-4 (m=8)**. Vertical cuts can never produce an inverted cell, which a
normal-offset map does as soon as `|n~·b·kappa| -> 1`.

**Thetis solves the Cartesian shallow-water equations on this curved mesh.** The
`(s,n)` metric is therefore carried exactly by the finite-element geometry and
**no `sigma = 1 + nC` factor appears anywhere in the code** — the same reason I81
eq. (1a-c) carries none (§2.2).

| quantity | value |
|---|---|
| `L` total / `L_in` entry / `L_m` meander / `L_out` exit | 1965.6 / 175.5 / 1684.8 / 105.3 m |
| `W = 2b` | 35.10 m (`b` = 17.55 m) |
| mesh | 224 x 28 -> 12 544 triangles, `dx` = 8.78 m, `dn` = 1.254 m (aspect 7.0) |

The **straight entry reach** is the flow-conditioning reach of a laboratory
flume. Its length is set against the friction adjustment length `H/(2C_f)` from
I81 eq. (7): that is **10.0 m = 0.28 W**, so `L_in` = 175.5 m is **17.6
adjustment lengths** — the inlet condition is fully forgotten before the first
bend. Entry and exit reaches are **non-erodible** (prior work in this repo hit
pile-up artefacts at erodible/rigid interfaces).

---

## 2. Governing equations

### 2.1 What Thetis solves

Depth-averaged shallow water, `h = eta - z_b`:

```
d_t eta + div(h u) = 0
d_t u + (u.grad)u = -g grad(eta) - C_f |u|u / h + div(nu grad u)
```

DG-DG P1, Crank-Nicolson. `eta` is prognostic, so **superelevation and the whole
`F^2` part of the bend response are computed, not parameterised** (§6).

### 2.2 I81, verbatim

Read from p. 365-368; `xi~` is the **water surface**, `eta~` the **bed**,
`h~ = xi~ - eta~` the depth, `C~` the curvature. Note (1a-c) carry **no metric
factor** — they are the exact channel-following system already truncated at
leading order in `nC <= b/R`, which is consistent because curvature *is* the
perturbation (`C~ = 0 + C'`).

```
(1a)  u~ du~/ds~ + v~ du~/dn~ + C~ u~v~ = -g dxi~/ds~ - tau_s/(rho h~)
(1b)  u~ dv~/ds~ + v~ dv~/dn~ - C~ u~^2 = -g dxi~/dn~ - tau_n/(rho h~)
(1c)  C~ v~h~ + d(v~h~)/dn~ + d(u~h~)/ds~ = 0
(2)   C_f U^2 = gHI ;   UH = q_w
(3a)  U^2 C' = g dxi'/dn~
(3b)  U du'/ds~ = -g dxi'/ds~ - C_f (U^2/H)(2u'/U - xi'/H + eta'/H)
(5)   xi' = C' U^2 n~ / g
(6)   eta'/H = -A C' n~
(7)   U du'_b/ds~ + 2(U/H)C_f u'_b = b[-U^2 dC'/ds~ + C_f C'(U^4/(gH^2) + A U^2/H)]
```

### 2.3 The SW note, and what had to be corrected

`docs/River_Meandering_SW_corrected.pdf` carries the full errata in red. The
load-bearing corrections:

- **the bed was deleted.** "Neglecting the bottom topography => grad eta = grad h"
  removes the only term that can balance bed friction; with `H_0 = const` its
  eq. (9) gives `U_0 = 0`. Its own equations admit only a drawdown state
  `dH_0/ds = -C_f Fr^2/(1-Fr^2)`, which drains this channel in **~202 m** — under
  half of one meander wavelength. **Restoring `h = eta - z_b` restores I81
  exactly**: (1a) has `-g dxi~/ds~` with `xi~` the *surface*, while the note wrote
  `-g dh/ds` with `h` the *depth*, and `h + z_b = eta = xi~`.
- **eq. (10) forced constant depth across the channel.** Once `z_b` is back it
  reads `d_n(h_0 + z_b) = 0`: the free surface is flat across a straight section
  *while the depth is free to vary*. One symbol repairs both.
- continuity must carry the depth inside the derivatives (I81 (1c) already does).
- an undefined `H` in its eq. (6) friction terms (should be `H_0`); `Lambda` never
  defined (a `lambda` vs `lambda/2pi` choice moves `alpha` by 2pi and
  `eps ~ alpha^2` by ~40x).

`sw_note.py` solves the corrected system directly and is the reference the
Thetis run must reduce to (`tests/test_sw_note.py`). Its own self-test
reproduces a **closed-form solution** for the flat-jet limit to 5e-13.

---

## 3. Base state — an exact steady solution

The jet is prescribed quadratic (the vorticity-gradient provider) and the **bed
follows from it**, so nothing decays during spin-up:

```
ubar(n~) = U_0 + Delta (1 - n~^2)
H(n~)    = C_f ubar^2 / (gI + nu ubar_nn) ,   ubar_nn = -2 Delta/b^2  (constant)
z_b(x, n~) = eta_ref - I x - H(n~)
```

With `nu -> 0` this is `ubar = sqrt(gIH/C_f)` — I81 eq. (2), `ubar ~ sqrt(H)`.
**The jet and the bed are not independently choosable**: a quadratic `ubar`
forces a quartic `H`, and the depth contrast is `(ubar_centre/ubar_bank)^2`.

`z_b` **falls downstream at the valley slope** (that is what drives the flow)
while its cross-sectional shape and the depth are identical at every `x`.
*"The bed does not change" means `d_t z_b = 0`, not `d_x z_b = 0`.*

| quantity | value |
|---|---|
| `C_f`, `F`, `nu` | 0.05, 0.30, 0.05 m^2/s |
| `I = C_f F^2` (exact, from (2)) | 4.500e-3 |
| `U_0` (bank) / `Delta` / width-mean `Ubar` | 0.7802 / 0.2341 / 0.9362 m/s |
| `H` range / width-mean | 0.691 - 1.167 m / 1.000000 m |
| Froude across the section | 0.2997, **uniform to 1e-16** (`F^2 = I/C_f`) |
| momentum-balance residual | **1.6e-16** of `gI` — exact to machine precision |

Background PV gradient of this base state, `qbar_n = -ubar_nn/H + ubar_n H_n/H^2`:

| term | value |
|---|---|
| `beta_shear = -ubar_nn/H` (mid-channel) | 1.30e-3 (m s)^-1 |
| `beta_topo  =  ubar_n H_n/H^2` | 0 -> 2.63e-3 (m s)^-1 |
| `|beta_topo/beta_shear|` max | **2.39** |

So the frozen n-varying bed is the *stronger* PV-gradient source at this design
point. This is recorded as a **property of the prescribed base state**, i.e. the
design rationale for the quadratic jet — **not** as a claim about the dynamics.
By explicit decision this package ships **movies only** and runs no `T_shear`,
PV-budget or gravity-vs-vortical diagnostics; the repo's prior retraction of
"the meander *is* the vortical/Rossby wave"
(`dedalus_meander_full_SW/README.md`) stands untouched and is not re-tested here.

**Initial condition** = this analytic base state with `u' = v' = 0`. The bank
sinuosity then spins `u', v'` up from zero — that *is* the requested fast
spin-up. Do **not** start from literal rest: prior work established that a
constant-value BC from rest is a Heaviside step -> startup bore, and
over-specifies a subcritical boundary.

---

## 4. Boundary conditions

| boundary | condition | why |
|---|---|---|
| inflow, `x = 0` | `{'un': -ubar(n~)}` — the base-jet profile | a subcritical inflow admits **exactly one** characteristic. Prescribing the profile injects the jet; prescribing `elev` as well would over-specify |
| outflow, `x = L` | `{'elev': eta_0(L)}` | the one outgoing characteristic |
| banks, `n~ = +-1` | `{'un': 0}` — free slip | no penetration; assumes nothing about interior divergence. Free slip (not no slip) because the cross-channel shear is *prescribed* by the jet, not generated by wall friction |

**Never pin `elev` + `u` + `v` at the same subcritical boundary** — that is the
over-specification two independent reviewers flagged in the FUNWAVE work.

---

## 5. Drag

`options.quadratic_drag_coefficient = C_f = 0.05`, i.e. `tau_b/rho = C_f|u|u/h`,
which is **I81's `C_f` with no conversion** — the point of not using Manning or
Nikuradse, either of which would break the correspondence with (2), (3b) and (7).

`C_f = 0.05` is high for a large river (0.005-0.03) and appropriate for a
shallow, rough, flume-like channel. It is a **derived** choice, not a fudge: with
`A = 0` the Ikeda-selected wavelength is `lambda_OM = 21.06 H_0/C_f`, so

| `C_f` | `lambda_OM` | vs observed 10-14 W |
|---|---|---|
| 0.01 (river-typical) | ~62 W | far above observation |
| **0.05 (used here)** | **12.00 W** | in range |

That table *is* the quantitative price of `A = 0` (§6), made visible rather than
hidden.

Horizontal viscosity `nu = 0.05 m^2/s` is carried for DG stability and **enters
the base state** (it appears in `H(n~)`), so the base state stays exact rather
than being perturbed by the stabiliser.

---

## 6. Secondary flow: needed, and deliberately omitted

**This section is the answer to "do we already resolve it, or do we need
A = 2.89?"**

Substituting (5) and (6) into (3b) and using `h' = xi' - eta'`:

```
h'/H = (F^2 + A) C' n~
```

so the last two terms of (3b)'s friction bracket are just `-h'/H`. **`A` and
`F^2` occupy the same slot — a depth (drag) modulation.** `F^2` is the
free-surface superelevation; `A` is the secondary-flow bed tilt.

- A 2D model with a prognostic free surface computes the **`F^2` half exactly**.
- It **cannot produce the `A` half at all**: eq. (6) comes from the *3D helical
  circulation*, which depth averaging removes by construction. No 2D closure
  recovers it; it must be imposed.

For an alluvial river `A = 2.89` (I81 p. 367: *"An analysis of 45 bends of ten
alluvial rivers in Japan based on data collected by Suga (1963) suggests an
average value of 2.89"*; Engelund suggested ~4) versus `F^2 = 0.09` here — so
**`A` would be ~97% of the bend forcing**.

**Decision (user, 2026-07-23): `A = 0`.** That is not an approximation to the
alluvial case; it is I81's **incised** case, already a named parameter set on
disk (`ikeda_1981/ikeda_lib.PARAMS_INCISED`). It also makes the bed *literally*
frozen, which is what the specification asked for. Its cost, computed:

| | `A = 2.89` (alluvial) | `A = 0` (used) |
|---|---|---|
| `A + F^2` | 2.98 | 0.09 |
| `k_c = C_f sqrt(2(A+F^2))` | 2.441 `C_f` | 0.424 `C_f` (**5.75x smaller**) |
| peak `alpha_0/C_f^2` | 1.336 | **1.98e-3** (~675x weaker) |
| `lambda_OM` at `C_f`=0.01 | ~12 W | ~62 W |

**Consequence to hold onto:** with `A = 0` the only driver pushing the fast
filament toward the *outer* bank is `F^2 = 0.09`, competing against a free vortex
(`u ~ 1/r`) that is faster on the *inside*. **Which bank erodes is therefore a
measured output, not a design assumption.** Inner-bank erosion is a live
possibility and, if it happens, is a result — not a bug.

`A` is a one-line switch (`Config.A_ikeda`) should the alluvial case be wanted.

---

## 7. Erosion and deposition

### 7.1 I81 and RP are the same law

I81 (11)-(12), verbatim: `gamma dy~/dt~ = zeta~`, `gamma = cos theta`, with
`zeta~ = zeta~(U) + E(U) u'(s~,b)` and `zeta~(U) == 0`, so `E > 0` is *the*
bank-erosion coefficient. p. 368: *"erosion of the 'north' bank corresponds to
equal deposition at the 'south' bank."*

RP p. 19 prints instead `d_t psi'_1 = (eps C_f U_0/b)(psi'_2 - psi'_1)`. Two
substitutions collapse one onto the other:

1. **the bank is a displaced streamline.** `psibar(b) + psibar'(b) y_b + psi'_1 =
   psibar(b)` with `ubar = -d psibar/dy` gives `psi'_1 = ubar(b) y_b = U_0 y_b`.
2. **`(psi'_2 - psi'_1)/b` *is* `u'_b`** — the 3-level finite difference of
   `u' = -d psi'/dy` at the bank.

=> `U_0 d_t y_b = eps C_f U_0 u'_b`, i.e.

> **`gamma d_t y_b = E u'_b` with `E = eps · C_f`** — I81 (13) exactly.

*Independently corroborated:* the `deliverable1_noboru_model` session concluded
from the **deck's own dispersion relation** that the complete parameter set is
`{k*, D, gamma, eps·C_f}` — the erosion coefficient enters *only* as that
product. Two different routes, same answer. Signs agree with I81 p. 368.

### 7.2 What this package implements

Ikeda's form, because Thetis **resolves** `u'` at the wall; the
`(psi'_2-psi'_1)/b` form is a coarse half-width difference that exists only
because the 3-level model has nowhere else to get `u'_b`.

```
d_t y_N = +E_{e,d} u'_N ,        d_t y_S = -E_{e,d} u'_S
E = E_erode    where u' > 0   (erosion)
E = E_deposit  where u' < 0   (deposition)
```

with `u'` the near-bank *streamwise excess* over the reach mean, and
`gamma = cos theta`.

`E_e != E_d` is the **user's extension**: it makes the width `W(s,t)` prognostic.
Setting `E_e = E_d` plus the antisymmetry `u'_S = -u'_N` recovers constant width,
so **I81 and RP are both the `E_e = E_d` special case** and this is a strict
generalisation of each, not a departure from either.

WARNING: **`E_e != E_d` is NOT in I81 and has NO calibration on disk.** The
values used are *chosen*, and are reported as chosen. `E = eps C_f` is the bridge
that lets a run be quoted in either paper's units (`E = 1e-6` => `eps = 2e-5`).

Erosion is applied **only in the meander reach**.

---

## 8. Numerics and timescale separation

| | value |
|---|---|
| hydrodynamic transit `L/Ubar` | **2099.5 s ~ 35 min** |
| gravity wave `sqrt(gH)` | 3.132 m/s |
| explicit CFL on `dn` | 0.308 s (Crank-Nicolson is implicit; used only to size `dt`) |
| bank timescale | `>>` transit — set by `E`, hence the morphological factor |

Field erodibility is `E/U ~ 1e-8`, so real bank migration takes decades. A
**morphological acceleration factor** is therefore mandatory; the run reports the
implied physical time rather than hiding it.

Morphological loop: run continuously; every `N_morph` hydrodynamic steps extract
`u'_N(s), u'_S(s)`, advance the banks, rebuild the mesh coordinates and the
solver, and copy `(uv, elev)` across by **direct DOF copy** — exact because the
topology never changes, so DOFs correspond 1:1. Guards: minimum width, maximum
bank displacement per step, `theta` threshold, NaN abort.

---

## 9. Limitations, stated

1. **No helical secondary flow** (`A = 0`) — this is the incised case, not the
   alluvial one. §6.
2. **No Exner equation.** The bed is frozen; there are no bars, no bar-bend
   resonance, hence **no free alternate-bar mechanism**. The model can select a
   wavelength only through the bank instability, not through bar resonance.
3. **`E_e/E_d` uncalibrated** — chosen, not measured. §7.2.
4. **Small-amplitude planform.** The vertical-cut map is accurate to O(theta^2)
   and is asserted below 20 deg; the linear-theory predictions quoted for m=4/m=8
   are *predictions to be tested*, not guarantees — the design point sits
   **between** the SW note's two distinguished limits (`alpha` = 0.26/0.52,
   `F_c` = 3.35/1.68, `Ci` = 3.4e-3/1.4e-2, computed by
   `sw_note.epsilon_table`).
5. **Movies only** — no vortical/gravity discrimination is attempted. §3.
