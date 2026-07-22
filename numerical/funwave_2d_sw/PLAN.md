# funwave_2d_sw — plan, for approval

**Question.** `../dedalus_meander_full_SW` solves the *linear* shallow-water meander problem
with a **prescribed** bed. Its own README states the resulting limit:

> *Cannot select the observed meander wavelength. The bed is prescribed (no Exner), so the
> free alternate-bar mode that sets λ by bar–bend resonance does not exist.*

So the missing physics is **Exner + a mobile bed**, nonlinearly. This document (1) writes down
the complete system that would have to be solved, (2) asks whether FUNWAVE-TVD solves it, and
(3) answers with a verdict and a staged plan. Everything about FUNWAVE below is **read out of
the cloned source**, with `file:line`, not from memory or from the website.

Status: repo cloned, built (`SEDIMENT`+`MIXING`+`CHECK_MASS_CONSERVATION`, gnu/mpif90/double),
and smoke-tested. See §7.

---

## 1. The system to be solved

### 1.1 Hydrodynamics — depth-averaged 2-D, non-rotating

Let `h(x,y,t)` be the still-water depth (positive down, **time-dependent** once the bed moves),
`η(x,y,t)` the free surface, `H = η + h` the total depth, `(u,v)` the depth-averaged velocity,
`(P,Q) = (Hu, Hv)` the volume fluxes.

**Mass**

```
∂η/∂t + ∂P/∂x + ∂Q/∂y = − ∂h/∂t                                            (1)
```

The right-hand side is the term the Dedalus model does not have: a rising bed displaces water.
Under the standard morphodynamic scale separation (§4) it is dropped from the *hydrodynamic*
step and reinstated only when the bed is updated.

**Momentum** (conservative, well-balanced form; this is FUNWAVE's NSWE limit)

```
∂P/∂t + ∂/∂x[ P²/H + ½g(η² + 2ηh) ] + ∂/∂y[ PQ/H ]
        = gη ∂h/∂x  −  C_d u√(u²+v²)  +  ∇·(ν_t H ∇u)                      (2)

∂Q/∂t + ∂/∂x[ PQ/H ] + ∂/∂y[ Q²/H + ½g(η² + 2ηh) ]
        = gη ∂h/∂y  −  C_d v√(u²+v²)  +  ∇·(ν_t H ∇v)                      (3)
```

Splitting the pressure as `½g(η²+2ηh)` with the `gη∂h/∂x` source is what makes the scheme
well-balanced over a non-flat bed (lake-at-rest is exact). FUNWAVE's full Boussinesq form adds
`Gamma1`-scaled dispersive terms; setting `DISPERSION = F` zeroes `Gamma1 = Gamma2 = 0`
(`src/init.F:647-652`) and (2)–(3) are exactly what is integrated.

Friction as coded (`src/sources.F:242`): `− C_d u|u|`; with `-DMANNING`
(`src/sources.F:237-240`) it becomes `− g n² H^(−1/3) u|u|`.
Lateral mixing under `-DMIXING` (`src/sources.F:367-385`) is Smagorinsky,
`ν_t = C_smg Δx Δy √(u_x² + v_y² + ½(u_y+v_x)²) + ν_bkg`, `C_smg = 0.25` default
(`src/mod_global.F:301`).

**Relation to the Dedalus system.** (2)–(3) in channel-fitted `(s,n)` with `σ = 1+nC(s)` are
exactly the curvilinear equations in `../dedalus_meander_full_SW/derivations/sw_sn_meander.tex`
eq. (cont/smom/nmom); the centrifugal term `−C u_s²/σ` there is *not* a separate physical term
— in Cartesian coordinates it is carried by the advective flux `∂(PQ/H)/∂y` along a curved
channel. The two formulations are the same physics. The differences are: **linear vs nonlinear**
(Dedalus drops `u'·∇u'`), and **fixed wall vs wetting–drying waterline** (§3).

### 1.2 Boundary conditions

| | Dedalus (`(s,n)`, fixed walls) | what a mobile-bank model needs |
|---|---|---|
| bank, normal | `u_n(±b) = 0` | free waterline: `H → 0`; no-flux at the last wet face |
| bank, tangential | free slip `∂_n u_s(±b) = 0` | free slip (or a wall law) |
| inflow | none (periodic in `s`) | steady discharge `Q_in`: prescribe `u` (or `Hu`) |
| outflow | none | fixed stage `η = η_out` (or radiation) |
| sediment in | — | equilibrium load `c̄ = c̄_eq(τ_b)`, else inlet scour |
| sediment out | — | free outflow `∂c̄/∂x = 0` |
| bed | prescribed `H(n)` | Exner (1.3), plus a non-erodible floor `z_b ≤ z_s` |

The derivation note is explicit that the fixed-wall BC is only legitimate as `E → 0`
(`sw_sn_meander.tex`, §"What the model does not contain", ¶ii): at the package's inflated
`E = 0.1 m/s` the neglected wall-normal velocity `E·u_s'(±b)` is *not* small compared with the
retained `u_n'`. A mobile-boundary model removes that inconsistency by construction.

### 1.3 Bed erosion — Exner

```
(1 − n_p) ∂z_b/∂t  =  D − P  −  ∇·q_b                                       (4)
```

`n_p` porosity, `z_b` bed elevation, `P` pickup (bed → suspension, m/s), `D` deposition,
`q_b` bedload flux vector (m²/s). Closures (the ones FUNWAVE implements — verified in source):

```
τ_b/ρ = κ² |u|² / [ ln(30H/k_s) − 1 ]²,     κ² = 0.16,  k_s = 2.5 D50       (5)
        (mod_sediment.F:1254 writes it as 0.16/(1+ln(k_s/30H))², identical)

τ_cr/ρ = (s−1) g D50 θ_cr                   (mod_sediment.F:936-937)        (6)
D*     = D50 [ (s−1) g / ν² ]^(1/3)         (mod_sediment.F:935)

pickup, van Rijn (1984)     (mod_sediment.F:1287-1297)
  c_b = 0.015 [ (τ_b − τ_cr)/τ_cr ]^1.5 D*^(−0.3)
  c_a = min(1, 0.65/c_b) · c_b · D50 / (0.01 H)
  P   = max(0, c_a w_s)

deposition, Cao             (mod_sediment.F:1369-1372)
  γ = min(2, (1−n_p)/c̄),      D = γ c̄ w_s (1 − γ c̄)²

bedload, Meyer-Peter–Müller (mod_sediment.F:1321-1323)
  |q_b| = 8 (τ_b − τ_cr,b)^1.5 / [ g(s−1) ],     q_b ∥ (u,v)
```

and the suspended load is carried by

```
∂(c̄H)/∂t + ∇·( c̄H u ) = ∇·( k H ∇c̄ ) + P − D                              (7)
```

Two closures that (4)–(7) **should** contain for a meandering river and that FUNWAVE does not:

```
transverse bedload deflection on a sloping bed (Ikeda 1982 / Talmon 1995)
  q_b,n = |q_b| [ sin α  −  (1/f(θ)) ∂z_b/∂n ],   f(θ) ≈ β√θ                (8)

helical / secondary flow correction to the transport direction (Ikeda et al. 1981 `A`)
  tan δ = A H/R_c   →   the transport direction is rotated inward by δ       (9)
```

(8) and (9) are jointly what builds a **point bar**: without them a depth-averaged model has
no mechanism to hold sediment on the inner bank against gravity. See §3.

### 1.4 Bank erosion — the closure choice

Two physically distinct routes, and they are **not** interchangeable:

**(a) Fluvial bank retreat (what Dedalus uses), Ikeda–Parker–Sawai 1981:**

```
∂ζ_c/∂t = E · ½[ u_s'(+b) − u_s'(−b) ],       E = ε U,  ε ~ 1e−8           (10)
C'(s,t) = −∂²ζ_c/∂s²                     (curvature feedback closes the loop)
```

A *kinematic* law: a calibrated coefficient times the near-bank velocity excess. The channel
keeps its width; only the centreline moves.

**(b) Toe erosion + geotechnical failure (what FUNWAVE has):**

```
bank toe erodes by (4)–(7) like any other bed cell
then, wherever  |∇z_b| > tan φ :                                            (11)
   dh = ½(z_i − z_j) − ½ tanφ Δx,   z_i −= dh,  z_j += dh
applied every `Aval_interval`      (mod_sediment.F:1554-1637)
```

(b) is more fundamental — it *derives* bank retreat instead of prescribing it — but its rate is
set by `D50, θ_cr, tan φ`, not by `E`, so **the two cannot be compared without recalibration.**
This is the single most important thing to be clear about before starting.

---

## 2. Timescales — yes, three of them, and they must be separated

Computed for the Dedalus reference state (`H=3 m, W=100 m, U=1 m/s, D50=0.5 mm, s=2.68,
n_p=0.47, θ_cr=0.047, λ=1047 m`), script-checked, not asserted:

| | formula | value |
|---|---|---|
| bed drag | `C_d = κ²/[ln(30H/k_s)−1]²` | **0.00154** |
| shear | `τ_b/ρ = C_d U²` | 1.54e−3 m²/s² → `u* = 0.039 m/s`, Shields 0.187 |
| bedload | MPM (5)-(6) | `q_b = 1.9e−5 m²/s` (1.6 L/m/day) |
| **T_flow** | `CFL·Δx/(U+√(gH))`, Δx=2.5 m | **0.20 s** |
| **T_bed** (one bar, L=W) | `(1−n_p) H L / q_b` | **8.3e6 s ≈ 97 d** |
| **T_bed** (one wavelength) | same, L=λ | 8.7e7 s ≈ 2.8 yr |
| **T_bank** | `b/(εU)`, ε=1e−8 | **5e9 s ≈ 158 yr** |

**Separation: `T_flow : T_bed : T_bank ≈ 1 : 4×10⁷ : 2×10¹⁰`.**

So a bar-forming run is 4.3×10⁷ hydrodynamic steps — unaffordable. The standard treatment,
and the one FUNWAVE implements, is a **morphological acceleration factor**:

```
Depth = Depth_ini + z_b · MF          (mod_sediment.F:1642, MF integer)
```

plus a **bed-forcing averaging window** `Morph_interval` which time-averages `P` and `D`
before they are allowed to move the bed (`mod_sediment.F:1387-1400`) — a second, weaker
separation device that filters wave-scale fluctuations out of the Exner forcing.

Validity check for MF (the standard criterion is that the bed must not move appreciably within
one hydrodynamic adjustment time):

| MF | bed change per hydro step, as a fraction of H | hydro time to simulate one bar |
|---|---|---|
| 100 | 2.3e−6 | 0.97 d |
| **1000** | 2.3e−5 | **0.10 d = 8.3e3 s ≈ 4.3e4 steps** |
| 10000 | 2.3e−4 | 0.01 d |

**MF = 1000 is the working choice** — 4×10⁴ steps per bar-formation event is cheap, and the
per-step bed displacement is still 2×10⁻⁵ of the depth. MF must then be **verified by
convergence**, not trusted: run MF ∈ {250, 500, 1000, 2000} and require the final bed to be
MF-independent. This is a gate, not a formality.

The bank timescale is a further ×600 beyond the bed. It cannot be reached by MF alone. This is
why route (b) in §1.4 gives a model of **bar and bend morphodynamics**, not of century-scale
planform migration — a distinction to make in writing before any figure is produced.

---

## 3. Can FUNWAVE-TVD solve it? — verdict

**Yes for the hydrodynamics + bed. No for the two closures that make a meander a meander.**

### Has (verified in source, and built here)

- 2-D depth-averaged NSWE, conservative + well-balanced, MUSCL-TVD / HLL shock-capturing,
  wetting–drying via `MASK` at `MinDepth` — so the channel, its banks and its floodplain are
  just bathymetry, and the waterline moves on its own. No fixed-wall BC needed.
- `DISPERSION = F` ⇒ exactly (2)–(3); the whole Boussinesq apparatus switches off.
- Smagorinsky lateral mixing = the model's `ν` (`-DMIXING`).
- Steady discharge in / stage out: `TIDAL_BC_ABS = T`, `TideBcType = CONSTANT`, with
  `TideWest_U/_ETA`, `TideEast_ETA` relaxed through a sponge (`src/mod_tide.F:323-330`).
- Full sediment module: (5)–(7), Exner (4), avalanching (11), non-erodible floor
  (`Hard_bottom` + `Zs`), morphological factor, cohesive option.
- MPI-parallel, double precision. **Compiles and runs on dolma** (§7).

### Does not have — the four gaps, in order of severity

1. **No secondary (helical) flow correction, eq. (9).** Depth-averaging removes the spiral
   motion that is the *dominant* bend driver — Ikeda et al. 1981 put `A ≈ 2.89`; this model is
   the `A = 0` limit, exactly like the Dedalus package. The `UNDERTOW_U/V` term in the
   concentration equation (`mod_sediment.F:1020-1098`) is the **wave roller** undertow, and is
   identically zero in a river with no waves. There is no river spiral-flow correction.
2. **No transverse bed-slope deflection of bedload, eq. (8).** Bedload direction is
   `atan2(v,u)` and nothing else (`mod_sediment.F:1319-1323`). Combined with (1) this means the
   model has **no mechanism to form a point bar**. Gravity cannot pull grains down the
   transverse slope, so the bar–bend resonance that selects λ — the exact thing the Dedalus
   package was missing — is *still* missing. Avalanching (11) is a threshold relaxation at
   `tan φ ≈ 0.7`, not a continuous slope term; it fires only at repose angle.
3. **Cartesian rectangular grid only** (`SPHERICAL=false` ⇒ `-DCARTESIAN`); no curvilinear
   `(s,n)`. Not fatal — carve the meander into the bathymetry — but it costs resolution:
   ≥20 cells across the width ⇒ Δx ≤ 5 m ⇒ ~4×10⁵ cells for a 4-bend reach. Affordable.
4. **Sediment is closed at the open boundaries.** `FLUX_SCALAR_BC` (`mod_sediment.F:1443-1503`)
   sets the advective scalar flux to **zero on all four sides**. There is no equilibrium-inflow
   BC. Consequence: an artificial scour/deposition zone at inlet and outlet. And `PERIODIC` in
   FUNWAVE is **south–north only** — there is no streamwise-periodic reach.
   *Workaround:* make the first and last ~1 bend `Hard_bottom` (non-erodible) and analyse only
   the interior. That workaround must be **tested**, not assumed (§5, gate G2).

Also worth fixing at setup: the momentum drag `Cd` (input) and the sediment drag (5) are
**independent** in this code. Set `Cd = 0.00154` so the flow and the transport see the same
bed. Left at the example value 0.002 they disagree by 30 %.

### Bottom line

FUNWAVE-TVD **can** deliver what the Dedalus package explicitly cannot: a nonlinear,
free-waterline, mobile-bed 2-D simulation of flow in a meander bend, with bank retreat emerging
from toe erosion rather than a prescribed `E`. That is a genuine and publishable step up.

It **cannot**, as shipped, reproduce point-bar formation or bar–bend wavelength selection,
because gaps 1 and 2 remove precisely those closures. Either the scope is set accordingly, or
(8)–(9) are added to `mod_sediment.F` — both are ~40 lines and local to the bedload block, and
that is the natural "rung 2" if rung 1 works.

---

## 4. Configuration (what the first run would be)

**Decisions taken 2026-07-22 (§8): S6 committed up front; FUNWAVE's own bank law only;
finite amplitude `A ~ 1–3 W`.** Configuration below reflects them.

```
geometry     W = 2b = 100 m, H0 = 3 m, centreline y_c(x) = A sin(kx),
             k = 6e−3 1/m (λ = 1047 m ≈ 10 W, Leopold–Wolman), 4 bends ⇒ L = 4189 m
             A = 236 m = 2.36 W  ⇒ sinuosity ≈ 1.5, R_min = 1/(Ak²) = 118 m, R/W = 1.2
             floodplain 50 m each side, bank slope 1:3
grid         DX = DY = 2.5 m ⇒ Mglob ≈ 1676, Nglob ≈ 270   (~4.5e5 cells)
flow         Q = U·W·H = 300 m³/s (U = 1 m/s)
             bed slope S = C_d U²/(gH) = 5.2e−5 ⇒ 0.22 m drop over the reach
             TideWest_U = 1.0, TideWest_ETA / TideEast_ETA set to that drop
physics      DISPERSION = F ; Cd = 0.00154 ; MinDepth = 0.01 ; FroudeCap = 1.0
             MIXING on, C_smg = 0.25
sediment     D50 = 5e−4, Sdensity = 2.68, n_porosity = 0.47, WS = 0.0745 m/s
             Shields_cr = 0.055, Shields_cr_bedload = 0.047, Tan_phi = 0.7
             Bed_Change = T, BedLoad = T, Avalanche = T
             Morph_factor = 1000, Morph_interval = 60 s, Aval_interval = 60 s
             Hard_bottom = T with the inlet/outlet bend masked non-erodible
```

`WS = 0.0745 m/s` is the Soulsby (1997) settling velocity for 0.5 mm quartz,
`w_s = (ν/d)[√(10.36² + 1.049 D*³) − 10.36]` with `D* = 12.72` — **not** the 0.0125 in the
shipped `sediment_rip` example, which is for ~0.1 mm sand. To be computed and asserted in the
setup script, not typed in. Note `u*/w_s = 0.53 < 1`: this reference state is
**bedload-dominated**, so gap 2 in §3 (no transverse slope term) bites hardest exactly here.

---

## 5. Staged plan, with gates

Each stage has a **falsifiable gate**. If a gate fails the stage is not "mostly working".

- **S0 — build & smoke.** ✅ done, §7.
- **S1 — straight channel, rigid bed.** Uniform flow at `U = 1 m/s`; gate **G1**: the computed
  normal depth and the log-law `u*` match `C_d U² = g H S` to <2 %, and mass is conserved
  (`-DCHECK_MASS_CONSERVATION`). Catches every BC and drag error before sediment is on.
- **S2 — straight channel, mobile bed, equilibrium transport.** Gate **G2**: with a spatially
  uniform flow the bed must stay **flat**. Any inlet/outlet scour that appears is gap 4, and its
  streamwise extent sets how much of the domain must be masked/discarded. This is the honest
  test of the §3-4 workaround.
- **S3 — implement (8)+(9) in `mod_sediment.F`.** Spec in §6. Gate **G3a** (reduction): with
  `A_s = 0` and the slope term disabled the modified code must reproduce the unmodified S2
  result **bit-for-bit**. Gate **G3b** (straight channel): the new terms must be identically
  zero for `R_s → ∞` and a flat bed, so S1/S2 must be unchanged.
- **S4 — mild constant-curvature bend: calibration of the new closures.** `R_c = 1000 m`
  (`R/W = 10`, `dz/H = 0.32` ⇒ point bar stays **submerged**), run to morphodynamic equilibrium.
  Gate **G4** — the pre-registered number: the emergent transverse bed slope must satisfy
  `∂z_b/∂n = A·H/R_c` with **`A = 3.2 ± 0.5`**, against Ikeda's alluvial `A = 2.89`
  (`../ikeda_1981/ikeda_lib.py:82`, "Suga 1963"). §6 derives 3.23 from the two closures *before*
  the run; a result outside that band falsifies the implementation, not the theory.
  This stage is why a mild bend is needed — see the amplitude note in §6.
- **S5 — MF convergence.** MF ∈ {250, 500, 1000, 2000} on S4. Gate **G5**: the equilibrium bed
  is MF-independent to <5 %. If not, MF is reduced until it is.
- **S6 — the production run: 4-bend finite-amplitude reach** (`A = 2.36 W`, sinuosity 1.5).
  Gate **G6**: outer-bank scour phase-lagged downstream of each apex, and a point bar on each
  inner bank. At this amplitude `dz/H ≈ 2.7`, so **the bar emerges above the water surface** —
  the run is in the wetting-drying regime and chute cutoff is physically possible. Report bar
  emergence and any cutoff as results, not as failures.
- **S7 — cross-check against `../dedalus_meander_full_SW`.** ⚠️ **Only partially valid now.**
  Dedalus is linear perturbation theory about a near-straight channel; at sinuosity 1.5 it does
  not apply. So the comparison must be run at the *S4* mild bend, not at S6, and it compares the
  **flow** (bend phase lag, near-bank velocity excess) only — never the migration rate, because
  the two bank closures differ (§1.4) and are not calibrated to each other.

Deliverable per stage: one figure + one line in a results table + the honest gate verdict.
Repo conventions follow `../README.md` (no synthetic data; every number machine-checked).

## 6. S6 specification — the two closures to add

Both go in the bedload block of `SEDIMENT_ADVECTION_DIFFUSION`
(`mod_sediment.F:1319-1323`), which currently reads `Angle_cur = ATAN2(V,U)` and nothing else.

**(9) Secondary-flow deflection.** The near-bed velocity is rotated toward the inner bank by

```
tan δ = A_s · H / R_s ,        A_s = (2/κ²)(1 − √g /(κ C_z)) ,   C_z = √(g/C_d) ,  κ = 0.4
1/R_s = [ u(u ∂_x v + v ∂_y v) − v(u ∂_x u + v ∂_y u) ] / |u|³            (streamline curvature)
```

so the transport direction becomes `Angle_cur = ATAN2(V,U) − δ·sign(1/R_s)`.

**(8) Transverse bed-slope deflection.** Superposed on the above,

```
q_b,n  ←  q_b,n − (|q_b|/f(θ)) ∂z_b/∂n ,      f(θ) = 9 (D50/H)^0.3 √θ      (Talmon et al. 1995)
```

implemented as a vector correction to `(BedFluxX, BedFluxY)` so it also acts in the streamwise
direction on a longitudinal slope.

**Pre-registered consistency check (this is gate G4).** At morphodynamic equilibrium in a long
constant-curvature bend, `q_b,n = 0`, so (8) and (9) balance:

```
∂z_b/∂n = f(θ) · A_s · H/R_c   ≡   A · H/R_c        ⇒   A = f(θ)·A_s
```

which is *exactly* Ikeda's bed closure `η'/H = −A·C·n`. For the reference state
(`θ = 0.187, D50/H = 1.67e−4, C_z = 79.8`):

```
A_s = 11.27 ,   f(θ) = 0.286   ⇒   A = 3.23     vs   Ikeda's 2.89   (+11.6 %)
```

**A 12 % agreement with a parameter Ikeda took from field data, with no fitting, is the whole
justification for adding these two terms.** It is computed here *before* implementation so the
gate cannot be tuned after the fact.

⚠️ **Provenance to pin before coding.** `f(θ) = 9(D50/H)^0.3 √θ` (Talmon et al. 1995) and
`A_s = (2/κ²)(1 − √g/(κC_z))` (Rozovskii / Engelund) are written from working knowledge. Per the
repo house rule ("transcribe + independently re-derive"), both must be **read out of a PDF on
disk** and re-derived before they enter `mod_sediment.F`. Neither is in `../literature/` yet.
If the transcription changes the constants, `A = 3.23` changes with it — and the gate must be
re-registered *before* S4 is run, not after.

**Amplitude constraint (why S4 is a mild bend and S6 is not a validation).** The equilibrium
slope above predicts the transverse bed drop across the width, `dz = A·H·W/R_min`, and the point
bar emerges once `dz > H`. For a sine centreline `R_min = 1/(Ak²)`:

| A/W | λ=1047 m | λ=2000 m | λ=3000 m |
|---|---|---|---|
| 0.5 | dz/H = 0.58 | 0.16 | 0.07 |
| 1.0 | **1.16 ✗** | 0.32 | 0.14 |
| 2.36 (chosen) | **2.7 ✗** | 0.76 | 0.34 |

So at `λ ≈ 10 W` a genuinely finite-amplitude meander (sinuosity 1.5) **necessarily** has an
emergent point bar. That is physically correct — real meanders do — and it plays to FUNWAVE's
actual strength (robust wetting–drying + shock capturing; it is a surf-zone/tsunami code). But
it puts S6 outside the regime where the equilibrium-slope relation holds, which is why the
closures must be calibrated at S4's mild bend first.

---

## 7. Build status (done, verified)

```
clone   FUNWAVE-TVD/                 (github.com/fengyanshi/FUNWAVE-TVD, depth 1)
config  FUNWAVE-TVD/Makefile.river   gnu / mpif90 / PARALLEL / double
                                     -DSEDIMENT -DMIXING -DCHECK_MASS_CONSERVATION
exe     work_river/funwave-SEDIMENT-CHECK_MASS_CONSERVATION-MIXING--mpif90-parallel-double
smoke   simple_cases/sediment_rip, 20 s, 2 ranks -> "Normal Termination!",
        sediment outputs written (C, Depo, BedStr, BedFx/y, Aval, AvalAc, DchgS/B, dep)
```

`double` matters: at MF=1 the bed moves 7e−8 m per step, which single precision would lose in
`Depth = Depth_ini + Zb*Morph_factor`.

Rebuild: `make -f Makefile.river` — **serially**. `-j` races on Fortran `.mod` files because
`Make_Essential` compiles from a flat file list with no declared module dependencies.

Harmless noise on dolma: `Read -1, expected …, errno = 1` from OpenMPI's CMA single-copy path
over NFS. Silence with `--mca btl_vader_single_copy_mechanism none`.

---

## 8. Decisions (taken 2026-07-22)

1. **Scope → commit to the two closures up front.** (8) and (9) are implemented in
   `mod_sediment.F` before any bend run; spec in §6. Bar–bend resonance is therefore on the
   table from the start. Cost: we now own modified third-party Fortran, so S3's reduction gate
   (G3a, bit-for-bit with the terms off) is mandatory, not optional.
2. **Bank law → FUNWAVE's toe erosion + avalanching only.** No Ikeda `E` law is bolted on.
   Consequence, to be stated in writing: bank retreat is *derived*, its rate is set by
   `D50, θ_cr, tan φ`, and it is **not comparable** with the Dedalus migration rate.
3. **Amplitude → finite, `A = 2.36 W` (sinuosity ≈ 1.5).** Consequences (§6): the point bar
   emerges above the water surface (`dz/H ≈ 2.7`), so (a) validation of the new closures moves
   to a separate mild bend (S4, `R/W = 10`), and (b) the Dedalus cross-check moves to S4 too and
   is flow-only — at sinuosity 1.5 the linear theory does not apply.

**Still open — the one thing blocking S3.** `f(θ)` and `A_s` in §6 are written from working
knowledge and are not yet pinned to a PDF on disk (`../literature/` has neither Talmon et al.
1995 nor Rozovskii/Engelund). The repo rule is transcribe-then-re-derive. Either those two
sources get added to `../literature/` first, or S3 proceeds with the constants flagged
`PROVENANCE: UNPINNED` in the source and gate G4's `A = 3.2 ± 0.5` is treated as provisional.
