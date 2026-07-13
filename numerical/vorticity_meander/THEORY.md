# The vorticity-meander equations, and how they differ from Ikeda's Eq. (7)

Reference document for the rebuilt `vorticity_meander` package (v2, 2026-07-12) and
for the planned 2-D numerical model. Everything here is re-derived from the 6/30 deck
(`literature/Rossby_Palooza_meet_0630.pdf`, cited by page) and machine-verified in
[`vorticity_lib.py`](vorticity_lib.py).

## 1. The reduction: 3-D river → 2-D vorticity dynamics

Two steps, sharing the first with Ikeda et al. (1981):

**Step 1 — depth averaging (3-D → St Venant).** Integrate the 3-D equations from bed to
surface under shallow-water geometry H ≪ b ≪ λ (hydrostatic pressure). The velocity
becomes the depth-averaged (u, v); the vertical shear ∂u/∂z survives only inside the
bottom-drag closure τ = ρ C_f |**u**| **u**. Ikeda additionally parameterises the
helical secondary flow (a ∂u/∂z effect) through his transverse-bed closure A.

**Step 2 — low-Froude reduction (St Venant → 2-D vorticity), deck p. 3.** Alluvial
rivers are slow, 0.1 < Fr < 0.3 (Getirana et al. 2012). Fr² ≪ 1 slaves the free
surface (rigid lid): ∇·(H**u**) ≈ 0 with H ≈ const, so a streamfunction ψ exists and
the curl of the momentum equation eliminates pressure. The result is 2-D
(barotropic-vorticity) dynamics — the same reduction that takes the primitive
equations to the barotropic vorticity equation in the atmosphere, which is the deck's
"Alluvial Rivers & Jet streams" analogy.

What "vorticity gradient" means after the reduction: the only vorticity left is the
**vertical component of the depth-averaged flow, ζ = ∂v/∂x − ∂u/∂y**. Its
cross-channel gradient ∂ζ̄/∂y is the β-analogue. It is *not* ∂u/∂z — that was
integrated out in Step 1 and lives on only inside C_f (and Ikeda's A).

## 2. The new governing equations ("our formula")

**Base state (deck p. 4).** A maintained parabolic jet (observed shape:
Bahmanpouri et al. 2022, deck p. 3):

    ū(y) = U₀ + (Δ/b²)(b² − y²),   ζ̄ = −ū_y = (2Δ/b²) y,
    ∂ζ̄/∂y = −ū_yy = 2Δ/b² = const > 0        (the channel β)

[FLAG_BASE: the jet is externally maintained; friction acts on perturbations only.]

**Interior (linearised vorticity equation, continuum form):**

    (∂/∂t + ū(y) ∂/∂x) ∇²ψ′ + (2Δ/b²) ∂ψ′/∂x = F[ψ′]          (deck p. 4)

with F the interior friction closure — the deck does not print it; two readings are
implemented:

    rayleigh (deck-literal, v1 FLAG_FRICTION):  F = −(C_f ū_c/H) ζ′
    momentum (Ikeda-consistent):                F = curl_z[−(C_f/H)|**u**|**u**]′
             = −(C_f/H)[2 ∂_y(ū u′) − ū ∂_x v′] → streamwise drag 2C_f ū u′/H,
             cross-stream C_f ū v′/H  (the |u|u linearisation's factor 2,
             exactly as in Ikeda's Eq. 3b)

**Boundary (the erodible bank, deck p. 7 — the engine):**

    ∂ψ′/∂t |_{y=±b} = (ε C_f U₀/b) (ψ′_centre − ψ′|_{y=±b})

Rigid banks (ε = 0) ⇒ ψ′ = 0 at walls, and the parabolic jet has **no inflection
point** ⇒ neutral by Rayleigh's criterion (verified). The instability is not shear
instability: it is a **boundary–interior cooperation** unlocked by bank erodibility.

**Nondimensionalisation** (lengths b, speeds U₀+Δ, time b/(U₀+Δ); FLAG_TSCALE):
k* = kb, D = Δ/(U₀+Δ), γ = C_f b/H, E = εC_f U₀/(U₀+Δ) = ECOEF·(1−D).

**Three-level closure (the deck's reduction, p. 4).** Levels ψ₁(+b), ψ₂(0), ψ₃(−b),
sinuous symmetry ψ̂₁ = ψ̂₃, 3-point Laplacian ζ̂₂ = 2ψ̂₁ − (2+k*²)ψ̂₂. With
W = −iω*:

    centre:  (W + ik* + γ) ζ̂₂ + 2iD k* ψ̂₂ = 0                    (rayleigh)
             (W + ik*) ζ̂₂ + 2iD k* ψ̂₂ + γ[2(2ψ̂₁−2ψ̂₂) − k*²ψ̂₂] = 0  (momentum)
    bank:    (W + E) ψ̂₁ = E ψ̂₂

    det M = 0:  (2+k*²) W² + A₁ W + E[k*²(ik*+γ) − 2iDk*] = 0
    A₁(rayleigh) = (2+k*²)(ik*+γ+E) − 2iDk* − 2E
    A₁(momentum) = A₁(rayleigh) + 2γ            (the ONLY difference)

**Exact consequences (all machine-verified):**

| statement | result |
|---|---|
| forced steady, γ=0 (both closures) | ψ̂₂/ψ̂₁ = 2/(2+k*²−2D); interior amplified iff k*²<2D (p. 5 box, exact); no pole for D<1 |
| rigid banks | neutral (no inflection point), both closures |
| small-E growth, γ=0 | σ/E → (2D−k*²)/(2+k*²−2D): growth band k*² < 2D |
| k*→0 phase speed | c* → −ED/γ (rayleigh), −ED/(2γ) (momentum): **UPSTREAM** |
| momentum flux (p. 6) | ∂_y(u′v′)‾ = (γ/2D) ζ₂′²‾ > 0 at the centre: wave carries downstream momentum from the jet core to the banks (zero without friction) |

## 3. Ikeda's Eq. (7), for contrast

Ikeda, Parker & Sawai (1981) stay at Step 1 (St Venant + closures) and reduce
everything to one ODE along the channel for the near-bank excess velocity
u_b′ = u′(ñ=b), forced by centreline curvature 𝒞′(s̃):

    U ∂u_b′/∂s̃ + 2 (U/H) C_f u_b′ =
        b [ −U² ∂𝒞′/∂s̃  +  C_f 𝒞′ ( U⁴/(gH²) + A U²/H ) ]
            ①free vortex        ②superelevation   ③A-scour

LHS: downstream advection + friction relaxation over length H/(2C_f) (the factor 2 is
the |u|u linearisation — our "momentum" closure). RHS: ① the potential-vortex
response (inner bank speeds up, no lag); ② the transverse surface tilt making the
outer column deeper hence faster (∝F²); ③ the secondary-flow bed scour (A ≈ 2.89) —
for alluvial rivers A/F² ≈ 30: ③ dominates. Because ②③ act through the first-order
relaxation operator, u_b′ is an exponentially-weighted convolution of **upstream**
curvature and peaks ≈ 0.18λ **downstream** of each apex; erosion follows u_b′, so
bends grow while migrating **downstream**: ω₀ = C_f k³(2+A+F²)/(k²+4C_f²) > 0 for all
k (their Eq. 18) — explicitly correcting Ikeda et al. (1976)'s upstream heuristic.

## 3½. Term-by-term ledger: Ikeda (7) *is* a vorticity equation

The comparison becomes exact once (7) is recognised as the **curl of Ikeda's momentum
pair**: ∂/∂ñ(3b) − ∂/∂s̃(3a) eliminates the pressure ξ′ and, with closures (5)–(6),
reproduces (7) verbatim. Since his forcings are all ∝ ñ, u′ = u_b′·(ñ/b) and
u_b′/b = ∂u′/∂ñ; define his perturbation (shear) vorticity ζ′_I ≡ −∂u′/∂ñ = −u_b′/b
and divide (7) by −b:

    U ∂ζ′_I/∂s̃ + 2C_f(U/H) ζ′_I  =  U²∂𝒞′/∂s̃ − C_f𝒞′U⁴/(gH²) − C_f𝒞′AU²/H
      advection    friction            ①free-vortex   ②superelevation   ③A-scour

against ours:  ∂ζ′/∂t + ū(y)∂ζ′/∂x + (2Δ/b²)∂ψ′/∂x = F[ψ′],  ζ′ = ψ′_yy − k²ψ′.

| term | Ikeda (7) | ours | note |
|---|---|---|---|
| (A) ∂ζ′/∂t | ✗ (quasi-steady) | ✓ **extra** | the W = −iω* terms; permits free waves, hence c* < 0 |
| (B) v′∂ζ̄/∂y = (2Δ/b²)ψ′_x | ✗ (**identically zero**: plug flow ζ̄ ≡ 0) | ✓ **extra** | the 2iDk*ψ̂₂ term; resonance band k*²<2D + upstream retrogression |
| (B′) v′_x share of ζ′ (−k²ψ′) | ✗ (only −∂u′/∂ñ carries dynamics; v′ diagnostic, never feeds back) | ✓ **extra** | the k*² in ζ̂₂ = 2ψ̂₁−(2+k*²)ψ̂₂; makes the operator elliptic (upstream influence) vs his parabolized 1st-order-in-s̃ |
| advection U∂ζ′/∂s̃ | ✓ (single U) | ✓ (ū(y): differential) | shared |
| friction 2C_f(U/H)ζ′ | ✓ (factor 2 from \|u\|u) | momentum closure reproduces it verbatim in the plug-flow limit; rayleigh reading lacks the factor 2 | shared; the microscopic origin of A₁'s +2γ and the E↔2E degeneracy |
| ① U²∂𝒞′/∂s̃ | ✓ | moved into the bank BC (curvature forcing lives on the boundary) | relocated, not deleted |
| ② C_f𝒞′U⁴/(gH²) | ✓ | ✗ **deleted** (rigid lid, F²→0) | deck model lacks |
| ③ C_f𝒞′AU²/H | ✓ (**his dominant term**, A/F²≈30) | ✗ **deleted** (flat rigid bed) | deck model lacks; numerical model's sediment feedback restores it |
| h′ in continuity | ✓ | ✗ (rigid lid ⇒ ψ′ exists) | with ② |
| erosion law | γ∂y/∂t = E u_b′ (feeds 𝒞′) | ∂ψ₁′/∂t = (εC_fU₀/b)(ψ₂′−ψ₁′) | both present, different form |

Degenerate-limit check (a sanity test for the numerical model): set D = 0, ∂t = 0 and
drop the −k²ψ′ share of ζ′ in our system ⇒ the operator collapses to Ikeda's
U∂/∂s̃ + 2C_f U/H; adding back sources ②③ recovers (7) exactly. The two theories are
the same parent (depth-averaged momentum + continuity, curled), keeping different terms.

## 4. The structural differences

| | Ikeda Eq. (7) | vorticity-meander equations |
|---|---|---|
| state variable | u_b′(s̃): 1-D, near-bank, flow **diagnostic** (quasi-steady, slaved to 𝒞′) | ψ′(x, y, t): 2-D, flow **prognostic** (∂ζ′/∂t retained) |
| base flow | plug (reach-averaged U; ζ̄ ≡ 0) | parabolic jet; β-analogue ∂ζ̄/∂y = 2Δ/b² |
| vorticity-gradient physics | structurally absent | the core: channel-Rossby wave, resonant band k*² < 2D |
| spatial operator | 1st order in s̃ (parabolized): influence strictly downstream | elliptic ∇²ψ′: induced flow reaches upstream |
| bed / secondary flow | algebraic slaved closures (5)–(6): superelevation F², A-scour | none yet (rigid flat bed; the numerical model's sediment feedback will resolve it) |
| bank law | γ ∂y/∂t = E u_b′ (erosion ∝ near-bank speed) | ∂ψ₁′/∂t = (εC_f U₀/b)(ψ₂′−ψ₁′) (bank relaxes toward interior) |
| instability engine | friction phase-lag + A-scour | interior Rossby resonance + erodible-bank relaxation |
| migration | downstream, always (c₀ > 0 ∀k) | upstream in the growth band (c* → −ED/γ_eff as k*→0) |
| momentum redistribution | via parameterised secondary flow (A) | resolved: u′v′ carries core momentum to the banks (∝γ) |

Note what each model *lacks*: Ikeda has no interior vorticity dynamics; the deck model
has no superelevation, no secondary flow, no bed evolution — i.e. it deletes Ikeda's
*dominant* term ③ and replaces the engine entirely. The planned numerical model
(deck p. 8 goals) is the arbiter that contains both.

## 5. Rebuild-v2 finding: the friction closure alone does not explain the peak gap

Both closures calibrated on the same six deck-p.8 phase intercepts (exact E↔2E
degeneracy: ECOEF = 0.5 rayleigh / 1.0 momentum, per-curve spread < 3%):

| closure | deck σ_pk / model σ_pk | peak k* shift |
|---|---|---|
| rayleigh (deck-literal) | 3.18 – 4.42 | +0.09 … +0.19 |
| momentum (Ikeda-consistent) | 2.28 – 3.22 | +0.13 … +0.22 |

The momentum closure raises the peaks ~40% (closes ≈ one third of the log-gap) while
leaving phases and zero crossings intact — so **the interior friction closure cannot
by itself reproduce the deck's peak heights**. Remaining suspects, for the deck's
author and the numerical model: (a) the p. 6 "pressure drag due to wavy banks" that
never enters the p. 7 bank equation; (b) the growth-axis normalisation (FLAG_TSCALE).
Resolution is excluded (N-point solver converges to the closure's peaks by N ≈ 11).
