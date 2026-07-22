# Source specification for `deliverable1_noboru_model`

**This document is the specification the code is written FROM.** Every symbol, equation and
numerical target in the package traces to a page here. Anything river.pdf does not define is
listed under [NOT DEFINED IN SOURCE](#not-defined-in-source) and is marked `[NOT IN DECK]` at its
use site in the code — never silently filled in.

Produced by `/ars-lit-review` + `/ars-citation-check`, 2026-07-21.

---

## 0. Sources and provenance

| file | what it is |
|---|---|
| `literature/river.pdf` | **THE AUTHORITY.** 21-page Keynote deck, *"Meanders of alluvial rivers as forced Rossby waves — Kinship between the Mississippi and the jet stream(?)"*. PDF `/Title` = `NoonBalloon2026`, `/Producer` = Quartz PDFContext, `/Creator` = **Keynote**, `/CreationDate` = 2026-07-20. |
| `literature/Rossby_Palooza_meet_0630.pdf` | **Superseded, and second-hand.** 8 pages. `/Title` = `Rossby Palooza`, `/Author` = **`yanxi`**, body carries OneNote page markers (`Rossby Palooza Page 1`) and the timestamp `2026年6月29日 17:06` — this is *a meeting capture made by the reader*, not the presenter's file. Its p.8 is byte-for-byte the same figure as river.pdf p.20. Cite only to note what it left unprinted (the friction closure). |

Because the 6/30 file is a screenshot record and river.pdf is the Keynote original, **river.pdf wins
every disagreement.**

### river.pdf page map

| p. | title | carries |
|---|---|---|
| 1 | (title) | — |
| 2 | Wavelength of meandering | `λ ≈ 10 − 14 × width`; Schumm 1967; bullets *Discharge* / *Sediment load* |
| 3 | Bend instability theory (Ikeda et al. 1981) | "Erodible banks amplify sinusoidal meander at certain wavelengths" |
| 4 | Bend instability theory (Ikeda et al. 1981) | `λ_max ≈ (8π b H₀/C_f)^{1/2}`; defines `b`, `H₀`, `C_f`; "Based on St Venant equations" |
| 5 | Bend instability theory (Ikeda et al. 1981) | `ω₀ = C_f k³(2+A+F²)/(k²+4C_f²)`, `c₀ = ω₀/k`; defines `k`, `C_f`, `F`, `A`; "It also predicts downstream migration" |
| 6 | Recasting to vorticity dynamics | `F_r = U₀/√(gH₀)`; `0.1 < F_r < 0.3`; Getirana et al. 2012 |
| 7 | Recasting to vorticity dynamics | "Flow curvature sustains a background vorticity gradient"; Bahmanpouri et al. 2022 |
| 8 | Recasting to vorticity dynamics | `Ro = U₀/(f₀L)`; `Ro ≪ 1` |
| **9** | Recasting to vorticity dynamics | **schematic + the three `ψⱼ` definitions + `ζ'₂` box + `ψ̂₁ = ψ̂₃` box + `ū(y)` + the `2b`/`H` cross-section** |
| **10** | Recasting to vorticity dynamics | **+ "Linearized vorticity equation" — the governing PDE, printed** |
| 11 | Recasting to vorticity dynamics | + "In steady state, this comes down to `ψ̂₂ = f(ψ̂₁)`" |
| 12 | Forced steady state | `k*=0.3, D=0.5, γ=0.0` / `k*=1.5, D=0.5, γ=0.0`; **the sidebar parameter box first appears** |
| 13 | Forced steady state | `k*=0.3, D=0.5` / `k*=0.3, **D=0.1**`, both `γ=0.0` |
| **14** | Forced steady state | `k*=0.3, D=0.5, γ=0.0` with `ψ₁/ψ₂/ψ₃` labelled; box **`\|ψ̂₂\| > \|ψ̂₁\| if k*² < 2D`**; "There is no resonance for k\*>0 even in the inviscid limit." |
| 15 | Forced-dissipative steady state | `γ=0.0` vs **`γ=0.1`** |
| **16** | Forced-dissipative steady state | **`∂/∂y u'v'‾ ≈ −v'₂ζ'₂‾ = (Dγ/2b) ζ'₂²‾ > 0`**; "…AND by the **pressure drag due to wavy banks**"; `u'v'` arrows |
| 17 | (flume photograph) | — |
| 18 | Forced-dissipative steady state | `u'v'` panel + flume photograph |
| **19** | Bank erosion | "UPSTREAM propagation"; **`∂ψ'₁/∂t = (εC_fU₀/b)(ψ'₂ − ψ'₁)`**; `M(ω)[ψ̂'₁;ψ̂'₂] = 0 ⇒ det M = 0` |
| **20** | Bend instability revisited | **the 2×2 dispersion figure** |
| 21 | Bend instability revisited | summary bullets |

---

## 1. The three governing equations, verbatim

All three verified word-for-word against the pages cited (`/ars-citation-check`, 2026-07-21).

```
ū(y) = −∂ψ̄/∂y = U₀ + (Δ/b²)(b² − y²)                                        [p.9]

ψ₁(x,t) = ψ̄(b)  + ψ̂₁ exp[i(kx − ωt)]
ψ₂(x,t) = ψ̄(0)  + ψ̂₂ exp[i(kx − ωt)]                                        [p.9]
ψ₃(x,t) = ψ̄(−b) + ψ̂₃ exp[i(kx − ωt)]

ζ'₂(x,t) = ∇²ψ' ≈ [ (ψ̂₁ + ψ̂₃ − 2ψ̂₂)/b²  −  k²ψ̂₂ ] exp[i(kx − ωt)]           [p.9]

ψ̂₁ = ψ̂₃                                                                     [p.9, 10, 11]

[ ∂/∂t + (U₀+Δ) ∂/∂x ] ζ'₂ + (2Δ/b²) ∂ψ'₂/∂x  =  − C_f (U₀+Δ)/H · ζ'₂       [p.10]

∂ψ'₁/∂t = (ε C_f U₀ / b) (ψ'₂ − ψ'₁)                                        [p.19]

M(ω) [ψ̂'₁; ψ̂'₂] = 0   ⇒   det M = 0                                         [p.19]
```

Three points that were easy to get wrong and are settled here:

1. **Advection is at the constant centreline speed `(U₀+Δ)`**, not at `ū(y)`. p.10 prints it.
2. **Friction is Rayleigh damping of `ζ'₂` with coefficient `C_f(U₀+Δ)/H`.** It is printed, so
   there is no closure ambiguity and no second variant to implement.
3. **The `/b²` in `ζ'₂` is printed** and is kept visible in the code by carrying `b = 1.0` as a
   named constant.

---

## 2. Notation table — symbol → code

| printed | first page | meaning as the deck states it | Python identifier | dim? |
|---|---|---|---|---|
| `ψ₁, ψ₂, ψ₃` | p.9 | streamfunction at `y = +b, 0, −b` | `psi1 psi2 psi3` — **hold the perturbation `ψ'ⱼ`** | nondim |
| `ψ̂₁, ψ̂₂, ψ̂₃` | p.9 | complex modal amplitudes | `psi1_hat, psi2_hat` | nondim |
| `ψ̄` | p.9 | mean streamfunction; `ū = −∂ψ̄/∂y` | `psibar` — **never evolved** | dim |
| `ψ'` | p.9 | perturbation streamfunction, inside `ζ'₂ = ∇²ψ'` | *the three `psiN` **are** `ψ'`* | nondim |
| `ζ'₂` | p.9 | `= ∇²ψ' ≈ (ψ̂₁+ψ̂₃−2ψ̂₂)/b² − k²ψ̂₂` | `zeta2` | nondim |
| `ū(y)` | p.9 | `= U₀ + (Δ/b²)(b²−y²)`; nondim `1 − D y²` | `ubar(y, cfg)` | dim |
| `b` | p.9 | half-width; cross-section labelled `2b` | `b = 1.0` (the length unit) | dim |
| `H` | p.9 | depth, labelled on the cross-section | `H` | dim |
| `U₀` | p.9 | bank-edge speed, `ū(±b) = U₀` | `U0` | dim |
| `Δ` | p.9 | centreline excess, `ū(0) = U₀+Δ` | `Delta` | dim |
| `k` | p.9 | wavenumber | `k` | dim |
| `ω` | p.9 | frequency in `exp[i(kx−ωt)]` | `omega` | dim |
| `y = +b, 0, −b` | p.9 | the three levels | `(+b, 0, −b)` | dim |
| `C_f` | p.4 | "frictional drag coefficient" | `Cf` | nondim |
| `H₀` | p.4 | "mean depth" *(Ikeda context only)* | — | dim |
| `λ` | p.12 sidebar | `λ = 2π/k` | `lam` | dim |
| `k*` | p.12 sidebar | `k* = kb` | `kstar` | **nondim** |
| `D` | p.12 sidebar | `D = Δ/(U₀+Δ)` | `D` | **nondim** |
| `γ` | p.12 sidebar | `γ = C_f b/H` | `gamma` | **nondim** |
| `u'v'‾` | p.16 | only `∂_y u'v'‾ ≈ −v'₂ζ'₂‾ = (Dγ/2b)ζ'₂²‾ > 0` is printed | `momentum_flux()` | nondim |
| `v'₂` | p.16 | used, never defined — read as `∂ₓψ'₂` | `v2_of()` | nondim |
| `ε` | p.19 | appears **only** inside `ε C_f U₀ / b` | `eps_Cf` (the product) | nondim |
| `M(ω)` | p.19 | `M(ω)[ψ̂'₁;ψ̂'₂] = 0 ⇒ det M = 0` | `dispersion_roots()` — **reconstruction** | — |
| `F_r` | p.6 | `= U₀/√(gH₀)`; `0.1 < F_r < 0.3` | — | nondim |
| `Ro` | p.8 | `= U₀/(f₀L)`; `Ro ≪ 1` | — | nondim |
| `A`, `F`, `ω₀`, `c₀`, `λ_max` | pp.4–5 | Ikeda background only | — | — |

**Derived, and stated in the code where it is used:** `E = ε C_f U₀/(U₀+Δ) = eps_Cf·(1−D)` — the
p.19 erosion rate in the nondimensional time unit.

### The nondimensionalisation is *implied*, not free

Lengths in `b`, speeds in `U₀+Δ`, time in `b/(U₀+Δ)`. The deck never states the time unit — but it
is not a free choice: p.10's friction coefficient `C_f(U₀+Δ)/H` becomes `C_f(U₀+Δ)T/H` under a time
scale `T`, and that equals the sidebar's `γ = C_f b/H` **only if `T = b/(U₀+Δ)`**. The deck's own
two statements pin it exactly.

*Consequence:* the normalisation of p.20's "Nondimensional growth rate" axis is therefore **not** a
free parameter, which eliminates it as an explanation for the p.20 peak-height gap (see §5).

### Why `b`, `H` and `U₀` are not configuration parameters

They never appear individually — only through the sidebar groups:

- `b` enters only through `k* = kb` and `γ = C_f b/H`
- `H` and `C_f` enter only through `γ = C_f b/H`
- `U₀` and `Δ` enter only through `D = Δ/(U₀+Δ)`

So the complete physical parameter set is `{k*, D, γ}` **plus one more the deck never prints**:
the bank erodibility, carried as `eps_Cf = ε C_f`. Four numbers, no more.

---

## NOT DEFINED IN SOURCE

Never silently fill these in. Each is marked `[NOT IN DECK]` at its use site.

| item | status |
|---|---|
| `ε` | No definition and **no numerical value anywhere in 21 pages**. Appears only in the product `εC_fU₀/b` (p.19). The package's `eps_Cf = 0.5` is **calibrated**, not cited — see §5. |
| `u'`, `v'` individually | Only `u'v'‾` and `v'₂` appear (p.16). No expression for `u'` at the banks exists anywhere; the bank signs are arrow annotations on pp.16/18/19. |
| `v'₂` | Used in the p.16 formula, never defined. Read as `∂ₓψ'₂`. |
| time normalisation of p.20's axes | Derivable (above), but never stated. |
| entries of `M(ω)`, explicit `det M = 0` | p.19 prints only the schematic. The quadratic in the code is a **reconstruction**. |
| `f` in `ψ̂₂ = f(ψ̂₁)` | p.11 names it and never writes it. Reconstructed. |
| `g`, `f₀`, `L` | Used in the `F_r` and `Ro` definitions (pp.6, 8), never given. |
| **any growth band / stability criterion** | The deck states none. `k*² < 2D` is p.14's *forced-steady amplification* criterion, **not** a stability boundary. |
| **any time integration** | There is none — all 21 pages are normal-mode or steady-state. The IVP itself is this package's contribution. **But the initial condition is not a free choice**: pp.12–18 + p.11 define the forced steady state, and pp.17–18 show it physically as a rigid carved wavy channel. That state — wavy banks, interior slaved via `ψ̂₂ = f(ψ̂₁)` — is where the integration starts, and releasing the banks (p.19) is what makes it an IVP. |

---

## 3. Annotated bibliography

Every field below was read verbatim from the on-disk PDF. No bibliographic detail is supplied from
memory; where a DOI is not printed in the file, that is stated.

**Ikeda, S., Parker, G., & Sawai, K. (1981).** *Bend theory of river meanders. Part 1. Linear
development.* **Journal of Fluid Mechanics 112, 363–377.** DOI **10.1017/S0022112081000451**.
> The theory river.pdf sets out to revisit. p.3 quotes its premise ("erodible banks amplify
> sinusoidal meander at certain wavelengths"); p.4 quotes `λ_max ≈ (8πbH₀/C_f)^{1/2}` and
> reproduces its Figure 4; p.5 quotes its eq (18) `ω₀ = C_f k³(2+A+F²)/(k²+4C_f²)`, `c₀ = ω₀/k`.
> **Verified verbatim** against the source: the same passage states "bends always migrate
> downstream, whether stable or unstable, and regardless of Froude number… a correction of the
> heuristic conclusion reached in Ikeda et al. (1976)", which is exactly river.pdf p.5's claim and
> the foil for river.pdf's own *upstream* result.
> ⚠ The constant `8π` in the p.4 formula could **not** be confirmed: the source's eq (30) text
> layer is OCR-garbled (`k,M = ac? 1'`). Dimensionally sound (`bH₀/C_f` has units L²) and
> consistent with Figure 4 being a 1:1 test "in metres". Background only; not part of the model.

**Parker, G., Sawai, K., & Ikeda, S. (1982).** *Bend theory of river meanders. Part 2. Nonlinear
deformation of finite-amplitude bends.* **Journal of Fluid Mechanics 115, 303–314.** DOI
**10.1017/S0022112082000767**.
> Present in `literature/` but **not cited on any river.pdf slide.** Listed for completeness.

**Schumm, S. A. (1967).** *Meander Wavelength of Alluvial Rivers.* **Science 157(3796), 1549–1550.**
DOI **10.1126/science.157.3796.1549**.
> river.pdf p.2, for the finding that meander wavelength depends on discharge **and** sediment
> load — which the slide's bullets ("Discharge", "Sediment load") reflect faithfully.
> ⚠ **The `λ ≈ 10–14 × width` figure on the same slide is not Schumm's result.** Schumm's paper
> gives `λ = 1890 Qm^0.34/M^0.74` and `λ = 234 Qma^0.48/M^0.74` and explicitly cautions that "the
> simple correlation of meander wavelength with channel width will be very good, but both are
> closely related to discharge and type of sediment load". The width multiple is the
> Leopold–Wolman scaling.

**Bahmanpouri, F., Eltner, A., Barbetta, S., Bertalan, L., & Moramarco, T. (2022).** *Estimating the
average river cross-section velocity by observing only one surface velocity value and calibrating
the entropic parameter.* **Water Resources Research 58, e2021WR031821.** DOI
**10.1029/2021WR031821**.
> river.pdf p.7: the observed parabolic cross-channel velocity profile that justifies the base jet
> `ū(y) = U₀ + (Δ/b²)(b²−y²)`, and hence the **constant** background vorticity gradient
> `∂ζ̄/∂y = 2Δ/b²` — the channel's β-analogue, and the reason the deck calls these Rossby waves.

**Getirana et al. (2012)** — cited on river.pdf p.6 for the Amazon Froude-number map
(`0.1 < F_r < 0.3`). **Not present in `literature/`**; not verifiable here.

---

## 4. Numerical targets for `postprocessing/03_verify.py`

### Forced steady `|ψ̂₂/ψ̂₁|`

At `γ = 0` these equal the p.14 identity `2/(2 + k*² − 2D)` exactly; the `γ = 0.1` row is the only
one with a phase tilt, which is the p.16 momentum flux.

| `k*` | `D` | `γ` | `\|ψ̂₂/ψ̂₁\|` | phase | slide |
|---|---|---|---|---|---|
| 0.3 | 0.5 | 0.0 | **1.8349** | 0° | p.12 top, p.13 top, p.14 |
| 1.5 | 0.5 | 0.0 | **0.6154** | 0° | p.12 bottom |
| 0.3 | **0.1** | 0.0 | **1.0582** | 0° | p.13 bottom |
| 0.3 | 0.5 | **0.1** | **1.6297** | **+14.15°** | p.15 bottom, p.16, p.18, p.19 |

p.13's entire point is the first-vs-third row: `k*² = 0.09` against `2D = 1.0` versus `2D = 0.2`
→ strongly amplified (1.835) versus barely amplified (1.058).

### p.20 dispersion figure

Six `(D, γ)` families, `E = 0.5(1−D)`. Model values (`σ_peak @ k*_peak / k*_zero / c(k*→0)`):

| `D` | `γ` | `σ_pk @ k*_pk` | `k*_zero` | `c(k*→0)` |
|---|---|---|---|---|
| 0.3 | 0.05 | 0.0754 @ 0.306 | 0.769 | −2.100 |
| 0.6 | 0.05 | 0.1048 @ 0.440 | 1.091 | −2.400 |
| 0.9 | 0.05 | 0.0676 @ 0.536 | 1.339 | −0.900 |
| 0.6 | 0.03 | 0.1109 @ 0.430 | 1.093 | −3.998 |
| 0.6 | 0.06 | 0.1018 @ 0.444 | 1.089 | −2.000 |
| 0.6 | 0.09 | 0.0932 @ 0.457 | 1.084 | −1.333 |

Axes as printed: growth `−0.6…0.6`, phase `−5…1`, wavenumber `kb 0…2`. Magenta markers
`λ/2b = 4π, 2π, π` at `k* = 0.25, 0.5, 1.0` (since `λ/2b = π/k*`). Panel mapping is **crossed**:
growth-left pairs with phase-right (`D` family), growth-right with phase-left (`γ` family).


---

## 5. `ε` is the one parameter the deck never gives

`ε` appears only inside `εC_fU₀/b` (p.19) and is never defined or valued anywhere in 21 pages.
`σ` and `c(k*→0)` both scale with it. This package therefore **states it as an assumption**
(`εC_f = 0.5`) rather than fitting it to anything — in particular, not to values scraped off the
p.20 figure, which is what an earlier version did. See the README for why that was dropped.

## 6. Two internal inconsistencies inside river.pdf

Recorded, not resolved — resolving them is the author's call.

1. **`λ ≈ 10 − 14 × width` (p.2) vs `λ ≈ 7 − 14 × width` (p.21).** This matters: `λ/2b = π/k*`, so
   `7–14` means `k* ∈ [0.224, 0.449]` while `10–14` means `k* ∈ [0.224, 0.314]` — different verdicts
   on whether the model's growth peaks fall at "observed scales".
2. **`ψ̄` ordering.** p.9 prints `ū = −∂ψ̄/∂y`, and `ū > 0` everywhere forces
   `ψ̄(+b) < ψ̄(0) < ψ̄(−b)` — yet p.14 labels `ψ₁` on the **top** curve. Two readings are possible
   (the plots show `−ψ`, or their vertical offsets are cosmetic) and the deck does not say which.
   The package therefore treats the offsets as a **declared display constant**, not as `ψ̄`.
