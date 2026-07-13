# Vorticity-Meander Theory — rebuild v2 of the 6/30 Rossby-Palooza deck package

A verified numerical reproduction of the **group's own in-progress theory** presented in

> *"Alluvial Rivers & Jet streams"*, Rossby Palooza meeting deck, 2026-06-30 (`literature/Rossby_Palooza_meet_0630.pdf`).

**This is a from-scratch rebuild (2026-07-12)** of the v1 package (built 2026-07-06, torn down
on request): every equation re-derived by hand from the deck, the deck-p.8 pins re-digitized
independently at 300 dpi (they agree with v1's 200-dpi reads within the stated reading error),
and — the substantive upgrade — the interior **friction closure is now implemented in two
variants** so the v1 open question ("is the growth-peak discrepancy the friction closure?")
could be answered quantitatively. It is answered: **no, not by itself** (see below).

The governing equations, their derivation, and the term-by-term contrast with Ikeda et al.
(1981) Eq. (7) are written up in [`THEORY.md`](THEORY.md).

## The physics in one paragraph

A slow alluvial river (0.1 < Fr < 0.3, deck p. 3) is depth-averaged 2-D vorticity dynamics.
The parabolic in-channel jet ū = U₀ + (Δ/b²)(b²−y²) carries a **constant positive vorticity
gradient 2Δ/b² — the channel's planetary-β analogue** (hence "Rossby" Palooza). With rigid
banks the jet has no inflection point, so it is neutrally stable; the deck's three-level
closure (ψ₁, ψ₂, ψ₃ at y = +b, 0, −b; sinuous symmetry ψ̂₁=ψ̂₃) instead makes the **erodible
bank** the engine: the bank relaxes toward the interior streamfunction (∂ψ₁′/∂t =
εC_fU₀/b·(ψ₂′−ψ₁′), p. 7), which destabilizes wavelengths in the resonant band k*² < 2D and —
unlike Ikeda's always-downstream c₀ > 0 — sends the meander pattern **upstream**
(c* → −E·D/γ_eff as k*→0). With friction the wave tilts and its Reynolds stress carries
downstream momentum from the jet core to the banks (p. 6): ∂_y(u′v′)‾ = (γ/2D)ζ₂′²‾ > 0.

## What is verified (self-test, hard tier)

| deck statement | this package |
|---|---|
| p.5 box: \|ψ̂₂\|>\|ψ̂₁\| iff k\*²<2D (γ=0) | **exact** (equality at k\*²=2D to 1e-12), both closures coincide at γ=0 |
| p.5: "no resonance for k\*>0 even in the inviscid limit" | exact (no pole for D<1) |
| rigid banks stable | all modes Im ω ≤ 0 for ε=0, both closures |
| p.7 upstream propagation | analytic: c\* → −E·D/γ (rayleigh), −E·D/(2γ) (momentum), asserted to 0.1% |
| growth band | analytic small-E: σ/E → (2D−k\*²)/(2+k\*²−2D) at γ=0 |
| p.4 closure = discretization | N=3 channel eigenproblem ≡ the 2×2 closure to 1e-10 (+ varicose ψ mode ω=−iE), **both closures**; N-point converged by N≈11 |
| closure algebra | A₀, A₂ closure-independent; A₁(momentum) − A₁(rayleigh) = 2γ exact |
| p.8 axis markers | λ/2b = π/k\* arithmetic |

## Deck p. 8 comparison (calibrated tier) — the open question, sharpened

Re-digitized pins (`data/deck_p8_pins.csv`). Single calibration per closure from the six
phase intercepts via c₀ = −E·D/γ_eff, E = ECOEF(1−D): **ECOEF = 0.5 (rayleigh) / 1.0
(momentum)**, per-curve spread < 3% — an exact E↔2E degeneracy, so **the intercepts cannot
distinguish the closures**.

| quantity | rayleigh (deck-literal) | momentum (Ikeda-consistent) |
|---|---|---|
| phase intercepts c₀ (6 curves) | ±4–15% | ±4–15% (identical fit quality) |
| growth zero crossings k\*₀ (6) | ±0.03 | ±0.03 |
| σ_peak orderings | reproduced | reproduced |
| **σ_peak magnitudes** | **deck/model = 3.18–4.42** | **deck/model = 2.28–3.22** |
| peak k\* shift | +0.09…+0.19 | +0.13…+0.22 |

**Rebuild-v2 answer to the v1 open question**: switching the interior friction from Rayleigh
damping of ζ′ to the curl of the linearised bottom drag −(C_f/H)\|u\|u (the closure Ikeda's
Eq. 3b uses, with its factor-2 streamwise anisotropy) raises the peaks by ~40% but leaves a
**2.3–3.2× gap** — the friction closure **alone does not explain the deck's peak heights**.
Remaining suspects, handed back as the sharpened question: (a) the p. 6 **"pressure drag due
to wavy banks"** that does not appear in the p. 7 bank equation; (b) the plotted growth
axis's normalization (`FLAG_TSCALE`). Resolution is excluded (fig11).

## Flagged assumptions

| flag | assumption | status |
|---|---|---|
| `FLAG_FRICTION` | interior friction closure not printed in the deck | **two variants implemented**; neither reaches the deck peaks under the intercept calibration |
| `FLAG_EPS` | E = εC_f·U₀/(U₀+Δ), εC_f = `ECOEF[friction]` | **calibrated**: 0.5 / 1.0 fit all six intercepts to <3% |
| `FLAG_TSCALE` | time unit b/(U₀+Δ) | makes centre advection speed exactly 1; now also a peak-gap suspect |
| `FLAG_BASE` | parabolic jet externally maintained; friction on perturbation only | deck-implicit |
| bank eq | literal p.7 (no advection term at the bank) | deck-explicit; alternatives fit the intercepts worse (v1 finding) |

## Quickstart & file map

```bash
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/vorticity_meander
micromamba run -n fourcastnetv2 python vorticity_lib.py         # self-test
micromamba run -n fourcastnetv2 python 01_setup_schematic.py    # fig01-02
micromamba run -n fourcastnetv2 python 02_forced_response.py    # fig03-05 (deck pp.5-6)
micromamba run -n fourcastnetv2 python 03_dispersion.py         # fig06-07 (deck p.8 + discrepancy)
micromamba run -n fourcastnetv2 python 04_regime_map.py         # fig08-09
micromamba run -n fourcastnetv2 python 05_anim_meander_mode.py  # meander_mode.mp4
micromamba run -n fourcastnetv2 python 06_continuum_solver.py   # fig10-12 (N-point)
micromamba run -n fourcastnetv2 python 07_friction_closures.py  # fig13-14 (NEW: closure duel)
```

| File | Output |
|---|---|
| `THEORY.md` | the governing equations, reduction chain, Ikeda-(7) contrast, closure finding |
| `vorticity_lib.py` | closures (rayleigh + momentum), dispersion (quadratic in W=−iω\*), branch continuation, N-point GEP, calibration, self-test |
| `01_setup_schematic.py` | `fig01` jet+levels+β-analogue · `fig02` no-inflection ⇒ bank-driven instability |
| `02_forced_response.py` | `fig03` p.5 planforms · `fig04` \|ψ₂/ψ₁\| map with k\*²=2D curve · `fig05` p.6 friction phase-lag + u′v′ flux |
| `03_dispersion.py` | `fig06` **deck p.8 regenerated with pins** · `fig07` match/mismatch bars |
| `04_regime_map.py` | `fig08` σ_pk, k\*_pk over (D,γ) · `fig09` upstream-speed map |
| `05_anim_meander_mode.py` | `meander_mode.mp4` — growing mode marching upstream |
| `06_continuum_solver.py` | `fig10` N=3≡2×2 (both closures) · `fig11` N-convergence vs deck peak · `fig12` eigenfunction |
| `07_friction_closures.py` | `fig13` closure overlay on deck p.8 · `fig14` peak-ratio bars |
| `data/deck_p8_pins.csv` | re-digitized deck pins (provenance header) |

## Data provenance

`data/deck_p8_pins.csv`: six curves' peaks/crossings/intercepts read visually from 300-dpi
renders of deck p. 8 (reading error ~±0.02 in k\*, ±0.02 in σ, ±0.1 in c₀), 2026-07-12,
independently of v1; the two readings agree within error. No synthetic data; all model
inputs are the deck's printed parameters. `grep -rn "np.random" *.py` returns nothing.

## Shared helper block

`set_style`/`save_fig`/`fig_to_rgb`/`write_mp4` byte-identical across packages (fenced
`=== shared helper block v1 ===`); check:

```bash
diff <(sed -n '/=== shared helper block v1/,/=== end shared helper block ===/p' vorticity_lib.py) \
     <(sed -n '/=== shared helper block v1/,/=== end shared helper block ===/p' ../parker_1982/parker_lib.py)
```
