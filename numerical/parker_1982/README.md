# Parker, Sawai & Ikeda (1982) — Bend Theory of River Meanders, Part 2

A verified numerical explainer of the weakly nonlinear meander theory

> Parker, G., Sawai, K. & Ikeda, S. (1982) **"Bend theory of river meanders. Part 2. Nonlinear deformation of finite-amplitude bends."** *J. Fluid Mech.* **115**, 303–314.

Every relation in [`parker_lib.py`](parker_lib.py) is transcribed from the rendered PDF **and pinned by an independent computational route** — the scan's dense fractions make single-glyph misreads likely, and this package's design caught three of them (see Deviations).

## The physics in one paragraph

Part 1's linear theory grows a perfect sinusoid forever; real bends fatten, lean, and slow down. Part 2 applies a **modified Stokes expansion** (y = εμ₁ + ε³μ₃, strained rates α(τ), ω(τ), strain function ℱ(τ)=(e^{2τ}−1)/(2τ)) to the nonlinear bend equation (Eq. 7), whose geometric nonlinearities γ=cosθ and the *reach-averaged* sinuosity factor χ=(mean 1/γ)^{−1/3} couple the growing bend to its own decreasing valley slope. Results: (1) growth and downstream migration **slow with amplitude** (Eqs. 26a, 32); (2) the selected wavenumber **shifts** — always longer waves for incised channels, and for alluvial channels longer only if the erosion-law constant e exceeds e\*(F) (5.1 at F≪1, 2.7 at F=1; Eq. 30, Fig. 5); (3) self-generated third harmonics **fatten** (J_F·cos3φ) and **skew** (J_S·sin3φ) the bend (Eqs. 33–35) — the shapes you can read off aerial photos (Beaver River fit: δ₀=0.98, J_F=0.073, J_S=0.103).

## The verified core (5 independent routes)

| route | what it checks | result |
|---|---|---|
| A | α₂, ω₂ **derived** by solving Eq. (24)'s first-mode-removal 2×2 system (re-derives printed 25a/b) ≡ printed (26a) at k₀M | exact (1e-12) across the (e,F) grid |
| B | printed (30), (32) closed forms | (30) confirmed by the printed thresholds e\*=5.1/2.7 (its bracket zero gives 5.12/2.73 analytically); signs of (26a)/(26b)/(32) |
| C | (J_F, J_S) by **inverting ℒ** (Eq. 14) against Eq. (24)'s third-mode sources ≡ printed (34a/b) | exact; anchors J_FM = 0.0478 (alluvial) & 0.0469 (incised) both printed; J_SM(β=1.5) = **0.00636** |
| D | k_M (Eq. 30) ≡ parabolic-refined argmax of α₀+ε²α₂ | ≤2% on the shift coefficient |
| E | **full nonlinear PDE (Eq. 7)**, pseudo-spectral (2× dealiasing, preconditioned ℳ⁻¹, RK4) | linear limit exact (0.000%); growth vs (26a): 0.07%; drift vs ω(0): 0.24%; third-harmonic \|J\|: 0.1%; even harmonic at 1e-15 floor (μ₂≡0) |

Plus: sine-generated-curve geometry → Cartesian flattening 7/144 = 0.0486 with O(δ₀²) convergence; `harmonics` recovers the Beaver constants from Eq. (5) to 1e-10.

## Quickstart & file map

```bash
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/parker_1982
micromamba run -n fourcastnetv2 python parker_lib.py                    # self-test (routes A-E)
micromamba run -n fourcastnetv2 python 01_fattening_skewing_geometry.py # fig01-03
micromamba run -n fourcastnetv2 python 02_nonlinear_corrections.py      # fig04-06 (Fig. 5 replica)
micromamba run -n fourcastnetv2 python 03_shape_evolution.py            # fig07 (Fig. 6 replica) + shape_evolution.mp4
micromamba run -n fourcastnetv2 python 04_pde_validation.py             # fig08-10 + pde_bend.mp4
```

| File | Output |
|---|---|
| `parker_lib.py` | verified theory + geometry utils + nonlinear PDE integrator + self-test |
| `01_...geometry.py` | `fig01` sine-generated curve & fattening (Fig. 1) · `fig02` J_F/J_S families · `fig03` Beaver planform (Fig. 4 solid line) + harmonic recovery |
| `02_...corrections.py` | `fig04` α, c vs δ₀M · `fig05` k_M shift by regime · `fig06` (F,e) regime map with printed 5.1/2.7 anchors (Fig. 5) |
| `03_shape_evolution.py` | `fig07` Fig.-6 replica (k=π/3, J=0.05, t=0 & 0.45) · `shape_evolution.mp4` |
| `04_pde_validation.py` | `fig08` harmonics vs Eq. 35 · `fig09` δ-sweep of growth & \|J\| · `fig10` planform PDE-vs-Eq.35 · `pde_bend.mp4` |

## Deviations & transcription notes (all caught by the routes)

- **Eq. 26a**: the eβ² coefficient is ambiguous (½ vs 3/2) at scan resolution; route A's identity selects **½** (residual coefficient 0.500000 exactly).
- **Eq. 24 third-mode source**: the k⁵ coefficient is ambiguous (⅛ vs 9/8); route C selects **9/8** — only it reproduces both printed J_FM anchors to machine precision.
- **Eq. 34b**: the β⁴ glyph is ambiguous (¼ vs ½); route C recovers the bracket coefficient 12.0006 with **½**. J_SM(alluvial) = 0.00636. *(An earlier internal briefing's "J_SM ≈ 0.0103" appears nowhere in the paper and fails route C.)*
- **Eq. 7**: the χ² term carries **plain A** (not Ā) — reading Ā double-counts F² and shifts the PDE growth ~5%, which is how route E caught it. At χ=1 the term linearizes to C_f(A+F²)=C_fĀ, consistent with Eq. (16a).
- **Beaver fit**: the text prints **δ₀ = 0.98** (an earlier briefing's "δ₀² = 0.98" ambiguity resolved from the rendered page).
- Printed Eq. (31)'s β⁴/β⁶ glyphs are illegible; `omega_kM` uses the numeric route ω₀(k_M)+ε²ω₂(k_M), exact to the same order.
- The paper's Fig. 4 *observed* Beaver centerline is a hand-drawn dashed trace over an aerial photo; it is not digitized here (no reliable programmatic separation from the overlapping solid fit line). Fig03 shows the published fit itself; **no synthetic stand-in is drawn**.
- The PDE runs at C_f = 0.1 for speed; every asserted quantity (β, f, e, δ₀M-scaled corrections, J's) is C_f-free.

## Shared helper block

`set_style`/`save_fig`/`fig_to_rgb`/`write_mp4` byte-identical across packages (fenced `=== shared helper block v1 ===`).
