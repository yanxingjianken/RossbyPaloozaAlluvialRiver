# dedalus_meander_full_SW — full shallow water on a meandering channel, curvilinear (s,n)

Linear perturbation dynamics for the **full shallow-water primitive equations**
`(u_s, u_n, η)` in **channel-fitted curvilinear coordinates**, with an erodible
centreline. Unlike the rigid-lid/streamfunction packages in `../`, the **free surface
η is prognostic**, so gravity waves and the vortical branch coexist in one system and
the flow is genuinely divergent (there is no streamfunction).

Full derivation of **every** equation in the driver:
[`derivations/sw_sn_meander.pdf`](derivations/) (9 pp, with bibliography).

## The physics

With metric `σ = 1 + n·C(s)`, `h = η + H(n)`:

```
η_t + (1/σ)[∂_s(h u_s) + ∂_n(σ h u_n)] = 0
∂_t u_s + (u_s/σ)∂_s u_s + u_n ∂_n u_s + C u_s u_n/σ = -(g/σ)∂_s η - r_s u_s + r_η η + ν∇²u_s
∂_t u_n + (u_s/σ)∂_s u_n + u_n ∂_n u_n - C u_s²/σ    = -g ∂_n η      - r_n u_n       + ν∇²u_n
```

`-C u_s²/σ` is the centrifugal term (⇒ superelevation `g ∂_n η̄ = C̄ Ū_s²/σ̄`);
`r_s = 2C_f Ū/h̄`, `r_n = C_f Ū/h̄` (anisotropic Chézy drag) and
**`r_η = C_f Ū²/h̄²`** is the superelevation-drag term — the only route by which the
free surface drives bend flow, hence the only place the Froude number enters. Every
equation is multiplied through by σ to clear the rational `1/σ` and keep the NCCs banded.

Walls at `n = ±b`: `u_n = 0` (no penetration — this assumes nothing about interior
divergence) and free slip `∂_n u_s = 0`, closed with 2 tau per viscous velocity.
The centreline erodes by the Ikeda law `∂_t ζ_c = E·½[u_s(+b) − u_s(−b)]` and feeds
back through the curvature `C' = −∂²_s ζ_c`.

## Perturbation: always broadband

The base state is an exact steady solution, so something must perturb it; the dynamics
are linear, so **any** perturbation is admissible. We always seed **all** resolvable
centreline modes at once. For a straight base channel the s-Fourier modes decouple
exactly, so **one run contains the whole dispersion relation** (demodulate each mode).

Consequently a run is identified by its **physics**, never by "which wavelength was
perturbed":

```
run_H<bed>_bank<sinuosity>_Cf<friction>_U<base speed>dU<jet excess>.h5
      │        │              │            │          └─ Delta, jet excess: the
      │        │              │            │             cross-channel shear
      │        │              │            │             (0 = plug, <0 = wake)
      │        │              │            └─ U0, bank-edge speed
      │        │              └─ bottom friction coefficient C_f
      │        └─ initial bank sinuosity = base curvature amplitude (0 = straight)
      └─ bed: 'flat', or 'cross<amp>' for a parabolic thalweg H(n)
```

Modes that never achieved ≥3 e-foldings are flagged `disp_converged=0` and drawn
hollow — their fitted growth rate is transient, not an eigenvalue.

## Run

```bash
micromamba run -n dedalus env OMP_NUM_THREADS=1 python sw_sn_driver.py   # ONE run of CONFIG at the file head
micromamba run -n dedalus env OMP_NUM_THREADS=1 python experiments.py    # the parameter study
python tests/test_base_profiles.py                                       # base-state checks
cd postprocessing
  python 01_dispersion.py          # sigma(k), c(k) for every configuration
  python 02_eulerian_momflux.py    # absolute-Eulerian momentum-flux movie per configuration
```

`sw_sn_driver.py` has **one** run and no CLI — every knob is the CONFIG at its head
(bed `cross_amp`, bank sinuosity `Cbar_amp`, friction `Cf`, speeds `U0`/`Delta`,
`Froude`, `nu`, resolution, `t_end`). `outputs/` is gitignored.

## Outputs

1. **`fig01_dispersion.png`** — σ(k) and c(k) per configuration, with the
   **Doppler-shifted** gravity bands `Ū ± 1/F` marked (not `1/F`), unconverged modes hollow.
2. **`momflux_eulerian_<tag>.mp4`** — absolute-Eulerian: `(s,n)` fields mapped to the
   lab-frame meandering channel (banks bound the flow), **one fixed scale for the whole
   movie** (no per-frame normalisation, no amplification), colorbars in true units;
   panels `u_s' · u_n' · u_s'u_n' · η'` + a y-z cross-section + the centreline waveform
   and true growth.

## What this model can and cannot say

**Can:** whether the growing disturbance is balanced (`‖δ'‖/‖ζ'‖`, `‖η'‖/‖u'‖ ∝ F²`),
and — the decisive test — whether the mean-flow PV gradient actually powers it
(`T_shear > 0`) or is an energy sink (`T_shear ≤ 0`). `classify_mode()` reports these.

**Cannot:**
- **Select a meander wavelength.** The bed is prescribed (no Exner), so the free
  alternate-bar mode that sets λ by bar–bend resonance does not exist. σ(k) has a
  maximum at k ≈ 1.8 in 6 of the 9 runs, but it barely moves across a 10× change in
  `C_f`, a flat→parabolic bed, and bank sinuosity 0→0.15 — a peak independent of every
  physical knob is the ν cutoff, not selection. No comparison with λ ≈ 10 W is meaningful.

### Why the movies look busy (it is not numerical noise)

Measured, not asserted:

- **It is a wave packet.** 90% of the final `u_s'` variance sits in **~7 neighbouring
  modes**, k = 1.65–2.10, r.m.s. width Δk ≈ 0.15. Seven nearby wavenumbers with unequal
  phases beat — irregular-looking but fully deterministic. Noise would be flat in k.
  Near the peak σ ≈ σ_max − a(k−k_p)² with a ≈ 0.077, so after time T only
  Δk ≲ (aT)^(−1/2) ≈ 0.3 survives: the packet narrows as **T^(−1/2)** and `max_efold`
  stops the run long before it goes monochromatic.
- **The grid is clean.** Energy in the top decade of resolved k is ~**1e-16** of the
  total — no aliasing pile-up. And N_s 64 → 128 at fixed ν changes σ(k) by **0.0–0.1%**
  across the whole band, peak still k=1.80. Converged, not grid-generated.
- **What N_s=128 fixes is legibility only.** λ = 2π/1.86 ≈ 3.4, which at N_s=64 is just
  **2.6 grid points per wavelength** — barely past the 2-point Nyquist limit, so *any*
  wave would render as one-pixel stripes. At N_s=128 it is **5.2 points/wavelength**.
  Production runs use N_s=128 for that reason alone; the physics is identical.
- **Reach the geomorphic regime.** Field erodibility is `E/U ~ 1e-8`; the default here
  is O(0.1). Classical theory takes E → 0, which is what justifies imposing `u_n=0` at
  a fixed wall; at O(1) erodibility that BC is inconsistent at leading order.
- **Discriminate wave families by a Froude sweep.** With `P ≡ gη` the momentum equations
  are F-free and F survives only as `F²` on the continuity time-derivative, i.e. purely
  as a divergence parameter. Any slow balanced mode is F-insensitive by scaling, so
  F-insensitivity *confirms the rigid-lid reduction* rather than identifying a wave family.
- A smooth, single-signed `∂_n q̄` (what a parabolic jet gives) has no PV edge and so
  supports **no discrete vortical eigenmode** — only a continuous spectrum. A shear-β is
  not a planetary β.

Also absent: helical secondary flow (depth-averaged away — this is the `A=0` limit of
Ikeda et al. 1981, whose `A≈2.89` is their dominant driver), width variation, rotation,
and any nonlinearity. The base state is frozen; the lateral viscosity uses the flat, not
the curvilinear, Laplacian.
