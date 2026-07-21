# dedalus_meander_full_SW — full shallow-water meander in curvilinear (s,n)

**What wave is a river meander?**  The other packages in `../` (`vorticity_meander`,
`dedalus_meander`, `dedalus_meander2`, `dedalus_rigid_lid_to_share`) are all
**rigid-lid / low-Froude / streamfunction** reductions: they assume the mass flux is
non-divergent (`∇·(hu)=0`), use a streamfunction Ψ and PV `q=ζ/H`, and thereby **filter
out gravity waves**.  This package keeps the **free surface η** (the mass equation has
`η_t`), so **both gravity waves and the vortical/Rossby wave live in the same system** and
the flow is genuinely **divergent** — there is no streamfunction, we solve the primitive
variables `(u_s, u_n, η)` and **diagnose** PV to classify modes.

The channel meanders with **finite amplitude** (no small-meander assumption): we work in
**channel-fitted curvilinear coordinates (s, n)** — `s` along the meandering centerline,
`n` across — with metric `σ = 1 + n·C̄(s)` carrying the curvature `C̄`.  The base jet
**follows the channel** and the outer bank is superelevated.  Dynamics are **linear
perturbations** on this finite-meander base; the erodible bank feeds back through the
**curvature perturbation** `C'`.  **IVP only** (no eigenvalue solver, no validation oracle).

## The physics (curvilinear shallow water, non-rotating)

With `σ = 1 + nC`, `h = η + H(s,n)`:

```
η_t + (1/σ)[∂_s(h u_s) + ∂_n(σ h u_n)] = 0
∂_t u_s + (u_s/σ)∂_s u_s + u_n ∂_n u_s + C u_s u_n/σ = -(g/σ)∂_s η - r u_s + ν∇²u_s
∂_t u_n + (u_s/σ)∂_s u_n + u_n ∂_n u_n - C u_s²/σ    = -g ∂_n η    - r u_n + ν∇²u_n
```

Curvature terms: `+C u_s u_n/σ` (s), `-C u_s²/σ` (n, **centrifugal → superelevation**).
Base (finite `C̄`, `Ū_n=0`): **parabolic jet** `Ū_s(n)=U₀+Δ(1-(n/b)²)` (so `∂²_n Ū_s=-2Δ/b²`
= **constant cross-channel vorticity gradient = the channel-β**, the Rossby restoring), and
superelevation `g ∂_n η̄ = C̄ Ū_s²/σ̄`.  We multiply every equation **through by σ** to clear
the rational `1/σ` (keeps NCCs polynomial/banded).

**Walls & erosion** (banks fixed at `n=±b`, no non-divergence assumed):
- `u_n(±b)=0` (no-penetration) + free-slip `∂_n u_s(±b)=0` — closed with 2 tau per viscous velocity.
- **Meander (centerline) erosion** `∂_t ζ_c = E·½[u_s(+b) − u_s(−b)]` — the **antisymmetric**
  near-bank velocity (fast outer minus slow inner = Ikeda bend growth).
- Feedback through curvature `C' = −∂²_s ζ_c`, which enters the n-momentum as `+Ū_s²·∂²_s ζ_c`.

The bank drag `C_f` acts on **both** momentum components (`r_s=2C_fŪ_s/h̄`, `r_n=C_fŪ_s/h̄`).
`g = 1/F²` (gravity-wave speed `1/F`, `F=U_c/√(gH₀)` the Froude number).

## Run (micromamba `dedalus`, v3.0.5)

```bash
# THE run: everything is configured in the CONFIG dict at the top of the driver.
# Edit CONFIG, then:
micromamba run -n dedalus env OMP_NUM_THREADS=1 python sw_sn_driver.py

# everything else lives outside the driver:
python tests/test_base_profiles.py        # base-state sanity (jet, channel-beta, metric, superelevation)
python sweep_dispersion.py                # the (k x Froude) experiment -> many outputs/run_*.h5
cd postprocessing
  python 01_dispersion.py                 # sigma(k), c(k) figure, grouped by Froude
  python 02_eulerian_momflux.py [tag ...] # absolute-Eulerian momentum-flux movie(s)
```

**Layout.** `sw_sn_driver.py` has **one** run and no CLI options — all knobs are the
CONFIG at its head. `tests/` holds the sanity checks, `sweep_dispersion.py` the
(k, Froude) experiment, `postprocessing/` all figures/movies, `derivations/` the
full derivation of **every** equation in the driver
([`sw_sn_meander.pdf`](derivations/), 7 pp), `outputs/` the raw HDF5 (gitignored).

## The two outputs

1. **Dispersion relation** `fig01_dispersion.png` — growth `σ(k)` and migration speed `c(k)`
   of the meander, measured from the IVP (per-wavelength runs), grouped by Froude.  The
   gravity speed `1/F` is marked: a meander branch that tracks `1/F` is gravity-coupled; an
   F-insensitive branch is the vortical/Rossby wave.
2. **`momflux_eulerian_<tag>.mp4`** — **ABSOLUTE-Eulerian** movies (bird's-eye), **one per init
   bank wavelength**: the `(s,n)` fields mapped back to the lab-frame meandering channel (banks
   bound the flow, free aspect). **ONE fixed scale for the whole movie** — colorbars carry the
   TRUE physical values, display gain 1, **no per-frame normalisation and no extra amplification**
   — so the colours and the banks both grow with the real `e^{σt}` (the channel starts ~straight
   and develops the meander). The model is linear, so the single overall amplitude is a free
   constant: it is fixed ONCE (final meander = 0.5 channel half-widths) and applied identically
   to every frame and every field.
   6 panels: `u_s' · u_n' · momentum flux u_s'u_n'` · `η' (free surface)` · a **y-z cross-section**
   (Ikeda Fig-2b: bed `H(n)` + jet + banks + free surface) · growth stats, each with a colorbar.
   Rendered on a straight base (`C̄=0`) so the meander forms cleanly from a near-straight channel.

## CONFIG (top of `sw_sn_driver.py`, all adjustable)

`U0, Delta, b` (parabolic jet) · `Froude, H0` (free surface) · `A_bank / kmeander / Cbar_amp`
(**init bank/meander amplitude & curvature**) · `bed_depth(s,n)` (**adjustable bed H expression**)
· `Cf` (drag) · `ECOEF→E` (erodibility) · `nu` (lateral viscosity) · `kstar, Ns, Nn, dt, t_end, A0`.

## Honest caveats

- The lateral viscosity `ν∇²` uses the flat (not the exact curvilinear vector) Laplacian —
  a wall-closure + short-wave regulariser, small; the physics is in the inviscid terms.
- The base is a **frozen** finite-meander state (the steady residual is an implicit forcing,
  as in every frozen-jet linear stability); the perturbation curvature `C'` is small.
- Bank erosion drives the **centerline/bending** mode (constant width); the breathing/widening
  mode is not modelled.  Full mesh-deforming morphodynamics (evolving `C̄`) is out of scope.
