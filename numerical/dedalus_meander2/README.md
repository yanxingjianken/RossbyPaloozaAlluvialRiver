# dedalus_meander2 — variable-depth PV meander model (single-driver architecture)

Variable riverbed **H(x,y)** generalization of the constant-depth vorticity-meander
model in [`../dedalus_meander/`](../dedalus_meander/). Dropping the flat-bed assumption
turns absolute-vorticity conservation into **potential-vorticity (PV) conservation**
`D/Dt(ζ/H)`, and the Rossby restoring force gains a **topographic-shear β**. The full
derivation is in [`derivations/variable_h_pv.pdf`](derivations/) (11-slide deck).

## Architecture: one core Dedalus file + a postprocessing folder

- **[`meander_driver.py`](meander_driver.py)** — the **single core Dedalus file**. All
  core parameters live in a `CONFIG` dict at the top (edit-and-run; CLI overrides for
  batch). It builds the variable-H PV solver and writes **raw HDF5** to `outputs/` — it
  **never plots**. Modes: `selftest | evp | ivp | sweep`.
- **[`postprocessing/`](postprocessing/)** — **all** figures/movies. Reads `outputs/*.h5`
  (never re-runs the solver) and reuses the verified renderers from
  `../dedalus_meander/channel_lib.py` (DRY).
- **[`derivations/`](derivations/)** — the PV / topographic-β derivation (LaTeX + PDF).
- `outputs/` (gitignored, regenerable HDF5) · `figures/` (rendered, committed).

**Env / run** (micromamba `dedalus`, Dedalus v3.0.5):
```bash
micromamba run -n dedalus env OMP_NUM_THREADS=1 python meander_driver.py --mode selftest
micromamba run -n dedalus env OMP_NUM_THREADS=1 python meander_driver.py --mode ivp --kstar 0.3 --cross-amp 0.3
micromamba run -n dedalus env OMP_NUM_THREADS=1 python meander_driver.py --mode sweep
cd postprocessing && micromamba run -n dedalus env OMP_NUM_THREADS=1 python 01_dispersion.py
                     micromamba run -n dedalus env OMP_NUM_THREADS=1 python 03_multipanel.py            # normalized
                     micromamba run -n dedalus env OMP_NUM_THREADS=1 python 03_multipanel.py --eulerian  # fully-Eulerian
```

## The physics (one paragraph)

Low-Froude rigid lid with variable depth ⇒ the **mass flux** `H·U` is non-divergent, so
`u=-(1/H)∂yΨ, v=(1/H)∂xΨ` with a **mass-transport streamfunction Ψ**, and
`ζ'=∇·((1/H)∇Ψ')` (a variable-coefficient elliptic operator). Curling the momentum
equation gives PV tendency `D/Dt(ζ/H)=(1/H)curl F`. Linearized about a jet `ū(y)` over a
base bed `H̄(y)` it is **term-for-term the flat-bed model** with `β → β_top(y)` and
`∇² →` the H̄-weighted elliptic operator:

```
∂t ζ' + ū(y) ∂x ζ' + β_top(y) ∂x Ψ' = curl F',     ζ' = ∇·((1/H̄)∇Ψ')
β_top(y) = ∂y(ζ̄/H̄) = 2D/H̄ + ū_y H̄_y/H̄²           (flat bed H̄=1 → 2D)
```

Flat bed (`cross_amp=0`) recovers `../dedalus_meander` **exactly**.

## CONFIG knobs (single-file)

`D` (⇒ U₀=1−D), `Lx`, bed `cross_amp`/`along_amp`/`along_kbed`/`along_phase`
(`H(x,y)=[1+cross_amp(1−y²)]·[1+along_amp cos(k_bed x)]`), `gamma` (bottom friction) +
`friction` (`rayleigh`/`momentum`), `ECOEF`/`E` (bank erodibility), `f0` (rotation knob,
see below), `kstar`, `Nx`, `Ny`, `dt`, `t_end`, `A0`. Banks are the two `psib_top/psib_bot`
fields at y=±b.

## Verification ladder (all hard-asserted, `--mode selftest`, all green)

| check | what | result |
|---|---|---|
| flat-bed reduction | variable-H GEP vs `VL.channel_modes` (H=1) | **0.0** (identical) |
| d3 EVP vs GEP | flat + bumped bed, both closures | **4.4e-9** |
| topographic Rossby | f0=0 → β_top=0; f0≠0 → propagating wave | as predicted |
| IVP vs EVP | measured σ,c, flat + bumped bed | **0.1%** |

Two independent discretizations (Dedalus Chebyshev spectral & an FD+Richardson GEP) agree
to ~1e-9 — the Ĥ²-weighted, ζ-auxiliary reformulation is correct.

## The model is strictly linear (by design)

This package solves the **linear** variable-depth PV instability only. The two
second-order effects sometimes conflated with it — the O(ampl²) self-advection `J(ψ',ζ')`
(finite-amplitude fattening/skewing, Parker 1982) and Ikeda's 3-D **secondary-flow scour**
(`A≈2.89`) — are **deliberately excluded**: they are separate physics, orthogonal to the
linear channel-Rossby instability (see the four-effects slide in the derivation). No
nonlinear terms, no empirical closures, no hyperviscosity.

## The two movies (both make the banks the true channel boundary)

`03_multipanel.py` renders each run as a 6-panel movie (ψ_total · ψ' · momentum flux
u′v′ · y-z cross-section · dispersion · stats), in **two views that differ only in scaling**:

- **`multipanel_…`** (default) — **normalized**: each frame is re-scaled to a constant
  bank amplitude (magnifying-glass **mode shape**; the true growth is the `gain` counter).
- **`multipanel_eulerian_…`** (`--eulerian`) — **fully-Eulerian**: **one fixed scale** for
  the whole movie, so the meander visibly grows `e^{σt}` out of a near-straight channel.

In **both**, the interior field is drawn with `warp_fill` on the mesh whose top/bottom edges
**are** the moving bank lines `y=±1+η(x)` — so **the two banks are the exact boundary of the
flow** (no gap, no water outside). Every planform panel carries a **colorbar**, and the
title states the bed's functional form — **H(y)** (cross-channel thalweg), **H(x)**
(along-channel bars), or **H(x,y)** (both) — plus which view it is.

## Honest caveats

- **Non-rotating river** (`f0=0`): the bed *modulates* the shear-β; a bed bump alone
  (D=0) gives **no** β (no planetary vorticity to squash). `f0≠0` is a formal knob for
  true topographic Rossby waves (waves even at D=0) — beyond the real river.
- A **symmetric** thalweg bump modulates the dispersion only modestly (partial cancellation
  in β_top); asymmetric (point-bar) beds would do more.
- **Along-channel** bed `H(x,y)` (`along_amp>0`) couples Fourier modes (IVP-only, no EVP).
  **Done** (`build_ivp_Hxy`): discharge-conserving base flow `u₀=H̄(y)ū(y)/H(x,y)` (transport
  H̄ū conserved), prognostic `q'=ζ'/H`, x-Ĥ²-weighted well-banded form. Verified: `along_amp=0`
  reduces to the H̄(y) case to **0.1%** (self-test Stage 5). `along_kbed` defaults to `kstar`
  (**bed↔bank resonance** = the Ikeda bar↔bend coupling); at resonance the bed measurably
  shifts σ and c (e.g. amp=0.25: σ 0.103→0.109, c −0.261→−0.189). The `03_multipanel` y-z
  cross-section becomes **dynamic** (the bed sweeps in x).
- `momentum`-closure variable-H friction drops an O(∇H) curl correction (flat-bed-exact);
  the along-channel-bed solver is rayleigh-closure.

## File map

| file | role |
|---|---|
| `meander_driver.py` | core solver: CONFIG, `bed_depth`, profiles, variable-H GEP, d3 EVP+IVP, HDF5, self-test |
| `postprocessing/pp_lib.py` | HDF5→`res` adapter + `dispersion`/`group_velocity`; imports channel_lib renderers |
| `postprocessing/01_dispersion.py` | `fig01` σ/c*/c_g vs bed · `fig02` group-flip k_g vs thalweg depth |
| `postprocessing/03_multipanel.py` | the **two movies** (above): ψ_total · ψ' · momentum flux u′v′ · **y-z cross-section** (Ikeda Fig-2b view) · dispersion · stats. Default = **normalized**; **`--eulerian`** = fully-Eulerian (fixed scale). Both `warp_fill` the interior so the banks are the exact field boundary; every panel has a colorbar; the title states H(y)/H(x)/H(x,y) + the view |
| `derivations/variable_h_pv.tex/.pdf` | the PV / topographic-β derivation (11 slides) |

Provenance: reuses the verified d3 idioms, GEP, and renderers of `../dedalus_meander`
(theory bridge to `../vorticity_meander/vorticity_lib.py`). Built 2026-07-19.
