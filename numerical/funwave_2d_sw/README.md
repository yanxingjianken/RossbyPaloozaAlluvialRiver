# funwave_2d_sw — two bank wavelengths, mobile bed, mobile banks

Nonlinear depth-averaged shallow water on a meandering channel with an **erodible bed and
erodible banks**, run twice: the two cases differ **only in bank wavenumber**.

This is the mobile-bed companion to [`../dedalus_meander_full_SW`](../dedalus_meander_full_SW),
whose own README states the limit it cannot pass:

> *Cannot select the observed meander wavelength. The bed is prescribed (no Exner), so the
> free alternate-bar mode that sets λ by bar–bend resonance does not exist.*

Full equation set, closure-by-closure source citations, and the timescale analysis:
[`PLAN.md`](PLAN.md).

## The design: only the wavenumber changes

For a centreline `y_c = A cos(kx)` the apex curvature is `C = A k²`. Holding `A` fixed and
changing `k` would change the bend **forcing** by `k²`, so the two runs would differ in drive
strength as well as wavelength and no comparison could be attributed. We therefore hold

```
C₀ = A k² = 8.496e-3 m⁻¹   FIXED     ⇒   A = C₀/k²
```

and the reach is a **common multiple of both wavelengths**, so the two cases share the same
down-valley domain, the same `nx`, the same inlet/outlet geometry and the same interior
length — only the bend *density* differs.

```
L_valley = 6 × 1047.0 = 12 × 523.5 = 6282 m
buffer   = 1047 m at each end (non-erodible)   ⇒   interior 4188 m for both
```

| | **B1** | **B2** |
|---|---|---|
| λ | 1047.0 m | 523.5 m |
| k | 6.00e−3 m⁻¹ | 1.20e−2 m⁻¹ |
| A | 235.9 m = 2.36 W | 59.0 m = 0.59 W |
| `A k²` | 8.496e−3 | **8.496e−3** ✓ |
| `R_min = 1/C₀` | 117.7 m = 2.35 b | **117.7 m** ✓ |
| `A k² b` (fold margin, must be <1) | 0.425 | **0.425** ✓ |
| `L_valley` | 6282 m | **6282 m** ✓ |
| interior (analysed) | 4188 m = **4 bends** | 4188 m = **8 bends** ✓ |
| grid @ Δx = 2.5 m | 2513 × 261 | **2513** × 119 |
| sinuosity (exact integral) | 1.391 | 1.115 |
| channel length | 8740 m | 7007 m |

**The buffer is a fixed physical length, not a bend count.** The inlet artefact's scale is the
sediment adaptation length (`L_a = U H/(γ w_s) ≈ 20 m`) plus the flow adjustment, neither of
which scales with λ. Counting bends instead would discard 33% of B1's reach but only 17% of
B2's — an asymmetry in exactly the variable under test.

The residual 1.25× difference in *channel* length is irreducible: it is the sinuosity, which is
a genuine consequence of changing λ at fixed curvature. Spin-up is per-run (one transit time
each), so it is absorbed rather than compared.

## The cross-section

```
h(n) = ( w₀ + β n²/2 )⁻²        w₀ = H_c^(−1/2),  β = 2(H_b^(−1/2) − w₀)/b²
H_c = 3.0 m (centreline),  H_b = 1.5 m (bank edge),  b = 50 m
```

**Why this shape.** Under the local normal-flow balance `U = √(g h S/C_d)` the depth-averaged
potential vorticity is

```
q = ζ/h = √(gS/C_d) · dw/dn ,      w ≡ h^(−1/2)
```

so `w` quadratic in `n` ⟺ `q` linear in `n` ⟺ **`∂q/∂n` exactly constant** — the mobile-bed
analogue of the constant channel-β (`∂ζ̄/∂n = 2Δ/b²`) that the `(s,n)` linear model imposes.
Here it is `∂(ζ/h)/∂n = 1.105e−4 m⁻²s⁻¹`.

The parabolic jet of the linear model is **not** reproducible: in FUNWAVE there is no base
state, `U ∝ √h`, and matching `Δ = 0.6 m/s` would need a thalweg 3.06× the bank depth,
dropping `b/H` from 17 to 5.4. `Δ` is an output here, not a knob.

## The standing prediction

Stock FUNWAVE directs bedload along `atan2(v,u)` and nothing else — no transverse bed-slope
deflection, no secondary-flow (helical) correction. This is the **`A = 0` limit of Ikeda et
al. (1981)**, whose `A ≈ 2.89` is their dominant bend driver.

> **Prediction, stated before the runs: outer-bank scour WITHOUT an inner point bar.**

`postprocessing/01_validate.py` reports the measured inner/outer bed-change split against it.
Adding the two missing closures (`PLAN.md` §6) is the natural next step and is *not* done here.

## Run

```bash
micromamba run -n fourcastnetv2 python tests/test_bathy.py            # 40 geometry checks
micromamba run -n fourcastnetv2 python run_meander.py                 # build both cases
micromamba run -n fourcastnetv2 python run_meander.py --launch        # spinup + morph
micromamba run -n fourcastnetv2 python postprocessing/01_validate.py  # gates -- run FIRST
micromamba run -n fourcastnetv2 python postprocessing/02_morphology.py --max-frames 1
micromamba run -n fourcastnetv2 python postprocessing/02_morphology.py
```

`run_meander.py` has one CONFIG at its head and no CLI beyond `--launch`. `runs/` and
`figures/` are gitignored.

### Two phases, and why

Each case runs **spin-up** (`Bed_Change = F`, from the analytic normal-flow state, one channel
transit time) and then **morph** (`Bed_Change = T`, hot-started from the last spin-up snapshot
via `INI_UVZ`). Without this, B1's channel — 1.25× longer than B2's even on the common reach —
would spend 1.25× longer spinning up, and the bed would evolve under an unconverged flow for
different durations in the two cases: a confound in exactly the variable being compared.
(Before the reach was made a common multiple it was 2.5×.) Spin-up is **per-run**; the
morphodynamic phase is **identical** (8000 s × MF 1000 = 93 morphological days ≈ one
bar-formation timescale).

Hot-start continuity is exact on wet cells (measured `max|Δu| = max|Δv| = 0`); `eta` differs
only on dry cells, where FUNWAVE resets `Eta = MinDepth + Z`.

## Outputs

`figures/morph_AB_*.mp4` — one side-by-side movie, both panels at the same metres-per-inch,
one fixed colour scale for the whole movie and both panels, and every frame captioned with
`t_hydro`, `t_morph` and the morphological factor.

## Deviations and traps found in FUNWAVE (all verified in source)

- `C_smg`, **not** `Csmg` — a misspelled key is silently defaulted, never an error.
- `MinDepthPickup = 0.1` (the shipped `sediment_rip` value) **switches off bank-toe pickup**,
  i.e. the entire bank-retreat mechanism. It cannot be 0: the log-law drag
  `0.16/[ln(30H/k_s) − 1]²` is singular at `H = e·k_s/30 = 1.13e−4 m`. We use **0.01 m**.
- `Cd` in `input.txt` and the sediment module's internal log-law drag are **independent**.
  We set `Cd = 0.00154` so flow and transport see the same bed; the example's 0.002 is 30% off.
- `Hard_bottom` / `Zs` is applied as `IF(Zb>Zs) Zb=Zs` with `Zb>0` = erosion, so **`Zs=0`
  blocks erosion only** — the buffer still accretes (measured: 0 eroding cells, 120 accreting).
  It quarantines the inlet artefact; it is **not** a frozen bed and is excluded from analysis.
- FUNWAVE's sediment scalar flux is hard-zeroed on **all four** open boundaries
  (`FLUX_SCALAR_BC`), hence the buffer. `PERIODIC` is south–north only, so there is no
  streamwise-periodic reach.
- `WS` must be computed (Soulsby 1997 gives 0.0745 m/s for 0.5 mm quartz); the example's
  0.0125 is for ~0.1 mm sand.
- `Morph_interval` must exceed the suspension relaxation time `T_c = H/(γ w_s) = 20 s`;
  we use 200 s ≈ 10 `T_c`.
- Build **serially** — `make -j` races on Fortran `.mod` files. `PRECISION = double` is
  mandatory: the bed moves ~7e−8 m per step.
- `y_c` sinuosity: the small-`Ak` expansion `1 + (Ak)²/4` is **7.9% high** at B1's `Ak = 1.42`.
  We integrate the arc length.

## Verification

`tests/test_bathy.py` — 40 checks, each pinned by a second independent route (finite-difference
curvature vs the design `C₀`; brute-force nearest-distance vs the KD-tree projection; numeric
`∂q/∂n` from the built section vs the closed form; centre-of-curvature distances vs the
inner/outer sign convention).

`postprocessing/01_validate.py` — post-run gates: spin-up steadiness, achieved normal flow and
momentum balance, normal termination, no erosion in the buffer, the inlet artefact not
dominating the first interior bend, and bed change per output interval < 10% of `H`
(the Morph_factor sanity check).
