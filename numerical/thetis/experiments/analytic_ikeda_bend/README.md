# analytic_ikeda_bend/ — family C: the flow eliminated analytically

**Archive** of the analytic-slaving reference (family C in
[`../../docs/timescale_review.md`](../../docs/timescale_review.md)). This is the
limit of the steady-solver approach: the quasi-steady flow is not *solved* per
step, it is eliminated *symbolically*, collapsing the coupled flow+bank system to
a single evolution equation for the planform `y(x, t)`. No hydrodynamic solve at
all — the ultimate morphological acceleration.

The bend equation, Ikeda, Parker & Sawai (1981) eq. (16),

```
y_xt + 2 C_f y_t = y_xxx − C_f (A + F²) y_xx ,
```

*is* the bank law with the flow already slaved out. Its linear dispersion gives
the growth rate `α₀(k)` and celerity `c₀(k)` that the family-B Thetis runs in
`../A0_incised/` and `../A2p89_alluvial/` are measured against
(`postprocessing/03_growth_migration.py` overlays them).

## Provenance

The canonical, self-testing implementations live **outside this package** and are
not duplicated here — only their headline figures are copied in:

| file here | copied from | shows |
|---|---|---|
| `figures/fig06_growth_rate.png` | `../../../ikeda_1981/` | `α₀(k)` — growth band, cutoff `k_c` |
| `figures/fig07_celerity.png` | `../../../ikeda_1981/` | `c₀(k) > 0` — always downstream (A=0) |
| `figures/fig08_dispersion_combined.png` | `../../../ikeda_1981/` | full dispersion |
| `figures/fig04_bank_erosion.png` | `../../../ikeda_1981/` | the `γ ∂y/∂t = E u'_b` law |
| `figures/dispersion.png` | `../../../meander_migration/` | SWE-extended dispersion |
| `figures/meander_migration.mp4` | `../../../meander_migration/` | the planform migrating (bend model) |

The code and its verification (`ikeda_lib.py` self-test; the numbers
`β=1.50`, `k_OM=0.564k²`, phase lag 64°, etc.) remain in
[`../../../ikeda_1981/`](../../../ikeda_1981/) and
[`../../../meander_migration/`](../../../meander_migration/). This folder is a
pointer + snapshot so the three families sit together for comparison, not a
second copy of the package.

## The key contrast the three families draw

- **C (here)** predicts, for A=0, a *weak* growth (`α₀/C_f² = +2×10⁻³`) and
  *downstream* migration (`c₀ > 0`).
- **B (`../A0_incised/`)** tests that prediction with the full nonlinear 2-D
  flow — and finds **decay**, because the predicted growth is ~675× weaker than
  the alluvial case and is overwhelmed by numerical/viscous diffusion. Migration
  is downstream, as C predicts.
- **B (`../A2p89_alluvial/`)** turns on the secondary flow to test whether the
  `A` term is what supplies the growth C attributes to it.
