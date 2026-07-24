# experiments/ — meander morphodynamics across regimes, solvers, and closures

The same meandering-channel problem, organised to answer two questions:
**what makes a meander amplify vs decay**, and **what sets its migration direction**.
See `../param_search.py` for the analytic verdict and
[`summary_growth_and_closure.png`](summary_growth_and_closure.png) for the one figure.

| subfolder | regime | flow solver | A | result |
|---|---|---|---|---|
| [`A0_incised/`](A0_incised/) | limit 1 (α=0.26) | steady (B) | 0 | **decay**, downstream (81 / 21 yr) |
| [`A2p89_alluvial/`](A2p89_alluvial/) | limit 1 | steady (B) | 2.89 | **amplify**, downstream |
| [`rossby_limit2/`](rossby_limit2/) | **limit 2 (α≈1)**, Fr=0.09 | steady (B) | 0 | decay, downstream |
| [`rossby_limit2_unsteady/`](rossby_limit2_unsteady/) | limit 2 | **unsteady (A, CN)** | 0 | decay, downstream (77 morph frames) |
| [`analytic_ikeda_bend/`](analytic_ikeda_bend/) | limit 1 | analytic (C) | 0 & 2.89 | Ikeda reference (downstream) |
| `rossby_limit2/vorticity_upstream_reference/` | limit 2 QGPV | 3-level model | — | **UPSTREAM** (vorticity closure) |

## The two findings

1. **A is the growth knob.**  The secondary-flow parameter A flips the meander
   from DECAY (A=0, gravity only, F²=0.09) to GROWTH (A=2.89, alluvial secondary
   flow) — both in the full nonlinear 2-D SWE.  Confirms Ikeda's thesis that the
   `A` (secondary-flow / vortical) term, not gravity, drives meandering.

2. **The bank closure sets the migration direction.**  Every Thetis case — limit
   1 & 2, steady & unsteady solver, Fr 0.09–0.30 — migrates DOWNSTREAM, because
   Thetis uses Ikeda's near-bank-velocity closure (`γ ∂_t y = E u'_b`).  Changing
   the flow *regime* (limit 1 → 2) or *solver* does not flip it.  UPSTREAM
   migration appears only with the deck's 3-level **vorticity** closure
   (`c₀ = −E D/γ`, `../vorticity_meander`), archived under
   `rossby_limit2/vorticity_upstream_reference/`.

The SW-note limit-2 QGPV equation IS a Rossby wave mathematically, but the 2-D
SWE meander it drives still goes downstream; upstream is a property of the
vorticity bank closure, not the flow's Rossby/gravity regime.

## Reproduce

```bash
bash run_case.sh 4 0.0        # A0_incised   (edit m, A)
bash run_case.sh 4 2.89       # A2p89_alluvial
bash run_limit2.sh 12         # rossby_limit2 (steady, α≈1, low Fr)
bash run_unsteady_limit2.sh 15 # rossby_limit2_unsteady (CN)
micromamba run -n fourcastnetv2 python postprocessing/03_growth_migration.py <case>
micromamba run -n fourcastnetv2 python postprocessing/02_bank_evolution.py  <case> m4 m8
micromamba run -n fourcastnetv2 python postprocessing/04_summary.py
```

Launches use `THETIS_*` env vars (see `meander_thetis._env_overrides`); the
driver derives the case folder from A (`geometry.case_name`) or `THETIS_CASE`.
