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

2. **The migration direction lives in the FLOW-RESPONSE phase, not the erosion
   law.**  The erosion law is UNIFIED — `river.pdf` p.19 == Ikeda `γ ∂_t y = E u'_b`
   with `E = ε C_f` (proved in `docs/River_Meandering_SW_corrected.pdf`, eqs
   31-35).  So up/down cannot come from the erosion closure; it comes from the
   PHASE of the near-bank velocity `u'_b` relative to the bend curvature, which
   is set by the *flow model*:
   - Thetis full 2-D SWE gives a DOWNSTREAM-lagged `u'_b` in EVERY case tested
     (limit 1 & 2, steady & unsteady, Fr 0.09-0.30) -> downstream migration
     (verified by convention-free d(crest)/dt on clean-amplitude frames).
   - The deck's 3-level QGPV model (`../vorticity_meander`) gives an upstream
     `u'_b` phase -> `c₀ = -E D/γ < 0`; but it is a *cross-channel-truncated*
     reduction, archived under `rossby_limit2/vorticity_upstream_reference/`.

The SW-note limit-2 QGPV equation IS a Rossby wave mathematically.  **Open
question:** does the *exact* 2-D SWE at limit 2 give an upstream `u'_b` phase, or
is the deck's upstream a 3-level-truncation artefact?  My α≈1, Fr=0.09 SWE runs
give downstream — either not deep enough in the limit, or the full SWE genuinely
differs from the 3-level reduction.

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
