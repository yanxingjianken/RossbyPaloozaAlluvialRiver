# experiments/ — three ways to reach the morphological timescale

The same meandering-channel problem, organised by **how the fast flow is handled**
relative to the slow bank migration (see [`../docs/timescale_review.md`](../docs/timescale_review.md)).
The two Thetis experiments differ only in the secondary-flow parameter `A`.

| subfolder | family | flow treatment | `A` | question it answers |
|---|---|---|---|---|
| [`A0_incised/`](A0_incised/) | **B** | steady SWE, Newton solve per bank move | **0** (incised) | does an A=0 meander amplify or decay, and which way does it migrate? |
| [`A2p89_alluvial/`](A2p89_alluvial/) | **B** | steady SWE, Newton solve per bank move | **2.89** (alluvial, Suga 1963) | does turning on the secondary flow flip decay → growth? |
| [`analytic_ikeda_bend/`](analytic_ikeda_bend/) | **C** | flow eliminated analytically (bend equation) | 0 and 2.89 | the linear-theory reference the two runs are measured against |

**Family A** (Crank–Nicolson + morfac) is the original time-marched path; it is
still selectable (`CONFIG['flow_solver'] = 'cranknicolson'`) but is not a
separate experiment here — it and family B solve the *same* physics, B just
reaches the morphological timescale ~1000× faster (one Newton solve vs ~1000 time
steps per bank move).

## Reproduce

```bash
# family B, both cases (each writes to its own experiments/<case>/):
#   edit CONFIG['A_ikeda'] = 0.0   -> A0_incised
#   edit CONFIG['A_ikeda'] = 2.89  -> A2p89_alluvial
micromamba run -n firedrake python meander_thetis.py

# figures for a case (case = folder name):
micromamba run -n fourcastnetv2 python postprocessing/03_growth_migration.py A0_incised
micromamba run -n fourcastnetv2 python postprocessing/02_bank_evolution.py  A0_incised m4 m8
```

Each `<case>/` holds `outputs/` (per-frame + summary npz, gitignored) and
`figures/` (the growth/migration diagnostic and the 4/5-row movies).
