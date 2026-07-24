# rossby_limit2/ — the SW-note LIMIT 2 (QGPV / shear-Rossby regime)

Limit 2 of `River_Meandering_SW.pdf` (eqs 23-26): `alpha=1` (lambda~pi*W, tight
meanders) + `Fr^2~eps` (weak gravity), which brings the PV dynamics to leading
order and yields the **QGPV / shear-Rossby equation (26)**.  This is the regime
the deck predicts migrates **upstream**.

## Two realisations, two answers

| | model | closure | result |
|---|---|---|---|
| `outputs/`, `steady_downstream/` | **Thetis 2-D SWE** (this package), steady solver, alpha~1, Fr=0.09 | Ikeda near-bank `gamma dy/dt = E u'_b` | **DECAY, DOWNSTREAM** |
| `../rossby_limit2_unsteady/` | Thetis 2-D SWE, **unsteady** (CrankNicolson, flow dt retained), alpha~1 | Ikeda near-bank | **DECAY, DOWNSTREAM** (77 morph frames) |
| `vorticity_upstream_reference/` | **3-level QGPV** (`../../vorticity_meander`, `../../deliverable1_noboru_model`) | **vorticity** `c0 = -E D/gamma` | **UPSTREAM** (`c0<0`) |

## The finding

Putting Thetis in limit 2 (alpha~1, low Fr) does **not** flip the meander
upstream -- with **either** the steady **or** the unsteady solver.  So the
`dt`/steady-vs-unsteady hypothesis is ruled out empirically.  The migration
direction is set by the **flow-response phase of u'_b** (NOT the erosion law, which is
unified -- river.pdf p19 == Ikeda, E=eps*Cf, corrected PDF eqs 31-35):

- Thetis full 2-D SWE -> **downstream-lagged u'_b phase** in every regime tested
  (limit 1 & 2, steady & unsteady, Fr 0.09-0.30) -> downstream (verified by
  convention-free d(crest)/dt on clean-amplitude frames).
- The deck's 3-level QGPV **reduction** (cross-channel truncated) -> **upstream
  u'_b phase** (`c0 = -E D/gamma < 0`), archived from
  `../../vorticity_meander/figures/fig09_upstream_speed_map.png`.

Same erosion law in both; the difference is how the FLOW computes u'_b.
**Open:** does the exact 2-D SWE limit-2 give upstream, or is the deck's upstream
a 3-level-truncation artefact?  The alpha~1 SWE runs here say downstream.

The SW-note limit-2 QGPV equation *is* a Rossby wave mathematically; but the 2-D
SWE meander it drives (via Ikeda's near-bank law) still migrates downstream.  The
upstream migration is a property of the vorticity bank closure, not of the flow's
Rossby/gravity regime.  See `../../param_search.py` for the analytic verdict and
`../summary_growth_and_closure.png` for the one-figure summary.
