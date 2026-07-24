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
direction is set by the **bank-erosion closure**, not the flow regime or solver:

- Ikeda's near-bank-velocity closure (what Thetis uses) -> **downstream**, in
  every regime tested (limit 1 & 2, steady & unsteady, Fr 0.09-0.30).
- The deck's 3-level **vorticity** closure -> **upstream** (`c0 = -E D/gamma`),
  archived here from `../../vorticity_meander/figures/fig09_upstream_speed_map.png`.

The SW-note limit-2 QGPV equation *is* a Rossby wave mathematically; but the 2-D
SWE meander it drives (via Ikeda's near-bank law) still migrates downstream.  The
upstream migration is a property of the vorticity bank closure, not of the flow's
Rossby/gravity regime.  See `../../param_search.py` for the analytic verdict and
`../summary_growth_and_closure.png` for the one-figure summary.
