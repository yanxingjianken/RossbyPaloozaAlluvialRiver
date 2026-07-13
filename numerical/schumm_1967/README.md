# Schumm (1967) — Meander Wavelength of Alluvial Rivers

A verified numerical explainer of the empirical milestone

> Schumm, S. A. (1967) **"Meander Wavelength of Alluvial Rivers."** *Science* **157** (3796), 1549–1550. doi:10.1126/science.157.3796.1549

The "model" here *is* the pair of fitted power laws — the first quantitative demonstration that meander wavelength is set by **discharge AND sediment type**, not discharge alone. Every quantitative claim is computed from the verified relations and the transcribed real dataset in [`schumm_lib.py`](schumm_lib.py) / [`data/`](data/).

## The physics in one paragraph

Carlston (1965) fit λ to mean annual discharge for 14 eastern-US rivers and explained 98% of the variance; on Schumm's 36 Great-Plains + Murrumbidgee sections the same single-variable idea explains only **43%**. The missing axis is what the river carries: channels transporting sand and gravel (bedload, silt-clay index M < 5%) meander **long**, channels moving fine suspended loads (M > 20%) meander **short** — a **tenfold** wavelength range at fixed discharge. Multiple regression recovers

```
λ = 1890 Q_m^0.34  / M^0.74     (1)  r = .95, 89% explained, SE 0.16 log units
λ =  234 Q_ma^0.48 / M^0.74     (2)  r = .93, 86% explained, SE 0.19 log units
```

(λ ft; Q_m mean annual discharge, Q_ma mean annual flood, cfs; M weighted % silt-clay in the channel perimeter). In the later mechanistic theory (Ikeda et al. 1981, this repo's sibling package) the same sediment control reappears as the transverse-bed-slope parameter A.

## The verified core

| lib function | source | published target (asserted in self-test) |
|---|---|---|
| `wavelength_qm` | Eq. (1) | frozen constants 1890 / 0.34 / 0.74 |
| `wavelength_qma` | Eq. (2) | frozen constants 234 / 0.48 / 0.74 |
| `classify` | p. 1549 | bedload M<5, mixed 5–20, suspended M>20 |
| `fit_power_law` on `data/` | Eq. (1) refit | coef≈1890, exps 0.34/−0.74 (±0.03), r=.95±.01, R²=.89±.02, SE=.16±.02, Q-alone 43%±3 |
| `fit_power_law` on `data/` | Eq. (2) refit | coef≈234, exps 0.48/−0.74 (±0.03), r=.93±.01, R²=.86±.02, SE=.19±.02, Q-alone 40%±3 |
| `loo_log_errors` | honesty check | ≥80% of sections within ±2 SE (measured: 97%) |
| `tenfold_range_factor` | p. 1550 claim | (44.6/1.3)^0.74 = 13.7× > 10× |

Refitting the regressions on the transcription reproduces **every published statistic to rounding precision** — Schumm's own fit is the checksum of the data (see Data provenance).

## Quickstart

```bash
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/schumm_1967

# 0. self-test: recompute the published statistics from the transcribed data
micromamba run -n fourcastnetv2 python schumm_lib.py

# 1. static figures -> figures/fig01..fig06
micromamba run -n fourcastnetv2 python 01_regressions.py
micromamba run -n fourcastnetv2 python 02_residuals.py
micromamba run -n fourcastnetv2 python 03_predict_demo.py

# 2. animation -> figures/m_sweep.mp4 (+preview). --max-frames 1 for a smoke run.
micromamba run -n fourcastnetv2 python 04_anim_m_sweep.py
```

## File map

| File | Output |
|---|---|
| [`schumm_lib.py`](schumm_lib.py) | verified relations + regression engine + self-test |
| [`01_regressions.py`](01_regressions.py) | `fig01` λ–Q_m (paper Fig. 1 replica, Carlston line, M-family) · `fig02` λ–Q_ma (Fig. 2 replica, Dury line) |
| [`02_residuals.py`](02_residuals.py) | `fig03` measured-vs-calculated (Fig. 3 replica, ±2 SE band) · `fig04` Q-only residuals vs M (slope −0.74) · `fig05` the λM^0.74 collapse |
| [`03_predict_demo.py`](03_predict_demo.py) | leave-one-out table + `fig06` |
| [`04_anim_m_sweep.py`](04_anim_m_sweep.py) | `m_sweep.mp4` — Eq. (1) line sweeping M through 1.3→44.6%, lighting up each channel class |
| `data/schumm_1967_sections.csv` | the 36 real sections (see provenance) |
| `slides/` | (beamer deck — pending) |

## Data provenance

The 3-page Science report prints **no data table**. Following its refs (5) and (7), the sections were transcribed from **USGS Professional Paper 598** (Schumm 1968, public domain, pubs.usgs.gov):

| rows | source | method |
|---|---|---|
| 1–33 | PP 598 Table 6 "Data for rivers of Midwestern United States" (p. 45) | manual transcription 2026-07-06 from the rendered scan page, cross-checked against the OCR text layer (`pdftotext -layout`); single-digit ambiguities resolved toward the layout text |
| 34–36 | PP 598 Table 1, Murrumbidgee sections 2/4/5 (the three with measured λ) | same; Q_m uses the diversion-adjusted values where printed (per PP 598 Fig. 31) |

Internal consistency checks that passed: exactly 5 US rows lack bankfull discharge (PP 598: "bankfull discharge of 28 was calculated"); refitting reproduces r=.95/.93, R²=89%/86%, SE=0.16/0.19, and the 43%/40% Q-alone contrasts. **No synthetic or randomly generated data anywhere** (house rule); the only inputs are the published tables.

## Deviations & caveats

- The Science text layer garbles both equations (`1890 Qm'°3/M0'`, `234 Qma0Od/Mo0r4`); exponents 0.34/0.48/0.74 were read from the **rendered** page and validated by the refit.
- An earlier internal briefing mis-stated Eq. 1 as the Q_ma relation (and vice versa) and gave exponents 0.5/−0.44; the paper is authority: **Eq. 1 = Q_m** (r=.95), **Eq. 2 = Q_ma** (r=.93), exponents as above. It also had the class definitions reversed; the paper's are as in `classify`.
- PP 598's own three-variable regression (λ = 438·Q_b^0.43/M^0.47, bankfull, n=34) is a *different* fit from Schumm 1967 Eqs. (1)–(2) and is not implemented here.
- The standard error convention that reproduces the paper's 0.16/0.19 is SEE = √(SS_res/(n−3)).

## Shared helper block

`set_style` / `save_fig` / `fig_to_rgb` / `write_mp4` are duplicated byte-identically across the rossby_palooza packages (fenced `=== shared helper block v1 ===`). Sync check:

```bash
diff <(sed -n '/shared helper block/,/end shared helper block/p' schumm_lib.py) \
     <(sed -n '/shared helper block/,/end shared helper block/p' ../bahmanpouri_2022/bahmanpouri_lib.py)
```
