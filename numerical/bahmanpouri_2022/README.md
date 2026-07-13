# Bahmanpouri et al. (2022) — Mean River Velocity from One Surface Velocity

A verified numerical explainer of

> Bahmanpouri, F., Barbetta, S., Gualtieri, C., Ianniruberto, M., Filizola, N., Termini, D. & Moramarco, T. (2022) **"Estimating the Average River Cross-Section Velocity by Observing Only One Surface Velocity Value and Calibrating the Entropic Parameter."** *Water Resources Research* **58**, e2021WR031821. doi:10.1029/2021WR031821

Every relation is transcribed from the PDF and verified in [`bahmanpouri_lib.py`](bahmanpouri_lib.py); the demo inputs are the paper's own printed anchors and programmatically digitized figures — **no synthetic data anywhere**.

## The physics in one paragraph

Chiu's entropy theory gives a two-parameter closed form for a river's velocity field: a vertical profile (Eq. 1) whose fullness is set by one entropy parameter M and whose maximum can sit a dip h *below* the surface (the signature of secondary currents), and a linear relation (Eq. 2) U_mean = Φ(M)·U_max with Φ(M) = e^M/(e^M−1) − 1/M. Because Φ(M) is constant at a gauged site, a **single surface-velocity observation** (from a drone) plus bathymetry is enough: convert the observed surface value to the vertical's maximum (Eq. 4 — exactly Eq. 1 inverted at the surface), spread the surface velocity across the section with a parabolic or elliptic scenario (Corato et al. 2011), depth-integrate, and read off mean velocity and discharge. On the Sajó (Hungary) and Freiberger Mulde (Germany) the method lands within **13%** of ADCP measurements — parabolic scenario within ~1–2%.

## The verified core

| lib function | paper source | self-test |
|---|---|---|
| `phi_of_M` / `M_of_phi` | Eq. (2) | series ½+M/12−M³/720 at small M; monotone ½→1; overflow-safe at M=700; round-trip 1e-9 |
| `u_vertical` | Eq. (1) | U(0)=0; max at y=D−h; h=0 ⇒ U(D)=U_maxv |
| `umaxv_from_surface` | Eq. (4) | h=0 identity (paper text); **dual route: inverts Eq. 1 at y=D to 1e-12** |
| `dip_solve` | §3 (Moramarco 2017 idea) | recovers a known dip to 1e-8 |
| Table 2 (`data/`) | p. 6 | Φ(Mᵢ)=Φᵢ ±0.005 and Φᵢ=U_m/U_max ±0.01 for **6/7 rows**; ranges (0.605–0.732)/(1.16–3.24) |
| **FM CS3 anomaly** | p. 6 | printed (Φ=0.678, M=1.16) must FAIL the identity: Φ(1.16)=0.595=0.68/1.14 — a codified published typo |
| Tables 3–4 (`data/`) | pp. 8, 10 | error %s recomputed ±0.05; Q-err ≡ vel-err (Q=U·A construction); max 10.52% < 13% (abstract) |
| anchors | §4.2 | 0.82·13.67=11.21; 0.57·9.85≈5.6 |
| full pipeline | Fig. 2 flowchart | on digitized Sajó CS1 bathymetry: parabolic U_m=0.757 (paper 0.81, −6.5%), Q=10.04 (11.07, −9.3%), inside 10/12% tolerance; elliptic>parabolic ordering as in Table 3 |

## Quickstart

```bash
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/bahmanpouri_2022
micromamba run -n fourcastnetv2 python bahmanpouri_lib.py     # self-test (3 tiers)
micromamba run -n fourcastnetv2 python 01_phi_curve.py        # fig01-02
micromamba run -n fourcastnetv2 python 02_vertical_profiles.py# fig03-04
micromamba run -n fourcastnetv2 python 03_cross_section_maps.py# fig05-06 (money figures)
micromamba run -n fourcastnetv2 python 04_surface_scenarios.py# fig07-08
micromamba run -n fourcastnetv2 python 05_validation.py       # fig09-10
micromamba run -n fourcastnetv2 python 06_anim_entropy_family.py  # entropy_family.mp4
```

## File map

| File | Output |
|---|---|
| `bahmanpouri_lib.py` | verified Eqs. 1/2/4, scenarios, pipeline, 3-tier self-test |
| `01_phi_curve.py` | `fig01` Φ(M) backbone + 7 transects + literature bands + CS3 callout · `fig02` three-route identity bars |
| `02_vertical_profiles.py` | `fig03` profile family (M, dip) · `fig04` Sajó deepest-vertical profile (Fig. 8a analogue) |
| `03_cross_section_maps.py` | `fig05` Sajó CS1 2D map (paper Fig. 5 style; no-dip vs dip) · `fig06` FM CS3 (Fig. 7 style) |
| `04_surface_scenarios.py` | `fig07`/`fig08` parabolic vs elliptic surface distributions (Figs. 4/6 style) |
| `05_validation.py` | `fig09`/`fig10` measured vs paper vs this-pipeline bars with 13% band |
| `06_anim_entropy_family.py` | `entropy_family.mp4` — M sweep: profile morphs, Φ point slides, Sajó Q sensitivity |
| `data/` | see provenance |

## Data provenance

| file | source | method |
|---|---|---|
| `table2_entropy_params.csv` | Table 2, p. 6 | manual transcription from rendered page (clean text layer) |
| `tables34_validation.csv` | Tables 3–4 + §4.2 anchors | manual transcription |
| `sajo_cs1_bathymetry.csv` | Fig. 5(a) | **programmatic** color-boundary digitization of the 200-dpi render (script `data/digitize_figures.py`); axis-tick auto-calibration; checksum: area 13.27 m² vs printed **13.67 m²** (−2.9%) |
| `mulde_cs3_bathymetry.csv` | Fig. 7(a) | same; area 9.26 m², width 15.7 m (no printed CS3 area; plausibility-checked) |

## Deviations & caveats

- **FM CS3 printed Φ** (0.678) is inconsistent with its printed M (1.16) and its own ADCP ratio (0.68/1.14=0.596); the self-test asserts the anomaly *exists* rather than silently correcting it.
- Eq. (3) (ungauged-site Φ from roughness/hydraulic radius, Moramarco & Singh 2010) is not implemented — the paper's own workflow is ADCP-calibrated, and Eq. 3's inputs are not tabulated in the paper.
- The dip in the map figures uses a **flagged illustrative taper** (calibrated at the deepest vertical for Sajó; fixed 12% for FM CS3 where the Table-2 target is unreachable from the UAV anchor at M=1.16 — the paper itself notes a 15% UAV/ADCP gap on that river). The per-vertical Moramarco-2017 iteration needs SI profile data.
- Pipeline dip is h=0 (no-dip baseline); its −6.5% U_m bias vs the paper's 0.81 partly reflects that and the −2.9% digitized-area deficit.
- FM CS1 has no printed bathymetry figure (SI only), so the pipeline validation bar exists only for Sajó CS1.
- Table 5 (vertical-profile R²=0.55–0.86, SE 0.047–0.063) is cited but not re-derived (needs SI ADCP profiles).

## Shared helper block

`set_style`/`save_fig`/`fig_to_rgb`/`write_mp4` are byte-identical across packages (fenced `=== shared helper block v1 ===`); sync-check with the diff one-liner in the schumm_1967 README.
