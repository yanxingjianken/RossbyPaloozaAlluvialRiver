# Rossby Palooza — verified numerical models of the literature

One package per item in [`../literature/`](../literature/): each is a self-contained, **machine-verified explainer** — a `<paper>_lib.py` whose equations are transcribed from the PDF *and* independently re-derived, a `_self_test()` that reproduces the paper's own published numbers, figure/animation scripts, and a README documenting provenance and every deviation found.

## The five-source map

```
        Schumm (1967, Science)                 empirical milestone: wavelength is set by
                 │                             discharge AND sediment type (λ∝Q^0.34/M^0.74);
                 ▼                             the sediment control later becomes Ikeda's A
        Ikeda, Parker & Sawai (1981)           St Venant linear stability: 3 governing eqs
                 │                             + bed BC (η'/H=−AC·ñ) + bank-erosion BC
                 │                             (γ∂y/∂t=E·u'_b) → bend eq. → dispersion
                 │                             (note 2C_f² in α₀) → k_OM selection,
                 ▼                             DOWNSTREAM migration (c₀>0 always)
        Parker, Sawai & Ikeda (1982)           3rd-order modified-Stokes expansion:
                 │                             growth/migration slow with amplitude,
                 │                             wavelength shifts (e*(F)=5.1/2.7 boundary),
                 │                             fattening J_F + skewing J_S third harmonics
        ┌────────┴──────────────┐
        ▼                       ▼
  Rossby-Palooza deck      Bahmanpouri et al. (2022, WRR)
  (2026-06-30)             UAV + entropy theory: mean velocity & discharge
  vorticity dynamics:      from ONE surface velocity (Φ(M)=U_m/U_max),
  parabolic jet = β        dip phenomenon from secondary currents, <13% error —
  analogue; erodible       modern calibration of the theories' inputs
  banks → UPSTREAM-
  propagating meanders
```

## Packages

| package | theory | headline machine-checked reproductions |
|---|---|---|
| [`ikeda_1981/`](ikeda_1981/) | linear bend instability | β=1.50, α_OM=0.564k², ω_OM=1.17k², phase lag 64°≈0.18λ; + Eq.-16 PDE integrated: selection emerges, spectrum matches normal modes to 5e-10 |
| [`schumm_1967/`](schumm_1967/) | empirical wavelength laws | refit of the 36 transcribed sections reproduces r=.95/.93, R²=89/86%, SE=.16/.19, Q-alone 43/40% — the paper's own fit is the transcription checksum |
| [`parker_1982/`](parker_1982/) | nonlinear fattening & skewing | J_FM=0.0478/0.0469 anchors; e*=5.12/2.73 thresholds; full Eq.-7 PDE tracks the theory to 0.07% (growth) / 0.1% (\|J\|); 4 transcription glyphs resolved by dual routes |
| [`bahmanpouri_2022/`](bahmanpouri_2022/) | entropy velocity method | Φ(M) identity on 6/7 Table-2 rows (+codified published typo in FM CS3); digitized Sajó bathymetry A=13.27 vs printed 13.67; full one-surface-velocity pipeline within 10% of Table 3 |
| [`vorticity_meander/`](vorticity_meander/) | **the group's own theory** (6/30 deck) | **rebuilt from scratch 2026-07-12 (v2)**: pins re-digitized independently (agree with v1); p.5 box identity exact; rigid banks neutral; upstream c₀=−ED/γ_eff; TWO friction closures (deck-literal Rayleigh + Ikeda-consistent momentum drag) both fit all 6 phase intercepts (exact E↔2E degeneracy, ECOEF=0.5/1.0) — peak gap 3.2–4.4× (ray) vs 2.3–3.2× (mom): **friction closure alone ruled out as the explanation**; suspects now wavy-bank pressure drag + growth-axis normalization; see its `THEORY.md` for the equations vs Ikeda Eq. 7 |

## Quickstart

```bash
# every package: run the lib directly for its self-test
cd <package> && micromamba run -n fourcastnetv2 python <paper>_lib.py
# then the numbered scripts in order (figures + mp4s; --max-frames 1 smoke-tests animations)
```

Environment: `fourcastnetv2` micromamba env (numpy, scipy, matplotlib, imageio+libx264, PIL).

## House rules embodied here

- **No synthetic data.** Demo/validation inputs are the papers' printed parameters, transcribed tables, or programmatically digitized figures — every `data/*.csv` carries a provenance header (source table/figure, method, date, checksum). `grep -rn "np.random" */` returns nothing.
- **Transcribe + independently re-derive.** Every dense printed equation is pinned by a second computational route (derivation, numerical inversion, printed anchors, or direct PDE integration). This session that discipline caught: a swapped Q_m/Q_ma pair and reversed class definitions (Schumm briefing), four glyph-level misreads in Parker (½eβ², 9/8k⁵, ½β⁴, plain-A in Eq. 7), a fabricated "J_SM=0.0103", the Beaver δ₀ ambiguity, and a published typo in Bahmanpouri Table 2.
- **Reproduce in-progress work honestly.** The vorticity package asserts its one quantified disagreement with the deck rather than tuning it away.
- Plot/animation helpers are duplicated per package inside a fenced `shared helper block v1` (md5-identical; check with the diff one-liner in any README).

## Status / deferred

- Beamer decks: `ikeda_1981/slides/` has the compiled 23-slide talk; the other four packages' decks are deferred (figures are slide-ready PNGs).
- `vorticity_meander`: v1 (2026-07-06) torn down and rebuilt as v2 (2026-07-12) with a dual friction closure; the v1 open question is answered (friction closure alone ≠ the peak gap) and sharpened (wavy-bank pressure drag / growth-axis normalization). Next natural step: the 2-D numerical model of the deck p. 8 goals, which contains both closures and the bank pressure drag by construction.

*Built 2026-07-06. The root-level `../ikeda_1981/` original build (superseded by the extended copy here) and the empty misspelled `../banmanpouri_2022/` were removed 2026-07-13 when the repo went under git; this tree is canonical.*
