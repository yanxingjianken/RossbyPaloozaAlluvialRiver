# thetis — depth-averaged 2D shallow water on a meandering channel with migrating banks

Nonlinear, finite-element, **moving-boundary** meander model: Thetis (Firedrake)
solves the Cartesian shallow-water equations on a *curved physical mesh* whose
**banks migrate** under an Ikeda-type erosion/deposition law, over a **frozen**
bed that varies across the channel.

This is the gap the other packages in `../` leave: `dedalus_meander_full_SW` is
linear and spectral, `funwave_2d_sw` is nonlinear finite-volume with a mobile bed
and *rigid* walls. Here the bed is rigid and the **banks** are the morphology.

| deliverable | file |
|---|---|
| governing equations, BCs, drag, erosion/deposition, the `A = 2.89` decision | [`docs/model.md`](docs/model.md) |
| the SW note, reconstructed with minimal **red** corrections + a new bank-erosion section | [`docs/River_Meandering_SW_corrected.pdf`](docs/) |
| xOy + yOz initial-condition figures, both runs | `figures/IC_m4.png`, `figures/IC_m8.png` |
| 4-row bank-evolution movie, one per run | `figures/bank_evolution_m{4,8}.mp4` |

## Quickstart

```bash
bash build_env.sh                                    # Phase 0, ~1-2 h, detached
micromamba run -n fourcastnetv2 python tests/test_setup.py   # tier 1 (no Firedrake)
micromamba run -n firedrake     python tests/test_setup.py   # tier 1 + 2
bash run_all.sh                                      # tests -> 2 runs -> figures -> movies
```

## Layout

```
build_env.sh          Phase 0: micromamba env + PETSc from source + Firedrake + Thetis
geometry.py           channel map, frozen bed, exact base state, planform   [self-test]
sw_note.py            the CORRECTED note system (28)-(30), solved directly  [self-test]
meander_thetis.py     THE DRIVER -- CONFIG at the head, no CLI, writes data only
postprocessing/
  pp_lib.py           shared helpers (+ the byte-identical rossby_palooza block)
  01_ic.py            xOy + yOz IC figures
  02_bank_evolution.py  the 4-row mp4
tests/test_setup.py   gates, tiered so tier 1 runs without Firedrake
docs/                 model.md, the corrected SW note (.tex + .pdf)
thetis-src/           git clone of thetisproject/thetis        [gitignored]
outputs/              run data                                  [gitignored]
```

## The design, in one table

Every number is computed by `geometry.py`, not asserted.

| | |
|---|---|
| `C_f` = 0.05, `F` = 0.30, `A_ikeda` = **0** (incised), `ν` = 0.05 m²/s | |
| `I = C_f F²` = 4.5e-3 — exact, from Ikeda (2) | |
| `W` = 35.10 m, `H̄` = 1.000 m, `Ū` = 0.936 m/s | |
| `λ_OM` = 421.2 m = **12.00 W** | m=4 sits exactly at Ikeda's `k_OM` |
| `L` = 1965.6 m = 175.5 (straight entry) + 1684.8 (meander) + 105.3 (exit) | entry = **17.6 friction adjustment lengths** |
| mesh 224 × 28 → 12 544 triangles | transit `L/Ū` = 2099.5 s |

**Two runs**, wavenumber 4 and 8 over the meander reach. The reach length is set
so `m=4` lands exactly on Ikeda's fastest-growing wavenumber and `m=8` at
`2k_OM > k_c`, giving a growing/decaying pair for free:

```
alpha0/Cf^2 :  m=4  +1.98e-03      m=8  -1.44e-02
```

⚠️ Those are **linear-theory predictions, not guarantees**. The design point sits
*between* the SW note's two distinguished limits (`α` = 0.26/0.52,
`F_c` = 3.35/1.68), so grow-vs-decay is the experiment.

## Things that are decided, and why

- **`A = 2.89` is omitted (`A = 0`).** Substituting Ikeda (5)+(6) into (3b) gives
  `h′/H = (F² + A)·𝒞′ñ`: `A` and `F²` share one slot, a depth/drag modulation.
  A 2D model computes the `F²` half exactly and **cannot** produce the `A` half —
  it comes from 3D helical circulation. So secondary flow **is** needed for the
  alluvial case, and setting `A = 0` selects Ikeda's **incised** branch instead.
  Cost, computed: `k_c` 5.75× smaller, peak growth ~675× weaker.
  **Which bank erodes therefore becomes a measured output, not an assumption.**
  See `docs/model.md` §6.
- **`river.pdf` p.19 and Ikeda (11)–(13) are the same law**, with `E = ε·C_f`
  — derived in `docs/model.md` §7.1 and corroborated independently by the
  `deliverable1_noboru_model` result that only the product `εC_f` ever appears.
  `E_e ≠ E_d` (independent erosion/deposition) is a strict generalisation of both;
  it is **not** in Ikeda and has **no calibration on disk**.
- **Movies only.** No `T_shear`, PV-budget or gravity-vs-vortical diagnostics.
  The repo's prior retraction of "the meander *is* the vortical/Rossby wave"
  stands untouched and is not re-tested here.

## House rules

- `grep -rn "np[.]random"` returns nothing (asserted by `tests/test_setup.py`).
- The driver holds one run in a `CONFIG` dict and **writes data only**; every
  figure and movie lives in `postprocessing/`.
- The shared helper block in `postprocessing/pp_lib.py` is byte-identical to the
  other packages' (asserted).

## Traps hit while building this (all now guarded)

1. **`np.trapezoid` is numpy ≥ 2.0**; `fourcastnetv2` and the Firedrake env sit on
   opposite sides of that split.
2. **A trapezoid check on a quartic hid a 1e-6 error.** The width-mean-depth gate
   now uses Gauss–Legendre (exact through degree 15) instead of loosening the
   tolerance — loosening would have masked real design errors.
3. **Pattern self-match, three times**: `pgrep -f build_env.sh` matched its own
   command line; an `awk` range over the shared-helper fence matched the
   *docstring* that described it; `grep np.random` matched the test that greps for
   it. Bracket the literal (`np[.]random`) or the pattern finds itself.
4. **Unquoted `$CONF_OPTS`** word-split `--COPTFLAGS='-O3 -march=native ...'` into
   three arguments and PETSc rejected it; `eval "set -- $OPTS"` re-parses quoting.
5. **DG1 solution vs CG1 coordinates have different lengths.** The driver projects
   to CG1 before saving, and asserts the shapes match — otherwise postprocessing
   would silently mis-pair values with positions.
6. **A low-resolution render made me "fix" a colour bug that did not exist.**
   Pixel-sampling at 600 dpi showed the LaTeX colouring had been correct all
   along. Measure before patching.
