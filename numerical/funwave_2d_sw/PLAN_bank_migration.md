# PLAN (VERIFIED) — FUNWAVE nonlinear-SW meander, erodible banks, spin-up + evolution

Full nonlinear shallow water (NSWE, DISPERSION=F) on a carved sine-generated channel with two
straight buffers and an always-wet shelf; **interior fully erodible end-to-end**, banks retreat by
fluvial toe erosion + slumping; watch the bank *and* the flow (**total + anomaly**) develop and
evolve. Rossby-vs-gravity is deferred to the linearized dispersion in `meander_migration/`.

**Verification (3 independent reviewers, `academic-paper:plan` discipline, read the real source):**
- **Governing math — CORRECT.** NSWE (mass+2 mom, Γ₁=Γ₂=0); all sediment closures verified
  formula/sign/unit-correct against `mod_sediment.F`: log-law bed shear (`:1372`), van Rijn pickup
  (`:1401`), Cao deposition (`:1521`, always-on sink), MPM bedload + Ikeda + Talmon (`:1433-1478`),
  Exner (`:1685`). Bank-erosion applicability sound: pickup/bedload depend on `(|u|,H)` not slope →
  valid at the toe; the steep face is handled by the slope-triggered avalanche rule (`:1741`).
- **Boundaries — AUDITED** (table below).
- **Feasibility — CONDITIONAL.** Four gated risks (below); nothing infeasible.

## 1. Initial condition — the "spin-up" done SAFELY (key correction)
**Literal from-rest is wrong twice** (both reviewers, independently): with `TideBcType=CONSTANT`
(no `Time_ramp` — wavemaker-only), imposing `TideWest_U/ETA` onto `u=v=η=0` closes >99.9% of the
gap in the FIRST timestep — a **Heaviside step = dam-break startup bore**, worse than the
acceleration transient that already killed a run at t=963 s. It is also **over-specified**: ETA, U,
V are all pinned at a subcritical open boundary where only one characteristic should be set.
**Resolution — start from the analytic BASE flow (`u' = v' = 0`), not from rest.** The IC is the
normal-flow profile along the local channel tangent (build_case already writes this) = the
*zeroth-order* flow with **no meander-induced perturbation**. The curvature then forces `u', v'` to
grow from ~0 to steady — **that is the spin-up the experiment wants** (the *anomaly* spinning up),
and it sidesteps the bore + over-specification entirely. (Optional tiny broadband η seed to break
symmetry.) *If literal from-rest is truly required, hand-roll a ramp — 10–20 chained sub-runs
stepping `TideWest_U/ETA` up over ~0.3 transit, each hot-starting from the previous snapshot (the
existing spinup→morph hot-start mechanism). Flagged as the harder path; not the default.*

## 2. Boundary conditions (audited: IC value | evolution)
| Boundary | IC value (t=0) | Evolves by |
|---|---|---|
| **Inlet (W)** | analytic normal flow: `u=U` along tangent, `v≈0`, `η=+drop/2` | `TIDE_BC` nudges ETA,U,V → TideWest targets inside the 30-cell (75 m) sponge, 3×/step (>99.9%/step at i=1, →0 by i=30) |
| **Outlet (E)** | `TideEast_ETA=-drop/2`, **`TideEast_U=0`**, V=0 | same nudge; U-target 0 ⇒ damps only what *arrives* → self-adjusting sink, cannot over-extract (U=design drains the reach) |
| **Banks** | `Depth=h_sec(n)+S(s-s0)`, steepened bank → always-wet shelf `h_plain=0.20`; `H=h_sec≥MinDepth` everywhere; NO MASK_STRUC walls | `UPDATE_MASK` re-tests η vs −Depth + 4 neighbours every sub-stage (free waterline); `Bed_Change=T` + `Avalanche` migrate the topographic toe |
| **Sediment** | `Zs=0` in the two 1560 m buffers, `Zs=1e6` interior (static) | `FLUX_SCALAR_BC` zeros sed. flux on all 4 boundaries; `Zs` caps erosion only ⇒ buffers accrete (bounded by the buffer sink-freeze, scoped to `Zs<1`) |

## 3. Gated fixes (from feasibility + math reviews)
1. **Bank resolution** — assert **≥4 cells across the bank face** before launch (like the R/W assert). If repose-angle (`tanφ=0.7`→1.4 cells) violates it, accept an *erosionally-limited* bank, not a geometric-repose one. Avoids waterline chatter + slump aliasing.
2. **Buffer-edge Zs is now the ONLY erodibility jump** — **ramp `Zs` smoothly over 5–10 dx** (reuse the `taper()` cosine) instead of a hard step; **track buffer-edge `dZb/dt` as a named diagnostic with an abort threshold** (it can seed a knickpoint).
3. **Bedload divergence has no TVD limiter** (`:1691`, centered difference) — ringing/checkerboard risk exactly at the toe where `BedFluxX/Y` switches across `Tau_cr,bl`. Watch for it; add `nu_bkg`/hyperdiffusion if it appears (allowed).
4. **Avalanche relaxes only 1 steepest neighbour / `Aval_interval`** — a multi-directional slump needs several intervals ⇒ set **`Aval_interval` short** vs the first-collapse rate.
5. **Morph_factor** — MF=5's window was calibrated for point-bar Exner, not slump-dominated retreat; **re-measure empirically once avalanching is on**.
6. **Mass sink scoping** — confirm the buffer-writer keeps `Zs<1` strictly in the buffers so the erodible interior stays Exner-conservative.

## 4. Diagnostics (total AND anomaly — the new science)
- **Total:** `u,v,|U|,η`; bird's-eye (xOy) + yOz sections.
- **Anomaly:** `ū(n)=<u_s>_s` (interior-only along-channel mean cross-profile), `u'=u_s−ū`, `v'` →
  the meander-induced perturbation; bird's-eye of `u',v'` + near-bank `u'_b`.
- **Bank-toe position** vs time → retreat rate/direction (+ the ≥2-dx-to-shelf guard).
- **Per-step sediment mass budget** (eroded − deposited − discarded) → migration vs degradation, MEASURED.

## 5. Stage-0 gate (cheapest de-risk — feasibility's recommendation)
Before the full 6-bend run: a **STRAIGHT `buffer_len` reach ALONE (no meander)**, analytic-base IC,
rigid→erodible, ~1000 s, instrumented with max|u|/Froude + bank-toe + mass budget. Isolates
**bore vs slump-chatter vs interface-pile** in one cheap run. Gate the full run on it.

## 6. Deferred
Rossby-vs-gravity dominance → the linearized dispersion (`meander_migration/`, R=β_eff b²/(F²U)=31 →
vortical/Rossby-dominated). The nonlinear run here supplies the total+anomaly fields to compare.
