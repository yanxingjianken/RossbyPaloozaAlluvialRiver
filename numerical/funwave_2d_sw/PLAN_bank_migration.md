# PLAN — FUNWAVE nonlinear SW meander with a RIGID bed and MIGRATING banks

**Flip of the funwave_2d_sw experiment.**  Previously: mobile bed (Exner point bar), FIXED banks
(always-wet shelf froze them).  Now: **rigid (unchanging) non-flat bed, MOBILE banks** — carve a
wavy channel with two straight buffers, spin the full nonlinear shallow-water flow up **from
rest**, then let **bank erosion + deposition** migrate the channel, and watch the bank *and* the
flow (**total and anomaly**) evolve.  Rossby-vs-gravity is deferred to a linearized dispersion
diagnosis later (the swe_stability.py / Ikeda work); the priority here is the nonlinear SW run.

---

## 1. Geometry (reuse the built machinery)
- Curvature-defined sine-generated centreline (C0 = A k²), one wavelength case first (λ=1040 m),
  interior + **two straight buffer reaches** at the ends (sediment-flux BC artefact absorber).
- Non-flat cross-section `h(n)` = the constant-PV-gradient bed (unchanged from before).
- **Always-wet shelf** (`h_plain=0.20 m`) beyond the bank toe — REQUIRED: an oblique wet/dry
  bank is what broke every earlier run; the shelf keeps the waterline off the staircased bank.

## 2. Rigid bed + mobile banks  (the defining change)
Use the `Hard_bottom` erodibility field `Zs` to split the domain:
- **Zs = 0 (rigid)** in the DEEP channel core (`|n| < n_core`) and both buffers → the "non-flat
  but unchanging bed".
- **Zs > 0 (erodible)** in the BANK zone (`n_core < |n| < n_toe`) → only the banks erode/deposit.
- Apply the **buffer/deep-bed sink fix** (freeze `Zb` where `Zs<1`, discard would-be deposition)
  so the protected bed cannot silently accrete and emerge (that blew the earlier run up).
- **Fallback (if the split is fiddly):** keep the whole channel erodible (as before) and simply
  READ OUT the bank signal; the user OK'd "whichever is easier".  Decision deferred to the review.

**Bank migration mechanism:** outer-bank toe erodes (high near-bank shear), sediment advects and
deposits on the inner bank → the channel translates laterally.  Same van Rijn pickup + Cao
deposition + MPM bedload + Exner + avalanching physics as the point-bar run, now acting at the
banks because that is where `Zs>0`.

## 3. Initial condition: NOT steady — spin up from rest
- **IC:** `u = v = 0`, `η = 0` (flat), i.e. `u' = v' = 0`.  (Option: seed a tiny broadband `η`
  perturbation ~1e-3 m to break symmetry and let any instability select itself.)
- **Phase 1 — spin-up (rigid, Bed_Change=F):** ramp the inlet discharge (tide-BC) from 0 to
  design over ~0.3 transit times to avoid a bore, then hold until the flow reaches steady
  (G1 drift < 5%).  Watch it develop the meander-forced `u', v'` from a flat start.
- **Phase 2 — erosion ON (Bed_Change=T):** hot-start from the steady flow, turn on sediment,
  let the banks fully evolve (Morph_factor acceleration).  *Or* keep erosion on throughout if the
  spin-up is stable with it — the easier of the two, decided in the review.

## 4. Boundary conditions
- **Inlet:** steady discharge via TIDE_BC (TidalBcType=CONSTANT), ramped from rest; U, V, η all
  specified at BOTH ends (TideEast_U default 0 walls the outflow — must be set).
- **Outlet:** self-adjusting stage sink (TideEast_U=0 as before — it cannot over-extract).
- **Banks:** topographic + always-wet shelf (free waterline, no wall).
- **Sediment:** FLUX_SCALAR_BC hard-zeros sediment flux on all 4 boundaries → the two buffers +
  the sink-freeze absorb it.

## 5. Diagnostics (total AND anomaly — the new requirement)
- **Total flow:** `u, v, |U|, η` fields; bird's-eye (xOy) + yOz cross-sections.
- **Anomaly flow:** define the mean as the along-channel (s) average of the cross-channel profile,
  `ū(n) = <u_s>_s`, then `u'(s,n) = u_s - ū(n)`, `v'` similarly → the MEANDER-INDUCED perturbation.
  Bird's-eye of `u', v'` and the near-bank `u'_b` (the bank-erosion driver).
- **Bank:** track the bank toe / waterline position vs time → migration rate + direction.
- **Movies:** (a) planform total flow + bank, (b) planform anomaly `u',v'`, (c) yOz sections.

## 6. Numerics / stability (hyperdiffusion allowed)
- DISPERSION=F (NSWE), MUSCL-TVD/HLL, CFL≈0.5.
- Stabilisers if the from-rest transient or the mobile bank chatters: background viscosity
  `nu_bkg`, Smagorinsky `C_smg`, or a small hyperdiffusion; the user explicitly allows these.
- Morph_factor (MF=5 start, MF-convergence-checked) for morphological acceleration; Morph_interval
  regime-aware (bedload-dominated → ~1 T_c).

## 7. Deferred (not this run)
- Rossby-vs-gravity dominance → linearized dispersion (swe_stability.py, R = β_eff b²/(F²U)),
  done separately; the nonlinear run here provides the total+anomaly fields to compare against.

## 8. Open questions for the design review
1. From-rest spin-up: stable with a ramped inlet, or does it need the analytic IC after all?
2. Rigid-deep-bed + erodible-bank `Zs` split vs always-erodible — which is correct AND easier?
3. Is bank migration well-posed in FUNWAVE (topographic bank + shelf), or does the protected
   deep bed / erodible bank interface create a new artefact (like the buffer pile)?
4. Correct erosion/deposition at the bank toe: is MinDepthPickup low enough, is deposition on the
   inner bank captured, does avalanching over-smooth the bank?
5. Anomaly definition: along-channel-s average the right "mean" to subtract?
