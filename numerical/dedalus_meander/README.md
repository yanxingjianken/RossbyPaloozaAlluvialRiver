# Dedalus 2-D channel model — the deck-p.8 numerical model, rung 1

The group's vorticity-meander equations ([`../vorticity_meander/THEORY.md`](../vorticity_meander/THEORY.md))
solved as a genuine 2-D initial-value / eigenvalue problem in **Dedalus v3.0.5**, and the
user-requested experiment: **erodible-bank initial planforms = sinusoids of different
wavelengths, evolving freely**.

## Fidelity statement

Per the design requirement ("faithful to equations"), the discretized system is exactly the
THEORY.md system and nothing else: **linear** dynamics, the **two friction closures**
(`rayleigh` deck-literal / `momentum` Ikeda-consistent) as the only dissipation, **no
artificial hyperviscosity** (deliberately not implemented; Orr filamentation of the vorticity
continuum is handled by the fit-window rule `t ≤ 0.9·Ny/(4Dk*)` and resolution), the deck-p.7
bank law taken literally (relaxation toward the **centerline** value), analytic RNG-free
initial conditions (bank sinusoid + harmonic interior extension, ζ′(0)=0).

## Formulation (all d3 idioms verified by smoke test 2026-07-13)

RealFourier(x) × Chebyshev(y∈[−1,1]); bank fields ψb±(x,t) live on `(xbasis,)` as
**prognostic variables** — their equations carry `dt()` and the interior interpolation
`psi(y=0)` on the LHS; two tau fields lifted to `ybasis.derivative_basis(2)`. Everything is
linear and on the LHS ⇒ the stepper (RK222, fixed dt=0.02) runs fully implicit — no CFL, LU
factorizations cached. The EVP (1-D Chebyshev, complex128, `dt→−iω`, `dx→ik*`) **shares the
identical equation strings** via namespace rebinding. kx=0 is closed and regular (BC rows pin
the Poisson nullspace; the constant-offset gauge mode is invariant and never excited by
zero-mean seeds — asserted).

## Verification ladder (all hard-asserted, all green 2026-07-13)

| rung | result |
|---|---|
| self-test (`python channel_lib.py`, <2 min) | d3 EVP vs FD-GEP Richardson(201/401): **\|Δω\| ≈ 1e−10** (tol 5e−4); IVP fit vs EVP 0.01%; rigid banks neutral; varicose decays at exactly −E with c=0; purity channels (kx=0, varicose, off-mode) at 1e−15–1e−20 |
| `01_evp_sweep` | 6 deck parameter sets × 2 closures × 7 assert-k each: worst \|Δσ\| = **1.6e−8** vs Richardson GEP (tol 5e−4). The Dedalus model and `vorticity_lib`'s N-point GEP are the same continuum spectrum. |
| `02_single_k_ivp` | 8 runs (2 closures × k*∈{0.3,0.44,0.9,1.3}): σ, c within 1e−5–1e−4 absolute of the EVP, R²=1.000000; rigid control decays (σ=−0.051); dt-halving drift 0.003% |
| `03_multi_k_selection` | **the headline experiment**: ONE channel (Lx=20π) seeded with 15 bank sinusoids k*=0.1…1.5 → per-mode demodulation reproduces the whole dispersion relation (30/30 modes within tolerance): growth band k*²<2D, upstream c*<0 in-band, out-band decay |
| `04_momentum_flux` | ⟨u′v′⟩(y) profile matches the EVP eigenfunction to 1.2e−4; centre ratio −⟨v′ζ′⟩/⟨ζ′²⟩ = **(γ+σ)/2D exactly** (0.00% err, rayleigh; also the γ=0 twin — growth alone sustains the bank-ward momentum flux, sharpening the deck-p.6 statement which is the σ→0 forced limit). Momentum closure transports ~19% more than the rayleigh-balance value (report-only; its centre balance differs). |
| `05_anim_planform` | `planform_upstream.mp4`: ψ′ + banks at k*=0.15, measured crest tracker marching upstream against the flow, honest e^{σt} gain counter |

Physics delivered for "不同波长三角函数波" (fig05): long sinusoids inside k*²<2D grow while
propagating **upstream** (c* up to ≈ −0.6…−0.9 at k*=0.1); short ones decay; the winner is
k*_pk ≈ 0.5 (rayleigh) — and under the momentum closure the decaying short waves' phase speed
even crosses to slightly downstream, a closure-discriminating detail.

## Quickstart & file map

```bash
cd /net/flood/data2/users/x_yan/rossby_palooza/numerical/dedalus_meander
micromamba run -n dedalus env OMP_NUM_THREADS=1 python channel_lib.py         # self-test
micromamba run -n dedalus env OMP_NUM_THREADS=1 python 01_evp_sweep.py        # fig01-02 (~5 min)
micromamba run -n dedalus env OMP_NUM_THREADS=1 python 02_single_k_ivp.py     # fig03
micromamba run -n dedalus env OMP_NUM_THREADS=1 python 03_multi_k_selection.py# fig04-05 (~8 min)
micromamba run -n dedalus env OMP_NUM_THREADS=1 python 04_momentum_flux.py    # fig06-07
micromamba run -n dedalus env OMP_NUM_THREADS=1 python 05_anim_planform.py    # planform_upstream.mp4
micromamba run -n dedalus env OMP_NUM_THREADS=1 python 06_anim_per_wavelength.py  # per-wavelength mp4s + fig08 (~6 min)
```

| File | Output |
|---|---|
| `channel_lib.py` | theory bridge (imports `../vorticity_meander/vorticity_lib.py` — single source of truth), shared IVP/EVP equation strings, builders, demodulation + fits, self-test |
| `01_evp_sweep.py` | `fig01/02` σ(k*), c(k*): d3 dots on GEP lines, 2×2 dotted |
| `02_single_k_ivp.py` | `fig03` single-k growth curves vs EVP slopes |
| `03_multi_k_selection.py` | `fig04` planform waterfall (selection + upstream march) · `fig05` measured dispersion |
| `04_momentum_flux.py` | `fig06` ⟨u′v′⟩(y) IVP vs eigenmode · `fig07` centre flux ratio vs (γ+σ)/2D |
| `05_anim_planform.py` | `planform_upstream.mp4` (+preview): the k*=0.15 hero shot on its own 1-wavelength reach |
| `06_anim_per_wavelength.py` | **one 2-D movie per seeded wavelength** — each of the 15 fig04 components (k*=0.1…1.5) run alone on the same Lx=20π reach → `figures/per_wavelength/planform_k*.mp4` (+previews), plus `fig08` final-frame contact sheet (the wavelength ladder). Each title carries λ/2b, σ* (grows/decays), c* (upstream/~stationary/downstream) + honest e^{σt} gain. Flags: `--kstars 0.3,0.5,...`, `--friction momentum`, `--frames N`. |
| `outputs/` | (gitignored; regenerable run data) |

**Planform rendering (`channel_lib.warp_fill`)**: the movies draw ψ′ *inside* the meandering
channel — the field is solved on the fixed domain y∈[−1,1] (linear theory imposes the bank BC
at the undeformed wall), but for display it is warped onto the mesh
Y(x,y)=y+(1+y)/2·d_top(x)+(1−y)/2·d_bot(x) whose edges are the two bank lines, so the colour
fills the wavy channel exactly instead of a fixed rectangle overlaid by displaced bank curves.
This is a display-only cartoon (O(amplitude) warp); the physics is unchanged.

Templates & provenance: IVP scaffold after the user's `literature_review/blocking/1983_eddy_straining_mechanism/scripts/shutts_1983.py`;
EVP idiom after Dedalus `examples/evp_1d_rayleigh_benard`; boundary-field pattern after
`examples/lbvp_2d_poisson` with the field promoted to a prognostic variable (smoke-tested).
Dedalus v3.0.5 in micromamba env `dedalus`. Docs: https://dedalus-project.readthedocs.io

## Shared helper block

`set_style`/`save_fig`/`fig_to_rgb`/`write_mp4` byte-identical across packages (fenced
`=== shared helper block v1 ===`); check with the diff one-liner in
[`../vorticity_meander/README.md`](../vorticity_meander/README.md).

## Next rungs (deferred, deck p.8 goals)

Nonlinear J(ψ′,ζ′) toggle; wavy-bank pressure drag via resolved boundary geometry (the
sharpened ×2.3–3.2 growth-peak suspect); erodible **bottom** with sediment-transport feedback
(Exner), restoring Ikeda's A-physics as resolved dynamics.
