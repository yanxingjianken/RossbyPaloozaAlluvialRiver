# deliverable1_noboru_model — the 3-level meander model, integrated in time

A short, self-contained Dedalus package that takes the model printed in
[`literature/river.pdf`](../../literature/river.pdf) ("Meanders of alluvial rivers as forced Rossby
waves", 21 pp., 2026-07-20) and **runs it forward in the lab frame**, so the meander is seen to grow,
to march upstream, and to build its Reynolds stress in real time.

**river.pdf contains no time integration.** All 21 pages are normal-mode or steady-state statements,
so the periodic domain and every numerical choice here are this package's contribution, marked
`[NOT IN DECK]` at each use site. The **initial condition is not** a free choice, though — see below. Everything
that *is* the deck's is page-cited, symbol by symbol, in
[`docs/lit_review.md`](docs/lit_review.md), which is the specification this code was written from.

## Quickstart

```bash
bash run_all.sh        # ~4 minutes end to end
```

or step by step (`OMP_NUM_THREADS=1` is required — Dedalus warns loudly otherwise):

```bash
env OMP_NUM_THREADS=1 micromamba run -n dedalus python noboru_model.py
cd postprocessing
env OMP_NUM_THREADS=1 micromamba run -n dedalus python 03_verify.py       # gate: run first
env OMP_NUM_THREADS=1 micromamba run -n dedalus python 01_movie.py        # the two mp4s
env OMP_NUM_THREADS=1 micromamba run -n dedalus python 02_dispersion.py   # the p.20 figure
```

## The physics in one paragraph

A slow alluvial river (`0.1 < F_r < 0.3`, p.6) is depth-averaged 2-D vorticity dynamics. The observed
parabolic jet `ū(y) = U₀ + (Δ/b²)(b²−y²)` (p.7, from Bahmanpouri et al. 2022) carries a **constant
cross-channel vorticity gradient `2Δ/b²`** — the channel's planetary-β analogue, and the reason the
deck calls these Rossby waves. The deck closes the problem on three levels `ψ₁, ψ₂, ψ₃` at
`y = +b, 0, −b` (p.9), advects the centre vorticity at the centreline speed and damps it with bottom
friction (p.10), and makes the **erodible bank** the engine: the bank relaxes toward the interior
streamfunction, `∂ψ'₁/∂t = (εC_fU₀/b)(ψ'₂−ψ'₁)` (p.19). The result is a meander that grows in a
resonant band and — unlike Ikeda et al. (1981), whose bends always migrate downstream — travels
**upstream**.

## The two questions this package was built to answer

### Is `ψ̂₁ = ψ̂₃` an initial condition, or does it hold throughout?

**Throughout. It is an exact invariant, not an assumption** — and the package *measures* this rather
than asserting it. `ū(y)` is even in `y` and the erosion law is identical at both banks, so the
linear operator commutes with `y → −y`. The state splits into two decoupled subspaces:

- **sinuous**, `(ψ₁+ψ₃)/2` — the meander. It alone enters `ζ'₂`, so it carries all of the p.19
  dynamics.
- **varicose**, `(ψ₁−ψ₃)/2` — a width pulsation. It obeys `∂ₜa = −Ea`, **never appears in `ζ'₂` at
  all**, and decays at exactly the erosion rate for every `k*, D, γ`. Measured: `−0.2500021` against
  `−E = −0.2500000`.

So starting sinuous stays sinuous exactly; starting asymmetric, the asymmetry dies while the meander
grows. `03_verify.py` seeds a deliberately lopsided channel and watches `|anti/sym|` fall from
`1.0` to `9.8e-10`.

**And it does not fight the mean flow.** The mean lives entirely in `ψ̄(y)`, which nothing in the
model evolves — friction and erosion act on the primed part only. `ψ̄(±b) = ∓0.833` at `D = 0.5`, so
`ψ̄(−b) − ψ̄(+b) = 1.667 = ∫ū dy`, the discharge. `ψ̂₁ = ψ̂₃` says `ψ'(+b) − ψ'(−b) = 0`: **the meander
adds no net discharge.** The river snakes without changing how much water it carries. The varicose
mode *is* the discharge fluctuation — and it dies.

### Do I need `b` and `H` as configuration parameters?

**No — they are already inside `k*` and `γ`.** Nothing in the code takes `b`, `H`, `U₀`, `Δ` or `C_f`
individually, because the deck's own sidebar groups them:

| parameter | where the dimensional quantities go |
|---|---|
| `k* = kb` | the only place `b` appears as a length |
| `D = Δ/(U₀+Δ)` | the only place `U₀` and `Δ` appear |
| `γ = C_f b/H` | the only place `H` and `C_f` appear |

But `{k*, D, γ}` is **one short**. The p.19 erosion law carries the bank erodibility `ε`, and
river.pdf never defines it and never gives it a value. The complete set is **`{k*, D, γ, εC_f}`** —
four numbers, and `CONFIG` in [`noboru_model.py`](noboru_model.py) has exactly those four.

`εC_f = 0.5` is an **assumption, not a citation and not a fit**. river.pdf prints only the product
`εC_fU₀/b` (p.19) and never gives `ε` a value. Both `σ` and `c(k*→0) = −ED/γ` scale with it, so it
sets the *rate* of everything while changing none of the structure: the growth band, the sign of `c`,
the `ψ̂₂/ψ̂₁` ratios and the `k*` of the peak are all unaffected. Change it in `CONFIG` and every
figure rescales accordingly.

## The two runs

Identical but for `k*`. `n_wave` differs only so both reaches are ≈ 25×2b like the slides.

| | `k* = 0.3` | `k* = 1.5` |
|---|---|---|
| `k*²` vs `2D = 1.0` | 0.09 < 1 → resonant | 2.25 > 1 → non-resonant |
| forced `\|ψ̂₂/ψ̂₁\|` (p.14 box) | **1.835** — interior amplified | **0.615** — suppressed |
| growth `σ` | **+0.0814** (e-folds in 12.3) | **−0.0962** |
| phase speed `c` | **−0.243 — UPSTREAM** | −0.0001 — stationary |
| what the movie shows | meander grows ×24.9, marches upstream | decays to 1/47, ψ₂ visibly *inside* the banks |

**`k* = 1.5` cannot grow, for any parameters.** The largest unstable wavenumber is bounded by
`√(2D) < √2 ≈ 1.414`, and a scan of 2520 parameter sets
(`D ∈ (0.02, 0.995) × γ ∈ [0.001, 2] × εC_f ∈ [0.05, 20]`) found a maximum of **1.4080**, with no set
reaching 1.5. That is why the second movie is a decay: it is the deck's p.12 contrast, not a failed
run.

## The initial condition is the deck's own forced steady state

There is exactly **one** physical setup here, so `CONFIG` has no `ic` option — only an amplitude,
and the system is linear so even that just sets the units.

river.pdf poses a *forced* problem before it poses an unstable one. pp.12–16 and 18 are titled
"Forced steady state" and "Forced-dissipative steady state"; p.11 states the closure as
`ψ̂₂ = f(ψ̂₁)` — the banks are **given** and the interior is slaved to them. The experiment on
pp.17–18 says the same thing in foam and dye: a **rigid carved wavy channel** with water running
through it. The banks *are* the imposed meander; the interior flow is what answers. Only on p.19
does the deck release the banks and let them erode.

So the initial-value problem the deck actually sets up is: **take the forced-dissipative steady
state on a given wavy channel, then switch the banks from rigid to erodible and watch.**

```
ψ'₁ = ψ'₃ = A cos(k*x)                       the carved meander      (p.9 sinuous, pp.12-18)
ψ'₂ = Re[ f(k*, D, γ) · A exp(i k*x) ]       the interior response   (p.11)
```

At `t = 0` the banks are wavy and `ψ₂` sits at the forced ratio — 1.63 at `k*=0.3`, **0.615** at
`k*=1.5`, where it is visibly *inside* the banks. That state is **not** an eigenmode of the p.19
erosion problem, so the dynamics still has to find the growing mode rather than be handed it: the
measured rate converges to `det M = 0` to five decimals.

The verification script needs two deliberately *unphysical* states — a pure varicose perturbation
and a lopsided channel — to measure a symmetry the physical IC makes identically zero. Those are
properties of the tests, so they live in `03_verify.py` behind a `_test_ic` hook, and the driver
stays single-valued.

### What the four rows show

No display gain anywhere. The curves are `ψⱼ = ψ̄(yⱼ) + ψ'ⱼ` in their own units, and the vertical
separation between them **is** `ψ̄(yⱼ)` computed from the printed `ū` — `∓0.833` at `D=0.5`, not an
invented offset. Since `ū = −∂ψ̄/∂y > 0` means `ψ` *decreases* with `y`, the axis is inverted, which
puts `ψ₁` on top for the same reason the physical channel does. `A0` is the sinuosity the channel is
carved with at `t=0` — a physical input, not a display knob.

| row | shows |
|---|---|
| 1 | `ψ₁, ψ₂, ψ₃`, fixed limits, real growth |
| 2 | the same curves with the **banks coloured by the local momentum flux** `−v'₂ζ'₂` — p.16's quantity with the `x`-average dropped, so sign *and* magnitude are a field along the channel instead of one number per bank |
| 3 | `\|ψ̂₂\|(t)` against the analytic growth rate |
| 4 | the bank mode's **dispersion relation**, `σ(k*)` and `c(k*)`, with this run's `k*` marked. An earlier version also marked `√(2D)` — wrong on a growth-rate plot, since `k*²<2D` is p.14's *forced-amplification* criterion, not a stability edge, and the deck states no stability edge at all. The real `σ=0` crossing is marked instead (0.9865 vs `√(2D)`=1.0000 at `D=0.5`). |

Row 2 shows something a single averaged number hides: the flux alternates sign *along* the meander
(it is a product of two `k`-waves, so it oscillates at `2k` about its mean). At `k*=0.3` the mean is
positive on the upper bank in **100%** of frames — the deck's arrow. At `k*=1.5` it is positive in
only 55%, and locally positive over 50% of `(x,t)`: outside the resonant band the momentum flux is
not a cleanly one-signed thing at all.

**A note on reading the panels.** `ψ` is a streamfunction value, not a channel position, and the
`pp.12–19` panels are three bare curves on a `ψ` axis — no channel is drawn, and none should be.
`|ψ̂₂| > |ψ̂₁|` whenever `k*² < 2D` is the deck's *headline* (p.14), so `ψ₂` swinging wider than
`ψ₁` is the result, not a violation. (An earlier version of this package filled a blue "channel"
between `ψ₃` and `ψ₁`, importing the p.9–11 schematic — which is drawn in `y` — into a `ψ`-axis
plot. That category error manufactured a reading where the centreline "escapes the river". The fill
is gone.)

## Can the meander ever travel downstream?

`c > 0` **does** exist in `det M = 0` — the *advective* root, the vorticity anomaly simply swept
along by the jet, `c ≈ +0.9`, and past `k* ≈ 1.2` it is the **less damped** of the two roots, so it
is what a generic initial state would converge to.

But the deck's own physical setup essentially cannot excite it. At `k*=2.0, D=0.3, γ=0.03`:

| state | `\|ψ̂₂/ψ̂₁\|` |
|---|---|
| the forced steady state (the initial condition) | **0.370** |
| the **bank** eigenvector (`c ≤ 0`) | **0.370** |
| the **advective** eigenvector (`c = +0.899`) | 5.213 |

The carved channel *is*, structurally, the bank eigenvector — identical to three decimals. So it
projects onto the branch with `c ≤ 0` and almost not at all onto the `c > 0` one. **Within this
model the meander never travels downstream**, which is the p.19/p.21 claim with a mechanism attached:
not merely "`c<0` on the growing branch", but "the only branch a real channel excites is the one with
`c ≤ 0`".

**Raising `U₀` does flip the sign — but only to `+8e-4`.** Raising `U₀` at fixed `Δ` lowers
`D = Δ/(U₀+Δ)`, and at `k*=1.5, γ=0.1` the bank-branch `c` goes from `−0.0001` (D=0.5) to `+0.0008`
(D=0.3). A scan of 2500 parameter sets finds the bank branch never exceeds `c ≈ +0.001` anywhere in
the deck's calibrated regime — 300× smaller than the `k*=0.3` upstream speed. A movie of it would
show a pattern sitting still. (It does reach `c ≈ +1` at `εC_f = 10`, i.e. `E = 9.8` — twenty times
the assumed erodibility, i.e. far outside the regime the deck describes.)

So there is no third movie: neither route produces a visibly downstream-travelling meander without
leaving the deck's own problem.

## What is verified

`03_verify.py` compares against the reconstructed `det M = 0` or an exact symmetry — never against a
hand-written expected array.

| check | result |
|---|---|
| IVP `σ, c` vs `det M = 0`, both `k*` | agree to **7 decimals** (`+0.0813601` vs `+0.0813602`) |
| the physical IC finds the mode on its own | `k*=1.5` fit `−0.09605` vs analytic `−0.09616` |
| **the IC really is the forced steady state** | at `t=0` the p.16 balance `−⟨v'₂ζ'₂⟩ = (γ/2D)⟨ζ'₂²⟩` holds to **1.4e-16**, and has left it by `t=40` |
| varicose decay = `−E`, both `k*` | `−0.2500021` vs `−0.2500000`; `ζ'₂` untouched to 1e-16 |
| mixed IC: subspaces separate | sym grows at `σ`, anti decays at `−E`, `\|anti/sym\|` → 9.8e-10 |
| `dt` halving | `σ` stable to 4e-7 (all terms linear + implicit) |
| forced `\|ψ̂₂/ψ̂₁\|` vs the p.14 box | exact to 1e-12, including the p.13 `D = 0.1` panel |
| p.16 momentum flux sign | `−⟨v'₂ζ'₂⟩ > 0` in every case |

## A note on the deck's own p.20 figure

This package deliberately does **not** score itself against river.pdf p.20. An earlier version
scraped values off a 300-dpi render of that figure and fitted to them; that was backwards. We have
the deck's equations solved analytically (`det M = 0`) *and* integrated numerically (Dedalus), and
the two agree to six decimals — re-measuring someone else's raster plot cannot improve on that, and
it drags in two weaknesses: the scrape is lossy (a curve maximum is *flat*, so its position is the
worst-resolved quantity of all), and `εC_f` was itself calibrated from those same scraped
intercepts, so any comparison built on them was partly circular.

`εC_f` is therefore stated as an **assumption** (0.5) rather than fitted. `σ` scales with it;
`c(k*→0) = −ED/γ` scales with it too. Nothing in this package depends on a number read off a figure.

For the record, and as an eyeball observation rather than a result of this package: the growth-rate
peaks drawn on p.20 look several times higher than these equations produce at any `εC_f` that also
matches its phase-speed panel. That forensic question is not this deliverable's job — it already has
a home in `../vorticity_meander/`, which exists to reconcile the deck's algebra with its figures.

## Two inconsistencies inside river.pdf

Recorded, not resolved.

1. **p.2 prints `λ ≈ 10–14 × width`; p.21 prints `λ ≈ 7–14 × width`.** This matters, because
   `λ/2b = π/k*`: `7–14` means `k* ∈ [0.224, 0.449]` while `10–14` means `k* ∈ [0.224, 0.314]` —
   different verdicts on whether the model's peaks fall at "observed scales". (Separately: the
   `10–14 × width` figure is not Schumm 1967's result; his paper gives `λ = 1890 Qm^0.34/M^0.74` and
   cautions against the width correlation.)
2. **`ψ̄` ordering.** p.9 prints `ū = −∂ψ̄/∂y`, and `ū > 0` forces `ψ̄(+b) < ψ̄(0) < ψ̄(−b)` — yet p.14
   labels `ψ₁` on the **top** curve. The vertical offsets in `01_movie.py` are therefore a **declared
   display constant**, not `ψ̄`.

## File map

| file | what it does |
|---|---|
| [`docs/lit_review.md`](docs/lit_review.md) | **the spec the code is written from** — page map, equations verbatim, notation table, NOT-DEFINED-IN-SOURCE list, bibliography, numerical targets |
| [`noboru_model.py`](noboru_model.py) | the driver. `CONFIG` on top, Dedalus IVP, writes `outputs/*.npz` and nothing else |
| [`postprocessing/pp_lib.py`](postprocessing/pp_lib.py) | shared helpers, `det M = 0`, the p.16 flux diagnostic |
| [`postprocessing/01_movie.py`](postprocessing/01_movie.py) | `figures/meander_k0.30.mp4`, `figures/meander_k1.50.mp4` |
| [`postprocessing/02_dispersion.py`](postprocessing/02_dispersion.py) | `figures/bend_instability.png` — analytic `det M = 0` curves + measured Dedalus points, laid out like p.20 |
| [`postprocessing/03_verify.py`](postprocessing/03_verify.py) | the gate. `figures/verify.png` |

## Four traps this package walked into, so you don't have to

0. **The worst one: posing the problem backwards.** The first version kicked the *centreline* and
   left the banks straight, on the reasoning that "the instability should build the banks itself".
   That inverts the deck. river.pdf's problem is *forced* — the banks are the given meander, the
   interior responds (`ψ̂₂ = f(ψ̂₁)`, p.11) — and the flume on pp.17–18 is a rigid carved channel,
   which is the same statement in foam. The symptom was visible in frame 1: flat banks with `ψ₂`
   swinging far outside them. Nothing in the *equations* was wrong, and every verification check
   still passed, because the tests compared the IVP against the dispersion relation — and a wrong
   initial state relaxes onto the right eigenmode eventually, so the asymptotics were fine while the
   physical setup was nonsense. **Numerical agreement with your own algebra does not tell you that
   you posed the right problem.**
1. **Dedalus `RealFourier` stores mode `n` at coefficient indices `[2n], [2n+1]`.** Reading `[2],[3]`
   for an `n_wave=2` run returns a *round-off* mode that still fits a perfectly clean growth rate —
   the wrong one. Everything goes through `mode_amplitude`.
2. **A long integration at a decaying `k*` gets hijacked.** With `n_wave=12` the box admits
   wavenumbers inside the growth band, and the ~1e-16 round-off there grows until it dominates the
   decaying mode being measured — after which the fit reports *the leader's* growth rate. The fix is
   `n_wave=1`, which makes `k*` the gravest mode in the box so nothing can outgrow it.
3. **An IVP converges to the least-damped root, which is not always the one you want.** Past
   `k* ≈ 1.2` the advective root decays more slowly than the bank root, so `02_dispersion.py`
   identifies which branch each measurement landed on and refuses to plot the other one on the bank
   curves.

## Environment

micromamba `dedalus` on dolma: dedalus 3.0.5, numpy 2.4.3, matplotlib 3.10.9, scipy, h5py, imageio,
PIL, `ffmpeg` on PATH.

## Provenance

**No number in this package is read off a figure.** Every value is either computed from the
equations printed in river.pdf or cited to the page that prints it. The only quantity river.pdf does
not supply is `ε`, and it is declared as an assumption in `CONFIG` rather than fitted to anything.

Nothing here draws a random number; check with

```bash
grep -rn "random" --include="*.py" .        # expected: no output
```
The `=== shared helper block v1 ===` fence in `pp_lib.py` is byte-identical to the other
rossby_palooza packages:

```bash
diff <(sed -n '/=== shared helper block v1/,/=== end shared helper block ===/p' postprocessing/pp_lib.py) \
     <(sed -n '/=== shared helper block v1/,/=== end shared helper block ===/p' ../vorticity_meander/vorticity_lib.py)
```
