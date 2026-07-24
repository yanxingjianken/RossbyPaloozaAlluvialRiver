# Timescale handling and the A=0 equation comparison — an annotated model review

A review of how the fast-flow / slow-morphology timescale gap is handled, and of exactly how the
FUNWAVE model at A=0 differs from the corrected shallow-water (SW) reduction and the Thetis model.
Every quantitative claim is tagged with its source: **[I81]** Ikeda–Parker–Sawai 1981;
**[SW]** `thetis/docs/River_Meandering_SW_corrected.tex`; **[MD]** `thetis/docs/model.md`;
**[D2]** `funwave_2d_sw/derivations/DESIGN_v2_unified_no_morph.md`; **[M]** measured this session.

---

## 0. Corpus (annotated)

| tag | source | what it contributes | key numbers |
|---|---|---|---|
| **I81** | Ikeda, Parker & Sawai 1981, *Bend theory of river meanders, Part 1* | the depth-averaged bend theory; the secondary-flow coefficient **A** enters only the transverse bed tilt, eq (6) $\eta'/H=-A\,C'\tilde n$; alluvial A=2.89 (Suga 1963, 45 bends), incised **A=0**; bank law (11)-(13) | A=2.89 / **A=0** [MD §6] |
| **SW** | Zeng, *Shallow water analysis*, corrected | linearizes the depth-averaged SW in $(s,n)$ to $O(\varepsilon)$; two distinguished limits — **Ikeda/bend** ($\alpha\!\sim\!\varepsilon^{1/2}$, eqs 20-22) and **QGPV/Rossby** ($\alpha=1$, $Fr^2\!\sim\!\varepsilon$, eq 26) — plus a kinematic bank law (31-35). Load-bearing correction: the bed was restored, $h=\eta-z_b$ | without the bed, $U_0=0$; the channel drains in **~202 m** [MD §2.3] |
| **MD** | Thetis `model.md` | an **implicit** (DG-DG P1, **Crank-Nicolson**) Firedrake SW model on a curved mesh; **A=0** (incised); **no Exner** — the bed is frozen and only the banks move by the kinematic Ikeda law; morphological acceleration declared **mandatory** | transit 2099.5 s; $\sqrt{gH}=3.132$; MF mandatory [MD §8] |
| **D2** | our FUNWAVE model | **explicit** MUSCL-TVD depth-averaged NSWE + **full Exner** sediment (van Rijn pickup, Cao deposition, MPM bedload); A=2.89; running now at **MF=1** | $\Delta t=0.199$ s; $T_{\rm adjust}=2292$ s [D2 §3.1] |

The four are complementary: **I81** is the theory, **SW** is its linear dispersion (and the
gravity-vs-Rossby split), **MD** is an efficient planform (bank-only) solver, **D2** is the
nonlinear bed-and-bank morphodynamics solver.

---

## 1. Q1 — the fast/slow timescale gap

### 1.1 The gap, quantified

Two timescales that differ by ~9 orders of magnitude must be bridged in one run:

| process | timescale | source |
|---|---|---|
| hydrodynamic transit $L/\bar U$ | **~35 min** (2099.5 s) | [MD §8] |
| flow adjustment after a bed change $T_{\rm adjust}=H/(C_dU)$ | **2292 s** | [D2 §3.1] |
| bar (transverse redistribution) saturation | **~1.5×10⁵ s** | [M] |
| bank migration ($E/U\sim10^{-8}$) | **decades** | [MD §8] |

The naïve loop — converge the flow, take a bed step, *re-converge the flow because the bed moved* —
pays a full flow re-adjustment per bed step, and there are astronomically many bed steps. Three
mechanisms attack this; they are complementary, not alternatives.

### 1.2 (a) Morphological acceleration factor (MF)

The Exner update is scaled: $h = h_{\rm ini}+Z_b\cdot\mathrm{MF}$ [D2 §1.3], so the bed evolves MF
times faster per unit hydrodynamic time. **This is the standard tool, and the Thetis spec itself
declares it "mandatory"** [MD §8] — bank migration over decades is otherwise unreachable.

**Validity limit.** MF is legitimate only while the flow stays quasi-steady with respect to the
*accelerated* bed. Over one adjustment time the accelerated bed moves by $\mathrm{MF}\cdot(\dot z_b)\cdot T_{\rm adjust}$;
requiring this $\ll H$ and using $\dot z_b\sim H/T_{\rm morph}$ gives

$$\boxed{\ \mathrm{MF}\ \ll\ \frac{T_{\rm morph}}{T_{\rm adjust}}\ }$$

The relevant $T_{\rm morph}$ is the **fastest** morphological process, i.e. the bar, not the bank.
With $T_{\rm morph}\approx T_{\rm bar}=1.5\times10^5$ s [M] and $T_{\rm adjust}=2292$ s [D2],

$$\frac{T_{\rm bar}}{T_{\rm adjust}}\approx 65\ \Rightarrow\ \mathrm{MF}\lesssim 10\ \text{(safety factor ~6)}.$$

So **MF=1000 (the old derivation's figure) is invalid** — it exceeds the ratio 15-fold — while
**MF≈5–10 is safe here**. The v1 experience is consistent: MF=50 blew up, MF=5 ran [prior session].

**A subtlety that distinguishes our model from Thetis.** Our model couples bed and bank as one
Exner surface, so the *fast* process (the bar) caps MF at ~10. Thetis freezes the bed [MD §9(2)];
its only morphology is bank migration (decades), so its $T_{\rm morph}/T_{\rm adjust}\sim10^6$ and
it can carry a **far larger MF**. Coupling in the bed is what forbids a big MF — the price of being
able to form bars.

### 1.3 (b) Hot-start makes re-adjustment cheap

Neither model cold-starts after a bed step. Thetis copies $(uv,\eta)$ across by **direct DOF copy**
— exact because the mesh topology is unchanged [MD §8]. Our `run_v2.py` hot-starts each chunk from
the previous flow and carries the evolved bed via `depth_cur.txt`. The consequence is the key point
the question raises: **the re-adjustment is a PERTURBATION relaxation, not a spin-up.** The flow has
only to relax to a small $\delta z_b$, which decays on $T_{\rm adjust}$ as a linear perturbation —
not the multi-transit cold spin-up. "Re-spin-up on the fast timescale" is therefore cheap, provided
the state is carried across.

> This is exactly the mechanism the reset bug destroyed: re-reading the *original* bathymetry each
> chunk discarded $\delta z_b$, so every chunk paid a full cold re-adjustment **and** threw away its
> morphology. The `dz_max_cum` gate now guards it (must grow chunk-to-chunk) [M, this session].

### 1.4 (c) The decisive solver-level point: explicit vs implicit

This is where FUNWAVE is fundamentally penalized and where the honest answer lives.

**FUNWAVE is explicit.** Its timestep is bounded by the gravity-wave CFL:
$$\Delta t=\frac{\mathrm{CFL}\,\Delta x}{U+\sqrt{gH}}=\frac{0.5\times2.5}{0.85+5.42}=0.199\ \text{s}\quad[\text{D2 §3.1, M}]$$
The step is set by $\sqrt{gH}=5.42$ m/s — the **gravity-wave** speed — not by the flow $U=0.85$.
At $Fr=0.157$ gravity waves are $\sqrt{gH}/U=6.4\times$ faster than the flow, so **we take 6.4×
more steps than the advective/morphological physics needs**. That factor is pure overhead of the
explicit scheme, and it is the direct cost of the (correct) requirement that $Fr$ stay small.

**Thetis is Crank-Nicolson implicit** [MD §2.1]. Its own note is explicit: "explicit CFL on `dn`
0.308 s (Crank-Nicolson is implicit; used only to size `dt`)" [MD §8]. An implicit SW solver is not
stability-bounded by $\sqrt{gH}$; its step is set by **accuracy** — resolving $T_{\rm adjust}$, not
the gravity wave. It can therefore take steps $\mathcal O(6\text{–}30)\times$ larger for the same
physics, removing exactly the penalty that makes FUNWAVE slow.

### 1.5 Recommendation (Q1)

1. **Within FUNWAVE, yes — turn MF back on, at a *moderate* value.** MF=1 (running now) is the
   rigorous baseline and the MF-convergence anchor, but it is not the production setting. Use
   **MF≈5–8** for production, validated by comparing the MF=1 and MF≈5 bar fields — that is the
   convergence check the derivation always demanded and never performed [D2 §3.1]. MF≈5–8 sits
   safely under the $T_{\rm bar}/T_{\rm adjust}\approx65$ ceiling.
2. **Always hot-start / carry the bed** (§1.3). This is what makes the re-adjustment a cheap
   perturbation; it is not optional, and getting it wrong silently wastes the whole run.
3. **The real remedy is the solver, not the factor.** FUNWAVE's explicit gravity-wave CFL cannot be
   removed from within FUNWAVE; only MF amortizes it. **Thetis (implicit) removes it at the source.**
   So: FUNWAVE for the bar/Exner physics it uniquely has (§2), Thetis for efficient long-time
   planform evolution.

---

## 2. Q2 — the equations with A=0, term by term

### 2.1 The hydrodynamic core is the *same* equation

Our Cartesian flux-form momentum [D2 §1.1] and the SW $(s,n)$ advective form [SW eqs 3-5] are one
system in two coordinate frames:

| | our FUNWAVE [D2 §1.1] | SW $(s,n)$ [SW 3-5] |
|---|---|---|
| $s$/$x$-momentum | $\partial_t P+\partial_x\!\big[\tfrac{P^2}{H}+\tfrac12 g(\eta^2+2\eta h)\big]+\partial_y\!\big[\tfrac{PQ}{H}\big]=g\eta\,\partial_x h-C_d u\lvert\bm u\rvert$ | $\partial_t u+u\partial_s u+v\partial_n u+\mathbf{Cuv}=-g\partial_s(h+z_b)-\tfrac{C_f\lvert\bm u\rvert}{h}u$ |
| $n$/$y$-momentum | mirror in $Q$ | $\partial_t v+u\partial_s v+v\partial_n v-\mathbf{Cu^2}=-g\partial_n(h+z_b)-\tfrac{C_f\lvert\bm u\rvert}{h}v$ |
| continuity | $\partial_t\eta+\partial_x(Hu)+\partial_y(Hv)=-\partial_t h$ | $\partial_t h+\partial_s(uh)+\partial_n(vh)+\mathbf{Cvh}=0$ |

Two structural differences, both *coordinate*, not *physics*:

- **The curvature metric term** $Cuv,\,-Cu^2,\,Cvh$ (bold) is explicit in $(s,n)$. In our Cartesian
  solver it is **carried implicitly by the geometry** — and, notably, the *Thetis code does the same*:
  it "solves the Cartesian shallow-water equations on this curved mesh… no $\sigma=1+nC$ factor
  appears anywhere" [MD §1]. So at the software level ours and Thetis agree — the $C$-term lives only
  in the analytic reduction, and I81 (1a-c) likewise carries no metric factor because curvature *is*
  the perturbation [MD §2.2].
- **Friction.** We write $C_d u\lvert\bm u\rvert$; SW writes $C_f\lvert\bm u\rvert u/h$. Linearizing
  ours about $(\bar U,0)$ gives exactly the SW anisotropic drag $r_s=2C_f\bar U/\bar H$,
  $r_n=C_f\bar U/\bar H$, $r_\eta=C_f\bar U^2/\bar H^2$ [D2 §1.1 "Relation to the linear $(s,n)$
  model"; SW eqs 6-7]. **The two agree to first order; ours carries all higher orders.**

### 2.2 What A=0 removes — and what is left

I81 eq (6) is the transverse bed tilt $\eta'/H=-A\,C'\tilde n$ [MD §2.2]. Substituting I81 (5)-(6)
into (3b) collapses the friction bracket to a single depth modulation [MD §6]:

$$\frac{h'}{H}=(F^2+A)\,C'\tilde n$$

so **A and $F^2$ occupy the same slot — a drag modulation through the depth**. $F^2$ is the
free-surface superelevation; A is the 3D-helical bed tilt. A 2D prognostic-$\eta$ model (both ours
and Thetis) **computes the $F^2$ half exactly and cannot produce the A half at all** — depth
averaging removes the helical circulation by construction [MD §6]. So:

| | with A=2.89 (alluvial) | **with A=0 (incised)** |
|---|---|---|
| $A+F^2$ | 2.98 | **0.09** [MD §6] |
| bend forcing that is helical | ~97 % | **0 %** |
| what drives the fast filament outward | secondary flow (dominant) | only $F^2=0.09$, against a free vortex ($u\sim1/r$) faster on the **inside** |
| **which bank erodes** | outer (assumed) | **a measured output — inner-bank erosion is a live, valid result** [MD §6] |

A=0 is not an approximation to the alluvial case; it is I81's named **incised** parameter set
[MD §6]. Our model reaches it with one switch (`A_ikeda=0`, dropping the `cd.txt`/`kappa.txt`/
`bedslope.txt` closures). At A=0 our MPM bedload direction is $\alpha=\operatorname{atan2}(v,u)$ with
**no transverse deflection** — i.e. bedload follows the depth-averaged flow, exactly what the linear
SW leading order gives.

### 2.3 The deepest difference — morphology closure

This is where the models genuinely part company, and it is not about A:

| | our FUNWAVE [D2 §1.2-1.3] | SW / Thetis [SW 31-35; MD §7, §9] |
|---|---|---|
| what evolves the boundary | **Exner sediment conservation** $(1-n_p)\partial_t z_b=D-P-\nabla\!\cdot\!\bm q_b$ | **kinematic bank law** $\gamma\,\partial_t y_b=E\,u_b'$, $E=\varepsilon C_f$ |
| bed | fully mobile; **bed and bank are one erodible surface** | **frozen** ($\partial_t z_b=0$) |
| transport physics | van Rijn pickup + Cao deposition + MPM bedload | none — bank moves $\propto$ near-bank velocity excess |
| **bars / bar-bend resonance** | **available** | **structurally impossible** [MD §9(2)] |
| wavelength selection | bar resonance **and** bank instability | bank instability **only** [MD §9(2)] |
| calibration | grain-based ($D_{50},\theta_{cr},\tan\phi$) — not comparable to $E$ without recalibration [D2 §1.3] | $E=\varepsilon C_f$; $E_e\neq E_d$ is an uncalibrated user extension [MD §7.2] |

So even at matched A=0, the two models answer different questions. **Ours can grow a point bar and
select a wavelength through bar-bend resonance; Thetis cannot form a bar at all** and selects a
wavelength only through the bank instability. Conversely, Thetis evolves the *planform* (bank
migration) cheaply and directly, which our Exner surface does only as a slow, MF-limited by-product.

### 2.4 A caveat inherited from the SW source

The SW reduction is only trustworthy in its **corrected** form. The original deleted the bed
("$\nabla\eta=\nabla h$"), which removes the only term balancing bed friction: with $H_0=$const its
eq (9) gives $U_0=0$, and the channel drains in **~202 m — under half a wavelength** [MD §2.3].
Restoring $h=\eta-z_b$ restores I81 exactly [SW L39-90; MD §2.3]. Our model never had this issue —
it carries $h=\eta+ (-z_b)$ with a prognostic $\eta$ from the start — but any comparison against the
SW dispersion must use the corrected equations, not the original.

### 2.5 The gravity-vs-Rossby content

The SW linear reduction is the one place the **Ikeda (bend/gravity)** and **QGPV (Rossby)** branches
are separated, by distinguished limits: Ikeda at $\alpha\sim\varepsilon^{1/2}$ [SW 20-22], QGPV at
$\alpha=1,\,Fr^2\sim\varepsilon$ [SW 26], the latter carrying a PV-gradient $\tilde\beta$ from the
jet-profile curvature [SW 27]. Our **nonlinear** model contains both limits at once and cannot, by
itself, say which dominates — which is why the linear $(s,n)$ companion exists. Both the FUNWAVE and
Thetis packages have **deliberately deferred** the gravity-vs-vortical diagnostic (movies only), and
the prior retraction of "the meander *is* the Rossby wave" stands [MD §3]. This review does not
reopen it.

---

## 3. Q3 — the plan

### 3.1 What each tool is for

| tool | its unique capability | use it for |
|---|---|---|
| **FUNWAVE + Exner** [D2] | full nonlinear bed+bank morphodynamics; **bars, bar-bend resonance, the point bar** | the morphology question: how the bed reshapes, the bar, the erosion/deposition phase |
| **Thetis** [MD] | implicit (no gravity CFL) + kinematic bank law; cheap long-time **planform** evolution | bank migration / planform over long morphological time; the efficient production engine |
| **SW-corrected (analytic)** [SW] | the linear dispersion; the **Ikeda vs QGPV** split | the gravity-vs-Rossby diagnosis, when it is reopened |

**They are not redundant.** FUNWAVE is the only one that can form a bar; Thetis is the only one
cheap enough for decades of planform; SW is the only one that separates the two wave branches.

### 3.2 Open items (prioritized)

| # | item | why it matters | status |
|---|---|---|---|
| **P0** | **$C_d$ dynamic-drag inconsistency** | momentum drag is frozen at the $t=0$ bed while the sediment module recomputes shear from the live depth every step (1.5–2.2× error on the bank face). At A=0, $F^2$ is the *only* bend driver, so a wrong bank-face drag directly corrupts the one signal left | **open**, un-applied |
| **P0** | **$\beta=b/H=22.5$, not 16.7** | I used $H_c=3.0$; realised $H=2.22$ m ($Fr=0.182$). $\beta=22.5$ is *further* above the free-bar threshold (~10–15), so free bars are *more* likely to contaminate the forced signal | **doc states a wrong number** |
| **P1** | **grid convergence with $\nu_{\rm smg}\equiv0$** | $\nu_{\rm smg}$ is set to ZERO in `init.F:437`, so the MUSCL/HLL limiter's numerical diffusion is the *only* lateral damping ⇒ bar wavelength is resolution-dependent, and $\Delta x=2.5$ m gives only ~2.6 cells across the bank face | **no convergence test exists** |
| **P1** | **MF=1 baseline → moderate-MF production** | the MF=1 run (finishing now) is the anchor; production should be MF≈5–8 (§1.5), which needs the MF=1 vs MF=5 bar comparison | **baseline running** |
| **P1** | **run the A=0 FUNWAVE case** | direct, matched comparison with Thetis (both A=0): does the Exner model, given the *same* incised forcing, produce a bar where Thetis structurally cannot, and does it agree on which bank erodes? High diagnostic value | **not started** |

### 3.3 Decision points

1. **Fix $C_d$ before the production (moderate-MF) run.** At A=0 it corrupts the only remaining bend
   driver; the MF=1 baseline can stand as-is for the convergence anchor, but production should not.
   Recommend the log-law field rebuilt from the live depth (the "(b)" fix from the earlier drag
   analysis).
2. **Add an A=0 FUNWAVE case** to the matrix — it is the clean bridge to Thetis and isolates the
   $F^2$-only bend response with our bar physics intact. Low marginal cost (one more case).
3. **Adopt MF≈5–8 for production**, gated by the MF=1↔MF=5 bar-field comparison. Keep the hot-start /
   `dz_max_cum` guard.
4. **Split the workload by tool** (§3.1): FUNWAVE for bars and the morphodynamic phase; Thetis for
   long-time planform; SW for the dispersion diagnosis if/when it is reopened.
5. **Correct $\beta$ and the H used with it** throughout `DESIGN_v2` before it is treated as settled.

---

## 4. One-paragraph answer to the three questions

**Q1:** FUNWAVE has no way around the fast/slow gap other than the morphological factor — its
explicit scheme is bounded by the gravity-wave CFL (6.4× overhead at $Fr=0.16$), and only MF
amortizes that; MF≈5–10 is safe here (the bar caps it at $T_{\rm bar}/T_{\rm adjust}\approx65$),
MF=1000 is not, and hot-starting makes each re-adjustment a cheap perturbation. The *structural*
fix is the implicit solver, which is why Thetis (Crank-Nicolson) is the better long-time engine.
**Q2:** at A=0 the hydrodynamic cores are the same equation in different coordinates and agree to
first order in friction; A=0 removes the helical half of the bend forcing, leaving only $F^2=0.09$,
so which bank erodes becomes a measured output — but the deepest difference is not A: we carry a
full Exner bed (bars, bar-bend resonance), Thetis freezes the bed and moves only the banks (no
bars). **Q3:** use FUNWAVE for bars, Thetis for planform, SW for the dispersion split; fix $C_d$ and
$\beta$, add an A=0 FUNWAVE case for a direct Thetis comparison, and move production to a moderate
MF once the MF=1 baseline validates it.
