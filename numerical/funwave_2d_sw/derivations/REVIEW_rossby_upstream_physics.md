# Can the meander propagate UPSTREAM (Rossby) instead of downstream (gravity)?

A settled answer, with a **validated** linear eigen-solver. Written after several of my own errors
(recorded honestly below), and while the 2×2 FUNWAVE run is computing.

## 0. The corrections I owe (the user was right on the physics)

- **Rossby waves DO propagate upstream relative to the mean flow, with NO instability.** Their
  intrinsic phase speed is $c = \bar U - \beta/(k^2+l^2)$; the retrograde ($c<\bar U$, and $c<0$ for
  strong enough $\beta$) is a property of the **neutral** wave. This is separate from the
  Rayleigh–Kuo *instability* condition (PV gradient changes sign).
- I mis-stated two things earlier: (i) I tied "upstream" to the $q_y$ sign change — wrong, that's the
  *amplification* condition; (ii) I claimed a semicircle bound on *neutral* modes — wrong, Howard's
  semicircle bounds **unstable** modes only.

## 1. The solver is validated

The QGPV/Rayleigh–Kuo eigen-problem $(\bar U-c)\,Q[\phi]+\beta\phi=0$, $Q=\partial_{nn}-2(\bar U'/\bar U)\partial_n-k^2$,
solved with proper interior ($\phi=0$ at banks) reduction, **reproduces the textbook barotropic
Rossby wave exactly** when given a uniform flow and an *external constant* $\beta$:

| $\beta$ | $\tilde k$ | theory $c=1-\beta/(k^2+l^2)$ | numeric |
|---|---|---|---|
| 2 | 0.5 | +0.264 | +0.264 |
| 5 | 0.5 | **−0.840** | **−0.840** (upstream) |
| 10 | 0.5 | **−2.680** | **−2.680** (upstream) |

So the tool is correct, and **upstream Rossby waves are real and computable.**

## 2. The crux: where does $\beta$ come from, and does a river have it?

$\beta$ is a **background potential-vorticity gradient**. Its physical sources:

| source | expression | needs rotation $f\neq0$? |
|---|---|---|
| planetary $\beta$ | $\mathrm df/\mathrm dy$ | **yes** |
| topographic $\beta$ | $f\,\nabla H/H$ | **yes** (vanishes at $f=0$) |
| shear (barotropic) | $-\bar U''$ (and the SW form $-\bar U''/H+\bar U'H'/H^2$) | no |

**A river meander is non-rotating at its scale:** $Ro=U/(fL)=0.85/(10^{-4}\cdot10^3)\approx8.5\gg1$.
So the planetary and topographic $\beta$'s are absent, and the **only** PV gradient is the
shear-derived $q_y=-\bar U''/H+\bar U'H'/H^2$.

**For a smooth (single-signed $q_y$) jet, this gives only ADVECTED-DOWNSTREAM vortical modes.** The
eigen-solve confirms $c\in[U_{\min},U_{\max}]$ at every wavelength, jet strength and depth. The
"upstream" values I kept extracting were an artefact: at shallow depth the retrograde eigenvalue was
exactly $c=U-\sqrt{gH}$ (e.g. $0.85-\sqrt{9.81\cdot1.0}=-2.28$) — **the upstream gravity
characteristic**, a fast gravity wave that always exists in subcritical flow, *not* a Rossby wave.

## 3. Verdict on Q3

> In a non-rotating shallow-water channel with a **smooth (single-signed $q_y$) jet**, there is **no
> genuine retrograde Rossby wave**. The vortical mode is advected **downstream**; the only upstream
> signal is the fast gravity characteristic. A genuinely **upstream** meander wave requires the PV
> gradient to change sign — an **INFLECTED jet** (Rayleigh–Kuo), which the channel gets from a
> **central ridge / compound cross-section** (a bed shallower in the middle makes the jet
> double-peaked, so $\bar U''$ — hence $q_y$ — flips sign).

Tested directly: a central-ridge jet **does** make $q_y$ change sign (confirmed for ridge depth
$\ge0.3\,U_0$), and the slowest mode's phase speed drops toward zero as the ridge deepens
(c: +0.43 → +0.34 → +0.09) — trending to retrograde, though a clean crossing needs a stronger
inflection and a cleaner solver (swe_stability's v=0 boundary rows inject $c\approx0$ artefacts).

## 4. Reconciling with the SW meander doc

The doc's two distinguished limits are **fast-vs-slow, both downstream**, not up-vs-down:

- **Ikeda / "gravity" limit** ($\alpha\!\sim\!\varepsilon^{1/2}$, $Fr\!\sim\!1$): the response projects
  on the **fast gravity** branch.
- **QGPV / "Rossby" limit** ($\alpha=1$, $Fr^2\!\sim\!\varepsilon$): the response projects on the
  **slow vortical** branch — "Rossby-**like**" in the sense of *PV-carrying and slow*, and it is what
  the $R=\beta_{\rm eff}b^2/(F^2U)=31$ number flags. But it still **advects downstream** for a smooth
  jet. "Rossby-driven" here means *slow/vortical*, **not** *retrograde*, unless the jet is inflected.

## 5. What this means for the runs (and the design)

**Launched now (robust, feasible):** the 2×2 `{A=0, A=2.89} × {λ=780 (k8), λ=1560 (k4)}` on the
smooth quadratic jet at $H_c=3$ m. This is guaranteed to run and produces the mp4s. It documents the
A and wavelength effects on the **bed morphodynamics** and lets us **measure** the actual migration
direction/phase — which, by the above, should be **downstream** (slow/vortical), confirming the
physics empirically rather than by assertion.

**To actually get an UPSTREAM (Rossby) bank — the one genuine route:** an **inflected-jet base
state**. Concretely, carve a **central longitudinal ridge** (bed shallower on the centreline than at
$|n|\sim b/2$) so the jet is double-peaked and $q_y$ changes sign. Then:
- the Rayleigh–Kuo modes can be **retrograde** (upstream) *and/or* **amplifying**,
- and A=0 vs A=2.89 tests whether the secondary-flow closure shifts that threshold.

This is a new geometry (new `section_depth`, re-calibration, and a check that the ridge does not
re-introduce a thin/oblique wet–dry bank), so it should be built and eigen-validated deliberately,
not gambled on unattended. The `swe_stability.py` extractor should also be hardened (project out the
$c\approx0$ boundary null-space) before it is trusted for the inflected case.

**Bottom line:** the "Rossby vs gravity" contrast is a **jet-profile** question (inflected vs
smooth), not a $U_0$/$\beta$-tuning of the smooth jet — and $U_0$ in particular cancels from the
retrograde ratio $\beta/(U_0K^2)$ for the shear-derived $\beta$. The smooth-jet 2×2 is running; the
inflected-ridge case is the correct next build for a true upstream meander.
