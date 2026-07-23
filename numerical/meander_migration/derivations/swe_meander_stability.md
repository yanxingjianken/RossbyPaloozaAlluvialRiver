# Shallow-water meander stability: gravity **and** Rossby, from a PV gradient

**Revision of the Ikeda linear bend model** (`bend_model.py`) in response to the reviewer
comment: *"Ikeda is friction balancing pressure gradient → a gravity-wave mechanism. Add the
shallow-water PV dynamics so the model also carries a Rossby-wave option from a background PV
gradient — β_topo from the bed H(y) and β_shear = −U_yy from a quadratic mean jet U(y)=U₀−βy²."*

The reviewer is correct. This note (i) shows precisely **where Ikeda's reduction throws the PV
dynamics away**, keeping only the gravity/friction response, and (ii) restores the full
depth-averaged shallow-water operator, which carries **both** a gravity branch and a vortical
(Rossby) branch, and (iii) gives the dimensionless number that decides which one drives river
meandering. Ikeda, Parker & Sawai (1981) is on disk; the shallow-water algebra below is standard.

---

## 1. Base state (straight channel) — and why Ikeda *is* the gravity balance

Depth-averaged shallow water, no planetary rotation (`f = 0`, a river), bed friction `C_d`,
free surface `η`, rigid bed `z_b(y)` so the still depth is `H(y) = η̄ − z_b(y)`:

```
mass:   h_t + (h u)_x + (h v)_y = 0                         h = H + η'
x-mom:  u_t + u u_x + v u_y = −g η_x − C_d u|u|/h
y-mom:  v_t + u v_x + v v_y = −g η_y − C_d v|u|/h
```

**Base state:** `u=U(y)`, `v=0`, surface tilted only down-valley, `η̄ = −S x`. The x-momentum
base balance is

> **`C_d U² / H = g S`   ⇔   U(y) = √(g H(y) S / C_d)`   (normal flow).**   (B1)

This is **exactly the friction ↔ pressure-gradient balance the reviewer names**, and it is the
whole of Ikeda's driving physics: curvature perturbs this balance, the near-bank velocity
responds, the bank erodes. The response is set by how fast the **free surface** (pressure,
`g η`) can re-tilt against friction — a **gravity/topographic-steering** mechanism, measured by
the Froude number `F² = U²/(gH)`. Note (B1) makes `U ∝ √H`, so a quadratic jet and a quadratic
bed are the *same* statement to leading order; below we keep `U(y)` and `H(y)` general.

---

## 2. The background potential-vorticity gradient (the Rossby ingredient Ikeda lacks)

Depth-averaged potential vorticity, `f = 0`:

> **`q = ζ / h`,   ζ = v_x − u_y`.**   (2.1)

Base state `ζ̄ = −U'(y)`, so `q̄ = −U'(y)/H(y)`, and the **background PV gradient** is

> **`q̄_y = d/dy(−U'/H) = −U''/H  +  U' H'/H²  =  β_shear + β_topo`.**   (2.2)

- **`β_shear = −U''/H`** — from the jet curvature `U_yy`. A quadratic jet `U = U₀(1−(y/b)²)`
  gives `U'' = −2U₀/b²`, so `β_shear = 2U₀/(b²H) > 0` (a **constant** β, exactly the reviewer's
  `−U_yy`).
- **`β_topo = U' H'/H²`** — from the cross-channel bed `H(y)`; it is the river analogue of the
  `f/H` topographic-β, here carried by the *shear* rather than planetary vorticity (`f=0`).

A **Rossby (vortical) wave exists because `q̄_y ≠ 0`.** Ikeda's model has no `q̄_y` term at all:
his cross-stream closure replaces the vorticity dynamics with the algebraic secondary-flow
parameter `A`, so `∂q'/∂t + v' q̄_y` never appears. **That is the missing branch.**

---

## 3. Linearised shallow-water operator (`e^{ikx+σt}φ(y)`)

Perturb `(u',v',η')`, define the friction rate `r = C_d U/H` (so the base friction is `rU`):

```
(mass)  σ η' + ik(H u') + ik U η' + (H v')_y = 0
(x)     σ u' + ikU u' + U' v' = −ikg η' − 2r u' + r(U/H) η'
(y)     σ v' + ikU v'          = −g η'_y      − r v'
```
(3.1)

Cross-differentiate (x),(y) to get the **perturbation vorticity / PV equation** — this is the
object that exposes the two branches:

> **`(σ + ikU) q'  +  v' q̄_y  =  −r q'  +  (friction-curl & surface-stretching terms)`,**  (3.2)

with `q' = ζ'/H − (ζ̄/H²) η'`. Equation (3.2) is a **Rossby-wave equation** (advection `ikU`,
restoring `v' q̄_y`, friction `−r`). Coupled to the divergence/continuity (3.1a) and the free
surface, the operator (3.1) has **three** y-eigenbranches for each `k`:

| branch | scale | physics | in Ikeda? |
|---|---|---|---|
| two **gravity** modes | `σ ≈ ikU ± i√(gH)|k| − r` | surface pressure ↔ inertia, damped by `r` | **yes** (the `F²` term) |
| one **vortical/Rossby** mode | `σ ≈ ikU − i β_eff k/(k²+l²) …` | PV advection against `q̄_y` (2.2) | **NO** — dropped |

`l ~ π/b` is the cross-channel wavenumber (bank BCs). The vortical mode is the slow one that a
**meander** (a slowly growing, migrating bend) actually rides.

---

## 4. Ikeda **is** the `F²→` (quasi-steady, PV-suppressed) reduction

Take (3.1) in Ikeda's limits: **quasi-steady flow** (`σ ≪ ikU`, the flow is slaved to the
current planform) **and** drop the vorticity time-tendency (`∂q'/∂t → 0`, no PV advection),
replacing the cross-stream balance by the algebraic secondary-flow closure `η'/H = −A C n`.
Eliminating `u',v',η'` for the near-bank velocity `u_b'` then collapses (3.1) to a single
streamwise ODE in arc length — **Ikeda eq (10)** — and, with the bank law, to the bend equation

> **`y_xt + 2C_f y_t = y_xxx − C_f(A + F²) y_xx`.**   (Ikeda 16)

The surviving driver is **`A + F²`**: `F²` is the **gravity** (free-surface) response and `A` is
the **parameterised** secondary flow. **There is no `q̄_y` — the Rossby branch was removed by the
quasi-steady + algebraic-closure step, not by any physical smallness.** This is exactly the
reviewer's point, made precise.

---

## 5. The extension: keep the PV branch → Rossby resonance

Do **not** slave the flow: retain `(σ + ikU)q'` and `v' q̄_y` in (3.2). Solving (3.1) as a
cross-channel two-point boundary-value/eigenvalue problem in `y` (banks: `v'=0` for the rigid-bank
flow response, or the Ikeda erosion law for the coupled meander) yields the **full dispersion**
`σ(k)` with the gravity branches *and* the vortical/Rossby branch. The near-bank velocity that
drives migration,

> **`u_b'(k) = u'(±b; k)`   →   γ ∂y/∂t = E u_b'   (Ikeda 11–13),**   (5.1)

now inherits **two** resonances in `k`:

- a **gravity resonance** near `k ≈ k_g` set by `F²` (Ikeda's), and
- a **Rossby resonance** near the wavenumber where the meander phase speed matches the vortical
  wave, `U ≈ β_eff/(k² + l²)`, i.e. **`k_R ≈ √(β_eff/U − l²)`**.   (5.2)

The **meander wavelength selection** (max growth of `y`) is pulled toward whichever resonance is
stronger.

---

## 6. Gravity **vs** Rossby — the dimensionless number

Compare the two restoring terms in (3.2)/(5.1) at the cross-channel scale `b`:

> **`R ≡  (Rossby restoring)/(gravity restoring)  ≈  β_eff b² / (F² U)  =  (b²/H)(β_shear+β_topo)/F²`.**  (6.1)

With `β_shear = 2U₀/(b²H)` (quadratic jet) the shear part gives `β_shear b²/U = 2/H`
(dimensionless when `H` is scaled by depth → `O(1)`), while the gravity part is `F² = U²/(gH)`.
For a river `F² ~ 0.02–0.25`, so

> **`R ~ 1/F² ≫ 1`  →  the vortical/PV (Rossby) restoring dominates the gravity one.**

i.e. **river meandering is a vortical / shear-Rossby instability, not a gravity wave** — and its
migration speed `c₀ = ω₀/k ≪ √(gH)` (Ikeda already shows `ω₀ = C_f k³(A+F²)/(k²+4C_f²)`, orders
below the gravity speed). Ikeda captured the *right slow instability* but *mislabelled its
restoring*: the `A` secondary-flow term is standing in for what is physically the PV-gradient
`β_eff`. Making `q̄_y` explicit (via `U(y)`, `H(y)`) is what the SWE extension buys.

**Test knobs** (all in `swe_stability.py`): set `β_topo=0` (flat bed) vs `β_shear=0` (uniform
jet) to isolate the two β sources; sweep `F²` (via `U₀` or `H`) to watch the gravity resonance
move relative to the Rossby one; the crossover `R=1` is the gravity↔Rossby boundary.

---

## 7. What the two plots are for (in this framing)

- **yOz cross-section** — the **base state that sets the PV gradient**: `U(y)` (the quadratic
  jet, its `U_yy` = β_shear) and the bed `H(y)` (its `H_y` = β_topo). This panel is where the
  Rossby ingredient lives; without a structured `U(y)`/`H(y)` there is no `q̄_y` and no Rossby
  option. (In the mobile-bed FUNWAVE run this same panel showed the point bar; here it shows the
  *rigid* equilibrium section that defines the wave problem.)
- **xOy bird's-eye** — the **wave itself**: the meander planform `y(x,t)` and the perturbation
  flow/PV `u'(x,y), q'(x,y)` riding on it, and its **down-valley migration** `c₀`. This is where
  "the bank + river moves" is read off, and where a gravity-dominated vs Rossby-dominated meander
  looks different (resonant wavelength and migration speed differ).

---

## 8. Response to reviewer (point-by-point)

1. **"Ikeda is friction ↔ pressure = gravity waves."** *Agree, and made exact:* the base balance
   (B1) is friction↔pressure, and §4 shows the surviving driver in Ikeda (16) is `A+F²` with `F²`
   the free-surface/gravity response — no PV term. `[ADDRESSED — §1, §4]`
2. **"Add SWE to include a Rossby option from a PV gradient."** *Done:* §2 derives
   `q̄_y = −U''/H + U'H'/H² = β_shear+β_topo`; §3 keeps the vortical branch (3.2) that Ikeda
   dropped; §5 gives the Rossby resonance. `[ADDRESSED — §2, §3, §5]`
3. **"β_topo from H(y), β_shear from U_yy (quadratic jet)."** *Exactly (2.2):* `β_shear=−U''/H`
   is the `−U_yy` term; `β_topo=U'H'/H²` is the bed term. `[ADDRESSED — §2]`
4. **"Diagnose gravity vs Rossby."** *Number given:* `R ≈ β_eff b²/(F²U) ~ 1/F² ≫ 1` (6.1) →
   vortical/Rossby-dominated for river Froude numbers; falsifiable by the `swe_stability.py`
   knobs. `[ADDRESSED — §6]`

**Reviewer-disagree (one point, justified):** calling the vortical branch a *Rossby* wave is
correct only in the generalised (shear/topographic-β, `f=0`) sense — it is **not** a planetary
Rossby wave. The restoring is the mean-flow PV gradient `q̄_y`, so the honest label is
"**shear/topographic Rossby (vortical) wave**." Kept throughout to avoid overclaiming.
