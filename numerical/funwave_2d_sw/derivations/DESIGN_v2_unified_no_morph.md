# Meander morphodynamics in FUNWAVE-TVD — model specification (v2)

A depth-averaged shallow-water + Exner simulation of a sinuous channel with a fully erodible
bed-and-bank surface, integrated **without morphological acceleration**.

Line numbers refer to `run_meander.py` in this directory's parent. `mod_sediment.F` line numbers
are prefixed `sed:`.

---

## 0. Nomenclature

### 0.1 Dependent variables (all depth-averaged; there is no vertical coordinate)

| symbol | meaning | sign / units |
|---|---|---|
| $\eta(x,y,t)$ | free-surface elevation above the still-water datum | up +, m |
| $h(x,y,t)$ | still-water depth | down +, m |
| $H=\eta+h$ | total water depth | $\ge0$, m |
| $(u,v)$ | depth-averaged velocity | m s$^{-1}$ |
| $(P,Q)=(Hu,Hv)$ | volume flux per unit width | m$^2$ s$^{-1}$ |
| $\bar c(x,y,t)$ | depth-averaged volumetric sediment concentration | – |
| $z_b=-h$ | bed elevation | up +, m |
| $Z_b$ | FUNWAVE's cumulative bed change, **positive for erosion** | m |

> **Sign trap.** `Zb` is positive for erosion and enters as `Depth = Depth_ini + Zb*Morph_factor`.
> Erosion therefore *increases* `Depth` and *decreases* $z_b$.

### 0.2 Geometry symbols — and how each is obtained

| symbol | meaning | how obtained |
|---|---|---|
| $x,y$ | down-valley, cross-valley Cartesian coordinates | grid |
| $s,n$ | along-channel arc length, cross-channel offset from the centreline | **derived** by nearest-point projection, `channel_coords` **L485** |
| $n\,\mathrm{sgn}\,\kappa>0$ | the **inner** bank of the local bend | convention |
| $b$ | channel **half**-width = 50 m | **input** (`CONFIG` L51) |
| $W=2b$ | channel full width = 100 m | derived |
| $\lambda$ | down-valley meander wavelength | **input** per case |
| $k=2\pi/\lambda$ | meander wavenumber | derived, **L290** |
| $\theta(s)$ | deflection angle of the centreline from the valley axis | **derived**, `centreline` **L330** |
| $\theta_m$ | maximum deflection angle | **derived** from $(\lambda,C_0)$ by Bessel root-find, `theta_max` **L312** |
| $\kappa(s)=\mathrm d\theta/\mathrm ds$ | centreline curvature | **derived**, m$^{-1}$ |
| $C_0=\max_{[0,L]}\lvert\kappa\rvert$ | **apex curvature** — the tightest bend | **input** per case, m$^{-1}$ |
| $R_{\min}=1/C_0$ | radius of curvature at the apex | derived, m |
| $\boxed{R/W=R_{\min}/(2b)}$ | **bend tightness** — the standard dimensionless measure of how sharp a bend is. Natural meanders cluster at $R/W\approx2$–3; large $R/W$ = gentle sweeping bend, small = tight hairpin | derived |
| $A$ | peak lateral excursion (amplitude) of the centreline | **measured off the built curve**, `amplitude` **L281**, m |
| $\boxed{\sigma=1/J_0(\theta_m)}$ | **sinuosity** = (channel arc length)/(down-valley length). $\sigma=1$ is a straight channel; $\sigma=1.2$ means the water travels 20 % further than the valley | derived, `sinuosity` **L435** |
| $L$ | down-valley reach length | derived, `reach_length` **L425**, m |
| $\ell_b,\ell_s$ | buffer length, straight lead-in length | input, m |
| $T(s)$ | amplitude taper (raised cosine) ramping the bends up from the straight lead-in | derived, `taper_arc` **L370** |
| $h_{\rm plain}$ | depth of the always-wet shelf beyond the bank toe = 0.20 m | input |
| $H_c,H_b$ | channel-centre depth (3.0 m), depth at the channel edge $|n|=b$ (1.5 m) | input |
| $m_{\rm bank}$ | bank face slope, 1 : $m_{\rm bank}$ = 1:5 | input |
| toe | $b+m_{\rm bank}(H_b-h_{\rm plain})=56.5$ m — where the bank face meets the shelf | derived |

### 0.3 Dimensionless groups

| symbol | definition | value here | why it matters |
|---|---|---|---|
| $\mathrm{Fr}$ | $U/\sqrt{gH}$ | 0.157 design, **0.18 measured** | **Froude number.** $<1$ subcritical (gravity waves outrun the flow, surface quasi-rigid); $\to1$ critical. Natural lowland rivers 0.1–0.3 |
| $\beta$ | $b/H$ | **22.5** (at the realised $H=2.22$ m; $b/H_c=16.7$ only if the centre depth is used) | **aspect ratio** — the controlling parameter for free alternate-bar instability; bars require $\beta$ above roughly 10–15. At $\beta=22.5$ the channel is *well* above threshold, so free bars can coexist with the forced response |
| $\theta$ | $\tau_b/[(s_g-1)gD_{50}]$ | 0.19 channel centre | **Shields stress** — dimensionless bed shear. Motion requires $\theta>\theta_{cr}$ |
| $\theta_{cr}$ | critical Shields | 0.055 suspension, 0.047 bedload | threshold of grain motion |
| $u_*/w_s$ | shear velocity / settling velocity | 0.45 | $<1$ ⇒ **bedload-dominated** regime |

### 0.4 Sediment and fluid constants

$g=9.81$; $\rho=1000$; $s_g=\rho_s/\rho=2.68$; $D_{50}=0.5$ mm; $n_p=0.47$ (porosity);
$\tan\phi=0.70$ (angle of repose); $k_s=2.5D_{50}$ (Nikuradse roughness); $\kappa^2=0.16$
(von Kármán); $A_{\rm ik}=2.89$ (Ikeda secondary-flow coefficient); $C_d=0.00154$.

---

## 1. Governing equations

### 1.1 Hydrodynamics — three primitive equations

Solved by FUNWAVE-TVD with `DISPERSION = F` (which sets $\Gamma_1=\Gamma_2=0$, reducing the
Boussinesq system to nonlinear shallow water). `run_meander.py` supplies parameters only.

**Mass:**
$$\partial_t\eta + \partial_x(Hu) + \partial_y(Hv) = -\,\partial_t h$$

The right-hand side is the coupling to the moving bed: a rising bed displaces water.

**$x$-momentum** (conservative, well-balanced form):
$$\partial_t P + \partial_x\!\left[\frac{P^2}{H}+\tfrac12 g\!\left(\eta^2+2\eta h\right)\right]
 + \partial_y\!\left[\frac{PQ}{H}\right]
 = g\eta\,\partial_x h \;-\; C_d\,u\sqrt{u^2+v^2}\;+\;\nabla\!\cdot\!\left(\nu_t H\nabla u\right)$$

**$y$-momentum:**
$$\partial_t Q + \partial_x\!\left[\frac{PQ}{H}\right]
 + \partial_y\!\left[\frac{Q^2}{H}+\tfrac12 g\!\left(\eta^2+2\eta h\right)\right]
 = g\eta\,\partial_y h \;-\; C_d\,v\sqrt{u^2+v^2}\;+\;\nabla\!\cdot\!\left(\nu_t H\nabla v\right)$$

Splitting the hydrostatic pressure as $\tfrac12g(\eta^2+2\eta h)$ against the source
$g\eta\nabla h$ makes the scheme **well-balanced**: lake-at-rest is preserved exactly over an
arbitrary bed — essential here because the bed is strongly non-flat by construction.

**Closures:**
$$\bm\tau_b/\rho = C_d\,\bm u\,|\bm u|,\qquad
\nu_t = C_{\rm smg}\,\Delta x\,\Delta y\sqrt{u_x^2+v_y^2+\tfrac12(u_y+v_x)^2}+\nu_{\rm bkg}$$

$C_d$ is **not free**: the sediment module computes its own bed shear from a log law
independent of `input.txt`'s `Cd`, so consistency requires
$$C_d=\frac{\kappa^2}{\left[\ln(30H/k_s)-1\right]^2}=0.00154 .$$

> **Verified caveat:** `nu_smg` is assigned `ZERO` once at `src/init.F:437` and never re-set, so
> despite `C_smg=0.25` the eddy viscosity $\nu_t$ is **identically zero in the built executable**.
> The lateral-mixing term above is therefore inactive. Any stabilisation strategy relying on it
> must first make that path live.

### 1.2 Sediment transport

$$\tau_b=\frac{\kappa^2|\bm u|^2}{\left[\ln(30H/k_s)-1\right]^2},\qquad
  \tau_{cr}=(s_g-1)\,g\,D_{50}\,\theta_{cr},\qquad
  D_*=D_{50}\!\left[\frac{(s_g-1)g}{\nu^2}\right]^{1/3}$$

$$w_s=\frac{\nu}{D_{50}}\left[\sqrt{10.36^2+1.049D_*^3}-10.36\right]=0.0745\ \mathrm{m\,s^{-1}}
\quad\text{(Soulsby 1997)}$$

**Erosion — van Rijn (1984)**, for $\tau_b>\tau_{cr}$ and $H>H_{\rm pick}$:
$$c_b=0.015\left[\frac{\tau_b-\tau_{cr}}{\tau_{cr}}\right]^{3/2}\!\!D_*^{-0.3},\quad
  c_a=\min\!\left(1,\frac{0.65}{c_b}\right)c_b\frac{D_{50}}{0.01H},\quad
  P=\max\left(0,\,c_a w_s\right)$$

$H_{\rm pick}=0.01$ m is load-bearing: the shipped 0.1 m switches pickup off *at the bank toe*,
which is the entire bank-retreat mechanism; zero is impossible because the log law is singular at
$H=e\,k_s/30=1.13\times10^{-4}$ m.

**Deposition — Cao:**
$$\gamma=\min\!\left[2,\frac{1-n_p}{\bar c}\right],\qquad
  D=\gamma\,\bar c\,w_s\left(1-\gamma\bar c\right)^2$$

$P$ depends on $\tau_b$ and not on $\bar c$ (pure source); $D$ is linear in $\bar c$ (sink). Both
act at every wet point; under capacity transport $D=P$ and the bed does not move.

**Suspended load:**
$$\partial_t(\bar cH)+\nabla\!\cdot\!(\bar cH\bm u)=\nabla\!\cdot\!(kH\nabla\bar c)+P-D$$

a relaxation equation with $T_c=H/(\gamma w_s)=20.1$ s and adaptation length $L_a=UT_c=20$ m.

**Bedload — Meyer-Peter & Müller (1948)**, for $\tau_b>\tau^b_{cr}$:
$$|\bm q_b|=\frac{8\left(\tau_b-\tau^b_{cr}\right)^{3/2}}{g(s_g-1)},\qquad
  \bm q_b=|\bm q_b|(\cos\alpha,\sin\alpha)$$

### 1.3 Bed evolution

**Exner:**
$$\boxed{\;(1-n_p)\,\partial_t z_b = D-P-\nabla\!\cdot\!\bm q_b\;}$$

integrated cumulatively with $Z_b>0$ meaning erosion and capped by the hard bottom $Z_b\le Z_s$;
$\bar P,\bar D$ are means over `Morph_interval`. The bed seen by the hydrodynamics is
$$h=h_{\rm ini}+Z_b\cdot\mathrm{MF},\qquad \boxed{\mathrm{MF}=8\ \text{in v2 (see §3.1)}}$$

**Avalanching** (every `Aval_interval`): wherever the local slope exceeds the angle of repose,
$$\text{if}\ \max_{j\in\mathcal N(i)}\frac{z_{b,i}-z_{b,j}}{\Delta x}>\tan\phi\ \text{and}\ Z_{b,i}<Z_{s,i}:
\quad \delta=\tfrac12\left(z_{b,i}-z_{b,j}\right)-\tfrac12\tan\phi\,\Delta x$$
with $z_{b,i}\!-\!=\!\delta$, $z_{b,j}\!+\!=\!\delta$.

**There is no separate bank law.** A bank is an ordinary bed cell at the channel margin and
erodes through the same equations as any other. The cross-section is one continuous surface, so
bed and bank are not separable — they erode and deposit together. Bank *retreat* is therefore an
emergent outcome of (i) toe erosion steepening the face and (ii) avalanching relaying that
steepening upslope; its rate is set by $D_{50},\theta_{cr},\tan\phi$, **not** by an erodibility
coefficient $E$, and is therefore not directly comparable to a kinematic law
$\partial_t\zeta_c=E\cdot\tfrac12[u_s'(+b)-u_s'(-b)]$ without recalibration.

### 1.4 Secondary-flow closures (the depth-averaged model cannot resolve helical flow)

Written as spatial fields by `write_case` **L735** and consumed by the solver:

$$C_d^{\rm eff}=C_d\,\mathrm{clip}\!\left(1+A_{\rm ik}\,\kappa\,n\,T_\perp,\;0.2,\;1.8\right)
  \times\left[1+(m_{\rm fp}-1)f_{\rm fp}\right] \qquad\text{(\texttt{cd.txt})}$$

$$\text{bedload deflection}\quad \alpha=\operatorname{atan2}(v,u)+\delta_{sf},\qquad
  \delta_{sf}=\frac{A_{\rm ik}\,\kappa\,H}{f_{\rm slope}} \qquad\text{(\texttt{kappa.txt})}$$

$$\left.\frac{\partial z_b}{\partial n}\right|_{\rm eq}=+A_{\rm ik}\,\kappa\,H
  \quad\text{(shallow inner / point bar)} \qquad\text{(\texttt{bedslope.txt})}$$

$T_\perp$ tapers the closure to zero across the bank face; $m_{\rm fp}=30$ is a floodplain drag
multiplier confining the flow to the channel; $f_{\rm fp}$ ramps it on beyond the toe.

---

## 2. Geometry construction

The centreline is **sine-generated** (Langbein–Leopold): the deflection angle, not the position,
is prescribed, and the curve is built by integrating it in arc length.

$$\theta(s)=\theta_m\,T(s)\,\sin(k_s s),\qquad k_s=C_0/\theta_m,\qquad \kappa=\mathrm d\theta/\mathrm ds$$
$$x(s)=\int\!\cos\theta\,\mathrm ds,\qquad y(s)=\int\!\sin\theta\,\mathrm ds
\qquad\text{(\texttt{centreline} \textbf{L330})}$$

$\theta_m$ is not free: wavelength and apex curvature are **coupled** through
$$\lambda=\frac{2\pi J_0(\theta_m)\,\theta_m}{C_0}\qquad\text{(\texttt{theta\_max} \textbf{L312})}$$
solved on the gentle branch. Because $J_0(\theta)\theta$ peaks at 0.80736 ($\theta_{\rm peak}=1.25578$),
there is a hard admissibility bound
$$\lambda\le\lambda_{\max}=\frac{2\pi(0.80736)}{C_0}\qquad\text{(\texttt{lam\_max} \textbf{L307})}$$
beyond which **no such meander exists**. Sinuosity follows as $\sigma=1/J_0(\theta_m)$.

**Drift removal.** A tapered, non-integer-bend reach leaves $\int\!\sin\theta\,\mathrm ds\ne0$, so the
curve wanders off-axis. It is re-centred on the excursion **mid-range** (not the mean, which does
not equalise the closest approach to each wall):
$$y\leftarrow y-\tfrac12\!\left(\max_{[0,L]}y+\min_{[0,L]}y\right)$$

**Cross-section** (`section_depth` **L458**), with $w_0=H_c^{-1/2}$ and $\beta_s=2(H_b^{-1/2}-w_0)/b^2$:
$$h(n)=\begin{cases}
\left(w_0+\tfrac{\beta_s}{2}\min(|n|,b)^2\right)^{-2}, & |n|\le b \quad\text{(constant-PV-gradient profile)}\\[6pt]
\max\!\left(H_b-\dfrac{|n|-b}{m_{\rm bank}},\ h_{\rm plain}\right), & |n|>b \quad\text{(bank face, then always-wet shelf)}
\end{cases}$$

The shelf is **not cosmetic**: a dry floodplain puts a wet/dry boundary along an oblique,
staircased bank, which is the documented failure mode of this configuration. Keeping the whole
domain wet removes the wetting–drying algorithm from the problem while leaving the bank erodible.

**Head budget** (`slope_design` **L294**, `slope` **L299**):
$$S_{\rm design}=\frac{C_dU^2}{gH_c},\qquad S_{\rm bed}=\mathcal H\,S_{\rm design}$$
where $\mathcal H$ (`head_factor`) absorbs bend losses the straight-channel slope omits.

> **Quote curvature as $C_0$, never as $Ak^2$.** For a tapered cosine $y_c=A\,T(x)\cos kx$ one would
> have $C_0=Ak^2$, but this construction is sine-generated with $C_0$ prescribed and $A$ measured.
> The identity fails as built: $Ak^2/C_0$ = 1.12 (B1), **1.45 (B2)**, 1.05 (B3), while measured
> $\max|\kappa|$ equals $C_0$ exactly in all three. The cases are curvature-matched in the quantity
> the flow feels ($\max|\kappa|$), not in $Ak^2$.

---

## 3. Configuration

### 3.1 Timescales, and why there is no morphological acceleration

$$T_{\rm flow}=\frac{\mathrm{CFL}\,\Delta x}{U+\sqrt{gH}},\quad
T_c=\frac{H}{\gamma w_s},\quad
T_{\rm adjust}=\frac{H}{C_dU},\quad
T_{\rm bed}=\frac{(1-n_p)H\mathcal L}{q_b},\quad
T_{\rm bank}=\frac{b}{\varepsilon U}$$

| | $T_{\rm flow}$ | $T_c$ | $T_{\rm adjust}$ | $T_{\rm bed}$ | $T_{\rm bank}$ |
|---|---|---|---|---|---|
| value | 0.20 s | 20.1 s | **2292 s** | $1.68\times10^7$ s | $\sim5\times10^9$ s |

$T_{\rm flow}$ is the **CFL step**, not an adjustment time; the timescale the flow actually needs to
re-equilibrate after a bed change is $T_{\rm adjust}=H/(C_dU)=2292$ s (reach transit $L/U=7708$ s).
Comparing the bed against the timestep overstates the separation by ~10⁴ and is not a valid basis
for acceleration.

**Measured**: the transverse point bar **saturates at $t\approx1.5\times10^5$ s** (99th-percentile
$|\Delta z_b|$ growth falls to 0.34–0.51 of its early rate). The controlling process is transverse
redistribution across the width, not the longitudinal bar-building that $T_{\rm bed}$ estimates.
Hence the operative separation is $T_{\rm eff}/T_{\rm adjust}\approx1.5\times10^5/2292\approx65$,
which is **directly integrable**.

$$\boxed{\ \mathrm{MF}=8,\qquad t_{\rm hydro}=6\times10^{4}\ \mathrm{s}\ \Rightarrow\
  t_{\rm morph}=\mathrm{MF}\cdot t_{\rm hydro}=4.8\times10^{5}\ \mathrm{s}\ }$$

**Choice of MF (updated 2026-07-24).** The validity limit is $\mathrm{MF}\ll T_{\rm morph}/T_{\rm adjust}$
evaluated on the *fastest* morphological process (the bar): $T_{\rm bar}/T_{\rm adjust}\approx65$, so
$\mathrm{MF}\le\sim10$ is safe and $\mathrm{MF}=1000$ (the old derivation) is not. **MF=8** sits under
that ceiling; because this run is **A=0** (see below), the forcing $F^2=0.09$ is far weaker than the
A=2.89 case, $\dot z_b$ is smaller, and the true ceiling is *higher* — MF=8 is comfortably safe. An
MF=1 run is kept as the convergence anchor. Hot-starting (`run_v2.py` carries `depth_cur.txt` across
chunks) makes each re-adjustment a cheap **perturbation** relaxation, not a cold spin-up — this is
what makes the fast/slow timescale gap affordable. `Morph_interval` stays 30 s ($\approx1.5\,T_c$).

$T_{\rm bank}$ implies $\varepsilon=b/(T_{\rm bank}U)=1.2\times10^{-8}$ — the standard natural range —
and is **not reachable**; this study claims bar and bend morphodynamics and the *phase* of bank
response, not planform migration rate.

### 3.1a Secondary flow: A = 0 (incised case)

Set $A_{\rm ikeda}=0$ (`A_secondary=A_ikeda`, so one switch zeros all three closures: the friction
modulation, the bedload deflection $\delta=A\kappa H$, and the equilibrium tilt). This is I81's named
**incised** parameter set, not an approximation to the alluvial one. The 3D helical circulation that
$A$ parameterises cannot be produced by a depth-averaged model, so A=0 is the **honest 2D limit**:
$h'/H=(F^2+A)C'\tilde n$ with $A=0$ leaves only the free-surface superelevation $F^2=0.09$ driving
the bend, against a free vortex ($u\sim1/r$) faster on the *inner* bank. **Which bank erodes is
therefore a measured output, not an assumption** — inner-bank erosion is a valid result. Kept active:
the floodplain drag confinement in `cd.txt` (a numerical device, not secondary flow) and the Talmon
down-slope bedload diffusion $A_{\rm bedslope}$ (gravitational, not helical). At A=0 the bedload
direction is $\alpha=\operatorname{atan2}(v,u)$ with no transverse deflection. This matches the Thetis
model (also A=0, incised) so the two are directly comparable — see `REVIEW_timescale_and_A0.md`.

### 3.2 Velocity matching; the Froude number is a constraint, not a target

**The matched quantity across cases is $U$, not $\mathrm{Fr}$.** Calibration therefore keeps its
existing form (`measure_speed` **L876**):
$$\mathcal H_{n+1}=\mathcal H_n\left(\frac{U_{\rm target}}{U_{\rm meas}}\right)^{2},
\qquad U_{\rm meas}=\left\langle|\bm u|\right\rangle_{\rm erodible\ channel},
\qquad U_{\rm target}=0.85\ \mathrm{m\,s^{-1}}$$

Rationale: the sediment forcing scales as $\theta\propto\tau_b\propto U^2$, so matching $U$ matches
the morphodynamic drive across cases — which is the control the experiment depends on. Matching
$\mathrm{Fr}$ instead would require $U\propto\sqrt{H}$ to differ between cases (their realised
$H=h+\eta$ differ slightly), which would *unmatch* the sediment forcing and confound the very
comparison the case matrix is built for.

$\mathrm{Fr}$ is instead a **constraint to satisfy, not a quantity to equalise**: it must stay
small — order $10^{-1}$ — and safely subcritical, so that the free surface is quasi-rigid and no
supercritical patches form. Measured: **0.182 / 0.191 / 0.177** (design $U/\sqrt{gH_c}=0.157$).
All are $O(0.1)$ and subcritical, so the constraint is met and no re-tuning is required. The ~8 %
spread is a consequence of matching $U$ and is accepted.

Chasing $\mathrm{Fr}=0.10$ exactly would be actively harmful: reaching it by slowing ($U=0.54$)
drops the channel to 1.2× critical Shields — the bed stops moving; reaching it by deepening
($H=7.4$ m) drops $\beta$ from 22.5 to 6.8, below the free-bar threshold, suppressing the bars
under study.

| | Fr | $\theta_{\rm channel}$ | $\beta=b/H$ (realised) |
|---|---|---|---|
| as configured | 0.157 | 0.138 (2.9× cr) | 22.5 |
| slow to Fr=0.1 | 0.100 | **0.056 (1.2× cr)** | 22.5 |
| deepen to Fr=0.1 | 0.100 | 0.116 (2.5× cr) | **6.8** |

### 3.3 Case matrix

| case | $\lambda$ (m) | $k$ (m$^{-1}$) | $C_0$ (m$^{-1}$) | $R/W$ | $A$ (m) | $\sigma$ |
|---|---|---|---|---|---|---|
| B1 | 1040 | 6.04e-3 | 3.000e-3 | 3.33 | 92 | 1.05 |
| B2 | 1560 | 4.03e-3 | 3.000e-3 | 3.33 | 268 | 1.17 |
| B3 | 1560 | 4.03e-3 | 1.418e-3 | 7.05 | 92 | 1.02 |
| B4 *(proposed)* | 1040 | 6.04e-3 | 1.418e-3 | 7.05 | ~40 | ~1.01 |

B1↔B2 vary $k$ at fixed $C_0$; B2↔B3 vary $C_0$ at fixed $k$. B4 closes the 2×2 factorial so the
$k$ and $C_0$ effects are separable rather than inferred from three corners. All share $U,H_c,b,
D_{50}$ and matched $\mathrm{Fr}$.

### 3.4 Domain in $y$

Measured at the closed (`PERIODIC=F`) walls: bed change **exactly 0.0000 m** in all three cases,
wall speed 6–8 % of the channel mean. The walls are morphologically inert, so a sponge would have
nothing to absorb and is not required. However clearance from bank edge to wall is only 36–44 m
($\approx0.4W$), and this configuration intends the bank to move. **Enlarge**: add 50 m to
`plain`, i.e. $+40$ cells in $y$, and gate on clearance each run.

### 3.5 Froude limiter

`FroudeCap = 1.0` (**L651**) is an undocumented non-conservative velocity limiter. 41 cells
(0.011 %) sit at $\mathrm{Fr}=1.000$ *exactly*, at $|n|=56$–57 m — the **bank toe** — in
$H=0.11$–0.14 m at 0.97–1.09 m s$^{-1}$. The flow wants to go supercritical there and is being
clamped, precisely where bank erosion is expected. B1 and B2 saturate the cap; B3 peaks at 0.768
and never touches it. **Log the capped-cell count as a first-class diagnostic**: if bank erosion
deepens the toe and the count rises, the outcome is limiter-controlled, not physics-controlled.

---

## 4. Protocol

1. Rebuild geometry with enlarged `plain`; verify symmetric wall clearance.
2. Calibrate $\mathcal H$ per case on $\mathrm{Fr}$ (§3.2), two iterations.
3. Rigid-bed spin-up (`Bed_Change=F`) to steady state.
4. Mobile-bed phase, MF=8, $t_{\rm hydro}=6\times10^4$ s ($t_{\rm morph}=4.8\times10^5$ s), hot-started,
   run in chunks with the closure fields rebuilt from the migrating centreline between chunks.
5. Gates each output: $\max\mathrm{Fr}$ and capped-cell count; $H_{\min}$; per-step $|\Delta z_b|$;
   wall clearance; mass-conservation residual.
6. Compare against the archived MF=5 result — the MF-convergence check.

**Measured baseline to beat.** Under MF=5 the outer-bank scour sat $+82$–$86^\circ$ of a half-bend
**downstream** of the apex in all three cases (a downstream-migration phase signature), with **no
detectable net translation** and zero bank-edge displacement.

---

## 5. Documentation/code discrepancy register

Established by direct comparison of `funwave_meander_model.tex` against `run_meander.py`:

| item | document states | code does |
|---|---|---|
| centreline | tapered cosine, $C_0=Ak^2$ | sine-generated from tapered $\theta$; $Ak^2/C_0$ up to 1.45 |
| cross-section outer branch | floor at $-1.5$ m (dry floodplain) | floor at $+0.20$ m (always-wet shelf) |
| equilibrium tilt sign | $\partial_n z_b=-A\kappa H$ | $+A\kappa H$ (code correct; doc contradicts its own algebra) |
| `Morph_interval` | 200 s ($\approx10T_c$) | 30 s ($\approx1.5T_c$); 200 s is recorded as a blow-up |
| MF | "1000" in §Timescales, 5 in §Approximations | 5 (v1), 1 (v2) |
| $\partial_n q$ constant | $1.105\times10^{-4}$ | 1.22–1.31$\times10^{-4}$ for shipped runs |
| freeboard table | $C_0=5\times10^{-3}$, $\lambda=520$ | those cases abandoned; B3 absent from the doc |
| `TideEast_U` | "must be given at both ends" | deliberately 0 (self-adjusting sink) |

Undocumented but active: the $\lambda\le\lambda_{\max}$ admissibility bound; the 30× floodplain
drag multiplier; the bank-face taper and $[0.2,1.8]$ clip on `cd.txt`; the `bedslope.txt` target
field; `FroudeCap = 1.0`; `VISCOSITY_BREAKING = T`; the $R/W\ge2$ build-time assertion.
