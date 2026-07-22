#!/usr/bin/env python3
"""Verification gate.  Run this BEFORE any figure script.

Every check compares the Dedalus IVP against either the reconstructed det M = 0 of
river.pdf p.19 or an exact symmetry of the linear operator.  Nothing is compared to a
hand-written "expected" array -- a toy test that encodes the answer proves only that the
answer was typed twice.

    env OMP_NUM_THREADS=1 micromamba run -n dedalus python postprocessing/03_verify.py

Exits non-zero if any check fails.
"""
import numpy as np

from pp_lib import (COLORS, bank_mode, fit_sigma_c, growth_and_phase,
                    momentum_flux, save_fig, set_style)

from noboru_model import CONFIG, bank_E, forced_ratio, simulate


# ---------------------------------------------------------------------------- #
#  TEST-ONLY initial states.
#  These live here, not in the driver, because they are properties of the TESTS,
#  not of the model.  The physical setup has exactly one initial condition (a carved
#  wavy channel in forced equilibrium -- noboru_model.initial_condition), and it makes
#  the varicose component identically zero, so a symmetry that has to be MEASURED
#  needs a deliberately unphysical state to excite it.  The driver stays single-valued.
# ---------------------------------------------------------------------------- #
def ic_varicose(x, cfg):
    """psi1' = -psi3', psi2' = 0.  zeta2' vanishes identically, so ONLY the
    antisymmetric subspace is excited -- which must decay at exactly -E."""
    A, k = cfg["A0"], cfg["kstar"]
    w = A * np.cos(k * x)
    return w, np.zeros_like(x), -w


def ic_lopsided(x, cfg):
    """One bank wavy, the other straight: both subspaces at once.  Used to show that
    psihat1 = psihat3 is REACHED by the dynamics, not imposed by the initial state."""
    A, k = cfg["A0"], cfg["kstar"]
    f = forced_ratio(k, cfg["D"], cfg["gamma"])
    cs, sn = np.cos(k * x), np.sin(k * x)
    return A * cs, 0.5 * A * (f.real * cs - f.imag * sn), np.zeros_like(x)


def ic_eigen(x, cfg):
    """The p.19 bank eigenmode itself -- a pure single mode, used where the momentum
    flux must be evaluated on a converged state rather than a transient."""
    A, k = cfg["A0"], cfg["kstar"]
    _, p1h, p2h = bank_mode(k, cfg["D"], cfg["gamma"], bank_E(cfg))
    cs, sn = np.cos(k * x), np.sin(k * x)
    b1 = A * (p1h.real * cs - p1h.imag * sn)
    return b1, A * (p2h.real * cs - p2h.imag * sn), b1


plt = set_style()
FAILURES = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    if not ok:
        FAILURES.append(name)


def run(_test_ic=None, **over):
    cfg = dict(CONFIG)
    cfg.update(over)
    return simulate(cfg, quiet=True, _test_ic=_test_ic)


print("verifying the river.pdf 3-level model as integrated by noboru_model.py")
print("=" * 78)

# --------------------------------------------------------------------------- #
# 1. IVP growth rate and phase speed vs the p.19 dispersion relation.
#    Started from the PHYSICAL initial condition -- a carved wavy channel in forced
#    equilibrium (river.pdf pp.12-18 + the flume on pp.17-18) -- not from the eigenvector,
#    so the dynamics still has to FIND the growing mode rather than be handed it.
#
#    t_end for k*=1.5 is long on purpose: the two roots there differ by only
#    |dsigma| ~ 0.04, so the subdominant one needs T > 6 ln10/dsigma ~ 345 to fall to
#    1e-6 of the leader.  A shorter window reads a blend of the two and looks converged.
#
#    ...but a long window at k*=1.5 walks straight into the round-off trap, and
#    n_wave=1 is what defends against it.  With n_wave=12 the domain admits
#    k_n = 2*pi*n/L = 0.125n, so modes n=1..7 lie INSIDE the growth band and the ~1e-16
#    round-off the IC leaves in them grows at sigma ~ +0.09.  Meanwhile the k*=1.5 mode
#    we are measuring DECAYS.  By t ~ 200 the growing junk exceeds it, mode 12's
#    coefficient becomes round-off of the leader, and the fit reports the LEADER's
#    growth rate (+0.074) as though it were mode 12's.  Setting n_wave=1 makes k*=1.5
#    the gravest mode in the box -- every other admissible wavenumber is 3.0, 4.5, ...,
#    all more strongly damped -- so nothing can grow and the decay stays clean to t=600.
#    The log-fit residual is asserted too, so a curved (i.e. still-blended) fit fails
#    loudly rather than passing next to a plausible-looking number.
# --------------------------------------------------------------------------- #
print("\n1. IVP (sigma, c) vs det M = 0   [p.19 reconstruction]")
ivp_pts = {}
for kstar, n_wave, t_end in ((0.30, 2, 200.0), (1.50, 1, 600.0)):
    r = run(kstar=kstar, n_wave=n_wave, t_end=t_end)
    sig, c, resid = fit_sigma_c(r["t"], r["amp2"], kstar)
    om, _, _ = bank_mode(kstar, float(r["D"]), float(r["gamma"]), float(r["E"]))
    ivp_pts[kstar] = (sig, c)
    # resid tolerance 1e-4: over t=600 the k*=1.5 log-amplitude spans -57.7, so 1e-4 is
    # a relative curvature of 2e-6 -- while the blended-root failure this guards against
    # produced resid = 4.5, four orders of magnitude clear of the threshold.
    ok = (abs(sig - om.imag) < 1e-5 and abs(c - om.real / kstar) < 1e-4
          and resid < 1e-4)
    check(f"k*={kstar} (n_wave={n_wave})",
          ok,
          f"IVP sigma={sig:+.7f} c={c:+.7f} | det M=0 sigma={om.imag:+.7f} "
          f"c={om.real / kstar:+.7f} | log-fit resid={resid:.1e}")

# --------------------------------------------------------------------------- #
# 2. The varicose subspace decays at exactly -E, for every k*.
#    This is the MEASURED answer to "is psihat1 = psihat3 an initial condition or does
#    it hold throughout?"  It must be its OWN run: extracted from a run dominated by the
#    growing sinuous mode, the antisymmetric part falls into that mode's round-off floor
#    and the fit reads ~-0.094 instead of -0.25.
# --------------------------------------------------------------------------- #
print("\n2. varicose subspace decays at -E, and never enters zeta2'   [p.9 symmetry]")
for kstar, n_wave in ((0.30, 2), (1.50, 12)):
    r = run(kstar=kstar, n_wave=n_wave, t_end=40.0, _test_ic=ic_varicose)
    E = float(r["E"])
    sig, _, _ = fit_sigma_c(r["t"], 0.5 * (r["amp1"] - r["amp3"]), kstar)
    z2max = float(np.max(np.abs(r["psi1"] + r["psi3"] - 2 * r["psi2"])))
    check(f"k*={kstar}: decay rate",
          abs(sig + E) < 1e-5,
          f"measured {sig:+.7f}  vs  -E = {-E:+.7f}   "
          f"(|psi1+psi3-2psi2|max = {z2max:.2e}, so zeta2' is untouched)")

# --------------------------------------------------------------------------- #
# 3. Lopsided channel: the two subspaces separate.  psihat1 -> psihat3 is REACHED.
# --------------------------------------------------------------------------- #
print("\n3. lopsided channel: sinuous grows, varicose dies  =>  psihat1 = psihat3 is reached")
kstar = 0.30
r = run(kstar=kstar, n_wave=2, t_end=60.0, _test_ic=ic_lopsided)
E = float(r["E"])
om, _, _ = bank_mode(kstar, float(r["D"]), float(r["gamma"]), E)
sym = 0.5 * (r["amp1"] + r["amp3"])
anti = 0.5 * (r["amp1"] - r["amp3"])
s_sym, _, _ = fit_sigma_c(r["t"], sym, kstar, frac=0.5)
m = r["t"] <= 30.0          # short window: beyond ~t=55 anti is below sym's round-off
s_anti = float(np.polyfit(r["t"][m], np.log(np.abs(anti[m])), 1)[0])
check("symmetric part grows at sigma", abs(s_sym - om.imag) < 1e-4,
      f"measured {s_sym:+.7f}  vs det M=0 {om.imag:+.7f}")
check("antisymmetric part decays at -E", abs(s_anti + E) < 1e-4,
      f"measured {s_anti:+.7f}  vs -E = {-E:+.7f}")
r0, r1 = abs(anti[0] / sym[0]), abs(anti[-1] / sym[-1])
check("asymmetry collapses", r1 < 1e-6 * r0,
      f"|anti/sym|: {r0:.3e} at t=0  ->  {r1:.3e} at t={r['t'][-1]:.0f}")

# --------------------------------------------------------------------------- #
# 4. dt is an accuracy choice, not a stability limit (all terms linear + implicit).
# --------------------------------------------------------------------------- #
print("\n4. dt halving leaves sigma unchanged (every term is linear and implicit)")
s_ref = None
for dt in (0.04, 0.02, 0.01):
    r = run(kstar=0.30, n_wave=2, t_end=200.0, dt=dt)
    s, _, _ = fit_sigma_c(r["t"], r["amp2"], 0.30)
    if s_ref is None:
        s_ref = s
    check(f"dt={dt}", abs(s - s_ref) < 1e-6,
          f"sigma={s:+.9f}   (delta from dt=0.04: {s - s_ref:+.1e})")

# --------------------------------------------------------------------------- #
# 5. Forced steady |psihat2/psihat1| vs the p.14 box, including the p.13 D=0.1 panel.
# --------------------------------------------------------------------------- #
print("\n5. forced steady ratio vs the p.14 box  |psihat2|>|psihat1| if k*^2 < 2D")
TARGETS = [
    (0.3, 0.5, 0.0, "p.12 top / p.13 top / p.14"),
    (1.5, 0.5, 0.0, "p.12 bottom"),
    (0.3, 0.1, 0.0, "p.13 bottom (D highlighted)"),
    (0.3, 0.5, 0.1, "p.15 bottom / p.16 / p.18 / p.19"),
]
for kstar, D, gamma, slide in TARGETS:
    ratio = forced_ratio(kstar, D, gamma)
    box = 2.0 / (2.0 + kstar**2 - 2.0 * D)     # the p.14 identity -- gamma = 0 only
    if gamma == 0.0:
        check(f"k*={kstar} D={D} gamma=0: equals the printed identity",
              abs(abs(ratio) - box) < 1e-12,
              f"|psihat2/psihat1| = {abs(ratio):.4f} = 2/(2+k*^2-2D) = {box:.4f}   [{slide}]")
    else:
        print(f"         (gamma={gamma}: |ratio| = {abs(ratio):.4f}, "
              f"phase = {np.degrees(np.angle(ratio)):+.2f} deg  [{slide}])")
    check(f"k*={kstar} D={D}: criterion k*^2 < 2D",
          (abs(ratio) > 1.0) == (kstar**2 < 2.0 * D),
          f"k*^2 = {kstar**2:.2f} vs 2D = {2 * D:.2f}  ->  "
          f"{'amplified' if abs(ratio) > 1 else 'suppressed'}  (|ratio| = {abs(ratio):.4f})")

# --------------------------------------------------------------------------- #
# 6. p.16 momentum flux.  Only the SIGN is asserted, because the sign is all the deck
#    asserts.  The coefficients are reported side by side: the deck prints (D gamma/2b)
#    where the steady algebra of the p.10 balance gives (gamma/2D), and they differ.
# --------------------------------------------------------------------------- #
print("\n6. p.16 momentum flux: sign asserted, coefficients only reported")
for kstar, gamma in ((0.30, 0.1), (0.30, 0.0)):
    r = run(kstar=kstar, n_wave=2, gamma=gamma, t_end=60.0, _test_ic=ic_eigen)
    lhs, rhs, z2sq = momentum_flux(r)
    D, b = float(r["D"]), float(r["b"])
    # Report COEFFICIENTS, not raw magnitudes: the mode grows by ~1e2 over the run, so
    # the raw flux is meaningless on its own.  coeff := -<v2'zeta2'> / <zeta2'^2>, which
    # is the quantity the deck's right-hand side actually names.
    coeff = lhs[-1] / z2sq[-1]
    check(f"k*={kstar} gamma={gamma}: -<v2' zeta2'> > 0",
          bool(np.all(lhs[len(lhs) // 2:] > 0)),
          f"measured coeff = {coeff:+.5f} | deck p.16 (D*gamma/2b) = "
          f"{D * gamma / (2 * b):+.5f} | steady algebra (gamma/2D) = "
          f"{gamma / (2 * D):+.5f}   [only the SIGN is asserted]")
print("         NOTE gamma=0 still gives a NONZERO flux.  'Zero without friction' holds")
print("         for the STEADY forced state; a growing mode is phase-tilted by its own")
print("         growth, so it carries momentum even at gamma = 0.")

# --------------------------------------------------------------------------- #
# 7. The initial condition really IS the forced steady state.
#    At t=0 -- and only at t=0 -- the p.16 steady balance must hold exactly, because the
#    IC is constructed as the solution of that balance.  Any later time has left it.
#    This is the cheapest possible check that the problem is posed the way river.pdf
#    poses it (banks given, interior slaved), and it would have caught the earlier
#    "kick the centreline" inversion immediately.
# --------------------------------------------------------------------------- #
print("\n7. the IC satisfies the p.16 steady identity exactly at t=0")
from pp_lib import load_run, v2_of, zeta2_of  # noqa: E402

for tag in ("k0.30", "k1.50"):
    try:
        r = load_run(tag)
    except SystemExit:
        print(f"         (skipped {tag}: run noboru_model.py first)")
        continue
    D, gamma = float(r["D"]), float(r["gamma"])
    q = (-v2_of(r) * zeta2_of(r)).mean(axis=-1)
    z2sq = (zeta2_of(r) ** 2).mean(axis=-1)
    pred = (gamma / (2 * D)) * z2sq
    rel = abs(q[0] - pred[0]) / abs(pred[0])
    check(f"{tag}: <-v2 zeta2>(t=0) == (gamma/2D)<zeta2^2>",
          rel < 1e-10,
          f"measured {q[0]:+.9e}  vs steady balance {pred[0]:+.9e}  (rel diff {rel:.1e}); "
          f"by t={r['t'][-1]:.0f} it has left the balance: {q[-1]:+.3e} vs {pred[-1]:+.3e}")

# --------------------------------------------------------------------------- #
# 8. The NONDIMENSIONALISATION itself.
#    Everything else in this package works entirely in nondimensional variables, so a
#    wrong scaling would be invisible to every other check -- they would all agree with
#    each other and all be wrong together.  This one starts from ARBITRARY dimensional
#    values (deliberately not round numbers), solves the p.10 equation as printed with no
#    nondimensionalisation at all, then converts omega to omega* = omega b/(U0+Delta) and
#    compares with what the code computes from the groups k*, D, gamma, E.
# --------------------------------------------------------------------------- #
print("\n8. dimensional p.10 equation -> nondimensional groups")
U0, Delta, b_, H, Cf, eps = 0.83, 1.17, 47.3, 2.9, 0.0061, 82.0
k_dim = 0.00634
Uc = U0 + Delta
A_bank = eps * Cf * U0 / b_               # dimensional bank rate  [1/s]
fric = Cf * Uc / H                        # dimensional friction   [1/s]
beta_dim = 2 * Delta / b_**2


def detM_dim(w):
    """det of the 2x2 system in (psihat1, psihat2), all dimensional, straight off p.9/10/19."""
    S = -1j * w + 1j * Uc * k_dim + fric
    a11 = S * (2 / b_**2)
    a12 = S * (-2 / b_**2 - k_dim**2) + beta_dim * 1j * k_dim
    a21 = -1j * w + A_bank
    a22 = -A_bank
    return a11 * a22 - a12 * a21


ws = np.array([0.0, 1.0, 2.0], dtype=complex)
w_dim = np.roots(np.polyfit(ws, np.array([detM_dim(w) for w in ws]), 2))
from pp_lib import dispersion_roots  # noqa: E402

got = np.sort_complex(w_dim * b_ / Uc)
want = np.sort_complex(dispersion_roots(k_dim * b_, Delta / Uc, Cf * b_ / H,
                                        eps * Cf * (1 - Delta / Uc)))
err = float(np.max(np.abs(got - want)))
check("dimensional equation == nondimensional code", err < 1e-10,
      f"max |domega*| = {err:.1e}   "
      f"(k*={k_dim * b_:.4f}, D={Delta / Uc:.4f}, gamma={Cf * b_ / H:.4f}, "
      f"E={eps * Cf * (1 - Delta / Uc):.4f})")

# --------------------------------------------------------------------------- #
#  Figure: the two IVP points sitting on the analytic branch
# --------------------------------------------------------------------------- #
ks = np.linspace(1e-3, 2.0, 800)
E = bank_E(CONFIG)
sig_c, c_c = growth_and_phase(ks, CONFIG["D"], CONFIG["gamma"], E)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for ax, curve, vals, lab, title in (
        (axes[0], sig_c, {k: v[0] for k, v in ivp_pts.items()}, r"$\sigma$", "growth rate"),
        (axes[1], c_c, {k: v[1] for k, v in ivp_pts.items()}, r"$c$",
         "phase speed (negative = UPSTREAM)")):
    ax.plot(ks, curve, color=COLORS["psi1"], lw=2, label=r"det $M=0$ (reconstruction)")
    ax.plot(list(vals), list(vals.values()), "o", ms=12, mfc="none",
            mec=COLORS["growth"], mew=2.5, label="Dedalus IVP")
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_xlabel(r"$k^* = kb$")
    ax.set_ylabel(lab)
    ax.set_title(title)
kb = np.sqrt(2 * CONFIG["D"])
axes[0].axvline(kb, color="0.6", ls=":", lw=1.2)
axes[0].annotate(rf"$\sqrt{{2D}}={kb:.2f}$", (kb + 0.03, 0.045), fontsize=9, color="0.45")
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle(rf"IVP lands on the dispersion relation  ($D={CONFIG['D']}$, "
             rf"$\gamma={CONFIG['gamma']}$, $E={E:.2f}$)", y=1.03)
save_fig(fig, "verify")

print("\n" + "=" * 78)
if FAILURES:
    print(f"FAILED {len(FAILURES)} check(s): " + "; ".join(FAILURES))
    raise SystemExit(1)
print("all checks passed.")
