#!/usr/bin/env python3
"""fig13-14: the friction-closure comparison (rebuild-v2 exploration).

The deck does not print its interior friction closure; v1 flagged the
Rayleigh-damping reading (FLAG_FRICTION) as the prime suspect for the
growth-peak discrepancy.  This script quantifies the alternative:

    rayleigh : F = -gamma zeta'                      (deck-literal, v1)
    momentum : F = curl[-(C_f/H)|u|u] linearised     (Ikeda-consistent)

Both closures are calibrated on the SAME six deck-p.8 phase-speed
intercepts (an exact E <-> 2E degeneracy: ECOEF = 0.5 vs 1.0), so the
intercepts cannot distinguish them; the growth peaks can.

Result (codified in the lib self-test): the momentum closure raises the
peaks by ~40% (deck/model 2.3-3.2 instead of 3.2-4.4) and leaves the
crossings/phases intact -- it closes about one third of the log-gap and
therefore does NOT resolve the discrepancy.  Remaining suspects: the p. 6
wavy-bank pressure drag absent from the p. 7 bank equation, and the
growth-axis normalisation (FLAG_TSCALE).
"""
import numpy as np

from vorticity_lib import (COLORS, FRICTIONS, Params, ECOEF, bank_branch,
                           growth_curve, kstar_peak, load_deck_pins,
                           set_style, save_fig)

plt = set_style()

ks = np.linspace(1e-3, 2.0, 900)
pins = load_deck_pins()
FAMS = {"D-family": ((0.3, 0.05), (0.6, 0.05), (0.9, 0.05)),
        "g-family": ((0.6, 0.03), (0.6, 0.06), (0.6, 0.09))}
CURVE_COLS = ("#2040d0", "#7b2d8b", "#e03020")

# --------------------------------------------------------------- fig13 ---- #
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
for jc, (famname, fam) in enumerate(FAMS.items()):
    axg, axc = axes[0, jc], axes[1, jc]
    fpins = [r for r in pins if r["family"] == famname]
    for (D, g), col, r in zip(fam, CURVE_COLS, fpins):
        for fr, ls in (("rayleigh", "-"), ("momentum", "--")):
            p = Params(D=D, gamma=g, friction=fr)
            sig, c = growth_curve(ks, p)
            lab = rf"$D={D}$, $\gamma={g}$" if fr == "rayleigh" else None
            axg.plot(ks, sig, ls, color=col, lw=1.8, label=lab)
            axc.plot(ks, c, ls, color=col, lw=1.8)
        axg.plot([r["kstar_peak"]], [r["sigma_peak"]], "v", color=col, ms=8,
                 mec="k", mew=0.6)
        axc.plot([0.02], [r["c0"]], "s", color=col, ms=7, mec="k", mew=0.6)
    axg.axhline(0, color="k", lw=0.8)
    axc.axhline(0, color="k", lw=0.8)
    axg.set_title(f"growth rate -- {famname}")
    axc.set_title(f"phase speed -- {famname}")
    axg.set_ylim(-0.6, 0.6)
    axc.set_ylim(-5, 1)
    axc.set_xlabel(r"wavenumber $k^*=kb$")
    axg.legend(fontsize=9, loc="upper right")
axes[0, 0].set_ylabel(r"$\sigma^*$")
axes[1, 0].set_ylabel(r"$c^*$")
fig.suptitle("friction closures: rayleigh (solid, ECOEF=0.5) vs momentum "
             "(dashed, ECOEF=1.0), each calibrated on the same deck "
             "intercepts; markers = deck pins", y=0.995, fontsize=12)
save_fig(fig, "fig13_closure_overlay")

# --------------------------------------------------------------- fig14 ---- #
labels, ratios = [], {fr: [] for fr in FRICTIONS}
print(f"{'family':>10} {'D':>4} {'gam':>5} | "
      f"{'spk(ray)':>9} {'spk(mom)':>9} {'deck':>6} | "
      f"{'ratio(ray)':>10} {'ratio(mom)':>10}")
for r in pins:
    labels.append(f"D={r['D']}\n" + rf"$\gamma$={r['gamma']}")
    row = {}
    for fr in FRICTIONS:
        p = Params(D=r["D"], gamma=r["gamma"], friction=fr)
        _, spk, _ = kstar_peak(p)
        ratios[fr].append(r["sigma_peak"] / spk)
        row[fr] = spk
    print(f"{r['family']:>10} {r['D']:>4} {r['gamma']:>5} | "
          f"{row['rayleigh']:9.3f} {row['momentum']:9.3f} "
          f"{r['sigma_peak']:6.2f} | {ratios['rayleigh'][-1]:10.2f} "
          f"{ratios['momentum'][-1]:10.2f}")

xx = np.arange(6)
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.bar(xx - 0.18, ratios["rayleigh"], 0.36, color=COLORS["growth"],
       label="rayleigh (deck-literal)")
ax.bar(xx + 0.18, ratios["momentum"], 0.36, color=COLORS["momentum"],
       label=r"momentum (curl of $-C_f|u|\,u/H$)")
ax.axhline(1, color="k", lw=1.2)
ax.text(5.45, 1.05, "deck", fontsize=9)
ax.set_xticks(xx, labels, fontsize=8)
ax.set_ylabel(r"deck $\sigma_{\rm pk}$ / model $\sigma_{\rm pk}$")
ax.set_title("the peak discrepancy under the two friction closures\n"
             "(both fit the six phase intercepts equally well: "
             r"$E\leftrightarrow 2E$ degeneracy)")
ax.legend(fontsize=9)
save_fig(fig, "fig14_closure_ratio_bars")

print("07_friction_closures: done.")
