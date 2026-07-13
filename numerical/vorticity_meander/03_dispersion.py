#!/usr/bin/env python3
"""fig06-07: the deck p. 8 dispersion figure, regenerated and pinned.

fig06: four-panel replica (growth rate / phase speed for the D family and
       the gamma family) from the deck-literal rayleigh closure with the
       single calibrated ECOEF = 0.5, with the re-digitized deck pins
       overlaid -- phases and zero crossings match; peak heights do not
       (the codified discrepancy).
fig07: per-curve agreement summary (c0, kzero, sigma_peak ratio).
"""
import numpy as np

from vorticity_lib import (COLORS, DECK_D_FAMILY, DECK_G_FAMILY, Params,
                           bank_branch, growth_curve, kstar_peak,
                           load_deck_pins, set_style, save_fig)

plt = set_style()

ks = np.linspace(1e-3, 2.0, 900)
pins = load_deck_pins()
FAMS = {"D-family": DECK_D_FAMILY, "g-family": DECK_G_FAMILY}
CURVE_COLS = ("#2040d0", "#7b2d8b", "#e03020")   # deck blue/purple/red

# --------------------------------------------------------------- fig06 ---- #
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
for jc, (famname, fam) in enumerate(FAMS.items()):
    axg, axc = axes[0, jc], axes[1, jc]
    fpins = [r for r in pins if r["family"] == famname]
    for p, col, r in zip(fam, CURVE_COLS, fpins):
        sig, c = growth_curve(ks, p)
        lab = rf"$D={p.D}$, $\gamma={p.gamma}$"
        axg.plot(ks, sig, color=col, lw=1.9, label=lab)
        axc.plot(ks, c, color=col, lw=1.9, label=lab)
        axg.plot([r["kstar_peak"]], [r["sigma_peak"]], "v",
                 color=col, ms=8, mec="k", mew=0.6)
        axg.plot([r["kzero"]], [0.0], "o", color=col, ms=7, mec="k", mew=0.6)
        axc.plot([0.02], [r["c0"]], "s", color=col, ms=7, mec="k", mew=0.6)
    axg.axhline(0, color="k", lw=0.8)
    axc.axhline(0, color="k", lw=0.8)
    axg.set_title(f"growth rate -- {famname}")
    axc.set_title(f"phase speed -- {famname}")
    axg.set_ylim(-0.6, 0.6)
    axc.set_ylim(-5, 1)
    axc.set_xlabel(r"wavenumber $k^*=kb$")
    axg.legend(fontsize=9, loc="upper right")
for k_, lab in ((0.25, r"$\lambda/2b=4\pi$"), (0.5, r"$2\pi$"), (1.0, r"$\pi$")):
    for ax in axes.ravel():
        ax.axvline(k_, color="0.85", lw=0.8, zorder=0)
axes[0, 0].set_ylabel(r"$\sigma^*$")
axes[1, 0].set_ylabel(r"$c^*$")
fig.suptitle("deck p. 8 regenerated (rayleigh closure, single ECOEF = 0.5) "
             "with re-digitized pins:  markers = deck  "
             r"($\blacktriangledown$ peak, $\bullet$ zero, $\blacksquare$ intercept)",
             y=0.995, fontsize=12)
save_fig(fig, "fig06_deck_p8_replica")

# --------------------------------------------------------------- fig07 ---- #
labels, c0m, c0d, kzm, kzd, rat = [], [], [], [], [], []
for r in pins:
    p = Params(D=r["D"], gamma=r["gamma"])
    kpk, spk, kz = kstar_peak(p)
    om = bank_branch([1e-3], p.D, p.gamma, p.E)[0]
    labels.append(f"D={r['D']}\n" + rf"$\gamma$={r['gamma']}")
    c0m.append(om.real / 1e-3)
    c0d.append(r["c0"])
    kzm.append(kz)
    kzd.append(r["kzero"])
    rat.append(r["sigma_peak"] / spk)

xx = np.arange(6)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
for ax, (m, d, ttl, ylab) in zip(
        axes[:2],
        ((c0m, c0d, "phase-speed intercept $c_0$", "$c_0$"),
         (kzm, kzd, "growth zero crossing $k^*_0$", "$k^*_0$"))):
    ax.bar(xx - 0.18, m, 0.36, color=COLORS["growth"], label="model")
    ax.bar(xx + 0.18, d, 0.36, color=COLORS["deckpin"], alpha=0.65, label="deck")
    ax.set_xticks(xx, labels, fontsize=8)
    ax.set_title(ttl + ": match")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=9)
ax = axes[2]
ax.bar(xx, rat, 0.5, color=COLORS["erosion"])
ax.axhline(1, color="k", lw=1)
ax.set_xticks(xx, labels, fontsize=8)
ax.set_ylabel("deck / model")
ax.set_title(r"$\sigma_{\rm peak}$: the codified discrepancy")
for i, v in enumerate(rat):
    ax.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=9)
save_fig(fig, "fig07_agreement_bars")

print("03_dispersion: done.")
