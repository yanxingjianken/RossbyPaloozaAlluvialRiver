#!/usr/bin/env python3
"""fig01-02: how the BED modulates the dispersion (the new-physics headline).

Overlays sigma(k*), phase speed c*, and GROUP velocity c_g of the bank branch
for a flat bed vs cross-channel depth bumps (deeper thalweg).  The topographic-
shear beta = d/dy(zetabar/Hbar) shifts the growth band and the group-velocity
sign-flip point k_g (where wave momentum switches from upstream to downstream).

Run:  micromamba run -n dedalus env OMP_NUM_THREADS=1 python 01_dispersion.py
Reads nothing (GEP is cheap); writes figures/fig01_dispersion_bed.png,
figures/fig02_groupflip_vs_bed.png.
"""
import numpy as np

import pp_lib as PP

plt = PP.CL.set_style()
MD = PP.MD

AMPS = (0.0, 0.3, 0.6)                 # flat, moderate, strong thalweg
COLS = ("#2040d0", "#7b2d8b", "#e03020")
fr = "rayleigh"

# --------------------------------------------------------------- fig01 ---- #
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
kg_of_amp = []
for amp, col in zip(AMPS, COLS):
    cfg = dict(MD.CONFIG, cross_amp=amp, friction=fr)
    ks, sig, cph, cg = PP.dispersion(cfg)
    lab = rf"$a_H={amp}$"
    axes[0].plot(ks, sig, color=col, lw=2, label=lab)
    axes[1].plot(ks, cph, color=col, lw=2, label=lab)
    axes[2].plot(ks, cg, color=col, lw=2, label=lab)
    # group-velocity sign flip (upstream->downstream)
    s = np.where(np.diff(np.sign(cg)) != 0)[0]
    kg = ks[s[0]] if len(s) else np.nan
    kg_of_amp.append(kg)
    if np.isfinite(kg):
        axes[2].axvline(kg, color=col, lw=0.8, ls=":")

for ax, ttl, yl in zip(
        axes,
        (r"growth rate $\sigma^*$", r"phase speed $c^*$ (crests)",
         r"group velocity $c_g$ (momentum)"),
        (r"$\sigma^*$", r"$c^*$", r"$c_g$")):
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$k^*=kb$")
    ax.set_ylabel(yl)
    ax.set_title(ttl)
    ax.legend(fontsize=9)
axes[0].set_ylim(-0.2, 0.6)
axes[2].set_ylim(-0.45, 0.2)
fig.suptitle(r"Bed modulates the dispersion: deeper thalweg ($a_H$) shifts the "
             r"growth band and the group-velocity flip  ($H(y)=1+a_H(1-y^2)$, "
             rf"{fr})", y=1.0, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{PP.FIG_DIR}/fig01_dispersion_bed.png", bbox_inches="tight", dpi=150)
print(f"  wrote figures/fig01_dispersion_bed.png")
plt.close(fig)

# --------------------------------------------------------------- fig02 ---- #
amps = np.linspace(0.0, 0.8, 9)
kgs = []
for amp in amps:
    cfg = dict(MD.CONFIG, cross_amp=float(amp), friction=fr)
    ks, sig, cph, cg = PP.dispersion(cfg, ks=np.linspace(0.05, 1.0, 40))
    s = np.where(np.diff(np.sign(cg)) != 0)[0]
    kgs.append(ks[s[0]] if len(s) else np.nan)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(amps, kgs, "o-", color=PP.COLORS["upstream"], lw=2)
ax.set_xlabel(r"thalweg depth bump $a_H$   ($H=1+a_H(1-y^2)$)")
ax.set_ylabel(r"group-velocity flip $k_g$ ($\lambda/2b=\pi/k_g$)")
ax.set_title("deeper thalweg moves the upstream/downstream momentum boundary")
fig.tight_layout()
fig.savefig(f"{PP.FIG_DIR}/fig02_groupflip_vs_bed.png", bbox_inches="tight", dpi=150)
plt.close(fig)

print("01_dispersion: done.")
print(f"  group-velocity flip k_g: a_H={AMPS} -> "
      + ", ".join(f"{k:.3f}" for k in kg_of_amp))
