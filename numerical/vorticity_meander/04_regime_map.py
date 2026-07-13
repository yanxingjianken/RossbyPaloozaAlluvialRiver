#!/usr/bin/env python3
"""fig08-09: regime maps over (D, gamma) -- rayleigh closure.

fig08: peak growth rate sigma_pk and its wavenumber k*_pk.
fig09: the k*->0 upstream phase speed c0 = -E D/gamma (log-magnitude map).
"""
import numpy as np

from vorticity_lib import Params, kstar_peak, set_style, save_fig, ECOEF

plt = set_style()

Ds = np.linspace(0.1, 0.9, 33)
gs = np.linspace(0.02, 0.12, 21)
SPK = np.zeros((gs.size, Ds.size))
KPK = np.zeros_like(SPK)
for i, g in enumerate(gs):
    for j, d in enumerate(Ds):
        kpk, spk, _ = kstar_peak(Params(D=float(d), gamma=float(g)),
                                 ks=np.linspace(1e-3, 2.0, 700))
        SPK[i, j] = spk
        KPK[i, j] = kpk

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, Z, ttl in ((axes[0], SPK, r"$\sigma_{\rm pk}$"),
                   (axes[1], KPK, r"$k^*_{\rm pk}$")):
    pc = ax.pcolormesh(Ds, gs, Z, cmap="viridis", shading="auto")
    fig.colorbar(pc, ax=ax, label=ttl)
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(ttl + r" over $(D,\gamma)$  [rayleigh, ECOEF$=0.5$]")
axes[0].plot([0.3, 0.6, 0.9], [0.05] * 3, "wv", ms=8, mec="k")
axes[0].plot([0.6] * 3, [0.03, 0.06, 0.09], "ws", ms=7, mec="k")
save_fig(fig, "fig08_regime_map_growth")

C0 = -(ECOEF["rayleigh"] * (1 - Ds[None, :])) * Ds[None, :] / gs[:, None]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
pc = ax.pcolormesh(Ds, gs, -C0, cmap="magma", shading="auto")
fig.colorbar(pc, ax=ax, label=r"$|c_0| = E\,D/\gamma$ (upstream)")
ax.set_xlabel(r"$D$")
ax.set_ylabel(r"$\gamma$")
ax.set_title(r"long-wave upstream speed $c_0=-E D/\gamma$ "
             "(deck p. 7 headline)")
save_fig(fig, "fig09_upstream_speed_map")

print("04_regime_map: done.")
