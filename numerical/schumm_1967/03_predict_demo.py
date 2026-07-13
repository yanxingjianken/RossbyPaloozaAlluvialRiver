#!/usr/bin/env python3
"""Leave-one-out prediction demo for the Schumm (1967) regressions.

For every one of the 36 sections, refit Eq.-(1)-form on the other 35 and
predict the held-out wavelength: an honest out-of-sample statement of how
well the two-variable power law transfers.  Prints a per-river table and
produces:

    fig06  LOO predicted vs observed wavelength, class-coded, +-2 SE band

Usage
-----
    micromamba run -n fourcastnetv2 python 03_predict_demo.py
"""
from __future__ import annotations

import numpy as np

import schumm_lib as L

plt = L.set_style()

D = L.load_sections()


def loo_predictions():
    n = len(D["lam"])
    pred = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        f = L.fit_power_law(D["Qm"][keep], D["M"][keep], D["lam"][keep])
        pred[i] = f["coef"] * D["Qm"][i] ** f["expQ"] * D["M"][i] ** f["expM"]
    return pred


def main():
    print("03_predict_demo.py -> figures/fig06 + table")
    pred = loo_predictions()
    err = np.abs(np.log10(pred) - np.log10(D["lam"]))
    se = L.PUBLISHED["eq1"]["see_log"]

    print(f"\n{'river':<44}{'class':<11}{'obs λ':>8}{'LOO λ':>8}{'log err':>9}")
    print("-" * 80)
    order = np.argsort(-err)
    for i in order:
        flag = "  *" if err[i] > 2 * se else ""
        print(f"{D['river'][i]:<44}{D['cls'][i]:<11}"
              f"{D['lam'][i]:>8.0f}{pred[i]:>8.0f}{err[i]:>9.2f}{flag}")
    print(f"\nwithin +-2 SE (+-{2*se:.2f} log units): "
          f"{100*np.mean(err < 2*se):.0f}%   (* = outside)")

    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    g = np.logspace(2.3, 4.6, 50)
    ax.fill_between(g, g / 10**(2 * se), g * 10**(2 * se),
                    color=L.COLORS["band"], alpha=0.55, zorder=1,
                    label=r"$\pm 2$ SE")
    ax.loglog(g, g, "-", color=L.COLORS["fit"], lw=1.5, zorder=2)
    for c in ("bedload", "mixed", "suspended"):
        m = D["cls"] == c
        ax.loglog(pred[m], D["lam"][m], "o", color=L.COLORS[c], ms=7.5,
                  mec="white", mew=0.6, zorder=3, label=c)
    ax.set_xlabel("leave-one-out predicted wavelength (ft)")
    ax.set_ylabel("observed wavelength (ft)")
    ax.set_title("Out-of-sample skill of the two-variable power law")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(200, 4e4)
    ax.set_ylim(200, 4e4)
    L.save_fig(fig, "fig06_loo_prediction")


if __name__ == "__main__":
    main()
