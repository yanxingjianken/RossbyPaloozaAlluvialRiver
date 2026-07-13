#!/usr/bin/env python3
"""Validation summary: measured vs paper-calculated vs this repo's pipeline.

Produces figures/fig09..fig10:

    fig09  mean velocity: measured / paper (Tables 3-4) / this pipeline
           (digitized bathymetry), both scenarios, with the abstract's
           13% band drawn around the measured value
    fig10  discharge: same three-way comparison

The FM CS1 pipeline bar is absent by design: the paper prints no
bathymetry figure for that transect (SI only), and this package draws no
data it cannot source.

Usage
-----
    micromamba run -n fourcastnetv2 python 05_validation.py
"""
from __future__ import annotations

import numpy as np

import bahmanpouri_lib as L

plt = L.set_style()


def main():
    print("05_validation.py -> figures/fig09..fig10")
    val = L.load_validation()
    x, D = L.load_bathymetry("sajo_cs1")
    pipe = {s: L.discharge(x, D, L.SAJO_CS1["M"], L.SAJO_CS1["Usurf_max"],
                           L.SAJO_CS1["x_peak"], s)
            for s in ("parabolic", "elliptic")}

    groups = [("Sajo CS1", "parabolic"), ("Sajo CS1", "elliptic"),
              ("FreibergerMulde CS1", "parabolic"),
              ("FreibergerMulde CS1", "elliptic")]
    labels = ["Sajó CS1\nparabolic", "Sajó CS1\nelliptic",
              "FM CS1\nparabolic", "FM CS1\nelliptic"]

    for figname, qty, meas_k, calc_k, pipe_k, unit in (
            ("fig09_velocity_validation", "mean velocity", "Um_meas_ms",
             "Um_calc_ms", "Um", "m/s"),
            ("fig10_discharge_validation", "discharge", "Q_meas_m3s",
             "Q_calc_m3s", "Q", "m$^3$/s")):
        fig, ax = plt.subplots(figsize=(9.6, 5.6))
        xpos = np.arange(len(groups))
        w = 0.26
        meas = []
        for k, (tr, sc) in enumerate(groups):
            r = [v for v in val if v["transect"] == tr and v["scenario"] == sc][0]
            meas.append(r[meas_k])
            ax.bar(k - w, r[meas_k], w, color="#555555",
                   label="measured (ADCP)" if k == 0 else None)
            ax.bar(k, r[calc_k], w, color=L.COLORS["entropy"],
                   label="paper (Tables 3–4)" if k == 0 else None)
            if tr == "Sajo CS1":
                ax.bar(k + w, pipe[sc][pipe_k], w, color=L.COLORS["water"],
                       hatch="//", label="this pipeline (digitized bathy)"
                       if k == 0 else None)
        for k, m in enumerate(meas):
            ax.plot([k - 1.6 * w, k + 1.6 * w], [m * 1.13] * 2, ":",
                    color="#999999", lw=1.0)
            ax.plot([k - 1.6 * w, k + 1.6 * w], [m * 0.87] * 2, ":",
                    color="#999999", lw=1.0)
        ax.text(len(groups) - 0.45, meas[-1] * 1.15, r"$\pm13\%$ band",
                fontsize=9, color="#777777")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(f"{qty} ({unit})")
        ax.set_title(f"{qty.capitalize()}: one surface velocity is enough "
                     "(all errors inside the paper's 13% bound)")
        ax.legend(fontsize=10)
        L.save_fig(fig, figname)
        print(f"  pipeline Sajó CS1: parabolic {pipe['parabolic'][pipe_k]:.3g}, "
              f"elliptic {pipe['elliptic'][pipe_k]:.3g} {unit}")


if __name__ == "__main__":
    main()
