#!/usr/bin/env python3
"""Growth and migration diagnostic — the plot that answers the two questions:

  * does the meander AMPLIFY or DECAY?   -> log|A(t)|  (slope = growth rate)
  * does the bend phase move UPSTREAM or DOWNSTREAM?  -> crest_x(t)  (slope = c0)

A(t) is the complex Fourier amplitude of the centreline at the fundamental
wavenumber, demodulated over the clean meander reach (meander_thetis.bank_mode).
This is only meaningful for the long steady-solver (family B) runs, which reach
the morphological timescale; the short CrankNicolson runs barely move.

Both runs (m=4, m=8) are overlaid, with the Ikeda A=0 linear-theory growth rate
and celerity drawn as reference slopes.

    python postprocessing/03_growth_migration.py
"""
from __future__ import annotations

import glob
import re
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_lib as pp  # noqa: E402
import geometry as geo  # noqa: E402

plt = pp.set_style()
sys.path.insert(0, os.path.join(pp.HERE, "..", "ikeda_1981"))


def load_series(tag):
    """(t, |A|, arg A unwrapped, crest_x) from the per-run summary npz."""
    f = os.path.join(pp.OUT_DIR, f"run_{tag}.npz")
    if not os.path.exists(f):
        return None
    D = np.load(f, allow_pickle=True)
    if "A" not in D.files or D["A"].size == 0:
        return None
    A = D["A"]
    return dict(t=D["t"], absA=np.abs(A), phase=np.unwrap(np.angle(A)),
               crest=D["crest"], k=float(D["k_fundamental"]))


def ikeda_rates(m):
    """Ikeda A=0 linear growth rate alpha0 and celerity c0 at wavenumber m,
    in DIMENSIONAL units (1/s and m/s), from the verified library."""
    from ikeda_lib import growth_rate, celerity  # noqa: E402
    d = geo.build_design(geo.Config(n_wave=m))
    k = 2.0 * np.pi * d.cfg.H_ref / (d.L_m / m)      # Ikeda-normalised wavenumber
    U = geo.width_mean(geo.base_velocity, d)
    # Ikeda time is t~ = t U0/H0; convert per-tilde-t rates to per second.
    a0 = growth_rate(k, Cf=d.cfg.Cf, A=d.cfg.A_ikeda, F=d.cfg.F_ref)   # per t~
    c0 = celerity(k, Cf=d.cfg.Cf, A=d.cfg.A_ikeda, F=d.cfg.F_ref)      # dimensionless
    tconv = U / d.cfg.H_ref                            # 1/tilde-t -> 1/s
    return a0 * tconv, c0 * U                           # alpha0 [1/s], c0 [m/s]


def main():
    if len(sys.argv) > 1 and os.path.isdir(os.path.join(pp.HERE, "experiments", sys.argv[1])):
        pp.set_case(sys.argv[1])
    print("=" * 74)
    print(f"03_growth_migration.py -- amplify/decay and up/downstream [{pp.CASE}]")
    print("=" * 74)
    ms = sorted(int(mm.group(1)) for f in glob.glob(os.path.join(pp.OUT_DIR, "run_m*.npz"))
                if (mm := re.match(r"run_m(\d+)\.npz$", os.path.basename(f))))
    series = {m: load_series(f"m{m}") for m in ms}
    if all(v is None for v in series.values()):
        raise SystemExit("no A(t) data -- run meander_thetis.py with flow_solver='steady'")

    fig, (axg, axm) = plt.subplots(1, 2, figsize=(14.0, 5.4))
    palette = ["#1f6fb4", "#c0392b", "#2ca02c", "#9467bd", "#e67e22"]
    colours = {m: palette[i % len(palette)] for i, m in enumerate(ms)}

    for m, S in series.items():
        if S is None:
            continue
        yr = S["t"] / (365.25 * 86400.0)                 # physical years
        # --- growth ---
        axg.plot(yr, S["absA"] / S["absA"][0], "-o", ms=3, color=colours[m],
                 label=f"m={m}")
        # --- migration: crest position, referenced to its start ---
        axm.plot(yr, S["crest"] - S["crest"][0], "-o", ms=3, color=colours[m],
                 label=f"m={m}")
        # measured mean rates
        a_meas = np.polyfit(S["t"], np.log(S["absA"]), 1)[0]
        c_meas = np.polyfit(S["t"], S["crest"], 1)[0]
        a_ik, c_ik = ikeda_rates(m)
        print(f"  m={m}: measured  growth {a_meas:+.3e} /s   celerity {c_meas:+.3e} m/s")
        print(f"        Ikeda A=0 growth {a_ik:+.3e} /s   celerity {c_ik:+.3e} m/s")
        print(f"        -> {'AMPLIFY' if a_meas > 0 else 'DECAY'}, "
              f"{'DOWNSTREAM' if c_meas > 0 else 'UPSTREAM'} migration")

    axg.axhline(1.0, color="0.6", lw=0.8, ls=":")
    axg.set_yscale("log")
    axg.set_xlabel("physical time [years]")
    axg.set_ylabel(r"$|A(t)| / |A_0|$   (fundamental amplitude)")
    axg.set_title("amplify vs decay")
    axg.legend()
    axg.text(0.03, 0.06, "above 1 = amplify\nbelow 1 = decay",
             transform=axg.transAxes, fontsize=9, color="0.4")

    axm.axhline(0.0, color="0.6", lw=0.8, ls=":")
    axm.set_xlabel("physical time [years]")
    axm.set_ylabel(r"crest displacement  $x_{crest}(t) - x_{crest}(0)$  [m]")
    axm.set_title("bend phase migration")
    axm.legend()
    axm.text(0.03, 0.92, "up = downstream\ndown = upstream",
             transform=axm.transAxes, fontsize=9, color="0.4", va="top")

    label = {"A0_incised": "A=0 incised", "A2p89_alluvial": "A=2.89 alluvial"}.get(pp.CASE, pp.CASE)
    fig.suptitle(f"Steady-solver (family B) morphodynamics: {label} case, "
                 "C$_f$=0.05, F=0.30", y=1.00, fontsize=13)
    pp.save_fig(fig, "growth_migration")


if __name__ == "__main__":
    main()
