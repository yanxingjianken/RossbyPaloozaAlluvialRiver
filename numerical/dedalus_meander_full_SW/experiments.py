#!/usr/bin/env python3
"""Parameter study: one broadband run per PHYSICAL configuration.

    micromamba run -n dedalus env OMP_NUM_THREADS=1 python experiments.py

The perturbation is always broadband (every wavelength excited at once), so a run is
never labelled by "which wavelength was perturbed" — each run already contains the
whole dispersion relation.  What distinguishes runs is the physics:

    bed H          cross_amp   (flat, or a parabolic thalweg)
    bank sinuosity Cbar_amp    (curvature amplitude of the initial meandering bank)
    bottom friction Cf
    base speed     U0, Delta   (Delta = jet excess = the cross-channel shear;
                                Delta=0 is plug flow, Delta<0 a reversed-shear wake)

Each run writes outputs/run_H<bed>_bank<sin>_Cf<cf>_U<u0>dU<Delta>.h5 carrying the
fields, the per-mode dispersion (disp_*) and the mode-classification diagnostics
(diag_*).  Every varied knob must appear in the tag or runs silently overwrite each
other, so main() refuses to start if two configurations collide.

Note on interpretation, stated once: per-mode dispersion is a true dispersion
relation only for a STRAIGHT base channel (bank sinuosity 0), where the s-Fourier
modes decouple.  At finite sinuosity the modes couple (Floquet) and the per-mode
numbers are Bloch quantities, not eigenvalues — the driver still records them, and
the `disp_converged` flag must be honoured either way.
"""
import numpy as np

import sw_sn_driver as M

# every run shares one domain so the mode grids are identical and comparable
# Ns=128, not 64.  sigma(k) is ALREADY converged at Ns=64 (it agrees with Ns=128 to
# 0.0-0.1% across the whole band, and the peak sits at k=1.80 in both), so this is not
# a correctness fix -- it is a legibility one.  The growth curve is nearly flat over
# k=1.5-2.3, so the field stays a superposition of that whole band; at Ns=64 the band
# lies at 62-96% of the grid Nyquist and renders as one-pixel stripes that read as
# numerical noise.  At Ns=128 the same physical modes sit at ~37% of Nyquist and look
# like the waves they are.
COMMON = dict(seed_type="broadband", Ls=4 * 2 * np.pi / 0.30,
              Ns=128, Nn=96, dt=0.01, t_end=120.0, A0=1e-4)

# --- the configurations to compare (edit here) ----------------------------- #
CONFIGS = [
    ("reference",            dict()),
    ("bank: straight",       dict(Cbar_amp=0.00)),
    ("bank: sinuous",        dict(Cbar_amp=0.15)),
    ("bed: parabolic H(n)",  dict(cross_amp=0.30)),
    ("friction: low",        dict(Cf=0.010)),
    ("friction: high",       dict(Cf=0.100)),
    ("U0: fast banks",       dict(U0=0.80, Delta=0.20)),
    ("jet: plug (no shear)", dict(U0=1.00, Delta=0.00)),
    ("jet: reversed shear",  dict(U0=1.00, Delta=-0.60)),
]


def main():
    cfgs = [(lab, dict(M.CONFIG, **COMMON, **over)) for lab, over in CONFIGS]

    # a knob that is varied but absent from run_tag() would make two runs share one
    # filename and the second would silently overwrite the first -- fail loudly instead
    tags = {}
    for lab, cfg in cfgs:
        tags.setdefault(M.run_tag(cfg), []).append(lab)
    clash = {t: labs for t, labs in tags.items() if len(labs) > 1}
    if clash:
        raise SystemExit("run_tag() collision -- these configurations would overwrite "
                         "each other; add the varied knob to run_tag():\n" +
                         "\n".join(f"  {t}: {labs}" for t, labs in clash.items()))

    print(f"{len(cfgs)} configurations, one broadband run each\n")
    for label, cfg in cfgs:
        print(f"== {label}  ->  {M.run_tag(cfg)}")
        M.run_ivp_SW(cfg)
    print("\ndone — now:  cd postprocessing && python 01_dispersion.py "
          "&& python 02_eulerian_momflux.py")


if __name__ == "__main__":
    main()
