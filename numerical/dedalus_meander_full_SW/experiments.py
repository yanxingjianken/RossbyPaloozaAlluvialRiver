#!/usr/bin/env python3
"""Parameter study: one run per PHYSICAL configuration.

    micromamba run -n dedalus env OMP_NUM_THREADS=1 python experiments.py
    cd postprocessing && python analysis.py     # then derive the diagnostics

Every run is perturbed the same way — a single drop of ink released upstream (a
localised bump on the centreline).  Being localised it is automatically broadband, so
one run still contains the whole dispersion relation, and a run is never labelled by
"which wavelength was perturbed".  What distinguishes runs is the physics:

    bed H          cross_amp   (flat, or a parabolic thalweg)
    bank sinuosity Cbar_amp    (curvature amplitude of the initial meandering bank)
    bottom friction Cf
    base speed     U0, Delta   (Delta = jet excess = the cross-channel shear;
                                Delta=0 is plug flow, Delta<0 a reversed-shear wake)

Each run writes outputs/run_H<bed>_bank<sin>_Cf<cf>_U<u0>dU<Delta>.h5 carrying the raw
fields only; postprocessing/analysis.py then derives disp_* and diag_* from them.
Every varied knob must appear in the tag or runs silently overwrite each other, so
main() refuses to start if two configurations collide.

Note on interpretation, stated once: per-mode dispersion is a true dispersion
relation only for a STRAIGHT base channel (bank sinuosity 0), where the s-Fourier
modes decouple.  At finite sinuosity the modes couple (Floquet) and the per-mode
numbers are Bloch quantities, not eigenvalues — the driver still records them, and
the `disp_converged` flag must be honoured either way.
"""
import os
import subprocess
import sys
import time

import numpy as np

import sw_meander as M

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# every run shares one domain so the mode grids are identical and comparable
# Ns=128, not 64.  sigma(k) is ALREADY converged at Ns=64 (it agrees with Ns=128 to
# 0.0-0.1% across the whole band, and the peak sits at k=1.80 in both), so this is not
# a correctness fix -- it is a legibility one.  The growth curve is nearly flat over
# k=1.5-2.3, so the field stays a superposition of that whole band; at Ns=64 the band
# lies at 62-96% of the grid Nyquist and renders as one-pixel stripes that read as
# numerical noise.  At Ns=128 the same physical modes sit at ~37% of Nyquist and look
# like the waves they are.
COMMON = dict(Ls=4 * 2 * np.pi / 0.30, Ns=128, Nn=96, dt=0.01, t_end=120.0, A0=1e-4)

# --- the configurations to compare (edit here) ----------------------------- #
CONFIGS = [
    ("reference",            dict()),
    ("bank: straight",       dict(Cbar_amp=0.00)),
    ("bank: sinuous",        dict(Cbar_amp=0.15)),
    ("bed: parabolic H(n)",  dict(cross_amp=0.30)),
    ("friction: low",        dict(Cf=0.010)),
    ("friction: high",       dict(Cf=0.100)),
    ("U0: fast banks",       dict(U0=0.80, Delta=0.20)),
    # --- CONFOUNDED pair, kept only as a cautionary record -------------------- #
    # These vary Delta but ALSO raise U0 to 1.0.  U0 is the speed at the BANK, and the
    # erosion law reads u_s there, so they conflate "removed the channel-beta" with
    # "sped up the bank" -- and the confound inverts the answer (they appear to grow
    # FASTER without shear; the matched controls below show the opposite).
    ("jet: plug, U0 raised (CONFOUNDED)",     dict(U0=1.00, Delta=0.00)),
    ("jet: reversed, U0 raised (CONFOUNDED)", dict(U0=1.00, Delta=-0.60)),
    # --- the DECISIVE controls: bank speed held at the reference value --------- #
    # Ubar_s(n)=U0+Delta(1-n^2/b^2) has only two knobs, so removing Delta MUST change
    # either the bank speed or the centre speed -- there is no perfectly clean control.
    # Since the erosion is driven at the bank, holding U0 fixed is the comparison that
    # isolates the channel-beta, and it is the one to quote.
    ("CONTROL plug, matched bank speed",     dict(U0=0.40, Delta=0.00)),
    ("CONTROL reversed, matched bank speed", dict(U0=0.40, Delta=-0.30)),
]


def _configs():
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
    return cfgs


def main():
    """Run every configuration CONCURRENTLY, one OS process each.

    The configurations are completely independent, so this is embarrassingly parallel
    and the whole study costs about as much wall-clock as its slowest single run.

    Why processes and not threads/MPI.  Each run is a small problem (Ns x Nn ~ 128x96),
    and Dedalus solves it as Ns independent banded systems in n -- there is very little
    for BLAS threads to bite on, and MPI domain decomposition at this size would spend
    more time communicating than computing.  Task-level parallelism has no
    communication at all, so it is both the simplest and the fastest option here.
    Each worker is pinned to ONE thread (OMP_NUM_THREADS=1) so N runs use exactly N
    cores and do not oversubscribe: this is a SHARED node.
    """
    cfgs = _configs()
    n_par = min(int(os.environ.get("NPAR", len(cfgs))), len(cfgs))
    print(f"{len(cfgs)} configurations, {n_par} at a time, 1 core each\n")

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    procs, t0 = {}, time.time()
    pending = list(range(len(cfgs)))
    logs = os.path.join(HERE, "outputs", "logs")
    os.makedirs(logs, exist_ok=True)

    while pending or procs:
        while pending and len(procs) < n_par:
            i = pending.pop(0)
            lf = open(os.path.join(logs, f"{i:02d}_{M.run_tag(cfgs[i][1])}.log"), "w")
            procs[subprocess.Popen([sys.executable, __file__, "--only", str(i)],
                                   stdout=lf, stderr=subprocess.STDOUT, env=env,
                                   cwd=HERE)] = (i, lf)
            print(f"  [launch] {cfgs[i][0]}")
        done = [p for p in procs if p.poll() is not None]
        for p in done:
            i, lf = procs.pop(p)
            lf.close()
            print(f"  [{'ok  ' if p.returncode == 0 else 'FAIL'}] {cfgs[i][0]}"
                  f"   ({time.time()-t0:.0f}s elapsed)")
        if procs and not done:
            time.sleep(2.0)

    print(f"\nall {len(cfgs)} runs done in {time.time()-t0:.0f}s "
          f"(logs in outputs/logs/) — now:  cd postprocessing && python analysis.py "
          f"&& python 01_dispersion.py && python 02_eulerian_momflux.py")


if __name__ == "__main__":
    if "--only" in sys.argv:                       # one worker, one configuration
        i = int(sys.argv[sys.argv.index("--only") + 1])
        label, cfg = _configs()[i]
        print(f"== {label}  ->  {M.run_tag(cfg)}", flush=True)
        M.run(cfg)
    else:
        main()
