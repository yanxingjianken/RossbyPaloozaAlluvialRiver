#!/usr/bin/env python3
"""Recompute the diag_* attrs of existing outputs/run_*.h5 IN PLACE.

    micromamba run -n dedalus env OMP_NUM_THREADS=1 python recompute_diagnostics.py

Why this exists: classify_mode() originally took d/ds and d/dn with np.gradient.
The divergence of a nearly-balanced flow is the small residual of cancelling two
O(k|u'|) terms, so finite-differencing a Chebyshev grid measured the DIFFERENCING
ERROR rather than the divergence -- it reported div/zeta ~ 0.76 where the spectral
value is ~ 0.010, a 75x overstatement that inverts the conclusion ("not balanced"
instead of "balanced").  classify_mode() now differentiates spectrally.

Re-running the 9 IVPs would take ~35 min; the diagnostic is a pure function of the
FINAL fields, which are already on disk.  So we rebuild the Dedalus structure (for
its coords/sigma NCC), load the stored final frame into it, and re-diagnose.

SELF-CHECK: T_shear and T_bend do not involve the changed derivatives, so they must
reproduce the stored values EXACTLY.  If they do not, the field reload is wrong and
the script refuses to write.
"""
import glob
import os

import h5py
import numpy as np

import sw_sn_driver as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
RTOL = 1e-9


def cfg_from_attrs(a):
    cfg = dict(M.CONFIG)
    for k in cfg:
        if k in a:
            v = a[k]
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, str):
                v = None if v == "None" else v
            cfg[k] = v
    return cfg


def redo_dispersion(path):
    """Re-derive disp_* from the stored centreline (pure numpy, no Dedalus needed).

    Picks up the round-off-contamination gate added to per_mode_dispersion: modes
    swamped by the leading mode's FFT round-off used to pass the e-folding gate
    (noise inherits the leader's growth rate) and showed up as a sawtooth in sigma(k).
    """
    with h5py.File(path, "r") as h:
        ts = np.array(h["t"]); zc = np.array(h["zc"]); Ls = float(h.attrs["Ls"])
        n_ok_old = int(np.sum(np.array(h["disp_converged"]) > 0))
    k, sg, c, ne, ok = M.per_mode_dispersion(ts, zc, Ls)
    with h5py.File(path, "a") as h:
        for name, arr in (("disp_k", k), ("disp_sigma", sg), ("disp_c", c),
                          ("disp_nefold", ne), ("disp_converged", ok)):
            del h[name]
            h[name] = arr
    return n_ok_old, int(np.sum(ok > 0)), len(ok)


def redo(path):
    with h5py.File(path, "r") as h:
        cfg = cfg_from_attrs(h.attrs)
        us_f = np.array(h["us"][-1]); un_f = np.array(h["un"][-1])
        eta_f = np.array(h["eta"][-1]); zc_f = np.array(h["zc"][-1])
        # numeric diag_* only -- diag_summary / diag_method are strings (and this
        # script must stay idempotent: it writes diag_method itself)
        old = {k: float(h.attrs[k]) for k in h.attrs
               if k.startswith("diag_") and np.isreal(h.attrs[k])
               and not isinstance(h.attrs[k], (str, bytes))}

    built = M.build_ivp_SW(cfg)
    for f, arr in ((built["us"], us_f), (built["un"], un_f),
                   (built["eta"], eta_f), (built["zc"], zc_f)):
        f.change_scales(1)
        f["g"] = arr.reshape(f["g"].shape)

    new = M.classify_mode(built)

    # the two energetics terms must be bit-comparable: same fields, same formula
    for key in ("T_shear", "T_bend"):
        o, n_ = old.get(f"diag_{key}"), new[key]
        if o is not None and abs(n_ - o) > RTOL * max(abs(o), 1e-300):
            raise SystemExit(f"{os.path.basename(path)}: {key} changed "
                             f"{o:.10g} -> {n_:.10g}; field reload is WRONG, not writing")

    with h5py.File(path, "a") as h:
        for k, v in new.items():
            h.attrs[f"diag_{k}"] = v
        h.attrs["diag_method"] = "spectral (d3.Differentiate); was np.gradient"

    ok0, ok1, ntot = redo_dispersion(path)
    print(f"{os.path.basename(path)[4:-3]:44s} "
          f"div/zeta {old.get('diag_div_ratio', float('nan')):.4f}->{new['div_ratio']:.5f}  "
          f"converged modes {ok0}->{ok1}/{ntot}"
          f"{'  (round-off gate dropped %d)' % (ok0 - ok1) if ok1 < ok0 else ''}")
    return old.get("diag_div_ratio", np.nan), new["div_ratio"], ok0 - ok1


def main():
    runs = sorted(glob.glob(os.path.join(OUT, "run_*.h5")))
    if not runs:
        raise SystemExit("no outputs/run_*.h5")
    o, n_, dropped = zip(*[redo(p) for p in runs])
    print(f"\n{len(runs)} runs re-diagnosed.")
    # NB 'previously stored' equals the new value on a second run -- this script is
    # idempotent, so the before/after spread is only informative the first time.
    print(f"  previously stored div/zeta : {min(o):.5f} .. {max(o):.5f}")
    print(f"  spectral           div/zeta : {min(n_):.5f} .. {max(n_):.5f}  (<<1 => BALANCED)")
    print(f"  modes newly rejected as FFT round-off of the leading mode: {sum(dropped)}")


if __name__ == "__main__":
    main()
