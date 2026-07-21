#!/usr/bin/env python3
"""Diagnostics derived FROM a finished run -- growth rates, dispersion, mode class.

    python analysis.py            # annotate every ../outputs/run_*.h5 in place

None of this belongs in the driver.  The driver's job is to integrate the equations
and write raw fields; everything here is an interpretation of those fields, and keeping
it separate means a diagnostic can be changed (or found to be wrong -- both of the
functions below have been) without any suggestion that the simulation changed too.
It also means these can be re-derived from stored output without re-running anything.

Writes back into each HDF5:
    disp_k, disp_sigma, disp_c, disp_nefold, disp_converged   per-mode dispersion
    diag_*                                                    mode classification
"""
import glob
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT_DIR = os.path.join(PKG, "outputs")
sys.path.insert(0, PKG)
import sw_meander as MD          # noqa: E402  (base profiles + CONFIG schema)


# --------------------------------------------------------------------------- #
#  Growth rate and phase speed
# --------------------------------------------------------------------------- #
def measure_sigma_c(ts, series, k, Ls):
    """Fit growth sigma and phase speed c of one demodulated centreline mode.

    Returns (sigma, c, k_eff, n_efold).  The phase rate must be divided by the mode
    ACTUALLY present, k_eff = 2*pi*m/Ls, not by the requested k: on a coarse k-grid
    those differ by up to ~10%, and that error alone puts a spurious sawtooth into c(k).
    n_efold lets the caller reject modes that never grew (the fit is meaningless below
    ~3 e-foldings, and its very SIGN then depends on the fit window).
    """
    ts = np.asarray(ts)
    m = max(1, int(round(k * Ls / (2 * np.pi))))
    k_eff = m * 2 * np.pi / Ls
    coeff = np.array([np.fft.rfft(z)[m] for z in series])
    amp = np.abs(coeff)
    j = len(ts) // 3                                    # drop the transient
    sig = np.polyfit(ts[j:], np.log(amp[j:] + 1e-30), 1)[0]
    cph = -np.polyfit(ts[j:], np.unwrap(np.angle(coeff))[j:], 1)[0] / k_eff
    n_efold = float(np.log(max(amp[-1], 1e-300) / max(amp[j], 1e-300)))
    return sig, cph, k_eff, n_efold


def per_mode_dispersion(ts, zcs, Ls, n_min_efold=3.0):
    """Demodulate EVERY centreline Fourier mode -> the whole dispersion relation.

    The seed is a localised bump, which is broadband by construction, and on a straight
    base channel the base state is s-translation-invariant so the s-Fourier modes
    decouple exactly.  One run therefore contains sigma(k) and c(k) for every resolvable
    k.  (At finite bank sinuosity the modes couple -- Floquet -- and these are Bloch
    quantities, not eigenvalues; the caller must say so.)

    Two gates mark a mode not-converged:
      (a) fewer than n_min_efold e-foldings -- the fit is to transient adjustment;
      (b) ROUND-OFF CONTAMINATION.  Once the fastest mode reaches amplitude A_max(t),
          every other mode in the FFT is polluted at eps*A_max(t), so a mode that has
          fallen below that floor measures the LEADER's round-off and reads back the
          LEADER's growth rate.  Gate (a) cannot catch this -- noise inherits the
          leader's growth and looks perfectly converged; it shows up instead as a
          sawtooth in which sigma depends only on m modulo a fixed integer.  Because
          the floor RISES with the leader, validity is a time WINDOW per mode, so each
          mode is fitted only over its own valid stretch rather than discarded.
    """
    ts = np.asarray(ts)
    C = np.fft.rfft(np.array(zcs), axis=1)
    j = len(ts) // 3
    floor_t = (1e3 * np.finfo(float).eps) * np.max(np.abs(C[:, 1:]), axis=1)
    ks, sg, cs, ne, ok = [], [], [], [], []
    for m in range(1, C.shape[1] - 1):
        a = C[:, m]
        amp = np.abs(a)
        if not np.all(np.isfinite(amp)) or amp[j] <= 0:
            continue
        k = m * 2 * np.pi / Ls
        clean = amp > floor_t
        i1 = len(ts)
        if not clean[-1]:
            bad = np.nonzero(~clean[j:])[0]
            i1 = j + int(bad[0]) if len(bad) else len(ts)
        w = slice(j, max(i1, j + 2))
        nfit = len(ts[w])
        ks.append(k)
        sg.append(np.polyfit(ts[w], np.log(amp[w] + 1e-300), 1)[0])
        cs.append(-np.polyfit(ts[w], np.unwrap(np.angle(a))[w], 1)[0] / k)
        n = float(np.log(max(amp[w][-1], 1e-300) / max(amp[w][0], 1e-300)))
        ne.append(n)
        ok.append(float(n >= n_min_efold and nfit >= 8))
    return (np.array(ks), np.array(sg), np.array(cs), np.array(ne), np.array(ok))


# --------------------------------------------------------------------------- #
#  Which kind of mode is it?
# --------------------------------------------------------------------------- #
def classify_mode(us, un, eta, zc, s, n, sig, cfg):
    """Diagnose the mode from the FINAL fields.

      div_ratio  = ||delta'||/||zeta'||  -> ~0 means balanced (not a gravity wave)
      pv_ratio   = ||q'||/||zeta'||      -> ~1 means vortex stretching is minor
      eta_over_u = ||eta'||/||u'||       -> should scale as F^2 for a balanced mode
      T_shear    = -INT u_s'u_n' d_n(Ubar)  the ONLY channel by which the mean-flow
                   vorticity gradient can power a free vortical wave.  Its SIGN is what
                   the whole exercise turns on: T_shear<=0 means the mean flow is a
                   SINK, so the disturbance is boundary-driven whatever its Froude
                   behaviour.  Quote the sign only -- the magnitude is
                   resolution-sensitive (it moved 9x from Ns=64 to Ns=128).
      T_bend     = -INT u_n' Ubar^2 d_ss(zeta_c)  work done by the moving bank.

    NOTE ON THE DERIVATIVES.  div is the small residual of two cancelling O(k|u'|)
    terms, so differencing a Chebyshev grid measures the differencing error, not the
    divergence: np.gradient reported div/zeta ~ 0.76 where the spectral value is 0.010,
    a 75x overstatement that inverted the verdict.  We rebuild a Dedalus basis and
    differentiate spectrally.
    """
    import dedalus.public as d3
    Ns, Nn = us.shape
    b, Ls = cfg["b"], cfg["Ls"]
    coords = d3.CartesianCoordinates("s", "n")
    dist = d3.Distributor(coords, dtype=np.float64)
    sb = d3.RealFourier(coords["s"], size=Ns, bounds=(0.0, Ls))
    nb = d3.Chebyshev(coords["n"], size=Nn, bounds=(-b, b))
    Ds = lambda A: d3.Differentiate(A, coords["s"])
    Dn = lambda A: d3.Differentiate(A, coords["n"])

    def fld(arr):
        f = dist.Field(bases=(sb, nb)); f.change_scales(1); f["g"] = arr; return f

    def gval(expr):
        f = expr.evaluate(); f.change_scales(1); return np.array(f["g"])

    usf, unf, sigf = fld(us), fld(un), fld(sig)
    zeta = gval((Ds(unf) - Dn(sigf * usf)) / sigf)
    div = gval((Ds(usf) + Dn(sigf * unf)) / sigf)

    hb = MD.bed_depth(n, cfg)[None, :] * np.ones_like(us)
    Ub = MD.ubar_s(n, cfg)[None, :] * np.ones_like(us)
    Ub_n = MD.ubar_s_n(n, cfg)[None, :] * np.ones_like(us)
    zbar = -np.gradient(sig * Ub, n, axis=1) / sig      # base state: smooth, analytic
    q = zeta / hb - zbar * eta / hb ** 2

    L2 = lambda A: float(np.sqrt(np.sum(A ** 2 * sig)))
    unorm = float(np.sqrt(np.sum((us ** 2 + un ** 2) * sig)))
    zc_ss = np.gradient(np.gradient(zc, s), s)[:, None] * np.ones_like(us)
    T_bend = float(-np.sum(un * Ub ** 2 * zc_ss * sig))
    T_shear = float(-np.sum(us * un * Ub_n * sig))
    return dict(div_ratio=L2(div) / max(L2(zeta), 1e-300),
                pv_ratio=L2(q) / max(L2(zeta), 1e-300),
                eta_over_u=L2(eta) / max(unorm, 1e-300),
                T_bend=T_bend, T_shear=T_shear,
                shear_share=T_shear / max(abs(T_bend), 1e-300))


# --------------------------------------------------------------------------- #
def annotate(path):
    """Derive every diagnostic from the stored fields and write them back in place."""
    with h5py.File(path, "r") as h:
        cfg = {k: (None if str(h.attrs[k]) == "None" else h.attrs[k])
               for k in h.attrs if k in MD.CONFIG}
        cfg["Ls"] = float(h.attrs["Ls"])
        ts, zc = np.array(h["t"]), np.array(h["zc"])
        s, n, sig = np.array(h["s"]), np.array(h["n"]), np.array(h["sigma_metric"])
        us, un, eta = (np.array(h[k][-1]) for k in ("us", "un", "eta"))

    k, sg, c, ne, ok = per_mode_dispersion(ts, zc, cfg["Ls"])
    d = classify_mode(us, un, eta, zc[-1], s, n, sig, cfg)

    with h5py.File(path, "a") as h:
        for name, arr in (("disp_k", k), ("disp_sigma", sg), ("disp_c", c),
                          ("disp_nefold", ne), ("disp_converged", ok)):
            if name in h:
                del h[name]
            h[name] = arr
        for key, v in d.items():
            h.attrs[f"diag_{key}"] = v

    i = int(np.argmax(np.where(ok > 0, sg, -np.inf))) if np.any(ok > 0) else -1
    fast = f"k={k[i]:.2f} sigma={sg[i]:+.4f} c={c[i]:+.3f}" if i >= 0 else "none converged"
    print(f"{os.path.basename(path)[4:-3][:44]:46s} {int(ok.sum()):2d}/{len(ok)} modes  "
          f"{fast}  div/zeta={d['div_ratio']:.4f}  "
          f"T_sh/|T_bd|={d['shear_share']:+.3f}")


def main():
    runs = sorted(glob.glob(os.path.join(OUT_DIR, "run_*.h5")))
    if not runs:
        raise SystemExit("no ../outputs/run_*.h5 -- run ../sw_meander.py first")
    for p in runs:
        annotate(p)
    print(f"\nannotated {len(runs)} run(s)")


if __name__ == "__main__":
    main()
