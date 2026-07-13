#!/usr/bin/env python3
"""Shared core for the Bahmanpouri et al. (2022) entropy-velocity explainer.

Reference
---------
Bahmanpouri, F., Barbetta, S., Gualtieri, C., Ianniruberto, M., Filizola, N.,
Termini, D. & Moramarco, T. (2022) "Estimating the Average River Cross-Section
Velocity by Observing Only One Surface Velocity Value and Calibrating the
Entropic Parameter."  Water Resources Research 58, e2021WR031821.
doi:10.1029/2021WR031821

The problem: estimate the cross-section MEAN velocity (hence discharge) of a
river from a SINGLE surface-velocity observation (UAV / remote sensing), via
Chiu's entropy velocity theory.  Verified equations (transcribed from the
rendered PDF; equation numbers are the paper's):

Eq. (1)  vertical velocity profile with dip h (depth of the velocity maximum
         below the surface), for vertical i at lateral position x_i:

    U(x_i, y) = (Umaxv(x_i)/M) *
                ln[ 1 + (e^M - 1) * (y/(D-h)) * exp(1 - y/(D-h)) ]

    y measured up from the bed; D = local depth; U -> Umaxv at y = D - h.

Eq. (2)  linear entropic relation between cross-section mean and maximum:

    Um = [ e^M/(e^M - 1) - 1/M ] * Umax  =  Phi(M) * Umax

Eq. (4)  maximum velocity of a vertical from its SURFACE velocity
         (exactly Eq. (1) evaluated at y = D and solved for Umaxv):

    Umaxv(x_i) = Usurf(x_i) / [ (1/M) ln(1 + (e^M - 1) delta e^(1-delta)) ],
    delta(x_i) = D(x_i) / (D(x_i) - h(x_i));   h = 0 => Umaxv = Usurf.

(The paper's Eq. (3), an ungauged-site estimate of Phi(M) from roughness and
hydraulic radius after Moramarco & Singh (2010), is NOT implemented: the
paper's own workflow calibrates Phi(M) from ADCP data, and Eq. 3's terms
depend on quantities not tabulated in the paper.)

Surface-velocity scenarios (Corato et al. 2011, as rendered in the paper's
Figs. 4 and 6): the transverse surface-velocity distribution is anchored at
the observed maximum (value + lateral position) and falls to zero at the
water edges, either parabolically or elliptically, with independent left /
right half-widths (the peak is generally not centred).

Validation targets (all printed in the paper; asserted in the self-test)
------------------------------------------------------------------------
Table 2 (ADCP-calibrated entropy parameters, 7 transects):
    Phi in [0.605, 0.732], M in [1.16, 3.24]; row-by-row Phi = Um/Umax and
    Phi = Phi(M) -- EXCEPT FreibergerMulde CS3, where the printed pair
    (Phi = 0.678, M = 1.16) is internally inconsistent: Phi(1.16) = 0.594 =
    0.68/1.14 (the row's own ADCP ratio) while Phi^-1(0.678) = 2.33.  The
    self-test asserts this anomaly EXISTS (a codified published typo).
Sec. 4.2 anchors: Sajo CS1  Usurf_max = 1.113 m/s at x = 17 m, A = 13.67 m^2,
    Q = 11.21 m^3/s, Um = 0.82 m/s;  FM CS1  Usurf_max = 0.87 m/s at
    x = 7.2 m, A = 9.85 m^2, Q = 5.6 m^3/s, Um = 0.57 m/s.
Tables 3-4 (single-surface-velocity pipeline vs measurements):
    Sajo CS1  parabolic 0.81 m/s / 11.07 m^3/s (err 1.22%), elliptic 0.87 /
    11.89 (6.10%);  FM CS1  parabolic 0.58 / 5.70 (1.75%), elliptic 0.63 /
    6.19 (10.52%).  Abstract: "error percentage less than 13%".
Table 5 (vertical-profile fits, parabolic scenario): R^2 = 0.86 (Sajo CS1,
    x = 16 m) and 0.55 (FM CS3, x = 5 m); SE 0.047-0.063 (cited, not
    re-derivable without the SI ADCP profiles).

Data files (data/): Table 2 and Tables 3-4 transcriptions, plus bathymetries
D(x) for Sajo CS1 (Fig. 5a) and FM CS3 (Fig. 7a) digitized programmatically
(color-boundary extraction; see data/digitize_figures.py).  Digitization
checksum: Sajo CS1 integrated area 13.27 m^2 vs printed 13.67 m^2 (-2.9%).
No synthetic or random data anywhere.

Usage
-----
    from bahmanpouri_lib import phi_of_M, discharge, load_bathymetry, ...
    micromamba run -n fourcastnetv2 python bahmanpouri_lib.py   # self-test
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Entropy relations (Eqs. 1, 2, 4)
# --------------------------------------------------------------------------- #
def phi_of_M(M):
    """Phi(M) = e^M/(e^M-1) - 1/M  (Eq. 2), overflow-safe form.

    Implemented as 1/(1 - e^-M) - 1/M (identical algebraically; finite for
    large M).  Series near 0: Phi = 1/2 + M/12 - M^3/720 + O(M^5).
    Strictly increasing from 1/2 (M -> 0) to 1 (M -> inf).
    """
    M = np.asarray(M, dtype=float)
    out = np.where(
        np.abs(M) < 1e-4,
        0.5 + M / 12.0 - M**3 / 720.0,
        1.0 / (1.0 - np.exp(-np.where(np.abs(M) < 1e-4, 1.0, M)))
        - 1.0 / np.where(np.abs(M) < 1e-4, 1.0, M),
    )
    return out if out.ndim else float(out)


def M_of_phi(phi):
    """Invert Phi(M) (scalar): bracketed Brent root find on [1e-9, 700].

    Phi is strictly increasing 1/2 -> 1, so brentq on Phi(M) - phi is the
    textbook approach; the overflow-safe phi_of_M keeps the bracket generous.
    """
    from scipy.optimize import brentq
    phi = float(phi)
    if not 0.5 < phi < 1.0:
        raise ValueError(f"Phi must lie in (0.5, 1); got {phi}")
    return brentq(lambda m: phi_of_M(m) - phi, 1e-9, 700.0, xtol=1e-12)


def u_vertical(y, Umaxv, D, h, M):
    """Entropy velocity profile U(y) along one vertical (Eq. 1).

    y from the bed (0 <= y <= D); maximum U = Umaxv at y = D - h.
    """
    y = np.asarray(y, dtype=float)
    g = (y / (D - h)) * np.exp(1.0 - y / (D - h))
    return (Umaxv / M) * np.log1p(np.expm1(M) * g)


def delta_of(D, h):
    """delta = D / (D - h)  (Eq. 4 text)."""
    return np.asarray(D, dtype=float) / (np.asarray(D, dtype=float) - h)


def umaxv_from_surface(Usurf, D, h, M):
    """Maximum velocity of a vertical from its surface velocity (Eq. 4).

    Eq. (4) is exactly Eq. (1) evaluated at y = D, solved for Umaxv; with
    h = 0 (delta = 1) the denominator is (1/M) ln(e^M) = 1, so
    Umaxv = Usurf identically (as stated below the paper's Eq. 4).
    """
    d = delta_of(D, h)
    denom = np.log1p(np.expm1(M) * d * np.exp(1.0 - d)) / M
    return np.asarray(Usurf, dtype=float) / denom


def dip_solve(Usurf, D, M, umax_target, h_max_frac=0.4):
    """Dip depth h such that Eq. (4) reproduces a known maximum velocity.

    The paper identifies the dip "by minimizing the error on M estimation"
    via the iterative approach of Moramarco et al. (2017); with an
    ADCP-calibrated M (the paper's own gauged workflow) that reduces to
    choosing h so the Eq.-4 maximum matches the observed section maximum.
    Solved by bracketed root find on h in [0, h_max_frac*D]; returns 0 if
    the target is at or below the surface velocity.
    """
    from scipy.optimize import brentq
    if umax_target <= Usurf:
        return 0.0
    f = lambda h: umaxv_from_surface(Usurf, D, h, M) - umax_target
    hi = h_max_frac * D
    if f(hi) < 0.0:
        raise ValueError("umax_target unreachable within h_max_frac*D")
    return brentq(f, 0.0, hi, xtol=1e-10)


# --------------------------------------------------------------------------- #
#  Surface-velocity scenarios (Corato et al. 2011; paper Figs. 4 and 6)
# --------------------------------------------------------------------------- #
def _side_scaled(x, x_peak, x_left, x_right):
    """|x - x_peak| scaled by the bank distance on each side (0 at peak, 1 at bank)."""
    x = np.asarray(x, dtype=float)
    wl = x_peak - x_left
    wr = x_right - x_peak
    s = np.where(x < x_peak, (x_peak - x) / wl, (x - x_peak) / wr)
    return np.clip(s, 0.0, 1.0)


def usurf_parabolic(x, x_peak, x_left, x_right, Usurf_max):
    """Parabolic transverse surface-velocity scenario: zero at both banks."""
    s = _side_scaled(x, x_peak, x_left, x_right)
    return Usurf_max * (1.0 - s**2)


def usurf_elliptic(x, x_peak, x_left, x_right, Usurf_max):
    """Elliptic scenario (fuller near the banks than the parabola)."""
    s = _side_scaled(x, x_peak, x_left, x_right)
    return Usurf_max * np.sqrt(1.0 - s**2)


# --------------------------------------------------------------------------- #
#  Section integration: the single-surface-velocity pipeline
# --------------------------------------------------------------------------- #
def section_area(x, D):
    """Cross-section area by trapezoid rule."""
    return float(np.trapz(np.asarray(D, float), np.asarray(x, float)))


def depth_mean_velocity(Usurf_i, D_i, M, h_i=0.0, n_y=200):
    """Depth-averaged velocity of one vertical: numeric mean of Eq. (1)."""
    Umaxv = umaxv_from_surface(Usurf_i, D_i, h_i, M)
    y = np.linspace(0.0, D_i, n_y)
    return float(np.trapz(u_vertical(y, Umaxv, D_i, h_i, M), y) / D_i)


def discharge(x, D, M, Usurf_max, x_peak, scenario="parabolic", h=0.0):
    """Full pipeline: one surface-velocity value -> (Q, Um, per-vertical data).

    Steps (paper Fig. 2 flowchart): build the scenario surface-velocity
    distribution anchored at (x_peak, Usurf_max) and the water edges; per
    vertical convert Usurf -> Umaxv (Eq. 4, dip h) and depth-integrate the
    Eq.-1 profile; integrate across the section.  h may be a scalar or an
    array aligned with x (0 = no dip).

    Returns dict(Q, Um, A, usurf, ubar).
    """
    x = np.asarray(x, dtype=float)
    D = np.asarray(D, dtype=float)
    wet = D > 0.005
    fn = usurf_parabolic if scenario == "parabolic" else usurf_elliptic
    us = fn(x, x_peak, x[wet].min(), x[wet].max(), Usurf_max)
    hh = np.broadcast_to(np.asarray(h, dtype=float), x.shape)
    ubar = np.zeros_like(x)
    for i in np.where(wet)[0]:
        ubar[i] = depth_mean_velocity(us[i], D[i], M, hh[i])
    A = section_area(x, D)
    Q = float(np.trapz(ubar * D, x))
    return dict(Q=Q, Um=Q / A, A=A, usurf=us, ubar=ubar)


# --------------------------------------------------------------------------- #
#  Published tables and digitized sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CrossSection:
    """One Table-2 row: ADCP-calibrated entropy parameters of a transect."""
    name: str
    Um: float          # observed mean velocity, ADCP (m/s)
    Umax: float        # observed maximum velocity, ADCP (m/s)
    Umax_entropy: float
    phi: float         # printed Phi(M)
    M: float           # printed entropy parameter
    aspect: float      # width/depth


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(x for x in f if not x.startswith("#")))


def load_table2():
    rows = _read_csv(os.path.join(DATA_DIR, "table2_entropy_params.csv"))
    return tuple(CrossSection(r["transect"], float(r["Um_adcp_ms"]),
                              float(r["Umax_adcp_ms"]), float(r["Umax_entropy_ms"]),
                              float(r["phi"]), float(r["M"]), float(r["aspect_ratio"]))
                 for r in rows)


def load_validation():
    rows = _read_csv(os.path.join(DATA_DIR, "tables34_validation.csv"))
    for r in rows:
        for k in list(r):
            if k not in ("transect", "scenario"):
                r[k] = float(r[k])
    return rows


def load_bathymetry(which):
    """Digitized D(x) for 'sajo_cs1' or 'mulde_cs3'; returns (x, D)."""
    path = os.path.join(DATA_DIR, f"{which}_bathymetry.csv")
    rows = _read_csv(path)
    x = np.array([float(r["x_m"]) for r in rows])
    D = np.array([float(r["depth_m"]) for r in rows])
    o = np.argsort(x)
    return x[o], D[o]


# Printed section anchors (Sec. 4.2 text).
SAJO_CS1 = dict(Usurf_max=1.113, x_peak=17.0, A=13.67, Q=11.21, Um=0.82,
                phi=0.732, M=3.24)
FM_CS1 = dict(Usurf_max=0.87, x_peak=7.2, A=9.85, Q=5.60, Um=0.57,
              phi=0.610, M=1.37)


# --------------------------------------------------------------------------- #
#  Plot styling & saving
# --------------------------------------------------------------------------- #
COLORS = {
    "water": "#2c7fb8",
    "water_fill": "#c7e0f0",
    "bed": "#7f5539",         # bank/bed brown
    "parabolic": "#d7301f",   # paper's red scenario curve
    "elliptic": "#08519c",    # paper's dashed blue scenario
    "entropy": "#238b45",     # entropy-model results (green)
    "adcp": "#6a51a3",        # ADCP observations (purple, as in Figs. 4/6)
    "uav": "#e6820a",         # UAV points (orange for contrast)
    "anomaly": "#d7301f",
    "band": "#c7e0f0",
}

# === shared helper block v1 (keep byte-identical across rossby_palooza packages) ===
def set_style():
    """Apply a consistent matplotlib style (Agg backend, readable fonts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "cm",
    })
    return plt


def save_fig(fig, name, subdir=None, close=True):
    """Save a figure into figures/ (or a subdir) as PNG; return the path."""
    out_dir = FIG_DIR if subdir is None else os.path.join(FIG_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    if not name.lower().endswith(".png"):
        name += ".png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")
    return path


def fig_to_rgb(fig):
    """Rasterise a drawn matplotlib figure to an (H, W, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def write_mp4(frames, name, fps=20):
    """Write a list of RGB frames to figures/<name>.mp4 via imageio+libx264.

    Also drops a representative preview PNG next to the mp4 (middle frame),
    following the chedan_talk/ convention.  Returns the mp4 path.
    """
    import imageio.v2 as imageio
    from PIL import Image

    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    mp4_path = os.path.join(FIG_DIR, name)
    imageio.mimsave(mp4_path, frames, fps=fps, codec="libx264",
                    quality=8, macro_block_size=1)
    prev = frames[len(frames) // 2]
    prev_path = mp4_path[:-4] + "_preview.png"
    Image.fromarray(prev).save(prev_path)
    print(f"  wrote {os.path.relpath(mp4_path, HERE)}  ({len(frames)} frames)")
    print(f"  wrote {os.path.relpath(prev_path, HERE)}")
    return mp4_path
# === end shared helper block ===


# --------------------------------------------------------------------------- #
#  Self-test: three tiers (exact / published / digitized)
# --------------------------------------------------------------------------- #
def _self_test():
    print("Bahmanpouri et al. (2022) -- entropy velocity method, verified core")
    print("-" * 72)

    # ---- Tier 1: exact identities -------------------------------------- #
    assert abs(phi_of_M(1e-8) - 0.5) < 1e-6, "Phi -> 1/2 as M -> 0"
    m = 0.01
    series = 0.5 + m / 12.0 - m**3 / 720.0
    assert abs(phi_of_M(m) - series) < 1e-10, "series match at small M"
    Mg = np.linspace(1e-3, 60.0, 500)
    assert np.all(np.diff(phi_of_M(Mg)) > 0), "Phi strictly increasing"
    assert 0.99 < phi_of_M(700.0) < 1.0, "Phi -> 1 without overflow"
    for Mv in (0.1, 1.16, 3.24, 10.0, 50.0):
        assert abs(M_of_phi(phi_of_M(Mv)) - Mv) < 1e-9, "Phi inversion round-trip"

    D0, h0, M0, Um0 = 0.9, 0.12, 3.24, 1.11
    y = np.linspace(0.0, D0, 4001)
    prof = u_vertical(y, Um0, D0, h0, M0)
    assert abs(prof[0]) < 1e-12, "U(0) = 0 at the bed"
    assert abs(y[np.argmax(prof)] - (D0 - h0)) < 2e-3, "max at y = D - h"
    assert abs(float(u_vertical(D0, Um0, D0, 0.0, M0)) - Um0) < 1e-12, \
        "h = 0: surface velocity equals Umaxv"
    assert abs(float(umaxv_from_surface(0.8, D0, 0.0, M0)) - 0.8) < 1e-12, \
        "Eq. 4 with h = 0 is the identity (paper text)"
    # Dual route: Eq. 4 is Eq. 1 inverted at the surface.
    us = float(u_vertical(D0, Um0, D0, h0, M0))
    assert abs(float(umaxv_from_surface(us, D0, h0, M0)) - Um0) < 1e-12, \
        "Eq. 4 must invert Eq. 1 at y = D exactly"
    hsol = dip_solve(us, D0, M0, Um0)
    assert abs(hsol - h0) < 1e-8, "dip_solve recovers the dip"
    print("Tier 1 (exact identities): OK")

    # ---- Tier 2: published-table recomputation ------------------------- #
    t2 = load_table2()
    assert len(t2) == 7
    phis = np.array([c.phi for c in t2])
    Ms = np.array([c.M for c in t2])
    assert (phis.min(), phis.max()) == (0.605, 0.732), "Phi range as printed"
    assert (Ms.min(), Ms.max()) == (1.16, 3.24), "M range as printed"
    n_ok = 0
    for c in t2:
        if "CS3" in c.name and "Mulde" in c.name.replace(" ", ""):
            continue
        dphi = abs(float(phi_of_M(c.M)) - c.phi)
        assert dphi <= 0.005, f"{c.name}: Phi(M) identity broken ({dphi:.4f})"
        assert abs(c.phi - c.Um / c.Umax) <= 0.01, f"{c.name}: Phi != Um/Umax"
        n_ok += 1
    assert n_ok == 6, "six consistent Table-2 rows"
    # The codified published typo: FM CS3 must FAIL the identity, and its M
    # must instead match the row's own ADCP ratio.
    cs3 = [c for c in t2 if c.name == "FreibergerMulde CS3"][0]
    assert abs(float(phi_of_M(cs3.M)) - cs3.phi) > 0.05, \
        "FM CS3 anomaly vanished -- data or theory edited?"
    assert abs(float(phi_of_M(cs3.M)) - cs3.Um / cs3.Umax) < 0.005, \
        "FM CS3: Phi(M=1.16) must equal the ADCP ratio 0.68/1.14"
    print(f"Tier 2: Table 2 identity holds for 6/7 rows; FM CS3 anomaly "
          f"codified (Phi(1.16) = {float(phi_of_M(1.16)):.3f} vs printed 0.678)")

    val = load_validation()
    errs = []
    for r in val:
        ev = abs(r["Um_calc_ms"] - r["Um_meas_ms"]) / r["Um_meas_ms"] * 100.0
        assert abs(ev - r["vel_err_pct"]) < 0.05, \
            f"{r['transect']} {r['scenario']}: velocity error mismatch"
        assert r["Q_err_pct"] == r["vel_err_pct"], \
            "printed Q error equals velocity error (Q = Um*A construction)"
        errs.append(max(r["vel_err_pct"], r["Q_err_pct"]))
    assert max(errs) < 13.0, "abstract's <13% bound"
    assert abs(SAJO_CS1["Um"] * SAJO_CS1["A"] - SAJO_CS1["Q"]) < 0.01, \
        "Sajo CS1: Q = Um * A (0.82*13.67 = 11.21)"
    assert abs(FM_CS1["Um"] * FM_CS1["A"] - FM_CS1["Q"]) < 0.02, \
        "FM CS1: Q = Um * A"
    # Parabolic beats elliptic on both printed sections.
    for name in ("Sajo CS1", "FreibergerMulde CS1"):
        rows = {r["scenario"]: r for r in val if r["transect"] == name}
        assert rows["parabolic"]["vel_err_pct"] < rows["elliptic"]["vel_err_pct"]
    print(f"Tier 2: Tables 3-4 errors recomputed; max {max(errs):.2f}% < 13%")

    # ---- Tier 3: digitized-bathymetry pipeline (soft tier) ------------- #
    x, D = load_bathymetry("sajo_cs1")
    A = section_area(x, D)
    assert abs(A - SAJO_CS1["A"]) / SAJO_CS1["A"] < 0.05, \
        f"digitized Sajo area {A:.2f} vs printed 13.67"
    assert 23.0 < x[D > 0.005].max() - x[D > 0.005].min() < 25.0
    assert 0.85 < D.max() < 0.95
    par = discharge(x, D, SAJO_CS1["M"], SAJO_CS1["Usurf_max"],
                    SAJO_CS1["x_peak"], "parabolic")
    ell = discharge(x, D, SAJO_CS1["M"], SAJO_CS1["Usurf_max"],
                    SAJO_CS1["x_peak"], "elliptic")
    print(f"Tier 3: Sajo CS1 pipeline  parabolic Um = {par['Um']:.3f} "
          f"(paper 0.81), Q = {par['Q']:.2f} (paper 11.07);  "
          f"elliptic Um = {ell['Um']:.3f} (paper 0.87), Q = {ell['Q']:.2f} (11.89)")
    assert abs(par["Um"] - 0.81) / 0.81 < 0.10, "parabolic Um within 10%"
    assert abs(par["Q"] - 11.07) / 11.07 < 0.12, "parabolic Q within 12%"
    assert ell["Um"] > par["Um"], \
        "elliptic overestimates relative to parabolic (Tables 3-4 ordering)"

    xm, Dm = load_bathymetry("mulde_cs3")
    Am = section_area(xm, Dm)
    assert 8.0 < Am < 11.0 and 15.0 < xm[Dm > 0.005].max() - xm[Dm > 0.005].min() < 17.0
    print(f"Tier 3: FM CS3 digitized area = {Am:.2f} m^2, "
          f"width = {xm[Dm>0.005].max()-xm[Dm>0.005].min():.1f} m (plausibility only; "
          f"CS3 pipeline results are in the paper's SI)")

    print("-" * 72)
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
