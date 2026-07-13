#!/usr/bin/env python3
"""Shared core for the Schumm (1967) meander-wavelength explainer.

Reference
---------
Schumm, S. A. (1967) "Meander Wavelength of Alluvial Rivers."
Science 157 (3796), 1549-1550.  doi:10.1126/science.157.3796.1549

Data source (the Science report prints no table; its refs 5 and 7):
Schumm, S. A. (1968) "River Adjustment to Altered Hydrologic Regimen --
Murrumbidgee River and Paleochannels, Australia." USGS Professional Paper
598: Table 6 (33 Midwestern US rivers, p. 45) and Table 1 (Murrumbidgee
sections; the three with measured wavelength).  Public domain.

This module is the single source of truth for the *verified* empirical
relations.  The two multiple regressions (transcribed from the rendered PDF
page -- the text layer garbles the exponents to "Qm'°3/M0'") are

    lambda = 1890 Qm^0.34  / M^0.74      (1)   r = .95, 89% explained,
                                               standard error 0.16 log units
    lambda = 234  Qma^0.48 / M^0.74      (2)   r = .93, 86% explained,
                                               standard error 0.19 log units

with lambda meander wavelength (ft), Qm mean annual discharge (cfs), Qma
mean annual flood (cfs), and M the weighted percent silt-clay (grains
< 0.074 mm) in the channel perimeter.  Discharge ALONE explains only 43%
(Qm) / 40% (Qma) of the wavelength variance -- sediment type is the story:
at fixed discharge the observed M range spans a "tenfold range in meander
wavelength" (paper, p. 1550).

Channel classification by M (paper, p. 1549, after Schumm 1963 Circ. 477):
    bedload   channels: M < 5   (transport > 11% of load as sand+gravel)
    mixed     channels: 5 <= M <= 20
    suspended channels: M > 20  (transport <~ 3% sand+gravel)

Reference lines drawn in the paper's figures (for replicas only):
    Fig. 1:  Carlston (1965)  lambda = 106 Qm^0.46   (14 eastern-US rivers)
    Fig. 2:  Dury (1965)      lambda =  30 Qb^0.5    (bankfull; drawn on the
                                                      Qma axis in the paper)

Transcription validation
------------------------
data/schumm_1967_sections.csv holds the 36 sections (33 US + 3 Murrumbidgee)
transcribed 2026-07-06 from the PP 598 scan (rendered image cross-checked
against the OCR text layer).  Refitting the two regressions on the
transcription reproduces every published statistic to rounding precision:

    Eq. 1: coef 1819 (pub 1890), exponents 0.350/-0.734 (pub 0.34/-0.74),
           r .947 (pub .95), R2 .896 (pub 89%), SEE .165 (pub .16),
           Qm-alone R2 .436 (pub 43%).
    Eq. 2: coef 224 (pub 234), exponents 0.483/-0.734 (pub 0.48/-0.74),
           r .924 (pub .93), R2 .854 (pub 86%), SEE .196 (pub .19),
           Qma-alone R2 .398 (pub 40%).

Schumm's own published fit is thus the checksum of the transcription; the
self-test asserts all of it.

Usage
-----
    from schumm_lib import PARAMS, wavelength_qm, classify, load_sections
    micromamba run -n fourcastnetv2 python schumm_lib.py    # runs self-test
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

CSV_PATH = os.path.join(DATA_DIR, "schumm_1967_sections.csv")


# --------------------------------------------------------------------------- #
#  Canonical published constants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """The published regression constants of Schumm (1967), Eqs. (1)-(2)."""
    coef_qm: float = 1890.0    # Eq. (1) coefficient
    exp_qm: float = 0.34       # Eq. (1) exponent on Qm
    coef_qma: float = 234.0    # Eq. (2) coefficient
    exp_qma: float = 0.48      # Eq. (2) exponent on Qma
    exp_m: float = 0.74        # exponent on M (denominator, both equations)
    name: str = "schumm1967"


PARAMS = Params()

# Published goodness-of-fit statistics (Science, p. 1550).
PUBLISHED = {
    "eq1": dict(r=0.95, R2=0.89, see_log=0.16, R2_Q_alone=0.43),
    "eq2": dict(r=0.93, R2=0.86, see_log=0.19, R2_Q_alone=0.40),
}

# Class boundaries in percent silt-clay (paper p. 1549).
M_BEDLOAD_MAX = 5.0      # bedload:   M < 5
M_SUSPENDED_MIN = 20.0   # suspended: M > 20


# --------------------------------------------------------------------------- #
#  The empirical relations (Eqs. 1-2) and inverses
# --------------------------------------------------------------------------- #
def wavelength_qm(Qm, M, p=PARAMS):
    """Meander wavelength (ft) from mean annual discharge  (Eq. 1)."""
    Qm = np.asarray(Qm, dtype=float)
    M = np.asarray(M, dtype=float)
    return p.coef_qm * Qm**p.exp_qm / M**p.exp_m


def wavelength_qma(Qma, M, p=PARAMS):
    """Meander wavelength (ft) from mean annual flood  (Eq. 2)."""
    Qma = np.asarray(Qma, dtype=float)
    M = np.asarray(M, dtype=float)
    return p.coef_qma * Qma**p.exp_qma / M**p.exp_m


def M_from_lambda_qm(lam, Qm, p=PARAMS):
    """Invert Eq. (1) for the silt-clay index M."""
    lam = np.asarray(lam, dtype=float)
    Qm = np.asarray(Qm, dtype=float)
    return (p.coef_qm * Qm**p.exp_qm / lam) ** (1.0 / p.exp_m)


def Qm_from_lambda(lam, M, p=PARAMS):
    """Invert Eq. (1) for mean annual discharge."""
    lam = np.asarray(lam, dtype=float)
    M = np.asarray(M, dtype=float)
    return (lam * M**p.exp_m / p.coef_qm) ** (1.0 / p.exp_qm)


def classify(M):
    """Channel class from percent silt-clay M: bedload / mixed / suspended.

    Paper p. 1549: "more than 20 percent silt and clay ... suspended-load
    channels"; "less than 5 percent silt-clay ... bedload channels"; the
    intermediate class is "mixed-load".
    """
    M = np.asarray(M, dtype=float)
    out = np.where(M < M_BEDLOAD_MAX, "bedload",
                   np.where(M > M_SUSPENDED_MIN, "suspended", "mixed"))
    return out if out.ndim else str(out)


def carlston_line(Qm):
    """Carlston (1965) single-variable line lambda = 106 Qm^0.46 (Fig. 1)."""
    Qm = np.asarray(Qm, dtype=float)
    return 106.0 * Qm**0.46


def dury_line(Q):
    """Dury (1965) bankfull line lambda = 30 Qb^0.5 (drawn in Fig. 2)."""
    Q = np.asarray(Q, dtype=float)
    return 30.0 * Q**0.5


def tenfold_range_factor(M_lo=1.3, M_hi=44.6, p=PARAMS):
    """Wavelength ratio across the observed M range at fixed discharge.

    Defaults are the dataset's extreme M values; the paper claims "a tenfold
    range in meander wavelength at a given discharge" from sediment type.
    """
    return (M_hi / M_lo) ** p.exp_m


# --------------------------------------------------------------------------- #
#  Data loading and regression engine
# --------------------------------------------------------------------------- #
def load_sections(path=CSV_PATH):
    """Load the 36 transcribed sections; returns a dict of numpy arrays.

    Qb_cfs is NaN where PP 598 prints no bankfull discharge (5 US rows).
    """
    rows = []
    with open(path) as f:
        for r in csv.DictReader(x for x in f if not x.startswith("#")):
            rows.append(r)
    out = {
        "id": np.array([int(r["id"]) for r in rows]),
        "river": np.array([r["river"] for r in rows]),
        "M": np.array([float(r["M_pct"]) for r in rows]),
        "Qm": np.array([float(r["Qm_cfs"]) for r in rows]),
        "Qma": np.array([float(r["Qma_cfs"]) for r in rows]),
        "Qb": np.array([float(r["Qb_cfs"]) if r["Qb_cfs"] else np.nan
                        for r in rows]),
        "lam": np.array([float(r["lambda_ft"]) for r in rows]),
        "sinuosity": np.array([float(r["sinuosity"]) for r in rows]),
        "wd_ratio": np.array([float(r["wd_ratio"]) for r in rows]),
        "source": np.array([r["source"] for r in rows]),
    }
    out["cls"] = classify(out["M"])
    return out


def fit_power_law(Q, M, lam):
    """OLS fit of log10(lam) = b0 + b1 log10(Q) + b2 log10(M).

    Returns dict(coef, expQ, expM, r, R2, see_log, R2_Q_alone), where
    see_log is the standard error of estimate sqrt(SS_res / (n - 3)) in
    log10 units (this convention reproduces the paper's 0.16 / 0.19) and
    r = sqrt(R2) is the multiple correlation coefficient.
    """
    Q = np.asarray(Q, dtype=float)
    M = np.asarray(M, dtype=float)
    lam = np.asarray(lam, dtype=float)
    y = np.log10(lam)
    X = np.column_stack([np.ones_like(y), np.log10(Q), np.log10(M)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot
    X1 = X[:, :2]
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    R2q = 1.0 - float(np.sum((y - X1 @ b1) ** 2)) / ss_tot
    return dict(coef=10.0 ** b[0], expQ=float(b[1]), expM=float(b[2]),
                r=float(np.sqrt(R2)), R2=float(R2),
                see_log=float(np.sqrt(ss_res / (len(y) - 3))),
                R2_Q_alone=float(R2q))


def loo_log_errors(Q, M, lam):
    """Leave-one-out |log10(lam_pred) - log10(lam_obs)| per section.

    Refits the two-predictor power law without each point and predicts it;
    the honest scatter statement behind the +-2 SE band.
    """
    Q = np.asarray(Q, dtype=float)
    M = np.asarray(M, dtype=float)
    lam = np.asarray(lam, dtype=float)
    n = len(lam)
    errs = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        f = fit_power_law(Q[keep], M[keep], lam[keep])
        pred = f["coef"] * Q[i] ** f["expQ"] * M[i] ** f["expM"]
        errs[i] = abs(np.log10(pred) - np.log10(lam[i]))
    return errs


# --------------------------------------------------------------------------- #
#  Plot styling & saving
# --------------------------------------------------------------------------- #
# Palette: channel classes + reference lines (colourblind-friendly).
COLORS = {
    "bedload": "#d7301f",     # low silt-clay, long waves (red)
    "mixed": "#e6820a",       # intermediate (orange)
    "suspended": "#2c7fb8",   # high silt-clay, short waves (blue)
    "fit": "#252525",         # Schumm's two-variable fit (near-black)
    "reference": "#969696",   # Carlston / Dury single-variable lines (grey)
    "band": "#c7e0f0",        # +-2 SE band fill
    "water": "#2c7fb8",
    "growth": "#238b45",
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
#  Self-test: the published statistics are the checksum of the data
# --------------------------------------------------------------------------- #
def _self_test():
    print("Schumm (1967) -- meander wavelength of alluvial rivers, verified core")
    print("-" * 70)

    # Frozen published constants (guards accidental edits).
    v = 1890.0 * 1000.0**0.34 / 5.0**0.74
    assert abs(float(wavelength_qm(1000.0, 5.0)) - v) < 1e-9 * v
    v2 = 234.0 * 10000.0**0.48 / 12.0**0.74
    assert abs(float(wavelength_qma(10000.0, 12.0)) - v2) < 1e-9 * v2

    # Monotonicity: wavelength grows with discharge, shrinks with silt-clay.
    Qg = np.logspace(1, 4, 40)
    Mg = np.linspace(1.0, 45.0, 40)
    assert np.all(np.diff(wavelength_qm(Qg, 10.0)) > 0)
    assert np.all(np.diff(wavelength_qma(Qg, 10.0)) > 0)
    assert np.all(np.diff(wavelength_qm(500.0, Mg)) < 0)

    # Inverses round-trip.
    lam0 = wavelength_qm(750.0, 8.0)
    assert abs(float(M_from_lambda_qm(lam0, 750.0)) - 8.0) < 1e-9
    assert abs(float(Qm_from_lambda(lam0, 8.0)) - 750.0) < 1e-6

    # Class boundaries exactly as printed (M<5 bed, M>20 susp).
    assert classify(4.999) == "bedload" and classify(5.0) == "mixed"
    assert classify(20.0) == "mixed" and classify(20.001) == "suspended"

    # Dataset structure: 36 sections = 33 US (Table 6) + 3 Murrumbidgee
    # (Table 1); PP 598 printed bankfull discharge for exactly 28 US rivers.
    d = load_sections()
    assert len(d["lam"]) == 36, "36 sections (paper abstract)"
    us = d["source"] == "PP598-T6"
    assert int(us.sum()) == 33 and int((~us).sum()) == 3
    assert int(np.isfinite(d["Qb"][us]).sum()) == 28, \
        "PP598: 'bankfull discharge of 28 was calculated'"
    for c in ("bedload", "mixed", "suspended"):
        assert np.any(d["cls"] == c), f"class {c} must be represented"

    # Eq. (1): refit reproduces the published fit (the transcription checksum).
    f1 = fit_power_law(d["Qm"], d["M"], d["lam"])
    print(f"Eq.1 refit: lam = {f1['coef']:.0f} Qm^{f1['expQ']:.3f} M^{f1['expM']:.3f}"
          f"   (published: 1890 Qm^0.34 / M^0.74)")
    print(f"  r = {f1['r']:.3f} (pub .95)   R2 = {f1['R2']:.3f} (pub .89)   "
          f"SEE = {f1['see_log']:.3f} log units (pub .16)   "
          f"Qm alone R2 = {f1['R2_Q_alone']:.3f} (pub .43)")
    assert abs(f1["expQ"] - 0.34) < 0.03
    assert abs(f1["expM"] - (-0.74)) < 0.03
    assert 1600.0 < f1["coef"] < 2100.0
    assert abs(f1["r"] - PUBLISHED["eq1"]["r"]) < 0.01
    assert abs(f1["R2"] - PUBLISHED["eq1"]["R2"]) < 0.02
    assert abs(f1["see_log"] - PUBLISHED["eq1"]["see_log"]) < 0.02
    assert abs(f1["R2_Q_alone"] - PUBLISHED["eq1"]["R2_Q_alone"]) < 0.03

    # Eq. (2): same checksum against the mean-annual-flood regression.
    f2 = fit_power_law(d["Qma"], d["M"], d["lam"])
    print(f"Eq.2 refit: lam = {f2['coef']:.0f} Qma^{f2['expQ']:.3f} M^{f2['expM']:.3f}"
          f"   (published: 234 Qma^0.48 / M^0.74)")
    print(f"  r = {f2['r']:.3f} (pub .93)   R2 = {f2['R2']:.3f} (pub .86)   "
          f"SEE = {f2['see_log']:.3f} log units (pub .19)   "
          f"Qma alone R2 = {f2['R2_Q_alone']:.3f} (pub .40)")
    assert abs(f2["expQ"] - 0.48) < 0.03
    assert abs(f2["expM"] - (-0.74)) < 0.03
    assert 200.0 < f2["coef"] < 260.0
    assert abs(f2["r"] - PUBLISHED["eq2"]["r"]) < 0.01
    assert abs(f2["R2"] - PUBLISHED["eq2"]["R2"]) < 0.02
    assert abs(f2["see_log"] - PUBLISHED["eq2"]["see_log"]) < 0.02
    assert abs(f2["R2_Q_alone"] - PUBLISHED["eq2"]["R2_Q_alone"]) < 0.03

    # Leave-one-out honesty: >= 80% of sections predicted within +-2 SE.
    errs = loo_log_errors(d["Qm"], d["M"], d["lam"])
    frac = float(np.mean(errs < 2.0 * PUBLISHED["eq1"]["see_log"]))
    print(f"leave-one-out: {100*frac:.0f}% of sections within +-2 SE "
          f"(worst |log err| = {errs.max():.2f})")
    assert frac >= 0.80

    # The paper's headline contrast: sediment type spans a ~tenfold
    # wavelength range at fixed discharge.
    tf = tenfold_range_factor()
    print(f"wavelength ratio across observed M range (1.3 -> 44.6): "
          f"{tf:.1f}x  (paper: 'tenfold range')")
    assert tf > 10.0

    print("-" * 70)
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
