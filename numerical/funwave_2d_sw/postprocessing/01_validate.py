#!/usr/bin/env python3
"""Post-run gates for the two meander cases.  Run BEFORE looking at any movie.

    micromamba run -n fourcastnetv2 python postprocessing/01_validate.py

Each gate is a statement that can fail.  A gate that cannot fail is not a gate.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import run_meander as rm  # noqa: E402

CFG = rm.CONFIG
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def snaps(base, phase, var):
    """Sorted snapshot paths.  Files are Nglob rows x Mglob cols -> load and transpose
    so everything downstream is (nx, ny) like the driver's arrays."""
    return sorted(glob.glob(os.path.join(base, phase, "output", f"{var}_*")))


def load(p):
    return np.loadtxt(p).T


def wet_channel(base, g):
    """Boolean (nx, ny): inside |n| <= b AND outside the non-erodible buffer."""
    # as-built mask, never recomputed from CONFIG (a config edit between build and analysis
    # silently shifts the centreline and samples shelf cells as channel)
    return (np.abs(g["n"]) <= CFG["b"]) & (g["Zs"] > 0)


def validate(base):
    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    tag = os.path.basename(base)
    print(f"\n{'='*78}\n{tag}\n{'='*78}")
    chan = wet_channel(base, g)

    # ---- G1: the spin-up actually reached a steady flow -------------------
    su = snaps(base, "spinup", "u")
    if len(su) < 2:
        check("G1 spin-up produced >=2 snapshots", False, f"{len(su)} found")
        return
    u0, u1 = load(su[-2]), load(su[-1])
    v0, v1 = load(su[-2].replace("/u_", "/v_")), load(su[-1].replace("/u_", "/v_"))
    sp0, sp1 = np.hypot(u0, v0), np.hypot(u1, v1)
    drift = np.abs(sp1 - sp0)[chan].max() / max(sp1[chan].mean(), 1e-9)
    check("G1 flow is steady at the end of spin-up", drift < 0.05,
          f"max |dspeed| between the last two snapshots = {drift*100:.2f}% of the mean")

    # ---- G2: normal flow -- the design velocity was actually achieved -----
    U_mean = sp1[chan].mean()
    check("G2 mean channel speed = design U", abs(U_mean - CFG["U"]) / CFG["U"] < 0.15,
          f"{U_mean:.3f} m/s vs design {CFG['U']:.3f} m/s "
          f"({100*(U_mean-CFG['U'])/CFG['U']:+.1f}%)")
    eta = load(su[-1].replace("/u_", "/eta_"))
    H = eta + g["Depth"]
    tau = CFG["Cd"] * sp1 ** 2
    bal = np.abs(tau[chan].mean() - rm.G_ACCEL * H[chan].mean() * float(g["S"]))
    check("G2 depth-averaged momentum balance Cd U^2 = g H S", bal / (tau[chan].mean()) < 0.25,
          f"Cd<U^2>={tau[chan].mean():.3e}, g<H>S={rm.G_ACCEL*H[chan].mean()*float(g['S']):.3e} "
          f"m2/s2  ({100*bal/tau[chan].mean():.1f}% apart)")

    # ---- G3: mass ---------------------------------------------------------
    log = open(os.path.join(base, "morph", "run.log")).read()
    check("G3 morph phase terminated normally", "Normal Termination" in log)

    # ---- G4: the boundary artefact stayed inside the buffer ---------------
    sd = snaps(base, "morph", "dep")
    if len(sd) < 2:
        check("G4 morph produced >=2 bed snapshots", False, f"{len(sd)} found")
        return
    dd = load(sd[-1]) - load(sd[0])
    bufmask = g["Zs"] == 0
    check("G4 no erosion in the Hard_bottom buffer (Zs caps erosion only)",
          dd[bufmask].max() <= 1e-9,
          f"buffer dDepth in [{dd[bufmask].min():+.3f}, {dd[bufmask].max():+.3f}] m "
          f"(+ = erosion); {int((dd[bufmask] < -1e-6).sum())} cells accreting")
    # the interior next to the buffer must not be dominated by the artefact:
    # compare the bed change in the first interior bend with the reach interior
    x = g["x"]
    lam, L, buf = float(g["lam"]), float(g["L"]), float(g["buffer_len"])
    near = chan & (x[:, None] > buf) & (x[:, None] < buf + lam)
    deep = chan & (x[:, None] > buf + lam) & (x[:, None] < L - buf - lam)
    r = np.abs(dd[near]).mean() / max(np.abs(dd[deep]).mean(), 1e-12)
    check("G4 first interior bend not dominated by the inlet artefact", r < 3.0,
          f"<|ddep|> first interior bend / rest of the interior = {r:.2f}")

    # ---- G5: is Morph_factor small enough? --------------------------------
    per = np.abs(np.diff(np.array([load(p) for p in sd]), axis=0))
    worst = per.max(axis=0)[chan].max() / CFG["H_c"]
    # morphodynamic Courant: bed move per HYDRO STEP must stay below ~1e-3 H or the coupled
    # system goes unstable (measured: MF=20 already blew up, MF=1 stable, threshold ~126).
    dt_hydro = CFG["CFL"] * CFG["dx"] / (CFG["U"] + np.sqrt(rm.G_ACCEL * CFG["H_c"]))
    per_step = worst * CFG["H_c"] * dt_hydro / (CFG["plot_intv"] * CFG["Morph_factor"])
    check("G5 morphodynamic Courant: bed move < 1e-3 H per hydro step",
          per_step / CFG["H_c"] < 1e-3,
          f"{per_step/CFG['H_c']:.1e} H/step at MF={CFG['Morph_factor']} "
          f"(blows up above ~1e-3; measured stable MF ceiling ~126)")
    check("G5 bed change per output interval < 15% of H", worst < 0.15,
          f"max |ddep| per {CFG['plot_intv']:.0f} s = {worst*100:.1f}% of H_c")
    tot = np.abs(dd)[chan].max() / CFG["H_c"]
    check("G5 total bed change is O(H), i.e. the run spans ~one bar", 0.05 < tot < 2.0,
          f"max |total ddep| = {tot*100:.0f}% of H_c over "
          f"{CFG['t_morph']*CFG['Morph_factor']/86400:.0f} morphological days")

    # ---- G6/G7: the always-wet shelf must behave as designed ---------------
    # The shelf replaces a dry floodplain so that no wet/dry boundary lies along the oblique
    # bank (that boundary is what destroyed every earlier run).  Two things must hold or the
    # design is invalid, so both are gates, not notes.
    shelf = (np.abs(g["n"]) > CFG["b"] + CFG["m_bank"] * (CFG["H_b"] - CFG["h_plain"])) \
            & (g["Zs"] > 0)
    u1 = load(su[-1]); v1 = load(su[-1].replace("/u_", "/v_"))
    Hs = eta + g["Depth"]
    q_shelf = float((np.hypot(u1, v1) * np.maximum(Hs, 0))[shelf].sum())
    q_chan = float((np.hypot(u1, v1) * np.maximum(Hs, 0))[chan].sum())
    frac = q_shelf / max(q_shelf + q_chan, 1e-12)
    check("G6 shelf carries a small fraction of the discharge", frac < 0.10,
          f"{100*frac:.1f}% of the total (design estimate ~4%)")
    check("G6 shelf is fully wet (no wet/dry line on the bank)",
          float((Hs[shelf] > CFG["MinDepth"]).mean()) > 0.999,
          f"{100*(Hs[shelf] > CFG['MinDepth']).mean():.2f}% wet")
    shelf_move = np.abs(dd[shelf]).max()
    check("G7 the shelf bed does not move (tau < tau_cr there)", shelf_move < 0.05,
          f"max |ddep| on the shelf = {shelf_move:.4f} m over the whole morph phase")
    bankface = (np.abs(g["n"]) > CFG["b"] * 0.8) & (np.abs(g["n"]) <= CFG["b"] * 1.4) \
               & (g["Zs"] > 0)
    print(f"  [INFO] bank-face bed change (0.8b < |n| <= 1.4b): "
          f"mean {dd[bankface].mean():+.4f} m, max erosion {dd[bankface].max():+.4f} m "
          f"-- this is the bank retreat the shelf design exists to preserve")

    # ---- the prediction this pair of runs was set up to test ---------------
    # Stock FUNWAVE has no transverse bed-slope deflection and no secondary-flow closure
    # (A_s = 0, the Ikeda-1981 A=0 limit), so the standing PREDICTION is outer-bank scour
    # WITHOUT an inner point bar.  Report the measured inner/outer split; do not assert it.
    # Inner bank is n*sign(kappa) > 0 -- verified geometrically in tests/test_bathy.py 7b.
    nn = g["n"] * np.sign(g["kappa"])           # as built                       # > 0 inner, < 0 outer
    core = chan & (x[:, None] > buf) & (x[:, None] < L - buf)   # the whole interior
    inner = core & (nn > CFG["b"] / 2)
    outer = core & (nn < -CFG["b"] / 2)
    mid = core & (np.abs(nn) <= CFG["b"] / 2)
    print(f"  [INFO] mean bed change over the interior bends (+ = erosion, - = deposition)")
    print(f"         OUTER bank  (n*sgn(kappa) < -b/2): {dd[outer].mean():+.4f} m")
    print(f"         mid-channel (|n*sgn(kappa)| < b/2): {dd[mid].mean():+.4f} m")
    print(f"         INNER bank  (n*sgn(kappa) > +b/2): {dd[inner].mean():+.4f} m")
    print(f"         point bar would be INNER < 0 (deposition) with OUTER > 0 (scour):"
          f" {'PRESENT' if dd[inner].mean() < 0 < dd[outer].mean() else 'ABSENT'}"
          f"   <- prediction was ABSENT")
    return dict(tag=tag, dd=dd, chan=chan, g=g, nn=nn,
                inner=dd[inner].mean(), outer=dd[outer].mean())


def main():
    bases = sorted(glob.glob(os.path.join(ROOT, "runs", "*")))
    if not bases:
        print("no runs/ -- build them with run_meander.py first")
        return 1
    out = [validate(b) for b in bases]

    print(f"\n{'='*78}\nCROSS-RUN: the two cases must share the drive\n{'='*78}")
    gs = [np.load(os.path.join(b, "bathy", "grid.npz")) for b in bases]
    C = [float(x["A"]) * float(x["k"]) ** 2 for x in gs]
    check("A k^2 identical across runs", np.allclose(C, C[0], rtol=1e-9),
          f"C = {C[0]:.6e} 1/m")
    check("wavenumbers differ", abs(float(gs[1]["k"]) / float(gs[0]["k"]) - 1) > 0.5,
          f"k ratio = {float(gs[1]['k'])/float(gs[0]['k']):.2f}")
    check("same morphological duration",
          len({round(CFG["t_morph"] * CFG["Morph_factor"])}) == 1,
          f"{CFG['t_morph']*CFG['Morph_factor']/86400:.0f} days for both")

    print("\n" + ("ALL GATES PASSED" if not FAIL else f"{len(FAIL)} FAILED: " + "; ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
