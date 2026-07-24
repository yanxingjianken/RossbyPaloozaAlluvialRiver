#!/usr/bin/env python3
"""Gates for the Thetis meander package.

Split into two tiers so the geometry/reduction tier can run without Firedrake:

  tier 1 (numpy/scipy only) -- geometry, base state, shared-helper checksum,
                               Ikeda cross-check, sw_note closed form
  tier 2 (needs Firedrake)  -- mesh warp, bathymetry sign, NULL-REBUILD

The null-rebuild test is the one that protects the whole morphological loop:
rebuilding mesh+solver with an *unchanged* centreline must leave the state
bit-identical, because the DOF copy is only legitimate while the topology is
fixed.

    micromamba run -n fourcastnetv2 python tests/test_setup.py      # tier 1
    micromamba run -n firedrake     python tests/test_setup.py      # tier 1 + 2
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geometry as geo  # noqa: E402

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------------------- #
#  Tier 1
# --------------------------------------------------------------------------- #
def tier1():
    print("\n--- tier 1: geometry / base state / reduction (no Firedrake) ---")

    # geometry.py and sw_note.py self-tests must pass in their own right
    for mod in ("geometry.py", "sw_note.py"):
        r = subprocess.run([sys.executable, os.path.join(HERE, mod)],
                           capture_output=True, text=True)
        check(f"{mod} self-test", r.returncode == 0,
              "" if r.returncode == 0 else r.stdout[-400:])

    d = geo.build_design(geo.Config())

    # --- the Ikeda design cross-check, from the VERIFIED library -----------
    sys.path.insert(0, os.path.join(HERE, "..", "ikeda_1981"))
    from ikeda_lib import growth_rate, k_OM  # noqa: E402
    kom = float(k_OM(Cf=d.cfg.Cf, A=d.cfg.A_ikeda, F=d.cfg.F_ref))
    k4, k8 = geo.wavenumber_of(4, d), geo.wavenumber_of(8, d)
    a = dict(Cf=d.cfg.Cf, A=d.cfg.A_ikeda, F=d.cfg.F_ref)
    check("m=4 is at Ikeda k_OM", abs(k4 - kom) / kom < 1e-12)
    check("m=8 is past the cutoff", k8 > d.k_c, f"k8/k_c={k8 / d.k_c:.3f}")
    check("linear theory: alpha0(m=4) > 0 > alpha0(m=8)",
          growth_rate(k4, **a) > 0 > growth_rate(k8, **a),
          f"{growth_rate(k4, **a) / d.cfg.Cf**2:+.3e} / "
          f"{growth_rate(k8, **a) / d.cfg.Cf**2:+.3e}  (units Cf^2)")

    # --- ubar quadratic, H quartic, H/ubar^2 constant ----------------------
    nt = np.linspace(-1, 1, 4001)
    u, H = geo.base_velocity(nt, d), geo.base_depth(nt, d)
    r2u = np.max(np.abs(u - np.polyval(np.polyfit(nt, u, 2), nt)))
    r4H = np.max(np.abs(H - np.polyval(np.polyfit(nt, H, 4), nt)))
    r2H = np.max(np.abs(H - np.polyval(np.polyfit(nt, H, 2), nt)))
    check("ubar is exactly quadratic (prescribed)", r2u < 1e-14, f"res {r2u:.1e}")
    check("H is quartic, NOT quadratic", r4H < 1e-13 and r2H > 1e-3,
          f"deg4 {r4H:.1e}, deg2 {r2H:.1e}")
    check("H / ubar^2 is constant (exact steady balance)",
          np.ptp(H / u**2) < 1e-13, f"ptp {np.ptp(H / u**2):.1e}")

    # --- bed frozen in time, falling downstream ----------------------------
    z0 = geo.bed_elevation(0.0, nt, d)
    zL = geo.bed_elevation(d.L, nt, d)
    check("bed falls downstream at the valley slope",
          abs((z0 - zL).mean() - d.I * d.L) < 1e-9)
    check("bed cross-section identical at every x",
          np.max(np.abs((z0 - z0.mean()) - (zL - zL.mean()))) < 1e-12)

    # --- entry reach long enough to forget the inlet -----------------------
    Hbar = geo.width_mean(geo.base_depth, d)
    Ladj = Hbar / (2.0 * d.cfg.Cf)
    check("entry reach >= 10 friction adjustment lengths",
          d.L_in / Ladj >= 10.0, f"{d.L_in / Ladj:.1f} lengths")

    # --- shared helper block byte-identical across packages ---------------
    def block_md5(path):
        txt = open(path).read()
        m = re.search(r"# === shared helper block v1.*?# === end shared helper block ===",
                      txt, re.S)
        return hashlib.md5(m.group(0).encode()).hexdigest() if m else None

    mine = block_md5(os.path.join(HERE, "postprocessing", "pp_lib.py"))
    ref = block_md5(os.path.join(HERE, "..", "deliverable1_noboru_model",
                                 "postprocessing", "pp_lib.py"))
    check("shared helper block byte-identical to deliverable1",
          mine is not None and mine == ref, f"{mine} vs {ref}")

    # --- no synthetic data anywhere ---------------------------------------
    # Bracket the literal so the pattern does not match THIS line -- the same
    # self-match trap as `pgrep -f script.sh` matching its own command line.
    r = subprocess.run(["grep", "-rn", "np[.]random", HERE, "--include=*.py"],
                       capture_output=True, text=True)
    check("no synthetic random data in the package", r.returncode != 0, r.stdout.strip()[:200])


# --------------------------------------------------------------------------- #
#  Tier 2
# --------------------------------------------------------------------------- #
def tier2():
    try:
        from thetis import Function, get_functionspace  # noqa: F401
    except Exception as e:                                    # noqa: BLE001
        print(f"\n--- tier 2 SKIPPED (no Firedrake): {type(e).__name__} ---")
        return False

    print("\n--- tier 2: mesh warp / bathymetry sign / null rebuild ---")
    import meander_thetis as mt  # noqa: E402

    cfg = dict(mt.CONFIG)
    d = geo.build_design(geo.Config(n_wave=cfg["n_wave"], A_ikeda=cfg["A_ikeda"]))
    x_ref = np.linspace(0.0, d.L, d.nx + 1)
    yN, yS = geo.initial_banks(x_ref, d)

    ref = mt.Reference(d)
    mesh = ref.warp(yN, yS, x_ref)
    xy = mesh.coordinates.dat.data_ro
    check("mesh spans the domain in x",
          abs(xy[:, 0].min()) < 1e-9 and abs(xy[:, 0].max() - d.L) < 1e-6)
    check("mesh y within the banks",
          xy[:, 1].max() <= yN.max() + 1e-6 and xy[:, 1].min() >= yS.min() - 1e-6)

    solver, ntil = mt.make_solver(mesh, d, yN, yS, x_ref, cfg, ref)

    # bathymetry sign: Thetis is POSITIVE DOWNWARD, so the thalweg (deepest
    # water, mid-channel) must carry the LARGEST bathymetry value.
    bathy = solver.fields.bathymetry_2d.dat.data_ro
    y = mesh.coordinates.dat.data_ro[:, 1]
    x = mesh.coordinates.dat.data_ro[:, 0]
    mid = np.abs(y - np.interp(x, x_ref, geo.centreline(yN, yS))) < 0.15 * d.b
    edge = np.abs(y - np.interp(x, x_ref, geo.centreline(yN, yS))) > 0.85 * d.b
    detrended = bathy - d.I * x                       # remove the valley slope
    check("bathymetry positive-downward: thalweg deeper than the banks",
          detrended[mid].mean() > detrended[edge].mean(),
          f"mid {detrended[mid].mean():.4f} vs edge {detrended[edge].mean():.4f} m")

    # --- NULL REBUILD: unchanged banks must leave the state untouched ------
    uv0 = solver.fields.solution_2d.subfunctions[0].dat.data_ro.copy()
    el0 = solver.fields.solution_2d.subfunctions[1].dat.data_ro.copy()
    solver.timestepper.advance(0.0, update_forcings=None)
    uv1 = solver.fields.solution_2d.subfunctions[0].dat.data_ro.copy()
    el1 = solver.fields.solution_2d.subfunctions[1].dat.data_ro.copy()

    mesh2 = ref.warp(yN, yS, x_ref)                    # SAME banks
    solver2, _ = mt.make_solver(mesh2, d, yN, yS, x_ref, cfg, ref)
    mt.carry_state(solver, solver2)
    uv2 = solver2.fields.solution_2d.subfunctions[0].dat.data_ro.copy()
    el2 = solver2.fields.solution_2d.subfunctions[1].dat.data_ro.copy()

    check("null rebuild preserves uv exactly",
          np.max(np.abs(uv2 - uv1)) == 0.0, f"max diff {np.max(np.abs(uv2 - uv1)):.2e}")
    check("null rebuild preserves elev exactly",
          np.max(np.abs(el2 - el1)) == 0.0, f"max diff {np.max(np.abs(el2 - el1)):.2e}")
    check("null rebuild reproduces the same mesh coordinates",
          np.max(np.abs(mesh2.coordinates.dat.data_ro
                        - mesh.coordinates.dat.data_ro)) < 1e-12)

    # one more step on each must agree -- proves the rebuilt solver is equivalent
    solver.timestepper.advance(cfg["dt"], update_forcings=None)
    solver2.timestepper.advance(cfg["dt"], update_forcings=None)
    d1 = solver.fields.solution_2d.subfunctions[0].dat.data_ro
    d2 = solver2.fields.solution_2d.subfunctions[0].dat.data_ro
    rel = np.max(np.abs(d2 - d1)) / max(np.max(np.abs(d1)), 1e-30)
    check("rebuilt solver takes the same next step", rel < 1e-12, f"rel {rel:.2e}")

    check("spin-up stayed finite", np.all(np.isfinite(uv1)) and np.all(np.isfinite(el1)))
    return True


def main() -> int:
    print("=" * 74)
    print("test_setup.py -- Thetis meander package gates")
    print("=" * 74)
    tier1()
    ran2 = tier2()
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("ALL CHECKS PASS" + ("" if ran2 else "  (tier 2 skipped -- no Firedrake)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
