#!/usr/bin/env python3
"""Reduction gate: Thetis must reproduce the corrected SW note in its own regime.

This is the test that makes "code and PDF are consistent" a *measurable* claim
rather than a prose assertion.  Without it, `sw_note.py` and `meander_thetis.py`
are two independent programs that merely cite the same document.

Regime forced here (the note's own assumptions, §2.3 of docs/model.md):
  * flat jet  -> uniform base velocity -> UNIFORM depth, i.e. a flat bed;
  * small curvature (linear response);
  * steady    -> run to a steady state, banks held fixed;
  * one streamwise Fourier mode.

Comparison: demodulate the Thetis transverse velocity at the fundamental
wavenumber over the meander reach, giving a complex profile vhat(ntil), and
compare its *shape* against `sw_note.solve_mode`.  Overall amplitude is not
compared -- the note's eps1/eps2 are free normalisation scales, so only the
profile shape and the cross-channel phase structure are physically determined.

    micromamba run -n firedrake python tests/test_sw_note.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geometry as geo  # noqa: E402
import sw_note as sn  # noqa: E402

FAILS: list[str] = []
TOL_SHAPE = 0.15          # 15% on the normalised transverse profile


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def demodulate(x, ntil_grid, field_2d, k, x0, x1):
    """Complex amplitude of `field_2d(x, ntil)` at streamwise wavenumber k.

    field_2d has shape (n_ntil, n_x).  Returns the complex profile over ntil,
    computed only on [x0, x1] (the clean meander reach).
    """
    sel = (x >= x0) & (x <= x1)
    xs = x[sel]
    w = np.exp(-1j * k * xs)
    # projection onto exp(+i k x) over the window, normalised
    return 2.0 * (field_2d[:, sel] * w[None, :]).mean(axis=1)


def main() -> int:
    print("=" * 74)
    print("test_sw_note.py -- Thetis vs the corrected SW note (28)-(30)")
    print("=" * 74)

    try:
        from thetis import Function, VectorFunctionSpace, get_functionspace  # noqa: F401
    except Exception as e:                                     # noqa: BLE001
        print(f"SKIPPED: Firedrake not importable ({type(e).__name__})")
        return 0

    import meander_thetis as mt

    # ---- the note's regime -------------------------------------------------
    gcfg = geo.Config(
        n_wave=4,
        jet_ratio=0.0,            # flat jet -> uniform depth -> FLAT BED
        amp0_over_b=0.01,         # weak curvature: linear response
        A_ikeda=0.0,
    )
    d = geo.build_design(gcfg)
    cfg = dict(mt.CONFIG)
    cfg.update(n_wave=4, dt=2.0)

    H = geo.base_depth(np.linspace(-1, 1, 11), d)
    check("regime really is a flat bed", np.ptp(H) < 1e-12,
          f"ptp(H) = {np.ptp(H):.2e} m")

    x_ref = np.linspace(0.0, d.L, d.nx + 1)
    yN, yS = geo.initial_banks(x_ref, d)
    ref = mt.Reference(d)
    mesh = ref.warp(yN, yS, x_ref)
    solver, _ = mt.make_solver(mesh, d, yN, yS, x_ref, cfg, ref)

    # ---- run to steady state (banks fixed) --------------------------------
    Ubar = geo.width_mean(geo.base_velocity, d)
    n_steps = int(round(4.0 * d.L / Ubar / cfg["dt"]))       # 4 transits
    print(f"\n  running {n_steps} steps ({4.0 * d.L / Ubar:.0f} s = 4 transits) "
          f"to steady state ...")
    prev = None
    for i in range(n_steps):
        solver.timestepper.advance(i * cfg["dt"], update_forcings=None)
        if (i + 1) % max(1, n_steps // 4) == 0:
            uv = solver.fields.solution_2d.subfunctions[0].dat.data_ro
            cur = float(np.linalg.norm(uv))
            drift = abs(cur - prev) / cur if prev else np.nan
            print(f"    step {i + 1}/{n_steps}  |uv| = {cur:.6e}"
                  f"  drift = {drift:.2e}")
            prev = cur
    check("run stayed finite", np.all(np.isfinite(
        solver.fields.solution_2d.subfunctions[0].dat.data_ro)))
    check("reached a steady state (last-quarter drift < 1%)",
          np.isfinite(drift) and drift < 1e-2, f"drift = {drift:.2e}")

    # ---- sample on a (x, ntil) grid and demodulate -------------------------
    nx_s, nn_s = 400, 41
    xs = np.linspace(d.x_m0, d.x_m1, nx_s)
    nts = np.linspace(-0.96, 0.96, nn_s)          # avoid the wall trace
    cl = geo.centreline(yN, yS)
    b = geo.half_width(yN, yS)
    uvf = solver.fields.solution_2d.subfunctions[0]

    vn = np.zeros((nn_s, nx_s))
    dcdx = np.gradient(cl, x_ref, edge_order=2)
    for j, nt in enumerate(nts):
        yy = np.interp(xs, x_ref, cl) + nt * np.interp(xs, x_ref, b)
        pts = np.column_stack([xs, yy])
        vals = np.array(uvf.at(pts, dont_raise=True, tolerance=1e-6))
        vals = np.array([[np.nan, np.nan] if v is None else v for v in vals],
                        dtype=float)
        slope = np.interp(xs, x_ref, dcdx)
        tn = np.sqrt(1.0 + slope**2)
        # transverse (n) component in the channel frame
        vn[j] = (-vals[:, 0] * slope + vals[:, 1]) / tn
    check("sampling hit the mesh everywhere", np.all(np.isfinite(vn)))

    k_phys = 2.0 * np.pi / (d.L_m / cfg["n_wave"])
    vhat_th = demodulate(xs, nts, vn, k_phys, d.x_m0 + 0.5 * d.lam,
                         d.x_m1 - 0.5 * d.lam)

    # ---- the note's prediction --------------------------------------------
    p = sn.params_from_design(d, cfg["n_wave"])
    p = sn.NoteParams(**{**p.__dict__, "jet_ratio": 0.0})
    nt_note, _, vhat_note, _ = sn.solve_mode(p, n=401)
    vhat_note_i = np.interp(nts, nt_note, vhat_note.real) \
        + 1j * np.interp(nts, nt_note, vhat_note.imag)

    def shape(z):
        z = np.asarray(z)
        i = np.argmax(np.abs(z))
        return z / z[i]                       # normalise amplitude AND phase

    s_th, s_note = shape(vhat_th), shape(vhat_note_i)
    err = np.max(np.abs(s_th - s_note))
    corr = float(np.abs(np.vdot(s_th, s_note))
                 / (np.linalg.norm(s_th) * np.linalg.norm(s_note)))

    print(f"\n  |vhat| Thetis max = {np.max(np.abs(vhat_th)):.4e} m/s")
    print(f"  |vhat| note   max = {np.max(np.abs(vhat_note_i)):.4e} (note units)")
    check("transverse-profile SHAPE matches the note", err < TOL_SHAPE,
          f"max|dshape| = {err:.3f} (tol {TOL_SHAPE})")
    check("profiles are strongly correlated", corr > 0.97, f"|corr| = {corr:.4f}")
    check("both vanish at the walls",
          abs(s_th[0]) < 0.35 and abs(s_th[-1]) < 0.35,
          f"|s(-1)|={abs(s_th[0]):.3f} |s(+1)|={abs(s_th[-1]):.3f}")

    # ---- the epsilon table, computed --------------------------------------
    print("\n  epsilon table at the production design point:")
    dprod = geo.build_design(geo.Config())
    for m in (4, 8):
        t = sn.epsilon_table(dprod, m)
        print("    " + "  ".join(f"{k}={v:.4g}" for k, v in t.items()))

    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("ALL CHECKS PASS -- Thetis reduces to the corrected SW note")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
