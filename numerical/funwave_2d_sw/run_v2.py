#!/usr/bin/env python3
"""v2 driver: unaccelerated (MF=1) meander morphodynamics with the channel-geometry closure fields
REBUILT from the evolving bed.

Why this exists.  Every spatial closure field the setup writes -- cd.txt (gap-1 drag modulation),
kappa.txt (bedload deflection), bedslope.txt (equilibrium tilt) -- is a function of the channel
coordinates (n, kappa) computed from the centreline.  In v1 those were built once at t=0.  With
bed and bank now one erodible surface at MF=1, the channel MIGRATES, so after a while those fields
describe a channel that is no longer there: the secondary-flow closure would be applied about a
stale axis, with the wrong sign wherever the channel has moved past it.

So the morph phase runs in CHUNKS.  After each chunk the centreline is re-detected from the current
bed, (n, kappa) are recomputed, the three closure fields are rewritten, and the next chunk is
hot-started.  This is the "fully decoupled" scheme the derivation recommends over an acceleration
factor: the flow is advanced to a quasi-steady state against a frozen bed, the bed is then advanced,
and no feedback instability can arise from the bed outrunning the flow.

    micromamba run -n fourcastnetv2 python run_v2.py --case 0        # B1
    micromamba run -n fourcastnetv2 python run_v2.py --case 1        # B2
"""
import argparse
import glob
import os
import pathlib
import shutil
import sys

import numpy as np

import run_meander as rm


# --------------------------------------------------------------------------- #
#  Centreline detection from an evolved bed
# --------------------------------------------------------------------------- #
def detect_centreline(bed, x, y, cfg, smooth_m=25.0):
    """Channel centreline y_c(x) recovered from a bed field.

    Method: BANK MIDPOINT.  At each down-valley station the bed is scanned across y for the two
    crossings of the bank-crest reference elevation, and the centreline is their midpoint.

    Why not the thalweg.  The deepest point is NOT the centreline of a bend: the point bar builds on
    the inner bank while the pool scours on the outer, so the thalweg migrates outward even when the
    channel as a whole has not moved.  Tracking it would manufacture a migration signal.  A
    depth-weighted centroid has the same bias for the same reason.  The bank midpoint is insensitive
    to that internal asymmetry -- it follows the two things that actually bound the channel.

    Returns y_c(x) with NaN where no clean pair of crossings exists (e.g. inside the buffers).
    """
    z = -bed                                             # bed elevation, up +
    z_ref = -0.5 * (cfg["H_b"] + cfg["h_plain"])         # mid bank-face elevation
    yc = np.full(x.size, np.nan)
    for i in range(x.size):
        zi = z[i]
        below = zi < z_ref                               # inside the channel
        if not below.any():
            continue
        idx = np.flatnonzero(below)
        # take the widest contiguous run (guards against a detached pond on the shelf)
        splits = np.flatnonzero(np.diff(idx) > 1)
        runs = np.split(idx, splits + 1)
        run = max(runs, key=len)
        if run.size < 3:
            continue
        jl, jr = run[0], run[-1]
        # sub-cell crossing on each side by linear interpolation of z through z_ref
        def cross(j_in, j_out):
            if j_out < 0 or j_out >= y.size:
                return y[j_in]
            z1, z2 = zi[j_out], zi[j_in]
            if z1 == z2:
                return y[j_in]
            f = (z_ref - z1) / (z2 - z1)
            return y[j_out] + f * (y[j_in] - y[j_out])
        yc[i] = 0.5 * (cross(jl, jl - 1) + cross(jr, jr + 1))
    # Smooth along x: the detection is per-column and can be noisy at the cell scale on an eroded
    # bed, while the centreline is a large-scale object.  Boxcar over smooth_m, NaN-aware.
    #
    # VALIDATED against the analytic curve on the as-built bathymetry (where the answer is known).
    # The sub-cell crossing interpolation above is near-exact by itself, and smoothing is the ONLY
    # error source -- it flattens real curvature once the window is a sizeable fraction of lam:
    #
    #   smooth_m      B1 (lam=780)  RMS / max      B2 (lam=1560) RMS / max
    #        0 m        0.186 / 0.305 m              0.012 / 0.021 m
    #       25 m        0.424 / 0.775 m              0.175 / 0.259 m
    #      100 m        1.156 / 1.595 m              0.304 / 0.434 m
    #      200 m        3.894 / 5.147 m              0.986 / 1.350 m
    #
    # 25 m (3% of B1's wavelength) keeps the error well inside one cell (dx = 2.5 m) while still
    # suppressing grid-scale noise.  Do NOT raise it toward a tenth of a wavelength.
    w = max(1, int(round(smooth_m / (x[1] - x[0]))))
    if w > 1:
        good = np.isfinite(yc).astype(float)
        filled = np.where(np.isfinite(yc), yc, 0.0)
        ker = np.ones(w) / w
        num = np.convolve(filled, ker, mode="same")
        den = np.convolve(good, ker, mode="same")
        yc = np.where(den > 0.2, num / np.maximum(den, 1e-9), np.nan)
    return yc


def coords_from_centreline(X, Y, x, yc, cfg):
    """(n, kappa) for an ARBITRARY centreline polyline y_c(x).

    rm.channel_coords() regenerates the ANALYTIC curve from (lam, C0) and so cannot see a migrated
    channel.  Here the centreline is whatever was detected, so n and kappa are built from it
    directly: n by signed nearest-point distance, kappa from the curve's own derivatives.
    """
    ok = np.isfinite(yc)
    xs, ys = x[ok], yc[ok]
    if xs.size < 5:
        raise RuntimeError("centreline detection produced too few valid stations")
    # resample onto a uniform, dense arc-length parameterisation
    dx = np.gradient(xs); dy = np.gradient(ys)
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds[1:])])
    tx, ty = dx / np.maximum(ds, 1e-12), dy / np.maximum(ds, 1e-12)
    # curvature kappa = (x' y'' - y' x'') / (x'^2+y'^2)^{3/2}, differentiated in s
    d2x = np.gradient(tx, s, edge_order=2)
    d2y = np.gradient(ty, s, edge_order=2)
    kap_c = tx * d2y - ty * d2x
    # signed cross-channel offset by nearest point on the polyline
    P = np.stack([xs, ys], axis=1)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    # chunked nearest-neighbour to keep memory bounded
    n_out = np.empty(pts.shape[0]); k_out = np.empty(pts.shape[0])
    CH = 200000
    for a in range(0, pts.shape[0], CH):
        q = pts[a:a + CH]
        d2 = ((q[:, None, 0] - P[None, :, 0]) ** 2 + (q[:, None, 1] - P[None, :, 1]) ** 2)
        j = np.argmin(d2, axis=1)
        dxq = q[:, 0] - P[j, 0]; dyq = q[:, 1] - P[j, 1]
        n_out[a:a + CH] = -dxq * ty[j] + dyq * tx[j]     # left-positive normal offset
        k_out[a:a + CH] = kap_c[j]
    return n_out.reshape(X.shape), k_out.reshape(X.shape)


def write_closure_fields(base, n, kap, cfg, bed):
    """Rewrite cd.txt / kappa.txt / bedslope.txt for the CURRENT (n, kappa) and the CURRENT bed.
    Mirrors write_case, including the depth-dependent log-law Cd base -- so the momentum drag tracks
    the evolving depth at chunk cadence, matching the sediment module."""
    b, toe = cfg["b"], cfg["b"] + cfg["m_bank"] * (cfg["H_b"] - cfg["h_plain"])
    A_ik = cfg.get("A_ikeda", 2.89)
    taper = np.clip((toe - np.abs(n)) / (toe - b), 0.0, 1.0)
    if cfg.get("SecondaryFlow"):
        cd = rm.loglaw_cd(bed, cfg) * np.clip(1.0 + A_ik * kap * n * taper, 0.2, 1.8)
        fp = np.clip((np.abs(n) - toe) / (4.0 * cfg["dx"]), 0.0, 1.0)
        cd = cd * (1.0 + (cfg.get("Cd_floodplain_mult", 30.0) - 1.0) * fp)
        np.savetxt(os.path.join(base, "bathy", "cd.txt"), cd.T, fmt="%14.7e")
    if cfg.get("SecondaryBedload"):
        np.savetxt(os.path.join(base, "bathy", "kappa.txt"), (kap * taper).T, fmt="%14.7e")
    if cfg.get("BedSlopeDeflection"):
        zbt = np.where(np.abs(n) <= b, -A_ik * kap * cfg["H_c"] * n, 0.0)
        np.savetxt(os.path.join(base, "bathy", "bedslope.txt"), zbt.T, fmt="%14.7e")


# --------------------------------------------------------------------------- #
#  Diagnostics / gates
# --------------------------------------------------------------------------- #
def gates(base, phase, cfg, n):
    """Return a dict of the v2 gate quantities from the latest snapshot of `phase`."""
    ld = lambda p: np.loadtxt(p).T
    us = [p for p in sorted(glob.glob(f"{base}/{phase}/output/u_*")) if "99999" not in p]
    if not us:
        return {}
    u = ld(us[-1]); v = ld(us[-1].replace("/u_", "/v_"))
    eta = ld(us[-1].replace("/u_", "/eta_"))
    dep = us[-1].replace("/u_", "/dep_")
    bed = ld(dep) if os.path.exists(dep) else ld(f"{base}/bathy/depth.txt")
    H = np.maximum(bed + eta, 1e-6)
    spd = np.hypot(u, v)
    wet = spd > 1e-6
    Fr = np.where(wet, spd / np.sqrt(9.81 * H), 0.0)
    chan = np.abs(n) <= cfg["b"]
    return dict(Fr_max=float(Fr.max()), Fr_chan=float(Fr[chan & wet].mean()) if (chan & wet).any() else np.nan,
                n_super=int((Fr > 1.0).sum()), H_min=float(H[chan & wet].min()) if (chan & wet).any() else np.nan,
                U_chan=float(spd[chan & wet].mean()) if (chan & wet).any() else np.nan)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, required=True, help="index into run_meander.RUNS")
    ap.add_argument("--chunks", type=int, default=10, help="morph chunks (fields rebuilt between)")
    ap.add_argument("--no-rebuild", action="store_true", help="disable geometry rebuild (control run)")
    args = ap.parse_args()

    run = rm.RUNS[args.case]
    cfg = rm.cfg_for(run)
    lam = run["lam"]
    assert cfg["Morph_factor"] <= 10, \
        f"MF={cfg['Morph_factor']} exceeds the T_bar/T_adjust~65 quasi-steady ceiling (use <=10)"

    tag, meta, t_spin, nr = rm.write_case(lam, cfg)
    base = rm.case_base(cfg, tag)
    print(f"[{run['tag']}] {tag}\n  spin-up {t_spin:.0f}s + morph {cfg['t_morph']:.0f}s (MF=1), "
          f"{nr} ranks, {meta['nx']}x{meta['ny']}", flush=True)

    g = np.load(os.path.join(base, "bathy", "grid.npz"))
    x, y = g["x"], g["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    n0 = g["n"]

    # ---- phase 1: rigid-bed spin-up ---------------------------------------
    if not rm.launch(base, "spinup", nr, cfg):
        print("SPINUP FAILED", flush=True); sys.exit(1)
    print(f"  spin-up gates: {gates(base, 'spinup', cfg, n0)}", flush=True)

    # ---- phase 2: morph in chunks, rebuilding the closure fields between ---
    snaps = [p for p in sorted(glob.glob(base + "/spinup/output/u_*")) if "99999" not in p]
    last = os.path.basename(snaps[-1]).split("_")[1]
    tpl = pathlib.Path(base + "/morph/input.txt").read_text()
    t_chunk = cfg["t_morph"] / args.chunks
    ini_from = ("../spinup/output", last)

    # The bed must be CARRIED ACROSS restarts.  FUNWAVE computes Depth = Depth_ini + Zb*MF and
    # re-reads DEPTH_FILE on every start, resetting Zb to 0 -- so leaving DEPTH_FILE pointed at the
    # original bathymetry silently discards each chunk's morphology.  (Measured: with the original
    # file, chunks 1-5 all ended at max|dep - depth.txt| = 0.44714 m instead of accumulating, and
    # dep_first == depth.txt every time.  Every chunk logged "morph: OK".)  So after each chunk the
    # evolved bed is written to bathy/depth_cur.txt and DEPTH_FILE is repointed at it.
    # bathy/depth.txt is left untouched as the t=0 reference the diagnostics difference against.
    depth_cur = os.path.join(base, "bathy", "depth_cur.txt")

    for ic in range(args.chunks):
        txt = tpl
        for vv in ("eta", "u", "v"):
            txt = txt.replace(f"@INI@/{vv}.txt", f"{ini_from[0]}/{vv}_{ini_from[1]}")
        txt = txt.replace(f"TOTAL_TIME = {cfg['t_morph']}", f"TOTAL_TIME = {t_chunk}")
        if ic > 0:
            assert os.path.exists(depth_cur), "evolved bed missing; refusing to restart from t=0 bed"
            txt = txt.replace("DEPTH_FILE = ../bathy/depth.txt",
                              "DEPTH_FILE = ../bathy/depth_cur.txt")
        pathlib.Path(base + "/morph/input.txt").write_text(txt)

        ok = rm.launch(base, "morph", nr, cfg)
        outs = [p for p in sorted(glob.glob(base + "/morph/output/u_*")) if "99999" not in p]
        if not outs:
            print(f"  chunk {ic}: no output", flush=True); break
        # archive this chunk's output so it is not overwritten by the next
        keep = os.path.join(base, "morph", f"chunk{ic:02d}")
        os.makedirs(keep, exist_ok=True)
        for f in glob.glob(base + "/morph/output/*"):
            shutil.copy2(f, keep)

        last = os.path.basename(outs[-1]).split("_")[1]
        ini_from = ("output", last)

        # carry the evolved bed forward (see the DEPTH_FILE note above)
        depf = base + f"/morph/output/dep_{last}"
        if os.path.exists(depf):
            shutil.copy2(depf, depth_cur)
            z_now = np.loadtxt(depf); z_ini = np.loadtxt(base + "/bathy/depth.txt")
            dz_max = float(np.abs(z_now - z_ini).max())
        else:
            dz_max = float("nan")
        gd = gates(base, "morph", cfg, n0)
        gd["dz_max_cum"] = round(dz_max, 4)      # MUST grow chunk to chunk, else the bed is resetting
        print(f"  chunk {ic+1}/{args.chunks} t={(ic+1)*t_chunk:.0f}s {'OK' if ok else 'STOPPED'} {gd}", flush=True)

        # -- rebuild the channel geometry from the CURRENT bed --------------
        if not args.no_rebuild:
            depf = base + f"/morph/output/dep_{last}"
            if os.path.exists(depf):
                bed = np.loadtxt(depf).T
                yc = detect_centreline(bed, x, y, cfg)
                try:
                    n_new, k_new = coords_from_centreline(X, Y, x, yc, cfg)
                    write_closure_fields(base, n_new, k_new, cfg, bed)   # bed = current Depth (still-water)
                    shift = np.nanmean(np.abs(yc - np.interp(x, x, np.where(np.isfinite(yc), yc, 0.0))))
                    print(f"    rebuilt closure fields; centreline valid on "
                          f"{100*np.isfinite(yc).mean():.0f}% of columns", flush=True)
                except RuntimeError as e:
                    print(f"    rebuild SKIPPED: {e}", flush=True)
        if not ok:
            break

    consolidate(base)
    print(f"[{run['tag']}] done", flush=True)


def consolidate(base):
    """Merge the per-chunk archives into ONE continuous series in morph/output/.

    FUNWAVE restarts its snapshot counter at 1 for every chunk, so chunk N would overwrite
    chunk N-1.  Each chunk was archived to morph/chunkNN/; here they are re-emitted into
    morph/output/ with a single global index so every postprocessing script -- all of which glob
    morph/output/<var>_##### -- sees the complete time series with no changes.
    """
    chunks = sorted(glob.glob(os.path.join(base, "morph", "chunk*")))
    if not chunks:
        return
    out = os.path.join(base, "morph", "output")
    for f in glob.glob(os.path.join(out, "*")):
        os.remove(f)
    k = 0
    for ch in chunks:
        idxs = sorted({os.path.basename(p).split("_")[1]
                       for p in glob.glob(os.path.join(ch, "eta_*")) if "99999" not in p})
        for idx in idxs:
            k += 1
            for var in ("eta", "u", "v", "dep", "mask"):
                src = os.path.join(ch, f"{var}_{idx}")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(out, f"{var}_{k:05d}"))
    print(f"  consolidated {len(chunks)} chunks -> {k} frames in morph/output/", flush=True)


if __name__ == "__main__":
    main()
