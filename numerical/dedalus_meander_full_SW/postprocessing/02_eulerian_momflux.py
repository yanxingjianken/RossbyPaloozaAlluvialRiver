#!/usr/bin/env python3
"""02: fully-Eulerian momentum-flux movies -- one per PHYSICAL configuration.

Maps the (s,n) fields to the LAB-frame meandering channel (banks bound the flow).
6 panels: u_s', u_n', momentum flux u_s'u_n', free surface eta', a y-z (n-z)
cross-section (Ikeda Fig-2b: bed H(n)+jet+banks+free surface), and growth stats.

ABSOLUTE-EULERIAN convention -- the two display tricks are deliberately ABSENT:

- ONE fixed colour scale per field for the whole movie (NO per-frame normalisation),
  so the colours genuinely build up as e^{sigma t} and two movies are comparable;
- ONE fixed overall gain G for the whole movie (NO bank amplification).  The model
  is linear so the overall amplitude is a free constant; G is chosen once, from
  LINEAR VALIDITY (final |u'| = 15% of the base jet), and applied to every frame
  and every field alike;
- ONE fixed view (the whole river at final extent) for every frame, so the frame
  really is Eulerian and the small early waveform is not zoomed away.

The only per-frame normalisation anywhere is the small centreline-SHAPE inset, which
is labelled as such -- it exists so the waveform stays readable while the absolute
amplitude is still tiny.

The perturbation is broadband (every k at once), so a movie is titled by its physics
(bed H, bank sinuosity, C_f, U_0, Delta), never by "which wavelength was perturbed";
the fastest CONVERGED mode is reported in the stats panel.

    python 02_eulerian_momflux.py [run_tag ...]      # default: every run on disk
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import matplotlib.colors as mcolors

import pp_lib as PP


def _fixed_vlim(fld_all):
    """ONE fixed colour limit for the whole movie, from the FINAL frame.

    Absolute-Eulerian convention: every frame is drawn with the SAME scale, so
    the colours genuinely build up as e^{sigma t}.  (NO per-frame normalisation --
    that would rescale each frame to full and erase the growth.)
    """
    return max(np.percentile(np.abs(fld_all[-1]), 99.5), 1e-30)


def render(path):
    res = PP.load_run(path)
    a = res["attrs"]
    cfg = PP.cfg_from_attrs(a)
    tag = os.path.splitext(os.path.basename(path))[0].replace("run_", "")
    s, n = res["s"], res["n"]
    ts = res["t"]
    nfr = len(ts)
    b = float(a["b"])
    F = float(a["Froude"]); U0 = float(a["U0"]); Cf = float(a["Cf"])
    Dl = float(a["Delta"])          # jet excess = the cross-channel shear (channel-beta)
    Cb = float(a["bank_sinuosity"])
    bed = "flat" if float(a["cross_amp"]) == 0 else f"cross {float(a['cross_amp']):.2g}"
    km = float(a["kmeander"])
    # the perturbation is broadband; report the fastest CONVERGED mode instead of
    # "the" wavelength (there is no single seeded wavelength any more)
    if "disp_k" in res and np.any(res["disp_converged"] > 0):
        ok = res["disp_converged"] > 0
        i = int(np.argmax(np.where(ok, res["disp_sigma"], -np.inf)))
        kfast, sfast, cfast = (float(res["disp_k"][i]), float(res["disp_sigma"][i]),
                               float(res["disp_c"][i]))
    else:
        kfast = sfast = cfast = float("nan")
    # the decisive diagnostics belong ON the movie, not only in a log: T_shear is the
    # ONLY channel by which the mean-flow vorticity gradient can power a free vortical
    # wave, so its SIGN is what the whole exercise turns on.  div_ratio<<1 says balanced.
    if "diag_T_shear" in a and "diag_T_bend" in a:
        Ts, Tb = float(a["diag_T_shear"]), float(a["diag_T_bend"])
        ratio = Ts / max(abs(Tb), 1e-300)
        verdict = "mean flow is a SINK" if Ts <= 0 else "mean flow POWERS it"
        diag_txt = (rf"$T_{{\rm shear}}/|T_{{\rm bend}}|$={ratio:+.2f} ({verdict})"
                    + (rf"   $\|\delta'\|/\|\zeta'\|$={float(a['diag_div_ratio']):.3g}"
                       if "diag_div_ratio" in a else ""))
    else:
        diag_txt = ""
    zc_final = max(np.max(np.abs(res["zc"][-1])), 1e-30)
    # ABSOLUTE-EULERIAN scaling.  The model is LINEAR, so the overall amplitude is a
    # free constant (= the seed choice).  We fix it ONCE, then draw EVERY frame with
    # that same constant (display gain 1, no per-frame renormalisation) -> meander
    # and colours both grow with the true e^{sigma t}.
    #
    # The constant is set by LINEAR VALIDITY, not by what looks big: we scale so the
    # final perturbation velocity is LIN_FRAC of the base jet, i.e. |u'|/Ubar = 0.15.
    # (Scaling instead to a "nice" 0.5b meander would put |u'|~|Ubar| -- formally
    #  outside the linear regime the model is solving.)
    LIN_FRAC = 0.15
    us_final = max(np.percentile(np.abs(res["us"][-1]), 99.5), 1e-30)
    G = LIN_FRAC * float(np.max(np.abs(PP.MD.ubar_s(n, cfg)))) / us_final

    cbar_s = Cb * np.cos(km * s)
    momf = np.array([PP.momflux(res, i) for i in range(nfr)])
    zc_amp = np.array([np.max(np.abs(res["zc"][i])) for i in range(nfr)])

    # base profiles for the y-z cross-section
    Hn = MD_bed(cfg, n)
    Ubn = PP.MD.ubar_s(n, cfg)
    ebar = PP.MD.etabar(0.0, n, cfg)                       # base superelevation (n-profile ok)

    # ---- TOTAL flow = base + perturbation (what the river actually does) ------
    # The base state is invisible in a perturbation-only movie, so there is no way to
    # judge whether the disturbance is a ripple on the jet or has taken it over.  With
    # the single global gain G (chosen so the FINAL |u'| is LIN_FRAC of the jet), the
    # total-flow panels start as the bare parabolic jet and end with a 15% wobble --
    # which is the evolution, shown honestly on a linear scale.
    Ub2d = PP.MD.ubar_s(n, cfg)[None, :]                        # (1,Nn) -> broadcasts
    eb2d = PP.MD.etabar(s, n, cfg)                              # (Ns,Nn) superelevation
    tot_us = np.array([Ub2d + G * res["us"][i] for i in range(nfr)])
    tot_un = np.array([G * res["un"][i] for i in range(nfr)])   # Ubar_n = 0 by construction
    tot_spd = np.sqrt(tot_us ** 2 + tot_un ** 2)
    tot_eta = np.array([eb2d + G * res["eta"][i] for i in range(nfr)])

    # TOTAL panels: diverging about the BASE value is meaningless, so use a sequential
    # map on a fixed linear range covering base and final.
    tot_panels = [("tus", r"TOTAL $\bar U_s+u_s'$", tot_us, "viridis"),
                  ("tspd", r"TOTAL speed $|\mathbf{u}|$", tot_spd, "magma"),
                  ("teta", r"TOTAL surface $\bar\eta+\eta'$", tot_eta, "cividis")]
    tot_lims = {k: (float(np.min(f)), float(np.max(f))) for k, _t, f, _c in tot_panels}

    # PERTURBATION panels.  These grow as e^{sigma t} over many e-foldings, so a fixed
    # LINEAR scale leaves all but the last frame or two black -- which is exactly the
    # "nothing happens then it explodes" complaint.  Fix: keep ONE fixed mapping for
    # the whole movie (no per-frame renormalisation) but make it SYMLOG, so several
    # decades of growth are visible at once and the early broadband transient -- the
    # only part where the SHAPE actually changes -- is finally on screen.
    pert_panels = [("us", r"$u_s'$", res["us"], "RdBu_r"),
                   ("un", r"$u_n'$", res["un"], "RdBu_r"),
                   ("uv", r"momentum flux $u_s'u_n'$", momf, "RdBu_r"),
                   ("eta", r"$\eta'$ (free surface)", res["eta"], "PuOr_r")]
    vlims = {key: _fixed_vlim(fld) for key, _t, fld, _c in pert_panels}
    # How many decades the symlog must span is a property of THIS run, not a constant:
    # it has to cover the run's own amplitude gain, or the early frames fall inside the
    # linear (blank) core and we are back to "nothing happens, then it explodes".
    gain_tot = float(zc_amp[-1] / max(zc_amp[0], 1e-30))
    DECADES = float(np.clip(np.log10(max(gain_tot, 10.0)) + 1.0, 4.0, 12.0))
    norms = {k: mcolors.SymLogNorm(linthresh=vlims[k] * 10 ** (-DECADES),
                                   vmin=-vlims[k], vmax=vlims[k], base=10)
             for k, _t, _f, _c in pert_panels}

    # show EVERY frame: with symlog the early frames are no longer blank, so there is
    # no reason to throw away the first half of the run
    frames_idx = list(range(nfr))

    fig, axs = plt.subplots(3, 3, figsize=(16.5, 10.0), dpi=110)
    axtot = [axs[0, 0], axs[0, 1], axs[0, 2]]              # TOTAL flow
    axpl = [axs[1, 0], axs[1, 1], axs[1, 2], axs[2, 0]]    # perturbation
    axc = axs[2, 1]                                        # y-z cross-section
    axst = axs[2, 2]                                       # growth stats

    # colourbars: label with the SYMBOL only.  Repeating "(fixed SYMLOG, N decades)" on
    # every bar made the text run into the neighbouring panel; the convention belongs in
    # the suptitle, where it is stated once.
    for ax, (key, title, fld, cmap) in zip(axtot, tot_panels):
        lo, hi = tot_lims[key]
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    for ax, (key, title, fld, cmap) in zip(axpl, pert_panels):
        sm = plt.cm.ScalarMappable(norm=norms[key], cmap=cmap)
        sm.set_array([])
        # a symlog bar auto-ticks EVERY decade, which at ~9 decades per sign is an
        # unreadable stack of overlapping labels; thin it to ~4 decades per sign
        vl = vlims[key]
        dec = np.arange(0.0, DECADES + 1.0, max(1.0, np.ceil(DECADES / 4.0)))
        ticks = ([-vl * 10.0 ** -d for d in dec][::-1] + [0.0]
                 + [vl * 10.0 ** -d for d in dec[::-1]])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, ticks=ticks)
        cb.ax.tick_params(labelsize=6)
    # cross-section colourbar: ONE fixed range, added ONCE here (never inside draw(),
    # which FuncAnimation calls repeatedly -> would stack duplicate colourbars)
    u_lo = float(np.min(Ubn / Hn) - abs(G) * np.percentile(np.abs(res["us"][-1]), 99.5) / Hn.min())
    u_hi = float(np.max(Ubn / Hn) + abs(G) * np.percentile(np.abs(res["us"][-1]), 99.5) / Hn.min())
    smc = plt.cm.ScalarMappable(norm=plt.Normalize(u_lo, u_hi), cmap="viridis")
    smc.set_array([])
    fig.colorbar(smc, ax=axc, fraction=0.05, pad=0.02, label=r"$\bar u_s/h$ (ONE fixed scale)")

    def centerline_xy(i):
        return PP.centerline(s, cbar_s, zc=G * res["zc"][i])

    # FIXED view for every frame = the WHOLE river at its largest extent (final
    # frame).  Without this, ax.clear() lets matplotlib autoscale per frame, so the
    # view zooms as the meander grows -- which is not an Eulerian (fixed-frame) view
    # and makes the small early waveform unreadable.
    xcf, ycf, nxf, nyf = centerline_xy(nfr - 1)
    Xf = xcf[:, None] + n[None, :] * nxf[:, None]
    Yf = ycf[:, None] + n[None, :] * nyf[:, None]
    pad = 0.10 * (Yf.max() - Yf.min())
    XLIM = (Xf.min(), Xf.max())                        # the full along-channel reach
    YLIM = (Yf.min() - pad, Yf.max() + pad)

    def draw(i):
        xc, yc, nx, ny = centerline_xy(i)
        X = xc[:, None] + n[None, :] * nx[:, None]
        Y = yc[:, None] + n[None, :] * ny[:, None]
        def _strip(ax, fld_i, title, **kw):
            ax.clear()
            ax.pcolormesh(X, Y, fld_i, shading="gouraud", rasterized=True, **kw)
            ax.plot(X[:, 0], Y[:, 0], "k", lw=1.4)         # the two banks = walls
            ax.plot(X[:, -1], Y[:, -1], "k", lw=1.4)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            # FIXED range, identical every frame = the whole river (no per-frame
            # autoscale); FREE aspect so it fills the box.
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)

        for ax, (key, title, fld, cmap) in zip(axtot, tot_panels):
            lo, hi = tot_lims[key]
            _strip(ax, fld[i], title, cmap=cmap, vmin=lo, vmax=hi)
        for ax, (key, title, fld, cmap) in zip(axpl, pert_panels):
            _strip(ax, fld[i], title, cmap=cmap, norm=norms[key])

        # ---- y-z cross-section at the crest (max |zc|) ------------------------
        ic = int(np.argmax(np.abs(res["zc"][i])))
        axc.clear()
        # same single global scale G (seed choice) -- no extra amplification
        surf = ebar + G * res["eta"][i, ic]
        zg = np.linspace(-Hn.max() * 1.15, max(0.3, surf.max() + 0.2), 60)
        u_col = (Ubn / Hn) + G * res["us"][i, ic] / Hn     # depth-averaged jet + perturbation
        U2 = np.where((zg[:, None] > -Hn[None, :]) & (zg[:, None] < surf[None, :]),
                      u_col[None, :], np.nan)
        axc.pcolormesh(n, zg, U2, cmap="viridis", shading="auto",
                       vmin=u_lo, vmax=u_hi)          # same fixed range as its colourbar
        axc.plot(n, -Hn, color="saddlebrown", lw=2.2)
        axc.fill_between(n, -Hn, zg.min(), color="saddlebrown", alpha=0.25)
        axc.plot(n, surf, color="steelblue", lw=1.8)       # free surface
        axc.plot([-b, -b], [-Hn[0], surf[0]], color="k", lw=3)
        axc.plot([b, b], [-Hn[-1], surf[-1]], color="k", lw=3)
        axc.set_xlim(-1.35 * b, 1.35 * b); axc.set_ylim(zg.min(), zg.max())
        axc.set_xlabel(r"cross-channel $n$"); axc.set_ylabel(r"depth $z$")
        axc.set_title(r"$y$-$z$ cross-section (bed+jet+banks+$\eta$)", fontsize=9)

        # ---- growth stats ----------------------------------------------------
        axst.clear(); axst.axis("off")
        gain = zc_amp[i] / max(zc_amp[0], 1e-30)
        axst.text(0.01, 1.00,
                  f"bed $H$: {bed}   bank $\\bar C$={Cb:.3g}   $C_f$={Cf:.3g}   "
                  f"$U_0$={U0:.2g}   $\\Delta$={Dl:+.2g}   $F$={F:g}\n"
                  f"fastest mode $k$={kfast:.2f}, $\\sigma$={sfast:+.3f}, $c$={cfast:+.2f}"
                  f"    |    $t$={ts[i]:.1f},  $\\zeta_c\\times{gain:.2g}$\n"
                  f"{diag_txt}",
                  va="top", ha="left", fontsize=7.5, transform=axst.transAxes)
        # centreline WAVEFORM (shape only, per-frame normalised) -- this is where the
        # prescribed initial cos(k s) and its downstream march stay visible even while
        # the absolute amplitude is still ~1/15 of final in the panels above.
        ax3 = axst.inset_axes([0.13, 0.52, 0.80, 0.17])
        # BLUE = the meander itself, zeta_c(s): WHERE the channel centreline is, at this
        # instant, normalised so the shape stays readable while the amplitude grows.
        # It is a function of s (space), NOT of time.  Watching it slide downstream is
        # what the phase speed c means -- c is the Im part of the eigenvalue.
        zc_i = res["zc"][i]
        ax3.plot(s, zc_i / max(np.max(np.abs(zc_i)), 1e-30), color="navy", lw=1.4)
        ax3.set_xlim(s[0], s[-1]); ax3.set_ylim(-1.3, 1.3)
        ax3.set_yticks([]); ax3.tick_params(labelsize=7)
        ax3.set_xlabel(r"along-channel $s$", fontsize=7, labelpad=1)
        ax3.set_title(r"BLUE: meander shape $\zeta_c(s)$ — drift downstream $=c$ (Im part)",
                      fontsize=7)
        ax3.grid(alpha=0.3)
        # RED = amplitude vs TIME on a log axis.  Its SLOPE is sigma = the REAL part of
        # the growth rate (sigma>0 = growing).  It is NOT the imaginary part -- the
        # imaginary part is the migration shown by the blue curve above.  The dashed
        # reference line is exp(sigma t) with the fitted sigma, so the two can be
        # compared by eye: parallel => the mode has settled onto its eigenvalue.
        ax2 = axst.inset_axes([0.13, 0.06, 0.80, 0.26])
        gains = zc_amp / max(zc_amp[0], 1e-30)
        ax2.semilogy(ts[:i + 1], gains[:i + 1], color="crimson", lw=2, label="measured")
        if np.isfinite(sfast):
            ax2.semilogy(ts, gains[0] * np.exp(sfast * (ts - ts[0])), "k--", lw=1,
                         alpha=0.7, label=rf"$e^{{\sigma t}},\ \sigma$={sfast:+.3f}")
            ax2.legend(fontsize=6, loc="upper left", framealpha=0.6)
        ax2.set_xlim(ts[0], ts[-1]); ax2.set_ylim(0.5, max(2.0, gains.max() * 1.3))
        ax2.set_xlabel("time $t$", fontsize=8)
        ax2.set_ylabel(r"$|\zeta_c|\times$", fontsize=8)
        ax2.set_title(r"RED: amplitude vs $t$ (log) — SLOPE $=\sigma$ (Re part)", fontsize=7)
        ax2.tick_params(labelsize=7); ax2.grid(alpha=0.3, which="both")

        fig.suptitle(
            rf"shallow-water meander $(s,n)\!\to$ lab frame  |  bed $H$: {bed}, "
            rf"bank $\bar C$={Cb:.3g}, $C_f$={Cf:.3g}, $U_0$={U0:.2g}, "
            rf"$\Delta$={Dl:+.2g}  |  BROADBAND perturbation (every $k$ at once)"
            "\n"
            rf"row 1 = TOTAL flow (base $+$ perturbation), fixed linear scale   ·   "
            rf"rows 2–3 = perturbation only, fixed SYMLOG over {DECADES:.1f} decades"
            rf"   ·   NO per-frame renormalisation anywhere",
            fontsize=9.5)
        return []

    fig.tight_layout(rect=[0, 0, 1, 0.925])
    anim = manim.FuncAnimation(fig, draw, frames=frames_idx, blit=False)
    out = os.path.join(PP.FIG_DIR, f"momflux_eulerian_{tag}.mp4")
    anim.save(out, fps=10, dpi=110)
    plt.close(fig)
    print(f"wrote momflux_eulerian_{tag}.mp4  "
          f"(fastest converged k={kfast:.2f}, sigma={sfast:+.4f}, "
          f"{len(frames_idx)}/{nfr} frames, display gain G={G:.3g})")


def MD_bed(cfg, n):
    return PP.MD.bed_depth(n, cfg)      # bed is H(n): no s argument


def main():
    args = sys.argv[1:]
    if args:
        tags = args
    else:                                   # every configuration on disk
        tags = [os.path.basename(p)[:-3]
                for p in sorted(glob.glob(os.path.join(PP.OUT_DIR, "run_*.h5")))]
    if not tags:
        raise SystemExit("no ../outputs/run_*.h5 -- run ../sw_meander.py first")
    for t in tags:
        p = os.path.join(PP.OUT_DIR, t if t.endswith(".h5") else t + ".h5")
        render(p)


if __name__ == "__main__":
    main()
