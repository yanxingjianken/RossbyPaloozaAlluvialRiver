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

    panels = [("us", r"$u_s'$", res["us"], "RdBu_r"),
              ("un", r"$u_n'$", res["un"], "RdBu_r"),
              ("uv", r"momentum flux $\overline{u_s'u_n'}$", momf, "RdBu_r"),
              ("eta", r"$\eta'$ (free surface)", res["eta"], "PuOr_r")]

    # ONE fixed colour limit per field, from the final frame (never per-frame)
    vlims = {key: _fixed_vlim(fld) for key, _t, fld, _c in panels}

    # show the last ~x25 of growth so the movie is not mostly blank (a TIME WINDOW,
    # not a rescaling); everything inside it is drawn at the one fixed scale.
    zc_amp_all = np.array([np.max(np.abs(res["zc"][i])) for i in range(nfr)])
    i0 = int(np.argmax(zc_amp_all > zc_amp_all[-1] / 25.0))
    frames_idx = list(range(i0, nfr))

    fig, axs = plt.subplots(2, 3, figsize=(16, 6.4), dpi=110)
    axpl = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]]    # 4 planform panels
    axc = axs[1, 1]                                        # y-z cross-section
    axst = axs[1, 2]                                       # growth stats

    for ax, (key, title, fld, cmap) in zip(axpl, panels):
        vl = vlims[key]
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(-vl, vl), cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02,
                     label=title + "  (ONE fixed scale)")
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
        for ax, (key, title, fld, cmap) in zip(axpl, panels):
            ax.clear()
            vl = vlims[key]                                # ONE fixed scale, all frames
            ax.pcolormesh(X, Y, fld[i], cmap=cmap, vmin=-vl, vmax=vl,
                          shading="gouraud", rasterized=True)
            ax.plot(X[:, 0], Y[:, 0], "k", lw=1.4)         # the two banks = walls
            ax.plot(X[:, -1], Y[:, -1], "k", lw=1.4)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            # FIXED range, identical every frame = the whole river (no per-frame
            # autoscale); FREE aspect so it fills the box.
            ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)

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
        axst.text(0.02, 0.99,
                  f"bed $H$: {bed}    bank sinuosity $\\bar C$={Cb:.3g}\n"
                  f"$C_f$={Cf:.3g}    $U_0$={U0:.2g}    $\\Delta$={Dl:+.2g}"
                  f" (shear)    $F$={F:g}\n"
                  f"fastest mode: $k$={kfast:.2f}, $\\sigma$={sfast:+.3f}, $c$={cfast:+.2f}\n"
                  f"{diag_txt}\n"
                  f"$t$={ts[i]:.1f}    centreline $\\times{gain:.2g}$ (true)",
                  va="top", ha="left", fontsize=8.5, transform=axst.transAxes)
        # centreline WAVEFORM (shape only, per-frame normalised) -- this is where the
        # prescribed initial cos(k s) and its downstream march stay visible even while
        # the absolute amplitude is still ~1/15 of final in the panels above.
        ax3 = axst.inset_axes([0.14, 0.60, 0.78, 0.20])
        zc_i = res["zc"][i]
        ax3.plot(s, zc_i / max(np.max(np.abs(zc_i)), 1e-30), color="navy", lw=1.4)
        ax3.set_xlim(s[0], s[-1]); ax3.set_ylim(-1.3, 1.3)
        ax3.set_yticks([]); ax3.tick_params(labelsize=7)
        ax3.set_title(r"centreline $\zeta_c(s)$ — SHAPE only (per-frame norm)", fontsize=7)
        ax3.grid(alpha=0.3)
        # true growth (absolute)
        ax2 = axst.inset_axes([0.14, 0.10, 0.78, 0.32])
        ax2.semilogy(ts[:i + 1], zc_amp[:i + 1] / max(zc_amp[0], 1e-30), color="crimson", lw=2)
        ax2.set_xlim(ts[0], ts[-1]); ax2.set_ylim(0.5, max(2.0, zc_amp.max() / max(zc_amp[0], 1e-30) * 1.3))
        ax2.set_xlabel("t", fontsize=8); ax2.set_ylabel(r"meander $\times$", fontsize=8)
        ax2.tick_params(labelsize=7); ax2.grid(alpha=0.3, which="both")

        fig.suptitle(
            rf"SW meander $(s,n)\!\to$lab | bed $H$: {bed}, bank $\bar C$={Cb:.3g}, "
            rf"$C_f$={Cf:.3g}, $U_0$={U0:.2g}, $\Delta$={Dl:+.2g} "
            rf"| BROADBAND perturbation (all $k$ at once) "
            rf"| ABSOLUTE Eulerian: one fixed scale, true $e^{{\sigma t}}$",
            fontsize=10)
        return []

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    anim = manim.FuncAnimation(fig, draw, frames=frames_idx, blit=False)
    out = os.path.join(PP.FIG_DIR, f"momflux_eulerian_{tag}.mp4")
    anim.save(out, fps=14, dpi=110)
    draw(nfr - 1)
    fig.savefig(os.path.join(PP.FIG_DIR, f"momflux_eulerian_{tag}_preview.png"))
    plt.close(fig)
    print(f"wrote momflux_eulerian_{tag}.mp4  "
          f"(fastest converged k={kfast:.2f}, sigma={sfast:+.4f}, "
          f"{len(frames_idx)}/{nfr} frames, display gain G={G:.3g})")


def MD_bed(cfg, n):
    return PP.MD.bed_depth(0.0, n, cfg)


def main():
    args = sys.argv[1:]
    if args:
        tags = args
    else:                                   # every configuration on disk
        tags = [os.path.basename(p)[:-3]
                for p in sorted(glob.glob(os.path.join(PP.OUT_DIR, "run_*.h5")))]
    if not tags:
        raise SystemExit("no ../outputs/run_*.h5 -- run ../sw_sn_driver.py first")
    for t in tags:
        p = os.path.join(PP.OUT_DIR, t if t.endswith(".h5") else t + ".h5")
        render(p)


if __name__ == "__main__":
    main()
