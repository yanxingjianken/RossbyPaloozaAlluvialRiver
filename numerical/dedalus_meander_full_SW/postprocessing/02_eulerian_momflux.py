#!/usr/bin/env python3
"""02: fully-Eulerian momentum-flux movies (one per init bank wavelength).

Maps the (s,n) fields to the LAB-frame meandering channel (banks bound the flow).
6 panels: u_s', u_n', momentum flux u_s'u_n', free surface eta', a y-z (n-z)
cross-section (Ikeda Fig-2b: bed H(n)+jet+banks+free surface), and growth stats.

- colours are PER-FRAME normalised (with a small floor) so the pattern is visible
  throughout (it does not stay faint while the mode grows);
- the meander/bank displacement is amplified for display, so the BANK EROSION
  (the meander migrating/growing) is visible; the true growth is the x-gain counter;
- FREE aspect ratio (each planform panel fills its box).

    python 02_eulerian_momflux.py [run_tag ...]      # default: a spread of wavelengths
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
    k = float(a["kstar"]); F = float(a["Froude"])
    Cb = (float(a["A_bank"]) * float(a["kmeander"]) ** 2
          if a["Cbar_amp"] in ("None", b"None") else float(a["Cbar_amp"]))
    km = float(a["kmeander"])
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
    zc_disp = G * zc_final                                 # resulting meander, in b

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

    Xf, Yf, _, _ = centerline_xy(nfr - 1)

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
            # FREE aspect: fill the box (do NOT set_aspect('equal'))
            ax.margins(x=0.01, y=0.06)

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
        axst.plot([], [])
        ax2 = axst.inset_axes([0.12, 0.12, 0.8, 0.5])
        ax2.semilogy(ts[:i + 1], zc_amp[:i + 1] / max(zc_amp[0], 1e-30), color="crimson", lw=2)
        ax2.set_xlim(ts[0], ts[-1]); ax2.set_ylim(0.5, max(2.0, zc_amp.max() / max(zc_amp[0], 1e-30) * 1.3))
        ax2.set_xlabel("t"); ax2.set_ylabel(r"meander $\times$")
        ax2.grid(alpha=0.3, which="both")
        axst.text(0.02, 0.97,
                  f"$k={k:g}$   $F={F:g}$   $\\bar C={Cb:.2g}$\n"
                  f"$\\sigma_{{\\rm meas}}={float(a['sigma_meas']):+.3f}$   "
                  f"$c={float(a['c_meas']):+.2f}$\n"
                  f"$t={ts[i]:.1f}$   meander $\\times{gain:.2g}$",
                  va="top", ha="left", fontsize=10, transform=axst.transAxes)

        fig.suptitle(
            rf"SW meander $(s,n)\!\to$lab | $k={k:g}$ ($\lambda={2*np.pi/k:.1f}$), "
            rf"$F={F:g}$, $\bar C={Cb:.2g}$ | ABSOLUTE Eulerian: ONE fixed scale, "
            rf"true $e^{{\sigma t}}$ | $|u'|/\bar U={LIN_FRAC:g}$ "
            rf"$\Rightarrow$ meander ${zc_disp:.2f}b$",
            fontsize=10)
        return []

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    anim = manim.FuncAnimation(fig, draw, frames=frames_idx, blit=False)
    out = os.path.join(PP.FIG_DIR, f"momflux_eulerian_{tag}.mp4")
    anim.save(out, fps=14, dpi=110)
    draw(nfr - 1)
    fig.savefig(os.path.join(PP.FIG_DIR, f"momflux_eulerian_{tag}_preview.png"))
    plt.close(fig)
    print(f"wrote momflux_eulerian_{tag}.mp4  (k={k}, {nfr} frames)")


def MD_bed(cfg, n):
    return PP.MD.bed_depth(0.0, n, cfg)


def main():
    args = sys.argv[1:]
    if args:
        tags = args
    else:
        # a spread of init bank wavelengths (from the F=0.6 sweep, in the growth band)
        tags = []
        for kt in ("k0p44", "k0p63", "k0p82", "k1p01"):
            g = sorted(glob.glob(os.path.join(PP.OUT_DIR, f"run_{kt}_F0p60*.h5")))
            if g:
                tags.append(os.path.basename(g[0])[:-3])
        if not tags:
            g = sorted(glob.glob(os.path.join(PP.OUT_DIR, "run_*.h5")))
            tags = [os.path.basename(g[0])[:-3]] if g else []
    if not tags:
        raise SystemExit("no ../outputs/run_*.h5 -- run the driver/sweep first")
    for t in tags:
        p = os.path.join(PP.OUT_DIR, t if t.endswith(".h5") else t + ".h5")
        render(p)


if __name__ == "__main__":
    main()
