#!/usr/bin/env python3
"""Two mp4s: the meander evolving in the lab frame, four panels each.

    env OMP_NUM_THREADS=1 micromamba run -n dedalus python postprocessing/01_movie.py

NO NORMALISATION ANYWHERE.  The curves are psi_j(x,t) = psibar(y_j) + psi'_j(x,t) in their
own units -- river.pdf p.9's object, exactly as it defines it.  There is no display gain
and no invented offset: the vertical separation of the three curves IS psibar(y_j),
computed from the printed ubar.  The axis limits are fixed for the whole movie, so what
you see grow is what actually grows.

Row 1  psi_1, psi_2, psi_3.
Row 2  the same three curves, with the banks COLOURED by the local momentum flux, so its
       sign and magnitude are a field along the channel rather than one number per bank.
Row 3  |psihat_2|(t) against the analytic growth rate.
Row 4  the bank mode's dispersion relation, with this run's k* marked.
"""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm

from pp_lib import (COLORS, bank_mode, fig_to_rgb, fit_sigma_c, growth_and_phase,
                    load_run, set_style, v2_of, write_mp4, zeta2_of)
from noboru_model import forced_ratio    # after pp_lib: it puts the pkg on sys.path


plt = set_style()

FPS = 20
KS = np.linspace(1e-3, 2.0, 700)


def psibar(y, D):
    """Mean streamfunction, from p.9's ubar = -d(psibar)/dy, in units of b(U0+Delta).

        ubar* = 1 - D y^2   =>   psibar* = -(y - D y^3/3)

    So psibar(+b) = -(1 - D/3) and psibar(-b) = +(1 - D/3): psibar DECREASES as y
    increases, for every admissible D, because ubar > 0 everywhere.  Nothing here is
    chosen -- these are the offsets the deck's own base state produces.
    """
    return -(y - D * y**3 / 3.0)


def make_movie(tag, title_extra=None):
    run = load_run(tag)
    x, t = run["x"], run["t"]
    kstar, D, gamma, E, b = (float(run[k]) for k in ("kstar", "D", "gamma", "E", "b"))
    p1, p2, p3 = run["psi1"], run["psi2"], run["psi3"]
    xplot = x / 2.0                                    # deck x-axis is in units of 2b

    om, _, _ = bank_mode(kstar, D, gamma, E)
    sigma_a, c_a = om.imag, om.real / kstar
    growing = sigma_a > 0

    # ---- the three curves, in their own units: psi_j = psibar(y_j) + psi'_j  [p.9] ----
    off1, off2, off3 = psibar(b, D), psibar(0.0, D), psibar(-b, D)
    y1, y2, y3 = off1 + p1, off2 + p2, off3 + p3

    # Fixed limits for the whole movie, taken from the whole movie.  psi DECREASES with y,
    # so the axis is inverted: then height on the page increases with y and psi_1 (at
    # y = +b) sits on top -- matching the physical channel and the deck's own figures
    # without inventing an offset to force it.
    lo = float(min(y1.min(), y2.min(), y3.min()))
    hi = float(max(y1.max(), y2.max(), y3.max()))
    pad = 0.10 * (hi - lo)

    # ---- local momentum flux along the channel  [p.16, un-averaged] -------------------
    # p.16 prints d/dy <u'v'> ~= -<v2' zeta2'>, an x-AVERAGE.  Dropping the average leaves
    # the local eddy vorticity flux -v2'(x) zeta2'(x), whose x-mean is exactly the deck's
    # quantity.  u'v' is odd in y for the sinuous mode, so the two banks carry it with
    # opposite sign: +b*q at psi_1 and -b*q at psi_3 -- the deck's pp.16/18/19 arrows,
    # resolved in x instead of collapsed to one number per bank.
    q = -v2_of(run) * zeta2_of(run)
    f_top, f_bot = b * q, -b * q
    fmax = float(np.max(np.abs(f_top)))
    # symlog colour scale: the flux is quadratic in the amplitude, so it spans twice as
    # many decades as the meander does.  A linear scale would leave every early frame flat.
    norm = SymLogNorm(linthresh=max(fmax / 60.0, 1e-30), vmin=-fmax, vmax=fmax, base=10)
    cmap = plt.get_cmap("RdBu_r")

    # ---- row 3 ------------------------------------------------------------------------
    amp2 = np.abs(run["amp2"])
    sigma_m, _, resid = fit_sigma_c(t, run["amp2"], kstar)
    converged = resid < 1e-2

    # ---- row 4 ------------------------------------------------------------------------
    sig_k, c_k = growth_and_phase(KS, D, gamma, E)

    fig = plt.figure(figsize=(10.8, 13.4))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 2.2, 1.0, 1.3],
                          width_ratios=[45, 1], hspace=0.82, wspace=0.025)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    axC = fig.add_subplot(gs[2, 0])
    axD = fig.add_subplot(gs[3, 0])
    for cell in (gs[0, 1], gs[2, 1], gs[3, 1]):
        fig.add_subplot(cell).axis("off")

    def dress(ax, title):
        ax.set_title(title, fontsize=12.5, pad=8, linespacing=1.5)
        ax.set_xlim(0, xplot[-1])
        ax.set_ylim(hi + pad, lo - pad)          # inverted: psi decreases as y increases
        ax.set_xlabel(r"Downstream distance ($\times 2b$)")
        ax.set_ylabel(r"$\psi$", labelpad=8)
        ax.grid(False)
        for off, name in ((off1, "$y = +b$"), (off2, "$y = 0$"), (off3, "$y = -b$")):
            ax.axhline(off, color="0.55", ls=":", lw=1.1, zorder=1)
            ax.annotate(name, (xplot[-1] * 0.008, off), va="center", ha="left",
                        fontsize=9, color="0.45", zorder=8,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

    # ===================== row 1 ========================================================
    dress(axA, "Streamfunctions\n"
          r"$\psi_j=\bar\psi(y_j)+\psi'_j$ in its own units — no display gain;"
          " axis inverted so height increases with $y$")
    (lineA1,) = axA.plot([], [], color=COLORS["psi1"], lw=2.4, zorder=4)
    (lineA2,) = axA.plot([], [], color=COLORS["psi2"], lw=2.4, zorder=5)
    (lineA3,) = axA.plot([], [], color=COLORS["psi1"], lw=2.4, zorder=4)
    ybot = hi + 0.55 * pad
    axA.annotate("", xy=(xplot[-1] * 0.10, ybot), xytext=(xplot[-1] * 0.01, ybot),
                 arrowprops=dict(width=6, headwidth=15, headlength=12, color="#2b7bba"))
    axA.text(xplot[-1] * 0.115, ybot,
             rf"mean flow:  $\bar u(0)=1$,  $\bar u(\pm b)=1-D={1 - D:.2f}$"
             r"   (units of $U_0+\Delta$)",
             va="center", fontsize=9.5, color="#2b7bba")
    for lab, off, col in ((r"$\psi_1$", off1, COLORS["psi1"]),
                          (r"$\psi_2$", off2, COLORS["psi2"]),
                          (r"$\psi_3$", off3, COLORS["psi1"])):
        axA.annotate(lab, (xplot[-1] * 1.012, off), va="center", color=col,
                     fontsize=13, annotation_clip=False)
    hdr = axA.text(0.988, 0.93, "", transform=axA.transAxes, ha="right", va="top",
                   fontsize=11, zorder=9,
                   bbox=dict(fc="white", ec="0.85", alpha=0.92, pad=2.5))


    # ===================== row 2 ========================================================
    dress(axB, r"same curves — banks coloured by the local momentum flux "
               r"$-v'_2\zeta'_2$   [p.16, un-averaged]")
    lcA = LineCollection([], cmap=cmap, norm=norm, linewidths=4.5, zorder=4)
    lcB = LineCollection([], cmap=cmap, norm=norm, linewidths=4.5, zorder=4)
    axB.add_collection(lcA)
    axB.add_collection(lcB)
    (lineB2,) = axB.plot([], [], color="0.35", lw=1.6, zorder=5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"local $\overline{u'v'}$   (red $>0$, blue $<0$)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7)

    def segs(xx, yy):
        pts = np.array([xx, yy]).T.reshape(-1, 1, 2)
        return np.concatenate([pts[:-1], pts[1:]], axis=1)

    # ===================== row 3 ========================================================
    axC.set_xlim(t[0], t[-1])
    axC.set_yscale("log")
    pos = amp2 > 0
    axC.set_ylim(max(amp2[pos].min() * 0.5, 1e-30), amp2.max() * 3)
    axC.set_xlabel(r"$t$   (units of $b/(U_0+\Delta)$)")
    axC.set_ylabel(r"$|\hat\psi_2|$")
    # Anchor the analytic curve on the ANALYTIC forced steady state, |f * A0|, not on
    # amp2[0].  With the spin-up initial condition the interior starts from rest, so
    # amp2[0] = 0 and a line drawn as amp2[0]*exp(sigma t) is identically zero -- it
    # vanishes on a log axis.  |f A0| is the value p.11's closure predicts the interior
    # relaxes to (A0 = |psihat1| at t=0, which the bank carries), so for a rigid-bank run
    # (sigma = 0) this is a HORIZONTAL line and the spin-up curve climbs to meet it.
    anch = abs(forced_ratio(kstar, D, gamma) * run["amp1"][0])
    axC.plot(t, anch * np.exp(sigma_a * (t - t[0])), ls="--", lw=1.6, color="0.55",
             label=rf"det $M=0$: $|f A_0|={anch:.4f}$, $\sigma={sigma_a:+.4f}$")
    jf = int(np.argmax(t >= 0.6 * t[-1]))          # fit_sigma_c uses frac = 0.6
    axC.axvspan(t[0], t[jf], color="0.85", alpha=0.45, lw=0, zorder=0)
    axC.text(t[jf] * 0.5, axC.get_ylim()[0], " spin-up — not fitted", fontsize=7.5,
             color="0.35", va="bottom", ha="center")
    clab = (rf"Dedalus (fit $\sigma={sigma_m:+.4f}$)" if converged
            else rf"Dedalus (fit $\sigma={sigma_m:+.4f}$) — roots NOT separated")
    (lineC,) = axC.plot([], [], color=COLORS["growth"], lw=2.0, label=clab)
    (dotC,) = axC.plot([], [], "o", ms=6, color=COLORS["growth"])
    axC.legend(fontsize=8.5, loc="best")

    # ===================== row 4 ========================================================
    axD.set_xlim(-0.03, 2.06)
    axD.set_xlabel(r"$k^* = kb$")
    axD.set_ylabel(r"$\sigma = \mathrm{Im}\,\omega^*$", color=COLORS["growth"])
    axD.plot(KS, sig_k, color=COLORS["growth"], lw=2.0)
    axD.set_ylim(1.25 * float(sig_k.min()), 1.9 * float(sig_k.max()))
    axD.axhline(0, color="0.75", lw=0.8)
    axD.tick_params(axis="y", labelcolor=COLORS["growth"])
    axD.set_title(r"dispersion relation of the $\hat\psi_1=\hat\psi_3$ bank mode "
                  r"(det $M=0$) — this run marked" "\n"
                  r"$\psi'\propto e^{i(k^*x-\omega^* t)}$:  $\sigma>0$ grows, "
                  r"$c<0$ travels upstream", fontsize=10.5, linespacing=1.5)
    axD2 = axD.twinx()
    axD2.plot(KS, c_k, color=COLORS["upstream"], lw=2.0, ls="-.")
    axD2.set_ylabel(r"$c = \mathrm{Re}\,\omega^*/k^*$", color=COLORS["upstream"])
    axD2.tick_params(axis="y", labelcolor=COLORS["upstream"])
    axD2.set_ylim(-3, 1.3)
    axD2.axhline(0, color=COLORS["upstream"], lw=0.6, alpha=0.35)
    axD.axvline(kstar, color="0.35", lw=1.4)
    axD.plot([kstar], [sigma_a], "o", ms=10, mfc="none", mec=COLORS["growth"], mew=2.3,
             zorder=6)
    axD2.plot([kstar], [c_a], "s", ms=9, mfc="none", mec=COLORS["upstream"], mew=2.3,
              zorder=6)
    # Mark where sigma ACTUALLY crosses zero.  An earlier version marked sqrt(2D) here,
    # which is wrong on a growth-rate plot: k*^2 < 2D is p.14's FORCED-amplification
    # criterion (|psihat2| > |psihat1|), not a stability boundary, and the deck states no
    # stability boundary at all.  The two differ -- 0.9865 vs 1.0000 at D=0.5 -- and
    # labelling the forced criterion as if it bounded growth would be an invention.
    kz = np.nan
    ipk = int(np.argmax(sig_k))
    for j in range(ipk, KS.size - 1):
        if sig_k[j] > 0 >= sig_k[j + 1]:
            kz = KS[j]
            break
    ytop, ybt = axD.get_ylim()[1], axD.get_ylim()[0]
    if np.isfinite(kz):
        axD.axvline(kz, color="0.7", ls=":", lw=1.2)
        axD.annotate(rf"$\sigma=0$ at $k^*={kz:.3f}$", (kz + 0.03, ytop * 0.70),
                     fontsize=8.5, color="0.45")
    axD.annotate(rf"$k^*={kstar}$", (kstar + 0.03, ybt * 0.72), fontsize=9.5, color="0.2")
    axD.annotate(rf"$\sigma={sigma_a:+.4f}$", (0.03, ybt * 0.72), fontsize=9,
                 color=COLORS["growth"])
    axD.annotate(rf"$c={c_a:+.4f}$", (0.03, ybt * 0.94), fontsize=9,
                 color=COLORS["upstream"])

    where = ("UPSTREAM" if c_a < -1e-3 else
             "DOWNSTREAM" if c_a > 1e-3 else "nearly STATIONARY")
    verdict = (f"GROWS, travels {where}" if growing else f"DECAYS, travels {where}")
    head = (rf"$k^*={kstar}$,  $D={D}$,  $\gamma={gamma}$,  $E={E:.2f}$"
            rf"     $\Rightarrow$  {verdict}")
    if title_extra:
        head += "\n" + title_extra
    fig.suptitle(head, fontsize=12, y=0.987)

    frames = []
    for i in range(t.size):
        lineA1.set_data(xplot, y1[i]); lineA2.set_data(xplot, y2[i])
        lineA3.set_data(xplot, y3[i])
        lcA.set_segments(segs(xplot, y1[i])); lcA.set_array(f_top[i][:-1])
        lcB.set_segments(segs(xplot, y3[i])); lcB.set_array(f_bot[i][:-1])
        lineB2.set_data(xplot, y2[i])
        hdr.set_text(rf"$t = {t[i]:6.1f}$        "
                     rf"$|\hat\psi_2|/|\hat\psi_2|_{{t=0}} = {amp2[i] / amp2[0]:8.3g}$")
        lineC.set_data(t[:i + 1], amp2[:i + 1]); dotC.set_data([t[i]], [amp2[i]])
        frames.append(fig_to_rgb(fig))

    write_mp4(frames, f"meander_{tag}", fps=FPS)
    plt.close(fig)
    return dict(tag=tag, kstar=kstar, sigma_a=sigma_a, sigma_m=sigma_m, c_a=c_a,
                amp_ratio=amp2[-1] / amp2[0], fmax=fmax,
                a0=float(np.max(np.abs(p1[0]))), aT=float(np.max(np.abs(p1[-1]))),
                psibar_b=abs(off1))


print("building the two movies")
print("=" * 78)
summary = [make_movie(tag) for tag in ("k0.30", "k1.50")]
print("=" * 78)
for s_ in summary:
    print(f"  k*={s_['kstar']:<5} sigma analytic {s_['sigma_a']:+.5f} / fit {s_['sigma_m']:+.5f}"
          f"   c={s_['c_a']:+.4f}   |psihat2| x{s_['amp_ratio']:.3g}")
    print(f"          bank |psi'_1| {s_['a0']:.4f} -> {s_['aT']:.4f}"
          f"   (|psibar(b)| = {s_['psibar_b']:.4f})")

