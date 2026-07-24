#!/usr/bin/env python3
"""Capstone: the whole investigation in one figure.

LEFT  -- amplify vs decay, all steady runs overlaid: the secondary-flow
         parameter A flips DECAY (A=0) to GROWTH (A=2.89); the limit-2 (alpha~1)
         run decays like A=0.  A is the growth knob.
RIGHT -- the mechanism/closure map: every Thetis case (limit 1 & 2, steady &
         unsteady, Fr 0.09-0.30) migrates DOWNSTREAM; only the deck's vorticity
         bank-closure (../../vorticity_meander) gives UPSTREAM.  Direction is set
         by the BANK CLOSURE, not the flow regime.

    python postprocessing/04_summary.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_lib as pp  # noqa: E402

plt = pp.set_style()
YR = 365.25 * 86400.0


def load(case, m):
    f = os.path.join(pp.HERE, "experiments", case, "outputs", f"run_m{m}.npz")
    if not os.path.exists(f):
        return None
    D = np.load(f, allow_pickle=True)
    if "A" not in D.files or D["A"].size == 0:
        return None
    A = D["A"]
    return D["t"] / YR, np.abs(A), D["crest"]


def main():
    fig = plt.figure(figsize=(15.0, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)

    # ---------------- LEFT: growth, all cases ----------------------------
    axg = fig.add_subplot(gs[0])
    curves = [
        ("A0_incised", 4, "#1f6fb4", "-", "A=0  m=4  (gravity, F$^2$=0.09)"),
        ("A0_incised", 8, "#1f6fb4", "--", "A=0  m=8"),
        ("A2p89_alluvial", 4, "#c0392b", "-", "A=2.89  m=4  (secondary flow)"),
        ("A2p89_alluvial", 8, "#c0392b", "--", "A=2.89  m=8"),
        ("rossby_limit2", 15, "#2ca02c", "-", r"limit 2  $\alpha\approx$1  m=15  (Fr=0.09)"),
    ]
    for case, m, col, ls, lab in curves:
        r = load(case, m)
        if r is None:
            continue
        t, absA, _ = r
        axg.plot(t, absA / absA[0], ls, color=col, lw=2.0, label=lab)
    axg.axhline(1.0, color="0.6", lw=0.8, ls=":")
    axg.set_yscale("log")
    axg.set_xlabel("physical time  [years]")
    axg.set_ylabel(r"$|A(t)| / |A_0|$   (meander amplitude)")
    axg.set_title("(a) amplify vs decay:  A is the growth knob")
    axg.legend(fontsize=9, loc="center left")
    axg.text(0.97, 0.93, "amplify", transform=axg.transAxes, ha="right",
             color="#c0392b", fontsize=11, fontweight="bold")
    axg.text(0.97, 0.06, "decay", transform=axg.transAxes, ha="right",
             color="#1f6fb4", fontsize=11, fontweight="bold")

    # ---------------- RIGHT: mechanism / closure map ---------------------
    axm = fig.add_subplot(gs[1])
    axm.set_xlim(0, 10)
    axm.set_ylim(0, 10)
    axm.axis("off")
    axm.set_title("(b) migration direction is set by the BANK CLOSURE")

    # two columns: near-bank (Ikeda) closure -> downstream ; vorticity closure -> upstream
    axm.axvline(5.0, color="0.7", lw=1.0)
    axm.text(2.5, 9.4, "near-bank velocity closure\n"
             r"$\gamma\,\partial_t y = E\,u'_b$  (Ikeda)", ha="center",
             fontsize=10.5, color="#333")
    axm.text(7.5, 9.4, "vorticity closure\n"
             r"$c_0 = -E D/\gamma$  (deck 3-level)", ha="center",
             fontsize=10.5, color="#333")
    axm.annotate("", xy=(4.4, 5.0), xytext=(0.8, 5.0),
                 arrowprops=dict(arrowstyle="-|>", color="#1f6fb4", lw=2.5))
    axm.text(2.6, 5.4, "DOWNSTREAM", ha="center", color="#1f6fb4",
             fontsize=12, fontweight="bold")
    axm.annotate("", xy=(5.6, 5.0), xytext=(9.2, 5.0),
                 arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.5))
    axm.text(7.4, 5.4, "UPSTREAM", ha="center", color="#c0392b",
             fontsize=12, fontweight="bold")

    left = ("Thetis 2-D SWE  (this package)\n"
            "- limit 1  ($\\alpha$=0.26)  &  limit 2 ($\\alpha\\approx$1)\n"
            "- steady (B)  &  unsteady (A)\n"
            "- Fr = 0.09 - 0.30\n"
            "- A = 0 (decay)  &  A = 2.89 (amplify)\n"
            "=> ALL downstream")
    axm.text(2.5, 3.1, left, ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="#eaf2fb", ec="#1f6fb4"))
    right = ("vorticity_meander /\ndeliverable1_noboru_model\n"
             "(SW-note limit-2 QGPV,\nthe 3-level erodible-bank model)\n"
             "=> upstream ($c_0<0$)")
    axm.text(7.5, 3.1, right, ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="#fbecea", ec="#c0392b"))
    axm.text(5.0, 0.7, "Changing the flow REGIME (limit 1 -> 2) or SOLVER does not "
             "flip the direction;\nonly changing the bank-erosion CLOSURE does.",
             ha="center", fontsize=9.5, style="italic", color="0.25")

    fig.suptitle("Meandering-channel morphodynamics in Thetis: A sets growth, "
                 "the bank closure sets migration direction",
                 fontsize=13, y=1.02)
    pp.set_case("A0_incised")            # write to a stable top-level-ish place
    out = os.path.join(pp.HERE, "experiments", "summary_growth_and_closure.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, pp.HERE)}")


if __name__ == "__main__":
    main()
