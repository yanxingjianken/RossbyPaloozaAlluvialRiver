#!/usr/bin/env python3
"""Shared plotting/analysis helpers for the Thetis meander package.

The fenced block at the bottom is kept byte-identical across rossby_palooza
packages.  Verified by tests/test_setup.py -- the fence markers are deliberately
NOT written out here, because an awk range would then match this docstring first
and silently change the checksum.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(HERE, "figures")
OUT_DIR = os.path.join(HERE, "outputs")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import geometry as geo  # noqa: E402


# --------------------------------------------------------------------------- #
#  Package-specific helpers
# --------------------------------------------------------------------------- #
def channel_frame(x, c):
    """Unit tangent/normal of the centreline y = c(x), as (tx, ty), (nx, ny)."""
    dc = np.gradient(np.asarray(c, float), np.asarray(x, float), edge_order=2)
    norm = np.sqrt(1.0 + dc**2)
    return (1.0 / norm, dc / norm), (-dc / norm, 1.0 / norm)


def to_channel_components(u, v, tx, ty, nx_, ny_):
    """Project a Cartesian vector onto the (s, n) channel frame."""
    return u * tx + v * ty, u * nx_ + v * ny_


def interior_mask(x, d: geo.Design, pad_frac: float = 0.15):
    """True on the clean meander core, excluding the buffer-edge transitions.

    Colour limits must be taken from here only: a spike at the erodible/rigid
    interface otherwise saturates every scale and washes out the signal (the
    2026-07-23 FUNWAVE lesson).
    """
    pad = pad_frac * d.lam
    return (np.asarray(x) > d.x_m0 + pad) & (np.asarray(x) < d.x_m1 - pad)


def even_crop(img):
    """Crop an RGB frame to even pixel dimensions (libx264 yuv420p needs it)."""
    h, w = img.shape[:2]
    return img[: h - (h % 2), : w - (w % 2)]


def sym_limits(field, pct: float = 99.0):
    """Symmetric colour limits from a percentile of |field| (NaN-safe)."""
    a = np.abs(np.asarray(field, float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    v = float(np.percentile(a, pct))
    v = v if v > 0 else 1.0
    return -v, v


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
