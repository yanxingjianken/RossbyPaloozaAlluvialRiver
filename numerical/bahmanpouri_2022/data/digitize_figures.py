#!/usr/bin/env python3
"""Digitizer v2: axes box via panel-width-normalized black-line fraction;
bed extraction + area checksum for Fig. 5(a) (Sajo CS1) and Fig. 7(a) (FM CS3).
"""
import numpy as np
from PIL import Image
from scipy import ndimage

BASE = "/tmp/claude-257430/-net-flood-home-x-yan/01898a17-6c8d-4525-b199-0a07fee42a81/scratchpad/bahman_figs"


def load(page):
    return np.asarray(Image.open(f"{BASE}/page-{page:02d}.png").convert("RGB")).astype(int)


def color_mask(img, sat_min=60, val_min=80):
    mx = img.max(axis=2); mn = img.min(axis=2)
    return (mx - mn > sat_min) & (mx > val_min)


def black_mask(img, thresh=110):
    return img.max(axis=2) < thresh


def find_panels(img, min_w=600, min_h=120):
    lab, n = ndimage.label(color_mask(img))
    panels = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.size < 5000:
            continue
        if xs.max() - xs.min() >= min_w and ys.max() - ys.min() >= min_h:
            panels.append(dict(y0=int(ys.min()), y1=int(ys.max()),
                               x0=int(xs.min()), x1=int(xs.max())))
    return sorted(panels, key=lambda p: p["y0"])


def axes_box(img, p):
    """Axes box: black rows/cols spanning most of the panel extent."""
    bm = black_mask(img)
    W = p["x1"] - p["x0"]
    H = p["y1"] - p["y0"]
    # rows: fraction of black pixels across the panel's x-range
    rf = bm[:, p["x0"]:p["x1"]].mean(axis=1)
    cf = bm[p["y0"] - 60:p["y1"] + 60, :].mean(axis=0) * (H + 120) / (H + 120)
    rows = np.where(rf > 0.7)[0]
    # candidate rows near the panel
    rows = rows[(rows > p["y0"] - 60) & (rows < p["y1"] + 160)]
    cols = np.where(bm[p["y0"]:p["y1"], :].mean(axis=0) > 0.7)[0]
    cols = cols[(cols > p["x0"] - 60) & (cols < p["x1"] + 60)]
    top = rows[rows <= p["y0"] + 8]
    bot = rows[rows >= p["y1"] - 8]
    lef = cols[cols <= p["x0"] + 8]
    rig = cols[cols >= p["x1"] - 8]
    return dict(top=int(top.max()) if top.size else None,
                bot=int(bot.min()) if bot.size else None,
                left=int(lef.max()) if lef.size else None,
                right=int(rig.min()) if rig.size else None,
                bm=bm, rows=rows.tolist(), cols=cols.tolist())


def xticks(bm, row, x_lo, x_hi):
    band = bm[row + 3: row + 13, x_lo:x_hi]
    hit = band.mean(axis=0) > 0.4
    out, run = [], []
    for i, v in enumerate(hit):
        if v: run.append(i)
        elif run: out.append(x_lo + int(np.mean(run))); run = []
    if run: out.append(x_lo + int(np.mean(run)))
    # merge clusters closer than 8 px
    merged = []
    for t in out:
        if merged and t - merged[-1][-1] < 15:
            merged[-1].append(t)
        else:
            merged.append([t])
    return [int(np.mean(m)) for m in merged]


def yticks(bm, col, y_lo, y_hi):
    band = bm[y_lo:y_hi, col + 1: col + 7]  # ticks point INWARD (right of axis)
    hit = band.mean(axis=1) > 0.3
    out, run = [], []
    for i, v in enumerate(hit):
        if v: run.append(i)
        elif run: out.append(y_lo + int(np.mean(run))); run = []
    if run: out.append(y_lo + int(np.mean(run)))
    merged = []
    for t in out:
        if merged and t - merged[-1][-1] < 15:
            merged[-1].append(t)
        else:
            merged.append([t])
    return [int(np.mean(m)) for m in merged]


def extract_bed(img, p, box, x_of_px, y_of_px):
    cm = color_mask(img)
    xs, ds = [], []
    for px in range(p["x0"], p["x1"] + 1):
        colr = np.where(cm[p["y0"]:p["y1"] + 1, px])[0]
        if colr.size < 3:
            continue
        bed_py = p["y0"] + colr.max()
        xs.append(x_of_px(px))
        ds.append(max(0.0, y_of_px(bed_py)))
    xs = np.array(xs); ds = np.array(ds)
    o = np.argsort(xs)
    return xs[o], ds[o]


def process(page, name, tick_vals, area_expect=None):
    img = load(page)
    p = find_panels(img)[0]           # panel (a)
    box = axes_box(img, p)
    print(f"\n=== page {page} ({name}) panel(a) box: {box['top']},{box['bot']},"
          f"{box['left']},{box['right']}  rows={box['rows']} cols={box['cols'][:6]}...")
    tx = xticks(box["bm"], box["bot"], p["x0"] - 40, p["x1"] + 40)
    print(f"  x-tick px: {tx}  (expect {len(tick_vals)} for labels {tick_vals})")
    if len(tx) != len(tick_vals):
        print("  !! tick count mismatch; using first/last if spacing uniform")
    # linear map from first and last tick
    px0, pxN = tx[0], tx[-1]
    v0, vN = tick_vals[0], tick_vals[-1]
    slope = (vN - v0) / (pxN - px0)
    x_of_px = lambda px: v0 + slope * (px - px0)
    # y calibration from left-axis tick marks (labels 0, 0.2, ..., 1.0 -> 6 ticks)
    ty = yticks(box["bm"], box["left"], p["y0"] - 40, box["bot"] - 5)
    print(f"  y-tick px: {ty}  (n={len(ty)}; first->0.0, last->1.0)")
    ok = False
    for nmaj in (6, 3, 11):               # 0.2-step, 0.5-step, 0.1-step majors
        grid = np.linspace(ty[0], ty[-1], nmaj)
        if all(min(abs(np.array(ty) - g)) <= 5 for g in grid):
            ok = True
            print(f"  y grid matched with {nmaj} majors")
            break
    assert ok, f"y ticks {ty} fit no uniform major grid"
    py0, pyN = ty[0], ty[-1]
    yv0, yvN = 0.0, 1.0
    y_of_px = lambda py: yv0 + (yvN - yv0) * (py - py0) / (pyN - py0)
    xs, ds = extract_bed(img, p, box, x_of_px, y_of_px)
    # area (trapezoid over uniform px grid)
    dx = np.abs(np.diff(xs)).mean()
    area = float(np.sum(ds) * dx)
    wet_w = xs.max() - xs.min()
    print(f"  bed points: {len(xs)}  wet width = {wet_w:.1f} m  max depth = {ds.max():.2f} m")
    print(f"  AREA = {area:.2f} m^2" + (f"   (printed: {area_expect})" if area_expect else ""))
    return xs, ds


if __name__ == "__main__":
    xs1, ds1 = process(8, "Fig5a Sajo CS1", [20, 15, 10, 5], 13.67)
    xs2, ds2 = process(10, "Fig7a FM CS3", [15, 10, 5, 0], None)
    np.save("/tmp/sajo_bed.npy", np.c_[xs1, ds1])
    np.save("/tmp/mulde_bed.npy", np.c_[xs2, ds2])
