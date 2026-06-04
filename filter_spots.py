#!/usr/bin/env python3
"""
Filter RS-FISH spot detections by a user-drawn polygon on a TIF image.

Coordinate convention: origin top-left, y increases downward, matching
imshow's default orientation (origin='upper') and standard image array
indexing where array[row, col] corresponds to pixel (x=col, y=row).
If CSV y-values exceed the image height a warning is printed rather than
silently filtering wrong.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path as MplPath


# ── Image ──────────────────────────────────────────────────────────────────────

def load_image(path, slice_idx=None):
    """Return (display_img: float32 2-D array, img_shape: tuple)."""
    img = tifffile.imread(str(path))
    shape = img.shape

    if img.ndim == 2:
        if slice_idx is not None:
            print("Warning: --slice ignored for a 2-D image.")
        return img.astype(np.float32), shape

    if img.ndim == 3:
        # Assumed layout: (Z, Y, X)
        if slice_idx is not None:
            z = shape[0]
            if not (0 <= slice_idx < z):
                sys.exit(f"Error: --slice {slice_idx} out of range; "
                         f"image has {z} z-slices (0–{z - 1})")
            return img[slice_idx].astype(np.float32), shape
        # Default: max-intensity projection along Z
        return img.max(axis=0).astype(np.float32), shape

    if img.ndim == 4:
        print(f"4-D image detected with shape {shape}.")
        print("Cannot determine axis order (Z/T/C/Y/X) automatically.")
        print("Convert to a 2-D or 3-D array before running this tool.")
        sys.exit(1)

    sys.exit(f"Error: Unsupported image shape {shape}")


def image_hw(shape):
    """Return (height, width) from a raw image shape tuple."""
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) >= 3:
        return shape[-2], shape[-1]
    return None, None


def auto_contrast(img):
    """Return (vmin, vmax) using the 1st / 99th percentile."""
    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))
    if vmin >= vmax:
        vmin, vmax = float(img.min()), float(img.max())
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax


# ── CSV ────────────────────────────────────────────────────────────────────────

def find_xy_columns(df):
    """Return (x_col, y_col), case-insensitive. Exits with a clear error if absent."""
    lmap = {c.lower(): c for c in df.columns}
    for xk, yk in [("x", "y"), ("xpos", "ypos")]:
        if xk in lmap and yk in lmap:
            return lmap[xk], lmap[yk]
    print("Error: Cannot identify x/y columns in the CSV.")
    print(f"Columns found: {list(df.columns)}")
    sys.exit(1)


def check_coordinate_sanity(df, x_col, y_col, img_shape):
    """Warn if spot coordinates appear outside the image footprint."""
    height, width = image_hw(img_shape)
    if height is None:
        return
    msgs = []
    xmax, ymax = float(df[x_col].max()), float(df[y_col].max())
    xmin, ymin = float(df[x_col].min()), float(df[y_col].min())
    if xmax > width:
        msgs.append(f"x_max={xmax:.1f} exceeds image width {width}")
    if ymax > height:
        msgs.append(f"y_max={ymax:.1f} exceeds image height {height}")
    if xmin < 0:
        msgs.append(f"x_min={xmin:.1f} is negative")
    if ymin < 0:
        msgs.append(f"y_min={ymin:.1f} is negative")
    if msgs:
        print("WARNING: Possible coordinate mismatch between image and CSV:")
        for m in msgs:
            print(f"  • {m}")
        print("  Verify that both use the same pixel coordinate system.")


# ── Interactive polygon drawing ────────────────────────────────────────────────

class PolygonDrawer:
    """
    Attach to a matplotlib Axes and let the user draw a closed polygon.

    Controls
    --------
    Left-click          add a vertex
    Double left-click   close the polygon (vertex added by the preceding
                        single-click event is kept as the final vertex)
    Right-click         remove the last vertex
    Backspace           remove the last vertex
    Enter               close the polygon
    r                   reset (clear all vertices and start over)
    s                   save polygon and exit (closes the figure)
    """

    _HINT = (
        "Left-click: add  |  Right-click / Backspace: undo  |  "
        "Enter / double-click: close  |  r: reset  |  s: save & exit"
    )

    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        self.verts = []       # list of (x, y) tuples in pixel coords
        self.closed = False
        self.done = False     # True after the user presses 's' and the window closes
        self._artists = []    # drawn artists cleared on each redraw
        self._cids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_press),
            self.fig.canvas.mpl_connect("key_press_event",   self._on_key),
        ]
        self._set_title(self._HINT)

    # ── event handlers ──────────────────────────────────────────────────────

    def _on_press(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None or self.done:
            return
        if ev.button == 1:
            if ev.dblclick:
                # Single-click already added a vertex at this position; just close.
                self._close()
            elif not self.closed:
                self.verts.append((ev.xdata, ev.ydata))
                self._redraw()
        elif ev.button == 3 and not self.closed and self.verts:
            self.verts.pop()
            self._redraw()

    def _on_key(self, ev):
        if self.done:
            return
        k = ev.key
        if k == "backspace" and not self.closed and self.verts:
            self.verts.pop()
            self._redraw()
        elif k == "enter":
            self._close()
        elif k == "r":
            self.verts.clear()
            self.closed = False
            self._redraw()
            self._set_title(self._HINT)
        elif k == "s":
            self._save()

    # ── state transitions ────────────────────────────────────────────────────

    def _close(self):
        if len(self.verts) < 3:
            self._set_title("Need ≥ 3 vertices to close.  " + self._HINT)
            return
        self.closed = True
        self._redraw()
        self._set_title("Polygon closed.  Press ’s’ to save & exit, ’r’ to reset.")

    def _save(self):
        if len(self.verts) < 3:
            self._set_title("Cannot save: need ≥ 3 vertices.  " + self._HINT)
            return
        if not self.closed:
            self._close()
            if not self.closed:
                return
        self.done = True
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)
        plt.close(self.fig)

    # ── display ──────────────────────────────────────────────────────────────

    def _redraw(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()

        if not self.verts:
            self.fig.canvas.draw_idle()
            return

        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]

        if self.closed:
            from matplotlib.patches import Polygon as MPoly
            patch = MPoly(
                list(zip(xs, ys)), closed=True,
                facecolor="yellow", edgecolor="yellow",
                alpha=0.25, linewidth=2, zorder=5,
            )
            self.ax.add_patch(patch)
            self._artists.append(patch)
            lx, ly = xs + [xs[0]], ys + [ys[0]]
        else:
            lx, ly = xs, ys

        (ln,) = self.ax.plot(lx, ly, color="yellow", lw=2, zorder=6)
        sc = self.ax.scatter(xs, ys, c="yellow", s=40, zorder=7)
        self._artists += [ln, sc]
        self.fig.canvas.draw_idle()

    def _set_title(self, msg):
        self.ax.set_title(msg, fontsize=9)
        self.fig.canvas.draw_idle()


def draw_polygon_interactive(display_img, spots_xy):
    """
    Open an interactive matplotlib window. Returns list of (x, y) vertex
    tuples, or calls sys.exit() if the user cancels without finishing.
    """
    height, width = display_img.shape[:2]
    vmin, vmax = auto_contrast(display_img)

    fig, ax = plt.subplots(figsize=(10, 10))

    # extent places pixel [row, col] center at data coordinate (col, row),
    # so the CSV's (x, y) values map directly to axes (x, y) with no drift.
    ax.imshow(
        display_img, cmap="gray", origin="upper", vmin=vmin, vmax=vmax,
        extent=[-0.5, width - 0.5, height - 0.5, -0.5],
    )

    if spots_xy is not None and len(spots_xy) > 0:
        ax.scatter(
            spots_xy[:, 0], spots_xy[:, 1],
            s=8, c="cyan", alpha=0.6, linewidths=0, zorder=3, label="spots",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)   # y increases downward
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    drawer = PolygonDrawer(ax)
    fig.tight_layout()
    plt.show()

    if not drawer.done:
        sys.exit("Drawing cancelled — no polygon was saved.")

    return drawer.verts


# ── Filtering ──────────────────────────────────────────────────────────────────

def filter_spots(df, x_col, y_col, vertices):
    """Vectorised point-in-polygon test; returns a boolean ndarray."""
    xy = df[[x_col, y_col]].to_numpy(dtype=float)
    return MplPath(vertices).contains_points(xy)


# ── Overlay PNG ────────────────────────────────────────────────────────────────

def save_overlay(display_img, df, x_col, y_col, mask, vertices, out_path):
    """Save a verification PNG: image + polygon outline + kept/removed spots."""
    height, width = display_img.shape[:2]
    vmin, vmax = auto_contrast(display_img)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        display_img, cmap="gray", origin="upper", vmin=vmin, vmax=vmax,
        extent=[-0.5, width - 0.5, height - 0.5, -0.5],
    )

    out = ~mask
    if out.any():
        ax.scatter(
            df.loc[out, x_col], df.loc[out, y_col],
            s=12, c="red", alpha=0.8, linewidths=0, label="removed", zorder=4,
        )
    if mask.any():
        ax.scatter(
            df.loc[mask, x_col], df.loc[mask, y_col],
            s=12, c="lime", alpha=0.9, linewidths=0, label="kept", zorder=5,
        )

    if vertices:
        from matplotlib.patches import Polygon as MPoly
        verts_arr = np.array(vertices)
        ax.add_patch(MPoly(
            verts_arr, closed=True,
            facecolor="yellow", edgecolor="yellow",
            alpha=0.2, linewidth=2, zorder=3,
        ))
        closed_v = np.vstack([verts_arr, verts_arr[0]])
        ax.plot(closed_v[:, 0], closed_v[:, 1], "y-", lw=2, zorder=6)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{mask.sum()} kept (green)  |  {(~mask).sum()} removed (red)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter RS-FISH spot detections to those inside a drawn polygon."
    )
    p.add_argument("--image",   required=True, metavar="PATH.tif",
                   help="Input TIF image (2-D or 3-D z-stack)")
    p.add_argument("--csv",     required=True, metavar="PATH.csv",
                   help="Spot detections CSV (must contain x and y columns)")
    p.add_argument("--output",  metavar="PATH.csv",
                   help="Output CSV path (default: <csv_stem>_filtered.csv)")
    p.add_argument("--slice",   type=int, metavar="N",
                   help="Z-slice index to display instead of max projection")
    p.add_argument("--polygon", metavar="PATH.json",
                   help="Reuse a previously saved polygon JSON (skips drawing)")
    return p.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    csv_path   = Path(args.csv)
    for p in (image_path, csv_path):
        if not p.exists():
            sys.exit(f"Error: File not found: {p}")

    stem    = csv_path.stem
    csv_dir = csv_path.parent
    out_csv   = Path(args.output) if args.output else csv_dir / f"{stem}_filtered.csv"
    poly_save = csv_dir / f"{stem}_polygon.json"
    overlay_p = csv_dir / f"{stem}_filtered_overlay.png"

    # ── Load image ────────────────────────────────────────────────────────────
    print(f"Image: {image_path}")
    display_img, img_shape = load_image(image_path, args.slice)
    print(f"  shape {img_shape}  →  display {display_img.shape} (H×W)")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"CSV:   {csv_path}")
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("Warning: CSV contains zero rows. Writing empty filtered CSV.")
        df.to_csv(out_csv, index=False)
        print(f"Output: {out_csv}")
        return

    x_col, y_col = find_xy_columns(df)
    print(f"  x='{x_col}'  y='{y_col}'  ({len(df)} spots, {len(df.columns)} columns)")
    check_coordinate_sanity(df, x_col, y_col, img_shape)

    # ── Polygon ───────────────────────────────────────────────────────────────
    if args.polygon:
        ppath = Path(args.polygon)
        if not ppath.exists():
            sys.exit(f"Error: Polygon file not found: {ppath}")
        with open(ppath) as fh:
            vertices = json.load(fh)
        print(f"Polygon: {ppath}  ({len(vertices)} vertices)")
    else:
        spots_xy = df[[x_col, y_col]].to_numpy(dtype=float)
        print("\nOpening drawing window …")
        vertices = draw_polygon_interactive(display_img, spots_xy)
        with open(poly_save, "w") as fh:
            json.dump(vertices, fh, indent=2)
        print(f"Polygon saved: {poly_save}")

    # ── Filter ────────────────────────────────────────────────────────────────
    mask = filter_spots(df, x_col, y_col, vertices)
    df[mask].to_csv(out_csv, index=False)
    save_overlay(display_img, df, x_col, y_col, mask, vertices, overlay_p)

    n_total   = len(df)
    n_kept    = int(mask.sum())
    n_removed = n_total - n_kept
    print(f"\nSummary: {n_total} total | {n_kept} kept | {n_removed} removed")
    print(f"Filtered CSV:  {out_csv}")
    print(f"Overlay PNG:   {overlay_p}")


if __name__ == "__main__":
    main()
