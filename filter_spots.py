import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button

def load_image(path, slice_idx=None):
    img = tifffile.imread(str(path))
    print(type(img))
    shape = img.shape
    print(f"  raw image shape: {shape}  dtype: {img.dtype}")

    if img.ndim == 2:
        if slice_idx is not None:
            print("--slice option ignored for 2D image")
        return img.astype(np.float32), shape

    if img.ndim == 3:
        if slice_idx is not None:
            z = shape[0]
            if not (0 <= slice_idx < z):
                sys.exit(f"Error: --slice {slice_idx} out of range; "
                         f"image has {z} z-slices (0–{z - 1})")
            return img[slice_idx].astype(np.float32), shape
        
        return img.max(axis=0).astype(np.float32), shape

    sys.exit(f"Error: Unsupported image shape {shape}")

def image_hw(shape):
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3:
        return shape[1], shape[2]
    print(f"Warning: Cannot determine image height/width from shape {shape}")
    return None, None


def auto_contrast(img):
    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))
    if vmin >= vmax:
        vmin, vmax = float(img.min()), float(img.max())
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax


# ── CSV ────────────────────────────────────────────────────────────────────────

def check_coordinate_sanity(df, x_col, y_col, img_shape):
    height, width = image_hw(img_shape)
    if height is None:
        print("Warning: Cannot check coordinate sanity without image height/width.")
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

    For 3-D stacks the caller also wires ↑/↓ and scroll-wheel for z-navigation;
    those are handled outside this class and do not interfere with the above.
    """

    _HINT = (
        "Left-click: add  |  Right-click / Backspace: undo  |  Enter / double-click: close\n"
        "r: reset  |  s: save & exit"
    )

    def __init__(self, ax, z_suffix="", active_guard=None):
        """
        active_guard : optional callable → bool.  When provided, press/key
        events are ignored unless active_guard() returns True.  Used to
        suppress polygon interactions while a different tab is displayed.
        """
        self.ax = ax
        self.fig = ax.figure
        self.verts = []       # list of (x, y) tuples in pixel coords
        self.closed = False
        self.done = False     # True after the user presses 's' and the window closes
        self._artists = []    # drawn artists cleared on each redraw
        self._z_suffix = z_suffix
        self._active_guard = active_guard
        self._cids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_press),
            self.fig.canvas.mpl_connect("key_press_event",   self._on_key),
        ]
        # Hint lives inside the axes at the bottom so it never overlaps buttons
        self._hint_artist = self.ax.text(
            0.5, 0.01, "",
            transform=self.ax.transAxes,
            ha="center", va="bottom", fontsize=8, color="white", zorder=20,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                      alpha=0.6, edgecolor="none"),
        )
        self._set_title(self._HINT)

    # ── event handlers ──────────────────────────────────────────────────────

    def _on_press(self, ev):
        if self._active_guard and not self._active_guard():
            return
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
        if self._active_guard and not self._active_guard():
            return
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
        self._hint_artist.set_text(msg + self._z_suffix)
        self.fig.canvas.draw_idle()


def draw_polygon_interactive(display_img, spots_xy, raw_stack=None, start_z=0,
                             spots_z=None):
    """
    Open a two-tab interactive window and return the drawn polygon vertices.

    Tab 1 — Spot Detection  (black background, spots only — no image)
        Two display modes toggled by buttons below the tab bar:
          • Show All Spots  — every detected spot plotted at once.
          • Filter by Z-slice — only spots whose z rounds to the current
            slice; scroll wheel or ↑/↓ to browse z.
        Draw your polygon in either mode; it filters by x,y only.

    Tab 2 — Raw Image
        The original TIF slice (no spot overlay) so you can compare
        fluorescence signal against the bare image to judge noise.

    Both tabs stay in sync: scrolling in one updates the other.

    raw_stack : (Z, Y, X) float32 ndarray; None for 2-D images.
    start_z   : initial z-slice (ignored when raw_stack is None).
    spots_z   : 1-D float array of z-coords, one per row of spots_xy.
                Enables the Per-Z mode toggle.  None = All Spots only.
    """
    is_3d = raw_stack is not None
    height, width = display_img.shape[:2]

    if is_3d:
        z_count = raw_stack.shape[0]
        start_z = max(0, min(z_count - 1, start_z))

    # Pre-compute which spots belong to each z-slice.
    # Round to nearest integer (e.g. z=3.7 → slice 4, z=3.2 → slice 3).
    # Clamped so out-of-range z values land on the nearest valid slice.
    has_spots = spots_xy is not None and len(spots_xy) > 0
    has_z_filter = is_3d and spots_z is not None and has_spots
    spot_z_map: dict = {}
    if has_z_filter:
        z_idx_arr = np.clip(np.round(spots_z).astype(int), 0, z_count - 1)
        for zi in range(z_count):
            spot_z_map[zi] = z_idx_arr == zi

    # Layout: tab buttons (row 1), optional mode-toggle (row 2), content (row 3)
    fig = plt.figure(figsize=(12, 10))
    ax_btn1 = fig.add_axes([0.01, 0.935, 0.48, 0.055])
    ax_btn2 = fig.add_axes([0.51, 0.935, 0.48, 0.055])

    ax_mode1 = ax_mode2 = None
    if has_z_filter:
        ax_mode1 = fig.add_axes([0.20, 0.885, 0.28, 0.038])
        ax_mode2 = fig.add_axes([0.52, 0.885, 0.28, 0.038])

    _content_h = 0.835 if has_z_filter else 0.88
    _rect = [0.07, 0.04, 0.88, _content_h]
    ax_spots = fig.add_axes(_rect)
    ax_raw   = fig.add_axes(_rect)

    # Tab 1: black background, spots only (no image)
    ax_spots.set_facecolor("black")
    ax_spots.set_xlim(-0.5, width - 0.5)
    ax_spots.set_ylim(height - 0.5, -0.5)
    ax_spots.set_xlabel("x (pixels)")
    ax_spots.set_ylabel("y (pixels)")

    from matplotlib.patches import Rectangle as _Rect
    ax_spots.add_patch(_Rect(
        (-0.5, -0.5), width, height,
        fill=False, edgecolor="#555555", linewidth=1, zorder=1,
    ))

    sc_spots = None
    if has_spots:
        sc_spots = ax_spots.scatter(
            spots_xy[:, 0], spots_xy[:, 1],
            s=8, c="cyan", alpha=0.6, linewidths=0, zorder=3, label="spots",
        )
        ax_spots.legend(loc="upper right", fontsize=8, labelcolor="white",
                        facecolor="#222222", edgecolor="#555555")

    z1_label = None
    if is_3d:
        z1_label = ax_spots.text(
            0.01, 0.99,
            "All Spots" if has_z_filter else f"Z: {start_z + 1} / {z_count}",
            transform=ax_spots.transAxes, color="white", fontsize=13,
            va="top", ha="left", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75),
        )

    # Tab 2: raw image, no spots
    raw_init = raw_stack[start_z] if is_3d else display_img
    vmin_r, vmax_r = auto_contrast(raw_init)
    im_raw = ax_raw.imshow(
        raw_init, cmap="gray", origin="upper", vmin=vmin_r, vmax=vmax_r,
        extent=[-0.5, width - 0.5, height - 0.5, -0.5],
    )
    ax_raw.set_xlim(-0.5, width - 0.5)
    ax_raw.set_ylim(height - 0.5, -0.5)
    ax_raw.set_xlabel("x (pixels)")
    ax_raw.set_ylabel("y (pixels)")
    ax_raw.set_title(
        "↑/↓ or scroll: change Z-slice  —  no spot overlay" if is_3d
        else "Raw image — no spot overlay",
        fontsize=9,
    )
    ax_raw.set_visible(False)

    z2_label = None
    if is_3d:
        z2_label = ax_raw.text(
            0.01, 0.99, f"Z: {start_z + 1} / {z_count}",
            transform=ax_raw.transAxes, color="white", fontsize=13,
            va="top", ha="left", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.65),
        )

    tab_state  = {"active": 1}
    z_state    = {"z": start_z}
    mode_state = {"all": True}   # True = All Spots; False = Filter by Z-slice

    def _apply_mode():
        """Refresh Tab 1 scatter and badge to match current mode + z."""
        if sc_spots is None:
            return
        z = z_state["z"]
        if mode_state["all"] or not has_z_filter:
            sc_spots.set_offsets(spots_xy)
            if z1_label is not None:
                z1_label.set_text("All Spots" if has_z_filter
                                  else f"Z: {z + 1} / {z_count}")
        else:
            zmask   = spot_z_map.get(z, np.zeros(len(spots_xy), dtype=bool))
            xy_here = spots_xy[zmask]
            sc_spots.set_offsets(xy_here if len(xy_here) > 0 else np.empty((0, 2)))
            if z1_label is not None:
                z1_label.set_text(f"Z: {z + 1} / {z_count}  ({zmask.sum()} spots)")

    _ON  = "#2196F3"
    _OFF = "#E0E0E0"

    btn1 = Button(ax_btn1,
                  "①  Spot Detection  —  scroll Z + draw polygon" if is_3d
                  else "①  Spot Detection  —  draw polygon here",
                  color=_ON, hovercolor="#1976D2")
    btn2 = Button(ax_btn2,
                  "②  Raw Image  —  clean view, no spot overlay" if is_3d
                  else "②  Raw Image  —  no spot overlay",
                  color=_OFF, hovercolor="#BDBDBD")
    btn1.label.set(color="white",   fontsize=10, fontweight="bold")
    btn2.label.set(color="#333333", fontsize=10)

    def _activate_tab1(_=None):
        tab_state["active"] = 1
        ax_spots.set_visible(True)
        ax_raw.set_visible(False)
        if ax_mode1 is not None:
            ax_mode1.set_visible(True)
            ax_mode2.set_visible(True)
        btn1.color = _ON;  btn1.hovercolor = "#1976D2"
        btn2.color = _OFF; btn2.hovercolor = "#BDBDBD"
        btn1.label.set(color="white",   fontweight="bold")
        btn2.label.set(color="#333333", fontweight="normal")
        fig.canvas.draw_idle()

    def _activate_tab2(_=None):
        tab_state["active"] = 2
        ax_spots.set_visible(False)
        ax_raw.set_visible(True)
        if ax_mode1 is not None:
            ax_mode1.set_visible(False)
            ax_mode2.set_visible(False)
        btn1.color = _OFF; btn1.hovercolor = "#BDBDBD"
        btn2.color = _ON;  btn2.hovercolor = "#1976D2"
        btn1.label.set(color="#333333", fontweight="normal")
        btn2.label.set(color="white",   fontweight="bold")
        fig.canvas.draw_idle()

    btn1.on_clicked(_activate_tab1)
    btn2.on_clicked(_activate_tab2)

    if has_z_filter:
        btn_mode1 = Button(ax_mode1, "Show All Spots",
                           color=_ON, hovercolor="#1976D2")
        btn_mode2 = Button(ax_mode2, "Filter by Z-slice",
                           color=_OFF, hovercolor="#BDBDBD")
        btn_mode1.label.set(color="white",   fontsize=9, fontweight="bold")
        btn_mode2.label.set(color="#333333", fontsize=9)

        def _set_mode_all(_=None):
            mode_state["all"] = True
            _apply_mode()
            btn_mode1.color = _ON;  btn_mode1.hovercolor = "#1976D2"
            btn_mode2.color = _OFF; btn_mode2.hovercolor = "#BDBDBD"
            btn_mode1.label.set(color="white",   fontweight="bold")
            btn_mode2.label.set(color="#333333", fontweight="normal")
            fig.canvas.draw_idle()

        def _set_mode_pz(_=None):
            mode_state["all"] = False
            _apply_mode()
            btn_mode1.color = _OFF; btn_mode1.hovercolor = "#BDBDBD"
            btn_mode2.color = _ON;  btn_mode2.hovercolor = "#1976D2"
            btn_mode1.label.set(color="#333333", fontweight="normal")
            btn_mode2.label.set(color="white",   fontweight="bold")
            fig.canvas.draw_idle()

        btn_mode1.on_clicked(_set_mode_all)
        btn_mode2.on_clicked(_set_mode_pz)

    if is_3d:
        def _set_z(new_z):
            new_z = max(0, min(z_count - 1, new_z))
            if new_z == z_state["z"]:
                return
            z_state["z"] = new_z
            sl = raw_stack[new_z]
            vmin_sl, vmax_sl = auto_contrast(sl)
            im_raw.set_data(sl)
            im_raw.set_clim(vmin_sl, vmax_sl)
            if z2_label is not None:
                z2_label.set_text(f"Z: {new_z + 1} / {z_count}")
            _apply_mode()
            fig.canvas.draw_idle()

        def _on_scroll(ev):
            if ev.button == "up":
                _set_z(z_state["z"] - 1)
            elif ev.button == "down":
                _set_z(z_state["z"] + 1)

        def _on_zkey(ev):
            if ev.key == "up":
                _set_z(z_state["z"] - 1)
            elif ev.key == "down":
                _set_z(z_state["z"] + 1)

        fig.canvas.mpl_connect("scroll_event", _on_scroll)
        fig.canvas.mpl_connect("key_press_event", _on_zkey)

    _saved_keymaps = {}
    for action in ("save", "quit", "back", "forward"):
        k = f"keymap.{action}"
        _saved_keymaps[k] = list(plt.rcParams.get(k, []))
        plt.rcParams[k] = []

    z_suffix = "\n↑/↓ or scroll: change Z-slice" if is_3d else ""
    drawer = PolygonDrawer(
        ax_spots,
        z_suffix=z_suffix,
        active_guard=lambda: tab_state["active"] == 1,
    )

    try:
        plt.show()
    finally:
        plt.rcParams.update(_saved_keymaps)

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

    x_col, y_col = "x", "y"
    print(f"  ({len(df)} spots, {len(df.columns)} columns)")
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

        # Detect z column so Tab 1 can show spots per z-slice
        z_col = next((c for c in df.columns if c.lower() == "z"), None)
        spots_z = df[z_col].to_numpy(dtype=float) if z_col is not None else None
        if z_col is not None:
            print(f"  z column '{z_col}' found — Tab 1 will filter spots per z-slice")
        else:
            print("  No z column found — all spots shown on every slice")

        # For 3-D stacks, load the full array so the user can scroll z-slices
        # interactively to find the focal plane that best represents the ROI.
        raw_stack = None
        start_z = 0
        if len(img_shape) == 3:
            print(f"  Loading full z-stack ({img_shape[0]} slices) for interactive scroll …")
            raw_stack = tifffile.imread(str(image_path)).astype(np.float32)
            start_z = args.slice if args.slice is not None else 0

        print("\nOpening drawing window …")
        vertices = draw_polygon_interactive(
            display_img, spots_xy,
            raw_stack=raw_stack, start_z=start_z,
            spots_z=spots_z,
        )
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
