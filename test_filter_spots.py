#!/usr/bin/env python3
"""
Synthetic verification test for filter_spots.py.

Creates a 512x512 TIF and 50 random spots, applies a known rectangular
polygon via --polygon (no interactive drawing), then checks:
  1. The filtered CSV row count matches an independently computed count.
  2. The overlay PNG is created.
  3. Column order and extra columns (z, intensity) are preserved.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path as MplPath

WORK_DIR = Path(__file__).parent
TEST_DIR = WORK_DIR / "test_data"


def make_test_data(rng):
    TEST_DIR.mkdir(exist_ok=True)

    # 512x512 gradient + noise image (gradient along x axis)
    xx = np.arange(512, dtype=np.float32)[np.newaxis, :]  # shape (1, 512)
    image = ((xx / 512.0) * 180 + rng.integers(0, 75, (512, 512))).astype(np.uint8)
    tif_path = TEST_DIR / "test_image.tif"
    tifffile.imwrite(str(tif_path), image)

    # 50 random spots with extra columns to verify preservation
    n = 50
    df = pd.DataFrame({
        "x":         rng.uniform(0.0, 511.0, n),
        "y":         rng.uniform(0.0, 511.0, n),
        "z":         rng.integers(0, 10, n).astype(float),
        "intensity": rng.uniform(100.0, 5000.0, n),
    })
    csv_path = TEST_DIR / "spots.csv"
    df.to_csv(csv_path, index=False)

    # Rectangular polygon well inside the image, no edge ambiguity
    vertices = [
        [120.0, 120.0],
        [390.0, 120.0],
        [390.0, 390.0],
        [120.0, 390.0],
    ]
    poly_path = TEST_DIR / "test_polygon.json"
    with open(poly_path, "w") as fh:
        json.dump(vertices, fh)

    return tif_path, csv_path, poly_path, df, vertices


def independent_mask(df, vertices):
    """
    Compute inside-polygon mask independently using numpy range checks.
    Valid for a convex rectangle and continuous float coordinates
    (boundary hits have probability ~0).
    x0, y0 = 120; x1, y1 = 390
    """
    x0, y0 = vertices[0]
    x1, y1 = vertices[2]
    x = df["x"].values
    y = df["y"].values
    return (x > x0) & (x < x1) & (y > y0) & (y < y1)


def run_cli(tif_path, csv_path, poly_path):
    out_csv = TEST_DIR / "spots_filtered.csv"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [
            sys.executable, str(WORK_DIR / "filter_spots.py"),
            "--image",   str(tif_path),
            "--csv",     str(csv_path),
            "--output",  str(out_csv),
            "--polygon", str(poly_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    return result, out_csv


def main():
    rng = np.random.default_rng(42)
    tif_path, csv_path, poly_path, df_orig, vertices = make_test_data(rng)

    # Independent reference mask using simple bounds check
    ref_mask = independent_mask(df_orig, vertices)
    n_expected = int(ref_mask.sum())

    print(f"Test image:    {tif_path}")
    print(f"Test CSV:      {csv_path} ({len(df_orig)} spots)")
    print(f"Polygon:       {poly_path}")
    print(f"Expected kept: {n_expected}  (independent bounds check)")
    print()

    result, out_csv = run_cli(tif_path, csv_path, poly_path)

    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print("\nFAIL: filter_spots.py exited with a non-zero code.")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    # ── Check 1: filtered CSV row count ──────────────────────────────────────
    if not out_csv.exists():
        print(f"\nFAIL: output CSV not found at {out_csv}")
        sys.exit(1)

    df_out = pd.read_csv(out_csv)
    n_got = len(df_out)

    if n_got != n_expected:
        print(f"\nFAIL: row count mismatch — expected {n_expected}, got {n_got}")
        # Cross-check with MplPath to help diagnose
        mpl_mask = MplPath(vertices).contains_points(df_orig[["x", "y"]].values)
        print(f"  MplPath count: {mpl_mask.sum()} | bounds check: {n_expected}")
        sys.exit(1)

    # ── Check 2: column order and content preserved ───────────────────────────
    if list(df_out.columns) != list(df_orig.columns):
        print(f"\nFAIL: column order changed.")
        print(f"  Expected: {list(df_orig.columns)}")
        print(f"  Got:      {list(df_out.columns)}")
        sys.exit(1)

    # Verify the kept rows are actually inside the polygon
    x_out = df_out["x"].values
    y_out = df_out["y"].values
    if not (np.all(x_out > 120) and np.all(x_out < 390)
            and np.all(y_out > 120) and np.all(y_out < 390)):
        print("\nFAIL: filtered CSV contains points outside the polygon.")
        sys.exit(1)

    # ── Check 3: overlay PNG created ─────────────────────────────────────────
    overlay_path = TEST_DIR / "spots_filtered_overlay.png"
    if not overlay_path.exists():
        print(f"\nFAIL: overlay PNG not found at {overlay_path}")
        sys.exit(1)

    overlay_bytes = overlay_path.stat().st_size
    if overlay_bytes < 1000:
        print(f"\nFAIL: overlay PNG is suspiciously small ({overlay_bytes} bytes)")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total   = len(df_orig)
    n_removed = n_total - n_got
    print()
    print("=" * 54)
    print(f"PASS  —  all checks passed")
    print(f"  Total spots   : {n_total}")
    print(f"  Kept (inside) : {n_got}  ✓  matches independent count")
    print(f"  Removed       : {n_removed}")
    print(f"  Filtered CSV  : {out_csv}")
    print(f"  Overlay PNG   : {overlay_path}  ({overlay_bytes:,} bytes)")
    print("=" * 54)


if __name__ == "__main__":
    main()
