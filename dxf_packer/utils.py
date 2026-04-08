"""Shared utility helpers for DXF Packer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging to stdout with a simple format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def make_box_mesh_vertices_edges(
    position: np.ndarray,
    extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and edge pairs for a wireframe box.

    Args:
        position: Min corner (x, y, z).
        extents: Box dimensions (dx, dy, dz).

    Returns:
        (vertices, edges) where vertices is (8, 3) and edges is (12, 2) int32.
    """
    x0, y0, z0 = position
    dx, dy, dz = extents
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz

    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ], dtype=np.int32)

    return verts, edges


def color_palette(n: int) -> list[tuple[float, float, float, float]]:
    """Generate *n* visually distinct RGBA colours.

    Args:
        n: Number of colours needed.

    Returns:
        List of (r, g, b, a) tuples, values in [0, 1].
    """
    # Evenly spaced hues, high saturation / medium value
    colours: list[tuple[float, float, float, float]] = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = _hsv_to_rgb(hue, 0.75, 0.90)
        colours.append((r, g, b, 0.85))
    return colours


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV (all in [0,1]) to RGB."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    return [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][i]


def mm_to_inches(mm: float) -> float:
    """Convert millimetres to inches."""
    return mm / 25.4


def inches_to_mm(inches: float) -> float:
    """Convert inches to millimetres."""
    return inches * 25.4
