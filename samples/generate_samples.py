"""Generate sample DXF files for testing the DXF Packer.

Produces three files:
  1. box_simple.dxf          — axis-aligned box assembly
  2. l_shape_tilted.dxf      — L-shaped assembly, tilted 13° Z + 7° X
  3. thin_beam_45deg.dxf     — long thin beam, rotated 45° on Z

Run from the project root:
    python samples/generate_samples.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Add parent to path so we can import ezdxf
sys.path.insert(0, str(Path(__file__).parent.parent))
import ezdxf


def _apply_transform(verts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous matrix to an (N, 3) vertex array."""
    ones = np.ones((len(verts), 1))
    hom = np.hstack([verts, ones])
    return (matrix @ hom.T).T[:, :3]


def _rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ])


def _rot_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([
        [ c, -s, 0, 0],
        [ s,  c, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1],
    ])


def _add_triangles(msp, triangles: list[tuple], layer: str = "0") -> None:
    """Write a list of (v0, v1, v2) triangles as 3DFACE entities."""
    for v0, v1, v2 in triangles:
        msp.add_3dface(
            [v0.tolist(), v1.tolist(), v2.tolist(), v2.tolist()],
            dxfattribs={"layer": layer},
        )


def _box_triangles(
    origin: np.ndarray, dims: np.ndarray, transform: np.ndarray | None = None
) -> list[tuple]:
    """Return 12 triangles (2 per face) for a box."""
    ox, oy, oz = origin
    dx, dy, dz = dims
    verts = np.array([
        [ox,    oy,    oz   ],
        [ox+dx, oy,    oz   ],
        [ox+dx, oy+dy, oz   ],
        [ox,    oy+dy, oz   ],
        [ox,    oy,    oz+dz],
        [ox+dx, oy,    oz+dz],
        [ox+dx, oy+dy, oz+dz],
        [ox,    oy+dy, oz+dz],
    ])
    if transform is not None:
        verts = _apply_transform(verts, transform)

    faces = [
        (verts[0], verts[1], verts[2]), (verts[0], verts[2], verts[3]),  # bottom
        (verts[4], verts[5], verts[6]), (verts[4], verts[6], verts[7]),  # top
        (verts[0], verts[1], verts[5]), (verts[0], verts[5], verts[4]),  # front
        (verts[2], verts[3], verts[7]), (verts[2], verts[7], verts[6]),  # back
        (verts[0], verts[3], verts[7]), (verts[0], verts[7], verts[4]),  # left
        (verts[1], verts[2], verts[6]), (verts[1], verts[6], verts[5]),  # right
    ]
    return faces


# ---------------------------------------------------------------------------
# Sample 1: simple axis-aligned box
# ---------------------------------------------------------------------------

def generate_box_simple(output_path: Path) -> None:
    """800 × 600 × 500 mm axis-aligned box."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    tris = _box_triangles(np.zeros(3), np.array([800.0, 600.0, 500.0]))
    _add_triangles(msp, tris, layer="GEOMETRY")
    doc.saveas(str(output_path))
    print(f"  Generated: {output_path}")


# ---------------------------------------------------------------------------
# Sample 2: L-shaped assembly, tilted 13° Z + 7° X
# ---------------------------------------------------------------------------

def generate_l_shape_tilted(output_path: Path) -> None:
    """Two boxes forming an L-shape, then tilted 13° on Z and 7° on X."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Build L-shape as two overlapping boxes (no tilt first)
    #   Box A: 1000 × 200 × 300 mm
    #   Box B: 200 × 600 × 300 mm, attached at one end
    transform = _rot_x(7.0) @ _rot_z(13.0)

    tris = []
    tris += _box_triangles(np.zeros(3), np.array([1000.0, 200.0, 300.0]), transform)
    tris += _box_triangles(np.array([0.0, 200.0, 0.0]), np.array([200.0, 600.0, 300.0]), transform)

    _add_triangles(msp, tris, layer="GEOMETRY")
    doc.saveas(str(output_path))
    print(f"  Generated: {output_path}")


# ---------------------------------------------------------------------------
# Sample 3: long thin beam, 45° on Z
# ---------------------------------------------------------------------------

def generate_thin_beam_45deg(output_path: Path) -> None:
    """1800 × 120 × 120 mm beam rotated 45° on Z."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    transform = _rot_z(45.0)
    tris = _box_triangles(np.zeros(3), np.array([1800.0, 120.0, 120.0]), transform)
    _add_triangles(msp, tris, layer="GEOMETRY")
    doc.saveas(str(output_path))
    print(f"  Generated: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples_dir = Path(__file__).parent
    samples_dir.mkdir(exist_ok=True)

    print("Generating sample DXF files…")
    generate_box_simple(samples_dir / "box_simple.dxf")
    generate_l_shape_tilted(samples_dir / "l_shape_tilted.dxf")
    generate_thin_beam_45deg(samples_dir / "thin_beam_45deg.dxf")
    print("Done.")
