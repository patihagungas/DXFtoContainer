"""Rotation matrices for 3D bin packing.

Provides:
  - axis_aligned_rotations(): 24 unique 90° rotations of a cube.
  - upright_rotations(): 4 rotations around Z only (no tipping).
  - fine_z_rotations(): rotations around Z in fixed degree steps.
"""

from __future__ import annotations

import numpy as np


def _rot_x(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def axis_aligned_rotations() -> list[np.ndarray]:
    """Return the 24 unique 90° rotation matrices of a cube.

    Generated as 6 face-up orientations × 4 spins around the vertical axis.
    Each matrix is a 3×3 float64 array.

    Returns:
        List of 24 unique 3×3 rotation matrices.
    """
    faces = [
        np.eye(3),                              # +Z up (default)
        _rot_x(90),                             # +Y up
        _rot_x(180),                            # -Z up
        _rot_x(270),                            # -Y up
        _rot_y(90),                             # -X up → Z up with X tipped
        _rot_y(270),                            # +X up
    ]
    spins = [_rot_z(0), _rot_z(90), _rot_z(180), _rot_z(270)]

    rotations: list[np.ndarray] = []
    seen: list[np.ndarray] = []

    for face in faces:
        for spin in spins:
            candidate = spin @ face
            # Deduplicate by checking all-close against previously added
            is_dup = any(np.allclose(candidate, s, atol=1e-6) for s in seen)
            if not is_dup:
                seen.append(candidate)
                rotations.append(candidate)

    return rotations


def upright_rotations() -> list[np.ndarray]:
    """Return only the 4 rotations around Z (no tipping).

    Use this when an assembly has allow_tipping=False.

    Returns:
        List of 4 3×3 rotation matrices (Z-axis spins at 0, 90, 180, 270°).
    """
    return [_rot_z(0), _rot_z(90), _rot_z(180), _rot_z(270)]


def fine_z_rotations(step_deg: int = 15) -> list[np.ndarray]:
    """Return rotations around Z in fixed angular steps.

    Generates rotations from 0° up to (360 - step_deg)° inclusive.
    The 0°, 90°, 180°, 270° entries will duplicate axis_aligned_rotations
    when used together, so callers should merge them as needed.

    Args:
        step_deg: Angular step in degrees (default 15).

    Returns:
        List of 3×3 rotation matrices.
    """
    angles = range(0, 360, step_deg)
    return [_rot_z(deg) for deg in angles]


def rotation_3x3_to_4x4(r3: np.ndarray) -> np.ndarray:
    """Embed a 3×3 rotation matrix into a 4×4 homogeneous transform.

    Args:
        r3: 3×3 rotation matrix.

    Returns:
        4×4 homogeneous matrix with no translation.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r3
    return m
