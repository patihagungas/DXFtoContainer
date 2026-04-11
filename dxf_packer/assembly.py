"""Assembly dataclass with OBB normalization.

Every assembly is normalized to an axis-aligned local frame immediately on
load so the packer never sees a tilted mesh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .dxf_loader import load_dxf, load_dxf_components

log = logging.getLogger(__name__)


@dataclass
class Assembly:
    """A single assembly loaded from a DXF file, normalized to axis-aligned frame.

    Attributes:
        name: Display name (defaults to filename stem).
        filepath: Source DXF path.
        mesh: Normalized trimesh (axis-aligned, longest→X, second→Y, shortest→Z).
        aabb: Axis-aligned bounding box of the normalized mesh.
        obb: Original oriented bounding box (before normalization).
        quantity: Number of copies to pack.
        weight_kg: Optional weight in kilograms.
        allow_tipping: If False, restrict packing to upright-only rotations.
        normalization_transform: 4×4 matrix mapping original coords → normalized.
        position: (x, y, z) placement position set by the packer (mm).
        rotation_matrix: 3×3 rotation set by the packer.
        color: RGBA tuple for 3D viewport rendering.
    """

    name: str
    filepath: Path
    mesh: trimesh.Trimesh
    aabb: trimesh.primitives.Box
    obb: trimesh.primitives.Box
    quantity: int = 1
    weight_kg: Optional[float] = None
    allow_tipping: bool = False
    normalization_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    position: Optional[np.ndarray] = None
    rotation_matrix: Optional[np.ndarray] = None
    color: tuple[float, float, float, float] = (0.4, 0.6, 1.0, 0.85)

    @property
    def dimensions(self) -> np.ndarray:
        """Normalized bounding box extents [L, W, H] in mm (longest first)."""
        return self.aabb.extents.copy()

    @property
    def volume(self) -> float:
        """Volume of the normalized bounding box in mm³."""
        e = self.aabb.extents
        return float(e[0] * e[1] * e[2])

    def __repr__(self) -> str:
        e = self.dimensions
        return (
            f"Assembly(name={self.name!r}, qty={self.quantity}, "
            f"dims={e[0]:.1f}×{e[1]:.1f}×{e[2]:.1f}mm)"
        )


def _normalize_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, trimesh.primitives.Box, np.ndarray]:
    """Apply OBB normalization to a mesh.

    Steps:
      1. Compute the oriented bounding box.
      2. Apply inverse OBB transform → axis-aligned.
      3. Sort axes so longest→X, second→Y, shortest→Z.

    Returns:
        (normalized_mesh, original_obb, normalization_transform_4x4)
    """
    mesh = mesh.copy()

    # Step 1: capture original OBB
    obb = mesh.bounding_box_oriented

    # Step 2: transform to OBB local frame (axis-aligned)
    to_local = np.linalg.inv(obb.primitive.transform)
    mesh.apply_transform(to_local)

    # Step 3: sort axes longest→X, second→Y, shortest→Z
    extents = mesh.bounding_box.extents
    order = np.argsort(-extents)          # descending index
    perm = np.eye(3, dtype=np.float64)[order]
    # Permutation matrices with an odd number of swaps have det=-1 (improper
    # rotation / reflection).  A reflected mesh renders inside-out in OpenGL.
    # Fix by negating one column so det becomes +1, then re-translate below.
    if np.linalg.det(perm) < 0:
        perm[:, 0] = -perm[:, 0]
    axis_swap = np.eye(4, dtype=np.float64)
    axis_swap[:3, :3] = perm
    mesh.apply_transform(axis_swap)

    # Translate so the mesh sits in the positive octant (min corner at origin)
    mins = mesh.bounding_box.bounds[0]
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = -mins
    mesh.apply_transform(translation)

    # Full normalization transform: original → normalized
    norm_transform = translation @ axis_swap @ to_local

    return mesh, obb, norm_transform


def load_assembly(
    filepath: str | Path,
    name: Optional[str] = None,
    quantity: int = 1,
    weight_kg: Optional[float] = None,
    allow_tipping: bool = False,
    color: tuple[float, float, float, float] = (0.4, 0.6, 1.0, 0.85),
) -> Assembly:
    """Load a DXF file and return a normalized Assembly.

    Args:
        filepath: Path to the .dxf file.
        name: Display name; defaults to the filename stem.
        quantity: Number of copies to pack.
        weight_kg: Optional weight in kg.
        allow_tipping: Allow non-upright rotations during packing.
        color: RGBA color for 3D viewport.

    Returns:
        An Assembly with a normalized, axis-aligned mesh.

    Raises:
        ValueError: If the DXF contains no usable geometry.
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if name is None:
        name = filepath.stem

    raw_mesh = load_dxf(filepath)
    norm_mesh, obb, norm_transform = _normalize_mesh(raw_mesh)

    aabb = norm_mesh.bounding_box

    log.info(
        "Assembly '%s' normalized: extents=%s (original OBB extents=%s)",
        name,
        np.round(aabb.extents, 1),
        np.round(obb.primitive.extents, 1),
    )

    return Assembly(
        name=name,
        filepath=filepath,
        mesh=norm_mesh,
        aabb=aabb,
        obb=obb,
        quantity=quantity,
        weight_kg=weight_kg,
        allow_tipping=allow_tipping,
        normalization_transform=norm_transform,
        color=color,
    )


def load_assembly_components(
    filepath: str | Path,
    name: Optional[str] = None,
    quantity: int = 1,
    weight_kg: Optional[float] = None,
    allow_tipping: bool = False,
    color: tuple[float, float, float, float] = (0.4, 0.6, 1.0, 0.85),
) -> list[Assembly]:
    """Load a DXF file and return one Assembly per connected component.

    If the DXF contains a single part the result is a one-element list
    (identical behaviour to load_assembly).  If the DXF is an assembly of
    several disconnected parts, each part is returned as its own Assembly
    so the packer can place them individually — eliminating the wasted space
    that comes from reserving the whole assembly's bounding box.

    Args:
        filepath: Path to the .dxf file.
        name: Base display name; components are suffixed _p1, _p2, … when
              there is more than one.
        quantity: Applied to every component.
        weight_kg: Optional per-component weight (split equally if desired).
        allow_tipping: Passed through to every component.
        color: Base RGBA colour — used as-is for all components.

    Returns:
        List of Assembly objects, one per disconnected component.
    """
    filepath = Path(filepath)
    if name is None:
        name = filepath.stem

    components = load_dxf_components(filepath)
    assemblies: list[Assembly] = []

    for i, raw_mesh in enumerate(components):
        part_name = name if len(components) == 1 else f"{name}_p{i + 1}"
        norm_mesh, obb, norm_transform = _normalize_mesh(raw_mesh)
        aabb = norm_mesh.bounding_box

        log.info(
            "Component '%s' normalized: extents=%s",
            part_name,
            np.round(aabb.extents, 1),
        )

        assemblies.append(Assembly(
            name=part_name,
            filepath=filepath,
            mesh=norm_mesh,
            aabb=aabb,
            obb=obb,
            quantity=quantity,
            weight_kg=weight_kg,
            allow_tipping=allow_tipping,
            normalization_transform=norm_transform,
            color=color,
        ))

    return assemblies
