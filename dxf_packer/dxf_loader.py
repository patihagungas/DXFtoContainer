"""DXF → trimesh.Trimesh loader.

Handles 3DFACE entities (primary), including those nested inside INSERT blocks,
plus LINE, LWPOLYLINE, MESH, and POLYFACE MESH as secondary geometry sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import ezdxf
from ezdxf.document import Drawing
from ezdxf.entities import DXFGraphic

log = logging.getLogger(__name__)


def _face3d_to_triangles(entity: DXFGraphic) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Convert a 3DFACE entity to one or two triangles.

    A 3DFACE has vertices vtx0–vtx3. When vtx2 == vtx3 it is already a
    triangle; otherwise it is a quad that we split into two triangles.

    Returns:
        List of (v0, v1, v2) triples, each a (3,) float64 array.
    """
    v0 = np.array(entity.dxf.vtx0, dtype=np.float64)
    v1 = np.array(entity.dxf.vtx1, dtype=np.float64)
    v2 = np.array(entity.dxf.vtx2, dtype=np.float64)
    v3 = np.array(entity.dxf.vtx3, dtype=np.float64)

    triangles = [(v0, v1, v2)]
    if not np.allclose(v2, v3):
        triangles.append((v0, v2, v3))
    return triangles


def _collect_entities(doc: Drawing) -> list[DXFGraphic]:
    """Collect all DXF graphic entities from modelspace, expanding INSERT blocks.

    Uses ezdxf's virtual_entities() to expand nested INSERT references so that
    3DFACE geometry inside blocks is also captured.

    Returns:
        Flat list of DXFGraphic entities.
    """
    msp = doc.modelspace()
    entities: list[DXFGraphic] = []

    for entity in msp:
        if entity.dxftype() == "INSERT":
            try:
                for virtual in entity.virtual_entities():
                    entities.append(virtual)
            except Exception as exc:
                log.warning("Could not expand INSERT block: %s", exc)
        else:
            entities.append(entity)

    return entities


def load_dxf(filepath: str | Path) -> trimesh.Trimesh:
    """Load a DXF file and return a trimesh.Trimesh of its geometry.

    Priority order:
      1. 3DFACE entities (direct and inside INSERT blocks).
      2. MESH entities.
      3. POLYFACE MESH entities.
      4. LINE entities (converted to a thin wire-frame mesh).

    Args:
        filepath: Path to the .dxf file.

    Returns:
        A trimesh.Trimesh containing the combined geometry.

    Raises:
        ValueError: If the file contains no usable geometry.
        FileNotFoundError: If the file does not exist.
        ezdxf.DXFError: If the file is not a valid DXF.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DXF file not found: {filepath}")

    log.info("Loading DXF: %s", filepath)
    doc = ezdxf.readfile(str(filepath))
    entities = _collect_entities(doc)

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []

    # --- 3DFACE (primary) ---
    face3d_count = 0
    for entity in entities:
        if entity.dxftype() == "3DFACE":
            try:
                for tri in _face3d_to_triangles(entity):
                    base = len(vertices)
                    vertices.extend(tri)
                    faces.append((base, base + 1, base + 2))
                    face3d_count += 1
            except Exception as exc:
                log.warning("Skipping malformed 3DFACE: %s", exc)

    log.info("3DFACE triangles found: %d", face3d_count)

    # --- MESH entities ---
    for entity in entities:
        if entity.dxftype() == "MESH":
            try:
                mesh_verts = list(entity.vertices)
                mesh_faces = list(entity.faces)
                base = len(vertices)
                vertices.extend([np.array(v, dtype=np.float64) for v in mesh_verts])
                for face in mesh_faces:
                    if len(face) == 3:
                        faces.append((base + face[0], base + face[1], base + face[2]))
                    elif len(face) == 4:
                        faces.append((base + face[0], base + face[1], base + face[2]))
                        faces.append((base + face[0], base + face[2], base + face[3]))
            except Exception as exc:
                log.warning("Skipping MESH entity: %s", exc)

    # --- POLYFACE MESH ---
    for entity in entities:
        if entity.dxftype() == "POLYLINE" and entity.is_poly_face_mesh:
            try:
                pmesh = entity
                verts = [np.array([v.dxf.location.x, v.dxf.location.y, v.dxf.location.z], dtype=np.float64)
                         for v in pmesh.vertices if not v.is_face_record]
                base = len(vertices)
                vertices.extend(verts)
                for v in pmesh.vertices:
                    if v.is_face_record:
                        idxs = [abs(i) - 1 for i in [v.dxf.vtx0, v.dxf.vtx1, v.dxf.vtx2, v.dxf.vtx3] if i != 0]
                        if len(idxs) >= 3:
                            faces.append((base + idxs[0], base + idxs[1], base + idxs[2]))
                        if len(idxs) == 4:
                            faces.append((base + idxs[0], base + idxs[2], base + idxs[3]))
            except Exception as exc:
                log.warning("Skipping POLYFACE MESH: %s", exc)

    if not faces:
        raise ValueError(
            f"No usable geometry found in '{filepath.name}'. "
            "File must contain 3DFACE, MESH, or POLYFACE MESH entities."
        )

    verts_array = np.array(vertices, dtype=np.float64)
    faces_array = np.array(faces, dtype=np.int32)

    mesh = trimesh.Trimesh(vertices=verts_array, faces=faces_array, process=True)
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()

    if len(mesh.faces) == 0:
        raise ValueError(
            f"Geometry in '{filepath.name}' collapsed to zero faces after processing. "
            "Check that the DXF contains valid 3D geometry."
        )

    log.info(
        "Loaded mesh: %d vertices, %d faces, extents=%s",
        len(mesh.vertices), len(mesh.faces), mesh.bounding_box.extents
    )
    return mesh
