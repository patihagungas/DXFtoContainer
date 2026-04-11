"""3D bin packing — Boundary-Point Best-Fit Decreasing.

Key improvements over naive extreme-point:
  - Candidate positions are the full Cartesian product of ALL X/Y face boundaries
    (left, right, front, back), giving denser coverage and enabling placements
    inside the bounding box of hollow/non-rectangular items.
  - Gravity Z is computed for every (x,y) candidate, not just the pre-stored Z.
  - Score favours low Z first (gravity), then low Y, then low X (fill front-left-bottom).
  - Clearance inflation is applied symmetrically; dunnage logs are drawn inside
    that clearance gap and correctly aligned to the actual (non-inflated) item.
  - Mesh-based collision detection (via trimesh/FCL) is used when available so that
    non-rectangular shapes (I-beams, trusses, angles) can be packed into the real
    empty space beside/between each other rather than their bounding boxes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import trimesh

from .assembly import Assembly
from .container import Container
from .rotations import axis_aligned_rotations, upright_rotations, fine_z_rotations

# ---------------------------------------------------------------------------
# Mesh collision support (trimesh + python-fcl)
# ---------------------------------------------------------------------------

HAS_MESH_COLLISION = False
try:
    import trimesh.collision as _tcol
    _cm_probe = _tcol.CollisionManager()
    _cm_probe.add_object("probe", trimesh.creation.box([1.0, 1.0, 1.0]))
    result = _cm_probe.in_collision_single(trimesh.creation.box([1.0, 1.0, 1.0]))
    # Accept both bool and (bool, ...) tuple return styles across trimesh versions
    _COLLISION_RESULT_IS_TUPLE = isinstance(result, tuple)
    HAS_MESH_COLLISION = True
    del _cm_probe, result
except Exception:
    _COLLISION_RESULT_IS_TUPLE = False

if HAS_MESH_COLLISION:
    import trimesh.collision

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlacedItem:
    """An assembly placed at a specific position and rotation inside the container.

    Attributes:
        assembly: Source Assembly (normalized mesh, min corner at origin).
        position: (x, y, z) min corner of the *inflated* AABB in container space (mm).
        rotation: 3×3 rotation matrix applied to the normalized mesh.
        inflated_extents: Clearance-inflated AABB extents (mm).
        clearance: Clearance value used for this placement (mm).
    """
    assembly: Assembly
    position: np.ndarray          # shape (3,)
    rotation: np.ndarray          # shape (3, 3)
    inflated_extents: np.ndarray  # shape (3,)
    clearance: float = 0.0

    @property
    def extents(self) -> np.ndarray:
        """Actual (non-inflated) AABB extents after rotation."""
        base = self.assembly.aabb.extents
        return np.abs(self.rotation) @ base

    @property
    def actual_position(self) -> np.ndarray:
        """Min corner of the *actual* (non-inflated) mesh in container space.

        The clearance is split equally on all sides, so the actual mesh is
        offset by clearance/2 from the inflated AABB corner.
        """
        return self.position + self.clearance / 2.0

    @property
    def max_corner(self) -> np.ndarray:
        """Max corner of the inflated AABB."""
        return self.position + self.inflated_extents

    @property
    def center(self) -> np.ndarray:
        """Center of the actual mesh AABB."""
        return self.actual_position + self.extents / 2.0


@dataclass
class SpacerLog:
    """A rectangular dunnage log placed inside the container.

    Attributes:
        position: Min corner (x, y, z) in container space (mm).
        extents: (length, width, height) in mm.
        label: Human-readable description.
    """
    position: np.ndarray
    extents: np.ndarray
    label: str = "dunnage"


@dataclass
class PackingResult:
    """Full result returned by Packer.pack()."""
    placed: list[PlacedItem]
    unpacked: list[Assembly]
    logs: list[SpacerLog]
    utilization: float
    total_weight_kg: Optional[float]
    collisions: list[str]
    elapsed_sec: float


# ---------------------------------------------------------------------------
# AABB helpers
# ---------------------------------------------------------------------------

def _aabb_overlap(
    pos_a: np.ndarray, ext_a: np.ndarray,
    pos_b: np.ndarray, ext_b: np.ndarray,
    tol: float = 1e-3,
) -> bool:
    """Return True if two AABBs overlap (separation test, strict tolerance)."""
    for i in range(3):
        if pos_a[i] + ext_a[i] <= pos_b[i] + tol:
            return False
        if pos_b[i] + ext_b[i] <= pos_a[i] + tol:
            return False
    return True


def _fits_in_container(
    pos: np.ndarray, ext: np.ndarray, container: Container, tol: float = 1e-3
) -> bool:
    """Return True if [pos, pos+ext] fits inside the container."""
    return (
        pos[0] >= -tol and pos[0] + ext[0] <= container.length + tol and
        pos[1] >= -tol and pos[1] + ext[1] <= container.width  + tol and
        pos[2] >= -tol and pos[2] + ext[2] <= container.height + tol
    )


def _overlaps_any(pos: np.ndarray, ext: np.ndarray, placed: list[PlacedItem]) -> bool:
    for item in placed:
        if _aabb_overlap(pos, ext, item.position, item.inflated_extents):
            return True
    return False


# ---------------------------------------------------------------------------
# Candidate position generation (boundary-point grid)
# ---------------------------------------------------------------------------

_MAX_CANDIDATES = 200  # cap to keep placement O(N) not O(N²)


def _candidate_xy_positions(placed: list[PlacedItem]) -> list[tuple[float, float]]:
    """Return (x, y) boundary positions to try for the next item.

    Builds the Cartesian product of X/Y face boundaries, then caps at
    _MAX_CANDIDATES by keeping only right/front faces (the most useful ones)
    when the grid would otherwise be too large.
    """
    xs_right = sorted(
        {0.0} | {p.position[0] + p.inflated_extents[0] for p in placed}
    )
    ys_front = sorted(
        {0.0} | {p.position[1] + p.inflated_extents[1] for p in placed}
    )
    candidates = [(x, y) for x in xs_right for y in ys_front]

    # If still within budget, also add left/back faces for denser coverage
    if len(candidates) <= _MAX_CANDIDATES:
        xs_all = sorted(
            {0.0}
            | {p.position[0] for p in placed}
            | {p.position[0] + p.inflated_extents[0] for p in placed}
        )
        ys_all = sorted(
            {0.0}
            | {p.position[1] for p in placed}
            | {p.position[1] + p.inflated_extents[1] for p in placed}
        )
        candidates = [(x, y) for x in xs_all for y in ys_all]
        if len(candidates) > _MAX_CANDIDATES:
            # Too many — fall back to right/front only
            candidates = [(x, y) for x in xs_right for y in ys_front]

    return candidates


# ---------------------------------------------------------------------------
# Gravity constraint
# ---------------------------------------------------------------------------

def _gravity_z(x0: float, y0: float, ext: np.ndarray, placed: list[PlacedItem]) -> float:
    """Return the lowest valid Z where an item of given extents can rest at (x0, y0).

    The item rests on the container floor or on the top face of any placed
    item whose inflated AABB overlaps the new item's XY footprint.
    """
    x1 = x0 + ext[0]
    y1 = y0 + ext[1]
    best_z = 0.0
    for item in placed:
        ix0, iy0, iz0 = item.position
        ix1 = ix0 + item.inflated_extents[0]
        iy1 = iy0 + item.inflated_extents[1]
        iz1 = iz0 + item.inflated_extents[2]
        if x0 < ix1 - 1e-3 and x1 > ix0 + 1e-3 and y0 < iy1 - 1e-3 and y1 > iy0 + 1e-3:
            best_z = max(best_z, iz1)
    return best_z


# ---------------------------------------------------------------------------
# Dunnage log insertion
# ---------------------------------------------------------------------------

def _insert_dunnage(
    placed: list[PlacedItem],
    log_cross: float,
    container: Container,
) -> list[SpacerLog]:
    """Insert dunnage logs under every placed item (floor and stacked).

    Two logs per item running along the WIDTH axis (Y), spaced at 25 % and
    75 % of the item's length (X).  Each log sits in the clearance gap directly
    below the actual mesh, so its height equals exactly clearance/2 — the
    tightest fit that keeps items separated.

    The log cross-section along X is min(log_cross, clearance/2) so it always
    stays inside the gap and never touches adjacent items.

    Args:
        placed: All placed items.
        log_cross: Nominal log width along X (mm). Clamped to clearance/2.
        container: Container dimensions (used only for boundary clamping).

    Returns:
        List of SpacerLog objects.
    """
    logs: list[SpacerLog] = []

    for item in placed:
        half_c = item.clearance / 2.0
        if half_c < 1.0:
            continue  # no usable gap

        # Log sits in the clearance gap below the actual mesh
        log_z = item.position[2]          # bottom of inflated AABB
        log_h = half_c                    # fills gap exactly, minimum possible

        # Log runs along Y (width).  Cross-section width along X is clamped
        # so it never exceeds the gap.
        log_w = min(log_cross, half_c)    # extent along X

        # Actual mesh footprint
        ap = item.actual_position         # (x, y, z) min corner of actual mesh
        ae = item.extents                 # actual extents

        # Clamp to container bounds
        ax0 = max(ap[0], 0.0)
        ax1 = min(ap[0] + ae[0], container.length)
        ay0 = max(ap[1], 0.0)
        ay1 = min(ap[1] + ae[1], container.width)

        log_len = ay1 - ay0   # runs along Y
        if log_len <= 0 or ax1 - ax0 <= 0:
            continue

        # Two logs at 25 % and 75 % of the item's X span
        for frac in (0.25, 0.75):
            log_x = ax0 + (ax1 - ax0) * frac - log_w / 2.0
            log_x = max(ax0, min(log_x, ax1 - log_w))  # stay inside item footprint
            logs.append(SpacerLog(
                position=np.array([log_x, ay0, log_z]),
                extents=np.array([log_w, log_len, log_h]),
                label="dunnage",
            ))

    return logs


# ---------------------------------------------------------------------------
# Collision verification
# ---------------------------------------------------------------------------

def _check_collisions(placed: list[PlacedItem]) -> list[str]:
    """Return descriptions of any AABB overlaps between placed items."""
    collisions: list[str] = []
    for i, a in enumerate(placed):
        for j, b in enumerate(placed):
            if j <= i:
                continue
            if _aabb_overlap(a.position, a.extents, b.position, b.extents):
                msg = (
                    f"Collision: '{a.assembly.name}' @ {np.round(a.position, 1)} "
                    f"overlaps '{b.assembly.name}' @ {np.round(b.position, 1)}"
                )
                collisions.append(msg)
                log.warning(msg)
    return collisions


# ---------------------------------------------------------------------------
# Mesh collision helpers
# ---------------------------------------------------------------------------

def _get_world_mesh(
    assembly: Assembly,
    actual_position: np.ndarray,
    rotation: np.ndarray,
) -> "trimesh.Trimesh":
    """Return the assembly mesh transformed to container-space coordinates.

    Args:
        assembly: Normalized assembly (mesh min-corner at origin).
        actual_position: Non-inflated min-corner in the container (mm).
        rotation: 3×3 rotation matrix applied before translation.

    Returns:
        A copy of the mesh in container frame.
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = actual_position
    mesh = assembly.mesh.copy()
    mesh.apply_transform(T)
    return mesh


# ---------------------------------------------------------------------------
# Packer
# ---------------------------------------------------------------------------

class Packer:
    """Boundary-Point Best-Fit Decreasing 3D bin packer.

    Usage::

        packer = Packer(container, clearance_mm=50.0, log_cross_mm=50.0)
        result = packer.pack(assemblies, fine_rotations=False)
    """

    def __init__(
        self,
        container: Container,
        clearance_mm: float = 50.0,
        log_cross_mm: float = 50.0,
        optimize_mode: str = "max_items",
    ) -> None:
        self.container = container
        self.clearance_mm = clearance_mm
        self.log_cross_mm = log_cross_mm
        self.optimize_mode = optimize_mode

    def _inflated(self, base: np.ndarray) -> np.ndarray:
        """Inflate extents by clearance_mm (clearance/2 per side)."""
        return base + self.clearance_mm

    def _rot_ext(self, base: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """AABB extents of a box after applying rotation rot."""
        return np.abs(rot) @ base

    def _score(self, pos: np.ndarray) -> float:
        """Lower is better: gravity first (Z), then front (Y), then left (X)."""
        return pos[2] * 1_000_000 + pos[1] * 1_000 + pos[0]

    def _try_place(
        self,
        assembly: Assembly,
        rotations: list[np.ndarray],
        placed: list[PlacedItem],
        cm: Optional["trimesh.collision.CollisionManager"] = None,
    ) -> Optional[PlacedItem]:
        """Find the best (rotation, position) for one assembly instance.

        Tries every rotation × every boundary-point (x, y) pair, drops each
        candidate to its gravity Z, then picks the lowest-scoring valid placement.

        When a CollisionManager *cm* is provided the collision check uses actual
        mesh geometry (via FCL) instead of inflated AABBs, which lets items
        nestle into the empty space beside/between non-rectangular shapes.
        The rotated mesh is built once per rotation; only the translation is
        updated per candidate so FCL BVH construction cost is amortised.

        Returns:
            Best PlacedItem, or None if the item cannot fit anywhere.
        """
        base = assembly.aabb.extents
        candidates = _candidate_xy_positions(placed)
        best: Optional[PlacedItem] = None
        best_score = float("inf")

        for rot in rotations:
            rot_base = self._rot_ext(base, rot)
            inf_ext = self._inflated(rot_base)

            # Pre-build the rotated mesh once for this rotation so that per-
            # candidate checks only need a translation transform (cheap).
            if cm is not None:
                rot_T = np.eye(4)
                rot_T[:3, :3] = rot
                rot_mesh = assembly.mesh.copy()
                rot_mesh.apply_transform(rot_T)

            for x, y in candidates:
                z = _gravity_z(x, y, inf_ext, placed)
                pos = np.array([x, y, z])

                if not _fits_in_container(pos, inf_ext, self.container):
                    continue

                if cm is not None:
                    # Mesh-based collision: translate the pre-rotated mesh and
                    # query FCL.  Pass the translation as a transform matrix so
                    # FCL can reuse the BVH without re-triangulating.
                    actual_pos = pos + self.clearance_mm / 2.0
                    trans_T = np.eye(4)
                    trans_T[:3, 3] = actual_pos
                    hit = cm.in_collision_single(rot_mesh, transform=trans_T)
                    if _COLLISION_RESULT_IS_TUPLE:
                        hit = hit[0]
                    if hit:
                        continue
                else:
                    # AABB fallback (when FCL is not installed)
                    if _overlaps_any(pos, inf_ext, placed):
                        continue

                sc = self._score(pos)
                if sc < best_score:
                    best_score = sc
                    best = PlacedItem(
                        assembly=assembly,
                        position=pos.copy(),
                        rotation=rot.copy(),
                        inflated_extents=inf_ext.copy(),
                        clearance=self.clearance_mm,
                    )

        return best

    def _pack_flat(
        self,
        items: list[Assembly],
        fine_rotations: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PackingResult:
        """Pack a pre-expanded, pre-sorted flat list of Assembly objects.

        Internal helper — does not expand by quantity.
        """
        t0 = time.perf_counter()
        placed: list[PlacedItem] = []
        unpacked: list[Assembly] = []
        total = len(items)

        # Initialise mesh collision manager once per pack run.
        # Items are added incrementally as they are placed so the manager
        # always reflects the current container state.
        if HAS_MESH_COLLISION:
            cm: Optional["trimesh.collision.CollisionManager"] = (
                trimesh.collision.CollisionManager()
            )
            log.debug("Mesh collision detection active (FCL)")
        else:
            cm = None
            log.debug("Mesh collision unavailable — using AABB fallback")

        for idx, asm in enumerate(items):
            if progress_callback:
                progress_callback(idx, total)

            rots: list[np.ndarray] = (
                axis_aligned_rotations() if asm.allow_tipping else upright_rotations()
            )
            if fine_rotations:
                for r in fine_z_rotations(step_deg=15):
                    if not any(np.allclose(r, b, atol=1e-6) for b in rots):
                        rots.append(r)

            item = self._try_place(asm, rots, placed, cm)
            if item is None:
                unpacked.append(asm)
                log.info("Could not place '%s'", asm.name)
            else:
                placed.append(item)
                # Register the placed mesh in the collision manager so future
                # items are checked against it.
                if cm is not None:
                    world_mesh = _get_world_mesh(asm, item.actual_position, item.rotation)
                    cm.add_object(f"item_{len(placed)}", world_mesh)
                log.debug("Placed '%s' at %s", asm.name, np.round(item.position, 1))

        logs = _insert_dunnage(placed, self.log_cross_mm, self.container)
        packed_vol = sum(p.assembly.volume for p in placed)
        utilization = packed_vol / self.container.volume if self.container.volume > 0 else 0.0
        weights = [p.assembly.weight_kg for p in placed if p.assembly.weight_kg is not None]
        total_weight: Optional[float] = sum(weights) if weights else None
        collisions = _check_collisions(placed)
        elapsed = time.perf_counter() - t0

        log.info(
            "Packed %d/%d  util=%.1f%%  time=%.2fs",
            len(placed), total, utilization * 100, elapsed,
        )
        return PackingResult(
            placed=placed,
            unpacked=unpacked,
            logs=logs,
            utilization=utilization,
            total_weight_kg=total_weight,
            collisions=collisions,
            elapsed_sec=elapsed,
        )

    def pack(
        self,
        assemblies: list[Assembly],
        fine_rotations: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PackingResult:
        """Pack all assembly instances into the container.

        Args:
            assemblies: List of Assembly objects (each with a .quantity).
            fine_rotations: If True, also try 15° Z-rotation steps.
            progress_callback: Optional callable(current, total) for progress UI.

        Returns:
            PackingResult with placed items, dunnage logs, utilization, etc.
        """
        items: list[Assembly] = []
        for asm in assemblies:
            for _ in range(max(1, asm.quantity)):
                items.append(asm)
        items.sort(key=lambda a: a.volume, reverse=True)
        return self._pack_flat(items, fine_rotations, progress_callback)

    def pack_all(
        self,
        assemblies: list[Assembly],
        fine_rotations: bool = False,
        max_containers: int = 20,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[PackingResult]:
        """Pack all assemblies, spilling into additional containers as needed.

        Args:
            assemblies: List of Assembly objects (each with a .quantity).
            fine_rotations: If True, also try 15° Z-rotation steps.
            max_containers: Safety cap on the number of containers created.
            progress_callback: Optional callable(current, total) for progress UI.

        Returns:
            List of PackingResult, one per container used.
        """
        # Expand by quantity once, sort largest-first
        items: list[Assembly] = []
        for asm in assemblies:
            for _ in range(max(1, asm.quantity)):
                items.append(asm)
        items.sort(key=lambda a: a.volume, reverse=True)

        all_results: list[PackingResult] = []
        container_num = 0

        while items and container_num < max_containers:
            result = self._pack_flat(items, fine_rotations, progress_callback)
            all_results.append(result)
            # Stop if nothing was placed (items are too big for this container)
            if not result.unpacked or len(result.placed) == 0:
                break
            items = result.unpacked
            container_num += 1

        return all_results
