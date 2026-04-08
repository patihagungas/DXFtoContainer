"""3D bin packing — Boundary-Point Best-Fit Decreasing.

Key improvements over naive extreme-point:
  - Candidate positions are the full Cartesian product of all X/Y face boundaries,
    giving O(n²) positions instead of O(n) → far fewer gaps.
  - Gravity Z is computed for every (x,y) candidate, not just the pre-stored Z.
  - Score favours low Z first (gravity), then low Y, then low X (fill front-left-bottom).
  - Clearance inflation is applied symmetrically; dunnage logs are drawn inside
    that clearance gap and correctly aligned to the actual (non-inflated) item.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from .assembly import Assembly
from .container import Container
from .rotations import axis_aligned_rotations, upright_rotations, fine_z_rotations

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

def _candidate_xy_positions(placed: list[PlacedItem]) -> list[tuple[float, float]]:
    """Return all (x, y) boundary positions to try for the next item.

    Builds the Cartesian product of:
      X-breakpoints: 0 + right face of every placed item
      Y-breakpoints: 0 + front face of every placed item

    This gives O(n²) positions and eliminates the gap problem caused by
    sparse extreme-point lists.
    """
    xs = sorted({0.0} | {p.position[0] + p.inflated_extents[0] for p in placed})
    ys = sorted({0.0} | {p.position[1] + p.inflated_extents[1] for p in placed})
    return [(x, y) for x in xs for y in ys]


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
    """Insert floor dunnage logs under each floor-level item.

    Two logs per item, running along the item's longest axis, placed at 25%
    and 75% of the perpendicular axis.  The logs are sized to fit inside the
    clearance gap below the actual mesh (clearance/2 tall, clamped to log_cross).

    Args:
        placed: All placed items.
        log_cross: Requested log cross-section (mm). Clamped to clearance/2.
        container: Container dimensions.

    Returns:
        List of SpacerLog objects.
    """
    logs: list[SpacerLog] = []

    for item in placed:
        half_c = item.clearance / 2.0
        # Only place floor logs when the item sits on (or near) the container floor
        if item.position[2] > half_c + 1.0:
            continue

        # Log height is limited to the clearance gap below the actual mesh
        log_h = min(log_cross, max(half_c, 1.0))

        # Actual mesh footprint (not the inflated box)
        ap = item.actual_position        # (x, y, z) of actual mesh min corner
        ae = item.extents                # actual extents

        ax0 = max(ap[0], 0.0)
        ay0 = max(ap[1], 0.0)
        ax1 = min(ap[0] + ae[0], container.length)
        ay1 = min(ap[1] + ae[1], container.width)

        if ae[0] >= ae[1]:
            # Longest axis is X → logs run along X, spaced along Y
            log_len = ax1 - ax0
            if log_len <= 0:
                continue
            for frac in (0.25, 0.75):
                log_y = ay0 + (ay1 - ay0) * frac - log_cross / 2.0
                log_y = max(0.0, min(log_y, container.width - log_cross))
                logs.append(SpacerLog(
                    position=np.array([ax0, log_y, 0.0]),
                    extents=np.array([log_len, log_cross, log_h]),
                    label="floor dunnage",
                ))
        else:
            # Longest axis is Y → logs run along Y, spaced along X
            log_len = ay1 - ay0
            if log_len <= 0:
                continue
            for frac in (0.25, 0.75):
                log_x = ax0 + (ax1 - ax0) * frac - log_cross / 2.0
                log_x = max(0.0, min(log_x, container.length - log_cross))
                logs.append(SpacerLog(
                    position=np.array([log_x, ay0, 0.0]),
                    extents=np.array([log_cross, log_len, log_h]),
                    label="floor dunnage",
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
    ) -> Optional[PlacedItem]:
        """Find the best (rotation, position) for one assembly instance.

        Tries every rotation × every boundary-point (x, y) pair, drops each
        candidate to its gravity Z, then picks the lowest-scoring valid placement.

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

            for x, y in candidates:
                z = _gravity_z(x, y, inf_ext, placed)
                pos = np.array([x, y, z])

                if not _fits_in_container(pos, inf_ext, self.container):
                    continue
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
        t0 = time.perf_counter()

        # Expand by quantity, sort largest volume first (BFFD)
        items: list[Assembly] = []
        for asm in assemblies:
            for _ in range(max(1, asm.quantity)):
                items.append(asm)
        items.sort(key=lambda a: a.volume, reverse=True)

        placed: list[PlacedItem] = []
        unpacked: list[Assembly] = []
        total = len(items)

        for idx, asm in enumerate(items):
            if progress_callback:
                progress_callback(idx, total)

            # Build rotation list
            rots: list[np.ndarray] = (
                axis_aligned_rotations() if asm.allow_tipping else upright_rotations()
            )
            if fine_rotations:
                for r in fine_z_rotations(step_deg=15):
                    if not any(np.allclose(r, b, atol=1e-6) for b in rots):
                        rots.append(r)

            item = self._try_place(asm, rots, placed)
            if item is None:
                unpacked.append(asm)
                log.info("Could not place '%s'", asm.name)
            else:
                placed.append(item)
                log.debug(
                    "Placed '%s' at %s", asm.name, np.round(item.position, 1)
                )

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
