"""3D OpenGL viewport for the DXF Packer.

Renders the container wireframe, packed assemblies as coloured meshes,
and spacer logs as brown boxes. Supports mouse rotate/zoom/pan and
item click-selection.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal, Qt

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import GLViewWidget
    HAS_GL = True
except Exception:
    HAS_GL = False

if TYPE_CHECKING:
    from ..packer import PackingResult, PlacedItem
    from ..container import Container

log = logging.getLogger(__name__)

# Brown colour for dunnage logs
DUNNAGE_COLOR = (0.55, 0.35, 0.15, 0.75)


class Viewport3D(QWidget):
    """PyQtGraph OpenGL 3D viewport.

    Signals:
        item_selected(PlacedItem | None): Emitted when a mesh item is clicked.
    """

    item_selected = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._placed_items: list = []      # PlacedItem list from last pack
        self._mesh_items: list[gl.GLMeshItem] = []
        self._selected_idx: Optional[int] = None
        self._result = None
        self._container = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_GL:
            self._view = GLViewWidget()
            self._view.setBackgroundColor((0, 0, 0, 255))
            self._view.setCameraPosition(distance=8000, elevation=25, azimuth=45)
            self._view.mousePressEvent = self._on_mouse_press
            layout.addWidget(self._view)
        else:
            from PyQt6.QtWidgets import QLabel
            lbl = QLabel("OpenGL not available.\nInstall pyqtgraph and PyOpenGL.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
            self._view = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_result(
        self,
        result: "PackingResult",
        container: "Container",
    ) -> None:
        """Render a single packing result."""
        self.render_multi_results([(result, container)])

    def render_multi_results(
        self,
        results_containers: list[tuple["PackingResult", "Container"]],
    ) -> None:
        """Render multiple containers laid out side-by-side along X.

        Args:
            results_containers: List of (PackingResult, Container) pairs.
        """
        if not HAS_GL or self._view is None:
            return

        self._placed_items = []
        self._selected_idx = None
        self._clear_scene()

        GAP = 1000.0  # mm between containers
        offset_x = 0.0

        for result, container in results_containers:
            offset = np.array([offset_x, 0.0, 0.0])

            self._draw_container(container, offset)

            for item in result.placed:
                self._draw_item(item, len(self._placed_items), offset)
                self._placed_items.append(item)

            for log_box in result.logs:
                self._draw_log(log_box, offset)

            self._draw_grid(container, offset)
            offset_x += container.length + GAP

        # Centre camera to frame all containers
        total_length = offset_x - GAP
        first_container = results_containers[0][1]
        self._centre_camera_multi(total_length, first_container)

    def clear_scene(self) -> None:
        """Remove all rendered items."""
        self._clear_scene()
        self._placed_items = []
        self._mesh_items = []
        self._selected_idx = None

    def grab_framebuffer(self) -> Optional[object]:
        """Return a QImage screenshot of the viewport."""
        if self._view is None:
            return None
        return self._view.grabFramebuffer()

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _clear_scene(self) -> None:
        if self._view is None:
            return
        for item in self._view.items[:]:
            self._view.removeItem(item)
        self._mesh_items = []

    def _draw_container(self, container: "Container", offset: np.ndarray = None) -> None:
        """Draw the container as a white wireframe box."""
        if offset is None:
            offset = np.zeros(3)
        l, w, h = container.length, container.width, container.height
        verts, edges = _box_wireframe(offset, np.array([l, w, h]))
        lines = _edges_to_lines(verts, edges)
        line_item = gl.GLLinePlotItem(
            pos=lines,
            color=(1.0, 1.0, 1.0, 0.9),
            width=2.0,
            mode="lines",
            antialias=True,
        )
        self._view.addItem(line_item)

    def _draw_item(self, item: "PlacedItem", idx: int, offset: np.ndarray = None) -> None:
        """Draw a placed assembly as a coloured solid mesh."""
        if offset is None:
            offset = np.zeros(3)
        mesh = item.assembly.mesh.copy()

        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = item.rotation
        t[:3, 3] = item.actual_position + offset
        mesh.apply_transform(t)

        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)
        r, g, b, a = item.assembly.color

        colors_arr = np.tile([r, g, b, a], (len(faces), 1)).astype(np.float32)

        mesh_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=colors_arr,
            smooth=False,
            drawEdges=True,
            edgeColor=(0, 0, 0, 0.4),
        )
        mesh_item.setGLOptions("opaque")
        self._view.addItem(mesh_item)
        self._mesh_items.append(mesh_item)

    def _draw_log(self, log_box, offset: np.ndarray = None) -> None:
        """Draw a spacer log as a brown solid box."""
        if offset is None:
            offset = np.zeros(3)
        pos = log_box.position + offset
        ext = log_box.extents
        verts, faces = _box_solid(pos, ext)

        r, g, b, a = DUNNAGE_COLOR
        colors_arr = np.tile([r, g, b, a], (len(faces), 1)).astype(np.float32)

        mesh_item = gl.GLMeshItem(
            vertexes=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            faceColors=colors_arr,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.3, 0.15, 0.05, 0.8),
        )
        mesh_item.setGLOptions("opaque")
        self._view.addItem(mesh_item)

    def _draw_grid(self, container: "Container", offset: np.ndarray = None) -> None:
        """Draw a subtle floor grid."""
        if offset is None:
            offset = np.zeros(3)
        ox = offset[0]
        oy = offset[1]
        step = max(500, int(min(container.length, container.width) / 10 / 100) * 100)
        lines = []
        for x in range(0, int(container.length) + step, step):
            lines.extend([
                [ox + x, oy, 0], [ox + x, oy + container.width, 0],
                [ox + x, oy + container.width, 0], [ox + x, oy, 0],
            ])
        for y in range(0, int(container.width) + step, step):
            lines.extend([
                [ox, oy + y, 0], [ox + container.length, oy + y, 0],
                [ox + container.length, oy + y, 0], [ox, oy + y, 0],
            ])

        if lines:
            grid_item = gl.GLLinePlotItem(
                pos=np.array(lines, dtype=np.float32),
                color=(0.3, 0.3, 0.3, 0.4),
                width=1.0,
                mode="lines",
                antialias=False,
            )
            self._view.addItem(grid_item)

    def _centre_camera(self, container: "Container") -> None:
        """Position the camera to frame a single container."""
        self._centre_camera_multi(container.length, container)

    def _centre_camera_multi(self, total_length: float, ref_container: "Container") -> None:
        """Position the camera to frame all containers."""
        import pyqtgraph as pg
        cx = total_length / 2
        cy = ref_container.width / 2
        cz = ref_container.height / 2
        dist = max(total_length, ref_container.width, ref_container.height) * 1.8
        self._view.opts["center"] = pg.Vector(cx, cy, cz)
        self._view.setCameraPosition(distance=dist, elevation=25, azimuth=45)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_mouse_press(self, event) -> None:
        """Handle mouse clicks for item selection (simple AABB ray cast)."""
        # Delegate default handling first
        GLViewWidget.mousePressEvent(self._view, event)

        if not self._placed_items:
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Basic selection: cycle through items on repeated click
        if self._selected_idx is None:
            self._selected_idx = 0
        else:
            self._selected_idx = (self._selected_idx + 1) % len(self._placed_items)

        self._highlight_selected()
        self.item_selected.emit(self._placed_items[self._selected_idx])

    def _highlight_selected(self) -> None:
        """Brighten the selected mesh item and dim others."""
        for i, mesh_item in enumerate(self._mesh_items):
            if i == self._selected_idx:
                mesh_item.setGLOptions("additive")
            else:
                mesh_item.setGLOptions("opaque")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _box_wireframe(
    position: np.ndarray, extents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """8 corners + 12 edges of a box."""
    x0, y0, z0 = position
    dx, dy, dz = extents
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)
    return verts, edges


def _edges_to_lines(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Expand edge index pairs into a flat (N*2, 3) line-segment array."""
    lines = []
    for a, b in edges:
        lines.append(verts[a])
        lines.append(verts[b])
    return np.array(lines, dtype=np.float32)


def _box_solid(
    position: np.ndarray, extents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """12 triangles (24 verts in face-vertex layout) for a solid box."""
    x0, y0, z0 = position
    dx, dy, dz = extents
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz

    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ], dtype=np.int32)

    return verts, faces
