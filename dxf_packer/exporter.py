"""Export packing results to DXF, JSON, CSV, and PDF.

PDF report includes:
  - 3D viewport screenshot
  - Top view  (looking down  −Z)
  - Front view (looking along +Y)
  - Side view  (looking along −X)
  - Summary stats table
  - Placement table
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .packer import PackingResult, PlacedItem, SpacerLog
    from .container import Container

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Placement transform (matches the viewport exactly)
# ---------------------------------------------------------------------------

def item_transform_4x4(item: "PlacedItem") -> np.ndarray:
    """4×4 matrix that places the normalized mesh at its packed position.

    This is the same transform the viewport applies::

        mesh.apply_transform(T)

    where T rotates the mesh (min corner at origin) then translates to position.

    Args:
        item: A packed PlacedItem.

    Returns:
        4×4 float64 homogeneous matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = item.rotation
    T[:3, 3] = item.actual_position   # offset by clearance/2 from inflated corner
    return T


# ---------------------------------------------------------------------------
# DXF export
# ---------------------------------------------------------------------------

def _define_assembly_block(
    doc,
    block_name: str,
    item: "PlacedItem",
    layer: str,
    offset: np.ndarray,
) -> None:
    """Create a BLOCK definition with the pre-transformed assembly mesh.

    The mesh faces are stored in container-local coordinates shifted by
    *offset*, so an INSERT at (0, 0, 0) places them correctly in model space.
    """
    block = doc.blocks.new(name=block_name)
    T = item_transform_4x4(item)
    mesh = item.assembly.mesh.copy()
    mesh.apply_transform(T)

    for face in mesh.faces:
        v0 = (mesh.vertices[face[0]] + offset).tolist()
        v1 = (mesh.vertices[face[1]] + offset).tolist()
        v2 = (mesh.vertices[face[2]] + offset).tolist()
        block.add_3dface(
            [v0, v1, v2, v2],
            dxfattribs={"layer": layer, "color": 1},
        )


def export_dxf(
    result: "PackingResult",
    container: "Container",
    output_path: str | Path,
) -> None:
    """Write a DXF with the container outline and all packed geometry.

    Each assembly is exported as a named BLOCK definition with a single
    INSERT reference, so the assembly appears as one selectable entity in CAD.
    Spacer logs are exported as 12-LINE wireframe boxes on layer DUNNAGE.

    Args:
        result: Packing result.
        container: Container dimensions.
        output_path: Destination .dxf path.
    """
    export_dxf_multi([(result, container)], output_path)


def export_dxf_multi(
    results_containers: list[tuple["PackingResult", "Container"]],
    output_path: str | Path,
) -> None:
    """Write a DXF with multiple containers laid out side-by-side along X.

    Each container is separated by a 1 000 mm gap.  Every assembly is exported
    as a named BLOCK + INSERT so it appears as one selectable entity in CAD.

    Args:
        results_containers: List of (PackingResult, Container) pairs.
        output_path: Destination .dxf path.
    """
    import ezdxf

    output_path = Path(output_path)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    GAP = 1000.0  # mm between containers
    offset_x = 0.0

    for c_idx, (result, container) in enumerate(results_containers):
        l, w, h = container.length, container.width, container.height
        offset = np.array([offset_x, 0.0, 0.0])

        # ── Container outline ──────────────────────────────────────────
        _add_box_lines(msp, offset, np.array([l, w, h]),
                       layer="CONTAINER", color=7)

        # ── Container label ────────────────────────────────────────────
        label_pos = (offset_x + l / 2, -600.0, 0.0)
        msp.add_text(
            f"Container {c_idx + 1}",
            dxfattribs={"insert": label_pos, "height": 300.0,
                        "layer": "CONTAINER_LABEL", "color": 7,
                        "halign": 4},   # 4 = center
        )

        # ── Packed assemblies as BLOCK + INSERT ────────────────────────
        for idx, item in enumerate(result.placed):
            block_name = f"ASM_{item.assembly.name}_C{c_idx}_{idx}"
            layer = f"ASSEMBLY_{item.assembly.name}"
            _define_assembly_block(doc, block_name, item, layer, offset)
            msp.add_blockref(
                block_name,
                insert=(0.0, 0.0, 0.0),
                dxfattribs={"layer": layer},
            )

        # ── Spacer logs ────────────────────────────────────────────────
        for spacer in result.logs:
            _add_box_lines(msp, spacer.position + offset, spacer.extents,
                           layer="DUNNAGE", color=30)

        offset_x += l + GAP

    doc.saveas(str(output_path))
    log.info("DXF exported → %s", output_path)


def _add_box_lines(msp, pos: np.ndarray, ext: np.ndarray, layer: str, color: int) -> None:
    """Write 12 LINE entities forming a wireframe box to *msp*."""
    x0, y0, z0 = pos
    x1, y1, z1 = pos + ext
    corners = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    for a, b in edges:
        msp.add_line(corners[a].tolist(), corners[b].tolist(),
                     dxfattribs={"layer": layer, "color": color})


# ---------------------------------------------------------------------------
# JSON manifest
# ---------------------------------------------------------------------------

def export_json(
    result: "PackingResult",
    container: "Container",
    output_path: str | Path,
) -> None:
    """Write a JSON manifest of the packing result."""
    export_json_multi([(result, container)], output_path)


def export_json_multi(
    results_containers: list[tuple["PackingResult", "Container"]],
    output_path: str | Path,
) -> None:
    """Write a JSON manifest for multiple containers."""
    output_path = Path(output_path)

    total_placed = sum(len(r.placed) for r, _ in results_containers)
    last_unpacked = results_containers[-1][0].unpacked if results_containers else []

    containers_data = []
    for c_idx, (result, container) in enumerate(results_containers):
        containers_data.append({
            "container_index": c_idx + 1,
            "container": {
                "name": container.name,
                "length_mm": container.length,
                "width_mm": container.width,
                "height_mm": container.height,
                "volume_mm3": container.volume,
            },
            "summary": {
                "items_packed": len(result.placed),
                "utilization_pct": round(result.utilization * 100, 2),
                "total_weight_kg": result.total_weight_kg,
                "packing_time_sec": round(result.elapsed_sec, 3),
                "collisions": result.collisions,
            },
            "items": [
                {
                    "name": p.assembly.name,
                    "filepath": str(p.assembly.filepath),
                    "position_mm": p.position.tolist(),
                    "actual_position_mm": p.actual_position.tolist(),
                    "rotation_matrix": p.rotation.tolist(),
                    "extents_mm": p.extents.tolist(),
                    "weight_kg": p.assembly.weight_kg,
                    "allow_tipping": p.assembly.allow_tipping,
                }
                for p in result.placed
            ],
            "logs": [
                {
                    "label": s.label,
                    "position_mm": s.position.tolist(),
                    "extents_mm": s.extents.tolist(),
                }
                for s in result.logs
            ],
        })

    data = {
        "total_containers": len(results_containers),
        "total_items_packed": total_placed,
        "items_unpacked": len(last_unpacked),
        "unpacked": [
            {
                "name": a.name,
                "filepath": str(a.filepath),
                "dimensions_mm": a.dimensions.tolist(),
            }
            for a in last_unpacked
        ],
        "containers": containers_data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("JSON exported → %s", output_path)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(result: "PackingResult", output_path: str | Path) -> None:
    """Write a flat CSV table of all placements."""
    export_csv_multi([(result, None)], output_path)


def export_csv_multi(
    results_containers: list[tuple["PackingResult", object]],
    output_path: str | Path,
) -> None:
    """Write a flat CSV table of all placements across multiple containers."""
    output_path = Path(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "container", "name", "filepath",
            "pos_x_mm", "pos_y_mm", "pos_z_mm",
            "ext_x_mm", "ext_y_mm", "ext_z_mm",
            "weight_kg", "allow_tipping", "status",
        ])
        for c_idx, (result, _container) in enumerate(results_containers):
            for item in result.placed:
                p, e = item.actual_position, item.extents
                w.writerow([
                    c_idx + 1,
                    item.assembly.name, item.assembly.filepath,
                    round(p[0], 2), round(p[1], 2), round(p[2], 2),
                    round(e[0], 2), round(e[1], 2), round(e[2], 2),
                    item.assembly.weight_kg, item.assembly.allow_tipping, "placed",
                ])
        # Unpacked from the last container pass
        last_unpacked = results_containers[-1][0].unpacked if results_containers else []
        for asm in last_unpacked:
            d = asm.dimensions
            w.writerow([
                "", asm.name, asm.filepath, "", "", "",
                round(d[0], 2), round(d[1], 2), round(d[2], 2),
                asm.weight_kg, asm.allow_tipping, "unpacked",
            ])
    log.info("CSV exported → %s", output_path)


# ---------------------------------------------------------------------------
# Orthographic 2-D projection helper
# ---------------------------------------------------------------------------

def _draw_ortho_view(
    ax,
    result: "PackingResult",
    container: "Container",
    plane: str,          # "top" | "front" | "side"
    title: str,
) -> None:
    """Draw one orthographic projection onto a matplotlib Axes.

    plane="top"   → XY  (looking down   −Z)
    plane="front" → XZ  (looking along  +Y)
    plane="side"  → YZ  (looking along  −X)
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    l, w, h = container.length, container.width, container.height

    # Map each plane to (horiz_dim, vert_dim, horiz_label, vert_label)
    plane_map = {
        "top":   (0, 1, "X (length)", "Y (width)",  l, w),
        "front": (0, 2, "X (length)", "Z (height)", l, h),
        "side":  (1, 2, "Y (width)",  "Z (height)", w, h),
    }
    hi, vi, hl, vl, hmax, vmax = plane_map[plane]

    # Container outline
    ax.add_patch(mpatches.Rectangle(
        (0, 0), hmax, vmax,
        linewidth=1.5, edgecolor="#ffffff", facecolor="#1a1a2a",
    ))

    # Placed items
    for item in result.placed:
        p = item.actual_position
        e = item.extents
        color = item.assembly.color[:3]
        rect = mpatches.Rectangle(
            (p[hi], p[vi]), e[hi], e[vi],
            linewidth=0.8, edgecolor="black",
            facecolor=color, alpha=0.85,
        )
        ax.add_patch(rect)
        # Label (first 8 chars)
        cx, cy = p[hi] + e[hi] / 2, p[vi] + e[vi] / 2
        ax.text(cx, cy, item.assembly.name[:8], ha="center", va="center",
                fontsize=5, color="white", clip_on=True)

    # Spacer logs
    for spacer in result.logs:
        p = spacer.position
        e = spacer.extents
        rect = mpatches.Rectangle(
            (p[hi], p[vi]), e[hi], e[vi],
            linewidth=0.5, edgecolor="#8b5e3c",
            facecolor="#8b5e3c", alpha=0.6,
        )
        ax.add_patch(rect)

    ax.set_xlim(-hmax * 0.02, hmax * 1.05)
    ax.set_ylim(-vmax * 0.02, vmax * 1.08)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.set_xlabel(hl, color="#aaa", fontsize=7)
    ax.set_ylabel(vl, color="#aaa", fontsize=7)
    ax.tick_params(colors="#aaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.set_facecolor("#12121e")


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def export_pdf(
    result: "PackingResult",
    container: "Container",
    output_path: str | Path,
    screenshot_path: Optional[str | Path] = None,
) -> None:
    """Generate a PDF packing report (single container)."""
    export_pdf_multi([(result, container)], output_path, screenshot_path)


def export_pdf_multi(
    results_containers: list[tuple["PackingResult", "Container"]],
    output_path: str | Path,
    screenshot_path: Optional[str | Path] = None,
) -> None:
    """Generate a PDF packing report covering all containers.

    Pages:
      1. Screenshot + overall summary.
      Per container: orthographic views + placement table.

    Args:
        results_containers: List of (PackingResult, Container) pairs.
        output_path: Destination .pdf path.
        screenshot_path: Optional PNG viewport screenshot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image, PageBreak,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import io
    except ImportError as e:
        raise ImportError(f"reportlab and matplotlib are required for PDF export: {e}")

    output_path = Path(output_path)
    tmp_dir = output_path.parent

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Heading1"], fontSize=16,
                                 textColor=colors.HexColor("#2c6fad"))
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12,
                                 textColor=colors.HexColor("#2c6fad"))
    body_style  = styles["Normal"]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        topMargin=12*mm, bottomMargin=12*mm,
        leftMargin=15*mm, rightMargin=15*mm,
    )
    story = []

    total_placed = sum(len(r.placed) for r, _ in results_containers)
    last_unpacked = results_containers[-1][0].unpacked if results_containers else []

    # ── Page 1: screenshot + overall summary ──────────────────────────────
    story.append(Paragraph("DXF Assembly Packing Report", title_style))
    story.append(Spacer(1, 4*mm))

    if screenshot_path and Path(screenshot_path).exists():
        story.append(Image(str(screenshot_path), width=200*mm, height=100*mm))
        story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Overall Summary", h2_style))
    first_container = results_containers[0][1]
    summary_rows = [
        ["Container type",   f"{first_container.name}  ({first_container.length:.0f} × {first_container.width:.0f} × {first_container.height:.0f} mm)"],
        ["Containers used",  str(len(results_containers))],
        ["Total items packed", str(total_placed)],
        ["Items unpacked",   str(len(last_unpacked))],
    ]
    t = Table(summary_rows, colWidths=[55*mm, 130*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#2a2a3e")),
        ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#88aadd")),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#444")),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("PADDING",     (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    tmp_files: list[Path] = []

    # ── Per-container pages ────────────────────────────────────────────────
    for c_idx, (result, container) in enumerate(results_containers):
        story.append(PageBreak())
        story.append(Paragraph(
            f"Container {c_idx + 1}  —  {container.name}  "
            f"({container.length:.0f} × {container.width:.0f} × {container.height:.0f} mm)",
            h2_style,
        ))
        story.append(Spacer(1, 2*mm))

        # Per-container stats
        stats_rows = [
            ["Items packed",   str(len(result.placed))],
            ["Utilization",    f"{result.utilization*100:.1f}%"],
            ["Total weight",   f"{result.total_weight_kg:.1f} kg" if result.total_weight_kg else "—"],
            ["Collisions",     str(len(result.collisions)) if result.collisions else "None"],
        ]
        ts = Table(stats_rows, colWidths=[55*mm, 80*mm])
        ts.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#2a2a3e")),
            ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#88aadd")),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#444")),
            ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("PADDING",     (0, 0), (-1, -1), 3),
        ]))
        story.append(ts)
        story.append(Spacer(1, 3*mm))

        # Orthographic views
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.patch.set_facecolor("#12121e")
        views = [
            (axes[0, 0], "top",   f"Top View  (−Z)  · Container {c_idx + 1}"),
            (axes[0, 1], "front", "Front View  (+Y)"),
            (axes[1, 0], "side",  "Side View  (−X)"),
        ]
        for ax, plane, title in views:
            _draw_ortho_view(ax, result, container, plane, title)

        ax4 = axes[1, 1]
        ax4.set_facecolor("#12121e")
        ax4.set_title("Legend", color="white", fontsize=9)
        ax4.axis("off")
        seen_names: set[str] = set()
        legend_patches = []
        for item in result.placed:
            if item.assembly.name not in seen_names:
                seen_names.add(item.assembly.name)
                r, g, b = item.assembly.color[:3]
                legend_patches.append(
                    mpatches.Patch(facecolor=(r, g, b), edgecolor="black",
                                   label=item.assembly.name)
                )
        legend_patches.append(
            mpatches.Patch(facecolor="#8b5e3c", edgecolor="#5a3d20",
                           label="Dunnage log")
        )
        ax4.legend(handles=legend_patches, loc="center", fontsize=8,
                   facecolor="#1e1e2a", labelcolor="white",
                   edgecolor="#555", framealpha=0.9)
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        ortho_path = tmp_dir / f"_ortho_views_c{c_idx}.png"
        tmp_files.append(ortho_path)
        with open(ortho_path, "wb") as f_img:
            f_img.write(buf.read())

        story.append(Image(str(ortho_path), width=250*mm, height=150*mm))

        # Placement table for this container
        if result.placed:
            story.append(Spacer(1, 3*mm))
            headers = ["Name", "Pos X", "Pos Y", "Pos Z", "L (mm)", "W (mm)", "H (mm)", "Weight"]
            rows = [headers]
            for item in result.placed:
                p = item.actual_position
                e = item.extents
                rows.append([
                    item.assembly.name,
                    f"{p[0]:.0f}", f"{p[1]:.0f}", f"{p[2]:.0f}",
                    f"{e[0]:.0f}", f"{e[1]:.0f}", f"{e[2]:.0f}",
                    f"{item.assembly.weight_kg:.1f} kg" if item.assembly.weight_kg else "—",
                ])
            col_w = [55*mm] + [22*mm] * 7
            tbl = Table(rows, colWidths=col_w, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2c6fad")),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",     (0, 0), (-1, -1), 8),
                ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#555")),
                ("PADDING",      (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#1e1e2e"), colors.HexColor("#252535")]),
                ("TEXTCOLOR",    (0, 1), (-1, -1), colors.HexColor("#ccddee")),
            ]))
            story.append(tbl)

    # ── Unpacked items ─────────────────────────────────────────────────────
    if last_unpacked:
        story.append(PageBreak())
        story.append(Paragraph("Unpacked Items (did not fit in any container)", h2_style))
        story.append(Spacer(1, 3*mm))
        for asm in last_unpacked:
            d = asm.dimensions
            story.append(Paragraph(
                f"• {asm.name}  —  {d[0]:.0f} × {d[1]:.0f} × {d[2]:.0f} mm",
                body_style,
            ))

    doc.build(story)

    for p in tmp_files:
        try:
            p.unlink()
        except Exception:
            pass

    log.info("PDF exported → %s", output_path)
